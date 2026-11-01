# BGE-M3 hybrid embeddings on llama.cpp (GTX 1080 Ti, CUDA and Vulkan)

Goal: serve [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) as a hybrid (dense + sparse) embedding server from
`llama-server`, with as little CPU and as much GPU work as possible, on both CUDA and Vulkan, packaged as Docker images
(`.devops/bge-m3/`). The comparison baseline is a FlagEmbedding (PyTorch) reference server running
`BGEM3FlagModel.encode(return_dense=True, return_sparse=True, max_length=512)`.

Machine: Dell R720, 2x Xeon E5-2695 v2 (20C/40T, no AVX2), GTX 1080 Ti (Pascal sm_61, PCIe 3.0 x8), driver 580.

## What changed

| area | change |
|---|---|
| model | `sparse_out` tensor (weight `[n_embd, 1]` + bias): the converter reads `sparse_linear.pt` next to an XLM-R checkpoint |
| graph | `build_sparse()`: `relu(W·h + b)` per token, computed next to the pooled CLS embedding, in the same graph |
| API | `llama_get_embeddings_sparse_seq(ctx, seq_id, tokens, weights, n_max)` |
| server | `/embeddings` and `/v1/embeddings` return `"sparse": {"indices", "values"}`: max per token id, CONTROL/UNKNOWN tokens and weights <= 0 dropped (as FlagEmbedding does) |
| server | `--embd-truncate N` / request `"truncate": N`: keep the first N-1 tokens and the closing `</s>` (FlagEmbedding's `max_length`) |
| attention mask | no-cache (encoder) mask: tokens of one sequence share a row, a run copies the previous row instead of recomputing it |
| Vulkan | Pascal (NVIDIA without coopmat) f32/f16 matmul tile `{256, 128, 128, 32, 32, 64, 2, 2, 1, 1}` + an invariant check |
| tests | `test-backend-ops`: f32/f16/q4_0 matmuls large enough for the large tile, aligned and unaligned K |
| Docker | `.devops/bge-m3/`: offline converter for `hf download`, hardened CUDA and Vulkan server images |

Before this, the only way was `--pooling none` (every token's hidden state over HTTP) plus the dense/sparse heads in
the client: a lot of JSON and client CPU per request.

## Accuracy

97 Hungarian constitution articles + 25 questions, against FlagEmbedding fp32 (`bge-cmp.py`):

| build | tokens | dense cos min | sparse rel. L1 | top-1 / top-5 overlap (dense, sparse, hybrid, RRF) |
|---|---|---|---|---|
| CUDA f32 / f16 | identical | 0.99981 | 0.34 % | identical / >= 0.992 |
| Vulkan f32 / f16 (new tile) | identical | 0.99981 | 0.34 % | identical / >= 0.992 |
| Docker CUDA, Docker Vulkan | identical | 0.99981 | 0.34 % | identical / >= 0.992 |

## Throughput (f32, 36-process client, no errors anywhere)

Query = one short question per request, index = 16 chunks of ~500 tokens per request.

| server | peak queries/s | index texts/s | server CPU | VRAM |
|---|---|---|---|---|
| FlagEmbedding reference fp32 (best of 1-4 instances) | 5.5 | 7.7 | 1-2 cores per instance | 4-11 GB, OOM at 6 instances |
| llama.cpp CUDA, per-token output + client head | 433 (64 clients) | 25.1 | 3-8 cores | 3.2 GB |
| llama.cpp Vulkan, per-token output + client head | 154 (96 clients) | 12.3 | 1-5 cores | 3.0 GB |
| llama.cpp CUDA, server-side hybrid, ub 1024 | 428 (32 clients, p50 75 ms) | 33.5 | 2.4 / 1.1 cores | 2.7 GB |
| llama.cpp Vulkan, server-side hybrid, old tile, ub 4096 | 111 | 13.7 | ~2 cores | 2.8 GB |
| llama.cpp Vulkan, server-side hybrid, new tile, ub 2048 | 270 (p50 94 ms) | 29.9 | ~2 / 0.6 cores | 2.7 GB |

The 122-text accuracy set encodes in 50.7 s with the per-token path, 4.3 s server-side on CUDA, 8.3 s -> 5.7 s on
Vulkan with the new tile (4.3 s with f16 weights).

## Bottlenecks found by profiling

1. **Per-token output** (fixed): with `--pooling none` every token's 1024 floats went through JSON to the client.
   The sparse head is one `[1024] x [1024, n_tokens]` matmul, it belongs in the graph.
2. **Vulkan f32/f16 matmul on Pascal** (fixed): `GGML_VK_PERF_LOGGER` showed MUL_MAT at ~87 % of the encode time at
   2.4-2.6 TFLOPS. The default large tile (128 threads, 4x4 thread tiles) runs out of registers on Pascal. A 256-thread
   tile with 2x2 thread tiles and BK=32 is 2.4x faster at pp512 and 1.7x at pp3584 in `llama-bench -embd 1`, and 2.4x
   faster end to end for queries.
3. **Cross-sequence attention** (open): the non-causal encoder mask is `n_tokens x n_tokens` over every sequence in a
   ubatch, so the attention work grows quadratically with the ubatch size. That is why a small ubatch wins on CUDA
   (index 33.6 texts/s at ub 1024 vs 20.9 at ub 8192). Per-sequence attention (or a KV_min skip in the FA kernels next
   to the existing KV_max) would remove the waste.
4. **Index fairness** (open): with 32 clients the index p95 is 40+ s while the p50 is 7-10 s; batches of long chunks
   wait behind each other.

## The tile lesson: fast can mean wrong

The first tile sweep found configs that were 2.5-6x faster. They were broken:

- `TM = 1`: `mul_mm.comp` packs two rows of a thread tile per accumulator (`for cr < TM / 2`), so with TM=1 no
  product is accumulated at all.
- `BLOCK_SIZE * LOAD_VEC / BK > BM`: a load pass writes beyond the shared-memory tile (LOAD_VEC is 8 for the fp16
  shaders, so 512 threads need BK >= 32).

`test-backend-ops` passed them anyway because none of its f32/f16 `MUL_MAT` shapes is large enough to select the large
tile. The new tests (`m=1024, n=512, k=1024` and `m=1000, n=509, k=1024`) fail on exactly these configs, and the Pascal
branch asserts both invariants. Only configs that pass these tests were measured afterwards.

## Docker

See `.devops/bge-m3/README.md`. Things that came up while testing on this machine:

- `GGML_NATIVE=OFF` alone builds the CPU backend with AVX2: SIGILL on Ivy Bridge. The images use
  `GGML_BACKEND_DL` + `GGML_CPU_ALL_VARIANTS`.
- The NVIDIA Vulkan ICD (`libGLX_nvidia`) needs libX11/libXext and the glvnd `libEGL.so.1` inside the image; without
  them the loader drops the ICD and llama.cpp silently runs on the CPU (323 s instead of 6 s for the test set).
  The images now pin `LLAMA_ARG_DEVICE`, so a missing GPU is a start-up error.
- The converter must see the whole snapshot: without `1_Pooling/` and `modules.json` the GGUF has no pooling type.

## Qwen3.5 regression check

The changes touch shared code (graph inputs, context output buffers, the Vulkan matmul tile), so the usual Qwen3.5-9B
validation was run again; see `qwen-regression.md`.
