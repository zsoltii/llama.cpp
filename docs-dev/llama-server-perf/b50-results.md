# Intel Arc Pro B50 (Vulkan2) — llama-server speedup, Qwen3.5-9B-MTP

## Environment

- GPU: Intel Arc Pro B50 (BMG G21, Xe2), 16 GB, Mesa ANV + xe kernel driver (Fedora 44).
- **Thunderbolt eGPU**: the B50 sits in a TB dock (PCIe 3.0 x4 to the dock).
  There is **no ReBAR**; the host-visible VRAM window is 256 MiB, and its heap budget has ~2 MiB free at startup.
- CPU: i7-11850H (16 threads), 31 GB RAM. Other GPUs: Intel UHD (Vulkan0), NVIDIA T1200 (Vulkan1).
- Reference: brew `llama-server` 0.7.0 (build 11669, Vulkan).
- Model: `unsloth/Qwen3.5-9B-MTP-GGUF:UD-Q4_K_XL` (+ mmproj), 128k ctx, q8_0 KV, FA on.
- Coopmat shapes on the B50: only **M=8 N=16 K=16** (f16/bf16), int8: 8x16x32.

## Measurement protocol

- `~/llama-bench-results/bench.py`:
  - server restart per config, warm-up, temp=0, `cache_prompt=false`, thinking off, medians;
  - output hash (determinism);
  - a 1 s monitor: GPU busy % from the xe fdinfo cycles, GPU MHz, VRAM, process and system CPU.
- Corpus (`~/llama-bench-results/corpus/`):
  - `story` = a short Hungarian story request, 1024 generated tokens;
  - `wiki` = the Roman Empire Wikipedia article (99.7 KB, 20 726 tokens) + an analysis request, 512 tokens.
- Kernel level: `~/llama-bench-results/kbench.py` and `test-backend-ops perf` (Qwen3.5 shapes).

## Reference (brew)

| config | story pp | story tg | wiki pp | wiki tg | acceptance (story/wiki) |
|---|---|---|---|---|---|
| no MTP | 306 | 25.6 | 423 | 18.3 | – |
| MTP n=1 | 238 | **27.6** | 376 | **19.6** | 0.58 / 0.73 |
| MTP n=2 | 234 | 27.6 | 376 | 16.9 | 0.47 / 0.55 |
| MTP n=3 | 234 | 24.5 | 377 | 16.8 | 0.34 / 0.46 |
| MTP n=4 | 213 | 19.9 | 375 | 11.5 | 0.25 / 0.39 |

All with `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1`; without it, MTP gives **2.5 tok/s** (see below).
Reference command: the user's command + `--spec-type draft-mtp --spec-draft-n-max 1 -ctkd q8_0 -ctvd q8_0`.

## Findings and fixes

1. **Small BAR → compute buffers in system memory.**
   - The backend asked for the device-local + host-visible type; the window filled up and the driver put the rest into sysmem.
   - The GPU then read it over TB (mat-vec 2–5 GFLOPS).
   - Fix (`ggml-vulkan.cpp`): if the host-visible device-local heap is < VRAM/2, use device-local memory only. `GGML_VK_ALLOW_SMALL_BAR_VIDMEM` switches this off.
   - 10× with MTP.
2. **Deferred MTP catch-up** (`LLAMA_MTP_DEFER_CATCHUP=1`, existing): one decode fewer per step, ~+5%.
3. **Backend sampling** (`-bs`): no 2 MB logit readback over TB.
   - post_decode 3.5 → 0.75 ms, ~+9% tg, output bitwise identical.
   - The pp of short prompts drops (238 → 167) — to be investigated.
4. **FA split-k** (decode/verify):
   - The old heuristic gave ~2 WG per core, i.e. 32 workgroups on the B50: latency-bound (q8_0 KV 20k: 24 GB/s).
   - New, vendor-neutral: 16 WG per core, but each split gets at least 256 keys (`GGML_VK_FA_WG_PER_CORE`, `GGML_VK_FA_MIN_KV_SPLIT`).
   - FA nb=1/2/3: 2.1× / 2.3× / 3.1×.
5. **FA coopmat1 with 8x16x16**: the shader only knew 16x16x16, so prefill FA ran on the scalar path on Intel Xe2.
   - The shader now takes the M size from the device through the `FaCmM` spec constant (16 = unchanged NVIDIA/AMD code).
   - Prefill FA (512 × 8.7k): f16 65 → 21.6 ms, q8_0 80 → 29 ms.
   - `test-backend-ops -o FLASH_ATTN_EXT`: 5368/5368 OK.
6. Diagnostics: `GGML_VK_FA_DEBUG=1` (FA path/split, coopmat shapes); the perf logger also prints the shape of GET_ROWS/ARGSORT/TOP_K.
7. **`-bs` + short prompt (pp 238 → 175).**
   - Cause: every request calls `llama_set_sampler` (reset: nullptr, new task: a new chain).
     That triggers a full `sched_reserve()`: 63 ms, 3 graph_reserves + a new scheduler, and graph reuse is lost too.
   - Fix (`src/llama-context.cpp`): swapping the sampler only requests a check. It reallocates only when the structure of the chains (seq → sampler names) differs from the last allocated one.
   - The graph is rebuilt anyway because of the new sampler pointer (`can_reuse` compares pointers).
8. **Prompt checkpoint (recurrent state, 50 MiB) 106 ms → 39 ms.**
   - 64 ms of it was the page faulting of a fresh 50 MiB `std::vector`.
   - Fix (`common/common.cpp`): checkpoint buffers go back into a pool (max 512 MiB), and `update_tgt/dft` take from there.
   - The remaining 39 ms is the 48 readbacks over TB.
   - 7+8 together: story pp **175 → 294 tok/s** (ref 238).
9. **Uploading graph inputs (scheduler).**
   - Every user input (tokens, positions, mask, indices, s_copy …) and the output of the CPU split (token embedding) was a separate, blocking upload.
     That is 9 per draft = 1.7 ms and 12 per verify = 2.7 ms per step, while the GPU idles.
   - Fix (`ggml/src/ggml-backend.cpp`, backend-neutral): one sync per split; the inputs are copied into the backend's pinned host buffer and go up with `set_tensor_async`.
     On Vulkan this is a copy in the command stream, on CUDA an async pinned memcpy.
   - 2.6–2.8 ms → 0.3–0.4 ms per step. Can be switched off with `GGML_SCHED_SYNC_INPUTS=1`.
10. Did not work out:
    - Br=32 in the coopmat1 FA (more queries per workgroup for K/V reuse): slower on Xe2 (f16 21 → 38 ms, occupancy);
    - n=2/3 drafts (the draft LM head costs 4.3 ms per step).

## KV types — FA kernel (µs, B50, after finding 4, before 5)

| type | decode nb=1 | nb=2 | nb=3 | prefill 512@8.7k |
|---|---|---|---|---|
| f16 | 584 | 1280 | 1839 | 64 995 |
| q8_0 | 978 | 1533 | 2301 | 80 457 |
| q5_1 | 642 | 1230 | 1824 | 80 761 |
| q5_0 | 946 | 1637 | 2295 | 96 656 |
| q4_1 | **538** | **1015** | **1499** | 69 181 |
| q4_0 | 733 | 1193 | 1736 | 76 791 |
| iq4_nl | 767 | 1405 | 2230 | 94 772 |

Hardware paths:
- decode: q4_0/q4_1/q5_0/q5_1/q8_0 K → int8 dot (DP4A);
- prefill: XMX coopmat (after finding 5);
- iq4_nl: LUT, no acceleration.

There is no q6 KV type in llama.cpp.

## Own build — best so far (MTP n=1, q8_0 KV, defer + `-bs`, up to finding 4)

| | ref | own | |
|---|---|---|---|
| story tg | 27.6 | 32.9 | +19% |
| wiki tg | 19.6 | 25.5 | +30% |
| wiki pp | 376 | 386 | +3% (before finding 5) |

## Own build v3 (findings 1–5, MTP n=1, defer + `-bs`) — KV type grid (server)

| KV (target+draft) | story pp | story tg | wiki pp | wiki tg | acceptance story/wiki | peak VRAM |
|---|---|---|---|---|---|---|
| q8_0 | 175 | 32.3 | **604** | 26.7 | 0.59 / 0.72 | 9.5 GB |
| f16 | 171 | 33.1 | **605** | **31.5** | 0.58 / 0.76 | 11.5 GB |
| q4_1 | 181 | 32.8 | 579 | 27.8 | 0.62 / 0.74 | 8.4 GB |
| q4_0 | 166 | 33.0 | 568 | 28.3 | 0.63 / 0.78 | 8.3 GB |
| q5_1 | 170 | 32.7 | 542 | 27.6 | 0.62 / 0.77 | 8.7 GB |
| iq4_nl | 177 | 33.4 | 553 | 27.8 | 0.65 / 0.78 | 8.3 GB |

- Every run is deterministic (2/2 identical hashes). GPU load: story ~82%, wiki ~91%.
- The KV type makes little speed difference: in tg the q4 types are ~+5% over q8_0, in pp −4…−10%.
- **f16 is the fastest at long context** (wiki tg +18% over q8_0), at the cost of +2 GB VRAM.
- The quality cost (KLD) is still to be measured.

## Own build v4 (findings 1–9, q8_0 KV, MTP n=1, defer + `-bs`)

| | ref (brew) | v3 | **v4** | v4 vs ref |
|---|---|---|---|---|
| story pp | 238 | 175 | **294** | +24% |
| story tg | 27.6 | 32.3 | **33.5** | +21% |
| wiki pp | 376 | 604 | **598** | +59% |
| wiki tg | 19.6 | 26.7 | **27.2** | +39% |

- The output is bitwise identical to v3 (story and wiki hashes). GPU load: story 82 → 89%, wiki 93%.
- One MTP step (story): draft 8.2 ms + verify 38.1 ms + other ~1.3 ms = 47.6 ms, 1.6 tokens per step.
- MTP n=2: 31.7 / 26.9, n=3: 28.8 / 23.8 (story / wiki tg) → n=1 stays.

## Accuracy level 1 — KL divergence (wikitext-2, 6×4096 tokens)

Reference: brew, f16 KV (PPL 6.6991 ± 0.151). Every run uses the same 24k tokens, `-fa on`, B50 (`~/llama-bench-results/eval/kld.sh`).

| build / KV | PPL | Mean KLD | 99.9% KLD | Max KLD | RMS Δp | same top-1 |
|---|---|---|---|---|---|---|
| brew q8_0 | 6.6972 | 0.00040 | 0.0113 | 0.52 | 0.62% | 99.03% |
| **own q8_0** | 6.6911 | 0.00044 | 0.0189 | 0.26 | 0.67% | 98.67% |
| own f16 | 6.6928 | 0.00042 | 0.0155 | 0.37 | 0.63% | 98.93% |
| own q5_1 | 6.7028 | 0.00329 | 0.0845 | 12.8 | 1.58% | 98.21% |
| own q4_0 | 6.7102 | 0.00510 | 0.397 | 6.9 | 1.96% | 96.92% |
| own q4_1 | 6.7158 | 0.00510 | 0.266 | 15.1 | 1.97% | 97.06% |
| own iq4_nl | 6.7094 | 0.00584 | 0.260 | 17.9 | 2.00% | 97.34% |

- The own build (XMX coopmat FA, split-k, input staging) is **as accurate as brew**.
  The difference between own f16 and brew f16 (KLD 0.0004) comes from a different floating-point summation order; it is as large as the whole cost of q8_0 KV.
- KV quantization:
  - q8_0 is practically lossless;
  - q5_1 has ~8× and the 4-bit types ~12–14× larger KLD, plus rare but large single deviations (max KLD 7–18). These can steer the output during long generations.
  - Recommendation: q8_0, or f16 if VRAM allows (see the v3 table: f16 is the fastest at long context).

## Accuracy level 2 — multiple choice (llama-perplexity, log-likelihood)

The same task set (fixed seed) for every run (`~/llama-bench-results/eval/mc.sh`).
The server cannot measure this (it does not return prompt-token logprobs), but the compute path (FA, matmul, KV quantization) is the same ggml/Vulkan code.

Note: `llama-perplexity` allocates an n_ctx × n_vocab float buffer. With the 248k vocabulary `-c 2048 -np 16` = 32 GB → OOM.
So tasks with more than 4 answers (TruthfulQA, ARC) run with `-c 256 -np 16`.

| | HellaSwag (400) | Winogrande (600) | TruthfulQA mc1 (300) | ARC-C (299) | MMLU (300) |
|---|---|---|---|---|---|
| brew q8_0 | 78.0 | 72.0 | 32.3 | 50.5 | 42.7 |
| **own q8_0** | 78.0 | 72.3 | 32.3 | 50.5 | 43.0 |
| own q4_1 | 78.5 | 72.2 | 31.7 | 50.2 | 43.7 |

Statistical error ±1.8–2.9 points → no measurable difference.

## Accuracy level 3 — generative, with the real server config (MTP n=1, q8_0 KV, 128k, thinking off)

Tools:
- `~/llama-bench-results/eval/server_eval.sh`: lm-eval 0.4.13, `llama-eval` conda env, local-chat-completions, temp 0.
- `wiki_qa.py`: the full Roman Empire article in context.
  - 25 questions answerable from the article;
  - 5 that the article cannot answer (expected: NOT_IN_TEXT → hallucination test).

| | wiki QA | wiki: does not invent an answer | NQ-open (300, 5-shot) | GSM8K (150, 5-shot) flex/strict | IFEval (150) prompt strict/loose |
|---|---|---|---|---|---|
| brew ref | 25/25 | 5/5 | 23.0 | 86.0 / 83.3 | 85.3 / 89.3 |
| **own v4** | 25/25 | 5/5 | 22.7 | 85.3 / 83.3 | 85.3 / 88.7 |

Paired comparison (same tasks):

| task | identical output | result flips |
|---|---|---|
| NQ-open | 288/300 | 1 |
| GSM8K | 99/150 | 5 (3 for brew, 2 for own) |
| IFEval | 32/150 (long answers) | 6 (3–3) |

- The flips are symmetric: greedy decoding diverges on single near-tie tokens (KLD ~0.0004). There is no quality loss.
- The wiki QA takes 55 s on the own build vs 96 s (faster pp + tg).

Note: NQ-open 0-shot in chat mode scores 0 points, because the model answers in full sentences and exact match fails.
So it runs 5-shot multiturn with a short-answer system prompt.

## Tests

- `test-backend-ops -o FLASH_ATTN_EXT` (B50, v4): 5368/5368 OK.
- `test-llama-context` (`GGML_SCHED_DEBUG_REALLOC=1`, Vulkan) and `test-backend-sampler` (`get_sub_batch`) **fail the same way on the unmodified HEAD**, so the changes do not cause them.

## NVIDIA GTX 1080 Ti (Pascal, Thunderbolt, Vulkan2) — 2026-10-27

Driver: 580.178.04 closed module (RPM Fusion `580xx`). The 615 branch only ships the open module, which cannot handle Pascal without GSP.
Device: fp16: 0, matrix cores: none, int dot: 1, 48 KB shmem, BAR 246 MiB → the small-BAR detection (finding 1) kicks in here too.

Fixes (bugs in the original llama.cpp, they also show up on the unmodified HEAD):

11. **Scalar FA GPU fault (Xid 13, `SKEDCHECK18_L1_CONFIG_TOO_SMALL`) → the process waits forever.**
    - On NVIDIA, K/V shmem staging is always on for hsk < 256. Without fp16 (fp32 FLOAT_TYPE) the tile is 50 KB > 48 KB at hsk=192.
    - The size check does not even run for decode, and it only halves block_rows (the tile depends on Bc).
    - Fix: if it does not fit, staging is switched off (it reads from global memory). Unchanged on fp16 devices.
12. **block_rows halved from 1 to 0** (n_rows=1, n_kv ≥ 1024, does not fit) → `wg_denoms > 0` assert. Now it only halves when > 1.
13. `GGML_VK_PERF_LOGGER` + input staging (finding 9): assert, because compute_ctx already contains the async uploads.
    Instead of the assert, the first timestamp now goes in after the uploads.

`test-backend-ops` (full, 1080 Ti): 21161/21161 OK.

| | story pp | story tg | wiki pp | wiki tg | acceptance |
|---|---|---|---|---|---|
| brew, no MTP | 137 | 42.3 | 275 | 32.8 | – |
| brew, MTP n=1 (ref) | 180 | 39.6 | 253 | 32.1 | 0.57 |
| **own v4** (defer + `-bs`) | 163 | **42.7** | 256 | **38.2** | 0.60 |

- GPU 97% busy, average 178 W (max 278 W), max 78 °C, 1911 MHz.
- Pascal is compute-bound: with MTP brew is slower than without it. The own build's wiki tg is +19% over the ref.
- llama-bench: pp512 374, pp512@d8192 291, tg64 44.9.
- pp512 breakdown: 98% is MUL_MAT (MMQ, q8_1 + DP4A) at q4_K 6.7 and q5_K/q6_K 4.3–5.0 TOPS, a fraction of the int8 peak (~45 TOPS).

14. **Int MMQ tile on NVIDIA without tensor cores.**
    - The default `l` tile has 128 threads with 64–128 accumulators each → few active warps.
    - After a sweep: 512 threads, WM=WN=32, WMITER=2, TM=2, TN=1 (`l_warptile_mmq_int(_k)`, only for `NVIDIA && !coopmat_support`).
    - MUL_MAT 4096×512×14336: q4_K 11.1 → 5.6 ms, q8_0 12.1 → 7.3, q5_K 14.0 → 6.6, q6_K 13.6 → 6.7 ms. MUL_MAT test 2275/2275 OK.
15. **Scalar FA prefill without fp16.**
    - With fp32 FLOAT_TYPE the Br×Bc=16×32 tile spills registers.
    - Fix: for `!fp16 && n_rows > 8`, use a 16×16 tile, row_split 4, 128 threads (if the shmem fits).
    - 512 q × 8.7k keys: hsk=256 f16 48.5 → 21.4 ms, q8_0 56 → 25.5; hsk=128 q8_0 39 → 28; hsk=64 q8_0 37 → 15.5 ms.
    - In decode (1 row) Bc does not matter (64/32/16 ±3%), so it is unchanged.
    - The int8 MMQ FA path (without dequant) is slower here (59 vs 48 ms), so dequant stays.
    - FA test 5368/5368 OK.

llama-bench v5: pp512 **656** (374), pp512@d8192 **536** (291), pp512@d16384 449, tg64 45.0 / 42.2 / 38.9.

| 1080 Ti | story pp | story tg | wiki pp | wiki tg | acceptance |
|---|---|---|---|---|---|
| brew, MTP n=1 (ref) | 180 | 39.6 | 253 | 32.1 | 0.57 / 0.73 |
| own v4 | 163 | 42.7 | 256 | 38.2 | 0.60 / 0.73 |
| **own v5** (findings 14–15) | **307** | **43.7** | **467** | **38.4** | 0.62 / 0.73 |
| v5 vs ref | +71% | +10% | +85% | +20% | |

- story pp: the value of the 2nd run; the 1st run gave 104 because of warm-up.
- GPU 95%, average 232 W, max 295 W, max 79 °C.
- KLD against the B50 brew f16 reference, measured on the 1080 Ti: own v5 q8_0 0.0022, brew q8_0 0.0026 → same accuracy.
  The ~0.002 comes from the two devices' different compute paths: int8 MMQ vs coopmat.

## CUDA — GTX 1080 Ti (sm_61)

Toolchain: conda env `llama-cuda`:
- CUDA 12.9: version 13 no longer supports sm_61.
- GCC 14: nvcc 12 does not accept the system GCC 16.

Build: `build-cuda`, `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="61;75;86"` (1080 Ti, 2080 Ti/T1200, 3090).
Comparison: upstream `de5122d87` CUDA build. `test-backend-ops` (CUDA0): 19051/19051 OK.

llama-bench (q8_0 KV), upstream / own: pp512 980 / 966, pp512@d8192 778 / 764, tg64 41.0 / 41.0.
1–4 token batches (own): CUDA pp1 40.7, pp2 76, pp3 106, pp4 127 t/s vs Vulkan 45, 69, 83, 92 → the CUDA verify batch is cheap.

16. **The user's command + MTP on CUDA: `-fit` only puts 31/34 layers on the GPU** (34/34 without MTP or on Vulkan).
    - Cause: the CUDA FA allocates the f16 copy of the quantized K/V in the compute buffer (`ggml_cuda_flash_attn_ext_get_alloc_size`, for the TILE/MMA kernels), for the whole context: ~0.5 GB per context at 128k.
    - Compute buffer CUDA vs Vulkan: main 689 vs 229 MiB, MTP 681 vs 171 MiB.
    - With the default 1 GB fit margin, 3 layers (weights + recurrent state) go to the CPU → CPU splits, 13 spinning CPU threads, story tg 34 (40 without MTP).
    - It actually fits: full offload is 10.46 / 11.26 GB.
    - `-fitt 512` → 34/34; `-ctkd f16 -ctvd f16` → MTP compute 681 → 169 MiB, but only 33/34.
17. **Fix: chunked K/V f16 conversion in the CUDA FA** (`fattn-common.cuh`, `fattn.cu`).
    - How it works:
      - The TILE and VEC kernels (parallel_blocks, no stream-k) convert and process the quantized KV in 64 MiB chunks (Qwen3.5-9B: 16384 rows).
      - Every chunk runs with ≥2 parallel blocks.
      - A new `flash_attn_combine_kv_chunk` kernel merges the partial results (VKQ, KQ max/sum) into the running result in dst; the last chunk normalizes.
      - The sinks only apply to the first chunk, and the mask is shifted per chunk.
    - Result:
      - The compute buffer only reserves one chunk of f16 space: CUDA0 compute 689 → 253 MiB.
      - The fit projects 8284 MiB against 11017 MiB free → **the user's command gets 34/34 unchanged**.
    - The MMA (Turing+, stream-k) and sparse paths are unchanged (still to do for the 2080 Ti/3090).
    - `GGML_CUDA_FA_KV_CHUNK_MB` overrides the chunk size (0 = off).
    - Correctness: `test-backend-ops` full CUDA 19051/19051; FLASH_ATTN_EXT with 256-row chunks (`GGML_CUDA_FA_KV_CHUNK_MB=0.0001`) 4040/4040.
    - Cost (FA prefill, kv=8704, nb=512, q8_0, us/run; no chunks / 8 MiB / 2 MiB):
      - hsk256 16612 / 17981 / 22013;
      - hsk128 14793 / 17247 / 22699.
      That is ~0.3 ms fixed cost per chunk; with 64 MiB chunks (≥16k rows) ~1%. No change with f16 KV (no conversion).

| CUDA 1080 Ti, MTP n=1, `-fitt 512` | story pp | story tg | wiki pp | wiki tg |
|---|---|---|---|---|
| upstream | 472 | 49.7 | 667 | 42.3 |
| own (defer + `-bs`) | 499 | 51.9 | 662 | 44.2 |
| own + chunked FA, default fit (34/34, without `-fitt`) | 501 | 52.1 | 663 | 43.5 |
| + Pascal mmvq (18) | 505 | 53.7 | 666 | 44.5 |
| **+ Pascal FA GQA (19), final** | **506** | **53.2** | **677** | **45.1** |
| (default fit, 31/34 layers) upstream / own | 231 / 220 | 31.7 / 34.0 | 507 / 505 | 25.1 / 26.0 |
| (no MTP, upstream) | 525 | 40.0 | 726 | 35.9 |

18. **Pascal mmvq parameters** (`mmvq.cu`, new `MMVQ_PARAMETERS_PASCAL` table, sm_60–69).
    - Settings: 4 warps / 2 rows per block for ncols_dst=1; 4 rows for 2–4 columns (MTP verify).
    - llama-bench tg128 41.1 → 42.1, pp2 76.8 → 77.9.
    - In an isolated kernel 4 rows per block is best (q6_K −9%), but on the fused gate/up path it is slower (tg 40.5), hence 2.
    - The L2 prefetch (after the DGX Spark pattern) is +15..+70% slower on Pascal.
    - `-DGGML_CUDA_MMVQ_PASCAL_{NW1,RPB1,RPB_SMALL}` overrides it.
19. **Pascal FA: GQA optimization at a GQA ratio of 4 in prefill too** (`fattn-tile.cuh`).
    - Until now it was off on Pascal for gqa ≤ 4 and more than 16 Q rows.
    - Measured: every [4,1] perf case −5..−10% (e.g. hsk256 kv8704 nb512 q8_0 16.3 → 15.1 ms); the rest unchanged.
    - gqa ≤ 2 and pre-Pascal keep the old behaviour.
    - pp2048@d16384 625 → 644 (upstream: 625).
    - KLD (wikitext, q8_0 KV) is bitwise identical to upstream CUDA. On short server prompts the KV split differs, so the greedy output may diverge.
20. **Under CUDA the GeForce runs in the P2 state: memory 5005 MHz vs Vulkan P0 5508 MHz** (−10% bandwidth).
    - This is most of the Vulkan 45 vs CUDA 42 tg gap; the kernels' efficiency relative to peak is similar (q4_K 12288×4096: CUDA 61%, Vulkan 66%).
    - No override on Pascal:
      - `nvidia-smi --cuda-clocks` / `-lmc` are Volta+ only;
      - tried as root (driver 580.178): `-lgc 1936,1936`, `-lmc 5505,5505` and `-ac 5505,1936` are all "not supported";
      - with `-pm 1` it still stays in P2 under load: SM 1911 MHz, memory 5005 MHz (tg256 42.8).
    - The Coolbits memory offset of `nvidia-settings` needs an X server (here: Wayland + headless eGPU), and it applies to the top P-state, not to P2.
    - For generation without MTP, Vulkan is the faster backend on Pascal.
      With MTP CUDA still wins, because the 2–3 token verify is cheap (pp2 78 vs Vulkan 69).
    - MTP draft length: n=1 is best (n=2: story 52.4 / wiki 44.8; n=3: 43.7 / 42.7).
    - `-ub 1024/2048` does not speed up prefill (945 / 944 / 928).

The outputs are bitwise identical (story/wiki sha upstream = own).
On Pascal CUDA is the faster backend (Vulkan v5: 307 / 43.7 / 467 / 38.4).

## NVIDIA RTX 2080 Ti (Turing, sm_75, Thunderbolt) — 2026-10-29

Same command, MTP n=1, defer + `-bs`.
- Vulkan: `-dev Vulkan2` (NV_coopmat2).
- CUDA: `CUDA_VISIBLE_DEVICES=1 -dev CUDA0`. Under CUDA it runs in P2 with 6800 MHz memory (Vulkan: P0 / 7000 MHz), only −3%.

| 2080 Ti | story pp | story tg | wiki pp | wiki tg |
|---|---|---|---|---|
| brew Vulkan, no MTP | 360 | 61.6 | 2061 | 54.6 |
| brew Vulkan, MTP n=1 (ref) | 562 | 69.2 ¹ | 1796 | 53.6 |
| own Vulkan v5 | 341 | 64.2 | 1827 | 58.8 |
| **own Vulkan (22–23)** | **690** | **74.1** | **1821** | **65.4** |
| upstream CUDA, no MTP | 1018 | 74.7 | 1949 | 65.5 |
| upstream CUDA, MTP n=1 (29/34 layers) | 187 | 19.3 | 723 | 16.6 |
| own CUDA + MMA chunking (34/34) | 973 | 95.3 | 1786 | 61.6 |
| **own CUDA + vec verify (21)** | **910–939** | **96.5–97.2** | **1773–1794** | **83.9–84.2** |

¹ In the story scenario brew generates different text (acceptance 0.91, repetitive text), so its story tg is not comparable.
In the quick test (acceptance 0.56 vs 0.62) brew gives 56.8 vs own 61.9.

21. **CUDA chunked FA on the MMA (stream-k) path too** (`fattn-mma-f16.cuh`, `fattn-common.cuh`, `fattn.cu`).
    - In chunk mode the kernel runs one block per tile.
      It writes the unnormalized tile + the KQ max/rowsum pair through the existing `needs_fixup` path.
    - `kv_chunk_mode` = meta buffer present + gridDim.x == number of tiles, which never happens in a normal launch.
    - The new `flash_attn_combine_kv_chunk_tiles` merges the chunks. The sparse variant (op_params[4] > 0) does not chunk.
    - 2080 Ti: target compute buffer 964 → 253 MiB, draft 681 → 245 MiB, 29/34 → 34/34 layers.
    - FA 4376/4376 (normal and with 256-row chunks).
    - ~0.1 ms per chunk (kv=8704, 256-row chunks: 2.99 → 6.78 ms; negligible at 64 MiB).
    - **Quantized KV + 2-token verify uses the vec kernel on Turing/Ampere too** (until now Ada+ only).
      The MMA path converts the whole KV to f16 on every call.
      FA kv=20736 nb=2 q8_0: 1266 → 279 us; pp2@d20480 83 → 120 t/s.
22. **CUDA FA iq4_nl K/V**: it was not supported until now (FA fell back to the CPU).
    - New: `vec_dot_fattn_vec_KQ_iq4_nl`, `dequantize_V_iq4_nl`, the f16 conversion (`dequantize_iq4_nl_pair`) and vec instances.
    - The default of `GGML_CUDA_FA_QUANTS` now contains every same-type KV pair (q4_0, q4_1, q5_0, q5_1, iq4_nl, q8_0, f16, bf16).
      Before, q4_1/q5_x got a slow f16 fallback.
    - FA (kv=20736, us, nb=1 / 2): f16 159/161, q8_0 219/287, q4_0 227/314, q4_1 203/311, q5_0 344/421, q5_1 313/407, iq4_nl 415/517.
    - (q6_0 is not an existing KV type.)
23. **Vulkan, NVIDIA Turing+ mat-vec**:
    - With 2–3 columns (verify) the fp K-quant shader with 4 rows is faster than MMVQ (`kq_batch_rm` = 4 by default).
      n=2: q4_K 12288×4096 108 → 76 us, q5_K 121 → 87, q6_K 248320×4096 1620 → 1476.
    - With 1 column and k > 4096 the fp shader is used too (q4_K 4096×12288 MMVQ 70 → 60 us).
    - **q4_K/q5_K shader** (vendor-neutral): the B sums of the min term are computed once instead of per row.
      q4_K n=2 −6..−10%, n=3 −5..−8%, n=1 −2..−3%; q5_K −1..−3%.
    - llama-bench pp2 101 → 129–135, tg128 68.6 → 70.4–70.8.
    - Vulkan FA: nb=2 ≈ 2× nb=1, because the GQA merge only merged the heads; separate workgroups read the whole KV for each token.
      kv=20736 q8_0 215 vs 418 us. → finding 24.
    - Vulkan pp2@d20480 GPU time is 21 ms: mat-vec ~12.9 ms, FA 3.3 ms, small ops (ADD/CPY/RMS_NORM/SCALE, 5–8 us each) ~4 ms.

24. **Vulkan FA: tokens × GQA heads in one workgroup** (`flash_attn.comp`, `ggml_vk_flash_attn`).
    - In the scalar shader, row r maps to token r / gqa and head r % gqa.
    - A workgroup handles gqa_T tokens (gqa_T·gqa ≤ the block's rows), so the tokens read the KV once.
    - Q, the mask, the output and the split_k partial results use per-row addressing; gqa_T = 1 keeps the old behaviour.
    - On by default (`GGML_VK_FA_GQA_TOKENS`, 1 = off), except coopmat1 (not measured) and sparse.
    - kv=20736, 2 tokens, us: f16 395 → 219, q4_0 415 → 245, q8_0 409 → 305, q4_1 462 → 302, q5_0 511 → 372, q5_1 492 → 473, iq4_nl 619 → 583.
      With 3 tokens iq4_nl goes 872 → 947; the others get faster.
    - FA 5368/5368, full Vulkan 21371/21371.
    - Server (MTP n=1): q8_0 story 76.3 → 79.2, wiki 66.4 → 73.4; q4_0 wiki 68.4 → 72.9.

KV types on the server (2080 Ti, MTP n=1, after findings 21–24; t/s story / wiki tg):

| KV | CUDA | Vulkan (before 24) |
|---|---|---|
| q8_0 | 97.9 / 85.3 | 76.3 / 66.4 |
| q4_0 | 95.6 / 84.7 | 74.4 / 68.4 |
| q4_1 | 96.3 / 84.3 | 79.1 / 67.2 |
| iq4_nl | 94.7 / 75.5 | 75.6 / 64.6 |
| q5_0 | 95.2 / 76.0 | 75.3 / 63.6 |
| q5_1 | 95.4 / 79.0 | 77.4 / 67.1 |
| f16 | 11 / 9 (does not fit: fit puts layers on the CPU) | did not start |

On Vulkan the first run with a new KV type spends ~5 s compiling shaders (story pp ~20 t/s); after that the driver loads them from its cache.
q6_0 is not an existing KV type.

## Accuracy level 4: repetition loops (infinite token repetition), 2026-10-29

**Why:** the user often got infinite repetition on long Hungarian stories, on both Vulkan and CUDA. Two patterns:
- "épp csak épp csak …";
- paragraph-sized blocks repeated over and over.

The question was whether the speedups (FA chunking, vec verify, Vulkan GQA merge, MTP defer, `-bs`) hurt the numerics or the sampling.

**What was done:** a dedicated test (`~/llama-bench-results/loop-test/`).
- `loop-test.py` starts the server with a given config, 128k context, q8_0 KV.
- It sends the user's two prompts with streaming, fixed seeds (1–6), up to 12000 tokens, with default sampling.
- It detects short-period loops on the fly and aborts the generation.
- `loop-score.py` scores afterwards: a run loops if the last 200 characters of the output occur at least 3 times in the last 12000 characters. This also catches paragraph-sized (~1100 character) periods.
- `loop-run.sh` runs the configs: cu-full, cu-mtp, cu-nomtp, up-nomtp, up-mtp, vk-full.

Prompts (Hungarian, verbatim from the user):
- p0: "tervezz meg egy 10000 szavas gyerekenek szóló mesét, amit később meg kell írnod" (plan a 10000-word children's story that you will write later)
- p1: "csinálj 10000 szavas mesét avgy hosszabbat" (make a story of 10000 words or longer)

**Result (2080 Ti, CUDA, loops / 6 runs):**

| config | sampling | p0 | p1 | p1 tg (t/s) |
|---|---|---|---|---|
| own CUDA, MTP n=1, defer + `-bs` | default (temp 0.8, top_k 40, top_p 0.95, min_p 0.05, no penalties) | 0/6 | 4/6 | ~99–106 |
| **unmodified upstream CUDA, no MTP** | default | 0/6 | **4/6** | ~72 |
| own CUDA, MTP n=1, defer + `-bs` | Qwen recommendation: `--temp 1.0 --top-k 20 --top-p 0.95 --min-p 0 --presence-penalty 1.5` | – | **0/6** | ~82 |

- The changes do not cause the repetition: the unmodified upstream without MTP loops just as often.
- It is the behaviour of the 9B Q4 model on long Hungarian text with penalty-free sampling.
  The upstream loops look just like the user's examples, e.g. "A toll a toll volt. A toll a toll volt. …".
- During loops the draft acceptance is high (~0.87 instead of the usual ~0.7), so tg is higher too (~105 t/s).
  This can distort bench numbers (see the ¹ note on the brew story result).
- Presence penalty 1.5 removes the loops and the text stays sensible. It also runs on the GPU with `-bs` (the penalties sampler supports the backend).
- Cost: the higher temperature and the penalty drop acceptance to ~0.6, and tg goes ~100 → ~82 t/s. Still faster than upstream without MTP (~72).
- **Recommendation:** add `--temp 1.0 --top-k 20 --top-p 0.95 --min-p 0 --presence-penalty 1.5` to the server command.
  If the web UI has its own sampling values set, set Presence penalty to 1.5 there.

**Speed with the loop-free sampling** (2080 Ti, 2026-10-29).
Measured with `bench.py` and `BENCH_SAMPLING=server`, using the Qwen recommendation above; median of 3 runs, seed = 1 + run, thinking off. t/s.

| | story pp | story tg | wiki pp | wiki tg |
|---|---|---|---|---|
| brew Vulkan, no MTP | 696 | 59.8 | 2055 | 54.6 |
| brew Vulkan, MTP n=1 | 582 | 52.0 | 1801 | 49.8 |
| original code, own Vulkan build, no MTP | 718 | 59.0 | 2123 | 54.2 |
| original code, own Vulkan build, MTP n=1 | 591 | 52.0 | 1856 | 51.1 |
| original code, CUDA, no MTP | 1035 | 70.6 | 1969 | 63.1 |
| original code, CUDA, MTP n=1 (does not fit, puts layers on the CPU) | 208 | 18.4 | 737 | 16.6 |
| own Vulkan, no MTP | 767 | **63.8** | 2062 | 57.5 |
| own Vulkan, MTP n=1, defer + `-bs` | 683 | 61.7 | 1821 | **58.4** |
| own CUDA, no MTP | 1100 | 70.6 | 1983 | 62.9 |
| **own CUDA, MTP n=1, defer + `-bs`** | 1034 | **77.6** | 1798 | **71.2** |

- In the original code MTP slows things down with this sampling (Vulkan: 59 → 52 tg).
- In the own build:
  - On CUDA MTP helps: +10% story, +13% wiki compared to running without MTP.
  - On Vulkan MTP gives no speedup: 61.7 / 58.4 with MTP, 63.8 / 57.5 without. pp is higher without MTP too, so on Vulkan with this sampling running without MTP is recommended.
- Without MTP:
  - On CUDA the own and the original code are identical (70.6 / 62.9 vs 70.6 / 63.1), because the CUDA changes target the MTP verify step.
  - On Vulkan the own build gives +8% story and +6% wiki tg, thanks to the Turing mat-vec settings (finding 23).
- wiki pp is ~10% lower with MTP than without, because the draft head also processes the prompt. The original code behaves the same (2123 → 1856).
- Compared with the greedy measurements (above: CUDA 97/85, Vulkan 79/73), tg is ~20% lower because the draft acceptance is ~0.6.

### Intel Arc Pro B50 with the loop-free sampling (evening of 2026-10-29)

The B50 is back in the eGPU slot (Vulkan2). The only CUDA device left is the laptop's T1200 (4 GB), which cannot hold the
~6 GB model, so there is no CUDA measurement.

Setup:
- Same protocol as on the 2080 Ti: `bench.py`, `BENCH_SAMPLING=server`, Qwen-recommended sampling, median of 3, thinking off.
- "Original code" = `master` at the fork's merge base (`63b4b303b`), built with Vulkan.
- "Workaround" = `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1`. It is not needed by the own build (finding 1).

| B50, t/s | story pp | story tg | wiki pp | wiki tg | peak VRAM |
|---|---|---|---|---|---|
| brew, no MTP, as is | – | – | **crash** ² | **crash** ² | – |
| brew, no MTP, workaround | 345 | 27.0 | 424 | 19.1 | 8.8 GB |
| brew, MTP n=1, workaround | 301 | 29.7 | 380 | 20.0 | 9.6 GB |
| original code, no MTP, as is | 362 | 27.1 | **crash** ² | **crash** ² | 8.8 GB |
| original code, MTP n=1, as is | 83 | **3.1** | 155 | **3.2** | 9.3 GB |
| original code, no MTP, workaround | 351 | 27.2 | 438 | 19.1 | 8.8 GB |
| original code, MTP n=1, workaround | 321 | 29.4 | 391 | 19.7 | 9.6 GB |
| own, no MTP | 356 | 26.5 | **679** | 22.1 | 8.8 GB |
| **own, MTP n=1, defer + `-bs`** | 334 | **29.4** | 612 | **24.9** | 9.5 GB |

² **GPU job timeout → `ErrorDeviceLost`, the server aborts.**
- Symptoms:
  - It happens on the 20k-token wiki prompt, without MTP.
  - The kernel log shows `xe ... GT0: Timedout job ... in llama-server`, then `vk::DeviceLostError` from `waitForFences`.
  - It hits at the last 2048-token chunk of the prompt (brew: at 18–20k, original code: at 4–6k).
- Cause:
  - Without the workaround, the original code keeps the compute buffers in host-visible memory.
  - On a small-BAR eGPU that is mostly system RAM behind Thunderbolt (finding 1).
  - One prefill submit, the FA over the full context, then runs longer than the xe driver's job timeout, and the kernel kills the context.
- With MTP the original code does not crash, but it is ~9× slower (3 t/s) for the same reason.
- Both disappear with the workaround, and the own build is not affected (finding 1 detects small BAR and uses device-local memory).
- Raising the xe job timeout (sysfs, root) would only hide the symptom; the slowness stays.

Conclusions:
- **Long context:** the own build is clearly faster.
  - With MTP: wiki tg +24…+26% and wiki pp +57…+61% over brew/original with the workaround.
  - Without MTP: wiki tg +16%, pp +55…+60%.
  - This comes from the B50 FA fixes (findings 4–5) and the input staging (finding 9).
- **Short prompt (story):** own, brew and original code are on par (29.4–29.7 tg with MTP, 26.5–27.2 without).
  - With this sampling the draft acceptance is ~0.6, so the MTP gain on short context is only ~+10% everywhere.
  - Under greedy sampling (v4 table above) the own build was +21% on story tg; that gain came from the accepted drafts.
- **MTP pays off on the B50 with this sampling too** (own: story +11%, wiki +12% tg over no MTP), unlike on the 2080 Ti Vulkan.
  It costs ~10% wiki pp (the draft head processes the prompt as well).
- **As-is user experience:** the original code / brew without the workaround either crashes on long prompts or runs at 3 t/s with MTP. The own build needs no env var.

Recommended B50 command (loop-free sampling):
```bash
LLAMA_MTP_DEFER_CATCHUP=1 ./build-vk/bin/llama-server \
  -hf unsloth/Qwen3.5-9B-MTP-GGUF:UD-Q4_K_XL -c 128000 -fa on --parallel 1 -t 16 -tb 16 \
  -dev Vulkan2 --port 8080 -ctk q8_0 -ctv q8_0 \
  -ctkd q8_0 -ctvd q8_0 --spec-type draft-mtp --spec-draft-n-max 1 -bs \
  --temp 1.0 --top-k 20 --top-p 0.95 --min-p 0 --presence-penalty 1.5
```

### Upstream merge check on the B50 (2026-10-29 night)

`origin/master` (79 upstream commits, through `418f905a4`) merged into `speedup` without conflicts (merge `15acaa296`;
pre-merge state on branch `speedup-pre-merge-20261029`). Results on the B50:

| check | before merge | after merge |
|---|---|---|
| test-backend-ops (Vulkan0 = B50, full) | – | 21632/21632 |
| own MTP, loop-free sampling: story pp / tg, wiki pp / tg | 334 / 29.4, 612 / 24.9 | 337 / 29.5, 612 / 24.9 |
| own no MTP: story pp / tg, wiki pp / tg | 356 / 26.5, 679 / 22.1 | 356 / 26.5, 679 / 22.1 |
| KLD q8_0 (mean / same top-1 / PPL) | 0.00044 / 98.67% / 6.6911 | 0.000443 / 98.665% / 6.6911 |
| wiki QA (answerable / abstained) | 25/25, 5/5 | 25/25, 5/5 |
| loop test p1 (6 seeds), default sampling | – | 5/6 not ok |
| loop test p1 (6 seeds), Qwen sampling | – | 0/6 |

The loop scorer (`loop-score.py`) was fixed:
- Runs that never leave the thinking phase were missed, because their tail contained the empty-content separator.
- Runs aborted on the fly were not counted.
- It now scores reasoning and content separately and also reports RUNAWAY runs (token limit without exact repetition).

With default sampling, 3 of the 5 B50 failures loop inside the thinking phase ("Wait, I will ensure the story is …").

### B50 profile (2026-10-30, q8_0 KV, MTP n=1, greedy)

Tools: `GGML_VK_PERF_LOGGER=1` (per-op GPU time) + `LLAMA_SERVER_TIMING=1`.
The logger does not flush exactly once per graph, so the decode numbers are the totals over the whole generation divided by the number of MTP steps.

Story (short context): wall step 55.7 ms = draft 8.0 + verify 47.1 ms; GPU time 42.9 ms per step.

| op (per step) | ms | share | effective bandwidth |
|---|---|---|---|
| weight mat-vecs n=2 (all) | ~31 | ~72% | q4_K ~185 GB/s, q6_K ~170, q5_K ~167, **q8_0 ~128** |
| LM head q6_K 248320×4096, verify (n=2) + draft (n=1) | 4.6 + 4.5 | 21% | ~185 GB/s |
| small ops (RMS_NORM, ADD, CPY, GDN, …) | ~3 | ~7% | |

Wiki (20k context): GPU 59 ms per step. **FA of the 2-token verify: 12.4 ms per step (21%)**, 1.64 ms per call, against ~0.2 ms to read the q8_0 K/V once at 224 GB/s.
In the 20k prefill, FA is ~39% and the q4_K GEMM ~27% of the GPU time.

~~Not GPU work: the verify decode takes 47 ms wall against ~37 ms of GPU.~~
**Correction:** that gap was the perf logger's own cost (per-op timestamps).
- Without the logger the story step is 46.4 ms = verify 38.0 + draft 7.6 + ~0.8 ms other, and the verify wall time equals its GPU time.
- The step is GPU-bound.

The q8_0 mat-vec "57% of bandwidth" is a cold-read effect of the small 4096×4096 matrices in the real run:
- the MMVQ kernel itself streams 4096×8192 q8_0 at ~200 GB/s;
- on the B50 MMVQ beats the fp shader for every shape and n (by 25–74%);
- more rows per workgroup (`GGML_VK_KQ_BATCH_RM=4`) is 24–48% slower.

So the mat-vec defaults stay.

Order of attack (plan-b50-kv.md):
1. decode/verify FA per KV type (finding 25);
2. prefill FA (finding 26);
3. the draft LM head (4.5 ms of the 7.6 ms draft; needs a reduced-vocabulary head, larger project).

### 25. B50 decode/verify FA per KV type (2026-10-30)

Two findings on Intel Xe2:
- With GQA the decode/verify FA has n_rows = 4 (the 4 heads of a KV group). So it does not take the "scalar for one row" shortcut and runs on **coopmat1 with a 16-row tile** (75% of the tile idle).
- Every path is far from bandwidth: q8_0, 1 token, kv=20736 takes 731 us for ~45 MB of K/V (~60 GB/s of 224).

Experiments (`test-backend-ops perf`, kv=20736, head 256, GQA 4):
- Enabling subgroup ops for the scalar shader on Intel: ±1%, no effect.
- Token merge (finding 24) on Intel: helps q5_1, but q8_0 2 tokens go 1.4 → 3.4 ms. Kept off on Intel.
- Split sweep (workgroups per core 4–64, minimum keys per split 128/512): shorter splits help every type.
  coopmat1 also wants 32 workgroups per core.
- The scalar shader instead of coopmat1 helps K types f16, q4_0, q4_1 and q5_1. It hurts q8_0, q5_0 and iq4_nl.

Change (`ggml-vulkan.cpp`, Intel Xe2 only; the other vendors are unchanged):
- for n_rows ≤ 4 and K in {f16, q4_0, q4_1, q5_1}: scalar path instead of coopmat1;
- split: at least 128 keys per split (was 256), and 32 workgroups per core on coopmat1 (was 16);
- token merge off on Intel.

The env overrides `GGML_VK_FA_WG_PER_CORE` / `GGML_VK_FA_MIN_KV_SPLIT` still apply.

| FA us (kv=20736), 1 / 2 / 3 tokens | before | after |
|---|---|---|
| f16 | 702 / 1312 / 1793 | 588 / 1007 / 1399 (−16…−23%) |
| q8_0 | 731 / 1361 / 2021 | 660 / 1200 / 1777 (−10…−12%) |
| q4_0 | 711 / 1333 / 1937 | 677 / 1134 / 1553 (−5…−20%) |
| q4_1 | 682 / 1278 / 1846 | 538 / 908 / 1279 (−21…−31%) |
| q5_0 | 870 / 1666 / 2394 | 784 / 1494 / 2183 (−9…−10%) |
| q5_1 | 773 / 1466 / 2104 | 650 / 1185 / 1836 (−13…−19%) |
| iq4_nl | 741 / 1395 / 2122 | 663 / 1231 / 1807 (−11…−15%) |

Prefill FA (512 × 8704) is unchanged.

Server (B50, MTP n=1, greedy, story + wiki, 2 runs):
- The outputs are not bit-identical to before: the new split changes the summation order, and greedy decoding diverges at near-tie tokens.
  So the draft acceptance moves by up to ±0.06, which dominates raw tg. Normalized to MTP steps per second (tg / (1 + acceptance)), wiki (20k):

| KV | wiki tg before → after | acceptance before → after | steps/s |
|---|---|---|---|
| q8_0 | 27.5 → 28.3 | 0.71 → 0.74 | +1.1% |
| q4_0 | 29.3 → 28.4 | 0.78 → 0.72 | +0.3% |
| q4_1 | 28.9 → 29.9 | 0.73 → 0.71 | +4.8% |
| q5_0 | 26.8 → 27.6 | 0.73 → 0.76 | +1.0% |
| q5_1 | 28.7 → 29.3 | 0.77 → 0.71 | +5.7% |
| iq4_nl | 28.9 → 28.6 | 0.78 → 0.74 | +1.4% |

- Story (short context) is unchanged within noise (pp 335–341, tg 33.2–34.9 for all types).
- The gain grows with context length: FA is 21% of the step at 20k.
- Tests: FLASH_ATTN_EXT on the B50 passes. KLD is unchanged (q8_0 0.000443, q4_1 0.0051): perplexity runs the prefill path, which did not change.

KV types on the B50 after this change (MTP n=1, greedy, t/s):

| KV | story pp / tg | wiki pp / tg |
|---|---|---|
| q8_0 | 337 / 34.2 | 612 / 28.3 |
| q4_0 | 340 / 34.9 | 573 / 28.4 |
| q4_1 | 340 / 34.6 | 585 / 29.9 |
| q5_0 | 339 / 33.2 | 521 / 27.6 |
| q5_1 | 339 / 34.3 | 559 / 29.3 |
| iq4_nl | 335 / 34.7 | 564 / 28.6 |

Every requested type was already implemented on Vulkan (no CPU fallback in the graph). wiki pp differs by type (q5_0 521 vs q8_0 612): that is the prefill FA dequant, the next item.

### 26. Prefill FA: dequant + transpose to an f16 scratch for every KV type (2026-10-30)

For batches of ≥ 64 queries with a dense quantized KV cache, coopmat1 FA can dequantize K/V once into an f16 scratch and then run the f16 kernel (upstream #25494).
- That shader (`dequant_q8_0_transpose`) only existed for q8_0.
- The other types dequantized inside the FA inner loop on every tile. For example, 512 × 8704, head 256, q5_0 took 35.6 ms vs ~21.6 ms with f16.

Change:
- `dequant_head.glsl` gets `dequant_block_dst()`: the scratch index of a 32-element block, transposed under `DEQUANT_TRANSPOSE`.
- q4_0, q4_1, q5_0, q5_1, iq4_nl and q8_0 use it.
- The generator builds `dequant_<type>_transpose` for all six, and the device creates the pipelines.
- The selection logic (`use_dequant_kv`) is unchanged, so it applies to every vendor where q8_0 already used it (not coopmat2, not Intel Xe1).
- New test cases: dense, permuted quantized KV, 75 queries, head 64/128/256, all six types. The existing FA tests used non-dense KV views, so they never reached this path. B50: 27/27 pass, `dequant_kv=1` confirmed for every type.

Server (B50, MTP n=1, greedy, wiki 20k prompt):

| KV | wiki pp before → after |
|---|---|
| q5_0 | 521 → 608 (+16.7%) |
| q5_1 | 559 → 609 (+8.9%) |
| iq4_nl | 564 → 610 (+8.2%) |
| q4_0 | 573 → 607 (+6.0%) |
| q4_1 | 585 → 607 (+3.8%) |
| q8_0 | 612 → 612 (already used it) |

- Story (short prompt) and tg are unchanged.
- KLD is unchanged: q4_0 0.00510, q4_1 0.00510, q5_0 0.00269, q5_1 0.00329, iq4_nl 0.00584. The in-shader dequant also produced f16, so the values are identical.

### Validation after findings 25–26 (B50, 2026-10-30)

| check | result |
|---|---|
| test-backend-ops (B50, full, incl. the new dense-KV FA cases) | 21650/21650 |
| wiki QA (answerable / abstained) | 25/25, 5/5 |
| loop-free sampling, q8_0, MTP: story pp / tg, wiki pp / tg | 339 / 29.7, 612 / 25.4 (was 337 / 29.5, 612 / 24.9) |
| loop-free sampling, q8_0, no MTP | 356 / 26.3, 680 / 22.4 |
| loop test p1, default sampling (6 seeds) | 6/6 not ok (before: 5/6) |
| loop test p1, Qwen sampling (12 seeds) | 2/12 not ok (before: 0/6) |
| loop test p1, Qwen sampling + DRY (6 seeds) | 0/6 exact loops, 1 paraphrase runaway |

Notes:
- **The two Qwen-sampling loops** repeat whole paragraphs with a period of ~1500 characters (~400 tokens).
  - The presence penalty only covers `--repeat-last-n` (default 64 tokens), so it cannot see such a cycle.
  - Whether one forms depends on the trajectory. The FA change only moves the summation order (for q8_0 it only changes the split length), so it gives a different trajectory but not a systematic change.
  - Seeds 7–12 had 0/6.
- **DRY** (`--dry-multiplier 0.8 --dry-penalty-last-n 16384`):
  - It removes the exact repetition. One run still ended in a paraphrased cycle ("Hol van itt most?" / "Most már itt vagyunk?" with small variations), a limit of the 9B Q4 model for a 10 000-word Hungarian story.
  - DRY runs on the CPU, so the backend sampling chain is cut: tg ~29.5 vs ~31.5 t/s (−6%).
- **One GPU device-lost during the night run (03:24).**
  - The kernel log shows ~10 s of GuC timeouts (`Timed out wait for G2H`, `GuC PC query task state failed`), then a job timeout for a job that had **not started**, then a GT reset.
  - This points to the GuC firmware or the Thunderbolt link rather than a long-running shader.
  - It did not reproduce in the next 18 runs (~2.5 hours of generation).
  - The devcoredump (`/sys/class/drm/card2/device/devcoredump/data`) is root-only; reading it needs the user.
  - The earlier `Timedout job` lines of the night all came from `test-backend-ops perf` runs of experimental FA configurations (very long single submits).

**Validation rule from now on:** after every speedup, run this loop test next to the accuracy and hallucination tests (levels 1–3).
Run it on every affected backend, with the same sampling as upstream.
- A speedup is only acceptable if its loop rate is not worse than upstream's (p1: 4/6 with default sampling, 0/6 with the Qwen recommendation).
- Run: `~/llama-bench-results/loop-test/loop-run.sh cu-full vk-full up-nomtp`.
- p1 only: `ONLY_P=1`.
- Known issue: the `UP` path in `loop-run.sh` points to a temporary directory; if it is missing, the upstream build has to be rebuilt.

### 27. Reduced vocabulary for the MTP draft LM head (2026-10-30)

The draft step runs the full LM head (248320×4096 q6_K, 834 MB): ~4.5 ms of the 7.6 ms draft step on the B50.
The draft only has to propose a likely token, and the target verifies it over the full vocabulary. Restricting the draft to a subset of tokens therefore cannot change the output. It can only lower the acceptance when the right token is outside the subset.

How the subset is chosen:
- **The id prefix alone is not enough for Hungarian.** Story output tokenized with the Qwen3.5 tokenizer:
  - ids < 32768 cover 76% of the tokens;
  - ids < 151643 cover 87%;
  - the rest are in the extended range (151k–248k).
  - For English wiki text, ids < 32768 cover 90%.
- **Prefix + the tokens already seen in the context** cover 93% of the Hungarian story (offline estimate).
  - Adding a Hungarian seed text brings it to 97.7%.

Implementation (`src/llama-draft-vocab.{h,cpp}`, `llm_graph_context::build_draft_lm_head`):
- **Static part:** ids `[0, n_static)` are scored through a view of the LM head rows, with no copy.
- **Dynamic part:**
  - an F16 head `[n_embd, n_dyn+1]` lives on the LM head's device;
  - every ctx_dft ubatch assigns rows to its new tokens;
  - the same graph copies their LM head rows in with `get_rows` + `set_rows`;
  - only the used rows are scored, rounded up to 1024, so the graph only changes when a chunk fills.
- **Full-size logits:** the unscored tokens get −INF (fill + concat + set_rows scatter), so the samplers, backend sampling and the p_min check are unchanged.
- **Control tokens** (`<|im_end|>`, `</think>`, ...) are added at start.
- **Scope:** used by every MTP head that ends in `build_lora_mm(head_w, cur, head_s)` (qwen35, qwen35moe, qwen3next, deepseek2/32, glm*, mimo2, hy-v3, nemotron-h-moe), and not when the head has a scale tensor.
- **Env:**
  - `LLAMA_MTP_DRAFT_VOCAB=n_static` (default 32768; `-1` disables it);
  - `LLAMA_MTP_DRAFT_VOCAB_DYN` (default 16384 rows, 128 MiB);
  - `LLAMA_MTP_DRAFT_VOCAB_SEED=<text file>`: its tokens are added at start. Recommended for non-English chats.

Server, B50, q8_0 KV, MTP n=1, defer + `-bs`, greedy (2 runs, median):

| config | story tg | wiki tg | accept story / wiki |
|---|---|---|---|
| full vocab | 34.27 | 28.37 | 0.60 / 0.74 |
| 32k + seen | 37.10 (+8.3%) | 29.64 (+4.5%) | 0.55→0.62 / 0.64→0.76 |
| 32k + seen + Hungarian seed | 37.43 (+9.2%) | 30.03 (+5.9%) | 0.58→0.60 / 0.69→0.75 |
| 16k + seed / 64k + seed | 37.59 / 36.96 | 30.08 / 29.88 | |
| default (32k, no env) | 37.45 | 29.67 | |
| n-max 3, CPU draft sampler: full → 32k | 26.79 → 31.48 (+17.5%) | 23.89 → 25.95 (+8.6%) | 0.31 / 0.46 both |

- The output hash is identical in every configuration; only the speed changes.
- The two accept values are the first and the second run of the same server. The dynamic set keeps growing across requests, so the second run has the Hungarian words of the first.
- The wiki prompt is English but the answer is Hungarian, which is why the first run loses acceptance without a seed.
- With n-max 3 the draft head runs three times per step, so the gain is the largest.

Validation:
- test-backend-ops SET_ROWS / GET_ROWS / FILL / CONCAT on the B50: all pass.
- Wiki QA: 25/25, 5/5.
- Loop test p1 (6 seeds):
  - default sampling 6/6, Qwen sampling 2/6. The same seeds loop with the same token counts as before, because the output is unchanged.
  - Loop-test tg under Qwen sampling: 31.2–33.4 → 33.2–36.2 t/s.
- The CUDA build compiles. It is not measured, because no CUDA card is attached.

### Attachment test: Hungarian Fundamental Law (B50, 2026-10-30)

The user reported a thinking-phase loop on Vulkan. To reproduce it, I added a new test (`tests/attach-test.py`, see `tests/README.md`):
- the 2011 Fundamental Law (`testdata/alaptorveny-2011.txt`, 41 366 prompt tokens) is attached the way the web UI does it;
- the prompt is "foglald össze a magyar alaptörvényt";
- each run reprocesses the prompt.

Build b298513ba, q8_0 KV, 128k context, 6 seeds per row:

| config | pp t/s | tg t/s | accept | not ok |
|---|---|---|---|---|
| MTP n=1 + `-bs`, default sampling | 446 | 24.7 | 0.79 | 0/6 |
| MTP, Qwen sampling | 446 | 22.6 | 0.71 | 0/6 |
| MTP, Qwen + DRY | 446 | 22.2 | 0.69 | 0/6 |
| no MTP, Qwen sampling | 501 | 19.6 | – | 0/6 |
| MTP, default sampling + web UI fields (`reasoning_control` ...) | 446 | 23.9 | 0.79 | 0/6 |
| MTP, Qwen sampling + web UI fields | 446 | 22.6 | 0.71 | 0/6 |

- **No loop reproduced in 36 runs.** The thinking is 6–11k characters and the answer 4–6k; every run finished with `stop`.
- **The web UI fields** give the same tokens. `reasoning_control` arms a CPU sampler, which costs ~3% tg with default sampling.
- **Still open:** the exact command, sampling settings and conversation of the user's looping run.
  - The web UI sends no seed, so every request takes a different trajectory.
- **To do:** run this test on the other GPUs as well (speed and loops).

## Remaining work

- FA prefill (coopmat1, Xe2): ~3.5 TFLOPS, about a third of the wiki pp.
  Br=32 does not help; the kernel structure (small tiles, shmem softmax, many barriers) is the limit. It would need a deeper rewrite.
- Draft LM head: done (finding 27). Next options: persist the learned set across restarts, or a Q8_0 dynamic head.
  Measure it on CUDA when a card is attached.
- Measure the Vulkan FA token merge on coopmat1 (Intel B50, RDNA3); slow iq4_nl/q5_1 dequant in the scalar FA.
- Fuse the small Vulkan ops (~4 ms per verify step on the 2080 Ti).
- Finding 23 was measured on Turing; check it on Ampere (3090).
- Pascal FA prefill: the fp32 Q tile of the TILE kernel sits at the 48 KB shared memory limit (hsk=256); it would need a deeper rewrite.
- Verify CPU recording ~7 ms per step (mostly overlapped with the GPU).
- Accuracy: done (levels 1–4 above). The own build is as accurate as brew and does not loop more often than upstream.
  Re-run all four levels after every new speedup.
- Loop test on Vulkan (`vk-full`) too: on the 2080 Ti only CUDA was measured, while the user saw it on both.
  The B50 has not been loop-tested either.
- B50 short-prompt tg is on par with brew under the recommended sampling.
  The remaining cost there is the MTP step (draft LM head 4.3 ms, verify ~38 ms); see the draft-head item above.
- Upstream report candidate: on small-BAR eGPUs the original Vulkan backend can hit the xe job timeout (`ErrorDeviceLost`)
  on long prompts. Finding 1 fixes it; it could be proposed upstream.
- Measure an intermediate sampling: only `--presence-penalty 1.5` with the old temp 0.8 (or DRY, `--dry-multiplier 0.8`).
  Goal: keep ~100 t/s without loops, since the Qwen recommendation loses ~18% through draft acceptance.
- Extend the loop test to the other KV types (q4_0, q4_1, q5_x, iq4_nl) and to MTP n=2.
- Redesign the Vulkan mat-vec for n=2 (~400 GB/s, ~510 on CUDA).
  Fuse small ops on both backends (~700 ops, ~5 µs each; e.g. RMS_NORM+SCALE in the GDN l2 norm).
- Speed up the iq4_nl/q5_x FA dequant on CUDA and Vulkan.
- Hardware: RTX 3090 (Ampere), RX 6900 XT (ROCm and Vulkan; the packages and groups need sudo from the user).
- For stable measurements: `sudo nvidia-smi -i 1 -lgc 1800,1800` (the user has to run it); the noise is ±10–15% now.
