# Qwen3.5-9B regression check after the bge-m3 work

What changed in shared code: graph inputs (no-cache mask), context output buffers (`embd_sparse`), the server
embeddings path, and the Pascal Vulkan f32/f16 matmul tile. The standard validation of
`docs-dev/llama-server-perf/bench/README.md` was run on the R720 (GTX 1080 Ti) on 2026-10-31/11-01 and compared with
the 2026-10-30 R720 baselines.

## VRAM note: `-fit` spills to the CPU now

On 10-31 the desktop held ~1.5 GB VRAM (~0.95 GB on 10-30). With the default `-fit`, the 128k MTP config no longer
fits and part of it moves to the CPU, on every build (upstream too): story tg dropped to ~15 t/s at ~800 % server
CPU. The round was therefore run with `LLAMA_ARG_FIT=off LLAMA_ARG_N_GPU_LAYERS=99`, and the upstream baselines were
measured again the same way. Upstream CUDA with MTP does not fit at all without `-fit` (mmproj allocation fails), so
its 10-30 numbers stay the reference. The Vulkan attachment test ran with `-c 80000` (41k prompt + 32k max tokens),
because 128k with q8_0 KV ran out of device memory.

## Results

| test | own Vulkan | own CUDA | baseline / upstream |
|---|---|---|---|
| test-backend-ops | 21446/21446 | 19569/19569 | — |
| wiki QA (hallucination) | 25/25, abstain 5/5, 81.7 s | 25/25, abstain 5/5, 54.0 s | 10-30: 25/25, 5/5, 84.2 s / 50.3 s |
| story loop, default sampling (vk-full / cu-full) | 5/6 | 5/6 | 10-30: 5/6, 5/6 |
| story loop, Qwen sampling (vk-qwen) | 0/6 | — | 10-30: 0/6 |
| story loop, no MTP (cu-nomtp / up-nomtp) | — | 4/6 | upstream today 4/6 |
| KLD vs upstream Vulkan f16-KV logits, q8_0 KV | 0.00121, same top 98.62 % | 0.00780, same top 97.56 % | upstream Vulkan 0.00128, 98.77 %; upstream CUDA 0.00780, 97.56 % |
| attachment test, 24 seeds | 0/24 not ok, pp 352, tg 22.2, acc 0.71 (`-c 80000`) | 0/24 not ok, pp 515, tg 30.9, acc 0.72 | 10-30: Vulkan 0/24, pp 354, tg 22.4, acc 0.71; CUDA 0/24, pp 524, tg 31.2, acc 0.72 |

The KLD reference is computed on Vulkan, so the CUDA number is the CUDA-vs-Vulkan numeric difference: upstream CUDA
gives exactly the same value.
`wiki.test.raw` from the ggml-org/ci zip did not match the B50 checksum; every number here uses the same file, so they
compare with each other but not with the B50 tables.

Speed with the Qwen sampling (`q-bench.sh`, median of 3, story / wiki):

| config | pp story | tg story | pp wiki | tg wiki | 10-30 tg story / wiki |
|---|---|---|---|---|---|
| own Vulkan, MTP | 236 | 25.9 | 420 | 24.1 | 30.9 / 29.1 |
| own Vulkan, no MTP | 261 | 28.6 | 483 | 25.2 | 34.0 / 29.4 |
| upstream Vulkan, MTP | 134 | 23.0 | 245 | 19.3 | 23.2 / 19.9 |
| upstream Vulkan, no MTP | 170 | 26.8 | 270 | 22.5 | 26.8 / 22.6 |
| own CUDA, MTP | 439 | 41.3 | 648 | 36.0 | 40.9 / 34.9 |
| own CUDA, no MTP | 457 | 36.5 | 720 | 33.4 | 37.6 / 34.1 |
| upstream CUDA, no MTP | 482 | 35.8 | 715 | 32.3 | 36.2 / 32.8 |

The own Vulkan tg is below its 10-30 value while upstream Vulkan is unchanged. A bisect rules out this work:

- `llama-bench` (q8_0 KV, FA, d=0 and d=16384): the pre-bge commit, the bge commit and the bge commit + new tile give
  the same pp and tg within noise (tg128 41.3 / 42.0 / 41.7, tg128 @ d16384 37.1 / 37.3 / 36.7).
- `llama-server`, story, no MTP: pre-bge build 28.73 t/s, current build 28.66 t/s.

So the gap against 10-30 is not caused by these changes; it is the same for the build before them.

The MTP draft acceptance of the own Vulkan build is identical to 10-30 (0.462 story, 0.589 wiki), so the generated
text is the same as well.

## Verdict

No regression: the backend tests pass, KLD and wiki QA match upstream, the loop rates and the attachment test match
the baselines, and CUDA speed matches its baseline. The slowdowns at the start of the round came from `-fit` spilling
to the CPU (desktop VRAM); the remaining own-Vulkan tg gap against 10-30 is shared by the build before these changes
and is not explained yet.
