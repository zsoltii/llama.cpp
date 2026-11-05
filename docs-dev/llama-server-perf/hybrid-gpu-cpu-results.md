# Hybrid GPU+CPU results: gemma-4-12b, Qwen3-14B, gpt-oss-20b

Machine: R720 VM (hypervisor flag), 2x E5-2695 v2 (2 NUMA nodes x 20 vCPU), 30 GB RAM,
GTX 1080 Ti on PCIe Gen3 x8 (pinned H2D 6.1 GB/s, pageable 2.5 GB/s).
Vulkan on Pascal: no fp16, int dot available. Run-to-run noise on tg is +-1-2 t/s.

Models: gemma-4-12b Q4_K_M, Qwen3-14B Q4_K_M, gpt-oss-20b MXFP4.
gemma and Qwen fit fully in VRAM; gpt-oss is the real hybrid (MoE experts of the
first N layers on the CPU via `-ncmoe N`).

Builds: brew (Vulkan), upstream master Vulkan/CUDA (up-vk, up-cuda), speedup branch
Vulkan/CUDA (own-vk, own-cuda). llama-bench, pp2048 / tg128 at depth 0 / 4096 / 15360.

## Reference table


### gemma

| build | best config | pp d0 | pp d4k | pp d15k | tg d0 | tg d4k | tg d15k |
|---|---|---|---|---|---|---|---|
| brew | ngl99 kvq8, t10 ub2048 fa1 | 220.60 | 134.76 | 66.84 | 29.68 | 26.64 | 23.38 |
| up-vk | ngl99 kvf16, t10 ub512 fa0 | 216.98 | 191.53 | 153.15 | 30.78 | 27.70 | 23.60 |
| up-cuda | ngl99 kvf16, t10 ub1024 fa1 | 555.96 | 422.21 | 278.68 | 32.77 | 30.75 | 29.61 |
| own-vk | ngl99 kvf16, t6 ub512 fa0 | 378.11 | 342.06 | 290.25 | 34.15 | 30.44 | 25.41 |
| own-cuda | ngl99 kvf16, t28 ub256 fa1 | 543.93 | 429.74 | 283.42 | 35.05 | 32.93 | 31.04 |

### qwen

| build | best config | pp d0 | pp d4k | pp d15k | tg d0 | tg d4k | tg d15k |
|---|---|---|---|---|---|---|---|
| brew | ngl99 kvq8, t6 ub512 fa1 | 221.79 | 184.54 | 125.20 | 26.64 | 24.74 | 21.30 |
| up-vk | ngl99 kvq8, t6 ub512 fa1 | 221.74 | 184.31 | 125.54 | 26.91 | 25.02 | 21.48 |
| up-cuda | ngl99 kvq8, t6 ub256 fa1 | 479.58 | 342.88 | 204.19 | 29.36 | 26.67 | 21.62 |
| own-vk | ngl99 kvq8, t28 ub512 fa1 | 336.16 | 252.46 | 156.13 | 27.11 | 25.05 | 21.39 |
| own-cuda | ngl99 kvq8, t6 ub256 fa1 | 482.47 | 343.11 | 203.39 | 30.38 | 27.51 | 22.21 |

### oss

| build | best config | pp d0 | pp d4k | pp d15k | tg d0 | tg d4k | tg d15k |
|---|---|---|---|---|---|---|---|
| brew | ncmoe3 kvq8, t10 ub512 fa1 | 137.48 | 142.64 | 121.79 | 33.16 | 32.91 | 30.03 |
| up-vk | ncmoe4 kvq8, t28 ub512 fa1 | 130.50 | 120.62 | 117.48 | 31.21 | 28.75 | 29.30 |
| up-cuda | ncmoe4 kvq8, t6 ub512 fa1 | 546.18 | 326.42 | 281.87 | 30.13 | 31.53 | 31.44 |
| own-vk | ncmoe4 kvq8, t16 ub512 fa1 | 136.83 | 131.92 | 119.36 | 31.21 | 31.97 | 28.71 |
| own-cuda | ncmoe4 kvq8, t20 ub512 fa1 | 493.04 | 444.93 | 376.81 | 34.39 | 35.17 | 33.25 |

### oss, re-tune with pinned host weights (`-lm none`, kvq8 fa1; plain = t10, numa = `numactl --interleave=all` + t20 `--numa distribute`)

| build | best config | mode | pp d0 | pp d4k | pp d15k | tg d0 | tg d4k | tg d15k |
|---|---|---|---|---|---|---|---|---|
| brew | ncmoe4 ub1024 | plain | 215.39 | 198.08 | 163.62 | 26.22 | 26.35 | 24.34 |
| brew | ncmoe4 ub1024 | numa | 213.77 | 200.75 | 163.24 | 28.87 | 28.70 | 26.51 |
| up-vk | ncmoe4 ub1024 | plain | 213.05 | 196.92 | 162.48 | 30.73 | 28.27 | 28.08 |
| up-vk | ncmoe4 ub1024 | numa | 213.33 | 199.35 | 160.59 | 28.15 | 27.29 | 27.45 |
| up-cuda | ncmoe5 ub1024 | plain | 1006.77 | 828.81 | 588.92 | 27.90 | 26.77 | 27.17 |
| up-cuda | ncmoe5 ub1024 | numa | 1012.45 | 852.47 | 588.58 | 27.97 | 29.47 | 26.64 |
| own-vk | ncmoe5 ub1024 | plain | 548.28 | 506.88 | 400.87 | 25.80 | 25.33 | 25.08 |
| own-vk | ncmoe5 ub1024 | numa | 621.25 | 546.21 | 404.26 | 30.56 | 27.53 | 24.99 |
| own-cuda | ncmoe5 ub1024 | plain | 1044.50 | 887.73 | 608.84 | 30.88 | 29.90 | 27.53 |
| own-cuda | ncmoe5 ub1024 | numa | 1047.49 | 875.02 | 614.87 | 33.69 | 31.91 | 31.56 |

ub2048 and ncmoe3 at ub>=1024 do not fit in 11 GB VRAM at d15k. ncmoe5 trades ~3 t/s tg for 2x pp vs ncmoe4 ub512.

## Findings

- **Pinned host weights.** With `-ncmoe` / `-ot ...=CPU` and the default mmap load the
  CPU-side weights are pageable, so every ubatch that offloads them to the GPU goes
  through a staging copy at ~2.5 GB/s. `-lm none` loads them into the pinned host
  buffer type instead: gpt-oss pp2048 CUDA 512 -> 1007 t/s (upstream), Vulkan 136 -> 181
  (before the MMQ fix below). It also allows ub1024, which doubles pp again.
- **NUMA.** The CPU part of a gpt-oss token is almost entirely MUL_MAT_ID (~15 ms of
  ~31 ms per token) and is memory-bandwidth bound (13.4 GB/s with 10 threads on one
  node). `numactl --interleave=all` + `-t 20 --numa distribute` raises the kernel to
  ~20 GB/s and cuts MUL_MAT_ID time ~20%; end to end it gives +2-4 t/s tg on own-cuda
  and own-vk, but on other builds it stays within the noise.
- **ncmoe vs ubatch.** ub2048, and ncmoe3 with ub>=1024, run out of the 11 GB VRAM at
  d15k. ncmoe5 ub1024 maximizes pp; ncmoe4 ub512 gives ~3 t/s more tg at half the pp.
- **Scheduler profile.** `GGML_SCHED_PROF=1` prints per-backend split counts, copy and
  compute time and host-weight bytes per graph when the scheduler is freed
  (llama-bench needs `-v` to show it).

## Fixes on the speedup branch

1. `vulkan: int-dot MUL_MAT_ID and MXFP4 matmul pipelines on fp32-only devices`.
   The fp32 (no fp16) path had no q8_1 integer-dot MMID pipelines and no dense MXFP4
   q8_1 pipeline, so Pascal fell back to dequant + fp32 matmul. mxfp4 MUL_MAT_ID n512:
   36.6 -> 9.7 ms; dense mxfp4: 31 -> 6.4 ms. gpt-oss Vulkan pp2048 (-lm none):
   213 (upstream) -> 548-621 t/s.
2. `vulkan: fix FA GQA decode when the retuned block has fewer rows than gqa_ratio`.
   A regression from the earlier Pascal FA tile change on this branch: for gqa 8 and
   a KV of less than 1024 cells (gpt-oss SWA with `--parallel 1`) the retuned block
   had 4 rows, so heads 4-7 were never written and the model produced garbage. The GQA
   path is now dropped when the block is smaller than the ratio. New test-backend-ops
   case with a sliding-window mask, KV 768, gqa 8.
3. `vulkan: Pascal MUL_MAT_ID int-dot tile`. The earlier Pascal MMQ tile (16 warps) was only
   applied to MUL_MAT; MUL_MAT_ID kept the 128-thread default. 8 warps with 64x32 warp tiles,
   n512: mxfp4 9.4 -> 5.5 ms, q4_0 6.6 -> 4.1 ms, q4_K 7.5 -> 4.7 ms. gpt-oss own-vk pp2048
   (ncmoe5 ub1024, t10): 548 -> 790 t/s at d0, 401 -> 459 at d15k; server prompt of 43.6k
   tokens 280 -> 395 t/s. A tile with WMITER=1 and TN=2 looked 1.2-1.9x faster but leaves half
   of each warp tile unwritten (NaN perplexity on Qwen3-14B, while test-backend-ops passed);
   an assert now rejects such tiles. Tests: MUL_MAT/MUL_MAT_ID 3541/3541, gpt-oss KLD 0.035321
   (bit-identical), Qwen3-14B PPL 6.4980 (CUDA 6.4997), attach 0/3 bad, wiki QA 25/25, 4/5.
4. `vulkan: return GGML_STATUS_ALLOC_FAILED on device OOM during graph compute`. The scratch
   buffers grow inside graph_compute; a full device threw `vk::SystemError` and aborted the
   process (seen at ncmoe4 ub1024 d15k). Now llama_decode returns -2 and the context stays
   usable (injected failure in llama-server: the next requests answer correctly).
5. `tests: cover the large matmul tiles for every type`. MUL_MAT/MUL_MAT_ID tests with m, n > 64
   existed only for f16, f32 and q4_0; the invalid k-quant tile of item 3 now fails 4 cases.
6. `llama: load CPU-overridden weights into pinned memory in --load-mode auto`. With a GPU and a
   `-ot ...=CPU` / `-ncmoe` override, auto now loads without mmap (pinned host buffers) when the
   model is smaller than 80% of free RAM; `--load-mode mmap` keeps the old behaviour.
   gpt-oss -ncmoe 5 ub1024 pp2048, default load mode: CUDA 484 -> 1023 t/s, Vulkan 352 -> 783 t/s;
   tg128 unchanged (29.4-31.7 t/s for auto and mmap, t10/t20, with and without interleave).
   Attach test on CUDA with the default load mode: 0/2 bad, pp ~495 t/s at 43.6k tokens.

## Tests

- test-backend-ops Vulkan0: MUL_MAT, MUL_MAT_ID (1111 MMID cases, 98 mxfp4),
  FLASH_ATTN_EXT 5402/5402 OK.
- KLD vs up-cuda (c4096, 6 chunks, f16 KV): own-vk 0.0353, up-vk 0.0327, own-cuda ~0.
- Wiki QA, gpt-oss (reasoning low, 1024 tokens): own-vk 22/25 and 4/5, the same as up-vk;
  up-cuda 23/25.
- Attach test (43.6k-token document): Qwen3.5 on Vulkan 0/6 loops; gpt-oss own-vk
  `-ngl 99 -ncmoe 4 -lm none`, `--parallel 1`, 3 seeds: 0/3 bad, pp ~280, tg ~20.8.

Raw data: `~/llama-bench-results/hybrid/` (`*-t2.jsonl`, `chain3.txt`, `eval/`).
