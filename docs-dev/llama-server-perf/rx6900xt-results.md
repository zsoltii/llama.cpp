# RX 6900 XT results (Vulkan and ROCm)

Test setup:
- **Machine:** the laptop (i7-11850H), with an AMD Radeon RX 6900 XT (RDNA2, gfx1030, 16 GB) on Thunderbolt as `Vulkan2` / `ROCm0`.
- **Drivers:** RADV (Mesa) and ROCm 7.1.1 (Fedora 44 packages).
- **Builds:**
  - `build-vk`;
  - `build-rocm`, configured with `HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" cmake -B build-rocm -DGGML_HIP=ON -DGPU_TARGETS=gfx1030`;
  - upstream master `17701e48d` in `../llama.cpp-upstream/build-rocm`.

Models:
- the Qwen3.5-9B-MTP UD-Q4_K_XL;
- the models of the R720 hybrid work (`hybrid-gpu-cpu-results.md`): gemma-4-12b-it UD-Q4_K_XL, Qwen3-14B Q4_K_M, gpt-oss-20b UD-Q4_K_XL, Qwen3.8-27B UD-IQ4_XS;
- bge-m3, converted to the hybrid f16 GGUF.

The measurements use `bench/prof/model-bench.sh`:
- llama-bench, FA on, q8_0 KV, `-ngl 99`;
- pp512 and tg128 at depth 0 and 8192;
- 3 repetitions.

## Findings

1. **ROCm aborted in every FA prompt with head size <= 128** on gfx1030: `GGML_ASSERT(max_blocks_per_sm > 0)` in `launch_fattn`.
   - **Cause:** `hipOccupancyMaxActiveBlocksPerMultiprocessor` limits a block to `regsPerBlock` = 32768 registers, which is one SIMD. The 256-thread tile kernel for hd128 uses 220 VGPRs (57k registers per block), so the query returns 0, but the kernel runs fine. Upstream master aborts the same way.
   - **Fix:** take at least 1 block per SM on HIP. `test-backend-ops` ROCm0: 20088/20088.
2. **ROCm FA decode is 3-5x slower than Vulkan** with a quantized KV cache.
   - `test-backend-ops perf`, single token, us:

     | shape | ROCm | Vulkan |
     |---|---|---|
     | Qwen3-14B hd128 gqa5 kv8704, q8_0 | 230 | 66 |
     | gpt-oss hd64 gqa8, q8_0 | 145 | 45 |
     | Qwen3.5 hd256 gqa4 kv20736, q8_0 | 494 | 102 |

   - In the model this is the tg drop with depth: Qwen3-14B 45.1 -> 33.3 t/s on ROCm (d0 -> d8192), 43.0 -> 38.8 on Vulkan.
   - **Kernel choice:** the vec kernel computes every Q head in its own block. The tile kernel needs an f16 copy of the cache.
   - **GQA variant of the vec kernel** (several Q heads of a K/V head per block, `ncols2`):
     - it helps only at hd256 with 4 heads per block: Qwen3.5 494 -> 362 us;
     - it is slower for hd128 gqa5 (230 -> 359), hd64 gqa8 (145 -> 267) and hd256 gqa6 (211 -> 285);
     - it is therefore enabled only for hd256 with a GQA ratio divisible by 4, on HIP.
   - The remaining gap needs a different decode kernel; it is still open.
3. **Vulkan FA prefill on RDNA2 was 2x slower than ROCm** at depth.
   - Qwen3-14B hd128 gqa5, 512 rows, kv 8704: f16 10.2 ms vs 5.2 ms.
   - A sweep of the scalar FA tile (block rows/cols, row split, workgroup size, occupancy limit) on the shapes of every model found 128-thread workgroups without row split.
   - The sweep has to drop the configs where a thread gets no KV columns (`Bc * D_split * row_split < workgroup size`), or a row group is not whole subgroups. Those compute garbage quickly: `test-backend-ops perf` does not check results.
   - Results, ms, f16 / q8_0:

     | shape | config | before | after |
     |---|---|---|---|
     | hd64 gqa8 kv8704 | Br 8, Bc 64 | 8.3 / 11.0 | 6.5 / 8.1 |
     | hd128 gqa5 kv8704 | Br 8, Bc 64 | 10.2 / 11.9 | 6.1 / 7.1 |
     | hd256 gqa4 kv8704, quantized KV only | Br 8, Bc 32 | q8_0 6.4, q4_0..iq4_nl 8.2-10.7 | 5.0, 5.9-7.0 |

     With f16 KV, hd256 is 3% slower with this config. hd512 (gemma-4 global layers) has no better valid config.

## Model benchmarks (t/s, before -> after the two speedups)

| backend | model | pp512 d0 | tg128 d0 | pp512 d8192 | tg128 d8192 |
|---|---|---|---|---|---|
| Vulkan | Qwen3.5-9B | 1666 | 56.2 | 1331 -> 1340 | 54.8 -> 55.1 |
| Vulkan | gemma-4-12b | 1185 | 41.1 | 788 | 39.2 |
| Vulkan | Qwen3-14B | 1074 | 43.0 | **661 -> 777 (+17.6%)** | 38.8 |
| Vulkan | gpt-oss-20b | 2831 | 110.7 | **2059 -> 2163 (+5.1%)** | 103.4 -> 105.1 |
| Vulkan | Qwen3.8-27B | 539 -> 559 | 24.5 | 440 | 24.1 |
| ROCm | Qwen3.5-9B | 1622 | 62.7 | 1466 | **55.7 -> 58.4 (+4.8%)** |
| ROCm | gemma-4-12b | 1096 | 43.8 | 926 | 41.3 |
| ROCm | Qwen3-14B | 945 | 45.1 | 687 | 33.3 -> 33.5 |
| ROCm | gpt-oss-20b | 3130 | 118.1 | 2175 | 107.6 |
| ROCm | Qwen3.8-27B | 623 | 25.9 | 561 | 23.2 |

Embeddings, bge-m3 f16, `-embd 1`, Vulkan: pp512 12.1k, pp2048 13.2k -> 13.9k t/s. ROCm: pp2048 13.5k t/s (its pp512 has a large run-to-run spread).

## Comparison with brew and upstream

Builds compared:
- **brew:** llama.cpp b11669 (`620d2225c`), Vulkan only.
- **upstream:** master `17701e48d`, Vulkan and ROCm.
- **own:** `speedup` with the fixes above (`tuned1`).

All builds were run with `model-bench.sh` on the same card.

On upstream ROCm, FA aborts (`max_blocks_per_sm > 0`) on every model except gpt-oss. The usable upstream ROCm configuration is therefore FA off with an f16 KV cache (`FA=0`, tag `cmp-nofa`). Own ROCm with FA off gives the same numbers as upstream with FA off.

bge-m3 does not load on brew or upstream: the hybrid GGUF needs this fork.

t/s, pp512 / tg128:

| model | depth | brew Vulkan | upstream Vulkan | upstream ROCm | own Vulkan | own ROCm |
|---|---|---|---|---|---|---|
| Qwen3.5-9B | 0 | 1450 / 52.5 | 1499 / 56.4 | 1566 / 62.4 (no FA) | 1664 / 56.4 | 1623 / 62.0 |
| Qwen3.5-9B | 8192 | 1055 / 51.3 | 1080 / 54.4 | 950 / 53.2 (no FA) | 1340 / 55.1 | 1465 / 58.4 |
| gemma-4-12b | 0 | 1060 / 33.1 | 1079 / 41.6 | 955 / 44.5 (no FA) | 1185 / 41.3 | 1096 / 43.8 |
| gemma-4-12b | 8192 | 690 / 31.5 | 706 / 39.2 | 777 / 33.0 (no FA) | 787 / 39.1 | 926 / 41.1 |
| Qwen3-14B | 0 | 943 / 42.6 | 989 / 43.3 | 871 / 43.5 (no FA) | 1078 / 42.8 | 945 / 45.4 |
| Qwen3-14B | 8192 | 586 / 38.0 | 599 / 38.5 | 158 / 21.0 (no FA) | 777 / 38.9 | 688 / 33.5 |
| gpt-oss-20b | 0 | 2412 / 112.5 | 2482 / 117.6 | 3117 / 114.8 | 2833 / 110.6 | 3116 / 118.2 |
| gpt-oss-20b | 8192 | 1585 / 104.5 | 1641 / 108.8 | 2190 / 104.9 | 2163 / 105.1 | 2179 / 107.9 |
| Qwen3.8-27B | 0 | 485 / 24.5 | 545 / 24.6 | 611 / 25.9 (no FA) | 559 / 24.5 | 630 / 25.9 |
| Qwen3.8-27B | 8192 | 380 / 24.0 | 418 / 24.2 | 364 / 21.9 (no FA) | 440 / 24.1 | 561 / 23.2 |

Own ROCm against upstream ROCm:
- **Prefill at depth 8192:**
  - Qwen3.5-9B: +54%;
  - gemma-4-12b: +19%;
  - Qwen3-14B: 4.3x;
  - Qwen3.8-27B: +54%.
- **Decode at depth 8192:**
  - Qwen3.5-9B: +10%;
  - gemma-4-12b: +25%;
  - Qwen3-14B: +60%;
  - Qwen3.8-27B: +6%.
- **Why:** FA works on gfx1030, so a q8_0 KV cache is usable.
- **gpt-oss:** pp is the same; tg is +3%.

Own ROCm against brew:
- **Faster:**
  - prefill: +0-30% at depth 0 and +17-48% at 8k;
  - decode: +5-32%.
- **Slower:** decode at 8k on Qwen3-14B (-12%) and Qwen3.8-27B (-3%). This is the ROCm decode FA gap below.

Own Vulkan against upstream Vulkan:
- **Prefill:** +3-14% at depth 0, +5-32% at 8k.
- **Decode:** the same, except gpt-oss, which is 6% / 3.4% slower (110.6 vs 117.6 t/s). That is a regression in the branch and is still open.

## Validation (Qwen3.5-9B-MTP, the user's server command)

| test | Vulkan | ROCm |
|---|---|---|
| KLD, q8_0 KV vs the B50 f16 reference | mean 0.0029, 99.9% 0.183, same top p 98.1% | mean 0.0018, 99.9% 0.124, same top p 98.1% |
| wiki QA | 25/25 answered, 5/5 abstained | 25/25 answered, 5/5 abstained |
| story loop test, p1 6 seeds, default / Qwen sampling | 5/6 / 1/6 not ok | 6/6 / 0/6 not ok |
| attachment test, 6 seeds | 0/6 loops; pp ~827, tg ~63 t/s | 0/6 loops; pp ~1127, tg ~58.8 t/s |

Notes:
- **Vulkan max KLD:** it is 12.3, but upstream Vulkan on this card also shows single outliers (max 4.76, 99.9% 0.326). It is one token, not a shift in the distribution.
- **Story loop with default sampling:** it loops on every backend; the Qwen sampling settings are what matter.

During the brew Qwen3.8-27B run the eGPU dropped off the Thunderbolt bus once (`amdgpu: device lost from bus`), which froze the machine. The rerun went through.

## Remaining gaps

- **ROCm decode FA with a quantized KV cache.** Qwen3-14B loses 26% tg at 8k, versus 10% on Vulkan.
- **Vulkan gpt-oss decode is 6% below upstream** (110.6 vs 117.6 t/s at depth 0). The cause is not found yet.
- **Vulkan gemma-4 prefill at depth.** The hd512 global layers: 788 vs 926 t/s on ROCm.
- **Vulkan tg at short context is 4-12% below ROCm** on every model: mat-vec and launch overhead.
