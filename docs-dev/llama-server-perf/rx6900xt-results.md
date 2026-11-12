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

## Comparison with brew and upstream (first round, 325 W)

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

## Second round: transfers, concat, GQA decode, KV scratch (292 W)

The card's power cap was lowered from 325 W to 292 W for this round, so its numbers are not directly comparable with the tables above.

Changes:
- **`b9fe1f552` tiled concat (Vulkan, CUDA/HIP).** The delta-net conv input concat of Qwen3.5/3.8 has a transposed src1, so the plain kernel reads it uncoalesced. A 32x32 shared-memory tile: 409 -> 82 us on Vulkan, 316 -> 47 us on ROCm. pp +1.7-2.8% on Qwen3.5/3.8; KLD bit-identical.
- **`a2ea786f9` Vulkan host uploads.**
  - The scheduler input staging used `set_tensor_async`, which on RADV with ReBAR cost a queue wait per copy. Small inputs are now copied by the CPU when the backend is idle and the destination is host-coherent. gpt-oss tg d0 110.6 -> 117.3 (upstream 117.6), d8k 105.1 -> 110.0.
  - Unpinned host weights (a multi-device instance disables host-memory import on AMD) went through a CPU memcpy into ReBAR VRAM at 0.2 GB/s over Thunderbolt. They now use staged DMA at 1.4-1.9 GB/s. Hybrid Qwen3.8-27B (fit for 128k ctx) pp512: 49 -> 276 t/s at d0, 48 -> 242 at d8k.
- **`e92ee173b` HIP FA vec GQA decode.** 2 Q heads per block for every even GQA ratio (q5_0/q5_1 only above GQA 4), 2 threads per K row for q8_0/iq4_nl and 4 for the others; it replaces the 4-head variant, which register pressure made slower. Qwen3.8-27B iq4_nl tg: d8k 23.5 -> 25.0, d32k 18.6 -> 23.3; Qwen3.5-9B q8_0 tg d8k 57.4 -> 60.8.
- **`955d8d945` Vulkan quantized KV prefill on RDNA2.** The f16 dequant scratch path is about 2.7x slower in real models than in `test-backend-ops` with the same dispatch parameters (cause unknown, data dependent). The direct quantized path is now used at head size >= 256 on AMD RDNA2; the scratch path still wins at hd64/hd128. pp512 d8k / d32k: Qwen3.8 iq4_nl 434 / 234 -> 474 / 327, Qwen3.5 q8_0 1346 / 729 -> 1494 / 1099, gemma-4 q8_0 767 / 365 -> 941 / 616.

Tests: ROCm `test-backend-ops` 20099/20099; FA ROCm 4682/4682, Vulkan 5702/5702; CONCAT on both backends.

### Comparison at 292 W

Same builds as the first round (brew b11669, upstream master `17701e48d`), against `speedup` at `955d8d945`. Tag `w292`, q8_0 KV, FA on. Upstream ROCm still aborts in FA except on gpt-oss, so its column is FA off with an f16 KV cache (`w292-nofa`).

t/s, pp512 / tg128:

| model | depth | brew Vulkan | upstream Vulkan | upstream ROCm | own Vulkan | own ROCm |
|---|---|---|---|---|---|---|
| Qwen3.5-9B | 0 | 1202 / 57.0 | 1502 / 57.4 | 1541 / 62.9 (no FA) | 1674 / 57.3 | 1621 / 63.1 |
| Qwen3.5-9B | 8192 | 916 / 55.3 | 1087 / 55.7 | 940 / 53.0 (no FA) | 1500 / 55.1 | 1465 / 60.9 |
| gemma-4-12b | 0 | 933 / 42.1 | 1072 / 42.4 | 940 / 45.6 (no FA) | 1168 / 42.3 | 1076 / 45.5 |
| gemma-4-12b | 8192 | 634 / 40.0 | 699 / 40.2 | 763 / 33.5 (no FA) | 943 / 40.1 | 910 / 42.8 |
| Qwen3-14B | 0 | 817 / 43.3 | 963 / 43.7 | 854 / 44.2 (no FA) | 1046 / 43.7 | 927 / 45.2 |
| Qwen3-14B | 8192 | 527 / 38.8 | 587 / 39.1 | 156 / 21.1 (no FA) | 757 / 39.6 | 676 / 33.5 |
| gpt-oss-20b | 0 | 1937 / 116.1 | 2433 / 117.4 | 3011 / 114.5 | 2780 / 117.4 | 3027 / 118.2 |
| gpt-oss-20b | 8192 | 1383 / 109.7 | 1628 / 111.2 | 2149 / 104.8 | 2137 / 110.8 | 2140 / 107.8 |
| Qwen3.8-27B | 0 | 474 / 24.7 | 532 / 24.8 | 597 / 26.0 (no FA) | 555 / 24.7 | 626 / 26.1 |
| Qwen3.8-27B | 8192 | 373 / 24.3 | 409 / 24.4 | 361 / 22.0 (no FA) | 498 / 24.3 | 557 / 25.3 |

Qwen3.8-27B with an iq4_nl KV cache (the user's long-context setting, `KV=iq4_nl DEPTHS=0,8192,32768`, tag `w292-iq4`):

| depth | brew Vulkan | upstream Vulkan | upstream ROCm | own Vulkan | own ROCm |
|---|---|---|---|---|---|
| 0 | 472 / 24.7 | 528 / 24.7 | 136 / 22.3 | 553 / 24.6 | 626 / 25.7 |
| 8192 | 376 / 24.0 | 412 / 24.2 | 5.0 / 3.5 | 473 / 24.0 | 556 / 25.0 |
| 32768 | 235 / 22.5 | 248 / 22.7 | 1.3 / 1.0 | 326 / 22.1 | 415 / 23.3 |

Upstream ROCm has no FA kernel for iq4_nl KV on gfx1030, so its FA falls back to the CPU and collapses.

Own Vulkan:
- **Against upstream Vulkan:** prefill +4-14% at depth 0 and +22-38% at 8k; Qwen3.8 iq4_nl at 32k +31%. Decode is the same within 1%, except Qwen3.8 iq4_nl at 32k (-2.4%).
- **Against brew:** prefill +17-44% at depth 0, +34-64% at 8k, +39% on Qwen3.8 iq4_nl at 32k. Decode is the same or up to +2%, except Qwen3.8 iq4_nl at 32k (-1.7%).
- The gpt-oss decode regression of the first round is gone (117.4 = upstream).

Own ROCm:
- **Against upstream ROCm (no FA):** prefill at 8k +56% Qwen3.5, +19% gemma-4, 4.3x Qwen3-14B, +55% Qwen3.8; decode at 8k +15%, +28%, +59%, +15%. gpt-oss (FA works upstream): pp the same, tg +3%.
- **Against brew:** prefill +13-56% at depth 0, +28-60% at 8k; decode +4-10% at 8k, except Qwen3-14B (-14%) and gpt-oss (-2%).
- **Against the best upstream build at 32k** (Qwen3.8 iq4_nl): pp 415 vs 248 (+67%), tg 23.3 vs 22.7 (+3%).

### Hybrid GPU + CPU (Qwen3.8-27B, fit for 128k ctx)

With the server's default fit margin and the mmproj loaded, part of the FFN runs on the CPU; the laptop's RAM reads about 20 GB/s and the CPU FFN already uses 18 GB/s, so more threads do not help (`-t 16` is slower than 6). pp2048 / tg32:

| fit target, ubatch | Vulkan | ROCm |
|---|---|---|
| 1024, 512 (server default) | 240 / 5.8 | 271 / 5.5 |
| 256, 512 | 295 / 7.9 | 334 / 6.8 |
| 256, 256 | 220 / 8.8 | 251 / 8.1 |
| 256, 1024 | 324 / 5.7 | 392 / 5.7 |

With `--no-mmproj` and an iq4_nl or q4_1 KV cache the model fits on the GPU at 128k ctx: about 24.8 t/s tg on ROCm. On Vulkan, `GGML_VK_VISIBLE_DEVICES=<AMD index>` lets host memory be imported for the AMD card (fit 256, ub 512: pp 295 -> 316).

## Third round: Qwen3.8-27B for agentic use (300 W, 2026-11-12)

Target: the user's command, Qwen3.8-27B UD-IQ4_XS, 128k ctx, iq4_nl K and V, `--fit-target 64 -ub 256 -b 256 --no-mmproj`, with 15-20k-token prompts and long answers. The card is `Vulkan1` / `ROCm0` now (the T1200 is `Vulkan0`).

Baseline, 20.7k-token wiki prompt, 512 generated tokens:

| backend | pp t/s | tg t/s |
|---|---|---|
| Vulkan | 430 | 22.4 |
| ROCm | 519 | 23.6 |

Weights per decoded token: 13.45 GB (iq4_xs 61%, q5_K 14% with the LM head, iq3_s 9%, q4_K 7%, iq3_xxs 4.5%). At 512 GB/s that is about 26 ms, against 44.7 ms measured on Vulkan.

### Findings

1. **Mat-vec microbenchmarks lie on RDNA2.** A single 17408x5120 iq4_xs matrix (47 MB) stays in the 128 MB Infinity Cache, and `test-backend-ops` reports 1.3 TB/s.
   - New perf cases use 8 different matrices (`bs=[8,1]`), so the weights come from VRAM.
   - With them, every type used by the model runs at 485-500 GB/s on Vulkan (95-97% of peak). ROCm is the same, except iq2_s (395-411), iq3_xxs (432-470) and iq3_s (455-472).
   - In the model, the Vulkan mat-vecs reach 410 GB/s on average (GGML_VK_PERF_LOGGER). The difference is a fixed cost of about 5-9 us per dependent dispatch, the same as the floor of the small ops (ADD, GLU, MUL: 5.8 us each).
   - Not the cause:
     - MMVQ (`GGML_VK_DISABLE_MMVQ`: +2%; the IQ types do not use it);
     - rows per workgroup (`GGML_VK_RM_KQ` 1 and 4 are slower);
     - the clocks (MCLK 1000 MHz, SCLK ~2.6 GHz, 265 W during decode).
2. **Decode step on Vulkan, 20.7k context:** 44.3 ms of GPU time, made of:
   - mat-vec 32.8 ms;
   - FA 3.3 ms (117 GB/s);
   - about 1300 small ops, 8.3 ms.
3. **Vulkan decode FA (hd256, GQA rows 4-8).**
   - Before, the scalar path used row_split 4 with 8 rows, so every K/V value was dequantized by 4 row groups, and 2 of the 8 rows were padding at GQA 6.
   - The new config is one row group of 128 threads with exactly n_rows rows, Bc 32 and D_split 16.
   - RX 6900 XT, kv 20736, us:

     | gqa | f16 | q8_0 | iq4_nl |
     |---|---|---|---|
     | 6 | 111 -> 83 | 121 -> 82 | 189 -> 101 |
     | 4 | 203 -> 78 | 104 -> 68 | 167 -> 85 |

   - kv 33280, gqa 6: f16 233 -> 171, q8_0 175 -> 119, iq4_nl 271 -> 145 us. Multi-token decode (nb 2, 3) is also faster for every type.
   - Qwen3.8 wiki tg 22.37 -> 22.92.
4. **Vulkan prefill MMQ tiles on RDNA2.**
   - Without coopmat, AMD had the large tile disabled, so prefill ran on the 128-thread 64x64 tile.
   - The large integer dot tile is now enabled on RDNA2 for the MMQ path only, with 4 waves: 64x64 warp tiles, or 128x32 for the K-quant layout. q5_K gets 8 waves of 32x64.
   - 17408 x 512 x 5120, TFLOPS:

     | type | before | after | ROCm |
     |---|---|---|---|
     | iq4_xs | 32.6 | 33.9 | 37.7 |
     | iq3_s | 29.6 | 32.0 | 35.4 |
     | q4_K | 31.8 | 35.8 | |
     | q5_K | 23.8 | 30.4 | 26.1 |
     | q3_K | 26.0 | 28.5 | |

   - Qwen3.8 wiki pp 431 -> 449.
5. **The f16 KV scratch prefill mystery is not memory placement.**
   - With iq4_nl KV, `test-backend-ops` (kv 16384, 256 rows) gives:
     - scratch 7.2 ms;
     - direct 10.5 ms.
   - In the model, with the same dispatch parameters (`GGML_VK_FA_DEBUG`: Br 16, Bc 32, split_k 3), the scratch path takes 15.9 ms per call and the direct path 9.9 ms.
   - The scratch time grows superlinearly with depth: 1.23 ms at 2k keys, 7.4 at 8k, 15.9 at 16k.
   - The direct path grows linearly.
   - The scratch path is just as slow with `-c 32768`, where VRAM is free, and with only the AMD device visible.
   - So the direct path stays for hd256 on RDNA2.
6. **Prefill FA tile sweep** (quantized KV, 256 rows): no config beats the current one (Br 8, Bc 32, 128 threads). The dequant per row block is the cost.
7. **ROCm decode FA** is compute bound in the GQA vec kernel: about 180 us at kv 20736 for q8_0 and iq4_nl alike. None of these helped:
   - 3 heads per block: 459 us, from register pressure;
   - 4 threads per K row for iq4_nl: 273 us;
   - more parallel blocks: the HIP occupancy query returns 1 block per SM, so only 120 blocks run, but up to 8 blocks per SM only help f16 (307 -> 137 us).

   A faster ROCm decode needs a new kernel.
8. **Other ROCm paths:**
   - the FFN gate/up/GLU mat-vec fusion already exists in the CUDA backend;
   - MMQ prefill is 10-15% ahead of Vulkan;
   - no ROCm code was changed in this round.

### Result, 20.7k-token prompt, 512 generated tokens

| backend | pp t/s | tg t/s |
|---|---|---|
| Vulkan, before | 430 | 22.4 |
| Vulkan, after | 449 | 22.9 |
| ROCm (unchanged) | 519 | 23.6 |

Validation (Vulkan, RX 6900 XT):
- `test-backend-ops`: 22297/22297.
- Wiki QA: Qwen3.5-9B-MTP and Qwen3.8-27B (user command) both 25/25 answered, 5/5 abstained.
- Story loop test, p1, 6 seeds (Qwen3.5): default sampling 5/6 not ok, Qwen sampling 1/6, the same as the second-round baseline.
- Attachment test, Qwen sampling, 6 seeds: 0/6 loops; pp ~1160 t/s (41k tokens), tg ~64.4 t/s.

## Fourth round: prefill, and the fastest server command (2026-11-12)

Same card, model and use case as the third round (Qwen3.8-27B UD-IQ4_XS, iq4_nl K/V, 15-20k-token prompts, 3-4k
generated tokens). Prefill profile (Vulkan, pp512 at d16384, 1170 ms per ubatch): MUL_MAT 68%, FLASH_ATTN_EXT 27%,
GATED_DELTA_NET 1.4%, the rest small ops.

### Findings

1. **The FA prefill "f16 scratch mystery" (third round, finding 5) was the causal mask.**
   - `test-backend-ops` fills the mask with random values. A model mask is all zero except the causal triangle, so
     the scalar shader skips the mask load (`MASK_OPT_ALL_ZERO`) in almost every block.
   - That load had the only barrier of the KV loop. Without K/V staging in shared memory, the subgroups of a
     workgroup then drift apart and read K/V on their own.
   - f16 KV, hd256, gqa 6, kv 33280, 512 rows: 26 ms with a random mask, 77 ms with a causal one.
   - Fix: a barrier in the all-zero branch when there is no staging: 77 -> 24 ms. On the T1200 it is neutral
     (q8_0 hd256 99 -> 91 ms).
2. **With that fixed, the f16 scratch beats the direct quantized path on RDNA2 for every head size.** So the RDNA2
   hd >= 256 exception is gone.
   - hd256 f16 prefill has its own tile now: Br 16, Bc 32, row split 2, 128 threads, 23.7 -> 19.9 ms.
   - iq4_nl prefill FA, kv 33280: 38 -> 21 ms (ROCm: 24.9 ms).
   - pp512 @ d32768, q8_0 / iq4_nl KV:
     - Qwen3.5-9B: 1020 / 893 -> 1054 / 1062;
     - gemma-4-12b: 606 / 464 -> 673 / 663.
   - Qwen3.8 pp1024 @ d16384, ub 1024: 427 -> 495 (ROCm 505).
3. **The scratch is not part of the scheduler's reservation.** With the MTP draft on a full card (~90 MiB free at
   128k), the growing scratch over-committed VRAM: the driver paged buffers out, with 3-7 s stalls per ubatch after
   8k tokens and pp 77 t/s.
   - The scratch now grows only while the device has room, in 64 MiB steps.
   - Otherwise the scalar FA runs in KV chunks. Each chunk is dequantized into the scratch, and the O/L/M state
     goes from chunk to chunk through two split_k slots.
   - 8k-key chunks cost nothing (kv 33280: 24.5 ms whole, 24.0 chunked), 1k chunks 6%.
   - MTP server pp: 77 -> 451 t/s.
4. **ROCm MMQ on RDNA2:** J 128 tiles for q4_K and q5_K, 27.9 -> 31.2 and 26.2 -> 30.7 TFLOPS.
   - The IQ types lose 10-18% with J 128.
   - Other I / J / thread counts, and the FA tile configs (also 64 columns per block for hd256), were not faster.
   - ROCm pp +2%.
5. **MTP works with the Qwen3.8 GGUF** (it has the nextn layer) and is the biggest win for long answers.
   - `--spec-draft-n-max 1` is best: tg 22.8 -> 33-35 t/s (acceptance 0.75-0.80). With n-max 2 it is 25-26, with
     n-max 3 it is 22.
   - The draft KV must be iq4_nl (`-ctkd iq4_nl -ctvd iq4_nl`). With q8_0, `--fit` moves an FFN to the CPU.
   - Larger ubatches do not fit with MTP.
6. **ROCm has less free VRAM than Vulkan** (16196 vs 16351 MiB at start, compute buffer 176 vs 138 MiB), and its
   memory pool (MMQ src1, FA f16 K/V) is outside `--fit`.
   - With MTP at 128k, the fit offloads 2 FFNs (tg only 23-27).
   - At 116k it fits, but the first ubatch fails with out of memory in the pool.
   - 100k works.
7. **Without MTP, larger ubatches help prefill.**

   | backend | ub 256 | ub 512 | ub 1024 |
   |---|---|---|---|
   | Vulkan | 477 | 515 | 528 |
   | ROCm | 520 | 544 | 549 |

   - Wiki 20.7k pp, t/s.
   - At ub 2048, Vulkan offloads a layer.

### Fastest command, 20.7k wiki / 15.8k code prompt, 3500 generated tokens (default sampling, seed 1)

| backend | command | pp t/s | tg t/s | time for 20.7k + 3500 |
|---|---|---|---|---|
| Vulkan | user (ub 256) | 477 / 489 | 22.8 / 23.1 | 197 s |
| Vulkan | ub 1024 | 528 / 538 | 23.0 / 23.1 | 192 s |
| Vulkan | MTP, ub 256, 128k | 451 / 462 | 33.4 / 34.6 | 151 s |
| ROCm | user (ub 256) | 520 / 517 | 23.7 / 23.9 | 188 s |
| ROCm | ub 1024 | 549 / 568 | 23.6 / 23.9 | 186 s |
| ROCm | MTP, ub 256, 100k | 496 / 511 | 32.0 / 35.3 | 151 s |

Vulkan:

```
LLAMA_MTP_DEFER_CATCHUP=1 llama-server -hf unsloth/Qwen3.8-27B-GGUF:UD-IQ4_XS -c 128000 -fa on --parallel 1 \
  -ctk iq4_nl -ctv iq4_nl --fit-target 64 -ub 256 -b 256 --no-mmproj -dev Vulkan1 \
  --spec-type draft-mtp --spec-draft-n-max 1 -ctkd iq4_nl -ctvd iq4_nl
```

ROCm: the same with `-c 100000 -dev ROCm0`. Without MTP, `-ub 1024 -b 1024` is the fastest on both backends.

Validation (RX 6900 XT, after both commits):

| test | Vulkan | ROCm |
|---|---|---|
| `test-backend-ops` | 22303/22303 | 20127/20127 |
| wiki QA: Qwen3.5-9B-MTP, Qwen3.8 user command, Qwen3.8 MTP command | 25/25 + 5/5 for each | 25/25 + 5/5 for each |
| story loop, p1 6 seeds, default / Qwen sampling (Qwen3.5) | 5/6 / 1/6 not ok | 6/6 / 0/6 not ok |
| attachment test, 6 seeds (Qwen3.5) | 0/6 loops, pp ~1205, tg ~64 | 0/6 loops, pp ~1174, tg ~61 |

The loop results are the same as the earlier baselines. The FA tests also pass on the T1200, with and without
`GGML_VK_FA_KV_CHUNK=256`.

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

- **ROCm decode FA with q4_0 KV.** The GQA vec kernel is still slow for q4_0 (906 us at GQA 6, against 266 for iq4_nl); Qwen3-14B q8_0 still loses 26% tg at 8k, versus 10% on Vulkan.
- **ROCm FA prefill memory:** the f16 K/V conversion and the MMQ buffers come from the pool, outside `--fit`.
  - On a full card (MTP draft, 116k+ ctx) the first ubatch fails with out of memory.
  - A KV-chunked conversion like the Vulkan one would fix it.
- **ROCm hd256 FA prefill** stays at about 17 TFLOPS (tile kernel), against 21 on Vulkan now.
- **Vulkan prefill at depth is below ROCm** on Qwen3.8 (326 vs 415 t/s at 32k), and Vulkan decode at 32k is 2% below upstream.
- **Vulkan tg at short context is 4-12% below ROCm** on most models: mat-vec and small-op launch overhead (about 19% of a decode step).
- The second-round commits still need the story loop, attachment and wiki QA validation runs.
