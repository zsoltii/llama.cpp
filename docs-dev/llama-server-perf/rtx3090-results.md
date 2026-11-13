# RTX 3090 results (CUDA and Vulkan)

Test setup:
- **Machine:** the laptop (i7-11850H), with an RTX 3090 (Ampere, sm_86, 24 GB) on Thunderbolt.
  - CUDA: `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1` gives `CUDA0`.
  - Vulkan: `Vulkan1`.
  - The T1200 is index 0 on both.
- **Driver:** NVIDIA 580.196.07. Vulkan uses the NV_coopmat2 path, and rebar is off on the eGPU.
- **Builds:**
  - `build-cuda`: sm 61;75;86, CUDA 12.9 from the `llama-cuda` conda env;
  - `build-vk`;
  - upstream master `202c1a229` in `../llama.cpp-upstream/build-cuda` and `build-vk`. The upstream CUDA build needs the conda env activated, since it uses its compiler and linker flags.
- **Model:** Qwen3.8-27B UD-IQ4_XS (hd256, 4 KV heads, gqa 6, 16 of 64 layers with attention), iq4_nl K/V, FA on.
- **Target:** long input context and long generation (the user saw ~515 t/s prefill and ~26 t/s generation with a 110k-token prompt).

Upstream CUDA has no iq4_nl FA kernel: `GGML_CUDA_FA_QUANTS` defaults to q4_0, q8_0, f16 and bf16. With iq4_nl KV the FA runs on the CPU, so the upstream CUDA baseline uses q4_0 KV (also 4.5 bits per value).

## Baseline (llama-bench, pp2048 / tg128, ub 2048)

| build | KV | pp d0 / 16k / 64k | tg d0 / 16k / 64k |
|---|---|---|---|
| ours CUDA | iq4_nl | 1725 / 1394 / 909 | 45.6 / 37.5 / **24.1** |
| upstream CUDA | q4_0 | 1698 / 1400 / 916 | 46.9 / 42.0 / 31.8 |
| ours Vulkan | iq4_nl | 1262 / 1025 / 686 | 36.6 / 34.1 / 28.3 |
| upstream Vulkan | iq4_nl | 1244 / 1021 / 690 | 36.7 / 34.3 / 28.9 |

At 64k depth, CUDA generation lost 19.7 ms per token to attention. That means reading the 1.2 GB iq4_nl K/V of the 16 attention layers at ~61 GB/s, while the card has 936 GB/s.

## Findings

1. **CUDA decode FA read the K/V cache once per Q head.**
   - The vector kernel had the GQA mode (several Q heads of a K/V head per block) only on HIP.
   - `test-backend-ops perf`, single token, kv 33280, gqa 6, us: f16 164 (MMA), q8_0 296, q4_0 316, iq4_nl 628. The iq4_nl K/V data is 38 MB, which is ~41 us at full bandwidth.
   - **Fix:** the GQA mode on NVIDIA, with 4 threads per K row.
     - The block takes 4, 3 or 2 heads: the largest that divides the GQA ratio.
     - Sweep, gqa 6, iq4_nl, us: 2 heads 209, 3 heads 156, 6 heads 184.
     - Sweep over threads per K row, 3 heads: 32 threads 515, 8 threads 187, 4 threads 156.
   - Result, us: gqa 6 kv 33280 iq4_nl 623 -> 156, q4_0 314 -> 119; gqa 4 kv 20736 iq4_nl 275 -> 73.
2. **iq4_nl did two table lookups per int.**
   - `get_int_from_table_16` returns both nibble halves, and the KQ dot used one of them. The same int was looked up again for the other half by the same thread.
   - **Fix:** a thread handles both halves of its ints together. The V dequantization looks up only its 4 values.
   - Result, us: gqa 6 kv 33280 156 -> 144, gqa 4 kv 20736 73 -> 66.
3. **MTP verify (2 tokens) did not use the GQA mode.**
   - Measured, us: gqa 6 kv 33280 iq4_nl 676 with 2 tokens against 144 with 1 token.
   - **Fix:** the GQA mode now takes 2 tokens x heads per block. Column j is head j % ncols2 of token j / ncols2.
   - With 2 tokens, 2 heads per block is best: 3 heads gives 258 us, 4 heads at gqa 4 gives 176 us. The cause is register pressure.
   - Result, us: gqa 6 kv 33280 iq4_nl 676 -> 235, q8_0 395 -> 222; gqa 4 kv 20736 iq4_nl 295 -> 105.
4. **ncu: the GQA vector kernel used 255 registers with spills.**
   - Occupancy was 2 blocks per SM (16.7%) with 45% DRAM throughput.
   - Forcing more blocks per SM with `__launch_bounds__` made it slower (more spills).
   - **Fix:** Q's q8_1 ds is kept once per q8_1 block, and GQA mode unrolls the KQ loop by 2 and the V loop by 8. Sweep of the V unroll: 1, 2, 4, 8, 16 and full; 8 is best.
   - The kernel now uses 209 registers with no spills. Result, us: gqa 4 q8_0 85 -> 65, iq4_nl 66 -> 55, 2 tokens 106 -> 95; gqa 6 2 tokens iq4_nl 237 -> 213.
   - The remaining stalls are mostly instruction issue and long scoreboard. At 64k the FA is ~4.5 ms of a ~26 ms token, so little is left to gain there.
5. **Vulkan NVIDIA decode used 32-row coopmat2 tiles for 6 rows.**
   - With gqa 6, 26 of the 32 rows were empty. The scalar path with the RDNA2-style n_rows tile is faster on NVIDIA too.
   - Sweep: Bc 16/32/64, 64/128/256 threads, d_split 8/16, shmem staging, workgroups per core 4-32. The best is Bc 16, 128 threads, d_split 16, no staging, 4 workgroups per core.
   - Result, kv 33280 gqa 6, f16 / q8_0 / q4_0 / iq4_nl, us:
     - RTX 3090: 228 / 252 / 241 / 268 -> 180 / 137 / 133 / 172.
     - T1200: 1319 / 2046 / 1675 / 2691 -> 1213 / 1095 / 1050 / 1258.
   - The T1200 (TU117) has no coopmat extension, so the Vulkan backend classes it as "pre-Turing". The condition is therefore NVIDIA with fp16, which excludes Pascal.
   - **Dead end:** merging the 2 MTP tokens of gqa 6 into one 12-row tile is slower (q8_0 237 -> 313 us).

## Result (llama-bench, pp2048 / tg128, ub 2048)

| build | pp d0 / 16k / 64k | tg d0 / 16k / 64k |
|---|---|---|
| ours CUDA, before | 1725 / 1394 / 909 | 45.6 / 37.5 / 24.1 |
| ours CUDA, after | 1692 / 1388 / 903 | 45.9 / 43.6 / **37.8** |
| ours Vulkan, before | 1262 / 1025 / 686 | 36.6 / 34.1 / 28.3 |
| ours Vulkan, after | 1265 / 1036 / 683 | 37.5 / 35.4 / **30.8** |

## Server: 115k-token prompt + 2000 generated tokens

The prompt is the 2011 Hungarian Fundamental Law, part of `seed-hu.txt`, `long-wiki.txt`, a code context and the analysis question (`PROMPTS=long` in the scratch `cmdbench.sh`). The common server options are:

```
-c 131072 -fa on --parallel 1 -ctk iq4_nl -ctv iq4_nl --no-mmproj -ngl 99 -ub 2048 -b 2048
```

MTP adds `--spec-type draft-mtp --spec-draft-n-max 1 -ctkd iq4_nl -ctvd iq4_nl` and `LLAMA_MTP_DEFER_CATCHUP=1`. Default sampling, seed 1.

| build | MTP | pp t/s | tg t/s |
|---|---|---|---|
| **ours CUDA** | yes | 865 | **39.4** |
| ours CUDA | no | 958 | 32.7 |
| upstream CUDA (q4_0 KV) | yes | 884 | 33.8 |
| upstream CUDA (q4_0 KV) | no | 977 | 24.9 |
| ours Vulkan | yes | 646 | 30.6 |
| ours Vulkan | no | 729 | 26.9 |
| upstream Vulkan | no | 696 | 24.4 |

Prefill ubatch sweep at 115k:
- CUDA: ub 512 / 1024 / 2048 gives 883 / 935 / 962 t/s. With MTP: ub 512 801, ub 2048 865.
- Vulkan: 693 / 708 / 730 t/s.

A larger ubatch is the main prefill lever at this depth. With MTP the prefill is lower because the draft layer also processes the prompt.

## Validation (after commit 87debad8d)

| test | CUDA | Vulkan |
|---|---|---|
| wikiqa Qwen3.5 MTP (answerable / abstained) | 25/25, 5/5 | 25/25, 5/5 |
| wikiqa Qwen3.8 iq4_nl | 25/25, 5/5 | 25/25, 5/5 |
| wikiqa Qwen3.8 iq4_nl + MTP | 25/25, 5/5 | 25/25, 5/5 |
| story loop, default sampling (loops / 6) | 5/6 | 6/6 |
| story loop, Qwen sampling (loops / 6) | 1/6 | 2/6 |
| attach test, 41k-token Hungarian constitution (loops / 6) | 0/6, tg ~108 t/s | 0/6, tg ~97 t/s |
| test-backend-ops (full) | 20285/20285 | 22371/22371 |

The loop counts match the earlier baselines (default sampling 4-6/6, Qwen sampling 0-1/6). Vulkan with Qwen sampling at 2/6 is one run above that range; with 6 samples this is within noise.

## Remaining gaps

- **Prefill at depth looks compute-bound on the MMA FA kernel.**
  - This is an estimate from the llama-bench times: at 64k depth the attention part of a 2048-token ubatch works out to roughly 50 TFLOPS f16. It was not profiled with ncu.
  - Ours and upstream are within 2%, so this round did not change it.
- **iq4_nl decode FA is still slower than q4_0.** At gqa 6 it takes 144 us against 120 us, because of the table lookups.
- **Vulkan 2-token verify at gqa 6** stays on the old path.
- **HIP:** the CUDA changes that could help HIP are untested on the RX 6900 XT.
