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


## Second round (2026-11-13)

### Qwen3.8-27B, CUDA

1. **Decode FA with a quantized KV cache on the MMA kernel** (d475d64b7). The MMA kernel preloads the raw K/V rows with
   cp.async and dequantizes them into its f16 tile, so every K/V value is read once for all GQA heads, without an f16
   copy of the cache. gqa 6, kv 33280, us: iq4_nl 1 token 142 -> 87, 2 tokens (MTP verify) 211 -> 100.
2. **KV write** (38ae5c511): quantized set_rows with one warp per block. iq4_nl, 1024 values: 14.6 -> 1.7 us.
3. **Mat-vec of iq3_s / iq3_xxs / iq2_s / q3_K** (865d87973): 4 rows per block on Ampere and the iq3 grids in shared
   memory. GB/s, 1 / 2 tokens: iq3_s 586 / 552 -> 711 / 626, q3_K 719 / 581 -> 803 / 696.
4. nsys at 112k: only FA grows with depth (0.3 -> 4.1 ms per token); mat-vec stays at 17.8-18.0 ms.

| | before | after |
|---|---|---|
| llama-bench tg128 d0 / 16k / 64k | 45.9 / 43.6 / 37.8 | 47.5 / 45.9 / 42.5 |
| server, 115k prompt, MTP, tg | 39.6 | 48.5 |

### Qwen3.8-27B, Vulkan

- MTP verify of gqa 6 with 4-bit K/V on merged coopmat1 tiles, 2 rows per integer mat-vec on NVIDIA (69b1bd100):
  verify FA iq4_nl 311 -> 231 us; server 115k MTP tg 30.6 -> 36.1; tg128 d0 / 16k / 64k 38.2 / 36.1 / 31.9.
- Dead ends: an integer (MMVQ) iq3_s mat-vec, a coalesced K_PER_ITER 8 iq4_xs shader, rows sweeps for the k-quants,
  more rows for the 2-column MTP verify mat-vec.

### bge-m3 (CUDA)

- The mask scan also finds the first unmasked KV tile and the MMA kernel skips the keys before it (811647b6c); encode()
  copies the pooled embeddings of a ubatch at once (064651d45). Queries (64 per 1024-token ubatch): 1226 -> 1335
  requests/s, index 98 -> 101 texts/s, outputs bit-identical.
- The GPU is only ~75% busy under query load: per ubatch the server spends ~2.6 ms sending results, ~1 ms building and
  uploading the n_tokens^2 mask and ~0.6 ms building the graph. q8_0 weights are no faster and lose accuracy.

### Other models (RTX 3090, q8_0 KV, llama-bench, tg128 / pp512)

Fixes found by the model sweep:

| commit | change | effect |
|---|---|---|
| 4bf1d215a | graph node budget >= 64 per layer | Phi-4 with a quantized V cache aborted (upstream too) |
| 2838afff4 | quantized MMA decode FA for head size 128, from 2048 keys | tg @ 16k: Phi-4 56.1 -> 61.9, Qwen3-14B 55.4 -> 68.3 |
| 1053dc199 | CUDA decode converts the KV cache whole below 1 GiB | tg @ 16k: gemma-4-26B-A4B 71.7 -> 117.3, Nemotron-3.5-Bolt-E4B 59.3 -> 108.4 |
| a9d242457 | Vulkan q4_K / q5_K MMVQ on Ampere+ again | Qwen3-14B Vulkan tg 69.9 -> 78.3 (upstream 76.3) |
| 6faea67e6, 44c5277a8 | Vulkan FA prefill f16 scratch on coopmat2 for head sizes > 256 (MLA: V reads K's rows) | pp @ 16k: GLM-4.7-Flash 91 -> 329, gemma-4-12b 1106 -> 1427, gemma-4-26B 1638 -> 2003 |

Speculative decoding, server, tg on a 20.7k wiki prompt / a short story prompt (CUDA; Vulkan in the sweep log):

| model | none | with draft |
|---|---|---|
| Qwen3.5-9B, MTP n 1 | 107 / 113 | 151 / 157 |
| Ornith-1.5-9B, MTP | 111 / 117 | 144 / 144 |
| gemma-4-12b, MTP (unsloth MTP Q8_0) | 72 / 78 | 104 / 116 |
| gemma-4-26B-A4B, MTP | 108 / 128 | 141 / 161 |
| Nemotron-3.5-Lightning-30B-A3B, MTP | 183 / 190 | 240 / 220 |
| Qwen3-14B, Eagle3 (third-party GGUF) | 64 / 78 | 50 / 74 (acceptance 0.21) |
| gpt-oss-20b, Eagle3 (third-party GGUF) | 141 / 187 | 113 / 116 (acceptance 0.25) |
| Nemotron-3.5-Bolt-E4B, DFlash | 100 / 123 | 111 / 107 (acceptance 0.23) |

The third-party gpt-oss DFlash GGUFs do not load (expected 123 tensors, got 91).

Open:
- Vulkan MLA FA (GLM) is still far behind CUDA: pp @ 16k 329 vs 1309, tg @ 16k 55 vs 93; the coopmat2 decode with q8_0
  KV is 3x slower than with f16 KV.
- Vulkan 2-column mat-vec (MTP verify) at ~560 GB/s for q4_K; needs a shader redesign.
- CUDA head size 512 decode (gemma-4) still converts the whole KV cache to f16 every token.

## Remaining gaps

- **Prefill at depth looks compute-bound on the MMA FA kernel.**
  - This is an estimate from the llama-bench times: at 64k depth the attention part of a 2048-token ubatch works out to roughly 50 TFLOPS f16. It was not profiled with ncu.
  - Ours and upstream are within 2%, so this round did not change it.
- **iq4_nl decode FA is still slower than q4_0.** At gqa 6 it takes 144 us against 120 us, because of the table lookups.
- **Vulkan 2-token verify at gqa 6** stays on the old path.
- **HIP:** the CUDA changes that could help HIP are untested on the RX 6900 XT.
