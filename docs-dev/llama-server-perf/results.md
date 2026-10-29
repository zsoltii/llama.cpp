# Measurement results

## B-baseline (brew 0.6.0, Vulkan, the exact user command)

`llama-server -hf unsloth/Qwen3.5-9B-MTP-GGUF:UD-Q4_K_XL -fa on -ctk iq4_nl
-ctv iq4_nl --parallel 1 -c 80000`, temp=0, 2026-10-16.

| case | prompt (tok) | pp tok/s | TTFT | tg tok/s (median) | peak VRAM | trunc |
|------|-------------|----------|------|-------------------|------------|-------|
| R1 greeting (32 gen) | 8 | 25.1 | 0.32 s | 41.66 | 7756 MiB | no |
| R2 MoE in Hungarian (512 gen) | 61 | 199.6 | 0.31 s | 37.46 | 7756 MiB | no |
| R3 60k document (256 gen) | 65578 | 1024.6 | 64.46 s | 32.79 | 7756 MiB | no |

Notes:
- tg drops with context (42 → 37.5 → 32.9).
- VRAM: 7756 MiB used / 11264 MiB → ~3.5 GB free, so there is room for tuning.
- Raw JSON: `~/llama-bench-results/results/B-baseline.json`.

## B-opt grid (brew 0.6.0, Vulkan)

| config | R1 tg | R2 tg | R3 pp | R3 tg | VRAM |
|--------|-------|-------|-------|-------|------|
| B-baseline | 41.66 | 37.46 | 1024.6 | 32.79 | 7756 |
| B-ub1024 | 47.33 | 43.79 | 944.0 | 30.40 | 7963 |
| B-ub2048 (-b 4096) | 45.30 | 42.52 | 924.6 | 30.38 | 8353 |
| B-nommproj (ub1024, mmproj CPU) | 45.42 | 42.47 | 929.7 | 30.56 | 6836 |
| B-mtp2 | 38.17 | 35.94 | 697.3 | 27.57 | 8468 |
| B-mtp3 | 34.83 | 31.50 | 742.8 | 23.99 | 8518 |

Lessons:
- Drift study (B-baseline2, same command repeated): R1 tg 47.39, R2 tg 43.38, R3 pp 964.2 / tg 32.37.
  The first baseline was the outlier; the ub1024 "improvement" is measurement noise (±5–15% between runs).
- **B-opt decision**: on brew + Vulkan there is no real ub gain above the noise.
  B-opt = baseline + `--no-mmproj-offload` (−1.1 GB VRAM for free).
- MTP with brew DROPPED: acceptance is good (mtp2 R3: 143/221 = 65%), yet it is slower.
  In 0.6.0 the draft overhead is larger than the verify gain.

## q4_1 re-measurement (evening of 2026-10-16, clock lock active)

BREAKTHROUGH on CUDA. `-ctk q4_1 -ctv q4_1` everywhere.

| config | R1 tg | R2 tg | R3 pp | R3 tg | VRAM |
|--------|-------|-------|-------|-------|------|
| B-base-q41 (brew) | 41.13 | 45.37 | 760.1 | 26.94 | 8506 |
| V-nommproj-q41 | 50.65 | 47.20 | 789.6 | 33.08 | 7275 |
| C-base-q41 | 62.08 | 66.11 | 1322.5 | 30.27 | 8803 |
| C-nommproj-q41 | 68.99 | 66.14 | 1325.8 | 30.21 | 7647 |
| C-mtp2-q41 | 70.49 | 73.34 | 1173.9 | **54.82** | 7814 |
| C-mtp3-q41 | 57.93 | 62.32 | 1176.4 | 54.34 | 7863 |

- pp champion: C-nommproj-q41 (1325.8, +42% over Vulkan).
- tg champion: C-mtp2-q41 (R3 54.82 = +81% over no MTP, +66% over Vulkan).
  Acceptance R3: 150/209 = 72%. n-max 3 is not better than 2.
- CUDA R3 tg without MTP (30.2) is weaker than Vulkan (33.1): long-context FA is slower on Turing. MTP more than makes up for it.
- B-base-q41 R2-rep0 stall (pp 4.7, TTFT 13 s): transient noise, the median is OK.

## Round 2: edges of the n-max sweep + CPU spin + large ubatch (CUDA, q4_1)
| config | R1 tg | R2 tg | R3 pp | R3 tg | VRAM |
|--------|-------|-------|-------|-------|------|
| C-mtp1 (n-max 1) | 64.43 | 77.79 | 1207.0 | 49.99 | 7763 |
| C-mtp4 (n-max 4) | 54.65 | 58.33 | 1194.5 | 48.32 | 7913 |
| C-mtp2-t1 (-t1 -tb1 --poll 0) | 62.78 | 75.97 | 1185.2 | **57.61** | 7793 |
| C-ub4096 (-b 8192 -ub 4096, no MTP) | 69.75 | 65.61 | **1346.3** | 31.34 | 8173 |

- n-max order by R3 tg: 2 (54.82) > 3 (54.34) > 1 (49.99) > 4 (48.32).
- `-t 1` helped in R3 (+5%) and dropped in R1; the long context is what matters.
- C-mtp2-ub4096: 48.44 tg / 1118.5 pp / 9773 VRAM → a large ub hurts with MTP.
- Decision: winner = **C-mtp2-t1** (mtp2 + t1/tb1/poll0, default batch). q4_1 KV stays.
  Next: code patches (BlockingSync, rm_kq).

## V-Head (source 0.7.0-dev build 11697, Vulkan, +`--offline`, iq4_nl KV)

| config | R1 tg | R2 tg | R3 pp | R3 tg | VRAM |
|--------|-------|-------|-------|-------|------|
| V-base | 47.76 | 42.96 | 451.2* | 29.66 | 7769 |
| V-nommproj | 46.03 | 42.60 | 929.4 | 30.58 | 6642 |
| V-mtp2 | 37.97 | 36.28 | 687.7 | 27.06 | 7341 |
| V-mtp3 | 33.72 | 30.90 | 766.2 | 23.76 | 7391 |
| V-sparsedis | 45.73 | 42.32 | 936.8 | 30.20 | 6642 |

\* The V-base R3 pp includes first-run pipeline compilation.
R1 rep0 1.3 s and R2 rep0 13.5 s were one-off costs; the driver cache only works from the 2nd process on.
Real V-Head: pp ~930, tg ~30.5.

Lessons:
- HEAD Vulkan ≈ brew (within the noise band). MTP slows HEAD down on Vulkan too.
- `FA_SPARSE_DISABLE` has no effect here. `--no-mmproj-offload` saves 1.1 GB for free.

## Round 3: combinations + q8_0 + ngram (CUDA, winner base)

| config | R1 tg | R2 tg | R3 pp | R3 tg | VRAM |
|--------|-------|-------|-------|-------|------|
| C-mtp2-ub2048 | 70.33 | 73.88 | 1212.5 | 52.76 | 8649 |
| C-q80 (q8_0 KV) | 60.02 | 69.73 | 1111.6 | 49.29 | 8361 |
| C-ngram (mtp+ngram-mod) | 64.89 | 69.97 | 1122.6 | 49.72 | 7815 |
| C-mtp2-ub4096 | 68.42 | 63.03 | 1118.5 | 48.44 | 9773 |

- A large ub hurts monotonically with MTP (52.8 / 48.4 vs 57.6).
- q8_0 KV is slower than q4_1 (bandwidth). The ngram chain hurts.
- q4_1 + the default batch stay.

## Code patch verdicts (measured)

| patch | result | decision |
|-------|----------|--------|
| CUDA `BlockingSync` | R3 tg 57.6 → 42.9 (−25%, reproducible) | REVERT |
| Vulkan `rm_kq=1` | pp +10%, but tg −11–14% in every case | REVERT |
| `-march=native` | not built (the CPU side is minimal with t1, no ROI) | DROPPED |

After both reverts the tree is clean (`git diff` empty).

## Round 4: SSL verification + kv-unified + f16 KV (CUDA)

| config | R1 tg | R2 tg | R3 pp | R3 tg | VRAM |
|--------|-------|-------|-------|-------|------|
| C-verify (winner repeated) | 65.54 | 74.35 | 1135.7 | 51.69 | 7773 |
| C-kvunified | 68.12 | 71.22 | 1122.5 | 51.90 | 7773 |
| C-f16kv | 75.64 | 71.91 | 1132.9 | **64.02** | 9221 |

- kv-unified: nothing (±noise) → the default stays.
- f16 KV: R3 tg +24% (51.69 → 64.02), R1 +15%.
  The dequant cost is larger than the extra bandwidth. VRAM 9221 MiB (2 GB free).
- Repeat (C-f16kv2): R3 tg **70.75**, pp 1191.5 → band 64–71, ROBUST.
  **NEW WINNER: f16 KV + mtp2 + t1.**

## Measurement noise: thermal drift (night of 2026-10-16, important lesson)

On free clocks the card throttles thermally at 87 °C.
Even **under** the `-lgc 1410,1650` lock it drops to 1365–1445 MHz on long loads (1650+ when cold).
Between sessions this causes an apparent ±20–25% difference on the same config:

- the same config (f16 + mtp2 + t1): R3 tg 64.0 / 70.8 (one session), then 51.1 / 51.3 (another, hot session).
- A fixed clock lock (`-lgc 1250,1250`) would need sudo, which we do not have. Instead, a controlled protocol:
  - interleaved A/B anchors in the same session;
  - 60 s cooldown between runs;
  - a per-run `nvidia-smi` clock log (`clocks-*.csv`, evaluated with `clockstat.py`).

So the "f16 ROBUST" claim of round 4 needs review: part of the +24% may have been session luck.
Round 6 (controlled) decides the q4_1 vs f16 question.

## Round 5 (interrupted — drift, low-value data)

| config | R1 tg | R2 tg | R3 pp | R3 tg | VRAM |
|--------|-------|-------|-------|-------|------|
| C-f16-mtp1 | 50.85 | 63.55 | 1036.1 | 51.14 | 9116 |
| C-f16-mtp3 | 49.53 | 54.57 | 1002.6 | 51.34 | 9216 |

- Hot session; the mtp1/mtp3 difference is within the noise.
- The round was interrupted in favour of the controlled protocol.

## Round 6: controlled A/B (interleaved, clock-logged, 60 s cooldown)

The GPU busy clock was stable between 1533–1577 MHz in every run (clocks-*.csv), so the comparisons are valid.

| config | R1 tg | R2 tg | R3 pp | R3 tg | VRAM |
|--------|-------|-------|-------|-------|------|
| C-a-q41 (q4_1 anchor) | 55.84 | 64.82 | 1021.0 | 46.83 | 7718 |
| C-a-q41b (q4_1 anchor 2) | 63.92 | 61.81 | 1006.9 | 47.03 | 7718 |
| C-b-f16 (f16 anchor) | 68.50 | 66.89 | 1045.8 | 58.38 | 9166 |
| C-b-f16b (f16 anchor 2) | 69.43 | 62.75 | 1031.6 | 54.69 | 9166 |
| C-b-f16c (closing anchor) | 69.26 | 65.22 | 1047.2 | 57.92 | 9166 |
| C-f16vec1 (VEC ≤1 tok) | 69.12 | 66.08 | 1045.0 | 55.09 | 9166 |
| C-f16vec3 (VEC ≤3 tok) | 63.20 | 64.59 | 1043.5 | **40.90** | 9166 |
| C-f16-bsampling (draft -bs) | 64.28 | 64.70 | 1040.9 | 58.24 | 9166 |
| C-f16-bs-both (draft+main -bs) | **72.41** | **71.87** | 1033.1 | **61.98** | 9198 |
| C-f16-pmin30 (p-min 0.3) | 59.15 | 60.42 | 1041.1 | 54.82 | 9166 |

- **f16 vs q4_1 (interleaved): R3 tg 46.9 ± 0.2 vs 56.5 ± 1.9 → +20%.**
  f16 wins under controlled conditions too. Round 4's +24% was real; averaged across sessions the effect is just smaller.
- **FA VEC kernel on Turing: negative.**
  - With VEC on the verify batch (3 tokens), R3 tg goes 58.4 → 40.9 (−26%). The MMA kernel is better; the stock heuristic is right.
  - Neutral on batch 1 (draft steps). Patch reverted.
  - The `GGML_CUDA_FATTN_DEBUG` log did not show in the server log (the server filters the plain ggml INFO lines), so the conclusion is based on behaviour.
- **`--spec-draft-backend-sampling -bs`: +9–13%** (R3 62.0 vs 54.7–58.4; R2 71.9 vs 62.8–66.9; R1 72.4).
  - The draft_n/acceptance numbers are bitwise identical (499/261, 214/147) → same tokens, faster sampling.
  - Partially (draft only) it is neutral → the gain is the sampling on the main path.
- `--spec-draft-p-min 0.3`: no gain → skipped.
- Note: `-bs` is an experimental switch; with a grammar or special samplers the server automatically falls back to CPU sampling.

## Round 7: confirmation (interleaved A/B, final binary, clean tree)

| run | R1 tg | R2 tg | R3 pp | R3 tg | VRAM |
|-----|-------|-------|-------|-------|------|
| C-r7-base1 | 67.11 | 66.26 | 1051.8 | 57.80 | 9166 |
| C-r7-bs1 | 73.37 | 73.14 | 1038.6 | **60.83** | 9198 |
| C-r7-base2 | 55.58 | 61.77 | 1034.4 | 55.45 | 9166 |
| C-r7-bs2 | 73.45 | 72.21 | 1031.2 | **60.68** | 9198 |

- `-bs` + `--spec-draft-backend-sampling`: R3 +7.6% / +9.4% (both pairs), R2 +10% / +17%, R1 +9% → goes into the final command.
- base2-R1 (55.58) is transient noise; the bs pairs are consistent with each other.

Vulkan fallback update (f16 KV):

| run | R1 tg | R2 tg | R3 pp | R3 tg | VRAM |
|-----|-------|-------|-------|-------|------|
| V-f16 | 43.02 | 43.21 | 910.0 | 30.06 | 8354 |
| V-f16-mtp | 33.67 | 35.53 | 721.4 | 25.92 | 9053 |

- V-f16 ≈ V-q4_1 (within the noise), but +1.6 GB VRAM → the Vulkan fallback stays on `q4_1` (more context headroom).
- MTP still hurts on Vulkan.

## Finale — FINAL (early morning of 2026-10-17, controlled measurements)

**Winning CUDA command (daily use):**

```
build-cuda/bin/llama-server -hf unsloth/Qwen3.5-9B-MTP-GGUF:UD-Q4_K_XL \
  -fa on -ctk f16 -ctv f16 --parallel 1 -c 80000 --no-mmproj-offload \
  --spec-type draft-mtp --spec-draft-n-max 2 -t 1 -tb 1 --poll 0 \
  --spec-draft-backend-sampling -bs
```

**Vulkan fallback (unchanged):**

```
build-vulkan/bin/llama-server -hf unsloth/Qwen3.5-9B-MTP-GGUF:UD-Q4_K_XL \
  -fa on -ctk q4_1 -ctv q4_1 --parallel 1 -c 80000 --no-mmproj-offload
```

## Final comparison (original vs winner)

- Original: brew 0.6.0 Vulkan, iq4_nl KV, the user's command.
- Winner: CUDA HEAD, f16 KV, mtp2, t1/poll0, -bs (controlled r7 measurements).

| | pp tok/s | tg tok/s (R3) | tg (R2) | VRAM |
|---|---|---|---|---|
| Original | ~960–1025 | ~32.8–33 | ~43–45 | 7756 |
| Winner (controlled) | ~1035 | **60.7** | **72.7** | 9198 |
| Winner (cool-session peak) | ~1190 | **70.8** | ~75 | 9221 |

- tg at long context: +85% (controlled) … +115% (peak) over the original.
- pp: ~1035 in the controlled session (~1190 in the cool session); it ranges 1010–1190 depending on clocks/heat.
- VRAM: 9198 MiB / 11264 → ~2 GB free; no OOM risk.
- Per the block list in `docs-dev/llama-server-perf/plan.md`, the `corpus/` files are untouched; at the end the tree has no local modifications.

## What stayed in at the end (code)

Nothing. Every code experiment was either ineffective or harmful: BlockingSync (−25%), Vulkan rm_kq=1 (−11% tg), FA VEC on Turing (−26% R3).
All gains came from configuration: KV type, MTP, threads/poll, where sampling runs, mmproj offload.

> **Update (phase 2b, 2026-10-19):** this section is about the earlier (Oct 17) campaign.
> In phase 2b **two code changes stayed in**, both confirmed by measurement and default-on:
> - the `GGML_CUDA_FA_QUANTS` += `q4_1-q4_1` build option (non-MTP R3 +59%);
> - the vectorized q4_1→f16 conversion kernel in `convert.cu` (MTP R3 +9%).
>
> Details: rounds 13–15 + the "Phase 2b" section of `plan.md`.

## Round 8 — GRAPH_OPT code switch (morning of 2026-10-17)

Another code-level candidate: `GGML_CUDA_GRAPH_OPT=1`, a stream-overlap optimization that exists in the tree but is off by default. It is an env var, so it can be measured without a rebuild.

| run | R1 tg | R2 tg | R3 pp | R3 tg | VRAM |
|-----|-------|-------|-------|-------|------|
| C-r8-base1 (cold start) | 73.78 | 75.28 | 981.1 | **68.87** | 9201 |
| C-r8-gopt1 | 71.60 | 70.95 | 1037.9 | 62.06 | 9215 |
| C-r8-base2 (warm) | 65.10 | 69.36 | 1032.7 | 61.40 | 9201 |
| C-r8-gopt2 (warm) | 72.60 | 71.60 | 1036.6 | 61.44 | 9215 |

- Warm pair in the same thermal state (base2 vs gopt2): R3 tg 61.40 vs 61.44 → **GRAPH_OPT brings no gain**; dropped.
- The ~11% between the cold base1 (68.87) and the warm runs (~61.4) is thermal drift. This motivated the new regime with a fixed power cap.
- Power under load in this round: average ~190 W, peaks 258–289 W; 81–88 °C.

## New measurement regime: fixed 170 W power cap (2026-10-17)

- Card limits: default 250 W, max 280 W, min 100 W → **170 W** set.
  Needs root: `nvidia-smi -pl 170 -i 0`; permanent systemd service: `nvidia-powerlimit.service`.
- Goal: remove the 81–88 °C thermal throttling and the ±15–20% spread between sessions. Under the cap the temperature settles low.
- The clock lock (lost with a reboot) does NOT come back; the power cap is the only regulator (simpler, more deterministic).
- Numbers from this regime form their own reference; do not compare them in absolute terms with the earlier (throttling) measurements.
- **Uniform measurement standard: q4_1 KV** (CUDA and Vulkan, every run); iq4_nl is not used.
  The f16 CUDA advantage (+20%) is documented from the old regime but suspended for comparability.
- Done: the r9 reference series under 170 W with q4_1 KV:
  - win ×3 (stability);
  - nomtp/nobs (the effect of MTP and `-bs`);
  - Vulkan base and Vulkan MTP;
  - brew anchor.
  See the round 9 section.

## Round 9 — 170 W reference, uniform q4_1 KV (evening of 2026-10-17)

8/8 runs rc=0, every run with q4_1 KV.
Run order: win1, nomtp, nobs, win2, V-base, V-mtp, brew-anchor, win3.

**CUDA (build-cuda, unpatched reference binary):**

| run | config | R1 tg | R2 tg | R3 pp | R3 tg | VRAM |
|---|---|---|---|---|---|---|
| win1 | q4_1+MTP+BS | 58.6 | 74.9 | 1165 | 48.1 | 7945 |
| win2 | q4_1+MTP+BS | 61.2 | 76.4 | 1160 | 56.3 | 7947 |
| win3 | q4_1+MTP+BS | 64.0 | 64.7 | 1005 | 47.1 | 7958 |
| nomtp | q4_1 | 53.6 | 52.5 | 1131 | 23.8 | 7215 |
| nobs | q4_1+MTP (no BS) | 68.7 | 68.0 | 1075 | 52.9 | 7909 |

**Vulkan / brew (0.6.0):**

| run | config | R1 tg | R2 tg | R3 pp | R3 tg | VRAM |
|---|---|---|---|---|---|---|
| V-base | q4_1 | 41.3 | 40.0 | 805 | 29.2 | 6815 |
| V-mtp | q4_1+MTP+BS | 35.6 | 37.3 | 714 | 25.7 | 7575 |
| B-brew | q4_1 | 43.0 | 40.3 | 801 | 29.0 | 7953 |

**Findings:**

- MTP effect (CUDA): long-context tg ~2× (win 47–56 vs 23.8), R2 tg +25–45%, pp neutral (1131 → 1160, within run noise). MTP stays.
- The draft acceptance is **bitwise identical** in every CUDA MTP run (0.71770; 150/209; mean length 2.43).
  The R3 tg spread of the win anchors (47–56, ±9%) is operating-state noise (clocks/heat under the cap), not an acceptance difference.
  So the r10 A/B will be interleaved.
- BS (`-bs`) vs without: R1/R2/R3 all favour nobs (except R2: win1/win2 ~75–76 vs nobs 68).
  Because of the ±9% spread there is no clear verdict. The win config keeps BS; the earlier controlled r7 measurements showed the same.
- Vulkan: MTP **hurts** (R3 tg 29.2 → 25.7; pp 805 → 714; R1/R2 tg too), and acceptance is weaker on VK (mean length 2.17 vs 2.43).
  The VK base beats CUDA without MTP in long-context tg (29.2 vs 23.8), but is far worse in short-context tg and pp.
- brew 0.6.0 ≈ build-vulkan (pp 801 vs 805, tg 29.0 vs 29.2): the brew anchor is stable, and the two Vulkan builds are practically the same.
- Power regime in this round: average ~150 W, short peaks 176–193 W, max 86 °C, average busy clock 1430–1530 MHz.
  The cap actively limits in 25–50% of the busy samples.
- VRAM with q4_1 KV: 7.2–8.0 GB (f16: 9.2 GB) → ~1.3 GB more free space.

**Informational comparison with the old regime** (not clean pairs: f16↔q4_1 and the cap are mixed):
- r8 warm f16 without the cap: pp ~1036 / tg ~61.4.
- r9 warm q4_1 under 170 W: pp ~1160–1165 / tg ~48–56.

pp improved (+12%) and tg dropped. Separating the two (f16 vs q4_1 under the cap) would be a separate measurement.

## Round 10 — CUDA MMVQ↔MMQ decode A/B (night of 2026-10-17)

An env-gated experimental patch (`GGML_CUDA_DISABLE_MMVQ=1` → MMQ instead of MMVQ for decode batches), interleaved win/mmq pairs, q4_1 + MTP + BS, 170 W.
Without the env the patch is inactive, so the win runs measure the reference path.

| run | config | R1 tg | R2 tg | R3 pp | R3 tg | VRAM |
|---|---|---|---|---|---|---|
| C-r10-win1 | MMVQ (default) | 80.1 | 83.4 | 1240 | 57.4 | 7800 |
| C-r10-mmq1 | MMQ (forced) | 66.1 | 69.5 | 1138 | 50.7 | 7966 |
| C-r10-win2 | MMVQ (default) | 73.5 | 75.5 | 1139 | 53.5 | 7888 |
| C-r10-mmq2 | MMQ (forced) | 62.9 | 68.7 | 1126 | 48.1 | 7890 |

Per-pair difference (MMQ − MMVQ): R1 −17% / −15%, R2 −17% / −9%, R3 tg −12% / −10%. pp is within run noise (MMVQ only affects batches ≤8).

- **Verdict: the MMVQ (default) path wins on every metric, in both pairs.**
  The heuristic that is "untuned" for Turing actually picks the right path for decode batches.
  No code change stays (patch reverted, `mmvq.cu` clean).
- Under MMQ the draft acceptance dropped too (mean length 2.29 vs 2.43).
  The different kernel numerics give a different argmax, which explains part of the tg loss; the rest is kernel speed.
- The r10 win numbers are at the top of the r9 band (R1/R2 tg 73–83 vs 58–76). That is session-level variance; the interleaved pairs make the A/B valid.
  For every mmq run the env passing was checked in the running server's `/proc/<pid>/environ`.

## Round 11 — Vulkan: isolating MTP + env switches (early morning of 2026-10-18)

6 runs, q4_1 KV, 170 W, `build-vulkan`. Anchors: base vs base2.

| run | config | R1 tg | R2 tg | R3 pp | R3 tg | VRAM |
|---|---|---|---|---|---|---|
| V-r11-base | q4_1 (default) | 47.0 | 46.9 | 901 | 31.6 | 6605 |
| V-r11-base2 | q4_1 (default) | 45.7 | 46.1 | 910 | 31.2 | 6605 |
| V-r11-spec | +MTP (no BS) | 35.4 | 38.6 | 741 | 26.6 | 7353 |
| V-r11-fmmvq | `FORCE_MMVQ=1` | 41.7 | 40.0 | 897 | 29.0 | 6605 |
| V-r11-dmmvq | `DISABLE_MMVQ=1` | 47.5 | 47.6 | 896 | 31.2 | 6610 |
| V-r11-nodes | `MAX_NODES=1000` | 46.9 | 45.0 | 912 | 31.5 | 6605 |

Findings:

- The anchors are within ±2%; run-to-run precision on Vulkan is excellent.
- **MTP hurts on Vulkan on its own** (without BS: R3 tg −16%, R2 −18%, pp −18%).
  So backend sampling is not at fault; the VK spec path is inherently slower, and the ~2× MTP gain measured on CUDA does not show up on VK. MTP on VK dropped.
- `GGML_VK_FORCE_MMVQ=1` hurts (R2 −15%, R3 −8%): the VK auto-mix (n=1, Q4_K, k≤4096 → MMQ) is better than forced MMVQ.
- `GGML_VK_DISABLE_MMVQ=1` and `GGML_VK_MAX_NODES_PER_SUBMIT=1000` are neutral (±1–3%) → nothing to tune.
- **VK optimum: base (no MTP)**, but it trails CUDA+MTP (R3 tg 31.6 vs 48–57; pp 901 vs 1140–1240; R2 tg 47 vs 68–83).

## PCIe link under load (2026-10-18)

- An 18-minute sample (every 2 s) during the r11 runs: 375 samples at **Gen3 x8**, 161 at Gen1 (idle/ASPM), 14 at Gen2.
  Max capability: Gen3 x16; the link's current width is x8 (physical/slot limit).
- Conclusion: under load it goes up to Gen3, and x8 is plenty for token-sized GPU↔CPU traffic → **no GPU↔CPU bottleneck**; nothing to do here.

## Round 10b — Turing MMVQ nwarps variant (dropped, early morning of 2026-10-18)

Tuning the decode path of the Turing MMVQ table: the k-quants (Q2_K..Q6_K) get 2 warps at `ncols_dst=1`; the variant gives them 4 (like the other types).
A compile variant + a separate, clean reference binary (`bin-cuda-ref/`), interleaved ref/var pairs, win config, q4_1, 170 W.

| run | R1 tg | R2 tg | R3 pp | R3 tg | VRAM |
|---|---|---|---|---|---|
| C-r10b-ref1 | 64.9 | 77.9 | 1193 | 55.3 | 7737 |
| C-r10b-var1 | 72.6 | 76.0 | 1148 | 53.7 | 7737 |
| C-r10b-ref2 | 73.2 | 75.1 | 1149 | 54.4 | 7737 |
| C-r10b-var2 | 73.5 | 76.9 | 1139 | 54.4 | 7737 |

- Per-pair difference (var − ref): R1 +11.8% / +0.4%, R2 −2.4% / +2.4%, R3 tg −2.9% / ±0.0% → **no consistent direction**.
  var1's R1 gain comes from ref1 being a first-run outlier (64.9; the other three are 72.6–73.5).
- The acceptance is bitwise identical in all 4 runs (0.71770, mean length 2.43).
  The variant is numerically the same, only the warp split differs; the tg differences are pure run noise.
- **Verdict: no measurable gain → variant dropped** (source `git checkout`, `build-cuda` rebuilt from clean source).
- Methodology note: the md5 of `libggml-cuda.so.0.27.0` rebuilt from the same source differs per build.
  But per `cmp` only 2 bytes differ (the PID in the nvcc temp file name), so the build is practically deterministic and the difference is cosmetic.

## Round 12 — CPU affinity / NUMA pinning (early morning of 2026-10-18)

`taskset` wrappers: pin0 = cores 0-11 (NUMA node 0), pin1 = 12-23 (node 1).
The pinning was confirmed during the run with `taskset -cp` (0-11 and 12-23 active).
Win config, q4_1, 170 W; anchors: base and base2 (no pinning).

| run | R1 tg | R2 tg | R3 pp | R3 tg |
|---|---|---|---|---|
| C-r12-base | 70.9 | 76.4 | 1153 | 55.4 |
| C-r12-base2 | 66.1 | 77.0 | 1145 | 55.8 |
| C-r12-pin0 | 73.2 | 76.8 | 1176 | 55.1 |
| C-r12-pin1 | 73.9 | 76.6 | 1158 | 55.3 |

- Anchors: R2/R3/pp ±1%, tight; R1 ±5–7% (the usual spread).
- pin0/pin1 vs the anchor mean: R2 ±0.2%, R3 tg −0.6..−0.9%, pp within the noise.
  → **NUMA pinning is neutral on the long-context metrics**, and the choice of node does not matter either (node0 ≈ node1).
- On R1 the two pinned runs are +7–8%, but that is within R1's ±7% spread (the same config ranged 58.6–80.1 during r9/r10).
  Not decisive; affinity does not go into the daily command.
- Acceptance identical in all 4 runs (0.71770, mean length 2.43).
- The 170 W regime is stable: avgP 148.5–149.6 W, maxT 80–84 °C.

## Round 13 — native q4_1 FA VEC kernels A/B (morning of 2026-10-19)

`GGML_CUDA_FA_QUANTS` += `q4_1-q4_1` (build-cuda). ref = the clean Oct 17 runtime (`bin-cuda-ref`), var = the new build.

**Lib isolation:**
- The ref binary's RUNPATH points to the live `build-cuda/bin`, so every run started with `LD_LIBRARY_PATH=$(dirname $bin)`.
- Evidence:
  - `LD_TRACE_LOADED_OBJECTS`: ref → bin-cuda-ref, var → build-cuda/bin;
  - the server log line `no FlashAttention vector kernel compiled ... q4_1-q4_1`: 1 in ref runs, 0 in var runs.

Win config with MTP and without MTP (no `--spec-*`); 170 W, q4_1 KV, interleaved.

| run | R1 tg | R2 tg | R3 pp | R3 tg | R3 acc |
|---|---|---|---|---|---|
| C-r13-ref1 | 67.3 | 76.6 | 1148 | 55.7 | 0.7177 |
| C-r13-var1 | 73.0 | 75.5 | 1166 | 54.2 | 0.7177 |
| C-r13-ref2 | 72.8 | 73.6 | 1130 | 54.5 | 0.7177 |
| C-r13-var2 | 74.5 | 76.6 | 1120 | 53.9 | 0.7177 |
| C-r13-var3 | 67.1 | 75.9 | 1141 | 54.7 | 0.7177 |
| C-r13-refN (non-MTP) | 57.6 | 61.8 | 1301 | **26.6** | – |
| C-r13-varN (non-MTP) | 63.7 | 62.4 | 1311 | **42.3** | – |

**Verdict: the q4_1-q4_1 FA VEC kernels STAY.**
- Non-MTP, long context: R3 tg 26.6 → 42.3 t/s (**+59%**).
  Until now batch-1 decode converted the whole KV to f16 on every step (the VEC kernel had no q4_1 instance); now it reads it natively.
- MTP: no real change (54–56 t/s, −1..−3% within the noise).
  The verify batch=3 goes to MMA on Turing, which is f16 only, so it still converts.
- pp: no change (1120–1166 with MTP, 1301–1311 without).
- The acceptance is bitwise identical (0.7177 / 0.5313 / 0.4783) → the numerics did not change, only the conversion disappeared.

Side observation: the r9 non-MTP R3 tg (23.8) and the current clean ref (26.6) differ. That is session-level spread; the A/B within the same session (26.6 vs 42.3) is what counts.

## Round 14 — forcing the FA kernel + conversion kernel (late morning of 2026-10-19)

7 runs on the q4_1 FA build (win config + MTP, 170 W, q4_1 KV).

| run | R1 tg | R2 tg | R3 pp | R3 tg | note |
|---|---|---|---|---|---|
| C-r14-base | 64.4 | 74.6 | 1124 | 55.1 | anchor |
| C-r14-vec4 | 73.6 | 77.6 | 1148 | **41.1** | `vec:4` — MTP verify batch=3 → VEC |
| C-r14-convfast | 73.6 | 75.6 | 1164 | **59.6** | vectorized q4_1→f16 conversion |
| C-r14-mmqprec | 74.4 | 76.3 | 1139 | 55.0 | `MMQ_PREC=q4` |
| C-r14-ub1024 | 72.9 | 74.3 | 1140 | **50.8** | `-ub 1024` |
| C-r14-mma1 | 74.2 | 76.7 | 1163 | 54.8 | `mma:1` control |
| C-r14-base2 | 74.2 | 74.9 | 1128 | 54.2 | anchor |

**Verdict:**
- **`vec:4` REJECTED** (R3 tg −25%).
  - For the batch-3 verify, VEC (SIMT) is much slower than MMA (tensor cores); saving the f16 conversion does not make up for it.
  - It has a numerical side effect too: R3 acceptance 0.71770 → 0.70283 (VEC and MMA sum in a different order).
    The speed loss does not come from that, though (mean length only went 2.43 → 2.41).
- **`CONV_FAST` WON (+8.1% R3 tg, +3.5% pp).**
  - The current q4_1→f16 conversion kernel writes 8× 2-byte stores (half-efficient sectors); the v2 kernel does one 32-bit load + 2× 8-byte stores.
  - Correctness: detcheck (MMA forced, same prompt + n_predict, temp=0) → **bitwise identical output** with the old and the new kernel.
- `mmqprec=q4` neutral (55.0); `-ub 1024` hurts (−7.7%, VRAM +276 MiB); `mma:1` neutral (the draft's batch-1 decodes run on a short KV).
- R1 is very noisy (64.4 vs 74.2 on the two anchors); R2/R3 are what count.

**Methodology note (log trap):**
- The `GGML_LOG_WARN` of the FA-forcing switch did not show in the server log.
- The wrapper's static initialization happens **during model loading** (0.5 s), where ggml log messages are lost; `fprintf(stderr, ...)` always shows up.
- Meanwhile the switch's effect was proven by behaviour (acceptance difference + tg).

**Next (r15):** the v2 conversion kernel set as default (`GGML_CUDA_CONV_LEGACY=1` for the old one); a confirming A/B on the same binary.

## Round 15 — confirming A/B of the conversion kernel (late morning of 2026-10-19)

The same binary (md5 25fe9bb9…), two branches: `GGML_CUDA_CONV_LEGACY=1` (old kernel) vs default (v2 vectorized). Interleaved, 170 W, q4_1 KV.
Evidence: the "legacy (unvectorized) kernel" line only appears in the legacy runs (1 vs 0), so the kernel choice is confirmed.

| pair | R3 pp | R3 tg | d tg | acc |
|---|---|---|---|---|
| legacy (1) | 1146 | 54.32 | – | 0.7177 |
| base (2) | 1160 | 58.24 | **+7.2%** | 0.7177 |
| legacy2 | 1115 | 54.24 | – | 0.7177 |
| base2 | 1144 | 60.01 | **+10.7%** | 0.7177 |
| legacyN (non-MTP) | 1321 | 41.20 | – | – |
| baseN (non-MTP) | 1298 | 41.95 | +1.8% | – |

- **The v2 conversion kernel stays the default**: MTP R3 tg +7.2…+10.7% (~+9% average), pp +1…+3%.
  Non-MTP decode uses no conversion (native VEC), so the difference there is noise.
- The acceptance is bitwise identical (0.7177): the conversion gives numerically the same result (detcheck showed this too).

## CPU profile — where does the CPU time go? (2026-10-19 10:00, cpuprobe)

Method:
- sample `/proc/<pid>/stat` utime+stime before and after each request, broken down per thread;
- R1/R2/R3 × 3 repetitions (cache off + cached);
- MTP win config with `-bs` (A) and without (B), 170 W.

| case | wall s | CPU s | CPU% | cpu_ms/token |
|---|---|---|---|---|
| R1 cached | 0.62–0.72 | 0.63–0.73 | ~102% | 19.7–22.8 |
| R2 cached | 6.7–7.0 | 6.9–7.2 | ~102% | 13.5–14.0 |
| R3 cached (65k) | 4.9–5.3 | 4.9–5.4 | ~101% | 19.3–20.9 |
| R3 nocache (65k pp) | 59.6 | 60.2 | ~101% | 235 |

- The main thread's CPU time ≈ the wall time in **every** case, so the thread runs practically 100% of the time.
  But that is **sync spin** (CUDA event polling between phases), not real work. Writing the KV mask (~0.1 ms per step) and tokenization are negligible next to it.
- **Without** `-bs` (B): R2 15.3–16.3 ms/token, wall time 7.6–8.1 s.
  With `-bs` (A): 13.5–14.0 ms/token, 6.7–7.0 s.
  So the CPU sampler costs ~1.5–2 ms/token, which on R2 is **+12.8% wall time** (average 6.85 s vs 7.85 s). `-bs` matters and stays in the daily config.
- Conclusion: there is nothing to win with CPU-side micro-optimization (mask, CPU sampler).
  - The GPU is the bottleneck; the CPU is "always busy" because of the sync spin.
  - A bigger restructuring could have theoretical headroom: fewer host syncs in the MTP step (currently ~4 graph executions per step).
    But that is not a quick, risk-free gain.

## Round 16 — Vulkan env switches (late morning of 2026-10-19)

9 runs, build-vulkan (md5 a0fcf84e, with the `GGML_VK_RM_KQ` evidence log), non-MTP base config, q4_1 KV, 170 W.

**Measurement anomaly:**
- In some runs the R3 *cached* phase (rep1/rep2) degraded internally (base2: 31.0 → 28.3 → 27.9, i.e. the median −10%), while the clocks changed by only ~2%.
- So the evaluation is based on the **rep0** data, which is the same in every run: a fresh 65k pp + 256 generated tokens.
| run | R3 pp (rep0) | tg rep0 | tg rep1 | tg rep2 | verdict |
|---|---|---|---|---|---|
| base | 914 | 31.49 | 31.60 | 31.78 | anchor |
| nocm2 | **432** | **29.43** | 29.61 | 29.51 | REJECTED (pp halves without coopmat2) |
| nodot2 | 895 | 31.10 | 29.33 | 29.05 | neutral / slight loss |
| nomadd | 905 | 31.50 | 31.65 | 31.61 | neutral |
| nodint | 908 | 31.34 | 31.11 | 29.68 | neutral |
| memprio | 886 | 30.36 | 30.12 | 31.18 | neutral (noise) |
| nodes32 | 892 | 31.96 | 31.91 | 32.46 | +1.5% on rep0 → within the noise |
| rmkq1 | 886 | **29.80** | 29.44 | 29.56 | REJECTED (−5%) |
| base2 | 889 | 31.02 | 28.27 | 27.87 | anchor (rep0 drift −1.5%) |

**Verdict:** the Vulkan base config stays.
- The coopmat2 and dot2 paths are useful (switching them off hurts).
- `rm_kq=1` hurts (−5%); the other switches are neutral.
- In-session rep0 drift: tg ~1.5%, pp ~2.7%.

## Phase 2b — FINAL winning configuration (2026-10-19)

> **UPDATE (2026-10-20, Part 1):** the KV type changed **q4_1 → f16** (+14–16% with MTP, +12–17% without; see the
> "Part 1 — FINAL configuration" section). The q4_1 command below stays as the **VRAM-constrained** variant.

**CUDA (daily command, 170 W, q4_1 KV, MTP):**

```
build-cuda/bin/llama-server -hf unsloth/Qwen3.5-9B-MTP-GGUF:UD-Q4_K_XL \
  -fa on -ctk q4_1 -ctv q4_1 --parallel 1 -c 80000 --no-mmproj-offload \
  --spec-type draft-mtp --spec-draft-n-max 2 -t 1 -tb 1 --poll 0 \
  --spec-draft-backend-sampling -bs
```

Measured (r15, same binary):
- MTP: R3 tg **58.2–60.0 t/s**, R3 pp 1144–1160;
- non-MTP: R3 tg **42.0 t/s**, pp 1298.

**Vulkan (fallback, no MTP):**

```
build-vulkan/bin/llama-server -hf unsloth/Qwen3.5-9B-MTP-GGUF:UD-Q4_K_XL \
  -fa on -ctk q4_1 -ctv q4_1 --parallel 1 -c 80000 --no-mmproj-offload
```

R3 tg 31.5 t/s, pp 914.

**Code changes that stayed in (confirmed by measurement, default-on):**

1. `GGML_CUDA_FA_QUANTS` += `q4_1-q4_1` (build option): native q4_1 VEC FA kernel; non-MTP R3 tg 26.6 → 42.3 (+59%, r13).
2. Vectorized q4_1→f16 conversion kernel (`ggml/src/ggml-cuda/convert.cu`): MTP R3 tg +8…+11% (r14/r15). `GGML_CUDA_CONV_LEGACY=1` gives the old one.

**Env switches (default off):** `GGML_CUDA_DISABLE_MMVQ`, `GGML_CUDA_BLOCKING_SYNC`, `GGML_CUDA_FA_FORCE_KERNEL=vec|tile|mma[:N]`,
`GGML_CUDA_CONV_LEGACY`, `GGML_CUDA_MMVQ_TURING_KQ_NWARPS4` (compile time), `GGML_VK_RM_KQ=1..4`.

**Pending sudo tasks:**

```bash
sudo install -m 755 ~/llama-bench-results/nvidia-powerlimit.sh  /usr/local/bin/nvidia-powerlimit.sh
sudo install -m 644 ~/llama-bench-results/nvidia-powerlimit.service /etc/systemd/system/nvidia-powerlimit.service
sudo systemctl daemon-reload
sudo systemctl enable --now nvidia-powerlimit.service
```

(Without this the 170 W limit only lasts until the next boot.)

**Build configuration (CUDA):** the `build-cuda` cache contains `-DGGML_CUDA_FA_QUANTS=q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16;q4_1-q4_1`.
For a fresh build directory it must be given again; otherwise the native q4_1 FA kernel is not compiled in.

**Where the source changes are (uncommitted, working tree):**
- `ggml/src/ggml-cuda/convert.cu` (v2 kernel);
- `ggml/src/ggml-cuda/fattn.cu` (FA forcing + evidence log);
- `ggml/src/ggml-cuda/mmvq.cu` + `ggml-cuda.cu` (P0 env switches);
- `ggml/src/ggml-vulkan/ggml-vulkan.cpp` (RM_KQ).

## Conversion kernel micro-benchmark — v3 dropped (afternoon of 2026-10-19)

An isolated CUDA micro-benchmark (`/tmp/opencode/convbench.cu`, sm_75) on the real MTP-verify conversion volume:
65 536 rows × 256 dim × 8 layers × 2 tensors = 537 MB of f16 output from 168 MB of q4_1 input.

| kernel | time | effective bandwidth |
|---|---|---|
| legacy (8× 2B store) | 2.103 ms | 335 GB/s |
| v2 (current default) | **1.819 ms** | 387 GB/s |
| v3 (1× 16B store) | **1.819 ms** | 387 GB/s |
| memcpy D2D reference | 2.629 ms | 408 GB/s |

- v2 is **practically at the memory bandwidth limit** (387 vs 408 GB/s); v3 (16-byte stores) **adds nothing** (0.0%) → not adopted.
- The full conversion set per step is ~1.82 ms (instead of the earlier ~2.9 ms estimate), i.e. ~4.4% of the 41 ms MTP step.
- Conclusion: the conversion direction is closed. Further CUDA gains would need the verify batch's mat-vec kernel (MMVQ, ~240 GB/s effective), which is deep kernel work.

## Round 17 — Vulkan rm_kq 3/4 (evening of 2026-10-19)

The Turing default is `rm_kq=2`. In r16 `1` lost; 3/4 had not been measured yet.
4 runs, base config, q4_1, 170 W. The evidence log was correct (rm_kq_log 70 in the variants, 0 in the bases).

| run | R3 pp (rep0) | R3 tg rep0 | rep1 | rep2 |
|---|---|---|---|---|
| base | 826.7 | 28.61 | 28.57 | 29.03 |
| rmkq3 | 813.4 | 28.95 | 28.63 | 28.94 |
| rmkq4 | 811.0 | 28.67 | 28.89 | 28.75 |
| base2 | 849.4 | 29.75 | 32.36 | 28.57 |

**Verdict:** rm_kq 3/4 is neutral (+1.2%/+0.2% is within the +4% deviation of the base2 anchor) → **no code change**; the default 2 stays.

**Environmental note:**
- In this (evening) session the absolute VK numbers are ~10% lower than in the morning (28.6 vs 31.6), while the SM clock is *higher* (1533 vs 1497 MHz).
- So the difference is not clocks/temperature but external interference/latency.
- **Only compare absolute numbers within the same session**; the verdicts come from the in-round rep0 values and anchors.

## Round 18 — MMQ routing for the MTP verify batch (evening of 2026-10-19)

The dominant component of the CUDA MTP step is the verify (batch 3) mat-vecs: ~23 ms of the 41 ms step.
- On Turing the `ne11 <= 8` rule runs MMVQ for it.
- The new env-gated `GGML_CUDA_MMQ_MIN_BATCH=3` switch can route it to MMQ (tensor cores).
- This had not been measured separately before; the earlier MMQ test also affected the batch-1 decodes.

| run | R3 pp | R3 tg |
|---|---|---|
| base | 1184 | 59.90 |
| mmq3 | 1144 | **52.05 (−13.1%)** |
| base2 | 1175 | 59.52 |
| mmq3b | 1125 | **51.90 (−12.8%)** |

**Verdict:** MMQ is **13% slower** for the batch-3 verify (the large MMQ tile is wasteful at 3 columns) → MMVQ stays.
- The switch stays in as an env-gated experimental tool (default off), like the other P0 switches.
- Evidence: routing log 1 in the mmq runs, 0 in the bases; acceptance identical (0.4375).

**Tool note:**
- A kernel-level nsys profile is **not available** on this machine.
- The installed nsys 2023.4 lacks the QDSTRM importer ("importer binary and its dependencies were not found"); fixing it would need sudo/installation. It does not accept the `--` separator either.
- So the decomposition was done by subtraction/ablation.

## Round 19 (Part 1 / WP-B1) — KV type: f16 vs q4_1 (2026-10-20)

8 runs, MTP and no-MTP, interleaved q4_1↔f16 pairs + anchors, 170 W, the same binary (md5 `c05aa45e`), CUDA.
Evidence: the CMD lines (-ctk) + VRAM.

| config | R3 tg rep0 | median | R3 pp |
|---|---|---|---|
| MTP q4_1 | 65.27 | 64.15 | 1232.9 |
| **MTP f16** | **74.56 (+14.2%)** | 74.56 (+16.2%) | 1271.0 |
| MTP q4_1 (2) | 57.44 | 57.54 | 1147.6 |
| **MTP f16 (2)** | **65.77 (+14.5%)** | 66.03 (+14.8%) | 1134.8 |
| no-MTP q4_1 | 41.07 | 41.03 | 1293.1 |
| **no-MTP f16** | **47.95 (+16.8%)** | 47.82 (+16.6%) | 1314.9 |
| no-MTP q4_1 (2) | 41.42 | 41.40 | 1295.0 |
| **no-MTP f16 (2)** | **46.39 (+12.0%)** | 47.32 (+14.3%) | 1301.4 |

- **Verdict: f16 KV is +14–16% with MTP and +12–17% without**.
  pp unchanged, acceptance bitwise identical (0.4375); the rep spread is small (MTP f16: 74.56/72.78/75.12).
- **Cost: +1.42 GB VRAM** (MTP 7876→9316 MB; no-MTP 7162→8610 MB). At 80k it fits in 11.2 GB, but the headroom shrinks to ~1.9 GB.
- Cause:
  - with q4_1 KV the batch-1 path is SIMT VEC with per-element dequant; with f16, VEC/MMA work in f16 without dequant;
  - the full-KV f16 conversion of the MTP verify (700 MB per step) disappears.
- **B2 (CUDA KV shadow) DROPPED.** An f16 shadow copy on top of the q4_1 cache would be +2.1 GB (~2.75 GB KV in total).
  That is worse than plain f16 KV (+1.42 GB), and the f16 path is simpler and faster.
- **Daily config: `-ctk f16 -ctv f16`** (q4_1 stays for VRAM-constrained / larger-context cases).
- Note: the current session is ~10% faster than the earlier evening one (q4_1 MTP 64–65 vs 59–60), so only in-round pairs are valid.

## Round 20 (Part 1 / WP-E) — n-max and -ub re-measured with f16 KV (2026-10-20)

8 runs, MTP, f16 KV (the new winner), interleaved anchors (an nmax2 run next to every variant), 170 W, the same binary.

| run | R3 tg rep0 | median | pp |
|---|---|---|---|
| nmax2 (anchor) | 66.43 | 67.05 | 1166.4 |
| nmax3 | 60.88 | 60.88 | 1143.7 |
| nmax2b (anchor) | 65.21 | 65.97 | 1167.0 |
| nmax3b | 59.32 | 60.03 | 1132.4 |
| nmax1 | 64.04 | 66.88 | 1146.9 |
| nmax2c (anchor) | 64.22 | 64.60 | 1136.7 |
| ub1024 | 63.49 | 63.49 | 1140.7 |
| nmax2d (anchor) | 64.62 | 65.12 | 1143.3 |

- **n-max 3: −8.4% / −9.0% → dropped** (confirms the earlier order 2 > 3).
- **n-max 1: rep0 −0.3% (median +3.5%, within the 3.8% anchor spread) → neutral.**
  The measured benefit of the 2nd draft position (mean length 1.88 vs 1.55) just balances its step cost.
  **n-max 2** stays (more robust under varying acceptance).
- **`-ub 1024`: −1.7% / −2.5%, VRAM +316 MB → dropped** (confirms the direction of the earlier −7.7%).
- Every run's acceptance is consistent (nmax2: 0.4375; nmax1: 0.55; nmax3: 0.3125).

## Round 21 (Part 1 / WP-C0) — GEMV micro-benchmark: the 160 B format DROPPED (2026-10-20)

Goal: decide whether wider (8/16 B) loads raise the GEMV bandwidth, which was the justification for the 160 B aligned q4_K format.
Method:
- a standalone kernel (`/tmp/opencode/gemvbench.cu`, sm_75);
- the same 4-bit data traffic, 1 warp per row, dp4a on the nibbles;
- a 2.5 GiB buffer.

| variant | ms | GB/s (read) |
|---|---|---|
| memcpy D2D (r+w) | 13.373 | 401.5 |
| pure-read 16 B | 6.306 | 425.7 |
| **gemv-4B (current scheme)** | **4.972** | **539.9** |
| gemv-8B | 5.311 | 505.4 (−6.8%) |
| gemv-16B | 5.794 | 463.3 (−16.5%) |

- **The hypothesis failed: 16 B loads are 16.5% SLOWER.** The 4-byte pattern reaches 540 GB/s (88% of the 616 GB/s peak), so load width is not a bottleneck.
- **WP-C (160 B aligned q4_K + 16 B loads) DROPPED** — not justified.
- Consequence: the real decode's effective ~231 GB/s (5.5 GB / 20.8–23.8 ms) does not depend on weight streaming.
  It depends on **the other components**: FA compute at long context, small ops, syncs.
  → Next experiment: FA kernel family A/B with f16 KV (VEC vs MMA; with the existing `GGML_CUDA_FA_FORCE_KERNEL` switch).

## Round 22 (Part 1) — FA kernel family A/B with f16 KV (2026-10-20)

After the C0 lesson (the remaining time is not weight streaming) we looked at FA.
With f16 KV there is no conversion, so MMA (tensor cores) might beat the batch-1 VEC path.
Using the existing `GGML_CUDA_FA_FORCE_KERNEL` switch, 8 runs.

| config | R3 tg rep0 | median | pp |
|---|---|---|---|
| no-MTP VEC (default) | 49.18* | 49.18 | 1311.5 |
| no-MTP MMA (`mma:1`) | 49.11 | 49.11 | 1311.7 |
| no-MTP VEC (2) | 49.02 | 49.02 | 1309.7 |
| no-MTP MMA (2) | 48.59 | 48.59 | 1300.0 |
| MTP default | 67.33 | 67.33 | 1166.2 |
| MTP `mma:4` (draft also MMA) | 65.29 | 65.29 | 1142.2 |
| MTP default (2) | 65.33 | 65.33 | 1144.9 |
| MTP `mma:4` (2) | 66.22 | 66.22 | 1130.7 |

- **Verdict: neutral** (no-MTP ±0.5%; with MTP the two pairs have opposite signs, within the spread) → **the default stays** (batch 1: VEC, batch ≥2: MMA).
- Evidence: the `FA_FORCE_KERNEL: forcing` log is 1 in the forced runs and 0 in the defaults; acceptance 0.4375 everywhere.
- This covers the FA path too: f16 KV + the default kernel choice is the winner.

## Round 23 (Part 1 / WP-D) — CPU FA split-KV for quantized KV: DROPPED (2026-10-20)

The CPU FA split-KV path only handled f16/f32 KV (`kv_is_f32_or_f16`), even though the chunk function is type-generic (K: `vec_dot`, V: `to_float`).
The gate was relaxed with the env-gated `GGML_CPU_FA_SPLIT_QUANT=1` switch (default off).

**Correctness:** CPU (9B, `-ngl 0`, `-fa on`, q4_1 KV, 1217-token context, greedy, 16 tokens) → the output is **bitwise identical** to the default path.
The evidence log shows the split path really runs (`K=q4_1, V=q4_1, n_kv=1280`).

**Speed — direct CPU FA micro-benchmark** (`/tmp/opencode/fa_cpu_bench.c`, n_kv=65536, DK=128, 8 KV heads, 32 Q heads, 16 threads):

| path | ms/call | KV read |
|---|---|---|
| default (chunk loop) | **36.41** | 2.3 GB/s |
| split-KV (quantized) | 61.90 | 1.4 GB/s (**+70% slower**) |

- **Cause:** inside a chunk the split path loops `for q_head` and **re-reads the same KV chunk for every Q head**.
  With GQA that is 32× the traffic, the chunk no longer fits in L3, and the overhead eats the better load balancing.
- **Verdict: dropped.**
  - The daily CPU path stays the chunk loop; the switch stays as an env-gated experimental tool (default off).
  - A future head-fused rewrite might make sense, but that is bigger work.
- Note: CPU FA is slow in absolute terms (2.3 GB/s KV read). The q4_1 dequant dominates in the per-token vec_dot; that is the real limit of long-context CPU speedups, not parallelism.

## Part 1 — FINAL configuration (2026-10-20)

**CUDA daily command (f16 KV, MTP, 170 W):**

```
build-cuda/bin/llama-server -hf unsloth/Qwen3.5-9B-MTP-GGUF:UD-Q4_K_XL \
  -fa on -ctk f16 -ctv f16 --parallel 1 -c 80000 --no-mmproj-offload \
  --spec-type draft-mtp --spec-draft-n-max 2 -t 1 -tb 1 --poll 0 \
  --spec-draft-backend-sampling -bs
```

Measured (r19–r24, controlled A/B), compared with q4_1:
- **MTP R3 tg +14.2% / +14.5%** (64–75 t/s absolute, depending on the session's temperature);
- **no-MTP +12–17%** (48–49 t/s);
- pp unchanged (~1150–1310), acceptance bitwise identical;
- **VRAM at 80k: 9336 MB** (+1.42 GB over q4_1).

**VRAM-constrained / larger-context variant:** the same with `-ctk q4_1 -ctv q4_1` (R3 tg ~65 t/s, VRAM 7876 MB).

**Text-only service (optional):** `--no-mmproj`. The 879 MB vision projector is not loaded; speed is unchanged (r24: 65.6 vs 66.1 t/s).

**Part 1 results table:**

| WP | result | decision |
|---|---|---|
| B1 f16 KV | +14.2% / +14.5% MTP; +12–17% no-MTP | **default: f16** |
| B2 KV shadow | not run: the f16 shadow (+2.1 GB) costs more than plain f16 KV (+1.42 GB) | dropped |
| E n-max 3 | −8.4…−9.0% | dropped |
| E n-max 1 | −0.3% (rep0), median +3.5% (anchor spread) | 2 stays |
| E `-ub 1024` | −1.7…−2.5%, VRAM +316 MB | dropped |
| C0 GEMV 16 B | **−16.5%** (the 4 B scheme does 540 GB/s = 88% of peak) | WP-C dropped |
| FA kernel family (f16 KV) | neutral | default (VEC batch 1, MMA ≥2) |
| D CPU split-KV quantized KV | **−70%** (per-head KV re-reads) | env-gated, default off |
| F `--no-mmproj` | −879 MB RAM, same speed | optional |

**Code changes (Part 1):**
- `ggml/src/ggml-cpu/ops.cpp`: the `GGML_CPU_FA_SPLIT_QUANT` env-gated switch (split-KV for quantized KV; **default off**, measured −70% → do not use).
- Every other experiment ran with the existing env switches (no new default).

## Round 24 (Part 1) — final validation: f16 KV daily config + `--no-mmproj` (2026-10-20)

| run | R3 tg | R3 pp | VRAM | acceptance |
|---|---|---|---|---|
| daily (f16 KV, MTP) | 66.13 | 1145.2 | 9336 | 0.4375 |
| + `--no-mmproj` | 65.59 | 1163.4 | 9336 | 0.4375 |

- The daily config reproduces the +14–16% gain (over q4_1).
- `--no-mmproj`: the log has **0** "loaded multimodal model" lines (the 879 MB projector is not loaded), and the speed is within the spread → an optional text-only switch.

## Round 25 (Vulkan SDK 1.4.363 + rebuild) — 2026-10-20

- Downloaded the **Vulkan SDK 1.4.363.0** (2026-09-28, 366 MB tarball, sha256 `197962f5...`).
  Partly extracted (glslc, vulkaninfo, spirv-dis, headers, libshaderc — 66 MB); the tarball was deleted.
- `build-vulkan` reconfigured to the SDK's glslc (`shaderc 2026.4`, headers 1.4.363); the system loader stays (`libvulkan.so`).
- **Build error and fix:**
  - The new glslc supports the `GL_EXT_float_e2m1/e4m3` extensions, so the C++ references the FP4 (`_ocp`) shader arrays.
  - But the shader generator (ExternalProject) binary was outdated → missing symbols.
  - Fix: delete the generator prefix and the generated header, then regenerate cleanly (1570 SPIR-V, 12 `_ocp` arrays).
  - This is an upstream build-system bug (outdated ExternalProject combined with a new glslc).
- New `libggml-vulkan` md5: `a8a14bd9b313bd8a5023036917fa14f3` (old: `1a9808237a8111343c225cd93882a913`).

## Round 26 (Vulkan toolchain A/B: old vs SDK library) — 2026-10-20

4 runs, only `libggml-vulkan` swapped, q4_1 KV, no MTP, 170 W.

| run | library | R3 tg | R3 pp | VRAM |
|---|---|---|---|---|
| old | 1a980823 | 28.04 | 808.0 | 7146 |
| new | a8a14bd9 | 34.02 | 954.0 | 6762 |
| oldb | 1a980823 | 34.36 | 949.0 | 6755 |
| newb | a8a14bd9 | 34.19 | 948.8 | 6755 |

- **The clean pair (oldb vs newb): −0.5% tg, −0.02% pp, same VRAM → neutral.**
  The +21% of the first pair is the session's cold-start effect: the anchor `oldb` itself jumped +22.5% over `old`, and VRAM went 7146 → 6755.
- **Decision: the SDK build stays** (newer toolchain, same speed).
- Measurement lesson: the first Vulkan run of a session can be up to 20% slower when cold; always take the pairs from warmed-up runs.

## Round 27 (Vulkan KV type: q4_1 vs f16) — 2026-10-20

4 runs, SDK Vulkan library, no MTP, 170 W, alternating pairs.

| run | KV | R3 tg | R3 pp | VRAM |
|---|---|---|---|---|
| q41 | q4_1 | 34.30 | 951.9 | 6755 |
| f16 | f16 | **37.99** | **1077.5** | 8480 |
| q41b | q4_1 | 33.77 | 956.6 | 6755 |
| f16b | f16 | **37.65** | **1076.5** | 8480 |

- **tg: +10.8% / +11.5%** (consistent pair), **pp: +13.2% / +12.5%**.
- Cost: **+1725 MB** VRAM (6755 → 8480 MB at 80k).
- The pp gain comes from prefill FA reading f16 KV without dequantization. On CUDA this was neutral, because the FA there already handled q4_1 well.
- **Decision: the Vulkan daily profile is `-ctk f16 -ctv f16`** (8.5 GB fits).
- Note: this round was clean (the stuck server only appeared later, during the pipeline-cache test).
  Lesson for later rounds: before every measurement run `ps ax | grep -i llama-server`, kill by PID, and check the GPU.

## Round 28 — persistent Vulkan pipeline cache (load speed) — 2026-10-21

Implementation (env switch, OFF by default):
- `GGML_VK_PIPELINE_CACHE=<file>`:
  - loads the `VkPipelineCache` blob when the device is created;
  - uses it when creating pipelines;
  - writes it back on exit (`atexit`). The device destructor does not run (a static registry keeps it alive), hence the exit handler.
- Files touched: `ggml/src/ggml-vulkan/ggml-vulkan-types.h` (2 members), `ggml-vulkan.cpp` (load/save + the `createComputePipeline` call).
- Shader compilation (SPIR-V → SASS) happens in the driver, on the CPU; the cache saves it on later starts.

Measurement (8 runs, alternating pairs, f16 KV, 651-token prompt, 170 W):

| run | cache | startup | req1 pp (ms) | req2 pp (ms) |
|---|---|---|---|---|
| cold | cold (writes) | 10850 | 1279 | 571 |
| noc1 | none | 10735 | 659 | 616 |
| warm1 | warm | 9089 | 730 | 604 |
| noc2 | none | 9763 | 781 | 605 |
| warm2 | warm | 10094 | 651 | 663 |
| noc3 | none | 12761 | 780 | 666 |
| warm3 | warm | 9818 | 636 | 677 |
| nocend | none | 9523 | 747 | 588 |

- **First prefill (req1):**
  - no-cache average 742 ms, warm cache 672 ms (−9.4%);
  - pairs: 1× worse, 2× better (−17%, −18%) → the direction is right, but the spread is large.
  - The cache file is 3.33 MB.
- **Startup:** average −1.0 s (−9.6%); per pair −15% / +3% / −23% → noisy.
- **Where loading time goes** (log timestamps):
  - model + mmproj loading takes ~9.8 s of the ~10.7 s (0.7 s → 8.5 s weights, 8.5 → 10.5 s mmproj);
  - shader/pipeline creation only adds ~70 ms on the first request.
  - So loading is mostly I/O + GGUF processing, not shader compilation.
- **CPU measurement** (new `cpu-watch.sh`, built into `bench.py`):
  - the server uses **0.72–0.80 cores** on average over the whole run, sys 2.6–5.0%, so the test is not CPU-bound;
  - the CPU is mostly used by model loading (one thread) and tokenization/sampling.
- **Decision:** the code stays, **behind an env switch, off by default** (the gain is small and noisy; we do not risk the default behaviour).
- Lesson: the 2–9 s "prompt_ms" values of the earlier (contaminated) measurement were caused by a stuck server; the clean measurement gives 0.6–0.8 s.

## Round 29 — Vulkan: FORCE_MMVQ re-check (f16 KV) — 2026-10-21

4 runs, alternating pairs, f16 KV, 170 W, with CPU monitoring.

| run | env | R1 tg | R2 tg | R3 pp | R3 tg | VRAM |
|---|---|---|---|---|---|---|
| def | — | 44.90 | 42.11 | 952.3 | 35.14 | 8459 |
| mmvq | FORCE_MMVQ=1 | 37.51 | 39.13 | 1036.2 | 32.29 | 8459 |
| defb | — | 49.23 | 49.56 | 960.1 | 37.15 | 8449 |
| mmvqb | FORCE_MMVQ=1 | 45.16 | 42.05 | 1012.7 | 32.81 | 9091 |

- **tg: worse in both pairs** (R1 −8…−16%, R2 −7…−15%, R3 −8…−12%).
- pp improves a little (+5.5…+8.8%), but tg is the main goal → **dropped**. This matches round 11, now with f16 KV too.
- CPU: the server uses 0.87–0.93 cores on R1 and 0.47–0.49 on R3, sys 1.5–3.8% → not CPU-bound; forcing MMVQ is not slower because of extra CPU work.
- VRAM: the mmvqb run used +642 MB (MMVQ prealloc buffers) — another reason to drop it.

## Round 31 — measuring the effect of the master merge (82 commits) — 2026-10-21

Two backends, one alternating pair each, then a **repeat in reversed order** (r31c) to filter out drift.
Only the backend library was swapped (libllama identical).

| run | R1 tg | R2 tg | R3 pp | R3 tg |
|---|---|---|---|---|
| CUDA pre (cold, 1st run) | 59.85 | 78.82 | 1196 | 55.61 |
| CUDA merge | 82.51 | 78.01 | 1230 | 70.37 |
| CUDA mergeR (reversed pair) | 71.30 | 79.46 | 1265 | 71.86 |
| CUDA preR (reversed pair) | 72.83 | 82.81 | 1283 | 75.14 |
| Vulkan pre | 50.96 | 53.15 | 1074 | 37.75 |
| Vulkan merge | 51.55 | 48.93 | 972 | 30.63 |
| Vulkan mergeR | 51.99 | 51.44 | 1088 | 37.36 |
| Vulkan preR | 50.23 | 53.56 | 1065 | 37.99 |

- **The first-run trap:** the session's first CUDA run is **35% slower** (55.61 vs 75.14 with the same library), because of driver JIT/warm-up of the freshly built CUDA library. This caused the apparent "+26.5%".
- **Balanced (warm) comparison:**
  - **CUDA: merge −4.4%** (R3 tg), R2 −4.0%, pp −1.5% → a small slowdown.
    Probably the cost of the `99db719bd` FA vec WAR correctness fix (vec FA is the batch-1 decode path, hot at long context).
  - **Vulkan: merge −1.7%** (R3 tg), pp +2.2% → within the noise, **neutral**.
    The −19% first measurement was a VRAM/SM-clock drift artefact; recording the memory clock ruled it out this time.
- **Decision: the merge stays** (upstream alignment + correctness fixes; CUDA −4% is an acceptable price for the WAR hazard fix).
  The effect of the `qwen35.cpp` 2D reshape cannot be isolated in this measurement (libllama was identical in both runs).
- **Methodology lesson:**
  - drop the session's first run (especially on CUDA), or treat it as a separate "cold" anchor;
  - also repeat the pairs in reversed order.

## Round 30 — Vulkan coopmat A/B (`GGML_VK_DISABLE_COOPMAT`) — 2026-10-21

4 runs, alternating pairs, f16 KV, 170 W, with CPU monitoring.

| run | env | R1 tg | R2 tg | R3 pp | R3 tg | VRAM |
|---|---|---|---|---|---|---|
| def | — | 50.38 | 47.42 | 981.3 | 34.26 | 8461 |
| nocm | DISABLE_COOPMAT=1 | 50.29 | 47.29 | 986.8 | 34.74 | 8461 |
| defb | — | 47.40 | 47.79 | 988.0 | 34.57 | 8461 |
| nocmb | DISABLE_COOPMAT=1 | 48.62 | 48.22 | 986.3 | 34.84 | 8461 |

- R3 tg: +1.4% / +0.8% (without coopmat), pp ±0.6%, VRAM identical → **within the noise, neutral**. The default stays (coopmat on).
- Note: Turing FA uses the `scalar` path anyway (because of an NVIDIA compiler bug pinned in the code), so coopmat only affects the GEMMs → small effect.

---

## Vulkan — FINAL configuration (2026-10-21)

```
LD_LIBRARY_PATH=build-vulkan/bin build-vulkan/bin/llama-server \
  -hf unsloth/Qwen3.5-9B-MTP-GGUF:UD-Q4_K_XL \
  -fa on -ctk f16 -ctv f16 --parallel 1 -c 80000 --no-mmproj-offload -ub 512
# optional: GGML_VK_PIPELINE_CACHE=/tmp/vk-pipeline-cache.bin (small startup gain)
```

- **f16 KV: +10.8–11.5% tg, +12.5–13.2% pp** (cost: +1725 MB VRAM at 80k) — the main gain.
- Dropped (measured, not winning): FORCE_MMVQ (−8…−16% tg), DISABLE_COOPMAT (neutral), MTP on Vulkan (−16%), DISABLE_MMVQ (neutral), MAX_NODES (neutral).
- Pipeline cache: env-gated, off by default (first prefill −9% average, noisy).
- SDK swap (1.4.363): neutral.
- The `-O` shader experiment (rope/coopmat) is prepared (`GGML_VULKAN_OPT_ALL_SHADERS=1` at build time). The expected gain is small and there is a correctness risk, so it is not enabled for now.
- Vulkan is still ~2× slower in tg than the CUDA+MTP daily config (R3: 34–38 vs 70–75), so daily use is CUDA.

## Round 32 — master merge #2 (31 commits, 84e66d032 → e3797b184) — 2026-10-22

Fast-forward + stash pop, **without conflicts**.
- The 7 local modifications survived bit for bit (235 insertions / 4 deletions).
- The key markers were checked: `GGML_CPU_FA_SPLIT_QUANT`, `GGML_CUDA_BLOCKING_SYNC`, `GGML_VK_PIPELINE_CACHE`, `GGML_VK_RM_KQ`.

**Upstream content (relevance):**
- `eb182d85c` keeps d2t on the GPU — **EAGLE3 only**; Qwen3.5-MTP does not use d2t.
- `d454a204e` new fused `top_k_mask` op (QSA/DSA/GLM) — qwen35 does not use it; the CUDA registration is +6 lines and does not affect existing paths.
- `824eaecf4` vocabulary loading optimization — load time, not decode.
- `261700b89`, `9a3b91c68` server API/accounting — functional.
- `479a52779` CPU gelu_quick — CPU only.
- The rest: new models (Flint, SenseNova U1), SYCL/OpenCL/Metal/Hexagon/WebGPU, jinja, UI → **no CUDA/Vulkan decode perf change**.
- The only meaningful core change is the K-shift condition (`llama-kv-cache.cpp`), which only runs on a context shift (the measurements have no shift).

**Build:** CUDA `a73534b18fdc293a953e5ca948390d70`, Vulkan `df3700a40572825e1d6f6909ad2de48d`.

**Sanity measurement (170 W, CPU monitoring, f16 KV):**

| run | R1 tg | R2 tg | R3 pp | R3 tg | VRAM |
|---|---|---|---|---|---|
| V-r32-new | 48.73 | 47.96 | 1034.3 | 34.82 | 8388 |
| C-r32-newA | 76.04 | 73.79 | 1188.1 | 64.82 | 9226 |
| C-r32-newB | 76.16 | 72.52 | 1169.7 | 66.53 | 9226 |
| C-r32-newC | 70.32 | 74.78 | 1250.8 | **70.16** | 9226 |
| C-r32-newD | 83.93 | 81.23 | 1250.4 | **72.29** | 9226 |

- **CUDA: neutral.** The plateau of the 3rd–4th runs (70.2 / 72.3 R3; R2 81.2; pp 1250) is in the r31 warm range (R3 70.4–75.1; R2 78–83; pp 1230–1283).
- **Vulkan: neutral.** 34.82 R3 is in the r30 session cluster (34.3–34.8); r31's 37.4–38.0 is session-level drift (see round 31).
- **Warm-up effect confirmed:** the first 1–2 CUDA runs of a session are 5–12% slower (64.8 / 66.5 → 70.2 / 72.3), even with the same library.
  **For measurements, drop the first 2 runs** (or use an anchor).
- **VRAM:** CUDA 9290 → 9226 MiB (−64), Vulkan 8461 → 8388 MiB (−73), probably from the vocab/tokenizer change; not a regression.
- **Load time:** 6.4–10.4 s, dominated by the page cache → not evaluable.
- **Functional check:** greedy `llama-cli` output is coherent (chat template OK).

**Decision: the merge stays; the daily configurations are unchanged.**

## Rounds 33–34 — MTP config sweep + strict A/B (Vulkan, f16 KV) — 2026-10-22

**Round 33 (screen-r33, 60k prompt, 128 gen, cooldown ≤65 °C at start):**

| config | pp | tg |
|---|---|---|
| base-a / base-b | 917.2 / 927.7 | 31.08 / 29.44 |
| mtp2 (dbs+bs) | 737.6 | 29.04 |
| mtp1 (dbs+bs) | 722.3 | 30.38 |
| mtp3 (dbs+bs) | 742.6 | 24.92 |
| mtp2 (no -bs) | 769.2 | 31.52 |
| **mtp2 (-bs)** | 866.4 | **34.26** |
| mtp2 (repeat) | 730.9 | 28.35 |

**Critical correction:**
- `--spec-draft-backend-sampling` is **ON by default** (`common.h`: `draft.backend_sampling = true`).
- So the runs labelled "dbs" and "no dbs" were **identical configurations**.
- The measured 29.04 vs 34.26 (17%) between them is pure run noise, not a config effect.
So round 33's "+13% winner" is invalid, and round 34 refuted it.

**Round 34 (ab-60k.sh: strictly alternating, cooldown ≤62 °C, 2 rounds):**

| config | pp | tg run1 / run2 | evaluation |
|---|---|---|---|
| base | 917.5 / 917.3 | 30.75 / 31.00 | stable |
| base (anchor, later) | 908.6 | 29.88 | stable |
| mtp2 (dbs default ON) | 735.3 / 736.4 | **23.66 / 29.15** | unstable, −23…−5% |
| mtp2 (dbs explicitly ON, repeat) | 729.7 / 743.5 | 28.63 / 27.65 | −9% |
| **base + -bs** | 904.9 / 915.2 | **32.53 / 31.96** | **+5.6% tg**, pp neutral |

- With identical acceptance (70/113, mean length 2.23) and identical GPU statistics (88–90% util, 154–158 W), **the same configuration ranged 23.66–29.15 t/s** (17%).
  So the spread depends on the run, not the config.
- **MTP on Vulkan does not win at 60k**: the 4 same-config runs average 27.3 t/s against base 30.5 (**−11%**), with a large spread. Round 33's +13% was run noise.
- **Daily Vulkan config: `-bs` added** (+5.6% tg, same pp, same VRAM).

**Short context (2.8k prompt, 256 gen):**

| config | tg |
|---|---|
| base | 41.6 (40.55 / 42.61) |
| base + -bs | 45.59 |
| base + `-t 1 -tb 1 --poll 0` | 46.5 (47.26 / 45.68) |
| base + `-bs -t 1 -tb 1 --poll 0` | 50.9 (48.80 / 53.11) |
| mtp2 (-bs) | 40.84 |
| mtp2 (+dbs) | 46.35 |
| mtp1 (+dbs) | 48.26 |

- `-t 1 -tb 1 --poll 0` gives +11…+22% on short context, but pp −5% at 60k and −9% with MTP → for short-context use only.
- CUDA on the same prompt: base 52.73; MTP2 73.90 (+40%) → the CUDA MTP advantage is real; the cost of the Vulkan draft graph is disproportionately large.

**Root-cause measurements (new, env-gated instrumentation):**

- `LLAMA_GRAPH_TIMING=1` (`llama-context.cpp`), for decode: **build = 0.06 ms, set_inputs ≈ 0, submit = 14.84 ms, sched sync = 0.03 ms**.
  So the Vulkan backend blocks inside `graph_compute` (no CPU/GPU overlap), and the submit time = GPU execution.
- `llama-bench` (without the server) on short context: **tg 47.64 t/s**; the server gives 42.6 → the server-side overhead is ~2.5 ms/token.
- Effective bandwidth in decode: 5.7 GiB / 21 ms ≈ **291 GB/s** (47% of 616 GB/s).
  The C0 GEMV micro-bench reaches 540 GB/s (88%) → **~1.85× theoretical headroom**, lost to the gaps between many small kernels / dispatch behaviour.
- Prefill: 1425–1570 t/s ≈ 26 TFLOP/s (~96% of the fp16 peak) → the GEMM path is good.
- `GGML_VK_SERIALIZE_SUBMISSIONS=1`: 10.75 t/s (−75%) → waiting on submits/fences is expensive under VFIO, but the normal async path hides it.

**New env-gated switches (default = unchanged behaviour):**

- `GGML_VK_FLOPS_PER_SUBMIT_DIV` (default 40): how the graph is split into submits.
  Measured: 40 → 42.61; 8 → 40.61; 4 → 37.08; 1 → 43.20 t/s → **no consistent effect**; the default stays.
- `LLAMA_GRAPH_TIMING` (default off): phase timing in the log.

**Decisions:**
1. Vulkan daily command: **`-bs` goes in** (`--backend-sampling`).
2. MTP on Vulkan at 60k is **dropped** (unstable or −9%); CUDA+MTP stays the fast path when the user runs CUDA.
3. `-t 1 -tb 1 --poll 0`: only for short-context (≤3k) work.

**Vulkan daily command (from 2026-10-22, f16 KV + backend sampling):**

```
build-vulkan/bin/llama-server -hf unsloth/Qwen3.5-9B-MTP-GGUF:UD-Q4_K_XL \
  -fa on -ctk f16 -ctv f16 --parallel 1 -c 80000 --no-mmproj-offload \
  -ub 512 -bs
```

Measured (60k, strict A/B): tg 32.25 t/s (`-bs`) vs 30.54 (base) = **+5.6%**, pp 910 vs 917 (same), VRAM 8388 MiB.
MTP is not included (see rounds 33–34). For short-context (≤3k) work `-t 1 -tb 1 --poll 0` is optional (+11…+22% tg, but pp −5% at 60k).

> **WARNING — the MTP verdict above is invalid, see round 35.**

---

## Round 35 — 41k context: MTP re-evaluated, CUDA vs Vulkan, `-t`/`-ub` (2026-10-26)

**Background:** the "MTP loses on Vulkan" verdict of rounds 33–34 was based on **128-token** probes.
Long real-world use (41k prompt, 2000–8000 generated tokens) refutes it.
The short probe does not measure MTP's warm-up, and at 128 tokens the acceptance spread is larger than the effect.

Every run is the same conversation, `pp = 41173` tokens, Vulkan `f16` KV, `-bs`.
Run by hand, not under the 170 W protocol; the >10% effects are unambiguous anyway.

| config | pp t/s | gen tok | tg t/s | acc | mean len | ms/step |
|---|---|---|---|---|---|---|
| CUDA + MTP2, `-t 24` (#1) | 1034 | 4123 | 58.39 | 0.794 | 2.59 | 44.4 |
| CUDA + MTP2, `-t 24` (#2) | – | 2308 | 48.34 | 0.658 | 2.32 | 48.0 |
| Vulkan + MTP2, `-t 24` (#1) | 1075 | 8118 | 48.73 | 0.714 | 2.43 | 49.9 |
| Vulkan + MTP2, `-t 24` (#2) | – | 2317 | 49.13 | 0.702 | 2.40 | 48.8 |
| Vulkan, no MTP, `-t 24` | 1213 | 2172 | 37.39 | – | 1 | 26.7 |
| Vulkan, no MTP, `-t 1` | 1202 | 1382 | 37.94 | – | 1 | 26.4 |
| Vulkan, no MTP, `-t 1 -b 4096 -ub 4096` | 1145 | (8 t/s) | ~8.0 | – | 1 | **125** |

**Conclusions**

1. **MTP at 41k: +30% tg, at a cost of 4.37 s of pp** (38.31 s vs 33.94 s, −11.4% pp).
   - Break-even: `4.37 s / 6.5 ms` ≈ **670–700 generated tokens**.
   - A 41k prompt + a 2000-token answer: 79.3 s vs 87.7 s (−9.5%); at 5000 tokens −16%.
   - → MTP clearly wins for long answers and loses in pp for short ones (≤700 tokens).
2. **CUDA vs Vulkan at 41k: on par.**
   - CUDA's 58.39 t/s is an **acceptance artefact** (0.794 vs 0.714).
   - Step times: CUDA 44.4/48.0 ms, Vulkan 49.9/48.8 ms → the difference is within the noise.
   - **Vulkan is better in pp** (1075 vs 1034).
   - CUDA's advantage is real on short context (2.2k: 73.9 vs 46.4 in round 33) and disappears at 41k.
3. **Text dependence is ±9%.**
   - The same Vulkan+MTP config gives 48.7 / 49.1 / 53.4 t/s at 41k depending on what it generates (acceptance 0.66–0.79).
   - So compare MTP only with **ms/step + acceptance** together, or with ≥2000 generated tokens and several repetitions. t/s alone is misleading.
4. **`-t 1 -tb 1` at 41k (without MTP) is neutral**: pp −1%, tg +2%. It does not hurt, but it does not help either; `-t 24 -tb 24` stays for long context.
5. **`-b 4096 -ub 4096` is a disaster: 8.0 t/s decode (125 ms per step), and pp gets worse too (1145 vs 1213).**
   - `-ub` stays at 512 (which is also the default).
   - The mechanism is unclear. Hypothesis: the prefill graph of a 4096-wide ubatch asks for ~4 GB of transient buffers in the ~2.6 GB of free VRAM, and the decode buffers get pushed out.
   - Practical rule: **never raise it above 512**.
6. **Correction:** the "MTP on Vulkan dropped at 60k" verdict of round 34 is **withdrawn** (128-token probe + 60k VRAM pressure).
   The MTP behaviour at 60k has to be re-measured with long generations (open).

**Vulkan daily command (from 2026-10-26, with MTP, for long context):**

```
build-vulkan/bin/llama-server --port 8080 \
  -hf unsloth/Qwen3.5-9B-MTP-GGUF:UD-Q4_K_XL \
  -fa on -ctk f16 -ctv f16 --parallel 1 -c 80000 --no-mmproj-offload \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  --spec-draft-backend-sampling -bs \
  -t 24 -tb 24 --poll 0
```

The CUDA fallback is the same with the `build-cuda/bin/llama-server` binary.
For short (≤3k) chats, `-t 1 -tb 1` can replace `-t 24 -tb 24`.

---

## Round 36 — Part 2/A: custom quant mix (2026-10-26)

**Goal:** reduce the per-step weight traffic of decode.
- Starting point (UD-Q4_K_XL): **7.22 GB** of traffic per step.
- `output.weight` (Q6_K, 834 MB) is read **three times** (target verify + 2 MTP draft steps) = 2.50 GB, i.e. **34.7%** of the traffic.
- The qwen35 MTP graph uses `model.output` as its head (`nextn.shared_head_head` is not in the GGUF); the source confirms it (`qwen35.cpp:656`).

**Method:** `Qwen3.5-9B-BF16.gguf` (18.41 GB) + `imatrix_unsloth.gguf_file` (248 entries, 77 chunks) → `llama-quantize` with an **explicit tensor table** (`mix-gen.py`, generated from the released file).
- Regex `--tensor-type` alone was not reproducible. The internal counters of the default Q4_K mix (i_ffn_down, i_attention_wv) only grow on non-overridden tensors, so the number of overrides shifts which layers get Q6_K.
- The full table removes this: the `a0` replica is **5840.36 MiB = exactly the data of the released file (6124 MB)**.

**Variants** (`quant-mix.sh`, 24 threads, 3–7 min each):

| version | change | file | step traffic |
|---|---|---|---|
| a0 | exact replica of the released mix | 6.14 GB | 7.22 GB (0%) |
| v1 | head Q6_K → Q5_K | 6.00 GB | 6.82 GB (−5.6%) |
| v2 | head Q6_K → Q4_K | 5.87 GB | 6.43 GB (−10.9%) |
| v3 | v2 + ffn_down Q6_K→Q5_K, ffn_down Q5_K→Q4_K, ssm_out Q8_0→Q5_K | 5.59 GB | 6.15 GB (−14.8%) |
| v4 | every quantizable tensor Q4_K | 5.21 GB | 5.77 GB (−20.1%) |

**Raw decode screen** (`llama-bench -p 128 -n 128 -r 3`, Vulkan, f16 KV, no MTP, 6 models interleaved; the first `base` run is cold):

| model | size | pp t/s | tg t/s | Δ tg vs a0 |
|---|---|---|---|---|
| base (UD-Q4_K_XL) | 5.70 GiB | 1539 ± 173 | 54.02 ± 1.79 | +2.2% (noise) |
| a0 (replica) | 5.70 GiB | 1632 ± 32 | 52.85 ± 0.76 | – |
| v1 (head Q5_K) | 5.58 GiB | 1654 ± 29 | 56.93 ± 1.54 | +7.7% |
| v2 (head Q4_K) | 5.46 GiB | 1652 ± 18 | 59.24 ± 1.22 | +12.1% |
| v3 | 5.20 GiB | 1601 ± 105 | 59.17 ± 0.05 | +12.0% |
| **v4 (all Q4_K)** | **4.84 GiB** | **1839 ± 22** | **63.82 ± 1.05** | **+20.8%** |

**Conclusion from the screen:** the byte model is accurate.
- For v2 the computed +8.9% (1× head read) came out as +12.1%: Q4_K is also the native, faster Vulkan path, not just fewer bytes.
- v4's pp is also +13% (1839 vs 1632), so a smaller quant does not slow down prefill.
- v3 is practically on par with v2 (`ssm_out` Q8_0→Q5_K brings nothing measurable), but smaller.

> **WARNING:** the first reading of the MTP A/B below ("quantization costs 15%") **was wrong**. The 60k run throttles
> thermally, and the drift is larger than the measured effect. The correct evaluation is in the two sections after it.

**MTP A/B, 60k, 2 rounds (base / a0 / v2 / v4), 250 W:**

| run | tg r1 | tg r2 | ms/step r1 → r2 |
|---|---|---|---|
| base | 45.06 | **36.99** | 62.6 → 76.2 |
| a0 | 45.02 | **37.39** | 63.1 → 76.0 |
| v2 | 38.06 | 37.64 | 75.7 → 76.5 |
| v4 | 36.34 | 36.58 | 77.6 → 77.1 |

- **The same model (base) went 45.06 → 36.99 (−18%) between the two rounds**, while temperature (83→84 °C), SM clock (1561→1575 MHz) and power (185→179 W) were the same.
- The 60k run is ~140 s of continuous load. The 2080 Ti's **88 °C thermal limit** is already active then, and the card throttles; the "fast" band (45 t/s) only lasts for the first ~4–5 minutes.
- In the stable band the order is v2 37.64 > a0 37.39 > base 36.99 > v4 36.58 = **±1.5%, i.e. no model effect**.
- So the 60k A/B is **not suitable** for deciding the quant mix.

**Short-context, stable A/B (2.8k prompt, 3 rounds, MTP2, 250 W):**

| model | tg median | **ms/step** | acc |
|---|---|---|---|
| base | 53.98 | **49.3** | 0.833 |
| a0 | 51.91 | 50.5 | 0.815 |
| v1 | 58.24 | 47.9 | 0.898 |
| v2 | 55.43 | 50.9 | 0.911 |
| v3 | 54.80 | 51.6 | 0.915 |
| v4 | 53.44 | 51.1 | 0.863 |

- The t/s order is an **acceptance artefact** here too (0.815–0.915: each model generates different text).
- By step cost (ms/step) the quantized variants are **−3…+5%**, within the run spread (±2–4%).
- → **No gain on the MTP path**, on short or long context.

**Why:**
- The base MTP step moves 7.22 GB in 49.3 ms = **146 GB/s**.
- The path without MTP does 345 GB/s on the same card (54.0 t/s → 18.5 ms, 6.39 GB).
- So the MTP path uses **~42%** of the bandwidth: it is overhead-bound (draft graph + 2 submits + sync), not byte-bound.
  A byte reduction can only act on the remaining 58%, and the measured effect stays in the noise.
- **This is also the largest open opportunity: ~+40% theoretical decode if the overhead of the MTP path drops.**

**Decision:** the daily config stays **UD-Q4_K_XL + MTP2**.
- The quant mix only wins without MTP (+12…+21%), or for VRAM (v4: 5.21 vs 6.14 GB, +0.9 GB free).
- The quality gate (perplexity) only needs to run if it is introduced for such a purpose.

**Methodology lessons (by-products of this round):**

- **The 170 W limit is not enabled**: the card runs at 250 W (`nvidia-powerlimit.service` is not active).
  - So everything in this round was measured at 250 W; on short context that is +17% over the earlier 170 W values (base MTP2: 54.0 vs 46.35 t/s).
  - Old and new absolute numbers are **not comparable**.
- **Above 60k the run throttles thermally** (88 °C limit). Long-context A/B needs either a pre-warmed steady state (2–3 discarded runs) or measuring on short context.
- The harness now **also samples `clocks.mem`**. It was 6800 MHz for the whole round, so the throttling is on the SM/power side, not a memory clock drop.
- ~~`LLAMA_GRAPH_TIMING` only prints newly **built** graphs (`built=1`)~~ — **wrong (round 38):**
  the counter was shared between the two contexts and printed every 16th call. It aliased with the 4-graphs-per-step cycle, so it always sampled the rebuilt catch-up graph. Fixed.

---

## CTranslate2 — closed direction (summary, 2026-10-26)

**What it was:** `plan-ctranslate2.md` (rev. 2, archived) broke down CTranslate2's sources of speed and derived work packages from them.
- CTranslate2 **is not and never was integrated** into llama.cpp: not a dependency, not a backend, not code.
- It was only a research source (`~/git/CTranslate2` checkout + the rev. 2 analysis).
- Two experiments came from it, both **measured**:

| WP | CTranslate2 element | Measurement | Verdict |
|---|---|---|---|
| C0 | AWQ int4 GEMV: 16 B (`float4`) weight loads + split-K | 4 B **540 GB/s**; 8 B −6.8%; 16 B −16.5% (round 21) | **dropped** → 2.2 (160 B q4_K repack) is not justified |
| D | split-KV in the CPU FA for quantized KV | +70% slower under GQA (round 23) | **dropped**, env-gated experiment only |

**What remains:** the `GGML_CPU_FA_SPLIT_QUANT=1` env switch (`ggml/src/ggml-cpu/ops.cpp:9586`, default **off**). It is an experimental tool and does not run on the daily path.

**What our own measurements have since overridden:**
- The premise of rev. 2 was a 1.7× GEMV headroom (231 GB/s against the 408 ceiling).
- Today: no-MTP **345 GB/s** (the GEMV micro-bench 540), the MTP step **146 GB/s** (round 36).
So the bottleneck is not the load width but the overhead of the MTP path (draft graph + 2 submits + sync).
That makes the 16 B direction moot; the open question moved to the MTP overhead.

**What stayed useful:** rev. 2's "what we already have" list (split-KV FA on all three backends, fused QKV, caching allocator, CUDA graphs).
These are upstream elements and did not come from CTranslate2; CT2 only explained why the competing implementation is fast.

---

## Round 37 — int8 (Q8_0) vs Q4_K: CTranslate2-style int8 does not transfer (2026-10-26)

**Question:** can CTranslate2's int8 (per-row symmetric int8 + cuBLAS IMMA GEMM) be transferred?
First without code: the repo already has a ready `Qwen3.5-9B-Q8_0.gguf`.

**Measurement** (`llama-bench`, Vulkan, 170 W, `-p 4096 -n 64 -r 3`, f16 KV, `-ub 512`):

| model | size | pp4096 | tg64 |
|---|---|---|---|
| UD-Q4_K_XL | 5.70 GiB | 2040.4 ± 9.4 | **60.87** ± 0.32 |
| Q8_0 | 9.10 GiB | 2077.7 ± 7.5 | **41.30** ± 0.41 |

- **Prefill +1.8%**, **decode −32.2%**.
  On the existing path (Vulkan integer dot / CUDA dp4a) int8 weights bring no significant prefill gain, and the extra bytes (8.5 vs 4.5 bpw) eat decode.

**Why it is different in CTranslate2** (sources: `~/git/CTranslate2`, `docs/quantization.md`, `src/cuda/primitives.cu`):

| aspect | CTranslate2 int8 | llama.cpp Q8_0 |
|---|---|---|
| scale | 1 f32 per **row** (`127/max(abs(W[i,:]))`) | 1 f16 per **32 weights** (128× finer) |
| layout | dense `int8_t` matrix, BLAS-ready | GGUF blocks with embedded scales |
| compute | `cublasGemmEx` `CUDA_R_8I` → `CUDA_R_32I` = **IMMA** | hand-written MMVQ/MMQ, **dp4a** (IMMA only on sm_80+) |
| coverage | only embeddings + linear layers | every quantizable tensor |
| workload | **batched** (large M) → compute-bound | 1–3 tokens → **memory-bound** |

**Conclusion:** CT2's int8 advantage does not come from "8 bits" but from the **vendor IMMA GEMM and the batched workload**.
- For our decode int8 structurally loses (2× bytes).
- Prefill, however, is compute-bound (~37 TFLOP/s ≈ ~70% of the 2080 Ti's fp16 tensor peak), so **in theory** an IMMA path could give up to ~2× pp. But:
  - **On Vulkan, Turing has no int8 tensor cores** (`VK_KHR_cooperative_matrix` is Ampere+, and there is no vendor BLAS), so there is nothing to implement beyond dp4a.
  - **On CUDA** there is an sm_75 int8 IMMA helper (`mma.cuh`, `m8n8k16.s8`), but the MMQ configs only use it from Ampere on.
    It would need a new MMQ config + correctness checks, and it would only speed up pp (~9% of the wall time at 41k).

**Decision:** an int8 implementation is not justified (impossible on Vulkan; on CUDA only ~9% of pp, with high risk). The existing Q4_K stays.

---

## Round 38 — MTP step breakdown, preliminary (G0, evening of 2026-10-26)

**Fixed `LLAMA_GRAPH_TIMING`** (per context, aggregated by `(n_tokens, n_outputs)`).
Smoke test: Vulkan, MTP2, f16 KV, `-bs`, `-c 8192`, 8-token prompt, 300 tokens, temp 0, **170 W**. A single run → preliminary.

| context | graph | n | rebuilt | build | submit (ms) |
|---|---|---|---|---|---|
| main | verify, 3 tokens / 3 outputs | 64 | 1 | 0.23 | **14.95** |
| mtp | draft, 1 token / 1 output | 42 | **22** | 0.33 | 1.71 |
| mtp | catch-up, 3 tokens / **0 outputs** | 22 | **22** | 0.73 | 1.61 |

- sched sync: main ~0.8–1.1 ms, mtp ~0.25 ms.
- Server: 52.6 t/s, acceptance 0.673, mean length 2.35 → 127 steps, **~44.8 ms per step**.
- Time spent in graphs ≈ 15.2 + 2×2.1 + 2.4 + ~1.5 (sync) ≈ **23 ms per step**.
  → **~22 ms per step outside `process_ubatch`** (server loop, sampling, embedding readback, `yield_to_queue`).
- The catch-up graph is rebuilt on **every** step, and so is the 1st draft step (slot switch because of 0 ↔ 1 outputs) — plan-part3's hypothesis confirmed.

~~**Conclusion:** the largest item is not the GPU but the ~22 ms outside the graphs.~~ **Wrong, see round 39:**
the Vulkan `graph_compute` does not block all the way through. The "submit" is only command recording; the GPU finishes the rest inside the server's `llama_synchronize` call. The gap was GPU time.

---

## Round 39 — the root cause of the "22 ms": the verify n=3 k-quant mat-vec (evening of 2026-10-26)

**New instrumentation (env-gated, default off):**
- `LLAMA_SERVER_TIMING=1` (`server-context.cpp`): a per-step breakdown averaged over 64 steps.
  Phases: gap, pre_decode, draft(+work), tgt_decode(+work), spec_process(+work), post_decode, accept, step.
- `LLAMA_GRAPH_TIMING=1` now also breaks down the phases of `llama_context::decode`: reserve, mem_update, init_batch, out_reserve, ubatch, extract, apply.

**Step breakdown (Vulkan, MTP2, short ctx, 170 W):** step 44.5 ms = tgt_work **32.4** + draft_work 8.0 + spec_work 3.0 + about 1 ms of yield/other.
- The server itself is negligible: gap 0.02, yield ~0.3 per phase, sampling 0.1 ms.
- The CPU side of the target `decode()` takes 15.9 ms, almost all of it recording the `graph_compute` commands.
  The remaining ~16 ms is spent waiting for the GPU in `llama_synchronize`.
- → **The verify graph's GPU time is ~30.6 ms** (`GGML_VK_PERF_LOGGER`, average of 59 graphs).

**Op level:** ~26 ms is the weight mat-vec. For the same matrix (`q4_K 12288×4096`): n=1 (draft) 65 µs (~435 GB/s), **n=3 (verify) 168 µs (~165 GB/s)**.
- Cause: a workgroup of the float k-quant shader computes `rm_kq=2` rows and re-reads the activation vector per column.
  At n=3 that is ~300 MB of L1/L2 traffic next to 28 MB of weights.
- The MMVQ (integer dot product) path, the default for n>1, is just as slow (167–171 µs).

| q4_K 12288×4096, n=3 | µs |
|---|---|
| default (MMVQ) | 168 |
| float, rm=2 (`GGML_VK_DISABLE_MMVQ`) | 156 |
| float, rm=4 | 116–122 |
| float, rm=8 | 112 |

`GGML_VK_DISABLE_MMVQ` is not good globally: it makes q8_0 n=3 worse (56 → 101 µs).

**New switch: `GGML_VK_KQ_BATCH_RM=N`** (`ggml-vulkan.cpp`, default off):
- the q4_K/q5_K/q6_K float mat-vec pipelines get N rows for n≥2;
  q6_K at most 4, because at 8 it gets worse (head 1.67 → 5.16 ms);
- for n>1, q4_K/q5_K go to the float path instead of MMVQ;
- q8_0 stays on MMVQ.

**Measurement** (interleaved, 2 rounds × 3 configs, 600 tokens, 30 s cooldown):

| config | tgt_work | step | t/s |
|---|---|---|---|
| default | 34.2–35.0 ms | 48.5–49.1 ms | 47.1–48.7 |
| **RM=4** | **29.1–29.6 ms** | **43.1–44.4 ms** | **52.8–53.6** |
| RM=8 (q6 ≤4) | 28.4–31.0 ms | 42.3–48.8 ms | 48.7–55.6 |

The first round of the 2nd series was slower for all three configs (drift).
→ **RM=4: verify −14%, step −9…−10%, t/s +10%**. RM=8 is the same within the noise, so 4 is recommended.

**Output:**
- deterministic per configuration (bitwise identical across repeats);
- it differs from the default at character 1499 of 2532 (floating-point summation order, expected with greedy decoding);
- acceptance 345 vs 349.

**What remains in the step (RM=4):**
- draft_work ~8 ms, of which GPU ~4.6 ms: 2 draft graphs; the head takes 1.55 ms per step at ~538 GB/s, so it is bandwidth-bound.
- spec_work ~3 ms (GPU ~1.3 ms).
- The difference is CPU-side (graph rebuilds, sampling, syncs) → this is the target of G1.

**Open anomaly:** in the MTP context `q4_K m=8192 n=1 k=4096` takes 235 µs (~80 GB/s), while the same with n=3 in the trunk takes 97–118 µs (~0.4 ms per step).

**To do:**
- Confirm at 41k context with ≥2000 tokens; if it wins there too, `GGML_VK_KQ_BATCH_RM=4` should go into the daily command.
- On CUDA, check separately whether MMVQ n=3 is similarly bad.

### 39b — confirmation at long context (evening of 2026-10-26)

Setup:
- Prompt: the first 129k characters of `long-60k.txt` + a "summarize in detail" request → **45 253 tokens**; 2000 generated tokens.
- Vulkan + MTP2 + f16 KV + `-bs`, `-t 24 -tb 24 --poll 0`, 170 W.
- Interleaved (default, RM=4, default, RM=4), 60 s cooldown, max 79 °C (no throttling).
- Script: `ab41k.sh` (session scratchpad).

| run | pp t/s | tg t/s | acc | mean len | ms/step | tgt_work | draft_work | spec_work |
|---|---|---|---|---|---|---|---|---|
| default #1 | 998.5 | 49.70 | 0.929 | 2.86 | 57.4 | 41.7 | 9.0–9.5 | 3.3–3.5 |
| RM=4 #1 | 1011.6 | **54.06** | 0.939 | 2.88 | **53.1** | **37.5** | 9.1–9.3 | 3.3–3.4 |
| default #2 | 991.8 | 49.06 | 0.929 | 2.86 | 58.1 | 42.3 | 9.6 | 3.5–3.6 |
| RM=4 #2 | 998.8 | **53.80** | 0.939 | 2.88 | **53.5** | **37.0** | 9.3–9.8 | 3.3–3.5 |

- **ms/step −7.6%** (57.8 → 53.3), verify −11.9% (42.0 → 37.0), tg **+9.2%**.
  About 1 point of the tg gain comes from slightly better acceptance, because the generated text differs.
- pp unchanged (+0.7%, noise): the large prefill batch is not affected.
- The output is bitwise identical across repeats (default: 5684, RM=4: 5652 characters).
- The draft (~9.3 ms) and the catch-up (~3.4 ms) do not change → G1 is next.

**Decision:** the Vulkan daily command gets the `GGML_VK_KQ_BATCH_RM=4` env:

```
GGML_VK_KQ_BATCH_RM=4 build-vulkan/bin/llama-server --port 8080 \
  -hf unsloth/Qwen3.5-9B-MTP-GGUF:UD-Q4_K_XL \
  -fa on -ctk f16 -ctv f16 --parallel 1 -c 80000 --no-mmproj-offload \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  --spec-draft-backend-sampling -bs \
  -t 24 -tb 24 --poll 0
```

---

## Round 40 — CUDA vs Vulkan at 45k + fixes on both sides (night of 2026-10-26/27)

Protocol: the same 45 253-token prompt + 2000 tokens, MTP2, f16 KV, `-bs`, `-t 24 -tb 24 --poll 0`, 170 W, interleaved, 60 s cooldown (`ab45k.sh`).

### CUDA vs Vulkan (before the fixes, Vulkan already with RM=4)

| run | pp t/s | tg t/s | step | verify | draft | catch-up |
|---|---|---|---|---|---|---|
| CUDA #1 (disturbed) | 978 | 46.1 | 61.5 | 50.0 | 7.1 | 2.2 |
| CUDA #2 | 1404 | 96.4 | 29.9 | 22.4 | 6.1 | 0.6 |
| CUDA #3 / #4 | 1327 / 1265 | 87.8 / 87.8 | 34.1 / 33.0 | 24.1 / 24.5 | 7.2 / 6.7 | 1.4 / 0.9 |
| Vulkan RM=4 #1 / #2 | 983 / 986 | 53.4 / 54.7 | 53.4 / 54.2 | 38.1 / 37.9 | 9.0 / 9.6 | 3.2 / 3.5 |

- **At 45k CUDA is ~1.6–1.8× faster than Vulkan** (both tg and pp). The "on par" finding of round 35 was probably based on a disturbed CUDA run.
- **Source of disturbance:** `gnome-remote-desktop-daemon` holds a CUDA context (C+G, 289 MiB).
  With an active remote desktop a CUDA run can drop to half speed (#1), while Vulkan feels it less. There should be no active remote work during measurements.

### Op level (`test-backend-ops perf`, with the model's shapes, µs)

| op | CUDA | Vulkan |
|---|---|---|
| FA 45k KV, head 256, GQA 4, batch 1 | 367 | 465 |
| **FA 45k KV, batch 3** | **380** | **1262** |
| q4_K 12288×4096 n=1 / n=3 | 75 / 100 | 81 / 140 (RM=4) |
| q6_K head 248320×4096 n=1 / n=3 | 2020 / 2498 | 1723 / 2561 |

**Root cause of Vulkan FA at batch 3:**
- On the GQA branch (`ggml_vk_flash_attn`, `N <= 8`) every token gets its own group of workgroups (`workgroups_x = neq1`), and each reads the whole K/V.
- So 3 tokens = 3×185 MB; over 8 attention layers that is ~7 ms per verify.
- A "split-major" workgroup order (hoping for L2 sharing) brought nothing (1245 vs 1262 µs) → reverted.
- The real fix is to pull the token×head rows into one tile (3D Q/mask/output addressing in the scalar, cm1 and cm2 FA shaders) — an open shader project (plan-part3 G6).
  *(Later done for the scalar shader: b50-results.md finding 24.)*

### CUDA: MMVQ rows_per_block 2 → 4 on Turing (ncols_dst 2–4) — DEFAULT

The same mechanism as on Vulkan (re-reading the y vector).
- Op level, n=3: q4_K 100 → 84 µs (−17%), q4_K 4096×12288 101 → 79 (−22%), q5_K −16%, q6_K −20%, head 2498 → 1991 (−20%).
- rpb=8 is worse overall.
- `test-backend-ops -o MUL_MAT`: OK.

45k A/B (interleaved, 2+2):

| | step | verify | tg t/s | pp t/s |
|---|---|---|---|---|
| rpb=2 | 29.5 / 29.5 | 22.0 | 97.7 / 97.3 | 1449 / 1380 |
| **rpb=4** | **27.0 / 26.8** | **19.3** | **105.8 / 107.2** | 1439 / 1439 |

→ **step −8.8%, verify −12%, tg +9.3%**, pp unchanged.
The output is bitwise identical in all 4 runs (and so is the acceptance).
Code: `mmvq.cu` `GGML_CUDA_MMVQ_TURING_RPB_SMALL` (default 4, `-D...=2` is the old one).

### G1: merging the catch-up into the draft (`LLAMA_MTP_DEFER_CATCHUP=1`, default off)

- Implemented (`common/speculative.cpp`).
- Along the way an **existing bug** came up in the qwen35 MTP graph:
  - It captured `t_h_nextn` *before* `get_rows(inp_out_ids)`.
  - So in masked mode, with a mixed batch (with and without outputs), it returned the h of the wrong row → acceptance 0.68 → 0.52.
  - Fixed (`src/models/qwen35.cpp`): in masked mode it captures after the crop, like the trunk.
  - Other models' MTP graphs (deepseek2, glm4-moe, …) may follow the same pattern; G1 was only tested on qwen35.
- After the fix: output bitwise identical, acceptance identical.
  On short ctx Vulkan 38.1 → 37.8 ms per step, CUDA 26.6 → 26.3 (~1%).
- The catch-up work moves into the first draft step, and the launch overhead is small → **not worth switching on**; it stays as an experimental switch.
  *(Later, on the eGPU laptop, it gave ~+5% on the B50: b50-results.md finding 2.)*

### Other

- G2 (greedy draft sampler) dropped based on measurement: the draft `TOP_K` takes ~26 µs per step on Vulkan, nothing to gain.
- The glslc path of `build-vulkan` pointed to the deleted `/tmp/opencode/vulkansdk`, so shader changes were not compiled.
  Fixed: `-DVulkan_GLSLC_EXECUTABLE=~/vulkan-sdk/1.4.363.0/x86_64/bin/glslc`.
- `tests/test-backend-ops.cpp`: the model's shapes were added to the perf cases (FA 45k batch 1/3, the main mat-vecs n=1/3).
- `build-cuda` and `build-vulkan` now have `LLAMA_BUILD_TESTS=ON`.

### Daily command (from 2026-10-27): CUDA

At 45k CUDA is ~2× faster (27 vs 53 ms per step), so the daily driver is CUDA:

```
build-cuda/bin/llama-server --port 8080 \
  -hf unsloth/Qwen3.5-9B-MTP-GGUF:UD-Q4_K_XL \
  -fa on -ctk f16 -ctv f16 --parallel 1 -c 80000 --no-mmproj-offload \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  --spec-draft-backend-sampling -bs \
  -t 24 -tb 24 --poll 0
```

Vulkan fallback: the same with the `GGML_VK_KQ_BATCH_RM=4` env from `build-vulkan`.
