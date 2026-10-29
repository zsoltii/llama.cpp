# Status: llama-server speedup (Vulkan + CUDA)

Last update: the evening of 2026-10-26 (docs cleanup + round 38: MTP step breakdown, preliminary).
**The current daily command: results.md round 35** (Vulkan + MTP2 + f16 KV + `-bs`).

> This file covers the R720 / RTX 2080 Ti (VM) campaign up to round 40.
> The later laptop work (Intel Arc Pro B50, GTX 1080 Ti, RTX 2080 Ti over Thunderbolt) is in `b50-results.md`.

## Done

- [x] F0: installation. FA: corpus + `bench.py`. FB–FD: baseline + Vulkan + CUDA matrix. FE: MTP (on CUDA).
- [x] Round 6 (controlled):
  - f16 is +20% vs q4_1 (interleaved anchors);
  - FA VEC on Turing is negative → reverted;
  - `-bs` + draft backend sampling +9–13% (identical tokens: 499/261, 214/147);
  - p-min 0.3 brought nothing.
- [x] Round 7:
  - `-bs` confirmed in both pairs (R3 60.8/60.7 vs 57.8/55.5);
  - Vulkan f16 ≈ q4_1 but +1.6 GB → Vulkan stays on q4_1.
- [x] Code experiments measured, all reverted: BlockingSync (−25%), rm_kq=1 (−11% tg), FA VEC on Turing (−26% R3). **The tree is clean.**
- [x] Thermal drift handling: controlled protocol (interleaved anchor, `clocks-*.csv` + `clockstat.py`, 60 s cooldown).
- [x] Final documentation (results.md: rounds 6–7 + FINAL section).
- [x] Round 8: `GGML_CUDA_GRAPH_OPT=1` (dormant stream overlap) tested → warm pair 61.40 vs 61.44 → no gain, dropped.
- [x] New measurement regime: fixed 170 W power limit (systemd service `nvidia-powerlimit`), which removes the thermal drift.
- [x] Round 9 (DONE): reference re-measurement under 170 W with uniform q4_1 KV (CUDA + Vulkan), 8/8 rc=0. Main numbers:
  - CUDA win: R3 pp ~1160 / tg 47–56, R2 tg 65–76;
  - MTP gives ~2× long-context tg (23.8 → 47–56);
  - Vulkan MTP makes it worse (29.2 → 25.7); brew ≈ build-vulkan.
  Details: results.md round 9.
- [x] Round 10 (DONE): CUDA MMVQ↔MMQ A/B. MMQ is worse on every metric (R1 −15..17%, R2 −9..17%, R3 tg −10..12%).
  MMVQ stays; the patch was reverted, no code change. Details: results.md round 10.
- [x] 10b (DONE): Turing MMVQ nwarps variant (k-quant 2→4 warps).
  - No consistent gain: R1 +11.8%/+0.4%, R2/R3 ±3% = noise; acceptance bitwise identical.
  - Variant dropped, source and build restored (results.md round 10b).
- [x] Round 11 (DONE): Vulkan isolation + env switches.
  - MTP alone makes it worse (even without BS), FORCE_MMVQ makes it worse, DISABLE_MMVQ/nodes neutral → the VK optimum is the base; CUDA stays the daily driver.
  - PCIe: Gen3 x8 under load, not a bottleneck.
  Details: results.md round 11.
- [x] Round 12 (DONE): CPU affinity / NUMA pinning.
  - Neutral on the long-context metrics (R2 ±0.2%, R3 tg −0.6..−0.9%); node0 ≈ node1.
  - R1 +7–8% is within its spread, so it does not go into the daily command (results.md round 12).
  - The campaign was closed; open: the 170 W systemd service (sudo).

## Winners — HISTORICAL (state of 2026-10-18)

> **Outdated.** Since then f16 KV won (round 19) and Vulkan + MTP2 became the daily config (round 35).
> The commands below are snapshots of the 170 W q4_1 regime.

- CUDA — measured reference in the 170 W regime (q4_1, recommended):
  `build-cuda/bin/llama-server -hf unsloth/Qwen3.5-9B-MTP-GGUF:UD-Q4_K_XL
  -fa on -ctk q4_1 -ctv q4_1 --parallel 1 -c 80000 --no-mmproj-offload
  --spec-type draft-mtp --spec-draft-n-max 2 -t 1 -tb 1 --poll 0
  --spec-draft-backend-sampling -bs`
  (r9–r12: R3 pp ~1150–1240 / tg 53–57; R2 tg 75–77; R1 tg 66–80; VRAM ~7.7–8.0 GB; draft mean length 2.43; MTP long-context tg ~2×.)
- CUDA — f16 variant (winner of the pre-cap regime, not re-measured under 170 W):
  `build-cuda/bin/llama-server -hf unsloth/Qwen3.5-9B-MTP-GGUF:UD-Q4_K_XL
  -fa on -ctk f16 -ctv f16 --parallel 1 -c 80000 --no-mmproj-offload
  --spec-type draft-mtp --spec-draft-n-max 2 -t 1 -tb 1 --poll 0
  --spec-draft-backend-sampling -bs`
  (controlled: R3 pp ~1035 / tg 60.7; R2 tg ~72.7; VRAM 9198 MiB. Cool-session peak: pp ~1190, tg ~70.8.)
- Vulkan (fallback):
  `build-vulkan/bin/llama-server -hf unsloth/Qwen3.5-9B-MTP-GGUF:UD-Q4_K_XL
  -fa on -ctk q4_1 -ctv q4_1 --parallel 1 -c 80000 --no-mmproj-offload`
  (r11: R3 pp ~895–912, tg ~31.2–31.6; R2 tg ~46–48; VRAM 6605 MiB. NOT worth it with MTP: the spec path makes Vulkan slower — see r11.)

Notes:
- `-bs` is experimental: with a grammar or special samplers it automatically falls back to the CPU.
- Clocks/thermals: at 87–88 °C the card drops to ~1550 MHz (even when locked). A cool session gives 10–20% better numbers; the controlled measurements are the reliable baseline.
- KV standard for measurements: q4_1 (CUDA and Vulkan) in the 170 W regime. The f16 variant is the winner of the old regime (not re-measured under 170 W). iq4_nl dropped (CUDA FA did not support it).
- `corpus/` is untouched. The tree has local, env-gated (default off) modifications since then — list: plan.md switch table; `git diff --stat` 10 files.

Notes (environment):
- `--offline` was REMOVED: libssl-dev installed + rebuilt with OpenSSL 3.5.5, `-hf` works (loads from the cache, smoke-tested).
- Clock lock: NONE since the 2026-10-16 reboot (and it does not come back in the new regime). Instead a fixed 170 W power cap (systemd: `nvidia-powerlimit.service`). Revert: `nvidia-smi -pl 250 -i 0`.
  - **Warning:** rounds 35–36 ran at 250 W (the limit was not active then).
  - On the evening of 2026-10-26 `nvidia-smi` showed **170 W** active again.
  - Always check before measuring: `nvidia-smi --query-gpu=power.limit --format=csv`.
- KV standard: today **f16** (rounds 19 and 27); q4_1 is the VRAM-constrained variant; iq4_nl DROPPED (CUDA FA did not support it).
- Sampling in the measurements was temp=0; in real use your own temp/top-p settings apply (tg may differ by ±10%).

## Results in numbers (R3, 65k prompt)

| | pp | tg | VRAM |
|---|---|---|---|
| Original (brew, iq4_nl) | ~960 | ~33 | 7756 |
| Winner (CUDA, f16 KV, MTP, old regime) | ~1150 (+20%) | ~67 (+103%) | ~9220 |
| 170 W regime (CUDA, q4_1, MTP) | ~1150–1240 | ~53–57 | ~7700–8000 |

## Phase 2b — P0 code restore + P1 native q4_1 FA (2026-10-19)

**P0 — experimental switches restored.**
- All env-gated, default off; uncommitted in the tree, 4 files, +60/−1 lines.
- `GGML_CUDA_DISABLE_MMVQ`, `GGML_CUDA_BLOCKING_SYNC`, `GGML_CUDA_FA_FORCE_KERNEL=vec|tile|mma[:N]` (N = max batch, new), `GGML_VK_RM_KQ=1..4`, `-DGGML_CUDA_MMVQ_TURING_KQ_NWARPS4` (compile time).

**P1 — extend `GGML_CUDA_FA_QUANTS` with `q4_1-q4_1`.** build-cuda was rebuilt; `fattn-vec-instance-q4_1-q4_1.cu` exists.

**Methodology trap (important!):** the RUNPATH of `bin-cuda-ref/llama-server` is `/home/zsoltii/git/llama.cpp/build-cuda/bin`, NOT `$ORIGIN`.
- So in the first r13 round the "ref" runs also loaded the NEW libs, which made the A/B invalid.
- Fix: `LD_LIBRARY_PATH=$(dirname $bin)` per run (it takes precedence over RUNPATH).
- Check with `LD_TRACE_LOADED_OBJECTS=1` and look for the warning `no FlashAttention vector kernel compiled for K/V types q4_1-q4_1` in the server log (ref: present, var: absent).

**Quick confirmation (non-MTP, R3, 170 W):**
| lib | R3 tg | note |
|---|---|---|
| clean (ref, Oct 17) | 27.7 t/s | FA warning = 1, f16 conversion runs |
| new (q4_1 FA) | 41.3–41.6 t/s | FA warning = 0, native VEC kernel |

→ Removing the f16 K/V conversion in batch-1 decode gives **~+49%** tg (non-MTP path).
The MTP verify (batch=3) goes to MMA on Turing (f16 only), so the conversion stays there; we attack that with the `vec:4` switch (r14).

### r13 final result (with fixed lib isolation, 2026-10-19 08:30–08:50)

| run | R1 tg | R2 tg | R3 pp | R3 tg | R3 acc |
|---|---|---|---|---|---|
| ref1 (MTP) | 67.3 | 76.6 | 1148 | 55.7 | 0.7177 |
| var1 (MTP) | 73.0 | 75.5 | 1166 | 54.2 | 0.7177 |
| ref2 (MTP) | 72.8 | 73.6 | 1130 | 54.5 | 0.7177 |
| var2 (MTP) | 74.5 | 76.6 | 1120 | 53.9 | 0.7177 |
| var3 (MTP) | 67.1 | 75.9 | 1141 | 54.7 | 0.7177 |
| refN (non-MTP) | 57.6 | 61.8 | 1301 | **26.6** | – |
| varN (non-MTP) | 63.7 | 62.4 | 1311 | **42.3** | – |

**Verdict:** `GGML_CUDA_FA_QUANTS` += `q4_1-q4_1` **STAYS**.
- Non-MTP R3 +59%; MTP/pp neutral; acceptance bitwise identical.
- FA warning evidence: ref runs 1, var runs 0.

**r14 (running until ~09:35):** 7 runs on the q4_1 FA build, win config + MTP:
- base;
- `vec:4`: the MTP verify (batch=3) goes native VEC with no conversion; the main candidate;
- `CONV_FAST=1`: vectorized conversion kernel; detcheck gives bitwise identical output;
- mmqprec=q4, ub1024, mma:1 (control), base2 (anchor).

### r14 final result (2026-10-19 08:55–09:15)

| run | R3 pp | R3 tg | verdict |
|---|---|---|---|
| base | 1124 | 55.1 | anchor |
| vec:4 | 1148 | **41.1** | REJECTED (SIMT VEC is slower than MMA at batch 3) |
| CONV_FAST | 1164 | **59.6** | **WON (+8.1%)** |
| MMQ_PREC=q4 | 1139 | 55.0 | neutral |
| -ub 1024 | 1140 | 50.8 | REJECTED (−7.7%, VRAM +276 MiB) |
| mma:1 | 1163 | 54.8 | neutral (control) |
| base2 | 1128 | 54.2 | anchor |

- The v2 conversion kernel (one 32-bit load + 2× 8-byte stores instead of 8× 2-byte) **is now the default**; `GGML_CUDA_CONV_LEGACY=1` gives the old one. Correctness: detcheck → bitwise identical output.
- Numerical side effect of vec:4: R3 acceptance 0.71770 → 0.70283. That did not cause the −25%; mean length only went 2.43 → 2.41.
- **Log trap:** the static init of the FA-forcing wrapper runs during model loading (~0.5 s), where `GGML_LOG_*` messages are lost.
  So the evidence log was switched to `fprintf(stderr, ...)`; that is why the first evidence greps of r13/r14 found nothing.

**r15 (running until ~10:05):** a confirming A/B on the same binary: legacy vs default conversion kernel (6 runs, MTP + non-MTP pairs).
Then: Vulkan rebuild (RM_KQ) + r16, CPU profile (cpuprobe).

### r15 final result (2026-10-19 09:29–09:46)

On the same binary (md5 25fe9bb9):
- MTP R3 tg: legacy 54.32/54.24 vs **v2 default 58.24/60.01 (+7.2…+10.7%)**;
- pp +1…+3%; non-MTP ±2% (noise).

The log evidence of the conversion kernel choice was correct (1 vs 0).
→ The v2 kernel stays the default; `GGML_CUDA_CONV_LEGACY=1` is the emergency switch.

**Next:** Vulkan rebuild (RM_KQ switch) → r16 (9 runs, non-MTP base config) → CPU profile (cpuprobe, MTP win config, -bs vs without).

### r16 final result (Vulkan, 2026-10-19 10:06–10:40)

9 runs, base config + env switches. **Main lessons:**
- nocm2: pp 914 → 432 (halved without coopmat2) → coopmat2 is required.
- rmkq1: tg 31.5 → 29.8 (−5%) → the default rm_kq stays.
- nodot2/nodint/nomadd/memprio/nodes32: neutral (rep0 ±1.5%; the +1.5% of nodes32 is within the base/base2 anchor spread).
- **Measurement anomaly:** in some runs the R3 cached rep1/rep2 degraded internally (base2 rep1/rep2 −10%).
  So from now on rep0 (same fresh pp) is authoritative; median-based evaluation must be used with care on VK.
- The `GGML_VK_RM_KQ` evidence log (fprintf, 70 lines in the rmkq1 run, 0 elsewhere) works.

### Phase 2b — CLOSED (2026-10-19)

- Final: CUDA MTP + q4_1 KV + `-bs` (R3 tg 58–60), Vulkan base (31.5).
- 2 code changes stay in: q4_1 FA quants, vectorized conversion. Every other experiment was rejected or neutral.
- CPU profile:
  - the main thread is at ~100% because of the sync spin; the real CPU work is negligible;
  - the CPU sampler (without `-bs`) costs ~1.5–2 ms per token.
- Sudo task: install the 170 W systemd service.

### Phase 2b continued (evening of 2026-10-19) — more code experiments

- **CUDA conversion v3** (16-byte stores): implemented in the micro-benchmark but **identical to v2** (1.819 ms; the kernel is at the memory bandwidth limit: 387 GB/s vs memcpy 408) → not adopted.
- **Vulkan rm_kq 3/4** (r17): neutral (28.95/28.67 vs 28.61; anchor spread ±4%) → the default 2 stays.
  Environmental drift: the evening session gives ~10% slower absolute values, at higher SM clocks.
- **CUDA MMQ_MIN_BATCH=3** (r18, new env-gated switch): routing the verify batch to MMQ is **−13%** → rejected, MMVQ stays.
- nsys kernel profile: unusable because of an installation error (would need sudo).
- Summary: the remaining gain needs deep kernel work: the MMVQ verify path (~240 GB/s effective bandwidth) and the Vulkan mat-vec shaders. The space reachable with switches is exhausted.

### CTranslate2-inspired plan — split into 2 parts (2026-10-20)

- **plan-part1-nodisk.md**: runs with 2.1 GB of free space (measurements + code):
  B1 f16 KV A/B, E re-measurements (n-max, -ub), C0 GEMV micro-benchmark (gate), D CPU FA split-KV for quantized KV, B2 CUDA KV shadow (if needed), F config hygiene.
- **plan-part2-disk.md**: peak need ~24–25 GB (BF16 original 18 GB + output ~6 GB):
  custom quant mix (A1–A4, measure all, memory-aware), 160 B q4_K repack (only if the C0 gate wins), E2 re-measurements.
- User decisions: measure every quant variant; BF16 source; KV type A/B; code order C0 → D → B2; optimizing the mixed GPU+CPU mode is a goal too.

### Part 1 / B1 — KV type (2026-10-20) — CLOSED

- **f16 KV vs q4_1:** +14.2% / +14.5% (MTP R3, two pairs), +16.8% / +12.0% (no-MTP); pp unchanged, acceptance bitwise identical.
- Cost: +1.42 GB VRAM (MTP 9316 MB at 80k — fits).
- **Decision: daily config `-ctk f16 -ctv f16`**; q4_1 stays for VRAM-constrained cases.
- **B2 (KV shadow) dropped** (the shadow would be +2.1 GB over q4_1 — worse than f16).
- Next: round E (n-max/-ub, with f16 KV), C0 micro-benchmark, D (CPU split-KV).

### Part 1 — CLOSED (2026-10-20)

- **B1:** f16 KV +14.2% / +14.5% with MTP, +12–17% without MTP → daily config **f16 KV** (VRAM 9336 MB at 80k). q4_1 is the VRAM-constrained variant (7876 MB).
- **E:** n-max 3 (−8.4…−9.0%) dropped, n-max 1 neutral (2 stays), `-ub 1024` (−1.7…−2.5%) dropped.
- **C0:** 16 B loads are −16.5%; the 4 B GEMV reaches 540 GB/s (88% of peak) → **WP-C (160 B aligned format) dropped**.
- **FA kernel family** with f16 KV: neutral → default (VEC batch 1, MMA ≥2).
- **D** (CPU split-KV for quantized KV): correct (bitwise identical output), but **−70%** because each head re-reads the KV.
  It is env-gated (`GGML_CPU_FA_SPLIT_QUANT`), default off; do not use it.
- **F:** `--no-mmproj` optional (−879 MB RAM, same speed).
- Final validation: MTP R3 tg 66.1 t/s, pp 1145, acceptance 0.4375.
- Daily command: results.md → "Part 1 — FINAL configuration".

### Vulkan rounds (2026-10-20/21) — SDK + merge + measurements

- **Vulkan SDK 1.4.363.0** downloaded and partly extracted; `build-vulkan` rebuilt with the SDK's glslc.
  An error from an outdated shader generator came up and was fixed. Toolchain impact measured: **neutral** (round 26).
- **Round 27: Vulkan f16 KV → +10.8/11.5% tg, +12.5/13.2% pp**, cost +1725 MB.
  So the Vulkan daily profile is now **`-ctk f16 -ctv f16`**. Vulkan is still ~25% slower than CUDA, and MTP on Vulkan does not win.
- **Pipeline cache** implemented (`GGML_VK_PIPELINE_CACHE=<file>`, saved with `atexit` because the device destructor does not run).
  Measured: first prefill −9% on average, but noisy; **OFF by default**.
- **CPU measurement introduced** (`cpu-watch.sh` + `bench.py`): every run records the server's core count, sys% and iowait%.
  - Normal values: 0.5–0.9 cores, sys 1.5–4%, so the measurements are not CPU-bound.
  - A stuck server caused the earlier odd CPU/VRAM values.
- **Round 29: FORCE_MMVQ makes it worse** (also with f16 KV) → dropped.
- **Master merge:** 82 commits fast-forwarded, the 7 local modifications restored bit for bit.
  - Relevant new code: `qwen35.cpp` output reshape (2D), CUDA MMVQ low-bit, MMQ Stream-K cap, FA vec/tile fixes, CUDA graph cleanup.
  - The CUDA FA-oob idea already exists on Vulkan (`kv_active` fast path).
- **Unsloth:** the Dynamic 2.0 mix is in use (no MXFP4; Q4_K+Q5_K+Q6_K+Q8_0).
  Dynamic 3.0 is not released for 9B-MTP yet, but the **imatrix is available** (5.2 MB) → our own v3-style mix as step A0 of part 2 (plan-part2-disk.md).
- Next: merge-check A/B (round 31), coopmat A/B (round 30), `-O` shader experiment (round 32).
- **Round 30:** coopmat A/B neutral → default stays.
- **Round 31:** measuring the master merge.
  - Because of the +35% trap of a cold first run, a repeat in reversed order was needed.
  - On a warm basis: **CUDA −4.4%** (the cost of the FA vec WAR correctness fix), **Vulkan −1.7%** (noise) → the merge stays.
- **Closed:** final Vulkan config (results.md), ReBAR plan rev. 3, Unsloth Dynamic 3.0 as step A0 of part 2.
- **Round 32:** master merge #2 (31 commits → `e3797b184`), no conflicts, local modifications untouched.
  - New libs: CUDA `a73534b1`, Vulkan `df3700a4`.
  - Sanity: CUDA warm plateau 70.2/72.3 R3 tg, pp 1250; Vulkan 34.82 → **neutral**.
  - Warm-up rule confirmed: the first 2 CUDA runs are 5–12% slower.

### Rounds 33–34 — MTP re-measurement + root cause (Vulkan) — 2026-10-22

- **Model review:** the linear-attention block and the MTP graph of `src/models/qwen*.cpp` (qwen35, qwen35moe, qwen3next, qwen4exp) are practically identical.
  - qwen4exp `build_conv_state_at` only differs in the rollback-slot logic (we have `n_rs_seq = 0`); qwen3next is the older variant with more `cont`s.
  - **No graph-level gain to port.** The Vulkan graph optimizer already fuses the silu+mul, sigmoid+mul, rms_norm+mul and ssm_conv+silu chains.
- **MTP (60k, strict A/B, cooldown ≤62 °C):** base 30.5; the same mtp2 config measured 4×: 23.66 / 29.15 / 28.63 / 27.65 (mean 27.3, −11%, with a 17% run spread) → **MTP on Vulkan at 60k dropped.**
- **Correction:** `--spec-draft-backend-sampling` is ON by default.
  - So round 33's "dbs vs no dbs" comparison compared two identical configs; the +13% "winner" was run noise.
  - The earlier checkpoint analysis ("dbs is the main slowdown") is therefore invalid.
- **`-bs` (backend sampling) on the Vulkan daily config: +5.6% tg** (32.25 vs 30.54), pp identical → added to the daily command.
- **`-t 1 -tb 1 --poll 0`:** +11…+22% on short contexts, but pp −5% at 60k and −9% with MTP → only for short-context work.
- **Root cause:** decode on Vulkan is **not bandwidth-bound**: 291 GB/s effective out of 616, while the GEMV micro-bench reaches 540.
  - The time goes to gaps between many small kernels and a one-off graph cost.
  - Per `LLAMA_GRAPH_TIMING`: build 0.02–0.06 ms, backend `graph_compute` (GPU execution, blocking) 13–18 ms.
  - CUDA at the same point: 52.7 (base) / 73.9 (MTP2) t/s, against Vulkan 41.6 / 46.4.
- **New env switches (default off / unchanged):** `GGML_VK_FLOPS_PER_SUBMIT_DIV` (default 40; measured: no effect), `LLAMA_GRAPH_TIMING` (phase timing).
- **Tool fix:** the probe/screen script guard is now `pgrep -x llama-server`. The `BIN=.../llama-server` text in the wrapper's command line gave a false positive with `ps ax | grep`.

### Round 35 — 41k real context: MTP re-evaluated — 2026-10-26

- **The MTP verdict of rounds 33–34 is withdrawn.**
  - On a 41k prompt with 2000–8000 generated tokens, Vulkan+MTP2 gives **+30% tg** (48.7–49.1 vs 37.3–37.9 t/s).
  - It costs 4.37 s of pp (38.31 vs 33.94 s); break-even is ~670–700 generated tokens.
- **CUDA vs Vulkan are on par at 41k** (step time 44.4/48.0 vs 49.9/48.8 ms).
  - CUDA's 58.39 t/s is an acceptance artefact (0.794 vs 0.714).
  - Vulkan is better in pp (1075 vs 1034). CUDA's short-context advantage disappears at 41k.
- **Methodology lesson:** t/s depends strongly on the text (±9%, acceptance 0.66–0.79).
  - For MTP comparisons use **ms/step + acceptance**, or ≥2000 tokens with several repetitions.
  - `probe.sh` now prints `acc` / `mean_len` / `ms/step` / `ms/token`.
  - It can run on any GGUF with `MODEL=`; `SAVE=` also saves the response text.
- **`-t 1 -tb 1` at 41k (without MTP) is neutral** (pp −1%, tg +2%) → `-t 24 -tb 24` stays for long contexts.
- **`-b 4096 -ub 4096` = 8.0 t/s decode** (125 ms per step), with slower pp too; `-ub 512` (default) stays. Cause unexplored (VRAM hypothesis).
- **Daily config updated:** Vulkan + MTP2 + `-bs` + `-t 24 -tb 24 --poll 0` (results.md round 35). CUDA with the same flags is the fallback.

### Round 36 — Part 2 / A: custom quant mix (started) — 2026-10-26

- **Disk space arrived** (142 GB free) → Part 2 (plan-part2-disk.md) can start.
- BF16 base downloaded (`~/models/qwen35mtp/Qwen3.5-9B-BF16.gguf`, 18.41 GB) + `imatrix_unsloth.gguf_file` (5.2 MB, 248 entries, 77 chunks).
- **Tools** in `~/llama-bench-results/`:
  - `gguf-tensors.py`: tensor inventory, size/traffic;
  - `mix-gen.py`: explicit `--tensor-type-file` table from UD-Q4_K_XL;
  - `quant-mix.sh` (variants), `quant-ppl.sh`, `quant-mix-ab.sh`.
- **Important lesson:** with regex-based `--tensor-type` overrides plus the internal counters of the default Q4_K mix, the "replica" was not reproducible (9 tensors differed).
  - Fix: an **explicit, complete tensor table** (`^blk\.N\.tensor\.weight$=TYPE`).
  - With it, a0 is exactly 5840.36 MiB (= the data of the released 6124 MB file).
- **Ladder:** a0 (replica) → v1 (head Q5_K) → v2 (head Q4_K) → v3 (+ffn_down/ssm_out) → v4 (all Q4_K).
  Computed step traffic: −5.6% / −10.9% / −14.8% / −20.1%.
- **Raw screen (llama-bench, no MTP):** v1 +7.7%, v2 +12.1%, v3 +12.0%, **v4 +20.8%** tg; v4 pp +13%.
  The byte model is accurate, and Q4_K is also the faster native Vulkan path.
- **But no gain with MTP.**
  - Short-context stable A/B (3 rounds, ms/step): base 49.3; v1 47.9; v2 50.9; v3 51.6; v4 51.1 ms → at the edge of the spread.
  - Cause: the MTP step is 7.22 GB / 49.3 ms = **146 GB/s**, while the path without MTP does 345 GB/s → the MTP path is **overhead-bound**, not byte-bound.
- **Measurement pitfall (important):** the first reading of the 60k MTP A/B said v2 lost 15% — **false**.
  - The same base model went 45.06 → 36.99 t/s between the two rounds at the same clock and temperature.
  - The 60k run (~140 s) reaches the 2080 Ti's 88 °C limit and throttles. In a stable band the spread is ±1.5%.
- **The 170 W limit was not enabled** (250 W is the default), so this round ran at 250 W: +17% on short context compared with the 170 W values.
- **Harness:** `clocks.mem` sampling added; `MODEL=` (any GGUF), `SAVE=` (save the response), `ms/step` + `ms/token` output.
- **Tools:** `state-test.sh` (thermal state), `quant-short-ab.sh` (stable short-context A/B), `quant-diag.sh`.
- **Decision:** with MTP the daily config stays UD-Q4_K_XL.
  - The mix is only useful without MTP (+12…+21%) or to save VRAM (v4: −0.93 GB).
  - The largest open opportunity is the overhead of the MTP path (~+40% in theory).

### CTranslate2 line closed (2026-10-26)

- **CTranslate2 is not integrated.** It was never a dependency, backend or code, only a research source (`~/git/CTranslate2` + the rev. 2 analysis).
- **Rev. 2 archived into the repo:** `docs-dev/llama-server-perf/plan-ctranslate2.md` (with a status header).
  The references in `plan-part1-nodisk.md` / `plan-part2-disk.md` now point to this file; before, they pointed to `~/.opencode/plan/`, outside the repo.
- **Both derived experiments closed:**
  - C0 (16 B AWQ-GEMV loads) → round 21, dropped (4 B is the fastest, 540 GB/s);
  - D (CPU FA split-KV for quantized KV) → round 23, dropped (+70%).
- **Remaining code:** the `GGML_CPU_FA_SPLIT_QUANT=1` env switch (`ggml/src/ggml-cpu/ops.cpp:9586`, default off).
- **The premise was overridden:** instead of rev. 2's 1.7× GEMV headroom, today no-MTP runs at 345 GB/s and the MTP step at 146 GB/s (round 36).
  So the bottleneck is the MTP overhead, not the load width.
- **plan-part2-disk.md status updated:** 2.1 done (round 36), 2.2 dropped (C0), 2.3/2.4 open.

### Round 37 — int8 / CTranslate2 quantization: not transferable (2026-10-26)

- Code-free measurement done with a Q8_0 file (170 W, Vulkan, llama-bench): **pp4096 +1.8%**, **tg64 −32.2%** compared with Q4_K.
- CT2 uses int8 per-row scales + `cublasGemmEx` IMMA, for batched workloads.
  Our decode is 1–3 tokens and memory-bound, so int8 with 2× the bytes structurally loses.
- Vulkan on Turing has no int8 tensor cores (coopmat is Ampere+).
  - On CUDA the `mma.s8` helper exists, but MMQ only uses it on sm_80+.
  - Such a path would only give ~2× on the prefill (~9% of wall time), at the cost of a new type + kernels.
- **Decision: no int8 implementation.** Details: results.md round 37.

### Docs cleanup + round 38: MTP step breakdown (evening of 2026-10-26)

- **New plan:** `plan-part3-mtp-overhead.md` (G0–G4: the overhead of the MTP step).
- **Instrumentation bug fixed:** the `LLAMA_GRAPH_TIMING` counter was shared between the main and the MTP context and printed every 16th call.
  - That aliased with the 4-graphs-per-step cycle: it always sampled the same graph.
  - It now aggregates per context by `(n_tokens, n_outputs)` and prints every 64 graphs.
  - The server also prints the draft context's `graphs reused`.
- **`mmvq.cu`:** the `getenv` of the two experimental env switches is now cached statically (it used to run on every mul_mat dispatch).
- **Round 38 (preliminary, 1 run):**
  - The dft context's 0-output catch-up graph is rebuilt every step (22/22), and so is the 1st draft step (22/42).
  - Time spent in graphs is ~23 ms per step, while the measured step is **~45 ms** → **~22 ms per step outside the graphs** (server/sampling/readback).
  Details: results.md round 38.
- Docs contradictions fixed:
  - plan.md: environment, round statuses, switch table;
  - plan-part1: status, f16 VRAM cost;
  - plan-part2: head size, 2.0 recipe, 2.4 → G3;
  - results.md round 36: timing note.

### Round 39 — the root cause of the "22 ms" (evening of 2026-10-26)

- **The reading of round 38 was wrong:** the Vulkan `graph_compute` does not block all the way through, so the gap is GPU time.
  - New server timer (`LLAMA_SERVER_TIMING`): the server itself takes ~1 ms per step.
  - The target verify takes ~32 ms, of which the GPU is ~30.6 ms.
- **Root cause:** the verify n=3 k-quant mat-vec (`q4_K 12288×4096`: n=1 65 µs, n=3 168 µs).
  The float shader re-reads the activation per column for every 2 rows, and the MMVQ path is no better.
- **New switch `GGML_VK_KQ_BATCH_RM=4`:** verify −14%, step −9…−10%, **t/s +10%** (short ctx, 2×3 interleaved runs).
  Default off; to be confirmed at 41k, then it can go into the daily command.
- Remaining: draft ~8 ms (GPU ~4.6) + catch-up ~3 ms (GPU ~1.3) → G1.
- **39b — confirmed at 45k:** ms/step 57.8 → 53.3 (−7.6%), tg 49.4 → 53.9 (+9.2%), pp unchanged, output deterministic.
  The daily Vulkan command now has the `GGML_VK_KQ_BATCH_RM=4` env (results.md 39b).

### Round 40 — CUDA vs Vulkan + fixes (night of 2026-10-27)

- **At 45k CUDA is ~1.7× faster than Vulkan** (29.5 vs 53.8 ms per step). The "on par" reading of round 35 came from a disturbed CUDA run (the remote desktop uses CUDA).
- **CUDA: MMVQ rows_per_block 4 (Turing, ncols 2–4) — now the default:** at 45k step −8.8%, tg 97.5 → 106.5 t/s, output bitwise identical.
- **Vulkan FA with batch 3 = 3× KV reads** (1262 vs CUDA 380 µs), ~7 ms per verify.
  - Fixing it is shader work (plan-part3 G6). A split-major experiment had no effect.
  - *(Later fixed for the scalar shader: b50-results.md finding 24.)*
- **G1** done (env, default off, ~1%); along the way the masked `t_h_nextn` bug of the qwen35 MTP graph was fixed.
- Daily driver: **CUDA** (results.md round 40).
