# llama.cpp speedup — PART 1: plan that needs no disk space

> **STATUS: EXECUTED (2026-10-20).** Results: `results.md` rounds 19–24, "Part 1 — FINAL configuration", and
> `progress.md`. Summary:
> - B1 (f16 KV) won (+14–16%).
> - B2 dropped.
> - E (n-max 3, ub1024) dropped.
> - C0 → WP-C (160 B format) **dropped**.
> - FA kernel family neutral.
> - D (CPU split-KV for quantized KV) is **env-gated, default off** (measured −70%).
> - F (`--no-mmproj`) optional.

Status: **executed** (see the status block above; the text below is the original plan).

Goal: Qwen3.5-9B-MTP, RTX 2080 Ti, 170 W, 80k context. Measured tg/pp gains that stay portable across CPU, CUDA, Vulkan and ROCm.

Constraint: **2.1 GB free disk space**, so this part has **no new model file**, only measurements and code (incremental build).

**Decisions (user, 2026-10-20):**
- **Every variant** of the quant mix must be measured (q5_K, q4_K, A1–A4), **memory-aware**. The VRAM footprint matters: a smaller mix lets more layers fit on the GPU.
- The quant mix is made from the **BF16 original** (~18 GB download), so it belongs to part 2, once disk space is available.
- **Measure both** KV types (q4_1 vs f16); the faster one stays.
- Order of the Part 1 code work: **C0 → D → B2**.
- **Mixed GPU+CPU mode must also be optimized**, so partial offload is fast too.

Background: `plan-ctranslate2.md` (rev. 2, archived; its content was merged here), `docs-dev/llama-server-perf/results.md`.

---

## 1.0 — Documentation (FIRST STEP, for the implementing agent)

Put the two plans into the repo:
- `docs-dev/llama-server-perf/plan-part1-nodisk.md` ← this file
- `docs-dev/llama-server-perf/plan-part2-disk.md` ← part 2
- `docs-dev/llama-server-perf/progress.md`: new entry "CTranslate2-inspired plan, split into 2 parts (because of the disk limit)".

The existing `plan.md` and `results.md` stay unchanged. The rev. 2 analysis (`plan-ctranslate2.md`, archived) was merged into the two parts.

---

## 1.1 — B1: f16 KV vs q4_1 (no code)

**Why:** the CUDA MMA path converts the whole KV to f16 on every step for batch ≥2, i.e. the MTP verify. That is ~700 MB per step at 65k.
- With f16 KV this conversion disappears. An earlier measurement showed **+4–5% with MTP** (62.18 vs 59–60 t/s).
- In exchange, batch-1 decode reads more KV (534 vs 157 MB/token) and needs +450 MB VRAM at 80k.
  *(That was an estimate; the measured cost was +1.42 GB, round 19.)*

**Steps**
1. Interleaved A/B: `q4_1` anchor → `f16` → `q4_1` anchor → `f16` (4 runs, 170 W, MTP win config).
2. The same without MTP (2+2 runs). Here f16 is expected to lose because of the larger KV reads.
3. VRAM check at 80k. It must fit: 7.75 GB + ~0.45 GB ≈ 8.2 GB.
4. Check acceptance and mean length. f16 KV is numerically better, so acceptance may change.

**Acceptance:**
- If MTP tg improves and VRAM fits, the daily config's KV type becomes f16. The no-MTP config may stay on q4_1.
- **If it does not win**, 1.5 (B2, KV shadow) takes over.

---

## 1.2 — E: re-measurements with the current model (no code)

**Why:** the earlier sweeps (n-max 2/3/1/4) were made with the *expensive* Q6_K head and the old conversion kernel. The conversion has since improved by +9% (r15), which may shift the optimum.

**Steps**
1. `--spec-draft-n-max 3` vs `2` (interleaved, 4 runs). The 3rd draft position had an acceptance of 0.478.
2. A control run with `--spec-draft-n-max 1`, to validate the old 49.99 vs 54.82 ratio.
3. Re-check `-ub` (512 default vs 1024), to confirm or override the earlier −7.7%.
4. FA kernel choice for the verify step: `GGML_CUDA_FA_FORCE_KERNEL=mma` (default) vs `vec:4`, to confirm the earlier −25% after the conversion speedup.

**Acceptance:** any difference above 2% that is consistent over 2 pairs → update the daily config and the docs.

---

## 1.3 — C0: GEMV micro-benchmark (gate for the part 2 format work)

**Why:** the decode mat-vec runs at ~231 GB/s while the memory system can do ~408 GB/s, so there is ~1.7× headroom.
Hypothesis: 4-byte loads (VDR=2) limit memory-level parallelism. 16-byte (uint4) loads plus split-K, following the CTranslate2 AWQ GEMV pattern, should help.

**Steps**
1. A standalone .cu (`/tmp/opencode/gemvbench.cu`, ~10 MB) with the same 5.5 GB of traffic in 3 variants:
   - 4 B loads (current MMVQ scheme, VDR=2)
   - 8 B loads (VDR=4)
   - 16 B loads (uint4, VDR=8) + optional split-K
2. Metrics: GB/s and ms, repeated with a cold L2 (large buffer).
3. Compare 231 GB/s (actual MMVQ) against 408 GB/s (memcpy reference).

**Gate:**
- If the 16 B variant is **at least 15% faster**, part 2's C is justified: a 160 B aligned q4_K format plus kernel compilation.
- If not, the format work is **dropped** and decode stays as it is.

---

## 1.4 — D: CPU FA split-KV for quantized KV (code, CPU only)

**Why:** the split-KV path in `ggml/src/ggml-cpu/ops.cpp` currently handles **only f16/f32 KV** (`kv_is_f32_or_f16`, line 9566).
With quantized KV (q4_1) the slower path without chunking runs. The chunk-partial + LSE-reduce infrastructure already exists.

**Steps**
1. Extend the condition to quantized K/V. The chunk computation uses the dequantized K/V, like the generic path.
2. Correctness: `test-backend-ops -o FLASH_ATTN_EXT` (CPU) against the reference; near-bitwise agreement with the non-split path, within tolerance.
3. Measure CPU `llama-bench` (or the server with `-ngl 0`) at 4k/32k/65k context, with and without the split.
4. CUDA and Vulkan are **not affected** (they have their own FA), so their builds stay unchanged.

**Acceptance:** above +10% on long CPU contexts with correct results → keep. Otherwise it stays an env-gated experiment.

---

## 1.5 — B2: CUDA KV shadow (code, only if B1 does not win)

**Why:** if q4_1 KV stays, the verify (MMA) path converts the whole KV to f16 (~700 MB per step).
A persistent f16 shadow KV, updated together with the KV write, removes that conversion.

**Steps**
1. Allocate a persistent f16 scratch the size of the KV cache (~655 MB at 80k; it fits in VRAM).
2. Write f16 next to the KV write (the graph `cpy` into the cache), inside the CUDA backend, alongside the q4_1 cache.
3. MMA FA reads the shadow; the VEC path (batch 1) still reads q4_1.
4. Invalidation: on cache shift, rollback or defrag, an epoch counter triggers a full reconversion.
5. Measure an MTP R3 A/B (4 runs). Correctness: `detcheck.sh` (near-bitwise output) plus acceptance.

**Acceptance:** above +4% MTP tg with correct output. Otherwise drop it and document it.

---

## 1.6 — F: config hygiene (no code)

1. `--no-mmproj`: the vision projector is not needed (saves RAM and load time). Check that text inference is unchanged.
2. Batch reordering (sorting slots by length) **only matters with several parallel requests**. We use `--parallel 1`, so it is documented, not implemented.
3. Finalize the daily config with the winners (depending on f16 KV, n-max and -ub).

---

## Measurement protocol (mandatory)

- Before measuring:
  - check 170 W;
  - collect evidence: lib path, md5, env.
- During measuring:
  - interleaved A/B with an anchor;
  - 60 s cooldown;
  - no source edits or builds.
- Evaluation:
  - rep0 (Vulkan), medians, delta pairs;
  - `results/summary.csv`;
  - acceptance and mean length.
- Session drift is ±10%, so only compare within the same session; repeat when in doubt.

## Disk impact

- **No new model file**, only `/tmp/opencode` (~10 MB) and incremental builds into the existing `build-cuda`/`build-vulkan` directories.
- The CPU test needs no new build dir: `build-cuda/bin/libggml-cpu.so` can be used (`-ngl 0`).

## Mixed GPU+CPU mode (user requirement)

- Goal: the config must stay fast when not every layer fits in VRAM (partial offload, running next to other models).
- Test scenario: the winning config plus partial offload (`-ngl` set so that ~8–16 layers run on the CPU) → tg/pp + VRAM.
- CPU-side attention gets faster thanks to WP-D (split-KV for quantized KV); this is the key to mixed mode.
- When choosing the part 2 quant mix, the **VRAM footprint is a criterion too**. A smaller (Q4_K) mix lets more layers fit on the GPU, which gives a bigger gain in mixed mode.
- Validate the `-ngl` tuning and the CPU FA with separate measurements, both on the CPU and in mixed mode.

## Order (user decision: C0 → D → B2)

1.0 docs → 1.1 B1 f16 KV → 1.2 E re-measurements → 1.3 C0 gate → 1.4 D CPU split-KV → 1.5 B2 (if needed) → 1.6 F.
