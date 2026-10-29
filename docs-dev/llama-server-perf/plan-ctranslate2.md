# llama.cpp speedup — CTranslate2-inspired plan (rev. 2)

> **STATUS: ARCHIVED (2026-10-26).** The content was merged into `plan-part1-nodisk.md` / `plan-part2-disk.md`.
> The two experiments that came from it are **closed**:
> - **WP-C0** (16 B AWQ-GEMV loads + split-K) → round 21: **dropped**. The 4 B load is the fastest (540 GB/s); 16 B is −16.5%.
> - **WP-D** (CPU FA split-KV for quantized KV) → round 23: **dropped** (+70%).
>
> Later measurements **partly overrode** the premise of rev. 2:
> - The premise: decode at 231 GB/s against a 408 GB/s ceiling, i.e. 1.7× headroom.
> - Measured since: no-MTP 345 GB/s, MTP step 146 GB/s (round 36).
> - So the bottleneck is the overhead of the MTP path, not the load width.
>
> Original location: `~/.opencode/plan/llamacpp-perf-ctranslate2.md`. The text below is rev. 2, with only the status
> header added (translated to English on 2026-10-29).

---


> **Executable version, split into two parts (because of the disk limit):**
> - `llamacpp-perf-part1-nodisk.md` — now `plan-part1-nodisk.md` (measurements + code)
> - `llamacpp-perf-part2-disk.md` — now `plan-part2-disk.md` (quant mix, repack)
>
> As the first implementation step, both files go under `docs-dev/llama-server-perf/`.

Status: **plan, awaiting approval** (Plan mode; implementing needs an agent switch).

Goal: Qwen3.5-9B-MTP, RTX 2080 Ti, 170 W, 80k context, q4_1 KV. The largest measured tg/pp gain that stays portable across CPU, CUDA, Vulkan and ROCm.

---

## 0. Executive summary — what changed since rev. 1

1. **Priorities were reversed.**
   - Rev. 1 put the CUDA GEMV / reformatted weights first: large, risky and unproven.
   - The critical review showed that the model carries **~1.2 GB per step of "unnecessary" precision**. It can be removed with **zero code** and **takes effect on every backend immediately**.
   - That became WP-A (first).
2. **The 16-byte load hypothesis got a gate.** First a 30-minute synthetic GEMV micro-benchmark decides (4/8/16 B loads, same traffic). The format work starts only if it wins.
3. **The format plan got simpler.** Not AWQ-style separated scales, but a **160-byte q4_K block aligned to 16 B** (+8% size, lossless repack). It gives both the alignment and the wide loads.
4. **New item: the KV type.**
   - An earlier measurement showed **f16 KV + MTP ≈ +4–5%** (62.18 vs 59–60 t/s), with no code.
   - The user had prescribed q4_1 as the measurement standard, so **a decision is needed**.
   - If q4_1 stays, the CUDA "KV shadow" follows (removing the redundant conversion).
5. **New item: re-measure n-max** after WP-A. The draft now reads the head 3 times; with a cheaper head a 3rd draft may pay off again.
6. **New small item: `--no-mmproj`.** The vision projector is loaded needlessly (RAM + load time).

---

## 1. Facts (code + measurements + GGUF analysis)

### Model composition (UD-Q4_K_XL, from the GGUF header)
| part | size |
|---|---|
| layers 0-31 | 4431.7 MB |
| MTP block (blk.32, 15 tensors) | 150.7 MB |
| **LM head (`output.weight`, Q6_K, 248 320 vocabulary)** | **795.7 MB** |
| token_embd (Q4_K, gather → traffic ~0) | 560.8 MB |
| **total** | **5939 MB** |

Type mix:

| type | size | of which |
|---|---|---|
| Q4_K | 2963 MB | |
| Q6_K | 1678 MB | head 796, `ffn_down` 883 |
| Q5_K | 827 MB | |
| Q8_0 | 442 MB | `ssm_out` 408, `eh_proj` 34 |
| F32 | 28 MB | |

The model is **hybrid**: 24 SSM layers (`ssm_out` is Q8_0!) and 9 attention layers.

### Weight traffic of one MTP step
- verify (32 layers + head): **5228 MB**
- 2 draft decodes (MTP block 151 + head 796): 2 × 947 = **1894 MB**
- **total: 7120 MB per step; the head is 34% of it** (read 3 times: verify + 2 drafts)
- on top: the FA conversion (~700 MB) + f16 KV reads (~534 MB) → ~8.3 GB per step

### Measured numbers (170 W, q4_1)
- no-MTP: 42 t/s → 231 GB/s effective weight bandwidth
- MTP R3: 59–60 t/s, 41.1 ms per step, acceptance 0.7177, mean length 2.43
- pp: 1180–1300 t/s
- Upper bound of the memory system (memcpy D2D, convbench): **408 GB/s**, so the mat-vec runs ~1.7× below the peak.

### CTranslate2 — what explains its speed
- Its own README and author forum list:
  - precision (fp16/int8);
  - vendor BLAS (MKL/cuBLAS; `CUBLAS_COMPUTE_16F` is 2× on GeForce);
  - layer fusion (masked softmax, fused QKV);
  - padding removal and batch reordering;
  - in-place ops and caching.
- Its own **AWQ int4 GEMV**: 16-byte (`float4`) weight loads + split-K + warp reduction (`src/ops/awq/gemv_gpu.cu`).
- Its own FlashAttention: **sm_80+ only**, so it does not run on Turing. It was removed from the Python package ("minimal performance gain").
- Benchmark (opus-mt, 4 CPU threads): transformers 1023 → CT2 fp16 9297 tok/s; memory 4097 → 909 MB.

### What llama.cpp already has
- split-KV FA on all three backends:
  - CUDA: VEC `parallel_blocks` / MMA stream-K;
  - Vulkan: `fa_split_k_reduce`;
  - CPU: chunk partials, **for f16/f32 KV only**.
- fused QKV/FFN, quantized weights, a caching allocator, CUDA graphs;
- the `GGML_CUDA_CUBLAS_COMPUTE_TYPE` env switch.

---

## 2. Critical review — what was wrong with rev. 1

1. **Priority:** the largest and riskiest item came first, and the cheapest one with the biggest return (quant mix) was not even highlighted. → WP-A became first.
2. **Unproven hypothesis:** the benefit of 16 B loads was not proven, so a **C0 micro-benchmark gate** comes first.
3. **Format plan too big** (AWQ style) → a simpler 160 B aligned block (lossless, +8% size).
4. **Missing KV decision:** f16 KV vs q4_1 → split into B1/B2.
5. **Missing measurement risk:** evening drift is ±10%. Use only in-session, interleaved A/B with an anchor, and a separate anchor when switching models.
6. **The head is read 3 times.** This was missing from rev. 1, and it is the largest single item.
7. **mmproj** is loaded needlessly → `--no-mmproj`.

---

## 3. Work packages (by priority)

### WP-A — custom quant mix (no code, affects every backend) ★ first
- **What:** requantize the head and the overly precise tensors.
- **Tool:** `llama-quantize --output-tensor-type q5_K|q4_K --tensor-type 'ffn_down=q5_K' --tensor-type 'ssm_out=q5_K'` (per-tensor overrides exist).
- **Variants / expected traffic reduction (MTP step):**
  - A1: head q5_K (700 MB) → −402 MB (−5.7%)
  - A2: head q4_K (546 MB) → −750 MB (−10.5%)
  - A3: A2 + ffn_down q5_K + ssm_out q5_K → ~−1014 MB (−14%)
  - A4: all q4_K → ~−1197 MB (−17%)
- **Expected speed:**
  - tg +8–15% with MTP, +4–8% without;
  - pp roughly unchanged (compute-bound);
  - VRAM 5.9 → ~5.2 GB, which frees room for f16 KV.
- **Quality gate:**
  - `llama-perplexity` + KL against the baseline;
  - an **acceptance check**: the draft uses the same head, so acceptance may change; look at the mean length too.
- **Portability:** automatic; CPU, CUDA, Vulkan and ROCm all consume the same GGUF.

### WP-B — KV: f16 vs q4_1, or a CUDA KV shadow
- **B1 (config, no code):** re-measure `-ctk f16 -ctv f16` with MTP.
  - Expected +4–5%.
  - VRAM +450 MB at 80k (7.75 → 8.2 GB, fits; WP-A even frees some).
- **B2 (code, if q4_1 stays):** a **KV shadow** for the CUDA MMA path.
  - An f16 copy is made next to the KV write and FA reads it, so the full-KV conversion (700 MB per step) disappears.
  - Risk: invalidation on cache shift/rollback (epoch counter). The VEC path (batch 1) is unchanged.
- **Expected:** +4–7% with MTP.

### WP-C — GEMV micro-benchmark → (if it wins) 160 B aligned q4_K + 16 B loads
- **C0 (gate, ~30 min):** a synthetic kernel benchmark with 4/8/16 B loads and the same 5.5 GB of traffic → GB/s. It decides whether there is any bandwidth gain at all.
- **C1:** new type: q4_K with a **160-byte block** (128 B quants + 12 B scales + d/dmin + pad), aligned to 16 B. A **lossless repack** (+8.1% size).
- **C2:** CUDA MMVQ (+MMQ fallback for large batches) with 16 B loads; ROCm comes automatically via hipify.
- **C3:** CPU `vec_dot` (16 B / AVX2) and a Vulkan mat-vec shader (`vec4`) for the same layout.
- **Expected:** +20–40% on decode, **if** the hypothesis holds. A large amount of work: new ggml type + converter + 3 backends.

### WP-D — CPU FA split-KV for quantized KV (portability)
- `ggml/src/ggml-cpu/ops.cpp:9566`: extend the `kv_is_f32_or_f16` condition to quantized KV. The chunk-partial + LSE reduce path already exists.
- Expected: +10–30% on long CPU contexts; CUDA and Vulkan unchanged; ROCm not affected.

### WP-E — re-measurements after WP-A
- `--spec-draft-n-max 3` (with a cheaper head), `-ub`, FA kernel choice, combination with f16 KV.
- Possibly a **separate, smaller head** for the draft (extra GGUF tensor + code), only if the quality of a shared Q4_K head is not good enough for the verify.

### WP-F — small items
- `--no-mmproj` (RAM + load time).
- Batch reordering (sorting slots by length) only matters with several parallel requests; for us (`--parallel 1`) it is negligible.

---

## 4. Portability matrix

| WP | CPU | CUDA | Vulkan | ROCm | Note |
|---|---|---|---|---|---|
| A quant mix | ✓ | ✓ | ✓ | ✓ | GGUF only, no code |
| B1 f16 KV | ✓ | ✓ | ✓ | ✓ | config |
| B2 KV shadow | – | ✓ | – | ✓ (hipify) | CUDA-specific optimization |
| C repack | ✓ | ✓ | ✓ | ✓ | new type; every backend needs a kernel |
| D CPU split | ✓ | – | – | – | CPU only |
| F mmproj | ✓ | ✓ | ✓ | ✓ | config |

ROCm: `ggml-hip/CMakeLists.txt` compiles the `../ggml-cuda/*.cu` sources (hipify in the build).
So a CUDA change carries over automatically when it uses no CUDA-only API. **A clean build is the acceptance criterion.**

---

## 5. Measurement protocol (mandatory, by the campaign rules)

- Before measuring:
  - check 170 W;
  - log evidence: lib path, md5, env.
- During measuring:
  - interleaved A/B with an anchor;
  - 60 s cooldown;
  - no source edits or builds.
- Evaluation:
  - rep0 (Vulkan R3), medians, delta pairs;
  - rows in `results/summary.csv`;
  - check acceptance and mean length.
- **When switching models** (WP-A): old model anchor → new model → old anchor again. Otherwise the ±10% session drift swallows the effect.
- A WP stays only with a **measured win**; the losers are documented (results.md).

---

## 6. Open decisions (need the user's answer)

1. **WP-A head:** may it go to **q4_K** (largest gain, ~−10.5%), or the safer **q5_K** (−5.7%)?
2. **WP-A source:**
   - requantize from the existing UD-Q4_K_XL (fast, already available);
   - or download the **BF16 original** (better quality, ~18 GB download).
3. **WP-B:** is **f16 KV** acceptable (+450 MB VRAM, +4–5%) instead of q4_1, or does q4_1 stay with the CUDA shadow?
4. **WP-C:** if the C0 micro-benchmark wins, do we start the **160 B format**? It is large work: new type + 3 backends + converter.
5. **WP-E:** is a separate draft head needed (extra model tensor + code), or is the shared head enough?

---

## 7. Order (proposed)

1. **WP-A1/A2** (quant mix, no code) — 1–2 hours of measurement + quality gate; expected to be the largest cheap gain.
2. **WP-B1** (f16 KV) — 1 hour, if the user allows it.
3. **WP-E** (n-max, -ub re-measurement) — 1 hour.
4. **WP-C0** (micro-benchmark gate) — 30 minutes; decides on the large work.
5. **WP-D** (CPU split-KV) — half a day; a portability demonstration.
6. **WP-C1–C3** (only if C0 wins) — large work.

**Implementing requires an agent switch.**
