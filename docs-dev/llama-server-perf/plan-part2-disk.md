# llama.cpp speedup — PART 2: plan that needs disk space

Status: **PARTLY EXECUTED (2026-10-26)** — disk space is available (142 GB free).
- **2.1 (A0–A4) DONE**, results in **round 36**. In short:
  - The quant mix gives +12…+21% tg without MTP (v4), but **no gain with MTP**: the MTP path is overhead-bound at 146 GB/s.
  - So the daily config stays UD-Q4_K_XL. The mix is only useful without MTP or to save VRAM.
- **2.2 (C, 160 B repack) DROPPED** — the C0 gate did not win (round 21).
- **2.3 (E2) OPEN**. n-max 3 is partly measured already: the 3rd draft position had an acceptance of 0.478.
- **2.4 → replaced** by G3 in `plan-part3-mtp-overhead.md` (reduced-vocabulary draft head, FR-Spec pattern).

Goal: reduce the weight traffic of decode/MTP with new model files (a custom quant mix). If the micro-benchmark wins, also with a new weight format aligned to 16 B.

Background: `plan-part1-nodisk.md`, `plan-ctranslate2.md` (rev. 2, archived).

---

## Disk requirements (prerequisite)

| item | size | note |
|---|---|---|
| **BF16 original (source — user decision)** | **~18 GB** | download; can be deleted after quantizing (frees 18 GB) |
| 2.1 quant-mix GGUFs (A1–A4) | ~5.2–5.9 GB each | quantized from BF16; 1–2 variants can be stored at a time |
| 2.2 160 B repack (if C0 wins) | ~6.4 GB | +8.1% compared to the current one |
| quality measurement (perplexity log) | <10 MB | — |

→ **Peak need for 2.1: ~24–25 GB free space** (BF16 18 GB + output ~6 GB).
- Deleting the BF16 after quantizing frees 18 GB.
- **Alternative:** quantize on an external disk (BF16 and output both there), then copy back only the finished ~5.5 GB GGUF.

---

## 2.1 — A: custom quant mix (no code, affects every backend)

**Why:** the GGUF analysis shows ~1.2 GB per step of "unnecessary" precision in the model:
- **LM head (`output.weight`) Q6_K, 796 MB** (*834 MB as measured in round 36*). It is **read 3 times per step** (verify + 2 drafts), which is 34% of the step's weight traffic.
- `ffn_down` Q6_K: 883 MB.
- `ssm_out` Q8_0: 408 MB (hybrid SSM layers).

**Tool:** `llama-quantize --output-tensor-type ... --tensor-type 'ffn_down=...' --tensor-type 'ssm_out=...'` (per-tensor overrides exist).

**Variants — ALL of them must be measured (user decision), memory-aware**

| version | contents | Δ traffic | expected tg | VRAM |
|---|---|---|---|---|
| A1 | head q5_K (700 MB) | −402 MB (−5.7%) | +4–6% | −96 MB |
| A2 | head q4_K (546 MB) | −750 MB (−10.5%) | +8–10% | −250 MB |
| A3 | A2 + ffn_down q5_K + ssm_out q5_K | ~−1014 MB (−14%) | +9–12% | −290 MB |
| A4 | all q4_K | ~−1197 MB (−17%) | +10–15% | −~700 MB |

**Memory-aware aspect (user):** a smaller mix lets **more layers fit on the GPU** with partial offload, and fits better next to other models.
So the measurement needs two criteria, and the winning mix must be reported for both:
1. full GPU offload, tg/pp;
2. **mixed GPU+CPU** mode (e.g. `-ngl` with ~8–16 layers on the CPU), tg/pp + VRAM.

**Mixed GPU+CPU validation:** measure WP-D (CPU FA split-KV) and the `-ngl` tuning separately. The smaller mix may gain more here than with full offload.

**Quality gate (mandatory)**
1. `llama-perplexity` for the baseline (UD-Q4_K_XL) vs the new mix, on the same corpus (e.g. `corpus/`).
2. KL divergence against the baseline logits, if a tool is available (`llama-perplexity --kl-divergence`).
3. Check **acceptance and mean length** with MTP. The draft uses the same head, so quantizing the head also affects acceptance.
4. Threshold: perplexity at most +2% and acceptance ≥0.68 is acceptable. Otherwise step back (q5_K).

**Expected:**
- MTP +8–15%, no-MTP +4–8%; pp roughly unchanged.
- VRAM 5.9 → ~5.2 GB, which frees room for f16 KV.

---

## 2.2 — C: 160 B q4_K aligned to 16 B + 16 B loads (only if C0 wins)

**Prerequisite:** the 1.3 C0 micro-benchmark shows at least a 15% gain for 16 B loads.

**Plan**
1. **C1 — new type:** q4_K with a **160-byte block** (128 B quants + 12 B scales + d/dmin + pad), aligned to 16 B.
   - It is a **lossless repack** (+8.1% size).
   - Converter: from the existing GGUF (a llama.cpp tool or our own script).
2. **C2 — CUDA/ROCm:**
   - MMVQ with 16 B (`uint4`) loads plus optional split-K;
   - MMQ fallback for large batches;
   - ROCm comes automatically via hipify, and compiling cleanly there is the acceptance criterion.
3. **C3 — CPU and Vulkan:**
   - CPU: `vec_dot` with 16 B (AVX2) loads;
   - Vulkan: mat-vec shader with `vec4` loads.
4. Measure MTP, no-MTP and pp with the usual protocol. Correctness: `detcheck.sh`.

**Expected:**
- +20–40% decode, if the hypothesis holds;
- pp possibly +5–10%, from fewer load transactions.

**Risk:** a new ggml type on every backend is a large surface, so only after the C0 gate.

---

## 2.3 — E2: re-measurements after 2.1

- `--spec-draft-n-max 3`: with a cheaper head the 3rd draft may pay off again.
- Combine with f16 KV; thanks to the VRAM savings of 2.1 it will fit.
- `-ub` 512 vs 1024 with the new weights.
- Put the winners into the daily config and the docs.

## 2.4 — (optional) separate, smaller head for the draft — REPLACED

> Instead: `plan-part3-mtp-overhead.md` G3 (the rows of the ~32k most frequent tokens, taken from the existing head,
> without a new GGUF tensor).

If the quality of a shared q4_K head is not good enough for the verify, add a separate `mtp.head` tensor (Q4_K) to the GGUF plus code, so that the draft uses it and the verify keeps Q6_K.
Expected: −500 MB per step, but acceptance may drop, so it has to be measured.

---

## Measurement protocol (same as in part 1)

- 170 W; lib/md5/env evidence; interleaved A/B with an anchor; 60 s cooldown; no edits or builds while measuring.
- **When switching models** (2.1): old model anchor → new model → old anchor again, because of the ±10% session drift.
- A variant stays only with a measured win. The losers go into `results.md`.

## Order

0. **Download the BF16 original** (~18 GB) and secure space for quantizing (internal or external disk).
1. 2.1 **A1 → A2 → A3 → A4**, then the quality gate.
   Measure every variant, memory-aware, with 1–2 files at a time.
2. 2.3 E2 (n-max 3, f16 KV combination, `-ub`) + **mixed GPU+CPU** validation.
3. (if the C0 gate won) 2.2 C1 → C2 → C3.
4. 2.4 optional (separate draft head).

---

## 2.0 — Unsloth Dynamic 3.0: what it means and how to apply it

**What it is:** Unsloth's next-generation dynamic quantization. It is pure PTQ, without QAT/QAD:
- much better imatrix calibration data (agentic coding, chat, multilingual);
- finer layer selection;
- more quantization techniques.

Their claims:
- **More than 10% better top-1 at the same size** compared to 2.0 and competitors; KLD and Divergence-300@32 are also better.
- The GGUF is standard, so it runs in llama.cpp.
- The MTP module is removed from quants smaller than `UD-Q2_K_XL` (~500 MB saved). It can be replaced by a separate Q4_0 MTP module. It stays in Q4_K_XL.

**Is it available to us now?**
- The `unsloth/Qwen3.5-9B-MTP-GGUF` repo was last modified on **2026-05-16**.
  - Every file there (`UD-*`) is a **2.0 mix**: Q4_K 3.02 + Q5_K 0.85 + Q6_K 1.76 + Q8_0 0.46 GB, without MXFP4.
  - The official 3.0 release so far only exists for **Qwen3.8-27B**, so there is no ready v3 file for our model yet.
- **BUT** the repo contains the model's **imatrix** (`imatrix_unsloth.gguf_file`, 5.2 MB), so we can produce a v3-style mix ourselves.

**Recipe (once ~25 GB of free space is available):**

> **Warning (round 36):** regex `--tensor-type` overrides are **not reproducible**. The internal counters of the default
> Q4_K mix shift which layers get Q6_K. Use a complete, explicit table instead: `~/llama-bench-results/mix-gen.py` →
> `--tensor-type-file` (`quant-mix.sh`). The command below only shows the intent.
```bash
hf download unsloth/Qwen3.5-9B-MTP-GGUF \
  Qwen3.5-9B-BF16.gguf imatrix_unsloth.gguf_file --local-dir ~/models/qwen35mtp
# v3-style dynamic mix: imatrix + per-tensor overrides
llama-quantize --imatrix imatrix_unsloth.gguf_file \
  --tensor-type attn_v=Q6_K --tensor-type ffn_down=Q6_K \
  --tensor-type ssm_out=Q8_0 --output-tensor-type Q6_K \
  Qwen3.5-9B-BF16.gguf out-UD-Q4_K_XL-v3.gguf Q4_K
```
Notes on the recipe:
- Syntax: `--tensor-type tensor_name=type` for single tensors, `--tensor-type-file` for a full table.
- Calibrate the exact mix from the 2.1 measurements and the Unsloth v3 ratios.
- Keep the **MTP head** separately at Q4_0/Q5_K or better. Otherwise the acceptance rate drops, and with it tg.

**Pros:**
- better quality at the same size, i.e. free accuracy with no runtime cost;
- the public imatrix makes it reproducible;
- standard GGUF.

**Cons:**
- No ready v3 file for 9B-MTP, so we have to quantize ourselves.
- Needs ~25 GB free disk (BF16 18 GB + output 6 GB) and several hours of runs and validation.
- We must **not** requantize from the existing Q4_K_XL (double loss); a BF16/Q8_0 base is needed.
- The quality gain has to be proven with our own harness.

**Decision:** step **A0** of the 2.1 custom quant-mix plan is this v3-style mix with the Unsloth imatrix, alongside or before the current A1–A4. Watch for an official v3 release of 9B-MTP.
