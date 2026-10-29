# llama.cpp speedup — PART 3: the overhead of the MTP step

Status: **plan** (2026-10-26).

Background: round 36 showed that the MTP step runs at 146 GB/s while decode without MTP runs at 345 GB/s, so the step is not byte-bound.
This plan targets concrete overhead sources identified in the code: `common/speculative.cpp`, `tools/server/server-context.cpp`, `src/llama-context.cpp`.

## What happens in one MTP step (n_max = 2, qwen35)

| # | call | context | batch | outputs | graph |
|---|---|---|---|---|---|
| 1 | verify (`llama_decode`) | tgt | 3 tokens | 3 | reuse |
| 2 | `process()` catch-up (`speculative.cpp:1656`) | dft | 3 tokens | **0** | **always rebuilt** |
| 3 | `draft()` step 1 | dft | 1 token | 1 | **always rebuilt** |
| 4 | `draft()` step 2 | dft | 1 token | 1 | reuse |

- Each step has **4 blocking `graph_compute` calls** plus 3 CPU-side sampling/readback steps.
- The catch-up decodes **all three** tokens of the verify batch into the dft KV, including drafts that are rejected later (acceptance is not known yet at that point).
- **Graph rebuilds:**
  - `get_gf_res_prev()` (`llama-context.cpp:2610`) keeps two slots, keyed by `n_outputs > 0`.
  - A graph is only reused when its slot was the last active one (`gf_res_prev_active == res`), because the sched allocation is shared.
  - The 0-output catch-up and the 1-output draft alternate, so the dft context **rebuilds twice per step**: sched reset + alloc + Vulkan graph optimization / CUDA graph re-capture.
- Estimate (short context, Vulkan, 250 W):
  - verify ~19–22 ms, 2 draft steps ~6 ms by bytes, catch-up ~1 ms;
  - that is ~28–30 ms expected, against the measured **49.3 ms**.
  - The ~20 ms gap is the main target. (Estimate, must be measured — G0.)

## Measurement bug in the existing instrumentation — FIXED (round 38)

`LLAMA_GRAPH_TIMING` (`llama-context.cpp`, local patch):
- `static int64_t n_graph` is **function-static, so the tgt and dft contexts share one counter**, and only every 16th call is printed.
- With MTP the cycle is exactly 4 graphs per step, so sampling every 16th call **always sees the same position** (here: the rebuilt catch-up graph).
- The round 36 observation ("it only prints `built=1` lines") looks like this aliasing, not a general sign that reuse is missing.

Fix:
- one counter per context;
- log every call, or aggregate statistics by `(ctx_type, n_tokens, n_outputs, built)`;
- print `n_outputs`.

The same applies to `n_sync % 16` in `synchronize()`.

## First result (round 38, preliminary)

The instrumentation fix is done. Short smoke test (Vulkan, 170 W, 300 tokens):
- the step takes ~44.8 ms;
- ~23 ms of it is in graphs: verify 15.0, 2 drafts 2×2.1, catch-up 2.4, sync ~1.5;
- so **~22 ms is outside the graphs**.

The catch-up and the 1st draft graph are rebuilt on every step (22/22, 22/42).
→ G0/2 (server-side timer) is the most urgent; G1 targets the GPU-side part (~3 ms).

## Round 39 — result of the breakdown (overrides the "~22 ms outside the graphs" reading above)

- The gap was **GPU time**: the Vulkan `graph_compute` does not block all the way through.
- Step (baseline):
  - verify ~32 ms (GPU 30.6);
  - draft ~8 ms (GPU ~4.6);
  - catch-up ~3 ms (GPU ~1.3);
  - server ~1 ms.
- Main cause: the verify n=3 k-quant mat-vec is 2.6× slower than n=1.
  - New **G5: `GGML_VK_KQ_BATCH_RM=4`** — verify −14%, t/s +10% (results.md round 39).
  - **G5 confirmed at 45k** (step −7.6%, tg +9.2%, results.md 39b), so it went into the daily command.
- New priority: G1 (draft/catch-up CPU side, ~4–5 ms) → G2 → G3. Check CUDA MMVQ with n=3.

## G0 — step breakdown (instrumentation, ~40 lines of code, env-gated)

1. ~~Fix `LLAMA_GRAPH_TIMING` + print the dft `n_reused`~~ — **DONE** (round 38).
2. ~~Server-side step timer~~ — **DONE** (`LLAMA_SERVER_TIMING`, round 39).
   It reports the ms/step average over 64 steps for each of:
   - the verify decode;
   - `common_speculative_process`;
   - `common_speculative_draft`;
   - verify sampling + accept;
   - the `yield_to_queue` wait.
3. `GGML_VK_PERF_LOGGER=1` on a short MTP run: time of the draft graph's `TOP_K` (248k vocab), `GET_ROWS`, the head `MUL_MAT` and FA.
4. The same on CUDA (`GGML_CUDA_DISABLE_GRAPHS=1` vs default: the cost of graph re-capture).

**Output:** the breakdown of the 49.3 ms, which decides the order of G1–G4.

## G1 — merge the catch-up into the first draft step (code, `speculative.cpp`)

**Idea:**
- For a generation batch, `process()` does not decode; it only records the (token, pos, h-row) triples.
- `accept()` trims them to the accepted prefix.
- Step 1 of `draft()` sends them in one batch: `[accepted catch-up tokens (no outputs)] + [id_last (with output)]`.
- Prefill (several ubatches) keeps the immediate catch-up.

What it gives:
- **One `graph_compute` and one sync fewer per step**, and rejected tokens never enter the dft KV.
- The "0-output" graph disappears from the dft context, removing one of the rebuilds caused by slot switching.
- It is also a known upstream TODO: `[TAG_SPEC_AVOID_DRAFT_REEVAL]` (`server-context.cpp:4619`).

Watch out for:
- the server's dft-side `seq_rm` on positions that no longer exist (should be harmless);
- the handling of `pending_h`;
- `n_seq > 1`.

Correctness: same prompt + temp 0 → **bitwise identical drafts and acceptance** compared with the old path. The draft tokens must not change, only the ordering/batching.

Expected: −3…−6 ms per step (**+6…+12% tg**), on every backend.

## G2 — greedy draft sampler instead of the backend top-k(10) (code, ~10 lines)

- The draft chain runs `top_k(10)` on the backend over the full 248k vocabulary (`speculative.cpp:1509`).
- With greedy drafting (`!params.probabilistic`), only `cur_p->data[0]` is needed.
- Fix: `llama_sampler_init_greedy()` → `ggml_argmax` (a single-pass reduction) instead of top-k, twice per step.
- The `p_min` branch with a threshold of 0 does not use `p`, but this must be checked.

- A cheap experiment; the G0/3 `TOP_K` time tells how much it is worth.
- Correctness: identical drafts (argmax = the first element of top-k).

## G3 — reduced vocabulary for the draft head (FR-Spec pattern; replaces 2.4)

The head is read 3 times per step (2.50 GB, 35% of the traffic), twice of that for the draft.
The draft only needs the rows of the ~32k most frequent tokens (FR-Spec: ~13% of the vocabulary covers the vast majority of tokens).

How:
- The rows of the embedded `output.weight` are contiguous, so the frequent rows can be cut into a new tensor with a byte copy (also Q6_K), for +~110 MB VRAM.
- The remapping pattern already exists: EAGLE-3 `d2t` (`src/models/eagle3.cpp:307`).
- Frequency list: our own corpus plus general HU/EN text tokenized via `/tokenize`, or as a first approximation the BPE order.

Cost and expected gain:
- Acceptance drops slightly: if the target token falls outside the 32k, the draft surely misses it.
  Measure **ms/step + acceptance** over at least 2000 tokens.
- Expected: −1.45 GB per step. The effect depends on how byte-bound the draft is (G0/3); if the draft is overhead-bound too, this brings little. **Decide after G0.**
- Code size: medium (model loading + qwen35 MTP graph + id remapping).

## G4 — the cost of dft graph rebuilds (only if G0 shows it matters)

Condition:
- some alternation remains after G1: step 1's batch size is 2–4 tokens depending on acceptance, step 2's is 1;
- and G0 shows the rebuild is expensive (CUDA graph recapture, Vulkan graph_optimize).

Then: pad step 1 to a fixed 1+n_max tokens with masked dummy rows without outputs, which gives a constant graph topology.
This is more complicated, so only after measuring.

## Measurement protocol

- **Check 170 W before every measurement.** In round 36 we ran at 250 W and hit thermal throttling at 60k; since the evening of 2026-10-26, 170 W is active again.
- Short context (2.8k) for stable A/B, 41k for confirmation.
- Measure **ms/step + acceptance**, not t/s.
- Interleaved A/B, 60 s cooldown, discard the first 2 runs (warm-up rule).
- Every code change behind an env switch (the old path can be restored), with the usual lib isolation (`LD_LIBRARY_PATH`).

## Order

G0 (instrumentation + timing-aliasing fix) → G2 (cheap) → G1 → G0 again → G3 / G4 depending on the breakdown.

## After round 40 — status and next steps

- G1 done (env, ~1%, not worth it), G2 dropped. G5 is env-gated on Vulkan and the default on CUDA (rows_per_block 4).
- **G6 (new, Vulkan, the largest remaining item):** FA for several tokens in the GQA path.
  - Change: on the `N <= 8` branch of `ggml_vk_flash_attn`, put the token×head rows into one workgroup, so K/V is read once.
  - Shaders:
    - `flash_attn_base.glsl` (`init_indices`, `gqaStore`);
    - `flash_attn.comp`, `flash_attn_cm1.comp`, `flash_attn_cm2.comp` (3D addressing of Q and the mask, output by `(t, h)`).
  - Expected: ~7 ms per verify at 45k (Vulkan step −13%).
  - Test: `test-backend-ops -o FLASH_ATTN_EXT`.
  - *(Done later for the scalar shader: b50-results.md finding 24.)*
- **G7 (CUDA):** the draft step takes ~6 ms (2 drafts, the head 2× ~2 ms). G3 (reduced-vocabulary draft head) is the largest item here.
