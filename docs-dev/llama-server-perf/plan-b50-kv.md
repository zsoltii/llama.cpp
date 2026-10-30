# Plan: Intel Arc Pro B50 — profiling, then KV-type speedups (iq4_nl, q4_0, q4_1, q5_0, q5_1, q8_0)

Status: **plan** (2026-10-29 evening). Target: B50 on Vulkan (Vulkan2, eGPU). The laptop's T1200 is not a target.
Model and command: the user's command + MTP n=1 (`-ctkd/-ctvd` = the same type as `-ctk/-ctv`), defer + `-bs`.

## 0. Where we start

- Merge of upstream (79 commits) is done and tested on the B50:
  - test-backend-ops 21632/21632;
  - speed equals pre-merge (MTP story 29.5 / wiki 24.9 tg, wiki pp 612);
  - KLD 0.000443, wiki QA 25/25 + 5/5.
- Vulkan already supports every requested KV type in FA (`supports_op`: f16, bf16, q8_0, q5_1, q5_0, q4_1, q4_0, iq4_nl).
  So the work is speed, not missing support. Any type that turns out to fall back somewhere (e.g. a CPU op in the graph) gets implemented.
- Known facts from earlier B50 work (b50-results.md, "KV types — FA kernel" table, measured before findings 5 and 24):
  - Decode FA (nb=1): q4_1 fastest (538 us), q8_0 978, q5_0 946, iq4_nl 767 (LUT, no int-dot path).
  - Prefill FA: q5_0 and iq4_nl are the slowest (~95 ms vs 65 ms for f16 at 512 × 8.7k).
- **Lead found while reading the code:**
  - The MTP verify step (2 tokens) runs FA on the **coopmat1** path on the B50 (`get_fa_tuning_params`: coopmat1 for n_rows > 1).
  - The token merge of finding 24 (tokens × GQA heads in one workgroup, K/V read once) is **disabled on coopmat1**, because it was never measured there.
  - On the 2080 Ti the same change gave −40…−50% FA time at 2 tokens.

## 1. Profiling (current code, q8_0 KV)

1. `vkprof.sh`: server with `GGML_VK_PERF_LOGGER=1` (every graph) + `LLAMA_SERVER_TIMING=1`.
   - story (short ctx, 256 tokens);
   - wiki (20k prompt, 128 tokens).
2. `vkprof.py` groups the graphs into prefill / verify / draft and prints the per-op GPU time per graph.
3. Output: where the ~38 ms verify and ~8 ms draft step go on the B50 (mat-vec vs FA vs small ops), and how much prefill FA costs at 20k.
4. Decide the order of the optimizations below from the measured shares.

## 2. KV-type baseline on the B50

For each type t in {q8_0, q4_0, q4_1, q5_0, q5_1, iq4_nl} (+ f16 as reference):
- **Kernel level:** `test-backend-ops perf -o FLASH_ATTN_EXT` with the model's shapes (head 256, GQA 4):
  - decode nb=1/2/3 at kv≈20k;
  - prefill 512 × 8.7k and 512 × 20k.
- **Server level:** `bench.py` story + wiki, MTP n=1, greedy (comparable to the earlier tables).
- Check the graph for CPU fallbacks per type (`GGML_SCHED_DEBUG=1`: any split onto CPU = missing implementation).

## 3. Optimization candidates (in expected-value order; the profile may reorder them)

1. **Token merge on coopmat1 for the verify (finding 24 extended).**
   - Option a: for small N (≤ 8 rows after merging), use the scalar shader with token merge instead of coopmat1. It is cheap to test: the env switch exists.
   - Option b: implement the token × head row layout in `flash_attn_cm1.comp` (the same addressing as the scalar shader).
   - Vendor-neutral; measured on the B50; must not change NVIDIA/AMD behaviour without a measurement.
2. **iq4_nl dequant in the FA shaders.**
   - The LUT lookup per element is the bottleneck. Ideas:
     - keep the 16-entry table in registers/shared memory;
     - an int8 dot path (dequant the nibble to the int8 table value, then DP4A / int8 coopmat like q4_0).
   - Applies to the scalar decode path and the coopmat1 prefill path.
3. **q5_0 / q5_1: high-bit extraction.**
   - The 5th bit (qh) is gathered per element. Vectorize it: one 32-bit qh load per block, building 4 values at a time.
   - Same for the int-dot K path.
4. **Prefill FA K/V dequant to shared memory per type.**
   - Check that the coopmat1 prefill uses the fastest dequant per type (the earlier table shows q5_0/iq4_nl +40–50% over f16).
5. **KV write cost (SET_ROWS / CPY into the quantized cache) per type.**
   - Visible in the profile; quantizing iq4_nl is the most expensive (best-index search).
6. Anything else the profile shows above ~5% of the step.

Rules:
- Every change is vendor-neutral or gated by device capabilities. NVIDIA/AMD code paths stay identical unless measured there too.
- Laptop limits: one heavy job at a time, builds with `-j4`.

## 4. Tests after each accepted change (and once at the end)

1. `test-backend-ops -b Vulkan0` (B50 only): FLASH_ATTN_EXT after every shader change, the full suite at the end.
2. Benchmarks: `bench.py` story + wiki for every KV type, MTP n=1, greedy and loop-free sampling (`BENCH_SAMPLING=server`).
3. Accuracy:
   - KLD per KV type (`eval/kld.sh run NAME BIN TYPE`), compared with the earlier table (q8_0 0.00044, q4_0 0.0051, q4_1 0.0051, q5_1 0.0033, iq4_nl 0.0058);
   - wiki QA (hallucination): 25/25 + 5/5 expected.
4. Loop test: `loop-run.sh vk-full vk-qwen` (p1, 6 seeds). Expected: about upstream level with default sampling, 0/6 with the Qwen sampling.
5. Commit after each accepted step (local only; the user pushes). Docs in English in `b50-results.md`.

## 5. Deliverables

- `b50-results.md`: profile breakdown, KV-type table (before/after), each change as a numbered finding with kernel and server numbers.
- A short Hungarian summary for the user at the end.
