# Plan: next speedup candidates (CUDA, Vulkan, ROCm, OpenVINO, mmproj)

Written 2026-11-13, after the RTX 3090 round (`rtx3090-results.md`). It is a work order for another agent: every
item says what is known, what is only a hypothesis, how to measure it, and when to stop. Nothing in this file has been
measured unless it cites a number from one of the result files.

Target workload: Qwen3.8-27B UD-IQ4_XS, iq4_nl K/V, 100k-200k context, agentic use (opencode), long answers.
The user's current command is in `rtx3090-results.md` ("Server"), with `-c 200000` and the Qwen sampling flags.

## 1. Why the numbers look low, and how much headroom exists

The user compared with Qwen3-Next on the same RTX 3090: 2500-3000 t/s prompt processing, 30 t/s generation on Vulkan.
`src/models/qwen3next.cpp` is the 80B-A3B model: a MoE with about 3B active parameters per token. Qwen3.8-27B is
dense, so every token uses all 27B parameters, about 9 times more work per token. The two are not comparable
directly. Which GGUF and which offload split the user ran was not checked.

Rough upper bounds for the dense model on the 3090 (estimates, to be confirmed in step C0):

| quantity | bound | now (CUDA) | now (Vulkan) |
|---|---|---|---|
| tg, no MTP, depth 0 | 936 GB/s / 14.35 GB weights = ~65 t/s | 45.9 (70%) | 37.5 (58%) |
| pp, depth 0 | ~5.2e10 flops per token; nominal fp16 tensor peak ~140 TFLOPS = ~2700 t/s | 1692 (~88 TFLOPS) | 1265 (~66 TFLOPS) |
| pp, depth 64k | adds 16 FA calls per ubatch over the whole cache | 903 | 683 |

Reading:
- CUDA prefill at depth 0 already runs at well over half of the nominal peak. A realistic gain there is 10-20%.
- CUDA generation is at 70% of the bandwidth bound. The rest is small ops, launch overhead and FA.
- Vulkan is 18-25% behind CUDA everywhere, also at depth 0. So its gap is in the mat-vec, the mat-mat and the small
  ops, and only partly in FA.
- The largest remaining factors are not kernels: how often the server re-processes the prompt (S1), and MTP.

## 2. Rules for the executing agent

- No sudo. No push, no force push. Commit locally with
  `git -c user.name=zsoltii -c user.email=fater.zsolt@gmail.com commit`.
- Never commit the OpenVINO work in progress: `.devops/bge-m3/README.md`, `.devops/bge-m3/compose.yaml`,
  `.devops/bge-m3/openvino.Dockerfile`, `ggml/src/ggml-openvino/openvino/op/pad.cpp`,
  `docs-dev/bge-m3/bge-backend-cmp.py`. Ask the user before that changes.
- Before any benchmark run `nvidia-smi` and `pgrep -a llama`. The user often runs their own `llama-server` on the
  3090: do not measure, and do not stop it; ask.
- Builds: `nproc` first, then `nice -n 19 cmake --build <dir> -j$(nproc)`. Heavy jobs run one at a time.
- CUDA builds need `source ~/anaconda3/bin/activate llama-cuda` and `-G Ninja`. `build-cuda-dev` (sm_86, 4 FA quant
  pairs) is the fast iteration build.
- Vulkan changes must be gated on capabilities or architecture, so Intel, AMD and older NVIDIA cards keep their
  current path. Run `test-backend-ops` on the T1200 (`Vulkan0`) too.
- Results go to `$HOME/llama-bench-results`, not `/tmp` (tmpfs, lost on reboot or an eGPU drop).
- Docs, notes and commit messages in English. The final summary to the user in Hungarian.
- Ignore `AGENTS.md`.

Devices on the laptop (i7-11850H, 8C/16T, 31 GB RAM, Thunderbolt eGPU):
- RTX 3090: `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1` -> `CUDA0`; Vulkan `-dev Vulkan1` or
  `GGML_VK_VISIBLE_DEVICES=1`.
- T1200: index 0 on both. Not a tuning target.
- RX 6900 XT (when attached instead of the 3090): `ROCm0`, RADV. See `rx6900xt-results.md`.
- Upstream baseline builds: `../llama.cpp-upstream/build-cuda`, `build-vk`, `build-rocm`.

## 3. Measurement protocol

- **Model level:** `llama-bench -p 2048 -n 128 -ub 2048 -d 0,16384,65536 -ctk iq4_nl -ctv iq4_nl -fa 1 -r 3`, and
  the 115k-token server run of `rtx3090-results.md` (prompt parts are in `testdata/` and `bench/corpus/`).
- **Op level:** `test-backend-ops perf -b <dev> -o <OP> -p <regex>`. The Qwen3.8 FA shapes are
  `hs=256,nh=4,nr23=[6,1],kv=33280` with `nb=1`, `nb=2`, `nb=512`.
- **Accept a change** only if the model-level number moves by 3% or more in two interleaved A/B runs. Record dead
  ends in the result file with their numbers.
- **Traps that already cost time:**
  - `test-backend-ops perf` does not check results. A tile config that computes garbage looks fast. Run the eval
    mode on the same shape before believing a perf number.
  - Its FA mask is random. A model mask is all zero except the causal triangle, and that changes the shader path
    (`rx6900xt-results.md`, fourth round, finding 1).
  - A single weight matrix can sit in a cache (128 MB Infinity Cache on RDNA2). Use the `bs=[8,1]` mat-vec cases.
  - `GGML_VK_PERF_LOGGER` adds about 10 ms per step of wall time. Use its per-op times, not the wall time.
  - Thermal and clock drift is 10-15% on these cards. Interleave A and B.
- **Validation after every accepted change** (`tests/README.md`, `bench/eval/server_eval.sh`): wikiqa for Qwen3.5
  MTP, Qwen3.8 iq4_nl and Qwen3.8 MTP (expect 25/25 and 5/5), the story loop test (`vk-full`, `vk-qwen`), the attach
  test (expect 0/6), and the full `test-backend-ops` on both backends. Baselines are in `rtx3090-results.md`
  ("Validation").

## 4. Work packages

Order of work: S1, S2, then C0 and V0 (profiles), then the packages the profiles point at. R and O need other
hardware states and can wait.

### S. Server and configuration (all backends, no kernel work)

**S1. Prompt re-processing in agentic use. Highest expected value, not measured so far.**
- Qwen3.5 / 3.8 are hybrid models: 48 of 64 layers keep a recurrent state. The KV cache cannot be cut at an
  arbitrary position, so a prompt that differs from the cached one is restored from a context checkpoint, or
  processed again from token 0.
- Defaults (`common/common.h`): 32 checkpoints per slot, at least 8192 tokens apart, 8192 MiB of `--cache-ram`.
- Log lines to look for (`tools/server/server-context.cpp`): "created context checkpoint", "restored context
  checkpoint", "forcing full prompt re-processing due to lack of cache data".
- Measure: run a real opencode session of 10+ turns against the server with `-lv 1` or higher. For each turn record
  the prompt size, the tokens actually evaluated, and which of the three lines appeared.
- If turns re-process tens of thousands of tokens: sweep `--checkpoint-min-step` (2048, 4096), `--ctx-checkpoints`
  (64, 128) and `--cache-ram`, and record the checkpoint size and the time to create one (it was 106 ms -> 39 ms on
  the B50 after a pool fix, `b50-results.md` finding 8).
- Also look at what opencode changes in the prefix between turns (the user's plugin is in
  `~/.config/opencode/plugins/llamacpp/`; do not touch `auth.json` or `account.json`). A stable prefix is worth
  more than any kernel.
- Gate: tokens evaluated per turn. One avoided 100k re-process saves about 2 minutes.

**S2. Backend sampling and threads on the 3090.**
- `-bs --spec-draft-backend-sampling` was +9-13% on the 2080 Ti and essential on the B50 (no logit readback over
  Thunderbolt). The 3090 runs did not use it.
- The presence penalty runs on the backend. DRY and grammars fall back to CPU sampling.
- Also re-check `-t 1 -tb 1 --poll 0` against the defaults: it won on the 2080 Ti with Qwen3.5.
- Measure tg on the 115k server run and at short context, both backends, with the Qwen sampling flags.

**S3. MTP settings at depth.**
- `--spec-draft-n-max 1` won on the RX 6900 XT. It was not swept on the 3090 after the 2-token verify got faster.
- Sweep n-max 1 / 2, `--spec-draft-p-min` 0 / 0.5 / 0.75, and `LLAMA_MTP_DRAFT_VOCAB` on / off (`-1`) on CUDA: the
  reduced draft head was never measured on CUDA.
- Record tg and the acceptance rate at 20k and 115k.

**S4. MTP plus an n-gram speculator for copy-heavy output (low priority).**
- `--spec-type` takes a list. `draft-mtp` with `ngram-mod` hurt on stories (`results.md`, round 3).
- Code edits repeat long spans of the prompt, so test once with a "rewrite this file with one change" prompt.
- Drop it if tg does not improve by 10%.

### C. CUDA (RTX 3090)

**C0. Profile first.**
- `LLAMA_GRAPH_TIMING=1` gives per-graph times. ncu is in the conda env `ncu` (`~/anaconda3/envs/ncu/bin/ncu`) and
  works for non-root since `RmProfilingAdminOnly` is 0.
- Produce a table for one decode step at depth 0 and 64k, and one 2048-token ubatch at depth 0 and 64k: time in
  mat-vec / mat-mat by weight type, FA, GDN ops, small ops, and the gap to the wall time.
- Without this table, do not start C1-C4.

**C1. MMA FA prefill at depth.**
- Fact: with quantized K/V the MMA kernel converts the cache to f16 on every call, in 64 MiB chunks, and merges the
  chunk results (`fattn-common.cuh`, `GGML_CUDA_FA_KV_CHUNK_MB`; with chunks, `parallel_blocks >= 2`).
- Hypotheses, in the order to test:
  1. The chunk size is wrong for 100k+ contexts. Sweep `GGML_CUDA_FA_KV_CHUNK_MB` 0 / 64 / 256 / 1024 at 64k and
     115k. This is an env var, no build.
  2. The softmax part (scale, mask, max, exp per KQ element) is a large share next to the tensor-core part. ncu on
     `nb=512, kv=33280` shows it.
  3. The conversion itself. Estimate: 16 layers x ~470 MB f16 per ubatch at 115k, which should be tens of ms out of
     ~2 s. Time `to_fp16` to confirm, then drop the idea if it is under 3%.
- Only if 2 holds: look at the tile shape for head size 256 and at the exp path.
- Gate: pp at 64k (903 t/s now, upstream 916).

**C2. Mat-vec bandwidth per weight type on Ampere.**
- The Ampere defaults were never checked: the MMVQ rows-per-block choice (finding 23 in `b50-results.md`) was tuned
  on Turing.
- Measure every type in the GGUF (`gguf-dump` the tensor types) with the `bs=[8,1]` cases, for 1 and 2 columns
  (plain decode and MTP verify), in GB/s. Anything under ~800 GB/s is a candidate.
- Then tune per type: rows per block, warps, the IQ lookup tables.
- Gate: tg at depth 0.

**C3. Small ops and graph overhead.**
- Count the nodes of a decode step and their time (C0). Earlier cards showed ~700-1300 small ops at ~5 us each.
- Check that CUDA graphs stay captured when the batch alternates between the MTP draft and the 2-token verify, and
  that graph reuse is on (`LLAMA_GRAPH_REUSE_DISABLE=1` as the A/B).
- A linear-attention layer runs four mat-vecs on the same input (`wqkv`, `wqkv_gate`, `ssm_beta`, `ssm_alpha`,
  `src/models/qwen35.cpp`). Check whether they are already fused; the FFN gate/up/GLU fusion exists. If not, fuse
  them the same way.
- Gate: tg at depth 0, 3%.

**C4. iq4_nl decode FA against q4_0.** 144 us against 120 us at gqa 6, kv 33280. The KQ dot still does one table
lookup per int. Only worth it if C0 shows FA above 10% of a decode step at the user's depth.

**C5. Long shot: int8 tensor-core KQ.** The vec kernel already quantizes Q to q8_1. Ampere has int8 MMA with k = 32,
which is exactly one q8_0 / iq4_nl block. A prefill KQ in int8 would skip the f16 conversion and halve the K reads.
The per-block scales need a float step per block, which may eat the gain. Write a micro-benchmark first; stop if it
is not 1.3x faster than the f16 MMA KQ part.

### V. Vulkan (RTX 3090; every change vendor-safe)

**V0. Profile first.** `GGML_VK_PERF_LOGGER=1` for decode and prefill at depth 0 and 64k. Put the per-op times next
to the CUDA table of C0. The gap list decides the order of V1-V5. `GGML_VK_FA_DEBUG=1` prints the FA path and tile.

**V1. Mat-vec.** On the 2080 Ti the 2-column mat-vec stopped at ~400 GB/s against ~510 on CUDA; the note says it
needs a shader redesign. Measure 1 and 2 columns per type on the 3090 first. Known switches for A/B:
`GGML_VK_DISABLE_MMVQ`, `GGML_VK_FORCE_MMVQ`, `GGML_VK_RM_KQ`, `GGML_VK_KQ_BATCH_RM`.

**V2. Prefill mat-mat.** Find which pipeline each weight type takes on the 3090 (coopmat2 with f16 dequant, or the
integer-dot MMQ tiles) and its TFLOPS at `17408 x 512 x 5120`, against CUDA MMQ. The large integer tiles were
enabled for RDNA2 only (`rx6900xt-results.md`, third round, finding 4); try them on NVIDIA. Any tile sweep must drop
invalid configs (see the traps above) and must pass the eval mode.

**V3. FA prefill at depth.** Vulkan is 24% behind CUDA at 64k, the same ratio as at depth 0, so FA is not worse than
the rest. Check with `GGML_VK_FA_DEBUG` that iq4_nl uses the f16 scratch on NVIDIA and what the coopmat2 tile is.
Low priority unless V0 says otherwise.

**V4. 2-token verify at gqa 6.** It still uses the old path; the 12-row merge was slower (237 -> 313 us). Try two
6-row tiles in one dispatch that share the K/V reads.

**V5. Dispatch overhead.** About 5-9 us fixed cost per dependent dispatch was measured on RDNA2, and ~1300 small ops
per Qwen3.8 decode step. Options: fuse more of the GDN chain, and sweep `GGML_VK_MAX_NODES_PER_SUBMIT` and
`GGML_VK_FLOPS_PER_SUBMIT_DIV`. NVIDIA submissions are serialized because of a driver bug (`ggml-vulkan.cpp`,
"Avoid concurrent submissions on NVIDIA"); leave that.

### R. ROCm / HIP (RX 6900 XT, when it is attached)

1. A/B the five items of the memory note "HIP follow-ups from 3090": iq4_nl KQ with one lookup per int, 3 heads per
   block for gqa 6, Q ds per q8_1 block (already active), partial unroll in GQA mode, iq4_nl V dequant lookup.
2. KV-chunked f16 conversion for the tile kernel. The pool is outside `--fit`, so MTP at 116k+ runs out of memory;
   this would allow 128k with MTP.
3. q4_0 GQA decode is 906 us against 266 us for iq4_nl at gqa 6: find out why.
4. Head size 256 prefill stays at ~17 TFLOPS against 21 on Vulkan.
5. S2 and S3 on this card too.

### O. OpenVINO

Scope: the backend runs on the laptop CPU and the Tiger Lake iGPU. The B50 needs a desktop with Resizable BAR. So
far only bge-m3 accuracy was measured on OpenVINO, no throughput.

1. **Baseline:** bge-m3 queries/s and index texts/s (the 36-process client of `docs-dev/bge-m3/README.md`) for
   OpenVINO CPU, OpenVINO iGPU, the llama.cpp CPU backend, and Vulkan on the iGPU.
2. **Runtime properties.** The backend sets only the cache properties at compile time; no performance hint, stream
   count or inference precision (`ggml-openvino-extra.cpp`). Try `PERFORMANCE_HINT` LATENCY / THROUGHPUT,
   `NUM_STREAMS`, and f16 inference precision on the GPU. Add an env switch for each, default unchanged.
3. **Flash attention on OpenVINO** gives wrong bge-m3 output (FA falls back to the CPU on a permuted Q view). Fix it
   or keep FA off; measure whether FA would help at all.
4. **Start-up:** `GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR` for the bge-m3 image.
5. **Cross-sequence attention** (open item 3 of the bge-m3 README): the encoder mask covers every sequence of a
   ubatch, so the work grows quadratically. Per-sequence attention helps every backend.
6. LLM generation on OpenVINO (CPU or iGPU) against the CPU backend, with Qwen3.5-9B. Low priority: neither device
   is a target for the 27B model.

### M. mmproj (vision encoder)

It runs only for requests with images or video. Never measured in this project.

1. **Baseline:** encode time for a small, a 1024 px and a 4k image and a short video, CUDA and Vulkan (the server
   logs the encode time), plus the extra VRAM. The model's file is `mmproj-BF16.gguf` (931 MB).
2. **Weight type:** BF16 against an F16 or Q8_0 mmproj. Check how each backend handles BF16 first.
3. **Flash attention in the encoder:** it is per model (`flash_attn_auto`, `tools/mtmd/mtmd-subcontext.cpp`).
   Check that it is on for this model on both backends.
4. **Token count:** `--image-min-tokens` / `--image-max-tokens`. This changes quality, so it needs an image QA
   check, not only speed.
5. **Batching:** the generic graph accepts at most 2 images per batch (`mtmd-graph.cpp`). For video, measure whether
   frames are encoded one by one.
6. Kernel work only if one op dominates the profile.

## 5. Already tried: do not repeat

- CUDA GQA vec kernel: 6 heads per block, 4 heads for 2 tokens, `__launch_bounds__` with 3 or 4 blocks per SM,
  V unroll 1 / 2 / 16, 8 or 32 threads per K row (`rtx3090-results.md`).
- Vulkan: 12-row merge for the gqa 6 verify; 16-row tiles for decode.
- ROCm: 3 heads per block, 4 threads per K row for iq4_nl, more blocks per SM, other MMQ tile shapes, J 128 for the
  IQ types (`rx6900xt-results.md`).
- A 160-byte aligned q4_K format, CPU FA split-KV, int8 (Q8_0) weights in the CTranslate2 style, custom quant mixes
  (`results.md`, rounds 21, 23, 36, 37).
- `GGML_CUDA_GRAPH_OPT=1`, CPU pinning and NUMA, MMQ routing for the verify batch (`results.md`, rounds 8, 12, 18).
- MTP n-max 2 and 3 on the RX 6900 XT.

## 6. Deliverables

- One result file per round next to this plan (or a new section in `rtx3090-results.md`), with the profile tables
  of C0 and V0, every experiment with its numbers, and the dead ends.
- Updated recommended server commands if S1-S3 change them.
- The validation table after the last accepted change.
