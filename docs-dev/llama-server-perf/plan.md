# llama-server speedup: Vulkan + CUDA (RTX 2080 Ti, Qwen3.5-9B-MTP)

Goal: speed up `llama-server` for a model running on one GPU, without the model spilling out of GPU memory.
Both backends (Vulkan and CUDA) must improve both prompt processing (pp) and token generation (tg).

## Environment (fixed, because every measurement builds on it)

- GPU: NVIDIA GeForce RTX 2080 Ti, 11 GB VRAM (Turing, sm_75, has tensor cores).
- Driver 595.99.02 (CUDA driver 13.2). `nvcc` was NOT installed (added via apt).
- OS: Ubuntu 26.04.1 LTS (QEMU/KVM guest, VFIO GPU passthrough, host: Dell R720).
  - CPU and RAM: 2× Xeon E5-2670 = 24 logical CPUs (2 NUMA nodes), 15 GB RAM.
  - Disk: 20 GB free at the start, ~142 GB since 2026-10-26 (which is why Part 2 could start).
- Model (fixed, may not be changed). The starting command is below.
  - The KV type has been `f16` since rounds 19/27.
  - The current daily command is in results.md round 35:
  `llama-server -hf unsloth/Qwen3.5-9B-MTP-GGUF:UD-Q4_K_XL -fa on -ctk iq4_nl -ctv iq4_nl --parallel 1 -c 80000`
- Model architecture (Qwen/Qwen3.5-9B config + `src/models/qwen35.cpp`):
  - 32 layers: 24 gated delta net (linear attention) + 8 full attention (16 Q / 4 KV heads, head_dim 256);
  - the MTP head is baked into the GGUF (`nextn`).
- Starting point: brew llama.cpp 0.6.0 (build 11429, Vulkan only), using 7.3 GB VRAM. About 35 tok/s tg with a long prompt.
- VRAM budget (80k ctx, iq4_nl KV): ~5.8 GB weights + ~0.9 GB KV + ~0.5 GB compute + ~0.9 GB mmproj = ~7.3 GB.
  Goal: always keep at least ~800 MB of VRAM free.
- Repo: fresh HEAD (f3363593f). It contains the built-in Qwen3.5 MTP speculative decoding support (`draft-mtp`), which brew 0.6.0 also has (checked: `git show d81235049`).

## Phases

### F0 - installation (the user runs it with sudo) - DONE

- `cmake glslc libvulkan-dev spirv-headers gcc-13 g++-13 nvidia-cuda-toolkit`.
- For the CUDA build: `CUDAHOSTCXX=/usr/bin/g++-13` (CUDA 12.4 + gcc-13).
- During measurements Firefox and other GPU apps are OFF (VRAM noise); check the disk.

### FA - reference survey (standard protocol, identical for every binary)

- Fixed corpus (generated once, must NOT be modified):
  - `greeting.txt` (R1: a greeting, `n_predict:32`)
  - `moe-hu.txt` (R2: an MoE explanation in Hungarian, `n_predict:512`)
  - `long-60k.txt` (R3: 65 578 tokens + a question, `n_predict:256`; refined with `/tokenize`; 65k/80k = ~82% context fill)
- Protocol per config:
  - server restart, warm-up;
  - 1× `cache_prompt:false` (pp+tg) + 2× `cache_prompt:true` (long-context tg);
  - median of 3 runs;
  - sampling: temp=0 (deterministic).
- Metrics: TTFT, pp tok/s, tg tok/s, peak VRAM (1 s `nvidia-smi` polling), OOM check.
- After starting, a **run check is mandatory**:
  - the process, the server log (`server is listening`) and real GPU load (util/VRAM);
  - at the end, that the expected measurements exist in `summary.csv`.
  The mere presence of the process is not proof: the bench also waits for a dead server.
- Results: `~/llama-bench-results/` (`bench.py`, `results/*.json`, `summary.csv`).

### FB - brew measurements (Vulkan, 0.6.0) - DONE

- B-baseline: exactly the user's command.
- B-opt grid: `-b`/`-ub`, `--no-mmproj-offload`, `--cache-reuse`, `--spec-type draft-mtp` + `n-max`.
- Result: no real ub gain above the ±5–15% run-to-run noise (confirmed by a drift study).
  B-opt = baseline + `--no-mmproj-offload`. MTP with brew dropped.

### FC - Vulkan build from the repo HEAD (`build-vulkan/`) - DONE, evaluated

- 0.7.0-dev, build 11697. Real numbers: pp ~930, tg ~30.5.
- HEAD Vulkan ≈ brew (within the noise band). MTP slows HEAD down on Vulkan too.
- `FA_SPARSE_DISABLE` has no effect. `--no-mmproj-offload` saves 1.1 GB for free.
- Methodology lesson: the FIRST run of a new binary pays for Vulkan pipeline compilation (the driver cache only works from the 2nd process on), so first runs are not evaluated.

### FD - CUDA build from the repo HEAD (`build-cuda/`) - DONE

- `GGML_CUDA=ON`, `CMAKE_CUDA_ARCHITECTURES=75`, `CUDAHOSTCXX=g++-13`.
- Result: the HEAD CUDA binary became the daily winner (with MTP and `-bs`); see results.md rounds 6–7 and 9.
- The list of env-gated experimental switches: the "Phase 2b" table below.

### FE - MTP on the best Vulkan and CUDA config

- Condition: tg ≥ +15% and pp does not get worse, VRAM fits. Otherwise dropped.
- Sweep `n-max` 1–6 (`2` is the starting point per Unsloth).
  `--spec-draft-p-min` stays 0 on the GPU; `--parallel 1` is mandatory.
- Chained variant: `--spec-type draft-mtp,ngram-mod`.
- Record attempted/accepted/mean length from the server stats plus peak VRAM.

### FF - documentation and final configs

- `~/llama-bench-results/` + `docs-dev/llama-server-perf/results.md`: tables per binary and config.
- 2 final launch commands (Vulkan + CUDA) + an optional launch script.
- Requirements for the winners: R1/R2/R3 without errors at 80k, at least ~800 MB of free VRAM.

## Block list (proven harmful by measurement)

- `GGML_VK_DISABLE_FUSION` (and any fusion switch-off): −5%..−18%.
- The `p-min` gate on a fast GPU: −17%..−21% (measured).
- The `--mtp` download flag: there is no sidecar, the head is baked in. Correct: `--spec-type draft-mtp`.
- Writing our own GDN/CUDA kernel: the tensor-core path exists, not worth it.

## Risks

- CUDA 12.4 + Ubuntu 26.04 glibc friction (OK so far: configure passed).
- R3 runs are slow: a full measurement has taken ~1.5–2 hours so far.
- Run-to-run noise is ±5–15%: only announce repeated differences above that.

## Phase 2 — optimization rounds (from 2026-10-17)

Regime:
- fixed 170 W power cap (systemd: `nvidia-powerlimit.service`);
- every measurement **with q4_1 KV** (both CUDA and Vulkan): no iq4_nl and no f16 in the measurements.
  The advantage of f16 stays documented under the earlier regime.

Target areas (from the initial survey):
- **CUDA GPU:**
  - the MMVQ↔MMQ decode decision (no tuned threshold for Turing; an env-gated patch is ready);
  - MMVQ nwarps/config A/B, FA config, GDN kernel.
- **Vulkan GPU:**
  - env switch A/B: FORCE/DISABLE_MMVQ, MAX_NODES_PER_SUBMIT, DISABLE_ASYNC, DESCRIPTOR_REUSE, MEMORY_PRIORITY, GRAPH_OPTIMIZE;
  - retest MTP/`-bs` on q4_1.
- **CPU:**
  - affinity (2× Xeon E5-2670, 24 logical CPUs, 2 NUMA nodes; the GPU has `numa_node=-1`);
  - `taskset`/thread experiments, server-side hot path.
- **GPU↔CPU:**
  - check the PCIe link under load (Gen1 x8 idle, Gen3 x16 max);
  - sync/copy paths, graph coverage, pinned memory.

Methodology:
- 60 s cooldown, winner-anchor runs, power guard;
- one variable at a time;
- update the docs after every round;
- a run check is mandatory after starting (see the FA protocol).

Rounds:
- r9 (DONE): 170 W reference with uniform q4_1 KV.
  Configs: CUDA win ×3, nomtp, nobs, VK base, VK MTP, brew anchor (results.md round 9).
- r10 (DONE): CUDA MMVQ↔MMQ A/B.
  - MMQ is worse on every metric: R1 −15..17%, R2 −9..17%, R3 tg −10..12%.
  - MMVQ stays; the patch was reverted (results.md round 10).
- r10b (DONE): Turing MMVQ nwarps variant — no gain, dropped (results.md round 10b).
- r11 (DONE): Vulkan (results.md round 11).
  - MTP alone makes it worse, FORCE_MMVQ makes it worse, DISABLE_MMVQ/nodes neutral → the VK optimum is the base.
  - PCIe runs at Gen3 x8 under load → not a bottleneck.
- r12 (DONE): CPU affinity / NUMA pinning — neutral (results.md round 12). Open: installing the systemd 170 W service (sudo).
- r13 (DONE, won — results.md round 13): A/B of the FA q4_1-q4_1 VEC kernels.
  - ref = the archived clean CUDA binary; var = the new build (q4_1 FA compiled in + env-gated experimental code).
  - Expected: the f16 conversion of batch-1 decode FA disappears → MTP tg +3..8%, non-MTP decode +10..20%.
- r14 (DONE — results.md round 14): CUDA env A/Bs.
  `GGML_CUDA_MMQ_PREC=q4`, FA kernel forcing, a context test of the restored switches, a `-t/-tb` sweep.
- r15 (DONE — conversion A/B, results.md round 15; CPU profile: "CPU profile" section): CPU profiling (perf, GGML_SCHED_DEBUG) + parallelization.
- r16 (DONE — results.md round 16): Vulkan profiling (`GGML_VK_PERF_LOGGER`) + env sweep + `rm_kq`.

## Phase 2b — restoring the experimental code + new experiments (from 2026-10-19)

The restored experimental switches are **env-gated, default off**, so the daily command stays fast; a measurement only sets the env vars it needs.

| switch | location | effect | default |
|---|---|---|---|
| `GGML_CUDA_DISABLE_MMVQ=1` | `ggml/src/ggml-cuda/mmvq.cu` | every quantized mul_mat goes through MMQ | off (MMVQ) |
| `GGML_CUDA_BLOCKING_SYNC=1` | `ggml/src/ggml-cuda/ggml-cuda.cu` | `cudaDeviceScheduleBlockingSync` instead of spinning | off (spin) |
| `GGML_CUDA_FA_FORCE_KERNEL=vec\|tile\|mma[:N]` | `ggml/src/ggml-cuda/fattn.cu` | forces an FA kernel family (optional `:N` = only for `Q->ne[1] <= N`) | off (auto) |
| `GGML_CUDA_MMQ_MIN_BATCH=N` | `ggml/src/ggml-cuda/mmvq.cu` | batches with `ne11 >= N` go to MMQ (tensor cores), smaller ones to MMVQ; for a targeted look at the MTP verify (batch 3) | off (MMVQ up to 8) |
| `GGML_VK_RM_KQ=1..4` | `ggml/src/ggml-vulkan/ggml-vulkan.cpp` | overrides rows-per-shader of the k-quant MMVQ shader | off (2) |
| `-DGGML_CUDA_MMVQ_TURING_KQ_NWARPS4` | `mmvq.cu` (compile variant) | Turing k-quant `ncols_dst=1`: 2→4 warps | off (2 warps) |
| `GGML_CUDA_FA_QUANTS` += `q4_1-q4_1` | build configuration | native q4_1 VEC FA kernel (no f16 conversion) | **ON** (confirmed by r13: non-MTP R3 +49%) |
| `GGML_CUDA_CONV_LEGACY=1` | `ggml/src/ggml-cuda/convert.cu` | forces the *old* q4_1→f16 conversion; the vectorized v2 is the default | **ON** (r14/r15: MTP R3 +8…+11%) |
| `GGML_CPU_FA_SPLIT_QUANT=1` | `ggml/src/ggml-cpu/ops.cpp` | CPU FA split-KV for quantized KV | off (round 23: −70%) |
| `GGML_VK_BLOCKING_SYNC=1` | `ggml/src/ggml-vulkan/ggml-vulkan.cpp` | fence wait in the kernel (instead of spinning) | off |
| `GGML_VK_FLOPS_PER_SUBMIT_DIV=N` | `ggml/src/ggml-vulkan/ggml-vulkan.cpp` | how the graph is split into submits | 40 (round 34: no effect) |
| `GGML_VK_PIPELINE_CACHE=<file>` | `ggml/src/ggml-vulkan/ggml-vulkan.cpp` | persistent pipeline cache | off (round 28) |
| `LLAMA_GRAPH_TIMING=1` | `src/llama-context.cpp` | graph timing per context, by `(n_tokens, n_outputs)`, every 64 graphs + the phases of `decode()` | off (alias-free since round 38) |
| `LLAMA_SERVER_TIMING=1` | `tools/server/server-context.cpp` | per-step server breakdown (draft, verify, catch-up, sampling, yield), every 64 steps | off (round 39) |
| `GGML_CUDA_MMVQ_TURING_RPB_SMALL` | `ggml/src/ggml-cuda/mmvq.cu` (compile time) | Turing MMVQ rows_per_block for ncols_dst 2–4 | **4 (ON)** — round 40: verify −12%, tg +9.3%; `-D...=2` is the old one |
| `LLAMA_MTP_DEFER_CATCHUP=1` | `common/speculative.cpp` | merges the MTP catch-up into the first draft step | off (round 40: ~1%, not worth it) |
| `GGML_VK_KQ_BATCH_RM=N` | `ggml/src/ggml-vulkan/ggml-vulkan.cpp` | q4_K/q5_K/q6_K float mat-vec for n≥2 with N rows per workgroup (q6_K ≤4); q4_K/q5_K with n>1 use the float path instead of MMVQ | off — **round 39: N=4 → MTP verify −14%, t/s +10%; confirmed at 45k: step −7.6%, tg +9.2%** → set via env in the daily command |

**Status (2026-10-19 morning):**
- r13–r15 confirmed the two code changes that stay in: native q4_1 FA + vectorized conversion.
- r14 rejected `vec:4` and `-ub 1024`; `MMQ_PREC=q4` and `mma:1` are neutral.
- Open: the r16 Vulkan env round and the evaluation of the CPU profile.


**Mechanism (r13):**
- On Turing, FA with quantized KV uses VEC at batch=1 and MMA (f16 only) at batch≥2.
- So the MMA path converts the whole KV to f16 *on every step*: ~3 GB per token at 60k context over the 8 attention layers.
- The native q4_1 VEC kernel removes this for batch-1 decode: non-MTP R3 tg 27.7 → 41.3 t/s (+49%).
- The MTP verify has batch=3 → MMA → the conversion stays. That is why `vec:4` (r14) and `CONV_FAST` (r15) are the candidates.

**Measurement trap:** the RUNPATH of `bin-cuda-ref/llama-server` points to the live `build-cuda/bin`.
So every run needs `LD_LIBRARY_PATH=$(dirname $bin)`; otherwise the "ref" loads the new libs too.


Block list additions:
- iq4_nl: not supported in CUDA FA, plus a user decision.
  *(Since 2026-10-29 the CUDA FA supports iq4_nl; see b50-results.md finding 22.)*
- f16 KV in the measurements (suspended).
- `GGML_VK_DISABLE_FUSION` (measured earlier: −5..−18%).
