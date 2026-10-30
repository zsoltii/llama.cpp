# Benchmark and validation scripts

These are the scripts behind the measurements in `../results.md`, `../b50-results.md` and `../progress.md`. With them the measurements can be repeated on another machine or GPU.

The model is always `unsloth/Qwen3.5-9B-MTP-GGUF:UD-Q4_K_XL`, and every server run starts from the same serving config: 128k context, FA on, q8_0 KV cache, `--parallel 1`, `-t 16 -tb 16`.

## Setup on a new machine

1. **Builds** in the repo root, with `-j4` on a laptop.
   - Vulkan:
     ```bash
     cmake -B build-vk -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build-vk -j4
     ```
   - CUDA (61 = 1080 Ti, 75 = 2080 Ti / T1200, 86 = 3090):
     ```bash
     cmake -B build-cuda -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="61;75;86" -DCMAKE_BUILD_TYPE=Release
     cmake --build build-cuda -j4
     ```
   - **Upstream baseline** (unmodified llama.cpp), used for the loop test (`up-*`), `q-bench.sh upv-*/upc-*` and the KLD reference. Build it the same way in a worktree next to the repo:
     ```bash
     git worktree add ../llama.cpp-upstream $(git merge-base HEAD master)
     ```
     It also works from any other directory, via `UP_BIN` / `UPV_BIN`.
2. **Model:** the first `llama-server -hf ...` start downloads it to `~/.cache/huggingface`. The perplexity-based tests find it there (or set `MODEL=/path/to.gguf`).
3. **Devices:** list them with `build-vk/bin/llama-server --list-devices`, then set the variables in `config.sh`, for example:
   ```bash
   export VK_DEV=Vulkan0 VK_IDX=0                       # -dev name, and the GGML_VK_VISIBLE_DEVICES index
   export CUDA_DEV=CUDA0 CUDA_ENV="CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0"
   ```
   On the original laptop (B50 = `Vulkan2`, eGPU CUDA card = PCI index 1) the defaults are already right.
4. **Accuracy data:** run `eval/get-data.sh`. It downloads the data into `$BENCH_OUT/eval/data` and checks it against the files used so far.
5. **lm-eval** (only for `eval/server_eval.sh`): a conda env named `llama-eval` with Python 3.11:
   ```bash
   pip install "lm-eval[api,ifeval]==0.4.13" datasets
   ```
   The first run needs network access for the datasets. Later runs work offline (`HF_DATASETS_OFFLINE=1`).

Results go to `$BENCH_OUT` (default `~/llama-bench-results`), never into the repo.

## Scripts

| Script | What it measures |
|---|---|
| `bench.py` | Server speed. Fresh server, warm-up, then the scenarios (`story`: tg; `wiki`: 20.7k-token prompt pp + tg; `wikipp`; `quick`), greedy, `cache_prompt=false`. It records the medians, the draft acceptance, the output hash (determinism) and GPU/CPU load (Intel xe only) in `summary.csv`. `BENCH_SAMPLING=server` uses the server's sampling instead of greedy. |
| `kbench.py`, `mvbench.sh` | Kernel micro-benchmarks of the decode shapes (mat-vec, FA) via `test-backend-ops perf`. |
| `suites/q-bench.sh` | Speed with the Qwen-recommended sampling: own vs upstream build, with and without MTP, Vulkan and CUDA. |
| `suites/kv-bench.sh` | Speed per KV-cache type (q8_0, q4_0, q4_1, q5_0, q5_1, iq4_nl). |
| `suites/dv-bench.sh` | A/B of the reduced draft LM head (`LLAMA_MTP_DRAFT_VOCAB*`). The Hungarian seed text is `../testdata/seed-hu.txt`. |
| `suites/fa-bench.sh`, `suites/fa-cmp.py` | FA kernel microbench per KV type, and a comparison of labels. |
| `loop-test/loop-run.sh` | Story loop test: 2 Hungarian story prompts, 6 seeds, streaming abort on a loop. `loop-score.py` scores the output files after the run. |
| `../tests/attach-test.py` | Attachment test: the 2011 Fundamental Law (41k tokens) sent the way the web UI sends it. See `../tests/README.md`. |
| `eval/kld.sh` | Level 1: KL divergence against f16-KV reference logits (wikitext-2, 6×4096 tokens). |
| `eval/mc.sh` | Level 2: HellaSwag, Winogrande, TruthfulQA, ARC-Challenge, MMLU via `llama-perplexity`. |
| `eval/server_eval.sh`, `eval/wiki_qa.py` | Level 3: wiki grounded QA and hallucination check (25 answerable + 5 unanswerable questions), plus lm-eval nq_open, gsm8k and ifeval through the server. |
| `prof/vkprof.sh`, `prof/vkprof.py`, `prof/vkstep.py` | Vulkan per-op profile (`GGML_VK_PERF_LOGGER`) of the story and wiki scenarios, as a per-graph or per-MTP-step breakdown. |
| `suites/validate.sh` | Validation after a speedup, see below. |

## After every speedup

```bash
B=docs-dev/llama-server-perf/bench
$B/suites/validate.sh            # test-backend-ops, wiki QA, speed, story loop test, attachment test (~3 h)
$B/eval/kld.sh run <name> build-vk/bin/llama-perplexity q8_0
```

For CUDA:
- run `loop-run.sh cu-full cu-nomtp up-nomtp`;
- run `q-bench.sh ownc ownc-nomtp upc-nomtp`;
- run `attach-test.py` with `--bin build-cuda/bin/llama-server --dev CUDA0 --env CUDA_VISIBLE_DEVICES=...`;
- pass `DEV=CUDA0` to the eval scripts.

A change is OK if:
- `test-backend-ops` passes;
- KLD and wiki QA do not get worse;
- the loop rates are no worse than upstream's with the same sampling;
- the speed improves.

The baselines per GPU are in `../b50-results.md` (B50), `../results.md` and `../progress.md` (2080 Ti and the others).

Run heavy jobs one after the other: only one server or GPU test at a time.

## Usage guide

The examples run from the repo root with `B=docs-dev/llama-server-perf/bench`.
Each script starts its own server and stops it at the end. Stop any other `llama-server` first, because the ports are fixed: 8080 for `bench.py`, `server_eval.sh` and `validate.sh`; 8091 for the loop test; 8092 for the attachment test and `vkprof.sh`.

### Common settings (`config.sh`)

Every shell script reads `config.sh`. Each setting can be overridden from the environment, e.g. `VK_DEV=Vulkan0 VK_IDX=0 $B/suites/kv-bench.sh x`.

| Variable | Default | Meaning |
|---|---|---|
| `BENCH_OUT` | `~/llama-bench-results` | where all results go (the Python scripts read it too) |
| `VK_BIN`, `CUDA_BIN` | `build-vk/bin`, `build-cuda/bin` | own builds |
| `UPV_BIN`, `UP_BIN` | `../llama.cpp-upstream/build-{vk,cuda}/bin` | unmodified upstream builds (baseline) |
| `VK_DEV` | `Vulkan2` | `-dev` of the Vulkan server runs (also read by `bench.py`) |
| `VK_IDX` | `2` | `GGML_VK_VISIBLE_DEVICES` index for `test-backend-ops` (also read by `kbench.py`) |
| `CUDA_DEV`, `CUDA_ENV` | `CUDA0`, `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1` | CUDA device and its env |
| `MODEL` | the `-hf` download in `~/.cache/huggingface` | GGUF path for `llama-perplexity` |

### Server speed: `bench.py`

```bash
python3 $B/bench.py --name NAME [--bin BIN] [--dev DEV] [--env K=V ...] [--runs 3] [--scenarios story,wiki] [--port 8080] -- [server args]
# e.g. own Vulkan build, MTP n=1 + backend sampling:
python3 $B/bench.py --name vk-mtp --bin build-vk/bin/llama-server --env LLAMA_MTP_DEFER_CATCHUP=1 -- \
    -ctkd q8_0 -ctvd q8_0 --spec-type draft-mtp --spec-draft-n-max 1 -bs
```

- `--bin`: the server binary (default: `llama-server` from `PATH`).
- `--dev`: replaces `-dev` (default `$VK_DEV`, e.g. `CUDA0`).
- `--env`: extra environment variable for the server, repeatable.
- `--runs`: runs per scenario, interleaved; the medians are reported.
- `--scenarios`:
  - `story`: short prompt, 1024 generated tokens;
  - `wiki`: 20.7k-token prompt, 512 generated tokens;
  - `wikipp`: the same prompt, prefill only;
  - `quick`: 128 generated tokens.
- The server args come after `--`. They are appended to the base config: 128k ctx, FA on, q8_0 KV, `-t 16 -tb 16`, `--parallel 1`.
- Sampling is greedy (temp 0, top-k 1). `BENCH_SAMPLING=server` uses the sampling given on the server command line instead.
- Output:
  - `MEDIAN ...` lines;
  - `$BENCH_OUT/results/NAME.{log,json,mon.csv}`;
  - a row per scenario in `$BENCH_OUT/summary.csv` (`deterministic` = the same output hash in every run).

### Speed suites (`suites/`)

| Command | Parameters |
|---|---|
| `$B/suites/q-bench.sh CONFIG...` | Qwen sampling, 3 runs, story + wiki. Configs: `ownv` (own Vulkan, MTP + `-bs`), `ownv-nomtp`, `upv-mtp`, `upv-nomtp`, `ownc`, `ownc-nomtp`, `upc-mtp`, `upc-nomtp` (CUDA). Env: `T=prefix-` (result name prefix, default `b50-`), `SC=story,wiki`. |
| `$B/suites/kv-bench.sh LABEL` | Every KV type (q8_0, q4_0, q4_1, q5_0, q5_1, iq4_nl) for target and draft, MTP n=1, greedy. |
| `$B/suites/dv-bench.sh LABEL [ENV=VAL ...]` | Reduced draft LM head A/B. Env:<br>• `LLAMA_MTP_DRAFT_VOCAB=-1` turns it off; N = static prefix (default 32768);<br>• `LLAMA_MTP_DRAFT_VOCAB_DYN` = dynamic rows (default 16384);<br>• `LLAMA_MTP_DRAFT_VOCAB_SEED=FILE` = seed text, e.g. `$B/../testdata/seed-hu.txt`. |
| `$B/suites/fa-bench.sh LABEL >> fa.txt` | FA kernel times (kv 20736 at nb 1/2/3, prefill 512×8704) per KV type. With an env prefix (e.g. `GGML_VK_X=1 fa-bench.sh x >> fa.txt`) it compares settings. `python3 $B/suites/fa-cmp.py fa.txt` prints a table; the first label is the baseline. |

### Kernel micro-benchmarks

```bash
python3 $B/kbench.py NAME [KEY=VAL ...]   # decode mat-vec shapes + FA kv=20736 -> $BENCH_OUT/kernels/NAME.txt
$B/mvbench.sh [KEY=VAL ...]               # mat-vec shapes only, n=1/2/3
```

`TBO=path/test-backend-ops` selects another build. Both print the measured device. Check it: without the eGPU, index 2 can be llvmpipe.

### Story loop test (`loop-test/`)

```bash
SEEDS="1 2 3 4 5 6" MAXTOK=12000 ONLY_P=1 $B/loop-test/loop-run.sh vk-full vk-qwen
python3 $B/loop-test/loop-score.py $BENCH_OUT/loop-test/loop-L-vk-qwen.txt
```

- Configs:
  - `vk-full`: own Vulkan, MTP + `-bs`, server default sampling;
  - `vk-qwen`: the same with Qwen sampling;
  - `vk-qwen-dry`: + DRY;
  - `cu-full`, `cu-mtp`, `cu-nomtp`: own CUDA;
  - `up-mtp`, `up-nomtp`: upstream CUDA.
- Prompts:
  - p0: "tervezz meg egy 10000 szavas gyerekenek szóló mesét, amit később meg kell írnod";
  - p1: "csinálj 10000 szavas mesét avgy hosszabbat". This one loops more often.
- `ONLY_P=0|1` runs only that prompt.
- Each run is aborted as soon as a loop is detected.
- `loop-score.py` prints one verdict per run and `not ok: x/y`:
  - `LOOP`: aborted, or its last 200 chars repeat ≥3× in the last 12000;
  - `RUNAWAY`: hit the token limit without repeating;
  - `ok`.
- For any other config, call the runner directly: `python3 $B/loop-test/loop-test.py NAME BIN "ENV=.. ENV=.." "server args" SEED...`.

### Attachment test (`../tests/attach-test.py`)

```bash
python3 docs-dev/llama-server-perf/tests/attach-test.py --name NAME --bin BIN --dev DEV \
    [--sampling default|qwen|qwen-dry] [--seeds 1,2,3] [--max-tokens 16000] [--env K=V ...] \
    [--no-mtp] [--no-ui] [--port 8092] [--out DIR] [-- extra server args]
```

- `--sampling`:
  - `default`: server defaults;
  - `qwen` (default): `--temp 1.0 --top-k 20 --top-p 0.95 --min-p 0 --presence-penalty 1.5`;
  - `qwen-dry`: `qwen` + DRY.
- `--no-mtp`: no MTP flags.
- `--no-ui`: leaves out the extra fields the web UI adds.
- Standard run (the user's command): `--sampling qwen --seeds $(seq -s, 7 30) --max-tokens 32000 --env LLAMA_MTP_DEFER_CATCHUP=1`.
- Output:
  - one line per seed, with pp, tg, draft acceptance, loop score (thinking/answer) and lengths;
  - at the end, `not ok: x/y`;
  - `$BENCH_OUT/attach-test/attach-NAME.{jsonl,txt,srv.log}`.

### Accuracy (`eval/`)

```bash
$B/eval/get-data.sh                                        # once: wikitext-2 + multiple-choice data, checksummed
$B/eval/kld.sh base [REF_BIN]                              # once per machine: f16-KV reference logits (~5.7 GB)
$B/eval/kld.sh run NAME build-vk/bin/llama-perplexity q8_0 # mean KLD, 99.9% KLD, same-top-p, PPL vs the reference
$B/eval/mc.sh NAME build-vk/bin/llama-perplexity q8_0 [HELLASWAG_TASKS=400] [MC_TASKS=300]
$B/eval/server_eval.sh NAME build-vk/bin/llama-server "LLAMA_MTP_DEFER_CATCHUP=1" -- \
    -ctkd q8_0 -ctvd q8_0 --spec-type draft-mtp --spec-draft-n-max 1 -bs
```

- **`DEV=CUDA0`** switches all three to CUDA (default `$VK_DEV`).
- **`kld.sh`:** `REF_BIN` defaults to the upstream Vulkan `llama-perplexity`.
- **`mc.sh`:** `ONLY=1` runs only TruthfulQA and ARC.
- **`server_eval.sh`:**
  - `TASKS="wikiqa nq_open gsm8k ifeval"` selects the tasks;
  - `LIMIT_NQ=300 LIMIT_GSM=150 LIMIT_IF=150` are the sample counts;
  - `CONDA_SH`, `CONDA_ENV` locate the lm-eval env (default `~/anaconda3/...`, `llama-eval`);
  - the server runs with thinking disabled.
- **The wiki QA alone**, against an already running server: `python3 $B/eval/wiki_qa.py --name NAME [--port 8080]`. It reports 25 answerable and 5 unanswerable questions; the latter must be answered `NOT_IN_TEXT`.
- Results: `$BENCH_OUT/eval/`.

### Vulkan profile (`prof/`)

```bash
TAG=base [VK=path/llama-server] [EXTRA_ENV="GGML_VK_X=1"] $B/prof/vkprof.sh
python3 $B/prof/vkprof.py $BENCH_OUT/prof/prof-base-story.log [verify|draft|decode1|prefill]
python3 $B/prof/vkstep.py $BENCH_OUT/prof/prof-base-story.log N_STEPS [TOP=45]
```

`vkprof.sh` runs the story (256 tokens) and wiki (128 tokens) scenarios with `GGML_VK_PERF_LOGGER`, in the MTP serving config.
- `vkprof.py`: GPU time per op and graph type.
- `vkstep.py`: GPU time per MTP step. `N_STEPS` is the number of verify steps, i.e. the generated tokens minus the accepted drafts.

### Full validation: `suites/validate.sh`

```bash
[SKIP_ATTACH=1] $B/suites/validate.sh [TAG]   # TAG defaults to the current git commit
```

It runs, in order:
1. `test-backend-ops` on the `$VK_IDX` device;
2. the wiki QA;
3. `q-bench.sh ownv ownv-nomtp`;
4. the story loop test (p1, `vk-full`, `vk-qwen`);
5. the attachment standard run (24 seeds, ~1.7 h).

Logs go to `$BENCH_OUT/validate-TAG/`. It covers Vulkan only; for CUDA, use the commands in "After every speedup".
