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
