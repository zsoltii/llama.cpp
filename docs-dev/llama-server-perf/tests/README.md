# llama-server perf tests

## Attachment test (`attach-test.py`)

The user hit a repetition loop in the **thinking** phase on the Vulkan build with this request: the 2011 Fundamental Law of Hungary (`../testdata/alaptorveny-2011.txt`, 137 KB, ~41.4k tokens) attached as a `.txt` file in the web UI, with the prompt "foglald össze a magyar alaptörvényt" ("summarize the Hungarian Fundamental Law").

The script sends the request the same way the web UI does (`tools/ui/src/lib/services/chat.service.ts`):
- the attachment is a text part `"\n\n--- File: <name> ---\n<content>"`;
- the prompt follows as a second text part.

Each seed is a fresh request with `cache_prompt: false`, so every run also measures the 41k-token prefill.

It records per run:
- prompt processing and generation speed;
- MTP draft acceptance;
- the loop score of the thinking and of the answer;
- whether the run never left the thinking phase.

The run is aborted as soon as a loop is detected: a unit of 1–200 chars repeated ≥ 3 times, ≥ 300 chars long.

Run it on every GPU and backend after each speedup, next to the story loop test and the accuracy tests:

**Standard run:** the user's own server command, the one that looped once in the web UI:

```bash
LLAMA_MTP_DEFER_CATCHUP=1 llama-server -hf unsloth/Qwen3.5-9B-MTP-GGUF:UD-Q4_K_XL -c 128000 -fa on --parallel 1 \
    -t 16 -tb 16 -dev Vulkan2 -ctk q8_0 -ctv q8_0 -ctkd q8_0 -ctvd q8_0 \
    --spec-type draft-mtp --spec-draft-n-max 1 -bs \
    --temp 1.0 --top-k 20 --top-p 0.95 --min-p 0 --presence-penalty 1.5
```

With the web UI fields and 24 seeds, this is:

```bash
T=docs-dev/llama-server-perf/tests/attach-test.py
python3 $T --name <gpu>-vk-user --bin build-vk/bin/llama-server --dev Vulkan2 --sampling qwen \
    --seeds $(seq -s, 7 30) --max-tokens 32000 --env LLAMA_MTP_DEFER_CATCHUP=1
```

Baseline (B50, Vulkan, 2026-10-30):
- 0/24 loops;
- pp 446, tg 22.1–22.9 t/s;
- draft acceptance 0.68–0.73.

A build or GPU passes if it does not loop more often, and pp/tg can be compared seed by seed.

Other variants:

```bash
# Vulkan (B50 = Vulkan2), MTP n=1 + backend sampling, Qwen-recommended sampling, 6 seeds
python3 $T --name b50-vk-qwen --bin build-vk/bin/llama-server --dev Vulkan2 --sampling qwen \
    --seeds 1,2,3,4,5,6 --env LLAMA_MTP_DEFER_CATCHUP=1
# also: --sampling default | qwen-dry, --no-mtp; CUDA: --bin build-cuda/bin/llama-server --dev CUDA0
#       (with --env CUDA_VISIBLE_DEVICES=... if needed)
```

Outputs are written to `~/llama-bench-results/attach-test/` by default (`--out`):
- `attach-<name>.jsonl`: one record per run;
- `attach-<name>.txt`: the thinking and answer texts;
- `attach-<name>.srv.log`: the server log.

The result line ends with `not ok: x/y`.
