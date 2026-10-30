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

```bash
T=docs-dev/llama-server-perf/tests/attach-test.py
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
