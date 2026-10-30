#!/bin/bash
# Vulkan profiling ($VK_DEV, default B50): story (short ctx) and wiki (20k ctx) with the MTP server config + per-op perf logger
# logs: $BENCH_OUT/prof/prof-TAG-{story,wiki}.log; analyse with vkprof.py LOG [kind] or vkstep.py LOG N_STEPS
source "$(dirname "$0")/../config.sh"
S=$BENCH_OUT/prof; mkdir -p $S
export BENCH
VK=${VK:-$VK_BIN/llama-server}
TAG=${TAG:-base}
run() {  # $1=name $2=prompt-file-or-text-mode $3=max_tokens
    local name=$1
    GGML_VK_PERF_LOGGER=1 GGML_VK_PERF_LOGGER_FREQUENCY=1 LLAMA_SERVER_TIMING=1 LLAMA_MTP_DEFER_CATCHUP=1 $EXTRA_ENV \
    $VK -hf $MODEL_HF -c 128000 -fa on --parallel 1 -t 16 -tb 16 \
      -dev $VK_DEV --port 8092 -ctk q8_0 -ctv q8_0 -ctkd q8_0 -ctvd q8_0 --spec-type draft-mtp --spec-draft-n-max 1 -bs \
      > $S/prof-$TAG-$name.log 2>&1 &
    local srv=$!
    until curl -sf localhost:8092/health >/dev/null; do sleep 2; kill -0 $srv 2>/dev/null || { echo "server died"; return; }; done
    python3 - "$2" "$3" <<'PY'
import json, os, sys, urllib.request
B = os.environ["BENCH"] + "/corpus/"
mode, n = sys.argv[1], int(sys.argv[2])
if mode == "story":
    p = open(B + "story.txt").read()
else:
    p = open(B + "long-wiki.txt").read() + "\n\n" + open(B + "analysis-question.txt").read()
body = {"messages": [{"role": "user", "content": p}], "max_tokens": n, "temperature": 0, "top_k": 1,
        "cache_prompt": False, "chat_template_kwargs": {"enable_thinking": False}}
r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8092/v1/chat/completions",
        json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=1800))
t = r["timings"]
print(mode, "pp", round(t["prompt_per_second"], 1), "tg", round(t["predicted_per_second"], 2),
      "draft", t.get("draft_n_accepted"), "/", t.get("draft_n"))
PY
    kill $srv; wait $srv 2>/dev/null
}
run story story 256
run wiki wiki 128
