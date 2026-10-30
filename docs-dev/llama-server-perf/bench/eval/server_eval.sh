#!/bin/bash
# Level 3: generative accuracy through llama-server with the real serving config (MTP, q8_0 KV, 128k ctx).
#   server_eval.sh NAME BIN "ENV=.. ENV2=.." -- extra server args
# Runs: wiki grounded QA + abstention (wiki_qa.py), lm-eval nq_open (closed-book facts), gsm8k (5-shot,
# multiturn chat), ifeval.  Results: $BENCH_OUT/eval/wikiqa-NAME.json, $BENCH_OUT/eval/lmeval-NAME/
# lm-eval runs in the conda env $CONDA_ENV (default llama-eval, see README.md); DEV=CUDA0 for CUDA.
set -e
source "$(dirname "$0")/../config.sh"
cd "$BENCH/eval"
R=$BENCH_OUT/eval
name=$1; bin=$2; envs=$3; shift 3; [ "$1" = "--" ] && shift
LIMIT_NQ=${LIMIT_NQ:-300}; LIMIT_GSM=${LIMIT_GSM:-150}; LIMIT_IF=${LIMIT_IF:-150}
PORT=8080
TASKS=${TASKS:-wikiqa nq_open gsm8k ifeval}
run_task() { [[ " $TASKS " == *" $1 "* ]]; }
LOG=$R/server-$name.log
mkdir -p $R

env GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1 $envs "$bin" -hf $MODEL_HF -c 128000 -fa on \
    --parallel 1 -t 16 -tb 16 -dev ${DEV:-$VK_DEV} -ctv q8_0 -ctk q8_0 --port $PORT \
    --chat-template-kwargs '{"enable_thinking":false}' "$@" > "$LOG" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null; wait $SRV 2>/dev/null' EXIT
until curl -sf http://127.0.0.1:$PORT/health > /dev/null; do
    kill -0 $SRV 2>/dev/null || { echo "server died"; tail -5 "$LOG"; exit 1; }
    sleep 2
done

source "${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-llama-eval}"
run_task wikiqa && python wiki_qa.py --name "$name" --port $PORT | tail -1

MA="model=qwen,base_url=http://127.0.0.1:$PORT/v1/chat/completions,num_concurrent=1,max_retries=3,tokenized_requests=False"
lmeval() { # task limit extra...
    local t=$1 l=$2; shift 2
    HF_DATASETS_OFFLINE=1 lm_eval run --model local-chat-completions --model_args "$MA" --tasks "$t" --limit "$l" \
        --apply_chat_template --gen_kwargs temperature=0 --seed 1234 --log_samples \
        --output_path "$R/lmeval-$name" "$@" > "$R/lmeval-$name-$t.log" 2>&1
    grep -E "^\|" "$R/lmeval-$name-$t.log" | grep -vE "^\|(Tasks|-)" | sed "s/^/$name /"
}
run_task nq_open && lmeval nq_open "$LIMIT_NQ" --num_fewshot 5 --fewshot_as_multiturn \
    --system_instruction "Answer with only the short answer itself (a name, date, number or short phrase), no full sentence, no formatting."
run_task gsm8k && lmeval gsm8k "$LIMIT_GSM" --num_fewshot 5 --fewshot_as_multiturn
run_task ifeval && lmeval ifeval "$LIMIT_IF"
