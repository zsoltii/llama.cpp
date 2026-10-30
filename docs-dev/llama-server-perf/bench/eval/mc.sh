#!/bin/bash
# Multiple-choice accuracy (level 2) with llama-perplexity: HellaSwag (first N tasks), Winogrande,
# TruthfulQA mc, ARC-Challenge, MMLU.  Same tasks and order for every (build, KV type) run.
#   mc.sh NAME BIN KVTYPE [HELLASWAG_TASKS=400] [MC_TASKS=300]   (winogrande: 600)
#   -> $BENCH_OUT/eval/mc-NAME-*.txt, summary line per test. ONLY=1: TruthfulQA + ARC only. Data: eval/get-data.sh.
set -e
source "$(dirname "$0")/../config.sh"
name=$1; bin=$2; kv=$3; nhs=${4:-400}; nmc=${5:-300}
COMMON=(-m "$MODEL" -c 2048 -b 2048 -ub 512 -fa on -t 16 -dev "${DEV:-$VK_DEV}" -ctk "$kv" -ctv "$kv")
export GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1
D=$BENCH_OUT/eval/data/mc; R=$BENCH_OUT/eval
mkdir -p $R

run() { # test-name, args...
    local t=$1; shift
    "$bin" "${COMMON[@]}" "$@" > "$R/mc-$name-$t.txt" 2>&1
    local last
    last=$(grep -E "^[0-9]+\s" "$R/mc-$name-$t.txt" | tail -1)
    local fin
    fin=$(grep -E "Final (Winogrande|result)" "$R/mc-$name-$t.txt" | tail -1)
    printf "%-10s %-12s %s %s\n" "$name" "$t" "$last" "$fin"
}

[ -z "$ONLY" ] && run hellaswag  -f $D/hellaswag_val_full.txt --hellaswag --hellaswag-tasks "$nhs"
[ -z "$ONLY" ] && run winogrande -f $D/winogrande-debiased-eval.csv --winogrande --winogrande-tasks 600
run truthfulqa -c 256 -np 16 -bf $D/truthful-qa-validation.bin --multiple-choice --multiple-choice-tasks "$nmc"
run arc-c      -c 256 -np 16 -bf $D/arc-challenge-validation.bin --multiple-choice --multiple-choice-tasks "$nmc"
[ -z "$ONLY" ] && run mmlu       -bf $D/mmlu-validation.bin --multiple-choice --multiple-choice-tasks "$nmc"
