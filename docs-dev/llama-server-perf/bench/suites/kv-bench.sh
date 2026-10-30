#!/bin/bash
# server bench per KV type (MTP n=1, defer + -bs, greedy): kv-bench.sh LABEL
source "$(dirname "$0")/../config.sh"
for t in q8_0 q4_0 q4_1 q5_0 q5_1 iq4_nl; do
  echo "=== $(date +%T) $1 $t"
  python3 $BENCH/bench.py --runs 2 --scenarios story,wiki --name kv-$1-$t --bin $VK_BIN/llama-server --env LLAMA_MTP_DEFER_CATCHUP=1 -- \
    -ctk $t -ctv $t -ctkd $t -ctvd $t --spec-type draft-mtp --spec-draft-n-max 1 -bs 2>&1 | grep -E "MEDIAN|rror"
done
