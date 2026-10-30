#!/bin/bash
# reduced draft LM head A/B (MTP n=1, defer + -bs, greedy): dv-bench.sh LABEL [ENV=VAL ...]
# e.g. dv-bench.sh off LLAMA_MTP_DRAFT_VOCAB=-1; dv-bench.sh seed LLAMA_MTP_DRAFT_VOCAB_SEED=$BENCH/../testdata/seed-hu.txt
source "$(dirname "$0")/../config.sh"
L=$1; shift
E=(); for e in "$@"; do E+=(--env "$e"); done
python3 $BENCH/bench.py --runs 2 --scenarios story,wiki --name dv-$L --bin $VK_BIN/llama-server --env LLAMA_MTP_DEFER_CATCHUP=1 "${E[@]}" -- \
  -ctk q8_0 -ctv q8_0 $MTP -bs 2>&1 | grep -E "MEDIAN|rror|accept"
grep -E "draft_vocab|draft LM head" $BENCH_OUT/results/dv-$L.log | head -3
