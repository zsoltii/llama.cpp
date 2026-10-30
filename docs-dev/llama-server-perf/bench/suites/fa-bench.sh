#!/bin/bash
# FA microbench: kv=20736 nb=1/2/3 + prefill 512x8704, per KV type. usage: [ENV=..] fa-bench.sh LABEL >> fa.txt
# compare labels: fa-cmp.py fa.txt
source "$(dirname "$0")/../config.sh"
for t in f16 q8_0 q4_0 q4_1 q5_0 q5_1 iq4_nl; do
  GGML_VK_VISIBLE_DEVICES=$VK_IDX $VK_BIN/test-backend-ops perf -o FLASH_ATTN_EXT -b Vulkan0 -p "type_K=$t," 2>&1 | grep -E "hsk=256.*kv=(20736|8704),nb=(1|2|3|512)," | \
    sed -E 's/.*kv=([0-9]+),nb=([0-9]+),.*type_K=([a-z0-9_]+),.*runs - +([0-9.]+) us\/run.*/\1 \2 \3 \4/' | \
    awk -v L="$1" '{ printf "%s kv=%s nb=%s %s %s\n", L, $1, $2, $3, $4 }'
done
