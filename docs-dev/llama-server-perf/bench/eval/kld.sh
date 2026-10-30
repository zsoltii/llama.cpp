#!/bin/bash
# KL-divergence (level 1 accuracy): reference logits with f16 KV from a reference build (default: the upstream
# Vulkan build), then every (build, KV type) combination is compared against it on the same wikitext-2 chunks.
#   kld.sh base [REF_BIN]       -> $BENCH_OUT/eval/data/base-f16.kld (~5.7 GB)
#   kld.sh run NAME BIN KVTYPE  -> $BENCH_OUT/eval/kld-NAME.txt
# Data: eval/get-data.sh. Device: $DEV (default $VK_DEV).
set -e
source "$(dirname "$0")/../config.sh"
D=$BENCH_OUT/eval/data; R=$BENCH_OUT/eval
COMMON=(-m "$MODEL" -f $D/wiki.test.raw -c 4096 --chunks 6 -fa on -t 16 -dev "${DEV:-$VK_DEV}" -b 512 -ub 512)
export GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1
mkdir -p $R

case "$1" in
base)
    "${2:-$UPV_BIN/llama-perplexity}" "${COMMON[@]}" -ctk f16 -ctv f16 \
        --kl-divergence-base $D/base-f16.kld 2>&1 | tee $R/kld-base-f16.txt | grep -E "^\[|Final estimate"
    ;;
run)
    name=$2; bin=$3; kv=$4
    "$bin" "${COMMON[@]}" -ctk "$kv" -ctv "$kv" --kl-divergence-base $D/base-f16.kld --kl-divergence \
        > "$R/kld-$name.txt" 2>&1
    grep -E "Mean PPL\(Q\)|Mean    KLD|Maximum KLD|99.9%   KLD|Same top p|RMS Δp" "$R/kld-$name.txt"
    ;;
*)
    echo "usage: $0 base [REF_BIN] | run NAME BIN KVTYPE"; exit 1;;
esac
