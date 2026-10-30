#!/bin/bash
# speed with the Qwen-recommended (loop-free) sampling: upstream vs own, with and without MTP
#   q-bench.sh CONFIG...   configs: upv-nomtp upv-mtp ownv-nomtp ownv (Vulkan, $VK_DEV); upc-nomtp upc-mtp ownc-nomtp ownc (CUDA)
#   T=prefix for the result names, SC=story,wiki scenarios
source "$(dirname "$0")/../config.sh"
export BENCH_SAMPLING=server
T=${T:-b50-}
run() { echo "=== $(date +%H:%M:%S) $2"; python3 $BENCH/bench.py --runs 3 --scenarios ${SC:-story,wiki} "$@" 2>&1 | grep -E "MEDIAN|VRAM|offload|Error|error" ; }
CE=(); for e in $CUDA_ENV; do CE+=(--env "$e"); done
for c in "$@"; do case $c in
 upv-nomtp)  run --name q-${T}upv-nomtp --bin $UPV_BIN/llama-server -- $QWEN ;;
 upv-mtp)    run --name q-${T}upv-mtp --bin $UPV_BIN/llama-server -- $MTP $QWEN ;;
 ownv-nomtp) run --name q-${T}ownv-nomtp --bin $VK_BIN/llama-server -- $QWEN ;;
 ownv)       run --name q-${T}ownv --bin $VK_BIN/llama-server --env LLAMA_MTP_DEFER_CATCHUP=1 -- $MTP -bs $QWEN ;;
 upc-nomtp)  run --name q-${T}upc-nomtp --bin $UP_BIN/llama-server --dev $CUDA_DEV "${CE[@]}" -- $QWEN ;;
 upc-mtp)    run --name q-${T}upc-mtp --bin $UP_BIN/llama-server --dev $CUDA_DEV "${CE[@]}" -- $MTP $QWEN ;;
 ownc-nomtp) run --name q-${T}ownc-nomtp --bin $CUDA_BIN/llama-server --dev $CUDA_DEV "${CE[@]}" -- $QWEN ;;
 ownc)       run --name q-${T}ownc --bin $CUDA_BIN/llama-server --dev $CUDA_DEV "${CE[@]}" --env LLAMA_MTP_DEFER_CATCHUP=1 -- $MTP -bs $QWEN ;;
 *) echo "unknown config $c"; exit 1 ;;
esac; done
echo "=== $(date +%H:%M:%S) done"
