#!/bin/bash
# Story loop test: loop-run.sh CONFIG...   (SEEDS="1 2 3 4 5 6", MAXTOK=12000, ONLY_P=1 for the second prompt only)
# Configs: cu-full cu-mtp cu-nomtp up-nomtp up-mtp vk-full vk-qwen vk-qwen-dry; vk-* use $VK_DEV, cu-*/up-* use $CUDA_DEV.
# Results: $BENCH_OUT/loop-test/loop-L-CONFIG.txt, score them with loop-score.py.
source "$(dirname "$0")/../config.sh"
S=$BENCH/loop-test
OWN=$CUDA_BIN/llama-server
VK=$VK_BIN/llama-server
UP=$UP_BIN/llama-server
CE=$CUDA_ENV
SEEDS=${SEEDS:-"1 2 3 4 5 6"}
export MAXTOK=${MAXTOK:-12000}
for c in "$@"; do case $c in
 cu-full)  python3 $S/loop-test.py L-$c $OWN "$CE LLAMA_MTP_DEFER_CATCHUP=1" "-dev $CUDA_DEV $MTP -bs" $SEEDS ;;
 cu-mtp)   python3 $S/loop-test.py L-$c $OWN "$CE" "-dev $CUDA_DEV $MTP" $SEEDS ;;
 cu-nomtp) python3 $S/loop-test.py L-$c $OWN "$CE" "-dev $CUDA_DEV" $SEEDS ;;
 up-nomtp) python3 $S/loop-test.py L-$c $UP "$CE" "-dev $CUDA_DEV" $SEEDS ;;
 up-mtp)   python3 $S/loop-test.py L-$c $UP "$CE" "-dev $CUDA_DEV $MTP" $SEEDS ;;
 vk-full)  python3 $S/loop-test.py L-$c $VK "LLAMA_MTP_DEFER_CATCHUP=1" "-dev $VK_DEV $MTP -bs" $SEEDS ;;
 vk-qwen)  python3 $S/loop-test.py L-$c $VK "LLAMA_MTP_DEFER_CATCHUP=1" "-dev $VK_DEV $MTP -bs $QWEN" $SEEDS ;;
 vk-qwen-dry) python3 $S/loop-test.py L-$c $VK "LLAMA_MTP_DEFER_CATCHUP=1" "-dev $VK_DEV $MTP -bs $QWEN --dry-multiplier 0.8 --dry-penalty-last-n 16384" $SEEDS ;;
 *) echo "unknown config $c"; exit 1 ;;
esac; done
