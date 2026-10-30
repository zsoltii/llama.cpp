#!/bin/bash
# matvec-only kernel bench of the Qwen3.5-9B decode shapes: mvbench.sh [ENV=VAL ...]
source "$(dirname "$0")/config.sh"
env GGML_VK_VISIBLE_DEVICES=$VK_IDX "$@" "$VK_BIN/test-backend-ops" perf -b Vulkan0 -o MUL_MAT -p "type_a=(q4_K|q5_K|q6_K|q8_0),type_b=f32,m=(12288|4096|8192|248320),n=[123],k=(4096|12288|8192),bs=\[1,1\],nr=\[1,1\],per=\[0,1,2,3\],k_v=0,o=1" 2>&1 | tee >(grep -m1 "ggml_vulkan: 0 =" >&2) | grep -oE "type_a=\w+,type_b=f32,m=[0-9]+,n=[0-9]+,k=[0-9]+.*us/run" | sed -E 's/,bs=[^)]*\)//; s/ +/ /g' | awk '{print $1, $(NF-1), $NF}'
