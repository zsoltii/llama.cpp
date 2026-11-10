#!/bin/bash
# llama-bench of every profiled model on every given backend: pp512 / tg128 at depth 0 and 8192, FA on, q8_0 KV.
#   model-bench.sh TAG BACKEND...   backends: vk (own Vulkan, $VK_DEV), rocm (own ROCm/HIP, $ROCM_DEV)
#   MODELS="qwen35 gemma ..." picks models, REPS=3, DEPTHS=0,8192
#   -> $BENCH_OUT/model-bench/TAG-BACKEND-MODEL.jsonl, one summary line per test
source "$(dirname "$0")/../config.sh"
ROCM_BIN=${ROCM_BIN:-$REPO/build-rocm/bin}
ROCM_DEV=${ROCM_DEV:-ROCm0}
HUB=${HF_HUB:-$HOME/.cache/huggingface/hub}
R=$BENCH_OUT/model-bench; mkdir -p $R

gguf() { ls "$HUB"/models--$1/snapshots/*/$2 2>/dev/null | head -1; }
declare -A M=(
    [qwen35]="$(gguf unsloth--Qwen3.5-9B-MTP-GGUF Qwen3.5-9B-UD-Q4_K_XL.gguf)"
    [gemma]="$(gguf unsloth--gemma-4-12b-it-GGUF gemma-4-12b-it-UD-Q4_K_XL.gguf)"
    [qwen3]="$(gguf unsloth--Qwen3-14B-GGUF Qwen3-14B-Q4_K_M.gguf)"
    [oss]="$(gguf unsloth--gpt-oss-20b-GGUF gpt-oss-20b-UD-Q4_K_XL.gguf)"
    [qwen38]="$(gguf unsloth--Qwen3.8-27B-GGUF Qwen3.8-27B-UD-IQ4_XS.gguf)"
    [bge]="${BGE_GGUF:-$BENCH_OUT/models/bge-m3-hybrid-f16.gguf}"
)
MODELS=${MODELS:-"qwen35 gemma qwen3 oss qwen38 bge"}
REPS=${REPS:-3}
DEPTHS=${DEPTHS:-0,8192}

TAG=$1; shift
for be in "$@"; do
    case $be in
        vk)   BIN=$VK_BIN/llama-bench;   DEV=$VK_DEV ;;
        rocm) BIN=$ROCM_BIN/llama-bench; DEV=$ROCM_DEV ;;
        *) echo "unknown backend $be"; exit 1 ;;
    esac
    for m in $MODELS; do
        f=${M[$m]}
        [ -s "$f" ] || { echo "$m: model not found ($f)"; continue; }
        out=$R/$TAG-$be-$m.jsonl
        if [ $m = bge ]; then
            args=(-embd 1 -p 512,2048 -n 0 -b 2048 -ub 2048)
        else
            args=(-p 512 -n 128 -d "$DEPTHS" -ctk q8_0 -ctv q8_0)
        fi
        "$BIN" -m "$f" -dev "$DEV" -ngl 99 -fa 1 -r "$REPS" "${args[@]}" -o jsonl > "$out" 2> "$out.log" \
            || echo "$be $m: llama-bench failed, see $out.log"
        python3 - "$out" "$be" "$m" <<'PY'
import json, sys
for l in open(sys.argv[1]):
    r = json.loads(l)
    t = f"pp{r['n_prompt']}" if r["n_prompt"] else f"tg{r['n_gen']}"
    print(f"{sys.argv[2]:5} {sys.argv[3]:7} {t:7} d{r.get('n_depth', 0):<6} {r['avg_ts']:9.2f} t/s  +-{r['stddev_ts']:.2f}")
PY
    done
done
