# Shared settings of the bench scripts; source it. Every value can be overridden from the environment.
BENCH=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$BENCH/../../.." && pwd)

# results (logs, json, summary.csv) go outside the repo
export BENCH_OUT=${BENCH_OUT:-$HOME/llama-bench-results}

# builds: own Vulkan / CUDA, and an unmodified upstream build for baselines (see README.md)
VK_BIN=${VK_BIN:-$REPO/build-vk/bin}
CUDA_BIN=${CUDA_BIN:-$REPO/build-cuda/bin}
UP_BIN=${UP_BIN:-$REPO/../llama.cpp-upstream/build-cuda/bin}
UPV_BIN=${UPV_BIN:-$REPO/../llama.cpp-upstream/build-vk/bin}

# devices: llama-server -dev name, and the index for GGML_VK_VISIBLE_DEVICES (test-backend-ops then sees it as Vulkan0)
VK_DEV=${VK_DEV:-Vulkan2}
VK_IDX=${VK_IDX:-2}
CUDA_DEV=${CUDA_DEV:-CUDA0}
CUDA_ENV=${CUDA_ENV:-"CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1"}

# model: downloaded by llama-server -hf on the first start; the perplexity tools need the file path
MODEL_HF=unsloth/Qwen3.5-9B-MTP-GGUF:UD-Q4_K_XL
MODEL=${MODEL:-$(ls "$HOME"/.cache/huggingface/hub/models--unsloth--Qwen3.5-9B-MTP-GGUF/snapshots/*/Qwen3.5-9B-UD-Q4_K_XL.gguf 2>/dev/null | head -1)}

# the serving config that is measured
MTP="-ctkd q8_0 -ctvd q8_0 --spec-type draft-mtp --spec-draft-n-max 1"
QWEN="--temp 1.0 --top-k 20 --top-p 0.95 --min-p 0 --presence-penalty 1.5"

mkdir -p "$BENCH_OUT"
