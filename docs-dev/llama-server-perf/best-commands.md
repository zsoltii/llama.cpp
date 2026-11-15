# Best llama-server commands per card and model

Every command is a single line that works with copy and paste (run it from the repository root; the binaries are addressed relative to it, the cached MTP files with `~`).
Speed numbers are tokens/s from `llama-bench` / `llama-server` runs (see `bestsweep2.txt`, `specrun-final.txt` in `~/llama-bench-results`).

## Rules for writing a section for a new card

1. **Only models that fit the card's memory get a command.** A model fits when its weights plus a usable context
   (at least 32k tokens, or the model maximum if that is smaller) fit in the card's VRAM with about 2 GiB free.
   Check with `llama-fit-params -m MODEL.gguf -dev DEV -fa on -ctk q8_0 -ctv q8_0 -ub 2048 --parallel 1 -fitt 2048`
   (no `-ngl`; it prints the largest `-c` that fits). Models that need CPU offload are left out.
2. **The goal is the largest context.** `-c` is the model maximum; if f16 KV does not reach it, use `-ctk q8_0 -ctv q8_0`,
   and if q8_0 does not reach it, `-ctk iq4_nl -ctv iq4_nl` (on CUDA 1-4 % slower decode than f16, on Vulkan 5-15 %). If even
   iq4_nl does not reach the maximum, `-c` is the value `llama-fit-params` printed for iq4_nl. Run `llama-fit-params` once per KV type and per
   backend and take the first KV type that reaches the maximum.
3. Draft models (MTP): keep `-ctkd q8_0 -ctvd q8_0`; their KV is small, the 2 GiB margin covers it. If a draft does not fit, drop it before
   shrinking the main context.
4. **Repetition loops:** the gemma-4 models can fall into a thinking loop in long agent sessions (opencode, `high`/`max` thinking, ~100k context, after a failing
   build/test repeats). The server has no repetition protection by default (`repeat_penalty 1.0`, DRY off), so the gemma-4 commands carry `--dry-multiplier 0.8`.
   The KV type and the MTP draft are not the cause: 94k-token thinking runs with f16 KV, q8_0 KV and q8_0 KV + MTP showed no loops (`loopab.txt`).
   Other models: add the same option if you see the same behaviour. A client that sends its own `dry_multiplier` overrides it.
5. `-ub 2048`, except Nemotron Bolt (`-ub 1024`, +12 % prefill on CUDA, +5 % on Vulkan).
6. Verify each line: start it, wait for `/health`, send one request, and record the VRAM used.
7. Use the card's own device selection: CUDA `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<index>` and `-dev CUDA0`; Vulkan `-dev Vulkan<N>`; HIP `-dev ROCm0`.
   Build with `build-cuda` / `build-vk` (and the HIP build for AMD).

## RTX 3090 (24 GB)


### Measured speed (f16 KV, `-ub 2048`)

`SPEC ARGS` are the extra arguments of the best configuration. "tg server" is the 20.7k prompt / short prompt generation speed
with q8_0 KV and `-c 32768` (the specrun setting), without / with the draft.

| model | CUDA best | Vulkan best |
|---|---|---|
| Qwen3.5-9B (UD-Q4_K_XL, MTP GGUF) | f16 KV, `--spec-type draft-mtp --spec-draft-n-max 2 -ctkd q8_0 -ctvd q8_0` (short prompts: n-max 2 gives 174, long: n-max 1 and 2 equal 151-154). pp 4991, tg 119 / 111 @16k; server tg 107 -> 154 | f16 KV, `--spec-type draft-mtp --spec-draft-n-max 1 ...`. pp 4167, tg 103 / 97 @16k; server tg 93 -> 121 |
| Ornith-1.5-9B (Q4_K_M, MTP) | f16 KV, `--spec-type draft-mtp --spec-draft-n-max 1 ...`. pp 4944, tg 124; server 111 -> 144 | same flags. pp 4190, tg 107; server 96 -> 115 |
| Phi-4-reasoning-plus (UD-Q4_K_XL) | f16 KV, no draft. pp 3027, tg 85 / 64 @16k | f16 KV. pp 2683, tg 80 / 58 @16k |
| gemma-4-12b (UD-Q4_K_XL) | f16 KV, `-md MTP/mtp-gemma-4-12b-it-Q8_0.gguf --spec-type draft-mtp -ctkd q8_0 -ctvd q8_0`. pp 3429, tg 86 / 79 @16k; server 73 -> 109 | same flags. pp 3000, tg 73; server 64 -> 81 |
| gemma-4-26B-A4B (UD-IQ4_NL) | f16 KV, `-md MTP/mtp-gemma-4-26B-A4B-it-Q8_0.gguf --spec-type draft-mtp ...`. pp 6922, tg 143 / 129 @16k; server 118 -> 144 | same flags. pp 5101, tg 125 / 113 @16k; server 103 -> 117 |
| Nemotron-3.5-Lightning-30B-A3B (UD-IQ3_XXS) | f16 KV, `--spec-type draft-mtp --spec-draft-n-max 1 ...`. pp 7368, tg 200 / 195 @16k; server 182 -> 240 | same flags. pp 5360, tg 172 / 166 @16k; server 155 -> 172 |
| Nemotron-3.5-Bolt-E4B (gemma4 arch, UD-Q4_K_XL) | f16 KV, **`-ub 1024`**, no draft (DFlash 111 vs 111, no gain). pp 9274 (q8_0, ub 1024) / 8289 (f16), tg 139 / 127 @16k | f16 KV, `-ub 1024` for prefill (7610 vs 7275, tg unchanged 130). No draft (DFlash is slower on Vulkan: 71 vs 106) |
| flint (UD-Q3_K_XL) | f16 KV, no draft. pp 7195, tg 154 / 140 @16k | f16 KV. pp 5450, tg 134 / 119 @16k |
| flint (UD-IQ3_XXS) | f16 KV, no draft. pp 6744, tg 160 / 145 @16k (q8_0: 6538 / 159 / 142) | f16 KV. pp 5241, tg 123 / 110 @16k (q8_0: 5024 / 118 / 105) |
| gpt-oss-20b (UD-Q4_K_XL) | f16 KV, no draft (the third-party draft GGUFs do not load, Eagle3 is slower). pp 7661, tg 210 / 190 @16k | f16 KV. pp 6023, tg 184 / 158 @16k |
| GLM-4.7-Flash-REAP-23B-A3B (UD-Q4_K_XL) | f16 KV (**required** for speed: q8_0 decode at 16k is 93 on CUDA and 60 on Vulkan against 110 / 102). pp 5537, tg 140 / 110 @16k | f16 KV. pp 4218, tg 135 / 102 @16k, pp@16k 990 |
| Qwen3-14B (Q4_K_M) | f16 KV for prefill (2950 vs 2910); at long contexts q8_0 decodes faster on CUDA (68 vs 66 @16k). No draft (Eagle3 is slower) | f16 KV, no draft. pp 2843, tg 80 / 59 @16k |
| Qwen3.8-27B (UD-IQ4_XS) | f16 KV (or `iq4_nl` for >64k contexts, tg@64k 37.8). pp 1704, tg 48 / 45 @16k; with MTP see `rtx3090-results.md` | f16 KV. pp 1252, tg 39 / 37 @16k |
| Laguna-XS-2.1 (IQ4_NL) | f16 KV. pp 6645, tg 187 / 159 @16k; server tg 144 (21k prompt) / 165 | f16 KV. pp 5441, tg 162 / 135 @16k; server tg 115 / 139 |
| LFM2.5-24B-A2B (UD-Q4_K_XL) | f16 KV. pp 10303, tg 265 / 238 @16k; server tg 194 / 226 | f16 KV. pp 8488, tg 231 / 198 @16k; server tg 166 / 192 |
| Muse-Glimmer-30B (UD-Q4_K_XL) | f16 KV, `-md dflash-kquant.gguf --spec-type draft-dflash -ctkd q8_0 -ctvd q8_0`. pp 1661, tg 48 / 47 @16k; server tg 45 / 46 -> 59 / 70 (acceptance 0.36-0.46) | f16 KV, **no draft** (DFlash is slower on Vulkan: 38 / 44 vs 42 / 44). pp 1474, tg 46 / 44 @16k |
| Devstral-Small-2-24B (UD-Q4_K_XL) | f16 KV, no draft (none exists). pp 1957, tg 53 / 46 @16k; server tg 43 / 51 | f16 KV. pp 1772, tg 49 / 40 @16k; server tg 36 / 46 |
| Magistral-Small-2509 (UD-Q4_K_XL) | f16 KV, no draft. pp 1958, tg 54 / 47 @16k; server tg 44 / 52 | f16 KV. pp 1780, tg 52 / 42 @16k; server tg 39 / 50 |






Commands below use the context and KV type from the rules (smallest KV quantization that reaches the model maximum). Prefill and decode are a few percent slower than the f16 numbers above where the KV is quantized; the f16 line is the fastest choice if you need less context.


### Commands: CUDA (`-dev CUDA0`)

**Qwen3.5-9B (MTP)**

```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 LLAMA_MTP_DEFER_CATCHUP=1 build-cuda/bin/llama-server -hf unsloth/Qwen3.5-9B-MTP-GGUF:UD-Q4_K_XL -dev CUDA0 -ngl 99 -fa on -c 262144 -ctk f16 -ctv f16 -ub 2048 --parallel 1 --no-mmproj --host 127.0.0.1 --port 8080 --spec-type draft-mtp --spec-draft-n-max 2 -ctkd q8_0 -ctvd q8_0
```

**Ornith-1.5-9B (MTP)**

```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 LLAMA_MTP_DEFER_CATCHUP=1 build-cuda/bin/llama-server -hf ornith-ai/Ornith-1.5-9B-GGUF:Q4_K_M -dev CUDA0 -ngl 99 -fa on -c 262144 -ctk f16 -ctv f16 -ub 2048 --parallel 1 --no-mmproj --host 127.0.0.1 --port 8080 --spec-type draft-mtp --spec-draft-n-max 1 -ctkd q8_0 -ctvd q8_0
```

**Phi-4-reasoning-plus**

```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 LLAMA_MTP_DEFER_CATCHUP=1 build-cuda/bin/llama-server -hf unsloth/Phi-4-reasoning-plus-GGUF:UD-Q4_K_XL -dev CUDA0 -ngl 99 -fa on -c 32768 -ctk f16 -ctv f16 -ub 2048 --parallel 1 --no-mmproj --host 127.0.0.1 --port 8080
```

**gemma-4-12b (MTP)**

```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 LLAMA_MTP_DEFER_CATCHUP=1 build-cuda/bin/llama-server -hf unsloth/gemma-4-12b-it-GGUF:UD-Q4_K_XL -dev CUDA0 -ngl 99 -fa on -c 262144 -ctk f16 -ctv f16 -ub 2048 --parallel 1 --no-mmproj --host 127.0.0.1 --port 8080 -md ~/.cache/huggingface/hub/models--unsloth--gemma-4-12b-it-GGUF/snapshots/fc034cfff751157913579611efad8462ac1be606/MTP/mtp-gemma-4-12b-it-Q8_0.gguf --spec-type draft-mtp -ctkd q8_0 -ctvd q8_0 --dry-multiplier 0.8
```

**gemma-4-26B-A4B (MTP)**

```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 LLAMA_MTP_DEFER_CATCHUP=1 build-cuda/bin/llama-server -hf unsloth/gemma-4-26B-A4B-it-GGUF:UD-IQ4_NL -dev CUDA0 -ngl 99 -fa on -c 262144 -ctk q8_0 -ctv q8_0 -ub 2048 --parallel 1 --no-mmproj --host 127.0.0.1 --port 8080 -md ~/.cache/huggingface/hub/models--unsloth--gemma-4-26B-A4B-it-GGUF/snapshots/c099eb48e663fd284577b04978a94ffccb261841/MTP/mtp-gemma-4-26B-A4B-it-Q8_0.gguf --spec-type draft-mtp -ctkd q8_0 -ctvd q8_0 --dry-multiplier 0.8
```

**Nemotron-3.5-Lightning-30B-A3B (MTP)**

```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 LLAMA_MTP_DEFER_CATCHUP=1 build-cuda/bin/llama-server -hf unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF:UD-IQ3_XXS -dev CUDA0 -ngl 99 -fa on -c 393216 -ctk iq4_nl -ctv iq4_nl -ub 2048 --parallel 1 --no-mmproj --host 127.0.0.1 --port 8080 --spec-type draft-mtp --spec-draft-n-max 1 -ctkd q8_0 -ctvd q8_0
```

**Nemotron-3.5-Bolt-E4B**

```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 LLAMA_MTP_DEFER_CATCHUP=1 build-cuda/bin/llama-server -hf unsloth/NVIDIA-Nemotron-3.5-Bolt-E4B-GGUF:UD-Q4_K_XL -dev CUDA0 -ngl 99 -fa on -c 131072 -ctk f16 -ctv f16 -ub 1024 --parallel 1 --no-mmproj --host 127.0.0.1 --port 8080
```

**flint UD-Q3_K_XL**

```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 LLAMA_MTP_DEFER_CATCHUP=1 build-cuda/bin/llama-server -hf unsloth/flint-GGUF:UD-Q3_K_XL -dev CUDA0 -ngl 99 -fa on -c 262144 -ctk iq4_nl -ctv iq4_nl -ub 2048 --parallel 1 --no-mmproj --host 127.0.0.1 --port 8080
```

**flint UD-IQ3_XXS**

```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 LLAMA_MTP_DEFER_CATCHUP=1 build-cuda/bin/llama-server -hf unsloth/flint-GGUF:UD-IQ3_XXS -dev CUDA0 -ngl 99 -fa on -c 262144 -ctk q8_0 -ctv q8_0 -ub 2048 --parallel 1 --no-mmproj --host 127.0.0.1 --port 8080
```

**gpt-oss-20b**

```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 LLAMA_MTP_DEFER_CATCHUP=1 build-cuda/bin/llama-server -hf unsloth/gpt-oss-20b-GGUF:UD-Q4_K_XL -dev CUDA0 -ngl 99 -fa on -c 131072 -ctk f16 -ctv f16 -ub 2048 --parallel 1 --no-mmproj --host 127.0.0.1 --port 8080
```

**GLM-4.7-Flash-REAP-23B-A3B**

```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 LLAMA_MTP_DEFER_CATCHUP=1 build-cuda/bin/llama-server -hf unsloth/GLM-4.7-Flash-REAP-23B-A3B-GGUF:UD-Q4_K_XL -dev CUDA0 -ngl 99 -fa on -c 202752 -ctk q8_0 -ctv q8_0 -ub 2048 --parallel 1 --no-mmproj --host 127.0.0.1 --port 8080
```

**Qwen3-14B**

```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 LLAMA_MTP_DEFER_CATCHUP=1 build-cuda/bin/llama-server -hf unsloth/Qwen3-14B-GGUF:Q4_K_M -dev CUDA0 -ngl 99 -fa on -c 40960 -ctk f16 -ctv f16 -ub 2048 --parallel 1 --no-mmproj --host 127.0.0.1 --port 8080
```

**Qwen3.8-27B**

```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 LLAMA_MTP_DEFER_CATCHUP=1 build-cuda/bin/llama-server -hf unsloth/Qwen3.8-27B-GGUF:UD-IQ4_XS -dev CUDA0 -ngl 99 -fa on -c 262144 -ctk iq4_nl -ctv iq4_nl -ub 2048 --parallel 1 --no-mmproj --host 127.0.0.1 --port 8080
```


**Laguna-XS-2.1 (bartowski IQ4_NL)**

```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 LLAMA_MTP_DEFER_CATCHUP=1 build-cuda/bin/llama-server -hf bartowski/Laguna-XS-2.1-GGUF:IQ4_NL -dev CUDA0 -ngl 99 -fa on -c 200448 -ctk iq4_nl -ctv iq4_nl -ub 2048 --parallel 1 --no-mmproj --host 127.0.0.1 --port 8080
```

**LFM2.5-24B-A2B**

```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 LLAMA_MTP_DEFER_CATCHUP=1 build-cuda/bin/llama-server -hf unsloth/LFM2.5-24B-A2B-GGUF:UD-Q4_K_XL -dev CUDA0 -ngl 99 -fa on -c 131072 -ctk f16 -ctv f16 -ub 2048 --parallel 1 --no-mmproj --host 127.0.0.1 --port 8080
```

**Muse-Glimmer-30B (DFlash draft on CUDA only)**

```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 LLAMA_MTP_DEFER_CATCHUP=1 build-cuda/bin/llama-server -hf unsloth/Muse-Glimmer-30B-GGUF:UD-Q4_K_XL -dev CUDA0 -ngl 99 -fa on -c 131072 -ctk f16 -ctv f16 -ub 2048 --parallel 1 --no-mmproj --host 127.0.0.1 --port 8080 -md ~/.cache/huggingface/hub/models--unsloth--Muse-Glimmer-30B-GGUF/snapshots/faa5b025c584459c13febfa5c59883516710ae39/dflash-kquant.gguf --spec-type draft-dflash -ctkd q8_0 -ctvd q8_0
```

**Devstral-Small-2-24B**

```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 LLAMA_MTP_DEFER_CATCHUP=1 build-cuda/bin/llama-server -hf unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF:UD-Q4_K_XL -dev CUDA0 -ngl 99 -fa on -c 151552 -ctk iq4_nl -ctv iq4_nl -ub 2048 --parallel 1 --no-mmproj --host 127.0.0.1 --port 8080
```

**Magistral-Small-2509 (24B)**

```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 LLAMA_MTP_DEFER_CATCHUP=1 build-cuda/bin/llama-server -hf unsloth/Magistral-Small-2509-GGUF:UD-Q4_K_XL -dev CUDA0 -ngl 99 -fa on -c 131072 -ctk iq4_nl -ctv iq4_nl -ub 2048 --parallel 1 --no-mmproj --host 127.0.0.1 --port 8080
```

### Commands: Vulkan (`-dev Vulkan1`)

**Qwen3.5-9B (MTP)**

```
LLAMA_MTP_DEFER_CATCHUP=1 build-vk/bin/llama-server -hf unsloth/Qwen3.5-9B-MTP-GGUF:UD-Q4_K_XL -dev Vulkan1 -ngl 99 -fa on -c 262144 -ctk f16 -ctv f16 -ub 2048 --parallel 1 --no-mmproj --host 127.0.0.1 --port 8080 --spec-type draft-mtp --spec-draft-n-max 2 -ctkd q8_0 -ctvd q8_0
```

**Ornith-1.5-9B (MTP)**

```
LLAMA_MTP_DEFER_CATCHUP=1 build-vk/bin/llama-server -hf ornith-ai/Ornith-1.5-9B-GGUF:Q4_K_M -dev Vulkan1 -ngl 99 -fa on -c 262144 -ctk f16 -ctv f16 -ub 2048 --parallel 1 --no-mmproj --host 127.0.0.1 --port 8080 --spec-type draft-mtp --spec-draft-n-max 1 -ctkd q8_0 -ctvd q8_0
```

**Phi-4-reasoning-plus**

```
LLAMA_MTP_DEFER_CATCHUP=1 build-vk/bin/llama-server -hf unsloth/Phi-4-reasoning-plus-GGUF:UD-Q4_K_XL -dev Vulkan1 -ngl 99 -fa on -c 32768 -ctk f16 -ctv f16 -ub 2048 --parallel 1 --no-mmproj --host 127.0.0.1 --port 8080
```

**gemma-4-12b (MTP)**

```
LLAMA_MTP_DEFER_CATCHUP=1 build-vk/bin/llama-server -hf unsloth/gemma-4-12b-it-GGUF:UD-Q4_K_XL -dev Vulkan1 -ngl 99 -fa on -c 262144 -ctk f16 -ctv f16 -ub 2048 --parallel 1 --no-mmproj --host 127.0.0.1 --port 8080 -md ~/.cache/huggingface/hub/models--unsloth--gemma-4-12b-it-GGUF/snapshots/fc034cfff751157913579611efad8462ac1be606/MTP/mtp-gemma-4-12b-it-Q8_0.gguf --spec-type draft-mtp -ctkd q8_0 -ctvd q8_0 --dry-multiplier 0.8
```

**gemma-4-26B-A4B (MTP)**

```
LLAMA_MTP_DEFER_CATCHUP=1 build-vk/bin/llama-server -hf unsloth/gemma-4-26B-A4B-it-GGUF:UD-IQ4_NL -dev Vulkan1 -ngl 99 -fa on -c 262144 -ctk f16 -ctv f16 -ub 2048 --parallel 1 --no-mmproj --host 127.0.0.1 --port 8080 -md ~/.cache/huggingface/hub/models--unsloth--gemma-4-26B-A4B-it-GGUF/snapshots/c099eb48e663fd284577b04978a94ffccb261841/MTP/mtp-gemma-4-26B-A4B-it-Q8_0.gguf --spec-type draft-mtp -ctkd q8_0 -ctvd q8_0 --dry-multiplier 0.8
```

**Nemotron-3.5-Lightning-30B-A3B (MTP)**

```
LLAMA_MTP_DEFER_CATCHUP=1 build-vk/bin/llama-server -hf unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF:UD-IQ3_XXS -dev Vulkan1 -ngl 99 -fa on -c 1048576 -ctk q8_0 -ctv q8_0 -ub 2048 --parallel 1 --no-mmproj --host 127.0.0.1 --port 8080 --spec-type draft-mtp --spec-draft-n-max 1 -ctkd q8_0 -ctvd q8_0
```

**Nemotron-3.5-Bolt-E4B**

```
LLAMA_MTP_DEFER_CATCHUP=1 build-vk/bin/llama-server -hf unsloth/NVIDIA-Nemotron-3.5-Bolt-E4B-GGUF:UD-Q4_K_XL -dev Vulkan1 -ngl 99 -fa on -c 131072 -ctk f16 -ctv f16 -ub 1024 --parallel 1 --no-mmproj --host 127.0.0.1 --port 8080
```

**flint UD-Q3_K_XL**

```
LLAMA_MTP_DEFER_CATCHUP=1 build-vk/bin/llama-server -hf unsloth/flint-GGUF:UD-Q3_K_XL -dev Vulkan1 -ngl 99 -fa on -c 262144 -ctk iq4_nl -ctv iq4_nl -ub 2048 --parallel 1 --no-mmproj --host 127.0.0.1 --port 8080
```

**flint UD-IQ3_XXS**

```
LLAMA_MTP_DEFER_CATCHUP=1 build-vk/bin/llama-server -hf unsloth/flint-GGUF:UD-IQ3_XXS -dev Vulkan1 -ngl 99 -fa on -c 262144 -ctk q8_0 -ctv q8_0 -ub 2048 --parallel 1 --no-mmproj --host 127.0.0.1 --port 8080
```

**gpt-oss-20b**

```
LLAMA_MTP_DEFER_CATCHUP=1 build-vk/bin/llama-server -hf unsloth/gpt-oss-20b-GGUF:UD-Q4_K_XL -dev Vulkan1 -ngl 99 -fa on -c 131072 -ctk f16 -ctv f16 -ub 2048 --parallel 1 --no-mmproj --host 127.0.0.1 --port 8080
```

**GLM-4.7-Flash-REAP-23B-A3B**

```
LLAMA_MTP_DEFER_CATCHUP=1 build-vk/bin/llama-server -hf unsloth/GLM-4.7-Flash-REAP-23B-A3B-GGUF:UD-Q4_K_XL -dev Vulkan1 -ngl 99 -fa on -c 202752 -ctk q8_0 -ctv q8_0 -ub 2048 --parallel 1 --no-mmproj --host 127.0.0.1 --port 8080
```

**Qwen3-14B**

```
LLAMA_MTP_DEFER_CATCHUP=1 build-vk/bin/llama-server -hf unsloth/Qwen3-14B-GGUF:Q4_K_M -dev Vulkan1 -ngl 99 -fa on -c 40960 -ctk f16 -ctv f16 -ub 2048 --parallel 1 --no-mmproj --host 127.0.0.1 --port 8080
```

**Qwen3.8-27B**

```
LLAMA_MTP_DEFER_CATCHUP=1 build-vk/bin/llama-server -hf unsloth/Qwen3.8-27B-GGUF:UD-IQ4_XS -dev Vulkan1 -ngl 99 -fa on -c 262144 -ctk iq4_nl -ctv iq4_nl -ub 2048 --parallel 1 --no-mmproj --host 127.0.0.1 --port 8080
```


**Laguna-XS-2.1 (bartowski IQ4_NL)**

```
LLAMA_MTP_DEFER_CATCHUP=1 build-vk/bin/llama-server -hf bartowski/Laguna-XS-2.1-GGUF:IQ4_NL -dev Vulkan1 -ngl 99 -fa on -c 245248 -ctk iq4_nl -ctv iq4_nl -ub 2048 --parallel 1 --no-mmproj --host 127.0.0.1 --port 8080
```

**LFM2.5-24B-A2B**

```
LLAMA_MTP_DEFER_CATCHUP=1 build-vk/bin/llama-server -hf unsloth/LFM2.5-24B-A2B-GGUF:UD-Q4_K_XL -dev Vulkan1 -ngl 99 -fa on -c 131072 -ctk f16 -ctv f16 -ub 2048 --parallel 1 --no-mmproj --host 127.0.0.1 --port 8080
```

**Muse-Glimmer-30B (DFlash draft on CUDA only)**

```
LLAMA_MTP_DEFER_CATCHUP=1 build-vk/bin/llama-server -hf unsloth/Muse-Glimmer-30B-GGUF:UD-Q4_K_XL -dev Vulkan1 -ngl 99 -fa on -c 131072 -ctk f16 -ctv f16 -ub 2048 --parallel 1 --no-mmproj --host 127.0.0.1 --port 8080
```

**Devstral-Small-2-24B**

```
LLAMA_MTP_DEFER_CATCHUP=1 build-vk/bin/llama-server -hf unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF:UD-Q4_K_XL -dev Vulkan1 -ngl 99 -fa on -c 150528 -ctk iq4_nl -ctv iq4_nl -ub 2048 --parallel 1 --no-mmproj --host 127.0.0.1 --port 8080
```

**Magistral-Small-2509 (24B)**

```
LLAMA_MTP_DEFER_CATCHUP=1 build-vk/bin/llama-server -hf unsloth/Magistral-Small-2509-GGUF:UD-Q4_K_XL -dev Vulkan1 -ngl 99 -fa on -c 131072 -ctk iq4_nl -ctv iq4_nl -ub 2048 --parallel 1 --no-mmproj --host 127.0.0.1 --port 8080
```

Notes for the second group (Laguna, LFM2.5, Muse-Glimmer, Devstral, Magistral): none of them has an MTP draft. Only Muse-Glimmer has a DFlash draft (CUDA only).
Laguna and Devstral do not reach the model maximum even with iq4_nl KV, so `-c` is the largest value `llama-fit-params` accepted (CUDA 200448 / 151552, Vulkan 245248 / 150528).
The 5 new models are in `ctxfit-new.txt`, `bestsweep-new1.txt`, `bestsweep-new2.txt`, `specrun-new.txt` (in `~/llama-bench-results`).

Notes (RTX 3090, verified by starting every command above, see `verify-cmds.txt`; each one answered a request):
- Nemotron Lightning on CUDA: `llama-fit-params` allows `-c 622336` with iq4_nl KV, but with the MTP draft the compute buffers no longer fit
  (cudaMalloc out of memory at 622336, 589824, 524288 and 458752); 393216 loads (23.5 GB). Without the draft (remove the three `--spec-type`/`-ctkd`/`-ctvd` options) use `-c 622336`.
  On Vulkan the full 1048576 fits with q8_0 KV and the draft.
- Quantizing the draft KV (`-ctkd iq4_nl`) did not change that limit, so the commands keep `q8_0` for the draft.
- The tokens/s printed by the verification (16-token request on a cold server) are not benchmark numbers; use the speed table.

## RX 6900 XT (16 GB, gfx1030)

HIP build: `build-rocm`, device `-dev ROCm0`. Fit check: `llama-fit-params` with `-ub 256 -fitt 512` and the KV ladder (2026-11-15). The compute buffer of `-ub 2048` is several GiB too much on 16 GB for the bigger models, so those use `-ub 128`/`256` (slower prefill, same decode). Rule 5 does not apply to them.
Left out because they still need CPU offload (4096 tokens or less on the GPU): Nemotron-3.5-Lightning-30B, Laguna-XS-2.1, flint UD-Q3_K_XL.
MTP models use `-c 100000` on ROCm (the draft context does not fit larger). Each line was started, answered one chat request, and the VRAM below was read after it. tg is the short-prompt server speed.
Decode at 100k context (llama-bench, depth 0 -> 100000, t/s): Qwen3.5-9B 62.8 -> 50.3, gemma-4-12b f16 48.1 -> 37.5, gpt-oss-20b q8_0 115 -> 64.8 (see `rx6900xt-results.md`).

**Qwen3.5-9B (MTP)** (10.8 GiB, short-prompt tg 100 with MTP)

```
LLAMA_MTP_DEFER_CATCHUP=1 build-rocm/bin/llama-server -hf unsloth/Qwen3.5-9B-MTP-GGUF:UD-Q4_K_XL -dev ROCm0 -ngl 99 -fa on -c 100000 -ctk f16 -ctv f16 -ub 2048 --parallel 1 --no-mmproj --host 127.0.0.1 --port 8080 --spec-type draft-mtp --spec-draft-n-max 2 -ctkd q8_0 -ctvd q8_0
```

**Ornith-1.5-9B (MTP)** (10.3 GiB, tg 94)

```
LLAMA_MTP_DEFER_CATCHUP=1 build-rocm/bin/llama-server -hf ornith-ai/Ornith-1.5-9B-GGUF:Q4_K_M -dev ROCm0 -ngl 99 -fa on -c 100000 -ctk f16 -ctv f16 -ub 2048 --parallel 1 --no-mmproj --host 127.0.0.1 --port 8080 --spec-type draft-mtp --spec-draft-n-max 1 -ctkd q8_0 -ctvd q8_0
```

**Phi-4-reasoning-plus** (12.7 GiB, tg 44; f16 KV reaches only 24832 tokens, q8_0 the full 32768)

```
LLAMA_MTP_DEFER_CATCHUP=1 build-rocm/bin/llama-server -hf unsloth/Phi-4-reasoning-plus-GGUF:UD-Q4_K_XL -dev ROCm0 -ngl 99 -fa on -c 32768 -ctk q8_0 -ctv q8_0 -ub 2048 --parallel 1 --no-mmproj --host 127.0.0.1 --port 8080
```

**gemma-4-12b (MTP)** (14.2 GiB, tg 88; f16 KV (q8_0/iq4_nl would convert the hd512 cache to f16 every step: 28.5 vs 37.5 t/s at 100k), f16 reaches 204544 tokens, use q8_0 -c 262144 only if you need more)

```
LLAMA_MTP_DEFER_CATCHUP=1 build-rocm/bin/llama-server -hf unsloth/gemma-4-12b-it-GGUF:UD-Q4_K_XL -dev ROCm0 -ngl 99 -fa on -c 200000 -ctk f16 -ctv f16 -ub 2048 --parallel 1 --no-mmproj --host 127.0.0.1 --port 8080 -md ~/.cache/huggingface/hub/models--unsloth--gemma-4-12b-it-GGUF/snapshots/fc034cfff751157913579611efad8462ac1be606/MTP/mtp-gemma-4-12b-it-Q8_0.gguf --spec-type draft-mtp -ctkd q8_0 -ctvd q8_0 --dry-multiplier 0.8
```

**Nemotron-3.5-Bolt-E4B** (6.2 GiB, tg 71)

```
LLAMA_MTP_DEFER_CATCHUP=1 build-rocm/bin/llama-server -hf unsloth/NVIDIA-Nemotron-3.5-Bolt-E4B-GGUF:UD-Q4_K_XL -dev ROCm0 -ngl 99 -fa on -c 131072 -ctk f16 -ctv f16 -ub 1024 --parallel 1 --no-mmproj --host 127.0.0.1 --port 8080
```

**gpt-oss-20b** (13.2 GiB, tg 110; q8_0 KV (fits 105216 tokens; iq4_nl reaches 131072 but its hd64 decode kernel is 50 % slower at 100k))

```
LLAMA_MTP_DEFER_CATCHUP=1 build-rocm/bin/llama-server -hf unsloth/gpt-oss-20b-GGUF:UD-Q4_K_XL -dev ROCm0 -ngl 99 -fa on -c 105216 -ctk q8_0 -ctv q8_0 -ub 2048 --parallel 1 --no-mmproj --host 127.0.0.1 --port 8080
```

**Qwen3-14B** (12.6 GiB, tg 44; f16 KV reaches only 28928 tokens, q8_0 the full 40960)

```
LLAMA_MTP_DEFER_CATCHUP=1 build-rocm/bin/llama-server -hf unsloth/Qwen3-14B-GGUF:Q4_K_M -dev ROCm0 -ngl 99 -fa on -c 40960 -ctk q8_0 -ctv q8_0 -ub 2048 --parallel 1 --no-mmproj --host 127.0.0.1 --port 8080
```

**Qwen3.8-27B** (15.7 GiB, tg 25.3, pp 600-630. No draft. `-ub 256` is what lets the model plus 128k of iq4_nl KV fit (`-ub 2048` leaves only 4k); the same command gives 21.7 t/s at 100k depth)

```
LLAMA_MTP_DEFER_CATCHUP=1 build-rocm/bin/llama-server -hf unsloth/Qwen3.8-27B-GGUF:UD-IQ4_XS -dev ROCm0 -ngl 99 -fa on -c 128000 -ctk iq4_nl -ctv iq4_nl -ub 256 -b 256 --parallel 1 --no-mmproj --host 127.0.0.1 --port 8080
```

**gemma-4-26B-A4B (MTP, IQ4_NL draft)** (16.0 GiB, tg 126-151 with MTP, acceptance 62-81 %; f16 KV, `-c 100000`. The draft is re-quantized to IQ4_NL, see below; with the original Q8_0 draft only `-c 90000` is comfortable (100000 fits at 16.2 GiB, 109568 only without a draft))

```
LLAMA_MTP_DEFER_CATCHUP=1 build-rocm/bin/llama-server -hf unsloth/gemma-4-26B-A4B-it-GGUF:UD-IQ4_NL -dev ROCm0 -ngl 99 -fa on -c 100000 -ctk f16 -ctv f16 -ub 256 --parallel 1 --no-mmproj --host 127.0.0.1 --port 8080 -md ~/.cache/huggingface/hub/models--unsloth--gemma-4-26B-A4B-it-GGUF/snapshots/c099eb48e663fd284577b04978a94ffccb261841/MTP/mtp-gemma-4-26B-A4B-it-IQ4_NL.gguf --spec-type draft-mtp -ctkd q8_0 -ctvd q8_0 --dry-multiplier 0.8
```

The IQ4_NL draft is not downloadable; build it once from the Q8_0 draft that comes with the model (CPU only, a few seconds, no imatrix needed):

```
SNAP=~/.cache/huggingface/hub/models--unsloth--gemma-4-26B-A4B-it-GGUF/snapshots/c099eb48e663fd284577b04978a94ffccb261841/MTP
build-rocm/bin/llama-quantize --allow-requantize $SNAP/mtp-gemma-4-26B-A4B-it-Q8_0.gguf $SNAP/mtp-gemma-4-26B-A4B-it-IQ4_NL.gguf IQ4_NL
```

`--allow-requantize` is needed because the source is already quantized (Q8_0, nearly lossless). The tied token embedding / output head (262144 x 1024) is kept at q6_K by the quantizer, so 425 MiB becomes 291 MiB, not 220. A/B on this card (3 prompts, 400 tokens, temperature 0, same acceptance within noise: 260/416 vs 264/404 accepted drafts): tg 128.6 / 151.2 / 131.2 against 126.3 / 145.0 / 125.9 with the Q8_0 draft, VRAM -137 MiB. It only helps where VRAM is the limit; the Qwen3.8 draft is already Q4_0 (same size as IQ4_NL) and the DFlash draft of Muse is 1.5 GiB either way.

**flint UD-IQ3_XXS** (16.1 GiB, tg 66; iq4_nl KV, `-ub 128` reaches the full 262144 (`-ub 256` 230656, q8_0 124416))

```
LLAMA_MTP_DEFER_CATCHUP=1 build-rocm/bin/llama-server -hf unsloth/flint-GGUF:UD-IQ3_XXS -dev ROCm0 -ngl 99 -fa on -c 262144 -ctk iq4_nl -ctv iq4_nl -ub 128 --parallel 1 --no-mmproj --host 127.0.0.1 --port 8080
```

**GLM-4.7-Flash-REAP-23B-A3B** (16.3 GiB, tg 64; iq4_nl KV, `-ub 128` 165632 tokens (`-ub 256` 140800, q8_0 75776); model maximum 202752 does not fit)

```
LLAMA_MTP_DEFER_CATCHUP=1 build-rocm/bin/llama-server -hf unsloth/GLM-4.7-Flash-REAP-23B-A3B-GGUF:UD-Q4_K_XL -dev ROCm0 -ngl 99 -fa on -c 165632 -ctk iq4_nl -ctv iq4_nl -ub 128 --parallel 1 --no-mmproj --host 127.0.0.1 --port 8080
```

**LFM2.5-24B-A2B** (15.4 GiB, tg 116; iq4_nl KV (q8_0 reaches 125696))

```
LLAMA_MTP_DEFER_CATCHUP=1 build-rocm/bin/llama-server -hf unsloth/LFM2.5-24B-A2B-GGUF:UD-Q4_K_XL -dev ROCm0 -ngl 99 -fa on -c 131072 -ctk iq4_nl -ctv iq4_nl -ub 256 --parallel 1 --no-mmproj --host 127.0.0.1 --port 8080
```

**Muse-Glimmer-30B** (15.8 GiB, tg 26; q8_0 KV, no draft (the DFlash draft does not fit next to the 131072 context))

```
LLAMA_MTP_DEFER_CATCHUP=1 build-rocm/bin/llama-server -hf unsloth/Muse-Glimmer-30B-GGUF:UD-Q4_K_XL -dev ROCm0 -ngl 99 -fa on -c 131072 -ctk q8_0 -ctv q8_0 -ub 256 --parallel 1 --no-mmproj --host 127.0.0.1 --port 8080
```

**Devstral-Small-2-24B** (16.2 GiB, tg 27.5; iq4_nl KV, `-ub 128`: only 54016 tokens fit (the model maximum is 151552))

```
LLAMA_MTP_DEFER_CATCHUP=1 build-rocm/bin/llama-server -hf unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF:UD-Q4_K_XL -dev ROCm0 -ngl 99 -fa on -c 54016 -ctk iq4_nl -ctv iq4_nl -ub 128 --parallel 1 --no-mmproj --host 127.0.0.1 --port 8080
```

**Magistral-Small-2509 (24B)** (16.2 GiB, tg 28.2; iq4_nl KV, `-ub 128`: only 54016 tokens fit (maximum 131072))

```
LLAMA_MTP_DEFER_CATCHUP=1 build-rocm/bin/llama-server -hf unsloth/Magistral-Small-2509-GGUF:UD-Q4_K_XL -dev ROCm0 -ngl 99 -fa on -c 54016 -ctk iq4_nl -ctv iq4_nl -ub 128 --parallel 1 --no-mmproj --host 127.0.0.1 --port 8080
```

## Other cards

Add one section per card with the same layout (rules above, then speed table and the CUDA / Vulkan / HIP command blocks).
Cards to do: RX 6900 XT (16 GB, HIP and Vulkan; with MTP it needs `-c 100000`, see `rx6900xt-results.md`), RTX 2080 Ti (11 GB), GTX 1080 Ti (11 GB),
Intel Arc Pro B50 (16 GB, Vulkan), T1200 is not a target. For 11-16 GB cards most of the 20-30B MoE models above no longer fit; start from the fit check in rule 1.
