# Best llama-server commands per card and model

Every command is a single line that works with copy and paste (run it from any directory; the binaries are addressed by absolute path).
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
4. `-ub 2048`, except Nemotron Bolt (`-ub 1024`, +12 % prefill on CUDA, +5 % on Vulkan).
5. Verify each line: start it, wait for `/health`, send one request, and record the VRAM used.
6. Use the card's own device selection: CUDA `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<index>` and `-dev CUDA0`; Vulkan `-dev Vulkan<N>`; HIP `-dev ROCm0`.
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




Commands below use the context and KV type from the rules (smallest KV quantization that reaches the model maximum). Prefill and decode are a few percent slower than the f16 numbers above where the KV is quantized; the f16 line is the fastest choice if you need less context.


### Commands: CUDA (`-dev CUDA0`)

**Qwen3.5-9B (MTP)**

```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 LLAMA_MTP_DEFER_CATCHUP=1 /home/zsoltii/IdeaProjects/llama.cpp/build-cuda/bin/llama-server -hf unsloth/Qwen3.5-9B-MTP-GGUF:UD-Q4_K_XL -dev CUDA0 -ngl 99 -fa on -c 262144 -ctk f16 -ctv f16 -ub 2048 --parallel 1 --no-mmproj --host 127.0.0.1 --port 8080 --spec-type draft-mtp --spec-draft-n-max 2 -ctkd q8_0 -ctvd q8_0
```

**Ornith-1.5-9B (MTP)**

```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 LLAMA_MTP_DEFER_CATCHUP=1 /home/zsoltii/IdeaProjects/llama.cpp/build-cuda/bin/llama-server -hf ornith-ai/Ornith-1.5-9B-GGUF:Q4_K_M -dev CUDA0 -ngl 99 -fa on -c 262144 -ctk f16 -ctv f16 -ub 2048 --parallel 1 --no-mmproj --host 127.0.0.1 --port 8080 --spec-type draft-mtp --spec-draft-n-max 1 -ctkd q8_0 -ctvd q8_0
```

**Phi-4-reasoning-plus**

```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 LLAMA_MTP_DEFER_CATCHUP=1 /home/zsoltii/IdeaProjects/llama.cpp/build-cuda/bin/llama-server -hf unsloth/Phi-4-reasoning-plus-GGUF:UD-Q4_K_XL -dev CUDA0 -ngl 99 -fa on -c 32768 -ctk f16 -ctv f16 -ub 2048 --parallel 1 --no-mmproj --host 127.0.0.1 --port 8080
```

**gemma-4-12b (MTP)**

```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 LLAMA_MTP_DEFER_CATCHUP=1 /home/zsoltii/IdeaProjects/llama.cpp/build-cuda/bin/llama-server -hf unsloth/gemma-4-12b-it-GGUF:UD-Q4_K_XL -dev CUDA0 -ngl 99 -fa on -c 262144 -ctk f16 -ctv f16 -ub 2048 --parallel 1 --no-mmproj --host 127.0.0.1 --port 8080 -md /home/zsoltii/.cache/huggingface/hub/models--unsloth--gemma-4-12b-it-GGUF/snapshots/fc034cfff751157913579611efad8462ac1be606/MTP/mtp-gemma-4-12b-it-Q8_0.gguf --spec-type draft-mtp -ctkd q8_0 -ctvd q8_0
```

**gemma-4-26B-A4B (MTP)**

```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 LLAMA_MTP_DEFER_CATCHUP=1 /home/zsoltii/IdeaProjects/llama.cpp/build-cuda/bin/llama-server -hf unsloth/gemma-4-26B-A4B-it-GGUF:UD-IQ4_NL -dev CUDA0 -ngl 99 -fa on -c 262144 -ctk q8_0 -ctv q8_0 -ub 2048 --parallel 1 --no-mmproj --host 127.0.0.1 --port 8080 -md /home/zsoltii/.cache/huggingface/hub/models--unsloth--gemma-4-26B-A4B-it-GGUF/snapshots/c099eb48e663fd284577b04978a94ffccb261841/MTP/mtp-gemma-4-26B-A4B-it-Q8_0.gguf --spec-type draft-mtp -ctkd q8_0 -ctvd q8_0
```

**Nemotron-3.5-Lightning-30B-A3B (MTP)**

```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 LLAMA_MTP_DEFER_CATCHUP=1 /home/zsoltii/IdeaProjects/llama.cpp/build-cuda/bin/llama-server -hf unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF:UD-IQ3_XXS -dev CUDA0 -ngl 99 -fa on -c 393216 -ctk iq4_nl -ctv iq4_nl -ub 2048 --parallel 1 --no-mmproj --host 127.0.0.1 --port 8080 --spec-type draft-mtp --spec-draft-n-max 1 -ctkd q8_0 -ctvd q8_0
```

**Nemotron-3.5-Bolt-E4B**

```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 LLAMA_MTP_DEFER_CATCHUP=1 /home/zsoltii/IdeaProjects/llama.cpp/build-cuda/bin/llama-server -hf unsloth/NVIDIA-Nemotron-3.5-Bolt-E4B-GGUF:UD-Q4_K_XL -dev CUDA0 -ngl 99 -fa on -c 131072 -ctk f16 -ctv f16 -ub 1024 --parallel 1 --no-mmproj --host 127.0.0.1 --port 8080
```

**flint UD-Q3_K_XL**

```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 LLAMA_MTP_DEFER_CATCHUP=1 /home/zsoltii/IdeaProjects/llama.cpp/build-cuda/bin/llama-server -hf unsloth/flint-GGUF:UD-Q3_K_XL -dev CUDA0 -ngl 99 -fa on -c 262144 -ctk iq4_nl -ctv iq4_nl -ub 2048 --parallel 1 --no-mmproj --host 127.0.0.1 --port 8080
```

**flint UD-IQ3_XXS**

```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 LLAMA_MTP_DEFER_CATCHUP=1 /home/zsoltii/IdeaProjects/llama.cpp/build-cuda/bin/llama-server -hf unsloth/flint-GGUF:UD-IQ3_XXS -dev CUDA0 -ngl 99 -fa on -c 262144 -ctk q8_0 -ctv q8_0 -ub 2048 --parallel 1 --no-mmproj --host 127.0.0.1 --port 8080
```

**gpt-oss-20b**

```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 LLAMA_MTP_DEFER_CATCHUP=1 /home/zsoltii/IdeaProjects/llama.cpp/build-cuda/bin/llama-server -hf unsloth/gpt-oss-20b-GGUF:UD-Q4_K_XL -dev CUDA0 -ngl 99 -fa on -c 131072 -ctk f16 -ctv f16 -ub 2048 --parallel 1 --no-mmproj --host 127.0.0.1 --port 8080
```

**GLM-4.7-Flash-REAP-23B-A3B**

```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 LLAMA_MTP_DEFER_CATCHUP=1 /home/zsoltii/IdeaProjects/llama.cpp/build-cuda/bin/llama-server -hf unsloth/GLM-4.7-Flash-REAP-23B-A3B-GGUF:UD-Q4_K_XL -dev CUDA0 -ngl 99 -fa on -c 202752 -ctk q8_0 -ctv q8_0 -ub 2048 --parallel 1 --no-mmproj --host 127.0.0.1 --port 8080
```

**Qwen3-14B**

```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 LLAMA_MTP_DEFER_CATCHUP=1 /home/zsoltii/IdeaProjects/llama.cpp/build-cuda/bin/llama-server -hf unsloth/Qwen3-14B-GGUF:Q4_K_M -dev CUDA0 -ngl 99 -fa on -c 40960 -ctk f16 -ctv f16 -ub 2048 --parallel 1 --no-mmproj --host 127.0.0.1 --port 8080
```

**Qwen3.8-27B**

```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 LLAMA_MTP_DEFER_CATCHUP=1 /home/zsoltii/IdeaProjects/llama.cpp/build-cuda/bin/llama-server -hf unsloth/Qwen3.8-27B-GGUF:UD-IQ4_XS -dev CUDA0 -ngl 99 -fa on -c 262144 -ctk iq4_nl -ctv iq4_nl -ub 2048 --parallel 1 --no-mmproj --host 127.0.0.1 --port 8080
```


### Commands: Vulkan (`-dev Vulkan1`)

**Qwen3.5-9B (MTP)**

```
LLAMA_MTP_DEFER_CATCHUP=1 /home/zsoltii/IdeaProjects/llama.cpp/build-vk/bin/llama-server -hf unsloth/Qwen3.5-9B-MTP-GGUF:UD-Q4_K_XL -dev Vulkan1 -ngl 99 -fa on -c 262144 -ctk f16 -ctv f16 -ub 2048 --parallel 1 --no-mmproj --host 127.0.0.1 --port 8080 --spec-type draft-mtp --spec-draft-n-max 2 -ctkd q8_0 -ctvd q8_0
```

**Ornith-1.5-9B (MTP)**

```
LLAMA_MTP_DEFER_CATCHUP=1 /home/zsoltii/IdeaProjects/llama.cpp/build-vk/bin/llama-server -hf ornith-ai/Ornith-1.5-9B-GGUF:Q4_K_M -dev Vulkan1 -ngl 99 -fa on -c 262144 -ctk f16 -ctv f16 -ub 2048 --parallel 1 --no-mmproj --host 127.0.0.1 --port 8080 --spec-type draft-mtp --spec-draft-n-max 1 -ctkd q8_0 -ctvd q8_0
```

**Phi-4-reasoning-plus**

```
LLAMA_MTP_DEFER_CATCHUP=1 /home/zsoltii/IdeaProjects/llama.cpp/build-vk/bin/llama-server -hf unsloth/Phi-4-reasoning-plus-GGUF:UD-Q4_K_XL -dev Vulkan1 -ngl 99 -fa on -c 32768 -ctk f16 -ctv f16 -ub 2048 --parallel 1 --no-mmproj --host 127.0.0.1 --port 8080
```

**gemma-4-12b (MTP)**

```
LLAMA_MTP_DEFER_CATCHUP=1 /home/zsoltii/IdeaProjects/llama.cpp/build-vk/bin/llama-server -hf unsloth/gemma-4-12b-it-GGUF:UD-Q4_K_XL -dev Vulkan1 -ngl 99 -fa on -c 262144 -ctk f16 -ctv f16 -ub 2048 --parallel 1 --no-mmproj --host 127.0.0.1 --port 8080 -md /home/zsoltii/.cache/huggingface/hub/models--unsloth--gemma-4-12b-it-GGUF/snapshots/fc034cfff751157913579611efad8462ac1be606/MTP/mtp-gemma-4-12b-it-Q8_0.gguf --spec-type draft-mtp -ctkd q8_0 -ctvd q8_0
```

**gemma-4-26B-A4B (MTP)**

```
LLAMA_MTP_DEFER_CATCHUP=1 /home/zsoltii/IdeaProjects/llama.cpp/build-vk/bin/llama-server -hf unsloth/gemma-4-26B-A4B-it-GGUF:UD-IQ4_NL -dev Vulkan1 -ngl 99 -fa on -c 262144 -ctk f16 -ctv f16 -ub 2048 --parallel 1 --no-mmproj --host 127.0.0.1 --port 8080 -md /home/zsoltii/.cache/huggingface/hub/models--unsloth--gemma-4-26B-A4B-it-GGUF/snapshots/c099eb48e663fd284577b04978a94ffccb261841/MTP/mtp-gemma-4-26B-A4B-it-Q8_0.gguf --spec-type draft-mtp -ctkd q8_0 -ctvd q8_0
```

**Nemotron-3.5-Lightning-30B-A3B (MTP)**

```
LLAMA_MTP_DEFER_CATCHUP=1 /home/zsoltii/IdeaProjects/llama.cpp/build-vk/bin/llama-server -hf unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF:UD-IQ3_XXS -dev Vulkan1 -ngl 99 -fa on -c 1048576 -ctk q8_0 -ctv q8_0 -ub 2048 --parallel 1 --no-mmproj --host 127.0.0.1 --port 8080 --spec-type draft-mtp --spec-draft-n-max 1 -ctkd q8_0 -ctvd q8_0
```

**Nemotron-3.5-Bolt-E4B**

```
LLAMA_MTP_DEFER_CATCHUP=1 /home/zsoltii/IdeaProjects/llama.cpp/build-vk/bin/llama-server -hf unsloth/NVIDIA-Nemotron-3.5-Bolt-E4B-GGUF:UD-Q4_K_XL -dev Vulkan1 -ngl 99 -fa on -c 131072 -ctk f16 -ctv f16 -ub 1024 --parallel 1 --no-mmproj --host 127.0.0.1 --port 8080
```

**flint UD-Q3_K_XL**

```
LLAMA_MTP_DEFER_CATCHUP=1 /home/zsoltii/IdeaProjects/llama.cpp/build-vk/bin/llama-server -hf unsloth/flint-GGUF:UD-Q3_K_XL -dev Vulkan1 -ngl 99 -fa on -c 262144 -ctk iq4_nl -ctv iq4_nl -ub 2048 --parallel 1 --no-mmproj --host 127.0.0.1 --port 8080
```

**flint UD-IQ3_XXS**

```
LLAMA_MTP_DEFER_CATCHUP=1 /home/zsoltii/IdeaProjects/llama.cpp/build-vk/bin/llama-server -hf unsloth/flint-GGUF:UD-IQ3_XXS -dev Vulkan1 -ngl 99 -fa on -c 262144 -ctk q8_0 -ctv q8_0 -ub 2048 --parallel 1 --no-mmproj --host 127.0.0.1 --port 8080
```

**gpt-oss-20b**

```
LLAMA_MTP_DEFER_CATCHUP=1 /home/zsoltii/IdeaProjects/llama.cpp/build-vk/bin/llama-server -hf unsloth/gpt-oss-20b-GGUF:UD-Q4_K_XL -dev Vulkan1 -ngl 99 -fa on -c 131072 -ctk f16 -ctv f16 -ub 2048 --parallel 1 --no-mmproj --host 127.0.0.1 --port 8080
```

**GLM-4.7-Flash-REAP-23B-A3B**

```
LLAMA_MTP_DEFER_CATCHUP=1 /home/zsoltii/IdeaProjects/llama.cpp/build-vk/bin/llama-server -hf unsloth/GLM-4.7-Flash-REAP-23B-A3B-GGUF:UD-Q4_K_XL -dev Vulkan1 -ngl 99 -fa on -c 202752 -ctk q8_0 -ctv q8_0 -ub 2048 --parallel 1 --no-mmproj --host 127.0.0.1 --port 8080
```

**Qwen3-14B**

```
LLAMA_MTP_DEFER_CATCHUP=1 /home/zsoltii/IdeaProjects/llama.cpp/build-vk/bin/llama-server -hf unsloth/Qwen3-14B-GGUF:Q4_K_M -dev Vulkan1 -ngl 99 -fa on -c 40960 -ctk f16 -ctv f16 -ub 2048 --parallel 1 --no-mmproj --host 127.0.0.1 --port 8080
```

**Qwen3.8-27B**

```
LLAMA_MTP_DEFER_CATCHUP=1 /home/zsoltii/IdeaProjects/llama.cpp/build-vk/bin/llama-server -hf unsloth/Qwen3.8-27B-GGUF:UD-IQ4_XS -dev Vulkan1 -ngl 99 -fa on -c 262144 -ctk iq4_nl -ctv iq4_nl -ub 2048 --parallel 1 --no-mmproj --host 127.0.0.1 --port 8080
```


Notes (RTX 3090, verified by starting every command above, see `verify-cmds.txt`; each one answered a request):
- Nemotron Lightning on CUDA: `llama-fit-params` allows `-c 622336` with iq4_nl KV, but with the MTP draft the compute buffers no longer fit
  (cudaMalloc out of memory at 622336, 589824, 524288 and 458752); 393216 loads (23.5 GB). Without the draft (remove the three `--spec-type`/`-ctkd`/`-ctvd` options) use `-c 622336`.
  On Vulkan the full 1048576 fits with q8_0 KV and the draft.
- Quantizing the draft KV (`-ctkd iq4_nl`) did not change that limit, so the commands keep `q8_0` for the draft.
- The tokens/s printed by the verification (16-token request on a cold server) are not benchmark numbers; use the speed table.

## Other cards

Add one section per card with the same layout (rules above, then speed table and the CUDA / Vulkan / HIP command blocks).
Cards to do: RX 6900 XT (16 GB, HIP and Vulkan; with MTP it needs `-c 100000`, see `rx6900xt-results.md`), RTX 2080 Ti (11 GB), GTX 1080 Ti (11 GB),
Intel Arc Pro B50 (16 GB, Vulkan), T1200 is not a target. For 11-16 GB cards most of the 20-30B MoE models above no longer fit; start from the fit check in rule 1.
