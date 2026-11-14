# Best llama-server settings per model (RTX 3090, CUDA and Vulkan)

Measured 2026-11-14 on the RTX 3090 (`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1`, `-dev CUDA0`; Vulkan `-dev Vulkan1`),
HEAD fc0a84e32. Sources: `llama-bench` sweep (pp2048 / tg128, and the same at 16k depth, `-fa 1`, all layers on GPU) and server
runs with a 20.7k-token prompt + 800 generated tokens (`specrun.sh`). All numbers are tokens/s.

## Common rules

- Always `-fa on -ngl 99`. Pick the device with `-dev CUDA0` (set the two environment variables above) or `-dev Vulkan1`.
- **KV type: `f16` is fastest** for every model that has room for it (24 GB). It beats q8_0 by 0-5 % on decode and 2-10 % on
  prefill at 16k. Use `-ctk q8_0 -ctv q8_0` only when the context does not fit in f16 (about 2x more context), and `iq4_nl` only
  when q8_0 does not fit either (4x more context than f16; on Vulkan it costs 5-15 % decode, on CUDA 1-4 %).
  Exceptions: Qwen3-14B and Phi-4 on CUDA decode a little faster at 16k with a quantized KV (68 vs 66, 67 vs 64).
- `-ub 2048` (the batch default of `-b 2048`) is best on everything except Nemotron Bolt on CUDA (`-ub 1024`, +12 % prefill).
  Smaller `-ub` costs 20-30 % prefill on MoE models; do not go below 1024.
- Speculative decoding: set `LLAMA_MTP_DEFER_CATCHUP=1`; give the draft model a q8_0 KV (`-ctkd q8_0 -ctvd q8_0`).
- On HIP/ROCm quantized MMA flash attention is not used; see `rx6900xt-results.md` for that card.
- Template for every command below:

```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 LLAMA_MTP_DEFER_CATCHUP=1 \
  llama-server -m MODEL.gguf -dev CUDA0 -ngl 99 -fa on -c CTX -ctk f16 -ctv f16 -ub 2048 --parallel 1 [SPEC ARGS]
# Vulkan: the same without the CUDA variables and with -dev Vulkan1
```

## Per model

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

## Context length guidance

- 24 GB leaves room for f16 KV of about 32k tokens for the 9B-14B dense models and much more for hybrid/SWA models
  (Qwen3.5/3.8, gemma-4, Nemotron): there the KV is small and `-c 131072` with f16 fits.
- Beyond that switch to `-ctk q8_0 -ctv q8_0`, then `iq4_nl`. Decode at 64k depth with iq4_nl on CUDA is 37.8 t/s for Qwen3.8-27B
  and 113 t/s for gemma-4-26B (q8_0 KV, MMA path).
- ROCm needs `-c 100000` with MTP on the 16 GB RX 6900 XT (see `rx6900xt-results.md`).

## Validation of this state

- test-backend-ops: CUDA 20335/20335, Vulkan 22421/22421 (FLASH_ATTN_EXT also 5820/5820 on the T1200).
- wikiqa 25/25 + 5/5 on both backends (Qwen3.5 MTP, Qwen3.8 iq4_nl and MTP); attach test 6/6 finished, loop counts unchanged from
  the previous rounds (`val3090-final.txt`).

## Complete commands (model via `-hf`)

Each line is the best configuration from the table above. `-hf` resolves the main model from the Hugging Face cache (or downloads it).
`-hfd` does not resolve the gemma-4 `MTP/` subdirectory files, so those two use `-md` with the cached path (`$HF` below).
`-c` is the model maximum, or, where f16 KV does not fit in 24 GB, the largest value that fits (see the table below); the numbers come from `llama-fit-params -fa on -ctk f16 -ctv f16 -ub 2048 -fitt 2048` (2 GiB left free for compute buffers and a draft model), the smaller of the CUDA and Vulkan results. For Vulkan drop the two CUDA variables and use `-dev Vulkan1`
(Bolt: the only differences are noted in the table; gpt-oss, GLM, Phi-4, Qwen3-14B, flint, Qwen3.8 are identical on both backends except `-dev`).

```
P='CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 LLAMA_MTP_DEFER_CATCHUP=1'
C='-dev CUDA0 -ngl 99 -fa on -ctk f16 -ctv f16 -ub 2048 --parallel 1 --no-mmproj'
HF=~/.cache/huggingface/hub/models--unsloth--gemma-4-12b-it-GGUF/snapshots/fc034cfff751157913579611efad8462ac1be606

# Qwen3.5-9B (MTP, n-max 2 for short prompts, 1 for long)
env $P llama-server -hf unsloth/Qwen3.5-9B-MTP-GGUF:UD-Q4_K_XL -c 262144 $C --spec-type draft-mtp --spec-draft-n-max 2 -ctkd q8_0 -ctvd q8_0

# Ornith-1.5-9B
env $P llama-server -hf ornith-ai/Ornith-1.5-9B-GGUF:Q4_K_M -c 262144 $C --spec-type draft-mtp --spec-draft-n-max 1 -ctkd q8_0 -ctvd q8_0

# Phi-4-reasoning-plus
env $P llama-server -hf unsloth/Phi-4-reasoning-plus-GGUF:UD-Q4_K_XL -c 32768 $C

# gemma-4-12b (MTP draft from the cached MTP/ file)
env $P llama-server -hf unsloth/gemma-4-12b-it-GGUF:UD-Q4_K_XL -c 262144 $C -md $HF/MTP/mtp-gemma-4-12b-it-Q8_0.gguf --spec-type draft-mtp -ctkd q8_0 -ctvd q8_0

# gemma-4-26B-A4B
HF26=~/.cache/huggingface/hub/models--unsloth--gemma-4-26B-A4B-it-GGUF/snapshots/c099eb48e663fd284577b04978a94ffccb261841
env $P llama-server -hf unsloth/gemma-4-26B-A4B-it-GGUF:UD-IQ4_NL -c 259840 $C -md $HF26/MTP/mtp-gemma-4-26B-A4B-it-Q8_0.gguf --spec-type draft-mtp -ctkd q8_0 -ctvd q8_0

# Nemotron-3.5-Lightning-30B-A3B
env $P llama-server -hf unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF:UD-IQ3_XXS -c 335872 $C --spec-type draft-mtp --spec-draft-n-max 1 -ctkd q8_0 -ctvd q8_0

# Nemotron-3.5-Bolt-E4B (CUDA: -ub 1024; Vulkan: -ub 1024 too for prefill)
env $P llama-server -hf unsloth/NVIDIA-Nemotron-3.5-Bolt-E4B-GGUF:UD-Q4_K_XL -c 131072 ${C/-ub 2048/-ub 1024}

# flint (either quantization)
env $P llama-server -hf unsloth/flint-GGUF:UD-Q3_K_XL -c 97536 $C
env $P llama-server -hf unsloth/flint-GGUF:UD-IQ3_XXS -c 216064 $C

# gpt-oss-20b
env $P llama-server -hf unsloth/gpt-oss-20b-GGUF:UD-Q4_K_XL -c 131072 $C

# GLM-4.7-Flash-REAP-23B-A3B (keep f16 KV)
env $P llama-server -hf unsloth/GLM-4.7-Flash-REAP-23B-A3B-GGUF:UD-Q4_K_XL -c 128256 $C

# Qwen3-14B
env $P llama-server -hf unsloth/Qwen3-14B-GGUF:Q4_K_M -c 40960 $C

# Qwen3.8-27B (f16 KV fits 101376 tokens; with `-ctk q8_0 -ctv q8_0` 181248, with iq4_nl more still)
env $P llama-server -hf unsloth/Qwen3.8-27B-GGUF:UD-IQ4_XS -c 101376 $C
```

Largest context that fits (llama-fit-params, 2 GiB free), CUDA / Vulkan; a value equal to the model maximum is not a limit:

| model | f16 KV | q8_0 KV | model maximum |
|---|---|---|---|
| Qwen3.5-9B, Ornith-1.5-9B, gemma-4-12b | 262144 / 262144 | 262144 | 262144 |
| Phi-4-reasoning-plus | 32768 | 32768 | 32768 |
| Qwen3-14B | 40960 | 40960 | 40960 |
| gpt-oss-20b, Bolt-E4B | 131072 | 131072 | 131072 |
| GLM-4.7-Flash-REAP | 128256 / 132352 | 202752 | 202752 |
| flint UD-Q3_K_XL | 97536 / 107520 | 169984 / 186880 | 262144 |
| flint UD-IQ3_XXS | 216064 / 226048 | 262144 | 262144 |
| gemma-4-26B-A4B | 259840 / 262144 | 262144 | 262144 |
| Nemotron Lightning-30B | 335872 / 564224 | 480512 / 1048576 | 1048576 |
| Qwen3.8-27B | 101376 / 101888 | 181248 / 182272 | 262144 |

Models where f16 does not reach the maximum (GLM, flint Q3_K_XL, Qwen3.8, Lightning, gemma-26B, flint IQ3_XXS) can use
`-ctk q8_0 -ctv q8_0` with the larger value from the second column at a few percent lower speed.
Raw output: `~/llama-bench-results/ctxfit.txt`.
