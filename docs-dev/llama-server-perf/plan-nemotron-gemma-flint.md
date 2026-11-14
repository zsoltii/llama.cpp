# Plan: Nemotron, gemma4 and flint speedups (CUDA, Vulkan, ROCm)

Written for an executor (Opus or Sonnet) that continues on the RTX 3090 (CUDA0 / Vulkan1) and later on the RX 6900 XT.
Companion of `plan-next-speedups.md`; results go to `rtx3090-results.md` (and `rx6900xt-results.md`).

## Rules

- Qwen3.8 must not get slower; any change that touches shared kernels is re-measured with the Qwen3.8-27B iq4_nl
  tg128 / tg@64k / pp512 numbers (`msweep.sh`, `depth.sh`) before it is committed.
- Every speedup is followed by `test-backend-ops` (the touched ops, both backends), the story loop test,
  `attach-test.py`, and wikiqa accuracy.
- Vulkan changes must be vendor-safe (gate on capabilities, run the T1200 too).
- Check `nvidia-smi` / `nvtop -s` before and during every GPU run; never claim a job state without evidence.
- Commit per speedup, no push; OpenVINO files stay uncommitted.

## Models

| model | file | notes |
|---|---|---|
| gemma-4-26B-A4B (MoE) | `unsloth/gemma-4-26B-A4B-it-GGUF` UD-IQ4_NL + MTP Q8_0 | hd512 global layers (K=V view), hd256 SWA layers |
| gemma-4-12b (dense) | `unsloth/gemma-4-12b-it-GGUF` UD-Q4_K_XL + MTP Q8_0 | same attention layout |
| Nemotron-3.5-Lightning-30B-A3B | `unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF` | Mamba2 + MoE |
| Nemotron-3.5-Bolt-E4B | `unsloth/NVIDIA-Nemotron-3.5-Bolt-E4B-GGUF` UD-Q4_K_XL + DFlash Q8_0 | small, kernel-count bound |
| flint | `unsloth/flint-GGUF` (UD-Q3_K_XL present; a second quantization is to be measured when it is on disk) | MoE |

## Items

1. gemma4 hd512 decode on the quantized MMA path (done, 76837dbd7): tg128@64k 87 -> 113 t/s.
   Open: 2 tokens (MTP verify) at hd512 needs ncols 16 with a 2-stage tile (does not fit 99 KB with separate K/V tiles:
   try one shared f16 tile as in the quantized path, nbatch_fa 32 with 2 warps per 16 columns).
2. gemma4 hd256 SWA layers: check `ggml_cuda_flash_attn_ext_get_kv_chunk` and the sliding window mask scan at 64k+
   (KV_min), profile with `nsysdepth.sh`.
3. Kernel count fusions (CUDA decode, ~1.4 us per kernel, MoE models launch 26000 rms_norm per 64 steps):
   - RMS_NORM + MUL (+ROPE) already fused; fuse norm + quantize_q8_1 for the following mat-vec,
   - GLU (gelu/silu gate) + quantize_q8_1 before the down projection,
   - router: RMS_NORM + SCALE + MUL before topk_moe,
   - QKV: one mat-vec launch for three small matrices.
   Measure with `seq.sh <model>`; a fusion must remove launches, not add registers to the hot kernel.
4. Nemotron Mamba2 layers: profile `ssm_scan` / `ssm_conv` at decode and prefill, check the chunked scan on prefill,
   utilization with `util.sh` (CPU thread count, GPU SM %).
5. MoE mat-vec (IQ3/IQ4/q4_K/q6_K/mxfp4 experts): done for 2-4 column rows (f75c5aa4a); remaining: 1-column small-matrix
   cases at ~40% of the bandwidth (`moe_case` list in `test-backend-ops perf`).
6. Vulkan MoE prefill (MUL_MAT_ID 1.5-2.4x slower than CUDA): tile choice by rows per expert and BN 16/32/64 did
   not help (iq3 regressed 2x). Next: look inside the shader (B-matrix row ids load, `mmqid` warptile with
   coopmat2, f16 accumulation), compare with the CUDA `mmq_ids_helper` path.
7. Vulkan decode: MLA FA (GLM), 2-column mat-vec for MTP verify, gemma4 hd512 FA scratch (already f16 scratch on cm2).
8. flint (all quantizations on disk):
   - run `msweep.sh` entries for each file: pp512, tg128, depth 16k/64k, CUDA and Vulkan;
   - profile the decode step (`seq.sh`), the biggest kernels first (MoE expert mat-vec of the Q3_K / Q4 types);
   - Q3_K and Q3_K_XL mixes use iq3/q3_K/q6_K tensors: check the 4-rows-per-block table for each type present;
   - Vulkan: check which MMVQ / MMQ pipelines the file hits (`GGML_VK_PERF_LOGGER`), then tune like gemma/GLM;
   - ROCm (RX 6900 XT): the same sweep after the swap.
9. bge-m3 per-ubatch CPU work, mmproj timing.
10. ROCm port check of every CUDA change (`hip-followups-from-3090.md`): quantized MMA FA is CUDA only (`mma_q_applies`
    returns false on HIP), set_rows warp kernel is gated, iq3 grids in shared memory are non-HIP.

## Final deliverable

`docs-dev/llama-server-perf/best-commands.md`: for each model the best llama-server command on CUDA and on Vulkan
(measured, with the numbers it gives), and the context length / KV type / MTP settings that give it.
