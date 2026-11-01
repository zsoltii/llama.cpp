# BGE-M3 hybrid embedding server (CUDA / Vulkan)

`llama-server` serving [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) with both of its retrieval heads in one
request: the dense vector (normalized CLS) and the sparse lexical weights (`relu(h·W + b)`, max per token id,
special tokens left out), as FlagEmbedding's `BGEM3FlagModel.encode(return_dense=True, return_sparse=True)` computes them.

```
POST /embeddings  {"content": ["text 1", "text 2"], "truncate": 512}
-> [{"index": 0, "embedding": [[1024 floats]], "sparse": {"indices": [token ids], "values": [weights]}}, ...]
```

The OpenAI-style `/v1/embeddings` returns the same `sparse` field next to `embedding`.

## Files

| file                 | what                                                                            |
|----------------------|---------------------------------------------------------------------------------|
| `convert.Dockerfile` | one-shot converter: `hf download` (hub cache or `--local-dir`) -> hybrid GGUF   |
| `cuda.Dockerfile`    | server, CUDA 12.9 (builds for Pascal sm_61 by default: `CUDA_DOCKER_ARCH`)       |
| `vulkan.Dockerfile`  | server, Vulkan (NVIDIA through the container toolkit, AMD/Intel through mesa)   |
| `compose.yaml`       | converter + server with the hardening options below                            |

The model is not baked into the images. Download it once on the host, the converter reads the download offline:

```sh
hf download BAAI/bge-m3                       # into ~/.cache/huggingface/hub (or $HF_HOME/hub)
docker compose -f .devops/bge-m3/compose.yaml --profile cuda up -d --build
curl -s localhost:8080/embeddings -d '{"content": ["hello"]}'
```

Without compose:

```sh
docker build -f .devops/bge-m3/convert.Dockerfile -t llama-bge-m3:convert .
docker build -f .devops/bge-m3/cuda.Dockerfile    -t llama-bge-m3:cuda .
docker volume create bge-models
docker run --rm --network none --read-only --tmpfs /tmp --cap-drop ALL --security-opt no-new-privileges \
    -v ~/.cache/huggingface/hub:/hf:ro -v bge-models:/models llama-bge-m3:convert
docker run -d --gpus all --read-only --tmpfs /tmp --cap-drop ALL --security-opt no-new-privileges \
    -v bge-models:/models:ro -p 127.0.0.1:8080:8080 llama-bge-m3:cuda
```

The converter accepts `/hf` as the hub cache, a single `models--BAAI--bge-m3` directory, or a plain
`hf download BAAI/bge-m3 --local-dir DIR` directory. Settings: `BGE_REVISION` (branch, tag or commit in the cache,
default `main`), `BGE_OUTTYPE` (`f32` default, `f16`), `BGE_FORCE=1` to convert again. It needs the
repo's `sparse_linear.pt` (the sparse head), so download the whole repo, not only the safetensors.

## Hardening

- Server image: no model, no web UI, built without TLS (terminate it in a reverse proxy), only the OpenMP runtime (and for Vulkan the
  loader, mesa and the X11/EGL libraries the NVIDIA ICD links against) on top of the pinned base; base packages upgraded at build time; setuid/setgid bits removed.
- Runs as uid 10001 (no shell, no home); binaries are root-owned, the model volume is mounted read-only.
- Works with `--read-only`, `--cap-drop ALL`, `no-new-privileges`, a small `noexec` tmpfs; compose also sets
  pids and memory limits and publishes the port on 127.0.0.1 only.
- `/slots`, `/metrics`, `/props` (POST) are off. Set `LLAMA_API_KEY` (or `LLAMA_ARG_API_KEY_FILE` with a secret) when
  the port is reachable by others; `/health` stays open for the health check (bash `/dev/tcp`, no curl in the image).
- Converter: no network (`--network none`, `HF_HUB_OFFLINE`), read-only input, weights read with
  `torch.load(weights_only=True)` or safetensors, output written atomically.
- The GPU is fixed (`LLAMA_ARG_DEVICE=CUDA0` / `Vulkan0`): without a visible GPU the server exits at start
  (`invalid device`) instead of silently running on the CPU.
- The CPU backend is built for every x86 level and picked at run time (`GGML_CPU_ALL_VARIANTS`), so the image also
  starts on CPUs without AVX2.
- Base images are pinned by digest (`CUDA_*_IMAGE`, `UBUNTU_IMAGE`, `PYTHON_IMAGE` build args); bump them deliberately.

## GPU access

- NVIDIA: needs [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/) on the
  host (`--gpus all`). The Vulkan image additionally needs the `graphics` driver capability (set in the image) so the
  toolkit mounts the NVIDIA Vulkan ICD.
- AMD/Intel (Vulkan): `--device /dev/dri --group-add $(getent group render | cut -d: -f3)`.
- NVIDIA without the toolkit (tested this way, driver 580): pass the device nodes and mount the driver's user-space
  libraries yourself, all of them from the same driver version:

  ```sh
  L=/usr/lib/x86_64-linux-gnu V=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)
  M=""; for f in $(ls $L | grep -E "^lib(nvidia-[a-z0-9-]+|GLX_nvidia|EGL_nvidia|cuda)\.so\.$V$"); do
      M="$M -v $L/$f:/nv/$(objdump -p $L/$f | awk '/SONAME/{print $2}'):ro"; done
  DEV="--device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm --device /dev/nvidia-modeset"
  # CUDA
  docker run $DEV $M -e LD_LIBRARY_PATH=/app:/nv ... llama-bge-m3:cuda
  # Vulkan: also the ICD and the EGL vendor file
  docker run $DEV $M -e LD_LIBRARY_PATH=/app:/nv \
      -v /usr/share/vulkan/icd.d/nvidia_icd.json:/etc/vulkan/icd.d/nvidia_icd.json:ro \
      -v /usr/share/glvnd/egl_vendor.d/10_nvidia.json:/usr/share/glvnd/egl_vendor.d/10_nvidia.json:ro \
      ... llama-bge-m3:vulkan
  ```

## Tuning

All `LLAMA_ARG_*` variables can be overridden with `-e`. The defaults come from measurements on a GTX 1080 Ti:

- `LLAMA_ARG_EMBD_TRUNCATE=512`: FlagEmbedding's `max_length`; the server keeps the first 511 tokens plus the
  closing `</s>`. Raise it (and `CTX_SIZE`) for longer chunks; 8192 is the model limit.
- `LLAMA_ARG_N_PARALLEL` x `LLAMA_ARG_EMBD_TRUNCATE` = `LLAMA_ARG_CTX_SIZE`: each slot holds one input.
- `LLAMA_ARG_UBATCH`: the non-causal encoder attends over all sequences packed into a ubatch, so the attention work
  grows with the ubatch size: 1024 on CUDA, 2048 on Vulkan.

Measured on the GTX 1080 Ti, f32, 32 concurrent clients (single-text queries / 16-text batches of ~500-token chunks):

| image  | queries/s | p50 ms | index texts/s | server CPU (query load) |
|--------|----------:|-------:|--------------:|------------------------:|
| CUDA   | 428       | 75     | 33.5          | ~2.4 cores              |
| Vulkan | 270       | 94     | 29.9          | ~2 cores                |
- `f32` is the fastest on GPUs without fast fp16 (Pascal); `f16` halves the VRAM with the same retrieval results.
