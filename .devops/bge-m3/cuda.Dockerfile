# BGE-M3 hybrid (dense + sparse) embedding server: llama-server on CUDA, hardened runtime image.
# The model is not in the image: convert a `hf download` with convert.Dockerfile, mount the GGUF read-only.
#   docker build -f .devops/bge-m3/cuda.Dockerfile -t llama-bge-m3:cuda .
#   docker run --gpus all --read-only --cap-drop ALL --security-opt no-new-privileges \
#       -v bge-models:/models:ro -p 8080:8080 llama-bge-m3:cuda
# See .devops/bge-m3/README.md (and compose.yaml) for the full set of options.

# CUDA 12.x still builds for Pascal (sm_61), 13.x does not. Pinned by digest, override both together.
ARG CUDA_DEVEL_IMAGE=docker.io/nvidia/cuda:12.9.1-devel-ubuntu24.04@sha256:020bc241a628776338f4d4053fed4c38f6f7f3d7eb5919fecb8de313bb8ba47c
ARG CUDA_RUNTIME_IMAGE=docker.io/nvidia/cuda:12.9.1-runtime-ubuntu24.04@sha256:1287141d283b8f06f45681b56a48a85791398c615888b1f96bfb9fc981392d98

## build
FROM ${CUDA_DEVEL_IMAGE} AS build

# 61 = Pascal (GTX 10xx); add more as needed, e.g. "61;75;86;89"
ARG CUDA_DOCKER_ARCH=61

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential cmake git

WORKDIR /app

COPY . .

# no web UI, no TLS (the server never downloads models; terminate TLS in a reverse proxy)
# CPU backend for every x86 level (AVX .. AVX-512), picked at run time: old Xeons have no AVX2
RUN cmake -B build -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="${CUDA_DOCKER_ARCH}" \
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF \
        -DLLAMA_OPENSSL=OFF \
        -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined && \
    cmake --build build --config Release --target llama-server -j$(nproc)

# the dynamically loaded backends are looked up next to the executable
RUN mkdir -p /out && \
    find build -name "*.so*" -exec cp -P {} /out \; && \
    cp build/bin/llama-server /out/

## runtime
FROM ${CUDA_RUNTIME_IMAGE} AS server

# security updates of the base, then nothing beyond the OpenMP runtime; strip setuid/setgid bits
RUN apt-get update \
    && apt-get upgrade -y --no-install-recommends \
    && apt-get install -y --no-install-recommends libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* /tmp/* /var/tmp/* \
    && find / -xdev -perm /6000 -type f -exec chmod a-s {} + \
    && groupadd --system --gid 10001 llama \
    && useradd --system --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin llama \
    && mkdir -p /models

# binaries and libraries stay root-owned and read-only for the service user
COPY --from=build /out/ /app/
COPY .devops/bge-m3/healthcheck.sh /app/healthcheck

WORKDIR /app

# defaults, each can be overridden with docker run -e (see README.md for the measured trade-offs)
#   DEVICE: fail at start when the GPU is not visible instead of falling back to the CPU
#   CTX_SIZE = N_PARALLEL x EMBD_TRUNCATE: every slot holds one input of up to 512 tokens
#   UBATCH: the encoder attends across all sequences of a ubatch, a small ubatch wastes less attention
ENV LD_LIBRARY_PATH=/app:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
    LLAMA_ARG_HOST=0.0.0.0 \
    LLAMA_ARG_PORT=8080 \
    LLAMA_ARG_MODEL=/models/bge-m3-hybrid-f32.gguf \
    LLAMA_ARG_EMBEDDINGS=1 \
    LLAMA_ARG_EMBD_TRUNCATE=512 \
    LLAMA_ARG_DEVICE=CUDA0 \
    LLAMA_ARG_N_GPU_LAYERS=99 \
    LLAMA_ARG_FIT=off \
    LLAMA_ARG_N_PARALLEL=32 \
    LLAMA_ARG_CTX_SIZE=16384 \
    LLAMA_ARG_BATCH=16384 \
    LLAMA_ARG_UBATCH=1024 \
    LLAMA_ARG_THREADS=2 \
    LLAMA_ARG_UI=0 \
    LLAMA_ARG_ENDPOINT_SLOTS=0 \
    LLAMA_ARG_ENDPOINT_METRICS=0 \
    LLAMA_ARG_ENDPOINT_PROPS=0 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

USER 10001:10001

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 CMD [ "/app/healthcheck" ]

ENTRYPOINT [ "/app/llama-server" ]
