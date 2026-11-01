# BGE-M3 hybrid (dense + sparse) embedding server: llama-server on Vulkan, hardened runtime image.
# The model is not in the image: convert a `hf download` with convert.Dockerfile, mount the GGUF read-only.
#   docker build -f .devops/bge-m3/vulkan.Dockerfile -t llama-bge-m3:vulkan .
#   AMD/Intel: docker run --device /dev/dri --group-add video ... llama-bge-m3:vulkan
#   NVIDIA:    docker run --gpus all ... llama-bge-m3:vulkan   (nvidia-container-toolkit adds the Vulkan ICD)
# See .devops/bge-m3/README.md (and compose.yaml) for the full set of options.

# ubuntu:26.04, pinned
ARG UBUNTU_IMAGE=docker.io/library/ubuntu:26.04@sha256:6a06111c3b0a8600feadb270aec771dc239bf407b25625f93f44ab00e11f4d3c

## build
FROM ${UBUNTU_IMAGE} AS build

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential cmake git libvulkan-dev glslc spirv-headers

WORKDIR /app

COPY . .

# no web UI, no TLS (the server never downloads models; terminate TLS in a reverse proxy)
# CPU backend for every x86 level (AVX .. AVX-512), picked at run time: old Xeons have no AVX2
RUN cmake -B build -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DGGML_VULKAN=ON \
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF \
        -DLLAMA_OPENSSL=OFF && \
    cmake --build build --config Release --target llama-server -j$(nproc)

# the dynamically loaded backends are looked up next to the executable
RUN mkdir -p /out && \
    find build -name "*.so*" -exec cp -P {} /out \; && \
    cp build/bin/llama-server /out/

## runtime
FROM ${UBUNTU_IMAGE} AS server

# the NVIDIA ICD comes from the host driver (nvidia-container-toolkit, graphics capability) and needs libX11/libXext
# and the glvnd libEGL (headless Vulkan),
# mesa covers AMD and Intel
# security updates of the base, then only the Vulkan loader, the mesa drivers and OpenMP; strip setuid/setgid bits
RUN apt-get update \
    && apt-get upgrade -y --no-install-recommends \
    && apt-get install -y --no-install-recommends libgomp1 libvulkan1 mesa-vulkan-drivers libx11-6 libxext6 libegl1 \
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
ENV LD_LIBRARY_PATH=/app \
    LLAMA_ARG_HOST=0.0.0.0 \
    LLAMA_ARG_PORT=8080 \
    LLAMA_ARG_MODEL=/models/bge-m3-hybrid-f32.gguf \
    LLAMA_ARG_EMBEDDINGS=1 \
    LLAMA_ARG_EMBD_TRUNCATE=512 \
    LLAMA_ARG_DEVICE=Vulkan0 \
    LLAMA_ARG_N_GPU_LAYERS=99 \
    LLAMA_ARG_FIT=off \
    LLAMA_ARG_N_PARALLEL=32 \
    LLAMA_ARG_CTX_SIZE=16384 \
    LLAMA_ARG_BATCH=16384 \
    LLAMA_ARG_UBATCH=2048 \
    LLAMA_ARG_THREADS=2 \
    LLAMA_ARG_UI=0 \
    LLAMA_ARG_ENDPOINT_SLOTS=0 \
    LLAMA_ARG_ENDPOINT_METRICS=0 \
    LLAMA_ARG_ENDPOINT_PROPS=0 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

USER 10001:10001

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 CMD [ "/app/healthcheck" ]

ENTRYPOINT [ "/app/llama-server" ]
