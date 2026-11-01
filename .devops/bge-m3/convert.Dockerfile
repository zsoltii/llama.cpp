# One-shot converter: BAAI/bge-m3 from a `hf download` (hub cache or --local-dir) -> hybrid GGUF (dense + sparse head)
#   docker build -f .devops/bge-m3/convert.Dockerfile -t llama-bge-m3:convert .
#   hf download BAAI/bge-m3
#   docker run --rm --network none --read-only --tmpfs /tmp --cap-drop ALL --security-opt no-new-privileges \
#       -v ~/.cache/huggingface/hub:/hf:ro -v bge-models:/models llama-bge-m3:convert
# The image never downloads anything at run time (HF_HUB_OFFLINE) and runs as an unprivileged user.

# python:3.12-slim, pinned
ARG PYTHON_IMAGE=docker.io/library/python:3.12-slim@sha256:aa943ce411423c7582288f18707ff66eddcbaeb510502d747a72aeb34f30d159

FROM ${PYTHON_IMAGE}

WORKDIR /conv

# CPU-only torch (requirements file points pip at the CPU wheel index)
COPY requirements/requirements-convert_hf_to_gguf.txt requirements/requirements-convert_legacy_llama.txt ./requirements/
COPY gguf-py ./gguf-py
RUN pip install --no-cache-dir --disable-pip-version-check -r requirements/requirements-convert_hf_to_gguf.txt ./gguf-py \
    && python -m pip uninstall -y pip

COPY convert_hf_to_gguf.py ./
COPY conversion ./conversion
COPY .devops/bge-m3/bge-m3-convert.sh /usr/local/bin/bge-m3-convert

RUN find / -xdev -perm /6000 -type f -exec chmod a-s {} + \
    && groupadd --system --gid 10001 llama \
    && useradd --system --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin llama \
    && mkdir -p /models /hf && chown 10001:10001 /models

ENV HOME=/tmp \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

USER 10001:10001

VOLUME [ "/models" ]

ENTRYPOINT [ "/usr/local/bin/bge-m3-convert" ]
