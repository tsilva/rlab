# syntax=docker/dockerfile:1.7

ARG RUNTIME_BASE=dependencies
ARG GPU_BASE=gpu

# gpu-image-inputs-begin
ARG PYTHON_IMAGE=python:3.14-slim-bookworm@sha256:4ff4b92a68355dbdb52584ab3391dff8d371a61d4e063468bfd0130e3189c6d9
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.13@sha256:841c8e6fe30a8b07b4478d12d0c608cba6de66102d29d65d1cc423af86051563

FROM ${UV_IMAGE} AS uv

FROM ${PYTHON_IMAGE} AS gpu

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV RLAB_PROJECT_ROOT=/root/rlab
ENV UV_PROJECT_ENVIRONMENT=/root/rlab/.venv
ENV UV_LINK_MODE=copy
ENV PATH="/root/rlab/.venv/bin:${PATH}"
ENV PYTHONPATH=/root/rlab/src
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV WANDB_DIR=/root/rlab/runs
ENV WANDB_CACHE_DIR=/root/rlab/runs/.wandb-cache
ENV WANDB_CONFIG_DIR=/root/rlab/runs/.wandb-config
ENV WANDB_DATA_DIR=/root/rlab/runs/.wandb-data
ENV WANDB_ARTIFACT_DIR=/root/rlab/runs/.wandb-artifacts

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        ffmpeg \
        git \
        libgl1 \
        libglib2.0-0 \
        libxcb1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /usr/local/bin/

WORKDIR /root/rlab

COPY containers/train/gpu-linux-amd64.lock /tmp/gpu-linux-amd64.lock

RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv "${UV_PROJECT_ENVIRONMENT}" \
    && uv pip install \
        --python "${UV_PROJECT_ENVIRONMENT}/bin/python" \
        --require-hashes \
        --no-build \
        --no-deps \
        -r /tmp/gpu-linux-amd64.lock

ARG GPU_PLAN_SHA256=unknown

LABEL org.opencontainers.image.title="rlab train GPU foundation"
LABEL org.opencontainers.image.description="Locked GPU framework dependencies for rlab train runtimes"
LABEL org.opencontainers.image.source="https://github.com/tsilva/rlab"
LABEL io.rlab.gpu-plan.sha256="${GPU_PLAN_SHA256}"
# gpu-image-inputs-end

# dependency-image-inputs-begin
FROM ${GPU_BASE} AS dependencies

COPY containers/train/train-linux-amd64.lock /tmp/train-linux-amd64.lock

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip sync \
        --python "${UV_PROJECT_ENVIRONMENT}/bin/python" \
        --require-hashes \
        /tmp/train-linux-amd64.lock

ARG TRAIN_PLAN_SHA256=unknown
ARG GPU_IMAGE=unknown
ARG GPU_IMAGE_DIGEST=unknown

LABEL org.opencontainers.image.title="rlab train dependencies"
LABEL org.opencontainers.image.description="Locked system and Python dependencies for rlab train runtimes"
LABEL org.opencontainers.image.source="https://github.com/tsilva/rlab"
LABEL io.rlab.train-plan.sha256="${TRAIN_PLAN_SHA256}"
LABEL org.opencontainers.image.base.name="${GPU_IMAGE}"
LABEL org.opencontainers.image.base.digest="${GPU_IMAGE_DIGEST}"

# dependency-image-inputs-end

# runtime-image-inputs-begin
FROM ${PYTHON_IMAGE} AS app-package

ENV UV_LINK_MODE=copy

COPY --from=uv /uv /uvx /usr/local/bin/

WORKDIR /src

COPY pyproject.toml ./
COPY src ./src

RUN --mount=type=cache,target=/root/.cache/uv \
    printf '# rlab runtime package\n' > README.md \
    && uv pip install --target /opt/rlab --no-deps .

FROM scratch AS runtime-overlay

COPY --from=app-package /opt/rlab /opt/rlab
COPY src /root/rlab/src
COPY scripts /root/rlab/scripts
COPY experiments/machines.yaml /root/rlab/experiments/machines.yaml
COPY experiments/modal_eval.yaml /root/rlab/experiments/modal_eval.yaml
COPY --chmod=755 containers/train/rlab /usr/local/bin/rlab
COPY --chmod=755 containers/train/entrypoint.sh /usr/local/bin/rlab-container-entrypoint
COPY --chmod=755 containers/train/smoke.py /usr/local/bin/rlab-container-smoke

FROM ${RUNTIME_BASE} AS runtime

ENV PYTHONPATH=/root/rlab/src:/opt/rlab

COPY --link --from=runtime-overlay / /

ARG BUILD_DATE=unknown
ARG VCS_REF=unknown
ARG RUNTIME_INPUT_SHA256=unknown
ARG DEPENDENCY_IMAGE=unknown
ARG DEPENDENCY_IMAGE_DIGEST=unknown

ENV RLAB_SOURCE_SHA="${VCS_REF}"
ENV RLAB_RUNTIME_INPUT_SHA256="${RUNTIME_INPUT_SHA256}"

LABEL org.opencontainers.image.title="rlab train runtime"
LABEL org.opencontainers.image.description="Locked rlab training and evaluation runtime"
LABEL org.opencontainers.image.source="https://github.com/tsilva/rlab"
LABEL org.opencontainers.image.created="${BUILD_DATE}"
LABEL org.opencontainers.image.revision="${VCS_REF}"
LABEL io.rlab.runtime-input.sha256="${RUNTIME_INPUT_SHA256}"
LABEL org.opencontainers.image.base.name="${DEPENDENCY_IMAGE}"
LABEL org.opencontainers.image.base.digest="${DEPENDENCY_IMAGE_DIGEST}"

CMD ["rlab-container-smoke"]
# runtime-image-inputs-end
