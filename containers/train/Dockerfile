# syntax=docker/dockerfile:1.7

ARG RUNTIME_BASE=dependencies

# dependency-image-inputs-begin
ARG PYTHON_IMAGE=python:3.14-slim-bookworm@sha256:4ff4b92a68355dbdb52584ab3391dff8d371a61d4e063468bfd0130e3189c6d9
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.13@sha256:841c8e6fe30a8b07b4478d12d0c608cba6de66102d29d65d1cc423af86051563

FROM ${UV_IMAGE} AS uv

FROM ${PYTHON_IMAGE} AS dependencies

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

COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

ARG UV_LOCK_SHA256=unknown

LABEL org.opencontainers.image.title="rlab train dependencies"
LABEL org.opencontainers.image.description="Locked system and Python dependencies for rlab train runtimes"
LABEL org.opencontainers.image.source="https://github.com/tsilva/rlab"
LABEL io.rlab.uv-lock.sha256="${UV_LOCK_SHA256}"

# dependency-image-inputs-end

FROM ${PYTHON_IMAGE} AS app-package

ENV UV_LINK_MODE=copy

COPY --from=uv /uv /uvx /usr/local/bin/

WORKDIR /src

COPY pyproject.toml README.md ./
COPY src ./src

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --target /opt/rlab --no-deps .

FROM ${RUNTIME_BASE} AS runtime

ENV PYTHONPATH=/root/rlab/src:/opt/rlab

COPY --link --from=app-package /opt/rlab /opt/rlab
COPY --link src ./src
COPY --link METRICS.md INSTANCES.md ./
COPY --link scripts ./scripts
COPY --link experiments/machines.yaml ./experiments/
COPY --link experiments/modal_eval.yaml ./experiments/
COPY --link --chmod=755 containers/train/rlab /usr/local/bin/rlab
COPY --link --chmod=755 containers/train/entrypoint.sh /usr/local/bin/rlab-container-entrypoint
COPY --link --chmod=755 containers/train/smoke.py /usr/local/bin/rlab-container-smoke

ARG BUILD_DATE=unknown
ARG VCS_REF=unknown
ARG DEPENDENCY_IMAGE=unknown
ARG DEPENDENCY_IMAGE_DIGEST=unknown

LABEL org.opencontainers.image.title="rlab train runtime"
LABEL org.opencontainers.image.description="Locked rlab training and evaluation runtime"
LABEL org.opencontainers.image.source="https://github.com/tsilva/rlab"
LABEL org.opencontainers.image.created="${BUILD_DATE}"
LABEL org.opencontainers.image.revision="${VCS_REF}"
LABEL org.opencontainers.image.base.name="${DEPENDENCY_IMAGE}"
LABEL org.opencontainers.image.base.digest="${DEPENDENCY_IMAGE_DIGEST}"

CMD ["rlab-container-smoke"]
