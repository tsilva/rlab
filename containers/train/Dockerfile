# syntax=docker/dockerfile:1.7

ARG PYTHON_IMAGE=python:3.14-slim-bookworm
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.13

FROM ${UV_IMAGE} AS uv

FROM ${PYTHON_IMAGE} AS runtime

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

COPY README.md ./
COPY src ./src

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

COPY METRICS.md INSTANCES.md ./
COPY scripts ./scripts
COPY experiments/machines.yaml ./experiments/
COPY experiments/policies ./experiments/policies
COPY containers/train/entrypoint.sh /usr/local/bin/rlab-container-entrypoint
COPY containers/train/smoke.py /usr/local/bin/rlab-container-smoke

RUN chmod +x /usr/local/bin/rlab-container-entrypoint /usr/local/bin/rlab-container-smoke \
    && mkdir -p /root/rlab/runs /root/rlab/logs /tmp/matplotlib

ARG BUILD_DATE=unknown
ARG VCS_REF=unknown

LABEL org.opencontainers.image.title="rlab train runtime"
LABEL org.opencontainers.image.description="Locked rlab training and evaluation runtime"
LABEL org.opencontainers.image.source="https://github.com/tsilva/rlab"
LABEL org.opencontainers.image.created="${BUILD_DATE}"
LABEL org.opencontainers.image.revision="${VCS_REF}"

CMD ["rlab-container-smoke"]
