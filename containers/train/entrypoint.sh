#!/usr/bin/env bash
set -euo pipefail

export RLAB_PROJECT_ROOT="${RLAB_PROJECT_ROOT:-/root/rlab}"
export PYTHONPATH="${RLAB_PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export WANDB_DIR="${WANDB_DIR:-${RLAB_PROJECT_ROOT}/runs}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${WANDB_DIR}/.wandb-cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${WANDB_DIR}/.wandb-config}"
export WANDB_DATA_DIR="${WANDB_DATA_DIR:-${WANDB_DIR}/.wandb-data}"
export WANDB_ARTIFACT_DIR="${WANDB_ARTIFACT_DIR:-${WANDB_DIR}/.wandb-artifacts}"

mkdir -p "$MPLCONFIGDIR" "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$WANDB_DATA_DIR" "$WANDB_ARTIFACT_DIR"

if [ "$#" -eq 0 ]; then
  exec rlab-container-smoke
fi

exec "$@"
