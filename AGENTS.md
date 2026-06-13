# Project Rules

## Stable Retro

Use the PyPI `stable-retro-turbo` package as the stable-retro provider. The runtime import path remains `stable_retro`.

## Training Runs

Keep generated training artifacts out of source control. Use `runs/`, `logs/`, and `models/` for checkpoints, TensorBoard logs, and evaluation outputs.

Default remote training should not run robust evals in the training loop. Modal should focus on training, checkpointing, and uploading checkpoint artifacts. Evaluate checkpoints out of process, preferably locally while waiting for remote training progress, and log checkpoint eval metrics back to the same W&B run. Promote the current best checkpoint from that external eval process using completion rate first, then maximum x-position, then mean reward.

## Dependencies

Use `uv` for dependency resolution and keep `uv.lock` committed. Preserve Python supply-chain hardening in `pyproject.toml`.
