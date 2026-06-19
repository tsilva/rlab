<div align="center">
  <img src="./logo.png" alt="rlab" width="256" />

  **rlab is a reinforcement-learning workbench for training PPO agents on imported Stable Retro games.**
</div>

It uses `stable-retro-turbo`, Stable-Baselines3, W&B, and local or remote
runners to move from ROM import to checkpoint evaluation and playback.

The repo is optimized for experiment iteration: configure a game target, run a
bounded training job, upload checkpoints, evaluate them out of process, and
promote the best model by completion rate, max x-position, then mean reward.

## Install

```bash
git clone git@github.com:tsilva/rlab.git
cd rlab
UV_CACHE_DIR=.uv-cache uv sync --frozen
UV_CACHE_DIR=.uv-cache uv run python scripts/import_roms.py ~/Desktop/roms
```

Stable Retro matches ROMs by SHA, not filename. The import must recognize the
game id you plan to pass with `--game`.

## Run

Start with a local smoke run:

```bash
UV_CACHE_DIR=.uv-cache uv run python -m stable_retro_ppo.train \
  --game <GameId> \
  --preset smoke \
  --run-name local_smoke \
  --run-description "Local Stable Retro PPO smoke test"
```

Evaluate and watch the resulting model:

```bash
UV_CACHE_DIR=.uv-cache uv run python -m stable_retro_ppo.evaluate \
  --game <GameId> \
  --model runs/local_smoke/final_model.zip \
  --episodes 2 \
  --max-steps 600

UV_CACHE_DIR=.uv-cache uv run python -m stable_retro_ppo.play \
  --game <GameId> \
  --model runs/local_smoke/final_model.zip \
  --episodes 3 \
  --max-steps 1200 \
  --fps 30 \
  --scale 4
```

## Commands

```bash
UV_CACHE_DIR=.uv-cache uv sync --frozen
UV_CACHE_DIR=.uv-cache uv run ruff check .
UV_CACHE_DIR=.uv-cache uv run pytest

UV_CACHE_DIR=.uv-cache uv run python -m stable_retro_ppo.train --game <GameId> --preset smoke --run-description "Smoke test"
UV_CACHE_DIR=.uv-cache uv run python -m stable_retro_ppo.evaluate --game <GameId> --policy random --episodes 2 --max-steps 600
UV_CACHE_DIR=.uv-cache uv run python scripts/eval_wandb_checkpoints.py <run-name> --game <GameId> --episodes 50 --record-best-video
UV_CACHE_DIR=.uv-cache uv run python scripts/play_wandb_artifact.py <run-name> --game <GameId> --kind best --stochastic
```

## Remote Runs

Modal setup:

```bash
UV_CACHE_DIR=.uv-cache uv sync --frozen --extra modal
UV_CACHE_DIR=.uv-cache uv run modal setup
UV_CACHE_DIR=.uv-cache uv run modal run src/stable_retro_ppo/modal_app.py::upload_roms --rom-dir ~/Desktop/roms
```

Modal smoke job:

```bash
UV_CACHE_DIR=.uv-cache uv run modal run src/stable_retro_ppo/modal_app.py::train \
  --game <GameId> \
  --timesteps 512 \
  --n-envs 1 \
  --run-name modal_smoke \
  --run-description "Modal smoke test" \
  --max-episode-steps 600
```

SkyPilot launch manifests live in `experiments/launches/` and are rendered or
preflighted through `scripts/stable_retro_skypilot.py`. Read `INSTANCES.md`
before choosing hardware, changing concurrency, or launching remote training.

## Notes

- Python is pinned to `>=3.14,<3.15`; dependency resolution is managed by `uv`
  and `uv.lock`.
- The project name is `rlab`; the current Python package and console-script
  names still use `stable_retro_ppo` and `stable-retro-ppo-*`.
- Runtime support is pinned in `pyproject.toml` for macOS arm64 and Linux
  x86_64 with `stable-retro-turbo`.
- Every training run should include `--run-description`.
- Training logs to W&B and uploads model artifacts unless `--no-wandb-artifacts`
  is set.
- Set `WANDB_API_KEY` for online W&B. For R2/S3-backed reference artifacts, set
  `CHECKPOINT_BUCKET_URI` or pass `--wandb-artifact-storage-uri`, along with the
  required `AWS_*` credentials.
- Keep generated checkpoints, logs, videos, W&B files, caches, and ad hoc launch
  specs out of source control.
- Local eval outputs are written under `runs/local_evals/<run-name>/`.

## Architecture

![rlab architecture diagram](./architecture.png)

## License

No license file is present in this repository.
