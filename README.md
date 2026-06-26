<div align="center">
  <img src="./logo.png" alt="rlab" width="256" />

  **Reinforcement-learning workbench for training game agents**
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
UV_CACHE_DIR=.uv-cache uv run python -m rlab.train \
  --game <GameId> \
  --preset smoke \
  --run-name local_smoke \
  --run-description "Local rlab smoke test"
```

Evaluate and watch the resulting model:

```bash
UV_CACHE_DIR=.uv-cache uv run python -m rlab.evaluate \
  --game <GameId> \
  --model runs/local_smoke/final_model.zip \
  --episodes 2 \
  --max-steps 600

UV_CACHE_DIR=.uv-cache uv run python -m rlab.play \
  --game <GameId> \
  --model runs/local_smoke/final_model.zip \
  --episodes 3 \
  --max-steps 1200 \
  --fps 30 \
  --scale 4
```

Mixed Mario start-state rehearsal stays on `StableRetroNativeVecEnv`. The CLI
keeps `--states` and `--state-probs` for compatibility, then translates them to
the current native `state=` constructor argument. Use fixed native env
slots:

```bash
UV_CACHE_DIR=.uv-cache uv run python -m rlab.train \
  --game SuperMarioBros-Nes-v0 \
  --states Level1-1,Level1-2 \
  --n-envs 2 \
  --run-name mario_l1_l2_fixed \
  --run-description "Native vector fixed-slot rehearsal on Level1-1 and Level1-2"
```

Or native reset-time weighted sampling. `--state-probs` values must be positive
finite weights; training normalizes them before storing metadata and W&B config.

```bash
UV_CACHE_DIR=.uv-cache uv run python -m rlab.train \
  --game SuperMarioBros-Nes-v0 \
  --states Level1-1,Level1-2 \
  --state-probs 1,3 \
  --n-envs 8 \
  --run-name mario_l1_l2_weighted \
  --run-description "Reset-time weighted rehearsal on Level1-1 and Level1-2"
```

## Commands

```bash
UV_CACHE_DIR=.uv-cache uv sync --frozen
UV_CACHE_DIR=.uv-cache uv run ruff check .
UV_CACHE_DIR=.uv-cache uv run pytest

UV_CACHE_DIR=.uv-cache uv run python -m rlab.train --game <GameId> --preset smoke --run-description "Smoke test"
UV_CACHE_DIR=.uv-cache uv run python -m rlab.evaluate --game <GameId> --policy random --episodes 2 --max-steps 600
UV_CACHE_DIR=.uv-cache uv run python scripts/eval_wandb_checkpoints.py <run-name> --game <GameId> --episodes 50 --record-best-video
UV_CACHE_DIR=.uv-cache uv run python scripts/play_wandb_artifact.py <run-name> --game <GameId> --kind best --stochastic
```

## Remote Runs

Modal setup:

```bash
UV_CACHE_DIR=.uv-cache uv sync --frozen --extra modal
UV_CACHE_DIR=.uv-cache uv run modal setup
UV_CACHE_DIR=.uv-cache uv run modal run src/rlab/modal_app.py::upload_roms --rom-dir ~/Desktop/roms
```

Provider-neutral Modal launch dry run:

```bash
UV_CACHE_DIR=.uv-cache uv run rlab-compute launch \
  experiments/launches/rlab_rtx4090.example.json \
  --target modal-t4
```

SkyPilot launch manifests live in `experiments/launches/` and are rendered or
preflighted through `rlab-compute` for provider-neutral direct launches or
`rlab-skypilot` for SkyPilot-specific workflows. Read `INSTANCES.md` before
choosing hardware, changing concurrency, or launching remote training. Use
`rlab-compute targets` to list configured compute targets, and pass
`--target <name>` to swap between `beast-3`, `beast-2`, RunPod, Modal, or other
configured backends without editing the manifest.

For queue-backed training, prefer long-lived runner profiles in
`experiments/runner_profiles/`. Those profiles render the SkyPilot runtime
envelope and start `rlab.train_runner`; experiment payloads stay in
the campaign queue.

For local GPU queue capacity, use the fleet manager instead of SkyPilot. It
reads pending/running `train_jobs`, groups demand by `profile_id`,
`runtime_image_ref`, and `run_target`, then reconciles Docker runner containers
on `beast-3` and `beast-2` over SSH:

```bash
UV_CACHE_DIR=.uv-cache uv run rlab-fleet status
UV_CACHE_DIR=.uv-cache uv run rlab-fleet plan
UV_CACHE_DIR=.uv-cache uv run rlab-fleet reconcile --execute
```

Bootstrap each host once, then install the host-local reconciler so runners keep
working when the Mac sleeps or disconnects:

```bash
UV_CACHE_DIR=.uv-cache uv run rlab-fleet setup-host \
  --host beast-3 \
  --runtime-image-ref-file rlab-train-image.json \
  --execute

UV_CACHE_DIR=.uv-cache uv run rlab-fleet setup-host \
  --host beast-2 \
  --runtime-image-ref-file rlab-train-image.json \
  --execute

UV_CACHE_DIR=.uv-cache uv run rlab-fleet install-systemd --host beast-3 --execute
UV_CACHE_DIR=.uv-cache uv run rlab-fleet install-systemd --host beast-2 --execute
```

The fleet manager does not schedule experiments and does not inspect RL config.
It only starts, keeps, restarts, or removes runner containers for digest-pinned
train jobs. It never removes an obsolete container while one of its worker ids
still owns a running queue lease.

## Notes

- Python is pinned to `==3.14.*`; dependency resolution is managed by `uv`
  and `uv.lock`.
- The Python package is `rlab`; console scripts use the `rlab-*` prefix.
- Runtime support is pinned in `pyproject.toml` for macOS arm64 and Linux
  x86_64 with `stable-retro-turbo`.
- Every training run should include `--run-description`.
- Training logs to W&B and uploads model artifacts unless `--no-wandb-artifacts`
  is set.
- Queue-backed train jobs should reference immutable runtime image digests. The
  train-image CI workflow uploads `rlab-train-image.json`; pass it with
  `--runtime-image-ref-file` when enqueueing jobs.
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
