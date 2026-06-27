<div align="center">
  <img src="./logo.png" alt="rlab" width="256" />

  **Reinforcement-learning workbench for training game agents**
</div>

It uses `stable-retro-turbo`, Stable-Baselines3, W&B, and local queue runners
to move from ROM import to checkpoint evaluation and playback.

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
UV_CACHE_DIR=.uv-cache uv run python -m rlab.eval \
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
UV_CACHE_DIR=.uv-cache uv run python -m unittest discover -s tests -v

UV_CACHE_DIR=.uv-cache uv run python -m rlab.train --game <GameId> --preset smoke --run-description "Smoke test"
UV_CACHE_DIR=.uv-cache uv run python -m rlab.eval --game <GameId> --policy random --episodes 2 --max-steps 600
UV_CACHE_DIR=.uv-cache uv run rlab-eval --artifact-run <run-name> --checkpoint-series --game <GameId> --episodes 50 --record-best-video
UV_CACHE_DIR=.uv-cache uv run rlab-play <entity>/<project>/<run-name>-checkpoint:latest
```

## Research Loop

The current Mario Level1-1 contract is machine-readable:

```bash
cat experiments/goals/mario-level1-100of100/goal.json
```

Queue comparable experiments from checked-in spec files instead of ad hoc
commands:

```bash
UV_CACHE_DIR=.uv-cache uv run rlab-queue enqueue-train \
  --spec-file experiments/goals/mario-level1-100of100/specs/b83-b55-post21-five-seed-l11-confirm.json \
  --runtime-image-ref-file rlab-train-image.json
```

Train specs are validated against the mandatory queue-backed schema in
`src/rlab/spec_schema.py` before enqueue. Extra research metadata is preserved
in `spec_payload_json`, but required launch, naming, W&B, seed, selection, and
minimal train-config fields must be present and well-formed.

Seed ranges are part of the comparability contract. Training seeds must stay in
`0..9999`, accounting for vector env slots, and seeds `10000+` are reserved for
eval and playback. `rlab.eval`, eval-queue jobs, and `rlab.play` default to
seed `10007`; pass `--seed` only when replaying a legacy comparison
deliberately.

Then keep capacity aligned with the repo policy and queue state:

```bash
UV_CACHE_DIR=.uv-cache uv run rlab-fleet policy
UV_CACHE_DIR=.uv-cache uv run rlab-fleet plan
UV_CACHE_DIR=.uv-cache uv run rlab-fleet reconcile --execute
```

Use `rlab-queue status --goal <goal>` for operational queue state and compact
result receipts. Durable research decisions belong in the repo under the goal
folder; detailed metrics and artifacts belong in W&B.

To stop a running train job without dropping its latest weights, request a
queue cancel:

```bash
UV_CACHE_DIR=.uv-cache uv run rlab-queue cancel-train <train_job_id>
```

The train runner relays that request to the leased trainer with `SIGUSR1`. The
trainer stops at the next callback step, saves an interrupted step checkpoint,
uploads it through the usual checkpoint artifact path, saves/uploads the final
model with an `interrupted` alias, and then exits. If the trainer does not exit
within the runner's cancel grace window, the runner falls back to `SIGTERM`.

For a quick terminal monitor over queue jobs and fleet state:

```bash
UV_CACHE_DIR=.uv-cache uv run rlab-monitor --view all
UV_CACHE_DIR=.uv-cache uv run rlab-monitor --json
```

## Queue Runners

Queue-backed training is the supported GPU workflow. Create train jobs with
`rlab-queue`, then run capacity through `rlab-fleet` on `beast-3` and
`beast-2`.

For queue-backed training, keep worker capacity in `experiments/fleet.json` and
`experiments/policies/capacity_policy.json`. `rlab-fleet` starts digest-pinned
Docker containers running `rlab.train_runner`; experiment payloads stay in the
queue row snapshot loaded from the checked-in spec file.

For local GPU queue capacity, run the fleet manager from the MacBook. It reads
pending/running `train_jobs`, groups demand by `profile_id`,
`runtime_image_ref`, and `run_target`, then reconciles Docker runner containers
on `beast-3` and `beast-2` over SSH. The beast hosts are intentionally just
Docker engines; they do not poll the queue or run a local fleet service.

```bash
UV_CACHE_DIR=.uv-cache uv run rlab-fleet status
UV_CACHE_DIR=.uv-cache uv run rlab-fleet ps
UV_CACHE_DIR=.uv-cache uv run rlab-fleet plan
UV_CACHE_DIR=.uv-cache uv run rlab-fleet reconcile --execute
UV_CACHE_DIR=.uv-cache uv run rlab-fleet reconcile --execute --watch --interval 30
UV_CACHE_DIR=.uv-cache uv run rlab-fleet watch --execute
```

After publishing a new train image, roll all active beast hosts to the latest
successful digest with `ensure-latest`:

```bash
UV_CACHE_DIR=.uv-cache uv run rlab-fleet ensure-latest --execute
```

This starts or keeps one unprofiled latest-image runner per selected host. It
also removes older managed containers when there are no pending/running jobs
matching that old container's profile, digest, and target, and no active worker
lease owned by that container. Add `--host beast-3` to limit a rollout, or
`--watch --interval 30` if you want a long-running local loop that keeps
checking for newly published latest artifacts.

For an operator-friendly live view, prefer `watch`. It continuously
checks configured fleet hosts, keeps one unprofiled latest-image runner alive on
each live host, leaves old runners alone while they still own active leases or
have matching queued/running jobs, marks stale running train jobs failed so they
can stop blocking the queue, removes idle stale managed containers, and shows
the last three published train-image digests with their source commit hashes,
publish times, and commit subjects. Omit `--execute` for a dry-run dashboard.

```bash
UV_CACHE_DIR=.uv-cache uv run rlab-fleet watch --execute
```

To explicitly make a latest-image runner available on a local beast host,
without waiting for queue demand, use `ensure-runner`. When no image ref is
provided it resolves the latest successful `rlab train image` artifact on
`main` through the GitHub CLI and uses the artifact's immutable digest. By
default this runner does not filter by profile; it claims any train job matching
the image and host target:

```bash
UV_CACHE_DIR=.uv-cache uv run rlab-fleet ensure-runner \
  --host beast-3 \
  --image latest \
  --execute
```

`--image latest` is the default; pass `--image docker:...@sha256:...` or
`--image-file rlab-train-image.json` to pin a specific artifact explicitly.
Pass `--profile <profile-id>` only when you intentionally want a runner locked
to one queue lane.

Bootstrap each host once so Docker, the NVIDIA runtime, persistent directories,
the non-secret env-file path, digest pulls, and the container smoke check are
ready:

```bash
UV_CACHE_DIR=.uv-cache uv run rlab-fleet setup-host \
  --host beast-3 \
  --runtime-image-ref-file rlab-train-image.json \
  --execute

UV_CACHE_DIR=.uv-cache uv run rlab-fleet setup-host \
  --host beast-2 \
  --runtime-image-ref-file rlab-train-image.json \
  --execute
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
- Queue-backed train jobs are profileless by default and should reference
  immutable runtime image digests. `rlab-queue enqueue-train` resolves the
  latest successful train-image artifact when no explicit digest/file is given;
  pass `--profile <profile-id>` only for an intentionally profile-locked lane.
- Set `WANDB_API_KEY` for online W&B. For R2/S3-backed reference artifacts, set
  `CHECKPOINT_BUCKET_URI` or pass `--wandb-artifact-storage-uri`, along with the
  required `AWS_*` credentials.
- Keep generated checkpoints, logs, videos, W&B files, caches, and scratch
  outputs out of source control.
- Local eval outputs are written under `runs/local_evals/<run-name>/`.

## Architecture

![rlab architecture diagram](./architecture.png)

## License

No license file is present in this repository.
