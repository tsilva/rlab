<div align="center">
  <img src="./logo.png" alt="rlab" width="256" />

  **Reinforcement-learning workbench for training game agents**
</div>

rlab is a Python CLI for training, evaluating, replaying, and operating reinforcement-learning game agents. It is built around Stable Retro environments, Stable-Baselines3 PPO, W&B artifacts, and queue-backed one-job GPU containers, so a researcher can move from a checked-in experiment recipe to a replayable checkpoint without hand-wiring each step.

The normal workflow is to install the CLI once with `./install.sh`, then use `rlab` commands directly from the repo root. Do not wrap the examples below in `uv run`; the installed tool owns its runtime environment.

## Install

```bash
git clone git@github.com:tsilva/rlab.git
cd rlab
./install.sh
```

If you are reinstalling after local changes:

```bash
./install.sh
```

`install.sh` exports exact constraints from the committed `uv.lock` and installs the
`rlab` command as an editable uv tool. Re-running it preserves the locked dependency
versions and does not modify tracked files.

Refresh the forward Stable Retro and Mario runtimes explicitly when that is intended:

```bash
./refresh-runtimes.sh
```

The refresh command advances the package-specific cutoffs, updates `uv.lock`, keeps
the user-level uv configuration in sync, and then reinstalls the tool.

This repo uses uv's seven-day `exclude-newer` protection, with package-specific exceptions for the exact Stable Retro requirement and minimum Mario runtime recorded in `uv-tool.toml`, `pyproject.toml`, `uv.lock`, and the user-level uv config used by `uv tool install`.

After installation, run commands as plain `rlab ...`:

```bash
rlab --help
rlab validate
```

Import ROMs through the installed CLI so playback and training see them in the same runtime:

```bash
rlab import-roms ~/Desktop/roms --game SuperMarioBros-Nes-v0
```

## Run

Start with a queue-backed local smoke run:

```bash
rlab train \
  --recipe-file experiments/goals/SuperMarioBros-Nes-v0/Level1-1/recipes/base.yaml \
  --run-target local-macbook \
  --set recipe_id=local-smoke \
  --set group_id=local-smoke \
  --set train.policy.timesteps=512 \
  --set train.environment.env_config.n_envs=1 \
  --set logging.wandb=false \
  --set logging.wandb_mode=disabled \
  --set logging.wandb_artifact_storage_uri=

rlab fleet shepherd --machine local-macbook --limit 1 --once
```

Inspect the resulting launch and `result.json`:

```bash
rlab fleet watch --machine local-macbook --once --no-tui
```

Queue comparable experiments from checked-in recipe files:

```bash
rlab train \
  --recipe-file experiments/goals/<goal-slug>/recipes/<recipe>.yaml \
  --runtime-image-ref-file rlab-train-image.json
```

For short-lived queue-backed ablations, keep the checked-in recipe as the base and
apply repeatable Hydra/OmegaConf dotlist overrides from the CLI:

```bash
rlab train \
  --recipe-file experiments/goals/SuperMarioBros-Nes-v0/Level1-1/recipes/base.yaml \
  --set recipe_id=lr2e4 \
  --set group_id=Level1-1-lr2e4 \
  --set train.policy.learning_rate=2e-4 \
  --seed 1
```

Overrides are recorded in the queued recipe payload and W&B config as
`recipe_overrides`. Give each comparable variant a distinct `recipe_id` so
leaderboards do not mix sweep arms.

If `rlab-train-image.json` is absent, omit `--runtime-image-ref-file` and `rlab train` will resolve the latest successful train-image artifact by default.

## Commands

```bash
rlab validate                                      # validate goals, recipes, benchmarks, and machine config
rlab train --recipe-file experiments/goals/<goal-slug>/recipes/<recipe>.yaml
rlab eval --game <GameId> --policy random --episodes 2 --max-steps 600
rlab play <run-name>                                  # installed CLI; works outside this checkout
rlab play <entity>/<project>/<run-name>-checkpoint:latest
rlab play hf://tsilva/SuperMarioBros-NES_Level1-2     # download and play from Hugging Face
rlab play <checkpoint> --step-over
rlab play <checkpoint> --attribution gradcam
rlab play <checkpoint> --attribution occlusion --attribution-interval 12
rlab jobs status --goal <goal-slug>
rlab leaders runs --goal <goal-slug> --min-seeds 3
rlab leaders checkpoints --goal <goal-slug>
rlab leaders checkpoints --goal <goal-slug> --limit 1 --json
rlab jobs cancel-train <train_job_id>
rlab fleet shepherd --machine beast-3 --once
rlab fleet watch --machine beast-3
rlab benchmark list
rlab benchmark run retro-env-throughput-mario-l11 --dry-run
```

The command surface is intentionally one binary:

- `rlab train` enqueues queue-backed train jobs from checked-in recipes.
- `rlab eval` runs local/scripted or explicit-model evaluation. Queue-backed train jobs evaluate saved checkpoints asynchronously, and async checkpoint eval is the supported checkpoint-promotion path.
- `rlab play` replays a local model path, W&B checkpoint artifact, or Hugging Face model repo.
- `rlab jobs` and `rlab fleet` operate and inspect the queue and one-job container fleet.
- `rlab leaders` queries W&B for run/recipe winners and best evaluated checkpoints.
- `rlab benchmark` runs named smoke, throughput, fleet, and eval-contract profiles.

Specialized and maintenance commands are intentionally outside the normal research loop:

- `rlab distill` contains Mario-specific behavior-cloning and sequence-evaluation experiments.
- `rlab jobs reset-schema --dry-run` previews an administrative queue-schema export and reset; rerun without `--dry-run` only when that destructive operation is intended.

## Research Loop

Active research contracts live under `experiments/goals/`. For current Mario work, read the goal's `_goal.yaml` before choosing recipes, caps, metrics, or promotion criteria.

Train recipes are validated against the queue-backed schema before enqueue. Extra research metadata is preserved, but required launch, naming, W&B, seed, selection, and train-config fields must be present and well-formed.

Promotion compares checkpoints by per-start completion minimum, then per-start completion mean, then least checkpoint timesteps once the completion goal is met, then eval reward. W&B is the source of truth for run and eval metrics; the queue database stores train-job state.

To ask for the current best evaluated checkpoint for a goal, query the checkpoint leaders with a
single-row limit:

```bash
rlab leaders checkpoints --goal Level1-1 --limit 1 --json
```

`leaders checkpoints` returns evaluated checkpoint rows already sorted by the checkpoint promotion
order, so `--limit 1` is the canonical best-checkpoint query. Use `leaders runs` separately when
the question is about training/recipe winners rather than the checkpoint artifact to play or promote.
`leaders runs` uses the current primary goal metric by default for fast W&B queries.

To regenerate the W&B checkpoint leaderboard report with one section per goal, run:

```bash
UV_CACHE_DIR=.uv-cache uv run --with 'wandb[workspaces]' --exclude-newer 2026-06-25T00:00:00Z python scripts/create_wandb_checkpoint_leaderboard_report.py
```

Learning-oriented standalone examples live under `examples/learning/`; production training should use checked-in recipes through `rlab train`.

## Fleet

Queue-backed training is the supported GPU workflow. `rlab train` creates train
jobs, and `rlab fleet shepherd --machine <name>` reconciles digest-pinned,
one-job Docker containers on registered local or SSH Docker machines.

```bash
rlab fleet shepherd --machine beast-3 --once
```

Hard fleet capacity and target mapping come from `experiments/machines.yaml`.
`INSTANCES.md` is the authoritative operator guide for shepherd operation,
status and watch commands, host setup, cleanup, hardware targets, concurrency,
and beast host recommendations.

## Notes

- Python is pinned to `==3.14.*`; dependency resolution and lock state are managed by `uv`.
- The installed console command is `rlab`; examples should not use `uv run`.
- Runtime support is pinned for macOS arm64 and Linux x86_64 with `stable-retro-turbo`.
- Stable Retro matches ROMs by SHA, not filename. Import ROMs with `rlab import-roms` for the game ids you train or play.
- Every queue-backed training recipe must include a non-empty `description`; `rlab train` records it as the run description.
- Training logs to W&B and uploads model artifacts unless the recipe sets
  `logging.no_wandb_artifacts: true` (or `--set logging.no_wandb_artifacts=true`).
- Queue-backed train jobs are profileless by default and should reference immutable runtime image digests.
- Set `WANDB_API_KEY` for online W&B. For R2/S3-backed reference artifacts, set
  `CHECKPOINT_BUCKET_URI` or configure `logging.wandb_artifact_storage_uri` in the recipe,
  along with the required `AWS_*` credentials.
- Keep generated checkpoints, logs, videos, W&B files, caches, and scratch outputs out of source control.
- Local eval outputs are written under `runs/local_evals/<run-name>/`.

## License

No license file is present in this repository.
