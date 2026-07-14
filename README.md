<div align="center">
  <img src="./logo.png" alt="rlab" width="256" />

  **Reinforcement-learning workbench for training game agents**
</div>

rlab is a Python CLI for training, evaluating, replaying, and operating reinforcement-learning game agents. It is built around Stable Retro environments, Stable-Baselines3 actor-critic models, W&B artifacts, and queue-backed one-job GPU containers, so a researcher can move from a checked-in experiment recipe to a replayable checkpoint without hand-wiring each step.

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

Start with the built-in ROM-free native-vector smoke environment:

```bash
rlab env inspect rlab:Bandit-v0
rlab env check \
  --goal-file experiments/goals/rlab__bandit/_goal.yaml \
  --recipe-file experiments/recipes/bandit/ppo.yaml
```

Run its complete queue-backed local training and checkpoint-evaluation path:

```bash
rlab train \
  --goal-file experiments/goals/rlab__bandit/_goal.yaml \
  --recipe-file experiments/recipes/bandit/ppo.yaml \
  --machine local-macbook \
  --checkpoint-eval-backend local \
  --wait terminal \
  --json \
  --set recipe_id=local-smoke \
  --set campaign_id=local-smoke \
  --set logging.wandb=false \
  --set logging.wandb_mode=disabled \
  --set logging.wandb_artifact_storage_uri=
```

`rlab:Bandit-v0` has no renderer. Replay it headlessly with
`rlab play <checkpoint> --debug` for interactive stepping or add a positive `--episodes` limit
for unattended playback.

For a ROM-backed Mario smoke run, use the same queue path with explicit smoke overrides:

```bash
rlab train \
  --goal-file experiments/goals/SuperMarioBros-Nes-v0/Level1-1/_goal.yaml \
  --recipe-file experiments/recipes/mario/single/ppo.yaml \
  --machine local-macbook \
  --checkpoint-eval-backend none \
  --wait terminal \
  --json \
  --set recipe_id=local-mario-smoke \
  --set campaign_id=local-mario-smoke \
  --set train.timesteps=512 \
  --set train.environment.env_config.n_envs=1 \
  --set logging.wandb=false \
  --set logging.wandb_mode=disabled \
  --set logging.wandb_artifact_storage_uri=
```

Inspect the resulting launch and `result.json`:

```bash
rlab jobs status --machine local-macbook --json
```

Queue comparable experiments from checked-in recipe files:

```bash
rlab train \
  --goal-file experiments/goals/<goal-slug>/_goal.yaml \
  --recipe-file experiments/recipes/<family>/<recipe>.yaml \
  --machine beast-3 \
  --runtime-image-ref-file rlab-train-image.json
```

For short-lived queue-backed ablations, compose the checked-in goal and recipe and
apply repeatable Hydra/OmegaConf dotlist overrides from the CLI:

```bash
rlab train \
  --goal-file experiments/goals/SuperMarioBros-Nes-v0/Level1-1/_goal.yaml \
  --recipe-file experiments/recipes/mario/single/ppo.yaml \
  --machine beast-3 \
  --set recipe_id=lr2e4 \
  --set campaign_id=Level1-1-lr2e4 \
  --set train.backend.config.learning_rate=2e-4 \
  --seed 1
```

Overrides are recorded in the queued recipe payload and W&B config as
`recipe_overrides`. Give each comparable variant a distinct `recipe_id` so
leaderboards do not mix sweep arms. Each submission receives one generated
`bx<16 hex>` `batch_id`, shared by all of its seeds and used as the W&B group.
Use optional `campaign_id` to connect related submissions over time.

If `rlab-train-image.json` is absent, omit `--runtime-image-ref-file`. `rlab train` pins the clean pushed commit and waits for its exact-source receipt. The receipt may reuse an immutable image when the content-addressed runtime inputs match a prior source state; it never falls back without that proof. The exact source, recipe composition, runtime fingerprint, runtime build source, and digest remain recorded.

## Commands

```bash
rlab validate                                      # validate goals, recipes, benchmarks, and machine config
rlab env list                                      # list declared providers and environments without importing them
rlab env inspect rlab:Bandit-v0
rlab env check --goal-file experiments/goals/rlab__bandit/_goal.yaml --recipe-file experiments/recipes/bandit/ppo.yaml
rlab env inspect supermariobrosnes-turbo:SuperMarioBros-Nes-v0
rlab env check --goal-file experiments/goals/SuperMarioBros-Nes-v0/Level1-1/_goal.yaml --recipe-file experiments/recipes/mario/single/ppo.yaml
rlab train --goal-file experiments/goals/<goal-slug>/_goal.yaml --recipe-file experiments/recipes/<family>/<recipe>.yaml --machine beast-3
rlab eval --game <GameId> --policy random --episodes 2 --max-steps 600
rlab play <run-name>                                  # resolves the promoted checkpoint; never moving :latest
rlab play <entity>/<project>/rlab-<run-id>-checkpoint:latest
rlab play hf://tsilva/NES-SuperMarioBros_Level1-2_gray84-hudcrop-stack4-simple_ppo
rlab play <checkpoint> --debug                       # Enter steps once; use help for commands
rlab play <checkpoint> --attribution gradcam
rlab play <checkpoint> --attribution occlusion --attribution-interval 12
rlab jobs status --machine beast-3 --json
rlab jobs wait --job <train-job-id> --until terminal --timeout 12h --json
rlab jobs cancel --job <train-job-id> --wait --json
rlab jobs logs --job <train-job-id> --follow
rlab leaders runs --goal <goal-slug> --min-seeds 3
rlab leaders checkpoints --goal <goal-slug>
rlab leaders checkpoints --goal <goal-slug> --limit 1 --json
rlab fleet drain --machine beast-3
rlab fleet resume --machine beast-3
rlab fleet service status --json
rlab eval modal status
rlab benchmark list
rlab benchmark run retro-env-throughput-mario-l11 --dry-run
```

The command surface is intentionally one binary:

- `rlab train` enqueues queue-backed train jobs from checked-in recipes.
- `rlab eval` runs local/scripted or explicit-model evaluation. Queue-backed train jobs evaluate saved checkpoints asynchronously; jobs materialized for Modal use bounded remote CPU workers, while direct training and explicit `rlab eval` stay local.
- `rlab play` replays a local model path, W&B checkpoint artifact, or Hugging Face model repo.
- `rlab env` lists static provider contracts, inspects one qualified environment, or explicitly
  preflights a materialized recipe against the installed native runtime.
- `rlab jobs` and `rlab fleet` operate and inspect the queue and one-job container fleet.
- `rlab leaders` queries W&B for run/recipe winners and best evaluated checkpoints.
- `rlab benchmark` runs named smoke, throughput, fleet, and eval-contract profiles.

`rlab env list` and `rlab env inspect` are static and do not import provider modules or access
ROMs. `rlab env check` is an explicit, recipe-backed behavioral probe: it constructs the same
native provider used by training, checks vector reset/step and visible masked-reset preservation,
then binds and steps the normal rlab runtime. Its report separates declared configuration,
runtime-observed evidence, and hidden reset invariants backed by the pinned provider contract;
it does not claim that emulator or RNG internals are black-box observable. Pass `--json` for one
versioned report on stdout; provider diagnostics are routed to stderr.

Maintenance commands are intentionally outside the normal research loop:

- `rlab jobs reset-schema --dry-run` previews an administrative queue-schema export and reset; rerun without `--dry-run` only when that destructive operation is intended.

## Research Loop

Active research contracts live under `experiments/goals/`. For current Mario work, read the goal's `_goal.yaml` before choosing recipes, caps, metrics, or promotion criteria.

Train recipes are validated against the queue-backed schema before enqueue. Extra research metadata is preserved, but required launch, naming, W&B, seed, selection, and train-config fields must be present and well-formed.

Promotion compares checkpoints by per-start completion minimum, then per-start completion mean, then least checkpoint timesteps once the completion goal is met, then eval reward. For local evaluation, W&B remains the query projection. For Modal evaluation, immutable R2 evidence plus the accepted PostgreSQL attempt is authoritative; W&B is projected into the exact producing run only after its live publisher exits.

`rlab jobs status --job <id> --json` reports `eval_status`, `promoted_step`,
`artifact_status`, `artifact_ref`, and `published_at`. Modal-backed enqueue fails closed when the
fleet evaluator's latest pass is stale or unsuccessful. Projection failures retry with backoff and
are isolated to their producing run; after correcting a persistent failure, reset only that run
with `rlab eval modal retry-projection <train-job-id>`.

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

## Publish a Policy

Policies publish under `tsilva` with a generated repository identity:

```text
<game-family>_<goal>_<policy-variant>_<algorithm>
```

For example:

```text
tsilva/NES-SuperMarioBros_Level1-1_gray84-hudmask-stack4-simple_ppo
```

The game family comes from rlab's provider-neutral registry, the policy variant comes from the
saved observation/action contract, and the algorithm comes from checkpoint metadata. Provider,
run, recipe, seed, runtime, and environment hash remain in `model_metadata.json` and
`release_manifest.json`; they are not manually encoded in the repository name.

Use the project `$upload-checkpoint` skill for releases. Before any Hub mutation, preview the exact
identity with the repository-owned release helper:

```bash
uv run python scripts/prepare_huggingface_release.py \
  --goal-file experiments/goals/SuperMarioBros-Nes-v0/Level1-1/_goal.yaml \
  --model-metadata runs/<run>/checkpoints/<checkpoint>.metadata.json \
  --identity-only
```

The complete workflow requires stochastic evaluation evidence, a browser-safe `replay.mp4`, the
matching public YouTube preview, and the exact seven-file Hugging Face release bundle. The helper
rejects unknown families, inconsistent model classes, manual names, non-portable paths, extra
files, and invalid hashes before upload.

`README.md` is generated from the same manifest, metadata, and evaluation evidence as the release;
it is not a separate hand-maintained input. After publishing and tagging a release, audit the live
Hub state through the API:

```bash
uv run python scripts/audit_huggingface_release.py \
  tsilva/NES-SuperMarioBros_Level1-1_gray84-hudmask-stack4-simple_ppo \
  --revision v1
```


## Fleet

Queue-backed training is the supported GPU workflow. Every `rlab train` job
names one registered machine. A single Mac-side launchd service performs short,
bounded reconciliation passes for digest-pinned, one-job Docker containers on
registered local or SSH Docker machines. Runner machines remain SSH/Docker-only.

```bash
rlab fleet service install
rlab fleet service status --json
rlab jobs status --machine beast-3 --json
```

Hard fleet capacity and exact machine configuration come from
`experiments/machines.yaml`. `INSTANCES.md` is the authoritative operator guide
for service operation, job status, host setup, cleanup, hardware, concurrency,
and beast host recommendations.

## Notes

- Python is pinned to `==3.14.*`; dependency resolution and lock state are managed by `uv`.
- The installed console command is `rlab`; examples should not use `uv run`.
- Runtime support is pinned for macOS arm64 and Linux x86_64 with `stable-retro-turbo`.
- Stable Retro matches ROMs by SHA, not filename. Import ROMs with `rlab import-roms` for the game ids you train or play.
- Every queue-backed training recipe must include a non-empty `description`; `rlab train` records it as the run description.
- Queue-backed W&B run names are `<batch_id>-<recipe_id>-s<effective_seed>-<utc>` and use an opaque `rlab-...` run id. Projects identify canonical game families, while config/tags identify the goal, recipe, provider, campaign, and exact environment hash.
- Canonical W&B projects are `SuperMarioBros-Nes-v0` for SMB1, `SuperMarioBros3-Nes-v0` for SMB3, `Breakout-Atari2600-v0` for ALE or Stable Retro Breakout, and `MsPacman-Atari2600-v0` for ALE or Stable Retro Ms. Pac-Man. An explicit `wandb_project` still overrides this routing; unknown environments use their provider-local environment id.
- New W&B model artifact collections and R2/S3 object paths use the immutable run id. Playback continues to resolve historical display-name artifacts and the legacy `breakout` and `ms_pacman` projects.
- Training logs to W&B and uploads model artifacts unless the recipe sets
  `logging.no_wandb_artifacts: true` (or `--set logging.no_wandb_artifacts=true`).
- Queue-backed train jobs are profileless by default and should reference immutable runtime image digests.
- Modal checkpoint evaluation is configured in `experiments/modal_eval.yaml`, uses PostgreSQL as
  its only wait queue, and is the default for newly enqueued queue-backed jobs at effective
  capacity 1. Use `rlab eval modal smoke-local` for the credential-free integration path.
- The train-image workflow publishes the exact immutable image receipt immediately after the image
  exists, then deploys and startup-probes the digest-specific Modal evaluator and publishes a
  separate Modal readiness receipt. Local and no-eval submissions do not wait for Modal. Modal-backed
  submissions also run the live schema, capacity, asset, and deployment preflight before inserting
  queue rows. Use `rlab eval modal preflight --runtime-image-ref <digest-ref> --game <game-id>` for
  operator diagnosis, `--checkpoint-eval-backend local` for the explicit local fallback, or
  `--checkpoint-eval-backend none` only for a non-promotable smoke/debug run.
- Set `WANDB_API_KEY` for online W&B. For R2/S3-backed reference artifacts, set
  `CHECKPOINT_BUCKET_URI` or configure `logging.wandb_artifact_storage_uri` in the recipe,
  along with the required `AWS_*` credentials.
- Every normal queue-backed screen evaluation captures a lightweight policy-observation preview,
  stores its immutable MP4 in R2, and exposes an external player as `eval/screen/preview` in the
  producing W&B run. Set `MODAL_EVAL_PREVIEW_STORAGE_URI` to the public R2 bucket/prefix and
  `MODAL_EVAL_PREVIEW_PUBLIC_BASE_URL` to the matching HTTPS base URL. Configure that bucket to
  serve `video/mp4` with byte ranges and allow the W&B application domain through CORS; Modal
  preflight fails closed when this preview path is not ready.
- Keep generated checkpoints, logs, videos, W&B files, caches, and scratch outputs out of source control.
- Local eval outputs are written under `runs/local_evals/<run-name>/`.

## License

No license file is present in this repository.
