<div align="center">
  <img src="./logo.png" alt="rlab" width="256" />

  **Reinforcement-learning workbench for training game agents**
</div>

rlab turns checked-in goals and recipes into portable, reproducible training
runs. Research code sees one container and does not contain provider-specific
scheduling logic. dstack places that container on a local GPU, spot instance,
or explicitly authorized on-demand machine; Modal evaluates immutable
checkpoints independently.

## Architecture

| Concern | Authority |
| --- | --- |
| Provisioning, placement, logs, cancellation, interruption retries | dstack |
| Run, attempt, lease, promotion, and terminal state | private control R2 |
| Evaluation intents, results, evidence, and videos | private eval R2 |
| Training and evaluation metrics | one W&B run |
| Downloadable checkpoints and public run index | public model R2 |
| Temporary event buffering | SQLite WAL in the training container |

One supervisor inside the training container is the only W&B process. The
learner performs no network I/O: it emits local metric and checkpoint events
and responds to a cooperative stop signal. The supervisor uploads and verifies
checkpoints, dispatches Modal evaluations, observes accepted results, signals
the learner at a safe boundary, and drains all frozen work before the task can
succeed.

dstack exit status alone never establishes scientific success. The
authoritative terminal receipt must prove complete checkpoint/evaluation
inventories, promotion, W&B delivery, and drain.

## Install

```bash
git clone git@github.com:tsilva/rlab.git
cd rlab
./install.sh
rlab validate
```

The project uses `uv`, a committed `uv.lock`, and a seven-day package-age gate.
`stable-retro-turbo`, `supermariobrosnes-turbo`, and
`breakout-turbo-env` have explicit forward-runtime exceptions recorded in
`pyproject.toml` and `uv-tool.toml`.

Register a local ROM without uploading it to source control:

```bash
rlab rom sync --game SuperMarioBros-Nes-v0
rlab rom status --json
```

Local dstack hosts use the hash-verified read-only ROM cache. Each Modal-backed
run stages its exact ROM bytes and manifest to eval-private R2. ROMs, R2
credentials, W&B credentials, and Modal credentials are never embedded in the
image.

## Launch and observe

```bash
rlab experiment launch \
  --goal-file experiments/goals/SuperMarioBros-Nes-v0/Level1-1/_goal.yaml \
  --recipe-file experiments/goals/SuperMarioBros-Nes-v0/Level1-1/recipes/ppo.yaml \
  --seed 123 \
  --run-description "Mario Level1-1 PPO seed 123" \
  --compute local \
  --target b3 \
  --max-duration 48h \
  --json
```

`launch` requires a clean, pushed source revision and waits for its verified
immutable training image and source-specific Modal deployment. It returns the
rlab run ID, attempt ID, dstack task, selected compute offer, source/image
digest, W&B URL, and public run-index URL.

Compute policy:

- `auto` uses an idle compatible local host first.
- `spot` requires finite `--max-price` and `--max-cost-usd`.
- without a cloud budget, `auto` waits for local capacity.
- `on-demand` also requires `--allow-on-demand`.
- every task requires a finite maximum duration.

Observe or control one logical run:

```bash
rlab experiment status --run <rlab-run-id> --json
rlab experiment follow --run <rlab-run-id>
rlab experiment wait --run <rlab-run-id> --until terminal --timeout 48h
rlab experiment logs --run <rlab-run-id> --tail 200
rlab experiment cancel --run <rlab-run-id>
rlab experiment retry --run <rlab-run-id>
```

Retry preserves the logical run ID and creates a new attempt ID. It requires a
terminal prior dstack attempt, an expired writer lease, and a 30-second
quiescence interval. A run whose learner already finished resumes in
drain-only mode and cannot retrain.

For a short B3 integration smoke, use the checked-in
`experiments/goals/SuperMarioBros-Nes-v0/Level1-1/recipes/dstack-smoke.yaml`
recipe. Repeatable `--set KEY=VALUE` ablations are composed, validated, and
included in the immutable portable recipe hash; use a checked-in leaf recipe
for durable or shared variants.

## Checkpoints and playback

Periodic and final checkpoints are immutable:

```text
runs/<run-id>/checkpoints/<step>-<sha256>/model.zip
runs/<run-id>/checkpoints/<step>-<sha256>/manifest.json
runs/<run-id>/index.json
```

The public index is mutable through ETag compare-and-swap and served with
`Cache-Control: no-store`; checkpoint objects are immutable and cacheable.
Playback needs no private credentials:

```bash
rlab play --run <rlab-run-id>
rlab play <local-checkpoint>
rlab play hf://<owner>/<repository>
```

W&B contains metrics, hashes, metadata, and R2 URLs only. Model bytes, videos,
episode evidence, ROMs, and recovery journals remain in R2.

## Evaluation and early stop

Every ready periodic checkpoint and the natural final checkpoint is evaluated
against the immutable goal-owned episode manifest. Modal validates the
checkpoint, goal, recipe, environment, protocol, and episode-manifest hashes.
Acceptance fails fast on the first valid failed episode and requires all 100
episodes to pass.

The supervisor polls results every two seconds. An accepted result requests
learner stop within ten seconds; the learner stops cooperatively at a safe
on-policy boundary and saves a final checkpoint. The ready set is then frozen,
every member reaches a terminal eval state, and the lowest-step accepted
checkpoint is promoted exactly once.

## Goals, recipes, metrics, and reports

- Active goals: `experiments/goals/`
- Goal-local launchable recipes: each goal’s `recipes/`
- Reusable presets: `experiments/recipes/_presets/`
- Metric contract: `METRICS.md`
- Hardware and dstack operations: `INSTANCES.md`
- Control-plane units and templates: `ops/dstack/`

Useful commands:

```bash
rlab validate
rlab env list
rlab env inspect supermariobrosnes-turbo:SuperMarioBros-Nes-v0
rlab env preflight \
  --goal-file experiments/goals/SuperMarioBros-Nes-v0/Level1-1/_goal.yaml \
  --recipe-file experiments/goals/SuperMarioBros-Nes-v0/Level1-1/recipes/ppo.yaml
rlab leaders runs --goal SuperMarioBros-Nes-v0/Level1-1 --min-seeds 3
rlab leaders checkpoints --goal SuperMarioBros-Nes-v0/Level1-1 --limit 1 --json
rlab reports plan --goal SuperMarioBros-Nes-v0/Level1-1
rlab reports sync --goal SuperMarioBros-Nes-v0/Level1-1
rlab reports verify --goal SuperMarioBros-Nes-v0/Level1-1
rlab benchmark list
```

## Datasets and model release

`rlab dataset` records and verifies Gymrec v3 gameplay data with provider-native
actions, rewards, boundaries, seeds, environment contracts, and approved policy
provenance:

```bash
rlab dataset record mario-level1-1 \
  --env-id SuperMarioBros-Nes-v0 \
  --provider supermariobrosnes-turbo \
  --agent human
rlab dataset verify mario-level1-1
rlab dataset play mario-level1-1 --episode 1
rlab dataset upload mario-level1-1 <owner/repository>
```

External SB3 checkpoints are Python-executable content. rlab stages and hashes
the complete model closure and requires approval before deserialization unless
the exact source matches `RLAB_MODEL_SOURCE_ALLOWLIST`.

Published model releases use Hugging Face model cards and include a
representative `replay.mp4` when the policy has visual behavior. Generated
artifacts belong under ignored `runs/`, `logs/`, and `models/` directories,
never source control.
