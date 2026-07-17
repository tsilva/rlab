# Project Rules

## Product Specifications

Before any task in this repository, use the `$specs-author` skill to read the root `SPECS.md`. Use `$specs-author` whenever reading or writing `SPECS.md`.

- Do not knowingly make or recommend a change that violates an applicable requirement in `SPECS.md`.
- If an existing spec violation is found, the requested work would infringe a requirement, `SPECS.md` is missing when one should exist, or a requirement is missing or unclear, report the issue and propose a scoped rectification or clarification task.
- Ask for explicit user permission before editing `SPECS.md` or starting any rectification or clarification task that is not already explicitly requested. Continue the original task only when it can be completed without violating the current spec; otherwise stop and request direction.

## GPU Instances

Before choosing hardware, launching training, changing concurrency, or recommending beast targets, read `INSTANCES.md`. It is the source of truth for known GPU instances, access commands, child counts, cleanup, and gotchas. Update it when benchmark or access facts change.

When running or changing fleet shepherd behavior, make unused host runtime-image cleanup part of the reconciliation contract: after jobs stop using an image, the shepherd should prune stale Docker images from the host while preserving active containers and currently demanded immutable runtime image digests.

## Stable Retro

- Use PyPI `stable-retro-turbo`; import path remains `stable_retro`.
- Current required forward runtime is `stable-retro-turbo==1.0.1.post30`.
- Current minimum Mario runtime is `supermariobrosnes-turbo>=0.3.1`.
- Native-vector code should use `stable_retro.RetroVecEnv`, whose constructor follows the original `RetroEnv` positional signature plus vector-only keyword arguments; do not use the removed `StableRetroNativeVecEnv` name.
- Runtime version source of truth: `pyproject.toml` minimums and the resolved versions in `uv.lock`. Use `uv sync --frozen`; make overrides explicit in recipes, fleet policy, run descriptions, and W&B tags.
- Native-vector obs may be channel-last `(n_envs, 84, 84, 4)` or channel-first `(n_envs, 4, 84, 84)`. Detect shape; skip `VecTransposeImage` for channel-first; transpose only channel-last.
- Keep version history and benchmark conclusions in `INSTANCES.md` or experiment reports.

## Training Runs

- When asked to launch, run, start, execute, or monitor a queue-backed training recipe, use the project-level `$launch-training-recipe` skill in `.codex/skills/launch-training-recipe`. It defaults to `beast-3`, reports the W&B URL as soon as it appears, monitors compactly, and delegates potential-bug diagnosis to the read-only `training_run_investigator` custom agent without applying fixes.
- Active research goal contracts live under goal-scoped folders in `experiments/goals/`. For current Mario Level1-1 work, read `experiments/goals/SuperMarioBros-Nes-v0/Level1-1/_goal.yaml` before choosing recipes, caps, metrics, or promotion criteria. Seed ranges are owned by `rlab.seeds`, not goal files.
- Legacy goal-local `decisions/`, `recipes/`, `reports/`, `best.yml`, and old `experiments/history/` artifacts live under repo-root `.deprecated/` with their source-relative folder structure. That directory is gitignored and should be treated only as historical context about past experiments, not as active contract, recipe, or promotion state.
- Keep generated artifacts out of source control; use `runs/`, `logs/`, and `models/`.
- Log to W&B and upload checkpoint/final artifacts unless explicitly opted out.
- Every training run needs a specific description via `--run-description`.
- Queue-backed train jobs are profileless: do not pass or persist a `profile_id`. Lock train jobs to immutable runtime image digests instead, resolving to the latest successful train image by default when no digest is specified.
- Use queue-backed run names shaped as `<batchid>-<shortdescription>-s<seed>-<utc>`, for example `bx0123456789abcdef-b55reval-s6-20260702T150934Z`. The queue generates one immutable `bx<16 hex>` batch per submission and uses it as the W&B group; checked-in `campaign_id` is optional cross-submission research lineage. Keep goal, provider, target/scope, runtime versions, and long recipe context in W&B config, tags, descriptions, and recipe metadata rather than the run name.
- Route W&B projects by canonical game family, keep provider identity and `environment_hash` in config, and name new checkpoint/final artifact collections from the immutable `rlab-...` run id. Preserve legacy project and display-name lookup for playback.
- Acceptance evaluation is the only supported checkpoint-promotion workflow for new queue-backed runs. Evaluate each periodic checkpoint and the natural final model once against the immutable goal-owned episode manifest, fail fast on the first valid failed episode, and atomically promote the first checkpoint with complete accepted evidence. W&B publication, preview generation, and duplicate promotion evals must remain outside the learner stop-critical path.

## Metrics

- `METRICS.md` is the source of truth for W&B metric names and semantics.
- When adding, removing, renaming, or changing the meaning of a logged metric, update `METRICS.md` in the same change.
- When touching metric logging, dashboards, reports, eval summaries, or answering metric semantics questions, audit the relevant emitted metric names/templates against `METRICS.md` and patch any missing or stale entries before finishing.
- When the user asks a metric question and the answer is not already clear from `METRICS.md`, improve `METRICS.md` with that clarification before finishing.

## Model Cards

- When asked to upload, publish, release, or promote a trained checkpoint/model, use the project-level `$upload-checkpoint` composite skill in `.codex/skills/upload-checkpoint`. It coordinates Hugging Face model-card publishing with `$model-card-author` and YouTube preview upload with `$upload-youtube-video`.
- Published model cards should include a preview video when the model has a visual or interactive behavior. For Stable Retro policies, record a representative completed episode and upload it with the model files as root `replay.mp4` so Hugging Face's reinforcement-learning widget can show the page preview; do not also embed the video in the README body unless the widget is unavailable.
- For uploading, updating, or troubleshooting YouTube model-preview videos, use the project-level `$upload-youtube-video` skill in `.codex/skills/upload-youtube-video`. Encode future YouTube upload and description-rule changes in that skill first.

## Dependencies

Use `uv` for dependency resolution and keep `uv.lock` committed. Preserve Python supply-chain hardening in `pyproject.toml`.
The intentional exceptions to the seven-day `exclude-newer` window are `breakout-turbo-env`, `stable-retro-turbo`, and `supermariobrosnes-turbo`, because this project tracks current forward native Breakout, Stable Retro, and Mario runtimes while keeping the rest of the dependency graph age-gated. Keep the per-package cutoffs in `[tool.uv.exclude-newer-package]`, `uv-tool.toml`, and the user-level uv config in sync so `uv tool install . --editable` remains installable without extra flags.
