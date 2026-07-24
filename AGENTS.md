# Project Rules

## Product Specifications

Before every task in this repository, use the `$specs-author` skill to read the entire root `SPECS.md`. Before finishing, reread it and check the task and conversation for new or changed stakeholder intent.

- Treat `SPECS.md` as the persistent source of stakeholder requirements that cannot be inferred reliably from code or remembered conversations.
- If the task, repository, or user request contradicts, omits, or ambiguously interprets the specification, tell the user. Continue safe exploration and work that does not depend on resolving the issue, but never silently choose an interpretation.
- Never edit `SPECS.md` from inference. Propose the exact change, explain why it reflects stakeholder intent, and edit the file only after the user explicitly approves that exact change.
- Keep `SPECS.md` complete, concise, and compacted. It must contain stakeholder intent rather than implementation, architecture, operations, or transient project detail.

## GPU Instances

Before choosing hardware, launching training, changing concurrency, or recommending beast targets, read `INSTANCES.md`. It is the source of truth for known GPU instances, access commands, child counts, cleanup, and gotchas. Update it when benchmark or access facts change.

When changing dstack host behavior, preserve the root-owned runtime-image cleanup contract: prune only unused rlab-managed images while preserving active containers and every immutable digest demanded by pending or active dstack tasks.

## Stable Retro

- Use PyPI `stable-retro-turbo`; import path remains `stable_retro`.
- Current required forward runtime is `stable-retro-turbo==1.0.1.post35`.
- Current minimum Mario runtime is `supermariobrosnes-turbo>=0.4.4`.
- Current minimum Breakout runtime is `breakout-turbo-env>=0.4.1`.
- Native-vector code should use `stable_retro.RetroVecEnv`, whose constructor follows the original `RetroEnv` positional signature plus vector-only keyword arguments; do not use the removed `StableRetroNativeVecEnv` name.
- Runtime version source of truth: `pyproject.toml` minimums and the resolved versions in `uv.lock`. Use `uv sync --frozen`; make overrides explicit in recipes, compute policy, run descriptions, and W&B tags.
- Native-vector obs may be channel-last `(n_envs, 84, 84, 4)` or channel-first `(n_envs, 4, 84, 84)`. Detect shape; skip `VecTransposeImage` for channel-first; transpose only channel-last.
- Keep version history and benchmark conclusions in `INSTANCES.md` or experiment reports.

## Training Runs

- When asked to launch, run, start, execute, or monitor a dstack-backed training recipe, use the project-level `$launch-experiment` skill in `.codex/skills/launch-experiment`. It reports the W&B URL immediately, monitors through the authoritative R2 terminal receipt, and delegates active-run potential-bug diagnosis to the read-only `training_run_investigator` custom agent. Its default mode is observation-only; explicit user authorization enables its evidence-preserving repair mode.
- When asked to tune or optimize a checked-in SB3 PPO/A2C recipe for sample efficiency and stability across training seeds, use the project-level `$autoresearch` skill in `.codex/skills/autoresearch`. It runs a bounded training-only 20%/50% fixed-rung search, launches no checkpoint evaluations, confirms the winner from five untouched full-cap training seeds, and patches only the pointed leaf recipe. Its result is training-signal-confirmed, not checkpoint-promoted or goal-accepted.
- Active research goal contracts live under goal-scoped folders in `experiments/goals/`. For current Mario Level1-1 work, read `experiments/goals/SuperMarioBros-Nes-v0/Level1-1/_goal.yaml` before choosing recipes, caps, metrics, or promotion criteria. Seed ranges are owned by `rlab.seeds`, not goal files.
- Launchable recipes live under their owning active goal's `recipes/` directory and may inherit reusable defaults from `experiments/recipes/_presets/`. Legacy goal-local `decisions/`, `reports/`, `best.yml`, and old `experiments/history/` artifacts live under repo-root `.deprecated/` with their source-relative folder structure. That directory is gitignored and should be treated only as historical context about past experiments, not as active contract, recipe, or promotion state.
- Keep generated artifacts out of source control; use `runs/`, `logs/`, and `models/`.
- The training-container supervisor is the sole W&B writer. W&B stores metrics, metadata, hashes, and R2 URLs only; checkpoints, evidence, replays, ROMs, and recovery bytes belong in their scoped R2 buckets.
- Every training run needs a specific description via `--run-description`.
- Training tasks are profileless and locked to exact-source immutable runtime-image digests. One task owns one single-GPU host in v1.
- Logical run IDs are `rlab-<32 lowercase hex>` and attempt IDs are `attempt-<16 lowercase hex>`. Retries preserve the run ID and create a new attempt ID. Do not introduce numeric database job IDs or batch IDs.
- Route W&B projects by canonical game family and keep provider identity and `environment_hash` in config. The W&B run ID and group are the immutable rlab run ID.
- Acceptance evaluation is the only checkpoint-promotion workflow for orchestrated runs. Evaluate every ready periodic checkpoint and the natural final model against the immutable goal-owned episode manifest, fail fast on the first valid failed episode, and atomically promote the lowest-step accepted checkpoint. Modal never receives W&B or control-private credentials.
- dstack task success is not scientific success. Require the private R2 terminal receipt proving complete eval/checkpoint inventories, promotion, W&B high-water delivery, and drain.

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
