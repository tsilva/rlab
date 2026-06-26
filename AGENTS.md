# Project Rules

## GPU Instances

Before choosing hardware, launching remote training, changing concurrency, or recommending SkyPilot targets, read `INSTANCES.md`. It is the source of truth for known GPU instances, access commands, child counts, `env_threads`, cleanup, and gotchas. Update it when benchmark or access facts change.

## Stable Retro

- Use PyPI `stable-retro-turbo`; import path remains `stable_retro`.
- Runtime pin source of truth: `pyproject.toml` and `uv.lock`. Use `uv sync --frozen`; make overrides explicit in manifests, run descriptions, and W&B tags.
- Native-vector obs may be channel-last `(n_envs, 84, 84, 4)` or channel-first `(n_envs, 4, 84, 84)`. Detect shape; skip `VecTransposeImage` for channel-first; transpose only channel-last.
- Keep version history and benchmark conclusions in `INSTANCES.md` or experiment reports.

## Training Runs

- Keep generated artifacts out of source control; use `runs/`, `logs/`, and `models/`.
- Log to W&B and upload checkpoint/final artifacts unless explicitly opted out.
- Every run needs a specific description via `--run-description` or Modal `run_description`.
- Use run names shaped as `<batch>_<scope>_<arm>_s<seed>_<utc>`, for example `b58_l11_lowkldecay_s108_20260623T142700Z`. Keep target/scope separate from the arm/recipe unless the recipe is inherently target-specific.
- Do not run robust evals inside remote training by default. Evaluate checkpoints out of process; promote by completion rate, then mean reward, then max x-position.
- Default Modal shape unless overridden: `cpu=16.0`, `memory=32768`, `gpu=T4`, `n_envs=32`, `env_threads=0`, `torch_num_threads=0`, `n_steps=512`, `batch_size=256`, `n_epochs=10`.

## Metrics

- `METRICS.md` is the source of truth for W&B metric names and semantics.
- When adding, removing, renaming, or changing the meaning of a logged metric, update `METRICS.md` in the same change.
- When the user asks a metric question and the answer is not already clear from `METRICS.md`, improve `METRICS.md` with that clarification before finishing.

## Model Cards

- When asked to upload, publish, release, or promote a trained checkpoint/model, use the project-level `$upload-checkpoint` composite skill in `.codex/skills/upload-checkpoint`. It coordinates Hugging Face model-card publishing with `$model-card-author` and YouTube preview upload with `$upload-youtube-video`.
- Published model cards should include a preview video when the model has a visual or interactive behavior. For Stable Retro policies, record a representative completed episode, upload it with the model files as `replay.mp4` so Hugging Face's reinforcement-learning widget can find it, embed it near the top of the README, and include the seed/training-metadata caveats.
- For uploading, updating, or troubleshooting YouTube model-preview videos, use the project-level `$upload-youtube-video` skill in `.codex/skills/upload-youtube-video`. Encode future YouTube upload and description-rule changes in that skill first.

## Autoresearch

When the user gives a game plus target and asks Codex to find a reproducible model-training recipe, use the project-level `$autoresearch` skill in `.codex/skills/autoresearch`. That workflow is RTX4090-only, allows only reward-function and hyperparameter changes by default, and requires three fresh successful seeds before declaring the target solved.

## SkyPilot Mario Monitoring

When the user asks to poll, monitor, summarize, or conditionally continue rlab Mario PPO SkyPilot training jobs, use the project-level `$skypilot-mario-monitor` skill in `.codex/skills/skypilot-mario-monitor`. That workflow standardizes SkyPilot queue checks, W&B per-seed metric collection, success-criterion gating, and avoiding accidental RTX4090 oversubscription.

## Eval Queue

When the user asks to flush unevaluated checkpoints, evaluate pending checkpoints, run the Modal eval queue, or produce an eval database report, use the project-level `$flush-eval` skill in `.codex/skills/flush-eval`. Its default eval profile is Level 1 with no terminal-on-life.

## Remote Provider Task Retrospective

After remote-provider monitoring, orchestration, benchmarking, or launches, include a short retrospective: avoidable agent-token spend, reproducibility choices worth encoding, and any useful follow-up to ask about.

## Dependencies

Use `uv` for dependency resolution and keep `uv.lock` committed. Preserve Python supply-chain hardening in `pyproject.toml`.
