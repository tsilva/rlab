# Project Rules

## GPU Instances

Before choosing hardware, launching remote training, changing concurrency, or recommending a SkyPilot target, read `INSTANCES.md`. Treat it as the repo-local source of truth for known GPU instances, access commands, benchmark-backed child counts, `env_threads`, cleanup expectations, and operational gotchas. Update `INSTANCES.md` when a benchmark or access fact changes.

## Stable Retro

Use the PyPI `stable-retro-turbo` package as the stable-retro provider. The runtime import path remains `stable_retro`.
As of 2026-06-17, use `stable-retro-turbo==1.0.0.post11` for all future macOS and Linux Mario PPO training runs unless the user explicitly requests a different build. `pyproject.toml` and `uv.lock` are pinned to post11, so SkyPilot jobs using `uv sync --frozen` should install post11 directly.
Do not use `stable-retro-turbo==1.0.0.post5` for `StableRetroNativeVecEnv` training. On 2026-06-14, post5 reproduced single-env behavior but returned empty native-vector `info` dicts on Linux, which zeroed the project reward wrapper.
On 2026-06-15, post6 fixed the empty-info regression but introduced/kept a training-breaking observation aliasing change: with `copy_observations=False`, post6 returns the same mutable observation buffer on reset/step and mutates prior observation references, while post4 returned distinct buffers. This breaks SB3 PPO because `_last_obs` can be mutated by `env.step()` before it is written to the rollout buffer.
On 2026-06-15, a local macOS arm64 audit of `stable-retro-turbo==1.0.0.post7` showed the SB3-critical aliasing issue appears fixed: the immediately previous observation used for action selection survived the following `step()` unchanged in the wrapped training env, and vector `info` keys were populated. post7 still uses shared/two-buffer observation storage with `copy_observations=False`, so older retained obs references can mutate later; do not rely on indefinite old-observation immutability. A local macOS/MPS 5M reproduction of the best config learned and cleared Level1-1 intermittently (`101` total completions, final `40/100` rolling completion rate), but did not reproduce the RTX/post4 `80/100` early-stop result. A later Linux x86_64 RTX2060 SkyPilot reproduction with post7 did reproduce the completed-episode stop criterion, stopping at `2,711,552` timesteps with `182` total completions and `80/100` recent completions. Linux x86_64 RTX4090 post7 repeats were variable: one same-config run maxed at `5,005,312` with `31/100`, while the immediate repeat crossed `80/100` just after the last logged update at `4,227,072` timesteps. A same-config post4 control with two concurrent RTX4090 child runs also failed to reproduce the original post4 winner: both maxed at `5,005,312`, with final `0/100` and `30/100` recent completion rates. Treat post7 as training-validated on Linux/RTX, but do not make fine sample-efficiency claims from a single run; use matched repeat distributions for post4-vs-post7 regression claims.
On 2026-06-16, `stable-retro-turbo==1.0.0.post10` was tested on Linux x86_64 RTX4090 with the same best config. An isolated full 5M run learned and cleared levels (`253` total completions) but did not early-stop, ending around `47/100` recent completions after a transient peak around `66/100`; wall time was `1,719s` (`28m39s`), final fps about `2939`. A clean repeat was manually stopped before conclusion; it had begun learning late (`4` completions by about `2.78M`). Treat post10 as not showing the post5/post6 catastrophic regressions, but not yet proven better than post7/post4 without completed repeat distributions.
On 2026-06-17, `stable-retro-turbo==1.0.0.post11` was used for a three-seed RTX4090 retry of the best Level1-1 no-curriculum recipe (`lr=0.00015`, entropy `0.01 -> 0.0003` over 2M, strict `100/100` terminal-episode stop, 5M cap). It trained successfully and reproduced level-clearing behavior, but did not confirm a strict `100/100` baseline in that batch: seed24 was a strong near miss (`918` total completions, final `93/100`, peak observed `98/100`, `5,005,312` timesteps, `47m17s` wall time), seed25 finished `164` total and `37/100`, and seed23 finished `38` total and `9/100`. Treat post11 as the current default runtime, but do not promote that recipe as a confirmed `100/100` baseline without another successful confirmation.

## Training Runs

Keep generated training artifacts out of source control. Use `runs/`, `logs/`, and `models/` for checkpoints, TensorBoard logs, and evaluation outputs.

Default remote training should not run robust evals in the training loop. Modal should focus on training, checkpointing, and uploading checkpoint artifacts. Evaluate checkpoints out of process, preferably locally while waiting for remote training progress, and log checkpoint eval metrics back to the same W&B run. Promote the current best checkpoint from that external eval process using completion rate first, then maximum x-position, then mean reward.

By default, all training runs should log to W&B and upload checkpoint/final model artifacts unless the user explicitly opts out.

Every training run must include a human-readable run description explaining the experiment or ablation being tested. Pass it through `--run-description` for local/SkyPilot runs or the `run_description` parameter for Modal runs. The description should be specific enough to distinguish the run from nearby baselines and ablations.

Default Modal training runs should use the benchmarked T4 settings unless the user explicitly requests a different shape: `cpu=16.0`, `memory=32768`, `gpu=T4`, `n_envs=32`, `env_threads=0` (native default resolves to 16 threads), `torch_num_threads=0`, `n_steps=512`, `batch_size=256`, and `n_epochs=10`.

## Dependencies

Use `uv` for dependency resolution and keep `uv.lock` committed. Preserve Python supply-chain hardening in `pyproject.toml`.
