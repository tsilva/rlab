# Mario PPO

PPO training scaffold for `SuperMarioBros-Nes-v0` using `stable-retro-turbo` and Stable-Baselines3.

The goal is to train a CNN PPO policy that moves right through `Level1-1` and improves over random and simple scripted baselines.

## Setup

```bash
uv sync
uv run python scripts/import_roms.py ~/Desktop/roms
```

The ROM import must recognize `SuperMarioBros-Nes-v0`; stable-retro matches by ROM SHA, not by filename.

On Apple Silicon, the project pins PyTorch to `torch==2.11.0` because the `2.12.0`
macOS arm64 wheel reports MPS unavailable on macOS 26.5.1. Local `--device auto`
resolves to CUDA, then MPS, then CPU; SB3's built-in `auto` would otherwise pick
CPU even when MPS is available.

## Smoke Test

```bash
uv run python -m mario_ppo.evaluate --policy right --episodes 2 --max-steps 600
uv run python -m mario_ppo.train \
  --timesteps 512 \
  --n-envs 1 \
  --batch-size 128 \
  --max-episode-steps 600 \
  --run-name smoke
uv run python -m mario_ppo.evaluate --model runs/smoke/final_model.zip --episodes 2 --max-steps 600
```

Watch the smoke model in a GUI window:

```bash
PYTHONPATH=src .venv/bin/python -m mario_ppo.play --model runs/smoke_doc/final_model.zip --episodes 3 --max-steps 1200 --fps 30 --scale 4
```

## Train

Start with a bounded run:

```bash
uv run python -m mario_ppo.train \
  --timesteps 1000000 \
  --run-name ppo_level1_1_1m
```

Longer run:

```bash
uv run python -m mario_ppo.train \
  --timesteps 10000000 \
  --run-name ppo_level1_1_10m
```

W&B online run:

```bash
wandb login
uv run python -m mario_ppo.train \
  --timesteps 10000000 \
  --run-name ppo_level1_1_10m \
  --wandb \
  --wandb-project SuperMarioBros-NES
```

Current training defaults mimic the upstream baseline hyperparameters from
`BASELINE.md` while keeping SB3's `CnnPolicy` model:

```text
device=auto -> mps
n_envs=8
seed=123
n_steps=512
batch_size=256
n_epochs=10
learning_rate=1e-4
gamma=0.9
gae_lambda=1.0
ent_coef=0.01
vf_coef=1.0
clip_range=0.2
normalize_advantage=False
adam_eps=1e-8
action_set=simple
reward_mode=baseline
```

That collects `4096` env steps per PPO update. The previous 64-env MPS setup
was faster for raw throughput, but these defaults prioritize matching the
working upstream PPO recipe.

W&B offline smoke run:

```bash
uv run python -m mario_ppo.train \
  --timesteps 512 \
  --n-envs 1 \
  --batch-size 128 \
  --max-episode-steps 600 \
  --run-name wandb_smoke \
  --wandb \
  --wandb-mode offline
```

Evaluate:

```bash
uv run python -m mario_ppo.evaluate --model runs/ppo_level1_1_10m/final_model.zip --episodes 20
uv run python -m mario_ppo.evaluate --policy random --episodes 20
uv run python -m mario_ppo.evaluate --policy right --episodes 20
```

Model evaluation defaults to 20 stochastic episodes. Use `--no-stochastic` for deterministic argmax evaluation.

Training-loop eval is disabled by default. Robust eval is handled out of process from checkpoint artifacts so training throughput is not blocked. The local checkpoint evaluator tracks Mario-specific progress metrics in addition to reward:

- `eval/max_x_mean` and `eval/max_x_max`
- `eval/max_level_x_mean` and `eval/max_level_x_max`
- `eval/completion_rate`, using either a reported level change or `--completion-x-threshold`
- `eval/death_rate` and `eval/death_count`
- W&B `eval/death_x_pos_histogram` when deaths are observed
- W&B/local best-episode video for each checkpoint eval when requested

Local eval files are written under the local eval directory:

```text
runs/local_evals/<run-name>/checkpoint_eval_metrics.jsonl
runs/local_evals/<run-name>/videos/best_episode_<timesteps>_steps.mp4
```

For `Level1-1`, `--completion-x-threshold` defaults to `3160`. Set it to `0` if you only want to count completion when stable-retro reports a level change.

Training uses the upstream-style score/env reward by default. Progress metrics
are still tracked on a monotonic global coordinate across level changes for
logging and evaluation.

The default `baseline` reward mode is:

```text
raw_reward = env_reward + score_delta / 40
if completed_level: raw_reward += 50
elif died_or_done: raw_reward -= 50
reward = raw_reward / 10
```

Use `--reward-mode bounded` for the older SuperMarioRL-style capped progress reward, or `--reward-mode additive` to restore the older additive death/completion shaping. For legacy additive level-completion runs, prefer adding a large one-time completion reward and a death penalty large enough that dying near the end is worse than finishing:

```bash
--death-penalty 250 \
--completion-reward 2000 \
--time-penalty 0.02
```

The promoted `best_model` artifact is selected by completion rate first, then maximum x-position, then mean reward.

TensorBoard:

```bash
uv run tensorboard --logdir runs
```

Download a W&B model artifact and watch it locally:

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/play_wandb_artifact.py modal_gpu_short_improve
```

Use the best artifact instead of the final model:

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/play_wandb_artifact.py modal_gpu_short_improve --kind best
```

Watch a sampled PPO policy instead of deterministic argmax:

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/play_wandb_artifact.py modal_fixed_reward_gpu_50k --kind best --stochastic
```

## Modal

Modal runs Linux containers, while local Apple Silicon runs use the same `stable-retro-turbo` package.

Install the local Modal CLI extra and authenticate:

```bash
UV_CACHE_DIR=.uv-cache uv sync --extra modal
UV_CACHE_DIR=.uv-cache uv run modal setup
```

Upload NES ROMs from your Mac to the Modal Volume:

```bash
UV_CACHE_DIR=.uv-cache uv run modal run src/mario_ppo/modal_app.py::upload_roms --rom-dir ~/Desktop/roms
```

Run a remote smoke training job without W&B upload:

```bash
UV_CACHE_DIR=.uv-cache uv run modal run src/mario_ppo/modal_app.py::train \
  --timesteps 512 \
  --n-envs 1 \
  --run-name modal_smoke \
  --max-episode-steps 600
```

Run a longer remote job:

```bash
UV_CACHE_DIR=.uv-cache uv run modal run src/mario_ppo/modal_app.py::train \
  --timesteps 10000000 \
  --n-envs 4 \
  --run-name modal_ppo_level1_1_10m \
  --max-episode-steps 4500
```

For W&B online from Modal, store your API key once as a Modal Secret:

```bash
UV_CACHE_DIR=.uv-cache uv run modal secret create wandb-secret WANDB_API_KEY=...
```

Then run training with W&B enabled:

```bash
UV_CACHE_DIR=.uv-cache uv run modal run src/mario_ppo/modal_app.py::train \
  --timesteps 10000000 \
  --n-envs 4 \
  --run-name modal_ppo_level1_1_10m_wandb \
  --wandb \
  --wandb-project SuperMarioBros-NES \
  --wandb-mode online
```

To prove artifact upload end to end with a short run, force frequent checkpoints:

```bash
UV_CACHE_DIR=.uv-cache uv run modal run src/mario_ppo/modal_app.py::train \
  --timesteps 512 \
  --n-envs 1 \
  --run-name modal_wandb_artifact_smoke \
  --max-episode-steps 600 \
  --checkpoint-freq 256 \
  --wandb \
  --wandb-project SuperMarioBros-NES \
  --wandb-mode online
```

When W&B is enabled, training uploads checkpoint and final-model artifacts. The Modal result includes `wandb_url` when W&B provides a run URL. A separate local eval process promotes the best-model artifact from evaluated checkpoints.

Evaluate pending checkpoint artifacts locally and log metrics back to the same W&B run:

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/eval_wandb_checkpoints.py modal_ppo_level1_1_10m_wandb \
  --episodes 50 \
  --max-steps 2500 \
  --action-set right \
  --record-best-video
```

The evaluator skips checkpoint steps already present in `runs/local_evals/<run-name>/checkpoint_eval_metrics.jsonl`. Use `--force` to re-evaluate them.

Run a short scaled GPU improvement job from the last smoke model:

```bash
UV_CACHE_DIR=.uv-cache uv run modal run src/mario_ppo/modal_app.py::train \
  --timesteps 50000 \
  --n-envs 8 \
  --run-name modal_gpu_short_improve \
  --max-episode-steps 1200 \
  --checkpoint-freq 10000 \
  --resume /vol/runs/modal_wandb_artifact_online_smoke_retry/final_model.zip \
  --wandb \
  --wandb-project SuperMarioBros-NES \
  --wandb-mode online
```

Run a continuation from a W&B model artifact:

```bash
UV_CACHE_DIR=.uv-cache uv run modal run src/mario_ppo/modal_app.py::train \
  --timesteps 250000 \
  --n-envs 16 \
  --cpu 16 \
  --memory 32768 \
  --gpu T4 \
  --n-steps 64 \
  --batch-size 256 \
  --learning-rate 0.00005 \
  --ent-coef 0.01 \
  --run-name modal_continue_best_250k_lr5e5_ent01 \
  --max-episode-steps 1200 \
  --checkpoint-freq 25000 \
  --resume-artifact tsilva/mario-ppo/modal_fixed_reward_gpu_250k_lr1e4_env16-best:latest \
  --wandb \
  --wandb-project SuperMarioBros-NES \
  --wandb-mode online
```

Run a restricted forward-action experiment:

```bash
UV_CACHE_DIR=.uv-cache uv run modal run src/mario_ppo/modal_app.py::train \
  --timesteps 250000 \
  --n-envs 16 \
  --cpu 16 \
  --memory 32768 \
  --gpu T4 \
  --n-steps 64 \
  --batch-size 256 \
  --learning-rate 0.0001 \
  --ent-coef 0.01 \
  --action-set right \
  --run-name modal_right_action_250k_lr1e4 \
  --max-episode-steps 1200 \
  --checkpoint-freq 25000 \
  --wandb \
  --wandb-project SuperMarioBros-NES \
  --wandb-mode online
```

Models trained with `--action-set right` must also be played or evaluated with `--action-set right`.

Run a completion-weighted restricted-action experiment:

```bash
UV_CACHE_DIR=.uv-cache uv run modal run src/mario_ppo/modal_app.py::train \
  --timesteps 500000 \
  --n-envs 16 \
  --cpu 16 \
  --memory 32768 \
  --gpu T4 \
  --n-steps 64 \
  --batch-size 256 \
  --n-epochs 4 \
  --learning-rate 0.00005 \
  --ent-coef 0.01 \
  --action-set right \
  --death-penalty 250 \
  --completion-reward 2000 \
  --time-penalty 0.02 \
  --completion-x-threshold 3160 \
  --run-name modal_right_completion_reward_500k \
  --max-episode-steps 2500 \
  --checkpoint-freq 25000 \
  --wandb \
  --wandb-project SuperMarioBros-NES \
  --wandb-mode online
```

Run a softer, entropy-preserving completion experiment:

```bash
UV_CACHE_DIR=.uv-cache uv run modal run src/mario_ppo/modal_app.py::train \
  --timesteps 250000 \
  --n-envs 16 \
  --cpu 16 \
  --memory 32768 \
  --gpu T4 \
  --n-steps 64 \
  --batch-size 256 \
  --n-epochs 2 \
  --learning-rate 0.00005 \
  --ent-coef 0.05 \
  --clip-range 0.1 \
  --target-kl 0.02 \
  --action-set right \
  --death-penalty 75 \
  --completion-reward 500 \
  --time-penalty 0.01 \
  --completion-x-threshold 3160 \
  --run-name modal_right_soft_completion_ent05_250k \
  --max-episode-steps 2500 \
  --checkpoint-freq 25000 \
  --wandb \
  --wandb-project SuperMarioBros-NES \
  --wandb-mode online
```

Remote outputs are persisted to the Modal Volume `mario-ppo-data` under `/runs/<run-name>`.

Run a short fixed-reward GPU job from scratch:

```bash
UV_CACHE_DIR=.uv-cache uv run modal run src/mario_ppo/modal_app.py::train \
  --timesteps 50000 \
  --n-envs 8 \
  --run-name modal_fixed_reward_gpu_50k \
  --max-episode-steps 1200 \
  --checkpoint-freq 10000 \
  --wandb \
  --wandb-project SuperMarioBros-NES \
  --wandb-mode online
```

## Notes

- The bundled stable-retro scenario rewards only `xscrollLo`, the low byte of scroll position. That byte wraps every 256 pixels, so training ignores that reward by default and uses wrapper-computed global best x-progress instead.
- Level changes are detected with stable-retro's `levelHi/levelLo` info fields. The wrapper logs both global progress (`max_x_pos`) and within-level progress (`level_max_x_pos`).
- By default episodes terminate on first life loss so the policy cannot farm repeated early progress after dying.
- Training rollouts use `StableRetroNativeVecEnv`. The default stable-retro-turbo observation pipeline uses `obs_crop=(32, 0, 0, 0)`, `obs_resize_algorithm="area"`, `obs_resize=(84, 84)`, `obs_grayscale=True`, `frame_skip=4`, `frame_stack=4`, and `maxpool_last_two=True`.
- Training and model evaluation rollouts use `StableRetroNativeVecEnv`. Python vector wrappers add discrete Mario actions, true progress reward, progress metrics, and SB3's HWC-to-CHW transpose. Image preprocessing, frame skip, frame stack, and max-pool stay in native stable-retro-turbo. Evaluation videos replay selected action traces through the render-capable single-env path for visualization only.
- `StableRetroNativeVecEnv` autoresets native terminal slots and provides `info["terminal_observation"]` before reset. Python-defined terminal events such as first-life-loss, level-completion termination, and max episode steps save `terminal_observation` before a full-vector reset. Slots reset only because another slot hit a Python terminal are marked `TimeLimit.truncated=True` so SB3 bootstraps from their terminal observation instead of treating the transition as a true terminal.
- Generated checkpoints and logs stay under `runs/`.
