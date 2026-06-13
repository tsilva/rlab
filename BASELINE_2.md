# Super Mario PPO Baseline 2: SkyPilot RTX 4090 SB3 Native-Vector Run

This baseline documents the successful SkyPilot RTX 4090 run
`sky_score_style_simple_maxpool_10m_seed23_20260613_191408`.

The important result is that the `5M` checkpoint is already a very strong
Level 1-1 policy under stochastic evaluation: `19/20` clears with only one
death. The `8M` checkpoint tied the same clear count and ranked slightly higher
by the project's promotion tiebreakers, but the `5M` checkpoint is the best
early, stable checkpoint from this run and is the recommended practical
artifact to inspect first.

## Artifact

Recommended checkpoint:

[5M checkpoint](runs/sky_score_style_simple_maxpool_10m_seed23_20260613_191408/checkpoints/ppo_mario_5000000_steps.zip)

Local absolute path:

```text
/Users/tsilva/repos/tsilva/sandbox-sb3/runs/sky_score_style_simple_maxpool_10m_seed23_20260613_191408/checkpoints/ppo_mario_5000000_steps.zip
```

SHA256:

```text
05d4012803979b16d487eabd35df3618f267ebf18063899fc591cecd9dba4657
```

The artifact is under `runs/`, which is intentionally ignored by source
control. Keep the checkpoint in artifact storage or local ignored storage, not
in git.

## Result Summary

The 10M run completed successfully on the RTX 4090 and produced checkpoints
every `100,000` aggregate policy timesteps. A local checkpoint sweep evaluated
every `1M` checkpoint with 20 stochastic episodes.

Ranked by the project promotion rule:

1. Completion rate.
2. Maximum x-position.
3. Mean reward.

| Rank | Checkpoint | Clears | Completion | Mean max_x | Max x | Mean reward | Death rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8M | 19/20 | 0.95 | 3572.9 | 6244 | 3589.26 | 0.05 |
| 2 | 5M | 19/20 | 0.95 | 3239.8 | 6214 | 3255.98 | 0.05 |
| 3 | 9M | 18/20 | 0.90 | 3822.4 | 6240 | 3836.39 | 0.10 |
| 4 | 6M | 18/20 | 0.90 | 3096.1 | 3129 | 3111.21 | 0.10 |
| 5 | 7M | 17/20 | 0.85 | 2821.7 | 3136 | 2835.07 | 0.15 |
| 6 | 3M | 15/20 | 0.75 | 2925.4 | 3155 | 2932.48 | 0.25 |
| 7 | 4M | 12/20 | 0.60 | 3054.4 | 6242 | 3055.37 | 0.40 |
| 8 | 2M | 5/20 | 0.25 | 2588.9 | 3124 | 2574.88 | 0.75 |
| 9 | 10M | 4/20 | 0.20 | 2575.1 | 3155 | 2558.39 | 0.80 |
| 10 | 1M | 0/20 | 0.00 | 1401.6 | 1827 | 1376.68 | 1.00 |

Chronological view:

| Checkpoint | Clears | Completion | Mean max_x | Max x | Mean reward | Reward std | Deaths |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1M | 0/20 | 0.00 | 1401.6 | 1827 | 1376.68 | 253.54 | 20/20 |
| 2M | 5/20 | 0.25 | 2588.9 | 3124 | 2574.88 | 367.76 | 15/20 |
| 3M | 15/20 | 0.75 | 2925.4 | 3155 | 2932.48 | 396.53 | 5/20 |
| 4M | 12/20 | 0.60 | 3054.4 | 6242 | 3055.37 | 792.65 | 8/20 |
| 5M | 19/20 | 0.95 | 3239.8 | 6214 | 3255.98 | 705.07 | 1/20 |
| 6M | 18/20 | 0.90 | 3096.1 | 3129 | 3111.21 | 117.52 | 2/20 |
| 7M | 17/20 | 0.85 | 2821.7 | 3136 | 2835.07 | 817.23 | 3/20 |
| 8M | 19/20 | 0.95 | 3572.9 | 6244 | 3589.26 | 1125.31 | 1/20 |
| 9M | 18/20 | 0.90 | 3822.4 | 6240 | 3836.39 | 1414.00 | 2/20 |
| 10M | 4/20 | 0.20 | 2575.1 | 3155 | 2558.39 | 480.59 | 16/20 |

The final 10M checkpoint is much worse than the 5M and 8M checkpoints. This
run is another clear example that PPO policy quality is non-monotonic and that
checkpoint selection should be based on out-of-process evaluation, not final
timestep alone.

## 5M Checkpoint Details

Evaluation protocol:

- `20` stochastic episodes.
- Seed base: `7000`.
- Environment: `SuperMarioBros-Nes-v0`, state `Level1-1`.
- Completion threshold: `max_level_x_pos >= 3160`.
- Episode terminates on life loss, completion threshold, or `4500` policy
  steps.
- Reward mode and wrappers matched the training run.

5M summary:

| Metric | Value |
| --- | ---: |
| Checkpoint step | `5,000,000` |
| Clears | `19/20` |
| Completion rate | `0.95` |
| Deaths | `1/20` |
| Death rate | `0.05` |
| Mean reward | `3255.98` |
| Reward std | `705.07` |
| Mean max_x | `3239.8` |
| Max x | `6214` |
| Mean level max_x | `3084.45` |
| Max level max_x | `3126` |

The single failed eval episode died at `max_x=2356`. Most successful episodes
terminated close to the threshold around `3123-3126` level x-position. One
successful episode continued into the next level before terminal accounting,
which is why global `max_x` reached `6214`.

GUI playback command:

```bash
uv --cache-dir .uv-cache run python -m mario_ppo.play \
  --model runs/sky_score_style_simple_maxpool_10m_seed23_20260613_191408/checkpoints/ppo_mario_5000000_steps.zip \
  --episodes 0 \
  --max-steps 4500 \
  --fps 30 \
  --scale 4 \
  --reward-mode score \
  --terminal-reward 50 \
  --reward-scale 10 \
  --action-set simple \
  --completion-x-threshold 3160 \
  --terminate-on-completion \
  --device auto \
  --stochastic
```

Observed GUI sample:

```text
episode=1 seed=7 reward=319.01 max_x=3122 steps=518 status=terminated died=False complete=True
```

## Training Run

SkyPilot job:

| Field | Value |
| --- | --- |
| SkyPilot cluster | `sandbox-sky-k8s-ml-gpu-image` |
| SkyPilot job id | `10` |
| Task name | `mario-10m-4090` |
| Infra | `k8s/rtx4090` |
| GPU | NVIDIA GeForce RTX 4090 |
| Run name | `sky_score_style_simple_maxpool_10m_seed23_20260613_191408` |
| Log path | `~/sky_logs/sky-2026-06-13-20-13-56-037947` |
| Status | `SUCCEEDED` |
| Runtime | `1h 17m 13s` |
| Final timesteps | `10,002,432 / 10,000,000` |
| Final reported SB3 fps | about `2160` |
| Progress-bar throughput | about `2165 it/s` |

Package/runtime facts observed in the run:

- Linux package: `stable-retro-turbo==1.0.0.post4`.
- Python: `3.14`.
- Torch on the SkyPilot node: `2.12.0+cu130`.
- CUDA was available.
- ROM import succeeded for `SuperMarioBros-Nes-v0`.

The repo lock used the supplied post4 Linux wheel SHA:

```text
d9bcfb534bc0c6e52094819c08ec58486b10090f23e86b5bc7608dd4fd48e9b2
```

## Training Command

The SkyPilot task was launched from `sky_mario_10m_4090.yaml` and ran this
training command:

```bash
uv run --python 3.14 --no-dev python -m mario_ppo.train \
  --timesteps 10000000 \
  --n-envs 16 \
  --run-name "${RUN_NAME}" \
  --seed 23 \
  --n-steps 512 \
  --batch-size 512 \
  --n-epochs 10 \
  --learning-rate 0.0001 \
  --gamma 0.9 \
  --gae-lambda 1.0 \
  --ent-coef 0.01 \
  --clip-range 0.2 \
  --reward-mode score \
  --terminal-reward 50 \
  --reward-scale 10 \
  --action-set simple \
  --frame-skip 4 \
  --max-pool-frames \
  --max-episode-steps 4500 \
  --completion-x-threshold 3160 \
  --terminate-on-completion \
  --checkpoint-freq 100000 \
  --eval-freq 0 \
  --eval-episodes 0 \
  --device cuda
```

The command did not pass `--no-terminate-on-life-loss`, so life loss was
terminal during training. In code:

```python
terminate_on_life_loss = not args.no_terminate_on_life_loss
```

That means training episodes ended on:

- life loss,
- completion threshold via `--terminate-on-completion --completion-x-threshold 3160`,
- native env terminal events,
- or `--max-episode-steps 4500`.

Training-loop evaluation was disabled. Checkpoints were evaluated out of
process after the run.

## Environment And Observation Path

Environment:

- Game: `SuperMarioBros-Nes-v0`.
- State: `Level1-1`.
- stable-retro provider: `stable-retro-turbo`.
- Native vector path: `StableRetroNativeVecEnv`.
- Parallel envs: `16`.
- Action set: `simple`.

Observation preprocessing:

- HUD crop top: `32` pixels.
- Resize: `84x84`.
- Grayscale.
- Resize algorithm: default `area`.
- Frame skip: `4`.
- Max-pool over the last two raw frames inside each frame skip.
- Frame stack: `4`.
- Tensor layout entering SB3 policy: channel-first image stack compatible with
  SB3 `CnnPolicy`.

Action set:

```text
noop
right
right_b
right_a
right_a_b
a
left
```

## Reward And Termination Semantics

This run used `--reward-mode score`, not the older upstream-baseline reward
documented in `BASELINE.md`.

For the vector wrapper, score mode computes:

```text
reward = progress_delta + 0.01 * score_delta
if completion_event: reward += completion_reward
if died: reward -= death_penalty
reward -= time_penalty
```

For this run:

| Parameter | Value |
| --- | ---: |
| `progress_reward_scale` | `1.0` |
| `completion_reward` | `0.0` |
| `death_penalty` | `25.0` |
| `time_penalty` | `0.0` |
| `use_retro_reward` | `False` |
| `completion_x_threshold` | `3160` |
| `terminate_on_life_loss` | `True` |
| `terminate_on_completion` | `True` |
| `terminate_on_level_change` | `False` |

Completion was counted if either stable-retro reported a level change or the
level-local x-position reached the threshold.

## PPO Configuration

| Parameter | Value |
| --- | ---: |
| Algorithm | SB3 PPO |
| Policy | `CnnPolicy` |
| Device | `cuda` |
| Seed | `23` |
| Parallel envs | `16` |
| Rollout length per env | `512` |
| Aggregate rollout size | `8192` |
| Batch size | `512` |
| Minibatches per epoch | `16` |
| PPO epochs per rollout | `10` |
| Learning rate | `1e-4` |
| Gamma | `0.9` |
| GAE lambda | `1.0` |
| Entropy coefficient | `0.01` |
| Value coefficient | default `1.0` |
| Clip range | `0.2` |
| Advantage normalization | `False` |
| Adam epsilon | default project setting `1e-8` |
| Checkpoint frequency | `100,000` aggregate policy steps |
| Training-loop eval | disabled |

Timestep accounting:

```text
16 envs * 512 steps = 8192 aggregate policy decisions per PPO rollout
```

Each policy decision repeats the selected action for `4` emulator frames, so
`10,000,000` policy decisions corresponds to about `40,000,000` emulator-frame
steps across all envs.

## Eval Command

Each 1M checkpoint was evaluated locally with:

```bash
uv --cache-dir .uv-cache run python -m mario_ppo.evaluate \
  --model runs/sky_score_style_simple_maxpool_10m_seed23_20260613_191408/checkpoints/ppo_mario_${steps}_steps.zip \
  --episodes 20 \
  --seed 7000 \
  --max-steps 4500 \
  --reward-mode score \
  --terminal-reward 50 \
  --reward-scale 10 \
  --action-set simple \
  --completion-x-threshold 3160 \
  --terminate-on-completion \
  --device auto \
  --output runs/local_evals/sky_score_style_simple_maxpool_10m_seed23_1m_sweep/eval_${steps}.json
```

The evaluator defaults to stochastic action sampling for PPO models, so this
matched training-time action sampling.

Raw local eval outputs:

```text
runs/local_evals/sky_score_style_simple_maxpool_10m_seed23_1m_sweep/
```

## Difference From BASELINE.md

This is not a same-settings reproduction of the upstream
`vietnh1009/Super-mario-bros-PPO-pytorch` baseline.

Important differences:

| Field | `BASELINE.md` upstream reproduction | This run |
| --- | --- | --- |
| PPO implementation | custom upstream PPO | SB3 PPO |
| Model | upstream custom CNN | SB3 `CnnPolicy` |
| Env backend | `gym-super-mario-bros` / `nes-py` | `stable-retro-turbo` |
| Parallel envs | `8` | `16` |
| Aggregate rollout | `4096` | `8192` |
| Effective minibatch | `256` | `512` |
| Seed | `123` | `23` |
| Reward | `env_reward + score_delta/40 +/-50, then /10` | progress delta + `0.01 * score_delta`, death penalty |
| Completion | upstream `done`/`flag_get` | x-threshold `3160` or level change |
| Training target | `5M` in documented run | `10M`, best early checkpoint at `5M` |

Shared or closely matched pieces:

- Simple movement action set.
- Frame skip `4`.
- 4-frame stack.
- `84x84` grayscale observations.
- Max-pooling over the last two frames.
- `gamma=0.9`.
- `gae_lambda=1.0`.
- `ent_coef=0.01`.
- `clip_range=0.2`.
- `n_epochs=10`.
- Stochastic checkpoint evaluation.
- Terminal life-loss behavior effectively on.

## Interpretation

The run demonstrates that the SB3 + `stable-retro-turbo` native-vector path can
learn a strong Level 1-1 policy on the home RTX 4090. The 5M checkpoint is the
main practical artifact because it reaches `19/20` clears and looks strong in
GUI playback. The 8M checkpoint is the formal winner under the promotion rule,
but the 5M checkpoint is already strong and avoids the later 10M regression.

The final checkpoint should not be used as the baseline policy. It achieved
only `4/20` clears in the same eval sweep.

Future runs should keep this exact run card as the comparison target and change
one thing at a time: reward mode, env count, completion semantics, or PPO
minibatch geometry.
