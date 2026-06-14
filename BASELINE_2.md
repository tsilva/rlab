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

A separate 6M replica run,
`sky_score_style_simple_maxpool_6m_seed23_20260613_213227`, confirmed that the
same setup can already produce level-clearing policies before 5M timesteps. A
GUI playback sample from checkpoint `4,700,000` cleared Level 1-1
stochastically with `max_x=6232`, `steps=528`, and `died=False`. Treat this as
evidence that the policy has learned the clearing behavior by about 4.7M, but
not as a reliability claim; checkpoint evaluation is still needed to measure
whether it clears often enough.

## Follow-up: Rolling Completion Early Stop

The later RTX 4090 run
`sky_score_style_simple_maxpool_10m_stop10_seed23_20260614_081635` used the same
BASELINE2 training setup with a `10M` timestep cap, but added training-loop
tracking of completion events per PPO rollout. It stopped once the rolling mean
over `10` rollouts crossed `10` completion events per rollout.

Stop summary:

| Field | Value |
| --- | ---: |
| Stop reason | rolling completion threshold |
| Stop timestep | `6,905,856 / 10,000,000` |
| Rolling window | `10` PPO rollouts |
| Final rolling completion mean | `10.5` |
| Final rollout completion events | `15` |
| Total training completion events | `1,538` |
| SkyPilot job duration | `51m 56s` |
| SB3 reported training elapsed | `3,109s` (`51m 49s`) |
| Final reported SB3 fps | about `2,221` |

The downloaded final model:

```text
runs/sky_score_style_simple_maxpool_10m_stop10_seed23_20260614_081635/final_model.zip
```

SHA256:

```text
fbffaf832bba06f4236f333962efe29d6f60fbe09ca6ddd16ee338617de8fd42
```

GUI playback sample from the final model:

```text
episode=1 seed=7 reward=319.34 max_x=3127 steps=492 status=terminated died=False complete=True
```

Lesson: the rolling completion signal was a useful training-time guardrail for
this exact run. It detected a strong policy around `6.9M` timesteps and stopped
before the kind of late overtraining/regression seen in the original 10M final
checkpoint. But the specific stop rule used here was completion events per PPO
rollout, so it is not scale-invariant: changing `n_envs` or `n_steps` changes
the number of completion opportunities per rollout.

Future BASELINE2-style early stops should instead track completion rate over
completed training episodes, e.g. stop when the last `100` terminal episodes
are at least `80%` complete. That criterion is invariant to env count and
rollout length. This still does not replace out-of-process checkpoint sweeps:
the final model needs stochastic eval to estimate reliability.

## Follow-up: Completed-Episode Completion-Rate Early Stop

The subsequent RTX 4090 run
`sky_score_style_simple_maxpool_10m_stop80ep100_seed23_20260614_091939` used
the same BASELINE2 setup with a `10M` timestep cap, but changed the early stop
criterion to completion rate over completed training episodes. It stopped when
the last `100` terminal episodes reached at least `80%` completion.

Stop summary:

| Field | Value |
| --- | ---: |
| Stop reason | completed-episode completion-rate threshold |
| Stop timestep | `5,278,832 / 10,000,000` |
| Episode window | `100` terminal episodes |
| Stop completion rate | `0.800` |
| Threshold | `0.800` |
| Total terminal episodes | `2,010` |
| Total completed episodes | `152` |
| SkyPilot job duration | about `40m` |

The run showed a sharp reliability transition after 5M timesteps. The
last-100 completion rate was around `0.24` at `5.05M`, then climbed through
`0.58` at `5.20M`, `0.74` at `5.24M`, and hit the `0.80` stop threshold at
`5.278M`. This supports using completed-episode completion rate as a practical
training-time stop signal: it is less sensitive to `n_envs` and rollout length
than counting completion events per rollout, while still stopping before the
late-run regression observed in the original 10M baseline.

Downloaded artifacts:

```text
runs/sky_score_style_simple_maxpool_10m_stop80ep100_seed23_20260614_091939/final_model.zip
runs/sky_score_style_simple_maxpool_10m_stop80ep100_seed23_20260614_091939/checkpoints/ppo_mario_5200000_steps.zip
```

SHA256:

```text
95a8387e955ef7befd3fd8b2a419bc0e8e5c0f8d22424390d218b81d81e9885d  final_model.zip
40fc56c8501822c7199039f098958a8eb8ca54ea2d112fa73dfe9deedfe26fe7  ppo_mario_5200000_steps.zip
```

GUI playback samples from the early-stopped final model:

```text
episode=1 seed=7 reward=229.05 max_x=2359 steps=332 status=terminated died=True complete=False
episode=2 seed=8 reward=319.74 max_x=6254 steps=528 status=terminated died=False complete=True
```

Lesson: this stop rule is the preferred in-training early stop for future
BASELINE2-style runs. It is still a training distribution metric, so final
promotion should continue to use out-of-process stochastic checkpoint evals,
but it is a much better budget guardrail than rollout-level completion counts.

## Follow-up: Sample-Efficiency Schedule Ablations

Two follow-up RTX 4090 ablations tested whether schedule changes could reach
the same completed-episode stop criterion faster. Both used the same BASELINE2
geometry and the same stop rule: stop when the last `100` terminal training
episodes are at least `80%` complete.

The two runs were launched concurrently inside one SkyPilot task so they shared
the same RTX 4090 while keeping separate run directories, logs, checkpoints,
and W&B-offline runs.

| Ablation | Schedule | Stop timestep | Relative to `5.278M` baseline | Result |
| --- | --- | ---: | ---: | --- |
| Entropy decay | `ent_coef: 0.01 -> 0.001` over first `3M` timesteps | `3,979,616` | `24.6%` fewer samples (`1.33x`) | Positive |
| LR decay | `learning_rate: 1e-4 -> 2e-5` over `10M` timesteps | `6,956,400` | `31.8%` more samples (`0.76x`) | Negative |

Entropy-decay stop summary:

| Field | Value |
| --- | ---: |
| Run name | `sky_score_style_simple_maxpool_10m_stop80ep100_entdecay_seed23_20260614_102508` |
| Stop reason | completed-episode completion-rate threshold |
| Stop timestep | `3,979,616 / 10,000,000` |
| Episode window | `100` terminal episodes |
| Stop completion rate | `0.800` |
| Total terminal episodes | `1,635` |
| Total completed episodes | `158` |

Entropy-decay artifacts:

```text
runs/sky_score_style_simple_maxpool_10m_stop80ep100_entdecay_seed23_20260614_102508/final_model.zip
runs/sky_score_style_simple_maxpool_10m_stop80ep100_entdecay_seed23_20260614_102508/checkpoints/ppo_mario_3900000_steps.zip
```

SHA256:

```text
c6197172192df6d407aa098f128785c8bfcfd5185b3381b7cd366b414ba3a146  final_model.zip
c9396fe16b3dd341f483db30b0e6ef3f26a4a1b7d826a6c2a66645a2d01d28d0  ppo_mario_3900000_steps.zip
```

LR-decay stop summary:

| Field | Value |
| --- | ---: |
| Run name | `sky_score_style_simple_maxpool_10m_stop80ep100_lrdecay_seed23_20260614_102508` |
| Stop reason | completed-episode completion-rate threshold |
| Stop timestep | `6,956,400 / 10,000,000` |
| Episode window | `100` terminal episodes |
| Stop completion rate | `0.800` |
| Total terminal episodes | `2,456` |
| Total completed episodes | `296` |

LR-decay artifacts:

```text
runs/sky_score_style_simple_maxpool_10m_stop80ep100_lrdecay_seed23_20260614_102508/final_model.zip
runs/sky_score_style_simple_maxpool_10m_stop80ep100_lrdecay_seed23_20260614_102508/checkpoints/ppo_mario_6900000_steps.zip
```

SHA256:

```text
f09d66361ec30800472c98286fcde23885004cb866ae155452c968ccd13e853e  final_model.zip
e91ea38f0c7a9a22a466033a328ccd765b4a796fb6b06f724138d6b0fcea1d69  ppo_mario_6900000_steps.zip
```

Lesson: entropy decay is the best sample-efficiency ablation so far. It
delayed early completions until roughly the schedule floor, then produced a
much sharper reliability ramp and hit the stop threshold about `1.30M`
timesteps earlier than the fixed-entropy completed-episode run. Linear LR
decay was counterproductive for this goal: it produced some early partial
success, then collapsed to a low completion rate and only recovered much later.
Future sample-efficiency runs should explore entropy schedules first, especially
delayed or less aggressive variants, before spending more runs on LR decay.

This entropy-decay run has been promoted to Baseline 3. See `BASELINE_3.md` for
the standalone run card and future sample-efficiency comparison target.

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
The later 6M replica also showed a stochastic GUI clear at 4.7M, reinforcing
that learning starts before the 5M recommended checkpoint even though the 4.7M
policy has not been proven reliable.

The rolling-completion early-stop follow-up produced another strong practical
artifact at about 6.9M timesteps, suggesting this training family benefits from
stopping on a live completion signal instead of blindly running to a fixed
timestep budget. A follow-up completed-episode stop reached the stricter
scale-invariant threshold earlier, at about 5.28M timesteps, after the last 100
terminal training episodes hit 80% completion. Prefer that completed-episode
rate criterion for future BASELINE2-style budget guards.

The first sample-efficiency ablations showed that entropy scheduling is the
highest-ROI next direction. Decaying `ent_coef` from `0.01` to `0.001` over the
first `3M` timesteps reached the same completed-episode stop at about `3.98M`
timesteps, while linear LR decay to `2e-5` delayed the stop to about `6.96M`.

The final checkpoint should not be used as the baseline policy. It achieved
only `4/20` clears in the same eval sweep.

Future runs should keep this exact run card as the comparison target and change
one thing at a time. For sample efficiency, prioritize entropy schedule
variants before reward mode, env count, completion semantics, LR schedules, or
PPO minibatch geometry.
