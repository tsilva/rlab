# Super Mario PPO Baseline 3: Entropy-Decay Sample-Efficient Stop

Baseline 3 promotes the entropy-decay RTX 4090 run
`sky_score_style_simple_maxpool_10m_stop80ep100_entdecay_seed23_20260614_102508`.

The important result is sample efficiency: it reached the completed-episode
early-stop criterion at `3,979,616` aggregate policy timesteps. The prior
completed-episode Baseline 2 follow-up stopped at `5,278,832`, so this run used
`24.6%` fewer samples, or about `1.33x` better sample efficiency. It is the new
default comparison target for future sample-efficiency work.

## Artifact

Recommended model:

[final model](runs/sky_score_style_simple_maxpool_10m_stop80ep100_entdecay_seed23_20260614_102508/final_model.zip)

Local absolute path:

```text
/Users/tsilva/repos/tsilva/sandbox-sb3/runs/sky_score_style_simple_maxpool_10m_stop80ep100_entdecay_seed23_20260614_102508/final_model.zip
```

SHA256:

```text
c6197172192df6d407aa098f128785c8bfcfd5185b3381b7cd366b414ba3a146
```

Latest periodic checkpoint before early stop:

[3.9M checkpoint](runs/sky_score_style_simple_maxpool_10m_stop80ep100_entdecay_seed23_20260614_102508/checkpoints/ppo_mario_3900000_steps.zip)

SHA256:

```text
c9396fe16b3dd341f483db30b0e6ef3f26a4a1b7d826a6c2a66645a2d01d28d0
```

Artifacts are under `runs/`, which is intentionally ignored by source control.

## Result Summary

Stop criterion:

```text
stop when the last 100 terminal training episodes are >=80% complete
```

Stop summary:

| Field | Value |
| --- | ---: |
| Stop reason | completed-episode completion-rate threshold |
| Stop timestep | `3,979,616 / 10,000,000` |
| Episode window | `100` terminal episodes |
| Stop completion rate | `0.800` |
| Threshold | `0.800` |
| Total terminal episodes | `1,635` |
| Total completed episodes | `158` |

Comparison:

| Run | Stop timestep | Relative samples | Notes |
| --- | ---: | ---: | --- |
| Baseline 2 completed-episode stop | `5,278,832` | `1.00x` | Fixed `ent_coef=0.01` |
| Baseline 3 entropy decay | `3,979,616` | `0.754x` | `1.33x` better sample efficiency |
| LR-decay ablation | `6,956,400` | `1.318x` | Negative ablation |

The entropy-decay run delayed early completions until the entropy coefficient
approached its floor, then produced a much sharper reliability ramp. This is
the highest-ROI sample-efficiency improvement observed so far.

## GUI Playback

GUI command:

```bash
uv --cache-dir .uv-cache run python -m mario_ppo.play \
  --model runs/sky_score_style_simple_maxpool_10m_stop80ep100_entdecay_seed23_20260614_102508/final_model.zip \
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
episode=1 seed=7 reward=319.35 max_x=3127 steps=513 status=terminated died=False complete=True
```

## Training Run

SkyPilot job:

| Field | Value |
| --- | --- |
| SkyPilot cluster | `sandbox-sb3-stop10-4090` |
| SkyPilot parent job id | `3` |
| Task name | `mario-sampleeff-ablate-parallel-4090` |
| Infra | `k8s/rtx4090` |
| GPU | NVIDIA GeForce RTX 4090 |
| Run name | `sky_score_style_simple_maxpool_10m_stop80ep100_entdecay_seed23_20260614_102508` |
| Parent log path | `~/sky_logs/sky-2026-06-14-11-24-56-120668` |
| Per-run log path | `~/sky_workdir/logs/parallel_ablation/sky_score_style_simple_maxpool_10m_stop80ep100_entdecay_seed23_20260614_102508.log` |
| Status | `SUCCEEDED` |
| Parent job duration | `1h 5m 30s` |

The parent SkyPilot task ran this entropy-decay ablation and the LR-decay
ablation concurrently inside one GPU pod. The parent duration is therefore not
the wall time of this child run alone.

Package/runtime facts observed in the run:

- Linux package: `stable-retro-turbo==1.0.0.post4`.
- Python: `3.14`.
- Torch on the SkyPilot node: `2.12.0+cu130`.
- CUDA was available.
- ROM import succeeded for `SuperMarioBros-Nes-v0`.

## Training Command

The SkyPilot task was launched from
`sky_mario_sampleeff_ablate_parallel_4090.yaml`. The entropy-decay child
process ran:

```bash
uv run --python 3.14 --no-dev python -m mario_ppo.train \
  --timesteps 10000000 \
  --n-envs 16 \
  --run-name "${ENT_RUN}" \
  --seed 23 \
  --n-steps 512 \
  --batch-size 512 \
  --n-epochs 10 \
  --learning-rate 0.0001 \
  --gamma 0.9 \
  --gae-lambda 1.0 \
  --ent-coef 0.01 \
  --ent-coef-final 0.001 \
  --ent-coef-schedule-timesteps 3000000 \
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
  --stop-completion-episode-window 100 \
  --stop-completion-rate-threshold 0.8 \
  --eval-freq 0 \
  --eval-episodes 0 \
  --device cuda \
  --env-threads 8 \
  --torch-num-threads 1 \
  --wandb \
  --wandb-project mario-ppo \
  --wandb-mode offline
```

The command did not pass `--no-terminate-on-life-loss`, so life loss was
terminal during training.

## Environment And Observation Path

Environment:

- Game: `SuperMarioBros-Nes-v0`.
- State: `Level1-1`.
- stable-retro provider: `stable-retro-turbo`.
- Native vector path: `StableRetroNativeVecEnv`.
- Parallel envs: `16`.
- Native env threads: `8`.
- Action set: `simple`.

Observation preprocessing:

- HUD crop top: `32` pixels.
- Resize: `84x84`.
- Grayscale.
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

This run used `--reward-mode score`.

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
| Entropy coefficient | `0.01 -> 0.001` |
| Entropy schedule | linear over first `3,000,000` timesteps |
| Value coefficient | `1.0` |
| Clip range | `0.2` |
| Advantage normalization | `False` |
| Adam epsilon | `1e-8` |
| Checkpoint frequency | `100,000` aggregate policy steps |
| Training-loop eval | disabled |

Timestep accounting:

```text
16 envs * 512 steps = 8192 aggregate policy decisions per PPO rollout
```

Each policy decision repeats the selected action for `4` emulator frames.

## Interpretation

Baseline 3 should replace the fixed-entropy completed-episode run as the
current sample-efficiency target. It did not achieve the desired 2x improvement,
but it moved the stop point from `5.28M` to `3.98M` aggregate timesteps without
changing env count, rollout geometry, reward mode, or completion semantics.

The main lesson is that fixed `ent_coef=0.01` is useful for discovery but likely
too exploratory once the policy has found level-clearing behavior. Decaying the
entropy coefficient allowed reliability to consolidate earlier.

The next sample-efficiency runs should tune entropy scheduling before changing
other axes. Promising variants:

- Delay decay until `1M` or `2M`, then decay to `0.001`.
- Use a higher floor such as `0.003`.
- Decay over `4M` instead of `3M`.

Continue to use the completed-episode stop as the training budget guardrail,
and continue using out-of-process stochastic checkpoint evals before promoting
policies for robustness claims.
