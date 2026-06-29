# 2026-06-29 B250-B254 Launch Snapshot

Goal: `mario-level1-4-100of100`

Primary metric: `train/info/level_complete/rate/min/last`

Status: discovery screen. No run is confirmed; success still requires the
goal's confirmation protocol on seeds `90,91,92,93,94` plus out-of-process eval.

## Contract

- Game/state: `SuperMarioBros-Nes-v0` / `Level1-4`.
- Target: strict 100/100 source-attempt completion.
- Source metric: `train/info/level_complete/from/0-3/rate`.
- Training stop: `train/info/level_complete/rate/min/last > 0.99`.
- Cap: `5,000,000` timesteps per scratch screen.
- Runtime: post21 train image resolved by the queue CLI to immutable digest
  slug `10e00c906541`.
- Hardware lane: `rtx4090` / `beast-3`, five workers, `env_threads=4`.

`Level1-4` was verified in the local Stable Retro state list before writing the
specs.

## Batch

The first World 1-4 batch is scratch PPO only. It intentionally does not resume
from a Level1-3 or other-level checkpoint because the goal contract says
`curriculum: none` and the active research scope allows reward and
hyperparameter levers only.

| Job | Spec | Run name | W&B | Hypothesis |
| --- | --- | --- | --- | --- |
| `255` | `b250-b55post21-l14-screen` | `b250_l14_b55post21_s90_20260629T065634Z` | <https://wandb.ai/tsilva/SuperMarioBros-NES/runs/o24fwlhs> | B55 low-KL scratch transfer baseline, no completion bonus. |
| `256` | `b251-b55complete100-l14-screen` | `b251_l14_b55complete100_s90_20260629T065653Z` | <https://wandb.ai/tsilva/SuperMarioBros-NES/runs/wpqcqazy> | B55 plus modest clean `completion_reward=100`. |
| `258` | `b252-clipped-slowent-complete100-l14-screen` | `b252_l14_clipped_slowent_complete100_s90_20260629T065654Z` | <https://wandb.ai/tsilva/SuperMarioBros-NES/runs/kt48wgd9> | Clipped progress, slower entropy decay, fixed LR, `target_kl=0.20`. |
| `257` | `b253-longcredit-complete500-l14-screen` | `b253_l14_longcredit_complete500_s90_20260629T065654Z` | <https://wandb.ai/tsilva/SuperMarioBros-NES/runs/vb1a28rw> | Longer credit, wider critic, advantage normalization, `completion_reward=500`. |
| `259` | `b254-longhorizon-complete1000-l14-screen` | `b254_l14_longhorizon_complete1000_s90_20260629T065654Z` | <https://wandb.ai/tsilva/SuperMarioBros-NES/runs/4r0cmb9b> | Longest horizon arm, `gamma=0.999`, `gae_lambda=1.0`, `completion_reward=1000`. |

The absent root `rlab-train-image.json` caused the explicit
`--runtime-image-ref-file` path to fail before any job was written. The jobs
were then enqueued without that flag, matching the documented Level1-3 fallback:
the queue CLI resolved the latest successful digest-pinned train image.

## Live State

Queue status immediately after launch:

```text
goal: mario-level1-4-100of100
train_jobs: {"running": 5}
active_train_jobs:
  job=255 status=running profile=any target=rtx4090 run=b250_l14_b55post21_s90_20260629T065634Z
  job=256 status=running profile=any target=rtx4090 run=b251_l14_b55complete100_s90_20260629T065653Z
  job=258 status=running profile=any target=rtx4090 run=b252_l14_clipped_slowent_complete100_s90_20260629T065654Z
  job=257 status=running profile=any target=rtx4090 run=b253_l14_longcredit_complete500_s90_20260629T065654Z
  job=259 status=running profile=any target=rtx4090 run=b254_l14_longhorizon_complete1000_s90_20260629T065654Z
```

Fleet plan showed no action required:

```text
desired_deployments=1
existing_containers=1
actions=0
desired:
  rlab-beast-3-rtx4090-any-profile-10e00c906541 host=beast-3 workers=5 profile=any target=rtx4090 digest=10e00c906541
actions:
  keep host=beast-3 container=rlab-beast-3-rtx4090-any-profile-10e00c906541 reason=container already matches desired state
```

`rlab-fleet ps` showed one lease per worker for jobs `255-259`, with recent
heartbeats. `rlab-monitor --view all` reported beast-3 busy with the five
World 1-4 jobs. Beast-2 was unreachable during this check, but this batch does
not depend on beast-2.

After W&B initialization, `rlab-queue status --goal mario-level1-4-100of100`
showed result rows `15001-15005` for the five running jobs and the W&B links in
the batch table above.

## Next Read

Monitor the goal metric first:

1. `train/info/level_complete/rate/min/last`
2. `train/info/level_complete/from/0-3/count`
3. `train/info/level_complete/from/0-3/rate`
4. `eval/done/level_change/from/Level1-4/rate` after queued out-of-process eval
5. `eval/progress/x/max`, `eval/reward/mean`, reward share, PPO KL/clip/EV

If a screen reaches the strict window, freeze that recipe and launch the
goal-defined confirmation seeds. If all screens stay at zero completions with a
flat reward/progress tail, the next legal batch should change only reward and
hyperparameter levers.
