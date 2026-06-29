# 2026-06-29 B255 Confirmation Failed; B256-B260 Robustness Plan

Goal: `mario-level1-4-100of100`

Primary metric: peak `train/info/level_complete/rate/min/last`

Status: B250/B255 is a strong candidate but not a confirmed recipe. The frozen
B250 recipe reached the strict 100/100 training window on three of five goal
confirmation seeds, but failed the required five-seed protocol.

## B255 Confirmation Result

| Seed | Job | Run | Process state | Peak rolling rate | Peak step | Final summary rate | Decision |
| ---: | ---: | --- | --- | ---: | ---: | ---: | --- |
| `90` | `255` | `b250_l14_b55post21_s90_20260629T065634Z` | finished | `1.00` | `3,488,576` | `1.00` | success |
| `91` | `260` | `b255_l14_b250post21_s91_20260629T074728Z` | finished | `0.17` | `2,464,592` | `0.01` | fail |
| `92` | `261` | `b255_l14_b250post21_s92_20260629T074728Z` | finished | `1.00` | `3,388,432` | `1.00` | success |
| `93` | `262` | `b255_l14_b250post21_s93_20260629T074728Z` | finished | `1.00` | `4,095,552` | `0.99` | success by peak |
| `94` | `263` | `b255_l14_b250post21_s94_20260629T074728Z` | finished | `0.98` | `4,935,872` | `0.93` | fail |

Queue process status reached `succeeded` for all five train jobs, but the goal
decision uses the metric peak, not process success. B250/B255 therefore confirms
`3/5` training seeds and cannot be promoted as the goal solution.

## Eval Evidence So Far

B250 seed 90 final artifact:
`tsilva/SuperMarioBros-NES/b250_l14_b55post21_s90_20260629T065634Z-final:latest`

| Eval job | Policy | Episodes | Completion | Reward mean | Max x |
| --- | --- | ---: | ---: | ---: | ---: |
| `1` | stochastic | `100` | `99/100` (`0.99`) | `4266.8040` | `4610` |
| `2` | deterministic | `100` | `100/100` (`1.0`) | `4607.9000` | `4608` |

Because the training confirmation failed, promotion eval for the B255 artifacts
is not enough to declare the goal solved.

## Interpretation

B250's no-completion-bonus score/progress recipe is the first Level1-4 recipe
that reliably shows real route learning: three seeds reached the strict window,
one nearly did, and one stayed in a weak local optimum. The failed B251-B254
screens indicate that a large completion bonus, clipped progress, and long-credit
scratch variants were not better first responses. The next legal direction is
therefore not a task or curriculum change; it is a robustness sweep around B250.

## Next Batch

All B256-B260 screens keep `SuperMarioBros-Nes-v0` / `Level1-4`, native vector
training, terminal-on-life-loss-or-level-change semantics, no curriculum, no
pretrained checkpoint, the 5M cap, W&B artifacts, and the strict 100/100
attempt-window stop rule.

| Spec | Change from B250 | Why it is legal and useful |
| --- | --- | --- |
| `b256-b250-normadv-l14-screen` | `normalize_advantage=true` | Hyperparameter-only stability test for seed variance. |
| `b257-b250-softupdate-l14-screen` | LR `1.2e-4 -> 8e-5`, `clip_range=0.12`, `target_kl=0.10` | Lower update pressure while preserving reward semantics. |
| `b258-b250-entropyfloor-l14-screen` | Entropy decays to `0.001` over `5M` | Tests persistent exploration for late/local-optimum seeds. |
| `b259-b250-progress125-l14-screen` | `progress_reward_scale=1.25` | Slightly louder target-aligned traversal signal. |
| `b260-b250-complete25-l14-screen` | `completion_reward=25` | Tiny clean-clear reinforcement after `completion_reward=100` looked too disruptive. |

Decision rule: any screen that reaches peak `1.0` can enter the full goal
confirmation seeds `90,91,92,93,94`; do not promote until that frozen recipe
passes all confirmation seeds and out-of-process eval.

## B256-B260 Launch

The specs validated with `rlab.job_queue.load_spec_document` before enqueue.
The root `rlab-train-image.json` file was absent, so jobs were enqueued without
`--runtime-image-ref-file`; the queue resolved the same latest successful
digest-pinned train image used by B250/B255, digest slug `10e00c906541`.

| Job | Spec | Run | W&B |
| ---: | --- | --- | --- |
| `264` | `b256-b250-normadv-l14-screen` | `b256_l14_normadv_s90_20260629T084741Z` | <https://wandb.ai/tsilva/SuperMarioBros-NES/runs/f4fvzco8> |
| `265` | `b257-b250-softupdate-l14-screen` | `b257_l14_softupdate_s90_20260629T084810Z` | <https://wandb.ai/tsilva/SuperMarioBros-NES/runs/mktnipn1> |
| `266` | `b258-b250-entropyfloor-l14-screen` | `b258_l14_entropyfloor_s90_20260629T084829Z` | <https://wandb.ai/tsilva/SuperMarioBros-NES/runs/5qhvu45z> |
| `267` | `b259-b250-progress125-l14-screen` | `b259_l14_progress125_s90_20260629T084847Z` | <https://wandb.ai/tsilva/SuperMarioBros-NES/runs/xur4w7jz> |
| `268` | `b260-b250-complete25-l14-screen` | `b260_l14_complete25_s90_20260629T084910Z` | <https://wandb.ai/tsilva/SuperMarioBros-NES/runs/abn9fw2l> |

Fleet state after launch: one beast-3 container,
`rlab-beast-3-rtx4090-any-profile-10e00c906541`, with five active worker leases
for jobs `264-268`.

## B256-B260 Screen Read

B256, B257, and B259 reached the strict screen window. B257 is the selected
confirmation candidate because it changes only PPO update pressure from B250,
ended with the summary metric still at `1.0`, and directly targets the B255
seed-variance failure without changing reward semantics. B259 is a useful backup
candidate but changes the dense progress reward scale, making reward comparisons
less clean. B258 and B260 were canceled to free capacity for confirmation.

| Spec | Job | Peak rolling rate | Peak step | Final/current status | Decision |
| --- | ---: | ---: | ---: | --- | --- |
| `b256-b250-normadv-l14-screen` | `264` | `1.00` | not re-scanned after queue success | process succeeded | backup, not selected |
| `b257-b250-softupdate-l14-screen` | `265` | `1.00` | `4,480,464` | process succeeded, final summary `1.00` | selected for B261 confirmation |
| `b258-b250-entropyfloor-l14-screen` | `266` | `0.98` | `3,664,400` | canceled for capacity | not selected |
| `b259-b250-progress125-l14-screen` | `267` | `1.00` | `4,384,336` | process succeeded, final summary `0.98` | backup, not selected |
| `b260-b250-complete25-l14-screen` | `268` | `0.26` | `3,667,648` | canceled for capacity | rejected |

## B261 Confirmation Launch

B261 freezes B257's exact `train_config` and runs the remaining goal
confirmation seeds. Seed 90 is represented by the B257 screen success.

| Job | Seed | Run | W&B |
| ---: | ---: | --- | --- |
| `269` | `91` | `b261_l14_b257soft_s91_20260629T095039Z` | <https://wandb.ai/tsilva/SuperMarioBros-NES/runs/xcn2vru4> |
| `270` | `92` | `b261_l14_b257soft_s92_20260629T095039Z` | <https://wandb.ai/tsilva/SuperMarioBros-NES/runs/bga06l27> |
| `271` | `93` | `b261_l14_b257soft_s93_20260629T095039Z` | <https://wandb.ai/tsilva/SuperMarioBros-NES/runs/mzlpig1s> |
| `272` | `94` | `b261_l14_b257soft_s94_20260629T095039Z` | <https://wandb.ai/tsilva/SuperMarioBros-NES/runs/c27n9lx9> |

Queue state after launch: active B261 train jobs `269-272` on `rtx4090`.
Fleet plan wanted the lower-contention four-worker beast-3 shape for digest
`10e00c906541`, but Mac-side SSH listing for beast-3 was intermittently timing
out. W&B and queue rows confirmed the jobs were running.

## Playback For Current Best Candidate

```bash
UV_CACHE_DIR=.uv-cache uv run rlab-play \
  --artifact-run b250_l14_b55post21_s90_20260629T065634Z \
  --artifact-kind final \
  --artifact-version latest \
  --game SuperMarioBros-Nes-v0 \
  --state Level1-4 \
  --episodes 3 \
  --seed 10014 \
  --fps 30 \
  --scale 4
```
