# B117-B121 Early Training Health

Created: `2026-06-28`

Goal: `mario-level1-3-100of100`

Scope: training-only early monitor. Eval remains out of scope for this pass.

## Live Queue State

`rlab-queue status --goal mario-level1-3-100of100` shows five running jobs and
the five earlier interrupted jobs still marked failed:

```text
train_jobs: {"failed": 5, "running": 5}
eval_jobs: {}
```

All five running jobs are claimed by the existing beast-3 RTX4090 managed
container `rlab-beast-3-rtx4090-any-profile-ff22345ac89b`, with fresh worker
heartbeats.

## Training Metrics

Latest W&B training snapshot from the five active runs:

| Arm | Job | W&B | Step | Peak L1-3 rate | L1-3 count | `rollout/ep_rew_mean` | FPS | KL | Clip frac | EV |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B117 Level1 B55 transfer | `99` | [`o8hgq0cf`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/o8hgq0cf) | `732240` | `0` | `0` | `621.44604` | `1434` | `0.004976155` | `0.11774902` | `0.99637526` |
| B118 B55 + complete25 | `100` | [`f6gsf8sh`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/f6gsf8sh) | `709456` | `0` | `0` | `564.904` | `1384` | `0.0056772367` | `0.11871338` | `0.98466724` |
| B119 B55 + slow entropy | `101` | [`ddapidps`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/ddapidps) | `635856` | `0` | `0` | `579.90204` | `1321` | `0.006022022` | `0.1482544` | `0.9843643` |
| B120 B46-style high pressure | `102` | [`wdzbu4dv`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/wdzbu4dv) | `597776` | `0` | `0` | `403.55798` | `1314` | `0.008907178` | `0.1661499` | `0.9730225` |
| B121 complete25 + slow entropy | `103` | [`iv1lxp8o`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/iv1lxp8o) | `583584` | `0` | `0` | `347.44797` | `1290` | `0.0046014176` | `0.107666016` | `0.96659315` |

Primary training metric:
`train/info/level_complete/from/0-2/rate`, mirrored by
`train/info/level_complete/rate/min/last` for this single-state goal. All five
arms are still at `0` so far.

## Interpretation

This is still an early screen: the jobs are about `0.58M-0.73M` summary steps
into a `5M` cap, with bounded history scans reaching about `0.59M-0.77M` rows.
PPO health looks acceptable so far: approximate KL is controlled, clip
fractions are moderate, explained variance is high, and the workers are running
at expected throughput.

B117 has the best shaped reward so far and is also the direct Level1 B55
baseline transfer, so it remains the most useful baseline read. B118 and B119
are now close on shaped reward, while B120 and B121 are lagging. None of the
arms has produced a Level1-3 clear yet, but there is not enough evidence to
reject them this early.

## Next Training Decision

Do not queue extra arms while all five RTX4090 workers are actively running and
no arm has produced a Level1-3 completion signal yet. Re-check around the first
meaningful threshold:

- first nonzero `train/info/level_complete/from/0-2/count`
- first nonzero `train/info/level_complete/from/0-2/rate`
- around `1M` steps if all arms remain at zero clears

If one arm produces clears, use it as the parent for the next legal training
batch. If all five remain at zero through a larger fixed-budget checkpoint,
design the next batch from reward and hyperparameter levers only.
