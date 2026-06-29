# 2026-06-29 B250 Success And B255 Confirmation Launch

Goal: `mario-level1-4-100of100`

Primary metric: `train/info/level_complete/rate/min/last`

Status: candidate found; not promoted yet. B250 seed 90 satisfied the training
contract, but the frozen recipe still needs the remaining confirmation seeds and
promotion eval evidence before this goal can be called solved.

## B250 Training Result

B250 was the plain B55 low-KL scratch-transfer recipe for `Level1-4` with no
completion bonus:

- Job: `255`
- Run: `b250_l14_b55post21_s90_20260629T065634Z`
- W&B: <https://wandb.ai/tsilva/SuperMarioBros-NES/runs/o24fwlhs>
- Final state: `finished`
- Final step: `3,488,576`
- Final `train/info/level_complete/rate/min/last`: `1.0`
- Final `train/info/level_complete/from/0-3/rate`: `1.0`
- Final `train/info/level_complete/from/0-3/count`: `4,429`
- Final `rollout/ep_rew_mean`: `2328.7805`
- PPO tail: `approx_kl=0.0015800865`, `clip_fraction=0.019042969`,
  `explained_variance=0.99998033`

This is the first Level1-4 run to hit the strict 100/100 source-attempt training
window. It stopped before the 5M cap, so the seed-90 train result is valid
evidence for the goal's confirmation set.

## Other B250-B254 Screens

The other discovery arms were canceled after B250 succeeded so beast-3 capacity
could turn over to confirmation:

| Job | Run | Status at cancel | Reason |
| --- | --- | --- | --- |
| `256` | `b251_l14_b55complete100_s90_20260629T065653Z` | low rolling rate despite many cumulative clears | Lower decision value than confirmation. |
| `258` | `b252_l14_clipped_slowent_complete100_s90_20260629T065654Z` | weak rolling rate, far below B250 | Lower decision value than confirmation. |
| `257` | `b253_l14_longcredit_complete500_s90_20260629T065654Z` | zero clean clears after a mature partial run | Lower decision value than confirmation. |
| `259` | `b254_l14_longhorizon_complete1000_s90_20260629T065654Z` | zero clean clears after a mature partial run | Lower decision value than confirmation. |

Interpretation: for World 1-4, the plain score/progress B55 recipe was much
better than the initial clean-completion reward or long-credit scratch variants.
The lesson is that reward shaping made the target louder but did not make the
policy more stable or sample-efficient in this first screen.

## Out-Of-Process Eval

Both eval jobs used the B250 final artifact:
`tsilva/SuperMarioBros-NES/b250_l14_b55post21_s90_20260629T065634Z-final:latest`.

| Eval job | Candidate | Policy | Episodes | Completion | Reward mean | Max x |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `1` | `b250-final-seed90` | stochastic | `100` | `99/100` (`0.99`) | `4266.8040` | `4610` |
| `2` | `b250-final-seed90-deterministic` | deterministic | `100` | `100/100` (`1.0`) | `4607.9000` | `4608` |

Local eval outputs:

- `logs/eval_runner/eval_job_1_5ac7754f.json`
- `logs/eval_runner/eval_job_2_2215ed2b.json`

The stochastic eval failure died around x=`2028` in episode 8. Deterministic
eval was perfect, so the candidate policy has learned the route, but stochastic
sampling still has a small failure tail. Treat this as strong candidate
evidence, not final promotion evidence.

## Confirmation Batch

Checked-in confirmation spec:
`experiments/goals/mario-level1-4-100of100/specs/b255-b250-post21-four-seed-l14-confirm.json`

The spec was validated with `rlab.job_queue.load_spec_document`, and its
`train_config` matches B250 exactly. Only stage, seeds, and run metadata changed.

Queued jobs:

| Job | Seed | Run | W&B |
| --- | ---: | --- | --- |
| `260` | `91` | `b255_l14_b250post21_s91_20260629T074728Z` | <https://wandb.ai/tsilva/SuperMarioBros-NES/runs/pq4eyn5d> |
| `261` | `92` | `b255_l14_b250post21_s92_20260629T074728Z` | <https://wandb.ai/tsilva/SuperMarioBros-NES/runs/pxrock12> |
| `262` | `93` | `b255_l14_b250post21_s93_20260629T074728Z` | <https://wandb.ai/tsilva/SuperMarioBros-NES/runs/95qypjvd> |
| `263` | `94` | `b255_l14_b250post21_s94_20260629T074728Z` | <https://wandb.ai/tsilva/SuperMarioBros-NES/runs/je7z5l71> |

Early snapshot after launch:

| Seed | Step | Rolling rate | Clean count | Reward mean |
| ---: | ---: | ---: | ---: | ---: |
| `91` | `638,976` | `0.02` | `22` | `1863.4225` |
| `92` | `503,344` | `0.02` | `10` | `1654.4003` |
| `93` | `499,712` | `0.00` | `0` | `1228.9000` |
| `94` | `499,712` | `0.01` | `12` | `1534.4644` |

Next decision: monitor B255 until each confirmation seed either reaches
`train/info/level_complete/rate/min/last > 0.99` or hits the 5M cap. If all
remaining seeds succeed, run promotion eval for the confirmed final artifacts
before declaring the recipe solved. If any seed fails, treat B250 as a strong
single-seed candidate and continue with legal reward/hyperparameter iteration.

## Playback

Best visual inspection target for the candidate found so far:

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
