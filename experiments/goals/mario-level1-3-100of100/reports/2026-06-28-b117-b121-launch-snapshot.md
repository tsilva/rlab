# B117-B121 Level1-3 Screen Launch Snapshot

Created: `2026-06-28T09:58:51Z`

Goal: `mario-level1-3-100of100`

Primary metric: peak `train/info/level_complete/rate/min/last`. For this
single-source Level1-3 goal, it should match
`train/info/level_complete/from/0-2/rate` once the rolling source window is
full.

## Orientation

- Live Stable Retro data includes `Level1-3`.
- Initial goal queue state was empty: no train jobs and no eval jobs.
- `rlab-fleet policy` selected `rtx4090` / `beast-3` screening capacity.
- Before enqueue, `rlab-fleet plan` saw one idle managed `beast-3` container
  and no queue demand.

## Specs

| Spec | Seed | Hypothesis | Primary delta |
| --- | ---: | --- | --- |
| `b117-b55post21-l13-screen` | `80` | B55/B83 low-KL late-LR-decay transfers from Level1-1 to Level1-3. | Baseline B55 transfer, no completion bonus. |
| `b118-b55complete25-l13-screen` | `80` | A modest true clean-completion reward improves Level1-3 clear reliability. | B117 plus `completion_reward=25`. |
| `b119-b55slowent-l13-screen` | `80` | Longer exploration helps Level1-3 discover reliable clears. | B117 plus `ent_coef_final=0.001`. |
| `b120-b46style-l13-screen` | `80` | Higher PPO update pressure explores Level1-3 faster than B55. | B46-style fixed LR, `target_kl=0.20`. |
| `b121-b55slowent-complete25-l13-screen` | `80` | Completion reward plus slower entropy improves clear discovery and reinforcement. | B118 plus B119 combined. |

All five specs loaded successfully through `rlab.job_queue.load_spec_document`.

## Queue Launch

Jobs were enqueued profileless, using the CLI's latest successful immutable train
image resolver because `rlab-train-image.json` was not present at repo root.

| Job | Run | W&B |
| ---: | --- | --- |
| `94` | `b117_l13_b55post21_s80_20260628T095535Z` | <https://wandb.ai/tsilva/SuperMarioBros-NES/runs/rmx94up8> |
| `95` | `b118_l13_b55complete25_s80_20260628T095550Z` | <https://wandb.ai/tsilva/SuperMarioBros-NES/runs/ctfz60j2> |
| `96` | `b119_l13_b55slowent_s80_20260628T095605Z` | <https://wandb.ai/tsilva/SuperMarioBros-NES/runs/3dd9af5y> |
| `97` | `b120_l13_b46style_s80_20260628T095621Z` | <https://wandb.ai/tsilva/SuperMarioBros-NES/runs/pk4cfmcg> |
| `98` | `b121_l13_b55slowent_complete25_s80_20260628T100401Z` | pending at report update |

## Fleet State

At `2026-06-28T09:58:51Z`, `rlab-monitor --view all --goal
mario-level1-3-100of100` showed all four jobs running on `beast-3`.

`rlab-fleet ps` showed one managed container:
`rlab-beast-3-rtx4090-any-profile-91ce61ae72fd`, digest short label
`91ce61ae72fd`, with four active worker leases and fresh heartbeats.

`beast-2` was unreachable during this check, but it is not part of this goal's
primary RTX4090 path.

At `2026-06-28T10:04:23Z`, B121 was added to fill the fifth available
`beast-3` worker. `rlab-queue status --goal mario-level1-3-100of100` showed
five running train jobs, and `rlab-fleet ps` showed B121 claimed by worker
`1-92b5278d` in the same managed container.

## Next Check

Monitor `train/info/level_complete/from/0-2/rate`,
`train/info/level_complete/rate/min/last`, `train/reward_share/done`,
`rollout/ep_rew_mean`, `eval/progress/x/max` only after out-of-process evals,
and PPO health metrics (`train/approx_kl`, `train/clip_fraction`,
`train/explained_variance`). Promote only after the goal's confirmation protocol
and out-of-process eval evidence.
