# B122-B126 Level1-3 Backfill Launch

Created: `2026-06-28T11:35Z`

Goal: `mario-level1-3-100of100`

Scope: training-only continuation. Eval remains out of scope for this launch
pass.

## Trigger

B117-B121 are still running on beast-3, but the live W&B scan around
`2026-06-28T11:25Z` showed no Level1-3 clears through roughly `1.07M-1.25M`
history steps:

| Arm | Job | Run | History step | Peak L1-3 rate | L1-3 count |
| --- | ---: | --- | ---: | ---: | ---: |
| B117 | `99` | `o8hgq0cf` | `1254368` | `0` | `0` |
| B118 | `100` | `f6gsf8sh` | `1201280` | `0` | `0` |
| B119 | `101` | `ddapidps` | `1130416` | `0` | `0` |
| B120 | `102` | `wdzbu4dv` | `1103968` | `0` | `0` |
| B121 | `103` | `iv1lxp8o` | `1071632` | `0` | `0` |

A final summary refresh before recording this launch note at
`2026-06-28T11:34Z` still showed all five active runs at zero clears:

| Arm | Summary step | L1-3 rate | L1-3 count | `rollout/ep_rew_mean` | FPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| B117 | `1913904` | `0` | `0` | `776.5059` | `1318` |
| B118 | `1867776` | `0` | `0` | `642.05` | `1290` |
| B119 | `1766768` | `0` | `0` | `646.54` | `1269` |
| B120 | `1757888` | `0` | `0` | `280.61` | `1266` |
| B121 | `1736704` | `0` | `0` | `646.76` | `1251` |

This is weaker than the known Level1 B55/B57 timing. The Level1 recipe-family
runs first emitted nonzero completion around `85k-111k` steps:

| Source | Run | First nonzero completion step | Result |
| --- | --- | ---: | --- |
| B55 seed108 | `qt3h08mc` | `86624` | final `100/100` |
| B55 seed109 | `actk7fw5` | `86544` | final `100/100` |
| B57 seed116 | `c916vdl6` | `111072` | final `96/100`, peak `99/100` |
| B57 seed117 | `8j5mlmu0` | `85072` | final `100/100` |

Reward attribution on B117-B121 also showed raw x-progress dominating the
training signal:

| Arm | Summary step | Reward | Progress share | Death share | Completion share |
| --- | ---: | ---: | ---: | ---: | ---: |
| B117 | `1490944` | `650.908` | `0.96279734` | `0.037136067` | `0` |
| B118 | `1417520` | `629.324` | `0.9629173` | `0.037082665` | `0` |
| B119 | `1343488` | `539.78204` | `0.9587475` | `0.04123014` | `0` |
| B120 | `1335296` | `282.99` | `0.9244232` | `0.07557677` | `0` |
| B121 | `1285712` | `644.79` | `0.96312016` | `0.036879838` | `0` |

All active arms were ending on life loss with no clean level-change clears.

## Queued Backfill

Five legal reward/hyperparameter arms were added under `specs/` and enqueued as
pending RTX4090 jobs:

| Job | Spec | Run |
| ---: | --- | --- |
| `104` | `b122-b55clipped-complete25-l13-screen` | `b122_l13_b55clipped_complete25_s80_20260628T113146Z` |
| `105` | `b123-b55clipped-death50-complete25-l13-screen` | `b123_l13_b55clipped_death50_complete25_s80_20260628T113150Z` |
| `106` | `b124-b55clipped-slowent-complete50-l13-screen` | `b124_l13_b55clipped_slowent_complete50_s80_20260628T113153Z` |
| `107` | `b125-b55clipped-gamma095-slowent-complete25-l13-screen` | `b125_l13_b55clipped_gamma095_slowent_complete25_s80_20260628T113157Z` |
| `108` | `b126-b46style-clipped-slowent-complete25-l13-screen` | `b126_l13_b46style_clipped_slowent_complete25_s80_20260628T113200Z` |

All five keep the same game, Level1-3 state, action set, Stable Retro event
contract, `done_on_events=life_loss,level_change`, 5M cap, and strict
`train/info/level_complete/rate/min/last > 0.99` early stop. The changes are
limited to legal reward/hyperparameter levers:

- B122: B55 low-update-pressure recipe, `score_progress_clipped=true`,
  `completion_reward=25`
- B123: B122 plus `death_penalty=50`
- B124: clipped progress, slower entropy floor, `completion_reward=50`
- B125: clipped progress, slower entropy floor, `gamma=0.95`,
  `completion_reward=25`
- B126: B46-style fixed LR / `target_kl=0.20` with clipped progress, slower
  entropy floor, and `completion_reward=25`

## Queue And Fleet State

After enqueue:

```text
train_jobs: {"failed": 5, "pending": 5, "running": 5}
eval_jobs: {}
```

`rlab-fleet plan` and `rlab-fleet reconcile` both reported no capacity action:

```text
desired_deployments=1
existing_containers=1
actions=0
keep rlab-beast-3-rtx4090-any-profile-ff22345ac89b
```

The new jobs are pending backfill behind the five active beast-3 workers, not an
attempt to oversubscribe the RTX4090 host.

## B117-B121 Retirement

At `2026-06-28T11:37Z`, the B117-B121 W&B history scan still showed zero
Level1-3 clears around the first fixed-budget checkpoint:

| Arm | Job | History step | Peak L1-3 rate | L1-3 count | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| B117 | `99` | `2130448` | `0` | `0` | cancel |
| B118 | `100` | `2069312` | `0` | `0` | cancel |
| B119 | `101` | `1999984` | `0` | `0` | cancel |
| B120 | `102` | `1990592` | `0` | `0` | cancel |
| B121 | `103` | `1942736` | `0` | `0` | cancel |

These jobs were graceful-canceled to free the five beast-3 workers for the
reward-balanced B122-B126 batch. This keeps the search moving after an obvious
zero-clear screen and preserves the B117-B121 W&B runs as negative evidence,
not as completed 5M-cap failures.

## B122-B126 Running State

The queue and fleet handoff completed cleanly:

```text
train_jobs: {"canceled": 5, "failed": 5, "running": 5}
eval_jobs: {}
```

All five new jobs were claimed by the existing beast-3 RTX4090 runner with
fresh heartbeats. Initial W&B metrics at `2026-06-28T11:40Z`:

| Arm | Job | W&B | Step | L1-3 rate | L1-3 count | Reward | FPS | KL | Clip frac |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B122 | `104` | [`8zqr1zs8`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/8zqr1zs8) | `143088` | `0` | `0` | `260.07397` | `1579` | `0.0054584644` | `0.13605957` |
| B123 | `105` | [`i8opel65`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/i8opel65) | `81584` | `0` | `0` | `212.106` | `1354` | `0.009414927` | `0.16572265` |
| B124 | `106` | [`1ogy6ovp`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/1ogy6ovp) | `81920` | `0` | `0` | `252.268` | `1362` | `0.0068200408` | `0.0770874` |
| B125 | `107` | [`pa3mvebv`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/pa3mvebv) | `78448` | `0` | `0` | `216.72198` | `1279` | `0.0076165004` | `0.12402344` |
| B126 | `108` | [`6c1301ue`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/6c1301ue) | `73728` | `0` | `0` | `220.71199` | `1225` | `0.0033013616` | `0.093811035` |

## B122-B126 Mid-Screen Snapshot

A W&B refresh at `2026-06-28T11:50Z` showed the batch still running cleanly but
without a Level1-3 clear:

| Arm | Job | History step | L1-3 rate | L1-3 count | Reward | FPS | KL | Clip frac | Death share | Progress share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B122 | `104` | `923888` | `0` | `0` | `644.62` | `1287` | `0.00431222` | `0.080078125` | `0.037715238` | `0.96228474` |
| B123 | `105` | `883168` | `0` | `0` | `617.97` | `1256` | `0.003752376` | `0.056262206` | `0.064875774` | `0.9351242` |
| B124 | `106` | `874176` | `0` | `0` | `630.64` | `1246` | `0.004807151` | `0.10489502` | `0.0353598` | `0.9646402` |
| B125 | `107` | `876064` | `0` | `0` | `590.582` | `1248` | `0.005442669` | `0.12756348` | `0.0409335` | `0.9590174` |
| B126 | `108` | `850992` | `0` | `0` | `639.232` | `1238` | `0.0035853935` | `0.08759765` | `0.033811033` | `0.9661841` |

Decision: keep B122-B126 running. Around `0.85M-0.92M` steps is not enough to
retire the clipped-progress batch because PPO health is acceptable and all five
jobs are progressing. The next intervention point remains the first nonzero
Level1-3 clean-clear metric or the `2M` fixed-budget checkpoint.

## Next Training Decision

Monitor B122-B126 first. The useful early triggers are:

- first nonzero `train/info/level_complete/from/0-2/count`
- first nonzero `train/info/level_complete/from/0-2/rate`
- the `2M` fixed-budget checkpoint if all five remain at zero

If one B122-B126 arm produces clears, use it as the parent for the next legal
training batch or confirmation path depending on the peak rate. If all five are
still zero around `2M`, treat clipped progress alone as insufficient and design
the next batch around stronger survival/credit-assignment hyperparameters while
preserving the same environment and event contract.
