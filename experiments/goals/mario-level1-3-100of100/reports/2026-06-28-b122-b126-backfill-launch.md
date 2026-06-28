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

## B122-B126 Retirement

A live W&B monitor at `2026-06-28T12:07Z` confirmed that all five B122-B126
arms were still at zero clean Level1-3 clears after the `2M` fixed-budget
checkpoint:

| Arm | Job | History step | Peak L1-3 rate | Peak min rate | L1-3 count | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| B122 | `104` | `2123200` | `0` | `0` | `0` | cancel |
| B123 | `105` | `2088064` | `0` | `0` | `0` | cancel |
| B124 | `106` | `2069344` | `0` | `0` | `0` | cancel |
| B125 | `107` | `2077776` | `0` | `0` | `0` | cancel |
| B126 | `108` | `2071280` | `0` | `0` | `0` | cancel |

The cancellation requests all returned `cancel_requested=1`. This retires the
clipped-progress/modest-completion family as negative Level1-3 screen evidence:
PPO remained numerically healthy, but the policy never generated a clean
level-change completion signal.

## B127-B131 Launch

The next screen batch moves to stronger survival and completion shaping while
preserving the same game, Level1-3 state, action set, Stable Retro event
contract, `done_on_events=life_loss,level_change`, 5M cap, and
`train/info/level_complete/rate/min/last > 0.99` early stop:

| Job | Spec | Run | W&B |
| ---: | --- | --- | --- |
| `109` | `b127-lowprogress-death50-complete50-l13-screen` | `b127_l13_lowprogress_death50_complete50_s80_20260628T120806Z` | [`yi2npvew`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/yi2npvew) |
| `110` | `b128-lowprogress-death75-complete75-l13-screen` | `b128_l13_lowprogress_death75_complete75_s80_20260628T120820Z` | [`ny9ysmdn`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/ny9ysmdn) |
| `111` | `b129-lowprogress-complete100-slowent-l13-screen` | `b129_l13_lowprogress_complete100_slowent_s80_20260628T120837Z` | [`4398ntt5`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/4398ntt5) |
| `112` | `b130-gamma097-lowprogress-complete50-l13-screen` | `b130_l13_gamma097_lowprogress_complete50_s80_20260628T120859Z` | [`a7lh2v2t`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/a7lh2v2t) |
| `113` | `b131-midpress-lowprogress-complete100-l13-screen` | `b131_l13_midpress_lowprogress_complete100_s80_20260628T120924Z` | [`nf20tius`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/nf20tius) |

After launch, the queue showed:

```text
train_jobs: {"canceled": 10, "failed": 5, "running": 5}
eval_jobs: {}
```

`rlab-fleet plan` and `rlab-fleet reconcile` both reported that the existing
beast-3 RTX4090 runner already matched desired capacity:

```text
desired_deployments=1
existing_containers=1
actions=0
keep rlab-beast-3-rtx4090-any-profile-063e55231d69
```

The fleet command warned that beast-2 SSH timed out while listing managed
containers, but this does not affect the active beast-3 Level1-3 training
capacity.

Initial W&B telemetry at `2026-06-28T12:11Z` confirms all five new runs are
actively logging:

| Arm | Job | History step | L1-3 rate | L1-3 count | Reward | FPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B127 | `109` | `282864` | `0` | `0` | `90.27` | `1674` |
| B128 | `110` | `231840` | `0` | `0` | `-4.4585` | `1449` |
| B129 | `111` | `196560` | `0` | `0` | `18.226002` | `1370` |
| B130 | `112` | `172032` | `0` | `0` | `75.664` | `1330` |
| B131 | `113` | `123344` | `0` | `0` | `15.839499` | `1243` |

## B127-B131 Plateau Retirement

The operator pointed out that `rollout/ep_rew_mean` can identify flat local
optima before the hard 2M zero-clear checkpoint. This was applied only as a
screening kill rule after verifying that the target metric was still zero:
reward is not a success metric, but a flat reward curve plus zero clean clears
is useful negative evidence.

At `2026-06-28T12:35Z`, all five B127-B131 arms still had no clean Level1-3
clear signal and had effectively flat `rollout/ep_rew_mean` over the last
roughly 500k steps:

| Arm | Job | History step | Peak L1-3 rate | Peak L1-3 count | Last reward | Reward delta over ~500k | Slope / 100k | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| B127 | `109` | `2069824` | `0` | `0` | `285.459` | `+0.849` | `+0.173` | cancel |
| B128 | `110` | `2022560` | `0` | `0` | `92.417` | `+2.800` | `+0.570` | cancel |
| B129 | `111` | `1999104` | `0` | `0` | `114.9425` | `+1.1045` | `+0.221` | cancel |
| B130 | `112` | `1982336` | `0` | `0` | `283.5` | `+3.335` | `+0.679` | cancel |
| B131 | `113` | `1924592` | `0` | `0` | `116.5625` | `+1.3505` | `+0.275` | cancel |

The cancellation requests all returned `cancel_requested=1`. This retires the
low-progress/survival/completion-reward batch as zero-clear plateau evidence.
B127 and B130 reached higher shaped-reward plateaus, while B128/B129/B131
settled lower, but none converted that shaped reward into the real
`train/info/level_complete/from/0-2/*` target.

## B132-B136 Launch

The next screen batch keeps the Level1-3 environment, action set, info events,
termination contract, 5M cap, and strict
`train/info/level_complete/rate/min/last > 0.99` early stop. It moves away from
more reward multiplier tuning and tests exploration/credit-assignment levers:

| Job | Spec | Run | W&B |
| ---: | --- | --- | --- |
| `114` | `b132-highentropy-lowprogress-complete50-l13-screen` | `b132_l13_highentropy_lowprogress_complete50_s80_20260628T123651Z` | [`qndz41am`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/qndz41am) |
| `115` | `b133-longrollout-gamma097-lowprogress-l13-screen` | `b133_l13_longrollout_gamma097_lowprogress_s80_20260628T123707Z` | [`vamb6nj1`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/vamb6nj1) |
| `116` | `b134-advnorm-lowprogress-complete75-l13-screen` | `b134_l13_advnorm_lowprogress_complete75_s80_20260628T123723Z` | [`2dla9n3o`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/2dla9n3o) |
| `117` | `b135-sparseclear-complete150-l13-screen` | `b135_l13_sparseclear_complete150_s80_20260628T123739Z` | [`td8f49qs`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/td8f49qs) |
| `118` | `b136-gamma099-advnorm-lowprogress-l13-screen` | `b136_l13_gamma099_advnorm_lowprogress_s80_20260628T123752Z` | [`vym2fmz9`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/vym2fmz9) |

After launch, the queue and fleet state settled to:

```text
train_jobs: {"canceled": 15, "failed": 5, "running": 5}
eval_jobs: {}
desired_deployments=1
existing_containers=1
actions=0
keep rlab-beast-3-rtx4090-any-profile-063e55231d69
```

Initial W&B telemetry at `2026-06-28T12:41Z` confirms all five new runs are
actively logging:

| Arm | Job | History step | L1-3 rate | L1-3 count | Reward | FPS | KL | Clip frac |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B132 | `114` | `347552` | `0` | `0` | `104.271996` | `1592` | `0.0046199616` | `0.08486328` |
| B133 | `115` | `315088` | `0` | `0` | `70.105995` | `1554` | `0.006918095` | `0.10568237` |
| B134 | `116` | `255280` | `0` | `0` | `-7.464` | `1347` | `0.0046434067` | `0.122521974` |
| B135 | `117` | `228032` | `0` | `0` | `-74.993996` | `1302` | `0.0036558057` | `0.055773925` |
| B136 | `118` | `183248` | `0` | `0` | `13.516001` | `1260` | `0.0040978338` | `0.115075685` |

## B135 One-Slot Backfill

A W&B refresh at `2026-06-28T12:45Z` showed B132, B133, and B136 still
improving shaped reward, with B134 weak but not fully flat. B135 was the one
clear zero-value plateau:

| Arm | Job | History step | Peak L1-3 count | Last reward | Reward delta over ~500k | Death share | Done share | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| B132 | `114` | `679888` | `0` | `250.60098` | `+149.886975` | `0.14913331` | `0` | keep |
| B133 | `115` | `680496` | `0` | `128.828` | `+56.42399` | `0.2351183` | `0` | keep |
| B134 | `116` | `589776` | `0` | `-0.96000046` | `+8.31199954` | `0.48253492` | `0` | keep |
| B135 | `117` | `565200` | `0` | `-75.0` | `-0.002` | `1.0` | `0` | cancel |
| B136 | `118` | `518720` | `0` | `42.964005` | `+34.949505` | `0.368053` | `0` | keep |

B135 was canceled with `cancel_requested=1`. Interpretation: pure sparse-clear
reward plus death penalty gave the agent essentially no learnable gradient
before the first clear, so the run settled into a constant death-penalty
plateau.

One replacement screen was added and launched:

| Job | Spec | Run | W&B |
| ---: | --- | --- | --- |
| `119` | `b137-tinyprogress-highentropy-complete150-l13-screen` | `b137_l13_tinyprogress_highentropy_complete150_s80_20260628T124700Z` | [`k6iv8tfh`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/k6iv8tfh) |

B137 keeps B135's target-aligned high completion reward, but adds a tiny
`progress_reward_scale=0.1` and higher entropy floor so early behavior is not a
constant death-penalty desert. After launch, the queue/fleet state was:

```text
train_jobs: {"canceled": 16, "failed": 5, "running": 5}
eval_jobs: {}
keep rlab-beast-3-rtx4090-any-profile-063e55231d69
```

Initial B137 telemetry at `2026-06-28T12:48Z`: step `82768`, Level1-3 count
`0`, reward `-52.085995`, FPS `1194`, KL `0.0054756296`, clip fraction
`0.076208495`, death share `0.76168644`, done share `0`.

## Next Training Decision

Monitor B132-B134, B136, and B137 first. Useful early triggers:

- first nonzero `train/info/level_complete/from/0-2/count`
- first nonzero `train/info/level_complete/from/0-2/rate`
- a clearly flat `rollout/ep_rew_mean` plateau with zero clean clears
- the 2M fixed-budget checkpoint if all five remain at zero

If one active arm produces clears, use it as the parent for the next legal
training batch or confirmation path depending on the peak rate. If all five
again hit flat zero-clear plateaus, the next batch should continue shifting
toward exploration/credit assignment before asking to broaden the search beyond
the current reward/hyperparameter-only scope.

## B132-B137 Plateau Check

A refresh at `2026-06-28T12:54Z` kept all five active runs. The clean Level1-3
target metric remained zero for every arm, but none of the older active arms
met the flat-reward abort rule yet. B137 was still too young to retire; its
reward movement is small, but it had only reached about 0.49M steps.

| Arm | Job | History step | L1-3 count | L1-3 rate | Min clear rate | Last reward | Reward delta over ~500k | Slope / 100k | Death share | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| B132 | `114` | `1261568` | `0` | `0` | `0` | `282.65` | `+23.185` | `+4.640` | `0.143534` | keep |
| B133 | `115` | `1310720` | `0` | `0` | `0` | `276.475` | `+77.994` | `+15.868` | `0.130364` | keep |
| B134 | `116` | `1187840` | `0` | `0` | `0` | `58.9015` | `+54.6425` | `+10.935` | `0.358059` | keep |
| B136 | `118` | `1130496` | `0` | `0` | `0` | `113.94` | `+38.9965` | `+7.804` | `0.233905` | keep |
| B137 | `119` | `491520` | `0` | `0` | `0` | `-47.826` | `+5.041` | `+1.043` | `0.729455` | keep, young |

Queue and fleet remained full on beast-3:

```text
train_jobs: {"canceled": 16, "failed": 5, "running": 5}
eval_jobs: {}
queue demand: profile=any target=rtx4090 pending=0 running=5 digest=063e55231d69
```

## B136 Plateau Backfill

A refresh at `2026-06-28T13:00Z` showed that B132 and B133 were nearly flat but
still moving, B134 was still improving, and B137 was still young. B136 had the
clearest zero-clear stagnation pattern: at `1.55M` steps it still had no
Level1-3 clean clear and its reward had effectively stopped moving over the
short tail.

| Arm | Job | History step | L1-3 count | L1-3 rate | Last reward | Reward delta over ~500k | Reward delta over ~250k | Reward delta over ~100k | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| B132 | `114` | `1712128` | `0` | `0` | `283.965` | `+4.348` | `+3.325` | `+2.475` | keep |
| B133 | `115` | `1785856` | `0` | `0` | `284.31` | `+9.427` | `+3.253` | `+1.343` | keep |
| B134 | `116` | `1630208` | `0` | `0` | `89.7025` | `+37.2705` | `+16.49` | `+6.9265` | keep |
| B136 | `118` | `1548288` | `0` | `0` | `117.162` | `+9.646` | `+1.032` | `+0.002` | cancel |
| B137 | `119` | `925696` | `0` | `0` | `-43.763` | `+5.388` | `+0.659` | `+3.849` | keep, young |

B136 was canceled with `cancel_requested=1`. This applies the operator's
plateau rule only after the target clean-clear metric is verified to be zero.

One replacement screen was added:

| Job | Spec | Run | W&B |
| ---: | --- | --- | --- |
| `120` | `b138-longcredit-tinyprogress-complete125-l13-screen` | `b138_l13_longcredit_tinyprogress_complete125_s80_20260628T130120Z` | [`9pgcc4f9`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/9pgcc4f9) |

B138 keeps the same game, Level1-3 state, action set, info events,
`done_on_events=life_loss,level_change`, cap, checkpoint cadence, and strict
`train/info/level_complete/rate/min/last > 0.99` stop. Its legal
reward/hyperparameter change is a lower progress scale (`0.15`), B55-level
death penalty (`25`), higher true-clear bonus (`125`), high entropy floor
(`0.003`), advantage normalization, and long-credit PPO (`n_steps=1024`,
`gamma=0.97`, `gae_lambda=0.95`).

After B136 exited, B138 was claimed and started logging:

```text
train_jobs: {"canceled": 17, "failed": 5, "running": 5}
eval_jobs: {}
```

Initial B138 telemetry at `2026-06-28T13:03Z`: step `42080`, Level1-3 count
`0`, reward `9.200501`.

## B132/B133 Plateau Backfill

A refresh at `2026-06-28T13:08Z` showed B132 and B133 past the 2M fixed-budget
checkpoint with no clean Level1-3 clears. Their shaped reward curves had also
flattened into the same high-progress local optimum seen in earlier screens,
so both were retired under the zero-clear plateau rule.

| Arm | Job | History step | L1-3 count | L1-3 rate | Last reward | Reward delta over ~500k | Reward delta over ~250k | Reward delta over ~100k | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| B132 | `114` | `2203648` | `0` | `0` | `283.735` | `-2.215` | `+0.958` | `-1.072` | cancel |
| B133 | `115` | `2342912` | `0` | `0` | `283.967` | `-0.768` | `+1.315` | `-0.550` | cancel |
| B134 | `116` | `2113536` | `0` | `0` | `89.2345` | `-0.607` | `+1.9905` | `+2.1795` | keep, still moving |
| B137 | `119` | `1425408` | `0` | `0` | `-43.593` | `+0.170` | `-2.712` | `-3.935` | keep, younger |
| B138 | `120` | `327680` | `0` | `0` | `13.8655` | `+5.628` | `+2.8775` | `-1.327` | keep, young |

B132 and B133 were canceled with `cancel_requested=1`. Both reached high
`rollout/ep_rew_mean` while the target metric stayed at zero, so their useful
lesson is negative: progress/score-shaped reward at this scale can plateau
without producing any clean Level1-3 transition.

Two replacement screens were added:

| Job | Spec | Run | W&B |
| ---: | --- | --- | --- |
| `122` | `b139-timepressure-midprogress-complete150-l13-screen` | `b139_l13_timepressure_midprogress_complete150_s80_20260628T130904Z` | [`lw4u8u62`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/lw4u8u62) |
| `121` | `b140-bounded-terminal150-longcredit-l13-screen` | `b140_l13_bounded_terminal150_longcredit_s80_20260628T130900Z` | [`usnsnb48`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/usnsnb48) |

B139 keeps score-mode reward but adds mild time pressure, lower progress
dominance, B55-level death pressure, a larger true-clear bonus, advantage
normalization, and a higher entropy floor. B140 tests whether score-mode
progress shaping itself is the attractor by using bounded reward with a tighter
progress cap and larger terminal reward. Both keep the same game, Level1-3
state, action set, info events, `done_on_events=life_loss,level_change`, cap,
checkpoint cadence, and strict `train/info/level_complete/rate/min/last > 0.99`
stop.

After B132/B133 exited, both replacements were claimed:

```text
train_jobs: {"canceled": 19, "failed": 5, "running": 5}
eval_jobs: {}
queue demand: profile=any target=rtx4090 pending=0 running=5 digest=063e55231d69
```

Initial telemetry at `2026-06-28T13:11Z`:

| Arm | Step | L1-3 count | Reward |
| --- | ---: | ---: | ---: |
| B139 | `38992` | `0` | `42.17194` |
| B140 | `43152` | `0` | `8.410002` |

## B134 Plateau Backfill

A refresh at `2026-06-28T13:14Z` showed B134 past `2.67M` steps with no clean
Level1-3 clears and an effectively flat reward tail. B137 and B138 were still
improving, while B139 and B140 were too young to judge.

| Arm | Job | History step | L1-3 count | L1-3 rate | Last reward | Reward delta over ~500k | Reward delta over ~250k | Reward delta over ~100k | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| B134 | `116` | `2670592` | `0` | `0` | `92.93` | `+0.653` | `+0.0675` | `-0.055` | cancel |
| B137 | `119` | `1957888` | `0` | `0` | `-26.658` | `+13.815` | `+9.044` | `+4.752` | keep |
| B138 | `120` | `917504` | `0` | `0` | `51.824` | `+36.5865` | `+23.8195` | `+9.8095` | keep |
| B139 | `122` | `286720` | `0` | `0` | `61.5269` | `+26.5488` | `+13.747` | `+6.912` | keep, young |
| B140 | `121` | `311296` | `0` | `0` | `11.549` | `+4.392` | `+2.137` | `+0.662` | keep, young |

B134 was canceled with `cancel_requested=1`.

One replacement screen was added:

| Job | Spec | Run | W&B |
| ---: | --- | --- | --- |
| `123` | `b141-baseline-terminal250-longcredit-l13-screen` | `b141_l13_baseline_terminal250_longcredit_s80_20260628T131528Z` | [`dhrjnowp`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/dhrjnowp) |

B141 tests a different legal reward family instead of another direct
x-progress-shaped variant: `reward_mode=baseline`, `terminal_reward=250`,
`reward_scale=10`, long-credit PPO (`n_steps=1024`, `gamma=0.97`,
`gae_lambda=0.95`), advantage normalization, and a high entropy floor. It keeps
the same game, Level1-3 state, action set, info events,
`done_on_events=life_loss,level_change`, cap, checkpoint cadence, and strict
`train/info/level_complete/rate/min/last > 0.99` stop.

After B134 exited, B141 was claimed:

```text
train_jobs: {"canceled": 20, "failed": 5, "running": 5}
eval_jobs: {}
queue demand: profile=any target=rtx4090 pending=0 running=5 digest=063e55231d69
```

Initial B141 telemetry at `2026-06-28T13:17Z`: step `43984`, Level1-3 count
`0`, reward `-2.482998`.

## B137 Plateau Backfill

A refresh at `2026-06-28T13:22Z` showed no clean Level1-3 clears across the
active batch. B137 was the only run old enough to recycle: it had passed the
2M checkpoint, still had peak clean-clear count/rate of zero, and its short
reward tail was effectively flat. B138-B141 were still improving or too young.

| Arm | Job | History step | Peak L1-3 count | Peak L1-3 rate | Last reward | Reward delta over ~500k | Reward delta over ~250k | Reward delta over ~100k | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| B137 | `119` | `2392064` | `0` | `0` | `-12.068` | `+16.95` | `+1.745` | `+0.087` | cancel |
| B138 | `120` | `1441792` | `0` | `0` | `73.759` | `+22.9115` | `+1.985` | `+1.529` | keep |
| B139 | `122` | `786432` | `0` | `0` | `160.922` | `+99.3948` | `+66.4744` | `+12.133` | keep |
| B140 | `121` | `851968` | `0` | `0` | `38.073` | `+23.755` | `+18.785` | `+8.855` | keep |
| B141 | `123` | `376832` | `0` | `0` | `4.8975` | `+7.805` | `+6.475` | `+2.5355` | keep |

B137 was canceled with `cancel_requested=1`.

One replacement screen was added:

| Job | Spec | Run | W&B |
| ---: | --- | --- | --- |
| `124` | `b142-ultralowdeath-tinyprogress-complete250-l13-screen` | `b142_l13_ultralowdeath_tinyprogress_complete250_s80_20260628T132311Z` | [`n7q5b51g`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/n7q5b51g) |

B142 follows up B137 by lowering death pressure (`death_penalty=10`), raising
the true clear bonus (`completion_reward=250`), keeping only a tiny progress
gradient (`progress_reward_scale=0.05`), and using long-credit PPO plus a
higher entropy floor (`n_steps=1024`, `gamma=0.97`, `gae_lambda=0.95`,
`ent_coef_final=0.005`). It keeps the same game, Level1-3 state, action set,
info events, `done_on_events=life_loss,level_change`, cap, checkpoint cadence,
and strict `train/info/level_complete/rate/min/last > 0.99` stop.

After B137 exited, B142 was claimed:

```text
train_jobs: {"canceled": 21, "failed": 5, "running": 5}
eval_jobs: {}
queue demand: profile=any target=rtx4090 pending=0 running=5 digest=063e55231d69
```

Initial B142 telemetry at `2026-06-28T13:26Z`: step `80416`, Level1-3 count
`0`, reward `2.2790012`, FPS `1315`, KL `0.0060406104`, clip fraction
`0.065264896`.

## B138/B139/B140 Plateau Backfill

A refresh at `2026-06-28T13:35Z` matched the operator's W&B chart read:
`rollout/ep_rew_mean` had flattened while the actual Level1-3 clean-clear
metrics were still zero. B138 and B140 were the clearest stalled slots; B139
was also at a high shaped-reward plateau with no conversion into the true
level-change objective. B141 and B142 were preserved because their reward tails
were still moving and they were testing distinct legal reward families.

| Arm | Job | History step | Peak L1-3 count | Peak L1-3 rate | Last reward | Reward delta over ~500k | Reward delta over ~250k | Reward delta over ~100k | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| B138 | `120` | `2598736` | `0` | `0` | `74.9975` | `+0.5455` | `+0.5455` | n/a | cancel |
| B139 | `122` | `1856320` | `0` | `0` | `168.761` | `+1.70049` | n/a | n/a | cancel |
| B140 | `121` | `1980544` | `0` | `0` | `51.937` | `+0.002` | `+0.002` | `+0.008` | cancel |
| B141 | `123` | `1483648` | `0` | `0` | `36.244` | `+14.3338` | `+1.275` | `+1.275` | keep |
| B142 | `124` | `838368` | `0` | `0` | `21.129` | `+16.0635` | `+8.9245` | `+1.288` | keep |

B138, B139, and B140 were canceled with `cancel_requested=1`. This is another
negative result for shaped progress: several reward variants can learn a stable
local optimum in Level1-3 without ever emitting
`train/info/level_complete/from/0-2/count`.

Three replacement screens were added and launched:

| Job | Spec | Run | W&B |
| ---: | --- | --- | --- |
| `127` | `b143-b55lowprogress-complete300-l13-screen` | `b143_l13_b55lowprogress_complete300_s80_20260628T133833Z` | [`xym16ra4`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/xym16ra4) |
| `125` | `b144-b46lowprogress-timecomplete300-l13-screen` | `b144_l13_b46lowprogress_timecomplete300_s80_20260628T133831Z` | [`o664t2km`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/o664t2km) |
| `126` | `b145-b55baseline-terminal300-l13-screen` | `b145_l13_b55baseline_terminal300_s80_20260628T133832Z` | [`f0l1kg93`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/f0l1kg93) |

B143 returns to the confirmed Level1-1 B55 PPO shape
(`n_steps=512`, `gamma=0.9`, `gae_lambda=1.0`, no advantage normalization,
`target_kl=0.16`) while reducing direct progress pressure
(`progress_reward_scale=0.15`, clipped), lowering death pressure
(`death_penalty=10`), and making true clean clears dominate the score-mode
reward (`completion_reward=300`). B144 uses the higher-update-pressure B46
shape (`target_kl=0.20`, fixed LR), lower progress pressure
(`progress_reward_scale=0.10`), mild time pressure (`time_penalty=0.03`), and
`completion_reward=300`. B145 combines the B55 PPO shape with
`reward_mode=baseline` and `terminal_reward=300` so it removes direct x-progress
shaping while keeping a strong level-change versus death signal. All three keep
the same game, Level1-3 state, action set, info events,
`done_on_events=life_loss,level_change`, cap, checkpoint cadence, and strict
`train/info/level_complete/rate/min/last > 0.99` stop.

After the canceled jobs exited, all replacements were claimed:

```text
train_jobs: {"canceled": 24, "failed": 5, "running": 5}
eval_jobs: {}
queue demand: profile=any target=rtx4090 pending=0 running=5 digest=063e55231d69
```

Final active telemetry at `2026-06-28T13:40Z`:

| Arm | Job | Step | Peak L1-3 count | Peak L1-3 rate | Last reward | W&B |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| B141 | `123` | `1897168` | `0` | `0` | `12.7867` | [`dhrjnowp`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/dhrjnowp) |
| B142 | `124` | `1277952` | `0` | `0` | `23.2565` | [`n7q5b51g`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/n7q5b51g) |
| B143 | `127` | `113184` | `0` | `0` | `30.237` | [`xym16ra4`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/xym16ra4) |
| B144 | `125` | `94592` | `0` | `0` | `-5.33802` | [`o664t2km`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/o664t2km) |
| B145 | `126` | `67280` | `0` | `0` | `-5.6775` | [`f0l1kg93`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/f0l1kg93) |

## B141 Low-LR / Long-Horizon Backfill

A refresh at `2026-06-28T13:45Z` still showed no clean Level1-3 clears in the
active batch. B141 was now past the 2M fixed-budget checkpoint with peak
`train/info/level_complete/from/0-2/count = 0` and a flat reward tail, so it was
recycled. The operator also called out the known Level1-3 PPO failure mode:
delayed precise jumps, harsh pit exploration, and premature deterministic
policies. The next backfill therefore prioritizes the commonly reported fixes
that are legal under this goal: lower learning rate, higher early entropy,
longer rollouts, and higher gamma.

| Arm | Job | History step | Peak L1-3 count | Peak L1-3 rate | Last reward | Reward delta over ~500k | Reward delta over ~250k | Reward delta over ~100k | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| B141 | `123` | `2260992` | `0` | `0` | `41.52` | `+0.734001` | `+0.734001` | n/a | cancel |
| B142 | `124` | `1615568` | `0` | `0` | `23.5125` | `+0.840002` | `+0.057499` | `+0.396504` | keep |
| B143 | `127` | `427088` | `0` | `0` | `55.3295` | `+28.568` | `+25.0385` | `+18.549` | keep, young |
| B144 | `125` | `425936` | `0` | `0` | `4.0919` | `+10.8112` | `+4.20161` | `-0.297595` | keep, young |
| B145 | `126` | `374944` | `0` | `0` | `-0.0387506` | `+7.90125` | `+3.46325` | `+1.46575` | keep, young |

B141 was canceled with `cancel_requested=1`.

One replacement screen was added:

| Job | Spec | Run | W&B |
| ---: | --- | --- | --- |
| `128` | `b146-lowlr-gamma995-highentropy-complete300-l13-screen` | `b146_l13_lowlr_gamma995_highentropy_complete300_s80_20260628T134736Z` | [`hfav6z8b`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/hfav6z8b) |

B146 keeps direct Level1-3 training, simple actions, frame skip 4, the same info
events, `done_on_events=life_loss,level_change`, the 5M cap, and strict
`train/info/level_complete/rate/min/last > 0.99` stop. Its legal hyperparameter
and reward changes are `learning_rate=7e-5 -> 4e-5`, `n_steps=2048`,
`gamma=0.995`, `gae_lambda=0.97`, `ent_coef=0.03 -> 0.005`,
`normalize_advantage=true`, clipped score-mode progress with
`progress_reward_scale=0.10`, `death_penalty=10`, and
`completion_reward=300`. The goal is to keep exploration alive long enough to
discover safe jump timing without changing the environment, action semantics,
state, or promotion metric.

Current training metrics do not by themselves answer whether policies die at
the first pit or later platforms; that requires out-of-process eval/playback
using the eval `max_x` and death-position metrics once a run has a plausible
checkpoint. Until then, pruning is based on target clean-clear count/rate plus
reward-tail stagnation.

After B141 exited, B146 was claimed:

```text
train_jobs: {"canceled": 25, "failed": 5, "running": 5}
eval_jobs: {}
queue demand: profile=any target=rtx4090 pending=0 running=5 digest=063e55231d69
```

Final active telemetry at `2026-06-28T13:49Z`:

| Arm | Job | Step | Peak L1-3 count | Peak L1-3 rate | Last reward | W&B |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| B142 | `124` | `1998848` | `0` | `0` | `23.5435` | [`n7q5b51g`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/n7q5b51g) |
| B143 | `127` | `757152` | `0` | `0` | `27.1985` | [`xym16ra4`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/xym16ra4) |
| B144 | `125` | `760144` | `0` | `0` | `30.142` | [`o664t2km`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/o664t2km) |
| B145 | `126` | `707824` | `0` | `0` | `8.8225` | [`f0l1kg93`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/f0l1kg93) |
| B146 | `128` | `77488` | `0` | `0` | `13.104` | [`hfav6z8b`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/hfav6z8b) |

## B142 Low-LR Baseline Companion Backfill

A second refresh at `2026-06-28T13:51Z` showed B142 had crossed the 2M
checkpoint with no clean clears and an essentially flat reward tail. B143-B145
were still too young, and B146 had just started. B142 was therefore recycled
under the same zero-clear plateau rule.

| Arm | Job | History step | Peak L1-3 count | Peak L1-3 rate | Last reward | Reward delta over ~500k | Reward delta over ~250k | Reward delta over ~100k | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| B142 | `124` | `2098512` | `0` | `0` | `23.5985` | `+0.028999` | `+0.028999` | n/a | cancel |
| B143 | `127` | `868704` | `0` | `0` | `85.041` | `-3.34951` | `-3.34951` | `-4.36049` | keep, young |
| B144 | `125` | `848960` | `0` | `0` | `30.142` | `+30.2517` | `+15.3871` | `+1.99681` | keep, young |
| B145 | `126` | `819632` | `0` | `0` | `30.531` | `+27.971` | `+18.4695` | `+9.74001` | keep, young |
| B146 | `128` | `178704` | `0` | `0` | `16.174` | `+3.507` | `+3.507` | `+3.07` | keep, fresh |

B142 was canceled with `cancel_requested=1`.

One replacement screen was added:

| Job | Spec | Run | W&B |
| ---: | --- | --- | --- |
| `129` | `b147-lowlr-baseline-gamma995-terminal300-l13-screen` | `b147_l13_lowlr_baseline_gamma995_terminal300_s80_20260628T135221Z` | [`0tgg10xf`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/0tgg10xf) |

B147 is a low-LR/high-entropy/high-gamma companion to B146 that removes direct
x-progress shaping. It uses `reward_mode=baseline`, `terminal_reward=300`,
`learning_rate=7e-5 -> 4e-5`, `n_steps=2048`, `gamma=0.995`,
`gae_lambda=0.97`, `ent_coef=0.03 -> 0.005`,
`normalize_advantage=true`, and `target_kl=0.12`. It preserves the same direct
Level1-3 state, simple action set, frame skip 4, info events,
`done_on_events=life_loss,level_change`, cap, checkpoint cadence, and strict
`train/info/level_complete/rate/min/last > 0.99` stop.

After B142 exited, B147 was claimed:

```text
train_jobs: {"canceled": 26, "failed": 5, "running": 5}
eval_jobs: {}
queue demand: profile=any target=rtx4090 pending=0 running=5 digest=063e55231d69
```

Final active telemetry at `2026-06-28T13:54Z`:

| Arm | Job | Step | Peak L1-3 count | Peak L1-3 rate | Last reward | W&B |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| B143 | `127` | `1096256` | `0` | `0` | `88.6485` | [`xym16ra4`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/xym16ra4) |
| B144 | `125` | `1089536` | `0` | `0` | `29.8659` | [`o664t2km`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/o664t2km) |
| B145 | `126` | `1040384` | `0` | `0` | `2.918` | [`f0l1kg93`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/f0l1kg93) |
| B146 | `128` | `449808` | `0` | `0` | `17.597` | [`hfav6z8b`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/hfav6z8b) |
| B147 | `129` | `77264` | `0` | `0` | `-7.6475` | [`0tgg10xf`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/0tgg10xf) |

## Active Batch Monitor

A refresh at `2026-06-28T13:58Z` showed the active batch still had no clean
Level1-3 clears. B143 and B144 were flattening but had not yet reached the 2M
fixed-budget maturity point used for the current zero-clear plateau rule. B145
was still improving, and B146/B147 were fresh low-LR/high-entropy tests.

| Arm | Job | History step | Peak L1-3 count | Peak L1-3 rate | Last reward | Reward delta over ~500k | Reward delta over ~250k | Reward delta over ~100k | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| B143 | `127` | `1409024` | `0` | `0` | `88.6155` | `-1.12448` | `+0.71802` | `+0.48052` | keep, not mature |
| B144 | `125` | `1405792` | `0` | `0` | `35.9127` | `+1.42721` | `-1.2872` | `-0.319697` | keep, not mature |
| B145 | `126` | `1351488` | `0` | `0` | `36.4025` | `+4.69` | `+1.679` | n/a | keep, still moving |
| B146 | `128` | `786432` | `0` | `0` | `21.028` | `+6.126` | `+2.012` | `+1.133` | keep, young |
| B147 | `129` | `421664` | `0` | `0` | `-4.9965` | `+2.409` | `+0.373251` | `+0.721001` | keep, fresh |

No jobs were canceled from this snapshot. The next recycle candidates, if they
remain zero-clear, are B143 and B144 once they pass the 2M checkpoint or show a
stronger plateau signal; otherwise continue to protect B146/B147 long enough to
test the low-LR/high-entropy/gamma=0.995 hypothesis.

## B143/B144/B145 Plateau Backfill

A refresh at `2026-06-28T14:09Z` showed B143, B144, and B145 past the 2M
checkpoint with peak Level1-3 clean-clear count/rate still zero. B143 had a
negative long-tail reward delta, while B144/B145 had effectively flat short
tails. They were recycled under the same rule: the target metric is checked
first, and `rollout/ep_rew_mean` is used only as stale-local-optimum evidence.
B146 and B147 were protected because they were younger low-LR/high-entropy
tests, and B146 was still moving.

| Arm | Job | History step | Peak L1-3 count | Peak L1-3 rate | Last reward | Reward delta over ~500k | Reward delta over ~250k | Reward delta over ~100k | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| B143 | `127` | `2195456` | `0` | `0` | `89.0555` | `-1.72548` | n/a | n/a | cancel |
| B144 | `125` | `2183904` | `0` | `0` | `37.8464` | `+0.171699` | `+0.356201` | `+0.051399` | cancel |
| B145 | `126` | `2141696` | `0` | `0` | `36.572` | `+0.371002` | `+0.402004` | `+0.494006` | cancel |
| B146 | `128` | `1644400` | `0` | `0` | `33.741` | `+5.799` | `+3.447` | `+5.30499` | keep, young/moving |
| B147 | `129` | `1254864` | `0` | `0` | `-3.053` | `+0.745` | `+0.394501` | `+0.133499` | keep, young |

B143, B144, and B145 were canceled with `cancel_requested=1`.

Three replacement screens were added and launched:

| Job | Spec | Run | W&B |
| ---: | --- | --- | --- |
| `131` | `b148-lowlr-bounded-terminal500-gamma995-l13-screen` | `b148_l13_lowlr_bounded_terminal500_gamma995_s80_20260628T141257Z` | [`ic2bvhvs`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/ic2bvhvs) |
| `130` | `b149-lowlr-additive-minprogress-complete500-l13-screen` | `b149_l13_lowlr_additive_minprogress_complete500_s80_20260628T141257Z` | [`h0gvy3ng`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/h0gvy3ng) |
| `132` | `b150-lowlr-explore-death0-complete750-l13-screen` | `b150_l13_lowlr_explore_death0_complete750_s80_20260628T141302Z` | [`hnprwhxt`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/hnprwhxt) |

B148 tests whether a bounded reward with a larger true terminal reward can
avoid the score-shaped high-reward/no-clear attractor. B149 uses additive
min-progress shaping with a stronger completion reward while keeping low LR,
high gamma, and early entropy. B150 is the most exploration-biased screen: zero
death penalty, larger completion reward, lower LR, longer rollout horizon, and
the highest early entropy. All three keep the same direct Level1-3 state,
simple action set, frame skip 4, info events,
`done_on_events=life_loss,level_change`, 5M cap, checkpoint cadence, and strict
`train/info/level_complete/rate/min/last > 0.99` training stop.

After the canceled jobs exited, all replacements were claimed:

```text
train_jobs: {"canceled": 29, "failed": 5, "running": 5}
eval_jobs: {}
queue demand: profile=any target=rtx4090 pending=0 running=5 digest=063e55231d69
```

Fresh W&B telemetry at `2026-06-28T14:16Z`:

| Arm | Job | Step | Peak L1-3 count | Peak L1-3 rate | Min clear rate | Last reward | W&B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| B146 | `128` | `2196768` | `0` | `0` | `0` | `39.85` | [`hfav6z8b`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/hfav6z8b) |
| B147 | `129` | `1818448` | `0` | `0` | `0` | n/a | [`0tgg10xf`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/0tgg10xf) |
| B148 | `131` | `140368` | `0` | `0` | `0` | n/a | [`ic2bvhvs`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/ic2bvhvs) |
| B149 | `130` | `118736` | `0` | `0` | `0` | n/a | [`h0gvy3ng`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/h0gvy3ng) |
| B150 | `132` | `124592` | `0` | `0` | `0` | `4.4244` | [`hnprwhxt`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/hnprwhxt) |

B146 is now above 2M with zero clears, but its reward tail has not fully
flattened under the low-LR/high-entropy/gamma=0.995 hypothesis, so it remains
active for the next monitor pass. B147 is approaching the maturity point but is
still younger than the recycled slots. B148-B150 are too fresh to judge.

## B146 Survival-Bonus Backfill

A refresh at `2026-06-28T14:26Z` still showed no clean Level1-3 clears in the
active batch. B146 had crossed the 3M fixed-budget checkpoint with peak
`train/info/level_complete/from/0-2/count = 0` and a nearly flat short
`rollout/ep_rew_mean` tail. The longer 500k reward delta was still slightly
positive, but the latest tail had settled around `56`, the reward mass was
dominated by x-progress, and no true clean-clear signal had appeared after more
than 3M steps. B147 was protected because its reward tail was still moving
materially; B148-B150 were still below the 2M maturity point.

| Arm | Job | History step | Peak L1-3 count | Peak L1-3 rate | Last reward | Reward delta over ~500k | Reward delta over ~250k | Reward delta over ~100k | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| B146 | `128` | `3014656` | `0` | `0` | `56.647` | `+3.43901` | `+1.231` | `+0.546005` | cancel |
| B147 | `129` | `2621440` | `0` | `0` | `31.0315` | `+16.29` | `+6.305` | `+5.25201` | keep, still moving |
| B148 | `131` | `950272` | `0` | `0` | `-33.18` | `+0.273998` | `+0.782996` | `+0.285` | keep, young |
| B149 | `130` | `950272` | `0` | `0` | `2.1138` | `+1.8476` | `+1.7154` | `+0.835599` | keep, young |
| B150 | `132` | `917504` | `0` | `0` | `5.3378` | `+0.3356` | `+0.2302` | `+0.1732` | keep, young |

B146 was canceled with `cancel_requested=1`.

One replacement screen was added and launched:

| Job | Spec | Run | W&B |
| ---: | --- | --- | --- |
| `133` | `b151-lowlr-survival-minprogress-complete1000-l13-screen` | `b151_l13_lowlr_survival_minprogress_complete1000_s80_20260628T143045Z` | [`pob5mfbv`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/pob5mfbv) |

B151 is a spec-only survival-bonus follow-up to the Level1-3 PPO failure mode
the operator highlighted. The stable-retro data for this game exposes x scroll,
score, lives, time, and level variables, but not a player y-position, so this
arm does not add height/platform shaping or change the emulator data contract.
Instead it uses the existing reward hooks: `time_penalty=-0.005` as a small
per-step survival bonus, `progress_reward_scale=0.01`, `death_penalty=2`,
`completion_reward=1000`, `score_progress_clipped=true`, low LR
`5e-5 -> 3e-5`, `n_steps=4096`, `batch_size=2048`, `gamma=0.995`,
`gae_lambda=0.98`, `ent_coef=0.05 -> 0.01`, `normalize_advantage=true`, and
`target_kl=0.10`. The max survival bonus is intentionally much smaller than a
true clean clear, so reward can diagnose patience versus waiting, but promotion
still depends only on the goal completion metric and out-of-process eval.

Because `rlab-train-image.json` was absent at repo root, B151 was enqueued
without `--runtime-image-ref-file`, using the queue CLI's latest successful
immutable image resolver as recorded in the goal decision note.

After B146 exited, B151 was claimed:

```text
train_jobs: {"canceled": 30, "failed": 5, "running": 5}
eval_jobs: {}
queue demand: profile=any target=rtx4090 pending=0 running=5 digest=063e55231d69
```

Fresh W&B telemetry at `2026-06-28T14:33Z`:

| Arm | Job | Step | Peak L1-3 count | Peak L1-3 rate | Min clear rate | Last reward | W&B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| B147 | `129` | `3097440` | `0` | `0` | `0` | `34.35` | [`0tgg10xf`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/0tgg10xf) |
| B148 | `131` | `1442928` | `0` | `0` | `0` | `-36.626` | [`ic2bvhvs`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/ic2bvhvs) |
| B149 | `130` | `1457984` | `0` | `0` | `0` | `6.3578` | [`h0gvy3ng`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/h0gvy3ng) |
| B150 | `132` | `1455120` | `0` | `0` | `0` | `5.8776` | [`hnprwhxt`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/hnprwhxt) |
| B151 | `133` | `83264` | `0` | `0` | `0` | `0.838507` | [`pob5mfbv`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/pob5mfbv) |

Next monitor priorities are B147 first, because it is mature but still moving,
then B148-B150 once they cross the 2M checkpoint. Any nonzero
`train/info/level_complete/from/0-2/count` should stop pruning and trigger a
candidate-checkpoint eval path.

## B147/B148/B149 Fast-Update Backfill

A refresh at `2026-06-28T14:37Z` still showed no clean Level1-3 clears.
B147 had crossed 3.4M steps with peak count/rate/min still zero. W&B's sampled
history only returned the current reward row in this pass, but the live reward
summary had barely moved from the previous recorded `3.097M` snapshot
(`34.35` to roughly `35.64`) while the target metric stayed at zero, so B147
was retired. B148 crossed 2.2M with a death-dominated negative reward plateau,
and B149 crossed 2.4M with a nearly flat additive tiny-progress reward tail.
B150 was protected despite being mature because it is still the cleanest
death-zero exploration test and continued moving upward; B151 was still young.

| Arm | Job | History step | Peak L1-3 count | Peak L1-3 rate | Last reward | Reward/readout note | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| B147 | `129` | `3407872` | `0` | `0` | `35.642` | only `+1.292` from the prior `3.097M` report row | cancel |
| B148 | `131` | `2204272` | `0` | `0` | `-35.265` | bounded reward, death share about `0.764` | cancel |
| B149 | `130` | `2457600` | `0` | `0` | `8.31619` | only `+0.188` since `2.178M` | cancel |
| B150 | `132` | `2818048` | `0` | `0` | `9.3874` | death-zero exploration arm still moving | keep |
| B151 | `133` | `1455680` | `0` | `0` | `1.49846` | young survival-bonus arm | keep |

B147, B148, and B149 were canceled with `cancel_requested=1`.

Three replacement screens were added and launched:

| Job | Spec | Run | W&B |
| ---: | --- | --- | --- |
| `134` | `b152-lowlr-fastupdate-deathlight-complete1000-l13-screen` | `b152_l13_lowlr_fastupdate_deathlight_complete1000_s80_20260628T144057Z` | [`hx6og3az`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/hx6og3az) |
| `135` | `b153-additive-survival-fastupdate-complete1000-l13-screen` | `b153_l13_additive_survival_fastupdate_complete1000_s80_20260628T144436Z` | [`m1cs6htr`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/m1cs6htr) |
| `136` | `b154-lowlr-fastupdate-rawprogress-complete1000-l13-screen` | `b154_l13_lowlr_fastupdate_rawprogress_complete1000_s80_20260628T144804Z` | [`0svlupww`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/0svlupww) |

B152 tests whether the low-LR/high-entropy/death-light recipe needs frequent
PPO updates (`n_steps=512`, `batch_size=512`) instead of the long rollout
horizons used in B146-B151. B153 applies the same fast-update idea to additive
reward with the B151 small survival bonus and no score reward, after bounded
reward in B148 proved too death-dominated. B154 keeps the fast-update/death-light
setup but uses raw score-mode x-progress instead of additive tiny progress, to
test whether stronger forward-landing reinforcement helps the platform timing.
All three keep the same Level1-3 state, simple action set, frame skip 4, info
events, `done_on_events=life_loss,level_change`, 5M cap, checkpoint cadence,
and strict `train/info/level_complete/rate/min/last > 0.99` training stop.

As before, `rlab-train-image.json` was absent at repo root, so these jobs were
enqueued without `--runtime-image-ref-file` and used the queue CLI's latest
successful immutable image resolver.

After the canceled jobs exited, all replacements were claimed:

```text
train_jobs: {"canceled": 33, "failed": 5, "running": 5}
eval_jobs: {}
queue demand: profile=any target=rtx4090 pending=0 running=5 digest=063e55231d69
```

Fresh W&B telemetry at `2026-06-28T14:50Z`:

| Arm | Job | Step | Peak L1-3 count | Peak L1-3 rate | Min clear rate | Last reward | W&B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| B150 | `132` | `2818048` | `0` | `0` | `0` | `9.3874` | [`hnprwhxt`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/hnprwhxt) |
| B151 | `133` | `1455680` | `0` | `0` | `0` | `1.49846` | [`pob5mfbv`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/pob5mfbv) |
| B152 | `134` | `588416` | `0` | `0` | `0` | `3.8006` | [`hx6og3az`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/hx6og3az) |
| B153 | `135` | `336000` | `0` | `0` | `0` | `1.40541` | [`m1cs6htr`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/m1cs6htr) |
| B154 | `136` | `74912` | `0` | `0` | `0` | `3.0098` | [`0svlupww`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/0svlupww) |

Next monitor priorities are B150 first, then B151 as it approaches 2M, while
B152-B154 need enough updates to test the fast-update hypothesis. Any nonzero
Level1-3 clean-clear count should switch the loop from pruning to candidate
eval/confirmation planning.

## B150 Gentle-Pressure Tiny-Progress Backfill

A refresh at `2026-06-28T15:02Z` still showed no clean Level1-3 clears in the
active batch. B150 was now the mature recycle candidate: it had crossed
3.5M steps with peak Level1-3 count/rate/min still zero, and its latest reward
mass was almost pure x-progress (`train/reward_share/prog_x ~= 0.9995`). That
is the same run-right/no-clear attractor the Level1-3 search is trying to avoid.
B151 was mature enough to watch but still moving, while B152-B154 were younger
fast-update tests and had positive reward tails.

| Arm | Job | History step | Peak L1-3 count | Peak L1-3 rate | Last reward | Reward delta over ~500k | Reward delta over ~250k | Reward delta over ~100k | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| B150 | `132` | `3538944` | `0` | `0` | `12.865` | `+1.8646` | `+0.4978` | `+0.1754` | cancel |
| B151 | `133` | `2162688` | `0` | `0` | `2.89626` | `+1.1746` | `+0.5299` | `+0.1748` | keep, survival test still moving |
| B152 | `134` | `1253376` | `0` | `0` | `10.232` | `+4.6386` | `+1.8642` | `+0.3566` | keep, young fast-update test |
| B153 | `135` | `1015808` | `0` | `0` | `4.69141` | `+2.6640` | `+1.7457` | `+0.5686` | keep, young fast-update test |
| B154 | `136` | `745472` | `0` | `0` | `5.6158` | `+2.9736` | `+2.0110` | `+1.7502` | keep, young fast-update test |

B150 was canceled with `cancel_requested=1`.

One replacement screen was added and launched:

| Job | Spec | Run | W&B |
| ---: | --- | --- | --- |
| `137` | `b155-lowlr-gentlepress-tinyprogress-complete1500-l13-screen` | `b155_l13_lowlr_gentlepress_tinyprogress_complete1500_s80_20260628T150159Z` | [`niggrdul`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/niggrdul) |

B155 keeps the direct Level1-3 start state, simple action set, frame skip 4,
info events, `done_on_events=life_loss,level_change`, 5M cap, checkpoint
cadence, and strict `train/info/level_complete/rate/min/last > 0.99` stop.
The controlled change is to suppress the dense x-progress attractor while
retaining exploration: clipped progress cap `10`, `progress_reward_scale=0.003`,
`time_penalty=-0.002` as a small survival bonus, `death_penalty=1`,
`completion_reward=1500`, LR `7e-5 -> 5e-5`, `n_steps=1024`,
`batch_size=1024`, `gamma=0.995`, `gae_lambda=0.98`,
`ent_coef=0.05 -> 0.005`, `normalize_advantage=true`, `clip_range=0.12`,
`clip_range_vf=0.2`, `target_kl=0.12`, and `vf_coef=0.5`.

As before, `rlab-train-image.json` was absent at repo root, so B155 was
enqueued without `--runtime-image-ref-file` and used the queue CLI's latest
successful immutable image resolver.

After B150 exited, B155 was claimed:

```text
train_jobs: {"canceled": 34, "failed": 5, "running": 5}
eval_jobs: {}
queue demand: profile=any target=rtx4090 pending=0 running=5 digest=063e55231d69
```

Active jobs are now B151-B155 on beast-3. Next monitor priorities are B151 first
as the oldest survival-bonus arm, then B152/B153 as they cross the 2M maturity
point. Any nonzero `train/info/level_complete/from/0-2/count` should switch the
loop from pruning to candidate checkpoint/eval planning.

## B151/B152/B153 Capacity-Entropy-Sparse Backfill

A refresh at `2026-06-28T15:21Z` still showed no clean Level1-3 clears. B151
had crossed 3.5M steps with a flat reward tail. B152 and B153 had crossed their
2M maturity points with zero target movement and flat or negative short reward
tails. B154 was protected because it was just under the maturity point and still
had a positive short tail; B155 was still too young to judge.

| Arm | Job | History step | Peak L1-3 count | Peak L1-3 rate | Last reward | Reward delta over ~500k | Reward delta over ~250k | Reward delta over ~100k | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| B151 | `133` | `3538944` | `0` | `0` | `5.27461` | `+0.1018` | `+0.16245` | `-0.05035` | cancel |
| B152 | `134` | `2449408` | `0` | `0` | `11.024` | `+0.0134` | `-0.0790` | `-0.1000` | cancel |
| B153 | `135` | `2211840` | `0` | `0` | `5.21846` | `+0.0287` | `+0.15765` | `-0.0525` | cancel |
| B154 | `136` | `1941504` | `0` | `0` | `11.056` | `+0.1534` | `+0.1268` | `+0.2168` | keep, not quite mature/still moving |
| B155 | `137` | `1081344` | `0` | `0` | `0.951303` | `+0.5385` | `+0.1173` | `+0.0164` | keep, young |

B151, B152, and B153 were canceled with `cancel_requested=1`.

Three replacement screens were added and launched:

| Job | Spec | Run | W&B |
| ---: | --- | --- | --- |
| `139` | `b156-lowlr-criticwide-tinyprogress-complete1500-l13-screen` | `b156_l13_lowlr_criticwide_tinyprogress_complete1500_s80_20260628T151901Z` | [`zjo4h4vl`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/zjo4h4vl) |
| `138` | `b157-lowlr-entropyburst-fastupdate-complete2000-l13-screen` | `b157_l13_lowlr_entropyburst_fastupdate_complete2000_s80_20260628T151856Z` | [`g031mpvr`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/g031mpvr) |
| `140` | `b158-sparse-longcredit-survival-complete2000-l13-screen` | `b158_l13_sparse_longcredit_survival_complete2000_s80_20260628T151904Z` | [`of27idx0`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/of27idx0) |

B156 tests the value-estimation angle by adding B110-style asymmetric network
capacity (`policy_net_arch=64,64`, `value_net_arch=512,512`) to the B155
tiny-progress/gentle-pressure reward. B157 tests whether the fast-update arms
needed a much larger early entropy burst rather than a different reward scale:
`ent_coef=0.10 -> 0.02`, `n_steps=512`, tiny clipped progress, zero explicit
death penalty, and `completion_reward=2000`. B158 tests an almost-sparse
long-credit survival reward: additive reward, `progress_reward_scale=0.0005`,
`gamma=0.997`, `gae_lambda=0.99`, `n_steps=4096`, a small survival bonus, light
death pressure, and `completion_reward=2000`.

All three keep the direct Level1-3 start state, simple action set, frame skip 4,
info events, `done_on_events=life_loss,level_change`, 5M cap, checkpoint
cadence, and strict `train/info/level_complete/rate/min/last > 0.99` stop.
They remain legal reward/hyperparameter-only screens; no emulator state, action
semantics, observation wrapper, or eval protocol was changed.

As before, `rlab-train-image.json` was absent at repo root, so these jobs were
enqueued without `--runtime-image-ref-file` and used the queue CLI's latest
successful immutable image resolver.

After the canceled jobs exited, the replacements were claimed:

```text
train_jobs: {"canceled": 37, "failed": 5, "running": 5}
eval_jobs: {}
queue demand: profile=any target=rtx4090 pending=0 running=5 digest=063e55231d69
```

Active jobs are now B154-B158 on beast-3. Next monitor priorities are B154 first
as it crosses the 2M maturity point, then B155 once it passes 2M. Any nonzero
`train/info/level_complete/from/0-2/count` should switch the loop from pruning
to candidate checkpoint/eval planning.

## B154 Micro-Update Entropy Backfill

A refresh at `2026-06-28T15:28Z` still showed no clean Level1-3 clears. B154
had crossed 2.5M steps with zero target movement and a very flat reward tail.
It was the active raw-progress fast-update arm, and its reward share was still
mostly progress, so it was recycled as another no-clear progress local optimum.
B155 was protected because it was still below the 2M maturity point, and
B156-B158 were too young to judge.

| Arm | Job | History step | Peak L1-3 count | Peak L1-3 rate | Last reward | Reward delta over ~500k | Reward delta over ~250k | Reward delta over ~100k | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| B154 | `136` | `2555904` | `0` | `0` | `11.313` | `+0.3208` | `+0.0548` | `+0.0554` | cancel |
| B155 | `137` | `1753088` | `0` | `0` | `1.20554` | `+0.0552` | `+0.0209` | `-0.0232` | keep, not mature |
| B156 | `139` | `344064` | `0` | `0` | `0.0116` | n/a | `+0.0966` | `+0.1103` | keep, young |
| B157 | `138` | `327680` | `0` | `0` | `0.4321` | n/a | `+0.0605` | `+0.0319` | keep, young |
| B158 | `140` | `344976` | `0` | `0` | `-0.3195` | n/a | `+0.1947` | `+0.1574` | keep, young |

B154 was canceled with `cancel_requested=1`.

One replacement screen was added and launched:

| Job | Spec | Run | W&B |
| ---: | --- | --- | --- |
| `141` | `b159-lowlr-microupdate-entropyburst-complete2500-l13-screen` | `b159_l13_lowlr_microupdate_entropyburst_complete2500_s80_20260628T152551Z` | [`575n7ges`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/575n7ges) |

B159 follows the fast-update failure line but pushes update cadence one notch
further: `n_steps=256`, `batch_size=256`, LR `7e-5 -> 5e-5`,
`ent_coef=0.10 -> 0.02`, `gamma=0.995`, `gae_lambda=0.98`,
`normalize_advantage=true`, `clip_range=0.12`, `target_kl=0.12`, zero explicit
death penalty, `progress_reward_cap=5`, `progress_reward_scale=0.0005`, a
small survival bonus (`time_penalty=-0.001`), and `completion_reward=2500`.
The hypothesis is that 512-step fast updates may still be too slow to reinforce
rare safe jump timing, while raw x-progress reward was too distracting.

B159 keeps the direct Level1-3 start state, simple action set, frame skip 4,
info events, `done_on_events=life_loss,level_change`, 5M cap, checkpoint
cadence, and strict `train/info/level_complete/rate/min/last > 0.99` stop.
It remains a legal reward/hyperparameter-only screen.

As before, `rlab-train-image.json` was absent at repo root, so B159 was
enqueued without `--runtime-image-ref-file` and used the queue CLI's latest
successful immutable image resolver.

After B154 exited, B159 was claimed:

```text
train_jobs: {"canceled": 38, "failed": 5, "running": 5}
eval_jobs: {}
queue demand: profile=any target=rtx4090 pending=0 running=5 digest=063e55231d69
```

Active jobs are now B155-B159 on beast-3. Next monitor priority is B155 as it
crosses the 2M maturity point. Any nonzero
`train/info/level_complete/from/0-2/count` should switch the loop from pruning
to candidate checkpoint/eval planning.

## B155 Plateau Backfill With Level1-Style No-AdvNorm PPO

A refresh at `2026-06-28T15:39Z` showed that B155 had crossed the mature
screen threshold with no Level1-3 clean clears. Its `rollout/ep_rew_mean` had
flattened without producing any target movement, so it matched the plateau
abort rule: zero clean-clear count/rate, mature step count, and a saturated
reward tail.

| Arm | Job | History step | Peak L1-3 count | Peak L1-3 rate | Last reward | Reward delta over ~500k | Reward delta over ~250k | Reward delta over ~100k | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| B155 | `137` | `2621440` | `0` | `0` | `1.34945` | `+0.10656` | `+0.02865` | `+0.09200` | cancel |
| B156 | `139` | `1220304` | `0` | `0` | `0.929392` | `+0.69634` | `+0.18906` | `-0.01293` | keep, young |
| B157 | `138` | `1163264` | `0` | `0` | `0.798579` | `+0.13258` | `+0.07912` | `+0.01627` | keep, young |
| B158 | `140` | `1319360` | `0` | `0` | `4.88` | `+0.77161` | `+0.43956` | `+0.54310` | keep, still moving |
| B159 | `141` | `598016` | `0` | `0` | `0.410405` | `+0.16863` | `+0.10757` | `+0.00492` | keep, young |

B155 was canceled with `cancel_requested=1`.

One replacement screen was added and launched:

| Job | Spec | Run | W&B |
| ---: | --- | --- | --- |
| `142` | `b160-lowlr-noadvnorm-longcredit-complete1500-l13-screen` | `b160_l13_lowlr_noadvnorm_longcredit_complete1500_s80_20260628T153835Z` | [`l70341ej`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/l70341ej) |

B160 follows the user's Level1-3 PPO failure-mode notes and the checked-in
Level1 recipe. The controlled test is the missing no-advantage-normalization
quadrant: B55's successful Level1 PPO geometry used `normalize_advantage=false`,
while most recent low-LR/high-gamma Level1-3 probes turned it on. B160 keeps
direct Level1-3 training, simple actions, frame skip 4, info events,
`done_on_events=life_loss,level_change`, 5M cap, checkpoint cadence, and strict
`train/info/level_complete/rate/min/last > 0.99` stop. The legal
reward/hyperparameter deltas are LR `7e-5 -> 4e-5`, `n_steps=2048`,
`batch_size=1024`, `gamma=0.995`, `gae_lambda=1.0`, `ent_coef=0.08 -> 0.01`,
`normalize_advantage=false`, `clip_range=0.15`, `target_kl=0.16`,
`progress_reward_cap=10`, `progress_reward_scale=0.01`, `death_penalty=2`,
`completion_reward=1500`, and `time_penalty=-0.001`.

As before, `rlab-train-image.json` was absent at repo root, so B160 was
enqueued without `--runtime-image-ref-file` and used the queue CLI's latest
successful immutable image resolver.

After B155 exited, B160 was claimed:

```text
train_jobs: {"canceled": 39, "failed": 5, "running": 5}
eval_jobs: {}
active jobs: B156, B157, B158, B159, B160
queue demand: profile=any target=rtx4090 pending=0 running=5 digest=063e55231d69
```

Next monitor priorities are B156 and B157 as they cross the 2M maturity point,
with B158 protected while its reward curve is still moving. Any nonzero
`train/info/level_complete/from/0-2/count` should switch the loop from pruning
to candidate checkpoint/eval planning.

## B158 Survival-Attractor Backfill

A refresh at `2026-06-28T15:47Z` still showed no clean Level1-3 clears in the
active batch. B158 had effectively reached the 2M maturity point with zero
target movement. Its reward tail had rolled over, and its reward share was
dominated by the time/survival component (`train/reward_share/time ~= 0.8423`)
with very long episodes (`rollout/ep_len_mean ~= 1954`). That made it a
survival-attractor failure rather than a useful clear-discovery run.

| Arm | Job | History step | Peak L1-3 count | Peak L1-3 rate | Last reward | Reward delta over ~500k | Reward delta over ~250k | Reward delta over ~100k | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| B156 | `139` | `1851392` | `0` | `0` | `1.1257` | `+0.10469` | `+0.08069` | `+0.05580` | keep, just below maturity |
| B157 | `138` | `1744896` | `0` | `0` | `0.813549` | `+0.02981` | `+0.02976` | `-0.00970` | keep, just below maturity |
| B158 | `140` | `1999696` | `0` | `0` | `4.94425` | `+0.23673` | `-0.12959` | `-0.71640` | cancel |
| B159 | `141` | `1122304` | `0` | `0` | `0.476114` | `+0.04337` | `+0.02064` | `+0.02808` | keep, young |
| B160 | `142` | `403568` | `0` | `0` | `0.863498` | n/a | `+0.21792` | `+0.46563` | keep, young |

B158 was canceled with `cancel_requested=1`.

One replacement screen was added and launched:

| Job | Spec | Run | W&B |
| ---: | --- | --- | --- |
| `143` | `b161-lowlr-wideenv-antistall-complete3000-l13-screen` | `b161_l13_lowlr_wideenv_antistall_complete3000_s80_20260628T154702Z` | [`fbvypoov`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/fbvypoov) |

B161 keeps the direct Level1-3 start state, simple action set, frame skip 4,
info events, `done_on_events=life_loss,level_change`, 5M cap, checkpoint
cadence, and strict `train/info/level_complete/rate/min/last > 0.99` stop.
The legal changes target B158's failure mode: broader parallel exploration
(`n_envs=32`), low LR `7e-5 -> 4e-5`, high early entropy `0.12 -> 0.02`,
`n_steps=1024`, `batch_size=2048`, `gamma=0.995`, `gae_lambda=0.98`,
`normalize_advantage=true`, `clip_range=0.12`, `target_kl=0.12`,
`progress_reward_cap=5`, `progress_reward_scale=0.005`, `death_penalty=1`,
`completion_reward=3000`, and a light anti-stall penalty
(`time_penalty=0.003`) instead of the B158 survival bonus.

As before, `rlab-train-image.json` was absent at repo root, so B161 was
enqueued without `--runtime-image-ref-file` and used the queue CLI's latest
successful immutable image resolver.

After B158 exited, B161 was claimed:

```text
train_jobs: {"canceled": 40, "failed": 5, "running": 5}
eval_jobs: {}
active jobs: B156, B157, B159, B160, B161
queue demand: profile=any target=rtx4090 pending=0 running=5 digest=063e55231d69
```

Next monitor priorities are B156 and B157 as they cross the 2M maturity point.
B159 should be protected until it crosses the 2M point unless its target metric
moves earlier; B160 and B161 are too young to judge. Any nonzero
`train/info/level_complete/from/0-2/count` should switch the loop from pruning
to candidate checkpoint/eval planning.

## B157 Fast-Update Entropy Backfill

A refresh at `2026-06-28T15:55Z` showed no clean Level1-3 clears in the active
batch. B157 had crossed the 2M maturity point with zero target movement and an
essentially flat reward tail. B156 was also above 2M with zero clears, but its
250k reward tail was still positive enough to keep watching for one more cycle;
B159, B160, and B161 were still young.

| Arm | Job | History step | Peak L1-3 count | Peak L1-3 rate | Last reward | Reward delta over ~500k | Reward delta over ~250k | Reward delta over ~100k | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| B156 | `139` | `2441216` | `0` | `0` | `1.28202` | `+0.10057` | `+0.14994` | `+0.00537` | keep, still some 250k movement |
| B157 | `138` | `2310144` | `0` | `0` | `0.830459` | `+0.03191` | `+0.00411` | `-0.00120` | cancel |
| B159 | `141` | `1609728` | `0` | `0` | `0.460039` | `+0.00958` | `+0.02950` | `-0.01625` | keep, young |
| B160 | `142` | `1015808` | `0` | `0` | `1.00794` | `+0.25377` | `+0.22137` | `+0.08462` | keep, young |
| B161 | `143` | `458752` | `0` | `0` | `-0.342791` | n/a | `+0.07955` | `+0.03913` | keep, young |

B157 was canceled with `cancel_requested=1`.

One replacement screen was added and launched:

| Job | Spec | Run | W&B |
| ---: | --- | --- | --- |
| `144` | `b162-lowlr-fastupdate-antistall-complete3000-l13-screen` | `b162_l13_lowlr_fastupdate_antistall_complete3000_s80_20260628T155418Z` | [`ddzmvogb`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/ddzmvogb) |

B162 keeps the direct Level1-3 start state, simple action set, frame skip 4,
info events, `done_on_events=life_loss,level_change`, 5M cap, checkpoint
cadence, and strict `train/info/level_complete/rate/min/last > 0.99` stop.
The legal change is a fast-update anti-stall variant of B157: `n_steps=512`,
`batch_size=512`, LR `7e-5 -> 4e-5`, `gamma=0.995`, `gae_lambda=0.98`,
`ent_coef=0.12 -> 0.02`, `normalize_advantage=true`, `clip_range=0.12`,
`target_kl=0.12`, `progress_reward_cap=5`, `progress_reward_scale=0.005`,
`death_penalty=1`, `completion_reward=3000`, and `time_penalty=0.002`.
This tests whether B157's high-entropy fast-update line needed anti-stall
pressure instead of a small survival bonus and zero explicit death pressure.

As before, `rlab-train-image.json` was absent at repo root, so B162 was
enqueued without `--runtime-image-ref-file` and used the queue CLI's latest
successful immutable image resolver.

After B157 exited, B162 was claimed:

```text
train_jobs: {"canceled": 41, "failed": 5, "running": 5}
eval_jobs: {}
active jobs: B156, B159, B160, B161, B162
queue demand: profile=any target=rtx4090 pending=0 running=5 digest=063e55231d69
```

Next monitor priority is B156. If its target metric is still zero and the
reward tail remains flat, recycle it; otherwise protect B159/B160/B161/B162
until they cross their maturity points. Any nonzero
`train/info/level_complete/from/0-2/count` should switch the loop from pruning
to candidate checkpoint/eval planning.

## B159 Micro-Update Entropy Backfill

A refresh at `2026-06-28T16:03Z` still showed no clean Level1-3 clears in the
active batch. B159 had crossed the 2M maturity point with zero target movement
and only tiny reward-tail movement. B156 was older and also still at zero
clears, but its shaped reward was still rising strongly, so it was protected
for another cycle under the plateau rule.

| Arm | Job | History step | Peak L1-3 count | Peak L1-3 rate | Last reward | Reward delta over ~500k | Reward delta over ~250k | Reward delta over ~100k | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| B156 | `139` | `3080272` | `0` | `0` | `2.95262` | `+1.47109` | `+0.83500` | `+0.15418` | keep, still moving |
| B159 | `141` | `2127008` | `0` | `0` | `0.509384` | `+0.05626` | `+0.03362` | `+0.03296` | cancel |
| B160 | `142` | `1671168` | `0` | `0` | `3.67558` | `+2.12131` | `+0.93424` | `+0.41879` | keep, young/still moving |
| B161 | `143` | `1277952` | `0` | `0` | `0.094449` | `+0.37470` | `+0.30154` | `+0.13739` | keep, young |
| B162 | `144` | `409600` | `0` | `0` | `-0.223139` | n/a | `+0.00176` | `-0.04857` | keep, young |

B159 was canceled with `cancel_requested=1`.

One replacement screen was added and launched:

| Job | Spec | Run | W&B |
| ---: | --- | --- | --- |
| `145` | `b163-lowlr-microupdate-antistall-complete3000-l13-screen` | `b163_l13_lowlr_microupdate_antistall_complete3000_s80_20260628T160233Z` | [`hf7j3eac`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/hf7j3eac) |

B163 keeps the direct Level1-3 start state, simple action set, frame skip 4,
info events, `done_on_events=life_loss,level_change`, 5M cap, checkpoint
cadence, and strict `train/info/level_complete/rate/min/last > 0.99` stop.
The legal change is a micro-update anti-stall variant of B159: `n_steps=256`,
`batch_size=512`, LR `7e-5 -> 4e-5`, `gamma=0.995`, `gae_lambda=0.98`,
`ent_coef=0.12 -> 0.02`, `normalize_advantage=true`, `clip_range=0.12`,
`target_kl=0.12`, `progress_reward_cap=5`, `progress_reward_scale=0.005`,
`death_penalty=1`, `completion_reward=3000`, and `time_penalty=0.002`.
This tests whether B159's very frequent high-entropy update line needed
anti-stall pressure instead of a small survival bonus and zero explicit death
pressure.

As before, `rlab-train-image.json` was absent at repo root, so B163 was
enqueued without `--runtime-image-ref-file` and used the queue CLI's latest
successful immutable image resolver.

After B159 exited, B163 was claimed:

```text
train_jobs: {"canceled": 42, "failed": 5, "running": 5}
eval_jobs: {}
active jobs: B156, B160, B161, B162, B163
queue demand: profile=any target=rtx4090 pending=0 running=5 digest=063e55231d69
```

Next monitor priority remains B156. If its target metric is still zero and its
reward tail flattens, recycle it; otherwise protect B160/B161/B162/B163 until
they cross their maturity points. Any nonzero
`train/info/level_complete/from/0-2/count` should switch the loop from pruning
to candidate checkpoint/eval planning.

## B161 No-Advantage-Normalization Anti-Stall Backfill

A refresh at `2026-06-28T16:13Z` still showed no clean Level1-3 clears in the
active batch. B161 had crossed the 2M-step maturity point with zero target
movement and a nearly flat `rollout/ep_rew_mean` tail. B156 and B160 also
remained at zero clears, but both still had material shaped-reward movement, so
they were protected under the plateau rule. B162 and B163 were still young.

| Arm | Job | History step | Peak L1-3 count | Peak L1-3 rate | Last reward | Reward delta over ~500k | Reward delta over ~250k | Reward delta over ~100k | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| B156 | `139` | `3817472` | `0` | `0` | `4.62210` | `+1.14773` | `+0.55960` | `+0.16979` | keep, still moving |
| B160 | `142` | `2392064` | `0` | `0` | `4.17055` | `+0.36728` | `+0.24320` | `+0.11377` | keep, still moving |
| B161 | `143` | `2195456` | `0` | `0` | `0.659238` | `+0.21467` | `+0.03352` | `+0.01218` | cancel |
| B162 | `144` | `1064960` | `0` | `0` | `0.510120` | `+0.58428` | `+0.14660` | `+0.02690` | keep, young |
| B163 | `145` | `454656` | `0` | `0` | `0.022771` | n/a | `+0.26481` | `+0.17946` | keep, young |

B161 was canceled with `cancel_requested=1`.

One replacement screen was added and launched:

| Job | Spec | Run | W&B |
| ---: | --- | --- | --- |
| `146` | `b164-lowlr-noadvnorm-antistall-complete3000-l13-screen` | `b164_l13_lowlr_noadvnorm_antistall_complete3000_s80_20260628T161137Z` | [`h6w19s0k`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/h6w19s0k) |

B164 keeps the direct Level1-3 start state, simple action set, frame skip 4,
info events, `done_on_events=life_loss,level_change`, 5M cap, checkpoint
cadence, and strict `train/info/level_complete/rate/min/last > 0.99` stop.
The legal change combines the Level1-style unnormalized PPO advantage geometry
from B160 with the anti-stall reward shape from B161/B162/B163:
`n_steps=2048`, `batch_size=1024`, LR `7e-5 -> 4e-5`, `gamma=0.995`,
`gae_lambda=1.0`, `ent_coef=0.12 -> 0.02`, `normalize_advantage=false`,
`clip_range=0.15`, `target_kl=0.16`, `progress_reward_cap=5`,
`progress_reward_scale=0.005`, `death_penalty=1`, `completion_reward=3000`,
and `time_penalty=0.002`.

This directly tests the user's Level1-3 hypothesis bundle: lower LR, direct
Level1-3 training, high early entropy, longer credit assignment, and a stronger
true-clear signal. The current trainer has legal progress/death/completion/time
reward controls, but no existing jump-height or landing-specific config knob, so
B164 stays inside the already-supported reward/hyperparameter surface instead of
changing observation, action, termination, or eval semantics.

As before, `rlab-train-image.json` was absent at repo root, so B164 was
enqueued without `--runtime-image-ref-file` and used the queue CLI's latest
successful immutable image resolver.

After B161 exited, B164 was claimed:

```text
train_jobs: {"canceled": 43, "failed": 5, "running": 5}
eval_jobs: {}
active jobs: B156, B160, B162, B163, B164
queue demand: profile=any target=rtx4090 pending=0 running=5 digest=063e55231d69
```

Next monitor priorities are B156 and B160 as mature zero-clear runs. Recycle
only if their `rollout/ep_rew_mean` tails flatten while the Level1-3 completion
count remains zero; otherwise let B162/B163/B164 reach their maturity points.
Any nonzero `train/info/level_complete/from/0-2/count` should switch the loop
from pruning to candidate checkpoint/eval planning.

## B156 Bottleneck Investigation And Checkpoint Forks

The B156 run was investigated after manual playback confirmed it crossed the
initial Level1-3 bottleneck. The W&B history and deterministic artifact evals
agree: B156 was not just producing shaped-reward noise. It was the first active
screen run in this batch to produce real clean Level1-3 completions and to move
past the failed-run wall near x=670.

Key B156 training facts:

- Run: `b156_l13_lowlr_criticwide_tinyprogress_complete1500_s80_20260628T151901Z`
  ([`zjo4h4vl`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/zjo4h4vl)).
- Final step: `5013504`.
- Final clean Level1-3 count: `23`.
- First clean-count increment: `3823072`.
- Final `train/info/level_complete/rate/min/last`: `0`; peak observed rolling
  rate was only `0.01`, so this is not solved, but it is a real discovery run.

Final deterministic artifact checks showed the x-position gap clearly:

| Run | Deterministic eval | Max x mean | Max x max | Death bin | Completion |
| --- | --- | ---: | ---: | --- | ---: |
| B156 final | 20 eps | `1599` | `1599` | `1500-1599` | `0/20` |
| B155 final | 20 eps | `673` | `673` | `600-699` | `0/20` |
| B160 final | 20 eps | `667` | `667` | `600-699` | `0/20` |

B156 checkpoint-series evals show where the breakthrough happened:

| Checkpoint | Max x |
| ---: | ---: |
| 500k | `0` |
| 1.0M | `649` |
| 1.5M | `639` |
| 2.0M | `674` |
| 2.5M | `670` |
| 3.0M | `1101` |
| 3.5M | `1368` |
| 4.0M | `1599` |
| 4.5M | `1645` |
| 5.0M | `1592` |

The main config diff against the closest failed control, B155, is the
asymmetric network head: B156 adds `policy_net_arch=64,64` and
`value_net_arch=512,512`. B155 uses the same low-LR, tiny-progress,
completion-1500 family without that critic-wide/value-wide architecture and got
zero clears. The leading hypothesis is therefore value estimation and delayed
platform-jump credit, not just entropy, progress scale, or clear bonus size.

Follow-up job decisions:

| Decision | Jobs | Rationale |
| --- | --- | --- |
| Canceled | B160/job `142` | Zero clears, final deterministic x still near the failed wall. |
| Canceled | B162/job `144`, B163/job `145` | Mature zero-clear arms with flat or weak reward tails. |
| Canceled | B164/job `146` | At about 4.3M, zero clears and low reward (`~0.75`), weaker than B156-family arms. |
| Canceled | B165/job `147` | Critic-wide anti-stall arm, but at about 3.44M had zero clears and low reward (`~0.25`). |
| Kept | B166/job `148` | Critic-wide survival arm; by about 4.24M it had `19` clean clears and rolling min rate `0.02`. |

B167/B168 attempted to resume the B156 4.5M checkpoint through
`resume_artifact`, but both failed before W&B startup. B169/B170 then attempted
direct `resume` using a host path under `/home/tsilva/rlab/artifacts`, reached
W&B, and failed with `FileNotFoundError` because only `/home/tsilva/rlab/runs`
and `/home/tsilva/rlab/logs` are mounted into the worker container. The
checkpoint was moved to the mounted path:

```text
host:      /home/tsilva/rlab/runs/train_resumes/b156_step4500000/ppo_supermariobros-nes-v0_4500000_steps.zip
container: /root/rlab/runs/train_resumes/b156_step4500000/ppo_supermariobros-nes-v0_4500000_steps.zip
```

The corrected mounted checkpoint forks are now running:

| Job | Spec | Run | W&B | Live signal at launch check |
| ---: | --- | --- | --- | --- |
| `153` | `b171-b156resume-mounted-stabilize-4500k-l13-screen` | `b171_l13_b156resume_mounted_stabilize4500k_s80_20260628T170311Z` | [`6x9lwrd9`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/6x9lwrd9) | `330960` continuation steps, `15` clears, rolling min rate `0.02` |
| `154` | `b172-b156resume-mounted-clearboost-4500k-l13-screen` | `b172_l13_b156resume_mounted_clearboost4500k_s80_20260628T170324Z` | [`gvfbpsys`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/gvfbpsys) | `311296` continuation steps, `13` clears, rolling min rate `0.03` |

The current live batch is now B166, B171, and B172. This is the first batch
where every live arm is in the B156 lineage and has already produced clean
Level1-3 completions. Next decision point: monitor
`train/info/level_complete/rate/min/last` first, then clear count density and
x-position evals from the best checkpoint if any arm's rolling completion rate
rises materially above `0.01`.

## B171/B172 Finished And B173-B176 Stability Backfill

A refresh after the first B156 checkpoint-fork batch completed showed that the
B156 lineage is now reliably discovering Level1-3 clears, but none of the
screen runs is close to the 100/100 window yet. The important failure mode is
not discovery anymore; it is consolidation. B171/B172 both produced immediate
clean clears from the B156 4.5M checkpoint, peaked near a `0.05` rolling clear
rate, and then decayed by the end of their 500k continuation. B166, the
from-scratch B156-family control, also found clears and reached a higher total
clear count, but its rate decayed near the 5M cap.

| Run | Status | Best min rate | Best L1-3 rate | Best clean count | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| B166/job `148` | succeeded | `0.04` @ `4654768` | `0.04` @ `4654768` | `42` @ `4867648` | Strongest from-scratch B156-family result, but still sparse and unstable. |
| B171/job `153` | succeeded | `0.05` @ `239152` continuation steps | `0.05` @ `239152` | `27` @ `486912` | 4.5M checkpoint fork, original completion reward; clears appeared immediately then faded. |
| B172/job `154` | succeeded | `0.05` @ `200384` continuation steps | `0.05` @ `200384` | `25` @ `493136` | 4.5M checkpoint fork, completion reward 3000; higher shaped reward but no rate improvement over B171. |

Because none of these reached the goal's strict
`train/info/level_complete/rate/min/last > 0.99` stop metric, no confirmation
or promotion was started. The next legal hypothesis is late-policy stability:
reduce PPO update pressure after the bottleneck behavior appears, and test
whether the sparse clean-clear event is still underweighted once updates are
gentler.

Four B156-lineage backfills were added:

| Job | Spec | Run | W&B | Rationale |
| ---: | --- | --- | --- | --- |
| `155` | `b173-b156resume-4000k-ultrastable-complete1500-l13-screen` | `b173_l13_b156resume_4000k_ultrastable_complete1500_s80_20260628T171519Z` | [`70moqy1u`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/70moqy1u) | Resume B156 at 4.0M so there is a full 1M continuation budget; keep completion reward 1500; use ultra-low LR, low entropy, fewer epochs, low KL, and tight clipping. |
| `156` | `b174-b156resume-4000k-ultrastable-complete3000-l13-screen` | `b174_l13_b156resume_4000k_ultrastable_complete3000_s80_20260628T171533Z` | [`tt2fisli`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/tt2fisli) | Same 4.0M ultra-stable continuation, but completion reward 3000 to compare sparse clear weighting under low update pressure. |
| `157` | `b175-lowlr-criticwide-survival-steady-complete3000-l13-screen` | `b175_l13_lowlr_criticwide_survival_steady_complete3000_s80_20260628T171548Z` | [`lojzn6x9`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/lojzn6x9) | From-scratch B166 stability variant: lower LR, lower entropy floor, fewer epochs, larger batches, tighter clip, lower target KL. |
| `158` | `b176-lowlr-criticwide-survival-steady-complete5000-l13-screen` | `b176_l13_lowlr_criticwide_survival_steady_complete5000_s80_20260628T171603Z` | pending | Same as B175 but completion reward 5000 to test whether clean clears remain underweighted. |

Initial telemetry confirmed the claimed jobs are live and comparable:

| Run | Step | L1-3 clean count | L1-3 rate | Min rate | Reward | KL | Clip frac |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B173 | `199952` | `4` | `0.01` | `0.01` | `20.175571` | `0.00065495796` | `0.053771973` |
| B174 | `196608` | `1` | `0` | `0` | `5.027743` | `0.0008231136` | `0.07159424` |
| B175 | `137616` | `0` | `0` | `0` | `-0.04849163` | `0.00043479062` | `0` |

B176 is pending because the active beast-3 runner is currently a three-worker
container and `rlab-fleet plan` will not restart it while leases are active.
The repo config still permits five beast-3 workers, but preserving the live
jobs is preferable to interrupting them; B176 should start when the first
short continuation arm frees a lane.

Next monitor priority is B173 versus B174. If either 4.0M continuation exceeds
the B171/B172 `0.05` peak or maintains a higher final rolling rate, use that
as the parent for the next stabilization recipe. If neither improves, keep
B175/B176 as the from-scratch stability test and consider whether 4.0M/4.5M
checkpoint forking is useful only for diagnostics, not for a reproducible
screen recipe.

## B174 Pruned, B176 Started

A follow-up refresh while B173/B174/B175 were running showed that the
ultra-stable 4.0M checkpoint continuations were preserving some clears, but not
improving over the B171/B172 4.5M checkpoint fork peak. Both B173 and B174 were
below the earlier `0.05` rolling-rate high-water mark. B174 was the weaker of
the paired 4.0M continuation arms, so it was canceled to let the queued
from-scratch completion-5000 stability arm start.

| Run | Step at decision | Best min rate | Clean count | Last min rate | Last reward | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| B173/job `155` | `560368` history / `502608` summary | `0.02` @ `204448` | `10` | `0` | `5.452332` | keep to 1M cap as the remaining 4.0M continuation probe |
| B174/job `156` | `546928` history / `499968` summary | `0.02` @ `458496` | `6` | `0` | `5.3466725` | cancel; weaker duplicate of B173 with completion reward 3000 |
| B175/job `157` | `538240` history / `513536` summary | `0` | `0` | `0` | `-0.032561556` | keep; too young for a from-scratch B166-style arm |

B174 cancellation returned `cancel_requested=1` and the queue transitioned it
to canceled. B176 was then claimed:

| Job | Run | W&B | Initial signal |
| ---: | --- | --- | --- |
| `158` | `b176_l13_lowlr_criticwide_survival_steady_complete5000_s80_20260628T171603Z` | [`tjhj7vf7`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/tjhj7vf7) | `70032` steps, zero clears, reward `-0.09032169`, KL `0.0004145006` |

At the next W&B snapshot, B173 had recovered to `20` clean clears by `806144`
continuation steps, but the rolling min rate was still only `0.01`. That is
useful evidence that the 4.0M ultra-stable continuation can preserve some
post-bottleneck behavior, but it still has not beaten the B171/B172 rate peak
or approached the goal stop threshold. Keep B173 until its 1M cap finishes,
then compare final and peak rates against B171/B172 before deciding whether
checkpoint forking remains worth pursuing. Keep B175/B176 until at least the
2M maturity point unless they clearly plateau with zero clean clears.

## B173 Finished And B177 Critic-XL Backfill

B173 finished its 1M continuation from the B156 4.0M checkpoint. It preserved
post-bottleneck behavior but did not improve on the earlier B171/B172 result:

| Run | Status | Best min rate | Final min rate | Final clean count | Best reward | Interpretation |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| B173/job `155` | succeeded | `0.05` @ `712608` | `0.01` | `22` | `66.15167` | Matched B171/B172 peak but did not improve consolidation; ultra-low update pressure alone is not enough. |

The from-scratch steady B166 variants are still too young to judge, but their
early zero-clear state is recorded for comparison:

| Run | Step | Clean count | Min rate | Reward | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| B175/job `157` | `~1.32M` history | `0` | `0` | `0.23387904` | Keep until at least the 2M maturity point. |
| B176/job `158` | `~0.72M` history | `0` | `0` | `0.0169285` | Keep, young. |

The next backfill was added to continue the strongest hypothesis from B156/B166:
value estimation is helping delayed platform-jump credit, but the current value
head may still be too weak to consolidate a reliable policy. B177 keeps the
B166 reward/PPO shape almost intact and changes the value side only:
`value_net_arch=1024,1024` and `vf_coef=0.75`, with the same small
`policy_net_arch=64,64`.

| Job | Spec | Run | W&B | Initial signal |
| ---: | --- | --- | --- | --- |
| `159` | `b177-lowlr-criticxl-survival-complete3000-l13-screen` | `b177_l13_lowlr_criticxl_survival_complete3000_s80_20260628T172441Z` | [`z6e2n2bp`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/z6e2n2bp) | `~0.25M` steps, zero clears, reward near `0`, KL `0.00339` |

Current live batch after B173 finished: B175, B176, and B177. None is an eval
or confirmation candidate yet because none has approached the goal's strict
`train/info/level_complete/rate/min/last > 0.99` training stop metric. The
next pruning checkpoint is B175 at 2M steps: if it remains zero-clear with a
flat reward tail, recycle it; otherwise keep it long enough to compare against
B166's late first-clear timing.

## B175-B177 Maturity Check

A later live refresh kept all three active runs. None has produced a clean
Level1-3 clear yet, so none is an eval or confirmation candidate. The prune
decision is still nuanced because B166's first useful clear arrived late, and
all three current runs still have some positive medium-horizon reward movement.

| Run | Summary step | Clean count | Min rate | Last reward | Reward delta over ~500k | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| B175/job `157` | `2579472` | `0` | `0` | `0.98784333` | `+0.3180` | keep to ~3M unless the short tail fully flattens |
| B176/job `158` | `1957024` | `0` | `0` | `0.18074903` | `+0.1990` | keep, just reaching maturity |
| B177/job `159` | `1372400` | `0` | `0` | `1.1093334` | `+0.5340` | keep, young and moving fastest |

B175 has crossed the 2M fixed-budget checkpoint with zero clears, but it is not
as flat as the earlier recycled runs; the 500k reward tail is still positive
and the B166 control discovered clears only much later. The next concrete
decision point is B175 around 3M steps. If it remains zero-clear and its 100k
and 250k reward tails are flat or negative, recycle it. If it discovers clears,
switch immediately to candidate/checkpoint analysis for the B175/B176/B177
family.

## B175/B176 Pruned And B166 Checkpoint Forks Launched

The later maturity refresh changed the decision on B175/B176. Both from-scratch
steady variants remained zero-clear well past the point where B166 had already
crossed the initial Level1-3 bottleneck. B166 first produced clean clears near
`3.11M` steps and reached `44` total clean clears by the 5M cap; by contrast,
B175 was still at zero clears at `4.47M` and B176 was still at zero clears at
`3.79M`.

| Run | Summary step | Clean count | Min rate | Reward | KL | Clip frac | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| B175/job `157` | `4466592` | `0` | `0` | `1.6349138` | `0.00087259046` | `0.068715416` | cancel; stable-from-start recipe appears to suppress discovery |
| B176/job `158` | `3793376` | `0` | `0` | `1.1385639` | `0.0008539298` | `0.061503094` | cancel; completion reward 5000 did not recover discovery under the steady settings |
| B177/job `159` | `3135072` | `0` | `0` | `4.240745` | `0.012199904` | `0.2923645` | keep; younger and still the strongest reward/progress signal |
| B166/control | `5013504` | `44` | `0` final | `2.046308` | `0.01013343` | `0.285968` | parent for checkpoint forks |

The resulting hypothesis is that B166's original higher update pressure helped
discover the bottleneck behavior, but a gentler continuation may be needed once
that behavior is present. Two legal B166 checkpoint forks were launched:

| Job | Spec | Run | W&B | Fork point | Rationale |
| ---: | --- | --- | --- | ---: | --- |
| `160` | `b178-b166resume-3000k-stablediscovery-complete3000-l13-screen` | `b178_l13_b166resume_3000k_stablediscovery_complete3000_s80_20260628T175445Z` | [`yx5z8kmx`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/yx5z8kmx) | `3.0M` | Resume just before B166's first-clear burst, keep completion reward 3000, and use moderate PPO stabilization for the remaining 2M steps. |
| `161` | `b179-b166resume-3500k-consolidate-complete5000-l13-screen` | `b179_l13_b166resume_3500k_consolidate_complete5000_s80_20260628T175449Z` | [`l2gtnmtm`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/l2gtnmtm) | `3.5M` | Resume after B166 has crossed the bottleneck, raise completion reward to 5000, and test whether sparse clear magnitude improves consolidation. |

Both forks preserve the Level1-3 single-state goal contract, the simple action
set, terminal-on-life-loss/level-change semantics, and the 5M total cap. No
eval or confirmation job is warranted yet; the live selection metric remains
`train/info/level_complete/rate/min/last`, with clean count density as the
secondary diagnostic until a rolling source-attempt window materially improves.

Current live batch after the handoff: B177, B178, and B179.

## B177 Clear Signal And B180/B181 Pending B177 Forks

A live refresh after B178/B179 started showed that the current batch is now
clear-producing across all three active arms. None is close to the strict
`train/info/level_complete/rate/min/last > 0.99` stop rule, so no eval or
confirmation job is warranted yet, but this is no longer a zero-clear pruning
state.

| Run | Job | Step | Clean count | Last min rate | Best min rate | First clean-clear step | Reward note | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| B177 | `159` | `~4.16M` history / `3.69M` summary | `10` history / `5` summary | `0.01` | `0.02` | `3161776` | reward peaked near `65.0`, then fell to about `3.0` | keep; real critic-XL discovery signal |
| B178 | `160` | `277040` continuation summary | `1` | `0` | n/a | n/a | reward `3.819253` | keep; too early, already preserved at least one clear |
| B179 | `161` | `277696` continuation summary | `2` | `0` | n/a | n/a | reward `54.56229` | keep; too early, completion-5000 consolidation arm is producing clears |

B177 has now exceeded B156's original first-clear timing and is approaching the
B166 family, but its falling reward tail suggests the same post-bottleneck
consolidation problem rather than a pure discovery problem. The B177
`4,000,000` checkpoint exists on beast-3 under the mounted run path, so two
pending checkpoint forks were added:

| Job | Spec | Run | Fork point | Rationale |
| ---: | --- | --- | ---: | --- |
| `162` | `b180-b177resume-4000k-stabilize-complete3000-l13-screen` | `b180_l13_b177resume_4000k_stabilize_complete3000_s80_20260628T180420Z` | `4.0M` | Keep B177 critic-XL and completion reward 3000, but lower PPO update pressure for a 1M continuation. |
| `163` | `b181-b177resume-4000k-consolidate-complete5000-l13-screen` | `b181_l13_b177resume_4000k_consolidate_complete5000_s80_20260628T180424Z` | `4.0M` | Same stabilization settings, but raise completion reward to 5000 to test whether clear density is still underweighted after discovery. |

Both B180/B181 preserve the goal contract: same Level1-3 start state, simple
action set, `done_on_events=life_loss,level_change`, no in-loop eval, and total
training budget capped at the equivalent of 5M steps. They are currently
pending because beast-3 is running a three-worker container and
`rlab-fleet plan` reports that active leases prevent a restart:

```text
queue demand: profile=any target=rtx4090 pending=2 running=3 digest=063e55231d69
active jobs: B177, B178, B179
pending jobs: B180, B181
warning: config changed but active lease prevents restart
```

Next decision point: let B177/B178/B179 mature unless one reaches the goal stop
metric. When a slot opens, compare B180/B181 against B177's original
post-4.0M continuation on clean-count density and peak
`train/info/level_complete/rate/min/last`.

## B178 High-Water And B182-B187 B178 Forks

B178 finished as the strongest Level1-3 screen so far. It did not approach the
goal's strict 100/100 stop rule, but it materially improved the high-water mark
over the previous B156/B166/B177 family:

| Run | Status | Clean count | Final min rate | Best observed min rate | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| B177/job `159` | succeeded | `37` | `0.01` | `0.02` | Critic-XL from scratch; real discovery, but sparse. |
| B178/job `160` | succeeded | `120` | `0.01` | `0.06` | Current high-water screen; B166 3.0M resume with stable-discovery settings. |
| B179/job `161` | succeeded | `82` | `0.01` | `0.01` | B166 3.5M resume, completion reward 5000; higher clear count than B177 but weaker rolling rate than B178. |
| B180/job `162` | succeeded | `63` | `0.05` | `0.05` | B177 4.0M resume, completion reward 3000; useful but below B178's observed `0.06`. |
| B181/job `163` | succeeded | `84` | `0.03` | `0.03` | B177 4.0M resume, completion reward 5000; more clears than B180 but lower rolling rate. |

B178's failure mode is now more specific: it can produce many clears and reach a
nontrivial rolling window, but the window decays before the cap. The critic
diagnostics are also weak on the best B178/B179 continuations:

| Run | Value/loss signal | Interpretation |
| --- | --- | --- |
| B178 final | explained variance `0.0347`, value loss `10946.632` | many clears, but poor value fit and final rolling-rate decay |
| B179 final | explained variance `0.0188`, value loss `92018.31` | completion-5000 clear bonus did not solve consolidation |
| B180 final | explained variance `0.0302`, value loss `43844.176` | B177 checkpoint fork preserved some clears, still value-stressed |
| B181 final | explained variance `0.0228`, value loss `30610.059` | stronger clear bonus improved count but not reliability |

Three B178 1.7M high-window forks then tested whether late optimizer/reward
changes can preserve that observed `0.06` window:

| Run | Job | W&B | Clean count | Final min rate | Critic signal | Interpretation |
| --- | ---: | --- | ---: | ---: | --- | --- |
| B182 | `164` | [`cms25q2v`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/cms25q2v) | `29` | `0.03` | explained variance `0.0298`, value loss `21853.361` | reward_scale `50` did not fix critic fit or completion density. |
| B183 | `165` | [`h8p884u7`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/h8p884u7) | `28` | `0.03` | explained variance `0.6826`, value loss `1.0713` | conservative original-scale continuation fixed critic fit but did not improve rolling clears. |
| B184 | `166` | [`5h6t9hkn`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/5h6t9hkn) | `42` | `0.05` | explained variance `0.0169`, value loss `122025.164` | completion reward 5000 recovered the best short continuation rate, but value fit remained poor. |

The current hypothesis is therefore: late conservative optimization can repair
value diagnostics, and clearboost can improve short-window completion density,
but neither alone converts B178's discovery into a reliable 100/100 policy.
The next batch starts from the B178 1.6M checkpoint instead of 1.7M to give the
same ideas 400k steps of budget under the effective 5M total cap:

| Job | Spec | Run | W&B | Rationale |
| ---: | --- | --- | --- | --- |
| `168` | `b185-b178resume-1600k-clearboost-complete5000-l13-screen` | `b185_l13_b178resume_1600k_clearboost_complete5000_s80_20260628T182754Z` | [`pmp6p4es`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/pmp6p4es) | Give the B184 clearboost recipe 400k steps from an earlier B178 checkpoint. |
| `169` | `b186-b178resume-1600k-clearboost-complete4000-l13-screen` | `b186_l13_b178resume_1600k_clearboost_complete4000_s80_20260628T182754Z` | [`t1mrzs9j`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/t1mrzs9j) | Test intermediate clear bonus 4000 as a possible value-stress/clear-density compromise. |
| `167` | `b187-b178resume-1600k-adaptive-clearboost5000-l13-screen` | `b187_l13_b178resume_1600k_adaptive_clearboost5000_s80_20260628T182753Z` | [`fz7eggj1`](https://wandb.ai/tsilva/SuperMarioBros-NES/runs/fz7eggj1) | Test whether B184's near-zero KL was too conservative by allowing moderate adaptation. |

All three preserve the Level1-3 goal contract, simple action set,
`done_on_events=life_loss,level_change`, no in-loop eval, and a total B166+B178
budget of about 5M steps. No eval jobs were launched because the best training
window remains `0.06`, far below the `>0.99` training stop metric.
