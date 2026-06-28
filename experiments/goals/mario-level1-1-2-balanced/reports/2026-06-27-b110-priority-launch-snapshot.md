# 2026-06-27 B110 Priority Launch Snapshot

Goal metric: peak `train/info/level_complete/rate/min/last`, requiring both
seeds in the same two-seed batch to exceed strict `0.80` within the 5M training
cap. Success is judged by each run's best observed W&B history value, not the
final checkpoint value and not an early-stop trigger.

## B91 Outcome

B91 did not solve the goal.

| Run | Queue state | W&B state | Peak step | Peak min-rate | Peak L1-1 | Peak L1-2 | Latest min-rate at scan |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `b91_l11l12_lowpress_complete25_s206_20260627T112013Z` | failed | finished | `4,089,456` | `0.88` | `0.89` | `0.88` | `0.00` |
| `b91_l11l12_lowpress_complete25_s207_20260627T112013Z` | failed | running at scan | `3,001,392` | `0.67` | `0.76` | `0.67` | `0.03` |

Interpretation:

- B91 is the strongest individual evidence so far: seed 206 crossed the strict
  threshold with both carried per-level rates above `0.80`.
- The same seed later collapsed on Level1-2, while seed 207 never crossed. The
  recipe is therefore a one-seed success, not a reproducible two-seed recipe.
- The useful signal is that `completion_reward=25` plus lower update pressure
  can reach the right region; the failure is preserving Level1-2 competence
  across seeds.

Noncompetitive current arms:

- B90 remained far below target: seed 204 peak `0.04`, seed 205 peak `0.08`.
- B89 seed 203 failed after peaking at `0.28`.
- B92 seed 198 failed before producing a W&B run URL.

## B110 Spec

Added:

- `experiments/goals/mario-level1-1-2-balanced/specs/b110-gentlerpress-complete25-l12soft-l11l12-two-seed.json`

Hypothesis:

- Combine B100's soft Level1-2 sampling with B109's gentler PPO pressure while
  keeping B91's true completion reward. This targets the exact B91 failure mode:
  seed 206 could cross, but Level1-2 later collapsed, and seed 207 stayed
  Level1-2 bottlenecked.

Config deltas from B91:

- `state_probs`: `[0.5, 0.5]` -> `[0.45, 0.55]`
- `target_kl`: `0.16` -> `0.12`
- `clip_range`: `0.15` -> `0.12`
- `learning_rate_final`: `0.0001` -> `0.000075`
- Kept `completion_reward=25`, `terminal_reward=50`, `death_penalty=25`,
  `reward_scale=10`, per-task advantage normalization, task conditioning, and
  the same native vector env path.

Validation:

- Loaded successfully through `rlab.job_queue.load_spec_document`.
- Runtime image is the same immutable digest used by the current beast-3 queue:
  `docker:ghcr.io/tsilva/rlab/rlab-train@sha256:c672be38cd0fb7b5505d4d7b902ac10316ec979538c784838531098b4c1bf0e5`.

## Queue Launch

B110 was enqueued at priority `90`, above the older broad ablations, so it was
claimed immediately after beast-3 was recovered.

| Job | Run | Seed | W&B |
| ---: | --- | ---: | --- |
| `77` | `b110_l11l12_gentlerpress_complete25_l12soft_s206_20260627T133104Z` | `206` | <https://wandb.ai/tsilva/SuperMarioBros-NES/runs/u9qhp8y3> |
| `78` | `b110_l11l12_gentlerpress_complete25_l12soft_s207_20260627T133104Z` | `207` | <https://wandb.ai/tsilva/SuperMarioBros-NES/runs/rmypl7nc> |

Current live queue after launch:

- Failed train jobs: `6`
- Succeeded train jobs: `10`
- Running train jobs: `5`
- Pending train jobs: `32`
- Eval jobs: none

Active beast-3 workers:

- `job=77`: `b110_l11l12_gentlerpress_complete25_l12soft_s206_20260627T133104Z`
- `job=78`: `b110_l11l12_gentlerpress_complete25_l12soft_s207_20260627T133104Z`
- `job=44`: `b92_l11l12_slowent_s199_20260627T112706Z`
- `job=41`: `b93_l11l12_slowent_l12bias_s200_20260627T112704Z`
- `job=42`: `b93_l11l12_slowent_l12bias_s201_20260627T112704Z`

## Fleet Recovery

The previous beast-3 runner stopped claiming work because the root filesystem
was full:

- Before cleanup: `/` was `221G` used, `0` available, `100%`.
- Docker had accumulated many old train images.
- Pruned unused Docker images older than one hour with
  `docker image prune -a -f --filter until=1h`.
- Reclaimed about `19.84GB` reported by Docker; post-cleanup `df` showed
  `43G` available and `80%` use.
- Removed the broken managed runner container, then recreated it with
  `rlab-fleet reconcile`.

The recreated beast-3 runner is on digest `c672be38cd0f` with five workers and
claimed B110 immediately.

## Next Monitor Condition

- Scan the two B110 W&B histories once they have meaningful training progress.
- Completion audit triggers only if both B110 seeds reach
  `train/info/level_complete/rate/min/last > 0.80`.
- If B110 fails, compare the failure shape against B100/B107/B108/B109 once
  those queued stabilizers run, especially whether the bottleneck is Level1-2
  retention, too-gentle learning, or seed-specific Level1-1 degradation.
