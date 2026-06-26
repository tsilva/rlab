# Goal: Sample-Efficient Level 1 Completion

Find the no-curriculum SuperMarioBros-NES PPO config that reaches reliable
`Level1-1` completion with the fewest aggregate policy timesteps.

## Target

Primary stop/score criterion:

```text
last 100 completed terminal training episodes are 100/100 completions
--stop-completion-episode-window 100
--stop-completion-rate-threshold 1.0
```

Use completed terminal episodes as the denominator. Do not use rollout-count
completion stops; they change meaning with `n_envs`, `n_steps`, and rollout
cadence.

Scope:

- Game/state: `SuperMarioBros-Nes-v0`, `Level1-1`
- No curriculum
- Native vector env training path
- W&B project: `SuperMarioBros-NES`
- Every run logs to W&B and has a specific `--run-description`

## Decision Metric

Primary metric: `time_to_100_of_100`, the first aggregate policy timestep where
the last 100 completed terminal training episodes are all completions.

If a run does not reach `100/100` before the active cap, record it as censored
at the cap with final completion rate, total completions, and total terminal
episodes.

Tie breakers:

1. Higher success count across confirmation seeds.
2. Lower median `time_to_100_of_100`.
3. Lower worst successful seed time.
4. Higher completion rate at fixed budgets: `2M`, `3M`, `4M`, and cap.
5. Higher external checkpoint eval completion rate, if available.

## Seeds And Cap

Use fixed paired seeds:

| Stage | Seeds | Purpose |
| --- | --- | --- |
| Screen | `23` | Cheap rejection of obvious losers |
| Confirm | `23,24,25` | Reproducibility check and baseline decision |

If a candidate is confirmed on `23,24,25`, compare only against an incumbent
evaluated on the same seeds. Same seed is not bitwise reproducibility under
default CUDA; treat it as a controlled randomization label.

Active cap:

```text
5,000,000 timesteps
```

Cap ratchet:

- If a confirmed baseline succeeds on all `23,24,25` seeds below the active cap,
  lower future caps to the slowest successful seed rounded up to the next `100k`.
- Do not increase the cap unless the goal changes or an infrastructure bug
  invalidates the baseline.

## Search Protocol

Screen with seed `23`. Reject obvious negatives. Advance candidates that reach
`100/100`, materially improve fixed-budget completion metrics, or look promising
enough to justify confirmation.

Confirm with seeds `23,24,25`. Promote only if success count is at least as good
as the incumbent, median `time_to_100_of_100` is better, there is no severe
reliability regression, and all runs either stop by `100/100` or reach the cap.

A baseline is the best confirmed config, not the best single run. Single lucky
runs generate hypotheses; they do not replace the baseline.

## Historical Context

The old `80/100` completed-episode target is retired. It was useful for finding
promising shapes, but it is not the current success criterion.

Most important retained facts from the retired experiment history:

- The best old single run was
  `b9_lr2e4_lrd1e4_2m_5m_stop80ep100_seed23_20260614_190906`, which reached the
  old `80/100` threshold at `2,558,256` timesteps with
  `learning_rate=2e-4 -> 1e-4 over 2M` and
  `ent_coef=0.01 -> 0.0003 over 2M`.
- That result was not a confirmed baseline. Same-seed/same-config repeats on
  later runs showed large PPO/CUDA/systems variance and did not reliably
  reproduce the old `80/100` stop.
- The upstream PPO reproduction proved that the simple-action, `gamma=0.9`,
  `gae_lambda=1.0`, fixed-entropy recipe can learn Level1-1, but checkpoint
  quality is non-monotonic. Do not evaluate a training run by final checkpoint
  alone.
- Use out-of-process stochastic checkpoint evaluation for promotion evidence.
  Rank checkpoints by completion rate first, then maximum x-position, then mean
  reward.
- Historical `BASELINE*` names are retired. In this repo, "baseline" now means a
  config confirmed under the current `100/100` protocol.
- Stable-retro runtime cautions and the current default runtime pin live in
  `AGENTS.md`; hardware and SkyPilot facts live in `INSTANCES.md`.

## RTX4090 Execution

Run SkyPilot jobs on:

```text
--infra k8s/beast-3
```

Benchmark-backed scheduling decision for the current `n_envs=16`, `n_steps=512`
shape:

```text
concurrent child trainings: 5
env_threads per child: 4
expected aggregate wall fps: about 6.2k
```

Why: measured RTX4090/post10 sweeps showed `5` children with `env_threads=4`
was the aggregate-throughput winner. Lowering `env_threads` to `1` or `2`
reduced CPU pressure but underfed rollout collection; `env_threads=2,count=5`
was about `6%` slower than `env_threads=4,count=5`.

Use `3-4` children only when individual run latency matters or a long
confirmation batch should leave more CPU headroom. Refresh the safe limit when
changing `n_envs`, `n_steps`, model size, deterministic CUDA flags, runtime
package version, or target node CPU shape.

Use normal SkyPilot task files that start all child runs from the beginning.
Avoid ad hoc second trainers via `sky exec`; one such run previously collapsed
to about `140` fps and is not valid for comparison. Backfill finished slots only
while the next run still has decision value.

## Determinism

Default search should measure the real training distribution. Do not enable
deterministic CUDA flags by default because they can slow training and alter the
optimizer/kernel path.

Use deterministic CUDA only for debugging reproducibility:

```text
CUBLAS_WORKSPACE_CONFIG=:4096:8
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True, warn_only=True)
```

If used, mark deterministic CUDA as a separate runtime condition.

## Current Winning Recipe

The current preferred training recipe for SuperMarioBros-NES `Level1-1`
screening is tracked in:

```text
experiments/recipes/supermariobros_nes_level1_winning_recipe.md
```

Short form:

```text
stable-retro-turbo: 1.0.0.post14
base: B33 target_kl=0.20 recipe
learning_rate: 1.5e-4 fixed
ent_coef: 0.01 -> 0.0003 over 2M
clip_range: 0.15
target_kl: 0.20
max_pool_frames: false
stop: 100/100 completed terminal training episodes
```

Decision: use no-maxpool for the active winning screening recipe. B40 found a
strict `100/100` seed and trained faster than B39. B39 still had the stronger
five-seed average, so no-maxpool is promoted as the active recipe component,
not yet as a statistically confirmed population-level baseline.

## Starting Config

Start from this near-best shape unless explicitly ablating one axis:

```text
stable-retro-turbo: 1.0.0.post14
state: Level1-1
n_envs: 16
env_threads: 4
n_steps: 512
batch_size: 512
n_epochs: 10
learning_rate: 1.5e-4 fixed
ent_coef: 0.01 -> 0.0003 over 2M
gamma: 0.9
gae_lambda: 1.0
clip_range: 0.15
target_kl: 0.20
vf_coef: 1.0
normalize_advantage: false
adam_eps: 1e-8
reward_mode: score
terminal_reward: 50
reward_scale: 10
action_set: simple
frame_skip: 4
max_pool_frames: false
max_episode_steps: 4500
completion_x_threshold: 3160
terminate_on_completion: true
terminate_on_life_loss: true
eval_freq: 0
eval_episodes: 0
```

Always use:

```text
--stop-completion-episode-window 100
--stop-completion-rate-threshold 1.0
--timesteps <active cap>
```

## Batch Record

For each batch, record:

```text
hypothesis:
config_delta:
stage: screen|confirm
active_cap:
seeds:
wandb_group:
runs: seed, run_name, wandb_id, status, time_to_100_of_100, final_rate, totals
aggregate: success_count, median_time_to_100, worst_success_time, fixed-budget rates
decision: reject|confirm_more|promote
baseline_update: previous, new, old_cap, new_cap
```

## Current Status

The historical best single run reached the older `80/100` criterion at
`2,558,256` timesteps. That does not satisfy this `100/100` goal and was not
reliably reproduced as a same-seed `80/100` result.

```text
current_baseline_status: winning screening recipe, not statistically confirmed baseline
active_cap: 5,000,000
target_stop: 100/100 completed episodes
next_step: evaluate and reproduce the post14/no-maxpool winner, then confirm on fresh seeds
```

## Active Search Notes

### B33/B34 Entropy-Floor Candidate

```text
hypothesis: lower late entropy floor improves strict 100/100 reliability
config_delta: target_kl=0.16, clip_range=0.15, fixed LR=1.5e-4, ent_coef 0.01 -> 0.0001 over 4M
runtime: stable-retro-turbo==1.0.0.post12
stage: screen then confirm
active_cap: 5,000,000
```

B33 seed-23 screen produced the first strict `100/100` post12 run:

```text
run: b33_post12_late_stability_5m_stop100ep100_targetkl016_ent1e4_4m_seed23_20260619_073650
wandb_id: 1oh4428i
status: success
time_to_100_of_100: 3,807,424
completion_total: 665
terminal_total: 1955
```

B34 confirmation did not promote the recipe:

```text
seed24: success at 4,553,920, final 100/100, 371 completions, 1969 terminal episodes, wandb_id=4qxd2a13
seed25: capped at 5,005,312, final 84/100, 388 completions, 2489 terminal episodes, wandb_id=88yxf80g
decision: reject as confirmed baseline; keep as strong hypothesis
```

Seed25 did not show a late collapse. Its completion rate was still around
`85/100` at the 5M cap after reaching about `70/100` near 4.4M, so the next
screen should target faster hard-seed convergence without changing the task,
environment, reward mode, or stop criterion.

### B35 Hard-Seed Screen

```text
hypothesis: B34 seed25 learned too slowly, so faster entropy decay or modestly higher LR might reach 100/100 before 5M
stage: screen
seed: 25
runtime: stable-retro-turbo==1.0.0.post12
active_cap: 5,000,000
```

Results:

```text
ent_coef 0.01 -> 0.0001 over 3M:
  run: b35_post12_hard_seed25_5m_stop100ep100_ent1e4_3m_seed25_20260619_121411
  wandb_id: vb91zh28
  status: capped
  final_rate: 47/100
  completions: 386
  terminal_episodes: 2660

learning_rate 1.75e-4, ent_coef 0.01 -> 0.0001 over 4M:
  run: b35_post12_hard_seed25_5m_stop100ep100_lr175e4_ent1e4_4m_seed25_20260619_121411
  wandb_id: t2gj9gim
  status: capped
  final_rate: 59/100
  completions: 201
  terminal_episodes: 1843
```

Decision: reject. The faster entropy schedule produced earlier completions
but did not maintain a reliable last-100 window; it peaked around `60/100` near
the cap and finished worse than B34 seed25. The higher-LR arm learned late but
also finished below B34. Keep the B34 4M entropy schedule as the current best
single recipe and test milder late-stability changes around it.

### B36 Hard-Seed Screen

```text
hypothesis: stay close to B34 and improve seed25 late reliability with either a lower final entropy floor or clipped score-progress shaping
stage: screen
seed: 25
runtime: stable-retro-turbo==1.0.0.post12
active_cap: 5,000,000
```

Results:

```text
ent_coef 0.01 -> 0.00005 over 4M:
  run: b36_post12_hard_seed25_5m_stop100ep100_ent5e5_4m_seed25_20260619_130424
  wandb_id: dxjfa7sx
  status: reject
  max_observed_rate: 57/100

B34 recipe + clipped score-progress shaping:
  run: b36_post12_hard_seed25_5m_stop100ep100_ent1e4_4m_clippeddx_seed25_20260619_130424
  wandb_id: 0jqd0mxq
  status: success
  time_to_100_of_100: 3,862,848
  completions_at_stop: 501
  terminal_episodes_at_stop: 2383
```

Decision: promote to confirmation candidate, not baseline. The W&B summary for
`0jqd0mxq` lagged at `98/100`, but raw history shows the strict `100/100`
window at `3,862,848`. Because this candidate was discovered on seed25, confirm
on additional seeds before promoting.

### B37 Clipped-DX Confirmation

```text
hypothesis: B36 clipped score-progress shaping at cap=30 is a reproducible improvement
config_delta: B34 recipe + score_progress_clipped=true, progress_reward_cap=30
stage: confirm
seeds: 23,24
runtime: stable-retro-turbo==1.0.0.post12
active_cap: 5,000,000
```

Results:

```text
seed23:
  run: b37_post12_clippeddx_confirm_5m_stop100ep100_seed23_20260619_134752
  wandb_id: dbc5bt5y
  status: capped
  final_rate: 35/100
  max_rate: 52/100
  completions: 207
  terminal_episodes: 1799

seed24:
  run: b37_post12_clippeddx_confirm_5m_stop100ep100_seed24_20260619_134752
  wandb_id: c5hohtky
  status: capped
  final_rate: 40/100
  max_rate: 60/100
  completions: 214
  terminal_episodes: 1781
```

Decision: reject as confirmed baseline. Cap-30 clipped progress solved seed25
but strongly hurt seeds 23 and 24. Next test should use the same B34 recipe
with a looser progress cap, paired across an easy/known seed and hard seed, to
look for an intermediate tradeoff instead of a seed25-specific fix.

## Interrupted Batch

On 2026-06-16, screen batch 1 was interrupted because the RTX4090 host had to
be shut down.

```text
task: sky_mario_100of100_screen_batch1_4090.yaml
cluster: rlab-100of100-screen1
wandb_group: 100of100-screen-batch1-20260616_151115
runtime: stable-retro-turbo==1.0.0.post4
stage: screen
seed: 23
target_stop: 100/100
status: incomplete, no promotion
```

Last observed partial signal near `2.47M-2.48M` timesteps:

```text
ent0001: 147 total completions, recent rate 0.78 after briefly logging 0.80
control: 40 total completions, recent rate 0.22
lr125: 13 total completions, recent rate 0.02
fixedlr: 4 total completions, recent rate 0.01
clip015: 0 total completions
```

Retry guidance: do not treat the interrupted `ent0001` run as a confirmed
recipe; it only reached the old `80/100` zone, not the current `100/100`
target.

## Completed Batch B30

On 2026-06-18, B30 screened post12 reliability variants on seed `23` with five
concurrent RTX4090 children and the strict `100/100` terminal-episode stop.

```text
task: sky_post12_reliability_screen_b30_4090.yaml
manifest: experiments/launches/post12_reliability_screen_b30.local.json
cluster: rlab-post12-5seed-followup
wandb_group_prefix: b30-post12-reliability-screen
runtime: stable-retro-turbo==1.0.0.post12
stage: screen
seed: 23
target_stop: 100/100
status: completed, no promotion candidate
wall_time: about 1h 9m including setup/uploads
cleanup: cluster down completed
```

Final W&B summaries:

```text
lr15e4_clip015_clippeddx: final 74/100, 386 total completions, peak 86/100 at 4,893,760
lr15e4_clip015_no_targetkl: final 14/100, 368 total completions, peak 93/100 at 4,293,056
lr15e4_clip012_no_targetkl: final 8/100, 27 total completions, peak 13/100
lr15e4_clip015_targetkl004: final 2/100, 2 total completions, peak 2/100
lr125e4_clip015_no_targetkl: final 0/100, 83 total completions, peak 44/100
```

Interpretation:

- `clip_range=0.15` can discover strong behavior on seed `23`, but without a
  KL fuse it is not stable: the rate climbed to `93/100` near `4.29M` and then
  collapsed to `14/100` by the cap.
- Clipped delta-x shaping improved final stability versus the plain arm
  (`74/100` final), but it did not reach the strict stop and peaked lower.
- `target_kl=0.04` with `clip_range=0.15` is too restrictive in this setup; it
  repeatedly tripped the PPO early-update stop and almost completely prevented
  discovery.
- Lowering LR to `1.25e-4` or clipping harder to `0.12` underfit this seed.

Next screen should keep `clip_range=0.15` and test looser KL fuses such as
`target_kl=0.08`, `0.12`, and `0.16`, with one or two variants combining the
looser fuse with clipped delta-x or fewer epochs. The hypothesis is that the
fuse must be loose enough not to block discovery, but tight enough to catch the
late destructive updates that erased the `93/100` policy.

## Completed Batch B31

On 2026-06-18, B31 screened looser KL fuses on seed `23` with five concurrent
RTX4090 children and the strict `100/100` terminal-episode stop.

```text
task: sky_post12_loose_kl_screen_b31_4090.yaml
manifest: experiments/launches/post12_loose_kl_screen_b31.local.json
cluster: rlab-post12-loose-kl-b31
wandb_group_prefix: b31-post12-loosekl-screen
runtime: stable-retro-turbo==1.0.0.post12
stage: screen
seed: 23
target_stop: 100/100
status: completed, no promotion candidate
wall_time: about 1h 8m including setup/uploads
cleanup: cluster down completed
```

Final W&B summaries:

```text
clip015_targetkl012_clippeddx: final 83/100, 799 total completions, peak 96/100 at 4,551,648
clip015_targetkl016: final 80/100, 912 total completions, peak 98/100 at 4,483,024
clip015_targetkl012: final 53/100, 173 total completions, peak 61/100
clip015_targetkl008: final 32/100, 265 total completions, peak 76/100
clip015_targetkl012_epochs5: final 29/100, 127 total completions, peak 41/100
```

Interpretation:

- `target_kl=0.16` was loose enough to preserve discovery and reached the best
  peak, `98/100`, but still degraded to `80/100` by the cap.
- Clipped delta-x with `target_kl=0.12` gave the best final stability,
  `83/100`, and a strong `96/100` peak, but still missed the strict stop.
- `target_kl=0.08` and `0.12` without clipped delta-x were weaker than the
  `0.16` fuse, and `n_epochs=5` underfit despite higher per-child fps.
- The next legal screen should combine the two useful signals: clipped delta-x
  plus a looser fuse around `target_kl=0.16`. Include a small late-pressure
  reducer, such as `n_epochs=8` or a late LR decay, only as labeled variants.

## Interrupted Batch B32

On 2026-06-18, B32 started the clipped-delta-x plus loose target-KL screen on
seed `23`, but the run was interrupted around `2.0M` timesteps because the user
stopped the batch.

```text
task: sky_post12_clippeddx_kl_screen_b32_4090.yaml
manifest: experiments/launches/post12_clippeddx_kl_screen_b32.local.json
cluster: rlab-post12-clippeddx-kl-b32
wandb_group_prefix: b32-post12-clippeddx-kl-screen
runtime: stable-retro-turbo==1.0.0.post12
stage: screen
seed: 23
target_stop: 100/100
status: interrupted/crashed, no promotion
cleanup: stale cluster down completed on 2026-06-19
```

Last W&B summaries before interruption:

```text
targetkl020_clippeddx: crashed at 1,954,688, final 3/100, 3 completions, 905 terminal episodes
targetkl016_clippeddx_lrd1e4_4m: crashed at 1,990,656, final 1/100, 1 completion, 1,073 terminal episodes
targetkl014_clippeddx: crashed at 2,064,384, final 0/100, 0 completions, 970 terminal episodes
targetkl016_clippeddx_epochs8: crashed at 2,155,312, final 0/100, 0 completions, 1,065 terminal episodes
targetkl016_clippeddx: crashed at 2,056,976, final 0/100, 0 completions, 994 terminal episodes
```

Interpretation:

- The interrupted B32 partial signal was much weaker than B31 at comparable
  early budgets, so do not automatically relaunch it unchanged.
- The best current evidence remains B31: `target_kl=0.16` reached a `98/100`
  training peak and `target_kl=0.12` plus clipped delta-x reached the best final
  stability at `83/100`.
- Next high-ROI step is to inspect/evaluate B31 checkpoints near their training
  peaks before spending more sweep budget, because final checkpoint quality is
  known to be non-monotonic.

Local checkpoint eval on 2026-06-19:

```text
B31 target_kl=0.16 checkpoint 4.3M: 18/20 stochastic eval completions
B31 target_kl=0.16 checkpoint 4.4M: 20/20 stochastic eval completions, then 95/100 on the full gate
B31 target_kl=0.16 checkpoint 4.5M: 17/20 stochastic eval completions
B31 target_kl=0.16 checkpoint 4.6M: 18/20 stochastic eval completions
B31 target_kl=0.12 + clipped delta-x checkpoint 4.4M: 20/20 stochastic eval completions, then 97/100 on the full gate
B31 target_kl=0.12 + clipped delta-x checkpoint 4.5M: 19/20 stochastic eval completions
B31 target_kl=0.12 + clipped delta-x checkpoint 4.6M: 16/20 stochastic eval completions
B31 target_kl=0.12 + clipped delta-x checkpoint 4.7M: 19/20 stochastic eval completions
```

Interpretation:

- The best B31 checkpoints are real near-winners, not just W&B training-metric
  artifacts, but they are still only around `95-97/100` under stochastic
  out-of-process eval.
- The next sweep should target the last few percent of reliability and late
  stability rather than adding more exploration.

## Completed Batch B33

```text
task: sky_post12_late_stability_screen_b33_4090.yaml
manifest: experiments/launches/post12_late_stability_screen_b33.local.json
cluster: rlab-post12-late-stab-b33
wandb_group_prefix: b33-post12-late-stability-screen
runtime: stable-retro-turbo==1.0.0.post12
stage: screen
seed: 23
target_stop: 100/100
status: completed, one promotion candidate
cleanup: cluster left warm for B34 confirmation
```

Hypothesis: B31 is already in the `95-98/100` reliability band. The remaining
gap is likely late stochastic failure/update stability, so B33 screened slightly
looser KL, lower late entropy, lower late LR, and lower minibatch-noise variants
around the B31 `clip_range=0.15`, `target_kl=0.16` near-winner.

Variants:

```text
target_kl=0.18
target_kl=0.20
target_kl=0.16 with entropy final 1e-4 over 4M
target_kl=0.16 with LR decay to 5e-5 over 4M
target_kl=0.16 with batch_size=1024
```

Final W&B summaries:

```text
targetkl016_ent1e4_4m: stopped at 3,807,424 with 100/100, 665 completions, 1,955 terminal episodes
targetkl020: final 89/100 at 5,005,312, 628 completions, 2,133 terminal episodes
targetkl016_lrd5e5_4m: final 84/100 at 5,005,312, 205 completions, 2,228 terminal episodes
targetkl018: final 37/100 at 5,005,312, 173 completions, 2,084 terminal episodes
targetkl016_batch1024: final 2/100 at 5,005,312, 117 completions, 2,028 terminal episodes
```

Interpretation:

- Lowering the late entropy floor to `0.0001` and spreading the entropy decay
  over `4M` was the first seed-23 recipe to satisfy the strict `100/100`
  completed-episode stop.
- Loosening `target_kl` to `0.20` improved final reliability versus B31 but did
  not close the last gap. `0.18`, lower late LR, and larger minibatches were
  weaker.
- The candidate is not a solved baseline yet. It must now be confirmed under the
  frozen recipe on fresh seeds.

## Active Batch B34

```text
manifest: experiments/launches/post12_ent1e4_confirm_b34.local.json
cluster: rlab-post12-ent1e4-b34
runtime: stable-retro-turbo==1.0.0.post12
stage: confirm
fresh_seeds: 24,25,26
target_stop: 100/100
expected_wall_time: up to about 70 minutes if all hit near 3.8M, longer up to the 5M cap
expected_monetary_cost: $0 on owned hardware, one RTX4090 occupied
status: first clean training attempt OOMKilled around 0.82M-0.88M before the expected first-completion zone; 64GB request was unschedulable, so retry split into 2-child seed24/25 then seed26
```

Frozen recipe:

```text
n_envs=16, n_steps=512, batch_size=512, n_epochs=10
learning_rate=1.5e-4 fixed
ent_coef=0.01 -> 0.0001 over 4M
clip_range=0.15
target_kl=0.16
reward_mode=score
terminate_on_life_loss=true
terminate_on_completion=true
stop=100/100 completed terminal training episodes
```
