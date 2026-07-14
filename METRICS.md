# Metrics Reference

This file describes the metrics this repo currently logs to Weights & Biases from the active
`src/rlab` training and evaluation paths.

Queue-backed training uses a durable SQLite telemetry outbox and exactly one long-lived W&B
publisher. With local checkpoint evaluation, the trainer, evaluator, and artifact producer write
local frames and only that publisher owns the live W&B run. With Modal checkpoint evaluation, the
live publisher handles training telemetry only; the checkpoint coordinator imports accepted early-stop
decisions into SQLite, and the fleet service resumes the same deterministic W&B run after training to
project accepted evaluation metrics and R2 checkpoint references. Training scalars are reduced into one
frame per rollout boundary. W&B's internal history step is only a delivery sequence: payloads do
not set `Run.log(step=...)`, and `global_step` is the explicit chart axis. This lets an asynchronous
evaluation for an older checkpoint arrive without rewinding W&B history.

The local `rlab.sqlite` ledger is authoritative for live training and imported early-stop decisions.
For Modal evaluation, immutable R2 attempt evidence plus the accepted PostgreSQL eval-job row are the
promotion source of truth. W&B is an eventually consistent projection after the live publisher exits.
A W&B
failure does not block the learner: frames remain retryable in the outbox. A local ledger write
failure is reported because it breaks the durable-metrics contract. TensorBoard remains a local
debugging output but is not used as the W&B transport.

Rollout value-prediction and advantage histograms are published every 64 rollout boundaries,
rather than every rollout. Their scalar distribution summaries still appear in every changed
rollout frame.

## Naming Conventions

Prefer training paths shaped as `train/<dimension>/<value-family>/<stat>` and evaluation paths
shaped as `eval/<protocol>/<dimension>/<value-family>/<stat>`, keeping names concise but explicit
enough to search by phase, protocol, info-value, reward, or progress.

Use `train` and `eval` as the first path segment. Evaluation always declares its evidence protocol
next: `screen` for the cheap first gate, `confirm` for the stronger candidate gate, and `full` for
promotion-quality evaluation. For example, use `train/done/all` and `eval/full/reward/mean`.

When training and evaluation measure the same semantic quantity, the semantic suffix after the
training phase or evaluation-protocol prefix must match. Clean Mario completion therefore uses
`train/info/level_complete/rate/min`, `eval/screen/info/level_complete/rate/min`,
`eval/confirm/info/level_complete/rate/min`, and `eval/full/info/level_complete/rate/min`.
Protocol-specific aggregation horizons may add explicit suffixes; they must not rename the underlying
quantity. Terminal reasons remain under `done/*` because termination and clean completion are distinct
concepts even when their values happen to coincide.

Use `rate` for fractions in `[0, 1]`, `count` for point-in-time counts, and standard stat suffixes such as
`mean`, `std`, `min`, `max`, `abs_mean`, and `nonzero_rate` only where a metric family explicitly logs
distribution statistics. Avoid aliases and alternate names for the same value.

New writers do not emit the retired `train/info/level_complete/rate/min/last`,
`train/info/level_complete/rate/mean/last`, `eval/info/level_complete/rate/min`,
`eval/info/level_complete/rate/mean`, `eval/done/level_change/from_rate/min`, or
`eval/done/level_change/from_rate/mean` completion aliases. Leader and ranking readers still accept
those keys when loading historical W&B runs. Historical reward readers likewise accept
`eval/reward/mean` and `eval/best/reward` while new writers use the `eval/full/*` forms.

`global_step` is the W&B step axis for training and evaluation frames. Post-training checkpoint eval logs `global_step` and
`eval/full/checkpoint/step` as the checkpoint timestep on the same W&B run that produced the checkpoint.

## Selection Metrics

These are the first metrics to check when choosing policies.

Goal contracts use `objective.rank` as the single ordered checkpoint-selection
contract. Goal-owned eval settings define the checkpoint measurement protocol.
Post-training checkpoint eval writes
canonical `eval/full/*` metrics to the producing W&B run with `global_step` set to
the checkpoint timestep. The run-local SQLite ledger is the live and recovery source of truth;
W&B is the query and visualization projection after pending telemetry has drained. The queue
database stores train-job state, not result metric projections.

For Modal-backed training, the run-local checkpoint coordinator imports accepted screen/confirm
decisions into SQLite and the run's single W&B publisher emits those `eval/screen/*` and
`eval/confirm/*` frames while training is active. The Mac-side terminal projector emits only
promotion-quality `eval/full/*` evidence after training, avoiding concurrent W&B writers and
duplicate staged rows.

| Metric | Meaning |
| --- | --- |
| `train/done/all` | Cumulative count of non-`global_reset` training `done=True` env-slot episode boundaries. This is exhaustive. |
| `train/done/<reason>` | Cumulative count of done events attributed to `<reason>`, such as `life_loss`, `level_change`, `max_steps`, or `unclassified`. Reason counters are explanatory and do not have to sum to `train/done/all`. |
| `train/done/serve_stall` | Cumulative Breakout training episodes terminated as task failures after `ball_y` remained zero for the configured consecutive policy-step threshold. Life loss itself remains nonterminal; this reason means the policy failed to relaunch the ball within the allowed serve window. |
| `train/done/max_steps` | Cumulative count of terminal training episodes attributed to max-step truncation. Emitted as `0` before the first max-step episode. |
| `train/done/unclassified` | Cumulative count of terminal training episodes that had no configured done reason and were not max-step truncations. Emitted as `0` before the first unclassified episode. |
| `train/done/<reason>/from/<prev>` | Cumulative count of structured done events for `<reason>` whose native payload reported previous value `<prev>`. Multi-key values are joined with `-`, e.g. `0-0`. |
| `train/done/<reason>/from/<prev>/ep_window/rate` | Fraction of the last 100 non-`global_reset` terminal training episodes whose configured source value for `<reason>` was `<prev>` that ended with that structured done event. Each `<reason>/from/<prev>` has its own 100-episode denominator and emits only after that per-source window is full. |
| `train/done/<reason>/from_rate/min` | Minimum across full per-source terminal episode-window rates for `<reason>`. |
| `train/done/<reason>/from_rate/mean` | Mean across full per-source terminal episode-window rates for `<reason>`. |
| `train/info/level_complete` | Root for Mario training level-complete metrics derived from task-event records. |
| `train/info/level_complete/from/<prev>/count` | Cumulative clean level clears from native source value `<prev>`, e.g. `0-0` for Level1-1. Death or life-loss info always records as a failed attempt, even if an upstream completion flag is also present. |
| `train/info/level_complete/from/<prev>/attempts` | Number of attempts currently represented in the rolling attempt window for `<prev>`, capped at 100. Emits from the first observed attempt and is useful for seeing whether the strict rate window has filled. |
| `train/info/level_complete/from/<prev>/rate/current` | Fraction of the current rolling attempt window for `<prev>` that produced a clean `level_complete`, using the available denominator up to 100 attempts. This is for live monitoring, not strict promotion. |
| `train/info/level_complete/from/<prev>/rate` | Fraction of the last 100 attempts from `<prev>` that produced a clean `level_complete`. Attempts can end at a clean completion, life loss, truncation, or episode done; death/life-loss attempts contribute `0`. Emits only after that source has a full 100-attempt window. |
| `train/info/level_complete/rate/min/current` | Minimum across the latest available `train/info/level_complete/from/<prev>/rate/current` values. Emits from the first observed attempt and is intended for live charts while strict windows are still filling. |
| `train/info/level_complete/rate/mean/current` | Mean across the latest available `train/info/level_complete/from/<prev>/rate/current` values. Emits alongside `rate/min/current`. |
| `train/info/level_complete/rate/min` | Minimum across the latest full-window `train/info/level_complete/from/<prev>/rate` values. Emits after at least one per-source strict rate is available and updates whenever any per-source rate updates. |
| `train/info/level_complete/rate/mean` | Mean across the latest full-window `train/info/level_complete/from/<prev>/rate` values. Emits alongside `rate/min` and updates whenever any per-source rate updates. |
| `eval/full/done/level_change/rate` | Pooled fraction of eval episodes whose terminal record contained a level-change event, whether or not it qualified as a clean completion. |
| `eval/full/done/level_change/from/<start>/rate` | Level-change terminal-event fraction for eval episodes that started from `<start>`. |
| `eval/full/info/level_complete/from/<start>/count` | Clean eval completions from `<start>`. |
| `eval/full/info/level_complete/from/<start>/attempts` | Eval episodes from `<start>` represented in the completion rate. |
| `eval/full/info/level_complete/from/<start>/rate` | Clean completion fraction for eval episodes from `<start>`. |
| `eval/full/info/level_complete/rate/min` | Minimum per-start clean completion fraction. Use this first when comparing multi-start-state policies. |
| `eval/full/info/level_complete/rate/mean` | Mean per-start clean completion fraction. |
| `eval/full/done/<reason>/rate` | Fraction of eval episodes whose terminal task record contained configured event `<reason>`, including generic identity-task reasons such as `serve_stall`. |

Current training does not log per-rollout done-count distribution stats such as `train/done/min`,
`train/done/mean`, or `train/done/max`. The aggregate all-done counter is `train/done/all`.

### Selection and Redundancy Notes

Training level-complete metrics live under `train/info/level_complete/*` and count attempts, not full
episodes. Use them for live training clear counts/rates when a policy can clear multiple levels
inside one episode. They are emitted only for tasks that declare the `level_change` event; identity
tasks such as Breakout do not run this Mario-specific reducer. Use `train/done/*` only to understand
what is ending training episodes.

`eval/full/info/level_complete/rate/min` is the eval selection metric for Mario-style multi-start-state
policies that define a clean completion event. Environments without a target-level completion
contract, such as Breakout, normally use `eval/full/reward/mean` as their reward objective. The current
Breakout goal first minimizes `eval/full/done/serve_stall/rate`, then ranks reward, so a checkpoint that
cannot reliably relaunch the ball cannot hide that failure behind score accumulated earlier.
The top-level eval metrics are pooled summaries and should be treated as secondary when
per-start-state eval done metrics exist.
After a checkpoint reaches the completion goal (`>=0.99` bottleneck completion), use
`leader/checkpoint/steps_to_completion_goal` to choose the most sample-efficient solved run before
falling back to reward.

### Mario Level1-1/Level1-2/Level1-3/Level1-4 Notes

For the current Level1 single-level and mixed-level training goals, native level values map to
training metrics as:

| Level | Count metric | Rate metric |
| --- | --- | --- |
| `Level1-1` | `train/info/level_complete/from/0-0/count` | `train/info/level_complete/from/0-0/rate` |
| `Level1-2` | `train/info/level_complete/from/0-1/count` | `train/info/level_complete/from/0-1/rate` |
| `Level1-3` | `train/info/level_complete/from/0-2/count` | `train/info/level_complete/from/0-2/rate` |
| `Level1-4` | `train/info/level_complete/from/0-3/count` | `train/info/level_complete/from/0-3/rate` |

For active multi-level training, use `train/info/level_complete/rate/min/current` as the live
bottleneck chart while attempt windows are still filling. It is the minimum of the currently
available per-level partial-window rates, so it appears much earlier than the strict full-window
metric. Use `train/info/level_complete/from/<prev>/attempts` next to it to see the current
denominator.

Use `train/info/level_complete/rate/min` as the strict training gate and comparable
full-window bottleneck. It is the minimum of the most recent full-window per-level rates that have
emitted so far. For example, if Level1-1 is `0.50` and Level1-2 is `0.30`, `rate/min` is
`0.30`. If Level1-1 later drops from `1.00` to `0.50` while Level1-2 drops from `0.60` to `0.55`,
`rate/min` is `0.50`. Use `train/info/level_complete/rate/mean` as the companion average
over those same latest per-level rates, mainly to distinguish policies with the same bottleneck. Training
intentionally no longer logs generic `train/event/*` or `train/outcome/*` metrics. Once post-train
checkpoint eval has logged per-start metrics,
use `eval/full/info/level_complete/rate/min` as the balanced eval selection metric.

`rate/min` and `rate/mean` are intentionally available-source live metrics. If only one
source has a full attempt window so far, both report that source's rate; as more source windows emit,
the minimum automatically tightens to include them and the mean averages across them. Use the companion
`train/info/level_complete/from/<prev>/attempts`, `/rate/current`, `/rate`, and `/count` metrics to
see which sources currently have coverage. On fast-solved single-level runs, `/rate/min` can
have only one W&B history point because the first full 100-attempt window immediately satisfies the
early-stop gate; use `/rate/min/current` for the live trace in that case.

`train/done/level_change/from_rate/min` and `train/done/level_change/from_rate/mean` are terminal
episode-window diagnostics. Keep them as done-reason diagnostics; for clear counts/rates, prefer
`train/info/level_complete/from/<prev>/count` and `train/info/level_complete/from/<prev>/rate`
because they also count non-terminal clears and exclude death or life-loss transitions. For a
single training selection scalar across source levels, prefer
`train/info/level_complete/rate/min`; use `train/info/level_complete/rate/mean` as a
tiebreaker/secondary signal when bottleneck rates match.

Goal contracts should not restate the `train/info/level_complete/from/<prev>/rate` 100-attempt FIFO
window as a separate success setting. The metric definition owns that window. Goal files should
declare training stop gates as a list in `train.early_stop`; multiple metric-threshold rules are
combined with AND. Ordered ranking signals stay in `objective.rank`.

Use current `train/reward_share/*` metrics for reward attribution rather than the older
`train/reward_component/*` namespace. Shares are based on absolute rollout contribution
magnitude, so negative components such as death or time penalties are visible by magnitude
rather than canceling against positive reward.

Training events are declared under `environment.task.events`; task termination maps those events to
success or failure under `environment.task.termination`. Providers emit raw columnar signals only.
The bound task kernel evaluates events at vector-step boundaries and emits compact event and episode
records through `RlabVecEnv.drain_records()`. `RlabCallback` consumes those records without copying
task metrics into every SB3 `info` dictionary.

Kernel-derived terminal events retain their configured reason in `train/done/<reason>`. For example,
a `stalled` event configured as failure emits `train/done/stalled`; a stall configured as timeout also
emits `train/done/stalled` alongside `train/done/max_steps` because reasons are explanatory and may
overlap.

Native engine terminations that do not match a task event and are not task/provider truncations count
as `train/done/unclassified`. Terminal episode records preserve event reasons and readable start IDs;
Mario start names are normalized back to native level coordinates for existing metric paths such as
`train/done/level_change/from/0-0`. Training intentionally does not emit `to` or full-transition
counters because those multiply metric cardinality quickly.

Evaluation, including post-training checkpoint eval, uses the same provider, bound task kernel, and
batch runtime. Its task contract keeps `level_change` as success termination while removing life-loss
failure termination. This stops successful single-level Mario eval episodes at the vector step that
observes the transition. `eval/full/done/level_change*` describes the terminal reason, while
`eval/full/info/level_complete*` describes clean completion using the same suffixes as training. Eval
`from` values remain configured episode start identities.

`train/done/*` windows remain terminal-episode metrics. Natural clean clears observed while the
training env keeps running set `level_complete` / `completion_event`, increment
`train/info/level_complete/from/<prev>/count`, and append to that source's
`train/info/level_complete/from/<prev>/rate` attempt window. This is the metric family to use when
the question is "did the policy clear this level?"

To sniff whether training is terminating on specific Mario events, chart the terminal counters
rather than the clean-clear counters:

- `train/done/life_loss` increasing means life-loss events are ending training episodes.
- `train/done/level_change` increasing means level-change events are ending training episodes.
- `train/done/all` is the exhaustive terminal episode count. Reason counters are explanatory, so
  `train/done/life_loss + train/done/level_change` can exceed `train/done/all` when the same terminal
  info payload reports both events.
- Removing events from task termination also removes their reset boundaries. If no provider engine
  boundary or task truncation occurs, `train/done/*` can stay absent
  while rollout reward collapses into sparse components such as `train/reward_share/death`. In that
  regime, use `train/reward/prog_x/*`, `train/reward/score/*`, and
  `train/info/level_complete/*` to confirm whether the agent is still receiving progress and clean
  clear signal.
- `train/info/level_complete/from/<prev>/count` should be less than or equal to
  `train/done/level_change/from/<prev>` when level changes are terminal. Any excess
  `train/done/level_change/from/<prev>` over clean `level_complete` count is terminal level-change
  traffic that was not accepted as a clean clear, for example a life-loss/death transition.

`level_change` means the bound level signal changed at a vector-step boundary. The Mario task kernel
classifies that transition as a clean completion only when no death or life loss occurred in the same
step. A clean transition from `(0, 0)` to `(0, 1)` therefore records
`train/info/level_complete/from/0-0/count`; simultaneous level change and life loss is a failed
attempt. Non-terminal clean transitions are emitted immediately as task-event records, so this metric
continues to work for multi-level and no-terminal training contracts.

### ALE / Ms. Pac-Man Notes

For ALE targets such as Ms. Pac-Man, `train/done/all` is the cumulative completed training episode
count. It increments whenever a vector-env slot reports `done=True`, including natural game-over
episodes and max-frame truncations. Use `eval/full/done/all` for the number of episodes summarized by a
standalone or post-training eval run.

Current Ms. Pac-Man recipes do not define a clean completion event, so Mario-style completion metrics
such as `train/info/level_complete/*`, `eval/full/done/level_change/*`, and `completion_count` are not
expected. Selection should use reward metrics such as `eval/full/reward/mean` and `eval/full/best/reward`.

Current ALE/Ms. Pac-Man training also does not emit a cumulative life-loss metric. The base Ms. Pac-Man
identity task does not bind `ale.lives` to a life-loss event. If a run is launched with ALE
`episodic_life=true`, life loss may become a training episode boundary, but without an explicit
life-loss counter those episode boundaries still should be read through `train/done/all` and
`train/done/unclassified`, not as total lives lost.

## SB3 PPO Metrics

These come from Stable-Baselines3 PPO and `VecMonitor`. rlab installs a durable SB3 logger output
format, so the complete scalar payload is written to the SQLite telemetry outbox and W&B from inside
SB3's `logger.dump()`, before SB3 clears it. PPO `train/*` values are produced after the normal
rollout dump and are therefore published at the following dump; rlab performs a final dump at
training end so the last PPO update is not lost.

| Metric | Meaning |
| --- | --- |
| `rollout/ep_rew_mean` | Mean shaped episode return over SB3's monitor window. This is the reward used by training, not raw game score; current Breakout training clips native reward events. It appears only after at least one monitored episode completes. |
| `rollout/ep_len_mean` | Mean episode length in policy environment steps over SB3's rolling window of the last 100 completed episodes. It appears only after at least one monitored episode completes. A long episode contributes only after it finishes and may span many PPO updates or saved checkpoints, so a spike must not be attributed to the nearest checkpoint without a frozen-policy evaluation. For example, Breakout's 54,000-step cap at frame skip 4 represents 216,000 emulator frames. |
| `time/fps` | Cumulative SB3 training throughput in environment steps per second. |
| `time/iterations` | Number of PPO learn iterations completed. |
| `time/time_elapsed` | Wall-clock seconds elapsed in the SB3 learn loop. Explicitly mirrored to W&B at rollout end with `global_step` as the step. |
| `time/total_timesteps` | Total environment steps reached by SB3 training. This should match `global_step` on the same history frame. |
| `train/approx_kl` | Approximate KL divergence between old and updated policies for the last PPO update. Spikes indicate large policy updates. |
| `train/clip_fraction` | Fraction of policy updates clipped by PPO's ratio clipping. High values mean many updates hit the trust-region bound. |
| `train/clip_range` | Active PPO policy clip range. |
| `train/clip_range_vf` | Active value-function clip range. Logged only when value clipping is configured. |
| `train/entropy_loss` | Negative entropy term from PPO. More negative generally means higher action entropy. |
| `train/explained_variance` | How much return variance the value function explains. Near 1 is good; near 0 or negative means weak value prediction. |
| `train/learning_rate` | Active optimizer learning rate after any schedule. |
| `train/loss` | Combined PPO loss for the last update. |
| `train/n_updates` | Cumulative optimizer update count. |
| `train/policy_gradient_loss` | PPO policy-gradient loss component. |
| `train/std` | Mean learned action-distribution standard deviation, logged by SB3 only for policies with `log_std`. Usually absent for discrete-action Mario policies. |
| `train/value_loss` | PPO value-function loss component. |

Do not confuse `train/clip_range` with `train/clip_fraction`: the range is the configured PPO ratio
limit, while a zero fraction means no sampled policy ratios crossed that limit. A zero fraction is
not itself a disabled clip range; interpret it together with `train/approx_kl`, policy entropy, and
reward progress to determine whether policy updates are too small.

## Throughput Metrics

| Metric | Meaning |
| --- | --- |
| `throughput/rollout_fps` | Rollout-only environment-step throughput, measured from rollout start to rollout end. This excludes PPO optimization time. |
| `throughput/native_env_step_fps` | Provider vector-env step throughput in environment steps per second, measured only around the underlying Gymnasium vector `env.step(...)` calls during rollout collection. This excludes PPO policy inference, callbacks, rollout-buffer work, and optimizer time. |
| `throughput/native_env_step_batch_fps` | Provider vector-env batch-step throughput in vector steps per second. Multiply by `n_envs` to compare with environment-step throughput. |
| `throughput/native_env_step_seconds` | Wall-clock seconds spent inside provider vector-env `env.step(...)` calls during the rollout. |
| `throughput/native_env_step_fraction` | Fraction of rollout collection wall time spent inside provider vector-env `env.step(...)`. Values near `1` mean native env stepping dominates rollout time; lower values point to policy inference, the task kernel/runtime, callbacks, or rollout-buffer overhead. |
| `throughput/loop_fps` | Full-loop instantaneous throughput, measured from one rollout start to the next. This includes rollout collection plus PPO optimization overhead. |

For training-speed and host-saturation comparisons, prefer `throughput/loop_fps` over total wall
time. It is the closest per-run W&B metric to effective training throughput because it includes
both environment rollout collection and PPO optimization. Use `throughput/rollout_fps` beside it to
separate rollout collection from optimizer overhead. Use `throughput/native_env_step_fps` and
`throughput/native_env_step_fraction` to isolate the provider vector-env call inside rollout
collection: if native env-step FPS drops as parallel jobs increase, the emulator/native preprocessing
path is the likely bottleneck; if native env-step FPS is high but rollout FPS is low, policy
inference, task-kernel/runtime work, callback logging, or rollout-buffer overhead is more likely; if rollout
FPS is steady but loop FPS drops, PPO optimization, GPU scheduling, or host contention outside
rollout collection is more likely. Use `time/fps` only as a cumulative SB3 sanity check, and discount
points near checkpoint/final artifact events separately from artifact-worker timing.

Do not compare `throughput/native_env_step_fps` directly with the default
`SuperMarioBros-Nes-turbo` `scripts/benchmark_sps.py` output unless the benchmark is configured to
match the training contract. The provider-only benchmark excludes task-kernel event detection, reward
shaping, record creation, masked reset, and SB3 adaptation. The native-step timer measures only the
provider call; use matching provider-plus-runtime and end-to-end PPO benchmarks for consolidation
regression claims.

## Artifact Timing Metrics

These sparse metrics are logged when the artifact worker flushes model artifacts. Checkpoint rows use
the checkpoint step as `global_step`; final and best artifacts use the model's current timestep. They
measure worker-side artifact latency, not trainer stall time.

| Metric | Meaning |
| --- | --- |
| `train/artifact/stall_seconds` | Wall-clock artifact-worker flush duration. It does not represent trainer stall time for async checkpoint/final artifacts. |
| `train/artifact/local_save_seconds` | Local SB3 checkpoint or final model save duration. Logged when the local save can be paired with the artifact log. |
| `train/artifact/log_seconds` | Total wall-clock time spent inside artifact metadata/upload/logging after the model zip already exists. |
| `train/artifact/metadata_seconds` | Time spent writing the checkpoint metadata sidecar. |
| `train/artifact/storage_upload_seconds` | Time spent uploading the model zip to external S3/R2 storage before W&B receives a reference artifact. Zero when no external artifact storage URI is configured. |
| `train/artifact/wandb_log_seconds` | Time spent in `wandb_run.log_artifact(...)` plus artifact upload completion when the W&B SDK exposes `wait()`. With reference artifacts this mostly covers W&B metadata/reference logging and sidecar upload; without reference artifacts it includes uploading the model zip to W&B. |

## Rollout Diagnostics

Logged at rollout end from the SB3 rollout buffer.

| Metric | Meaning |
| --- | --- |
| `rollout/value_pred/mean` | Mean value-function prediction over the collected rollout buffer. |
| `rollout/value_pred/std` | Standard deviation of value predictions. |
| `rollout/value_pred/min` | Minimum value prediction. |
| `rollout/value_pred/max` | Maximum value prediction. |
| `rollout/value_pred/abs_mean` | Mean absolute value prediction. |
| `rollout/value_pred/hist` | W&B histogram of rollout-buffer value predictions. Logged directly to W&B. |
| `rollout/advantage/mean` | Mean computed advantage over the collected rollout buffer. |
| `rollout/advantage/std` | Standard deviation of computed advantages. |
| `rollout/advantage/min` | Minimum computed advantage. |
| `rollout/advantage/max` | Maximum computed advantage. |
| `rollout/advantage/abs_mean` | Mean absolute computed advantage. |
| `rollout/advantage/hist` | W&B histogram of rollout-buffer advantages. Logged directly to W&B. |

For `value_pred` and `advantage`, `mean` preserves sign and can cancel positive and negative
entries, while `abs_mean` removes sign before averaging and tracks typical magnitude. For example,
predictions `[-10, 10]` have `mean = 0` but `abs_mean = 10`. Use `rollout/value_pred/mean` to
see value-function bias or drift in one direction, and `rollout/value_pred/abs_mean` to see whether
the critic's predicted returns are large regardless of sign.

## Reward Component Diagnostics

Logged at rollout end from vectorized task-metric batches drained from `RlabVecEnv`. Component
arrays are copied into reusable preallocated NumPy rollout buffers and reduced in one batch; values
are not retained in per-step or per-lane Python containers.

`train/reward/<component>/<stat>` is logged for each component that appears during the rollout.

Components:

| Component | Source field | Meaning |
| --- | --- | --- |
| `shaped` | `shaped_reward` | Final shaped reward passed toward training. |
| `raw` | `raw_reward` | Raw environment reward before repo reward shaping. |
| `native` | `native_reward_component` | Native stable-retro reward component. |
| `prog` | `progress_component` | Generic progress component supplied by the bound task kernel. |
| `prog_x` | `progress_reward_component` | X-position progress reward component. |
| `score` | `score_reward_component` | Score-derived reward component. |
| `score_d` | `score_delta` | Raw score delta observed in the step. |
| `done` | `completion_reward_component` | Completion bonus component. |
| `death` | `death_penalty_component` | Death penalty component. |
| `time` | `time_penalty_component` | Per-step or time penalty component. |

Stats:

| Metric template | Meaning |
| --- | --- |
| `train/reward/<component>/mean` | Mean component value over collected info records. |
| `train/reward/<component>/std` | Standard deviation of component values. |
| `train/reward/<component>/min` | Minimum component value. |
| `train/reward/<component>/max` | Maximum component value. |
| `train/reward/<component>/abs_mean` | Mean absolute component value. |
| `train/reward/<component>/nonzero_rate` | Fraction of collected values where the component was nonzero. |

Reward share metrics compare absolute component magnitudes within a rollout.
`train/reward_share/<component>` is logged for each share component only when the task emits reward
component data during that rollout. Identity tasks with native rewards do not emit empty, zero-valued
reward-share families:

| Metric | Meaning |
| --- | --- |
| `train/reward_share/prog_x` | Share of absolute reward-component mass from X-progress reward. |
| `train/reward_share/score` | Share from score reward. |
| `train/reward_share/death` | Share from death penalties. |
| `train/reward_share/done` | Share from completion bonuses. |
| `train/reward_share/time` | Share from time penalties. |
| `train/reward_share/native` | Share from native stable-retro reward. |

## Optional Training Metrics

| Metric | Logged when | Meaning |
| --- | --- | --- |
| `train/ent_coef` | `--ent-coef-final` is set | Active entropy coefficient from the entropy coefficient schedule. |
| `train/adv_norm/mode` | `--advantage-normalization per-task` | Marker value `1.0`, meaning per-task advantage normalization is active. |
| `train/adv/task<id>/count` | Per-task advantage normalization | Number of rollout-buffer samples assigned to task `<id>`. |
| `train/adv/task<id>/mean_pre` | Per-task advantage normalization | Mean task advantage before normalization. |
| `train/adv/task<id>/std_pre` | Per-task advantage normalization | Standard deviation before normalization. |
| `train/adv/task<id>/mean_post` | Per-task advantage normalization | Mean task advantage after normalization. |
| `train/adv/task<id>/std_post` | Per-task advantage normalization | Standard deviation after normalization. |

## Evaluation Metrics

These are logged by post-training checkpoint eval and by local `rlab eval` model checks.
Post-training checkpoint eval uses the goal's declared `eval.environment` contract, including its
provider, preprocessing, task termination, and reset semantics. The current Mario goals choose
`level_change` as terminal while allowing life-loss observations to remain non-terminal.

| Metric | Meaning |
| --- | --- |
| `eval/full/reward/mean` | Mean eval episode return under the goal-owned eval reward contract. Current Breakout eval disables training-time reward clipping, so this is real game score. |
| `eval/full/reward/std` | Standard deviation of eval episode returns. |
| `eval/full/reward/max` | Maximum eval episode return. |
| `eval/full/progress/x/mean` | Mean max global X position reached per eval episode. |
| `eval/full/progress/x/max` | Maximum global X position reached by any eval episode. |
| `eval/full/progress/level_x/mean` | Mean max level-local X position reached per eval episode. |
| `eval/full/progress/level_x/max` | Maximum level-local X position reached by any eval episode. |
| `eval/full/done/all` | Number of eval episodes summarized. This is exhaustive. |
| `eval/full/done/level_change` | Eval episodes whose terminal record contains a level-change event. For current single-level Mario checkpoint evals, this event normally ends the episode, but clean completion is reported separately under `eval/full/info/level_complete/*`. |
| `eval/full/done/level_change/rate` | `eval/full/done/level_change / eval/full/done/all`. |
| `eval/full/done/max_steps` | Eval episodes that hit the max-step limit. Should not overlap with `eval/full/done/level_change` for current single-level Mario checkpoint evals because level change is terminal there. |
| `eval/full/done/max_steps/rate` | `eval/full/done/max_steps / eval/full/done/all`. |
| `eval/full/done/terminated` | Eval episodes that ended via native env termination without being marked as max-step truncations. This is generic and is emitted for all target types. |
| `eval/full/done/terminated/rate` | `eval/full/done/terminated / eval/full/done/all`. |
| `eval/full/done/<reason>` | Eval episodes whose terminal task record contained configured event `<reason>`, including `serve_stall`. |
| `eval/full/done/<reason>/rate` | `eval/full/done/<reason> / eval/full/done/all`. |
| `eval/full/done/unclassified` | Eval episodes that ended without a configured task event and without max-step truncation. |
| `eval/full/done/unclassified/rate` | `eval/full/done/unclassified / eval/full/done/all`. |
| `eval/full/info/<name>` | Numeric task or provider information emitted by an eval episode, flattened under the canonical `eval/full/info/` prefix. |
| `eval/full/death/count` | Eval episodes where the target-specific death flag was observed at least once during the eval horizon. Currently Mario-specific. |
| `eval/full/death/rate` | `eval/full/death/count / eval episodes`. Currently Mario-specific. |
| `eval/full/best/reward` | Return of the best eval episode. Mario ranks best episodes by completion, then max X, then reward; generic targets rank by reward. |
| `eval/full/best/x` | Max global X position of the best eval episode. Logged only when the target defines global X progress. |
| `eval/full/checkpoint/step` | Checkpoint timestep being evaluated by the async eval worker. Eval also logs this value as `global_step` so W&B panels plot the result at the evaluated model timestep without forcing W&B's internal history step backward. |
| `eval/full/checkpoint/artifact` | W&B checkpoint artifact name or local checkpoint ref evaluated by the async eval worker. |
| `eval/full/config/hud_crop_top` | HUD crop used for checkpoint eval. |
| `eval/full/source` | Producer of this eval payload, such as `async_worker`, a staged async worker name, or `modal` for accepted remote evidence. |
| `eval/full/episodes` | Number of episodes summarized in this eval payload. |
| `eval/full/duration/seconds` | Wall-clock seconds spent inside `evaluate_model_episodes(...)`, including episode rollout and any requested best-episode video rendering. Logged to W&B for checkpoint eval when present. |
| `eval/<stage>/<eval_metric_suffix>` | Training-time staged checkpoint eval metric. `screen` and `confirm` preserve the semantic suffix of canonical `eval/full/*` metrics, for example `eval/screen/info/level_complete/rate/min`. They are not promotion-quality full eval metrics. |
| `eval/<stage>/pass` | `1` when the stage pass rules matched for that checkpoint, otherwise `0`. The default Mario `screen` and `confirm` stages require perfect per-start completion. |
| `eval/<stage>/stage_index` | Numeric order of the staged checkpoint eval step. Lower stages are cheaper screens; later stages are stronger confirmation gates. |
| `eval/<stage>/source` | Non-numeric evidence producer label. Modal decisions use `modal`; it is retained in decision/W&B payloads but is not inserted into SQLite's numeric latest-metric table. |
| `eval/confirm/candidate/pass` | Candidate early-stop signal emitted only by a stage marked `candidate_stop` after its pass rules match. Mario training early stop watches this metric instead of `eval/screen/*` or canonical `eval/full/*`, so 10-episode screens cannot stop training by themselves. |
| `eval/confirm/candidate/checkpoint_step` | Checkpoint timestep that produced the current candidate early-stop signal. |
| `eval/confirm/candidate/stage_index` | Stage index that produced the current candidate early-stop signal. |
| `eval/confirm/candidate/episodes` | Number of episodes used by the candidate-stop stage. |
| `leader/checkpoint/objective` | Raw value of the first `objective.rank` metric for the best evaluated checkpoint on a source run. Used by `rlab leaders checkpoints` as a query prefilter; final ordering replays the full saved rank. |
| `leader/checkpoint/objective_name` | Metric name represented by `leader/checkpoint/objective`, such as `eval/full/info/level_complete/rate/min` or `eval/full/reward/mean`. |
| `leader/checkpoint/rank` | Ordered `objective.rank` expressions used to select this checkpoint. |
| `leader/checkpoint/rank_values` | Raw metric values for `leader/checkpoint/rank`, retained so leader queries reproduce the goal contract rather than a hardcoded order. |
| `leader/checkpoint/completion_rate` | W&B summary field for Mario-style completion-aware runs, using `eval/full/info/level_complete/rate/min` when available. Omitted for generic targets without a completion event. |
| `leader/checkpoint/completion_rate_mean` | W&B summary tiebreaker for Mario-style completion-aware runs, using `eval/full/info/level_complete/rate/mean` when available. Omitted for generic targets without a completion event. |
| `leader/checkpoint/best_reward` | Best episode return for the selected checkpoint. |
| `leader/checkpoint/steps_to_completion_goal` | Checkpoint timestep used by the corresponding `objective.rank` criterion. Completion-aware ranks omit it until bottleneck completion reaches `0.99`; reward-only ranks use it directly as the sample-efficiency tiebreaker. |
| `leader/checkpoint/reward_mean` | Mean eval reward for the selected checkpoint. Its position in selection is determined only by `objective.rank`. |
| `leader/checkpoint/max_x_max` | Progress field for the source run's best evaluated checkpoint. Reported for inspection but not part of the current objective rank. |
| `leader/checkpoint/step` | Checkpoint step for the source run's current best evaluated checkpoint. |
| `leader/checkpoint/artifact_ref` | Artifact ref for the source run's current best evaluated checkpoint. |
| `leader/checkpoint/local_path` | Local checkpoint path for the source run's current best evaluated checkpoint when it is available on the eval worker. |
| `leader/checkpoint/eval_source` | Source that produced the current best-checkpoint summary update, normally `async_worker`. |
| `leader/checkpoint/updated_at` | UTC timestamp when the source run's best-checkpoint summary fields were last updated. |

Per-start-state eval done metrics mirror the training done namespace as
`eval/full/done/<reason>/from/<start>`. Eval preserves configured task failures other than life loss and
preserves level-change termination, but labels
`<start>` with the eval episode start state, for example `Level1-1`, rather than a native
previous-value tuple such as `0-0`.

| Metric template | Meaning |
| --- | --- |
| `eval/full/done/all/from/<start>` | Number of eval episodes that started from `<start>`. This is the denominator for that start state. |
| `eval/full/done/level_change/from/<start>` | Eval episodes from `<start>` whose terminal record contained a level-change event. |
| `eval/full/done/level_change/from/<start>/rate` | `eval/full/done/level_change/from/<start> / eval/full/done/all/from/<start>`. |
| `eval/full/info/level_complete/from/<start>/count` | Clean level completions from `<start>`. |
| `eval/full/info/level_complete/from/<start>/attempts` | Eval episodes from `<start>` used as the completion denominator. |
| `eval/full/info/level_complete/from/<start>/rate` | Clean level completion fraction for `<start>`. |
| `eval/full/info/level_complete/rate/min` | Minimum per-start-state clean completion rate. Use this for balanced multi-state eval ranking. |
| `eval/full/info/level_complete/rate/mean` | Mean per-start-state clean completion rate. |
| `eval/full/done/<reason>/from/<start>` | Eval episodes from `<start>` whose terminal task record contained configured event `<reason>`. |
| `eval/full/done/<reason>/from/<start>/rate` | `eval/full/done/<reason>/from/<start> / eval/full/done/all/from/<start>`. |
| `eval/full/done/max_steps/from/<start>` | Eval episodes from `<start>` that hit the max-step limit. Can overlap with level-change counts. |
| `eval/full/done/max_steps/from/<start>/rate` | `eval/full/done/max_steps/from/<start> / eval/full/done/all/from/<start>`. |
| `eval/full/done/terminated/from/<start>` | Eval episodes from `<start>` that ended via native env termination without max-step truncation. |
| `eval/full/done/terminated/from/<start>/rate` | `eval/full/done/terminated/from/<start> / eval/full/done/all/from/<start>`. |
| `eval/full/done/unclassified/from/<start>` | Eval episodes from `<start>` that ended without a configured task event or max-step truncation. |
| `eval/full/done/unclassified/from/<start>/rate` | `eval/full/done/unclassified/from/<start> / eval/full/done/all/from/<start>`. |

## Eval JSON Summary Fields

`rlab eval` and post-training checkpoint eval write richer JSON summaries. These fields are
stored in stdout JSON or post-train eval outputs; only the `eval/full/*` subset above is logged to W&B by default.

| Field | Meaning |
| --- | --- |
| `episodes` | Number of eval episodes summarized. |
| `deterministic` | Policy-action mode provenance. Always `false`: all supported evaluation and playback paths use stochastic sampling. |
| `reward_mean` | Mean eval episode return before mapping to `eval/full/reward/mean`. |
| `reward_std` | Standard deviation of eval episode returns before mapping to `eval/full/reward/std`. |
| `reward_max` | Maximum eval episode return before mapping to `eval/full/reward/max`. |
| `max_x_mean` | Mean max global X position before mapping to `eval/full/progress/x/mean`. Emitted only when target eval semantics define global X progress. |
| `max_x_max` | Maximum global X position before mapping to `eval/full/progress/x/max`. Emitted only when target eval semantics define global X progress. |
| `max_level_x_mean` | Mean max level-local X position before mapping to `eval/full/progress/level_x/mean`. Emitted only when target eval semantics define level-local X progress. |
| `max_level_x_max` | Maximum level-local X position before mapping to `eval/full/progress/level_x/max`. Emitted only when target eval semantics define level-local X progress. |
| `completion_count` | Eval episodes where a target-specific clean completion event was observed at least once. Currently emitted for Mario-style targets only. |
| `completion_rate` | `completion_count / episodes`. Same rate as `eval/full/done/level_change/rate` for Mario-style targets. |
| `death_count` | Eval episodes where target-specific death was observed at least once. Currently emitted for Mario-style targets only. |
| `death_rate` | `death_count / episodes`. Same rate as `eval/full/death/rate` when death semantics exist. |
| `terminated_count` | Eval episodes that hit native env done without being marked as max-step truncations. Completion alone does not increment this. |
| `terminated_rate` | `terminated_count / episodes`. |
| `truncated_count` | Eval episodes that hit the max-step limit. Same count as `eval/full/done/max_steps`; can overlap with `completion_count`. |
| `truncated_rate` | `truncated_count / episodes`. Same rate as `eval/full/done/max_steps/rate`. |
| `unclassified_count` | Eval episodes that ended without level completion or max-step truncation. Same count as `eval/full/done/unclassified`. |
| `unclassified_rate` | `unclassified_count / episodes`. Same rate as `eval/full/done/unclassified/rate`. |
| `death_x_histogram` | Local JSON histogram of death X positions. This structured value is not emitted as a W&B scalar. |
| `episode_results` | Per-episode records used to build the summary. Removed from stdout when `--summary-only` is set. |
| `episode_results[].seed_protocol` | Versioned stochastic RNG trace contract. Remote checkpoint eval currently requires `vector-lane-v1`. |
| `episode_results[].seed` | Scalar eval uses `base_seed + episode_ordinal`; vector eval records the shared base seed because lane/reset order is traced separately. |
| `episode_results[].seed_lane` | Vector lane that produced the episode; scalar eval uses lane `0`. |
| `episode_results[].seed_episode_ordinal` | Zero-based reset/episode ordinal within that lane. The `(seed_lane, seed_episode_ordinal)` pair is unique within one evaluation. |
| `episode_results[].start_state` | Provider start identity used to validate per-start episode counts before remote evidence can be accepted. |
| `best_episode` | Best episode record ranked by target eval semantics. Mario uses completion, then max X, then reward; generic targets use reward. |
| `best_episode_video` | Local best-episode video path when video recording is enabled. The checkpoint artifact owns the file; the path is not emitted as a W&B metric. |
| `eval_n_envs` | Number of vector env slots used by post-train or local eval summaries. |
| `checkpoint_step` | Checkpoint step attached by post-train eval summaries. W&B receives `eval/full/checkpoint/step`. |
| `checkpoint_artifact` | Checkpoint artifact or local checkpoint ref attached by post-train eval summaries. W&B receives `eval/full/checkpoint/artifact`. |
| `model` | Local model path used by local or post-train eval summaries. |
| `policy` | Scripted policy name for scripted eval, or `ppo` for model eval. |
| `hud_crop_top` | HUD crop used for eval. W&B receives `eval/full/config/hud_crop_top` in checkpoint eval. |
| `eval_seed` | Seed used for checkpoint or local eval. Modal checkpoint eval materializes base seed `10000`; the standalone CLI default may differ within the eval-reserved `10000+` range. Train seeds are forbidden from that range. |

## W&B Config And Artifacts

The run config is not a metric, but W&B stores all train CLI args plus resolved environment
configuration fields such as `game`, `state`, `states`, `state_probs`, `task_conditioning`,
frame skip, action set, reward settings, termination settings, preprocessing settings, and
state-distribution metadata.
`state_probs` defines native state sampling weights; `state_sampling_mode=weighted` marks this path, while
`state_distribution` records the normalized probabilities derived from those weights.

Queue-backed training also records flat W&B config fields for leaderboard queries:
`goal_slug`, `recipe_slug`, `recipe_path`, `queue_train_job_id`, `runtime_image_ref`, and
`machine`. Use `rlab leaders runs` for run/recipe winners across seeds and
`rlab leaders checkpoints` for the best evaluated checkpoints by source run.

Training logs model artifacts when W&B artifacts are enabled:

| Artifact kind | When logged | Contents and metadata |
| --- | --- | --- |
| `<run>-checkpoint` | New checkpoint zip files under the run checkpoint directory | Model zip plus metadata sidecar. Aliases include `latest` and `step-<step>` when the step can be parsed. |
| `<run>-best` | In-training best model or post-train promoted best checkpoint | Model zip plus metadata. Aliases include `best`, `latest`, and sometimes `step-<step>`. |
| `<run>-final` | End of training | Final model zip plus metadata. Aliases include `final` and `latest`. |

When `--wandb-artifact-storage-uri`, `WANDB_ARTIFACT_STORAGE_URI`, or `CHECKPOINT_BUCKET_URI`
is set, the model zip is uploaded to S3/R2 and W&B stores a reference artifact instead of the
bulk model bytes.
