# Metrics Reference

This file describes the metrics this repo currently logs to Weights & Biases from the active
`src/rlab` training and evaluation paths.

Training runs call `wandb.init(..., sync_tensorboard=True)` and define `global_step` as the
step metric for all logged keys. Most scalar metrics are recorded through the SB3 logger and
synced from TensorBoard. A few callbacks also call `wandb_run.log(...)` directly for histograms,
videos, and done-count updates.

## Naming Conventions

Prefer metric paths shaped as `<phase>/<dimension>/<value-family>/<stat>`, keeping names concise
but explicit enough to search by phase, info-value, reward, or progress.

Use `train` and `eval` as the first path segment. Keep aggregate metrics at the phase level, for example
`train/done/all` and `eval/reward/mean`.

Use `rate` for fractions in `[0, 1]`, `count` for point-in-time counts, and standard stat suffixes such as
`mean`, `std`, `min`, `max`, `abs_mean`, and `nonzero_rate` only where a metric family explicitly logs
distribution statistics. Avoid aliases and alternate names for the same value.

`global_step` is the W&B step axis for training metrics logged directly to W&B and for
TensorBoard-synced SB3 metrics. Post-training checkpoint eval logs `global_step` and
`eval/checkpoint/step` as the checkpoint timestep on the same W&B run that produced the checkpoint.

## Selection Metrics

These are the first metrics to check when choosing policies.

Queue-backed train recipes use `selection_metrics` as an ordered list of training
signals. Goal-owned eval settings do not define selection metrics; they only define
the checkpoint measurement protocol. Post-training checkpoint eval writes
canonical `eval/*` metrics to the producing W&B run with `global_step` set to
the checkpoint timestep. W&B is the source of truth for train and eval metrics;
the queue database stores train-job state, not result metric projections.

| Metric | Meaning |
| --- | --- |
| `train/done/all` | Cumulative count of non-`global_reset` training `done=True` env-slot episode boundaries. This is exhaustive. |
| `train/done/<reason>` | Cumulative count of done events attributed to `<reason>`, such as `life_loss`, `level_change`, `max_steps`, or `unclassified`. Reason counters are explanatory and do not have to sum to `train/done/all`. |
| `train/done/max_steps` | Cumulative count of terminal training episodes attributed to max-step truncation. Emitted as `0` before the first max-step episode. |
| `train/done/unclassified` | Cumulative count of terminal training episodes that had no configured done reason and were not max-step truncations. Emitted as `0` before the first unclassified episode. |
| `train/done/<reason>/from/<prev>` | Cumulative count of structured done events for `<reason>` whose native payload reported previous value `<prev>`. Multi-key values are joined with `-`, e.g. `0-0`. |
| `train/done/<reason>/from/<prev>/ep_window/rate` | Fraction of the last 100 non-`global_reset` terminal training episodes whose configured source value for `<reason>` was `<prev>` that ended with that structured done event. Each `<reason>/from/<prev>` has its own 100-episode denominator and emits only after that per-source window is full. |
| `train/done/<reason>/from_rate/min` | Minimum across full per-source terminal episode-window rates for `<reason>`. |
| `train/done/<reason>/from_rate/mean` | Mean across full per-source terminal episode-window rates for `<reason>`. |
| `train/info/level_complete` | Root for Mario training level-complete metrics. This is the only `info_events`-derived training metric family. |
| `train/info/level_complete/from/<prev>/count` | Cumulative clean level clears from native source value `<prev>`, e.g. `0-0` for Level1-1. Death or life-loss info always records as a failed attempt, even if an upstream completion flag is also present. |
| `train/info/level_complete/from/<prev>/attempts` | Number of attempts currently represented in the rolling attempt window for `<prev>`, capped at 100. Emits from the first observed attempt and is useful for seeing whether the strict rate window has filled. |
| `train/info/level_complete/from/<prev>/rate/current` | Fraction of the current rolling attempt window for `<prev>` that produced a clean `level_complete`, using the available denominator up to 100 attempts. This is for live monitoring, not strict promotion. |
| `train/info/level_complete/from/<prev>/rate` | Fraction of the last 100 attempts from `<prev>` that produced a clean `level_complete`. Attempts can end at a clean completion, life loss, truncation, or episode done; death/life-loss attempts contribute `0`. Emits only after that source has a full 100-attempt window. |
| `train/info/level_complete/rate/min/current` | Minimum across the latest available `train/info/level_complete/from/<prev>/rate/current` values. Emits from the first observed attempt and is intended for live charts while strict windows are still filling. |
| `train/info/level_complete/rate/mean/current` | Mean across the latest available `train/info/level_complete/from/<prev>/rate/current` values. Emits alongside `rate/min/current`. |
| `train/info/level_complete/rate/min/last` | Minimum across the latest available `train/info/level_complete/from/<prev>/rate` values. Emits after at least one per-source rate is available and updates whenever any per-source rate updates. |
| `train/info/level_complete/rate/mean/last` | Mean across the latest available `train/info/level_complete/from/<prev>/rate` values. Emits alongside `rate/min/last` and updates whenever any per-source rate updates. |
| `eval/done/level_change/rate` | Pooled eval episode completion fraction. |
| `eval/done/level_change/from/<start>/rate` | Eval completion fraction for episodes that started from `<start>`. |
| `eval/done/level_change/from_rate/min` | Minimum per-start eval completion fraction. Use this first when comparing multi-start-state policies. |
| `eval/info/level_complete/rate/min` | Eval counterpart to the train completion aggregate; alias of `eval/done/level_change/from_rate/min` for eval summaries and the preferred local early-stop metric. |
| `eval/info/level_complete/rate/mean` | Eval counterpart to the train completion aggregate; alias of `eval/done/level_change/from_rate/mean` for eval summaries. |
| `eval/info/level_complete/rate/min/last` | Backward-compatible eval alias of `eval/done/level_change/from_rate/min` for W&B charts and older reports. |
| `eval/info/level_complete/rate/mean/last` | Backward-compatible eval alias of `eval/done/level_change/from_rate/mean` for W&B charts and older reports. |

Current training does not log per-rollout done-count distribution stats such as `train/done/min`,
`train/done/mean`, or `train/done/max`. The aggregate all-done counter is `train/done/all`.

### Selection and Redundancy Notes

Training level-complete metrics live under `train/info/level_complete/*` and count attempts, not full
episodes. Use them for live training clear counts/rates when a policy can clear multiple levels
inside one episode. Use `train/done/*` only to understand what is ending training episodes.

`eval/done/level_change/from_rate/min` is the eval selection metric for Mario-style multi-start-state
policies that define a clean completion event. Environments without a target-level completion
contract, such as Breakout, use `eval/reward/mean` as the checkpoint objective instead.
`eval/info/level_complete/rate/min/last` and
`eval/info/level_complete/rate/mean/last` mirror the train `info/level_complete` aggregate names
for dashboards that compare train and eval surfaces side by side. The top-level eval metrics are
pooled summaries and should be treated as secondary when per-start-state eval done metrics exist.
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

Use `train/info/level_complete/rate/min/last` as the strict training gate and comparable
full-window bottleneck. It is the minimum of the most recent full-window per-level rates that have
emitted so far. For example, if Level1-1 is `0.50` and Level1-2 is `0.30`, `rate/min/last` is
`0.30`. If Level1-1 later drops from `1.00` to `0.50` while Level1-2 drops from `0.60` to `0.55`,
`rate/min/last` is `0.50`. Use `train/info/level_complete/rate/mean/last` as the companion average
over those same latest per-level rates, mainly to distinguish policies with the same bottleneck. Training
intentionally no longer logs generic `train/event/*` or `train/outcome/*` metrics. Once post-train
checkpoint eval has logged per-start metrics,
use `eval/done/level_change/from_rate/min` as the balanced eval selection metric.

`rate/min/last` and `rate/mean/last` are intentionally available-source live metrics. If only one
source has a full attempt window so far, both report that source's rate; as more source windows emit,
the minimum automatically tightens to include them and the mean averages across them. Use the companion
`train/info/level_complete/from/<prev>/attempts`, `/rate/current`, `/rate`, and `/count` metrics to
see which sources currently have coverage. On fast-solved single-level runs, `/rate/min/last` can
have only one W&B history point because the first full 100-attempt window immediately satisfies the
early-stop gate; use `/rate/min/current` for the live trace in that case.

`train/done/level_change/from_rate/min` and `train/done/level_change/from_rate/mean` are terminal
episode-window diagnostics. Keep them as done-reason diagnostics; for clear counts/rates, prefer
`train/info/level_complete/from/<prev>/count` and `train/info/level_complete/from/<prev>/rate`
because they also count non-terminal clears and exclude death or life-loss transitions. For a
single training selection scalar across source levels, prefer
`train/info/level_complete/rate/min/last`; use `train/info/level_complete/rate/mean/last` as a
tiebreaker/secondary signal when bottleneck rates match.

Goal contracts should not restate the `train/info/level_complete/from/<prev>/rate` 100-attempt FIFO
window as a separate success setting. The metric definition owns that window. Goal files should
declare training stop gates as a list in `train.early_stop`; multiple metric-threshold rules are
combined with AND. Ordered ranking signals stay in `objective.rank`.

Use current `train/reward_share/*` metrics for reward attribution rather than the older
`train/reward_component/*` namespace. Shares are based on absolute rollout contribution
magnitude, so negative components such as death or time penalties are visible by magnitude
rather than canceling against positive reward.

Training info events are configured with `--info-events-json`, which maps event names to native
info-variable rules. `--done-on-events` separately chooses which configured events terminate an
episode; every `--done-on-events` name must exist in `--info-events-json`. For Mario, a typical observed event set is
`{"life_loss":["lives","decrease"],"level_change":[["levelHi","levelLo"],"change"]}`. Native/default
environment terminations that do not match a configured rule and are not max-step truncations count
as `train/done/unclassified`. When native `done_on_info` payloads include `prev` and `next`,
training also emits fully softcoded previous-value counters such as
`train/done/level_change/from/0-0`. For per-source episode-window rates, successful structured
events use the native payload `prev`; terminal episodes where that reason did not fire use the
configured rule keys read from terminal `info` as the source value. For Mario
`level_change`, that means life-loss or max-step terminal episodes still count in the denominator
for their current `(levelHi, levelLo)` source level. Training intentionally does not emit `to` or
full-transition counters because those multiply metric cardinality quickly. Training does not emit
initializer-state mirrors under `train/state/<initializer>/done/*`; those labels are not reliable
for natural level transitions. Evaluation, including post-training checkpoint eval, preserves
`level_change` as a terminal event while dropping life-loss termination. This stops successful
single-level Mario eval episodes as soon as the policy clears the level, so
`eval/done/level_change` and `eval/done/level_change/from/<start>` track episodes that ended on
an observed natural transition. Eval `from` values are the configured episode start state, not
native `done_on_info` previous-value payloads.

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
- In native-vector training, clearing `done_on_events` also removes the normal life-loss/level-change
  reset boundaries. If no other provider `done=True` boundary occurs, `train/done/*` can stay absent
  while rollout reward collapses into sparse components such as `train/reward_share/death`. In that
  regime, use `train/reward/prog_x/*`, `train/reward/score/*`, and
  `train/info/level_complete/*` to confirm whether the agent is still receiving progress and clean
  clear signal.
- `train/info/level_complete/from/<prev>/count` should be less than or equal to
  `train/done/level_change/from/<prev>` when level changes are terminal. Any excess
  `train/done/level_change/from/<prev>` over clean `level_complete` count is terminal level-change
  traffic that was not accepted as a clean clear, for example a life-loss/death transition.

`level_change` is the generic stable-retro-style info event: it means configured native level
variables changed. The Mario target wrapper is responsible for deciding whether that raw transition
was actually a level clear. It sets the per-step `completion_event` / `level_complete` flag only
when the level changed without a detected death or life loss. The metrics callback then reuses the
raw `level_change` payload's `prev` value as the source, so a clean transition from `(0, 0)` to
`(0, 1)` records `train/info/level_complete/from/0-0/count`. `completion_event` is an info flag/alias
consumed by code; `level_complete` is the semantic event/result name used in W&B metrics. As a
defensive guard, the metrics callback treats any attempt with `died`, `life_loss`, or a `life_loss`
info event as failed even if a contradictory completion flag appears in the same info payload.

### ALE / Ms. Pac-Man Notes

For ALE targets such as Ms. Pac-Man, `train/done/all` is the cumulative completed training episode
count. It increments whenever a vector-env slot reports `done=True`, including natural game-over
episodes and max-frame truncations. Use `eval/done/all` for the number of episodes summarized by a
standalone or post-training eval run.

Current Ms. Pac-Man recipes do not define a clean completion event, so Mario-style completion metrics
such as `train/info/level_complete/*`, `eval/done/level_change/*`, and `completion_count` are not
expected. Selection should use reward metrics such as `eval/reward/mean` and `eval/best/reward`.

Current ALE/Ms. Pac-Man training also does not emit a cumulative life-loss metric. `train/done/life_loss`
only counts terminal episodes whose done payload is attributed to a configured `life_loss` reason; the
ALE provider does not support `done_on_events`, and the base Ms. Pac-Man goal does not enable a
separate generic info-event counter for decreases in `ale.lives`. If a run is launched with ALE
`episodic_life=true`, life loss may become a training episode boundary, but without an explicit
life-loss counter those episode boundaries still should be read through `train/done/all` and
`train/done/unclassified`, not as total lives lost.

## SB3 PPO Metrics

These come from Stable-Baselines3 PPO and `VecMonitor`.

| Metric | Meaning |
| --- | --- |
| `rollout/ep_rew_mean` | Mean shaped episode return over SB3's monitor window. This is the reward used by training, not raw game score. |
| `rollout/ep_len_mean` | Mean episode length over SB3's monitor window. |
| `time/fps` | Cumulative SB3 training throughput in environment steps per second. |
| `time/iterations` | Number of PPO learn iterations completed. |
| `time/time_elapsed` | Wall-clock seconds elapsed in the SB3 learn loop. Explicitly mirrored to W&B at rollout end with `global_step` as the step. |
| `time/total_timesteps` | Total environment steps reached by SB3 training. |
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

## Throughput Metrics

| Metric | Meaning |
| --- | --- |
| `throughput/rollout_fps` | Rollout-only environment-step throughput, measured from rollout start to rollout end. This excludes PPO optimization time. |
| `throughput/native_env_step_fps` | Provider vector-env step throughput in environment steps per second, measured only around the underlying Gymnasium vector `env.step(...)` calls during rollout collection. This excludes PPO policy inference, callbacks, rollout-buffer work, and optimizer time. |
| `throughput/native_env_step_batch_fps` | Provider vector-env batch-step throughput in vector steps per second. Multiply by `n_envs` to compare with environment-step throughput. |
| `throughput/native_env_step_seconds` | Wall-clock seconds spent inside provider vector-env `env.step(...)` calls during the rollout. |
| `throughput/native_env_step_fraction` | Fraction of rollout collection wall time spent inside provider vector-env `env.step(...)`. Values near `1` mean native env stepping dominates rollout time; lower values point to policy inference, Python wrappers, callbacks, or rollout-buffer overhead. |
| `throughput/loop_fps` | Full-loop instantaneous throughput, measured from one rollout start to the next. This includes rollout collection plus PPO optimization overhead. |

For training-speed and host-saturation comparisons, prefer `throughput/loop_fps` over total wall
time. It is the closest per-run W&B metric to effective training throughput because it includes
both environment rollout collection and PPO optimization. Use `throughput/rollout_fps` beside it to
separate rollout collection from optimizer overhead. Use `throughput/native_env_step_fps` and
`throughput/native_env_step_fraction` to isolate the provider vector-env call inside rollout
collection: if native env-step FPS drops as parallel jobs increase, the emulator/native preprocessing
path is the likely bottleneck; if native env-step FPS is high but rollout FPS is low, policy
inference, Python wrappers, callback logging, or rollout-buffer overhead is more likely; if rollout
FPS is steady but loop FPS drops, PPO optimization, GPU scheduling, or host contention outside
rollout collection is more likely. Use `time/fps` only as a cumulative SB3 sanity check, and discount
points near checkpoint/final artifact events separately from artifact-worker timing.

Do not compare `throughput/native_env_step_fps` directly with the default
`SuperMarioBros-Nes-turbo` `scripts/benchmark_sps.py` output unless the benchmark is configured to
match the training contract. The default benchmark can use the fast stepping path without per-lane
info dictionaries and without native `done_on` event reporting. Mario training uses full
`env.step(...)` with `done_on_events` such as `life_loss` and `level_change`, so the native-step
timer includes provider work for terminal/event reporting and autoreset metadata even though it
still excludes PPO policy inference, rollout-buffer writes, callbacks, and optimizer time.

## Artifact Timing Metrics

These sparse metrics are logged when the artifact worker flushes model artifacts. Checkpoint rows use
the checkpoint step as `global_step`; final and best artifacts use the model's current timestep. They
measure worker-side artifact latency, not trainer stall time.

| Metric | Meaning |
| --- | --- |
| `train/artifact/stall_seconds` | Wall-clock artifact-worker flush duration reported for compatibility with older charts. It no longer represents trainer stall time for async checkpoint/final artifacts. |
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

Logged at rollout end from reward fields in env `info` dictionaries.

`train/reward/<component>/<stat>` is logged for each component that appears during the rollout.

Components:

| Component | Source field | Meaning |
| --- | --- | --- |
| `shaped` | `shaped_reward` | Final shaped reward passed toward training. |
| `raw` | `raw_reward` | Raw environment reward before repo reward shaping. |
| `native` | `native_reward_component` | Native stable-retro reward component. |
| `prog` | `progress_component` | Generic progress component, when supplied by the env wrapper. |
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
`train/reward_share/<component>` is logged for each share component:

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
Post-training checkpoint eval preserves `level_change` as a terminal event while dropping life-loss
termination. The eval loop keeps running after observed life-loss events, but stops when the level
changes, on native env done, or at the configured max-step horizon.

| Metric | Meaning |
| --- | --- |
| `eval/reward/mean` | Mean eval episode return. |
| `eval/reward/std` | Standard deviation of eval episode returns. |
| `eval/reward/max` | Maximum eval episode return. |
| `eval/progress/x/mean` | Mean max global X position reached per eval episode. |
| `eval/progress/x/max` | Maximum global X position reached by any eval episode. |
| `eval/progress/level_x/mean` | Mean max level-local X position reached per eval episode. |
| `eval/progress/level_x/max` | Maximum level-local X position reached by any eval episode. |
| `eval/done/all` | Number of eval episodes summarized. This is exhaustive. |
| `eval/done/level_change` | Eval episodes where a clean natural level transition was observed. For current single-level Mario checkpoint evals, this event normally ends the episode. |
| `eval/done/level_change/rate` | `eval/done/level_change / eval/done/all`. |
| `eval/done/max_steps` | Eval episodes that hit the max-step limit. Should not overlap with `eval/done/level_change` for current single-level Mario checkpoint evals because level change is terminal there. |
| `eval/done/max_steps/rate` | `eval/done/max_steps / eval/done/all`. |
| `eval/done/terminated` | Eval episodes that ended via native env termination without being marked as max-step truncations. This is generic and is emitted for all target types. |
| `eval/done/terminated/rate` | `eval/done/terminated / eval/done/all`. |
| `eval/done/unclassified` | Eval episodes that ended without level completion or max-step truncation. |
| `eval/done/unclassified/rate` | `eval/done/unclassified / eval/done/all`. |
| `eval/death/count` | Eval episodes where the target-specific death flag was observed at least once during the eval horizon. Currently Mario-specific. |
| `eval/death/rate` | `eval/death/count / eval episodes`. Currently Mario-specific. |
| `eval/death/x_hist` | W&B histogram of death X positions. Logged when target semantics define death positions and positions exist. |
| `eval/best/reward` | Return of the best eval episode. Mario ranks best episodes by completion, then max X, then reward; generic targets rank by reward. |
| `eval/best/x` | Max global X position of the best eval episode. Logged only when the target defines global X progress. |
| `eval/best/video` | W&B video for the best eval episode, when video recording is enabled. |
| `eval/checkpoint/step` | Checkpoint timestep being evaluated by the async eval worker. Eval also logs this value as `global_step` so W&B panels plot the result at the evaluated model timestep without forcing W&B's internal history step backward. |
| `eval/checkpoint/artifact` | W&B checkpoint artifact name or local checkpoint ref evaluated by the async eval worker. |
| `eval/config/hud_crop_top` | HUD crop used for checkpoint eval. |
| `eval/duration/seconds` | Wall-clock seconds spent inside `evaluate_model_episodes(...)`, including episode rollout and any requested best-episode video rendering. Logged to W&B for checkpoint eval when present. |
| `leader/checkpoint/objective` | W&B summary field for the best evaluated checkpoint on a source run. Uses the target-specific primary eval objective: Mario completion bottleneck when present, otherwise `eval/reward/mean`. Used by `rlab leaders checkpoints`. |
| `leader/checkpoint/objective_name` | Metric name represented by `leader/checkpoint/objective`, such as `eval/done/level_change/from_rate/min` or `eval/reward/mean`. |
| `leader/checkpoint/completion_rate` | W&B summary field for Mario-style completion-aware runs, using `eval/done/level_change/from_rate/min` when available. Omitted for generic targets without a completion event. |
| `leader/checkpoint/completion_rate_mean` | W&B summary tiebreaker for Mario-style completion-aware runs, using `eval/done/level_change/from_rate/mean` when available. Omitted for generic targets without a completion event. |
| `leader/checkpoint/steps_to_completion_goal` | Checkpoint timestep for the source run's best evaluated checkpoint once `leader/checkpoint/completion_rate >= 0.99`. Lower is better and is used after min and mean per-start completion, before eval reward, so solved runs are ranked by sample efficiency. |
| `leader/checkpoint/reward_mean` | W&B summary tiebreaker for the source run's best evaluated checkpoint, after min and mean per-start completion and solved checkpoint timesteps. |
| `leader/checkpoint/max_x_max` | Progress field for the source run's best evaluated checkpoint. Reported for inspection but not part of the current objective rank. |
| `leader/checkpoint/step` | Checkpoint step for the source run's current best evaluated checkpoint. |
| `leader/checkpoint/artifact_ref` | Artifact ref for the source run's current best evaluated checkpoint. |
| `leader/checkpoint/local_path` | Local checkpoint path for the source run's current best evaluated checkpoint when it is available on the eval worker. |
| `leader/checkpoint/eval_source` | Source that produced the current best-checkpoint summary update, usually `async_worker`; legacy runs may show `post_train_inline`. |
| `leader/checkpoint/updated_at` | UTC timestamp when the source run's best-checkpoint summary fields were last updated. |

Per-start-state eval done metrics mirror the training done namespace as
`eval/done/<reason>/from/<start>`. Eval preserves level-change termination but labels
`<start>` with the eval episode start state, for example `Level1-1`, rather than a native
previous-value tuple such as `0-0`.

| Metric template | Meaning |
| --- | --- |
| `eval/done/all/from/<start>` | Number of eval episodes that started from `<start>`. This is the denominator for that start state. |
| `eval/done/level_change/from/<start>` | Eval episodes from `<start>` where a clean natural level transition was observed at least once during the eval horizon. |
| `eval/done/level_change/from/<start>/rate` | `eval/done/level_change/from/<start> / eval/done/all/from/<start>`. |
| `eval/done/level_change/from_rate/min` | Minimum per-start-state level-change rate. Use this for balanced multi-state eval ranking. |
| `eval/done/level_change/from_rate/mean` | Mean per-start-state level-change rate. |
| `eval/info/level_complete/rate/min` | Alias of `eval/done/level_change/from_rate/min`, named for local metric-store early stopping. |
| `eval/info/level_complete/rate/mean` | Alias of `eval/done/level_change/from_rate/mean`, named for local metric-store consumers. |
| `eval/info/level_complete/rate/min/last` | Backward-compatible alias of `eval/done/level_change/from_rate/min`, named to mirror older train `info/level_complete` reports. |
| `eval/info/level_complete/rate/mean/last` | Backward-compatible alias of `eval/done/level_change/from_rate/mean`, named to mirror older train `info/level_complete` reports. |
| `eval/done/max_steps/from/<start>` | Eval episodes from `<start>` that hit the max-step limit. Can overlap with level-change counts. |
| `eval/done/max_steps/from/<start>/rate` | `eval/done/max_steps/from/<start> / eval/done/all/from/<start>`. |
| `eval/done/terminated/from/<start>` | Eval episodes from `<start>` that ended via native env termination without max-step truncation. |
| `eval/done/terminated/from/<start>/rate` | `eval/done/terminated/from/<start> / eval/done/all/from/<start>`. |
| `eval/done/unclassified/from/<start>` | Eval episodes from `<start>` that ended without level completion or max-step truncation. |
| `eval/done/unclassified/from/<start>/rate` | `eval/done/unclassified/from/<start> / eval/done/all/from/<start>`. |

## Eval JSON Summary Fields

`rlab eval` and post-training checkpoint eval write richer JSON summaries. These fields are
stored in stdout JSON or post-train eval outputs; only the `eval/*` subset above is logged to W&B by default.

| Field | Meaning |
| --- | --- |
| `episodes` | Number of eval episodes summarized. |
| `deterministic` | Whether eval used deterministic policy actions. |
| `reward_mean` | Mean eval episode return before mapping to `eval/reward/mean`. |
| `reward_std` | Standard deviation of eval episode returns before mapping to `eval/reward/std`. |
| `reward_max` | Maximum eval episode return before mapping to `eval/reward/max`. |
| `max_x_mean` | Mean max global X position before mapping to `eval/progress/x/mean`. Emitted only when target eval semantics define global X progress. |
| `max_x_max` | Maximum global X position before mapping to `eval/progress/x/max`. Emitted only when target eval semantics define global X progress. |
| `max_level_x_mean` | Mean max level-local X position before mapping to `eval/progress/level_x/mean`. Emitted only when target eval semantics define level-local X progress. |
| `max_level_x_max` | Maximum level-local X position before mapping to `eval/progress/level_x/max`. Emitted only when target eval semantics define level-local X progress. |
| `completion_count` | Eval episodes where a target-specific clean completion event was observed at least once. Currently emitted for Mario-style targets only. |
| `completion_rate` | `completion_count / episodes`. Same rate as `eval/done/level_change/rate` for Mario-style targets. |
| `death_count` | Eval episodes where target-specific death was observed at least once. Currently emitted for Mario-style targets only. |
| `death_rate` | `death_count / episodes`. Same rate as `eval/death/rate` when death semantics exist. |
| `terminated_count` | Eval episodes that hit native env done without being marked as max-step truncations. Completion alone does not increment this. |
| `terminated_rate` | `terminated_count / episodes`. |
| `truncated_count` | Eval episodes that hit the max-step limit. Same count as `eval/done/max_steps`; can overlap with `completion_count`. |
| `truncated_rate` | `truncated_count / episodes`. Same rate as `eval/done/max_steps/rate`. |
| `unclassified_count` | Eval episodes that ended without level completion or max-step truncation. Same count as `eval/done/unclassified`. |
| `unclassified_rate` | `unclassified_count / episodes`. Same rate as `eval/done/unclassified/rate`. |
| `death_x_histogram` | Local JSON histogram of death X positions. W&B receives `eval/death/x_hist` when death positions exist. |
| `episode_results` | Per-episode records used to build the summary. Removed from stdout when `--summary-only` is set. |
| `best_episode` | Best episode record ranked by target eval semantics. Mario uses completion, then max X, then reward; generic targets use reward. |
| `best_model_score` | Checkpoint eval ranking tuple. Completion-aware targets use completion, solved checkpoint timesteps when available, then reward; generic targets use reward. |
| `best_episode_video` | Local best-episode video path when video recording is enabled. W&B receives `eval/best/video`. |
| `timesteps` | Training timestep attached by legacy eval summaries when present. New checkpoint eval summaries use `checkpoint_step`. |
| `eval_n_envs` | Number of vector env slots used by post-train or local eval summaries. |
| `checkpoint_step` | Checkpoint step attached by post-train eval summaries. W&B receives `eval/checkpoint/step`. |
| `checkpoint_artifact` | Checkpoint artifact or local checkpoint ref attached by post-train eval summaries. W&B receives `eval/checkpoint/artifact`. |
| `model` | Local model path used by local or post-train eval summaries. |
| `policy` | Scripted policy name for scripted eval, or `ppo` for model eval. |
| `hud_crop_top` | HUD crop used for eval. W&B receives `eval/config/hud_crop_top` in checkpoint eval. |
| `eval_seed` | Seed used for checkpoint or local eval. Default eval runs use `10007` in the eval-reserved `10000+` range; train seeds are forbidden from that range. |

## W&B Config And Artifacts

The run config is not a metric, but W&B stores all train CLI args plus resolved environment
configuration fields such as `game`, `state`, `states`, `state_probs`, `task_conditioning`,
frame skip, action set, reward settings, termination settings, preprocessing settings, and
state-distribution metadata.
For post7+ Stable Retro Turbo, `state_probs` is the legacy field name for native
state sampling weights; `state_sampling_mode=weighted` marks this path, while
`state_distribution` records the normalized probabilities derived from those weights.

Queue-backed training also records flat W&B config fields for leaderboard queries:
`goal_slug`, `recipe_slug`, `recipe_path`, `queue_train_job_id`, `runtime_image_ref`, and
`run_target`. Use `rlab leaders runs` for run/recipe winners across seeds and
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
