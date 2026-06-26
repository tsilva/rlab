# Metrics Reference

This file describes the metrics this repo currently logs to Weights & Biases from the active
`src/rlab` training and evaluation paths.

Training runs call `wandb.init(..., sync_tensorboard=True)` and define `global_step` as the
step metric for all logged keys. Most scalar metrics are recorded through the SB3 logger and
synced from TensorBoard. A few callbacks also call `wandb_run.log(...)` directly for histograms,
videos, and done-count updates.

## Naming Conventions

Prefer metric paths shaped as `<phase>/<dimension>/<value-family>/<stat>`, keeping names concise
but explicit enough to search by phase, info-value, outcome, reward, or progress.

Use `train` and `eval` as the first path segment. Keep aggregate metrics at the phase level, for example
`train/done/all` and `eval/reward/mean`.

Use `rate` for fractions in `[0, 1]`, `count` for point-in-time counts, and standard stat suffixes such as
`mean`, `std`, `min`, `max`, `abs_mean`, and `nonzero_rate` only where a metric family explicitly logs
distribution statistics. Avoid aliases and alternate names for the same value.

## Selection Metrics

These are the first metrics to check when choosing policies.

| Metric | Meaning |
| --- | --- |
| `train/done/all` | Cumulative count of non-`global_reset` training `done=True` env-slot episode boundaries. This is exhaustive. |
| `train/done/<reason>` | Cumulative count of done events attributed to `<reason>`, such as `life_loss`, `level_change`, `max_steps`, or `unclassified`. Reason counters are explanatory and do not have to sum to `train/done/all`. |
| `train/done/<reason>/from/<prev>` | Cumulative count of structured done events for `<reason>` whose native payload reported previous value `<prev>`. Multi-key values are joined with `-`, e.g. `0-0`. |
| `train/done/<reason>/from/<prev>/ep_window/rate` | Fraction of the last 100 non-`global_reset` terminal training episodes whose configured source value for `<reason>` was `<prev>` that ended with that structured done event. Each `<reason>/from/<prev>` has its own 100-episode denominator and emits only after that per-source window is full. |
| `train/event/<event>` | Cumulative count of observed configured `info_events`, terminal or non-terminal. |
| `train/outcome/<event>/from/<prev>/attempts` | Cumulative attempts tracked from source value `<prev>` for `<event>`. |
| `train/outcome/<event>/from/<prev>/fires` | Cumulative attempts from source value `<prev>` where `<event>` fired. |
| `train/outcome/<event>/from/<prev>/attempt_window/rate` | Fraction of the last 100 attempts from `<prev>` where `<event>` fired. Attempts can end at a non-terminal event, terminal event, life loss, truncation, or episode done. |
| `train/outcome/<event>/from_rate/min` | Minimum across full per-source `attempt_window/rate` values for `<event>`. For balanced Mario training, use `train/outcome/level_change/from_rate/min` as the live bottleneck metric. |
| `train/outcome/<event>/from_rate/mean` | Mean across full per-source `attempt_window/rate` values for `<event>`. This is secondary; it can hide one weak level. |
| `eval/done/level_change/rate` | Pooled eval episode completion fraction. |
| `eval/done/level_change/from/<start>/rate` | Eval completion fraction for episodes that started from `<start>`. |
| `eval/done/level_change/from_rate/min` | Minimum per-start eval completion fraction. Use this first when comparing multi-start-state policies. |

Current training does not log per-rollout done-count distribution stats such as `train/done/min`,
`train/done/mean`, or `train/done/max`. The aggregate all-done counter is `train/done/all`.

### Selection and Redundancy Notes

Training outcome metrics live under `train/outcome/*` and count attempts, not full episodes. Use
them for live training bottlenecks when a policy can clear multiple levels inside one episode. Use
`train/done/*` only to understand what is ending training episodes.

`eval/done/level_change/from_rate/min` is the eval selection metric for multi-start-state
policies. The top-level eval metrics are pooled summaries and should be treated as secondary
when per-start-state eval done metrics exist.

### Mario Level1-1/Level1-2 Notes

For the current Level1-1/Level1-2 training goal, native level values map to training outcome metrics as:

| Level | Training window-rate metric |
| --- | --- |
| `Level1-1` | `train/outcome/level_change/from/0-0/attempt_window/rate` |
| `Level1-2` | `train/outcome/level_change/from/0-1/attempt_window/rate` |

Use `train/outcome/level_change/from_rate/min` as the live training bottleneck: it is the lower
of the full per-level attempt-window rates at each logging point. For example, `Level1-1 = 0.50` and
`Level1-2 = 0.40` produces a min of `0.40`, while `Level1-1 = 0.30` and `Level1-2 = 0.80`
produces a min of `0.30`. The mean is secondary because it can hide one weak level. Once
checkpoint eval jobs have logged per-start metrics, use `eval/done/level_change/from_rate/min`
as the balanced eval selection metric.

Use current `train/reward_share/*` metrics for reward attribution rather than the older
`train/reward_component/*` namespace. Shares are based on absolute rollout contribution
magnitude, so negative components such as death or time penalties are visible by magnitude
rather than canceling against positive reward.

Training info events are configured with `--info-events-json`, which maps event names to native
info-variable rules. `--done-on-events` separately chooses which configured events terminate an
episode. Legacy `--done-on-info-json` remains supported as shorthand for "observe these events and
terminate on all of them." For Mario, a typical observed event set is
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
for natural level transitions. Evaluation forces `done_on_info={}` in env construction but stops
the eval episode when it observes completion, so `eval/done/level_change` and
`eval/done/level_change/from/<start>` track natural transitions per eval episode. Eval `from` values
are the configured episode start state, not native `done_on_info` previous-value payloads.

`train/done/*` windows remain terminal-episode metrics. Natural level transitions observed while
the training env keeps running set `level_complete` / `completion_event`, emit
`info_events.level_change`, and append to `train/outcome/level_change/from/<prev>/attempt_window/rate`.
The same outcome path is used when `level_change` is also listed in `--done-on-events`; in that mode
the event both counts as a successful attempt and ends the episode.

## SB3 PPO Metrics

These come from Stable-Baselines3 PPO and `VecMonitor`.

| Metric | Meaning |
| --- | --- |
| `rollout/ep_rew_mean` | Mean shaped episode return over SB3's monitor window. This is the reward used by training, not raw game score. |
| `rollout/ep_len_mean` | Mean episode length over SB3's monitor window. |
| `time/fps` | Cumulative SB3 training throughput in environment steps per second. |
| `time/iterations` | Number of PPO learn iterations completed. |
| `time/time_elapsed` | Wall-clock seconds elapsed in the SB3 learn loop. |
| `time/total_timesteps` | Total environment steps reached by SB3 or the in-loop eval callback. |
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
| `train/value_loss` | PPO value-function loss component. |

## Throughput Metrics

| Metric | Meaning |
| --- | --- |
| `throughput/rollout_fps` | Rollout-only environment-step throughput, measured from rollout start to rollout end. This excludes PPO optimization time. |
| `throughput/loop_fps` | Full-loop instantaneous throughput, measured from one rollout start to the next. This includes rollout collection plus PPO optimization overhead. |

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

Reward share metrics compare absolute component magnitudes within a rollout:

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

These are logged by the in-training `RetroEvalCallback` when training-loop eval is enabled, and
by `rlab-eval` artifact mode when evaluating checkpoint artifacts out of process.
Evaluation env construction forces `done_on_info={}`.

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
| `eval/done/level_change` | Eval episodes that completed by natural level transition. |
| `eval/done/level_change/rate` | `eval/done/level_change / eval/done/all`. |
| `eval/done/max_steps` | Eval episodes that hit the max-step limit. |
| `eval/done/max_steps/rate` | `eval/done/max_steps / eval/done/all`. |
| `eval/done/unclassified` | Eval episodes that ended without level completion or max-step truncation. |
| `eval/done/unclassified/rate` | `eval/done/unclassified / eval/done/all`. |
| `eval/death/count` | Eval episodes where the final info indicated death. |
| `eval/death/rate` | `eval/death/count / eval episodes`. |
| `eval/death/x_hist` | W&B histogram of death X positions. Logged when death positions exist. |
| `eval/best/reward` | Return of the best eval episode, ranked by completion first, then max X, then reward. |
| `eval/best/x` | Max global X position of the best eval episode. |
| `eval/best/video` | W&B video for the best eval episode, when video recording is enabled. |
| `eval/checkpoint/step` | Checkpoint step being evaluated. Logged by `rlab-eval` artifact mode. |
| `eval/checkpoint/artifact` | W&B checkpoint artifact name being evaluated. Logged by `rlab-eval` artifact mode. |
| `eval/config/hud_crop_top` | HUD crop used for the out-of-process checkpoint eval. |

Per-start-state eval done metrics mirror the training done namespace as
`eval/done/<reason>/from/<start>`. Because eval disables `done_on_info`, `<start>` is the eval
episode start state, for example `Level1-1`, rather than a native previous-value tuple such as
`0-0`.

| Metric template | Meaning |
| --- | --- |
| `eval/done/all/from/<start>` | Number of eval episodes that started from `<start>`. This is the denominator for that start state. |
| `eval/done/level_change/from/<start>` | Eval episodes from `<start>` that completed by natural level transition. |
| `eval/done/level_change/from/<start>/rate` | `eval/done/level_change/from/<start> / eval/done/all/from/<start>`. |
| `eval/done/level_change/from_rate/min` | Minimum per-start-state level-change rate. Use this for balanced multi-state eval ranking. |
| `eval/done/level_change/from_rate/mean` | Mean per-start-state level-change rate. |
| `eval/done/max_steps/from/<start>` | Eval episodes from `<start>` that hit the max-step limit. |
| `eval/done/max_steps/from/<start>/rate` | `eval/done/max_steps/from/<start> / eval/done/all/from/<start>`. |
| `eval/done/unclassified/from/<start>` | Eval episodes from `<start>` that ended without level completion or max-step truncation. |
| `eval/done/unclassified/from/<start>/rate` | `eval/done/unclassified/from/<start> / eval/done/all/from/<start>`. |

## W&B Config And Artifacts

The run config is not a metric, but W&B stores all train CLI args plus resolved environment
configuration fields such as `game`, `state`, `states`, `state_probs`, `task_conditioning`,
frame skip, action set, reward settings, termination settings, preprocessing settings, and
state-distribution metadata.

Training logs model artifacts when W&B artifacts are enabled:

| Artifact kind | When logged | Contents and metadata |
| --- | --- | --- |
| `<run>-checkpoint` | New checkpoint zip files under the run checkpoint directory | Model zip plus metadata sidecar. Aliases include `latest` and `step-<step>` when the step can be parsed. |
| `<run>-best` | In-training best model or out-of-process promoted best checkpoint | Model zip plus metadata. Aliases include `best`, `latest`, and sometimes `step-<step>`. |
| `<run>-final` | End of training | Final model zip plus metadata. Aliases include `final` and `latest`. |

When `--wandb-artifact-storage-uri`, `WANDB_ARTIFACT_STORAGE_URI`, or `CHECKPOINT_BUCKET_URI`
is set, the model zip is uploaded to S3/R2 and W&B stores a reference artifact instead of the
bulk model bytes.
