# Metrics schema v4

This file is the human contract for rlab telemetry. The Python registry in
`src/rlab/metric_names.py` is the executable source of truth. Every emitted metric must match an
exact registry entry or a bounded template.

## Surfaces and dimensions

- W&B history contains searchable scalar time series, one `eval/full/by_start` table, and the
  R2-backed `eval/screen/preview` media series.
- Metric producers write to a local SQLite outbox, then deliver gzip batches to the transient Neon
  mailbox. SQLite rows are deleted after Neon acknowledgement and the local database is removed
  after a successful final-flush barrier. Neon batches are deleted after confirmed W&B publication;
  neither buffer is permanent metric history.
- Fleet is the sole W&B writer for new `neon_mailbox_v1` runs. Training and evaluation workers never
  receive W&B credentials. Fleet preassigns the immutable W&B run id, publishes all streams to that
  run, and stores per-stream commit cursors in the W&B summary before deleting mailbox batches.
- W&B config contains run-defining dimensions: `metrics_schema_version: 4`, `training_backend_id`,
  `training_backend_config_hash`, `algorithm_id`, goal,
  environment, starts, seed, frame skip, environment count, hyperparameters, eval protocol, and
  runtime versions.
- `goal_contract_sha256` is the semantic SHA-256 of the fully composed, rendered, validated goal
  contract. Generated goal reports use it with `goal_slug` to keep current-contract leaderboards
  comparable; runs without the current fingerprint remain historical evidence only.
- `leader/checkpoint/*` contains the selected checkpoint's rank values and provenance.
- W&B is the permanent metric history. R2 is the permanent byte store for checkpoints, metadata,
  raw episode evidence, and videos. Postgres retains orchestration state and artifact locations.

The only evaluation protocols are `screen`, `confirm`, and `full`. Dimension IDs must be unique and
match `[A-Za-z0-9_.-]+`; unsafe IDs are rejected rather than silently rewritten. Starts use the same
readable ID in training and evaluation. Provider `info` fields never become metrics automatically.

An episode metric is a **return**. `reward` is reserved for per-step shaping and component
attribution. `global_step` counts policy environment transitions; frame skip remains run config.
Fleet never supplies W&B's internal `_step`; W&B assigns it in arrival order. It is not an
environment-transition or checkpoint count. Charts must use explicit `global_step` for the x-axis,
and evaluation rows also carry `eval/{protocol}/checkpoint/step` for the evaluated checkpoint.
Asynchronous evaluation rows may arrive after later training rows without overwriting them.

## Research interpretation

- Use `eval/full/outcome/success/rate/min` first for multi-start success goals, then the mean, then
  `leader/checkpoint/steps_to_goal`, then episode return. `steps_to_goal` exists only when minimum
  success is at least `0.99`.
- Aggregate training `current/rate/*` is cumulative. Aggregate `window_100/rate/*` is the latest
  100 attempts. Global
  window-100 min/mean appear only after every configured start has 100 attempts. Always pair early
  current aggregates with `start_coverage/rate`.
- Failure reasons may overlap, so reason counts and rates need not sum to the terminal count.
  Successful episodes contribute to success metrics, not the failure-reason families.
- Positive PPO policy entropy, dominant-action rate, and the action histogram diagnose discrete
  policy collapse. Value prediction and advantage histograms are sampled every 64 rollouts.
- `between_rollouts_seconds` includes PPO updates, callbacks, and logging. It is deliberately not
  named optimization time.
- Reward components are emitted only when active. Each component has mean, nonzero rate, and share;
  raw reward appears only when it differs from shaped reward.

## Full-evaluation table

`eval/full/by_start` has one row per start and observed failure reason with these columns:

`checkpoint_step`, `start_id`, `episodes`, `success_count`, `success_rate`, `return_mean`,
`return_std`, `return_median`, `reason`, `reason_count`, `reason_rate`.
The reason value is empty, with zero count and rate, when a start has no recorded failure reason.

Episode-level evidence stays in R2. Confidence intervals and start-by-reason scalar products
are intentionally computed offline rather than added to W&B history.

## Registry

<!-- METRIC_REGISTRY_START -->
| Metric or template | Meaning | Unit | Cadence | Surface |
|---|---|---|---|---|
| `global_step` | Policy environment transitions consumed. | steps | frame | history |
| `train/episode/return/shaped/mean` | Rolling mean shaped return over the latest 100 completed training episodes. | scalar | rollout | history |
| `train/episode/length/mean` | Rolling mean length over the latest 100 completed training episodes. | steps | rollout | history |
| `train/episode/count` | Cumulative completed training episodes. | episodes | rollout | history |
| `train/outcome/terminal/count` | Cumulative terminal episode records. | episodes | rollout | history |
| `train/outcome/reason/{reason}/count` | Cumulative failed episodes containing a reason. | episodes | rollout | history |
| `train/outcome/reason/{reason}/rate/window_100` | Failure-reason incidence over the latest 100 terminal episodes. | fraction | rollout | history |
| `train/outcome/success/from/{start}/count` | Cumulative successful episodes from a start. | episodes | rollout | history |
| `train/outcome/success/from/{start}/attempts` | Cumulative episode attempts from a start. | episodes | rollout | history |
| `train/outcome/success/from/{start}/rate/current` | Cumulative success rate from a start. | fraction | rollout | history |
| `train/outcome/success/from/{start}/rate/window_100` | Success rate over the latest 100 attempts from a start. | fraction | rollout | history |
| `train/outcome/success/current/rate/min` | Minimum cumulative success rate across observed starts. | fraction | rollout | history |
| `train/outcome/success/current/rate/mean` | Mean cumulative success rate across observed starts. | fraction | rollout | history |
| `train/outcome/success/window_100/rate/min` | Minimum window-100 success rate after every start has 100 attempts. | fraction | rollout | history |
| `train/outcome/success/window_100/rate/mean` | Mean window-100 success rate after every start has 100 attempts. | fraction | rollout | history |
| `train/outcome/success/start_coverage/rate` | Configured starts with an attempt divided by configured starts. | fraction | rollout | history |
| `train/reward/shaped/mean` | Distribution of shaped per-step reward. | scalar | rollout | history |
| `train/reward/shaped/std` | Distribution of shaped per-step reward. | scalar | rollout | history |
| `train/reward/shaped/min` | Distribution of shaped per-step reward. | scalar | rollout | history |
| `train/reward/shaped/max` | Distribution of shaped per-step reward. | scalar | rollout | history |
| `train/reward/shaped/nonzero_rate` | Distribution of shaped per-step reward. | scalar | rollout | history |
| `train/reward/raw/mean` | Distribution of raw per-step reward when distinct from shaped reward. | scalar | rollout | history |
| `train/reward/raw/std` | Distribution of raw per-step reward when distinct from shaped reward. | scalar | rollout | history |
| `train/reward/component/{component}/mean` | Active reward-component attribution. | scalar | rollout | history |
| `train/reward/component/{component}/nonzero_rate` | Active reward-component attribution. | scalar | rollout | history |
| `train/reward/component/{component}/share` | Active reward-component attribution. | scalar | rollout | history |
| `train/reward/signal/{signal}/mean` | Configured reward-source signal. | scalar | rollout | history |
| `train/reward/signal/{signal}/max` | Configured reward-source signal. | scalar | rollout | history |
| `train/reward/signal/{signal}/nonzero_rate` | Configured reward-source signal. | scalar | rollout | history |
| `train/algorithm/ppo/update/approx_kl` | Approximate KL divergence for the PPO update. | scalar | rollout | history |
| `train/algorithm/ppo/update/clip_fraction` | Fraction of policy ratios clipped by PPO. | scalar | rollout | history |
| `train/algorithm/jerk/retained/count` | Distinct action sequences retained by JERK search. | sequences | rollout | history |
| `train/algorithm/jerk/best/return_mean` | Mean observed return of JERK's highest-ranked retained sequence. | return | rollout | history |
| `train/algorithm/jerk/best/sequence_length` | Action length of JERK's highest-ranked retained sequence. | steps | rollout | history |
| `train/algorithm/jerk/archive/selected_prefix_return_mean` | Cumulative mean retained-prefix return selected for JERK archive replay. | return | rollout | history |
| `train/algorithm/jerk/exploit/probability` | Probability that JERK starts an episode by sampling a retained archive sequence. | fraction | rollout | history |
| `train/algorithm/{algorithm}/value/explained_variance` | Actor-critic value-function explained variance. | scalar | rollout | history |
| `train/algorithm/{algorithm}/update/policy_gradient_loss` | Actor-critic policy-gradient loss. | scalar | rollout | history |
| `train/algorithm/{algorithm}/update/value_loss` | Actor-critic value loss. | scalar | rollout | history |
| `train/algorithm/{algorithm}/update/learning_rate` | Current actor-critic learning rate. | scalar | rollout | history |
| `train/algorithm/{algorithm}/policy/entropy` | Positive actor-critic policy entropy. | scalar | rollout | history |
| `train/algorithm/{algorithm}/policy/distribution_std` | Continuous-action distribution standard deviation. | scalar | rollout | history |
| `train/algorithm/{algorithm}/policy/dominant_action_rate` | Fraction assigned to the most frequent sampled discrete action. | scalar | rollout | history |
| `train/algorithm/{algorithm}/policy/action_hist` | Sampled discrete-action histogram. | histogram | every 64 rollouts | history |
| `train/algorithm/{algorithm}/rollout/value_prediction/mean` | Rollout value-prediction distribution diagnostic. | scalar | rollout | history |
| `train/algorithm/{algorithm}/rollout/value_prediction/std` | Rollout value-prediction distribution diagnostic. | scalar | rollout | history |
| `train/algorithm/{algorithm}/rollout/value_prediction/min` | Rollout value-prediction distribution diagnostic. | scalar | rollout | history |
| `train/algorithm/{algorithm}/rollout/value_prediction/max` | Rollout value-prediction distribution diagnostic. | scalar | rollout | history |
| `train/algorithm/{algorithm}/rollout/value_prediction/hist` | Rollout value-prediction histogram. | histogram | every 64 rollouts | history |
| `train/algorithm/{algorithm}/rollout/advantage/mean` | Rollout advantage distribution diagnostic. | scalar | rollout | history |
| `train/algorithm/{algorithm}/rollout/advantage/std` | Rollout advantage distribution diagnostic. | scalar | rollout | history |
| `train/algorithm/{algorithm}/rollout/advantage/min` | Rollout advantage distribution diagnostic. | scalar | rollout | history |
| `train/algorithm/{algorithm}/rollout/advantage/max` | Rollout advantage distribution diagnostic. | scalar | rollout | history |
| `train/algorithm/{algorithm}/rollout/advantage/hist` | Rollout advantage histogram. | histogram | every 64 rollouts | history |
| `train/algorithm/{algorithm}/hyperparameter/entropy_coefficient` | Current scheduled entropy coefficient. | scalar | rollout | history |
| `train/throughput/loop_fps` | Training-loop rate or phase duration. | steps/second | rollout | history |
| `train/throughput/rollout_fps` | Training-loop rate or phase duration. | steps/second | rollout | history |
| `train/throughput/env_step_fps` | Training-loop rate or phase duration. | steps/second | rollout | history |
| `train/throughput/loop_seconds` | Training-loop rate or phase duration. | seconds | rollout | history |
| `train/throughput/rollout_seconds` | Training-loop rate or phase duration. | seconds | rollout | history |
| `train/throughput/env_step_seconds` | Training-loop rate or phase duration. | seconds | rollout | history |
| `train/throughput/rollout_overhead_seconds` | Training-loop rate or phase duration. | seconds | rollout | history |
| `train/throughput/between_rollouts_seconds` | Training-loop rate or phase duration. | seconds | rollout | history |
| `train/artifact/save/seconds` | Local model save duration. | seconds | artifact | history |
| `train/artifact/upload/seconds` | External storage and W&B artifact publication duration. | seconds | artifact | history |
| `eval/{protocol}/episode/return/mean` | Evaluation episode-return distribution. | return | evaluation | history |
| `eval/{protocol}/episode/return/std` | Evaluation episode-return distribution. | return | evaluation | history |
| `eval/{protocol}/episode/return/median` | Evaluation episode-return distribution. | return | evaluation | history |
| `eval/full/episode/return/best` | Best full-evaluation episode return. | return | evaluation | history |
| `eval/{protocol}/episode/length/mean` | Mean evaluation episode length. | steps | evaluation | history |
| `eval/{protocol}/episode/count` | Evaluation episodes represented. | episodes | evaluation | history |
| `eval/{protocol}/outcome/success/from/{start}/rate` | Evaluation success rate from a start. | fraction | evaluation | history |
| `eval/{protocol}/outcome/success/rate/min` | Aggregate per-start evaluation success rate. | fraction | evaluation | history |
| `eval/{protocol}/outcome/success/rate/mean` | Aggregate per-start evaluation success rate. | fraction | evaluation | history |
| `eval/{protocol}/outcome/reason/{reason}/count` | Failed evaluation episodes containing a reason. | episodes | evaluation | history |
| `eval/{protocol}/outcome/reason/{reason}/rate` | Evaluation failure-reason incidence. | fraction | evaluation | history |
| `eval/full/progress/{progress}/mean` | Goal-configured full-evaluation progress summary. | value | evaluation | history |
| `eval/full/progress/{progress}/max` | Goal-configured full-evaluation progress summary. | value | evaluation | history |
| `eval/{protocol}/checkpoint/step` | Evaluated checkpoint step. | steps | evaluation | history |
| `eval/{protocol}/checkpoint/artifact` | Evaluated checkpoint artifact reference. | metadata | evaluation | history |
| `eval/{protocol}/duration/seconds` | Evaluation wall duration. | seconds | evaluation | history |
| `eval/{protocol}/source` | Evaluation execution source. | text | evaluation | history |
| `eval/screen/preview` | External HTML player for the canonical R2 MP4 captured from policy observations during every normal queue-backed screen evaluation. | html | evaluation | media |
| `eval/screen/candidate/pass` | Staged checkpoint pass signal. | boolean | evaluation | history |
| `eval/confirm/candidate/pass` | Staged checkpoint pass signal. | boolean | evaluation | history |
| `eval/screen/candidate/stage_index` | Staged checkpoint protocol index. | index | evaluation | history |
| `eval/confirm/candidate/stage_index` | Staged checkpoint protocol index. | index | evaluation | history |
| `eval/confirm/candidate/checkpoint_step` | Confirmed candidate checkpoint step. | steps | evaluation | history |
| `eval/confirm/candidate/episodes` | Confirmed candidate evaluation episodes. | episodes | evaluation | history |
| `eval/full/by_start` | Structured full-evaluation evidence by start and reason. | table | evaluation | history |
| `leader/checkpoint/success_rate_min` | Selected checkpoint summary field. | summary | selection | summary |
| `leader/checkpoint/success_rate_mean` | Selected checkpoint summary field. | summary | selection | summary |
| `leader/checkpoint/objective` | Selected checkpoint summary field. | summary | selection | summary |
| `leader/checkpoint/objective_name` | Selected checkpoint summary field. | summary | selection | summary |
| `leader/checkpoint/return_mean` | Selected checkpoint summary field. | summary | selection | summary |
| `leader/checkpoint/best_return` | Selected checkpoint summary field. | summary | selection | summary |
| `leader/checkpoint/rank` | Selected checkpoint summary field. | summary | selection | summary |
| `leader/checkpoint/rank_values` | Selected checkpoint summary field. | summary | selection | summary |
| `leader/checkpoint/progress_max` | Selected checkpoint summary field. | summary | selection | summary |
| `leader/checkpoint/step` | Selected checkpoint summary field. | summary | selection | summary |
| `leader/checkpoint/steps_to_goal` | Selected checkpoint summary field. | summary | selection | summary |
| `leader/checkpoint/artifact_ref` | Selected checkpoint summary field. | summary | selection | summary |
| `leader/checkpoint/local_path` | Selected checkpoint summary field. | summary | selection | summary |
| `leader/checkpoint/eval_source` | Selected checkpoint summary field. | summary | selection | summary |
| `leader/checkpoint/updated_at` | Selected checkpoint summary field. | summary | selection | summary |
<!-- METRIC_REGISTRY_END -->
