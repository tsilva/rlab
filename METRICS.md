# Metrics schema v2

This file is the human contract for rlab telemetry. The Python registry in
`src/rlab/metric_names.py` is the executable source of truth. Every emitted metric must match an
exact registry entry or a bounded template.

## Surfaces and dimensions

- W&B history contains searchable scalar time series and one `eval/full/by_start` table.
- W&B config contains run-defining dimensions: `metrics_schema_version: 2`, `algorithm_id`, goal,
  environment, starts, seed, frame skip, environment count, hyperparameters, eval protocol, and
  runtime versions.
- `leader/checkpoint/*` contains the selected checkpoint's rank values and provenance.
- SQLite and R2 retain raw episode evidence, worker state, artifacts, and videos.

The only evaluation protocols are `screen`, `confirm`, and `full`. Dimension IDs must be unique and
match `[A-Za-z0-9_.-]+`; unsafe IDs are rejected rather than silently rewritten. Starts use the same
readable ID in training and evaluation. Provider `info` fields never become metrics automatically.

An episode metric is a **return**. `reward` is reserved for per-step shaping and component
attribution. `global_step` counts policy environment transitions; frame skip remains run config.

## Research interpretation

- Use `eval/full/outcome/success/rate/min` first for multi-start success goals, then the mean, then
  `leader/checkpoint/steps_to_goal`, then episode return. `steps_to_goal` exists only when minimum
  success is at least `0.99`.
- Training `rate/current` is cumulative. `rate/window_100` is the latest 100 attempts. Global
  window-100 min/mean appear only after every configured start has 100 attempts. Always pair early
  current aggregates with `start_coverage/rate`.
- Outcome reasons may overlap, so reason counts and rates need not sum to the terminal count.
- Positive PPO policy entropy, dominant-action rate, and the action histogram diagnose discrete
  policy collapse. Value prediction and advantage histograms are sampled every 64 rollouts.
- `between_rollouts_seconds` includes PPO updates, callbacks, and logging. It is deliberately not
  named optimization time.
- Reward components are emitted only when active. Each component has mean, nonzero rate, and share;
  raw reward appears only when it differs from shaped reward.

## Full-evaluation table

`eval/full/by_start` has one row per start and observed reason with these columns:

`checkpoint_step`, `start_id`, `episodes`, `success_count`, `success_rate`, `return_mean`,
`return_std`, `return_median`, `reason`, `reason_count`, `reason_rate`.

Episode-level evidence stays in R2/SQLite. Confidence intervals and start-by-reason scalar products
are intentionally computed offline rather than added to W&B history.

## Registry

<!-- METRIC_REGISTRY_START -->
| Metric or template | Meaning | Unit | Cadence | Surface |
|---|---|---|---|---|
| `global_step` | Policy environment transitions consumed. | steps | frame | history |
| `train/episode/return/shaped/mean` | Mean shaped episode return. | scalar | rollout | history |
| `train/episode/length/mean` | Mean episode length. | steps | rollout | history |
| `train/episode/count` | Cumulative completed training episodes. | episodes | rollout | history |
| `train/outcome/terminal/count` | Cumulative terminal episode records. | episodes | rollout | history |
| `train/outcome/reason/{reason}/count` | Cumulative episodes containing an outcome reason. | episodes | rollout | history |
| `train/outcome/reason/{reason}/rate/window_100` | Reason incidence over the latest 100 terminal episodes. | fraction | rollout | history |
| `train/outcome/success/from/{start}/count` | Cumulative successful episodes from a start. | episodes | rollout | history |
| `train/outcome/success/from/{start}/attempts` | Cumulative episode attempts from a start. | episodes | rollout | history |
| `train/outcome/success/from/{start}/rate/current` | Cumulative success rate from a start. | fraction | rollout | history |
| `train/outcome/success/from/{start}/rate/window_100` | Success rate over the latest 100 attempts from a start. | fraction | rollout | history |
| `train/outcome/success/rate/current/min` | Minimum cumulative success rate across observed starts. | fraction | rollout | history |
| `train/outcome/success/rate/current/mean` | Mean cumulative success rate across observed starts. | fraction | rollout | history |
| `train/outcome/success/rate/window_100/min` | Minimum window-100 success rate after every start has 100 attempts. | fraction | rollout | history |
| `train/outcome/success/rate/window_100/mean` | Mean window-100 success rate after every start has 100 attempts. | fraction | rollout | history |
| `train/outcome/success/start_coverage/rate` | Configured starts with an attempt divided by configured starts. | fraction | rollout | history |
| `train/reward/shaped/{stat}` | Distribution of shaped per-step reward. | scalar | rollout | history |
| `train/reward/raw/{stat}` | Distribution of raw per-step reward when distinct from shaped reward. | scalar | rollout | history |
| `train/reward/component/{component}/{stat}` | Active reward-component attribution. | scalar | rollout | history |
| `train/reward/signal/{signal}/{stat}` | Configured reward-source signal. | scalar | rollout | history |
| `train/algorithm/{algorithm}/update/{metric}` | Algorithm update-health diagnostic. | scalar | rollout | history |
| `train/algorithm/{algorithm}/policy/{metric}` | Policy behavior or distribution diagnostic. | scalar | rollout | history |
| `train/algorithm/{algorithm}/value/{metric}` | Value-function diagnostic. | scalar | rollout | history |
| `train/algorithm/{algorithm}/rollout/{distribution}/{stat}` | Rollout-buffer distribution diagnostic. | scalar | rollout | history |
| `train/algorithm/{algorithm}/hyperparameter/{metric}` | Scheduled algorithm hyperparameter. | scalar | rollout | history |
| `train/algorithm/{algorithm}/advantage/{metric}` | Advantage normalization diagnostic. | scalar | rollout | history |
| `train/algorithm/{algorithm}/advantage/task/{task}/{stat}` | Per-task advantage normalization diagnostic. | scalar | rollout | history |
| `train/throughput/{metric}` | Training-loop rate or phase duration. | scalar | rollout | history |
| `train/artifact/save/seconds` | Local model save duration. | seconds | artifact | history |
| `train/artifact/upload/seconds` | External storage and W&B artifact publication duration. | seconds | artifact | history |
| `eval/{protocol}/episode/return/{stat}` | Evaluation episode-return distribution. | return | evaluation | history |
| `eval/{protocol}/episode/length/mean` | Mean evaluation episode length. | steps | evaluation | history |
| `eval/{protocol}/episode/count` | Evaluation episodes represented. | episodes | evaluation | history |
| `eval/{protocol}/outcome/success/from/{start}/rate` | Evaluation success rate from a start. | fraction | evaluation | history |
| `eval/{protocol}/outcome/success/rate/{stat}` | Aggregate per-start evaluation success rate. | fraction | evaluation | history |
| `eval/{protocol}/outcome/reason/{reason}/count` | Evaluation episodes containing a reason. | episodes | evaluation | history |
| `eval/{protocol}/outcome/reason/{reason}/rate` | Evaluation reason incidence. | fraction | evaluation | history |
| `eval/{protocol}/progress/{progress}/{stat}` | Goal-configured evaluation progress summary. | value | evaluation | history |
| `eval/{protocol}/checkpoint/{field}` | Evaluated checkpoint identity. | metadata | evaluation | history |
| `eval/{protocol}/duration/seconds` | Evaluation wall duration. | seconds | evaluation | history |
| `eval/{protocol}/source` | Evaluation execution source. | text | evaluation | history |
| `eval/{protocol}/candidate/{field}` | Staged checkpoint decision signal. | scalar | evaluation | history |
| `eval/full/by_start` | Structured full-evaluation evidence by start and reason. | table | evaluation | history |
| `leader/checkpoint/{field}` | Selected checkpoint summary field. | summary | selection | summary |
<!-- METRIC_REGISTRY_END -->
