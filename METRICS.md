# Metrics schema v6

This file is the human contract for rlab telemetry. The Python registry in
`src/rlab/metric_names.py` is the executable source of truth. Every emitted metric must match an
exact registry entry or a bounded template.

## Surfaces and dimensions

- W&B is the authoritative scientific metric surface. One supervisor process inside the training
  container is the only process allowed to open and write the logical W&B run.
- The learner writes structured events only to its embedded SQLite WAL outbox. It performs no
  network I/O for metrics, checkpoint publication, or evaluation dispatch.
- Modal never receives W&B credentials. The supervisor validates Modal results and appends their
  metrics to the same W&B run.
- SQLite and private-R2 JSONL metric segments are delivery and recovery transports, not competing
  scientific metric stores. Verified terminal journals expire after seven days.
- Public model R2 contains immutable checkpoint closures and a mutable no-cache run index. Private
  eval R2 contains intents, results, and episode evidence. Private control R2 contains leases,
  journals, promotions, and terminal receipts.
- W&B config contains run-defining dimensions: `metrics_schema_version: 6`, `training_backend_id`,
  `training_backend_config_hash`, `algorithm_id`, goal,
  environment, starts, seed, frame skip, environment count, hyperparameters, eval protocol, and
  runtime versions.
- `goal_contract_sha256` is the semantic SHA-256 of the fully composed, rendered, validated goal
  contract. Generated goal reports use it with `goal_slug` to keep current-contract leaderboards
  comparable; runs without the current fingerprint remain historical evidence only.
- Catalog-backed runs also record `effective_goal_contract_sha256`, `reward_program_kind`,
  `reward_program_revision`, `reward_shape`, `reward_shape_sha256`, and
  `reward_shape_is_default`. Reward-derived returns are comparable only when the selected reward
  semantic identity and effective goal contract match; the readable key alone is not sufficient.
- `leader/checkpoint/*` contains diagnostic projections of the selected checkpoint. The
  create-only private-R2 `PromotionReceipt` is the authoritative selection.
- Heavy model bytes, videos, replays, episode rows, diagnostics, and recovery payloads never go to
  W&B.

The active checkpoint protocol is `acceptance`; complete accepted evidence additionally emits the
`full` metric family. Historical `screen` and `confirm` rows remain readable but are not produced by
new goal contracts. Dimension IDs must be unique and match `[A-Za-z0-9_.-]+`; unsafe IDs are rejected
rather than silently rewritten. Starts use the same readable ID in training and evaluation. Provider
`info` fields never become metrics automatically.

Configuration-selected internal learner feedback, such as a snapshot curriculum's per-start
priority statistic, is not telemetry merely because it has a readable name. Internal feedback
identifiers do not use metric paths and are not published to W&B unless a separately registered
metric explicitly projects them. If projected, the emitted name and semantics must appear in the
registry below. For SB3 PPO and A2C, snapshot-curriculum `priority_metric: value_error` specifically
means the arithmetic mean of `abs(A_t)` over one completed snapshot-origin trajectory, where `A_t`
is raw GAE before PPO minibatch normalization. That scalar updates the archive's cell-level EMA and
is intentionally not emitted; `train/curriculum/snapshot/feedback/trajectory/count` reports only how
many such trajectory updates were committed.

An episode metric is a **return**. `reward` is reserved for per-step shaping and component
attribution. Frame skip remains run config. W&B uses three explicit axes:

- `train/global_step`: policy environment transitions consumed by training.
- `eval/checkpoint_step`: step of the checkpoint represented by an evaluation row.
- `orchestration/event_seq`: durable supervisor delivery order.

Asynchronous evaluations may arrive after later training rows without changing their scientific
X-axis. The compatibility `global_step` field may remain in payloads while surviving producers are
migrated, but new charts and orchestration logic use the three names above.

Purged PostgreSQL/Fleet/W&B/R2 state has no compatibility guarantee. Newly materialized runs declare
schema v6.

## Research interpretation

- Mario ranks checkpoints only after acceptance: earliest `leader/checkpoint/step`, then highest
  `eval/full/episode/return/mean`. Breakout is training-only and ranks current-contract seeded recipe
  cohorts using `train/episode/return/shaped/from/target/mean`, which excludes snapshot-curriculum
  origins and non-episode control boundaries; tied cohorts prefer fewer policy transitions.
- Aggregate training `current/rate/*` is cumulative. Aggregate `window_100/rate/*` is the latest
  100 attempts. Global
  window-100 min/mean appear only after every configured start has 100 attempts. Always pair early
  current aggregates with `start_coverage/rate`.
- A bounded training-only search may use per-start success counts and the history peak and first
  threshold crossing of `train/outcome/success/window_100/rate/min` to screen and rank recipes.
  That evidence is not checkpoint evaluation and cannot establish checkpoint promotion, goal
  acceptance, or release evidence.
- Failure reasons may overlap, so reason counts and rates need not sum to the terminal count.
  Successful episodes contribute to success metrics, not the failure-reason families.
- Positive PPO policy entropy, dominant-action rate, and the action histogram diagnose discrete
  policy collapse. Value prediction and advantage histograms are sampled every 64 rollouts.
- Snapshot-curriculum `sampling/probability/max` and `sampling/effective_cell/count` summarize the
  current cell-probability distribution. They do not report realized per-cell selection frequency
  or identify which resident cells were selected.
- Derived throughput phase timing satisfies `loop wall time = env_step_seconds +
  rollout_overhead_seconds + between_rollouts_seconds`. Compare those three phase durations on
  matching workloads to identify a training-loop bottleneck. `rollout_overhead_seconds` includes
  policy inference plus wrapper, buffer, reset, task, and callback work outside the native provider.
  `between_rollouts_seconds` includes optimizer updates, callbacks, and logging, so it is deliberately
  not named optimization time.
- Reward components are emitted only when active. Each component has mean, nonzero rate, and share;
  raw reward appears only when it differs from shaped reward.

## Full-evaluation table

`eval/full/by_start` has one row per start and observed failure reason with these columns:

`checkpoint_step`, `start_id`, `episodes`, `success_count`, `success_rate`, `return_mean`,
`return_std`, `return_median`, `reason`, `reason_count`, `reason_rate`.
The reason value is empty, with zero count and rate, when a start has no recorded failure reason.

Episode-level evidence stays in R2. Confidence intervals and start-by-reason scalar products
are intentionally computed offline rather than added to W&B history.

An acceptance rejection is complete evidence of failure, but not a complete 100-episode
evaluation. W&B history always receives `global_step`, pass, planned/completed episodes, and
acceptance duration. It receives no partial `eval/full/*` result. Accepted projections additionally
include variable return, length, progress, episode count, artifact, source, and `eval/full/by_start`.
Constant acceptance success rates, per-start success scalars, failure-reason scalars, duplicate full
duration, and constant leader-success fields remain in private-R2 evidence but are suppressed from
acceptance W&B history.

`eval/acceptance/pass` is per-checkpoint history. W&B summarizes that history with `max`, so the
summary means that some checkpoint passed; it is not the run verdict. The authoritative verdict is
the create-only private-R2 `PromotionReceipt`, whose selected result is hash-bound to the complete
acceptance evidence. At terminal publication, that receipt restamps `rlab/goal/outcome`, the diagnostic
`leader/checkpoint/*` fields, and the accepted W&B projection. Later rejected checkpoint projections
remain in history and never modify the active projection. Raw acceptance aggregates and episode
evidence remain authoritative in private eval R2.

## Delivery, backpressure, and recovery

Every event has a stable content-derived event ID. Delivery to W&B is at least once; the durable
`orchestration/event_seq` is also W&B's internal step, so replay after an interrupted local
acknowledgement cannot append a second scientific point. Reports and summary projection retain the
event ID for explicit deduplication. Promotion, terminal state, and early-stop authority are exactly
once through conditional private-R2 receipts.

The supervisor seals immutable metric-journal segments to private R2 every five seconds or 1,000
events and batches pending frames to W&B. A retry reconstructs its local SQLite state from those
segments before producing new events. It resumes the same W&B run with `resume="must"`.

Backpressure is sampled every 15 seconds. The supervisor reports queue depth, oldest unpublished
age, ingress and publish rates, observed publication-capacity ratio, local/R2/W&B high-water marks,
remote-visible W&B lag, checkpoint backlog, pending evals, scratch utilization, accepted-result to
stop latency, and post-learner idle-GPU tail. Publication capacity is healthy only when measured
publish capacity is at least twice peak ingress.

Unpublished W&B age warns at 45 seconds and is unhealthy at 60 seconds. Terminal W&B drain has a
300-second deadline. If neither W&B nor private R2 can preserve pending metrics, or task scratch
usage reaches 80%, the supervisor requests a safe learner stop and emits a resumable failure rather
than discarding evidence.

A logical run succeeds only when its private-R2 `TerminalReceipt` proves complete checkpoint and
evaluation inventories, a promotion, the W&B high-water mark, and a complete drain. dstack process
exit alone is never scientific success.

## Registry

<!-- METRIC_REGISTRY_START -->
| Metric or template | Meaning | Unit | Cadence | Surface |
|---|---|---|---|---|
| `global_step` | Policy environment transitions consumed. | steps | frame | history |
| `train/episode/return/shaped/mean` | Rolling mean shaped return over the latest 100 genuine completed training episodes across target and snapshot origins; a snapshot-origin return starts at restoration, and control boundaries are excluded. | scalar | rollout | history |
| `train/episode/length/mean` | Rolling mean length over the latest 100 completed training episodes. | steps | rollout | history |
| `train/outcome/terminal/count` | Cumulative terminal episode records. | episodes | rollout | history |
| `train/outcome/reason/{reason}/count` | Cumulative failed episodes containing a reason. | episodes | rollout | history |
| `train/outcome/reason/{reason}/rate/window_100` | Failure-reason incidence over the latest 100 terminal episodes. | fraction | rollout | history |
| `train/outcome/success/from/{start}/count` | Cumulative successful episodes from a start. | episodes | rollout | history |
| `train/outcome/success/from/{start}/attempts` | Cumulative episode attempts from a start. | episodes | rollout | history |
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
| `train/throughput/loop_fps` | Policy transitions divided by rollout-start-to-next-rollout-start wall time. | steps/second | rollout | history |
| `train/throughput/rollout_fps` | Policy transitions divided by rollout-collection wall time. | steps/second | rollout | history |
| `train/throughput/env_step_fps` | Policy transitions divided by native-provider step wall time accumulated during the rollout. | steps/second | rollout | history |
| `train/throughput/rollout_seconds` | Wall time spent collecting one rollout. | seconds | rollout | history |
| `train/throughput/env_step_seconds` | Native-provider step wall time accumulated while collecting one rollout. | seconds | rollout | history |
| `train/throughput/rollout_overhead_seconds` | Rollout wall time outside native-provider step calls, including policy inference and wrapper, buffer, reset, task, and callback work. | seconds | rollout | history |
| `train/throughput/between_rollouts_seconds` | Wall time after rollout collection and before the next rollout, including optimizer updates, callbacks, and logging. | seconds | rollout | history |
| `train/artifact/save/seconds` | Local model save duration. | seconds | artifact | history |
| `train/artifact/upload/seconds` | Public R2 checkpoint publication duration. | seconds | artifact | history |
| `eval/{protocol}/episode/return/mean` | Evaluation episode-return distribution. | return | evaluation | history |
| `eval/{protocol}/episode/return/std` | Evaluation episode-return distribution. | return | evaluation | history |
| `eval/{protocol}/episode/return/median` | Evaluation episode-return distribution. | return | evaluation | history |
| `eval/full/episode/return/best` | Best full-evaluation episode return. | return | evaluation | history |
| `eval/{protocol}/episode/length/mean` | Mean evaluation episode length. | steps | evaluation | history |
| `eval/{protocol}/episode/count` | Evaluation episodes represented. | episodes | evaluation | history |
| `eval/{protocol}/outcome/success/from/{start}/rate` | Evaluation success rate from a start. | fraction | evaluation | history |
| `eval/{protocol}/outcome/success/rate/min` | Aggregate per-start evaluation success rate. | fraction | evaluation | history |
| `eval/{protocol}/outcome/success/rate/mean` | Aggregate per-start evaluation success rate. | fraction | evaluation | history |
| `eval/{protocol}/outcome/reason/{reason}/rate` | Evaluation failure-reason incidence. | fraction | evaluation | history |
| `eval/full/progress/{progress}/mean` | Goal-configured full-evaluation progress summary. | value | evaluation | history |
| `eval/full/progress/{progress}/max` | Goal-configured full-evaluation progress summary. | value | evaluation | history |
| `eval/{protocol}/checkpoint/artifact` | Evaluated checkpoint artifact reference. | metadata | evaluation | history |
| `eval/{protocol}/duration/seconds` | Evaluation wall duration. | seconds | evaluation | history |
| `eval/{protocol}/source` | Evaluation execution source. | text | evaluation | history |
| `eval/acceptance/pass` | Per-checkpoint acceptance result; W&B summarizes its history with max, not as the verdict. | boolean | acceptance evaluation | history |
| `eval/acceptance/episodes/planned` | Exact episode identities required by the acceptance manifest. | episodes | acceptance evaluation | history |
| `eval/acceptance/episodes/completed` | Valid planned episode rows completed before acceptance or fail-fast rejection. | episodes | acceptance evaluation | history |
| `eval/acceptance/failure/count` | Failed planned episodes; zero for acceptance and one for fail-fast rejection. | episodes | acceptance evaluation | history |
| `eval/acceptance/duration/seconds` | Acceptance-worker evaluation wall duration. | seconds | acceptance evaluation | history |
| `eval/full/by_start` | Structured full-evaluation evidence by start and reason. | table | evaluation | history |
| `leader/checkpoint/acceptance_pass` | Diagnostic projection of the promoted-checkpoint verdict; `eval_scope_exact` is authoritative. | boolean | selection | summary |
| `leader/checkpoint/success_rate_min` | Diagnostic selected-checkpoint projection. | summary | selection | summary |
| `leader/checkpoint/success_rate_mean` | Diagnostic selected-checkpoint projection. | summary | selection | summary |
| `leader/checkpoint/objective` | Diagnostic selected-checkpoint projection; never a ranking input. | summary | selection | summary |
| `leader/checkpoint/return_mean` | Diagnostic selected-checkpoint projection. | summary | selection | summary |
| `leader/checkpoint/best_return` | Diagnostic selected-checkpoint projection. | summary | selection | summary |
| `leader/checkpoint/rank_values` | Diagnostic rendering of authoritative rank values. | summary | selection | summary |
| `leader/checkpoint/progress_max` | Diagnostic selected-checkpoint projection. | summary | selection | summary |
| `leader/checkpoint/step` | Diagnostic selected-checkpoint projection. | summary | selection | summary |
| `leader/checkpoint/artifact_ref` | Diagnostic immutable artifact reference projection. | summary | selection | summary |
| `leader/checkpoint/eval_source` | Diagnostic evaluation-source projection. | summary | selection | summary |
| `leader/checkpoint/updated_at` | Diagnostic projection update time. | summary | selection | summary |
| `train/global_step` | Scientific training X-axis: policy environment transitions consumed. | steps | frame | history |
| `eval/checkpoint_step` | Scientific evaluation X-axis: step of the evaluated checkpoint. | steps | evaluation | history |
| `orchestration/event_seq` | Monotonic local outbox event sequence used as W&B delivery order. | events | frame | history |
| `orchestration/event_id` | Stable content-derived identifier used to deduplicate at-least-once delivery. | metadata | frame | history |
| `orchestration/outbox/queue_depth` | Metric outbox frames not yet acknowledged by the W&B SDK. | events | supervisor sample | history |
| `orchestration/outbox/oldest_unpublished_seconds` | Age of the oldest metric frame not yet acknowledged by the W&B SDK. | seconds | supervisor sample | history |
| `orchestration/outbox/ingress_rate` | Observed local metric-frame creation rate over the latest supervisor interval. | events/second | supervisor sample | history |
| `orchestration/outbox/publish_rate` | Observed W&B SDK acknowledgment rate over the latest supervisor interval. | events/second | supervisor sample | history |
| `orchestration/outbox/publication_capacity_ratio` | Observed W&B publication rate divided by observed metric ingress rate. | ratio | supervisor sample | history |
| `orchestration/outbox/local_high_water` | Largest metric-frame sequence durably present in local SQLite. | events | supervisor sample | history |
| `orchestration/outbox/r2_high_water` | Largest metric-frame sequence sealed in immutable private R2 journals. | events | supervisor sample | history |
| `orchestration/outbox/wandb_high_water` | Largest metric-frame sequence acknowledged by the W&B SDK. | events | supervisor sample | history |
| `orchestration/outbox/wandb_remote_high_water` | Largest orchestration event sequence observed through the W&B API. | events | remote visibility probe | history |
| `orchestration/outbox/wandb_remote_visible_lag_seconds` | Age of the newest local metric event not yet observed through the W&B API. | seconds | remote visibility probe | history |
| `orchestration/checkpoint/backlog` | Ready local checkpoints not yet verified in public model R2. | checkpoints | supervisor sample | history |
| `orchestration/eval/pending` | Persisted evaluation intents without a terminal verified result. | evaluations | supervisor sample | history |
| `orchestration/eval/result_to_stop_seconds` | Time from observing an accepted eval result to signaling the learner. | seconds | accepted evaluation | history |
| `orchestration/drain/idle_gpu_tail_seconds` | Time the training container retained its GPU after the learner exited. | seconds | terminal drain | history |
| `orchestration/scratch/used_fraction` | Fraction of the task scratch filesystem currently used. | fraction | supervisor sample | history |
| `train/episode/return/shaped/from/target/mean` | Rolling mean shaped return over the latest 100 genuine target-origin training episodes. | return | rollout | history |
| `train/curriculum/snapshot/archive/cell/count` | Current snapshot archive cell count. | cells | rollout | history |
| `train/curriculum/snapshot/archive/snapshot/count` | Current resident snapshot handle count. | snapshots | rollout | history |
| `train/curriculum/snapshot/admission/candidate/count` | Non-terminal cell-crossing candidates observed during the rollout. | transitions | rollout | history |
| `train/curriculum/snapshot/admission/accepted/count` | Snapshot candidates accepted into cell reservoirs during the rollout. | snapshots | rollout | history |
| `train/curriculum/snapshot/archive/evicted/count` | Archive cells evicted during the rollout. | cells | rollout | history |
| `train/curriculum/snapshot/capture/call/count` | Batched provider snapshot-capture calls during the rollout. | calls | rollout | history |
| `train/curriculum/snapshot/reset/episode/count` | Snapshot-origin episodes started during the rollout. | episodes | rollout | history |
| `train/curriculum/snapshot/reset/forced_boundary/count` | Non-episode control truncations used to activate snapshot lanes. | boundaries | rollout | history |
| `train/curriculum/snapshot/feedback/trajectory/count` | Completed snapshot-origin trajectories committed to the priority sampler. | trajectories | rollout | history |
| `train/curriculum/snapshot/transition/share` | Fraction of policy transitions whose origin is the snapshot curriculum. | fraction | rollout | history |
| `train/curriculum/snapshot/sampling/probability/max` | Largest final cell probability in the snapshot sampler. | fraction | rollout | history |
| `train/curriculum/snapshot/sampling/effective_cell/count` | Inverse-Simpson effective cell count of the snapshot sampling distribution. | cells | rollout | history |
| `train/curriculum/snapshot/capture/seconds` | Provider snapshot-capture wall time accumulated during the rollout. | seconds | rollout | history |
| `train/curriculum/snapshot/reset/seconds` | Provider reset wall time for reset calls containing snapshot lanes. | seconds | rollout | history |
<!-- METRIC_REGISTRY_END -->
