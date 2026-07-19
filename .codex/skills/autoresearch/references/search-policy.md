# Autoresearch Training-Signal Policy

## Objective and defaults

Select the most sample-efficient seed-stable recipe supported by durable training completion
signals. Autoresearch does not evaluate or promote checkpoints and cannot establish goal
acceptance.

Use `beast-3`, at most 48 reserved jobs, three stale search rounds, a `0.90` strong threshold, and
five untouched confirmation seeds requiring at least four strong runs. Freeze at initialization:

```text
rollout_quantum = n_steps * n_envs
screen_cap = ceil((0.20 * timesteps) / rollout_quantum) * rollout_quantum
pair_cap = ceil((0.50 * timesteps) / rollout_quantum) * rollout_quantum
full_effective_cap = ceil(timesteps / rollout_quantum) * rollout_quantum
```

Require `screen_cap < pair_cap < full_effective_cap`. Launch the confirmation recipe at its
original configured `timesteps`.

For `n_envs=N`, use screen seed `123`, paired seeds `123+N` and `123+2N`, and confirmation seeds
`123+3N` through `123+7N`. These are separate fresh runs; no rung resumes another run.

## Evidence and ranking

A screen passes only when the remotely finished W&B run shows at least one cumulative training
success for every configured start. A missing start or zero success prevents the paired rung.

For paired evidence, define a seed as strong when
`train/outcome/success/window_100/rate/min` reaches the frozen threshold at any logged
`global_step`. Missing window-100 history is not strong. Rank eligible two-seed candidates
lexicographically by:

1. more strong seeds;
2. lower median first-strong step, censoring non-strong seeds at the 50% effective cap;
3. lower worst censored first-strong step;
4. higher worst-seed peak window-100 minimum;
5. stable candidate ID only as an exact tie-break.

Candidate-ID ties never reset staleness. Use only the binary pass/fail aggregate from a failed
confirmation to exclude that candidate; do not use holdout per-seed evidence for later proposals.

## Frozen experiment contract

Never tune or change:

- goal, game, starts, provider, environment, preprocessing, task, reward, termination, evaluation
  contract, checkpoint cadence, or release semantics;
- `n_envs`, backend ID, `n_steps`, device, resume state, architecture, categorical choices, W&B,
  checkpoint publication, or artifact publication;
- source revision or the exact runtime image/input/build-source triplet.

The only study-level launch overrides outside a candidate delta are the preregistered rung
`train.timesteps`, description, and `--checkpoint-eval-backend none`. Do not persist these in the
winning leaf recipe. Fixed rung caps are separate declared training runs, not metric-driven early
stops.

Training evidence ranks recipes only. Never reinterpret it as checkpoint acceptance, promotion,
evaluation evidence, release evidence, or proof that a playable artifact satisfies the goal.

## Allowed candidate space

Tune only numeric, non-null keys already present in the composed backend config. Never introduce a
new key, move a zero baseline, change a boolean or categorical value, or change `n_steps`.

Each candidate changes one coherent group:

- learning-rate schedule: `learning_rate`, `learning_rate_final`,
  `learning_rate_schedule_timesteps`;
- entropy schedule: `ent_coef`, `ent_coef_final`, `ent_coef_schedule_timesteps`;
- discounting: `gamma`, `gae_lambda`;
- value weighting: `vf_coef`;
- PPO update: `batch_size`, `n_epochs`, `clip_range`, `target_kl`, `adam_eps`;
- A2C optimizer: existing numeric optimizer fields such as `learning_rate`,
  `learning_rate_final`, `max_grad_norm`, `rms_prop_eps`, and `vf_coef`.

Keep values within 0.25x to 4x of baseline and their semantic domains. Preserve integer types.
For PPO, require `batch_size` to divide fixed `n_steps * n_envs` and keep update work per
environment step at or below 2x baseline.

Propose at most three candidates per round, favoring a conservative lower setting, conservative
higher setting, and one local schedule or interaction alternative in the same group. Never repeat
a registered candidate.

## Capacity, recovery, and stopping

Read effective capacity immediately before each reservation and never alter it. Start independent
screen commands concurrently. Promote a passed screen into its paired rung as soon as the
controller requests it, but propose no new round until every screen and required pair in the
current round has complete remote evidence.

Every generated command uses one deterministic request key and batch. Reconcile before enqueue:

- exact cohort: record it;
- zero rows: execute once;
- partial rows or mismatched seeds, sources, overrides, runtime, or training-only backend: pause.

Operational failure, cancellation, observer failure, `potential_bug`, `attention_required`,
source drift, runtime drift, or unverified publication pauses rather than consuming a new wave.
Transient W&B evidence reads remain retryable without state mutation.

Reserve five jobs for confirmation throughout search. Stop after three stale rounds or when the
remaining budget cannot support another screen, its possible two-run pair, and confirmation. A
failed five-seed confirmation excludes the candidate, resets staleness, and continues only if the
budget supports a valid next action.

## Winner application

Require at least four of five untouched full-cap seeds to become strong. Record the result as
`training-signal-confirmed`, never accepted or promoted.

Apply in two phases: preregister the exact leaf postimage and hash while the leaf equals its pinned
preimage, then apply and recompose. Require the recomposed train configuration to equal the frozen
baseline plus only the winning numeric backend delta. Preserve all goal-owned evaluation settings
for ordinary future launches. Keep reports under `runs/autoresearch/`; only the leaf recipe patch is
source content.
