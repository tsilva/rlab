# Autoresearch Search Policy

## Objective and defaults

The winner satisfies the immutable goal contract at the earliest authoritative promoted checkpoint and reproduces that success across training seeds. Optimize lexicographically:

1. more paired search seeds that are `accepted` and remotely verified;
2. lower median censored promoted environment step;
3. lower worst censored promoted environment step;
4. stable candidate ID only as an exact deterministic tie-break.

Defaults are `beast-3`, at most 48 reserved queue jobs, paired search seeds based at 123, three consecutive stale rounds, and five untouched confirmation seeds requiring at least four accepted-and-remotely-verified runs.

The provider `n_envs` spaces all seed bases so vector lanes never overlap. For `n_envs=N`, search uses `123` and `123+N`; confirmation uses the next five bases `123+2N` through `123+6N`.

## Frozen experiment contract

Never tune or override:

- goal, game, state, provider, environment, preprocessing, task, reward, termination, or evaluation environment;
- evaluation episodes, acceptance, seed manifest, backend, checkpoint cadence, promotion, or early-stop semantics;
- training timesteps, `n_envs`, backend ID, `n_steps`, device, resume state, policy/model architecture, categorical choices, W&B/logging, or artifact publication;
- source revision or the exact runtime image/input/build-source triplet.

Use the goal-owned acceptance evaluation and its promoted checkpoint step as the sole success and sample-efficiency authority. Training-return or first-training-success metrics are never promotion evidence.

The failure censor is frozen at initialization:

```text
ceil(timesteps / (n_steps * n_envs)) * n_steps * n_envs
```

Do not recompute it per candidate.

## Allowed candidate space

Tune only numeric, non-null keys already present in the composed backend config. Never introduce a new key, move a zero baseline, change a boolean/categorical value, or change `n_steps`.

Every candidate changes one coherent group:

- learning-rate schedule: `learning_rate`, `learning_rate_final`, `learning_rate_schedule_timesteps`;
- entropy schedule: `ent_coef`, `ent_coef_final`, `ent_coef_schedule_timesteps`;
- discounting: `gamma`, `gae_lambda`;
- value weighting: `vf_coef`;
- PPO update: `batch_size`, `n_epochs`, `clip_range`, `target_kl`, `adam_eps`;
- A2C optimizer: existing numeric optimizer fields such as `learning_rate`, `learning_rate_final`, `max_grad_norm`, `rms_prop_eps`, and `vf_coef`.

Each value remains within 0.25x to 4x of baseline and within its semantic domain. Integers stay integers. For PPO, `batch_size` divides fixed `n_steps * n_envs`, and update work per environment step stays at or below 2x baseline. Do not change more than one group to rescue an invalid candidate.

Prefer a small local experimental design around the incumbent:

- one conservative lower setting;
- one conservative higher setting;
- one schedule/interaction alternative inside the same group when capacity permits.

Use prior paired search evidence to choose the next group or local direction. Never use untouched confirmation outcomes beyond the binary pass/fail exclusion.

Candidate identity is the hash of the normalized effective backend delta. It excludes recipe ID, description, campaign ID, run name, request ID, and other trace-only metadata. Never repeat a candidate already registered in the study.

## Capacity and concurrency

Read effective capacity and active reservations immediately before every wave. Do not alter capacity or existing work.

For search, preregister up to three candidate cohorts of two seeds each. Choose enough whole paired cohorts to fill current free slots; an odd free slot may cause one member of a pair to queue. Execute all independent candidate launch commands concurrently. Start one independent follow stream per returned run. Use a strict round barrier: no next wave until every prior run is terminal, every incident investigator returned, and no attention or observer failure remains.

Confirmation is exactly five concurrent jobs. Deliberately leave a sixth beast-3 slot for unrelated work when hard capacity is six.

Reservations happen before enqueue and remain charged when enqueue outcome is uncertain. Search reservations always leave five of the 48 jobs unused for confirmation.

## Recovery and stopping

The deterministic request key maps to one deterministic `bx<16 hex>` batch ID using the same repository helper as queue insertion. On resume, inspect the derived batch before enqueue:

- exact complete cohort: reconcile it into study state;
- zero rows: execute the reserved launch once;
- partial rows, changed request hash, wrong seeds, wrong source/recipe/goal, wrong overrides, or wrong runtime: pause.

Stop after three consecutive search rounds that do not improve accepted count, median censored step, or worst censored step. Candidate-ID tie-breaks never reset staleness. Operational failure, cancellation, observer failure, `potential_bug`, `attention_required`, source drift, runtime drift, or ambiguous study discovery pauses instead of consuming a new wave.

The five-seed holdout passes with at least four accepted-and-remotely-verified runs. On failure, exclude the candidate using only that binary result, reset staleness, recompute the incumbent from search evidence, and continue only if the fixed budget supports the next action. At budget exhaustion, report no winner and leave the recipe untouched.

## Winner application

Use a two-phase apply:

1. store exact leaf postimage, SHA-256, and diff in `study.json` while the leaf equals its pinned preimage;
2. apply that exact postimage, then require full recomposition to equal the frozen baseline train config plus only the winning backend delta.

All non-leaf composed sources must still match their pinned hashes. An unexpected leaf state is a pause. The baseline may validly win as a no-op. Keep all reports under `runs/autoresearch/`; only the leaf recipe patch is a source change.
