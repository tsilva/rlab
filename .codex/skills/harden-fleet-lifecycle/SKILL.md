---
name: harden-fleet-lifecycle
description: Start and execute a persistent repair-mode goal that stress-tests the complete rlab queue, fleet, training, checkpoint-evaluation, W&B publication, artifact, playback, and cleanup lifecycle. Use when the user explicitly asks to harden, stress-test, repeatedly exercise, or fix the fleet experiment lifecycle until it is reliable, especially with bounded SuperMarioBros-NES PPO runs across Level1-1 through Level1-4.
---

# Harden Fleet Lifecycle

Orchestrate repeated production-path experiments and repair every supported operational defect until the completion gate passes. Use `$launch-experiment` in repair mode for every launch and follow cycle.

## Dispatch or resume the persistent goal

Start a goal only when the user explicitly asks to run or start this skill. If the user asks what the skill does, answer without creating a goal or mutating anything.

1. Inspect the current persisted goal when goal tools are available.
2. If this skill's goal is already active, do not create another goal; continue it.
3. If a different unfinished goal is active, do not replace it. Report the conflict and ask the user to clear or finish it.
4. Otherwise create a persistent goal with this objective and no token budget:

```text
Use $harden-fleet-lifecycle in repair mode to make the rlab fleet experiment lifecycle reliable. Run the staged beast-3 SuperMarioBros-NES PPO Level1-1 through Level1-4 matrix at train.timesteps=2000000, preserve failed-run evidence, fix supported root causes with regression coverage, and launch fresh verification attempts. Continue until three consecutive clean four-level matrix cycles satisfy the skill's completion contract, or until a protected action or external-state change requires user direction. Do not confuse valid research rejection with operational failure.
```

5. Immediately continue executing the goal after creating it. Do not treat goal creation as completion.

If persisted-goal tools are unavailable, give the user the exact `/goal` form of the objective above and stop without launching.

## Authority boundaries

Explicitly starting this skill authorizes:

- bounded experiment launches on `beast-3`;
- read-only inspection of Git, PostgreSQL projections, fleet health, containers, W&B state, and durable evidence;
- in-scope repository code, configuration, and test repairs after preserving failure evidence;
- targeted, broader affected, configuration, and deterministic fault-injection tests.

It does not authorize committing or pushing changes, editing `SPECS.md`, weakening goal or acceptance semantics, changing credentials or budgets, deleting durable evidence, canceling unrelated jobs, destructive cleanup, or reloading/replacing/installing/restarting controllers. Ask once for the specific missing authority when verification genuinely requires one of these actions. Explain that remote `--from-head` verification may require a focused commit and pushed branch when the repair is not already present in committed remote source.

Preserve unrelated working-tree changes. If a repair overlaps them and cannot be isolated safely, stop and request direction.

## Fixed experiment contract

Read `SPECS.md`, `INSTANCES.md`, `METRICS.md`, the four goal files, the recipe, and `$launch-experiment` before the first launch:

```text
experiments/goals/SuperMarioBros-Nes-v0/Level1-1/_goal.yaml
experiments/goals/SuperMarioBros-Nes-v0/Level1-2/_goal.yaml
experiments/goals/SuperMarioBros-Nes-v0/Level1-3/_goal.yaml
experiments/goals/SuperMarioBros-Nes-v0/Level1-4/_goal.yaml
experiments/recipes/mario/single/ppo.yaml
```

Use these fixed launch settings unless the user explicitly changes them:

- machine: `beast-3`;
- recipe override: `train.timesteps=2000000`;
- checkpoint evaluation: goal-owned Modal backend; never replace it with `none` or local fallback;
- one shared seed across the four goals in each cycle, starting at training seed 123 and incrementing once per matrix cycle while staying within `rlab.seeds` training range; never reset the seed when the clean streak resets;
- one immutable batch and run identity per submission;
- committed `HEAD` through `rlab experiment launch --from-head`;
- online W&B logging, checkpoint/final artifacts, and normal goal-owned acceptance semantics.

Never change the goal contract, environment contract, evaluation episodes, checkpoint cadence, acceptance threshold, or PPO hyperparameters merely to make the lifecycle pass. A valid `goal_rejected` terminal classification is healthy research execution when all required evaluation, publication, evidence, and cleanup state is consistent.

## Preflight

Before each launch wave:

1. Inspect `git status` and record the committed source SHA and branch. Do not include unrelated local changes.
2. Run `uv run --frozen rlab validate --json` and require `ok: true`.
3. Compose all four goal/recipe pairs with `train.timesteps=2000000`; require each resolved document to retain 2,000,000 timesteps, the goal-owned checkpoint cadence, and Modal evaluation.
4. Read fleet service health, active runs, reservations, capacity, evaluation state, and current Modal budget without mutating them.
5. Do not launch into a drained/degraded target, insufficient capacity, incompatible controllers, unavailable exact-source runtime, or a correctly enforced budget block. Diagnose or wait according to the owning contract.
6. Confirm the planned wave cannot exceed beast-3's registered capacity or affect unrelated active work.

## Staged stress loop

### Gate run

Start with one Level1-1 run at the fixed contract. Apply the entire `$launch-experiment` repair-mode workflow and follow it through a terminal event. Do not start the four-level matrix until this gate has no operational failure or unresolved potential bug.

### Matrix cycle

Launch Level1-1, Level1-2, Level1-3, and Level1-4 as four distinct submissions close enough to overlap on `beast-3`. Start one persistent `rlab experiment follow --run <id> --jsonl` session per run and poll all sessions round-robin under `$launch-experiment` rules.

For every cycle verify:

- four durable queue rows with distinct batch, run, and W&B identities;
- exact machine, committed source SHA, resolved configuration, and immutable runtime provenance;
- no duplicate claim, launch, container, evaluator, publisher, or reservation owner;
- truthful readiness and terminal projections;
- checkpoint creation and goal-owned Modal evaluation scheduling;
- fail-fast rejection is not retried, while operational attempts remain distinct and traceable;
- W&B history and summaries finalize under the exact run identity;
- final/checkpoint artifacts and evaluation evidence remain durable and unambiguous;
- emitted playback commands reference immutable artifacts where required;
- exited containers, reservations, mailbox work, and obsolete undemanded runtime images are reconciled without touching active or demanded resources.

### Repair cycle

On any pre-queue failure, `potential_bug`, `operational_failure`, observer failure, inconsistent terminal state, publication failure, leaked resource, or failed remote verification:

1. Stop launching new waves but keep all existing monitors attached.
2. Preserve exact symptoms, identifiers, logs, projections, and durable evidence.
3. Let `$launch-experiment` dispatch the read-only training-run investigator for run-backed potential bugs.
4. Wait for every affected run's terminal event and every investigator report.
5. Establish a supported root cause; never blindly retry.
6. Reproduce narrowly, implement the root-cause repair, and add deterministic regression or fault-injection coverage.
7. If metric emission or semantics changed, reconcile `METRICS.md`. If hardware, access, capacity, cleanup, or benchmark facts changed, reconcile `INSTANCES.md`. Do not edit `SPECS.md` without explicit permission.
8. Run targeted tests, the broader affected queue/fleet/eval/publication suite, and `uv run --frozen rlab validate --json`.
9. Obtain commit/push or controller authority if remote verification requires it.
10. Reset the consecutive-clean-cycle count to zero and submit fresh runs with new identities. Never mutate or reuse the failed attempt.

Do not intentionally crash live controllers, workers, networks, or databases. Exercise destructive recovery paths through deterministic tests unless the user separately authorizes a bounded live fault-injection experiment.

## Completion contract

Count a matrix cycle as clean only when all four runs:

- emit terminal events without unresolved potential bugs or operational failures;
- have internally consistent queue, launch, training, evaluation, publication, artifact, and cleanup state;
- preserve exact source, runtime, goal, recipe, override, seed, target, and identity provenance;
- leave no leaked containers, reservations, mailbox work, or incorrectly retained runtime images.

Require three consecutive clean matrix cycles after the most recent repair. Research outcomes may be accepted, completed, or validly rejected; do not require the PPO policy to solve every level as proof of fleet correctness.

Before marking the goal complete, require:

1. all relevant automated and configuration validation to pass;
2. one representative immutable artifact playback command to be executed successfully when a qualifying artifact exists and local playback is available;
3. fleet service health to be current and non-degraded;
4. no active hardening runs or unresolved investigators;
5. a concise report of every failure fingerprint, root cause, repair, regression test, commit/source state, batch and run ID, W&B URL, terminal classification, artifact, playback verification, clean-cycle evidence, residual risk, and exact rerun command.

Mark the persisted goal complete only after every completion item is satisfied. If a required protected action or external-state change blocks progress, keep the goal active until the product's blocked threshold is met; then mark it blocked with the exact needed action.
