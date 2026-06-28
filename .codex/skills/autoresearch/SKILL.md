---
name: autoresearch
description: Goal-driven autonomous RL experiment workflow for rlab. Use when the user asks Codex to research, launch, monitor, iterate, and find a reproducible model-training recipe for a checked-in goal under experiments/goals. The workflow is constrained to the goal contract, RTX4090 queue runners by default, legal reward/hyperparameter changes, out-of-process eval, and goal-defined confirmation before declaring success.
---

# Autoresearch

## Contract

Autoresearch is tied to achieving a repo goal. Before launching or changing
training, resolve exactly one goal capsule under `experiments/goals/<goal-slug>/`
and read its `goal.json`. The goal contract owns the game, states, target
metric, success threshold/window, cap, runtime, seed protocol, selection policy,
promotion rules, and current incumbent evidence.

Do not run open-ended research outside a goal. If the user gives only a game and
target, first map it to an existing active goal. If no matching goal exists,
create or propose a goal capsule before launching work; if the target is
ambiguous, ask for the missing contract detail instead of inventing it.

Do not declare success from a single run, cherry-picked seed, local-only metric,
or unverifiable W&B summary. Success means the frozen recipe satisfies the
goal's confirmation protocol and promotion policy. The number of confirmation
seeds comes from `goal.json`, not from this skill.

## Required Orientation

- Read `AGENTS.md`, `INSTANCES.md`, `METRICS.md`, and the selected
  `experiments/goals/<goal-slug>/goal.json` before planning specs, hardware,
  launch shape, cleanup, metrics, or promotion criteria.
- List goal capsules with `find experiments/goals -name goal.json | sort` when
  the user does not provide an exact goal slug.
- Read the goal's `README.md`, `recipes/`, `specs/`, `reports/`, and
  `decisions/` if present. They are the durable research record for that goal.
- Use `rlab-queue status --goal <goal>` to orient against the current queue
  state before launching more work.
- If adding or changing game states, verify imported state names from the live
  Stable Retro data before writing specs.
- If a metric's meaning is unclear or changes, update `METRICS.md` in the same
  change before relying on it.

## Hard Constraints

- Read `AGENTS.md` and `INSTANCES.md` before planning hardware, launch shape, cleanup, or concurrency.
- The goal contract wins over this skill. Do not change the target definition,
  success metric, cap, seed protocol, runtime, or promotion criteria unless the
  user explicitly asks to revise the goal.
- Use only queue-backed training jobs. The current repo workflow is
  `rlab-queue` for train/eval jobs and Mac-side `rlab-fleet` for Docker
  runner capacity on the beast hosts. Do not use provider launchers, manual host
  shell trainers, `sky exec`, or ad hoc direct Docker training for comparable
  runs.
- Use the selected goal's `execution.primary_train_target` when present. The
  default decisive path is RTX4090 queue-backed `rlab-fleet` capacity on
  `beast-3`. If the required target is unavailable, do not silently fall back to
  another GPU or local training; report the blocker.
- Keep queue-backed train jobs profileless by default. Use
  `--runtime-image-ref-file rlab-train-image.json`, the spec's `run_target`, and
  immutable digest-pinned runtime images unless the user explicitly requests a
  profile-locked lane.
- Do not reward hack. The reported target must reflect the intended task, not a proxy exploit or trivial stop condition.
- Do not change the environment, emulator state, ROM, action semantics, observation wrapper semantics, termination semantics, target definition, or evaluation protocol unless the user explicitly approves a broader research scope.
- Allowed levers by default: reward function design and hyperparameters only.
- If the evidence indicates success requires levers beyond reward design or hyperparameters, stop and tell the user which extra lever is needed and why.
- Preserve Python dependency hardening and use `uv sync --frozen` for repo dependency setup.
- Keep generated artifacts in ignored experiment locations such as `runs/`, `logs/`, W&B, or R2/S3 artifacts unless the user asks to promote a reusable file.
- Store durable experiment specs under
  `experiments/goals/<goal-slug>/specs/`. Store durable recipe evidence,
  reports, and decisions under that same goal folder. Raw checkpoints, W&B
  downloads, videos, and logs stay out of source control.

## Research Loop

1. Resolve the goal.
   - Identify the goal slug and read `goal.json`.
   - Restate the goal from the contract: game, states, primary metric, training stop metric, success threshold/window, max timesteps, selection policy, seed protocol, runtime, incumbent, and disallowed shortcuts.
   - If the user asks for "the current goal" and multiple active goals exist, list the active goals and ask for the goal slug unless one is clearly implied by the conversation.
   - If no checked-in goal matches the requested target, create or propose the goal capsule first. Do not launch unowned research.

2. Establish live baseline and queue state.
   - Locate the current training entrypoints, config files, reward functions, eval/playback tools, and queue/fleet patterns in the repo.
   - Check whether the target can be measured directly from existing metrics. If not, add measurement without changing environment behavior.
   - Search current code, goal recipes/specs/reports/decisions, W&B run names/configs, local logs, `METRICS.md`, `INSTANCES.md`, and queue status.
   - Prefer existing best-known recipes as the first baseline unless they conflict with the task.
   - Record exact package/runtime versions, seed, reward definition, hyperparameters, stop criteria, and artifact paths.
   - Treat training metrics as diagnostic unless the goal selection policy explicitly says otherwise; promotion should come from out-of-process eval.

3. Plan a goal-owned queue batch.
   - Use the goal's `search_protocol`, `seed_protocol`, `cap_policy`, `runtime`, `default_train_spec_file`, and `capacity_policy_file`.
   - For a new candidate, write a checked-in spec file under `experiments/goals/<goal-slug>/specs/` with a specific hypothesis, expected signal, stage, seeds, run target, W&B group/tags, run name template, run description template, selection gate, and `train_config`.
   - Use run names shaped like the repo convention: `<batch>_<scope>_<arm>_s<seed>_<utc>`.
   - Validate the spec with `rlab.job_queue.load_spec_document` before enqueueing. New launches should use `info_events_json` plus `done_on_events`; do not introduce legacy event keys.
   - Re-check live fleet capacity and the policy lane for the goal target. Use `INSTANCES.md`/capacity-policy defaults: usually 5 `beast-3` workers with `env_threads=4` for screening, and 3-4 workers with `env_threads=4` for lower-contention confirmation when latency/debuggability matters.
   - Use W&B logging and artifact upload unless explicitly opted out.
   - Include a specific run description for every run.

4. Enqueue and reconcile through the current repo workflow.
   - Add/update the checked-in spec, then enqueue jobs from it:
     `UV_CACHE_DIR=.uv-cache uv run rlab-queue enqueue-train --spec-file <spec-path> --runtime-image-ref-file rlab-train-image.json`
   - Inspect and reconcile capacity:
     `UV_CACHE_DIR=.uv-cache uv run rlab-fleet policy`
     `UV_CACHE_DIR=.uv-cache uv run rlab-fleet plan`
     `UV_CACHE_DIR=.uv-cache uv run rlab-fleet reconcile`
   - Use `rlab-fleet watch` only when a long-running reconciliation loop is appropriate.

5. Iterate only legal levers.
   - Reward function changes must be documented as hypotheses about real task progress. Include what failure mode they address and why they are not reward hacks.
   - Hyperparameter changes must be isolated enough to learn from the result: learning rate, schedule, PPO clip range, target KL, entropy/value coefficients, rollout length, batch size, epochs, total timesteps, env count, seed set, and early-stop criteria are valid candidates.
   - Do not mix many unrelated changes unless the run is explicitly labeled as exploratory and followed by narrower ablations.
   - If a promising next step needs a forbidden lever, stop and explain the required goal-scope change instead of folding it into a spec.

6. Monitor and analyze.
   - Track the goal's primary metric and training stop metric first, then the metrics in `selection_policy.rank_order`, then reward, task-specific progress, policy entropy, approximate KL, clip fraction, explained variance, fps, and crash/error logs.
   - Use `rlab-queue status --goal <goal>`, `rlab-monitor --view all`, W&B, and `rlab-fleet status/ps/plan` as complementary surfaces. Reconcile disagreements instead of trusting only one surface.
   - Explain what failed runs teach before launching the next batch.
   - Rank and promote using the goal's `selection_policy.rank_order`, not generic reward or final checkpoint quality.
   - Record meaningful decisions as durable summaries under the goal's `decisions/` or `reports/` directory.

7. Evaluate and confirm reproducibility.
   - Queue out-of-process evals for candidate checkpoints using the goal's `default_eval_profile` and selection policy. Do not run robust evals inside remote training by default.
   - Freeze the candidate recipe: code diff, reward definition, hyperparameters, versions, launch shape, stop/eval criteria, and artifact behavior.
   - Run the confirmation seeds specified by `goal.json`; use fresh seeds when the goal protocol asks for them.
   - Declare success only if the confirmation protocol meets the goal threshold under the frozen recipe.
   - If one or more seeds fail, treat the recipe as not solved. Analyze variance and continue iterating or report the remaining blocker.
   - When promoting, update or add a goal-owned recipe note under `recipes/` with the exact evidence, runtime, seed set, W&B group/run IDs, eval results, and playback path.

## Reporting

When updating the user, keep the main thread compact:

- Goal slug and contract metric being pursued.
- Current hypothesis and why it is legal under the constraints.
- Batch launched or monitored, including target, seeds, spec path, and run names.
- Best evidence so far, including W&B links or local paths.
- Whether the candidate is discovery-only, under confirmation, or confirmed by the goal protocol.
- Exact playback command at the end for Mario PPO or similar playable RL runs.
