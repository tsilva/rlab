---
name: autoresearch
description: Autonomous RL experiment workflow for rlab. Use when the user gives Codex a game and target outcome and asks it to research, launch, monitor, iterate, and find a reproducible model-training recipe that accomplishes the target. The workflow is constrained to RTX4090-only SkyPilot runs, legal reward/hyperparameter changes, and three-seed confirmation before declaring success.
---

# Autoresearch

## Contract

Given a game and target, search for a training recipe that reaches the target and is reproducible across three different fresh seeds.

Do not declare success from a single run, cherry-picked seed, local-only metric, or unverifiable W&B summary. Success requires a recipe that meets the target when run again with three distinct seeds under the same declared recipe.

## Hard Constraints

- Read `AGENTS.md` and `INSTANCES.md` before planning hardware, launch shape, cleanup, or concurrency.
- Use only the RTX4090 target from `INSTANCES.md`: SkyPilot infra `k8s/rtx4090`. If it is unavailable, do not use another GPU or local fallback. Report that the required target is unavailable and stop launching.
- Before any remote launch, estimate expected cost and get user approval when project instructions require it.
- Do not reward hack. The reported target must reflect the intended task, not a proxy exploit or trivial stop condition.
- Do not change the environment, emulator state, ROM, action semantics, observation wrapper semantics, termination semantics, target definition, or evaluation protocol unless the user explicitly approves a broader research scope.
- Allowed levers by default: reward function design and hyperparameters only.
- If the evidence indicates success requires levers beyond reward design or hyperparameters, stop and tell the user which extra lever is needed and why.
- Preserve Python dependency hardening and use `uv sync --frozen` for repo dependency setup.
- Keep generated artifacts in ignored experiment locations such as `runs/`, `logs/`, W&B, R2/S3 artifacts, or ignored `sky_*.yaml` files unless the user asks to promote a reusable launch file.

## Research Loop

1. Define the task precisely.
   - Restate the game, target, success metric, eval method, and disallowed shortcuts.
   - Locate the current training entrypoints, config files, reward functions, eval/playback tools, and launch YAML patterns in the repo.
   - Check whether the target can be measured directly from existing metrics. If not, add measurement without changing environment behavior.

2. Establish a baseline from live evidence.
   - Search current code, W&B run names/configs, local logs, `GOAL.md`, and `INSTANCES.md`.
   - Prefer existing best-known recipes as the first baseline unless they conflict with the task.
   - Record exact package/runtime versions, seed, reward definition, hyperparameters, stop criteria, and artifact paths.

3. Plan an RTX4090-only batch.
   - Re-check live availability for `k8s/rtx4090`.
   - Use `INSTANCES.md` defaults: 5 concurrent children with `env_threads=4` for screening; 3-4 children with `env_threads=4` for lower-contention confirmation.
   - Use W&B logging and artifact upload unless explicitly opted out.
   - Include a specific `--run-description` or launch-level `run_description` for every run.

4. Iterate only legal levers.
   - Reward function changes must be documented as hypotheses about real task progress. Include what failure mode they address and why they are not reward hacks.
   - Hyperparameter changes must be isolated enough to learn from the result: learning rate, schedule, PPO clip range, target KL, entropy/value coefficients, rollout length, batch size, epochs, total timesteps, env count, seed set, and early-stop criteria are valid candidates.
   - Do not mix many unrelated changes unless the run is explicitly labeled as exploratory and followed by narrower ablations.

5. Monitor and analyze.
   - Track completion/progress metrics first, then reward, x-position or equivalent progress, policy entropy, approximate KL, clip fraction, explained variance, fps, and crash/error logs.
   - Explain what failed runs teach before launching the next batch.
   - Prefer promotion by target completion rate, then task-specific max progress, then mean reward, unless the user defines another ranking.

6. Confirm reproducibility.
   - Freeze the candidate recipe: code diff, reward definition, hyperparameters, versions, launch shape, stop/eval criteria, and artifact behavior.
   - Run three fresh seeds not used to discover the recipe.
   - Declare success only if all three confirmation seeds meet the target under the frozen recipe.
   - If one or more seeds fail, treat the recipe as not solved. Analyze variance and continue iterating or report the remaining blocker.

## Reporting

When updating the user, keep the main thread compact:

- Current hypothesis and why it is legal under the constraints.
- Batch launched or monitored, including target `k8s/rtx4090`, seeds, run names, and expected cost when relevant.
- Best evidence so far, including W&B links or local paths.
- Whether the candidate is discovery-only or three-seed confirmed.
- Exact playback command at the end for Mario PPO or similar playable RL runs.

After remote-provider monitoring, orchestration, benchmarking, or launches, include the repo-required short retrospective: avoidable agent-token spend, reproducibility choices worth encoding, and any useful follow-up to ask about.
