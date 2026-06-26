---
name: skypilot-mario-monitor
description: Monitor rlab Mario PPO SkyPilot jobs, collect W&B per-seed metrics, decide whether the recipe criterion is met, and conditionally launch prepared follow-up batches without oversubscribing the RTX4090.
---

# SkyPilot Mario Monitor

## Contract

Use this skill when the user asks to poll, monitor, summarize, or continue a
rlab Mario PPO SkyPilot training batch, especially when a follow-up batch
should launch only after the active job reaches a terminal state.

The local source of truth is the current repo plus live SkyPilot/W&B state. Do
not rely on a heartbeat summary alone when the cluster or W&B can be queried.

## Required Reads

Before making launch, cancellation, concurrency, or hardware decisions:

1. Read `AGENTS.md`.
2. Read `INSTANCES.md`.
3. If the task is part of recipe search, also read `.codex/skills/autoresearch/SKILL.md`.
4. If using SkyPilot commands, read `/Users/tsilva/.codex/skills/run-skypilot/SKILL.md` and its `references/current-environment.md`.

## Polling Workflow

1. Identify the active cluster, job id, job name, local manifest, local SkyPilot
   log path, W&B group, run ids, seeds, and success criterion from the user's
   instruction, automation payload, or recent repo artifacts.
2. Poll SkyPilot with the repo-local `uv` setup:

```bash
UV_CACHE_DIR=.uv-cache uv run sky queue <cluster-name>
```

3. If SkyPilot reports `RUNNING` or `PENDING`, inspect W&B summaries for
   progress. Load `.env` from Python instead of shell-sourcing it, because
   special characters in secrets can break shell parsing.
4. For each child run, collect at minimum:
   - W&B run id and URL
   - run state
   - seed and arm/recipe label
   - `global_step`
   - `train/completion_episode_rate`
   - `train/completion_episodes_total`
   - early-stop or final status when available
   - useful diagnostics: `rollout/ep_rew_mean`, `time/fps`, `train/approx_kl`,
     `train/clip_fraction`, entropy, and explained variance.
5. Report only material running-state updates to the user. A material update
   includes failure, terminal success/failure, a solved recipe criterion, clear
   metric regression, a stalled run, or a user explicitly asking for status.

## Terminal Handling

If a job `FAILED` or `CANCELED`:

- Fetch key error lines from `sky logs <cluster> <job-id>`.
- Report the terminal status, log path, and actionable error lines.
- Do not launch a follow-up batch unless the user explicitly asks to retry.

If a job `SUCCEEDED`:

- Collect final per-seed metrics from W&B and/or SkyPilot logs.
- Evaluate the exact success criterion the user gave. For recent Mario recipe
  searches, this is often strict `100/100`, represented as
  `train/completion_episode_rate == 1.0`, for every required seed in the same
  arm.
- If successful, state the frozen recipe, runtime package version, seeds, links,
  and exact playback command when there is a best checkpoint to inspect.
- If not successful and the user asked for iteration, design or launch the next
  batch only after the RTX4090 is free.

## Follow-Up Launch Rules

- Do not launch a new long trainer while the single RTX4090 is occupied by an
  active SkyPilot job unless the user explicitly instructs cancellation or
  overlapping contention.
- Prefer prepared manifests when the user or prior run already named one.
- Run preflight before launch when using repo launch manifests.
- Preserve generated artifacts in ignored locations such as `logs/`, `runs/`,
  `models/`, W&B, or R2/S3.
- Keep the runtime pin explicit in the manifest, run descriptions, and W&B tags.
- Use `.env`/SkyPilot secret handling without printing secrets.

## Reporting

For status replies, lead with the current state:

- cluster/job id and status
- best arm/seed so far
- whether the success criterion is met
- whether a follow-up was launched or intentionally deferred
- W&B group/run links and local log path when useful

After remote-provider monitoring, orchestration, benchmarking, or launches,
include the repo-required short retrospective: avoidable agent-token spend,
reproducibility choices worth encoding, and useful follow-up.
