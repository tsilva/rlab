---
name: flush-eval
description: Flush rlab eval queue jobs and report ranked results. Use when the user asks to evaluate unevaluated checkpoints, flush the eval queue, run pending eval jobs, check what checkpoints are not evaluated yet, or produce a post-eval report from the Neon queue database.
---

# Flush Eval

## Contract

Evaluate queue eval jobs whose artifacts carry current training metadata, then report queue status and ranked results from the eval results table.

Default eval protocol unless the user says otherwise:

- `profile=mario-level1-quick`
- eval config: `episodes=100`, `seed=10007`, `n_envs=20`, `max_steps=4500`, `stochastic=true`
- runner: `rlab eval worker`

This skill is for eval-only work. Do not launch training, upload model cards, publish checkpoints, or mutate candidate selection beyond creating explicitly requested eval jobs.

Use the local eval runner by default. Report expected runtime and hardware assumptions when the queue is large.

## Workflow

1. Read `AGENTS.md` and `INSTANCES.md`.
   - Confirm no newer project rule overrides the defaults above.
   - Use `.env` for `DATABASE_URL`, `DIRECT_DATABASE_URL`, W&B, and R2 credentials.

2. Ensure the queue schema is current:

```bash
UV_CACHE_DIR=.uv-cache uv run rlab jobs setup
```

3. Inspect current queue state:

```bash
UV_CACHE_DIR=.uv-cache uv run rlab jobs status --goal <goal-slug>
```

If the user asks to add a concrete checkpoint eval job, enqueue it through the queue table instead of the removed legacy `checkpoint_candidates` queue:

```bash
UV_CACHE_DIR=.uv-cache uv run rlab eval enqueue \
  --goal <goal-slug> \
  --profile mario-level1-quick \
  --candidate-label <label> \
  --eval-config-json '{"artifact_ref":"<entity/project/artifact:version>","episodes":100,"seed":10007,"n_envs":20,"max_steps":4500,"stochastic":true}'
```

4. Run the eval queue until idle:

```bash
UV_CACHE_DIR=.uv-cache uv run rlab eval worker \
  --profile mario-level1-quick \
  --lease-seconds 1800 \
  --max-jobs 0 \
  --once \
  --artifact-root runs/eval_artifacts \
  --output-dir logs/eval_runner
```

5. Generate the final report:

```bash
UV_CACHE_DIR=.uv-cache uv run python .codex/skills/flush-eval/scripts/eval_report.py \
  --profile mario-level1-quick
```

The report should include:

- job status counts
- result count for the requested profile
- remaining pending/running/failed jobs
- top ranked checkpoints by completion rate, reward mean, then max x-position
- eval runtime mean/std when finished job timestamps are available
- artifact/model refs and output paths when present

## Safety

- Do not print `.env` values or database URLs.
- Do not reset failed jobs unless the user asks for retries.
- Do not use terminal-on-life eval as the default for this skill; the default is explicitly no terminal on life.
- Keep generated reports and scratch outputs under ignored paths such as `runs/`.

## Final Response

Report the exact profile, episode count, seed, runner shape, number of jobs evaluated, remaining job counts, top results, and any skipped local-runtime details.
