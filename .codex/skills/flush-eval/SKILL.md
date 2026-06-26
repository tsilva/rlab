---
name: flush-eval
description: Flush rlab campaign eval jobs and report ranked results. Use when the user asks to evaluate unevaluated checkpoints, flush the eval queue, run pending eval jobs, check what checkpoints are not evaluated yet, or produce a post-eval report from the Neon campaign database.
---

# Flush Eval

## Contract

Evaluate campaign eval jobs whose artifacts carry current training metadata, then report queue status and ranked results from the campaign eval database.

Default eval protocol unless the user says otherwise:

- `profile=mario-level1-quick`
- eval config: `episodes=100`, `seed=10007`, `n_envs=20`, `max_steps=4500`, `stochastic=true`
- runner: `rlab-eval-runner`

This skill is for eval-only work. Do not launch training, upload model cards, publish checkpoints, or mutate candidate selection beyond creating explicitly requested eval jobs.

If the user asks to run remote/provider evals, estimate expected cost first and get explicit approval. Local eval runner work does not require a provider-cost approval, but still report expected runtime and hardware assumptions.

## Workflow

1. Read `AGENTS.md` and `INSTANCES.md`.
   - Confirm no newer project rule overrides the defaults above.
   - Use `.env` for `DATABASE_URL`, `DIRECT_DATABASE_URL`, W&B, and R2 credentials.

2. Ensure the campaign schema is current:

```bash
UV_CACHE_DIR=.uv-cache uv run rlab-campaign setup
```

3. Inspect current campaign state:

```bash
UV_CACHE_DIR=.uv-cache uv run rlab-campaign status <goal-slug>
```

If the user asks to add a concrete checkpoint eval job, enqueue it through the campaign table instead of the removed legacy `checkpoint_candidates` queue:

```bash
UV_CACHE_DIR=.uv-cache uv run rlab-campaign enqueue-eval \
  --goal <goal-slug> \
  --profile mario-level1-quick \
  --candidate-label <label> \
  --eval-config-json '{"artifact_ref":"<entity/project/artifact:version>","episodes":100,"seed":10007,"n_envs":20,"max_steps":4500,"stochastic":true}'
```

4. Run the eval queue until idle:

```bash
UV_CACHE_DIR=.uv-cache uv run rlab-eval-runner \
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
- Do not launch remote/provider workers without explicit cost approval in the current turn.
- Do not use terminal-on-life eval as the default for this skill; the default is explicitly no terminal on life.
- Keep generated reports and scratch outputs under ignored paths such as `runs/`.

## Final Response

Report the exact profile, episode count, seed, runner shape, number of jobs evaluated, remaining job counts, top results, and any skipped cost/provider details. Include the remote-provider retrospective required by `AGENTS.md` only when remote providers were used.
