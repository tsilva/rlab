---
name: flush-eval
description: Flush sandbox-sb3 Modal eval jobs for checkpoint candidates and report ranked results. Use when the user asks to evaluate unevaluated checkpoints, flush the eval queue, run pending Modal evals, check what checkpoints are not evaluated yet, or produce a post-eval report from the Neon eval database.
---

# Flush Eval

## Contract

Evaluate checkpoint candidates that do not yet have results for the requested eval profile, then report queue status and ranked results from the eval database.

Default eval protocol unless the user says otherwise:

- `eval_profile=mario_level1_no_life_loss_v1`
- `stage=quick`
- `episodes=100`
- `seed_start=10007`
- Modal resources: `--cpu 1 --memory-mib 4096`
- Modal runners: `--runners 1`

This skill is for eval-only work. Do not launch training, upload model cards, publish checkpoints, or mutate candidate selection beyond seeding missing eval jobs.

## Workflow

1. Read `AGENTS.md` and `INSTANCES.md`.
   - Confirm no newer project rule overrides the defaults above.
   - Use `.env` for `DATABASE_URL`, `DIRECT_DATABASE_URL`, W&B, and R2 credentials.

2. Inspect current DB state before launching:

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/setup_neon_eval_queue.py \
  --eval-profile mario_level1_no_life_loss_v1 \
  --stage quick \
  --episodes 100 \
  --seed-start 10007 \
  --no-seed-jobs
```

3. Seed missing jobs for all current `checkpoint_candidates`:

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/setup_neon_eval_queue.py \
  --eval-profile mario_level1_no_life_loss_v1 \
  --stage quick \
  --episodes 100 \
  --seed-start 10007
```

If the user requested a different profile, stage, episode count, or seed, substitute it consistently in every command and in the final report.

4. Estimate Modal cost before launching and get approval when required by project instructions.
   - Default `--cpu 1 --memory-mib 4096 --runners 1` is the cost-effective worker shape.
   - If the user asks to flush faster, scale by increasing `--runners`; do not raise CPU per runner by default.
   - Mention that the eval queue writes results to Neon and downloads W&B/R2 model artifacts.

5. Run the Modal queue until idle:

```bash
.venv/bin/modal run src/stable_retro_ppo/modal_app.py::eval_queue \
  --runners 1 \
  --cpu 1 \
  --memory-mib 4096 \
  --max-jobs-per-runner 0 \
  --idle-polls 2 \
  --idle-sleep-seconds 5 \
  --lease-seconds 1800 \
  --device cpu
```

Use `UV_CACHE_DIR=.uv-cache uv run modal ...` only if the local `.venv/bin/modal` is unavailable.

6. Generate the final report:

```bash
python .codex/skills/flush-eval/scripts/eval_report.py \
  --eval-profile mario_level1_no_life_loss_v1 \
  --stage quick \
  --episodes 100 \
  --seed-start 10007
```

The report should include:

- total candidates
- job status counts
- result count for the requested profile/stage
- remaining pending/running/failed jobs
- top ranked checkpoints by completion rate, max x-position, then reward mean
- eval runtime mean/std when finished job timestamps are available

## Safety

- Do not print `.env` values or database URLs.
- Do not reset failed jobs unless the user asks for retries.
- Do not change `checkpoint_candidates` selection logic unless the user asks to repopulate candidates.
- Do not use terminal-on-life eval as the default for this skill; the default is explicitly no terminal on life.
- Keep generated reports and scratch outputs under ignored paths such as `runs/`.

## Final Response

Report the exact profile, stage, episode count, seed, runner/resource shape, number of jobs evaluated, remaining job counts, top results, and actual/estimated Modal cost. Include the remote-provider retrospective required by `AGENTS.md`.
