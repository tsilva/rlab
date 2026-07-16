---
name: launch-training-recipe
description: Launch and monitor queue-backed rlab training from a checked-in goal and recipe, defaulting to beast-3. Use when the user asks to launch, run, start, execute, or monitor an rlab training recipe or research goal and expects the W&B URL, lightweight progress monitoring, read-only bug escalation, final run statistics, best artifact, and playback command.
---

# Launch Training Recipe

Run the bundled helper. Do not reconstruct this workflow from ad-hoc shell commands.

## Inputs

Resolve from the request and checked-in files:

- one goal file under `experiments/goals/`
- one recipe file under `experiments/recipes/`
- machine, defaulting to `beast-3`
- optional seed and `--set` overrides

Read `SPECS.md`, `INSTANCES.md`, the selected goal, and the selected recipe before launch. If either file cannot be resolved uniquely, ask one concise question. Never guess between multiple candidates. Do not ask about the machine when none is given; use `beast-3`.

## Start

From the repository root, run:

```bash
uv run python .codex/skills/launch-training-recipe/scripts/launch_training.py \
  --goal <goal-file> \
  --recipe <recipe-file> \
  [--machine <machine>] \
  [--seed <seed>] \
  [--set <key=value> ...]
```

The helper creates an isolated clean worktree at the current commit, preserves the caller's dirty worktree, performs exact-source preflight, launches through `rlab train`, monitors authoritative queue state, verifies the terminal result, then removes its temporary worktree. Do not create another worktree or call `rlab train` separately.

Start the helper as a yielded terminal process. Poll that same process every 30-60 seconds. Do not repeatedly run verbose `rlab runs status`, W&B history queries, or log dumps in the main thread.

## Event handling

The helper emits compact JSON lines. Follow them literally:

- `workspace_ready`: the helper is using the reported commit. If `caller_dirty` is true, mention once that unrelated local changes are excluded from the run.
- `launch_started` or `launch_waiting`: keep waiting. A short user update is enough if 60 seconds pass.
- `submitted`: report the job ID, batch, machine, source commit, and runtime when present.
- `wandb_url`: immediately send the clickable URL to the user. Do not wait for another phase.
- `progress`: normally keep silent. Give at most one short update every two minutes unless the user asks for more.
- `potential_bug`: immediately spawn one `training_run_investigator` custom agent for that fingerprint. Pass the job ID, fingerprint, reasons, snapshot, and the repository path. Continue monitoring while it investigates.
- `terminal`: retain its complete summary for the final response.
- `workflow_error`: report the concise error. If a `potential_bug` event preceded it, wait for that investigator; otherwise this is an input or preflight failure and no investigator is required.
- `workspace_cleaned`: no action is required.
- `complete`: the helper and temporary-worktree cleanup are done.

For a repeated fingerprint, reuse or follow up with the existing investigator instead of spawning another. For a distinct fingerprint, spawn another investigator after any current one completes. Wait for investigator reports before the final response.

## Bug boundary

The investigator must be the project custom agent at `.codex/agents/training-run-investigator.toml`, which pins `gpt-5.6-sol` with high reasoning and read-only mode.

Neither the main agent nor the investigator may fix, edit, commit, push, retry, cancel, restart, clean up, or mutate infrastructure after a potential bug appears. Continue read-only monitoring and report:

- observed symptom
- root cause, or the narrowest supported hypothesis
- evidence
- suggested fix
- confidence and remaining unknowns

Do not call a valid fail-fast checkpoint rejection a bug. Treat infrastructure failures, retries, publication errors, inconsistent terminal state, stalled progress, missing W&B visibility, mailbox errors, exhausted eval work, or failed remote verification as potential bugs.

## Final response

For every successful job, report from the `terminal` event:

- job, batch, run name, machine, source, and runtime
- submission-to-finish duration
- terminal and evaluation outcomes
- promoted/best checkpoint step and exact evidence counts when applicable
- retry, failed-attempt, mailbox, stream, and reservation counts
- remote W&B state and key final metrics
- best W&B artifact reference
- W&B run URL again
- the exact `play_command`

Only call a run successful when `verified_success` is true. If it is false, report the terminal state and investigator findings without inventing a playback recommendation.

Keep the final answer concise. Do not expose credentials or raw presigned URLs.

## Resume monitoring

To monitor an existing job without launching anything:

```bash
uv run python .codex/skills/launch-training-recipe/scripts/launch_training.py \
  --monitor-job <job-id>
```
