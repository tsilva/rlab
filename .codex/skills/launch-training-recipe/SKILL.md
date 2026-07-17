---
name: launch-training-recipe
description: Launch and monitor queue-backed rlab training from a checked-in goal and recipe, defaulting to beast-3. Use when the user asks to launch, run, start, execute, or monitor an rlab training recipe or research goal and expects the W&B URL, lightweight progress monitoring, read-only bug escalation, final run statistics, best artifact, and playback command.
---

# Launch Training Recipe

Use the repository CLI only. Do not reload, replace, install, or restart controllers as part of a launch.

## Inputs

Resolve from the request and checked-in files:

- one goal file under `experiments/goals/`
- one recipe file under `experiments/recipes/`
- machine, defaulting to `beast-3`
- optional seed and `--set` overrides

Read `SPECS.md`, `INSTANCES.md`, the selected goal, and the selected recipe before launch. If either file cannot be resolved uniquely, ask one concise question. Never guess between multiple candidates. Do not ask about the machine when none is given; use `beast-3`.

## Launch

From the repository root, run:

```bash
uv run rlab experiment launch \
  --from-head \
  --goal-file <goal-file> \
  --recipe-file <recipe-file> \
  --machine <machine> \
  [--seed <seed>] \
  [--set <key=value> ...] \
  --json
```

The command launches committed `HEAD` from an isolated worktree and excludes unrelated local changes. Report `run_ids`, `batch_id`, machine, source SHA, source branch, runtime image, and `local_changes_excluded`. If the command reports an incompatible control plane, stop and tell the user to run the exact remediation it prints. Never repair the service during this skill.

For each returned run, start one yielded monitor:

```bash
uv run rlab experiment follow --run <run-id> --jsonl
```

Poll that same process every 30-60 seconds. Do not run extra status, W&B history, or log commands unless investigating an emitted bug.

## Event handling

`experiment follow` emits compact JSON lines. Follow them literally:

- `wandb_url`: immediately send the clickable URL to the user. Do not wait for another phase.
- `progress`: normally stay silent. Give at most one short update every two minutes.
- `potential_bug`: immediately dispatch the `training_run_investigator` custom agent for that fingerprint. Pass the run ID, incident, projection, and repo path. Keep monitoring.
- `reporting_warning`: report that terminal presentation data was unavailable; do not change the authoritative terminal classification.
- `terminal`: retain the full payload. `accepted` and `completed` exit 0, `goal_rejected` exits 2, `canceled` exits 3, and `operational_failure` exits 1.

For a repeated fingerprint, reuse or follow up with the existing investigator instead of spawning another. For a distinct fingerprint, spawn another investigator after any current one completes. Wait for investigator reports before the final response.

Only launch the next explicit seed after `goal_rejected` when the user asked for repeated attempts. Preserve all overrides. Never automatically relaunch after `canceled`, `operational_failure`, `potential_bug`, or observer failure. A valid fail-fast rejection is not a bug.

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

For every successful run, report from the `terminal` event:

- run, batch, run name, machine, source, and runtime
- submission-to-finish duration
- terminal and evaluation outcomes
- promoted/best checkpoint step and exact evidence counts when applicable
- retry, failed-attempt, mailbox, stream, and reservation counts
- remote W&B state and key final metrics
- best W&B artifact reference
- W&B run URL again
- the exact `play_command`

Only call an acceptance run successful when `verified_success` is true, `terminal_classification` is `accepted`, and the user-required early-stop condition is satisfied. Use the exact immutable `wandb_artifact` and emitted `play_command`; never substitute the R2 checkpoint URI or a mutable alias.

Keep the final answer concise. Do not expose credentials or raw presigned URLs.

## Monitor an existing run

Do not launch anything. Run:

```bash
uv run rlab experiment follow --run <run-id> --jsonl
```
