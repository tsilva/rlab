---
name: launch-experiment
description: Launch and continuously monitor queue-backed rlab training from a checked-in goal and recipe, defaulting to beast-3. Use when the user asks to launch, run, start, execute, follow, watch, or monitor an rlab training recipe, run, or research goal. Stay attached through the terminal event, report the W&B URL immediately, escalate potential bugs read-only, and return final statistics, the best artifact, and the exact playback command.
---

# Launch Experiment

Use the repository CLI only. Never reload, replace, install, or restart controllers.

## Non-negotiable completion rule

Launching is not completion. Seeing a run start, a W&B URL, or a progress event is not completion.

Do not send the final response until:

1. every monitored run emitted one `terminal` event; and
2. every dispatched investigator returned.

Keep the monitor processes alive while doing all other work. An empty poll means "still waiting," not "monitoring is done."

## Choose one mode

- **Launch mode:** resolve a goal and recipe, launch them, then monitor every returned run ID.
- **Existing-run mode:** if the user supplied a run ID to watch, do not launch anything; monitor that run ID.

Read `SPECS.md` before either mode. The remaining goal, recipe, and hardware reads below apply to launch mode.

## Launch mode

Resolve exactly:

- one goal under `experiments/goals/`
- one recipe under `experiments/recipes/`
- machine, defaulting to `beast-3`
- optional seed and `--set` overrides

Read `INSTANCES.md`, the selected goal, and the selected recipe. If the goal or recipe is ambiguous, ask one concise question. Never guess between candidates. Do not ask for a machine when none was given.

From the repository root, run:

```bash
rlab experiment launch \
  --from-head \
  --goal-file <goal-file> \
  --recipe-file <recipe-file> \
  --machine <machine> \
  [--seed <seed>] \
  [--set <key=value> ...] \
  --json
```

This launches committed `HEAD` from an isolated worktree and excludes unrelated local changes. Parse the returned JSON. Immediately report `run_ids`, `batch_id`, machine, source SHA, source branch, runtime image, and `local_changes_excluded`.

If launch reports an incompatible control plane, stop and give the user the exact printed remediation. Never perform that remediation in this skill.

## Start persistent monitors

For every run ID, start exactly one long-lived process:

```bash
rlab experiment follow --run <run-id> --jsonl
```

Start it with a yielded execution call. If it is still running, retain its session handle. If it already exited, parse all returned lines and do not restart it. Do not wait for the whole run in the initial call.

Track this state for each run:

```text
run_id -> session handle, terminal payload or null, W&B URL sent?, investigator by fingerprint
```

For multiple runs, start one monitor per run and poll them round-robin.

## Required monitoring loop

Repeat these steps until every run has a stored `terminal` payload:

1. Resume each nonterminal run's **same session handle** with the session-write/resume tool, empty input, and a 30-60 second yield. Never send a newline, interrupt, or control character.
2. Parse every complete JSON line received. Handle events using the table below.
3. If no line arrived and the process is still running, do nothing except continue the loop.
4. If the process exited, parse its buffered lines before considering its exit code.
5. If it exited without a `terminal` event, report an observer failure and stop. Do not relaunch, repair, or switch to ad hoc status commands.

Do not use separate status, W&B-history, or log commands during ordinary monitoring. The follow stream is authoritative. Extra read-only diagnostics are allowed only after a `potential_bug` event and belong to the investigator.

| Event | Required action | Continue monitoring? |
| --- | --- | --- |
| `wandb_url` | Immediately send the clickable `url` to the user and mark it sent. | Yes |
| `progress` | Store the latest projection. Normally say nothing; send at most one brief progress update per two minutes. | Yes |
| `potential_bug` | Immediately dispatch the project `training_run_investigator` for its fingerprint. Pass repo path, run ID, `incident`, and `run` projection, then return to the loop without waiting. | Yes, without interruption |
| `reporting_warning` | Store and report the warning; do not change the terminal classification. | Yes |
| `terminal` | Store the complete payload for that run. This is the only event that ends its monitor loop. | No, for this run |

`follow` exits 0 for `accepted` or `completed`, 2 for `goal_rejected`, 3 for `canceled`, and 1 for `operational_failure`. These nonzero exits are expected classifications when accompanied by `terminal`; the JSON event, not the shell code alone, determines the result.

For a repeated bug fingerprint, reuse or follow up with its existing investigator. Queue a distinct fingerprint until the current investigator completes if no investigator slot is available. Wait for all investigator reports before the final response.

## Relaunch boundary

Launch another explicit seed only after `goal_rejected` and only when the user asked for repeated attempts. Preserve every override. Never automatically retry after `canceled`, `operational_failure`, `potential_bug`, or observer failure. A valid fail-fast rejection is not a bug.

## Read-only bug boundary

Use only the project custom agent at `.codex/agents/training-run-investigator.toml` for bug diagnosis. It pins `gpt-5.6-sol`, high reasoning, and read-only mode.

After any potential bug appears, neither the main agent nor investigator may fix, edit, commit, push, retry, cancel, restart, clean up, or mutate infrastructure. Continue monitoring and report:

- observed symptom
- root cause, or narrowest supported hypothesis
- concrete evidence
- suggested fix
- confidence and remaining unknowns

Do not call a valid fail-fast checkpoint rejection a bug. Infrastructure failures, retries, publication errors, inconsistent terminal state, stalled progress, missing W&B visibility, mailbox errors, exhausted eval work, or failed remote verification are potential bugs.

## Final response

For every run, state its terminal classification. For every successful run, copy these values from its stored `terminal` event:

- run, batch, run name, machine, source, and runtime
- submission-to-finish duration
- terminal and evaluation outcomes
- promoted or best checkpoint step and exact evidence counts, when applicable
- retry, failed-attempt, mailbox, stream, and reservation counts
- remote W&B state and key final metrics
- immutable best W&B artifact reference
- W&B run URL
- exact `play_command`

Call an acceptance run successful only when `verified_success` is true, `terminal_classification` is `accepted`, and the user-required early-stop condition is satisfied. Use the emitted immutable `wandb_artifact` and exact `play_command`; never substitute an R2 URI or mutable alias.

Keep the answer concise. Never expose credentials or raw presigned URLs.
