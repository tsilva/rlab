---
name: launch-experiment
description: Launch, continuously monitor, and when explicitly authorized repair queue-backed rlab training from a checked-in goal and recipe, defaulting to beast-3. Use when the user asks to launch, run, start, execute, follow, watch, monitor, diagnose, fix, or harden an rlab training recipe, run, or research goal. Stay attached through terminal events, report W&B URLs immediately, preserve failed-run evidence, and return final statistics, the best artifact, and the exact playback command.
---

# Launch Experiment

Use the repository CLI for training and fleet operations. Never reload, replace, install, or restart controllers unless the current user request separately authorizes that live control-plane action.

## Choose safety mode

- **Observe mode (default):** Use when the user asks only to launch, run, follow, watch, or monitor. Diagnose potential bugs read-only and do not edit, repair, retry, cancel, restart, clean up, or mutate infrastructure.
- **Repair mode (explicit opt-in):** Use only when the current user request explicitly authorizes fixing or hardening failures, making changes until the workflow works, or diagnosing and repairing a run. A bare request to launch or monitor does not authorize repair mode. Repair mode carries through an active persistent goal until the user edits, pauses, clears, or replaces that goal.

Repair mode authorizes in-scope repository code, configuration, and test changes plus fresh verification launches. It does not by itself authorize destructive cleanup, canceling unrelated work, changing credentials, relaxing goal or acceptance semantics, editing `SPECS.md`, committing, pushing, controller reload/replacement/restart, or materially expanding external cost. Obtain separate explicit authorization when one of those actions is required.

## Non-negotiable completion rule

Launching is not completion. Seeing a run start, a W&B URL, or a progress event is not completion.

Do not send the final response until:

1. every monitored run emitted one `terminal` event; and
2. every dispatched investigator returned.

In repair mode, one terminal run is not completion when the user requested a working or hardened end-to-end flow. Continue the evidence-fix-test-fresh-launch cycle until the user's measurable completion criteria pass or progress requires new authority or an external-state change.

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

If launch reports an incompatible control plane, preserve the exact output. In observe mode, stop and give the user the printed remediation. In repair mode, apply only an in-scope, non-destructive repository repair; request authorization before controller replacement, reload, restart, or other live control-plane mutation.

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
5. If it exited without a `terminal` event, preserve the buffered lines and classify it as an observer failure. In observe mode, report it and stop without ad hoc status commands. In repair mode, diagnose and repair the observer, use the minimum read-only status or log checks needed to avoid losing the underlying run, and attach one replacement follow process if necessary. Never infer a terminal state.

Do not use separate status, W&B-history, or log commands during ordinary monitoring. The follow stream is authoritative. During an active run, extra read-only diagnostics are allowed only after a `potential_bug` event and belong to the investigator, except for the minimum repair-mode observer recovery described above. After a run is terminal, the repair-mode root-cause cycle may use the diagnostics needed to reproduce and verify the defect.

| Event | Required action | Continue monitoring? |
| --- | --- | --- |
| `wandb_url` | Immediately send the clickable `url` to the user and mark it sent. | Yes |
| `progress` | Store the latest projection. Normally say nothing; send at most one brief progress update per two minutes. | Yes |
| `potential_bug` | Immediately dispatch the project `training_run_investigator` for its fingerprint. Pass repo path, run ID, `incident`, and `run` projection, then return to the loop without waiting. | Yes, without interruption |
| `attention_required` | Immediately report and persist the attention payload. Keep the monitor attached, do not dispatch an investigator for a budget block, and do not launch later work while attention remains. | Yes, without interruption |
| `reporting_warning` | Store and report the warning; do not change the terminal classification. | Yes |
| `terminal` | Store the complete payload for that run. This is the only event that ends its monitor loop. | No, for this run |

`follow` exits 0 for `accepted` or `completed`, 2 for `goal_rejected`, 3 for `canceled`, and 1 for `operational_failure`. These nonzero exits are expected classifications when accompanied by `terminal`; the JSON event, not the shell code alone, determines the result.

For a repeated bug fingerprint, reuse or follow up with its existing investigator. Queue a distinct fingerprint until the current investigator completes if no investigator slot is available. Wait for all investigator reports before the final response.

## Relaunch boundary

In observe mode, launch another explicit seed only after `goal_rejected` and only when the user asked for repeated attempts. Preserve every override. A user-requested bounded durable research controller may launch its preregistered next cohort after every prior run reached the terminal classification required by that controller, every investigator returned, its durable evidence was collected, and its barrier closed. For `$autoresearch`, that classification is remotely published training-only `completed`; it must collect the exact W&B training evidence before advancing. A bounded controller must never advance after `canceled`, `operational_failure`, `potential_bug`, `attention_required`, observer failure, or evidence mismatch.

In repair mode, never blindly retry. Preserve the failed attempt and its evidence, establish and repair a supported root cause, add regression coverage, and pass relevant tests before submitting a fresh launch with a new batch and run identity. Preserve the original goal, recipe, seed, and overrides unless the diagnosed defect is in that configuration; never weaken the research or acceptance contract to manufacture success. A valid fail-fast rejection is not a system bug and permits another seed only when the user requested repeated research attempts.

## Bug handling boundary

For a run-backed `potential_bug` event, use only the project custom agent at `.codex/agents/training-run-investigator.toml` for diagnosis while the run is active. It pins `gpt-5.6-sol`, high reasoning, and read-only mode. Diagnose pre-queue failures locally because no run exists to investigate.

After any potential bug appears, continue monitoring the current run and preserve:

- observed symptom
- root cause, or narrowest supported hypothesis
- concrete evidence
- suggested fix
- confidence and remaining unknowns

In observe mode, neither the main agent nor investigator may fix, edit, commit, push, retry, cancel, restart, clean up, or mutate infrastructure.

In repair mode:

1. Keep the investigator read-only and keep the current run monitor attached through its terminal event.
2. Do not mutate the isolated source, queue record, runtime, container, evaluation work, or publication state of the failed attempt.
3. After the run is terminal and its investigator has returned, reproduce the defect as narrowly as practical, implement the root-cause repair in the current worktree, and add deterministic regression coverage. A failure before queue mutation may be repaired immediately after preserving its output because no run exists to monitor.
4. Run targeted verification and every broader test surface materially affected by the repair.
5. Submit a fresh launch and monitor it under the same rules. Repeat for newly observed defects until the explicit completion criteria pass.
6. Stop and request direction when the repair requires a separately protected action listed under the safety modes, conflicts with `SPECS.md`, cannot preserve unrelated user changes, or cannot make progress without external coordination.

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

In repair mode, also summarize each failure fingerprint, root cause, repair, regression test, superseding verification run, and any residual risk or protected action that still needs user authorization.

Call an acceptance run successful only when `verified_success` is true, `terminal_classification` is `accepted`, and the user-required early-stop condition is satisfied. Use the emitted immutable `wandb_artifact` and exact `play_command`; never substitute an R2 URI or mutable alias.

For an `$autoresearch` training-only run, report `completed` and its remote training metrics without calling the run accepted or successful against the goal. Autoresearch selects recipes from training signals; it does not produce a promoted artifact or playback command.

Keep the answer concise. Never expose credentials or raw presigned URLs.
