---
name: launch-experiment
description: Launch, continuously monitor, and when explicitly authorized repair dstack-backed rlab training from a checked-in goal and recipe. Use when the user asks to launch, run, start, execute, follow, watch, monitor, diagnose, fix, or harden an rlab training recipe, run, or research goal. Stay attached through the authoritative R2 terminal receipt, report the W&B URL immediately, preserve failed-attempt evidence, and return the promoted public checkpoint and exact playback command.
---

# Launch Experiment

Use `rlab experiment` for dstack-backed runs. dstack owns compute placement,
logs, cancellation, retries for genuine interruption/no-capacity, and resource
release. Private R2 receipts own run, checkpoint, evaluation, promotion, and
terminal semantics. Never infer scientific success from dstack exit status.

Read `SPECS.md` before either launch or existing-run monitoring. Before a new
launch also read `INSTANCES.md`, the selected goal, and the selected recipe.

## Safety mode

- **Observe mode (default):** launch and monitor only. Diagnose potential bugs
  read-only. Do not edit, retry, cancel, restart, or mutate infrastructure.
- **Repair mode:** only when the user explicitly asks to fix, harden, or keep
  working until the complete workflow passes. Preserve failed attempts, repair
  root causes, add regression coverage, and launch a fresh attempt. Never weaken
  the goal, episode manifest, acceptance threshold, or cost policy.

Repair mode does not authorize destructive cleanup, credential changes,
commits/pushes, dstack server replacement, unrelated-run cancellation, or
unbounded cloud cost.

## Launch

Resolve exactly one checked-in goal and one launchable recipe from the selected goal's `recipes/` directory
per requested run; reusable defaults may come from
`experiments/recipes/_presets/`. Independent ablations are separate runs unless the user
explicitly asks to combine them. Repeatable `--set KEY=VALUE` overrides are
allowed when each launch row records its complete isolated override list; they
are composed, validated, and hash-bound in the immutable recipe contract.

Default to `--compute auto`. For the known local B3 host use
`--compute local --target b3`. Spot requires finite `--max-price` and
`--max-cost-usd`. On-demand additionally requires `--allow-on-demand`. Always
use a finite `--max-duration`.

```bash
rlab experiment launch \
  --goal-file <goal-file> \
  --recipe-file <recipe-file> \
  --seed <seed> \
  --run-description "<specific description>" \
  [--set <key=value> ...] \
  --compute <auto|local|spot|on-demand> \
  [--target <fleet-or-instance>] \
  [--max-price <hourly-price>] \
  [--max-cost-usd <total-bound>] \
  [--allow-on-demand] \
  --max-duration <duration> \
  --json
```

The command requires a clean, pushed source revision, resolves the verified
exact-source immutable training image and Modal deployment, stages the
hash-verified ROM to eval-private R2, creates the run manifest, and then submits
the dstack task. It never falls back to an older runtime.

Immediately report the returned run ID, attempt ID, dstack task, selected
compute/offer and maximum cost, source/image digest, W&B URL, and public R2 run
index.

## Monitor

Start one yielded long-lived monitor per run and retain its session handle:

```bash
rlab experiment follow --run <run-id>
```

Resume the same process with empty polls. Do not send a newline or interrupt it.
Each JSON line is a combined snapshot of dstack state and authoritative R2
semantic state. Send compact progress updates at most every two minutes unless
state requires attention.

The monitor is complete only when `semantic.terminal` exists. A successful
dstack exit without that receipt is an operational failure, not a scientific
success. Keep monitoring while any checkpoint evaluation or W&B drain remains
pending.

Use these read-only commands only when the follow process fails or a snapshot
shows an operational anomaly:

```bash
rlab experiment status --run <run-id> --json
rlab experiment logs --run <run-id> --tail 200
```

If an active run reveals a potential bug, dispatch the project
`training_run_investigator` with the repo path, run ID, latest combined
snapshot, and relevant log excerpt. Keep monitoring while it investigates.
Reuse the same investigator for the same fingerprint.

## Retry and repair

Never blindly retry. A manual retry is permitted only after:

1. the previous dstack attempt is terminal;
2. its R2 writer lease expired;
3. the CLI-enforced 30-second quiescence interval; and
4. the failure is supported as resumable.

```bash
rlab experiment retry --run <run-id>
```

The logical run ID remains stable and a new attempt ID is created. If a final
checkpoint already exists or acceptance was recorded, retry enters drain-only
recovery and never retrains.

In repair mode, wait for the failed attempt to become terminal and for its
investigator to return. Reproduce narrowly, patch the root cause, add a
deterministic regression, run the affected tests, publish the exact-source
runtime, then retry or launch fresh as the durable state requires.

## Completion

Launching, seeing W&B, observing an accepted eval, or seeing dstack exit is not
completion. A successful accepted run requires all of:

- authoritative terminal receipt with accepted stop reason;
- accepted 100/100 evidence for the immutable episode manifest;
- eval-driven cooperative stop at a safe learner boundary;
- every frozen periodic/final checkpoint publicly downloadable and terminally
  evaluated;
- the lowest-step accepted checkpoint promoted exactly once;
- W&B through its recorded high-water mark, written by the supervisor only;
- dstack successful and the host released;
- credential-free `rlab play --run <run-id>` from the public index.

Report the run and attempt IDs, dstack task/compute, source and image digest,
terminal stop reason and step, promoted checkpoint/evidence counts, W&B URL,
public index/checkpoint URLs, drain result, idle-GPU tail, and exact play
command. Do not expose credentials or presigned private URLs.
