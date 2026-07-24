---
name: autoresearch
description: Tune one checked-in rlab SB3 PPO or A2C recipe from durable training completion signals without launching checkpoint evaluations. Use when the user points to a recipe and asks to tune, optimize, autoresearch, improve sample efficiency, maximize training return, find the best hyperparameters, or make training behavior stable across seeds. Runs a bounded fixed-rung B3 dstack search, confirms the winner on five untouched training seeds, and patches only the pointed leaf recipe.
---

# Autoresearch

Tune the pointed recipe with a training-only fixed-rung study. A direct invocation authorizes at
most 48 reserved dstack runs on B3; it does not authorize infrastructure repair, capacity
changes, runtime builds or deployments, cancellation, commits, pushes, checkpoint promotion, or
goal-acceptance claims.

## Load the contracts

Before acting:

1. Read `SPECS.md` with `$specs-author`; do not edit it unless separately authorized.
2. Read `INSTANCES.md`, the selected goal, the selected leaf recipe, and every composed source in
   `_composition.source_files`.
3. Read [search-policy.md](references/search-policy.md) completely.
4. Use `$launch-experiment` in observe mode for every launch and monitor.

Resolve exactly one checked-in goal and recipe. V3 accepts only `sb3.ppo` and `sb3.a2c` recipes
with explicit training start states and distinct quantized 20%, 50%, and full training caps.
It uses success evidence when the task declares success termination and return evidence when the
goal ranks episode return but declares no success event.
Historical queue-backed schema-v1/v2 studies are inert and cannot be resumed.

## Initialize or resume

```bash
uv run python .codex/skills/autoresearch/scripts/study.py init \
  --goal <goal-file> \
  --recipe <recipe-file> \
  [--strong-threshold 0.90]
```

The helper resumes the sole matching schema-v2 study, pins committed `HEAD`, all composed source
hashes, the runtime triplet, rung caps, seeds, and threshold, and stores authoritative state in
`runs/autoresearch/<study>/study.json`. Never hand-edit it.

Ask for exactly one deterministic action at a time:

```bash
uv run python .codex/skills/autoresearch/scripts/study.py next --study <study>
```

Obey the returned action. Never infer or skip a rung, evidence read, or barrier.

## Read capacity and reserve

Before every reservation, read dstack without mutating capacity:

```bash
dstack ps --json
```

B3 has one training slot. Pass `1` as `effective_capacity` and the count of active B3 training
tasks (zero or one) as `active_reservations`. Wait when the slot is occupied.

Baseline screen and pair:

```bash
uv run python .codex/skills/autoresearch/scripts/study.py reserve-wave \
  --study <study> --phase baseline-screen \
  --effective-capacity <capacity> --active-reservations <active>

uv run python .codex/skills/autoresearch/scripts/study.py reserve-wave \
  --study <study> --phase baseline-pair \
  --effective-capacity <capacity> --active-reservations <active>
```

For `propose_search`, compute `available_slots = effective_capacity - active_reservations` and
propose exactly `min(next.max_candidates, available_slots)` new deltas. Change one coherent numeric
group per candidate, then reserve the one-seed screens:

```bash
uv run python .codex/skills/autoresearch/scripts/study.py reserve-wave \
  --study <study> --phase search-screen \
  --candidates-json '[{"delta":{"learning_rate":0.0002}}]' \
  --effective-capacity <capacity> --active-reservations <active>
```

Reserve a passed candidate's two fresh 50% seeds only when requested:

```bash
uv run python .codex/skills/autoresearch/scripts/study.py reserve-wave \
  --study <study> --phase search-pair --candidate-id <candidate-id> \
  --effective-capacity <capacity> --active-reservations <active>
```

Reserve the five full-cap confirmation seeds only when requested:

```bash
uv run python .codex/skills/autoresearch/scripts/study.py reserve-wave \
  --study <study> --phase confirmation \
  --effective-capacity <capacity> --active-reservations <active>
```

Reservations are durable. Search always retains five of the 48 jobs for at least one confirmation.

## Reconcile, launch, and monitor

Each wave returns one command per seed. Before launching, inspect `dstack ps -a --json` and the
study state for an already-recorded immutable run with the same submission key and seed.

```bash
dstack ps -a --json
```

- Record exact existing runs when every seed is present.
- Execute a generated seed command once only when that seed has no matching run.
- Pause on ambiguous, duplicate, partial, or mismatched runs; never replace or retry them ad hoc.

Combine each launch command’s JSON output as `{"runs":[...]}` before recording the wave:

```bash
uv run python .codex/skills/autoresearch/scripts/study.py record-launch \
  --study <study> --submission-key <key> --payload-file <json-file>
```

Every generated launch forces `--checkpoint-eval-backend none`, applies only the rung-specific
`train.timesteps` and candidate overrides, and retains exact committed source,
`--existing-runtime-only`, fixed seeds, deterministic submission identity, W&B, checkpoint
publication, and runtime reconciliation. The baseline captures the exact runtime
image/input/build-source triplet; later waves must match it. No Modal credentials, readiness,
runtime build, or deployment is allowed.

Because B3 has one slot, launch returned seed commands as capacity becomes available. Start exactly
one persistent `rlab experiment follow --run <id>` monitor per run under `$launch-experiment`.
Stay attached through the attempt terminal, dstack release, and investigator result. Report W&B
URLs immediately.

- Record `potential_bug` or `attention_required` and pause later waves.
- Do not repair, retry, cancel, restart, or mutate infrastructure.
- Accept only remotely published, training-only `completed` terminals. Any eval-backed,
  canceled, failed, mismatched, or unverified terminal pauses the study.

```bash
uv run python .codex/skills/autoresearch/scripts/study.py record-terminal \
  --study <study> --event-file <terminal-event.json>
```

When `next` requests training evidence, query the exact remotely finished W&B run:

```bash
uv run python .codex/skills/autoresearch/scripts/study.py collect-training-evidence \
  --study <study> --run-id <run-id>
```

A transient W&B read failure leaves the action pending and is safe to repeat. Missing success or
window-100 metrics are valid negative evidence, not an infrastructure failure. Return-mode
studies collect the final 10% mean, p05, standard deviation, peak, and last value of
`train/episode/return/shaped/mean`.

An active v2 study incorrectly screened as success-only may be upgraded only when its frozen goal
is return-ranked, it has no paired evidence, and confirmation or apply has not started:

```bash
uv run python .codex/skills/autoresearch/scripts/study.py upgrade-return-mode --study <study>
```

This re-reads the exact finished W&B runs and reopens only prematurely closed screen barriers. It
does not alter or relaunch a queue job.

Persist attention and resume only from supported new evidence:

```bash
uv run python .codex/skills/autoresearch/scripts/study.py record-attention \
  --study <study> --event-file <event.json>

uv run python .codex/skills/autoresearch/scripts/study.py resume \
  --study <study> --reason '<resolved evidence>'
```

## Close barriers and confirm

After every screen has evidence and every passing candidate's paired rung has evidence:

```bash
uv run python .codex/skills/autoresearch/scripts/study.py close-round \
  --study <study> --round <round>
```

Success mode ranks two-seed evidence by strong-seed count, median and worst censored first-strong
step, then worst peak window-100 minimum. Return mode pairs only the best robust screen per round
and ranks paired evidence by worst and median final-10% mean, worst and median final-10% p05, then
median peak. Stop after three stale rounds or when budget cannot support another complete
candidate plus confirmation.

After exactly five full-cap confirmation runs have evidence:

```bash
uv run python .codex/skills/autoresearch/scripts/study.py close-confirmation --study <study>
```

Success mode requires at least four seeds to reach the frozen strong threshold. Return mode
requires at least four seeds to meet the incumbent's frozen worst paired final-10% mean. A failed
holdout is a binary exclusion; do not inspect its per-seed details to choose later candidates. If
no candidate can be confirmed, run the requested `finish-no-winner` action and leave the recipe
unchanged.

## Apply the winner

Patch only the pointed leaf recipe with the winning `train.backend.config` numeric delta. Never
write rung caps, seeds, study identity, `checkpoint_eval_backend=none`, or descriptions into the
recipe.

Create the exact planned postimage under the generated study directory with `apply_patch`, then:

```bash
uv run python .codex/skills/autoresearch/scripts/study.py prepare-apply \
  --study <study> --postimage-file <planned-recipe-file>
```

Apply that exact patch to the recipe with `apply_patch`, then:

```bash
uv run python .codex/skills/autoresearch/scripts/study.py complete-apply --study <study>
```

The helper recomposes the recipe and proves that only the confirmed backend delta changed. A
baseline winner is a valid no-op. Run targeted tests and leave the recipe patch uncommitted.

Report the study, job budget, rung caps, ranking evidence, confirmation count, winning delta,
runtime triplet, recipe path, W&B links, and verification commands. Call the result only
`training-signal-confirmed`. State explicitly that autoresearch ran no checkpoint evaluation,
promoted no checkpoint, and established no goal acceptance.
