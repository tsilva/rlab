---
name: autoresearch
description: Tune one checked-in rlab SB3 PPO or A2C recipe against its goal for the earliest reproducible accepted checkpoint. Use when the user points to a recipe and asks to tune, optimize, autoresearch, improve sample efficiency, find the best hyperparameters, or make success stable across training seeds. Runs a bounded, durable, parallel beast-3 search, confirms the winner on untouched seeds, and patches only the pointed leaf recipe.
---

# Autoresearch

Tune the pointed recipe until the evidence identifies the most sample-efficient seed-stable recipe allowed by the bounded protocol. A direct invocation authorizes up to 48 reserved queue jobs on `beast-3`; it does not authorize infrastructure repair, capacity changes, runtime builds/deployments, cancellation of unrelated work, commits, or pushes.

## Load the contracts

Before acting:

1. Read `SPECS.md` with `$specs-author` and do not edit it unless separately authorized.
2. Read `INSTANCES.md`, the selected goal, the selected leaf recipe, and every composed source named by `_composition.source_files`.
3. Read [search-policy.md](references/search-policy.md) completely.
4. Read and use `$launch-experiment` in observe mode for every launch and monitor. The bounded-controller exception in that skill permits the next preregistered cohort only at a clean round barrier.

Resolve exactly one checked-in goal and one checked-in recipe. If the user names only a recipe, compose it and use its one goal contract. Stop for a genuinely ambiguous goal; do not guess.

## Initialize or resume

Run from the repository root:

```bash
uv run python .codex/skills/autoresearch/scripts/study.py init \
  --goal <goal-file> \
  --recipe <recipe-file>
```

Use the returned `study` path for every later helper command. The helper holds a discovery lock across scan/create, resumes the sole matching incomplete study, pins `HEAD` and all composed source hashes, and rejects unsupported recipes. `runs/autoresearch/<study>/study.json` is the only authoritative controller state; do not hand-edit it.

V1 accepts only goal-owned, checkpoint-evaluated `sb3.ppo` or `sb3.a2c`. Reject JERK, planned/unavailable backends, first-training-success acceptance, `checkpoint_eval_backend=none`, missing stop-on-acceptance, or an ambiguous composition.

At any time, ask for exactly one deterministic action:

```bash
uv run python .codex/skills/autoresearch/scripts/study.py next --study <study>
```

Obey that action. Never skip a barrier or infer state from memory.

## Read beast-3 capacity

Before every baseline, search, or confirmation wave, run:

```bash
rlab experiment status --machine beast-3 --json
```

Use `capacity.effective_capacity` and `capacity.active_reservations`. Do not set/reset capacity, drain/resume the machine, cancel work, or displace unrelated jobs. If `available_slots` is zero, keep the study active and wait.

## Reserve before enqueue

For the baseline:

```bash
uv run python .codex/skills/autoresearch/scripts/study.py reserve-wave \
  --study <study> --phase baseline \
  --effective-capacity <capacity> --active-reservations <active>
```

For a search round, propose one to three new candidate deltas as a JSON list. Each candidate changes one coherent numeric group and follows the policy reference. Then reserve:

```bash
uv run python .codex/skills/autoresearch/scripts/study.py reserve-wave \
  --study <study> --phase search \
  --candidates-json '[{"delta":{"learning_rate":0.0002}}]' \
  --effective-capacity <capacity> --active-reservations <active>
```

For confirmation, the helper selects the incumbent and the fixed five untouched seeds:

```bash
uv run python .codex/skills/autoresearch/scripts/study.py reserve-wave \
  --study <study> --phase confirmation \
  --effective-capacity <capacity> --active-reservations <active>
```

Reservation is durable and counts uncertain submissions against the 48-job budget. Search always retains five jobs for confirmation. Confirmation intentionally uses five slots, never a speculative sixth.

## Reconcile and launch idempotently

For each reserved cohort, first inspect its deterministic `batch_id`:

```bash
rlab experiment status --batch <batch-id> --json
```

- If the exact expected rows already exist, record that status payload.
- If zero rows exist, execute the preregistered launch command.
- If a partial or mismatched cohort exists, pause and report it; never fill, replace, retry, or duplicate it ad hoc.

Record a launch or recovered status payload:

```bash
uv run python .codex/skills/autoresearch/scripts/study.py record-launch \
  --study <study> --submission-key <key> --payload-file <json-file>
```

Every generated command includes `--from-head`, `--existing-runtime-only`, a deterministic request ID, the fixed seeds, and candidate overrides. The baseline captures the exact runtime image/input/build-source triplet; every later wave pins all three guards. A missing existing receipt or Modal readiness pauses before queue mutation. Never remove these flags and never start a build or deployment.

When a wave contains independent candidate commands, start all of those commands concurrently so free beast-3 slots are consumed immediately and safe excess work queues. Do not serialize independent candidates. After each launch returns, record its payload and immediately start one `rlab experiment follow --run <id> --jsonl` process per run, concurrently, under `$launch-experiment`.

## Monitor every run

Stay attached until every run in the round emits `terminal` and every dispatched training-run investigator completes.

- Report each `wandb_url` immediately.
- On `potential_bug`, dispatch the read-only `training_run_investigator`, continue all monitors, and launch no later wave.
- On `attention_required`, record the event, report the budget block, keep all monitors attached, dispatch no investigator for that event, and launch no later wave.
- Do not repair, retry, cancel, restart, or mutate infrastructure.

Record each complete `terminal` event directly; its submission key, seed, and run ID are self-describing:

```bash
uv run python .codex/skills/autoresearch/scripts/study.py record-terminal \
  --study <study> --event-file <terminal-event.json>
```

For either attention class, persist the pause:

```bash
uv run python .codex/skills/autoresearch/scripts/study.py record-attention \
  --study <study> --event-file <event.json>
```

If the same monitored run later proves that a budget block cleared, or a pinned source was restored exactly, resume with recorded evidence:

```bash
uv run python .codex/skills/autoresearch/scripts/study.py resume \
  --study <study> --reason '<resolved evidence>'
```

Do not resume a cancellation, operational failure, observer failure, unverified acceptance, or potential bug without new user authority and a separately supported resolution.

## Close barriers and continue

After all paired baseline/search cohorts in a round are terminal:

```bash
uv run python .codex/skills/autoresearch/scripts/study.py close-round \
  --study <study> --round <round>
```

The helper ranks only accepted-and-remotely-verified evidence, censors failures at the frozen effective training cap, chooses the lower median promoted step, then lower worst step, and uses candidate ID only for an exact tie. A tie does not reset the stale-round counter.

After exactly five confirmation runs are terminal:

```bash
uv run python .codex/skills/autoresearch/scripts/study.py close-confirmation --study <study>
```

The candidate wins only with at least four accepted-and-remotely-verified seeds. A failed holdout is a binary exclusion: do not inspect or use per-seed holdout details to propose candidates. The helper excludes it, recomputes the incumbent, and resets the stale counter. Continue only with the next action and remaining budget.

Stop search after three consecutive rounds without evidence improvement or when the reservation budget cannot support another valid action. If no confirmation candidate remains, run `finish-no-winner`, do not edit the recipe, and report the evidence.

## Apply the winner in two phases

Patch only the pointed leaf recipe. Add only the winning `train.backend.config` numeric delta; preserve comments and composition, and omit study IDs, descriptions, campaign IDs, seed lists, and other trace fields.

Before editing the recipe, create an exact planned postimage file under the generated study directory using `apply_patch`, then preregister it:

```bash
uv run python .codex/skills/autoresearch/scripts/study.py prepare-apply \
  --study <study> --postimage-file <planned-recipe-file>
```

Only after the state is `apply_pending`, apply the exact same patch to the pointed recipe with `apply_patch`. Then run:

```bash
uv run python .codex/skills/autoresearch/scripts/study.py complete-apply --study <study>
```

This accepts only the pinned preimage or exact preregistered postimage, recomposes the full goal/recipe, proves every frozen train field is unchanged, verifies the winner materialized, and writes a generated evidence report. If the confirmed baseline has no delta, record the no-op and leave the recipe unchanged.

Run targeted tests for the helper and all affected repository interfaces. Leave the winning recipe patch uncommitted. Report the study path, job budget used, ranking evidence, five-seed confirmation count, winning delta or no-winner result, runtime triplet, recipe path, W&B links, and verification commands without credentials or presigned URLs.
