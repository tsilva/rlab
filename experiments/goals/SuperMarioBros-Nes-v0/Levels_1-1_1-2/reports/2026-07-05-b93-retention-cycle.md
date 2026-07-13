# 2026-07-05 B93 Retention Cycle

## Goal

Train one task-conditioned PPO policy over `Level1-1` and `Level1-2` in a
single mixed run until both per-level 100-attempt clear-rate windows satisfy the
goal gate:

```text
train/info/level_complete/rate/min/last > 0.99
```

The active train target is `beast-3` with the repo-local enforced cap of five
train containers.

## Live Decisions

- Kept `b93-slowent-l12bias-extended` seed 201 running because it remains the
  strongest active candidate, but it has not solved the goal. Its observed live
  bottleneck during this cycle moved in the approximate `0.88` to `0.92` range,
  with both Level1-1 and Level1-2 nonzero and high but still below the gate.
- Canceled mature `b93-fastlowent-finish` seeds 312 and 313 after they failed to
  approach the historical B93 reproduction signal. Seed 313 was around 2.0M
  steps with roughly `0.03` to `0.05` per-level rates. Seed 312 reached roughly
  2.8M steps and then regressed to about `0.07` bottleneck.
- Launched `b93-gentle-retain` seed 320 as the next active arm. This arm keeps
  the historical B93 reward, task conditioning, 40/60 Level1-1/Level1-2 sampling,
  entropy schedule, and network shape, but narrows the PPO update envelope to
  test whether late-window regressions are update-volatility rather than
  exposure failures.
- Replenished the pending buffer with additional B93-family seeds so the
  fleet service can backfill immediately after future cancels.
- Later in the same cycle, canceled `b93-fastlowent-finish` seed 314 after it
  reached about 1.86M steps with only about `0.07` bottleneck clear-rate. This
  reinforced that the lower entropy-floor arm was not improving balanced
  discovery.
- Canceled `b93-slowent-l12bias-extended` seed 215 after it reached about
  1.59M steps with current per-level clear-rate windows still at `0.00`/`0.00`.
  Seed 201 remains the long B93 trajectory, so freeing this weak duplicate slot
  improves coverage of the retention hypothesis.
- Backfilled the two freed slots with `b93-gentle-retain` seeds 321 and 322,
  then queued additional `b93-gentle-retain` seeds 325 and 326 to restore the
  pending buffer.
- A sampled W&B history check for long-running seed 201 found a best observed
  bottleneck of about `0.97` near 11.68M steps, with Level1-1 around `0.97` and
  Level1-2 around `0.98`. The live run remained below the `>0.99` gate, so this
  is not a hidden solve.
- Canceled `b93-slowent-l12bias-extended` seed 216 after it reached about
  1.65M steps with only about `0.03` bottleneck. This kept seed 201 as the long
  B93 trajectory while freeing a duplicate weak slot.
- Added `b93-lrdecay-retain`, which preserves B93 discovery settings but decays
  learning rate from `1.5e-4` to `7.5e-5` over 10M steps. The hypothesis is that
  B93's constant late updates are too disruptive once both levels are discovered.
  Queued seeds 330 and 331 for this arm.

## Current Hypothesis

The reproduced B93 family can discover both levels in one shared policy, and
the remaining failure mode looks like retention/oscillation of the rolling
per-level windows rather than missing Level1-2 exposure. Lowering the entropy
floor earlier (`b93-fastlowent-finish`) did not improve balanced discovery in
the observed mature seeds. The next decisive comparisons are whether
`b93-gentle-retain` keeps both source-level windows high at the same time, and
whether `b93-lrdecay-retain` preserves B93's discovery speed while reducing late
window oscillation.

## Watchpoints

- Continue using `train/info/level_complete/rate/min/last` as the live
  bottleneck metric, with `from/0-0/rate` and `from/0-1/rate` to identify which
  level is limiting.
- Do not prune fresh B93/gentle seeds before they reach the historical discovery
  window unless both per-level clear counts remain effectively zero and reward
  has flattened.
- If a run crosses the training gate, wait for post-train eval when enabled and
  rank by `eval/done/level_change/from_rate/min`.
