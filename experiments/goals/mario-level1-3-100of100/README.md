# Mario Level1-3 100/100 Goal

This folder is the durable capsule for the active `mario-level1-3-100of100`
research objective.

Primary optimization metric: peak
`train/info/level_complete/rate/min/last`. For this single-level goal it should
match the Level1-3 source rate once the rolling source window is full; external
eval remains the promotion check.

Training jobs for this goal stop early once
`train/info/level_complete/rate/min/last > 0.99`, which is the strict 100/100
source-attempt success window for the single Level1-3 start state.

- `goal.json`: current goal contract, metric, seed protocol, cap, runtime, and
  promotion policy.
- `specs/`: checked-in queue payloads for candidate runs.
- `recipes/`: durable recipe evidence and operator-facing recipe notes.
- `reports/`: checked-in summaries or analysis reports for this goal.
- `decisions/`: checked-in decision records that should outlive chat history.

Generated checkpoints, W&B files, videos, and raw local logs should stay out of
source control. Prefer the repo-level ignored `runs/`, `logs/`, and `models/`
trees unless a goal-local ignored scratch directory is explicitly useful.
