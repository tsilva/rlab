# Mario Level1-2 100/100 Goal

This folder is the durable capsule for the active `mario-level1-2-100of100`
research objective.

- `goal.json`: current goal contract, metric, seed protocol, cap, runtime, and
  promotion policy.
- `specs/`: checked-in queue payloads for candidate runs.
- `recipes/`: durable recipe evidence and operator-facing recipe notes.
- `reports/`: checked-in summaries or analysis reports for this goal.
- `decisions/`: checked-in decision records that should outlive chat history.

Generated checkpoints, W&B files, videos, and raw local logs should stay out of
source control. Prefer the repo-level ignored `runs/`, `logs/`, and `models/`
trees unless a goal-local ignored scratch directory is explicitly useful.

```bash
UV_CACHE_DIR=.uv-cache uv run rlab-queue enqueue-train \
  --spec-file experiments/goals/mario-level1-2-100of100/specs/b85-b46-post21-five-seed-l12-confirm.json \
  --runtime-image-ref-file rlab-train-image.json
```
