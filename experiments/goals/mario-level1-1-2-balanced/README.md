# Level1-1/Level1-2 Balanced High-Watermark Search

Goal: find a mixed-policy PPO recipe for `SuperMarioBros-Nes-v0` whose two
seeds in the same batch maximize peak
`train/info/level_complete/rate/min/last` within 5M steps. Use
`train/info/level_complete/rate/mean/last` only as a companion signal or
tiebreaker.

The metric is a training high-watermark, not an early-stop trigger. Queue jobs
through checked-in specs and let each run reach the 5M cap. Recover peak values
from W&B history, because run summaries can show only the final value after a
collapse.

Current near-miss baseline:

- Spec: `specs/b86-b74current-l11l12-latest-five-seed.json`
- W&B group: `b86-l11l12-b74current-latest-five-seed`
- Best seed: `b86_l11l12_b74current_s195_20260627T091726Z`
- Peak: `train/info/level_complete/rate/min/last = 0.80` at about `4,070,256`

Queue new batches with:

```bash
UV_CACHE_DIR=.uv-cache uv run rlab-queue enqueue-train \
  --spec-file experiments/goals/mario-level1-1-2-balanced/specs/<spec>.json \
  --latest-image
```

Then reconcile beast-3 capacity:

```bash
UV_CACHE_DIR=.uv-cache uv run rlab-fleet plan
UV_CACHE_DIR=.uv-cache uv run rlab-fleet reconcile
```
