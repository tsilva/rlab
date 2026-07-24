# rlab training container

This image is the complete v1 runtime for one dstack training task. It contains
the learner and its in-container supervisor, but no ROMs, credentials,
checkpoints, W&B data, or run outputs.

The learner writes structured events and checkpoint notifications to local
SQLite. The supervisor is the only networked process: it owns the R2 writer
lease, uploads immutable checkpoints, dispatches Modal evaluations, and is the
sole W&B writer. A task is successful only after the terminal drain proves that
all ready checkpoints have terminal evaluations and W&B reached its durable
high-water mark.

## Image contract

The Dockerfile preserves three independently cacheable layers:

1. CUDA, PyTorch, and Triton.
2. The remaining frozen Python dependencies.
3. The exact-source rlab application overlay.

`uv.lock` is the dependency source of truth. The checked-in Linux lock
projections must remain disjoint and reconstruct the complete training
environment:

```bash
uv run --frozen --only-group train-image-build \
  python containers/train/lock_projection.py --check
```

Build and smoke locally:

```bash
docker buildx build \
  --platform linux/amd64 \
  -f containers/train/Dockerfile \
  -t rlab-train:local \
  --load \
  .

docker run --rm rlab-train:local
```

Published runs use only a verified immutable reference of the form
`docker:ghcr.io/tsilva/rlab/rlab-train@sha256:<digest>`. The image workflow
records the exact source SHA, runtime input hash, dependency digests, and final
digest. dstack submission refuses a dirty or unpublished source revision.

## Runtime mounts and secrets

A Mario task mounts the content-addressed host ROM cache read-only at
`/rom-cache`. The ROM bytes must match both the committed SHA-256 and the
provider identity in the run manifest. Future ephemeral hosts may instead fetch
the encrypted private asset during task setup, verify it, and remove it with
task storage.

The task receives scoped credentials at runtime:

- control-private R2: read/write authority and recovery journals;
- eval-private R2: intent/result reads plus per-run ROM staging;
- models-public R2: immutable public checkpoint publication;
- W&B: supervisor only;
- Modal: supervisor only.

The learner subprocess has these network credentials removed from its
environment.

## dstack operation

Checked-in task and fleet templates live under `ops/dstack/`. v1 schedules one
training task on one single-GPU host; it does not oversubscribe B3 with multiple
learner containers.

```bash
rlab experiment launch \
  --goal SuperMarioBros-Nes-v0/Level1-1 \
  --recipe ppo.yaml \
  --seed 123 \
  --run-description "Mario Level1-1 dstack run" \
  --compute local \
  --target b3

rlab experiment status --run <rlab-run-id> --json
rlab experiment follow --run <rlab-run-id>
rlab experiment logs --run <rlab-run-id>
rlab experiment cancel --run <rlab-run-id>
```

dstack owns placement, process logs, interruption reporting, cancellation, and
resource release. R2 receipts—not task exit status—own scientific success,
promotion, and resumable recovery.

Host image cleanup is a separate root-owned timer. It queries dstack for
active/queued immutable digests and removes only unused rlab-managed images,
preserving active containers and demanded digests.
