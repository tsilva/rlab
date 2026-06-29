# GPU Instances

This repo currently supports local Docker fleet runners only. Training jobs are
created in the queue DB with `rlab jobs`; Mac-side `rlab fleet` reconciles
Docker containers on `beast-3` and `beast-2` over SSH. Do not use provider
launchers for this project while the beast path is being hardened.

## Quick Choice

| Use case | Target | Shape |
| --- | --- | --- |
| Highest-throughput Mario PPO screening | `rtx4090` / `beast-3` | 5 runner workers, `env_threads=4` |
| Lower-contention RTX4090 confirmation | `rtx4090` / `beast-3` | 3-4 runner workers, `env_threads=4` |
| Small-GPU batch screening | `rtx2060` / `beast-2` | 4 runner workers, `env_threads=2` |
| Faster RTX2060 turnaround | `rtx2060` / `beast-2` | 2 runner workers, `env_threads=4` |
| Smoke, debugging, playback | `local-macbook` | direct local CLI |

Machine-readable target defaults live in `experiments/instances.json`; these
use `default_workers` and `hardware_max_workers` for descriptive capacity.
Fleet host connection, mount details, and the enforced `max_workers` cap live in
`experiments/fleet.json`. Scheduling lanes and policy checks live in
`experiments/policies/capacity_policy.json`.

## Standard Workflow

Queue work from checked-in goal spec files:

```bash
UV_CACHE_DIR=.uv-cache uv run rlab train \
  --spec-file experiments/goals/<goal-slug>/specs/<spec>.json \
  --runtime-image-ref-file rlab-train-image.json
```

Inspect and reconcile local capacity from the MacBook:

```bash
UV_CACHE_DIR=.uv-cache uv run rlab fleet policy
UV_CACHE_DIR=.uv-cache uv run rlab fleet status
UV_CACHE_DIR=.uv-cache uv run rlab fleet ps
UV_CACHE_DIR=.uv-cache uv run rlab fleet plan
UV_CACHE_DIR=.uv-cache uv run rlab fleet reconcile
UV_CACHE_DIR=.uv-cache uv run rlab fleet watch
```

For a long-running local reconciliation loop:

```bash
UV_CACHE_DIR=.uv-cache uv run rlab fleet reconcile --watch --interval 30
```

For a live terminal dashboard that keeps each reachable beast host on the latest
successful train image and removes idle old managed containers:

```bash
UV_CACHE_DIR=.uv-cache uv run rlab fleet watch
```

After publishing a new train image, roll active hosts to the latest successful
digest:

```bash
UV_CACHE_DIR=.uv-cache uv run rlab fleet ensure-latest
```

To warm a host even before matching queue demand:

```bash
UV_CACHE_DIR=.uv-cache uv run rlab fleet ensure-runner \
  --host beast-3 \
  --image latest
```

Use `--profile <profile-id>` only for an intentionally profile-locked lane.
Default train jobs and runners should be profileless and locked by immutable
`runtime_image_ref` plus optional `run_target`.

## Host Setup

Bootstrap each host after OS/Docker changes or when validating a new runtime
image:

```bash
UV_CACHE_DIR=.uv-cache uv run rlab fleet setup-host \
  --host beast-3 \
  --runtime-image-ref-file rlab-train-image.json

UV_CACHE_DIR=.uv-cache uv run rlab fleet setup-host \
  --host beast-2 \
  --runtime-image-ref-file rlab-train-image.json
```

The setup command verifies Docker, NVIDIA runtime support, persistent
directories, `.env.runner` permissions, digest pulls, and the container smoke
path. The beast hosts should remain simple Docker/GPU hosts; they do not run a
queue service and do not schedule experiments.

## beast-3 / RTX4090

- Target: `rtx4090`, alias `beast-3`.
- Access: `ssh tsilva@beast-3`.
- Fleet role: primary screening and confirmation host.
- Enforced fleet capacity: `max_workers=5` in `experiments/fleet.json`.
- Default operating shape: 5 runner workers.
- Default runtime shape: `env_threads=4`, `torch_num_threads=1`.
- Lower-contention shape: 3-4 workers with `env_threads=4`.
- Current benchmark expectation: about 6200 aggregate wall FPS for the current
  Mario PPO shape.
- Docker command: configured in `experiments/fleet.json`; currently
  `sudo -n docker`.
- Persistent root: `/home/tsilva/rlab`.
- ROM mount root: `/home/tsilva/roms`.

Use beast-3 for the run that decides the main research loop unless you are
intentionally testing small-GPU behavior.

## beast-2 / RTX2060

- Target: `rtx2060`, alias `beast-2`.
- Access: `ssh -o HostKeyAlias=beast-2 tsilva@192.168.133.26` until hostname
  resolution is restored.
- Fleet role: cheaper small ablations, smoke jobs, and RTX2060-specific checks.
- Enforced fleet capacity: `max_workers=4` in `experiments/fleet.json`.
- Default operating shape: 4 runner workers.
- Default runtime shape: `env_threads=2`, `torch_num_threads=1`.
- Fast-turnaround shape: 2 workers with `env_threads=4`.
- Docker command: configured in `experiments/fleet.json`; currently
  `sudo -n docker`.
- Persistent root: `/home/tsilva/rlab`.
- ROM mount root: `/home/tsilva/roms`.

The old `local-8332822-dirty` image tag was a k3s/containerd artifact. Use
pushed immutable GHCR digest refs for all comparable Docker fleet jobs.

## Local MacBook

- Target: `local-macbook`, aliases `macbook` and `local`.
- Use for smoke tests, debugging, playback, and quick eval checks.
- Do not use local training throughput as evidence for beast concurrency.

## Operational Rules

- Keep train jobs profileless by default.
- Use immutable `docker:...@sha256:...` runtime image refs.
- Keep secrets in `.env` locally and `/home/tsilva/rlab/.env.runner` on hosts.
- Do not print DB, W&B, or AWS/R2 secrets.
- Keep generated checkpoints, logs, videos, W&B files, caches, and scratch
  outputs under ignored paths such as `runs/`, `logs/`, `models/`, and `wandb/`.
- `rlab fleet` may remove old managed containers only when there are no
  pending/running jobs for that container's profile/digest/target and no active
  queue lease owned by one of its workers.
