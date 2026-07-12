# GPU Instances

This repo currently supports one-job Docker containers on registered local or
SSH Docker machines. Training jobs are created in the queue DB with `rlab
train`; Mac-side `rlab fleet shepherd` claims queued jobs and reconciles Docker
containers locally or on `beast-3` and `beast-2` over SSH. Do not use provider
launchers for this project while the beast path is being hardened.

## Quick Choice

| Use case | Target | Shape |
| --- | --- | --- |
| Highest-throughput Mario PPO screening | `rtx4090` / `beast-3` | 6 train containers |
| Lower-contention RTX4090 confirmation | `rtx4090` / `beast-3` | 3-4 train containers |
| Small-GPU batch screening | `rtx2060` / `beast-2` | 4 train containers |
| Faster RTX2060 turnaround | `rtx2060` / `beast-2` | 2 train containers |
| Smoke and queue/fleet debugging | `local-macbook` | 1 local Docker train container |

Concrete beast host operation, target mapping, and hard capacity live in
`experiments/machines.yaml`: backend, SSH/Docker access, payload/output paths,
env file, mounts, enforced
`max_parallel_containers` slot caps, and host runtime paths. Use the shepherd's
`--limit` option when a run intentionally needs a smaller operating shape.

## Standard Workflow

Queue work from checked-in goal recipe files:

```bash
rlab train \
  --recipe-file experiments/goals/<goal-slug>/recipes/<recipe>.yaml \
  --runtime-image-ref-file rlab-train-image.json
```

Inspect and reconcile local capacity from the MacBook:

```bash
rlab fleet status
rlab fleet ps
rlab fleet watch --machine beast-3
rlab fleet shepherd --machine beast-3 --once
```

For a manual recoverable one-job-per-container pass:

```bash
rlab fleet shepherd \
  --machine beast-3 \
  --once
```

In the job-container path, `watch --machine` is read-only: it shows machine
capacity, queued demand, launch rows, labeled containers, result presence, and
which rows need shepherd action. Use `shepherd --once` for a single
reconcile-and-fill pass, or omit `--once` for the long-running mutating
orchestrator. Shepherd reconciles, claims, launches, finalizes, streams a
line-oriented action log, and prunes stale Docker images from the host once no
active container or queued demand needs them. `rlab fleet shepherd --once` is
the only supported mutating repair-and-fill pass.

## Host Setup

Bootstrap each host after OS/Docker changes or when validating a new runtime
image:

```bash
rlab fleet setup-host \
  --host beast-3 \
  --runtime-image-ref-file rlab-train-image.json

rlab fleet setup-host \
  --host beast-2 \
  --runtime-image-ref-file rlab-train-image.json
```

The setup command verifies Docker, NVIDIA runtime support, persistent
directories, `.env.runner` permissions, digest pulls, and the container smoke
path. The beast hosts should remain simple Docker/GPU hosts; they do not run a
queue service and do not schedule experiments.

## beast-3 / RTX4090

- Machine: `beast-3`; queue target: `rtx4090`.
- Host resources: RTX4090, at least 12 CPUs, and at least 48 GB memory.
- Access: `ssh tsilva@beast-3`.
- Fleet role: primary screening and confirmation host.
- Enforced host capacity: `max_parallel_containers=6` in
  `experiments/machines.yaml`.
- Default operating shape: 6 train containers.
- Default runtime shape: provider and PyTorch thread defaults.
- Lower-contention shape: 3-4 workers.
- Current five-container benchmark expectation: about 6200 aggregate wall FPS
  for the current Mario PPO shape. Re-measure aggregate wall FPS after the
  six-container shape has enough steady-state samples.
- Docker command: configured in `experiments/machines.yaml`; currently
  `sudo -n docker`.
- Persistent root: `/home/tsilva/rlab`.
- ROM mount root: `/home/tsilva/roms`.

Use beast-3 for the run that decides the main research loop unless you are
intentionally testing small-GPU behavior.

## beast-2 / RTX2060

- Machine: `beast-2`; queue target: `rtx2060`.
- Host resources: RTX2060, at least 4 CPUs, and at least 8 GB memory.
- Access: `ssh -o HostKeyAlias=beast-2 tsilva@192.168.133.26` until hostname
  resolution is restored.
- Fleet role: cheaper small ablations, smoke jobs, and RTX2060-specific checks.
- Enforced host capacity: `max_parallel_containers=4` in
  `experiments/machines.yaml`.
- Default operating shape: 4 train containers.
- Default runtime shape: provider and PyTorch thread defaults.
- Fast-turnaround shape: 2 workers.
- Docker command: configured in `experiments/machines.yaml`; currently
  `sudo -n docker`.
- Persistent root: `/home/tsilva/rlab`.
- ROM mount root: `/home/tsilva/roms`.

The old `local-8332822-dirty` image tag was a k3s/containerd artifact. Use
pushed immutable GHCR digest refs for all comparable Docker fleet jobs.

## Local MacBook

- Machine and queue target: `local-macbook`.
- Backend: `local_docker`.
- Host resources: local CPU/MPS host.
- Use for queue-backed smoke tests and local fleet debugging.
- Default operating shape: 1 train container.
- Docker command: configured in `experiments/machines.yaml`; currently `docker`
  without `--gpus all`.
- Do not use local training throughput as evidence for beast concurrency.

## Operational Rules

- Keep train jobs profileless by default.
- Use immutable `docker:...@sha256:...` runtime image refs.
- Keep secrets in `.env` locally and `/home/tsilva/rlab/.env.runner` on hosts.
- Before every `docker_ssh` job claim, the launcher must refresh the shared
  `WANDB_API_KEY`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
  `AWS_S3_ENDPOINT_URL`, `AWS_REGION`, and `CHECKPOINT_BUCKET_URI` values in the
  host `.env.runner` from the repo-root `.env`. The sync is fail-closed, sends
  values only through SSH stdin, normalizes dotenv quotes for Docker, and
  preserves host-specific entries such as `ROM_PATH`.
- Do not print DB, W&B, or AWS/R2 secrets.
- Keep generated checkpoints, logs, videos, W&B files, caches, and scratch
  outputs under ignored paths such as `runs/`, `logs/`, `models/`, and `wandb/`.
- `rlab fleet` may remove old managed containers only when there are no
  pending or running jobs demanding that immutable runtime digest and run
  target, and no active launch or container uses the digest.
- In the recoverable job-container path, one container is one job attempt. The
  shepherd/launcher is the only mutating DB actor; the read-only watcher never
  claims, launches, releases, or finalizes jobs. The container reads a payload,
  writes `result.json`, uploads W&B/artifacts, and exits. Restarted shepherds
  reconcile DB launch rows, Docker labels, and durable output directories.
## Native Vector Runtime V2 Acceptance (2026-07-10)

The consolidated Mario runtime was compared against the deleted fused implementation from source
revision `5f732c1d` in a detached worktree, using the same installed Turbo providers, eight envs,
4096 timesteps, `n_steps=128`, `batch_size=256`, one PPO epoch, and seeds 101-103.

- Consolidated PPO SPS: `828.27`, `844.20`, `834.95`; median `834.95`.
- Fused PPO SPS: `802.16`, `822.29`, `815.33`; median `815.33`.
- Consolidated runtime was `2.41%` faster, passing the requirement that it be no more than `3%`
  slower than the fused path.
- Provider/runtime stepping overhead separately passed the `5%` gate for both
  `supermariobrosnes-turbo==0.2.20` and `stable-retro-turbo==1.0.1.post13`.
