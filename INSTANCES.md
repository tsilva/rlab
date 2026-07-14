# GPU Instances

This repo supports one-job Docker containers on registered local or SSH Docker
machines. Every queued `rlab train` job names one exact machine. A single
Mac-side launchd service runs short reconciliation passes for `local-macbook`,
`beast-3`, and `beast-2`; the runner machines remain simple SSH/Docker hosts.
Do not use provider launchers for this project while the beast path is being
hardened.

## Quick Choice

| Use case | Machine | Shape |
| --- | --- | --- |
| Highest-throughput Mario PPO screening | `beast-3` | 6 train containers |
| Lower-contention RTX4090 confirmation | `beast-3` | 3-4 train containers |
| Small-GPU batch screening | `beast-2` | 4 train containers |
| Faster RTX2060 turnaround | `beast-2` | 2 train containers |
| Smoke and queue/fleet debugging | `local-macbook` | 1 local Docker train container |

Concrete host operation and hard capacity live in
`experiments/machines.yaml`: backend, SSH/Docker access, payload/output paths,
env file, mounts, enforced `max_parallel_containers` slot caps, and host runtime
paths. Use the durable fleet capacity override when a machine intentionally
needs a smaller operating shape.

## Standard Workflow

Queue work from checked-in goal recipe files:

```bash
rlab train \
  --recipe-file experiments/goals/<goal-slug>/recipes/<recipe>.yaml \
  --machine beast-3 \
  --runtime-image-ref-file rlab-train-image.json
```

Install and inspect the Mac-side service, then observe jobs:

```bash
rlab fleet service install
rlab fleet service status --json
rlab jobs status --machine beast-3 --json
```

Mutating commands wake the service immediately; its 30-second launchd interval
is the recovery path for missed wake-ups and remote completion. Each invocation
loads current source, performs one bounded pass, and exits. launchd does not
overlap invocations of the same service label.

For a lower-contention machine shape:

```bash
rlab fleet capacity --machine beast-3 --set 4
rlab fleet capacity --machine beast-3 --reset
```

Use `rlab fleet drain --machine <name>` to stop new claims without killing
running jobs and `rlab fleet resume --machine <name>` to admit work again.
`rlab jobs status` is observational. The service is the only normal mutating
reconciler: it claims, launches, finalizes, and prunes stale Docker images after
no active container or exact-machine queued demand needs them.

## Modal CPU Checkpoint Evaluation

Modal is a backend-bound evaluation lane owned by the same Mac fleet service; it is not a
registered training machine and must not be added to `experiments/machines.yaml`. Its checked-in
deployment, timeout, budget, and concurrency contract is `experiments/modal_eval.yaml`. The hard
orchestration ceiling and the independent Modal `max_containers` guard are both 20, while rollout
starts at effective capacity 1 and must be promoted through 2 before 20. Modal is the default for
new queue-backed jobs at capacity 1; do not raise that capacity until the parity, interruption,
cap-2, and cost canaries pass.

```bash
rlab eval modal status
rlab eval modal preflight \
  --runtime-image-ref docker:ghcr.io/tsilva/rlab/rlab-train@sha256:<digest> \
  --game <game-id>
rlab eval modal drain
rlab eval modal resume --capacity 1
rlab eval modal retry <eval-job-id>
rlab eval modal recover <train-job-id>
rlab eval modal assets sync --game <game-id>
rlab eval modal smoke-local
```

The selected backend is materialized in the queue row and never changes for that job. Use
`rlab train ... --checkpoint-eval-backend local` only for an explicit fallback. `preflight` fails
closed unless the additive PostgreSQL schema, active capacity, private ROM object, local Modal
credentials, and exact runtime-specific deployment are all present.

The train-image release workflow deploys and startup-probes the exact digest-specific Modal app
before publishing `rlab-train-image.json`; a failed deployment leaves the previous successful
runtime as the default. Every new Modal-backed submission and explicit retry repeats the full
preflight before writing queue rows, so an explicitly supplied undeployed digest also fails closed.
Use `rlab eval modal recover <train-job-id>` only after a terminal train job reports
`awaiting_artifact_recovery`. Recovery drains pending artifacts inside the runtime container and
rejects active, finalizing, complete, or otherwise ineligible eval runs without changing their state.

PostgreSQL is the only wait queue. The service never submits work beyond the effective capacity,
reserves worst-case cost before dispatch, and leaves budget-blocked jobs pending for operator
inspection. Draining stops new Modal calls without stopping training. Checkpoint models, metadata,
announcements, attempts, and decisions are immutable R2 objects; Modal return values are only
receipts. Runtime-specific apps are deployed from CI as `rlab-eval-<digest-prefix>` from the exact
shared train/eval image digest. Worker retries are disabled; the fleet service may create one
separately recorded second attempt for transient failures. Modal 1.5 exposes only single-use or
unbounded-reuse containers. V1 uses warm-container reuse with a 60-second scale-down window because
single-use containers impose the full cold-start cost on every evaluation; the global call cap and
dollar budgets remain the spend guards. There is no enforceable ten-input container lifetime until
Modal supports `max_inputs > 1`.

The 2026-07-13 Breakout cap-1 canary used runtime
`sha256:ed1d6342ba2ba90c9832fb6e088a93c680dad09fcbda0ec71b8021f94a484498` and train job 8.
Training completed 131,072 steps at 5,432 reported FPS without a local eval worker. Modal accepted
the two-episode, two-lane `vector-lane-v1` promotion evaluation in 26.0 seconds for an estimated
$0.00450, uploaded immutable R2 evidence, promoted eval job 4, and projected `eval/source=modal`
into the exact finished W&B run. The deployment workflow's startup probe and the preflight command
both passed for the digest-specific app before the canary was admitted.

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

- Machine: `beast-3`.
- Host resources: RTX4090, at least 12 CPUs, and at least 48 GB memory.
- Access: `ssh tsilva@beast-3`.
- Fleet role: primary screening and confirmation host.
- Enforced host capacity: `max_parallel_containers=6` in
  `experiments/machines.yaml`.
- Default operating shape: 6 train containers.
- Default runtime shape: goal-declared provider arguments and PyTorch thread defaults.
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

- Machine: `beast-2`.
- Host resources: RTX2060, at least 4 CPUs, and at least 8 GB memory.
- Access: `ssh -o HostKeyAlias=beast-2 tsilva@192.168.133.26` until hostname
  resolution is restored.
- Fleet role: cheaper small ablations, smoke jobs, and RTX2060-specific checks.
- Enforced host capacity: `max_parallel_containers=4` in
  `experiments/machines.yaml`.
- Default operating shape: 4 train containers.
- Default runtime shape: goal-declared provider arguments and PyTorch thread defaults.
- Fast-turnaround shape: 2 workers.
- Docker command: configured in `experiments/machines.yaml`; currently
  `sudo -n docker`.
- Persistent root: `/home/tsilva/rlab`.
- ROM mount root: `/home/tsilva/roms`.

The old `local-8332822-dirty` image tag was a k3s/containerd artifact. Use
pushed immutable GHCR digest refs for all comparable Docker fleet jobs.

## Local MacBook

- Machine: `local-macbook`.
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
- The service may remove old managed containers and images only when no queued
  job on that exact machine demands the immutable runtime digest and no active
  launch or container uses it.
- In the recoverable job-container path, one container is one stable job
  launch. The service is the only normal mutating DB actor; status and log
  commands never claim, launch, cancel, or finalize jobs. The container reads a
  payload and atomically publishes `result.json`. Queue-backed run directories live under the
  host-mounted launch output so a checkpoint coordinator can recover incomplete R2 uploads after
  the training container exits. When Modal eval is selected, the live W&B publisher owns training
  telemetry plus imported `eval/screen/*` and `eval/confirm/*` frames; it does not upload checkpoint
  artifacts. The Mac-side terminal projector owns promotion-quality `eval/full/*` evidence.
  Later service passes reconcile DB launch rows, Docker labels, and durable
  output directories without creating a replacement launch.
## Train Image Build Baseline (2026-07-14)

Before dependency-image rebasing, six source-only GitHub Actions builds had a median runtime
image step of about 83 seconds. In run `29314888343`, BuildKit spent 38.2 seconds downloading and
26.3 seconds extracting the cached 3.12 GB Python dependency layer, while building the `rlab`
package itself took 1.3 seconds.

The first same-builder rebased canary, run `29321462616`, created the new dependency-keyed image
once in 114 seconds and then built the runtime image in 9 seconds. Cold-builder run `29321693205`
exposed a repeated linked-copy destination that still materialized the dependency filesystem.
After collapsing the application files into one scratch overlay, fresh-runner run `29321898579`
skipped the dependency build, built and pushed the runtime image in 10 seconds, and completed the
workflow in 37 seconds. That is an approximately 88% runtime-step reduction from the prior median;
the accepted build transferred no 3.12 GB dependency blob and exported no runtime cache.

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

## Provider Contract Preflight Acceptance (2026-07-14)

The `retro-env-throughput-mario-l11` profile was run sequentially on the same idle Mac against
revision `51f981d4` and the provider-contract working tree based on `60c4f352`, using the installed
`stable-retro-turbo==1.0.1.post29`. Provider/runtime overhead before and after was `2.61%`/`3.41%`
at one env, `-0.17%`/`1.36%` at 16 envs, and `0.66%`/`0.60%` at 32 envs. Every case passed the
profile's `5%` runtime-overhead gate; after-change median runtime SPS was also higher at all three
env counts.
