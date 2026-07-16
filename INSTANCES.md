# GPU Instances

This repo supports one-job Docker containers on registered local or SSH Docker
machines. Every queued `rlab train` job names one exact machine. A single
Mac-side launchd controllers continuously reconcile machine, evaluation, and
W&B publication state for `local-macbook`, `beast-3`, and `beast-2`; the runner
machines remain simple SSH/Docker hosts.
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

Queue work by composing one checked-in goal contract with one reusable recipe:

```bash
rlab train \
  --goal-file experiments/goals/<goal-slug>/_goal.yaml \
  --recipe-file experiments/recipes/<family>/<recipe>.yaml \
  --machine beast-3
```

Install and inspect the Mac-side service, then observe jobs:

```bash
rlab fleet service install
rlab fleet service watch
rlab fleet service status --json
rlab runs status --machine beast-3 --json
```

launchd supervises three independent controllers: machine reconciliation,
evaluation/promotion, and the per-run W&B publisher manager. Each polls
PostgreSQL every two seconds and holds a macOS sleep assertion while it owns
nonterminal work. Advisory locks prevent duplicate claims after a controller or
Mac restart.

`rlab fleet service watch` is the normal read-only operational view. On an interactive terminal it
opens a responsive dashboard; `--once`, `--plain`, and `--json` provide scriptable output modes.
The dashboard reads launchd registration plus authoritative PostgreSQL state; it never mutates or
repairs queue, Docker, SSH, Modal, or W&B state. It reports only failures that still block an active
run as needing attention. Use
`rlab fleet service logs --follow` for raw events and `rlab runs status` for exact job history.

`rlab fleet service status --json` exits nonzero when the last pass is stale or degraded, and
`rlab fleet service doctor` includes the last-pass result rather than treating a merely loaded
LaunchAgent as healthy. Two consecutive degraded or failed passes trigger a macOS notification;
recovery triggers a second notification, and repeated identical failures are rate-limited to once
per hour. A minimal standard-library launchd entrypoint records and immediately notifies failures
that prevent the main reconciler from importing. Idle-host maintenance attempts are also limited to once per hour so an offline unused
host cannot stretch every reconciliation pass; queued work still wakes that exact host lane on
every pass.

For a lower-contention machine shape:

```bash
rlab fleet capacity --machine beast-3 --set 4
rlab fleet capacity --machine beast-3 --reset
```

Use `rlab fleet drain --machine <name>` to stop new claims without killing
running jobs and `rlab fleet resume --machine <name>` to admit work again.
`rlab runs status` is observational. The three controllers are the only normal
mutating reconcilers. The machine controller claims and launches jobs and prunes
stale Docker images after no active container or exact-machine queued demand
needs them.

## Modal CPU Checkpoint Acceptance

Modal is a backend-bound evaluation lane owned by the Mac evaluation controller; it is not a
registered training machine and must not be added to `experiments/machines.yaml`. Its checked-in
deployment, timeout, budget, and concurrency contract is `experiments/modal_eval.yaml`. The
independent hard safety ceiling is 20 and the effective acceptance-eval capacity is 3. Admission
uses the checked-in workload policy in `experiments/eval_capacity.yaml`: an eval-enabled train job
is claimed only when total reserved load remains at or below 80% of effective Modal capacity.

```bash
rlab eval modal status
rlab eval modal preflight \
  --runtime-image-ref docker:ghcr.io/tsilva/rlab/rlab-train@sha256:<digest> \
  --game <game-id>
rlab eval modal drain
rlab eval modal resume --capacity 3
rlab eval modal retry <eval-job-id>
rlab eval modal retry-projection <train-job-id>
rlab eval modal recover <train-job-id>
rlab eval modal abandon <train-job-id>
rlab eval modal assets sync --game <game-id>
rlab eval modal smoke-local
```

The selected backend is materialized in the queue row and never changes for that job. Use
`rlab train ... --checkpoint-eval-backend local` only for an explicit fallback. Use `none` only for
a smoke/debug submission that does not need eval-owned early stopping, checkpoint promotion, or goal
acceptance. `preflight` fails closed unless the additive PostgreSQL schema, active capacity, private
ROM object, R2 evidence path, local Modal credentials, and exact runtime-specific deployment are
all present. Acceptance evaluation never captures video; representative replay remains a
release-time workflow.

The train-image workflow runs on every push to `main` and publishes an exact-source version-5
`rlab-train-image.json` as soon as the immutable image exists. Runtime images are keyed by the
runtime overlay plus the immutable dependency digest rather than by commit: goal, recipe, test, and
unrelated documentation changes therefore reuse a proven digest. The receipt also records the GPU,
dependency, and overlay keys and exact plan/base hashes. A reused runtime preserves its original
`runtime_build_source_sha`; the new receipt still records the exact pushed `source_sha`.
New fingerprints build an image and deploy the digest-specific Modal app. Reused fingerprints skip
both operations but still startup-probe the existing app and publish an exact-source
`rlab-modal-eval-readiness.json`. Local and `none` submissions may proceed from the early image
receipt; Modal submissions wait for matching readiness and repeat the full live preflight before
writing queue rows. Image workflow failure during the later Modal stage does not invalidate an
already published image receipt, but it does block Modal-backed submissions for that digest.
Use `rlab eval modal recover <train-job-id>` only after a terminal train job reports
`awaiting_artifact_recovery`. Recovery drains pending artifacts inside the runtime container and
rejects active, finalizing, complete, or otherwise ineligible eval runs without changing their state.
Use `rlab eval modal abandon <train-job-id>` after inspecting a failed, finalization-failed, or
canceled train whose evaluation remains nonterminal. It preserves uploaded evidence while canceling
undispatched evaluation work and closes the evaluation run with the matching terminal outcome.

PostgreSQL is the wait queue, orchestration authority, and transient telemetry mailbox. The
evaluation controller never submits beyond effective capacity, reserves worst-case cost before
dispatch, and leaves budget-blocked jobs pending for operator inspection. Draining stops new Modal
calls without stopping training. Checkpoint models, metadata, immutable episode evidence, and raw
Modal results are immutable R2 objects; accepted attempts, retained stop commands, artifact
locations, decisions, and publication cursors live in PostgreSQL. Each checkpoint has one logical
acceptance job and at most two immutable attempts; a valid rejection is successful execution and is
never retried. Runtime-specific apps are deployed from CI as `rlab-eval-<digest-prefix>` from the exact
shared train/eval image digest. Modal 1.5 exposes only single-use or
unbounded-reuse containers. V1 uses warm-container reuse with a 60-second scale-down window because
single-use containers impose the full cold-start cost on every evaluation; the global call cap and
dollar budgets remain the spend guards. There is no enforceable ten-input container lifetime until
Modal supports `max_inputs > 1`.

Checkpoint mailbox announcements and ready promotion projections are ingested in bounded batches.
The publisher manager starts exactly one isolated W&B SDK owner for each active run; its concurrency
is independent of the three-call Modal limit. Publisher work never blocks eval reconciliation or
stop delivery; publication completion and failure remain durable finalization-only state.
Each run retains its session advisory lock, so concurrent publication cannot interleave writers for
the same W&B run. Neon queue and mailbox connections use TCP keepalives and a 30-second user timeout
so a laptop sleep or network transition fails the pass promptly and is retried with a fresh
connection.

The evaluation controller inventories owned `rlab-eval-<12-hex>` deployments hourly and stops at most ten
zero-task apps per pass after a 24-hour grace period. It protects the latest runtime and every app
referenced by nonterminal training, evaluation, recovery, queued, or active-attempt work; unrelated
Modal apps are never eligible. Cleanup fails closed and reports separately from evaluation health.
If a reused runtime's app was stopped, CI treats only Modal `NotFoundError` as a redeployment signal,
redeploys that digest-specific app, and repeats the startup probe; authentication and network errors
remain failures rather than being reinterpreted as absence.

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

Drain a host before intentionally powering it off. A drained host with no active launches is not
contacted by reconciliation, so an offline spare does not degrade unrelated active work; active
launches remain observable and prevent that skip until they become terminal.

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
- Last measured five-container reference: about 6200 aggregate wall FPS for the
  then-current Mario PPO shape. Re-measure aggregate wall FPS after the
  six-container shape has enough steady-state samples.
- Docker command: configured in `experiments/machines.yaml`; currently
  `sudo -n docker`.
- Persistent root: `/home/tsilva/rlab`.
- ROM mount root: `/home/tsilva/roms`.
- Prewarming: enabled. The Mac fleet service pulls and probes the latest successful main-runtime
  receipt without reserving a training slot. Exactly that latest digest is temporary cleanup
  demand; a superseded digest is pruned once no queued or active job requires it. Prewarm failures
  appear in fleet-service health and do not stop running jobs. A failed prewarm also suppresses
  runtime-image pruning for that pass, preserving the last known good host image while inactive
  container cleanup continues.

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
- Prewarming: disabled initially.

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
- Prewarming: disabled initially.

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
- In the recoverable job-container path, one container is one provider-neutral worker attempt.
  The service owns orchestration and W&B publication; status and log
  commands never claim, launch, cancel, or finalize jobs. The container reads a
  payload and atomically publishes `result.json`. It receives R2 credentials plus a restricted,
  expiring Neon attempt token, never W&B credentials. Queue-backed run directories live under the
  host-mounted launch output while active; SQLite is deleted after the final Neon watermark is
  acknowledged. Fleet publishes training and all evaluation protocols into the preassigned W&B run,
  keeps that remote run active while queue work is nonterminal, and finishes it only after terminal
  publication. Managed training containers have a five-minute Docker stop timeout so an orderly
  daemon or host shutdown gives `run-job` enough time to stop telemetry workers and atomically write
  `result.json`; sudden power loss remains externally unrecoverable.
  Later service passes reconcile DB launch rows, Docker labels, and durable
  output directories.
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

The 3.12 GB blob is the Linux x86-64 Python artifact set, not rlab source. For the current lock,
the 121 selected archives total 3.108 GB: the 20 PyTorch/Triton/CUDA/NVIDIA archives contribute
2.732 GB and the other 101 train-runtime archives contribute 0.376 GB. The largest individual
archives are Torch (532 MB), cuBLAS (423 MB), cuDNN (366 MB), cuFFT (214 MB), NCCL (206 MB),
Triton (202 MB), and cuSOLVER (201 MB).
The current image contract isolates those packages in the content-addressed
`rlab-train-gpu:build-<gpu-key>` foundation, installs the remaining projected train dependencies
in `rlab-train-dependencies:build-<dependency-key>`, and adds source as a linked runtime overlay.
Branch pushes prebuild changed foundations before merge. Keys depend only on their own layer inputs
and the immutable digest directly below them; mutable maximal `buildcache` tags are not used.
Therefore source changes should publish a small overlay, ordinary dependency changes should retain
the GPU foundation, and only an actual GPU plan or GPU-stage change should require the multi-GB
transfer again on a host that does not already contain that digest.

The faster-publishing implementation was validated locally on 2026-07-16 before its CI canary.
The deterministic train projection contains 121 packages and is exactly partitioned into 20 GPU
packages plus 101 non-GPU packages. A genuinely cold Linux/amd64 GPU build on the arm64 development
host took 564.44 seconds under emulation: package preparation took 438 seconds, installation took
14.77 seconds, and the local image export took 108 seconds. The cold non-GPU overlay prepared its
packages in 33.68 seconds, installed them in 3.27 seconds, finished the build stage in 44.9 seconds,
and attached the linked overlay in 2.2 seconds. The dependency solve resolved the published GPU base
from metadata without extracting it; the local dangling-image exporter later tried to materialize
the remote parent, so that exporter run is not registry-push acceptance evidence.

With the exact GPU-only base and non-GPU overlay cached, a complete merged runtime assembled locally
in 3.52 seconds. A synthetic source-only invalidation took 12.78 seconds under Linux/amd64 emulation,
including 9.7 seconds to rebuild the `rlab` wheel; the prior native GitHub runner evidence remains the
relevant expectation for the at-most-10-second runtime assembly gate. That same-path overlay passed
`uv pip check` for all 121 packages, the container smoke, and imports for Torch, Stable Retro, SB3,
OpenCV, Numba, W&B, Breakout, and Mario.

The first published cold canary, run `29495731114`, exposed one remaining composition bug: both the
GPU foundation and non-GPU overlay targeted `/root/rlab/.venv`, so the linked copy still had to merge
the parent filesystem. The dependency step took 177 seconds and downloaded a 2.74 GB GPU blob before
SBOM scanning; runtime assembly itself remained 6 seconds. Reused same-SHA run `29496130906` then
created its schema-v5 receipt in 20 seconds, proving the steady-state at-most-45-second path while
leaving the cold non-GPU path unaccepted.

The follow-up implementation keeps GPU packages at `/root/rlab/.venv`, installs the 101 non-GPU
packages at `/opt/rlab-dependencies`, and bridges only the GPU site-packages with `rlab-gpu.pth`.
The final dependency image is still one immutable tag and one linked overlay, but its destination is
now disjoint from the GPU filesystem. SBOM generation scans the non-GPU scratch stage and explicitly
does not rescan the final composite; the separately published GPU image retains its own SBOM. The
dependency identity is version 4 and continues to hash the non-GPU projection, dependency Docker
contract, and resolved GPU digest.

Local Linux/amd64 verification on 2026-07-16 built the dependency target in 18.78 seconds with warm
archive cache: the 101-package venv took 7.5 seconds and the log contained no GPU-layer download or
extraction. A separate SBOM-enabled cache-only build took 41.65 seconds including a one-time 43 MB
scanner pull, a 7.8-second linked merge, and a 13.6-second scan, again with no GPU-layer transfer.
The full runtime build took 12.52 seconds under emulation, of which 9.4 seconds rebuilt the `rlab`
wheel and 0.3 seconds linked/exported the runtime overlay. The combined-environment validator proved
the exact 20-GPU plus 101-non-GPU lock union, all active cross-venv dependency constraints, interpreter
and console-script routing, and the `.pth` bridge. The final image passed the container smoke and
imports for Torch 2.12, Stable Retro, SB3, OpenCV, Numba, W&B, Breakout, and Mario. Remote acceptance
of the cold non-GPU path and next natural source-only push remains pending a committed canary; local
measurements alone do not declare the rollout accepted.

Before readiness was split, exact-source run `29348722980` built the source-only image in about 8
seconds but did not publish the usable runtime receipt until the CI image pull/contract smoke and
Modal deployment/probe completed; the full workflow took 5 minutes 40 seconds. The split contract
makes the early image receipt, named-machine image ensure, named-machine config validation, and
optional Modal wait separate timed phases. The named-machine validation remains mandatory because it
tests the real materialized payload inside the exact image on the actual selected runner.

Launch readiness now overlaps two fail-closed branches after the image receipt appears: Modal
readiness plus live backend/ROM/storage/database/startup checks, and selected-host image
inspection/pull plus validation of every materialized train payload. Queue rows commit only after
both branches succeed. Host image inspection precedes pulling, so a present digest is not fetched
again. Train JSON output reports image resolution, Modal readiness, live Modal preflight, host
inspection, pull, config validation, dispatch, queue-to-container startup, learner readiness, and
W&B readiness while retaining the aggregate readiness fields. Operational rollout targets are 90
seconds for prepared or reused runtimes and no more than 6 minutes for an immediate launch after a
genuine runtime change; measured acceptance belongs in canary evidence rather than this runbook.

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

The since-retired `retro-env-throughput-mario-l11` profile was run sequentially on the same idle
Mac against revision `51f981d4` and the provider-contract working tree based on `60c4f352`, using
the installed `stable-retro-turbo==1.0.1.post29`. Provider/runtime overhead before and after was
`2.61%`/`3.41%` at one env, `-0.17%`/`1.36%` at 16 envs, and `0.66%`/`0.60%` at 32 envs. Every case
passed its `5%` runtime-overhead gate; after-change median runtime SPS was also higher at all three
env counts. The active `mario-env-throughput-l11` successor targets the current
`supermariobrosnes-turbo` Mario provider, so these historical Stable Retro numbers are not a direct
baseline for it.
