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

Mutating commands wake the service immediately; its 30-second launchd interval
is the recovery path for missed wake-ups and remote completion. Each invocation
loads current source, performs one bounded pass, and exits. launchd does not
overlap invocations of the same service label.

`rlab fleet service watch` is the normal read-only operational view. On an interactive terminal it
opens a responsive dashboard; `--once`, `--plain`, and `--json` provide scriptable output modes.
The dashboard reads only launchd registration and the service's redacted atomic state files; it
does not query PostgreSQL, Docker, SSH, Modal, or W&B. The service records current phases and queue
classifications so an interrupted pass remains distinguishable from stale completed state. Use
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
`rlab runs status` is observational. The service is the only normal mutating
reconciler: it claims, launches, finalizes, and prunes stale Docker images after
no active container or exact-machine queued demand needs them.

## Modal CPU Checkpoint Evaluation

Modal is a backend-bound evaluation lane owned by the same Mac fleet service; it is not a
registered training machine and must not be added to `experiments/machines.yaml`. Its checked-in
deployment, timeout, budget, and concurrency contract is `experiments/modal_eval.yaml`. The hard
orchestration ceiling and the independent Modal `max_containers` guard are both 20, while rollout
starts at effective capacity 1 and must be promoted through 2 and 3 before 20. Modal is the default
for new queue-backed jobs at capacity 1; do not raise to the next stage until the parity,
interruption, staged-capacity, and cost canaries pass.

```bash
rlab eval modal status
rlab eval modal preflight \
  --runtime-image-ref docker:ghcr.io/tsilva/rlab/rlab-train@sha256:<digest> \
  --game <game-id>
rlab eval modal drain
rlab eval modal resume --capacity 1
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
ROM object, preview R2/public-URL path, local Modal credentials, and exact runtime-specific
deployment are all present. Every normal screen evaluation captures a bounded policy-observation
MP4 in R2 and projects it as `eval/screen/preview` in the producing W&B run.

The train-image workflow runs on every push to `main` and publishes an exact-source version-4
`rlab-train-image.json` as soon as the immutable image exists. Runtime images are keyed by a
fingerprint of their complete runtime inputs rather than by commit: goal, recipe, test, and
unrelated documentation changes therefore reuse a proven digest. A reused runtime preserves its
original `runtime_build_source_sha`; the new receipt still records the exact pushed `source_sha`.
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

PostgreSQL is the wait queue, orchestration authority, and transient telemetry mailbox. The service
never submits work beyond the effective capacity, reserves worst-case cost before dispatch, and
leaves budget-blocked jobs pending for operator inspection. Draining stops new Modal calls without
stopping training. Checkpoint models, metadata, previews, and raw Modal results are immutable R2
objects; accepted attempts, commands, artifact locations, decisions, and publication cursors live
in PostgreSQL. Runtime-specific apps are deployed from CI as `rlab-eval-<digest-prefix>` from the exact
shared train/eval image digest. Worker retries are disabled; the fleet service may create one
separately recorded second attempt for transient failures. Modal 1.5 exposes only single-use or
unbounded-reuse containers. V1 uses warm-container reuse with a 60-second scale-down window because
single-use containers impose the full cold-start cost on every evaluation; the global call cap and
dollar budgets remain the spend guards. There is no enforceable ten-input container lifetime until
Modal supports `max_inputs > 1`.

Ready promotion projections are enqueued in bounded batches rather than one per service pass, and
the service drains up to three independent W&B runs concurrently in isolated publisher processes.
Each run retains its session advisory lock, so concurrent publication cannot interleave writers for
the same W&B run. Neon queue and mailbox connections use TCP keepalives and a 30-second user timeout
so a laptop sleep or network transition fails the pass promptly and is retried with a fresh
connection.

The fleet service inventories owned `rlab-eval-<12-hex>` deployments hourly and stops at most ten
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
  appear in fleet-service health and do not stop running jobs.

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
