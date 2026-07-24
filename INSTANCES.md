# Compute instances

This file is the source of truth for choosing and operating rlab compute.
Research code always runs in one container and does not branch on the provider.
dstack owns placement, logs, cancellation, interruption classification, and
resource release.

## Selection policy

| Need | Request |
| --- | --- |
| Use an idle compatible local host, otherwise wait | `--compute auto` |
| Require a named local fleet | `--compute local --target <fleet>` |
| Permit bounded spot compute | `--compute spot --max-price … --max-cost-usd …` |
| Permit on-demand compute | `--compute on-demand --allow-on-demand --max-cost-usd …` |
| Short CPU evaluation | Modal, dispatched by the training supervisor |

`auto` never enters paid cloud compute without a finite price and total-cost
bound. On-demand is opt-in even with a budget. Every task has a finite
`max_duration`.

v1 schedules one training container per single-GPU host. Do not recreate the
former multi-container RTX 4090 oversubscription. The container may retain its
GPU while outstanding evals and W&B drain; measure that idle tail before adding
a separate finalizer.

## dstack control plane

- Version: CLI and server `0.20.28`.
- Server: B3 systemd unit, pinned image
  `dstackai/dstack@sha256:86b820cf5f6e0cfc54dd387527493168a4045b362ca9459265ea9828eef0b4af`.
- Persistent state: `/var/lib/rlab/dstack` on B3.
- API binding: B3 loopback port 3000 only.
- Client access: SSH tunnel to `http://127.0.0.1:3000`.
- Secrets: `/etc/rlab/dstack/server.env`, local private environment/Keychain;
  never source control.
- Checked-in operations: `ops/dstack/`.

Example tunnel:

```bash
ssh -N -L 3000:127.0.0.1:3000 tsilva@beast-3
```

The checked-in fleet is portable to dstack Sky. A later Sky migration changes
the project endpoint, credentials, and available offers, not the run/R2/W&B
protocol.

Automatic retries are limited to dstack `no-capacity` and genuine
`interruption`. Generic errors require evidence-backed manual retry. The CLI
requires the previous task to be terminal, the R2 writer lease to expire, and a
30-second quiescence interval. A finished learner retries in drain-only mode.

## B3

- dstack fleet: `b3`.
- Host: `beast-3`.
- Access: `ssh tsilva@beast-3`.
- GPU: NVIDIA RTX 4090.
- Known resources: at least 12 CPU and 48 GiB RAM.
- Role: primary training and final Mario acceptance.
- dstack blocks: 1 unsplit machine.
- Container concurrency: 1 training task.
- ROM cache source: `/var/lib/rlab/rom-cache-source`.
- ROM cache mount: `/var/lib/rlab/rom-cache:/rom-cache`. A root-owned systemd
  service exposes the source as a kernel-enforced read-only bind mount before
  dstack maps it into the task.

dstack 0.20.28 probes `/dev/kfd` before `/dev/nvidiactl`. B3’s Ryzen iGPU can
therefore mask the RTX 4090. The installed shim drop-in removes only the unused
`/dev/kfd` compute node and regenerates host inventory before each probe; it
does not remove `/dev/dri`.

The pinned CUDA smoke completed successfully and reported one RTX 4090. B3 was
idle after that verification.

## B2

- Planned dstack fleet: `b2`.
- Host: `beast-2`.
- Access:
  `ssh -o HostKeyAlias=beast-2 tsilva@192.168.133.26`
  until hostname resolution is reliable.
- GPU: NVIDIA RTX 2060.
- Known resources: at least 4 CPU and 8 GiB RAM.
- Role: cheaper small-GPU smokes and RTX 2060-specific validation.
- Container concurrency: 1 training task.

B2 must not be enrolled until the complete B3 Mario Level1-1 acceptance gate
passes.

## Local Mac

The Mac is the operator workstation and is not a registered training fleet in
v1. It runs the CLI, opens the SSH tunnel, validates configs, and may run
ROM-free unit/integration tests. Its GPU is not used for the Linux/CUDA
acceptance path.

## Modal evaluation

Modal is the v1 `EvalBackend` because acceptance evaluations are short CPU
jobs, it starts quickly, and it does not contend with local training GPUs.

- environment: `rlab-eval`;
- immutable app: `rlab-eval-v2-<source-sha12>`;
- worker: 8 CPU, 4 GiB RAM;
- zero warm containers;
- maximum concurrent containers: 10;
- at most two attempts per eval;
- private, hash-verified ROM provisioning;
- no W&B or control-private R2 credentials.

Modal writes only eval-private results/evidence. The training supervisor polls
those results and is the only process allowed to project eval metrics into the
shared W&B run.

A native Modal monthly hard budget must be configured before the final
acceptance launch. `$5/run` is an alert/forecast threshold, not a distributed
reservation.

## Host image cleanup

dstack 0.20.28 removes terminated containers but does not safely prune unused
runtime images. B3 therefore runs the root-owned
`rlab-dstack-image-cleanup.timer` every 30 minutes.

The cleanup fails closed unless it can query valid dstack run inventory. It
only considers immutable `ghcr.io/tsilva/rlab/rlab-train` images and preserves:

- images used by running containers;
- images demanded by pending, submitted, provisioning, running, or terminating
  tasks.

It never removes unrelated images, containers, volumes, or build cache. Install
the same timer on B2 only when B2 is enrolled.

## Runtime-image pipeline

Training tasks may start only after the exact source SHA has a verified
immutable image receipt. The image is split into:

1. a CUDA/PyTorch GPU foundation;
2. a disjoint non-GPU dependency environment;
3. a linked exact-source application overlay.

The committed projections currently partition the environment into GPU and
non-GPU packages and are verified by
`containers/train/environment_contract.py`. Source-only changes reuse the
large foundations and attach a small overlay; a prior accepted source-only
workflow, including Modal deployment/probe, completed in roughly six minutes.

These are build-cache observations, not learner-throughput promises. Record new
hardware/runtime benchmark facts here when they materially change compute
selection.

## Operational checklist

Before a launch:

1. Confirm this file and the selected goal/recipe.
2. Confirm the Git revision is clean and pushed.
3. Confirm exact-source image and Modal deployment receipts.
4. Confirm the three R2 scopes, public base URL, and seven-day delivered-journal
   lifecycle.
5. Confirm the Modal hard budget.
6. Confirm the requested compute policy and finite maximum duration.

During a run:

```bash
rlab experiment follow --run <run-id>
rlab experiment logs --run <run-id> --tail 200
```

After terminal state, verify the private R2 terminal receipt, dstack release,
public checkpoint/index access, and credential-free playback. A successful
dstack task without the terminal receipt is an operational failure.
