# rlab dstack control plane

rlab pins both the CLI and server to `0.20.28`. The server image is pinned to
the multi-platform digest
`sha256:86b820cf5f6e0cfc54dd387527493168a4045b362ca9459265ea9828eef0b4af`.

The B3 deployment stores dstack's internal SQLite database and logs beneath
`/var/lib/rlab/dstack`. It binds the API only to B3 loopback. Operators connect
through an SSH tunnel; the dstack API must not be exposed to the public network.
The checked-in systemd unit runs the pinned image directly, so the host does not
need Docker Compose.

Secrets are host-owned and never checked in:

- `/etc/rlab/dstack/server.env` contains `DSTACK_SERVER_ADMIN_TOKEN`.
- `/var/lib/rlab/dstack/config.yml` contains dstack's AES-256-GCM encryption key.
- the local client receives `DSTACK_TOKEN` from the operator's private environment.

The B3 fleet deliberately has one unsplit host (`blocks: 1`). A task therefore
owns the single GPU rather than recreating the former six-container oversubscription.
B2 is not enrolled until B3 passes the full acceptance gate.

## Read-only local ROM cache

Local ROM bytes live under `/var/lib/rlab/rom-cache-source`. Install
`rlab-rom-cache-mount` and `rlab-rom-cache-readonly.service` to expose that
directory at `/var/lib/rlab/rom-cache` as a kernel-enforced read-only bind
mount. dstack maps only the read-only mount into the container. The pinned
runner was verified by attempting a container write and receiving
`Read-only file system`.

## B3 NVIDIA detection

dstack 0.20.28 checks `/dev/kfd` before `/dev/nvidiactl`. B3's Ryzen integrated
GPU exposes `/dev/kfd`, which makes the pinned shim report the iGPU instead of
the RTX 4090. Install `dstack-shim-override.conf` as
`/etc/systemd/system/dstack-shim.service.d/rlab-nvidia-only.conf` before
enrolling B3. The drop-in removes only the unused AMD compute device node and
regenerates dstack's host inventory on every shim start; it does not remove
`/dev/dri` or affect B3's display device.

Use an SSH tunnel such as `ssh -N -L 3000:127.0.0.1:3000 tsilva@beast-3`, then
point the client at `http://127.0.0.1:3000`.

## Host image cleanup

The pinned runner removes terminated task containers but does not prune their
images. Install the isolated Python 3.13 dstack CLI at
`/opt/rlab/dstack-cli/bin/dstack`, install `rlab-dstack-image-cleanup` under
`/usr/local/libexec`, and enable `rlab-dstack-image-cleanup.timer`.

The cleanup job fails closed unless it can obtain and validate dstack's current
run inventory. It preserves images demanded by pending, submitted,
provisioning, running, or terminating tasks and images used by running
containers. It considers only immutable images in
`ghcr.io/tsilva/rlab/rlab-train`; it does not prune other images, containers,
volumes, or build cache. Set `RLAB_IMAGE_CLEANUP_DRY_RUN=1` for an audit-only
invocation.

## R2 metric-journal expiry

The `control-private` bucket must have an object lifecycle rule named
`expire-delivered-metric-journals` for prefix
`expiring-metric-journals/`, expiring objects after seven days. Active
journals stay beneath the run attempt until W&B remote visibility and all
terminal drain gates pass; only then does the supervisor atomically relocate
them beneath the expiring prefix. Cloudflare requires a token with Workers R2
Storage Write permission to install this bucket-level rule.
