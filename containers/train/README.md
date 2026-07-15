# rlab Train Container

This image is the shared runtime contract for train/eval workers. It contains
the repo code, locked Python dependencies from `uv.lock`, system libraries
needed by Stable Retro, the `rlab` CLI, and the container-only
`rlab-container-entrypoint` and `rlab-container-smoke` executables. It intentionally does
not contain ROMs, secrets, checkpoints, W&B data, or run outputs.

The Dockerfile keeps locked dependencies in a heavyweight cacheable stage and
assembles the small application filesystem in a scratch stage. Repository
documentation, tests, goals, and recipes are not embedded in the runtime; goals and
fully composed recipes travel in each queue payload. For
published builds, the workflow selects the immutable dependency-image digest as
the actual runtime base, then attaches the application filesystem with one
`COPY --link`. No command executes after that overlay is attached, so BuildKit
can rebase normal source changes without downloading or
extracting the multi-gigabyte dependency layer. Local builds and pull requests
with unpublished dependency inputs use the same Dockerfile's internal
`dependencies` stage instead.

## Build Locally

```bash
docker buildx build \
  --platform linux/amd64 \
  -f containers/train/Dockerfile \
  -t ghcr.io/tsilva/rlab/rlab-train:git-$(git rev-parse --short HEAD) \
  --load \
  .
```

Smoke the image without ROMs:

```bash
docker run --rm ghcr.io/tsilva/rlab/rlab-train:git-$(git rev-parse --short HEAD)
```

Smoke with a mounted ROM bundle:

```bash
docker run --rm --gpus all \
  -e RETRO_GAME=SuperMarioBros-Nes-v0 \
  -v /home/tsilva/roms:/roms:ro \
  ghcr.io/tsilva/rlab/rlab-train:git-$(git rev-parse --short HEAD) \
  rlab-container-entrypoint rlab-container-smoke
```

Run one claimed job payload:

```bash
docker run --rm --gpus all \
  --env-file /home/tsilva/rlab/.env.runner \
  -e RLAB_ROM_DIR=/roms \
  -v /home/tsilva/rlab/payloads:/root/rlab/payloads:ro \
  -v /home/tsilva/rlab/outputs:/root/rlab/outputs \
  -v /home/tsilva/roms:/roms:ro \
  ghcr.io/tsilva/rlab/rlab-train@sha256:<digest> \
  rlab-container-entrypoint \
  rlab run-job \
    --payload /root/rlab/payloads/<launch-id>.json \
    --output-dir /root/rlab/outputs/<launch-id>
```

`rlab-container-entrypoint` imports ROMs from `RLAB_ROM_DIR` before executing
the command. Set `RLAB_IMPORT_ROMS=0` to skip that step, or `RLAB_IMPORT_ROMS=1`
to fail if the mount is missing.

## Publishing

The `.github/workflows/rlab-train-image.yml` workflow builds `linux/amd64` and
pushes to GitHub Container Registry:

```text
ghcr.io/tsilva/rlab/rlab-train:runtime-<runtime-input-sha256>
ghcr.io/tsilva/rlab/rlab-train@sha256:<digest>
```

The runtime fingerprint hashes normalized indexed paths, executable modes, and file
contents for every runtime input. An exact-source version-4 receipt maps each pushed
commit to that fingerprint, the original runtime build commit, dependency identity,
and immutable image digest. Equivalent source commits reuse the existing tag and
digest; runs still record their exact source commit and composed recipe. Feed the
receipt's full `docker:...@sha256:...` ref into queue creation with
`--runtime-image-ref-file` so jobs do not depend on tags.

Dependency builds export their BuildKit cache to the mutable dependency
`buildcache` tag in GHCR. That tag is build infrastructure only and must never
be used as a runtime image selector. Runtime images do not export a second
cache because their linked application layers are cheap to rebuild.

The workflow publishes a dependency-input-keyed `rlab-train-dependencies` image with a full
SBOM and provenance when dependency inputs change. Content-addressed runtime images
use its immutable digest as their base, add only linked application layers, and retain the
runtime build source plus the exact dependency-image digest in both OCI labels and
`rlab-train-image.json`.

## Fleet Integration

Mac-side `rlab fleet` reconciles these one-job containers over local Docker or
SSH while the queue remains the scheduling authority. See
[INSTANCES.md](../../INSTANCES.md) for
the canonical service, job-status, host-setup, capacity, and cleanup commands:

```bash
rlab fleet service status --json
rlab runs status --machine beast-3 --json
rlab fleet capacity --machine beast-3 --set 4
rlab fleet drain --machine beast-3
rlab fleet resume --machine beast-3
```

Each launched container owns exactly one worker attempt and is labeled with
`rlab.job-container=true`, `rlab.job-id`, `rlab.launch-id`, `rlab.machine`, and
`rlab.runtime-image-ref`. The Mac fleet service finalizes completed launches from
`result.json` and prunes stale host runtime images that are not demanded by the
queue or used by active containers.

For a run materialized with `checkpoint_eval_backend: modal`, the container runs a low-priority
checkpoint coordinator plus a Neon telemetry relay. The trainer atomically saves into the launch
output's mounted `runs/` tree; the coordinator hashes and uploads checkpoints to R2 and announces
their verified locations through Neon. At shutdown, producers stop first and the relay immediately
flushes every remaining SQLite frame plus a final watermark. The attempt succeeds only after Neon
acknowledges that watermark. Fleet schedules bounded Modal CPU calls and is the only W&B writer.
