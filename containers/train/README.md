# rlab Train Container

This image is the shared runtime contract for train/eval workers. It contains
the repo code, locked Python dependencies from `uv.lock`, system libraries
needed by Stable Retro, the `rlab` CLI, and the container-only
`rlab-container-entrypoint` and `rlab-container-smoke` executables. It intentionally does
not contain ROMs, secrets, checkpoints, W&B data, or run outputs.

The Dockerfile keeps the heavyweight stack in two cacheable images: a GPU
foundation (`torch`, Triton, and CUDA/NVIDIA wheels), then the remaining train
dependencies. The non-GPU packages are installed in a clean virtual environment
at the final `/root/rlab/.venv` path, collapsed into a scratch overlay, and attached
to the immutable GPU foundation with one `COPY --link`. No command executes after
that merge. It assembles the small application filesystem in a separate scratch stage. Repository
documentation, tests, goals, and recipes are not embedded in the runtime; goals and
fully composed recipes travel in each queue payload. For
published builds, the workflow selects the immutable dependency-image digest as
the actual runtime base, then attaches the application filesystem with one
`COPY --link`. No command executes after that overlay is attached, so BuildKit
can rebase normal source changes without downloading or
extracting the multi-gigabyte dependency layer. Local builds and pull requests
with unpublished dependency inputs use the same Dockerfile's internal
`dependencies` stage instead. The GPU foundation is the source of nearly all of
the multi-gigabyte transfer; ordinary runtime and non-GPU dependency changes do
not invalidate it.

`uv.lock` remains the universal resolution and provenance source. The committed
`train-linux-amd64.lock`, `gpu-linux-amd64.lock`, and
`train-dependencies-linux-amd64.lock` files are deterministic CPython 3.14/Linux
x86-64 install projections for the container only. The GPU and non-GPU projections
must be disjoint and their union must exactly reconstruct the full train projection. Regenerate
and verify them with:

```bash
uv run --frozen --only-group train-image-build \
  python containers/train/lock_projection.py
uv run --frozen --only-group train-image-build \
  python containers/train/lock_projection.py --check
```

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

The branch-triggered `.github/workflows/rlab-train-dependencies.yml` workflow
uses one metadata job and one publisher job to prebuild changed foundations. The
publisher resolves or builds the GPU and non-GPU images sequentially through one
Buildx builder. The main train-image workflow is independently correct: one build
job resolves or builds the GPU, dependency, and runtime images through one builder,
then Modal deployment runs downstream. Both workflows reuse the immutable images
and publish:

```text
ghcr.io/tsilva/rlab/rlab-train-gpu:build-<gpu-key>
ghcr.io/tsilva/rlab/rlab-train-dependencies:build-<dependency-key>
ghcr.io/tsilva/rlab/rlab-train:runtime-<runtime-input-sha256>
ghcr.io/tsilva/rlab/rlab-train@sha256:<digest>
```

The identity chain is GPU key → GPU digest → dependency key → dependency digest →
overlay key → runtime key. The overlay key hashes normalized indexed paths,
executable modes, selected runtime Dockerfile instructions, and file contents, but
not `uv.lock` or dependency projections. An exact-source version-5 receipt records
every key, plan hash, base digest, original runtime build commit, and immutable image
digest. Equivalent source commits reuse the existing tag and
digest; runs still record their exact source commit and composed recipe. Feed the
receipt's full `docker:...@sha256:...` ref into queue creation with
`--runtime-image-ref-file` so jobs do not depend on tags.

GPU and dependency images carry SBOM and provenance. There is no mutable maximal
registry build cache: published content-addressed images are the reusable cache
contract. Dependency images use the immutable GPU digest as their base and add only
the linked non-GPU virtual-environment overlay. Runtime images use the immutable
dependency digest as their base and add only linked application layers.

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
If latest-runtime prewarming fails, container cleanup continues but runtime-image
pruning is skipped so the host's last known good image remains available.

For a run materialized with `checkpoint_eval_backend: modal`, the container runs a low-priority
checkpoint coordinator plus a Neon telemetry relay. The trainer atomically saves into the launch
output's mounted `runs/` tree; the coordinator hashes and uploads checkpoints to R2 and announces
their verified locations through Neon. At shutdown, producers stop first and the relay immediately
flushes every remaining SQLite frame plus a final watermark. The attempt succeeds only after Neon
acknowledges that watermark. Fleet schedules bounded Modal CPU calls and is the only W&B writer.
