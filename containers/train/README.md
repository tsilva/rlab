# rlab Train Container

This image is the shared runtime contract for train/eval workers. It contains
the repo code, locked Python dependencies from `uv.lock`, system libraries
needed by Stable Retro, and the `rlab-*` console scripts. It intentionally does
not contain ROMs, secrets, checkpoints, W&B data, or run outputs.

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
  --env-file .env.fleet \
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
ghcr.io/tsilva/rlab/rlab-train:git-<full-sha>
ghcr.io/tsilva/rlab/rlab-train:ci-<run-id>-<attempt>
ghcr.io/tsilva/rlab/rlab-train@sha256:<digest>
```

Use tags for humans and digests for runs. The workflow uploads
`rlab-train-image.json` with the full `docker:...@sha256:...` runtime ref. Feed
that file into queue creation with `--runtime-image-ref-file` so jobs do not
depend on mutable tags.

## Local Fleet Manager

For `beast-2` and `beast-3`, Mac-side `rlab fleet` reconciles Docker containers
directly over SSH and keeps the queue in charge of scheduling; the beast hosts
only need Docker, NVIDIA runtime support, mounts, and the fleet env file.

```bash
uv run rlab fleet setup-host --host beast-3
uv run rlab fleet shepherd --machine beast-3 --limit 1
uv run rlab fleet watch --machine beast-3
```

Each launched container owns exactly one queue launch and is labeled with
`rlab.job-container=true`, `rlab.job-id`, `rlab.launch-id`, `rlab.machine`, and
`rlab.runtime-image-ref`. The shepherd finalizes completed launches from
`result.json` and prunes stale host runtime images that are not demanded by the
queue or used by active containers.
