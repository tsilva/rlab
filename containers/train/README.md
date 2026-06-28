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

Run a train queue worker:

```bash
docker run --rm --gpus all \
  --env-file .env.runner \
  -e RLAB_ROM_DIR=/roms \
  -v /home/tsilva/roms:/roms:ro \
  -v /home/tsilva/sandbox-runs:/root/rlab/runs \
  ghcr.io/tsilva/rlab/rlab-train@sha256:<digest> \
  rlab-container-entrypoint \
  rlab-train-runner --profile mario-ppo/post20/rtx2060-task-conditioned-v1
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

For `beast-2` and `beast-3`, Mac-side `rlab-fleet` reconciles Docker containers
directly over SSH and keeps the queue in charge of scheduling; the beast hosts
only need Docker, NVIDIA runtime support, mounts, and the runner env file.

```bash
uv run rlab-fleet plan
uv run rlab-fleet reconcile
```

The managed containers are labeled with `rlab.managed=true`,
`rlab.profile`, `rlab.runtime-image-ref`, `rlab.run-target`, and a config hash.
They are removed only after there are no pending or running queue jobs for that
profile/digest/target and no active lease owned by that container's worker id.
