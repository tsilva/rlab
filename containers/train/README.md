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

## Modal

Modal keeps the existing local image builder by default. To use a prebuilt GHCR
image instead:

```bash
export RLAB_MODAL_IMAGE_REF=ghcr.io/tsilva/rlab/rlab-train@sha256:<digest>
export RLAB_MODAL_REGISTRY_SECRET=ghcr-registry  # only for private images
```

The Modal secret, when used, must provide `REGISTRY_USERNAME` and
`REGISTRY_PASSWORD`.

## SkyPilot

Runner profiles can opt into the prebuilt runtime:

```json
{
  "image_id": "docker:ghcr.io/tsilva/rlab/rlab-train@sha256:<digest>",
  "prebuilt_image": true
}
```

With `prebuilt_image: true`, the rendered runner YAML skips venv creation,
`uv sync`, and package installation. It mounts ROMs, runs
`rlab-container-entrypoint rlab-container-smoke`, then launches
`rlab.train_runner` through the same entrypoint.

Use this mode on backends that actually run the `image_id` as a container, such
as Kubernetes or RunPod. For SSH node pools, prefer direct `docker run` on the
host unless that host has been put behind Kubernetes.

## Local Fleet Manager

For `beast-2` and `beast-3`, prefer `rlab-fleet` over SkyPilot. It reconciles
Docker containers directly over SSH and keeps the queue in charge of scheduling:

```bash
uv run rlab-fleet plan
uv run rlab-fleet reconcile --execute
```

The managed containers are labeled with `rlab.managed=true`,
`rlab.profile`, `rlab.runtime-image-ref`, `rlab.run-target`, and a config hash.
They are removed only after there are no pending or running queue jobs for that
profile/digest/target and no active lease owned by that container's worker id.
