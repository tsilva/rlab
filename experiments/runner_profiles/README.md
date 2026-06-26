# Runner Profiles

Runner profiles are the durable source for long-lived SkyPilot queue workers.
They describe the remote runtime envelope: hardware shape, package pin, ROM
mounts, smoke checks, and the exact `train_jobs.profile_id` a worker may claim.
Use `rlab-compute targets` to inspect all configured targets, but use
`rlab-skypilot` for these runner profiles until a Modal training-runner adapter
exists.

Use profiles when the queue already owns the experiment payload. Use
`experiments/launches/` manifests only for older direct SkyPilot batches that
embed concrete training runs in the rendered YAML.

Render a profile to ignored scratch YAML:

```bash
UV_CACHE_DIR=.uv-cache uv run rlab-skypilot render-runner \
  experiments/runner_profiles/mario_ppo_post20_task_conditioned_rtx4090.example.json \
  --output sky_train_runner_4090.yaml
```

Override the profile target when capacity is better elsewhere:

```bash
UV_CACHE_DIR=.uv-cache uv run rlab-skypilot render-runner \
  experiments/runner_profiles/mario_ppo_post20_task_conditioned_rtx4090.example.json \
  --target runpod-l4 \
  --output sky_train_runner_runpod_l4.yaml
```

Preflight a profile before launch:

```bash
UV_CACHE_DIR=.uv-cache uv run rlab-skypilot preflight-runner \
  experiments/runner_profiles/mario_ppo_post20_task_conditioned_rtx4090.example.json
```

Launch a queue runner:

```bash
UV_CACHE_DIR=.uv-cache uv run rlab-skypilot launch-runner \
  experiments/runner_profiles/mario_ppo_post20_task_conditioned_rtx4090.example.json \
  --output sky_train_runner_4090.yaml \
  --execute \
  --detach-run
```

Queue and ensure a digest-pinned runner in one dry-run-first command:

```bash
UV_CACHE_DIR=.uv-cache uv run rlab-skypilot queue-train \
  experiments/runner_profiles/mario_ppo_post20_task_conditioned_rtx4090.example.json \
  --goal <goal-slug> \
  --spec-id <spec-id> \
  --train-config-json '<json>' \
  --runtime-image-ref 'docker:ghcr.io/tsilva/rlab/rlab-train@sha256:<digest>' \
  --target beast-3 \
  --ensure-runner
```

Keep profile IDs coarse. Create a new profile when the runtime contract changes:
package pin, observation space or policy family, ROM/state mounts, hardware
shape, or queue-client compatibility. Seeds, hyperparameters, W&B tags, stop
criteria, and run descriptions belong in queued jobs.

## Prebuilt Container Runtime

Profiles can use the shared training image from `containers/train/` instead of
building a venv during SkyPilot setup:

```json
{
  "image_id": "docker:ghcr.io/tsilva/rlab/rlab-train@sha256:<digest>",
  "runtime_image_ref": "docker:ghcr.io/tsilva/rlab/rlab-train@sha256:<digest>",
  "prebuilt_image": true
}
```

In this mode the rendered YAML still mounts ROM/state files, but setup only runs
the container smoke/import helper. The train runner then starts with:

```bash
rlab-container-entrypoint python -m rlab.train_runner ...
```

Use an immutable digest for runs. Tags such as `git-<short-sha>` are for humans;
the digest is the reproducibility boundary.

New queue runners claim only jobs whose `profile_id`, `runtime_image_ref`, and
optional `run_target` match the runner. Pending jobs without a runtime digest
must be canceled, re-enqueued, or explicitly migrated before digest-pinned
runners can claim them.
