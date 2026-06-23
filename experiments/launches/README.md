# Launches

Root-level `sky_*.yaml` files are ignored scratch launch files. Promote a launch
shape here only when it is worth reviewing and reusing.

Prefer names that describe intent instead of timestamps, and keep credentials in
SkyPilot secrets or environment variables rather than in YAML.

## Manifest-Driven RTX4090 Launches

Use `stable-retro-ppo-skypilot` to turn a JSON experiment matrix into a
SkyPilot task, run preflight checks, print the standard secret-safe launch
command, sparsely monitor live launches, and summarize finished child logs.

The example manifest is a template. Replace `game`, optional `state`, and
`rom_source` before expecting preflight or launch to succeed.
Default reusable manifests should target `stable-retro-turbo==1.0.0.post16`;
older runtime versions belong only in explicitly historical repro or comparison
manifests.

Render a task:

```bash
UV_CACHE_DIR=.uv-cache uv run stable-retro-ppo-skypilot render \
  experiments/launches/stable_retro_rtx4090.example.json \
  --output sky_stable_retro_generated_4090.yaml
```

Check env, descriptions, RTX4090 child count, `env_threads`, runtime pin, and ROM
source before launching:

```bash
UV_CACHE_DIR=.uv-cache uv run stable-retro-ppo-skypilot preflight \
  experiments/launches/stable_retro_rtx4090.example.json
```

Print the exact `sky launch` command without running it:

```bash
UV_CACHE_DIR=.uv-cache uv run stable-retro-ppo-skypilot launch \
  experiments/launches/stable_retro_rtx4090.example.json \
  --output sky_stable_retro_generated_4090.yaml
```

Actually launch only after the rendered YAML and preflight output look right:

```bash
UV_CACHE_DIR=.uv-cache uv run stable-retro-ppo-skypilot launch \
  experiments/launches/stable_retro_rtx4090.example.json \
  --output sky_stable_retro_generated_4090.yaml \
  --execute \
  --sparse \
  --log-output logs/stable_retro_example_4090.sky.log
```

Add `--down-on-complete` when the cluster should be cleaned up automatically
after the launch command exits.

Summarize child logs after completion:

```bash
UV_CACHE_DIR=.uv-cache uv run stable-retro-ppo-skypilot collect \
  logs/stable_retro_example_4090
```

Write a machine-readable report from a full SkyPilot launch log:

```bash
UV_CACHE_DIR=.uv-cache uv run stable-retro-ppo-skypilot report \
  logs/stable_retro_example_4090.sky.log \
  --output reports/stable_retro_example_4090.report.json
```

Reproduce a W&B run config without hardcoding the ROM target:

```bash
UV_CACHE_DIR=.uv-cache uv run stable-retro-ppo-skypilot repro-from-wandb \
  tsilva/SuperMarioBros-NES/lexxixz3 \
  --rom-source roms/your-game.rom \
  --manifest-output experiments/launches/repro_lexxixz3.json \
  --output sky_repro_lexxixz3_4090.yaml \
  --ensure-api
```

Run `doctor-api --execute` if the local SkyPilot CLI is pointed at the wrong API
server:

```bash
UV_CACHE_DIR=.uv-cache uv run stable-retro-ppo-skypilot doctor-api --execute
```

Machine-readable hardware defaults may live in `experiments/instances.json` when
launch manifests need them. Keep `INSTANCES.md` as the human source of truth,
and update both files together when a benchmark-backed hardware fact changes.
