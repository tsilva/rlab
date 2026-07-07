# Benchmark Profiles

Benchmark profiles are named, repeatable checks for the rlab runtime. They are
not training recipes and they are not promotion evidence by themselves. Use them
to catch runtime, throughput, artifact, eval, and fleet regressions before a
larger experiment batch burns time.

Profiles live as YAML files in `experiments/benchmarks/profiles/`. Shared
baseline expectations live in `experiments/benchmarks/baselines.yaml`. Results
belong under `logs/benchmarks/` and should stay out of source control.

```bash
UV_CACHE_DIR=.uv-cache uv run rlab benchmark list
UV_CACHE_DIR=.uv-cache uv run rlab benchmark show retro-env-throughput-mario-l11
UV_CACHE_DIR=.uv-cache uv run rlab benchmark run retro-env-throughput-mario-l11 --dry-run
```

Run a profile only when its scope is appropriate for the machine. Fleet and
artifact-storage profiles can touch remote hosts, W&B, R2, Docker, or the queue
database.

## Profile Types

- `local_smoke`: direct local train/eval smoke using the active Python
  environment.
- `container_smoke`: train-image boot/import smoke through Docker.
- `env_throughput`: Stable Retro saved-state environment throughput probe.
- `ppo_loop_throughput`: bounded PPO loop probe for rollout/update throughput.
- `fleet_capacity`: queue-backed capacity check for a target host/runner shape.
- `eval_contract`: eval-environment reconstruction check for a known model or
  artifact.
- `artifact_storage_smoke`: tiny checkpoint-producing W&B/R2 reference-artifact
  check.

Benchmark requests should default to real imported saved states, not `State.NONE`.
Use `allow_state_none=true` only for explicit emulator hot-path diagnostics.
