# Benchmark Profiles

Benchmark profiles are named, repeatable checks for the rlab runtime. They are
not training recipes and they are not promotion evidence by themselves. Use them
to catch runtime, throughput, artifact, eval, and fleet regressions before a
larger experiment batch burns time.

Profiles and their informational expectations live as YAML files in
`experiments/benchmarks/profiles/`. Results belong under `logs/benchmarks/` and
should stay out of source control.

Benchmark success is determined by command exit status. Profile `expectations`
document externally inspected evidence and are not executable gates. Values that
affect execution, such as `max_runtime_overhead`, are top-level profile inputs.

Environment-sensitive recipe-backed profiles derive their environment and task
contract from the materialized recipe. Command plans and result files record the
exact recipe, overrides, and runtime arguments used; manually copied contract
summaries are intentionally unsupported.

```bash
rlab benchmark list
rlab benchmark show retro-env-throughput-mario-l11
rlab benchmark run retro-env-throughput-mario-l11 --dry-run
```

Run a profile only when its scope is appropriate for the machine. Fleet and
artifact-storage profiles can touch remote hosts, W&B, R2, Docker, or the queue
database.

## Profile Types

- `local_smoke`: queue-backed localhost smoke using `local-macbook` and the
  same fleet payload/result contract as beast jobs.
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

The canonical PPO loop profile exercises the consolidated runtime only. Provider/runtime stepping
overhead is measured separately by the native provider/runtime conformance benchmark.
