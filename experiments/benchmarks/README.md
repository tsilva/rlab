# Benchmark Profiles

Benchmark profiles are named, repeatable performance measurements plus one
queue-backed local integration smoke. They are not training recipes and they are
not promotion evidence. Use them to catch environment/runtime overhead,
training-loop throughput, and local queue regressions before a larger experiment
batch burns time.

Profiles live as YAML files in `experiments/benchmarks/profiles/`. Results belong
under `logs/benchmarks/` and should stay out of source control.

Benchmark success requires every command and executable profile gate to pass.
Results record source/runtime/host provenance, command output, parsed benchmark
evidence, and validation failures. Environment throughput enforces
`max_runtime_overhead`; training-loop throughput requires its declared metrics;
the local smoke requires every queued run to finish successfully.

Environment-sensitive recipe-backed profiles compose an explicit goal contract
and recipe. Command plans and result files record both inputs, their source
composition, overrides, and runtime arguments; manually copied contract summaries
are intentionally unsupported.

```bash
rlab benchmark list
rlab benchmark show mario-env-throughput-l11
rlab benchmark run mario-env-throughput-l11 --dry-run
```

Run a profile only when its scope is appropriate for the machine. The local smoke
touches Docker and the queue database. Throughput profiles run locally and write
generated result and isolated run evidence under `logs/benchmarks/`.

## Profile Types

- `local_smoke`: queue-backed localhost smoke using `local-macbook`, disabled
  checkpoint evaluation, and the same one-attempt container/result contract as
  Beast jobs.
- `env_throughput`: paired native-provider and rlab runtime throughput benchmark
  for the current Mario provider.
- `train_loop_throughput`: bounded W&B-disabled backend training-loop benchmark
  for rollout/update throughput.

Benchmark requests should default to real imported saved states, not `State.NONE`.
Use `allow_state_none=true` only for explicit emulator hot-path diagnostics.

The canonical training-loop profile exercises the consolidated runtime only. Provider/runtime stepping
overhead is measured separately by the native provider/runtime conformance benchmark.
