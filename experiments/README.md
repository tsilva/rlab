# Experiments

This directory holds active goal contracts, training recipes, benchmark
profiles, report declarations, and experiment utilities.
Keep broad repo rules in the top-level runbooks:

- `../AGENTS.md` for repo rules and stable-retro runtime cautions.
- `../INSTANCES.md` for the human-facing hardware runbook.

Use `goals/<env-id>/` for durable goal-family contracts and optional provider-specific
environment fragments named `_env-<provider>.yaml`. Goal-family report declarations
live beside those contracts as `_reports.yaml`. Active training recipes live under
`recipes/`, while benchmark profiles live under `benchmarks/`. Generated local run
logs and outputs belong under ignored paths such as `runs/`, `logs/`, and `models/`.

Current research state:

- `goals/`: active goal contracts, optional environment fragments, and report declarations.
- `recipes/`: active checked-in training recipes and presets.
- `benchmarks/`: reproducible benchmark profiles and supporting documentation.
- `scripts/`: active experiment utilities used by benchmarks and tooling.

Historical goal reports, decisions, old recipe fragments, and the former `history/` tree
live under repo-root `.deprecated/` for local reference only.
