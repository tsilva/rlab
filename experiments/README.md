# Experiments

This directory holds shared machine config, active experiment scripts, and
goal-scoped experiment capsules.
Keep broad repo rules in the top-level runbooks:

- `../AGENTS.md` for repo rules and stable-retro runtime cautions.
- `../INSTANCES.md` for the human-facing hardware runbook.

Use `goals/<env-id>/` for durable goal-family contracts, provider-specific
environment fragments named `_env-<provider>.yaml`, checked-in train recipes, recipe
evidence, reports, and decisions. Generated local run logs and outputs belong
under ignored paths such as `runs/`, `logs/`, `models/`, or goal-local ignored
scratch directories.

Current machine-readable research state:

- `goals/`: active goal capsules, including contracts and checked-in recipes.
- `scripts/`: active experiment utilities used by benchmark profiles and tooling.

Historical goal reports, old recipe fragments, and the former `history/` tree
live under repo-root `.deprecated/` for local reference only.
