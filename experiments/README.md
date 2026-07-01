# Experiments

This directory holds shared machine and policy config plus goal-scoped experiment capsules.
Keep broad repo rules in the top-level runbooks:

- `../AGENTS.md` for repo rules and stable-retro runtime cautions.
- `../INSTANCES.md` for the human-facing hardware runbook.

Use `goals/<goal-slug>/` for durable goal contracts, checked-in specs, recipe
evidence, reports, and decisions. Generated local run logs and outputs belong
under ignored paths such as `runs/`, `logs/`, `models/`, or goal-local ignored
scratch directories.

Current machine-readable research state:

- `goals/`: active goal capsules, including contract, specs, recipe evidence,
  reports, and decisions.
- `policies/`: capacity and scheduling policies for keeping compute busy without
  mixing incomparable runtime envelopes.
