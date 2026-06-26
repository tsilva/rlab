# Security Remediation Todo

Plans for the 2026-06-23 Codex Security audit of `rlab`.

| ID | Severity | Plan |
| --- | --- | --- |
| 01 | High | [Modal eval secretless deserialization](01-modal-eval-secretless-deserialization.md) |
| 02 | High | [Training resume trust gate](02-training-resume-trust-gate.md) |
| 03 | High | [Modal packaging excludes local secrets](03-modal-packaging-exclude-secrets.md) |
| 04 | High | [SkyPilot YAML and shell rendering hardening](04-skypilot-yaml-shell-injection.md) |
| 05 | Medium | [YouTube OAuth file permissions](05-youtube-oauth-file-permissions.md) |
| 06 | Medium | [Run name path traversal](06-run-name-path-traversal.md) |
| 07 | Medium | [Monitor API auth and payload redaction](07-monitor-api-auth-and-redaction.md) |
| 08 | Low | [YouTube privacy preservation on updates](08-youtube-privacy-preservation.md) |

Each issue plan was drafted by a separate subagent and normalized into this directory for follow-up implementation.
