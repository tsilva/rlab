# 07 - Monitor API Auth And Payload Redaction

Severity: Medium
Confidence: Medium
Subagent: Goodall

## Problem

`rlab-monitor` can be bound to `0.0.0.0` or a LAN address via `--host`, but `/` and `/api/state` are served without auth. `/api/state` calls `collect_state()` and returns live queue jobs. The DB queries currently expose full `to_jsonb(j)` and `to_jsonb(r)` payloads, so a broad bind can leak raw job/result rows, configs, paths, artifact refs, and possibly secret-like config values.

Affected files:

- `src/rlab/monitoring/server.py:601-617`
- `src/rlab/monitoring/server.py:659-660`
- `src/rlab/monitoring/state.py:421-427`
- `src/rlab/monitoring/state.py:447-453`

## Desired state

The monitor stays easy for local loopback use, but fails closed for non-loopback binds unless authentication is configured. Both the UI and `/api/state` require auth when exposed remotely. Returned job payloads are useful for debugging but sanitized by default: no full DB-row JSON, no secret-like values, and no unnecessary internal result payloads.

## Implementation plan

1. In `monitoring/server.py`, add a small auth layer:
   - `is_loopback_host(host)`.
   - `auth_required_for_host(host)`.
   - `valid_authorization(header, token)`.
   - Read token from `--token-env`, defaulting to `STABLE_RETRO_MONITOR_TOKEN`.
   - Avoid a raw `--token` flag so secrets do not land in shell history/process listings.
   - If `--host` is not loopback/localhost/`::1` and no token is present, exit with a clear `SystemExit` before starting `ThreadingHTTPServer`.
   - Support `Authorization: Bearer <token>` for API/curl use.
   - Support HTTP Basic with username `monitor` and password `<token>` for browser access.
   - Use `hmac.compare_digest`.
   - Protect both `/` and `/api/state`; return `401` with `WWW-Authenticate` before calling `collect_state()`.
2. Pass auth config into `MonitorHandler` through the existing `partial(MonitorHandler, ...)` setup.
3. Keep loopback behavior unchanged when no token is configured. Optionally require auth on loopback too if the token env var is set.
4. In `monitoring/state.py`, minimize and sanitize payloads:
   - Remove `to_jsonb(j) AS job_payload` and `to_jsonb(r) AS result_payload` from train/eval queries.
   - Replace `payload_from_row()` with an allowlist-based payload builder using already selected row fields.
   - Include only operationally useful fields: `id`, `goal_slug`, `spec_slug`, `profile_id`, `status`, `run_name`, lease/heartbeat timestamps, redacted config, selected metrics, and result summaries.
   - Add recursive redaction for keys containing `secret`, `token`, `password`, `credential`, `api_key`, `access_key`, `private_key`, `database_url`, `dsn`, or `authorization`.
   - Redact URL credentials if a string parses as a URL with embedded userinfo.
   - Do not expose raw `artifact_refs`, raw result rows, local output paths, or full schemas unless a later privileged mode is added.
5. Update the UI only if needed. Existing `fetch("/api/state")` should work after Basic auth because the browser reuses same-origin credentials.

## Tests/verification

Add `tests/test_monitoring_server.py` and update `tests/test_monitoring.py`.

Test cases:

- Broad bind without token fails before serving for `0.0.0.0` and `::`.
- Loopback without token remains allowed.
- With a token configured, unauthenticated `GET /` returns `401`.
- With a token configured, unauthenticated `GET /api/state` returns `401` and mocked `collect_state()` is not called.
- Valid Basic auth can fetch `/`.
- Valid Bearer auth can fetch `/api/state`.
- Invalid token returns `401`.
- Job builders produce payloads with redacted secret-like config values.
- Serialized job payload JSON does not contain sentinel secret values, raw database URLs, or raw `result_payload`.
- Replace the existing full-payload expectation with sanitized-payload expectations.

Run:

```bash
UV_CACHE_DIR=.uv-cache uv run pytest tests/test_monitoring.py tests/test_monitoring_server.py
UV_CACHE_DIR=.uv-cache uv run ruff check src/rlab/monitoring tests/test_monitoring.py tests/test_monitoring_server.py
```

## Rollout notes

Default local usage should remain:

```bash
rlab-monitor --host 127.0.0.1 --port 8765
```

Remote/LAN usage should become:

```bash
STABLE_RETRO_MONITOR_TOKEN="$(openssl rand -hex 24)" rlab-monitor --host 0.0.0.0 --port 8765
```

Document that broad binding is for trusted networks only, should preferably be tunneled over SSH/Tailscale, and now requires the token. Treat any existing broad-bound monitor as sensitive and restart it after the fix.

## Open questions

- Should loopback require auth whenever `STABLE_RETRO_MONITOR_TOKEN` is set, or only non-loopback binds?
- Is a sanitized payload enough for normal queue debugging, or should there be an authenticated `--include-raw-payloads` mode for local-only use?
- Are artifact refs and output/video paths considered sensitive enough to omit entirely, or should redacted summaries be shown?
