from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from rlab.dotenv import load_env_file
from rlab.job_queue import connect, json_arg
from rlab.workspace_contract import canonical_json_bytes, cleanup_batch_from_mapping, sha256_json


SIGNER_PROTOCOL_VERSION = 1
DEFAULT_LEASE_SECONDS = 10


class WorkspaceSignerError(RuntimeError):
    pass


def heartbeat_signer(
    conn,
    *,
    signer_id: str,
    key_revision: str,
    state: str,
    error: str | None = None,
    lease_seconds: int = DEFAULT_LEASE_SECONDS,
) -> None:
    if state not in {"starting", "healthy", "degraded", "stopped"}:
        raise ValueError("invalid signer state")
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO workspace_signer_leases (
                  signer_id, generation, key_revision, state,
                  heartbeat_at, lease_expires_at, last_error
                ) VALUES (
                  %(signer_id)s, 1, %(key_revision)s, %(state)s, now(),
                  now() + (%(lease_seconds)s * interval '1 second'), %(error)s
                )
                ON CONFLICT (signer_id) DO UPDATE
                SET generation=CASE
                      WHEN workspace_signer_leases.key_revision <> EXCLUDED.key_revision
                        OR workspace_signer_leases.lease_expires_at <= now()
                      THEN workspace_signer_leases.generation+1
                      ELSE workspace_signer_leases.generation END,
                    key_revision=EXCLUDED.key_revision, state=EXCLUDED.state,
                    heartbeat_at=now(), lease_expires_at=EXCLUDED.lease_expires_at,
                    last_error=EXCLUDED.last_error
                """,
                {
                    "signer_id": signer_id,
                    "key_revision": key_revision,
                    "state": state,
                    "lease_seconds": max(1, int(lease_seconds)),
                    "error": error,
                },
            )


def sign_envelope(envelope: Mapping[str, Any], *, private_key_path: Path) -> str:
    parsed = cleanup_batch_from_mapping(envelope)
    payload = canonical_json_bytes(parsed.as_dict())
    try:
        result = subprocess.run(
            ["openssl", "dgst", "-sha256", "-sign", str(private_key_path)],
            input=payload,
            capture_output=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise WorkspaceSignerError("OpenSSL signing failed to start or timed out") from exc
    if result.returncode != 0:
        raise WorkspaceSignerError(
            f"OpenSSL signing failed: {result.stderr.decode('utf-8', errors='replace').strip()}"
        )
    return base64.b64encode(result.stdout).decode("ascii")


def claim_signing_outbox(
    conn,
    *,
    signer_id: str,
    key_revision: str,
    lease_seconds: int = DEFAULT_LEASE_SECONDS,
) -> dict[str, Any] | None:
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                WITH due AS (
                  SELECT o.batch_id
                  FROM workspace_authorization_outbox o
                  JOIN workspace_cleanup_batches b ON b.batch_id=o.batch_id
                  WHERE o.state IN ('pending','leased')
                    AND o.next_retry_at <= now()
                    AND (o.lease_expires_at IS NULL OR o.lease_expires_at <= now())
                    AND b.state='signing_pending'
                    AND b.key_revision=%(key_revision)s
                    AND b.outer_not_after > now()
                  ORDER BY o.created_at
                  FOR UPDATE OF o SKIP LOCKED
                  LIMIT 1
                )
                UPDATE workspace_authorization_outbox o
                SET state='leased', lease_owner=%(signer_id)s,
                  lease_generation=lease_generation+1,
                  lease_expires_at=now() + (%(lease_seconds)s * interval '1 second'),
                  attempts=attempts+1, updated_at=now(), last_error=NULL
                FROM due WHERE o.batch_id=due.batch_id
                RETURNING o.*
                """,
                {
                    "signer_id": signer_id,
                    "key_revision": key_revision,
                    "lease_seconds": max(1, int(lease_seconds)),
                },
            )
            row = cur.fetchone()
            if not row:
                return None
            cur.execute(
                """
                UPDATE workspace_cleanup_batches
                SET state='signer_claimed', signer_lease_owner=%(signer_id)s,
                  signer_lease_expires_at=now() + (%(lease_seconds)s * interval '1 second'),
                  updated_at=now()
                WHERE batch_id=%(batch_id)s AND state='signing_pending'
                RETURNING batch_id
                """,
                {
                    "batch_id": row["batch_id"],
                    "signer_id": signer_id,
                    "lease_seconds": max(1, int(lease_seconds)),
                },
            )
            if not cur.fetchone():
                raise WorkspaceSignerError("signing batch changed while claiming outbox")
            return dict(row)


def complete_signing_outbox(
    conn,
    *,
    batch_id: str,
    signer_id: str,
    lease_generation: int,
    signature: str,
) -> None:
    signature_sha256 = hashlib.sha256(base64.b64decode(signature)).hexdigest()
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE workspace_authorization_outbox
                SET state='signed', lease_expires_at=NULL, updated_at=now(), last_error=NULL
                WHERE batch_id=%(batch_id)s AND state='leased'
                  AND lease_owner=%(signer_id)s AND lease_generation=%(lease_generation)s
                  AND lease_expires_at > now()
                RETURNING envelope_json, envelope_sha256
                """,
                {
                    "batch_id": batch_id,
                    "signer_id": signer_id,
                    "lease_generation": int(lease_generation),
                },
            )
            outbox = cur.fetchone()
            if not outbox:
                raise WorkspaceSignerError("signer lease changed before signature receipt")
            if sha256_json(outbox["envelope_json"]) != outbox["envelope_sha256"]:
                raise WorkspaceSignerError("outbox envelope digest changed")
            cur.execute(
                """
                UPDATE workspace_cleanup_batches
                SET state='token_issued', signature=%(signature)s,
                  signature_sha256=%(signature_sha256)s, issued_at=now(), updated_at=now()
                WHERE batch_id=%(batch_id)s AND state='signer_claimed'
                  AND signer_lease_owner=%(signer_id)s
                  AND signer_lease_expires_at > now()
                RETURNING control_revision
                """,
                {
                    "batch_id": batch_id,
                    "signer_id": signer_id,
                    "signature": signature,
                    "signature_sha256": signature_sha256,
                },
            )
            batch = cur.fetchone()
            if not batch:
                raise WorkspaceSignerError("cleanup batch changed before token issuance")
            evidence = {
                "batch_id": batch_id,
                "signer_id": signer_id,
                "lease_generation": int(lease_generation),
                "signature_sha256": signature_sha256,
                "envelope_sha256": outbox["envelope_sha256"],
            }
            cur.execute(
                """
                INSERT INTO workspace_authorization_history (
                  batch_id, transition, actor, control_revision,
                  evidence_sha256, evidence_json
                ) VALUES (
                  %(batch_id)s, 'token_issued', %(signer_id)s, %(control_revision)s,
                  %(evidence_sha256)s, %(evidence_json)s
                )
                """,
                {
                    "batch_id": batch_id,
                    "signer_id": signer_id,
                    "control_revision": int(batch["control_revision"]),
                    "evidence_sha256": sha256_json(evidence),
                    "evidence_json": json_arg(evidence),
                },
            )


def fail_signing_outbox(
    conn,
    *,
    batch_id: str,
    signer_id: str,
    lease_generation: int,
    error: str,
) -> None:
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE workspace_authorization_outbox
                SET state='pending', lease_owner=NULL, lease_expires_at=NULL,
                  next_retry_at=now() + interval '5 seconds', last_error=%(error)s,
                  updated_at=now()
                WHERE batch_id=%(batch_id)s AND state='leased'
                  AND lease_owner=%(signer_id)s AND lease_generation=%(lease_generation)s
                """,
                {
                    "batch_id": batch_id,
                    "signer_id": signer_id,
                    "lease_generation": int(lease_generation),
                    "error": error,
                },
            )
            cur.execute(
                """
                UPDATE workspace_cleanup_batches
                SET state='signing_pending', signer_lease_owner=NULL,
                  signer_lease_expires_at=NULL, last_error=%(error)s, updated_at=now()
                WHERE batch_id=%(batch_id)s AND state='signer_claimed'
                  AND signer_lease_owner=%(signer_id)s
                """,
                {"batch_id": batch_id, "signer_id": signer_id, "error": error},
            )


def signer_database_url(database_env_file: Path | None = None) -> str:
    load_env_file(database_env_file or ".env")
    value = str(os.environ.get("WORKSPACE_SIGNER_DATABASE_URL") or "").strip()
    if not value:
        raise SystemExit("WORKSPACE_SIGNER_DATABASE_URL must be set for the workspace signer")
    return value


def signer_once(
    *,
    private_key_path: Path,
    key_revision: str,
    signer_id: str,
    database_env_file: Path | None = None,
) -> bool:
    conn = connect(signer_database_url(database_env_file))
    try:
        heartbeat_signer(
            conn,
            signer_id=signer_id,
            key_revision=key_revision,
            state="starting",
        )
        claimed = claim_signing_outbox(
            conn, signer_id=signer_id, key_revision=key_revision
        )
        if claimed is None:
            heartbeat_signer(
                conn,
                signer_id=signer_id,
                key_revision=key_revision,
                state="healthy",
            )
            return False
        try:
            signature = sign_envelope(
                claimed["envelope_json"], private_key_path=private_key_path
            )
            complete_signing_outbox(
                conn,
                batch_id=str(claimed["batch_id"]),
                signer_id=signer_id,
                lease_generation=int(claimed["lease_generation"]),
                signature=signature,
            )
        except Exception as exc:
            fail_signing_outbox(
                conn,
                batch_id=str(claimed["batch_id"]),
                signer_id=signer_id,
                lease_generation=int(claimed["lease_generation"]),
                error=str(exc),
            )
            heartbeat_signer(
                conn,
                signer_id=signer_id,
                key_revision=key_revision,
                state="degraded",
                error=str(exc),
            )
            raise
        heartbeat_signer(
            conn,
            signer_id=signer_id,
            key_revision=key_revision,
            state="healthy",
        )
        return True
    finally:
        conn.close()


def write_signer_status(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(dict(payload), stream, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="isolated rlab workspace authorization signer")
    parser.add_argument("--private-key", type=Path, required=True)
    parser.add_argument("--key-revision", required=True)
    parser.add_argument("--signer-id", default=f"signer-{socket.gethostname()}-{os.getpid()}")
    parser.add_argument("--status-file", type=Path)
    parser.add_argument("--database-env-file", type=Path)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    args = parser.parse_args(argv)
    while True:
        started = datetime.now(UTC)
        error = None
        worked = False
        try:
            worked = signer_once(
                private_key_path=args.private_key,
                key_revision=str(args.key_revision),
                signer_id=str(args.signer_id),
                database_env_file=args.database_env_file,
            )
        except Exception as exc:
            error = str(exc)
        if args.status_file:
            write_signer_status(
                args.status_file,
                {
                    "protocol_version": SIGNER_PROTOCOL_VERSION,
                    "signer_id": args.signer_id,
                    "key_revision": args.key_revision,
                    "updated_at": datetime.now(UTC).isoformat(),
                    "pass_started_at": started.isoformat(),
                    "worked": worked,
                    "error": error,
                },
            )
        if error:
            print(error, file=sys.stderr, flush=True)
        if args.once:
            return 1 if error else 0
        time.sleep(max(0.1, float(args.poll_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())
