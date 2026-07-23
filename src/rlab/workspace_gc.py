from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Mapping, Sequence

from rlab.job_queue import (
    acquire_fleet_admission_xact_lock,
    acquire_machine_control_xact_lock,
    acquire_work_creation_xact_lock,
    json_arg,
)
from rlab.workspace_contract import (
    CleanupBatchEnvelope,
    CleanupRowEnvelope,
    WorkspaceManifest,
    sha256_json,
    workspace_manifest_from_mapping,
)


TERMINAL_TELEMETRY_DISPOSITIONS = frozenset(
    {
        "published",
        "zero_batch",
        "disabled",
        "aborted_before_release",
        "recovered_final",
    }
)
IRREVERSIBLE_WORKSPACE_STATES = frozenset(
    {"deleting", "host_deleted", "completed", "rollback_review"}
)


class WorkspaceNotReady(RuntimeError):
    pass


class WorkspaceStateConflict(RuntimeError):
    pass


@dataclass(frozen=True)
class DurabilityReceipt:
    train_job_id: int
    ledger_id: int
    object_kind: str
    object_uri: str
    object_version: str
    size_bytes: int
    sha256: str
    full_read_verified_at: datetime
    verifier_identity: str
    policy_scope: str
    policy_sha256: str
    non_expiring_write_once: bool
    runtime_delete_denied: bool
    runtime_overwrite_denied: bool
    storage_root_nonoverlap_sha256: str
    receipt_json: Mapping[str, Any]

    def validate(self) -> None:
        if self.train_job_id < 1 or self.ledger_id < 1:
            raise ValueError("durability receipt job and ledger ids must be positive")
        if self.object_kind not in {"model", "metadata", "recipe"}:
            raise ValueError("durability receipt has unsupported object kind")
        if self.size_bytes < 0:
            raise ValueError("durability receipt size must be nonnegative")
        for label, digest in (
            ("sha256", self.sha256),
            ("policy_sha256", self.policy_sha256),
            ("storage_root_nonoverlap_sha256", self.storage_root_nonoverlap_sha256),
        ):
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError(f"durability receipt {label} is not a lowercase SHA-256")
        if not self.object_uri or not self.object_version or not self.verifier_identity:
            raise ValueError("durability receipt is missing immutable object identity")
        if not self.non_expiring_write_once:
            raise ValueError("cleanup requires a non-expiring write-once storage policy")
        if not self.runtime_delete_denied or not self.runtime_overwrite_denied:
            raise ValueError("runtime storage role must be denied object delete and overwrite")


def persist_workspace_manifest(
    conn,
    *,
    manifest: WorkspaceManifest,
    manage_transaction: bool = True,
) -> dict[str, Any]:
    def persist() -> dict[str, Any]:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO workspace_manifests (
                  launch_id, generation, machine, layout_version, helper_protocol_version,
                  reservation_nonce, manifest_sha256, payload_sha256, manifest_json,
                  payload_path, env_path, output_path, ownership_marker_path,
                  reservation_receipt_path, container_payload_path, container_output_path
                ) VALUES (
                  %(launch_id)s, %(generation)s, %(machine)s, %(layout_version)s,
                  %(protocol_version)s, %(reservation_nonce)s, %(manifest_sha256)s,
                  %(payload_sha256)s, %(manifest_json)s, %(payload_path)s, %(env_path)s,
                  %(output_path)s, %(ownership_marker_path)s, %(reservation_receipt_path)s,
                  %(container_payload_path)s, %(container_output_path)s
                )
                ON CONFLICT (launch_id, generation) DO UPDATE
                SET manifest_sha256 = workspace_manifests.manifest_sha256
                WHERE workspace_manifests.manifest_sha256 = EXCLUDED.manifest_sha256
                RETURNING *
                """,
                {
                    **manifest.as_dict(),
                    "manifest_sha256": manifest.digest,
                    "manifest_json": json_arg(manifest.as_dict()),
                },
            )
            row = cur.fetchone()
            if not row:
                raise WorkspaceStateConflict(
                    f"workspace manifest changed for {manifest.launch_id}/{manifest.generation}"
                )
            return dict(row)

    if manage_transaction:
        with conn:
            return persist()
    return persist()


def attest_launch_workspace_layout(
    conn,
    *,
    launch_id: str,
    generation: int,
    container_generation: int,
) -> None:
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE job_launches launch
                SET workspace_layout_version=1
                WHERE launch.launch_id=%(launch_id)s
                  AND launch.lifecycle_generation=%(generation)s
                  AND (launch.workspace_layout_version IS NULL
                    OR launch.workspace_layout_version=1)
                  AND EXISTS (
                    SELECT 1 FROM workspace_container_generations container_generation
                    WHERE container_generation.launch_id=launch.launch_id
                      AND container_generation.workspace_generation=launch.lifecycle_generation
                      AND container_generation.container_generation=%(container_generation)s
                      AND container_generation.bootstrap_release_state='released'
                  )
                  AND NOT EXISTS (
                    SELECT 1 FROM workspace_container_members member
                    WHERE member.launch_id=launch.launch_id
                      AND member.workspace_generation=launch.lifecycle_generation
                      AND member.container_generation=%(container_generation)s
                      AND (member.mount_attestation IS NULL
                        OR member.state NOT IN ('released','running','absent'))
                  )
                RETURNING launch_id
                """,
                {
                    "launch_id": launch_id,
                    "generation": int(generation),
                    "container_generation": int(container_generation),
                },
            )
            if not cur.fetchone():
                raise WorkspaceStateConflict("workspace layout lacks complete mount attestation")


def load_workspace_manifest(
    conn, *, launch_id: str, generation: int
) -> tuple[WorkspaceManifest, dict[str, Any]] | None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT manifest_json, reservation_intent_state, reservation_receipt,
              reservation_receipt_sha256, intent_cleaned_at
            FROM workspace_manifests
            WHERE launch_id=%(launch_id)s AND generation=%(generation)s
            """,
            {"launch_id": launch_id, "generation": int(generation)},
        )
        row = cur.fetchone()
    if not row:
        return None
    return workspace_manifest_from_mapping(row["manifest_json"]), dict(row)


def persist_reservation_receipt(
    conn,
    *,
    launch_id: str,
    generation: int,
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    digest = str(receipt.get("receipt_sha256") or "")
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE workspace_manifests
                SET reservation_intent_state = 'reserved',
                    reservation_receipt = %(receipt)s,
                    reservation_receipt_sha256 = %(digest)s,
                    reserved_at = COALESCE(reserved_at, now()),
                    intent_last_error = NULL,
                    intent_next_retry_at = NULL
                WHERE launch_id = %(launch_id)s AND generation = %(generation)s
                  AND manifest_sha256 = %(manifest_sha256)s
                  AND reservation_nonce = %(reservation_nonce)s
                  AND (
                    reservation_receipt_sha256 IS NULL
                    OR reservation_receipt_sha256 = %(digest)s
                  )
                RETURNING *
                """,
                {
                    "launch_id": launch_id,
                    "generation": int(generation),
                    "receipt": json_arg(dict(receipt)),
                    "digest": digest,
                    "manifest_sha256": str(receipt.get("manifest_sha256") or ""),
                    "reservation_nonce": str(receipt.get("reservation_nonce") or ""),
                },
            )
            row = cur.fetchone()
            if not row:
                raise WorkspaceStateConflict("reservation receipt does not match durable manifest")
            return dict(row)


def mark_reservation_intent_cleaned(
    conn, *, launch_id: str, generation: int, receipt_sha256: str
) -> None:
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE workspace_manifests
                SET reservation_intent_state = 'cleaned', intent_cleaned_at = now(),
                    intent_last_error = NULL, intent_next_retry_at = NULL
                WHERE launch_id = %(launch_id)s AND generation = %(generation)s
                  AND reservation_intent_state IN ('reserved', 'cleaned')
                  AND reservation_receipt_sha256 = %(receipt_sha256)s
                RETURNING launch_id
                """,
                {
                    "launch_id": launch_id,
                    "generation": int(generation),
                    "receipt_sha256": receipt_sha256,
                },
            )
            if not cur.fetchone():
                raise WorkspaceStateConflict("reservation intent cleanup evidence changed")


def mark_reservation_recovered_for_retry(
    conn, *, launch_id: str, generation: int, recovery: Mapping[str, Any]
) -> None:
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE workspace_manifests
                SET reservation_intent_state='pending',
                  reservation_intent_receipt=%(recovery)s,
                  reservation_receipt=NULL, reservation_receipt_sha256=NULL,
                  reserved_at=NULL, intent_cleaned_at=now(),
                  intent_attempts=intent_attempts+1,
                  intent_last_error=NULL, intent_next_retry_at=NULL
                WHERE launch_id=%(launch_id)s AND generation=%(generation)s
                  AND reservation_receipt_sha256 IS NULL
                RETURNING launch_id
                """,
                {
                    "launch_id": launch_id,
                    "generation": int(generation),
                    "recovery": json_arg(dict(recovery)),
                },
            )
            if not cur.fetchone():
                raise WorkspaceStateConflict("reservation recovery raced a durable receipt")


def register_container_generation(
    conn,
    *,
    launch_id: str,
    workspace_generation: int,
    container_generation: int,
    purpose: str,
    env_path: str,
    env_identity: Mapping[str, Any],
    env_fingerprint_sha256: str,
    members: Sequence[Mapping[str, Any]],
) -> None:
    if not members or len(members) > 4:
        raise ValueError("container generation must have one to four expected members")
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO workspace_container_generations (
                  launch_id, workspace_generation, container_generation, purpose,
                  env_path, env_device, env_inode, env_fingerprint_sha256,
                  expected_member_count
                ) VALUES (
                  %(launch_id)s, %(workspace_generation)s, %(container_generation)s,
                  %(purpose)s, %(env_path)s, %(env_device)s, %(env_inode)s,
                  %(env_fingerprint_sha256)s, %(expected_member_count)s
                )
                ON CONFLICT (launch_id, workspace_generation, container_generation) DO NOTHING
                RETURNING launch_id
                """,
                {
                    "launch_id": launch_id,
                    "workspace_generation": int(workspace_generation),
                    "container_generation": int(container_generation),
                    "purpose": purpose,
                    "env_path": env_path,
                    "env_device": int(env_identity["device"]),
                    "env_inode": int(env_identity["inode"]),
                    "env_fingerprint_sha256": env_fingerprint_sha256,
                    "expected_member_count": len(members),
                },
            )
            if not cur.fetchone():
                cur.execute(
                    """
                    SELECT env_path, env_device, env_inode, env_fingerprint_sha256,
                      expected_member_count, purpose
                    FROM workspace_container_generations
                    WHERE launch_id=%(launch_id)s
                      AND workspace_generation=%(workspace_generation)s
                      AND container_generation=%(container_generation)s
                    """,
                    {
                        "launch_id": launch_id,
                        "workspace_generation": int(workspace_generation),
                        "container_generation": int(container_generation),
                    },
                )
                existing = cur.fetchone()
                expected = {
                    "env_path": env_path,
                    "env_device": int(env_identity["device"]),
                    "env_inode": int(env_identity["inode"]),
                    "env_fingerprint_sha256": env_fingerprint_sha256,
                    "expected_member_count": len(members),
                    "purpose": purpose,
                }
                if not existing or any(existing[key] != value for key, value in expected.items()):
                    raise WorkspaceStateConflict("container generation registration changed")
            for ordinal, member in enumerate(members):
                cur.execute(
                    """
                    INSERT INTO workspace_container_members (
                      launch_id, workspace_generation, container_generation,
                      member_kind, ordinal, container_name, runtime_image_ref
                    ) VALUES (
                      %(launch_id)s, %(workspace_generation)s, %(container_generation)s,
                      %(member_kind)s, %(ordinal)s, %(container_name)s, %(runtime_image_ref)s
                    )
                    ON CONFLICT (
                      launch_id, workspace_generation, container_generation, member_kind, ordinal
                    ) DO NOTHING
                    """,
                    {
                        "launch_id": launch_id,
                        "workspace_generation": int(workspace_generation),
                        "container_generation": int(container_generation),
                        "member_kind": str(member["member_kind"]),
                        "ordinal": ordinal,
                        "container_name": str(member["container_name"]),
                        "runtime_image_ref": str(member["runtime_image_ref"]),
                    },
                )


def begin_recovery_container_generation(
    conn,
    *,
    launch_id: str,
    workspace_generation: int,
    purpose: str,
    env_path: str,
    member_kind: str,
    container_name: str,
    runtime_image_ref: str,
) -> int:
    if purpose not in {"checkpoint_recovery", "wandb_recovery"}:
        raise ValueError("invalid recovery generation purpose")
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT container_generation, env_cleanup_state,
                  NOT EXISTS (
                    SELECT 1 FROM workspace_container_members member
                    WHERE member.launch_id=generation.launch_id
                      AND member.workspace_generation=generation.workspace_generation
                      AND member.container_generation=generation.container_generation
                      AND member.state <> 'absent'
                  ) AS members_absent
                FROM workspace_container_generations generation
                WHERE launch_id=%(launch_id)s
                  AND workspace_generation=%(workspace_generation)s
                ORDER BY container_generation
                FOR UPDATE
                """,
                {
                    "launch_id": launch_id,
                    "workspace_generation": int(workspace_generation),
                },
            )
            prior_rows = [dict(row) for row in cur.fetchall()]
            if any(
                row["env_cleanup_state"] != "unlinked" or not row["members_absent"]
                for row in prior_rows
            ):
                raise WorkspaceStateConflict(
                    "an earlier credential generation is not durably clean"
                )
            latest = max(
                (int(row["container_generation"]) for row in prior_rows), default=0
            )
            container_generation = max(2, latest + 1)
            cur.execute(
                """
                INSERT INTO workspace_container_generations (
                  launch_id, workspace_generation, container_generation, purpose,
                  env_path, expected_member_count, env_intent_cleanup_state
                ) VALUES (
                  %(launch_id)s, %(workspace_generation)s, %(container_generation)s,
                  %(purpose)s, %(env_path)s, 1, 'not_created'
                )
                """,
                {
                    "launch_id": launch_id,
                    "workspace_generation": int(workspace_generation),
                    "container_generation": container_generation,
                    "purpose": purpose,
                    "env_path": env_path,
                },
            )
            cur.execute(
                """
                INSERT INTO workspace_container_members (
                  launch_id, workspace_generation, container_generation,
                  member_kind, ordinal, container_name, runtime_image_ref
                ) VALUES (
                  %(launch_id)s, %(workspace_generation)s, %(container_generation)s,
                  %(member_kind)s, 0, %(container_name)s, %(runtime_image_ref)s
                )
                """,
                {
                    "launch_id": launch_id,
                    "workspace_generation": int(workspace_generation),
                    "container_generation": container_generation,
                    "member_kind": member_kind,
                    "container_name": container_name,
                    "runtime_image_ref": runtime_image_ref,
                },
            )
            return container_generation


def complete_recovery_env_reservation(
    conn,
    *,
    launch_id: str,
    workspace_generation: int,
    container_generation: int,
    receipt: Mapping[str, Any],
) -> None:
    identity = dict(receipt.get("env_identity") or {})
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE workspace_container_generations
                SET env_device=%(device)s, env_inode=%(inode)s,
                  env_fingerprint_sha256=%(fingerprint)s,
                  env_reservation_receipt=%(receipt)s,
                  env_intent_cleanup_state='present'
                WHERE launch_id=%(launch_id)s
                  AND workspace_generation=%(workspace_generation)s
                  AND container_generation=%(container_generation)s
                  AND env_device IS NULL AND env_inode IS NULL
                RETURNING launch_id
                """,
                {
                    "launch_id": launch_id,
                    "workspace_generation": int(workspace_generation),
                    "container_generation": int(container_generation),
                    "device": int(identity["device"]),
                    "inode": int(identity["inode"]),
                    "fingerprint": str(receipt["env_fingerprint_sha256"]),
                    "receipt": json_arg(dict(receipt)),
                },
            )
            if not cur.fetchone():
                raise WorkspaceStateConflict("recovery env reservation changed")


def mark_recovery_env_intent_cleaned(
    conn, *, launch_id: str, workspace_generation: int, container_generation: int
) -> None:
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE workspace_container_generations
                SET env_intent_cleanup_state='cleaned'
                WHERE launch_id=%(launch_id)s
                  AND workspace_generation=%(workspace_generation)s
                  AND container_generation=%(container_generation)s
                  AND env_intent_cleanup_state IN ('present','cleaned')
                RETURNING launch_id
                """,
                {
                    "launch_id": launch_id,
                    "workspace_generation": int(workspace_generation),
                    "container_generation": int(container_generation),
                },
            )
            if not cur.fetchone():
                raise WorkspaceStateConflict("recovery env intent cleanup changed")
def mark_container_member_state(
    conn,
    *,
    launch_id: str,
    workspace_generation: int,
    container_generation: int,
    member_kind: str,
    state: str,
    container_id: str | None = None,
    mount_attestation: Mapping[str, Any] | None = None,
) -> None:
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE workspace_container_members
                SET state=%(state)s,
                  container_id=COALESCE(%(container_id)s, container_id),
                  mount_attestation=COALESCE(%(mount_attestation)s, mount_attestation),
                  absent_at=CASE WHEN %(state)s='absent' THEN COALESCE(absent_at, now())
                    ELSE absent_at END
                WHERE launch_id=%(launch_id)s
                  AND workspace_generation=%(workspace_generation)s
                  AND container_generation=%(container_generation)s
                  AND member_kind=%(member_kind)s
                RETURNING launch_id
                """,
                {
                    "launch_id": launch_id,
                    "workspace_generation": int(workspace_generation),
                    "container_generation": int(container_generation),
                    "member_kind": member_kind,
                    "state": state,
                    "container_id": container_id,
                    "mount_attestation": (
                        json_arg(dict(mount_attestation)) if mount_attestation is not None else None
                    ),
                },
            )
            if not cur.fetchone():
                raise WorkspaceStateConflict("container member does not exist")


def mark_container_generation_released(
    conn,
    *,
    launch_id: str,
    workspace_generation: int,
    container_generation: int,
) -> None:
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE workspace_container_generations g
                SET bootstrap_release_state='released', env_cleanup_state='eligible'
                WHERE launch_id=%(launch_id)s
                  AND workspace_generation=%(workspace_generation)s
                  AND container_generation=%(container_generation)s
                  AND (
                    SELECT COUNT(*) FROM workspace_container_members m
                    WHERE m.launch_id=g.launch_id
                      AND m.workspace_generation=g.workspace_generation
                      AND m.container_generation=g.container_generation
                      AND m.state IN ('created','attested_unreleased','released','running','absent')
                  ) = g.expected_member_count
                RETURNING launch_id
                """,
                {
                    "launch_id": launch_id,
                    "workspace_generation": int(workspace_generation),
                    "container_generation": int(container_generation),
                },
            )
            if not cur.fetchone():
                raise WorkspaceStateConflict("not every expected container member consumed the env")


def mark_env_unlinked(
    conn,
    *,
    launch_id: str,
    workspace_generation: int,
    container_generation: int,
    receipt: Mapping[str, Any],
) -> None:
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE workspace_container_generations
                SET env_cleanup_state='unlinked', env_unlinked_at=COALESCE(env_unlinked_at, now()),
                  env_cleanup_receipt=%(receipt)s, env_cleanup_error=NULL,
                  env_cleanup_next_retry_at=NULL
                WHERE launch_id=%(launch_id)s
                  AND workspace_generation=%(workspace_generation)s
                  AND container_generation=%(container_generation)s
                  AND env_cleanup_state IN ('eligible','unlinking','unlinked')
                RETURNING launch_id
                """,
                {
                    "launch_id": launch_id,
                    "workspace_generation": int(workspace_generation),
                    "container_generation": int(container_generation),
                    "receipt": json_arg(dict(receipt)),
                },
            )
            if not cur.fetchone():
                raise WorkspaceStateConflict("env generation is not eligible for cleanup")


def terminal_credential_cleanup_targets(
    conn, *, machine: str, limit: int = 32
) -> list[dict[str, Any]]:
    """Return exact, manifest-bound terminal credential cleanup work.

    This deliberately does not infer paths or container names from the machine registry.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT generation.launch_id, generation.workspace_generation,
              generation.container_generation, generation.env_path,
              generation.env_device, generation.env_inode,
              generation.env_cleanup_state, manifest.manifest_json,
              generation.env_reservation_receipt, generation.env_intent_cleanup_state,
              COALESCE(jsonb_agg(jsonb_build_object(
                'member_kind', member.member_kind,
                'ordinal', member.ordinal,
                'container_name', member.container_name,
                'container_id', member.container_id,
                'state', member.state
              ) ORDER BY member.ordinal) FILTER (WHERE member.member_kind IS NOT NULL), '[]')
                AS members
            FROM workspace_container_generations generation
            JOIN workspace_manifests manifest
              ON manifest.launch_id=generation.launch_id
             AND manifest.generation=generation.workspace_generation
            JOIN job_launches launch ON launch.launch_id=generation.launch_id
            LEFT JOIN workspace_container_members member
              ON member.launch_id=generation.launch_id
             AND member.workspace_generation=generation.workspace_generation
             AND member.container_generation=generation.container_generation
            WHERE manifest.machine=%(machine)s
              AND launch.state IN ('succeeded','failed','canceled')
              AND (
                generation.env_cleanup_state <> 'unlinked'
                OR generation.env_intent_cleanup_state IN ('present','failed')
                OR EXISTS (
                  SELECT 1 FROM workspace_container_members pending_member
                  WHERE pending_member.launch_id=generation.launch_id
                    AND pending_member.workspace_generation=generation.workspace_generation
                    AND pending_member.container_generation=generation.container_generation
                    AND pending_member.state <> 'absent'
                )
              )
              AND (
                generation.env_cleanup_next_retry_at IS NULL
                OR generation.env_cleanup_next_retry_at <= now()
              )
            GROUP BY generation.launch_id, generation.workspace_generation,
              generation.container_generation, generation.env_path,
              generation.env_device, generation.env_inode,
              generation.env_cleanup_state, manifest.manifest_json, generation.created_at
              , generation.env_reservation_receipt, generation.env_intent_cleanup_state
            ORDER BY generation.created_at
            LIMIT %(limit)s
            """,
            {"machine": machine, "limit": max(1, min(int(limit), 128))},
        )
        return [dict(row) for row in cur.fetchall()]


def drain_inventory_request(conn, *, machine: str) -> dict[str, Any] | None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT machine, control_revision
            FROM machine_controls
            WHERE machine=%(machine)s AND drain_requested AND NOT drained
              AND NOT host_mutation_quarantined
            """,
            {"machine": machine},
        )
        control = cur.fetchone()
        if not control:
            return None
        cur.execute(
            """
            SELECT manifest_json
            FROM workspace_manifests
            WHERE machine=%(machine)s
            ORDER BY launch_id, generation
            """,
            {"machine": machine},
        )
        manifests = [dict(row["manifest_json"]) for row in cur.fetchall()]
        cur.execute(
            """
            SELECT member.container_name
            FROM workspace_container_members member
            JOIN workspace_manifests manifest
              ON manifest.launch_id=member.launch_id
             AND manifest.generation=member.workspace_generation
            WHERE manifest.machine=%(machine)s AND member.state <> 'absent'
            ORDER BY member.container_name
            """,
            {"machine": machine},
        )
        names = [str(row["container_name"]) for row in cur.fetchall()]
        cur.execute(
            """
            SELECT row.prepare_receipt_path,
              manifest.manifest_json->'roots'->>'protected_metadata_root' AS protected_root,
              row.launch_id, row.lifecycle_generation
            FROM workspace_cleanup_rows row
            JOIN workspace_manifests manifest
              ON manifest.launch_id=row.launch_id
             AND manifest.generation=row.lifecycle_generation
            WHERE manifest.machine=%(machine)s AND (
              row.prepare_cleanup_state NOT IN ('not_created','cleaned')
              OR row.journal_cleanup_state NOT IN ('not_created','cleaned')
            )
            """,
            {"machine": machine},
        )
        protected_paths: list[str] = []
        for row in cur.fetchall():
            if row.get("prepare_receipt_path"):
                protected_paths.append(str(row["prepare_receipt_path"]))
            if row.get("protected_root"):
                protected_paths.append(
                    f"{str(row['protected_root']).rstrip('/')}/{row['launch_id']}."
                    f"host-deleted.g{int(row['lifecycle_generation'])}.json"
                )
        return {
            "machine": machine,
            "control_revision": int(control["control_revision"]),
            "manifests": manifests,
            "container_names": names,
            "protected_paths": protected_paths,
        }


def mark_terminal_member_absent(
    conn,
    *,
    launch_id: str,
    workspace_generation: int,
    container_generation: int,
    member_kind: str,
    ordinal: int,
    receipt: Mapping[str, Any],
) -> None:
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE workspace_container_members
                SET state='absent', absent_at=COALESCE(absent_at, now()),
                  mount_attestation=COALESCE(mount_attestation, '{}'::jsonb)
                    || jsonb_build_object('terminal_absence_receipt', %(receipt)s::jsonb)
                WHERE launch_id=%(launch_id)s
                  AND workspace_generation=%(workspace_generation)s
                  AND container_generation=%(container_generation)s
                  AND member_kind=%(member_kind)s AND ordinal=%(ordinal)s
                  AND state IN (
                    'expected','created','attested_unreleased','released','running',
                    'exited','removing','absent','failed'
                  )
                RETURNING launch_id
                """,
                {
                    "launch_id": launch_id,
                    "workspace_generation": int(workspace_generation),
                    "container_generation": int(container_generation),
                    "member_kind": member_kind,
                    "ordinal": int(ordinal),
                    "receipt": json_arg(dict(receipt)),
                },
            )
            if not cur.fetchone():
                raise WorkspaceStateConflict("terminal container member evidence changed")


def mark_terminal_credential_cleanup_failure(
    conn,
    *,
    launch_id: str,
    workspace_generation: int,
    container_generation: int,
    error: str,
) -> None:
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE workspace_container_generations
                SET env_cleanup_state='failed', env_cleanup_attempts=env_cleanup_attempts+1,
                  env_cleanup_error=%(error)s,
                  env_cleanup_next_retry_at=now() + interval '30 seconds'
                WHERE launch_id=%(launch_id)s
                  AND workspace_generation=%(workspace_generation)s
                  AND container_generation=%(container_generation)s
                """,
                {
                    "launch_id": launch_id,
                    "workspace_generation": int(workspace_generation),
                    "container_generation": int(container_generation),
                    "error": error[:4000],
                },
            )


def make_terminal_env_eligible(
    conn, *, launch_id: str, workspace_generation: int, container_generation: int
) -> None:
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE workspace_container_generations generation
                SET env_cleanup_state=CASE
                    WHEN env_cleanup_state='unlinked' THEN 'unlinked' ELSE 'eligible' END,
                  env_cleanup_error=NULL, env_cleanup_next_retry_at=NULL
                WHERE launch_id=%(launch_id)s
                  AND workspace_generation=%(workspace_generation)s
                  AND container_generation=%(container_generation)s
                  AND NOT EXISTS (
                    SELECT 1 FROM workspace_container_members member
                    WHERE member.launch_id=generation.launch_id
                      AND member.workspace_generation=generation.workspace_generation
                      AND member.container_generation=generation.container_generation
                      AND member.state <> 'absent'
                  )
                RETURNING launch_id
                """,
                {
                    "launch_id": launch_id,
                    "workspace_generation": int(workspace_generation),
                    "container_generation": int(container_generation),
                },
            )
            if not cur.fetchone():
                raise WorkspaceStateConflict("credential containers are not all absent")


def create_telemetry_obligation(
    conn,
    *,
    train_job_id: int,
    launch_id: str,
    lifecycle_generation: int,
    producer_identity: str,
    sink: str,
    configuration_revision: str,
    expected_stream_id: str | None,
    manage_transaction: bool = True,
) -> None:
    def insert() -> None:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO telemetry_obligations (
                  train_job_id, launch_id, lifecycle_generation, producer_identity,
                  sink, configuration_revision, expected_stream_id
                ) VALUES (
                  %(train_job_id)s, %(launch_id)s, %(lifecycle_generation)s,
                  %(producer_identity)s, %(sink)s, %(configuration_revision)s,
                  %(expected_stream_id)s
                )
                ON CONFLICT (
                  launch_id, lifecycle_generation, producer_identity, sink,
                  configuration_revision
                ) DO NOTHING
                """,
                {
                    "train_job_id": int(train_job_id),
                    "launch_id": launch_id,
                    "lifecycle_generation": int(lifecycle_generation),
                    "producer_identity": producer_identity,
                    "sink": sink,
                    "configuration_revision": configuration_revision,
                    "expected_stream_id": expected_stream_id,
                },
            )

    if manage_transaction:
        with conn:
            insert()
    else:
        insert()


def close_telemetry_obligation(
    conn,
    *,
    launch_id: str,
    lifecycle_generation: int,
    producer_identity: str,
    sink: str,
    configuration_revision: str,
    disposition: str,
    final_sequence: int | None,
    receipt: Mapping[str, Any],
) -> None:
    if disposition not in TERMINAL_TELEMETRY_DISPOSITIONS:
        raise ValueError("telemetry obligation requires a terminal safe disposition")
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE telemetry_obligations
                SET disposition=%(disposition)s, final_sequence=%(final_sequence)s,
                    receipt_json=%(receipt)s, closed_at=COALESCE(closed_at, now())
                WHERE launch_id=%(launch_id)s
                  AND lifecycle_generation=%(lifecycle_generation)s
                  AND producer_identity=%(producer_identity)s
                  AND sink=%(sink)s
                  AND configuration_revision=%(configuration_revision)s
                  AND disposition IN ('pending', 'failed', %(disposition)s)
                RETURNING id
                """,
                {
                    "launch_id": launch_id,
                    "lifecycle_generation": int(lifecycle_generation),
                    "producer_identity": producer_identity,
                    "sink": sink,
                    "configuration_revision": configuration_revision,
                    "disposition": disposition,
                    "final_sequence": final_sequence,
                    "receipt": json_arg(dict(receipt)),
                },
            )
            if not cur.fetchone():
                raise WorkspaceStateConflict("telemetry obligation is absent or already conflicts")


def record_artifact_durability_receipt(conn, receipt: DurabilityReceipt) -> None:
    receipt.validate()
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO artifact_durability_receipts (
                  train_job_id, ledger_id, object_kind, object_uri, object_version,
                  size_bytes, sha256, full_read_verified_at, verifier_identity,
                  policy_scope, policy_sha256, non_expiring_write_once,
                  runtime_delete_denied, runtime_overwrite_denied,
                  storage_root_nonoverlap_sha256, receipt_json
                ) VALUES (
                  %(train_job_id)s, %(ledger_id)s, %(object_kind)s, %(object_uri)s,
                  %(object_version)s, %(size_bytes)s, %(sha256)s,
                  %(full_read_verified_at)s, %(verifier_identity)s, %(policy_scope)s,
                  %(policy_sha256)s, %(non_expiring_write_once)s,
                  %(runtime_delete_denied)s, %(runtime_overwrite_denied)s,
                  %(storage_root_nonoverlap_sha256)s, %(receipt_json)s
                )
                ON CONFLICT (train_job_id, ledger_id, object_kind) DO UPDATE
                SET object_uri = artifact_durability_receipts.object_uri
                WHERE artifact_durability_receipts.object_uri = EXCLUDED.object_uri
                  AND artifact_durability_receipts.object_version = EXCLUDED.object_version
                  AND artifact_durability_receipts.sha256 = EXCLUDED.sha256
                  AND artifact_durability_receipts.policy_sha256 = EXCLUDED.policy_sha256
                RETURNING train_job_id
                """,
                {**receipt.__dict__, "receipt_json": json_arg(dict(receipt.receipt_json))},
            )
            if not cur.fetchone():
                raise WorkspaceStateConflict("artifact durability receipt changed")


def _proof_snapshot(conn, *, launch_id: str) -> dict[str, Any]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT l.launch_id, l.lifecycle_generation, l.workspace_layout_version,
              l.state AS launch_state, l.finished_at AS launch_finished_at,
              t.id AS train_job_id, t.status AS train_status,
              t.live_publication_status, t.train_config,
              t.telemetry_protocol_version, t.telemetry_generation,
              i.exact AS telemetry_exact,
              i.cleanup_eligible AS telemetry_cleanup_eligible,
              i.disposition AS telemetry_integrity_disposition,
              r.status AS eval_status, r.complete_announcement_seen, r.outcome AS eval_outcome
            FROM job_launches l
            JOIN train_jobs t ON t.id=l.job_id
            LEFT JOIN telemetry_integrity i
              ON i.train_job_id=t.id
             AND i.telemetry_generation=t.telemetry_generation
            LEFT JOIN eval_runs r ON r.train_job_id=t.id
            WHERE l.launch_id=%(launch_id)s
            FOR UPDATE OF l, t
            """,
            {"launch_id": launch_id},
        )
        root = cur.fetchone()
        if not root:
            raise WorkspaceNotReady("launch does not exist")
        root = dict(root)
        generation = int(root["lifecycle_generation"])
        train_job_id = int(root["train_job_id"])
        if root.get("workspace_layout_version") != 1:
            raise WorkspaceNotReady("launch does not have an attested layout-v1 workspace")
        if root["launch_state"] != "succeeded" or root["train_status"] != "succeeded":
            raise WorkspaceNotReady("training lifecycle is not authoritatively successful")
        protocol_v2 = int(root.get("telemetry_protocol_version") or 1) == 2
        if protocol_v2:
            if not bool(root.get("telemetry_exact")):
                raise WorkspaceNotReady("canonical telemetry integrity is not exact")
            if not bool(root.get("telemetry_cleanup_eligible")):
                raise WorkspaceNotReady("canonical telemetry cleanup is not eligible")
        elif root["live_publication_status"] not in {"complete", "disabled"}:
            raise WorkspaceNotReady("W&B publication is not closed")
        backend = str((root.get("train_config") or {}).get("checkpoint_eval_backend") or "local")
        if backend == "modal" and root.get("eval_status") != "complete":
            raise WorkspaceNotReady("required Modal evaluation is not complete")
        if root.get("eval_status") is not None and root.get("eval_status") != "complete":
            raise WorkspaceNotReady("evaluation lifecycle is not closed")
        if not bool(root.get("complete_announcement_seen")):
            raise WorkspaceNotReady("artifact announcement ledger is not closed")

        cur.execute(
            """
            SELECT ledger_id, disposition, artifact_kind, checkpoint_sha256,
              checkpoint_uri, metadata_sha256, metadata_uri, recipe_sha256, recipe_uri,
              announcement_sha256, verified_at
            FROM artifact_announcement_ledger
            WHERE train_job_id=%(train_job_id)s
            ORDER BY ledger_id
            """,
            {"train_job_id": train_job_id},
        )
        ledger = [dict(row) for row in cur.fetchall()]
        if not ledger or any(row["disposition"] != "ready" for row in ledger):
            raise WorkspaceNotReady("artifact ledger is empty or contains a tombstone")
        cur.execute(
            """
            SELECT ledger_id, object_kind, object_uri, object_version, size_bytes, sha256,
              full_read_verified_at, verifier_identity, policy_sha256,
              non_expiring_write_once, runtime_delete_denied, runtime_overwrite_denied,
              storage_root_nonoverlap_sha256
            FROM artifact_durability_receipts
            WHERE train_job_id=%(train_job_id)s
            ORDER BY ledger_id, object_kind
            """,
            {"train_job_id": train_job_id},
        )
        receipts = [dict(row) for row in cur.fetchall()]
        receipt_keys = {(int(row["ledger_id"]), str(row["object_kind"])) for row in receipts}
        for row in ledger:
            expected = {"model", "metadata"}
            if row.get("recipe_uri"):
                expected.add("recipe")
            missing = expected - {kind for ledger_id, kind in receipt_keys if ledger_id == row["ledger_id"]}
            if missing:
                raise WorkspaceNotReady(
                    f"artifact ledger {row['ledger_id']} lacks durability receipt(s): "
                    + ", ".join(sorted(missing))
                )
        if any(
            not row["non_expiring_write_once"]
            or not row["runtime_delete_denied"]
            or not row["runtime_overwrite_denied"]
            for row in receipts
        ):
            raise WorkspaceNotReady("artifact durability policy is not cleanup-safe")

        cur.execute(
            """
            SELECT id, producer_identity, sink, configuration_revision, disposition,
              expected_stream_id, final_sequence, receipt_json, closed_at
            FROM telemetry_obligations
            WHERE launch_id=%(launch_id)s AND lifecycle_generation=%(generation)s
            ORDER BY id
            """,
            {"launch_id": launch_id, "generation": generation},
        )
        obligations = [dict(row) for row in cur.fetchall()]
        if not obligations and not protocol_v2:
            raise WorkspaceNotReady("expected telemetry obligation set is absent")
        for obligation in obligations:
            if protocol_v2:
                continue
            if obligation["disposition"] not in TERMINAL_TELEMETRY_DISPOSITIONS:
                raise WorkspaceNotReady("telemetry obligations are not durably closed")
            stream_id = obligation.get("expected_stream_id")
            if stream_id:
                cur.execute(
                    """
                    SELECT accepted_sequence, final_sequence, submitted_sequence,
                      published_sequence
                    FROM metric_streams WHERE stream_id=%(stream_id)s
                    """,
                    {"stream_id": stream_id},
                )
                stream = cur.fetchone()
                if obligation["disposition"] in {"published", "recovered_final"}:
                    if (
                        not stream
                        or stream.get("final_sequence") is None
                        or int(stream["published_sequence"]) < int(stream["final_sequence"])
                    ):
                        raise WorkspaceNotReady("telemetry stream final watermark is unpublished")
        cur.execute(
            """
            SELECT COUNT(*) AS count
            FROM metric_batches b
            JOIN metric_streams s ON s.stream_id=b.stream_id
            JOIN worker_attempts w ON w.attempt_id=s.attempt_id
            WHERE w.train_job_id=%(train_job_id)s
            """,
            {"train_job_id": train_job_id},
        )
        if int(cur.fetchone()["count"]):
            raise WorkspaceNotReady("retained telemetry batches remain")

        cur.execute(
            """
            SELECT container_generation, env_cleanup_state, env_intent_cleanup_state
            FROM workspace_container_generations
            WHERE launch_id=%(launch_id)s AND workspace_generation=%(generation)s
            ORDER BY container_generation
            """,
            {"launch_id": launch_id, "generation": generation},
        )
        generations = [dict(row) for row in cur.fetchall()]
        if not generations or any(
            row["env_cleanup_state"] != "unlinked"
            or row["env_intent_cleanup_state"] not in {"not_created", "cleaned"}
            for row in generations
        ):
            raise WorkspaceNotReady("credential environment cleanup is incomplete")
        cur.execute(
            """
            SELECT COUNT(*) AS count
            FROM workspace_container_members
            WHERE launch_id=%(launch_id)s AND workspace_generation=%(generation)s
              AND state <> 'absent'
            """,
            {"launch_id": launch_id, "generation": generation},
        )
        if int(cur.fetchone()["count"]):
            raise WorkspaceNotReady("credential-bearing container objects remain")
        cur.execute(
            """
            SELECT COUNT(*) AS count FROM host_operation_leases
            WHERE launch_id=%(launch_id)s AND lifecycle_generation=%(generation)s
              AND state IN ('registered', 'running', 'reconciling')
            """,
            {"launch_id": launch_id, "generation": generation},
        )
        if int(cur.fetchone()["count"]):
            raise WorkspaceNotReady("host-dependent lifecycle work remains active")
        manifest = None
        cur.execute(
            """
            SELECT manifest_sha256, reservation_receipt_sha256, reserved_at, intent_cleaned_at
            FROM workspace_manifests
            WHERE launch_id=%(launch_id)s AND generation=%(generation)s
            """,
            {"launch_id": launch_id, "generation": generation},
        )
        manifest = cur.fetchone()
        if (
            not manifest
            or not manifest.get("reservation_receipt_sha256")
            or manifest.get("intent_cleaned_at") is None
        ):
            raise WorkspaceNotReady("workspace reservation is incomplete")
    input_times = [
        root.get("launch_finished_at"),
        *(row.get("verified_at") for row in ledger),
        *(row.get("full_read_verified_at") for row in receipts),
        *(row.get("closed_at") for row in obligations),
        manifest.get("intent_cleaned_at"),
    ]
    complete_at = max(value for value in input_times if value is not None)
    return {
        "protocol_version": 1,
        "launch_id": launch_id,
        "lifecycle_generation": generation,
        "train_job_id": train_job_id,
        "manifest_sha256": manifest["manifest_sha256"],
        "reservation_receipt_sha256": manifest["reservation_receipt_sha256"],
        "artifact_ledger": [
            {
                "ledger_id": int(row["ledger_id"]),
                "announcement_sha256": row["announcement_sha256"],
            }
            for row in ledger
        ],
        "durability_receipts": [
            {
                "ledger_id": int(row["ledger_id"]),
                "object_kind": row["object_kind"],
                "object_uri": row["object_uri"],
                "object_version": row["object_version"],
                "sha256": row["sha256"],
                "policy_sha256": row["policy_sha256"],
                "storage_root_nonoverlap_sha256": row["storage_root_nonoverlap_sha256"],
            }
            for row in receipts
        ],
        "telemetry_obligations": [
            {
                "id": int(row["id"]),
                "producer_identity": row["producer_identity"],
                "sink": row["sink"],
                "configuration_revision": row["configuration_revision"],
                "disposition": row["disposition"],
                "final_sequence": row.get("final_sequence"),
                "receipt_sha256": sha256_json(row.get("receipt_json") or {}),
            }
            for row in obligations
        ],
        "proof_inputs_complete_at": complete_at.isoformat(),
    }


def reduce_cleanup_proof(conn, *, launch_id: str) -> dict[str, Any]:
    with conn:
        snapshot = _proof_snapshot(conn, launch_id=launch_id)
        proof_sha256 = sha256_json(snapshot)
        generation = int(snapshot["lifecycle_generation"])
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO workspace_cleanup_proofs (
                  launch_id, lifecycle_generation, proof_sha256, proof_json,
                  proof_inputs_complete_at
                ) VALUES (
                  %(launch_id)s, %(generation)s, %(proof_sha256)s, %(proof_json)s,
                  %(proof_inputs_complete_at)s
                )
                ON CONFLICT (launch_id, lifecycle_generation) DO UPDATE
                SET proof_sha256=workspace_cleanup_proofs.proof_sha256
                WHERE workspace_cleanup_proofs.proof_sha256=EXCLUDED.proof_sha256
                RETURNING *
                """,
                {
                    "launch_id": launch_id,
                    "generation": generation,
                    "proof_sha256": proof_sha256,
                    "proof_json": json_arg(snapshot),
                    "proof_inputs_complete_at": datetime.fromisoformat(
                        snapshot["proof_inputs_complete_at"]
                    ),
                },
            )
            proof = cur.fetchone()
            if not proof:
                raise WorkspaceStateConflict("cleanup proof inputs changed after prior proof")
            cursor = {"phase": "content", "target_index": 0, "stack": []}
            cur.execute(
                """
                INSERT INTO workspace_cleanup_rows (
                  launch_id, lifecycle_generation, manifest_sha256, cursor_json, cursor_sha256
                ) VALUES (
                  %(launch_id)s, %(generation)s, %(manifest_sha256)s,
                  %(cursor_json)s, %(cursor_sha256)s
                )
                ON CONFLICT (launch_id, lifecycle_generation) DO NOTHING
                """,
                {
                    "launch_id": launch_id,
                    "generation": generation,
                    "manifest_sha256": snapshot["manifest_sha256"],
                    "cursor_json": json_arg(cursor),
                    "cursor_sha256": sha256_json(cursor),
                },
            )
            return dict(proof)


def reopen_launch_lifecycle(
    conn,
    *,
    launch_id: str,
    reason: str,
    db_only: bool = False,
) -> int:
    with conn:
        acquire_work_creation_xact_lock(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT l.lifecycle_generation, r.workspace_state
                FROM job_launches l
                LEFT JOIN workspace_cleanup_rows r
                  ON r.launch_id=l.launch_id
                  AND r.lifecycle_generation=l.lifecycle_generation
                WHERE l.launch_id=%(launch_id)s
                FOR UPDATE OF l
                """,
                {"launch_id": launch_id},
            )
            row = cur.fetchone()
            if not row:
                raise ValueError(f"launch does not exist: {launch_id}")
            state = row.get("workspace_state")
            if db_only:
                if state not in {None, "completed"}:
                    raise WorkspaceStateConflict("DB-only restamp is not safe in current state")
                return int(row["lifecycle_generation"])
            if state in IRREVERSIBLE_WORKSPACE_STATES:
                raise WorkspaceStateConflict(
                    f"host-dependent reopen rejected from workspace state {state}"
                )
            generation = int(row["lifecycle_generation"])
            cur.execute(
                "DELETE FROM workspace_cleanup_rows WHERE launch_id=%(launch_id)s "
                "AND lifecycle_generation=%(generation)s AND workspace_state='pending'",
                {"launch_id": launch_id, "generation": generation},
            )
            cur.execute(
                "DELETE FROM workspace_cleanup_proofs WHERE launch_id=%(launch_id)s "
                "AND lifecycle_generation=%(generation)s",
                {"launch_id": launch_id, "generation": generation},
            )
            cur.execute(
                """
                UPDATE job_launches
                SET lifecycle_generation=lifecycle_generation+1,
                    workspace_layout_version=NULL,
                    error=%(reason)s
                WHERE launch_id=%(launch_id)s AND lifecycle_generation=%(generation)s
                RETURNING lifecycle_generation
                """,
                {"launch_id": launch_id, "generation": generation, "reason": reason},
            )
            updated = cur.fetchone()
            if not updated:
                raise WorkspaceStateConflict("launch generation changed during reopen")
            return int(updated["lifecycle_generation"])


def claim_cleanup_rows(
    conn,
    *,
    machine: str,
    limit: int = 8,
) -> list[dict[str, Any]]:
    limit = min(8, max(1, int(limit)))
    attempt_id = f"gc-{uuid.uuid4().hex}"
    with conn:
        acquire_machine_control_xact_lock(conn, machine=machine)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT protocol_mode, cleanup_globally_enabled, control_revision
                FROM workspace_rollout_controls WHERE singleton=TRUE FOR UPDATE
                """
            )
            rollout = cur.fetchone()
            if (
                not rollout
                or not rollout["cleanup_globally_enabled"]
                or rollout["protocol_mode"] not in ('qualification', 'promotion_verifying', 'active')
            ):
                return []
            cur.execute(
                """
                SELECT * FROM machine_controls WHERE machine=%(machine)s FOR UPDATE
                """,
                {"machine": machine},
            )
            control = cur.fetchone()
            if (
                not control
                or control["drain_requested"]
                or control["drained"]
                or control["host_mutation_quarantined"]
            ):
                return []
            machine_enabled = bool(control["cleanup_enabled"])
            if rollout["protocol_mode"] != "qualification" and not machine_enabled:
                return []
            cur.execute(
                """
                WITH due AS (
                  SELECT r.id
                  FROM workspace_cleanup_rows r
                  JOIN job_launches l ON l.launch_id=r.launch_id
                    AND l.lifecycle_generation=r.lifecycle_generation
                  JOIN workspace_cleanup_proofs p ON p.launch_id=r.launch_id
                    AND p.lifecycle_generation=r.lifecycle_generation
                  WHERE l.machine=%(machine)s
                    AND r.workspace_state='pending'
                    AND (
                      %(machine_enabled)s OR EXISTS (
                        SELECT 1 FROM workspace_capabilities capability
                        WHERE capability.capability_kind='cleanup_canary'
                          AND capability.machine=l.machine
                          AND capability.launch_id=r.launch_id
                          AND capability.lifecycle_generation=r.lifecycle_generation
                          AND capability.control_revision=%(rollout_revision)s
                          AND capability.remaining_count > 0
                          AND capability.revoked_at IS NULL
                      )
                    )
                    AND (r.next_retry_at IS NULL OR r.next_retry_at <= now())
                    AND NOT EXISTS (
                      SELECT 1 FROM host_operation_leases h
                      WHERE h.machine=l.machine
                        AND h.state IN ('registered','running','reconciling')
                    )
                  ORDER BY COALESCE(r.next_retry_at, p.cleanup_ready_at), r.id
                  FOR UPDATE OF r SKIP LOCKED
                  LIMIT %(limit)s
                )
                UPDATE workspace_cleanup_rows r
                SET workspace_state='deleting', deletion_progress='claimed',
                  cleanup_attempt_id=%(attempt_id)s, control_revision=%(control_revision)s,
                  claimed_at=now(), attempts=attempts+1, last_error=NULL
                FROM due WHERE r.id=due.id
                RETURNING r.*
                """,
                {
                    "machine": machine,
                    "limit": limit,
                    "attempt_id": attempt_id,
                    "control_revision": int(control["control_revision"]),
                    "machine_enabled": machine_enabled,
                    "rollout_revision": int(rollout["control_revision"]),
                },
            )
            claimed = [dict(row) for row in cur.fetchall()]
            if not machine_enabled:
                for row in claimed:
                    cur.execute(
                        """
                        UPDATE workspace_capabilities
                        SET remaining_count=remaining_count-1,
                          consumed_at=CASE WHEN remaining_count=1 THEN now()
                            ELSE consumed_at END
                        WHERE capability_kind='cleanup_canary'
                          AND machine=%(machine)s AND launch_id=%(launch_id)s
                          AND lifecycle_generation=%(generation)s
                          AND control_revision=%(control_revision)s
                          AND remaining_count > 0 AND revoked_at IS NULL
                        RETURNING capability_id
                        """,
                        {
                            "machine": machine,
                            "launch_id": row["launch_id"],
                            "generation": int(row["lifecycle_generation"]),
                            "control_revision": int(rollout["control_revision"]),
                        },
                    )
                    if not cur.fetchone():
                        raise WorkspaceStateConflict(
                            "cleanup canary capability changed while claiming row"
                        )
            return claimed


def record_delete_prepare(
    conn,
    *,
    cleanup_row_id: int,
    cleanup_attempt_id: str,
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    digest = str(receipt.get("prepare_receipt_sha256") or "")
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE workspace_cleanup_rows
                SET deletion_progress='prepared', prepare_receipt_path=%(path)s,
                  prepare_receipt_sha256=%(digest)s, prepare_boot_id=%(boot_id)s,
                  prepare_deadline_monotonic_ns=%(deadline)s,
                  prepare_cleanup_state='present', last_error=NULL
                WHERE id=%(row_id)s AND workspace_state='deleting'
                  AND deletion_progress IN ('claimed','prepare_pending','prepared')
                  AND cleanup_attempt_id=%(cleanup_attempt_id)s
                  AND lifecycle_generation=%(generation)s
                  AND manifest_sha256=%(manifest_sha256)s
                  AND control_revision=%(control_revision)s
                  AND (
                    prepare_receipt_sha256 IS NULL OR prepare_receipt_sha256=%(digest)s
                  )
                RETURNING *
                """,
                {
                    "row_id": int(cleanup_row_id),
                    "cleanup_attempt_id": cleanup_attempt_id,
                    "generation": int(receipt.get("generation") or 0),
                    "manifest_sha256": str(receipt.get("manifest_sha256") or ""),
                    "control_revision": int(receipt.get("control_revision") or 0),
                    "path": str(receipt.get("prepare_receipt_path") or ""),
                    "digest": digest,
                    "boot_id": str(receipt.get("boot_id") or ""),
                    "deadline": int(receipt.get("deadline_monotonic_ns") or 0),
                },
            )
            row = cur.fetchone()
            if not row:
                raise WorkspaceStateConflict("delete prepare receipt no longer matches claim")
            return dict(row)


def authorize_cleanup_batch(
    conn,
    *,
    machine: str,
    cleanup_row_ids: Sequence[int],
    helper_revision: str,
    key_revision: str,
) -> tuple[CleanupBatchEnvelope, str]:
    row_ids = [int(value) for value in cleanup_row_ids]
    if not row_ids or len(row_ids) > 8 or len(set(row_ids)) != len(row_ids):
        raise ValueError("cleanup authorization requires one to eight distinct rows")
    batch_id = f"gcb-{uuid.uuid4().hex}"
    with conn:
        acquire_machine_control_xact_lock(conn, machine=machine)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT c.*, l.machine
                FROM workspace_cleanup_rows c
                JOIN job_launches l ON l.launch_id=c.launch_id
                  AND l.lifecycle_generation=c.lifecycle_generation
                WHERE c.id = ANY(%(row_ids)s)
                ORDER BY c.id
                FOR UPDATE OF c
                """,
                {"row_ids": row_ids},
            )
            rows = [dict(row) for row in cur.fetchall()]
            if len(rows) != len(row_ids) or [row["id"] for row in rows] != sorted(row_ids):
                raise WorkspaceStateConflict("cleanup authorization row set changed")
            if any(row["machine"] != machine for row in rows):
                raise WorkspaceStateConflict("cleanup authorization crosses machines")
            if any(
                row["workspace_state"] != "deleting"
                or row["deletion_progress"] != "prepared"
                or not row.get("prepare_receipt_sha256")
                for row in rows
            ):
                raise WorkspaceStateConflict("cleanup authorization rows are not prepared")
            control_revisions = {int(row["control_revision"]) for row in rows}
            boot_ids = {str(row["prepare_boot_id"]) for row in rows}
            if len(control_revisions) != 1 or len(boot_ids) != 1:
                raise WorkspaceStateConflict("cleanup authorization evidence revisions differ")
            cur.execute(
                """
                SELECT COALESCE(MAX(epoch), 0) + 1 AS epoch
                FROM workspace_cleanup_batches
                WHERE machine=%(machine)s
                """,
                {"machine": machine},
            )
            epoch = int(cur.fetchone()["epoch"])
            envelope_rows = tuple(
                CleanupRowEnvelope(
                    row_id=int(row["id"]),
                    cleanup_attempt_id=str(row["cleanup_attempt_id"]),
                    generation=int(row["lifecycle_generation"]),
                    manifest_sha256=str(row["manifest_sha256"]),
                    prepare_receipt_sha256=str(row["prepare_receipt_sha256"]),
                    starting_cursor_sha256=str(row["cursor_sha256"]),
                )
                for row in rows
            )
            envelope = CleanupBatchEnvelope(
                protocol_version=1,
                batch_id=batch_id,
                machine=machine,
                helper_revision=helper_revision,
                key_revision=key_revision,
                control_revision=next(iter(control_revisions)),
                epoch=epoch,
                boot_id=next(iter(boot_ids)),
                monotonic_deadline_ns=min(
                    int(row["prepare_deadline_monotonic_ns"]) for row in rows
                ),
                rows=envelope_rows,
            )
            envelope_json = envelope.as_dict()
            cur.execute(
                """
                INSERT INTO workspace_cleanup_batches (
                  batch_id, machine, control_revision, helper_revision, key_revision,
                  epoch, boot_id, envelope_sha256, helper_deadline_monotonic_ns,
                  outer_not_after
                ) VALUES (
                  %(batch_id)s, %(machine)s, %(control_revision)s, %(helper_revision)s,
                  %(key_revision)s, %(epoch)s, %(boot_id)s, %(envelope_sha256)s,
                  %(helper_deadline)s, now() + interval '20 seconds'
                )
                """,
                {
                    "batch_id": batch_id,
                    "machine": machine,
                    "control_revision": envelope.control_revision,
                    "helper_revision": helper_revision,
                    "key_revision": key_revision,
                    "epoch": epoch,
                    "boot_id": envelope.boot_id,
                    "envelope_sha256": envelope.digest,
                    "helper_deadline": envelope.monotonic_deadline_ns,
                },
            )
            for ordinal, row in enumerate(envelope.rows):
                cur.execute(
                    """
                    INSERT INTO workspace_cleanup_batch_rows (
                      batch_id, ordinal, cleanup_row_id, cleanup_attempt_id,
                      lifecycle_generation, manifest_sha256, prepare_receipt_sha256,
                      starting_cursor_sha256
                    ) VALUES (
                      %(batch_id)s, %(ordinal)s, %(row_id)s, %(cleanup_attempt_id)s,
                      %(generation)s, %(manifest_sha256)s, %(prepare_receipt_sha256)s,
                      %(starting_cursor_sha256)s
                    )
                    """,
                    {"batch_id": batch_id, "ordinal": ordinal, **row.__dict__},
                )
            cur.execute(
                """
                UPDATE workspace_cleanup_rows
                SET deletion_progress='mutation_pending'
                WHERE id = ANY(%(row_ids)s) AND deletion_progress='prepared'
                RETURNING id
                """,
                {"row_ids": row_ids},
            )
            if len(cur.fetchall()) != len(rows):
                raise WorkspaceStateConflict("cleanup row changed during authorization")
            cur.execute(
                """
                INSERT INTO workspace_authorization_outbox (
                  batch_id, envelope_json, envelope_sha256
                ) VALUES (%(batch_id)s, %(envelope_json)s, %(envelope_sha256)s)
                """,
                {
                    "batch_id": batch_id,
                    "envelope_json": json_arg(envelope_json),
                    "envelope_sha256": envelope.digest,
                },
            )
            evidence = {
                "batch_id": batch_id,
                "row_ids": row_ids,
                "envelope_sha256": envelope.digest,
            }
            cur.execute(
                """
                INSERT INTO workspace_authorization_history (
                  batch_id, transition, actor, control_revision,
                  evidence_sha256, evidence_json
                ) VALUES (
                  %(batch_id)s, 'authorized', 'workspace-gc-controller',
                  %(control_revision)s, %(evidence_sha256)s, %(evidence_json)s
                )
                """,
                {
                    "batch_id": batch_id,
                    "control_revision": envelope.control_revision,
                    "evidence_sha256": sha256_json(evidence),
                    "evidence_json": json_arg(evidence),
                },
            )
            return envelope, batch_id


def record_cleanup_delivery_intent(conn, *, batch_id: str) -> tuple[dict[str, Any], str]:
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE workspace_cleanup_batches
                SET state='delivery_intent', updated_at=now()
                WHERE batch_id=%(batch_id)s AND state='token_issued'
                  AND outer_not_after > now()
                RETURNING signature
                """,
                {"batch_id": batch_id},
            )
            batch = cur.fetchone()
            if not batch or not batch.get("signature"):
                raise WorkspaceStateConflict("cleanup batch has no deliverable signed token")
            cur.execute(
                "SELECT envelope_json FROM workspace_authorization_outbox "
                "WHERE batch_id=%(batch_id)s AND state='signed'",
                {"batch_id": batch_id},
            )
            outbox = cur.fetchone()
            if not outbox:
                raise WorkspaceStateConflict("signed cleanup envelope is absent")
            return dict(outbox["envelope_json"]), str(batch["signature"])


def due_signed_cleanup_batches(
    conn, *, machine: str, limit: int = 1
) -> list[dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT b.batch_id, o.envelope_json
            FROM workspace_cleanup_batches b
            JOIN workspace_authorization_outbox o ON o.batch_id=b.batch_id
            WHERE b.machine=%(machine)s AND b.state='token_issued'
              AND o.state='signed' AND b.outer_not_after > now()
            ORDER BY b.issued_at, b.id
            LIMIT %(limit)s
            """,
            {"machine": machine, "limit": max(1, int(limit))},
        )
        return [dict(row) for row in cur.fetchall()]


def manifests_for_cleanup_batch(
    conn, *, batch_id: str
) -> dict[int, WorkspaceManifest]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT br.cleanup_row_id, m.manifest_json
            FROM workspace_cleanup_batch_rows br
            JOIN workspace_cleanup_rows r ON r.id=br.cleanup_row_id
            JOIN workspace_manifests m ON m.launch_id=r.launch_id
              AND m.generation=r.lifecycle_generation
            WHERE br.batch_id=%(batch_id)s
            ORDER BY br.ordinal
            """,
            {"batch_id": batch_id},
        )
        rows = cur.fetchall()
    return {
        int(row["cleanup_row_id"]): workspace_manifest_from_mapping(row["manifest_json"])
        for row in rows
    }


def reconcile_cleanup_result(
    conn,
    *,
    batch_id: str,
    result: Mapping[str, Any] | None,
    ambiguous: bool = False,
) -> None:
    with conn:
        with conn.cursor() as cur:
            if ambiguous:
                cur.execute(
                    """
                    UPDATE workspace_cleanup_batches
                    SET state='delivery_ambiguous', updated_at=now(),
                      last_error='helper delivery or exit was ambiguous'
                    WHERE batch_id=%(batch_id)s
                      AND state IN ('delivery_intent','delivery_ack','consumed')
                    RETURNING machine
                    """,
                    {"batch_id": batch_id},
                )
                batch = cur.fetchone()
                if not batch:
                    raise WorkspaceStateConflict("cleanup batch cannot become ambiguous")
                cur.execute(
                    """
                    UPDATE machine_controls
                    SET host_mutation_quarantined=TRUE, control_revision=control_revision+1,
                      reason='ambiguous workspace cleanup helper exit', updated_at=now()
                    WHERE machine=%(machine)s
                    """,
                    {"machine": batch["machine"]},
                )
                return
            document = dict(result or {})
            completed = document.get("completed") or []
            for journal in completed:
                cur.execute(
                    """
                    UPDATE workspace_cleanup_rows
                    SET workspace_state='host_deleted', deletion_progress='mutation_complete',
                      host_deleted_at=now(), host_deleted_journal_sha256=%(journal_sha)s,
                      journal_cleanup_state='present', last_error=NULL
                    WHERE id=%(row_id)s AND workspace_state='deleting'
                      AND cleanup_attempt_id=%(cleanup_attempt_id)s
                      AND lifecycle_generation=%(generation)s
                    RETURNING id
                    """,
                    {
                        "row_id": int(journal["cleanup_row_id"]),
                        "cleanup_attempt_id": str(journal["cleanup_attempt_id"]),
                        "generation": int(journal["generation"]),
                        "journal_sha": str(journal["host_deleted_journal_sha256"]),
                    },
                )
                if not cur.fetchone():
                    raise WorkspaceStateConflict("host-deleted journal does not match cleanup row")
            partial_ids = [int(value) for value in document.get("partial_row_ids") or []]
            if partial_ids:
                cur.execute(
                    """
                    UPDATE workspace_cleanup_rows
                    SET deletion_progress='partial', last_error='cleanup quantum expired'
                    WHERE id = ANY(%(row_ids)s) AND workspace_state='deleting'
                    """,
                    {"row_ids": partial_ids},
                )
            cur.execute(
                """
                UPDATE workspace_cleanup_batches
                SET state='reconciled', updated_at=now(), last_error=NULL
                WHERE batch_id=%(batch_id)s
                  AND state IN ('delivery_intent','delivery_ack','consumed','exited')
                RETURNING batch_id
                """,
                {"batch_id": batch_id},
            )
            if not cur.fetchone():
                raise WorkspaceStateConflict("cleanup batch result cannot be reconciled")


def finalize_host_deleted(conn, *, limit: int = 100) -> int:
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                WITH due AS (
                  SELECT id FROM workspace_cleanup_rows
                  WHERE workspace_state='host_deleted'
                    AND deletion_progress='mutation_complete'
                    AND host_deleted_journal_sha256 IS NOT NULL
                  ORDER BY host_deleted_at, id
                  FOR UPDATE SKIP LOCKED
                  LIMIT %(limit)s
                )
                UPDATE workspace_cleanup_rows r
                SET workspace_state='completed', completed_at=COALESCE(completed_at, now())
                FROM due WHERE r.id=due.id
                RETURNING r.id
                """,
                {"limit": max(1, int(limit))},
            )
            return len(cur.fetchall())


def completed_journal_cleanup_due(
    conn, *, machine: str, limit: int = 8
) -> list[dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT r.*, m.manifest_json
            FROM workspace_cleanup_rows r
            JOIN job_launches l ON l.launch_id=r.launch_id
            JOIN workspace_manifests m ON m.launch_id=r.launch_id
              AND m.generation=r.lifecycle_generation
            WHERE l.machine=%(machine)s AND r.workspace_state='completed'
              AND r.journal_cleanup_state IN ('present','failed')
            ORDER BY r.completed_at, r.id
            LIMIT %(limit)s
            """,
            {"machine": machine, "limit": max(1, int(limit))},
        )
        return [dict(row) for row in cur.fetchall()]


def mark_journal_cleaned(
    conn, *, cleanup_row_id: int, host_deleted_journal_sha256: str
) -> None:
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE workspace_cleanup_rows
                SET journal_cleanup_state='cleaned', journal_cleanup_error=NULL
                WHERE id=%(row_id)s AND workspace_state='completed'
                  AND host_deleted_journal_sha256=%(journal_sha)s
                  AND journal_cleanup_state IN ('present','cleaning','failed','cleaned')
                RETURNING id
                """,
                {"row_id": int(cleanup_row_id), "journal_sha": host_deleted_journal_sha256},
            )
            if not cur.fetchone():
                raise WorkspaceStateConflict("journal cleanup evidence changed")


def mark_prepare_receipt_cleaned(
    conn, *, cleanup_row_id: int, prepare_receipt_sha256: str
) -> None:
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE workspace_cleanup_rows
                SET prepare_cleanup_state='cleaned', last_error=NULL
                WHERE id=%(row_id)s AND prepare_receipt_sha256=%(prepare_sha)s
                  AND prepare_cleanup_state IN ('present','cleaning','failed','cleaned')
                RETURNING id
                """,
                {"row_id": int(cleanup_row_id), "prepare_sha": prepare_receipt_sha256},
            )
            if not cur.fetchone():
                raise WorkspaceStateConflict("prepare receipt cleanup evidence changed")


def due_proof_reduction_launches(conn, *, limit: int = 25) -> list[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO workspace_proof_reducer_states (
              launch_id, lifecycle_generation, state, next_due_at
            )
            SELECT l.launch_id, l.lifecycle_generation, 'pending', now()
            FROM job_launches l
            JOIN train_jobs t ON t.id=l.job_id
            WHERE l.workspace_layout_version=1
              AND l.state='succeeded' AND t.status='succeeded'
              AND NOT EXISTS (
                SELECT 1 FROM workspace_cleanup_proofs p
                WHERE p.launch_id=l.launch_id
                  AND p.lifecycle_generation=l.lifecycle_generation
              )
            ON CONFLICT (launch_id, lifecycle_generation) DO NOTHING
            """
        )
        cur.execute(
            """
            SELECT l.launch_id
            FROM job_launches l
            JOIN train_jobs t ON t.id=l.job_id
            JOIN workspace_proof_reducer_states reducer
              ON reducer.launch_id=l.launch_id
             AND reducer.lifecycle_generation=l.lifecycle_generation
            WHERE l.workspace_layout_version=1
              AND l.state='succeeded' AND t.status='succeeded'
              AND reducer.state IN ('pending','blocked')
              AND reducer.next_due_at <= now()
              AND NOT EXISTS (
                SELECT 1 FROM workspace_cleanup_proofs p
                WHERE p.launch_id=l.launch_id
                  AND p.lifecycle_generation=l.lifecycle_generation
              )
            ORDER BY l.finished_at, l.id
            LIMIT %(limit)s
            """,
            {"limit": max(1, int(limit))},
        )
        return [str(row["launch_id"]) for row in cur.fetchall()]


def record_proof_reducer_result(
    conn, *, launch_id: str, ready: bool, error: str | None = None
) -> None:
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE workspace_proof_reducer_states reducer
                SET state=CASE WHEN %(ready)s THEN 'ready' ELSE 'blocked' END,
                  attempts=attempts+1, last_attempt_at=now(),
                  ready_at=CASE WHEN %(ready)s THEN now() ELSE ready_at END,
                  next_due_at=CASE WHEN %(ready)s THEN now()
                    ELSE now() + interval '5 seconds' END,
                  last_error=CASE WHEN %(ready)s THEN NULL ELSE %(error)s END
                FROM job_launches launch
                WHERE reducer.launch_id=%(launch_id)s
                  AND launch.launch_id=reducer.launch_id
                  AND reducer.lifecycle_generation=launch.lifecycle_generation
                """,
                {
                    "launch_id": launch_id,
                    "ready": bool(ready),
                    "error": str(error or "")[:4000],
                },
            )


def close_due_telemetry_obligations(conn, *, limit: int = 100) -> int:
    closed = 0
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT o.*, t.live_publication_status, l.state AS launch_state,
                  s.final_sequence AS stream_final_sequence,
                  s.published_sequence AS stream_published_sequence,
                  EXISTS (
                    SELECT 1 FROM metric_batches b
                    WHERE b.stream_id=o.expected_stream_id
                  ) AS retained_batches
                FROM telemetry_obligations o
                JOIN train_jobs t ON t.id=o.train_job_id
                JOIN job_launches l ON l.launch_id=o.launch_id
                  AND l.lifecycle_generation=o.lifecycle_generation
                LEFT JOIN metric_streams s ON s.stream_id=o.expected_stream_id
                WHERE o.disposition IN ('pending','failed')
                  AND l.state IN ('succeeded','failed','canceled')
                ORDER BY o.created_at
                FOR UPDATE OF o SKIP LOCKED
                LIMIT %(limit)s
                """,
                {"limit": max(1, int(limit))},
            )
            rows = [dict(row) for row in cur.fetchall()]
            for row in rows:
                disposition = None
                final_sequence = row.get("stream_final_sequence")
                if row["sink"] == "disabled":
                    disposition = "disabled"
                    final_sequence = 0
                elif row["live_publication_status"] == "complete":
                    if row.get("expected_stream_id") is None:
                        disposition = "published"
                        final_sequence = 0
                    elif (
                        final_sequence is not None
                        and int(row.get("stream_published_sequence") or 0)
                        >= int(final_sequence)
                        and not row["retained_batches"]
                    ):
                        disposition = "zero_batch" if int(final_sequence) == 0 else "published"
                if disposition is None:
                    continue
                receipt = {
                    "protocol_version": 1,
                    "obligation_id": int(row["id"]),
                    "producer_identity": row["producer_identity"],
                    "sink": row["sink"],
                    "expected_stream_id": row.get("expected_stream_id"),
                    "final_sequence": int(final_sequence or 0),
                    "live_publication_status": row["live_publication_status"],
                }
                cur.execute(
                    """
                    UPDATE telemetry_obligations
                    SET disposition=%(disposition)s, final_sequence=%(final_sequence)s,
                      receipt_json=%(receipt)s, closed_at=now()
                    WHERE id=%(id)s AND disposition IN ('pending','failed')
                    """,
                    {
                        "id": int(row["id"]),
                        "disposition": disposition,
                        "final_sequence": int(final_sequence or 0),
                        "receipt": json_arg(receipt),
                    },
                )
                closed += int(cur.rowcount == 1)
    return closed


def workspace_gc_status(conn, *, machine: str | None = None) -> dict[str, Any]:
    params: dict[str, Any] = {}
    machine_filter = ""
    if machine:
        machine_filter = "WHERE l.machine=%(machine)s"
        params["machine"] = machine
    with conn.cursor() as cur:
        cur.execute(
            "SELECT * FROM workspace_rollout_controls WHERE singleton=TRUE"
        )
        rollout = dict(cur.fetchone() or {})
        cur.execute(
            f"""
            SELECT l.machine, r.workspace_state, r.deletion_progress,
              COUNT(*) AS count, MIN(COALESCE(r.claimed_at, p.cleanup_ready_at)) AS oldest_at
            FROM workspace_cleanup_rows r
            JOIN job_launches l ON l.launch_id=r.launch_id
              AND l.lifecycle_generation=r.lifecycle_generation
            JOIN workspace_cleanup_proofs p ON p.launch_id=r.launch_id
              AND p.lifecycle_generation=r.lifecycle_generation
            {machine_filter}
            GROUP BY l.machine, r.workspace_state, r.deletion_progress
            ORDER BY l.machine, r.workspace_state, r.deletion_progress
            """,
            params,
        )
        rows = [dict(row) for row in cur.fetchall()]
        cur.execute(
            f"""
            SELECT l.machine,
              COUNT(*) FILTER (WHERE g.env_cleanup_state <> 'unlinked') AS env_backlog,
              COUNT(*) FILTER (WHERE m.state <> 'absent') AS container_backlog
            FROM job_launches l
            LEFT JOIN workspace_container_generations g ON g.launch_id=l.launch_id
              AND g.workspace_generation=l.lifecycle_generation
            LEFT JOIN workspace_container_members m ON m.launch_id=g.launch_id
              AND m.workspace_generation=g.workspace_generation
              AND m.container_generation=g.container_generation
            {machine_filter}
            GROUP BY l.machine ORDER BY l.machine
            """,
            params,
        )
        credential = [dict(row) for row in cur.fetchall()]
        cur.execute(
            f"""
            SELECT l.machine,
              COUNT(*) FILTER (WHERE l.state='succeeded' AND p.launch_id IS NULL)
                AS proof_backlog,
              MIN(l.finished_at) FILTER (WHERE l.state='succeeded' AND p.launch_id IS NULL)
                AS oldest_proof_input,
              COUNT(*) FILTER (WHERE r.workspace_state='rollback_review') AS review_backlog,
              COUNT(*) FILTER (WHERE lease.state IN ('registered','running','reconciling'))
                AS operation_lease_backlog
            FROM job_launches l
            LEFT JOIN workspace_cleanup_proofs p ON p.launch_id=l.launch_id
              AND p.lifecycle_generation=l.lifecycle_generation
            LEFT JOIN workspace_cleanup_rows r ON r.launch_id=l.launch_id
              AND r.lifecycle_generation=l.lifecycle_generation
            LEFT JOIN host_operation_leases lease ON lease.launch_id=l.launch_id
              AND lease.lifecycle_generation=l.lifecycle_generation
            {machine_filter}
            GROUP BY l.machine ORDER BY l.machine
            """,
            params,
        )
        lifecycle = [dict(row) for row in cur.fetchall()]
        control_filter = "WHERE machine=%(machine)s" if machine else ""
        cur.execute(
            f"""
            SELECT machine, drained, drain_requested, drain_state,
              normal_admission_disabled, host_mutation_quarantined,
              cleanup_enabled, control_revision, rollout_owner,
              cleanup_evidence_sha256, reason, updated_at
            FROM machine_controls {control_filter}
            ORDER BY machine
            """,
            params,
        )
        controls = [dict(row) for row in cur.fetchall()]
        cur.execute(
            """
            SELECT COUNT(*) AS missing_objects, MIN(ledger.verified_at) AS oldest_missing
            FROM artifact_announcement_ledger ledger
            WHERE ledger.disposition='ready' AND (
              NOT EXISTS (
                SELECT 1 FROM artifact_durability_receipts receipt
                WHERE receipt.train_job_id=ledger.train_job_id
                  AND receipt.ledger_id=ledger.ledger_id
                  AND receipt.object_kind='model'
              ) OR NOT EXISTS (
                SELECT 1 FROM artifact_durability_receipts receipt
                WHERE receipt.train_job_id=ledger.train_job_id
                  AND receipt.ledger_id=ledger.ledger_id
                  AND receipt.object_kind='metadata'
              ) OR (ledger.recipe_uri IS NOT NULL AND NOT EXISTS (
                SELECT 1 FROM artifact_durability_receipts receipt
                WHERE receipt.train_job_id=ledger.train_job_id
                  AND receipt.ledger_id=ledger.ledger_id
                  AND receipt.object_kind='recipe'
              ))
            )
            """
        )
        durability = dict(cur.fetchone() or {})
        cur.execute(
            """
            SELECT state, COUNT(*) AS count, MIN(created_at) AS oldest_at,
              MAX(last_error) FILTER (WHERE last_error IS NOT NULL) AS sample_error
            FROM workspace_authorization_outbox
            GROUP BY state ORDER BY state
            """
        )
        signer = [dict(row) for row in cur.fetchall()]
        cur.execute(
            """
            SELECT state, COUNT(*) AS count, MIN(next_due_at) AS oldest_due_at,
              MAX(last_error) FILTER (WHERE last_error IS NOT NULL) AS sample_error
            FROM workspace_proof_reducer_states
            GROUP BY state ORDER BY state
            """
        )
        reducers = [dict(row) for row in cur.fetchall()]
    return {
        "rollout": rollout,
        "machine_controls": controls,
        "cleanup": rows,
        "credential_cleanup": credential,
        "lifecycle": lifecycle,
        "artifact_durability": durability,
        "signer_outbox": signer,
        "proof_reducer": reducers,
    }


def workspace_protocol_mode(conn) -> str:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT protocol_mode FROM workspace_rollout_controls WHERE singleton=TRUE"
        )
        row = cur.fetchone()
    return str((row or {}).get("protocol_mode") or "dormant")


def set_workspace_rollout_control(
    conn,
    *,
    expected_revision: int,
    rollout_owner: str,
    protocol_mode: str,
    work_creation_paused: bool,
    cleanup_globally_enabled: bool,
    reason: str,
) -> dict[str, Any]:
    if not rollout_owner:
        raise ValueError("rollout owner is required")
    with conn:
        acquire_work_creation_xact_lock(conn, exclusive=True)
        acquire_fleet_admission_xact_lock(conn, exclusive=True)
        with conn.cursor() as cur:
            if protocol_mode == "active" and not work_creation_paused:
                cur.execute(
                    """
                    SELECT
                      (SELECT COUNT(*) FROM machine_controls control
                       WHERE NOT control.drained
                         AND NOT control.normal_admission_disabled) AS admitted_machines,
                      (SELECT COUNT(*) FROM machine_controls control
                       WHERE NOT control.drained
                         AND NOT control.normal_admission_disabled
                         AND (
                           NOT control.cleanup_enabled
                           OR control.host_mutation_quarantined
                           OR NOT EXISTS (
                             SELECT 1 FROM workspace_qualification_receipts receipt
                             JOIN workspace_qualification_schedules schedule
                               ON schedule.schedule_id=receipt.schedule_id
                             WHERE receipt.machine=control.machine
                               AND receipt.machine_control_revision=control.control_revision-1
                               AND receipt.receipt_sha256=control.cleanup_evidence_sha256
                               AND schedule.state='passed'
                           )
                         )) AS unqualified_machines,
                      (SELECT COUNT(*) FROM workspace_cleanup_rows
                       WHERE workspace_state='rollback_review') AS review_rows,
                      (SELECT COUNT(*) FROM host_operation_leases
                       WHERE state IN ('registered','running','reconciling')) AS active_leases,
                      (SELECT COUNT(*) FROM workspace_signer_leases
                       WHERE state='healthy' AND lease_expires_at > now()) AS healthy_signers
                      ,(SELECT COUNT(*) FROM workspace_promotion_receipts
                        WHERE rollout_control_revision=%(expected_revision)s)
                        AS promotion_receipts
                    """
                )
                gate = dict(cur.fetchone())
                if (
                    int(gate["admitted_machines"]) < 1
                    or int(gate["unqualified_machines"])
                    or int(gate["review_rows"])
                    or int(gate["active_leases"])
                    or int(gate["healthy_signers"]) < 1
                    or int(gate["promotion_receipts"]) < 1
                ):
                    raise WorkspaceStateConflict(
                        "promotion gates are not satisfied: " + str(gate)
                    )
            cur.execute(
                """
                UPDATE workspace_rollout_controls
                SET protocol_mode=%(protocol_mode)s,
                  work_creation_paused=%(work_creation_paused)s,
                  cleanup_globally_enabled=%(cleanup_globally_enabled)s,
                  control_revision=control_revision+1,
                  rollout_owner=%(rollout_owner)s, reason=%(reason)s, updated_at=now()
                WHERE singleton=TRUE AND control_revision=%(expected_revision)s
                  AND (rollout_owner IS NULL OR rollout_owner=%(rollout_owner)s)
                RETURNING *
                """,
                {
                    "expected_revision": int(expected_revision),
                    "rollout_owner": rollout_owner,
                    "protocol_mode": protocol_mode,
                    "work_creation_paused": bool(work_creation_paused),
                    "cleanup_globally_enabled": bool(cleanup_globally_enabled),
                    "reason": reason,
                },
            )
            row = cur.fetchone()
            if not row:
                raise WorkspaceStateConflict("rollout control revision or owner changed")
            return dict(row)


def record_workspace_promotion_receipt(
    conn,
    *,
    launch_id: str,
    rollout_control_revision: int,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    if not bool(evidence.get("quiescence_gate_passed")):
        raise ValueError("promotion verification requires a passed quiescence gate")
    digest = sha256_json(dict(evidence))
    with conn:
        acquire_work_creation_xact_lock(conn, exclusive=True)
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO workspace_promotion_receipts (
                  launch_id, lifecycle_generation, machine, rollout_control_revision,
                  ordinary_claim_without_cleanup_capability, cleanup_completed,
                  journal_cleanup_completed, quiescence_gate_passed,
                  evidence_json, evidence_sha256
                )
                SELECT row.launch_id, row.lifecycle_generation, manifest.machine,
                  %(rollout_control_revision)s, TRUE, TRUE, TRUE, TRUE,
                  %(evidence_json)s, %(evidence_sha256)s
                FROM workspace_cleanup_rows row
                JOIN workspace_manifests manifest ON manifest.launch_id=row.launch_id
                  AND manifest.generation=row.lifecycle_generation
                JOIN workspace_rollout_controls control ON control.singleton
                WHERE row.launch_id=%(launch_id)s
                  AND row.workspace_state='completed'
                  AND row.prepare_cleanup_state='cleaned'
                  AND row.journal_cleanup_state='cleaned'
                  AND control.protocol_mode='promotion_verifying'
                  AND control.work_creation_paused
                  AND control.control_revision=%(rollout_control_revision)s
                  AND NOT EXISTS (
                    SELECT 1 FROM workspace_capabilities capability
                    WHERE capability.launch_id=row.launch_id
                      AND capability.capability_kind='cleanup_canary'
                  )
                ON CONFLICT (launch_id) DO UPDATE
                SET evidence_sha256=workspace_promotion_receipts.evidence_sha256
                WHERE workspace_promotion_receipts.evidence_sha256=EXCLUDED.evidence_sha256
                RETURNING *
                """,
                {
                    "launch_id": launch_id,
                    "rollout_control_revision": int(rollout_control_revision),
                    "evidence_json": json_arg(dict(evidence)),
                    "evidence_sha256": digest,
                },
            )
            row = cur.fetchone()
            if not row:
                raise WorkspaceStateConflict(
                    "ordinary promotion launch has not completed cleanup and quiescence"
                )
            return dict(row)


def set_machine_cleanup_control(
    conn,
    *,
    machine: str,
    expected_revision: int,
    enabled: bool,
    evidence_sha256: str | None,
    rollout_owner: str,
    reason: str,
) -> dict[str, Any]:
    if enabled and (
        not evidence_sha256
        or len(evidence_sha256) != 64
        or any(char not in "0123456789abcdef" for char in evidence_sha256)
    ):
        raise ValueError("enabling machine cleanup requires a SHA-256 qualification receipt")
    with conn:
        acquire_work_creation_xact_lock(conn, exclusive=True)
        acquire_fleet_admission_xact_lock(conn, exclusive=True)
        acquire_machine_control_xact_lock(conn, machine=machine)
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE machine_controls
                SET cleanup_enabled=%(enabled)s,
                  cleanup_evidence_sha256=CASE WHEN %(enabled)s THEN %(evidence)s ELSE NULL END,
                  rollout_owner=%(rollout_owner)s, reason=%(reason)s,
                  control_revision=control_revision+1, updated_at=now()
                WHERE machine=%(machine)s AND control_revision=%(expected_revision)s
                  AND NOT host_mutation_quarantined
                  AND (
                    NOT %(enabled)s OR EXISTS (
                      SELECT 1 FROM workspace_qualification_receipts receipt
                      JOIN workspace_qualification_schedules schedule
                        ON schedule.schedule_id=receipt.schedule_id
                      WHERE receipt.machine=machine_controls.machine
                        AND receipt.machine_control_revision=machine_controls.control_revision
                        AND receipt.receipt_sha256=%(evidence)s
                        AND schedule.state='passed'
                    )
                  )
                RETURNING *
                """,
                {
                    "machine": machine,
                    "expected_revision": int(expected_revision),
                    "enabled": bool(enabled),
                    "evidence": evidence_sha256,
                    "rollout_owner": rollout_owner,
                    "reason": reason,
                },
            )
            row = cur.fetchone()
            if not row:
                raise WorkspaceStateConflict("machine control revision changed or host quarantined")
            return dict(row)


def record_workspace_qualification_receipt(
    conn,
    *,
    schedule_id: str,
    machine: str,
    machine_control_revision: int,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    required = {
        "source_sha",
        "runtime_image_digest",
        "backend",
        "effective_capacity",
        "paired_blocks",
        "throughput_point_regression",
        "throughput_upper95_regression",
        "loop_seconds_p99_regression",
        "loop_seconds_max_regression",
        "machine_pass_p95_regression",
        "mixed_backlog_rows",
        "service_rate_gate_passed",
        "quiescence_gate_passed",
        "boundary_evidence_complete",
    }
    missing = required - set(evidence)
    if missing:
        raise ValueError(
            "qualification evidence is incomplete: " + ", ".join(sorted(missing))
        )
    checks = (
        int(evidence["paired_blocks"]) >= 5,
        float(evidence["throughput_point_regression"]) <= 0.05,
        float(evidence["throughput_upper95_regression"]) <= 0.05,
        float(evidence["loop_seconds_p99_regression"]) <= 0.05,
        float(evidence["loop_seconds_max_regression"]) <= 0.10,
        float(evidence["machine_pass_p95_regression"]) <= 0.05,
        int(evidence["mixed_backlog_rows"]) > 8,
        bool(evidence["service_rate_gate_passed"]),
        bool(evidence["quiescence_gate_passed"]),
        bool(evidence["boundary_evidence_complete"]),
    )
    if not all(checks):
        raise ValueError("qualification evidence does not satisfy the performance/safety gates")
    digest = sha256_json(dict(evidence))
    receipt_id = f"wqr-{digest[:24]}"
    with conn:
        acquire_work_creation_xact_lock(conn, exclusive=True)
        acquire_machine_control_xact_lock(conn, machine=machine)
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO workspace_qualification_receipts (
                  receipt_id, schedule_id, machine, machine_control_revision,
                  source_sha, runtime_image_digest, backend, effective_capacity,
                  paired_blocks, throughput_point_regression,
                  throughput_upper95_regression, loop_seconds_p99_regression,
                  loop_seconds_max_regression, machine_pass_p95_regression,
                  mixed_backlog_rows, service_rate_gate_passed,
                  quiescence_gate_passed, boundary_evidence_complete,
                  receipt_json, receipt_sha256
                )
                SELECT %(receipt_id)s, %(schedule_id)s, %(machine)s,
                  %(machine_control_revision)s, %(source_sha)s,
                  %(runtime_image_digest)s, %(backend)s, %(effective_capacity)s,
                  %(paired_blocks)s, %(throughput_point_regression)s,
                  %(throughput_upper95_regression)s, %(loop_seconds_p99_regression)s,
                  %(loop_seconds_max_regression)s, %(machine_pass_p95_regression)s,
                  %(mixed_backlog_rows)s, %(service_rate_gate_passed)s,
                  %(quiescence_gate_passed)s, %(boundary_evidence_complete)s,
                  %(receipt_json)s, %(receipt_sha256)s
                FROM machine_controls control
                JOIN workspace_qualification_schedules schedule
                  ON schedule.schedule_id=%(schedule_id)s
                WHERE control.machine=%(machine)s
                  AND control.control_revision=%(machine_control_revision)s
                  AND schedule.state IN ('running','passed')
                ON CONFLICT (receipt_id) DO UPDATE
                SET receipt_sha256=workspace_qualification_receipts.receipt_sha256
                WHERE workspace_qualification_receipts.receipt_sha256=EXCLUDED.receipt_sha256
                RETURNING *
                """,
                {
                    "receipt_id": receipt_id,
                    "schedule_id": schedule_id,
                    "machine": machine,
                    "machine_control_revision": int(machine_control_revision),
                    "source_sha": str(evidence["source_sha"]),
                    "runtime_image_digest": str(evidence["runtime_image_digest"]),
                    "backend": str(evidence["backend"]),
                    "effective_capacity": int(evidence["effective_capacity"]),
                    "paired_blocks": int(evidence["paired_blocks"]),
                    "throughput_point_regression": float(
                        evidence["throughput_point_regression"]
                    ),
                    "throughput_upper95_regression": float(
                        evidence["throughput_upper95_regression"]
                    ),
                    "loop_seconds_p99_regression": float(
                        evidence["loop_seconds_p99_regression"]
                    ),
                    "loop_seconds_max_regression": float(
                        evidence["loop_seconds_max_regression"]
                    ),
                    "machine_pass_p95_regression": float(
                        evidence["machine_pass_p95_regression"]
                    ),
                    "mixed_backlog_rows": int(evidence["mixed_backlog_rows"]),
                    "service_rate_gate_passed": bool(
                        evidence["service_rate_gate_passed"]
                    ),
                    "quiescence_gate_passed": bool(evidence["quiescence_gate_passed"]),
                    "boundary_evidence_complete": bool(
                        evidence["boundary_evidence_complete"]
                    ),
                    "receipt_json": json_arg(dict(evidence)),
                    "receipt_sha256": digest,
                },
            )
            row = cur.fetchone()
            if not row:
                raise WorkspaceStateConflict(
                    "qualification schedule or machine revision is not passed/current"
                )
            return dict(row)


def create_workspace_qualification_schedule(
    conn,
    *,
    rollout_owner: str,
    control_revision: int,
    schedule: Mapping[str, Any],
) -> dict[str, Any]:
    machines_value = schedule.get("machines")
    if not isinstance(machines_value, list) or not machines_value:
        raise ValueError("qualification schedule requires a non-empty machines list")
    machines = [str(value) for value in machines_value]
    if len(set(machines)) != len(machines) or any(not value for value in machines):
        raise ValueError("qualification schedule machine identities must be unique")
    cap = int(schedule.get("global_concurrency_cap") or 0)
    if cap < 1:
        raise ValueError("qualification schedule requires a positive global concurrency cap")
    budgets = schedule.get("machine_budgets")
    if not isinstance(budgets, Mapping) or set(map(str, budgets)) != set(machines):
        raise ValueError("qualification schedule requires one budget for every machine")
    if any(int(budgets[machine]) < 1 for machine in machines):
        raise ValueError("qualification machine budgets must be positive")
    digest = sha256_json(dict(schedule))
    schedule_id = f"wqs-{digest[:24]}"
    with conn:
        acquire_work_creation_xact_lock(conn, exclusive=True)
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO workspace_qualification_schedules (
                  schedule_id, rollout_owner, control_revision, state,
                  global_concurrency_cap, schedule_json, schedule_sha256
                )
                SELECT %(schedule_id)s, %(rollout_owner)s, %(control_revision)s,
                  'running', %(global_concurrency_cap)s, %(schedule_json)s,
                  %(schedule_sha256)s
                FROM workspace_rollout_controls control
                WHERE control.singleton AND control.control_revision=%(control_revision)s
                  AND control.rollout_owner=%(rollout_owner)s
                  AND control.protocol_mode='qualification'
                  AND control.work_creation_paused
                ON CONFLICT (schedule_id) DO UPDATE
                SET schedule_sha256=workspace_qualification_schedules.schedule_sha256
                WHERE workspace_qualification_schedules.schedule_sha256=EXCLUDED.schedule_sha256
                RETURNING *
                """,
                {
                    "schedule_id": schedule_id,
                    "rollout_owner": rollout_owner,
                    "control_revision": int(control_revision),
                    "global_concurrency_cap": cap,
                    "schedule_json": json_arg(dict(schedule)),
                    "schedule_sha256": digest,
                },
            )
            row = cur.fetchone()
            if not row:
                raise WorkspaceStateConflict(
                    "qualification rollout revision/owner is not current and paused"
                )
            return dict(row)


def authorize_workspace_enqueue(
    conn,
    *,
    schedule_id: str,
    machine: str,
    submission_key: str,
    request_hash: str,
    rollout_owner: str,
    promotion_only: bool = False,
) -> dict[str, Any]:
    capability_id = f"wcap-enqueue-{uuid.uuid4().hex}"
    kind = "promotion_enqueue_only" if promotion_only else "qualification_enqueue"
    with conn:
        acquire_work_creation_xact_lock(conn, exclusive=True)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT schedule.*, control.protocol_mode,
                  control.control_revision AS current_revision
                FROM workspace_qualification_schedules schedule
                JOIN workspace_rollout_controls control ON control.singleton
                WHERE schedule.schedule_id=%(schedule_id)s
                  AND schedule.rollout_owner=%(rollout_owner)s
                  AND control.work_creation_paused
                FOR UPDATE OF schedule
                """,
                {"schedule_id": schedule_id, "rollout_owner": rollout_owner},
            )
            schedule = cur.fetchone()
            expected_mode = "promotion_verifying" if promotion_only else "qualification"
            expected_state = "passed" if promotion_only else "running"
            if (
                not schedule
                or schedule["protocol_mode"] != expected_mode
                or schedule["state"] != expected_state
                or (
                    not promotion_only
                    and int(schedule["control_revision"])
                    != int(schedule["current_revision"])
                )
            ):
                raise WorkspaceStateConflict("enqueue schedule is not active in the expected mode")
            budgets = dict(schedule["schedule_json"].get("machine_budgets") or {})
            if machine not in budgets:
                raise WorkspaceStateConflict("machine is outside the qualification schedule")
            cur.execute(
                """
                SELECT COUNT(*) AS used FROM workspace_capabilities
                WHERE schedule_id=%(schedule_id)s AND machine=%(machine)s
                  AND capability_kind=%(kind)s
                """,
                {"schedule_id": schedule_id, "machine": machine, "kind": kind},
            )
            used = int(cur.fetchone()["used"])
            if (promotion_only and used >= 1) or (
                not promotion_only and used >= int(budgets[machine])
            ):
                raise WorkspaceStateConflict("machine qualification submission budget is exhausted")
            capability = {
                "submission_key": submission_key,
                "request_hash": request_hash,
                "schedule_sha256": schedule["schedule_sha256"],
            }
            cur.execute(
                """
                INSERT INTO workspace_capabilities (
                  capability_id, capability_kind, schedule_id, machine,
                  control_revision, capability_json, remaining_count, expires_at
                ) VALUES (
                  %(capability_id)s, %(kind)s, %(schedule_id)s, %(machine)s,
                  %(control_revision)s, %(capability_json)s, 1,
                  now() + interval '1 hour'
                )
                RETURNING *
                """,
                {
                    "capability_id": capability_id,
                    "kind": kind,
                    "schedule_id": schedule_id,
                    "machine": machine,
                    "control_revision": int(schedule["current_revision"]),
                    "capability_json": json_arg(capability),
                },
            )
            return dict(cur.fetchone())


def complete_workspace_qualification_schedule(
    conn, *, schedule_id: str, rollout_owner: str
) -> dict[str, Any]:
    with conn:
        acquire_work_creation_xact_lock(conn, exclusive=True)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT * FROM workspace_qualification_schedules
                WHERE schedule_id=%(schedule_id)s AND rollout_owner=%(rollout_owner)s
                  AND state='running'
                FOR UPDATE
                """,
                {"schedule_id": schedule_id, "rollout_owner": rollout_owner},
            )
            schedule = cur.fetchone()
            if not schedule:
                raise WorkspaceStateConflict("qualification schedule is not running")
            machines = [str(value) for value in schedule["schedule_json"].get("machines") or ()]
            cur.execute(
                """
                SELECT DISTINCT machine FROM workspace_qualification_receipts
                WHERE schedule_id=%(schedule_id)s
                """,
                {"schedule_id": schedule_id},
            )
            passed = {str(row["machine"]) for row in cur.fetchall()}
            missing = set(machines) - passed
            if missing:
                raise WorkspaceNotReady(
                    "qualification receipts are missing for: " + ", ".join(sorted(missing))
                )
            cur.execute(
                """
                UPDATE workspace_qualification_schedules
                SET state='passed', completed_at=now()
                WHERE schedule_id=%(schedule_id)s AND state='running'
                RETURNING *
                """,
                {"schedule_id": schedule_id},
            )
            return dict(cur.fetchone())


def authorize_workspace_cleanup_canary(
    conn,
    *,
    schedule_id: str,
    cleanup_row_id: int,
    rollout_owner: str,
) -> dict[str, Any]:
    capability_id = f"wcap-cleanup-{uuid.uuid4().hex}"
    with conn:
        acquire_work_creation_xact_lock(conn, exclusive=True)
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO workspace_capabilities (
                  capability_id, capability_kind, schedule_id, machine,
                  launch_id, lifecycle_generation, control_revision,
                  capability_json, remaining_count
                )
                SELECT %(capability_id)s, 'cleanup_canary', schedule.schedule_id,
                  manifest.machine, row.launch_id, row.lifecycle_generation,
                  control.control_revision,
                  jsonb_build_object(
                    'cleanup_row_id', row.id,
                    'manifest_sha256', row.manifest_sha256,
                    'schedule_sha256', schedule.schedule_sha256
                  ), 1
                FROM workspace_cleanup_rows row
                JOIN workspace_manifests manifest ON manifest.launch_id=row.launch_id
                  AND manifest.generation=row.lifecycle_generation
                JOIN workspace_qualification_schedules schedule
                  ON schedule.schedule_id=%(schedule_id)s
                JOIN workspace_rollout_controls control ON control.singleton
                WHERE row.id=%(cleanup_row_id)s AND row.workspace_state='pending'
                  AND schedule.state='running'
                  AND schedule.rollout_owner=%(rollout_owner)s
                  AND control.protocol_mode='qualification'
                  AND control.control_revision=schedule.control_revision
                  AND control.work_creation_paused
                  AND schedule.schedule_json->'machines' ? manifest.machine
                RETURNING *
                """,
                {
                    "capability_id": capability_id,
                    "schedule_id": schedule_id,
                    "cleanup_row_id": int(cleanup_row_id),
                    "rollout_owner": rollout_owner,
                },
            )
            row = cur.fetchone()
            if not row:
                raise WorkspaceStateConflict(
                    "cleanup canary row is not pending in the active qualification schedule"
                )
            return dict(row)


def resolve_rollback_review(
    conn,
    *,
    cleanup_row_id: int,
    expected_control_revision: int,
    evidence: Mapping[str, Any],
    resolution: str,
) -> dict[str, Any]:
    if resolution not in {"defer", "resume"}:
        raise ValueError("review resolution must be defer or resume")
    evidence_sha = sha256_json(dict(evidence))
    with conn:
        acquire_work_creation_xact_lock(conn, exclusive=True)
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE workspace_cleanup_rows
                SET workspace_state=%(workspace_state)s,
                  deletion_progress=%(progress)s,
                  cleanup_attempt_id=NULL,
                  next_retry_at=CASE WHEN %(resolution)s='resume' THEN now() ELSE NULL END,
                  review_reason=%(review_reason)s,
                  last_error=NULL
                WHERE id=%(row_id)s AND workspace_state='rollback_review'
                  AND control_revision=%(control_revision)s
                RETURNING *
                """,
                {
                    "row_id": int(cleanup_row_id),
                    "control_revision": int(expected_control_revision),
                    "resolution": resolution,
                    "workspace_state": "pending",
                    "progress": None,
                    "review_reason": (
                        f"resolved:{resolution}:evidence_sha256={evidence_sha}"
                    ),
                },
            )
            row = cur.fetchone()
            if not row:
                raise WorkspaceStateConflict("rollback review row or control revision changed")
            return dict(row)


def register_host_operation_lease(
    conn,
    *,
    machine: str,
    operation_kind: str,
    resource_scope: str,
    source_revision: str,
    owner: str,
    max_duration: timedelta,
    launch_id: str | None = None,
    lifecycle_generation: int | None = None,
) -> str:
    operation_id = f"hop-{uuid.uuid4().hex}"
    with conn:
        acquire_machine_control_xact_lock(conn, machine=machine)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT control_revision, host_mutation_quarantined, drained,
                  CASE WHEN %(launch_id)s IS NULL THEN NULL ELSE (
                    SELECT row.workspace_state FROM workspace_cleanup_rows row
                    WHERE row.launch_id=%(launch_id)s
                      AND row.lifecycle_generation=%(lifecycle_generation)s
                  ) END AS workspace_state
                FROM machine_controls WHERE machine=%(machine)s FOR UPDATE
                """,
                {
                    "machine": machine,
                    "launch_id": launch_id,
                    "lifecycle_generation": lifecycle_generation,
                },
            )
            control = cur.fetchone()
            if not control:
                raise WorkspaceStateConflict(f"machine control does not exist: {machine}")
            if control["host_mutation_quarantined"] or control["drained"]:
                raise WorkspaceStateConflict("host mutation is quarantined or drained")
            if control.get("workspace_state") in IRREVERSIBLE_WORKSPACE_STATES:
                raise WorkspaceStateConflict(
                    f"host mutation rejected from workspace state "
                    f"{control['workspace_state']}"
                )
            cur.execute(
                """
                INSERT INTO host_operation_leases (
                  operation_id, machine, launch_id, lifecycle_generation,
                  operation_kind, resource_scope, source_revision, control_revision,
                  owner, deadline_at
                ) VALUES (
                  %(operation_id)s, %(machine)s, %(launch_id)s, %(lifecycle_generation)s,
                  %(operation_kind)s, %(resource_scope)s, %(source_revision)s,
                  %(control_revision)s, %(owner)s, %(deadline_at)s
                )
                """,
                {
                    "operation_id": operation_id,
                    "machine": machine,
                    "launch_id": launch_id,
                    "lifecycle_generation": lifecycle_generation,
                    "operation_kind": operation_kind,
                    "resource_scope": resource_scope,
                    "source_revision": source_revision,
                    "control_revision": int(control["control_revision"]),
                    "owner": owner,
                    "deadline_at": datetime.now(UTC) + max_duration,
                },
            )
    return operation_id


def finish_host_operation_lease(
    conn, *, operation_id: str, error: str | None = None, evidence: Mapping[str, Any] | None = None
) -> None:
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE host_operation_leases
                SET state=%(state)s, completed_at=now(), heartbeat_at=now(),
                  last_error=%(error)s, transport_evidence=%(evidence)s
                WHERE operation_id=%(operation_id)s
                  AND state IN ('registered','running','reconciling')
                RETURNING operation_id
                """,
                {
                    "operation_id": operation_id,
                    "state": "failed" if error else "completed",
                    "error": error,
                    "evidence": json_arg(dict(evidence or {})),
                },
            )
            if not cur.fetchone():
                raise WorkspaceStateConflict("host operation lease is not active")


def quarantine_expired_host_operation_leases(conn, *, limit: int = 25) -> int:
    quarantined = 0
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT operation_id, machine
                FROM host_operation_leases
                WHERE state IN ('registered','running','reconciling')
                  AND deadline_at <= now()
                ORDER BY machine, deadline_at
                LIMIT %(limit)s
                """,
                {"limit": max(1, min(int(limit), 100))},
            )
            rows = [dict(row) for row in cur.fetchall()]
            for row in rows:
                acquire_machine_control_xact_lock(conn, machine=str(row["machine"]))
                cur.execute(
                    """
                    UPDATE host_operation_leases
                    SET state='failed', completed_at=now(), heartbeat_at=now(),
                      last_error='operation lease expired with ambiguous host outcome'
                    WHERE operation_id=%(operation_id)s
                      AND state IN ('registered','running','reconciling')
                    """,
                    {"operation_id": row["operation_id"]},
                )
                cur.execute(
                    """
                    UPDATE machine_controls
                    SET host_mutation_quarantined=TRUE, cleanup_enabled=FALSE,
                      normal_admission_disabled=TRUE,
                      control_revision=control_revision+1,
                      reason=%(reason)s, updated_at=now()
                    WHERE machine=%(machine)s
                    """,
                    {
                        "machine": row["machine"],
                        "reason": f"expired host operation {row['operation_id']}",
                    },
                )
                quarantined += 1
    return quarantined
