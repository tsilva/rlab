from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from rlab.r2_store import (
    ConditionalWriteConflict,
    R2Bucket,
    RunStorageConfig,
)
from rlab.run_contracts import (
    CheckpointManifest,
    EvalIntent,
    EvalResult,
    PromotionReceipt,
    RunManifest,
    TerminalReceipt,
    checkpoint_id,
    utc_now,
)
from rlab.policy_bundle import model_document_path, recipe_document_path


LEASE_TTL_SECONDS = 60
LEASE_RENEW_SECONDS = 15
LEASE_MISSES_BEFORE_STOP = 2


class LeaseUnavailable(RuntimeError):
    pass


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


@dataclass(frozen=True)
class Lease:
    run_id: str
    attempt_id: str
    holder_id: str
    generation: int
    acquired_at: str
    renewed_at: str
    expires_at: str
    etag: str

    @classmethod
    def from_document(cls, value: Mapping[str, Any], *, etag: str) -> Lease:
        return cls(
            run_id=str(value["run_id"]),
            attempt_id=str(value["attempt_id"]),
            holder_id=str(value["holder_id"]),
            generation=int(value["generation"]),
            acquired_at=str(value["acquired_at"]),
            renewed_at=str(value["renewed_at"]),
            expires_at=str(value["expires_at"]),
            etag=etag,
        )

    def document(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "holder_id": self.holder_id,
            "generation": self.generation,
            "acquired_at": self.acquired_at,
            "renewed_at": self.renewed_at,
            "expires_at": self.expires_at,
        }


class RunAuthority:
    def __init__(self, storage: RunStorageConfig):
        self.storage = storage
        self.control = R2Bucket(storage.control)
        self.evaluation = R2Bucket(storage.evaluation)
        self.models = R2Bucket(storage.models)

    @staticmethod
    def run_prefix(run_id: str) -> str:
        return f"runs/{run_id}"

    def create_manifest(self, manifest: RunManifest) -> str:
        etag = self.control.put_json(
            f"{self.run_prefix(manifest.run_id)}/manifest.json",
            manifest.to_dict(),
            create_only=True,
        )
        self.create_attempt_manifest(manifest)
        return etag

    def create_attempt_manifest(self, manifest: RunManifest) -> str:
        return self.control.put_json(
            f"{self.run_prefix(manifest.run_id)}/attempts/"
            f"{manifest.attempt_id}/manifest.json",
            manifest.to_dict(),
            create_only=True,
        )

    def manifest(self, run_id: str) -> dict[str, Any] | None:
        return self.control.get_json_optional(f"{self.run_prefix(run_id)}/manifest.json")

    def acquire_lease(
        self,
        *,
        run_id: str,
        attempt_id: str,
        holder_id: str,
        now: datetime | None = None,
    ) -> Lease:
        instant = (now or datetime.now(UTC)).astimezone(UTC)
        key = f"{self.run_prefix(run_id)}/writer-lease.json"
        current = self.control.get_json_optional(key)
        current_etag = str(self.control.head(key)["etag"]) if current is not None else None
        if current is not None and _parse_timestamp(str(current["expires_at"])) > instant:
            if (
                str(current["attempt_id"]) != attempt_id
                or str(current["holder_id"]) != holder_id
            ):
                raise LeaseUnavailable(
                    f"writer lease is held by {current['attempt_id']}/{current['holder_id']}"
                )
        generation = int(current.get("generation") or 0) + 1 if current is not None else 1
        acquired = (
            str(current["acquired_at"])
            if current is not None
            and str(current["attempt_id"]) == attempt_id
            and str(current["holder_id"]) == holder_id
            else instant.isoformat().replace("+00:00", "Z")
        )
        renewed = instant.isoformat().replace("+00:00", "Z")
        document = {
            "schema_version": 1,
            "run_id": run_id,
            "attempt_id": attempt_id,
            "holder_id": holder_id,
            "generation": generation,
            "acquired_at": acquired,
            "renewed_at": renewed,
            "expires_at": (instant + timedelta(seconds=LEASE_TTL_SECONDS))
            .isoformat()
            .replace("+00:00", "Z"),
        }
        try:
            etag = self.control.put_json(
                key,
                document,
                create_only=current is None,
                if_match=current_etag,
            )
        except ConditionalWriteConflict as exc:
            raise LeaseUnavailable("writer lease changed while acquiring it") from exc
        return Lease.from_document(document, etag=etag)

    def renew_lease(self, lease: Lease, *, now: datetime | None = None) -> Lease:
        instant = (now or datetime.now(UTC)).astimezone(UTC)
        if _parse_timestamp(lease.expires_at) <= instant:
            raise LeaseUnavailable("writer lease expired before renewal")
        key = f"{self.run_prefix(lease.run_id)}/writer-lease.json"
        document = {
            **lease.document(),
            "generation": lease.generation + 1,
            "renewed_at": instant.isoformat().replace("+00:00", "Z"),
            "expires_at": (instant + timedelta(seconds=LEASE_TTL_SECONDS))
            .isoformat()
            .replace("+00:00", "Z"),
        }
        try:
            etag = self.control.put_json(
                key,
                document,
                create_only=False,
                if_match=lease.etag,
            )
        except ConditionalWriteConflict as exc:
            raise LeaseUnavailable("writer lease was lost during renewal") from exc
        return Lease.from_document(document, etag=etag)

    def seal_metric_segment(
        self,
        *,
        run_id: str,
        attempt_id: str,
        events: Sequence[Mapping[str, Any]],
    ) -> tuple[str, str]:
        if not events:
            raise ValueError("cannot seal an empty metric segment")
        rows = [dict(event) for event in events]
        sequences = [int(row["event_seq"]) for row in rows]
        if sequences != sorted(set(sequences)):
            raise ValueError("metric segment event_seq values must be strictly increasing")
        payload = b"".join(
            json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
            for row in rows
        )
        digest = hashlib.sha256(payload).hexdigest()
        key = (
            f"{self.run_prefix(run_id)}/attempts/{attempt_id}/metric-segments/"
            f"{sequences[0]:020d}-{sequences[-1]:020d}-{digest}.jsonl"
        )
        self.control.put_bytes(
            key,
            payload,
            content_type="application/x-ndjson",
            create_only=True,
            metadata={"sha256": digest},
        )
        return key, digest

    def archive_metric_journals(self, *, run_id: str) -> dict[str, Any]:
        active_prefix = f"{self.run_prefix(run_id)}/attempts"
        active_keys = sorted(
            key
            for key in self.control.iter_keys(active_prefix)
            if "/metric-segments/" in key and key.endswith(".jsonl")
        )
        archive_prefix = f"expiring-metric-journals/{run_id}/"
        archived_keys = sorted(
            key
            for key in self.control.iter_keys(archive_prefix)
            if key.endswith(".jsonl")
        )
        for source_key in active_keys:
            suffix = source_key.split(f"{active_prefix}/", 1)[1]
            attempt_id, remainder = suffix.split("/", 1)
            if not remainder.startswith("metric-segments/"):
                raise ValueError(f"invalid metric-journal key: {source_key}")
            destination_key = (
                f"expiring-metric-journals/{run_id}/{attempt_id}/"
                f"{remainder.removeprefix('metric-segments/')}"
            )
            source_etag = str(self.control.head(source_key)["etag"])
            self.control.copy_within(source_key, destination_key)
            self.control.delete(source_key, if_match=source_etag)
            if destination_key not in archived_keys:
                archived_keys.append(destination_key)
        archived_keys.sort()
        return {
            "prefix": archive_prefix,
            "segment_count": len(archived_keys),
            "keys": archived_keys,
        }

    def publish_checkpoint(
        self,
        *,
        run_id: str,
        model_path: Path,
        step: int,
        purpose: str,
        contract_hashes: Mapping[str, str],
        recovery_sidecar: Mapping[str, Any],
        created_at: str | None = None,
    ) -> CheckpointManifest:
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
        model_sidecar = model_document_path(model_path)
        recipe_sidecar = recipe_document_path(model_path)
        if not model_sidecar.is_file() or not recipe_sidecar.is_file():
            raise ValueError("checkpoint is missing its immutable model.json or recipe.json")
        model_sidecar_digest = hashlib.sha256(model_sidecar.read_bytes()).hexdigest()
        recipe_sidecar_digest = hashlib.sha256(recipe_sidecar.read_bytes()).hexdigest()
        identifier = checkpoint_id(step=step, sha256=digest)
        public_prefix = f"{self.run_prefix(run_id)}/checkpoints/{int(step)}-{digest}"
        model_key = f"{public_prefix}/model.zip"
        model_document_key = f"{public_prefix}/model.json"
        recipe_document_key = f"{public_prefix}/recipe.json"
        manifest_key = f"{public_prefix}/manifest.json"
        sidecar_key = (
            f"{self.run_prefix(run_id)}/checkpoints/{identifier}/recovery-sidecar.json"
        )
        self.control.put_json(sidecar_key, recovery_sidecar, create_only=True)
        self.models.put_file(
            model_key,
            model_path,
            sha256=digest,
            content_type="application/zip",
            cache_control="public, max-age=31536000, immutable",
        )
        self.models.put_file(
            model_document_key,
            model_sidecar,
            sha256=model_sidecar_digest,
            content_type="application/json",
            cache_control="public, max-age=31536000, immutable",
        )
        self.models.put_file(
            recipe_document_key,
            recipe_sidecar,
            sha256=recipe_sidecar_digest,
            content_type="application/json",
            cache_control="public, max-age=31536000, immutable",
        )
        manifest = CheckpointManifest(
            run_id=run_id,
            checkpoint_id=identifier,
            step=int(step),
            purpose=str(purpose),  # type: ignore[arg-type]
            sha256=digest,
            size_bytes=model_path.stat().st_size,
            public_url=self.models.public_url(model_key),
            model_document_url=self.models.public_url(model_document_key),
            model_document_sha256=model_sidecar_digest,
            recipe_document_url=self.models.public_url(recipe_document_key),
            recipe_document_sha256=recipe_sidecar_digest,
            goal_sha256=str(contract_hashes["goal_sha256"]),
            recipe_sha256=str(contract_hashes["recipe_sha256"]),
            environment_sha256=str(contract_hashes["environment_sha256"]),
            evaluation_contract_sha256=str(
                contract_hashes["evaluation_contract_sha256"]
            ),
            recovery_sidecar_key=sidecar_key,
            created_at=str(created_at or utc_now()),
        )
        self.models.put_json(
            manifest_key,
            manifest.to_dict(),
            create_only=True,
            cache_control="public, max-age=31536000, immutable",
        )
        self._upsert_public_index(manifest)
        return manifest

    def _update_public_index(
        self,
        run_id: str,
        *,
        checkpoint: CheckpointManifest | None = None,
        promotion: PromotionReceipt | None = None,
    ) -> None:
        key = f"{self.run_prefix(run_id)}/index.json"
        for _attempt in range(8):
            current = self.models.get_json_optional(key)
            etag = str(self.models.head(key)["etag"]) if current is not None else None
            rows = list((current or {}).get("checkpoints") or [])
            if checkpoint is not None:
                if checkpoint.run_id != run_id:
                    raise ValueError("checkpoint does not belong to the public run index")
                if not any(
                    str(row.get("checkpoint_id") or "") == checkpoint.checkpoint_id
                    for row in rows
                    if isinstance(row, Mapping)
                ):
                    rows.append(checkpoint.to_dict())
            rows.sort(key=lambda row: (int(row["step"]), str(row["sha256"])))
            promoted = (current or {}).get("promotion")
            if promotion is not None:
                if promotion.run_id != run_id:
                    raise ValueError("promotion does not belong to the public run index")
                if not any(
                    str(row.get("checkpoint_id") or "") == promotion.checkpoint_id
                    for row in rows
                    if isinstance(row, Mapping)
                ):
                    raise ValueError("promoted checkpoint is missing from the public run index")
                promoted = {
                    "checkpoint_id": promotion.checkpoint_id,
                    "checkpoint_step": promotion.checkpoint_step,
                    "eval_result_sha256": promotion.eval_result_sha256,
                    "accepted_episode_count": promotion.accepted_episode_count,
                    "promoted_at": promotion.promoted_at,
                }
            document = {
                "schema_version": 1,
                "run_id": run_id,
                "updated_at": utc_now(),
                "checkpoints": rows,
                "promotion": promoted,
            }
            try:
                self.models.put_json(
                    key,
                    document,
                    create_only=current is None,
                    if_match=etag,
                    cache_control="no-store",
                )
                return
            except ConditionalWriteConflict:
                continue
        raise RuntimeError("public run index CAS did not converge")

    def _upsert_public_index(self, checkpoint: CheckpointManifest) -> None:
        self._update_public_index(checkpoint.run_id, checkpoint=checkpoint)

    def put_eval_intent(self, intent: EvalIntent) -> str:
        return self.evaluation.put_json(
            f"{self.run_prefix(intent.run_id)}/evals/{intent.idempotency_key}/intent.json",
            intent.to_dict(),
            create_only=True,
        )

    def put_eval_dispatch(
        self,
        *,
        run_id: str,
        idempotency_key: str,
        attempt: int,
        modal_call_id: str,
    ) -> str:
        return self.evaluation.put_json(
            f"{self.run_prefix(run_id)}/evals/{idempotency_key}/"
            f"dispatch-{int(attempt)}.json",
            {
                "schema_version": 1,
                "run_id": run_id,
                "idempotency_key": idempotency_key,
                "attempt": int(attempt),
                "modal_call_id": modal_call_id,
                "dispatched_at": utc_now(),
            },
            create_only=True,
        )

    def prepare_eval_attempt(
        self,
        *,
        run_id: str,
        idempotency_key: str,
        attempt: int,
        expires_at: float,
    ) -> dict[str, Any]:
        key = (
            f"{self.run_prefix(run_id)}/evals/{idempotency_key}/"
            f"attempt-{int(attempt)}.json"
        )
        document = {
            "schema_version": 1,
            "run_id": run_id,
            "idempotency_key": idempotency_key,
            "attempt": int(attempt),
            "expires_at": float(expires_at),
        }
        self.evaluation.put_json(key, document, create_only=True)
        return document

    def eval_attempt(
        self,
        *,
        run_id: str,
        idempotency_key: str,
        attempt: int,
    ) -> dict[str, Any] | None:
        return self.evaluation.get_json_optional(
            f"{self.run_prefix(run_id)}/evals/{idempotency_key}/"
            f"attempt-{int(attempt)}.json"
        )

    def eval_dispatch(
        self,
        *,
        run_id: str,
        idempotency_key: str,
        attempt: int,
    ) -> dict[str, Any] | None:
        return self.evaluation.get_json_optional(
            f"{self.run_prefix(run_id)}/evals/{idempotency_key}/"
            f"dispatch-{int(attempt)}.json"
        )

    def eval_result(
        self,
        *,
        run_id: str,
        idempotency_key: str,
    ) -> dict[str, Any] | None:
        return self.evaluation.get_json_optional(
            f"{self.run_prefix(run_id)}/evals/{idempotency_key}/result.json"
        )

    def put_verified_eval_result(self, result: EvalResult) -> str:
        return self.evaluation.put_json(
            f"{self.run_prefix(result.run_id)}/evals/"
            f"{result.idempotency_key}/verified-result.json",
            result.to_dict(),
            create_only=True,
        )

    def create_promotion(self, receipt: PromotionReceipt) -> str:
        etag = self.control.put_json(
            f"{self.run_prefix(receipt.run_id)}/promotion.json",
            receipt.to_dict(),
            create_only=True,
        )
        self._update_public_index(receipt.run_id, promotion=receipt)
        return etag

    def create_terminal(self, receipt: TerminalReceipt) -> str:
        receipt.validate()
        if receipt.state != "succeeded":
            raise ValueError("canonical terminal receipt is reserved for scientific success")
        if receipt.acceptance_required is not True:
            raise ValueError(
                "canonical terminal receipt requires acceptance-backed evaluation"
            )
        drain = dict(receipt.drain)
        if drain.get("complete") is not True:
            raise ValueError("successful terminal receipt requires a complete drain")
        if int(receipt.wandb_high_water_mark) <= 0:
            raise ValueError("successful terminal receipt requires W&B metric delivery")
        if int(drain.get("metric_segment_high_water") or 0) != int(
            receipt.wandb_high_water_mark
        ):
            raise ValueError("R2 and W&B delivery high-water marks do not match")
        if int(drain.get("wandb_remote_high_water_mark") or 0) < int(
            receipt.wandb_high_water_mark
        ):
            raise ValueError("W&B delivery is not remotely visible")
        capacity_ratio = drain.get("publication_capacity_ratio")
        if capacity_ratio is not None and float(capacity_ratio) < 2.0:
            raise ValueError("W&B publication capacity is below twice peak ingress")
        journal_archive = drain.get("journal_archive")
        if (
            not isinstance(journal_archive, Mapping)
            or int(journal_archive.get("segment_count") or 0) <= 0
            or not str(drain.get("journal_expires_at") or "")
        ):
            raise ValueError("delivered metric journals are not scheduled for expiry")

        checkpoints = [dict(row) for row in receipt.checkpoint_inventory]
        evals = [dict(row) for row in receipt.eval_inventory]
        if not checkpoints or not evals:
            raise ValueError("successful terminal receipt requires checkpoint and eval inventory")
        checkpoint_ids = {str(row.get("checkpoint_id") or "") for row in checkpoints}
        eval_checkpoint_ids = {str(row.get("checkpoint_id") or "") for row in evals}
        if (
            "" in checkpoint_ids
            or len(checkpoints) != len(checkpoint_ids)
            or checkpoint_ids != eval_checkpoint_ids
        ):
            raise ValueError("every checkpoint must have exactly one terminal eval inventory entry")
        if len(evals) != len(eval_checkpoint_ids):
            raise ValueError("eval inventory contains duplicate checkpoint entries")
        if not any(str(row.get("purpose") or "") == "final" for row in checkpoints):
            raise ValueError("successful terminal receipt requires a final checkpoint")
        maximum_step = max(int(row.get("step") or 0) for row in checkpoints)
        if int(receipt.final_step) != maximum_step:
            raise ValueError("terminal final_step does not match checkpoint inventory")
        accepted = [
            row for row in evals if str(row.get("status") or "") == "accepted"
        ]
        if not accepted:
            raise ValueError("successful terminal receipt requires an accepted evaluation")
        selected = min(
            accepted,
            key=lambda row: (
                int(row.get("checkpoint_step") or 0),
                str(row.get("checkpoint_id") or ""),
            ),
        )
        promotion = self.control.get_json_optional(
            f"{self.run_prefix(receipt.run_id)}/promotion.json"
        )
        if (
            promotion is None
            or str(promotion.get("checkpoint_id") or "")
            != str(selected.get("checkpoint_id") or "")
            or int(promotion.get("checkpoint_step") or -1)
            != int(selected.get("checkpoint_step") or 0)
        ):
            raise ValueError("promotion is not the lowest-step accepted checkpoint")
        return self.control.put_json(
            f"{self.run_prefix(receipt.run_id)}/terminal.json",
            receipt.to_dict(),
            create_only=True,
        )

    def create_attempt_terminal(self, receipt: TerminalReceipt) -> str:
        return self.control.put_json(
            f"{self.run_prefix(receipt.run_id)}/attempts/"
            f"{receipt.attempt_id}/terminal.json",
            receipt.to_dict(),
            create_only=True,
        )

    def has_accepted_eval(self, run_id: str) -> bool:
        prefix = f"{self.run_prefix(run_id)}/evals"
        for key in self.evaluation.iter_keys(prefix):
            if not key.endswith("/verified-result.json"):
                continue
            if str(self.evaluation.get_json(key).get("status") or "") == "accepted":
                return True
        return False

    def semantic_state(self, run_id: str) -> dict[str, Any]:
        prefix = self.run_prefix(run_id)
        manifest = self.control.get_json_optional(f"{prefix}/manifest.json")
        terminal = self.control.get_json_optional(f"{prefix}/terminal.json")
        promotion = self.control.get_json_optional(f"{prefix}/promotion.json")
        public_index = self.models.get_json_optional(f"{prefix}/index.json")
        eval_keys = list(self.evaluation.iter_keys(f"{prefix}/evals"))
        control_keys = list(self.control.iter_keys(f"{prefix}/attempts"))
        attempt_manifests = [
            self.control.get_json(key)
            for key in control_keys
            if key.endswith("/manifest.json")
        ]
        attempt_terminals = [
            self.control.get_json(key)
            for key in control_keys
            if key.endswith("/terminal.json")
        ]
        attempt_manifests.sort(key=lambda row: str(row.get("created_at") or ""))
        attempt_terminals.sort(key=lambda row: str(row.get("completed_at") or ""))
        return {
            "run_id": run_id,
            "manifest": manifest,
            "terminal": terminal,
            "promotion": promotion,
            "public_index": public_index,
            "eval_intents": sum(key.endswith("/intent.json") for key in eval_keys),
            "eval_results": sum(key.endswith("/result.json") for key in eval_keys),
            "verified_eval_results": sum(
                key.endswith("/verified-result.json") for key in eval_keys
            ),
            "attempts": attempt_manifests,
            "attempt_terminals": attempt_terminals,
            "observed_at": time.time(),
        }
