from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import os
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final


CANONICAL_FORMAT_VERSION: Final = "canonical-jsonl-v1"
ADAPTER_VERSION: Final = "telemetry-adapters-v1"
NORMALIZATION_VERSION: Final = "wandb-normalization-v1"
EVIDENCE_VERSION: Final = "telemetry-evidence-v1"

DURABILITY_POLICIES: Final = {
    "queued_dual_r2_v1",
    "local_mirrored_v1",
    "local_singlecopy_optout_v1",
}
INTEGRITY_CLASSIFICATIONS: Final = {
    "pending",
    "intact_with_proof",
    "degraded",
    "legacy_unknown",
}
TERMINAL_OBLIGATION_DISPOSITIONS: Final = {
    "complete",
    "canceled",
    "aborted_before_release",
    "disabled",
    "failed",
}


class TelemetryContractError(ValueError):
    pass


class TelemetryIntegrityError(RuntimeError):
    pass


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_json(value: object) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _canonical_float(value: float) -> dict[str, str]:
    if not math.isfinite(value):
        raise TelemetryContractError("non-finite floats are not canonical telemetry values")
    return {"type": "f64", "hex": value.hex()}


Adapter = Callable[[object, "CanonicalAdapterRegistry"], object]


class CanonicalAdapterRegistry:
    """Closed conversion from runtime values to deterministic typed JSON values."""

    def __init__(self) -> None:
        self._adapters: dict[type[object], tuple[str, Adapter]] = {}

    def register(self, value_type: type[object], name: str, adapter: Adapter) -> None:
        name = str(name).strip()
        if not name:
            raise TelemetryContractError("adapter name is required")
        if value_type in self._adapters:
            raise TelemetryContractError(f"adapter already registered for {value_type!r}")
        self._adapters[value_type] = (name, adapter)

    def encode(self, value: object) -> object:
        if value is None:
            return {"type": "null"}
        if isinstance(value, bool):
            return {"type": "bool", "value": value}
        if isinstance(value, int):
            return {"type": "i64", "value": str(value)}
        if isinstance(value, float):
            return _canonical_float(value)
        if isinstance(value, str):
            return {"type": "string", "value": value}
        if isinstance(value, bytes):
            raise TelemetryContractError(
                "raw bytes require an immutable media or artifact reference adapter"
            )
        if isinstance(value, Mapping):
            marker = str(value.get("_type") or "") if "_type" in value else ""
            if marker:
                return self._encode_marked_mapping(marker, value)
            entries = [
                [str(key), self.encode(item)]
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            ]
            if len({key for key, _ in entries}) != len(entries):
                raise TelemetryContractError("mapping keys collide after string normalization")
            return {"type": "mapping", "entries": entries}
        if isinstance(value, Sequence):
            return {"type": "sequence", "items": [self.encode(item) for item in value]}
        registered = self._adapters.get(type(value))
        if registered is None:
            raise TelemetryContractError(
                f"unregistered telemetry value type: {type(value).__module__}."
                f"{type(value).__qualname__}"
            )
        name, adapter = registered
        return {
            "type": "registered",
            "adapter": name,
            "adapter_version": ADAPTER_VERSION,
            "value": adapter(value, self),
        }

    def _encode_marked_mapping(self, marker: str, value: Mapping[object, object]) -> object:
        payload = {str(key): item for key, item in value.items() if str(key) != "_type"}
        if marker == "histogram_v1":
            bins = payload.get("bins")
            counts = payload.get("counts")
            if not isinstance(bins, Sequence) or isinstance(bins, str | bytes):
                raise TelemetryContractError("histogram_v1 bins must be a sequence")
            if not isinstance(counts, Sequence) or isinstance(counts, str | bytes):
                raise TelemetryContractError("histogram_v1 counts must be a sequence")
            if len(bins) != len(counts) + 1:
                raise TelemetryContractError(
                    "histogram_v1 requires exactly one more bin edge than count"
                )
            return {
                "type": marker,
                "bins": [self.encode(float(item)) for item in bins],
                "counts": [self.encode(int(item)) for item in counts],
            }
        if marker == "table_v1":
            columns = payload.get("columns")
            rows = payload.get("rows")
            if not isinstance(columns, Sequence) or isinstance(columns, str | bytes):
                raise TelemetryContractError("table_v1 columns must be a sequence")
            if not isinstance(rows, Sequence) or isinstance(rows, str | bytes):
                raise TelemetryContractError("table_v1 rows must be a sequence")
            names = [str(column) for column in columns]
            if len(set(names)) != len(names):
                raise TelemetryContractError("table_v1 column names must be unique")
            encoded_rows = []
            for row in rows:
                if not isinstance(row, Sequence) or isinstance(row, str | bytes):
                    raise TelemetryContractError("table_v1 rows must contain sequences")
                if len(row) != len(names):
                    raise TelemetryContractError("table_v1 row width does not match columns")
                encoded_rows.append([self.encode(item) for item in row])
            return {"type": marker, "columns": names, "rows": encoded_rows}
        if marker == "preview_v1":
            required = {"sha256", "media_type", "size_bytes"}
            if not required.issubset(payload):
                raise TelemetryContractError(
                    f"preview_v1 is missing: {sorted(required - set(payload))}"
                )
            return {
                "type": marker,
                "sha256": _require_sha256(payload["sha256"], "preview sha256"),
                "media_type": str(payload["media_type"]),
                "size_bytes": _require_nonnegative_int(
                    payload["size_bytes"], "preview size_bytes"
                ),
                "uri": _require_immutable_uri(payload.get("uri")),
                "metadata": self.encode(payload.get("metadata") or {}),
            }
        if marker == "immutable_ref_v1":
            return {
                "type": marker,
                "sha256": _require_sha256(payload.get("sha256"), "reference sha256"),
                "uri": _require_immutable_uri(payload.get("uri")),
                "media_type": str(payload.get("media_type") or "application/octet-stream"),
                "size_bytes": _require_nonnegative_int(
                    payload.get("size_bytes"), "reference size_bytes"
                ),
            }
        raise TelemetryContractError(f"unregistered marked telemetry value: {marker}")


DEFAULT_ADAPTERS = CanonicalAdapterRegistry()


def _require_sha256(value: object, label: str) -> str:
    digest = str(value or "").strip().lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise TelemetryContractError(f"{label} must be a lowercase SHA-256")
    return digest


def _require_nonnegative_int(value: object, label: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise TelemetryContractError(f"{label} must be an integer") from exc
    if result < 0:
        raise TelemetryContractError(f"{label} must not be negative")
    return result


def _require_immutable_uri(value: object) -> str:
    uri = str(value or "").strip()
    if not uri.startswith(("s3://", "r2://", "file://", "wandb-artifact://")):
        raise TelemetryContractError("immutable references require an approved URI scheme")
    if uri.startswith(("http://", "https://")):
        raise TelemetryContractError("mutable HTTP references are not canonical telemetry")
    return uri


@dataclass(frozen=True, order=True)
class ProducerKey:
    run_id: int
    generation: int
    producer_ordinal: int
    producer_identity: str

    def __post_init__(self) -> None:
        if self.run_id < 1 or self.generation < 1 or self.producer_ordinal < 0:
            raise TelemetryContractError("producer identifiers are outside their valid range")
        if not self.producer_identity.strip():
            raise TelemetryContractError("producer identity is required")


@dataclass(frozen=True)
class CanonicalEvent:
    producer: ProducerKey
    source_sequence: int
    event_id: str
    kind: str
    payload: Mapping[str, object]
    global_step: int | None = None

    def __post_init__(self) -> None:
        if self.source_sequence < 1:
            raise TelemetryContractError("source sequence must be at least one")
        if not self.event_id.strip() or not self.kind.strip():
            raise TelemetryContractError("event id and kind are required")
        if self.global_step is not None and self.global_step < 0:
            raise TelemetryContractError("global step must not be negative")

    @property
    def stable_identity(self) -> str:
        return (
            f"{self.producer.run_id}:{self.producer.generation}:"
            f"{self.producer.producer_ordinal}:{self.source_sequence}:{self.event_id}"
        )

    def document(self, registry: CanonicalAdapterRegistry = DEFAULT_ADAPTERS) -> dict[str, object]:
        return {
            "format_version": CANONICAL_FORMAT_VERSION,
            "run_id": self.producer.run_id,
            "generation": self.producer.generation,
            "producer_ordinal": self.producer.producer_ordinal,
            "producer_identity": self.producer.producer_identity,
            "source_sequence": self.source_sequence,
            "event_id": self.event_id,
            "stable_identity": self.stable_identity,
            "kind": self.kind,
            "global_step": self.global_step,
            "payload": registry.encode(self.payload),
        }


@dataclass(frozen=True)
class CanonicalSegment:
    first_sequence: int
    last_sequence: int
    event_count: int
    uncompressed_sha256: str
    compressed_sha256: str
    payload: bytes


def build_canonical_segment(
    events: Iterable[CanonicalEvent],
    *,
    registry: CanonicalAdapterRegistry = DEFAULT_ADAPTERS,
) -> CanonicalSegment:
    ordered = sorted(
        events,
        key=lambda event: (
            event.producer.producer_ordinal,
            event.source_sequence,
            event.event_id,
        ),
    )
    if not ordered:
        raise TelemetryContractError("canonical segments cannot be empty")
    seen: set[tuple[int, int]] = set()
    expected_by_producer: dict[int, int] = {}
    lines: list[bytes] = []
    for event in ordered:
        identity = (event.producer.producer_ordinal, event.source_sequence)
        if identity in seen:
            raise TelemetryContractError(f"duplicate canonical source identity: {identity}")
        seen.add(identity)
        previous = expected_by_producer.get(event.producer.producer_ordinal)
        if previous is not None and event.source_sequence != previous + 1:
            raise TelemetryContractError(
                "canonical segment contains a producer sequence gap: "
                f"producer={event.producer.producer_ordinal} "
                f"expected={previous + 1} observed={event.source_sequence}"
            )
        expected_by_producer[event.producer.producer_ordinal] = event.source_sequence
        lines.append(canonical_json_bytes(event.document(registry)).rstrip(b"\n"))
    uncompressed = b"\n".join(lines) + b"\n"
    output = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0, compresslevel=9) as stream:
        stream.write(uncompressed)
    payload = output.getvalue()
    return CanonicalSegment(
        first_sequence=min(event.source_sequence for event in ordered),
        last_sequence=max(event.source_sequence for event in ordered),
        event_count=len(ordered),
        uncompressed_sha256=sha256_bytes(uncompressed),
        compressed_sha256=sha256_bytes(payload),
        payload=payload,
    )


def decode_canonical_segment(payload: bytes) -> list[dict[str, object]]:
    try:
        uncompressed = gzip.decompress(payload)
    except OSError as exc:
        raise TelemetryContractError("canonical segment is not valid gzip") from exc
    documents: list[dict[str, object]] = []
    for line in uncompressed.splitlines():
        value = json.loads(line)
        if not isinstance(value, dict) or value.get("format_version") != CANONICAL_FORMAT_VERSION:
            raise TelemetryContractError("canonical segment contains an invalid document")
        documents.append(value)
    if not documents:
        raise TelemetryContractError("canonical segment contains no documents")
    return documents


@dataclass(frozen=True)
class IntegrityInputs:
    classification: str
    expected_obligations: Mapping[str, str]
    realized_obligations: Mapping[str, str]
    producer_final_claims: Mapping[int, tuple[int, str]]
    archived_coverage: Mapping[int, int]
    no_more_producers: bool
    durability_policy: str
    required_archive_receipts: int
    observed_archive_receipts: int
    recovery_pending: bool
    incidents: Sequence[str] = ()


@dataclass(frozen=True)
class IntegrityResult:
    exact: bool
    cleanup_eligible: bool
    reasons: tuple[str, ...]
    expected_set_sha256: str
    coverage_sha256: str


def reduce_integrity(inputs: IntegrityInputs) -> IntegrityResult:
    if inputs.classification not in INTEGRITY_CLASSIFICATIONS:
        raise TelemetryContractError(f"invalid integrity classification: {inputs.classification}")
    if inputs.durability_policy not in DURABILITY_POLICIES:
        raise TelemetryContractError(f"invalid durability policy: {inputs.durability_policy}")
    reasons: list[str] = []
    expected = dict(inputs.expected_obligations)
    realized = dict(inputs.realized_obligations)
    if set(expected) != set(realized):
        reasons.append("expected_obligation_set_mismatch")
    for key in sorted(set(expected) & set(realized)):
        if expected[key] != realized[key]:
            reasons.append(f"obligation_disposition_mismatch:{key}")
        if realized[key] not in TERMINAL_OBLIGATION_DISPOSITIONS:
            reasons.append(f"obligation_not_terminal:{key}")
    if not inputs.no_more_producers:
        reasons.append("producers_not_frozen")
    for ordinal, (final_sequence, final_digest) in sorted(inputs.producer_final_claims.items()):
        if final_sequence < 0:
            reasons.append(f"invalid_final_sequence:{ordinal}")
        try:
            _require_sha256(final_digest, f"producer {ordinal} final digest")
        except TelemetryContractError:
            reasons.append(f"invalid_final_digest:{ordinal}")
        if int(inputs.archived_coverage.get(ordinal, -1)) != final_sequence:
            reasons.append(f"archive_coverage_mismatch:{ordinal}")
    extra_coverage = set(inputs.archived_coverage) - set(inputs.producer_final_claims)
    if extra_coverage:
        reasons.append("archive_contains_unknown_producer")
    if inputs.observed_archive_receipts < inputs.required_archive_receipts:
        reasons.append("archive_receipts_incomplete")
    if inputs.recovery_pending:
        reasons.append("recovery_pending")
    if inputs.incidents:
        reasons.append("integrity_incident_open")
    if inputs.classification != "intact_with_proof":
        reasons.append(f"classification:{inputs.classification}")
    if inputs.durability_policy == "local_singlecopy_optout_v1":
        reasons.append("durability_opted_out")
    exact = not reasons
    return IntegrityResult(
        exact=exact,
        cleanup_eligible=exact,
        reasons=tuple(reasons),
        expected_set_sha256=sha256_json(sorted(expected.items())),
        coverage_sha256=sha256_json(
            [
                {
                    "producer_ordinal": ordinal,
                    "final_sequence": final_sequence,
                    "final_digest": final_digest,
                    "archived_sequence": inputs.archived_coverage.get(ordinal),
                }
                for ordinal, (final_sequence, final_digest) in sorted(
                    inputs.producer_final_claims.items()
                )
            ]
        ),
    )


def normalize_wandb_rows(
    event: CanonicalEvent,
    *,
    first_ordinal: int,
    registry: CanonicalAdapterRegistry = DEFAULT_ADAPTERS,
) -> list[dict[str, object]]:
    if first_ordinal < 0:
        raise TelemetryContractError("W&B output ordinal must not be negative")
    document = event.document(registry)
    frames = event.payload.get("frames") if isinstance(event.payload, Mapping) else None
    raw_outputs: list[tuple[str, Mapping[str, object], int | None]]
    if event.kind == "metric_batch" and isinstance(frames, Sequence):
        raw_outputs = []
        for frame in frames:
            if not isinstance(frame, Mapping):
                raise TelemetryContractError("metric batch frames must be mappings")
            frame_payload = frame.get("payload")
            if not isinstance(frame_payload, Mapping):
                raise TelemetryContractError("metric batch frame payload must be a mapping")
            step = frame.get("global_step")
            raw_outputs.append(
                (
                    str(frame.get("kind") or "history"),
                    dict(frame_payload),
                    None if step is None else int(step),
                )
            )
    else:
        raw_outputs = [
            (
                event.kind,
                {"telemetry/event": document["payload"]},
                event.global_step,
            )
        ]
    rows: list[dict[str, object]] = []
    for index, (output_kind, output_payload, global_step) in enumerate(raw_outputs):
        ordinal = first_ordinal + index
        payload = {
            **dict(output_payload),
            "_rlab_event_id": event.stable_identity,
            "_rlab_adapter_version": ADAPTER_VERSION,
            "_rlab_normalization_version": NORMALIZATION_VERSION,
            "_rlab_output_index": index,
            "_rlab_output_ordinal": ordinal,
            "global_step": global_step,
            "telemetry/output_kind": output_kind,
        }
        rows.append(
            {
                "stable_key": (
                    f"{event.stable_identity}:{ADAPTER_VERSION}:"
                    f"{NORMALIZATION_VERSION}:{output_kind}:{index}"
                ),
                "source_event_id": event.stable_identity,
                "adapter_version": ADAPTER_VERSION,
                "normalization_version": NORMALIZATION_VERSION,
                "output_kind": output_kind,
                "output_index": index,
                "ordinal": ordinal,
                "payload": payload,
                "payload_sha256": sha256_json(payload),
            }
        )
    return rows


def _required_contract_mapping(value: Mapping[str, object], key: str) -> Mapping[str, object]:
    nested = value.get(key)
    if not isinstance(nested, Mapping) or not nested:
        raise TelemetryContractError(f"evaluation contract requires {key}")
    return nested


def build_eval_scope_exact(
    *,
    checkpoint: Mapping[str, object],
    evaluation_contract: Mapping[str, object],
    episode_manifest: Mapping[str, object],
    results: Sequence[Mapping[str, object]],
    acceptance_rule: Mapping[str, object],
    execution_key: str,
    attestation: Mapping[str, object],
) -> dict[str, object]:
    for key in (
        "canonical_goal_sha256",
        "effective_goal_contract_sha256",
        "recipe_sha256",
        "policy_bundle_sha256",
        "runtime_image_digest",
        "dependency_lock_sha256",
        "evaluator_implementation_sha256",
        "metrics_schema_version",
        "seed_protocol",
        "n_envs",
        "episodes",
        "max_steps",
    ):
        if evaluation_contract.get(key) in (None, ""):
            raise TelemetryContractError(f"evaluation contract requires {key}")
    for key in (
        "environment",
        "observation",
        "action",
        "preprocessing",
        "reward",
        "events",
        "starts",
        "termination",
        "assets",
    ):
        _required_contract_mapping(evaluation_contract, key)
    deterministic = evaluation_contract.get("deterministic")
    action_sampling = str(evaluation_contract.get("action_sampling") or "")
    if deterministic is not False or action_sampling != "stochastic":
        raise TelemetryContractError("evaluation evidence requires stochastic action sampling")
    checkpoint_sha256 = _require_sha256(checkpoint.get("sha256"), "checkpoint sha256")
    receipts = checkpoint.get("durability_receipts")
    if not isinstance(receipts, Sequence) or isinstance(receipts, str | bytes) or not receipts:
        raise TelemetryContractError("evaluation evidence requires checkpoint receipts")
    if not episode_manifest or not results:
        raise TelemetryContractError("evaluation evidence requires manifest and results")
    expected_episodes = _require_nonnegative_int(
        evaluation_contract["episodes"], "evaluation episodes"
    )
    if len(results) != expected_episodes:
        raise TelemetryContractError(
            f"evaluation results are incomplete: expected={expected_episodes} observed={len(results)}"
        )
    body = {
        "evidence_version": EVIDENCE_VERSION,
        "scope_kind": "eval_scope_exact",
        "checkpoint": {**dict(checkpoint), "sha256": checkpoint_sha256},
        "evaluation_contract": dict(evaluation_contract),
        "evaluation_contract_sha256": sha256_json(evaluation_contract),
        "episode_manifest": dict(episode_manifest),
        "episode_manifest_sha256": sha256_json(episode_manifest),
        "results": [dict(result) for result in results],
        "results_sha256": sha256_json([dict(result) for result in results]),
        "acceptance_rule": dict(acceptance_rule),
        "acceptance_rule_sha256": sha256_json(acceptance_rule),
        "execution_key": str(execution_key),
        "attestation": dict(attestation),
    }
    body["scope_sha256"] = sha256_json(body)
    return body


def build_training_success_scope_exact(
    *,
    contract: Mapping[str, object],
    success_event: Mapping[str, object],
    policy_artifact: Mapping[str, object],
) -> dict[str, object]:
    if contract.get("acceptance_mode") != "first_training_success":
        raise TelemetryContractError("training-success scope requires first_training_success")
    if contract.get("checkpoint_eval_backend") != "none":
        raise TelemetryContractError("training-success scope requires evaluation disabled")
    if not contract.get("deterministic_search_workflow"):
        raise TelemetryContractError("training-success scope requires declared deterministic search")
    for key in (
        "canonical_goal_sha256",
        "effective_goal_contract_sha256",
        "recipe_sha256",
        "environment_contract_sha256",
        "reward_program_sha256",
        "runtime_image_digest",
    ):
        if not contract.get(key):
            raise TelemetryContractError(f"training-success contract requires {key}")
    for key in ("event_id", "producer_ordinal", "source_sequence", "episode_id", "global_step"):
        if success_event.get(key) in (None, ""):
            raise TelemetryContractError(f"training success event requires {key}")
    _require_sha256(policy_artifact.get("sha256"), "policy artifact sha256")
    receipts = policy_artifact.get("durability_receipts")
    if not isinstance(receipts, Sequence) or isinstance(receipts, str | bytes) or not receipts:
        raise TelemetryContractError("training-success policy requires durability receipts")
    body = {
        "evidence_version": EVIDENCE_VERSION,
        "scope_kind": "training_success_scope_exact",
        "contract": dict(contract),
        "contract_sha256": sha256_json(contract),
        "success_event": dict(success_event),
        "success_event_sha256": sha256_json(success_event),
        "policy_artifact": dict(policy_artifact),
    }
    body["scope_sha256"] = sha256_json(body)
    return body


RUN_FACT_COMPARABILITY_FIELDS: Final = (
    "goal_slug",
    "canonical_goal_sha256",
    "effective_goal_contract_sha256",
    "target_scope",
    "reward_program_name",
    "reward_program_revision",
    "reward_program_sha256",
    "recipe_slug",
    "resolved_config_sha256",
    "environment_id",
    "environment_provider",
    "environment_contract_sha256",
    "training_backend",
    "runtime_image_digest",
    "dependency_lock_sha256",
    "source_sha",
    "metrics_schema_version",
    "rank_metric",
    "rank_direction",
)


def build_run_final_exact(
    *,
    archive_root_sha256: str,
    dimensions: Mapping[str, object],
    metrics: Mapping[str, object],
    seed: int,
    cohort_manifest: Mapping[str, object],
    integrity: Mapping[str, object],
) -> dict[str, object]:
    root = _require_sha256(archive_root_sha256, "archive root")
    missing = [
        field for field in RUN_FACT_COMPARABILITY_FIELDS if dimensions.get(field) in (None, "")
    ]
    if missing:
        raise TelemetryContractError(f"run facts lack comparability fields: {missing}")
    if dimensions["rank_direction"] not in {"min", "max"}:
        raise TelemetryContractError("rank direction must be min or max")
    expected_seeds = cohort_manifest.get("expected_seeds")
    if not isinstance(expected_seeds, Sequence) or isinstance(expected_seeds, str | bytes):
        raise TelemetryContractError("cohort manifest requires expected_seeds")
    normalized_seeds = sorted({int(value) for value in expected_seeds})
    if int(seed) not in normalized_seeds:
        raise TelemetryContractError("run seed is absent from its cohort manifest")
    if integrity.get("classification") != "intact_with_proof" or not integrity.get("exact"):
        raise TelemetryIntegrityError("final run facts require exact intact evidence")
    body = {
        "evidence_version": EVIDENCE_VERSION,
        "scope_kind": "run_final_exact",
        "archive_root_sha256": root,
        "dimensions": dict(dimensions),
        "comparability_sha256": sha256_json(
            {field: dimensions[field] for field in RUN_FACT_COMPARABILITY_FIELDS}
        ),
        "metrics": dict(metrics),
        "metrics_sha256": sha256_json(metrics),
        "seed": int(seed),
        "cohort_manifest": {**dict(cohort_manifest), "expected_seeds": normalized_seeds},
        "cohort_manifest_sha256": sha256_json(
            {**dict(cohort_manifest), "expected_seeds": normalized_seeds}
        ),
        "integrity": dict(integrity),
    }
    body["scope_sha256"] = sha256_json(body)
    return body


def require_comparable_run_facts(
    facts: Sequence[Mapping[str, object]],
    *,
    require_complete_cohort: bool,
) -> None:
    if not facts:
        raise TelemetryContractError("at least one run fact is required")
    keys = {str(fact.get("comparability_sha256") or "") for fact in facts}
    if "" in keys or len(keys) != 1:
        raise TelemetryContractError("run facts are not contract-comparable")
    manifests = {str(fact.get("cohort_manifest_sha256") or "") for fact in facts}
    if "" in manifests or len(manifests) != 1:
        raise TelemetryContractError("run facts do not share one cohort manifest")
    if require_complete_cohort:
        expected = {
            int(seed)
            for seed in facts[0].get("cohort_manifest", {}).get("expected_seeds", [])
        }
        observed = {int(fact["seed"]) for fact in facts}
        if observed != expected:
            raise TelemetryIntegrityError(
                f"cohort is incomplete: expected={sorted(expected)} observed={sorted(observed)}"
            )


def write_fsync(path: Path, payload: bytes, *, mode: int = 0o600) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(mode)
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    if path.read_bytes() != payload:
        raise TelemetryIntegrityError(f"fsynced file readback mismatch: {path}")
    return path
