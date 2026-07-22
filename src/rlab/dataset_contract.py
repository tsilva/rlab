from __future__ import annotations

import hashlib
import json
import math
import struct
import uuid
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DATASET_FORMAT_VERSION = 3
STORAGE_FORMAT_IMAGES = "images"
STORAGE_FORMAT_LOSSLESS_VIDEO = "lossless-video"
STORAGE_FORMATS = (STORAGE_FORMAT_IMAGES, STORAGE_FORMAT_LOSSLESS_VIDEO)
VIDEO_ARTIFACT_DIR = "videos"
ENVIRONMENT_ARTIFACT_DIR = "environments"
COLLECTOR_ARTIFACT_DIR = "collectors"
ENVIRONMENT_DOCUMENT_FILENAME = "environment.json"
ENVIRONMENT_DOCUMENT_TYPE = "gymrec.environment"
ENVIRONMENT_DOCUMENT_FORMAT_VERSION = 1
COLLECTION_DOCUMENT_TYPE = "gymrec.collection"
COLLECTION_FORMAT_VERSION = 2
CANONICAL_VIDEO_SUFFIX = ".rgb.mkv.bin"
PRODUCER_PREFIX = "rlab:"
RUNTIME_COLUMNS = frozenset({"_rlab_dataset_root", "_rlab_hf_repo_id"})
MAX_INFO_BYTES = 1 << 20


@dataclass(frozen=True)
class DatasetField:
    name: str
    cast: str | None = None


COMMON_FIELDS = (
    DatasetField("episode_id", "string"),
    DatasetField("step_index", "int64"),
    DatasetField("seed", "int64"),
    DatasetField("actions"),
    DatasetField("policy_actions"),
    DatasetField("rewards", "float64"),
    DatasetField("terminations", "bool"),
    DatasetField("truncations", "bool"),
    DatasetField("infos", "string"),
    DatasetField("session_id", "string"),
    DatasetField("dataset_format_version", "int64"),
    DatasetField("collector", "string"),
    DatasetField("gymrec_version", "string"),
    DatasetField("storage_format", "string"),
    DatasetField("provider_id", "string"),
    DatasetField("env_id", "string"),
    DatasetField("environment_contract_id", "string"),
    DatasetField("collector_contract_id", "string"),
    DatasetField("policy_mode", "string"),
    DatasetField("policy_seed", "int64"),
    DatasetField("collector_terminated", "bool"),
)
IMAGE_FIELDS = (DatasetField("observations", "image"),)
VIDEO_FIELDS = (
    DatasetField("video_path", "string"),
    DatasetField("frame_sha256", "string"),
    DatasetField("frame_width", "int64"),
    DatasetField("frame_height", "int64"),
)
TRANSITION_FIELDS = (
    "actions",
    "policy_actions",
    "rewards",
    "terminations",
    "truncations",
    "infos",
)
REQUIRED_TRANSITION_FIELDS = (
    "actions",
    "rewards",
    "terminations",
    "truncations",
    "infos",
)
STORAGE_FIELDS = frozenset(field.name for field in (*IMAGE_FIELDS, *VIDEO_FIELDS))
ROW_CONTEXT_FIELDS = tuple(
    field.name
    for field in COMMON_FIELDS
    if field.name not in {*TRANSITION_FIELDS, "step_index", "collector_terminated"}
)


@dataclass(frozen=True)
class DatasetSummary:
    storage_format: str
    rows: int
    episodes: int
    transitions: int
    environment_contracts: tuple[str, ...]
    collector_contracts: tuple[str, ...]


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sha256_rgb(frame: Any) -> str:
    return hashlib.sha256(observation_to_rgb(frame).tobytes()).hexdigest()


def observation_to_rgb(observation: Any) -> np.ndarray:
    if isinstance(observation, Mapping):
        for key in ("obs", "image", "screen"):
            if key in observation:
                observation = observation[key]
                break
    if hasattr(observation, "convert"):
        observation = np.asarray(observation.convert("RGB"))
    frame = np.asarray(observation, dtype=np.uint8)
    if frame.ndim == 2:
        frame = np.repeat(frame[..., None], 3, axis=2)
    elif frame.ndim == 3 and frame.shape[2] == 1:
        frame = np.repeat(frame, 3, axis=2)
    elif frame.ndim == 3 and frame.shape[2] >= 3:
        frame = frame[..., :3]
    else:
        raise ValueError(f"unsupported RGB observation shape {frame.shape}")
    return np.ascontiguousarray(frame)


def normalize_storage_format(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized not in STORAGE_FORMATS:
        raise ValueError(
            f"unknown storage format {value!r}; expected one of {', '.join(STORAGE_FORMATS)}"
        )
    return normalized


def dataset_fields(storage_format: str) -> tuple[DatasetField, ...]:
    storage = normalize_storage_format(storage_format)
    return (*COMMON_FIELDS, *(IMAGE_FIELDS if storage == STORAGE_FORMAT_IMAGES else VIDEO_FIELDS))


def canonical_column_order(storage_format: str) -> tuple[str, ...]:
    return tuple(field.name for field in dataset_fields(storage_format))


def _column_names(dataset: Any) -> tuple[str, ...]:
    names = getattr(dataset, "column_names", None)
    if names is not None:
        return tuple(name for name in names if name not in RUNTIME_COLUMNS)
    if isinstance(dataset, Sequence) and dataset:
        return tuple(key for key in dataset[0] if key not in RUNTIME_COLUMNS)
    raise ValueError("dataset has no rows or column metadata")


def _rows(dataset: Any) -> list[Mapping[str, Any]]:
    return [dataset[index] for index in range(len(dataset))]


def _canonical_uuid(value: Any, *, field: str) -> str:
    try:
        canonical = str(uuid.UUID(str(value)))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"invalid {field} {value!r}") from exc
    if value != canonical:
        raise ValueError(f"non-canonical {field} {value!r}; expected {canonical!r}")
    return canonical


def _hex_digest(value: Any, *, field: str) -> str:
    text = str(value or "")
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"invalid {field} {value!r}")
    return text


def _episode_rows(rows: Sequence[Mapping[str, Any]]) -> OrderedDict[str, list[Mapping[str, Any]]]:
    episodes: OrderedDict[str, list[Mapping[str, Any]]] = OrderedDict()
    for row in rows:
        episode_id = _canonical_uuid(row.get("episode_id"), field="episode_id")
        episodes.setdefault(episode_id, []).append(row)
    for episode_id, episode in episodes.items():
        episode.sort(key=lambda row: int(row["step_index"]))
        expected = list(range(len(episode)))
        actual = [int(row["step_index"]) for row in episode]
        if actual != expected:
            raise ValueError(
                f"episode {episode_id} has step_index values {actual}; expected {expected}"
            )
    return episodes


def _validate_info(value: Any, *, episode_id: str, step: int) -> None:
    if not isinstance(value, str):
        raise ValueError(f"episode {episode_id} step {step} infos must be a JSON string")
    if len(value.encode("utf-8")) > MAX_INFO_BYTES:
        raise ValueError(f"episode {episode_id} step {step} infos exceeds {MAX_INFO_BYTES} bytes")
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"episode {episode_id} step {step} infos is invalid JSON") from exc
    if not isinstance(decoded, Mapping):
        raise ValueError(f"episode {episode_id} step {step} infos must encode an object")


def validate_v3(dataset: Any, *, label: str = "dataset") -> DatasetSummary:
    rows = _rows(dataset)
    if not rows:
        raise ValueError(f"{label} is empty")
    storage_values = {normalize_storage_format(row.get("storage_format")) for row in rows}
    if len(storage_values) != 1:
        raise ValueError(f"{label} contains mixed storage formats")
    storage_format = storage_values.pop()
    expected_columns = canonical_column_order(storage_format)
    actual_columns = _column_names(dataset)
    if actual_columns != expected_columns:
        missing = sorted(set(expected_columns) - set(actual_columns))
        extra = sorted(set(actual_columns) - set(expected_columns))
        detail = []
        if missing:
            detail.append("missing " + ", ".join(missing))
        if extra:
            detail.append("unsupported " + ", ".join(extra))
        if not detail:
            detail.append("columns are out of canonical order")
        raise ValueError(f"{label} is not canonical Gymrec v3 ({'; '.join(detail)})")

    versions = {row.get("dataset_format_version") for row in rows}
    if versions != {DATASET_FORMAT_VERSION}:
        raise ValueError(f"{label} uses dataset versions {sorted(versions, key=str)!r}")
    for row in rows:
        _canonical_uuid(row.get("session_id"), field="session_id")
        _hex_digest(row.get("environment_contract_id"), field="environment_contract_id")
        collector_id = row.get("collector_contract_id")
        if collector_id is not None:
            _hex_digest(collector_id, field="collector_contract_id")
        producer = row.get("gymrec_version")
        if not isinstance(producer, str) or not producer:
            raise ValueError(f"{label} has an invalid gymrec_version")

    episodes = _episode_rows(rows)
    transitions = 0
    for episode_id, episode in episodes.items():
        if len(episode) < 2:
            raise ValueError(f"{label} episode {episode_id} has no transitions")
        context = {field: episode[0].get(field) for field in ROW_CONTEXT_FIELDS}
        for row in episode[1:]:
            changed = [field for field, value in context.items() if row.get(field) != value]
            if changed:
                raise ValueError(
                    f"{label} episode {episode_id} changes row context: {', '.join(changed)}"
                )
        for step, row in enumerate(episode[:-1]):
            if row.get("actions") is None:
                raise ValueError(f"{label} episode {episode_id} has an early terminal row")
            missing = [field for field in REQUIRED_TRANSITION_FIELDS if row.get(field) is None]
            if missing:
                raise ValueError(
                    f"{label} episode {episode_id} step {step} has null fields: "
                    + ", ".join(missing)
                )
            if row.get("collector_terminated") is not False:
                raise ValueError(
                    f"{label} episode {episode_id} step {step} must not be collector-terminated"
                )
            if not isinstance(row.get("terminations"), (bool, np.bool_)) or not isinstance(
                row.get("truncations"), (bool, np.bool_)
            ):
                raise ValueError(f"{label} episode {episode_id} step {step} has invalid done flags")
            _validate_info(row.get("infos"), episode_id=episode_id, step=step)
            transitions += 1

        terminal = episode[-1]
        non_null = [field for field in TRANSITION_FIELDS if terminal.get(field) is not None]
        if non_null:
            raise ValueError(
                f"{label} episode {episode_id} final row has transition values: "
                + ", ".join(non_null)
            )
        if not isinstance(terminal.get("collector_terminated"), (bool, np.bool_)):
            raise ValueError(
                f"{label} episode {episode_id} final row must declare collector_terminated"
            )
        previous = episode[-2]
        provider_ended = bool(previous["terminations"]) or bool(previous["truncations"])
        if bool(terminal["collector_terminated"]) == provider_ended:
            expected = "false" if provider_ended else "true"
            raise ValueError(
                f"{label} episode {episode_id} final collector_terminated must be {expected}"
            )

    return DatasetSummary(
        storage_format=storage_format,
        rows=len(rows),
        episodes=len(episodes),
        transitions=transitions,
        environment_contracts=tuple(sorted({str(row["environment_contract_id"]) for row in rows})),
        collector_contracts=tuple(
            sorted(
                {
                    str(row["collector_contract_id"])
                    for row in rows
                    if row.get("collector_contract_id") is not None
                }
            )
        ),
    )


def _read_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read {label} at {path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def validate_environment_document(
    document: Mapping[str, Any], *, expected_id: str | None = None
) -> str:
    expected_fields = {
        "document_type",
        "format_version",
        "provider_id",
        "provider_contract_version",
        "environment_id",
        "declared_config",
        "effective_config",
        "provenance",
        "action_space",
        "observation_space",
        "control_profile",
        "fps",
    }
    if set(document) != expected_fields:
        raise ValueError("environment document has a noncanonical schema")
    if document.get("document_type") != ENVIRONMENT_DOCUMENT_TYPE:
        raise ValueError("unsupported environment document type")
    if document.get("format_version") != ENVIRONMENT_DOCUMENT_FORMAT_VERSION:
        raise ValueError("unsupported environment document format")
    for field in ("declared_config", "effective_config", "action_space", "observation_space"):
        if not isinstance(document[field], Mapping):
            raise ValueError(f"environment document {field} must be an object")
    provenance = document["provenance"]
    if not isinstance(provenance, Mapping) or provenance.get("distribution") != document.get(
        "provider_id"
    ):
        raise ValueError("environment document has invalid provider provenance")
    if not isinstance(provenance.get("version"), str) or not provenance["version"]:
        raise ValueError("environment document has no provider version")
    if not isinstance(provenance.get("assets"), Mapping):
        raise ValueError("environment document has no asset provenance")
    fps = document["fps"]
    if (
        isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or not math.isfinite(fps)
        or fps <= 0
    ):
        raise ValueError("environment document fps must be positive and finite")
    actual = hashlib.sha256(canonical_json_bytes(document)).hexdigest()
    if expected_id is not None and actual != expected_id:
        raise ValueError(f"environment contract hash mismatch: {actual} != {expected_id}")
    return actual


def _safe_binding_filename(value: Any) -> str:
    if not isinstance(value, str) or not value or Path(value).name != value:
        raise ValueError(f"invalid collector artifact filename {value!r}")
    return value


def validate_contract_artifacts(root: Path, summary: DatasetSummary) -> dict[str, bytes]:
    documents: dict[str, bytes] = {}
    for contract_id in summary.environment_contracts:
        path = root / ENVIRONMENT_ARTIFACT_DIR / contract_id / ENVIRONMENT_DOCUMENT_FILENAME
        document = _read_json(path, label=f"environment {contract_id}")
        validate_environment_document(document, expected_id=contract_id)
        payload = canonical_json_bytes(document)
        documents[f"environment:{contract_id}"] = payload

    for contract_id in summary.collector_contracts:
        directory = root / COLLECTOR_ARTIFACT_DIR / contract_id
        collection_path = directory / "collection.json"
        document = _read_json(collection_path, label=f"collector {contract_id}")
        if (
            document.get("document_type") != COLLECTION_DOCUMENT_TYPE
            or document.get("format_version") != COLLECTION_FORMAT_VERSION
        ):
            raise ValueError(f"collector {contract_id} has an unsupported document")
        actual = hashlib.sha256(canonical_json_bytes(document)).hexdigest()
        if actual != contract_id:
            raise ValueError(f"collector contract hash mismatch: {actual} != {contract_id}")
        source = document.get("source")
        if not isinstance(source, Mapping):
            raise ValueError(f"collector {contract_id} has no source manifest")
        expected_names = {"collection.json"}
        for key in ("model", "recipe", "release_manifest"):
            binding = source.get(key)
            if binding is None and key == "release_manifest":
                continue
            if not isinstance(binding, Mapping):
                raise ValueError(f"collector {contract_id} has invalid source.{key}")
            filename = _safe_binding_filename(binding.get("filename"))
            expected_names.add(filename)
            artifact_path = directory / filename
            if sha256_file(artifact_path) != binding.get("sha256"):
                raise ValueError(f"collector {contract_id}/{filename} hash mismatch")
            documents[f"collector:{contract_id}:{filename}"] = artifact_path.read_bytes()
        actual_names = {path.name for path in directory.iterdir() if path.is_file()}
        if actual_names != expected_names:
            raise ValueError(f"collector {contract_id} contains unexpected files")
        documents[f"collector:{contract_id}:collection.json"] = canonical_json_bytes(document)
    return documents


def public_features_document(features: Any) -> Mapping[str, Any]:
    to_dict = getattr(features, "to_dict", None)
    document = to_dict() if callable(to_dict) else features
    if not isinstance(document, Mapping):
        raise ValueError("dataset Features has no public mapping serialization")
    canonical_json_bytes(document)
    return document


def feature_identity(features: Any) -> str:
    payload = canonical_json_bytes(
        {
            "domain": "rlab.dataset.features",
            "version": 1,
            "features": public_features_document(features),
        }
    )
    return hashlib.sha256(payload).hexdigest()


def _is_null_feature(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and value.get("_type") == "Value"
        and value.get("dtype") == "null"
    )


def features_append_compatible(existing: Any, incoming: Any) -> tuple[bool, bool]:
    left = dict(public_features_document(existing))
    right = dict(public_features_document(incoming))
    if left == right:
        return True, False
    left_policy = left.pop("policy_actions", None)
    right_policy = right.pop("policy_actions", None)
    promotion = (
        left == right and _is_null_feature(left_policy) and not _is_null_feature(right_policy)
    )
    nullable_append = (
        left == right and not _is_null_feature(left_policy) and _is_null_feature(right_policy)
    )
    return promotion or nullable_append, promotion


def _typed_hash(hasher: Any, value: Any) -> None:
    if value is None:
        hasher.update(b"N")
    elif isinstance(value, (bool, np.bool_)):
        hasher.update(b"B1" if bool(value) else b"B0")
    elif isinstance(value, (int, np.integer)):
        encoded = str(int(value)).encode("ascii")
        hasher.update(b"I" + len(encoded).to_bytes(8, "big") + encoded)
    elif isinstance(value, (float, np.floating)):
        hasher.update(b"F" + struct.pack(">d", float(value)))
    elif isinstance(value, str):
        encoded = value.encode("utf-8")
        hasher.update(b"S" + len(encoded).to_bytes(8, "big") + encoded)
    elif isinstance(value, (bytes, bytearray, memoryview)):
        encoded = bytes(value)
        hasher.update(b"Y" + len(encoded).to_bytes(8, "big") + encoded)
    elif isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        _typed_hash(hasher, {"dtype": str(contiguous.dtype), "shape": list(contiguous.shape)})
        _typed_hash(hasher, contiguous.tobytes())
    elif isinstance(value, Mapping):
        hasher.update(b"{")
        for key in sorted(value, key=str):
            _typed_hash(hasher, str(key))
            _typed_hash(hasher, value[key])
        hasher.update(b"}")
    elif isinstance(value, Sequence):
        hasher.update(b"[")
        for item in value:
            _typed_hash(hasher, item)
        hasher.update(b"]")
    else:
        try:
            _typed_hash(hasher, observation_to_rgb(value))
        except (TypeError, ValueError) as exc:
            raise TypeError(f"unsupported fingerprint value {type(value).__name__}") from exc


def episode_content_fingerprint(
    rows: Sequence[Mapping[str, Any]],
    *,
    frame_loader: Callable[[Mapping[str, Any]], Any],
    contract_documents: Mapping[str, bytes],
) -> str:
    hasher = hashlib.sha256()
    hasher.update(b"rlab.dataset.episode\x00v1\x00")
    for row in rows:
        logical = {
            key: value
            for key, value in row.items()
            if key not in STORAGE_FIELDS and key not in RUNTIME_COLUMNS
        }
        _typed_hash(hasher, logical)
        frame = observation_to_rgb(frame_loader(row))
        _typed_hash(
            hasher,
            {
                "width": int(frame.shape[1]),
                "height": int(frame.shape[0]),
                "rgb_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
            },
        )
    for name in sorted(contract_documents):
        _typed_hash(hasher, name)
        _typed_hash(hasher, contract_documents[name])
    return hasher.hexdigest()


def collection_fingerprint(features: Any, episode_fingerprints: Iterable[str]) -> str:
    hasher = hashlib.sha256()
    hasher.update(b"rlab.dataset.collection\x00v1\x00")
    _typed_hash(hasher, public_features_document(features))
    for fingerprint in episode_fingerprints:
        _typed_hash(hasher, fingerprint)
    return hasher.hexdigest()


def grouped_episode_rows(dataset: Any) -> OrderedDict[str, list[Mapping[str, Any]]]:
    return _episode_rows(_rows(dataset))
