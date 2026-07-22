from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Callable

from rlab.file_utils import file_sha256 as sha256_file


RECIPE_DOCUMENT_TYPE = "rlab.recipe"
RECIPE_FORMAT_VERSION = 1
MODEL_DOCUMENT_TYPE = "rlab.model"
MODEL_FORMAT_VERSION = 1

RECIPE_FILENAME = "recipe.json"
MODEL_FILENAME = "model.json"
CHECKPOINT_FILENAME = "model.zip"

_RECIPE_FIELDS = frozenset({"document_type", "format_version", "recipe", "provenance"})
_MODEL_FIELDS = frozenset(
    {"document_type", "format_version", "checkpoint", "recipe", "policy", "provenance"}
)
_CHECKPOINT_FIELDS = frozenset(
    {"filename", "sha256", "size_bytes", "kind", "step", "algorithm_id", "model_class"}
)
_RECIPE_BINDING_FIELDS = frozenset(
    {"filename", "document_type", "format_version", "sha256", "size_bytes"}
)
_POLICY_FIELDS = frozenset(
    {
        "algorithm_id",
        "model_class",
        "training_backend_id",
        "training_backend_config_hash",
    }
)
_RECIPE_VALUE_FIELDS = frozenset(
    {
        "schema_version",
        "goal",
        "recipe_id",
        "description",
        "tags",
        "campaign_id",
        "seeds",
        "recipe_overrides",
        "train",
        "train_config",
        "environment",
        "environment_hash",
        "eval",
    }
)
_EVAL_FIELDS = frozenset(
    {
        "environment",
        "action_sampling",
        "episodes",
        "n_envs",
        "max_steps",
        "seed",
        "seed_protocol",
        "protocol_version",
        "deterministic",
        "acceptance",
        "manifest",
        "evidence_policy",
        "asset",
    }
)
_RECIPE_PROVENANCE_FIELDS = frozenset({"source_commit", "source_files", "runtime", "asset"})
_MODEL_PROVENANCE_FIELDS = frozenset(
    {
        "metadata_version",
        "kind",
        "run_name",
        "run_description",
        "wandb_run_id",
        "wandb_project",
        "wandb_run_path",
        "batch_id",
        "campaign_id",
        "game_family",
        "retry_of_job_id",
        "goal_slug",
        "goal_sha256",
        "goal_contract_sha256",
        "effective_goal_contract_sha256",
        "reward_program_kind",
        "reward_program_revision",
        "reward_shape",
        "reward_shape_sha256",
        "reward_shape_is_default",
        "recipe_slug",
        "recipe_sha256",
        "queue_train_job_id",
        "runtime_image_ref",
        "seed",
        "repo_git_commit",
        "training_metadata",
        "training_metadata_hash",
    }
)
_SECRET_FRAGMENTS = (
    "api_key",
    "access_key",
    "secret",
    "token",
    "password",
    "credential",
    "database_url",
)
_OPERATIONAL_TRAIN_FIELDS = frozenset(
    {
        "batch_id",
        "campaign_id",
        "game_family",
        "goal_path",
        "machine",
        "queue_train_job_id",
        "recipe_composition",
        "recipe_json_path",
        "recipe_path",
        "retry_of_job_id",
        "run_description",
        "run_name",
        "runs_dir",
        "runtime_build_source_sha",
        "runtime_image_ref",
        "runtime_input_sha256",
        "source_sha",
        "telemetry_transport",
        "train_config_json",
        "wandb",
        "wandb_artifact_storage_uri",
        "wandb_entity",
        "wandb_group",
        "wandb_mode",
        "wandb_project",
        "wandb_run_id",
        "wandb_tags",
        "no_wandb_artifacts",
    }
)
_RUNTIME_PACKAGES = (
    "rlab",
    "stable-baselines3",
    "stable-retro-turbo",
    "supermariobrosnes-turbo",
)


class PolicyDocumentError(ValueError):
    """Base error for a policy document that cannot be interpreted safely."""


class UnsupportedPolicyDocumentVersion(PolicyDocumentError):
    def __init__(
        self,
        *,
        source: str,
        document_type: object,
        format_version: object,
        supported_versions: Sequence[int],
    ) -> None:
        supported = ", ".join(str(item) for item in supported_versions) or "none"
        super().__init__(
            f"Unsupported {document_type!r} format_version {format_version!r} in {source}. "
            f"This rlab version supports: [{supported}]. Upgrade rlab or install an "
            "explicit compatibility handler."
        )


@dataclass(frozen=True)
class PolicyBundle:
    checkpoint_path: Path
    model_path: Path
    recipe_path: Path
    model: dict[str, Any]
    recipe: dict[str, Any]
    source: str
    revision: str | None = None

    @property
    def checkpoint_sha256(self) -> str:
        return str(self.model["checkpoint"]["sha256"])

    @property
    def recipe_sha256(self) -> str:
        return str(self.model["recipe"]["sha256"])


def canonical_json_bytes(value: object) -> bytes:
    _assert_finite_json(value)
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise PolicyDocumentError(f"document is not canonical JSON: {exc}") from exc
    return (text + "\n").encode("utf-8")


def canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def write_canonical_json(path: Path, value: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(dict(value)))
    return path


def _assert_finite_json(value: object, *, label: str = "document") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise PolicyDocumentError(f"{label} contains a non-finite number")
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _assert_finite_json(nested, label=f"{label}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
        for index, nested in enumerate(value):
            _assert_finite_json(nested, label=f"{label}[{index}]")


def _required_mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PolicyDocumentError(f"{label} must be an object")
    return value


def _required_text(value: object, *, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise PolicyDocumentError(f"{label} must be a non-empty string")
    return text


def _required_sha256(value: object, *, label: str) -> str:
    text = str(value or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", text):
        raise PolicyDocumentError(f"{label} must be a SHA-256 hex digest")
    return text


def _required_size(value: object, *, label: str) -> int:
    if isinstance(value, bool):
        raise PolicyDocumentError(f"{label} must be a positive integer")
    try:
        size = int(value)
    except (TypeError, ValueError) as exc:
        raise PolicyDocumentError(f"{label} must be a positive integer") from exc
    if size <= 0:
        raise PolicyDocumentError(f"{label} must be a positive integer")
    return size


def _reject_unknown(mapping: Mapping[str, Any], allowed: frozenset[str], *, label: str) -> None:
    unknown = sorted(str(key) for key in mapping if key not in allowed)
    if unknown:
        raise PolicyDocumentError(f"{label} has unknown field(s): {', '.join(unknown)}")


def _assert_portable(value: object, *, label: str = "recipe") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key).lower()
            nested_label = f"{label}.{key}"
            if any(fragment in key_text for fragment in _SECRET_FRAGMENTS):
                raise PolicyDocumentError(f"{nested_label} is secret-like and is not portable")
            if key == "defaults":
                raise PolicyDocumentError(f"{nested_label} contains uncomposed defaults")
            _assert_portable(nested, label=nested_label)
        return
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        for index, nested in enumerate(value):
            _assert_portable(nested, label=f"{label}[{index}]")
        return
    if not isinstance(value, str) or not value:
        return
    if "${" in value or re.search(r"\{[A-Za-z_][A-Za-z0-9_]*\}", value):
        raise PolicyDocumentError(f"{label} contains unresolved interpolation: {value!r}")
    if value.startswith(("file://", "s3://", "r2://")):
        raise PolicyDocumentError(f"{label} contains a private or local URI")
    if "://" in value:
        raise PolicyDocumentError(f"{label} contains a URL, which is not portable")
    if Path(value).is_absolute() or PureWindowsPath(value).is_absolute():
        raise PolicyDocumentError(f"{label} contains an absolute local path")


def preflight_document(
    value: object,
    *,
    source: str,
    expected_type: str,
    handlers: Mapping[int, Callable[[Mapping[str, Any], str], dict[str, Any]]],
) -> dict[str, Any]:
    supported = sorted(handlers)
    document = _required_mapping(value, label=source)
    document_type = document.get("document_type")
    format_version = document.get("format_version")
    if document_type != expected_type:
        raise PolicyDocumentError(
            f"{source} document_type must be {expected_type!r}, got {document_type!r}; "
            f"supported versions are {supported}. Upgrade rlab or use an explicit "
            "compatibility handler."
        )
    if isinstance(format_version, bool) or not isinstance(format_version, int):
        raise PolicyDocumentError(
            f"{source} {expected_type!r} format_version must be an integer, got "
            f"{format_version!r}; supported versions are {supported}. Upgrade rlab or "
            "use an explicit compatibility handler."
        )
    handler = handlers.get(format_version)
    if handler is None:
        raise UnsupportedPolicyDocumentVersion(
            source=source,
            document_type=document_type,
            format_version=format_version,
            supported_versions=sorted(handlers),
        )
    try:
        return handler(document, source)
    except UnsupportedPolicyDocumentVersion:
        raise
    except PolicyDocumentError as exc:
        raise PolicyDocumentError(
            f"Invalid {expected_type} format_version {format_version} in {source}; "
            f"supported versions are {supported}: {exc}. Upgrade rlab or use an "
            "explicit compatibility handler."
        ) from exc


def load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PolicyDocumentError(f"Could not read JSON document {path}: {exc}") from exc
    return dict(_required_mapping(value, label=str(path)))


def _validate_recipe_v1(document: Mapping[str, Any], source: str) -> dict[str, Any]:
    _reject_unknown(document, _RECIPE_FIELDS, label=source)
    recipe = _required_mapping(document.get("recipe"), label=f"{source}.recipe")
    provenance = _required_mapping(document.get("provenance"), label=f"{source}.provenance")
    _reject_unknown(recipe, _RECIPE_VALUE_FIELDS, label=f"{source}.recipe")
    _reject_unknown(provenance, _RECIPE_PROVENANCE_FIELDS, label=f"{source}.provenance")
    required_recipe = {"goal", "recipe_id", "description", "train", "train_config", "eval"}
    missing = sorted(required_recipe - set(recipe))
    if missing:
        raise PolicyDocumentError(
            f"{source}.recipe missing required field(s): {', '.join(missing)}"
        )
    goal = _required_mapping(recipe.get("goal"), label=f"{source}.recipe.goal")
    train_config = _required_mapping(
        recipe.get("train_config"), label=f"{source}.recipe.train_config"
    )
    evaluation = _required_mapping(recipe.get("eval"), label=f"{source}.recipe.eval")
    _reject_unknown(evaluation, _EVAL_FIELDS, label=f"{source}.recipe.eval")
    from rlab.env_identity import validate_task_config
    from rlab.goal_schema import validate_goal_document_shape
    from rlab.train_config import (
        TRAIN_CONFIG_FIELDS,
        env_config_allowed_keys,
        validate_and_normalize_train_config,
        validate_train_config_fields,
    )

    _reject_unknown(
        train_config,
        frozenset(field.dest for field in TRAIN_CONFIG_FIELDS),
        label=f"{source}.recipe.train_config",
    )
    try:
        validate_and_normalize_train_config(
            train_config,
            label=f"{source}.recipe.train_config",
        )
        validate_goal_document_shape(goal, label=f"{source}.recipe.goal")
    except ValueError as exc:
        raise PolicyDocumentError(str(exc)) from exc
    for label, environment in (
        ("training", train_config),
        ("evaluation", evaluation.get("environment")),
    ):
        environment = _required_mapping(environment, label=f"{source} {label} environment")
        if label == "evaluation":
            _reject_unknown(
                environment,
                env_config_allowed_keys(),
                label=f"{source} {label} environment",
            )
            try:
                validate_train_config_fields(
                    environment,
                    label=f"{source} {label} environment",
                )
            except ValueError as exc:
                raise PolicyDocumentError(str(exc)) from exc
        _required_text(environment.get("env_provider"), label=f"{source} {label} provider")
        _required_text(environment.get("game"), label=f"{source} {label} game")
        task = _required_mapping(environment.get("task"), label=f"{source} {label} task")
        try:
            validate_task_config(task, label=f"{source} {label} task")
        except ValueError as exc:
            raise PolicyDocumentError(str(exc)) from exc
    _required_mapping(train_config.get("training_backend"), label=f"{source} training backend")
    _required_mapping(goal.get("train"), label=f"{source}.recipe.goal.train")
    if evaluation.get("action_sampling") != "stochastic":
        raise PolicyDocumentError(f"{source}.recipe.eval.action_sampling must be 'stochastic'")
    if evaluation.get("deterministic", False) is not False:
        raise PolicyDocumentError(f"{source}.recipe.eval.deterministic must be false")
    _required_text(evaluation.get("seed_protocol"), label=f"{source} eval seed_protocol")
    if not isinstance(evaluation.get("seed"), int) or isinstance(evaluation.get("seed"), bool):
        raise PolicyDocumentError(f"{source}.recipe.eval.seed must be an integer")
    if not isinstance(evaluation.get("episodes"), int) or int(evaluation["episodes"]) <= 0:
        raise PolicyDocumentError(f"{source}.recipe.eval.episodes must be positive")
    if "manifest" in evaluation:
        from rlab.checkpoint_acceptance import manifest_index

        try:
            manifest_index(evaluation)
        except ValueError as exc:
            raise PolicyDocumentError(f"{source}.recipe.eval manifest is invalid: {exc}") from exc
    runtime = _required_mapping(provenance.get("runtime"), label=f"{source}.provenance.runtime")
    _reject_unknown(runtime, frozenset({"image_ref", "packages"}), label=f"{source}.runtime")
    image_ref = _required_text(runtime.get("image_ref"), label=f"{source} runtime image_ref")
    if not re.fullmatch(r"docker:[^\s]+@sha256:[0-9a-f]{64}", image_ref):
        raise PolicyDocumentError(f"{source} runtime image_ref must be an immutable digest")
    _assert_portable(recipe, label=f"{source}.recipe")
    _assert_portable(provenance, label=f"{source}.provenance")
    _assert_finite_json(document, label=source)
    return deepcopy(dict(document))


def load_recipe_document(path: Path) -> dict[str, Any]:
    value = load_json_object(path)
    return preflight_document(
        value,
        source=str(path),
        expected_type=RECIPE_DOCUMENT_TYPE,
        handlers={RECIPE_FORMAT_VERSION: _validate_recipe_v1},
    )


def validate_recipe_document(
    document: Mapping[str, Any], *, source: str = RECIPE_FILENAME
) -> dict[str, Any]:
    return preflight_document(
        document,
        source=source,
        expected_type=RECIPE_DOCUMENT_TYPE,
        handlers={RECIPE_FORMAT_VERSION: _validate_recipe_v1},
    )


def _validate_model_v1(document: Mapping[str, Any], source: str) -> dict[str, Any]:
    _reject_unknown(document, _MODEL_FIELDS, label=source)
    checkpoint = _required_mapping(document.get("checkpoint"), label=f"{source}.checkpoint")
    recipe = _required_mapping(document.get("recipe"), label=f"{source}.recipe")
    policy = _required_mapping(document.get("policy"), label=f"{source}.policy")
    provenance = _required_mapping(document.get("provenance"), label=f"{source}.provenance")
    _reject_unknown(checkpoint, _CHECKPOINT_FIELDS, label=f"{source}.checkpoint")
    _reject_unknown(recipe, _RECIPE_BINDING_FIELDS, label=f"{source}.recipe")
    _reject_unknown(policy, _POLICY_FIELDS, label=f"{source}.policy")
    _reject_unknown(provenance, _MODEL_PROVENANCE_FIELDS, label=f"{source}.provenance")
    if checkpoint.get("filename") != CHECKPOINT_FILENAME:
        raise PolicyDocumentError(f"{source}.checkpoint.filename must be {CHECKPOINT_FILENAME!r}")
    _required_sha256(checkpoint.get("sha256"), label=f"{source}.checkpoint.sha256")
    _required_size(checkpoint.get("size_bytes"), label=f"{source}.checkpoint.size_bytes")
    if checkpoint.get("kind") not in {"checkpoint", "best", "final"}:
        raise PolicyDocumentError(
            f"{source}.checkpoint.kind must be 'checkpoint', 'best', or 'final'"
        )
    step = checkpoint.get("step")
    if step is not None and (not isinstance(step, int) or isinstance(step, bool) or step < 0):
        raise PolicyDocumentError(f"{source}.checkpoint.step must be a non-negative integer")
    _required_text(checkpoint.get("algorithm_id"), label=f"{source}.checkpoint.algorithm_id")
    _required_text(checkpoint.get("model_class"), label=f"{source}.checkpoint.model_class")
    _required_text(policy.get("algorithm_id"), label=f"{source}.policy.algorithm_id")
    _required_text(policy.get("model_class"), label=f"{source}.policy.model_class")
    _required_text(
        policy.get("training_backend_id"),
        label=f"{source}.policy.training_backend_id",
    )
    _required_sha256(
        policy.get("training_backend_config_hash"),
        label=f"{source}.policy.training_backend_config_hash",
    )
    if recipe.get("filename") != RECIPE_FILENAME:
        raise PolicyDocumentError(f"{source}.recipe.filename must be {RECIPE_FILENAME!r}")
    if recipe.get("document_type") != RECIPE_DOCUMENT_TYPE:
        raise PolicyDocumentError(f"{source}.recipe.document_type must be {RECIPE_DOCUMENT_TYPE!r}")
    if recipe.get("format_version") != RECIPE_FORMAT_VERSION:
        raise PolicyDocumentError(f"{source}.recipe.format_version must be {RECIPE_FORMAT_VERSION}")
    _required_sha256(recipe.get("sha256"), label=f"{source}.recipe.sha256")
    _required_size(recipe.get("size_bytes"), label=f"{source}.recipe.size_bytes")
    _assert_portable(provenance, label=f"{source}.provenance")
    _assert_finite_json(document, label=source)
    return deepcopy(dict(document))


def load_model_document(path: Path) -> dict[str, Any]:
    value = load_json_object(path)
    return preflight_document(
        value,
        source=str(path),
        expected_type=MODEL_DOCUMENT_TYPE,
        handlers={MODEL_FORMAT_VERSION: _validate_model_v1},
    )


def model_document_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_suffix(".model.json")


def recipe_document_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_suffix(".recipe.json")


def _portable_source_path(path: object, *, repo_root: Path) -> str:
    candidate = Path(str(path))
    if not candidate.is_absolute():
        return candidate.as_posix()
    try:
        return candidate.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError as exc:
        raise PolicyDocumentError(
            f"recipe source path is outside the repository: {candidate}"
        ) from exc


def _resolve_recipe_templates(value: object, replacements: Mapping[str, object]) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _resolve_recipe_templates(nested, replacements)
            for key, nested in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [_resolve_recipe_templates(nested, replacements) for nested in value]
    if not isinstance(value, str):
        return value
    rendered = value
    for key, replacement in replacements.items():
        rendered = rendered.replace("{" + key + "}", str(replacement))
    return rendered


def build_recipe_document(
    materialized_recipe: Mapping[str, Any],
    *,
    repo_root: Path,
    source_commit: str,
    run_description: str | None = None,
    seed: int | None = None,
    runtime_image_ref: str | None = None,
) -> dict[str, Any]:
    recipe = deepcopy(dict(materialized_recipe))
    recipe.pop("logging", None)
    composition = recipe.pop("_composition", {})
    train_config = dict(_required_mapping(recipe.get("train_config"), label="train_config"))
    # recipe.json is the immutable policy contract consumed by evaluation. Materialize
    # backend defaults here, before the queue adds operational fields, so its backend
    # hash is identical to the normalized config executed by the learner.
    from rlab.train_config import validate_and_normalize_train_config

    train_config = validate_and_normalize_train_config(
        train_config,
        label="recipe.json train_config",
    )
    # Derive the recipe's environment identity from the same effective EnvConfig
    # path used by the learner. Sparse source config omits defaults such as empty
    # state vectors; hashing that sparse identity makes otherwise identical model
    # metadata fail the cross-document contract before evaluation can start.
    from rlab.env import resolve_env_config
    from rlab.env_config import env_config_from_mapping
    from rlab.env_metadata import training_metadata

    effective_training_metadata = training_metadata(
        resolve_env_config(env_config_from_mapping(train_config)),
        rom_asset_manifest=train_config.get("rom_asset_manifest"),
    )
    recipe["environment"] = effective_training_metadata["environment"]
    recipe["environment_hash"] = effective_training_metadata["environment_hash"]
    from rlab.checkpoint_acceptance import CheckpointEvalContractCompiler

    eval_compiler = CheckpointEvalContractCompiler.from_train_config(
        train_config,
        portable_asset=True,
        require_asset=False,
        materialize_seed_defaults=True,
    )
    portable_asset = eval_compiler.asset
    stop_on_acceptance = bool(train_config.get("stop_on_acceptance"))
    for key in _OPERATIONAL_TRAIN_FIELDS:
        train_config.pop(key, None)
    train_config.pop("checkpoint_eval_asset_manifest", None)
    train_config.pop("rom_asset_manifest", None)
    if seed is not None:
        train_config["seed"] = int(seed)
    recipe["train_config"] = train_config
    if run_description:
        recipe["description"] = str(run_description)
    recipe = dict(
        _resolve_recipe_templates(
            recipe,
            {
                "seed": "" if seed is None else int(seed),
                "recipe_id": recipe.get("recipe_id") or "",
                "env_id": train_config.get("game") or "",
            },
        )
    )
    recipe["schema_version"] = int(recipe.get("schema_version") or 2)
    recipe["eval"] = eval_compiler.contract(require_acceptance=stop_on_acceptance)
    source_files = []
    if isinstance(composition, Mapping):
        for item in composition.get("source_files") or []:
            if not isinstance(item, Mapping):
                continue
            source_files.append(
                {
                    "path": _portable_source_path(item.get("path"), repo_root=repo_root),
                    "sha256": _required_sha256(
                        item.get("sha256"), label="recipe provenance source sha256"
                    ),
                }
            )
    provenance: dict[str, Any] = {
        "source_commit": _required_text(source_commit, label="recipe source_commit"),
        "source_files": source_files,
        "runtime": {"image_ref": str(runtime_image_ref or "")},
        "asset": portable_asset,
    }
    if not re.fullmatch(r"[0-9a-f]{40}", provenance["source_commit"]):
        raise PolicyDocumentError("recipe source_commit must be a full lowercase Git SHA")
    document = {
        "document_type": RECIPE_DOCUMENT_TYPE,
        "format_version": RECIPE_FORMAT_VERSION,
        "recipe": recipe,
        "provenance": provenance,
    }
    return _validate_recipe_v1(document, RECIPE_FILENAME)


def finalize_recipe_runtime(path: Path) -> dict[str, Any]:
    document = load_recipe_document(path)
    runtime = dict(document["provenance"].get("runtime") or {})
    packages: dict[str, str] = {}
    for package in _RUNTIME_PACKAGES:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    runtime["packages"] = packages
    document["provenance"]["runtime"] = runtime
    write_canonical_json(path, document)
    return load_recipe_document(path)


def build_model_document(
    checkpoint_path: Path,
    recipe_path: Path,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    load_recipe_document(recipe_path)
    step = metadata.get("checkpoint_step")
    provenance = {
        key: deepcopy(value)
        for key, value in metadata.items()
        if key in _MODEL_PROVENANCE_FIELDS and value not in (None, "")
    }
    document = {
        "document_type": MODEL_DOCUMENT_TYPE,
        "format_version": MODEL_FORMAT_VERSION,
        "checkpoint": {
            "filename": CHECKPOINT_FILENAME,
            "sha256": sha256_file(checkpoint_path),
            "size_bytes": checkpoint_path.stat().st_size,
            "kind": str(metadata.get("kind") or "checkpoint"),
            "step": int(step) if step is not None else None,
            "algorithm_id": str(metadata.get("algorithm_id") or ""),
            "model_class": str(metadata.get("model_class") or ""),
        },
        "recipe": {
            "filename": RECIPE_FILENAME,
            "document_type": RECIPE_DOCUMENT_TYPE,
            "format_version": RECIPE_FORMAT_VERSION,
            "sha256": sha256_file(recipe_path),
            "size_bytes": recipe_path.stat().st_size,
        },
        "policy": {
            "algorithm_id": str(metadata.get("algorithm_id") or ""),
            "model_class": str(metadata.get("model_class") or ""),
            "training_backend_id": str(metadata.get("training_backend_id") or ""),
            "training_backend_config_hash": str(metadata.get("training_backend_config_hash") or ""),
        },
        "provenance": provenance,
    }
    return _validate_model_v1(document, MODEL_FILENAME)


def model_document_as_metadata(document: Mapping[str, Any]) -> dict[str, Any]:
    validated = preflight_document(
        document,
        source=MODEL_FILENAME,
        expected_type=MODEL_DOCUMENT_TYPE,
        handlers={MODEL_FORMAT_VERSION: _validate_model_v1},
    )
    metadata = deepcopy(dict(validated["provenance"]))
    metadata.update(validated["policy"])
    metadata["checkpoint_step"] = validated["checkpoint"].get("step")
    metadata["kind"] = validated["checkpoint"].get("kind")
    metadata["filename"] = validated["checkpoint"]["filename"]
    metadata["checkpoint_sha256"] = validated["checkpoint"]["sha256"]
    metadata["recipe"] = deepcopy(validated["recipe"])
    return metadata


def _validate_cross_document_contract(model: Mapping[str, Any], recipe: Mapping[str, Any]) -> None:
    checkpoint = model["checkpoint"]
    policy = model["policy"]
    if checkpoint["algorithm_id"] != policy["algorithm_id"]:
        raise PolicyDocumentError("model.json checkpoint and policy algorithm_id disagree")
    if checkpoint["model_class"] != policy["model_class"]:
        raise PolicyDocumentError("model.json checkpoint and policy model_class disagree")
    train_config = recipe["recipe"]["train_config"]
    for key in (
        "goal_contract_sha256",
        "effective_goal_contract_sha256",
        "reward_program_kind",
        "reward_program_revision",
        "reward_shape",
        "reward_shape_sha256",
        "reward_shape_is_default",
    ):
        model_value = model["provenance"].get(key)
        recipe_value = train_config.get(key)
        if model_value not in (None, "") and model_value != recipe_value:
            raise PolicyDocumentError(f"model.json {key} disagrees with recipe.json")
    backend = _required_mapping(
        train_config.get("training_backend"),
        label="recipe.json training backend",
    )
    if str(backend.get("id") or "") != policy["training_backend_id"]:
        raise PolicyDocumentError("model.json training backend disagrees with recipe.json")
    from rlab.training_backend import training_backend_config_hash

    expected_backend_hash = training_backend_config_hash(train_config)
    if expected_backend_hash != policy["training_backend_config_hash"]:
        raise PolicyDocumentError(
            "model.json training backend config hash disagrees with recipe.json"
        )
    training_metadata = model["provenance"].get("training_metadata")
    if not isinstance(training_metadata, Mapping):
        return
    environment = training_metadata.get("environment")
    qualified_env_id = (
        str(environment.get("env_id") or "") if isinstance(environment, Mapping) else ""
    )
    expected_env_id = f"{train_config['env_provider']}:{train_config['game']}"
    if qualified_env_id and qualified_env_id != expected_env_id:
        raise PolicyDocumentError("model.json training environment disagrees with recipe.json")
    recipe_environment = recipe["recipe"].get("environment")
    recipe_environment_hash = recipe["recipe"].get("environment_hash")
    if recipe_environment is not None and environment != recipe_environment:
        raise PolicyDocumentError(
            "model.json normalized training environment disagrees with recipe.json"
        )
    if (
        recipe_environment_hash is not None
        and training_metadata.get("environment_hash") != recipe_environment_hash
    ):
        raise PolicyDocumentError("model.json training environment hash disagrees with recipe.json")


def load_policy_bundle(
    root: Path,
    *,
    source: str | None = None,
    revision: str | None = None,
) -> PolicyBundle:
    checkpoint_path = root / CHECKPOINT_FILENAME
    model_path = root / MODEL_FILENAME
    recipe_path = root / RECIPE_FILENAME
    missing = [
        path.name for path in (checkpoint_path, model_path, recipe_path) if not path.is_file()
    ]
    if missing:
        raise PolicyDocumentError(f"policy bundle {root} is missing: {', '.join(missing)}")
    model = load_model_document(model_path)
    recipe = load_recipe_document(recipe_path)
    checkpoint = model["checkpoint"]
    recipe_binding = model["recipe"]
    if sha256_file(checkpoint_path) != checkpoint["sha256"]:
        raise PolicyDocumentError(f"{root}/model.zip hash does not match model.json")
    if checkpoint_path.stat().st_size != checkpoint["size_bytes"]:
        raise PolicyDocumentError(f"{root}/model.zip size does not match model.json")
    if sha256_file(recipe_path) != recipe_binding["sha256"]:
        raise PolicyDocumentError(f"{root}/recipe.json hash does not match model.json")
    if recipe_path.stat().st_size != recipe_binding["size_bytes"]:
        raise PolicyDocumentError(f"{root}/recipe.json size does not match model.json")
    _validate_cross_document_contract(model, recipe)
    return PolicyBundle(
        checkpoint_path=checkpoint_path,
        model_path=model_path,
        recipe_path=recipe_path,
        model=model,
        recipe=recipe,
        source=source or str(root),
        revision=revision,
    )


def load_policy_bundle_from_checkpoint(
    checkpoint_path: Path,
    *,
    source: str | None = None,
    revision: str | None = None,
) -> PolicyBundle | None:
    if checkpoint_path.name == CHECKPOINT_FILENAME and (
        checkpoint_path.with_name(MODEL_FILENAME).is_file()
        or checkpoint_path.with_name(RECIPE_FILENAME).is_file()
    ):
        return load_policy_bundle(
            checkpoint_path.parent,
            source=source,
            revision=revision,
        )
    model_path = model_document_path(checkpoint_path)
    recipe_path = recipe_document_path(checkpoint_path)
    if not model_path.is_file() and not recipe_path.is_file():
        return None
    missing = [path.name for path in (model_path, recipe_path) if not path.is_file()]
    if missing:
        raise PolicyDocumentError(
            f"versioned policy checkpoint {checkpoint_path} is missing: {', '.join(missing)}"
        )
    model = load_model_document(model_path)
    recipe = load_recipe_document(recipe_path)
    if sha256_file(checkpoint_path) != model["checkpoint"]["sha256"]:
        raise PolicyDocumentError(f"{checkpoint_path} hash does not match {model_path}")
    if checkpoint_path.stat().st_size != model["checkpoint"]["size_bytes"]:
        raise PolicyDocumentError(f"{checkpoint_path} size does not match {model_path}")
    if sha256_file(recipe_path) != model["recipe"]["sha256"]:
        raise PolicyDocumentError(f"{recipe_path} hash does not match {model_path}")
    if recipe_path.stat().st_size != model["recipe"]["size_bytes"]:
        raise PolicyDocumentError(f"{recipe_path} size does not match {model_path}")
    _validate_cross_document_contract(model, recipe)
    return PolicyBundle(
        checkpoint_path=checkpoint_path,
        model_path=model_path,
        recipe_path=recipe_path,
        model=model,
        recipe=recipe,
        source=source or str(checkpoint_path),
        revision=revision,
    )


def evaluation_contract(recipe_document: Mapping[str, Any]) -> dict[str, Any]:
    validated = preflight_document(
        recipe_document,
        source=RECIPE_FILENAME,
        expected_type=RECIPE_DOCUMENT_TYPE,
        handlers={RECIPE_FORMAT_VERSION: _validate_recipe_v1},
    )
    return deepcopy(dict(validated["recipe"]["eval"]))


def evaluation_contract_sha256(recipe_document: Mapping[str, Any]) -> str:
    return canonical_json_sha256(evaluation_contract(recipe_document))
