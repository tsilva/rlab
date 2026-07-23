from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote, urlparse

from rlab.artifacts import apply_config_defaults, load_model_metadata, write_model_metadata
from rlab.env import resolve_env_config
from rlab.env_config import env_config_from_args
from rlab.env_metadata import (
    assert_metadata_runtime_versions,
    env_config_from_metadata,
    sanitize_env_config_metadata,
)
from rlab.file_utils import file_sha256
from rlab.recipe_documents import load_goal_contract_document
from rlab.policy_bundle import (
    CHECKPOINT_FILENAME,
    MODEL_FILENAME,
    RECIPE_FILENAME,
    PolicyBundle,
    load_policy_bundle,
    load_policy_bundle_from_checkpoint,
)
from rlab.wandb_artifacts import (
    checkpoint_step_from_artifact,
    download_model_artifact_with_revision,
    model_artifact_ref,
    safe_artifact_stem,
)
from rlab.wandb_utils import (
    canonical_wandb_environment,
    default_wandb_project_path,
    load_wandb_env,
)


MODEL_KIND_CHOICES = ("final", "best", "checkpoint")
HUGGINGFACE_MODEL_SCHEME = "hf://"
OBJECT_STORE_MODEL_SCHEME = "object-store:"
HUGGINGFACE_MODEL_URL_HOST = "huggingface.co"
WANDB_RUN_URL_HOST = "wandb.ai"
MODEL_ARTIFACT_KIND_SUFFIXES = tuple(f"-{kind}" for kind in MODEL_KIND_CHOICES)
# Prefix entries retain bare-run playback for historical projects. Entries without a
# prefix are fallback-only projects that predate goal-owned W&B project metadata.
ARTIFACT_PROJECT_COMPATIBILITY = (
    ("alepy__breakout", "breakout"),
    ("alepy__mspacman", "ms_pacman"),
    (None, "SuperMarioBros-Nes-v0"),
    (None, "SuperMarioBros3-Nes-v0"),
    (None, "ms_pacman"),
    (None, "breakout"),
)
MAX_PARALLEL_ARTIFACT_LOOKUPS = 4
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass
class ResolvedModelSource:
    model_path: Path
    artifact_ref: str | None = None
    artifact_name: str | None = None
    checkpoint_step: int | None = None
    run_config: dict[str, Any] = field(default_factory=dict)
    bundle: PolicyBundle | None = None


@dataclass(frozen=True)
class WandbRunRef:
    project_path: str
    run_id: str


def _wandb_run_path_parts(value: str) -> list[str] | None:
    text = str(value or "").strip()
    if not text:
        return None

    parsed = urlparse(text)
    if parsed.scheme in {"http", "https"}:
        host = parsed.netloc.lower().removeprefix("www.")
        if host != WANDB_RUN_URL_HOST:
            return None
        path = parsed.path
    else:
        path = text
        if path.lower().startswith(f"{WANDB_RUN_URL_HOST}/"):
            path = path[len(WANDB_RUN_URL_HOST) + 1 :]
        elif path.lower().startswith(f"www.{WANDB_RUN_URL_HOST}/"):
            path = path[len(f"www.{WANDB_RUN_URL_HOST}/") :]
        path = path.split("?", 1)[0].split("#", 1)[0]

    return [unquote(part) for part in path.split("/") if part]


def artifact_ref_arg(value: str) -> str:
    parts = value.split("/")
    artifact_name = parts[-1] if parts else ""
    if len(parts) != 3 or ":" not in artifact_name or artifact_name.startswith(":"):
        raise argparse.ArgumentTypeError(
            "expected W&B artifact ref like entity/project/run-checkpoint:latest"
        )
    return value


def is_huggingface_model_ref(value: str) -> bool:
    text = str(value or "").strip()
    if text.startswith(HUGGINGFACE_MODEL_SCHEME):
        return True
    parsed = urlparse(text)
    return parsed.scheme in {"http", "https"} and parsed.netloc == HUGGINGFACE_MODEL_URL_HOST


def _content_addressed_model_parts(path: str) -> tuple[str, ...]:
    decoded = unquote(path).strip("/")
    pure = PurePosixPath(decoded)
    if not decoded or "\\" in decoded or ".." in pure.parts:
        return ()
    parts = pure.parts
    if (
        len(parts) < 3
        or parts[-3] != "sha256"
        or not _SHA256.fullmatch(parts[-2])
        or parts[-1] != CHECKPOINT_FILENAME
    ):
        return ()
    return parts


def is_s3_model_ref(value: str) -> bool:
    parsed = urlparse(str(value or "").strip())
    return bool(
        parsed.scheme == "s3"
        and parsed.netloc
        and not parsed.params
        and not parsed.query
        and not parsed.fragment
        and _content_addressed_model_parts(parsed.path)
    )


def is_object_store_model_ref(value: str) -> bool:
    text = str(value or "").strip()
    if not text.startswith(OBJECT_STORE_MODEL_SCHEME):
        return False
    path = text.removeprefix(OBJECT_STORE_MODEL_SCHEME)
    return bool(_content_addressed_model_parts(path))


def parse_wandb_run_ref(value: str, *, default_project: str | None = None) -> WandbRunRef | None:
    parts = _wandb_run_path_parts(value)
    if not parts:
        return None

    if len(parts) >= 4 and parts[2] == "runs":
        return WandbRunRef(project_path=f"{parts[0]}/{parts[1]}", run_id=parts[3])
    if default_project and len(parts) >= 2 and parts[0] == "runs":
        return WandbRunRef(project_path=default_project, run_id=parts[1])
    if default_project and len(parts) >= 3 and parts[1] == "runs":
        entity = default_project.split("/", 1)[0]
        return WandbRunRef(project_path=f"{entity}/{parts[0]}", run_id=parts[2])
    return None


def is_wandb_run_ref(value: str) -> bool:
    return parse_wandb_run_ref(value) is not None


def positional_model_source_arg(value: str) -> str:
    if is_huggingface_model_ref(value):
        return value
    if is_wandb_run_ref(value):
        return value
    if "/" not in value and ":" not in value:
        return value
    parts = value.split("/")
    leaf = parts[-1] if parts else ""
    if len(parts) in (2, 3) and ":" not in leaf:
        return value
    return artifact_ref_arg(value)


def positional_huggingface_model_arg(value: str) -> str:
    if is_huggingface_model_ref(value):
        return value
    raise argparse.ArgumentTypeError("expected Hugging Face model ref like hf://owner/repo")


def add_model_source_args(
    parser: argparse.ArgumentParser,
    *,
    positional_artifact: bool = False,
    allow_multiple_artifacts: bool = False,
    model_default: str | None = None,
    model_help: str | None = None,
    default_kind: str = "checkpoint",
    include_wandb_artifacts: bool = True,
) -> None:
    if positional_artifact:
        if include_wandb_artifacts:
            parser.add_argument(
                "artifact_ref",
                nargs="?",
                type=positional_model_source_arg,
                help=(
                    "W&B run name or run URL, or a full artifact ref like "
                    "entity/project/run-checkpoint:latest."
                ),
            )
        else:
            parser.add_argument(
                "model_ref",
                nargs="?",
                type=positional_huggingface_model_arg,
                help="Hugging Face model ref, for example hf://owner/repo.",
            )
    model_kwargs: dict[str, Any] = {}
    if model_default is not None:
        model_kwargs["default"] = model_default
    if model_help is not None:
        model_kwargs["help"] = model_help
    parser.add_argument("--model", **model_kwargs)
    if not include_wandb_artifacts:
        parser.add_argument(
            "--hf-file",
            help=(
                "Checkpoint filename to download from a Hugging Face model repo. "
                "Required only when the repo contains multiple .zip checkpoints."
            ),
        )
        parser.add_argument("--hf-revision", help="Hugging Face model revision. Defaults to main.")
        parser.add_argument("--hf-model-root", default="runs/hf_models")
        return
    artifact_kwargs: dict[str, Any] = {
        "type": artifact_ref_arg,
        "help": "Full W&B model artifact ref, for example entity/project/run-checkpoint:latest.",
    }
    if allow_multiple_artifacts:
        artifact_kwargs["action"] = "append"
        artifact_kwargs["help"] = (
            "Full W&B model artifact ref to evaluate. May be passed more than once."
        )
    parser.add_argument("--artifact", **artifact_kwargs)
    parser.add_argument(
        "--artifact-run",
        help="Training run name used to build a W&B artifact ref with --artifact-kind/version.",
    )
    parser.add_argument("--artifact-project", default=default_wandb_project_path())
    parser.add_argument("--artifact-kind", choices=MODEL_KIND_CHOICES, default=default_kind)
    parser.add_argument("--artifact-version", default="latest")
    parser.add_argument("--artifact-root", default="runs/wandb_artifacts")
    parser.add_argument(
        "--hf-file",
        help=(
            "Checkpoint filename to download from a Hugging Face model repo. "
            "Required only when the repo contains multiple .zip checkpoints."
        ),
    )
    parser.add_argument("--hf-revision", help="Hugging Face model revision. Defaults to main.")
    parser.add_argument("--hf-model-root", default="runs/hf_models")


def artifact_values(args: argparse.Namespace) -> tuple[str, ...]:
    value = getattr(args, "artifact", None)
    if not value:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value if item)
    return (str(value),)


def artifact_project_from_args(args: argparse.Namespace) -> str:
    return str(getattr(args, "artifact_project", "") or default_wandb_project_path())


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _project_path(entity: str, project: str) -> str:
    return project if "/" in project else f"{entity}/{project}"


def _goal_project_from_document(goal_path: Path) -> str | None:
    try:
        data = load_goal_contract_document(goal_path)
    except Exception:
        return None
    if not isinstance(data, Mapping):
        return None
    explicit_project = _mapping_value(data, "wandb_project")
    if explicit_project:
        return str(explicit_project)
    logging = _mapping_value(data, "logging")
    if isinstance(logging, Mapping) and logging.get("wandb_project"):
        return str(logging["wandb_project"])
    train = _mapping_value(data, "train")
    if not isinstance(train, Mapping):
        return None
    environment = _mapping_value(train, "environment")
    if not isinstance(environment, Mapping):
        return None
    env_config = _mapping_value(environment, "env_config")
    if not isinstance(env_config, Mapping):
        return None
    env_args = _mapping_value(env_config, "env_args")
    game = (
        str(env_args["game"])
        if isinstance(env_args, Mapping) and env_args.get("game")
        else str(env_config.get("game") or "")
    )
    if game:
        provider = environment.get("env_provider") or env_config.get("env_provider")
        return canonical_wandb_environment(provider, game)[0]
    return None


def _local_goal_project_map(goals_root: Path = Path("experiments/goals")) -> dict[str, str]:
    if not goals_root.exists():
        return {}
    projects: dict[str, str] = {}
    for goal_file in sorted(goals_root.rglob("_goal.yaml")):
        goal_id = goal_file.parent.name
        projects[goal_id] = _goal_project_from_document(goal_file) or goal_id
    return projects


def artifact_lookup_project_paths(default_project: str, run_name: str) -> list[str]:
    entity = default_project.split("/", 1)[0]
    local_projects = _local_goal_project_map()
    inferred: list[str] = []
    for prefix, project in ARTIFACT_PROJECT_COMPATIBILITY:
        if prefix and (run_name == prefix or run_name.startswith(f"{prefix}_")):
            inferred.append(_project_path(entity, project))
            break
    for goal_id, project in sorted(
        local_projects.items(), key=lambda item: len(item[0]), reverse=True
    ):
        if run_name == goal_id or run_name.startswith(f"{goal_id}_"):
            inferred.append(_project_path(entity, project))
            break
    projects = [
        *inferred,
        default_project,
        *(_project_path(entity, project) for project in local_projects.values()),
        *(
            _project_path(entity, project)
            for prefix, project in ARTIFACT_PROJECT_COMPATIBILITY
            if not prefix
        ),
    ]
    return _dedupe(projects)


def _artifact_exists(ref: str) -> bool:
    import wandb

    try:
        wandb.Api().artifact(ref, type="model")
    except Exception:
        return False
    return True


_RUN_PLAYABLE_KINDS = ("checkpoint", "final", "interrupted")
_RUN_KIND_TIE_PRIORITY = {"checkpoint": 0, "interrupted": 1, "final": 2}


def _artifact_alias_set(artifact: Any) -> set[str]:
    return {
        str(getattr(alias, "alias", alias)) for alias in (getattr(artifact, "aliases", ()) or ())
    }


def _artifact_numeric_version(artifact: Any) -> int:
    version = str(getattr(artifact, "version", "") or "")
    qualified = str(getattr(artifact, "qualified_name", "") or "")
    if not version and ":" in qualified:
        version = qualified.rsplit(":", 1)[-1]
    return int(version[1:]) if version.startswith("v") and version[1:].isdigit() else -1


def _artifact_concrete_ref(artifact: Any) -> str | None:
    qualified = str(getattr(artifact, "qualified_name", "") or "")
    version = _artifact_numeric_version(artifact)
    if qualified and version >= 0:
        return f"{qualified.rsplit(':', 1)[0]}:v{version}"
    name = str(getattr(artifact, "name", "") or "")
    return f"{name}:v{version}" if name and version >= 0 else None


def _artifact_collection_name(artifact: Any) -> str:
    concrete = _artifact_concrete_ref(artifact)
    if concrete:
        return concrete.rsplit("/", 1)[-1].rsplit(":", 1)[0]
    return str(getattr(artifact, "name", "") or "").rsplit("/", 1)[-1]


def _artifact_kind_from_collection(collection: str) -> str | None:
    for kind in _RUN_PLAYABLE_KINDS:
        if collection.endswith(f"-{kind}"):
            return kind
    return None


def _materialized_logged_model_artifacts(run: Any) -> list[Any]:
    return [
        artifact
        for artifact in list(run.logged_artifacts())
        if str(getattr(artifact, "type", "") or "") == "model"
    ]


def _preferred_run_artifact_ref(run: Any) -> str | None:
    """Return the best currently API-visible run artifact as an immutable version."""

    artifacts = _materialized_logged_model_artifacts(run)
    run_id = str(
        _wandb_run_config_value(run, "wandb_run_id") or getattr(run, "id", "") or ""
    ).strip()
    canonical_names = {f"{safe_artifact_stem(run_id)}-{kind}" for kind in _RUN_PLAYABLE_KINDS}
    canonical = [
        artifact for artifact in artifacts if _artifact_collection_name(artifact) in canonical_names
    ]
    candidates = canonical or [
        artifact
        for artifact in artifacts
        if _artifact_kind_from_collection(_artifact_collection_name(artifact)) is not None
    ]
    candidates = [artifact for artifact in candidates if _artifact_concrete_ref(artifact)]
    if not candidates:
        return None

    promoted = []
    for artifact in candidates:
        metadata = dict(getattr(artifact, "metadata", {}) or {})
        if (
            str(metadata.get("artifact_publication_schema") or "") in {"v2", "v3"}
            and str(metadata.get("publication_role") or "") == "promotion"
            and int(metadata.get("promotion_revision") or 0) >= 1
        ):
            promoted.append(artifact)
    if promoted:
        selected = max(
            promoted,
            key=lambda artifact: (
                int((getattr(artifact, "metadata", {}) or {}).get("promotion_revision") or 0),
                _artifact_numeric_version(artifact),
            ),
        )
        return _artifact_concrete_ref(selected)

    legacy_promoted = [
        artifact for artifact in candidates if "promoted" in _artifact_alias_set(artifact)
    ]
    if legacy_promoted:
        selected = max(legacy_promoted, key=_artifact_numeric_version)
        return _artifact_concrete_ref(selected)

    summary = getattr(run, "summary", {}) or {}
    getter = getattr(summary, "get", None)
    leader_uri = str(getter("leader/checkpoint/artifact_ref") or "") if callable(getter) else ""
    raw_step = getter("leader/checkpoint/step") if callable(getter) else None
    leader_step = (
        int(raw_step)
        if isinstance(raw_step, int | float) and not isinstance(raw_step, bool)
        else None
    )
    if leader_step is not None:
        leader_matches = []
        for artifact in candidates:
            metadata = dict(getattr(artifact, "metadata", {}) or {})
            if checkpoint_step_from_artifact(artifact) != leader_step:
                continue
            if leader_uri and str(metadata.get("artifact_storage_uri") or "") != leader_uri:
                continue
            leader_matches.append(artifact)
        if leader_matches:
            selected = max(
                leader_matches,
                key=lambda artifact: (
                    _RUN_KIND_TIE_PRIORITY.get(
                        _artifact_kind_from_collection(_artifact_collection_name(artifact)) or "",
                        -1,
                    ),
                    _artifact_numeric_version(artifact),
                ),
            )
            return _artifact_concrete_ref(selected)

    stepped = [
        artifact for artifact in candidates if checkpoint_step_from_artifact(artifact) is not None
    ]
    if stepped:
        selected = max(
            stepped,
            key=lambda artifact: (
                int(checkpoint_step_from_artifact(artifact) or -1),
                _RUN_KIND_TIE_PRIORITY.get(
                    _artifact_kind_from_collection(_artifact_collection_name(artifact)) or "",
                    -1,
                ),
                _artifact_numeric_version(artifact),
            ),
        )
        return _artifact_concrete_ref(selected)

    legacy_latest = [
        artifact
        for artifact in candidates
        if _artifact_kind_from_collection(_artifact_collection_name(artifact)) == "checkpoint"
        and "latest" in _artifact_alias_set(artifact)
    ]
    if legacy_latest:
        selected = max(legacy_latest, key=_artifact_numeric_version)
        return _artifact_concrete_ref(selected)
    return None


def _logged_run_artifact_ref(run: Any, *, kind: str, version: str) -> str | None:
    """Return the requested model collection actually logged by a W&B run.

    Queue-backed runs use the immutable W&B run id for artifact collection names,
    whereas older local runs used the human-readable run name.  Reading the run's
    logged artifacts supports both naming schemes without guessing either one.
    """
    try:
        artifacts = run.logged_artifacts()
    except Exception:
        return None
    suffix = f"-{kind}:"
    for artifact in artifacts:
        if str(getattr(artifact, "type", "")) != "model":
            continue
        qualified_name = str(getattr(artifact, "qualified_name", "") or "")
        if suffix not in qualified_name:
            continue
        aliases = {str(alias) for alias in getattr(artifact, "aliases", []) or []}
        if version == "latest" and "latest" not in aliases:
            continue
        return f"{qualified_name.rsplit(':', 1)[0]}:{version}"
    return None


def _promoted_run_artifact_ref(
    run: Any,
    *,
    kind: str,
    version: str,
) -> tuple[str | None, int | None]:
    """Resolve an unqualified run to its best currently visible immutable artifact."""
    if kind != "checkpoint" or version != "latest":
        return None, None
    return _preferred_run_artifact_ref(run), None


def _promoted_run_artifact_ref_by_name(
    run_name: str,
    *,
    project: str,
    kind: str,
    version: str,
) -> tuple[str | None, int | None]:
    """Return a unique named run's promoted artifact and pending step, if any."""
    import wandb

    try:
        runs = wandb.Api().runs(project, filters={"display_name": run_name})
        matches = [
            run
            for run in runs
            if getattr(run, "name", None) == run_name
            or _wandb_run_config_value(run, "run_name") == run_name
        ]
    except Exception:
        return None, None
    promoted = [_promoted_run_artifact_ref(run, kind=kind, version=version) for run in matches]
    refs = sorted({ref for ref, _step in promoted if ref is not None})
    if len(refs) == 1:
        return refs[0], next(step for ref, step in promoted if ref == refs[0])
    if len(refs) > 1:
        choices = ", ".join(refs)
        raise SystemExit(
            f"Run name {run_name!r} is ambiguous within W&B project {project}; "
            f"pass a run URL. Matches: {choices}"
        )
    pending_steps = sorted({step for _ref, step in promoted if step is not None})
    if len(pending_steps) == 1:
        return None, pending_steps[0]
    if len(pending_steps) > 1:
        raise SystemExit(
            f"Run name {run_name!r} is ambiguous within W&B project {project}; pass a run URL."
        )
    return None, None


def _run_artifact_ref_by_name(
    run_name: str,
    *,
    project: str,
    kind: str,
    version: str,
) -> str | None:
    """Find a uniquely named W&B run and return its logged model artifact."""
    import wandb

    try:
        runs = wandb.Api().runs(project, filters={"display_name": run_name})
    except Exception:
        return None
    try:
        matches = [
            run
            for run in runs
            if getattr(run, "name", None) == run_name
            or _wandb_run_config_value(run, "run_name") == run_name
        ]
    except Exception:
        return None
    refs = [
        ref
        for run in matches
        if (ref := _logged_run_artifact_ref(run, kind=kind, version=version)) is not None
    ]
    unique_refs = sorted(set(refs))
    if len(unique_refs) == 1:
        return unique_refs[0]
    return None


def resolve_unique_bare_run_artifact_ref(
    run_name: str,
    *,
    default_project: str,
    kind: str,
    version: str,
) -> str | None:
    if "/" in run_name or ":" in run_name:
        return None
    candidates = [
        model_artifact_ref(project=project, run_name=run_name, kind=kind, version=version)
        for project in artifact_lookup_project_paths(default_project, run_name)
    ]
    if not candidates:
        return None

    load_wandb_env()

    if kind == "checkpoint" and version == "latest":
        import wandb

        matched_runs: list[tuple[str, Any]] = []
        lookup_errors: list[Exception] = []
        for project in artifact_lookup_project_paths(default_project, run_name):
            try:
                runs = wandb.Api().runs(project, filters={"display_name": run_name})
                for run in runs:
                    if (
                        getattr(run, "name", None) == run_name
                        or _wandb_run_config_value(run, "run_name") == run_name
                    ):
                        matched_runs.append((project, run))
            except Exception as exc:
                lookup_errors.append(exc)
        preferred = [
            (project, ref)
            for project, run in matched_runs
            if (ref := _preferred_run_artifact_ref(run)) is not None
        ]
        unique_preferred = sorted({ref for _project, ref in preferred})
        if len(unique_preferred) == 1:
            return unique_preferred[0]
        if len(unique_preferred) > 1:
            choices = ", ".join(unique_preferred)
            raise SystemExit(
                f"Run name {run_name!r} is ambiguous across W&B projects; pass a run URL. "
                f"Matches: {choices}"
            )
        if matched_runs:
            raise SystemExit(
                f"No W&B checkpoint artifact is available for run {run_name!r} yet; retry "
                "after the first checkpoint upload is confirmed."
            )
        # A bare token predates run-URL resolution and may be a direct artifact
        # collection name. Preserve that fallback when one or more candidate
        # projects cannot be listed; explicit run URLs still surface API errors.
        if lookup_errors:
            return None

    matches: list[str] = []
    if _artifact_exists(candidates[0]):
        matches.append(candidates[0])
    remaining = candidates[1:]
    if remaining:
        with ThreadPoolExecutor(
            max_workers=min(MAX_PARALLEL_ARTIFACT_LOOKUPS, len(remaining))
        ) as pool:
            futures = {pool.submit(_artifact_exists, ref): ref for ref in remaining}
            for future in as_completed(futures):
                if future.result():
                    matches.append(futures[future])

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        choices = ", ".join(sorted(matches))
        raise SystemExit(
            f"Run name {run_name!r} is ambiguous across W&B projects; pass project/run. "
            f"Matches: {choices}"
        )
    logged_matches = [
        ref
        for project in artifact_lookup_project_paths(default_project, run_name)
        if (ref := _run_artifact_ref_by_name(run_name, project=project, kind=kind, version=version))
        is not None
    ]
    if len(logged_matches) == 1:
        return logged_matches[0]
    if len(logged_matches) > 1:
        choices = ", ".join(sorted(logged_matches))
        raise SystemExit(
            f"Run name {run_name!r} is ambiguous across W&B projects; pass a run URL. "
            f"Matches: {choices}"
        )
    return None


def model_ref_from_run_path(
    value: str,
    *,
    default_project: str,
    kind: str,
    version: str,
) -> str | None:
    run_ref = wandb_run_artifact_ref(
        value,
        default_project=default_project,
        kind=kind,
        version=version,
    )
    if run_ref is not None:
        return run_ref
    if ":" in value:
        return None
    parts = value.split("/")
    if len(parts) == 1:
        inferred_ref = resolve_unique_bare_run_artifact_ref(
            parts[0],
            default_project=default_project,
            kind=kind,
            version=version,
        )
        if inferred_ref is not None:
            return inferred_ref
        project = default_project
        run_name = parts[0]
    elif len(parts) == 2:
        entity = default_project.split("/", 1)[0]
        project = f"{entity}/{parts[0]}"
        run_name = parts[1]
    elif len(parts) == 3:
        project = f"{parts[0]}/{parts[1]}"
        run_name = parts[2]
    else:
        return None
    if not run_name:
        return None
    if run_name.endswith(MODEL_ARTIFACT_KIND_SUFFIXES):
        return f"{project}/{run_name}:{version}"
    return model_artifact_ref(
        project=project,
        run_name=run_name,
        kind=kind,
        version=version,
    )


def _wandb_run_config_value(run: Any, key: str) -> Any:
    config = getattr(run, "config", {}) or {}
    if isinstance(config, Mapping):
        return config.get(key)
    getter = getattr(config, "get", None)
    if callable(getter):
        try:
            return getter(key)
        except Exception:
            return None
    return None


def wandb_run_artifact_ref(
    value: str,
    *,
    default_project: str,
    kind: str,
    version: str,
) -> str | None:
    run_ref = parse_wandb_run_ref(value, default_project=default_project)
    if run_ref is None:
        return None

    load_wandb_env()

    import wandb

    run_path = f"{run_ref.project_path}/{run_ref.run_id}"
    try:
        run = wandb.Api().run(run_path)
    except Exception as exc:
        raise SystemExit(f"Could not resolve W&B run {run_path}: {exc}") from exc
    if kind == "checkpoint" and version == "latest":
        try:
            preferred = _preferred_run_artifact_ref(run)
        except Exception as exc:
            raise SystemExit(f"Could not inspect W&B run artifacts for {run_path}: {exc}") from exc
        if preferred is not None:
            return preferred
        wandb_enabled = _wandb_run_config_value(run, "wandb")
        disabled = wandb_enabled is False or bool(
            _wandb_run_config_value(run, "no_wandb_artifacts")
        )
        if disabled:
            raise SystemExit(
                f"W&B artifacts are disabled for run {run_path}; no remote checkpoint is available."
            )
        raise SystemExit(
            f"No W&B checkpoint artifact is available for run {run_path} yet; retry after "
            "the first checkpoint upload is confirmed."
        )
    promoted_ref, promoted_step = _promoted_run_artifact_ref(
        run,
        kind=kind,
        version=version,
    )
    if promoted_ref is not None:
        return promoted_ref
    if promoted_step is not None:
        raise SystemExit(
            f"Promoted checkpoint step-{promoted_step} for W&B run {run_path} is not "
            "available as an artifact yet; retry after artifact projection completes."
        )
    logged_ref = _logged_run_artifact_ref(run, kind=kind, version=version)
    if logged_ref is not None:
        return logged_ref
    run_name = (
        _wandb_run_config_value(run, "run_name")
        or getattr(run, "name", None)
        or getattr(run, "id", None)
        or run_ref.run_id
    )
    return model_artifact_ref(
        project=run_ref.project_path,
        run_name=safe_artifact_stem(str(run_name)),
        kind=kind,
        version=version,
    )


def single_model_artifact_ref(args: argparse.Namespace) -> str | None:
    artifacts = artifact_values(args)
    if artifacts:
        return artifacts[0]
    positional = getattr(args, "artifact_ref", None)
    if positional:
        positional_ref = str(positional)
        if is_huggingface_model_ref(positional_ref):
            return None
        ref = model_ref_from_run_path(
            positional_ref,
            default_project=artifact_project_from_args(args),
            kind=getattr(args, "artifact_kind", "checkpoint"),
            version=getattr(args, "artifact_version", "latest"),
        )
        if ref is not None:
            return ref
        if ":" in positional_ref:
            return positional_ref
        if "/" in positional_ref:
            return positional_ref
    run_name = getattr(args, "artifact_run", None)
    if not run_name:
        return None
    ref = model_ref_from_run_path(
        str(run_name),
        default_project=artifact_project_from_args(args),
        kind=getattr(args, "artifact_kind", "checkpoint"),
        version=getattr(args, "artifact_version", "latest"),
    )
    if ref is not None:
        return ref
    return model_artifact_ref(
        project=artifact_project_from_args(args),
        run_name=str(run_name),
        kind=getattr(args, "artifact_kind", "checkpoint"),
        version=getattr(args, "artifact_version", "latest"),
    )


def single_huggingface_model_ref(args: argparse.Namespace) -> str | None:
    positional = getattr(args, "artifact_ref", None)
    if positional and is_huggingface_model_ref(str(positional)):
        return str(positional)
    positional = getattr(args, "model_ref", None)
    if positional and is_huggingface_model_ref(str(positional)):
        return str(positional)
    model = getattr(args, "model", None)
    if model and is_huggingface_model_ref(str(model)):
        return str(model)
    return None


def model_source_ref(args: argparse.Namespace) -> str | None:
    return single_huggingface_model_ref(args) or single_model_artifact_ref(args)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except TypeError, ValueError:
        return None


def _mapping_value(mapping: Any, key: str) -> Any:
    if isinstance(mapping, Mapping):
        return mapping.get(key)
    json_dict = getattr(mapping, "_json_dict", None)
    if isinstance(json_dict, Mapping):
        return json_dict.get(key)
    getter = getattr(mapping, "get", None)
    if callable(getter):
        try:
            return getter(key)
        except Exception:
            return None
    return None


def download_artifact_ref_source(ref: str, root: Path) -> ResolvedModelSource:
    model_path, revision = download_model_artifact_with_revision(ref, root)
    pinned_ref = f"{ref.rsplit(':', 1)[0]}:{revision}"
    bundle = load_policy_bundle_from_checkpoint(
        model_path, source=pinned_ref, revision=revision or None
    )
    return ResolvedModelSource(
        model_path=model_path,
        artifact_ref=pinned_ref,
        artifact_name=pinned_ref,
        bundle=bundle,
        checkpoint_step=(
            int(bundle.model["checkpoint"]["step"])
            if bundle is not None and bundle.model["checkpoint"].get("step") is not None
            else None
        ),
    )


def download_remote_model_source(
    ref: str,
    *,
    root: Path,
    require_pinned: bool = False,
) -> ResolvedModelSource:
    """Resolve a supported remote model and optionally require its locator to be immutable."""
    text = str(ref).strip()
    if is_huggingface_model_ref(text):
        resolved = download_huggingface_model_source(text, root=root)
    elif is_s3_model_ref(text) or is_object_store_model_ref(text):
        resolved = download_s3_model_source(text, root=root)
    else:
        artifact_ref = text
        if is_wandb_run_ref(text):
            artifact_ref = str(
                model_ref_from_run_path(
                    text,
                    default_project=None,
                    kind="checkpoint",
                    version="latest",
                )
                or ""
            )
        if not artifact_ref or "/" not in artifact_ref or ":" not in artifact_ref:
            raise ValueError(
                "remote model source must be a content-addressed S3 model, Hugging Face "
                "model, or W&B model artifact"
            )
        resolved = download_artifact_ref_source(artifact_ref, root)
    pinned = str(resolved.artifact_name or "")
    if not pinned:
        raise ValueError("remote model source did not resolve an immutable locator")
    if require_pinned and pinned != text:
        raise ValueError(f"remote model locator is not immutable: expected {pinned!r}")
    return resolved


def download_s3_model_source(ref: str, *, root: Path) -> ResolvedModelSource:
    """Download one content-addressed policy bundle from worker-accessible object storage."""
    text = str(ref).strip()
    logical = is_object_store_model_ref(text)
    if not is_s3_model_ref(text) and not logical:
        raise ValueError(
            "object-store or S3 model locator must end with "
            "sha256/<model-sha256>/model.zip"
        )
    from rlab.modal_eval_storage import (
        ObjectNotFound,
        ObjectStore,
        object_store_base_uri,
        write_downloaded_file,
    )
    from rlab.wandb_utils import load_wandb_env

    parsed = urlparse(text)
    path = (
        text.removeprefix(OBJECT_STORE_MODEL_SCHEME).strip("/")
        if logical
        else parsed.path.strip("/")
    )
    parts = _content_addressed_model_parts(path)
    expected_model_sha256 = parts[-2]
    prefix = "/".join(parts[:-1])
    target_dir = root / safe_artifact_stem(
        f"{parsed.netloc or 'object-store'}-{expected_model_sha256}"
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    load_wandb_env()
    store = ObjectStore(object_store_base_uri() if logical else f"s3://{parsed.netloc}")
    downloaded: dict[str, Path] = {}
    for filename in (MODEL_FILENAME, CHECKPOINT_FILENAME, RECIPE_FILENAME):
        uri = (
            store.uri(f"{prefix}/{filename}")
            if logical
            else f"s3://{parsed.netloc}/{prefix}/{filename}"
        )
        try:
            store.head(uri)
        except ObjectNotFound:
            if filename == CHECKPOINT_FILENAME:
                raise ValueError(
                    f"object-store model closure is missing {CHECKPOINT_FILENAME}"
                ) from None
            continue
        downloaded[filename] = write_downloaded_file(
            store.presign_get(uri, expires_seconds=300),
            target_dir / filename,
        )
    model_path = downloaded[CHECKPOINT_FILENAME]
    actual_model_sha256 = file_sha256(model_path)
    if actual_model_sha256 != expected_model_sha256:
        raise ValueError("object-store model bytes do not match the content-addressed locator")
    bundle = load_policy_bundle_from_checkpoint(
        model_path,
        source=text,
        revision=expected_model_sha256,
    )
    return ResolvedModelSource(
        model_path=model_path,
        artifact_name=text,
        checkpoint_step=(
            int(bundle.model["checkpoint"]["step"])
            if bundle is not None and bundle.model["checkpoint"].get("step") is not None
            else None
        ),
        bundle=bundle,
    )


def parse_huggingface_model_ref(value: str) -> tuple[str, str | None, str | None]:
    text = str(value or "").strip()
    if text.startswith(HUGGINGFACE_MODEL_SCHEME):
        path = text.removeprefix(HUGGINGFACE_MODEL_SCHEME).strip("/")
        parts = [unquote(part) for part in path.split("/") if part]
        if len(parts) < 2:
            raise ValueError(f"expected Hugging Face model ref like hf://owner/repo, got {value!r}")
        repo_name, separator, revision = parts[1].partition("@")
        if separator and not revision:
            raise ValueError(f"Hugging Face model ref has an empty revision: {value!r}")
        repo_id = f"{parts[0]}/{repo_name}"
        filename = "/".join(parts[2:]) or None
        return repo_id, filename, revision or None

    parsed = urlparse(text)
    if parsed.scheme not in {"http", "https"} or parsed.netloc != HUGGINGFACE_MODEL_URL_HOST:
        raise ValueError(f"expected Hugging Face model ref, got {value!r}")
    parts = [unquote(part) for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        raise ValueError(f"expected Hugging Face model URL with owner/repo, got {value!r}")
    repo_id = "/".join(parts[:2])
    filename = None
    revision = None
    if len(parts) >= 5 and parts[2] in {"blob", "raw", "resolve"}:
        revision = parts[3]
        filename = "/".join(parts[4:])
    elif len(parts) > 2:
        filename = "/".join(parts[2:])
    return repo_id, filename, revision


def _select_huggingface_checkpoint(
    *,
    repo_id: str,
    revision: str,
    filename: str | None,
) -> str:
    if filename:
        return filename
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit(
            "huggingface-hub is required for hf:// model refs; reinstall rlab with "
            "`uv tool install --from git+https://github.com/tsilva/rlab rlab`."
        ) from exc
    try:
        files = HfApi().list_repo_files(repo_id=repo_id, repo_type="model", revision=revision)
    except Exception as exc:
        raise SystemExit(f"Could not list Hugging Face model repo {repo_id}: {exc}") from exc
    checkpoints = sorted(path for path in files if path.endswith(".zip"))
    if not checkpoints:
        raise SystemExit(f"Hugging Face model repo {repo_id} has no .zip checkpoint files")
    if len(checkpoints) > 1:
        choices = ", ".join(checkpoints)
        raise SystemExit(
            f"Hugging Face model repo {repo_id} has multiple .zip checkpoints; "
            f"pass --hf-file. Choices: {choices}"
        )
    return checkpoints[0]


def _download_huggingface_release_closure(
    *,
    repo_id: str,
    revision: str,
    repo_files: set[str],
    target_dir: Path,
    hf_hub_download: Any,
) -> None:
    filename = "release_manifest.json"
    if filename not in repo_files:
        return
    manifest_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="model",
            revision=revision,
            filename=filename,
            local_dir=target_dir,
        )
    )
    if manifest_path.stat().st_size > 8 * 1024**2:
        raise ValueError("Hugging Face release manifest exceeds 8 MiB")
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = document.get("artifacts") if isinstance(document, Mapping) else None
    if not isinstance(artifacts, Mapping):
        raise ValueError("Hugging Face release manifest has no artifact closure")
    for bound_name in artifacts:
        bound_name = str(bound_name)
        if Path(bound_name).name != bound_name or bound_name not in repo_files:
            raise ValueError(f"release manifest binds unavailable file {bound_name!r}")
        hf_hub_download(
            repo_id=repo_id,
            repo_type="model",
            revision=revision,
            filename=bound_name,
            local_dir=target_dir,
        )


def download_huggingface_model_source(
    ref: str,
    *,
    root: Path,
    filename: str | None = None,
    revision: str | None = None,
) -> ResolvedModelSource:
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface-hub is required for hf:// model refs; reinstall rlab with "
            "`uv tool install --from git+https://github.com/tsilva/rlab rlab`."
        ) from exc

    try:
        repo_id, parsed_filename, parsed_revision = parse_huggingface_model_ref(ref)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    resolved_revision = revision or parsed_revision or "main"
    api = HfApi()
    try:
        immutable_revision = str(
            api.model_info(repo_id=repo_id, revision=resolved_revision).sha or ""
        )
        if not immutable_revision:
            raise ValueError("model repository did not return an immutable commit SHA")
        repo_files = set(
            api.list_repo_files(
                repo_id=repo_id,
                repo_type="model",
                revision=immutable_revision,
            )
        )
    except Exception as exc:
        raise SystemExit(f"Could not inspect Hugging Face model repo {repo_id}: {exc}") from exc
    if MODEL_FILENAME in repo_files:
        target_dir = root / safe_artifact_stem(f"{repo_id}@{immutable_revision}")
        target_dir.mkdir(parents=True, exist_ok=True)
        for bundle_filename in (CHECKPOINT_FILENAME, MODEL_FILENAME, RECIPE_FILENAME):
            try:
                hf_hub_download(
                    repo_id=repo_id,
                    repo_type="model",
                    revision=immutable_revision,
                    filename=bundle_filename,
                    local_dir=target_dir,
                )
            except Exception as exc:
                raise SystemExit(
                    f"Could not download required {bundle_filename} from {repo_id}@"
                    f"{immutable_revision}: {exc}"
                ) from exc
        _download_huggingface_release_closure(
            repo_id=repo_id,
            revision=immutable_revision,
            repo_files=repo_files,
            target_dir=target_dir,
            hf_hub_download=hf_hub_download,
        )
        bundle = load_policy_bundle(
            target_dir,
            source=f"hf://{repo_id}",
            revision=immutable_revision,
        )
        return ResolvedModelSource(
            model_path=bundle.checkpoint_path,
            artifact_name=f"hf://{repo_id}@{immutable_revision}",
            checkpoint_step=bundle.model["checkpoint"].get("step"),
            bundle=bundle,
        )
    checkpoint_filename = _select_huggingface_checkpoint(
        repo_id=repo_id,
        revision=immutable_revision,
        filename=filename or parsed_filename,
    )
    target_dir = root / safe_artifact_stem(f"{repo_id}@{immutable_revision}")
    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        checkpoint_path = Path(
            hf_hub_download(
                repo_id=repo_id,
                repo_type="model",
                revision=immutable_revision,
                filename=checkpoint_filename,
                local_dir=target_dir,
            )
        )
    except Exception as exc:
        raise SystemExit(
            f"Could not download {checkpoint_filename} from Hugging Face model repo {repo_id}: {exc}"
        ) from exc

    try:
        metadata_path = Path(
            hf_hub_download(
                repo_id=repo_id,
                repo_type="model",
                revision=immutable_revision,
                filename="model_metadata.json",
                local_dir=target_dir,
            )
        )
    except Exception as exc:
        print(
            f"warning: could not download model_metadata.json from {repo_id}: {exc}",
            file=sys.stderr,
        )
    else:
        sidecar_path = checkpoint_path.with_suffix(".metadata.json")
        if metadata_path != sidecar_path:
            shutil.copy2(metadata_path, sidecar_path)
    _download_huggingface_release_closure(
        repo_id=repo_id,
        revision=immutable_revision,
        repo_files=repo_files,
        target_dir=target_dir,
        hf_hub_download=hf_hub_download,
    )

    return ResolvedModelSource(
        model_path=checkpoint_path,
        artifact_ref=None,
        artifact_name=f"hf://{repo_id}@{immutable_revision}/{checkpoint_filename}",
        checkpoint_step=checkpoint_step_from_artifact(None, checkpoint_path),
    )


def resolve_single_model_source(
    args: argparse.Namespace,
    *,
    resolved_ref: str | None = None,
) -> ResolvedModelSource:
    hf_ref = (
        resolved_ref
        if resolved_ref is not None and is_huggingface_model_ref(resolved_ref)
        else single_huggingface_model_ref(args)
    )
    if hf_ref is not None:
        return download_huggingface_model_source(
            hf_ref,
            root=Path(getattr(args, "hf_model_root", "runs/hf_models")),
            filename=getattr(args, "hf_file", None),
            revision=getattr(args, "hf_revision", None),
        )
    ref = resolved_ref if resolved_ref is not None else single_model_artifact_ref(args)
    if ref is not None:
        return download_artifact_ref_source(ref, Path(args.artifact_root))
    model_path = Path(str(args.model))
    return ResolvedModelSource(
        model_path=model_path,
        bundle=load_policy_bundle_from_checkpoint(model_path),
    )


def artifact_run_config(ref: str) -> dict[str, Any]:
    load_wandb_env()

    import wandb

    try:
        run = wandb.Api().artifact(ref, type="model").logged_by()
    except Exception as exc:
        print(f"warning: could not infer playback config from {ref}: {exc}", file=sys.stderr)
        return {}
    if run is None:
        return {}
    config = getattr(run, "config", {}) or {}
    return config if isinstance(config, dict) else {}


def apply_model_source_defaults(
    args: argparse.Namespace,
    source: ResolvedModelSource,
    parser: argparse.ArgumentParser,
    parser_defaults: dict[str, object],
    explicit_dests: set[str],
    *,
    infer_artifact_config: bool = False,
    metadata_kind: str | None = None,
    print_loaded_metadata: bool = False,
) -> bool:
    metadata = load_model_metadata(source.model_path)
    assert_metadata_runtime_versions(metadata)
    saved_config = env_config_from_metadata(metadata)
    if saved_config:
        apply_config_defaults(args, saved_config, parser_defaults, explicit_dests)
        if print_loaded_metadata:
            print(
                f"loaded playback metadata: {source.model_path.with_suffix('.metadata.json')}",
                flush=True,
            )
        return True
    if not infer_artifact_config or source.artifact_ref is None:
        return False

    inferred_config = sanitize_env_config_metadata(artifact_run_config(source.artifact_ref))
    if not inferred_config:
        return False
    apply_config_defaults(args, inferred_config, parser_defaults, explicit_dests)
    source.run_config = inferred_config

    metadata_args = parser.parse_args([])
    apply_config_defaults(metadata_args, inferred_config, parser_defaults, set())
    metadata_config = resolve_env_config(env_config_from_args(metadata_args))
    kind = metadata_kind or getattr(args, "artifact_kind", "checkpoint")
    metadata_path = write_model_metadata(source.model_path, args, metadata_config, kind=kind)
    if metadata_path is not None:
        print(f"Wrote playback metadata: {metadata_path}", flush=True)
    return True
