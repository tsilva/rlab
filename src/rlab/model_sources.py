from __future__ import annotations

import argparse
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from rlab.artifacts import (
    apply_config_defaults,
    explicit_arg_dests,
    load_model_metadata,
    write_model_metadata,
)
from rlab.env import resolve_env_config
from rlab.env_config import env_config_from_args
from rlab.env_metadata import env_config_from_metadata, sanitize_env_config_metadata
from rlab.recipe_documents import load_goal_contract_document
from rlab.wandb_artifacts import (
    artifact_qualified_name,
    checkpoint_step_from_artifact,
    download_model_artifact,
    model_artifact_ref,
    safe_artifact_stem,
)
from rlab.wandb_utils import default_wandb_project_path, load_wandb_env


MODEL_KIND_CHOICES = ("final", "best", "checkpoint")
HUGGINGFACE_MODEL_SCHEME = "hf://"
HUGGINGFACE_MODEL_URL_HOST = "huggingface.co"
WANDB_RUN_URL_HOST = "wandb.ai"
MODEL_ARTIFACT_KIND_SUFFIXES = tuple(f"-{kind}" for kind in MODEL_KIND_CHOICES)
DEFAULT_ARTIFACT_LOOKUP_PROJECTS = (
    "SuperMarioBros-Nes-v0",
    "SuperMarioBros3-Nes-v0",
    "ms_pacman",
    "breakout",
)
LEGACY_GOAL_PROJECTS = {
    "alepy__breakout": "breakout",
    "alepy__mspacman": "ms_pacman",
}
MAX_PARALLEL_ARTIFACT_LOOKUPS = 4


@dataclass
class ResolvedModelSource:
    model_path: Path
    artifact_ref: str | None = None
    artifact_name: str | None = None
    checkpoint_step: int | None = None
    run_config: dict[str, Any] = field(default_factory=dict)


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
        artifact_kwargs["help"] = "Full W&B model artifact ref to evaluate. May be passed more than once."
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
    if isinstance(env_args, Mapping) and env_args.get("game"):
        return str(env_args["game"])
    if env_config.get("game"):
        return str(env_config["game"])
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
    for goal_id, project in LEGACY_GOAL_PROJECTS.items():
        if run_name == goal_id or run_name.startswith(f"{goal_id}_"):
            inferred.append(_project_path(entity, project))
            break
    for goal_id, project in sorted(local_projects.items(), key=lambda item: len(item[0]), reverse=True):
        if run_name == goal_id or run_name.startswith(f"{goal_id}_"):
            inferred.append(_project_path(entity, project))
            break
    projects = [
        *inferred,
        default_project,
        *(_project_path(entity, project) for project in local_projects.values()),
        *(_project_path(entity, project) for project in DEFAULT_ARTIFACT_LOOKUP_PROJECTS),
    ]
    return _dedupe(projects)


def _artifact_exists(ref: str) -> bool:
    import wandb

    try:
        wandb.Api().artifact(ref, type="model")
    except Exception:
        return False
    return True


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

    matches: list[str] = []
    if _artifact_exists(candidates[0]):
        matches.append(candidates[0])
    remaining = candidates[1:]
    if not remaining:
        return matches[0] if matches else None

    with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_ARTIFACT_LOOKUPS, len(remaining))) as pool:
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
    except (TypeError, ValueError):
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


def _artifact_kind(artifact: Any) -> str:
    metadata = getattr(artifact, "metadata", {}) or {}
    if isinstance(metadata, Mapping) and metadata.get("kind"):
        return str(metadata["kind"])
    name = artifact_qualified_name(artifact).split(":", 1)[0]
    if name.endswith("-final"):
        return "final"
    if name.endswith("-best"):
        return "best"
    if name.endswith("-checkpoint"):
        return "checkpoint"
    return ""


def _logged_run_global_step(artifact: Any) -> int | None:
    try:
        run = artifact.logged_by()
    except Exception:
        return None
    if run is None:
        return None
    summary = getattr(run, "summary", {}) or {}
    return _optional_int(_mapping_value(summary, "global_step"))


def model_artifact_checkpoint_step(artifact: Any, model_path: Path | None = None) -> int | None:
    step = checkpoint_step_from_artifact(artifact, model_path)
    if step is not None:
        return step
    if _artifact_kind(artifact) == "final":
        return _logged_run_global_step(artifact)
    return None


def download_artifact_ref_source(ref: str, root: Path) -> ResolvedModelSource:
    model_path = download_model_artifact(ref, root)
    return ResolvedModelSource(
        model_path=model_path,
        artifact_ref=ref,
        artifact_name=ref,
    )


def parse_huggingface_model_ref(value: str) -> tuple[str, str | None, str | None]:
    text = str(value or "").strip()
    if text.startswith(HUGGINGFACE_MODEL_SCHEME):
        path = text.removeprefix(HUGGINGFACE_MODEL_SCHEME).strip("/")
        parts = [unquote(part) for part in path.split("/") if part]
        if len(parts) < 2:
            raise ValueError(f"expected Hugging Face model ref like hf://owner/repo, got {value!r}")
        repo_id = "/".join(parts[:2])
        filename = "/".join(parts[2:]) or None
        return repo_id, filename, None

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


def download_huggingface_model_source(
    ref: str,
    *,
    root: Path,
    filename: str | None = None,
    revision: str | None = None,
) -> ResolvedModelSource:
    try:
        from huggingface_hub import hf_hub_download
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
    checkpoint_filename = _select_huggingface_checkpoint(
        repo_id=repo_id,
        revision=resolved_revision,
        filename=filename or parsed_filename,
    )
    target_dir = root / safe_artifact_stem(f"{repo_id}@{resolved_revision}")
    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        checkpoint_path = Path(
            hf_hub_download(
                repo_id=repo_id,
                repo_type="model",
                revision=resolved_revision,
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
                revision=resolved_revision,
                filename="model_metadata.json",
                local_dir=target_dir,
            )
        )
    except Exception as exc:
        print(f"warning: could not download model_metadata.json from {repo_id}: {exc}", file=sys.stderr)
    else:
        sidecar_path = checkpoint_path.with_suffix(".metadata.json")
        if metadata_path != sidecar_path:
            shutil.copy2(metadata_path, sidecar_path)

    return ResolvedModelSource(
        model_path=checkpoint_path,
        artifact_ref=None,
        artifact_name=f"hf://{repo_id}/{checkpoint_filename}",
        checkpoint_step=checkpoint_step_from_artifact(None, checkpoint_path),
    )


def resolve_single_model_source(args: argparse.Namespace) -> ResolvedModelSource:
    hf_ref = single_huggingface_model_ref(args)
    if hf_ref is not None:
        return download_huggingface_model_source(
            hf_ref,
            root=Path(getattr(args, "hf_model_root", "runs/hf_models")),
            filename=getattr(args, "hf_file", None),
            revision=getattr(args, "hf_revision", None),
        )
    ref = single_model_artifact_ref(args)
    if ref is not None:
        return download_artifact_ref_source(ref, Path(args.artifact_root))
    model_path = Path(str(args.model))
    return ResolvedModelSource(model_path=model_path)


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
    saved_config = env_config_from_metadata(load_model_metadata(source.model_path))
    if saved_config:
        apply_config_defaults(args, saved_config, parser_defaults, explicit_dests)
        if print_loaded_metadata:
            print(f"loaded playback metadata: {source.model_path.with_suffix('.metadata.json')}", flush=True)
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
    metadata_config = resolve_env_config(
        env_config_from_args(metadata_args, max_episode_steps_attr="max_steps")
    )
    kind = metadata_kind or getattr(args, "artifact_kind", "checkpoint")
    metadata_path = write_model_metadata(source.model_path, args, metadata_config, kind=kind)
    if metadata_path is not None:
        print(f"Wrote playback metadata: {metadata_path}", flush=True)
    return True


def explicit_source_arg_dests(parser: argparse.ArgumentParser, argv: list[str]) -> set[str]:
    return explicit_arg_dests(parser, argv)
