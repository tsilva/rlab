from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

from rlab.artifacts import model_metadata_path
from rlab.wandb_utils import load_wandb_env


def safe_artifact_stem(value: str, fallback: str = "artifact") -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or fallback


def artifact_aliases(artifact: Any) -> list[str]:
    aliases = []
    for alias in getattr(artifact, "aliases", []) or []:
        aliases.append(str(getattr(alias, "alias", alias)))
    return aliases


def artifact_qualified_name(artifact: Any) -> str:
    value = getattr(artifact, "qualified_name", None)
    if value:
        return str(value)
    return str(getattr(artifact, "name"))


def checkpoint_step_from_name(value: str) -> int | None:
    match = re.search(r"_(\d+)_steps(?:\.zip)?$", value)
    return int(match.group(1)) if match else None


def checkpoint_step_from_artifact(artifact: Any, model_path: Path | None = None) -> int | None:
    metadata = getattr(artifact, "metadata", {}) or {}
    step = metadata.get("checkpoint_step")
    if step is not None:
        return int(step)
    for alias in artifact_aliases(artifact):
        if alias.startswith("step-"):
            return int(alias.removeprefix("step-"))
    if model_path is not None:
        return checkpoint_step_from_name(model_path.name)
    return None


def model_artifact_ref(
    *,
    project: str,
    run_name: str,
    kind: str,
    version: str = "latest",
) -> str:
    if not run_name:
        raise ValueError("run_name is required")
    return f"{project}/{run_name}-{kind}:{version}"


def artifact_download_dir(root: Path, ref: str) -> Path:
    return root / safe_artifact_stem(ref.replace("/", "_").replace(":", "_"))


def immutable_artifact_ref(ref: str, artifact: Any) -> str:
    version = str(getattr(artifact, "version", "") or "").strip()
    if not version:
        raise ValueError(f"W&B artifact {ref!r} did not resolve to an immutable version")
    return f"{ref.rsplit(':', 1)[0]}:{version}"


def download_artifact_model(artifact: Any, root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = Path(artifact.download(root=str(root)))
    metadata = getattr(artifact, "metadata", {}) or {}
    expected_filename = metadata.get("filename") if isinstance(metadata, dict) else None
    model_path = model_zip_from_download(path, expected_filename=expected_filename)
    write_downloaded_artifact_metadata(model_path, artifact)
    return model_path


def download_model_artifact(ref: str, root: Path) -> Path:
    load_wandb_env()

    import wandb

    artifact = wandb.Api().artifact(ref, type="model")
    resolved_ref = immutable_artifact_ref(ref, artifact)
    return download_artifact_model(artifact, artifact_download_dir(root, resolved_ref))


def metadata_from_wandb_artifact(artifact, model_path: Path) -> dict:
    metadata = getattr(artifact, "metadata", None)
    if isinstance(metadata, dict) and metadata.get("training_metadata"):
        return dict(metadata)
    if isinstance(metadata, dict):
        print(
            f"warning: W&B artifact for {model_path.name} has no training_metadata",
            file=sys.stderr,
        )
        return dict(metadata)
    return {}


def write_downloaded_artifact_metadata(model_path: Path, artifact) -> Path | None:
    metadata = metadata_from_wandb_artifact(artifact, model_path)
    if not metadata:
        return None
    path = model_metadata_path(model_path)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def model_zip_from_download(path: Path, *, expected_filename: str | None = None) -> Path:
    if expected_filename:
        expected_path = path / expected_filename
        if not expected_path.is_file():
            raise FileNotFoundError(
                f"W&B artifact declared model file {expected_filename!r}, "
                f"but it was not found in {path}"
            )
        return expected_path
    zip_files = sorted(path.glob("*.zip"))
    if not zip_files:
        raise FileNotFoundError(f"No .zip model file found in downloaded artifact: {path}")
    if len(zip_files) > 1:
        print(f"Multiple model files found; using {zip_files[0]}", file=sys.stderr)
    return zip_files[0]
