from __future__ import annotations

import re
import sys
from pathlib import Path

from mario_ppo.wandb_utils import load_wandb_env


def safe_artifact_stem(value: str, fallback: str = "artifact") -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or fallback


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


def download_model_artifact(ref: str, root: Path) -> Path:
    load_wandb_env()

    import wandb

    root.mkdir(parents=True, exist_ok=True)
    artifact = wandb.Api().artifact(ref, type="model")
    path = Path(artifact.download(root=str(root)))
    return model_zip_from_download(path)


def model_zip_from_download(path: Path) -> Path:
    zip_files = sorted(path.glob("*.zip"))
    if not zip_files:
        raise FileNotFoundError(f"No .zip model file found in downloaded artifact: {path}")
    if len(zip_files) > 1:
        print(f"Multiple model files found; using {zip_files[0]}", file=sys.stderr)
    return zip_files[0]
