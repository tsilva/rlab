from __future__ import annotations

import os
import subprocess
from pathlib import Path

import modal

APP_NAME = "stable-retro-ppo"
VOLUME_NAME = "stable-retro-ppo-data"
PROJECT_ROOT = Path("/root/stable-retro-ppo")
VOLUME_ROOT = Path("/vol")
ROM_DIR = VOLUME_ROOT / "roms"
RUNS_DIR = VOLUME_ROOT / "runs"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
wandb_secret = modal.Secret.from_name("wandb-secret")


def load_local_env_for_modal_secrets(path: Path = Path(".env")) -> None:
    if not path.is_file():
        return
    allowed = {
        "DATABASE_URL",
        "DIRECT_DATABASE_URL",
        "WANDB_API_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_S3_ENDPOINT_URL",
        "AWS_REGION",
        "CHECKPOINT_BUCKET_URI",
    }
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key in allowed:
            os.environ.setdefault(key, value.strip().strip("'\""))


load_local_env_for_modal_secrets()
eval_queue_secret = modal.Secret.from_local_environ(
    [
        "DATABASE_URL",
        "WANDB_API_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_S3_ENDPOINT_URL",
        "AWS_REGION",
        "CHECKPOINT_BUCKET_URI",
    ]
)

image = (
    modal.Image.debian_slim(python_version="3.14")
    .apt_install("ffmpeg", "git")
    .pip_install_from_pyproject(
        "pyproject.toml",
        optional_dependencies=[],
        extra_options="--only-binary=:all:",
    )
    .workdir(str(PROJECT_ROOT))
    .env(
        {
            "PYTHONPATH": str(PROJECT_ROOT / "src"),
            "MPLCONFIGDIR": "/tmp/matplotlib",
            "WANDB_DIR": str(RUNS_DIR),
            "WANDB_CACHE_DIR": str(RUNS_DIR / ".wandb-cache"),
            "WANDB_CONFIG_DIR": str(RUNS_DIR / ".wandb-config"),
            "WANDB_DATA_DIR": str(RUNS_DIR / ".wandb-data"),
            "WANDB_ARTIFACT_DIR": str(RUNS_DIR / ".wandb-artifacts"),
        },
    )
    .add_local_dir(
        ".",
        remote_path=str(PROJECT_ROOT),
        ignore=[
            ".git",
            ".env",
            ".env.*",
            ".venv",
            ".uv-cache",
            ".matplotlib",
            "__pycache__",
            "runs",
            "logs",
            "models",
            "videos",
            "wandb",
        ],
    )
)


def run_cmd(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)


def ensure_remote_roms(kind: str = "training") -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    if not ROM_DIR.exists() or not any(ROM_DIR.iterdir()):
        raise FileNotFoundError(f"No ROMs found in {ROM_DIR}. Run upload_roms before {kind}.")
    run_cmd(["python", "-m", "stable_retro.import", str(ROM_DIR)])
