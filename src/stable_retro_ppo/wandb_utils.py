from __future__ import annotations

import os
from pathlib import Path

DEFAULT_WANDB_ENTITY = "tsilva"
DEFAULT_WANDB_PROJECT = "SuperMarioBros-NES"
DEFAULT_WANDB_PROJECT_PATH = f"{DEFAULT_WANDB_ENTITY}/{DEFAULT_WANDB_PROJECT}"

WANDB_ENV_PREFIXES = ("WANDB_", "AWS_")
WANDB_ARTIFACT_ENV_KEYS = {
    "CHECKPOINT_BUCKET_URI",
}


def load_wandb_env(dotenv_path: str | Path = ".env") -> None:
    """Load W&B and artifact storage env vars without adding a dotenv dependency."""
    path = Path(dotenv_path)
    if not path.is_file():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key.startswith(WANDB_ENV_PREFIXES) and key not in WANDB_ARTIFACT_ENV_KEYS:
            continue
        value = value.strip().strip("'\"")
        os.environ.setdefault(key, value)
