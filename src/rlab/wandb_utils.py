from __future__ import annotations

from pathlib import Path

from rlab.dotenv import load_env_file

DEFAULT_WANDB_ENTITY = "tsilva"
DEFAULT_WANDB_PROJECT = "SuperMarioBros-Nes-v0"
DEFAULT_WANDB_PROJECT_PATH = f"{DEFAULT_WANDB_ENTITY}/{DEFAULT_WANDB_PROJECT}"

WANDB_ENV_PREFIXES = ("WANDB_", "AWS_")
WANDB_ARTIFACT_ENV_KEYS = {
    "CHECKPOINT_BUCKET_URI",
}


def load_wandb_env(dotenv_path: str | Path = ".env") -> None:
    """Load W&B and artifact storage env vars without adding a dotenv dependency."""
    load_env_file(
        dotenv_path,
        key_filter=lambda key: key.startswith(WANDB_ENV_PREFIXES)
        or key in WANDB_ARTIFACT_ENV_KEYS,
    )


def wandb_project_for_env_id(env_id: str | None, *, fallback: str = DEFAULT_WANDB_PROJECT) -> str:
    """Return the default W&B project name for a provider-local environment id."""

    project = str(env_id or "").strip()
    return project or fallback


def resolve_wandb_project(explicit_project: object, env_id: str | None) -> str:
    """Use explicit W&B project when supplied, otherwise default to the env id."""

    project = str(explicit_project or "").strip()
    return project or wandb_project_for_env_id(env_id)
