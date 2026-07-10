from __future__ import annotations

import os
import argparse
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


def wandb_entity_from_env(*, fallback: str = DEFAULT_WANDB_ENTITY) -> str:
    entity = str(os.environ.get("WANDB_ENTITY") or "").strip()
    return entity or fallback


def default_wandb_project_path(project: str | None = None) -> str:
    load_wandb_env()
    project_name = str(project or DEFAULT_WANDB_PROJECT).strip() or DEFAULT_WANDB_PROJECT
    return f"{wandb_entity_from_env()}/{project_name}"


def wandb_project_for_env_id(env_id: str | None, *, fallback: str = DEFAULT_WANDB_PROJECT) -> str:
    """Return the default W&B project name for a provider-local environment id."""

    project = str(env_id or "").strip()
    return project or fallback


def resolve_wandb_project(explicit_project: object, env_id: str | None) -> str:
    """Use explicit W&B project when supplied, otherwise default to the env id."""

    project = str(explicit_project or "").strip()
    return project or wandb_project_for_env_id(env_id)


def configure_wandb_metrics(run):
    if run is not None:
        run.define_metric("global_step")
        run.define_metric("*", step_metric="global_step")
    return run


def resume_wandb_run(args: argparse.Namespace, run_dir: Path):
    if not getattr(args, "wandb", False):
        return None
    run_id_path = run_dir / "wandb_run_id.txt"
    if not run_id_path.is_file():
        raise RuntimeError(f"W&B run id not ready: {run_id_path}")
    run_id = run_id_path.read_text(encoding="utf-8").strip()
    if not run_id:
        raise RuntimeError(f"W&B run id is empty: {run_id_path}")
    load_wandb_env()
    import wandb

    return configure_wandb_metrics(
        wandb.init(
            project=resolve_wandb_project(
                getattr(args, "wandb_project", None), str(args.game)
            ),
            entity=args.wandb_entity,
            id=run_id,
            resume="allow",
            name=args.run_name,
            dir=str(run_dir),
            mode=args.wandb_mode,
            reinit=True,
        )
    )
