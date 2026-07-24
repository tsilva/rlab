from __future__ import annotations

import os
from pathlib import Path

from rlab.dotenv import load_env_file
from rlab.env_registry import game_family_for_environment, wandb_project_for_environment
from rlab.metric_names import EVAL_ACCEPTANCE_PASS, TRAIN_EPISODE_RETURN_SHAPED_MEAN

DEFAULT_WANDB_ENTITY = "tsilva"
DEFAULT_WANDB_PROJECT = "SuperMarioBros-Nes-v0"
DEFAULT_WANDB_PROJECT_PATH = f"{DEFAULT_WANDB_ENTITY}/{DEFAULT_WANDB_PROJECT}"

WANDB_ENV_PREFIXES = ("WANDB_",)


def load_wandb_env(dotenv_path: str | Path = ".env") -> None:
    """Load W&B configuration without exposing object-storage credentials."""
    load_env_file(
        dotenv_path,
        key_filter=lambda key: key.startswith(WANDB_ENV_PREFIXES),
    )


def wandb_entity_from_env(*, fallback: str = DEFAULT_WANDB_ENTITY) -> str:
    entity = str(os.environ.get("WANDB_ENTITY") or "").strip()
    return entity or fallback


def default_wandb_project_path(project: str | None = None) -> str:
    load_wandb_env()
    project_name = str(project or DEFAULT_WANDB_PROJECT).strip() or DEFAULT_WANDB_PROJECT
    return f"{wandb_entity_from_env()}/{project_name}"


def canonical_wandb_environment(
    env_provider: object,
    env_id: object,
    *,
    fallback: str = DEFAULT_WANDB_PROJECT,
) -> tuple[str, str]:
    """Return the canonical W&B project and provider-neutral game family."""

    project = wandb_project_for_environment(
        env_provider,
        env_id,
        fallback=fallback,
    )
    family = game_family_for_environment(
        env_provider,
        env_id,
        fallback=project,
    )
    return project, family


def wandb_project_for_env_id(
    env_id: str | None,
    *,
    env_provider: object = None,
    fallback: str = DEFAULT_WANDB_PROJECT,
) -> str:
    """Return the default W&B project name for a provider-local environment id."""

    return canonical_wandb_environment(env_provider, env_id, fallback=fallback)[0]


def resolve_wandb_project(
    explicit_project: object,
    env_id: str | None,
    *,
    env_provider: object = None,
) -> str:
    """Use explicit W&B project when supplied, otherwise default to the env id."""

    project = str(explicit_project or "").strip()
    return project or wandb_project_for_env_id(env_id, env_provider=env_provider)


def resolve_wandb_namespace(
    explicit_entity: object,
    explicit_project: object,
    env_id: str | None,
    *,
    env_provider: object = None,
) -> tuple[str, str]:
    entity = str(explicit_entity or "").strip() or wandb_entity_from_env()
    project = resolve_wandb_project(
        explicit_project,
        env_id,
        env_provider=env_provider,
    )
    return entity, project


def configure_wandb_metrics(run):
    if run is not None:
        run.define_metric("global_step", summary="max")
        run.define_metric("train/global_step", summary="max")
        run.define_metric("eval/checkpoint_step", summary="max")
        run.define_metric("orchestration/event_seq", summary="max")
        run.define_metric(
            "train/*",
            step_metric="train/global_step",
        )
        run.define_metric(
            "eval/*",
            step_metric="eval/checkpoint_step",
        )
        run.define_metric(
            "orchestration/*",
            step_metric="orchestration/event_seq",
        )
        run.define_metric(
            EVAL_ACCEPTANCE_PASS,
            step_metric="eval/checkpoint_step",
            summary="max",
        )
        run.define_metric(
            TRAIN_EPISODE_RETURN_SHAPED_MEAN,
            step_metric="train/global_step",
            summary="last",
        )
        run.define_metric("*", step_metric="orchestration/event_seq")
    return run
