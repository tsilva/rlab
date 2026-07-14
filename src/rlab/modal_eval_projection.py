from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

from rlab.artifacts import wandb_artifact_collection_name
from rlab.checkpoint_eval_worker import log_checkpoint_eval_metrics
from rlab.env import resolve_env_config
from rlab.env_metadata import env_config_from_config_dict
from rlab.wandb_utils import configure_wandb_metrics, load_wandb_env, resolve_wandb_namespace


def project_payload(payload: Mapping[str, Any]) -> None:
    train_config = dict(payload["train_config"])
    if not bool(train_config.get("wandb", False)):
        return
    run_id = str(train_config.get("wandb_run_id") or "")
    if not run_id:
        raise ValueError("Modal eval projection requires the producing W&B run id")
    load_wandb_env()
    import wandb

    entity, project = resolve_wandb_namespace(
        train_config.get("wandb_entity"),
        train_config.get("wandb_project"),
        str(train_config.get("game") or ""),
        env_provider=train_config.get("env_provider"),
    )
    run = configure_wandb_metrics(
        wandb.init(
            entity=entity,
            project=project,
            id=run_id,
            resume="must",
            mode=str(train_config.get("wandb_mode") or "online"),
        )
    )
    try:
        projection_kind = str(payload.get("projection_kind") or "evaluation")
        if projection_kind == "artifact_reference":
            if bool(train_config.get("no_wandb_artifacts", False)):
                return
            kind = str(payload["artifact_kind"])
            checkpoint_step = int(payload["checkpoint_step"])
            model_metadata = dict(payload["model_metadata"])
            if not isinstance(model_metadata.get("training_metadata"), dict):
                raise ValueError("artifact projection requires checkpoint training_metadata")
            checkpoint_uri = str(payload["checkpoint_uri"])
            artifact_filename = checkpoint_uri.rsplit("/", 1)[-1] or "model.zip"
            artifact_metadata = {
                **model_metadata,
                "source_filename": model_metadata.get("filename", ""),
                "filename": artifact_filename,
                "checkpoint_sha256": payload["checkpoint_sha256"],
                "metadata_sha256": payload["metadata_sha256"],
                "checkpoint_step": checkpoint_step,
                "metadata_uri": payload["metadata_uri"],
                "artifact_storage_uri": checkpoint_uri,
            }
            artifact = wandb.Artifact(
                wandb_artifact_collection_name(
                    kind,
                    run_id=run_id,
                    run_name=train_config.get("run_name"),
                ),
                type="model",
                metadata=artifact_metadata,
            )
            artifact.add_reference(checkpoint_uri)
            aliases = [kind, "latest"]
            if kind == "checkpoint":
                aliases = ["latest", f"step-{checkpoint_step}"]
            run.log_artifact(artifact, aliases=aliases)
            return
        decision = dict(payload["decision"])
        purpose = str(payload["purpose"])
        checkpoint_uri = str(payload["checkpoint_uri"])
        if purpose == "promotion":
            environment = train_config.get("checkpoint_eval_environment")
            if not isinstance(environment, dict):
                raise ValueError("Modal eval projection is missing the materialized environment")
            config = env_config_from_config_dict(environment)
            if config is None:
                raise ValueError("Modal eval projection environment is invalid")
            args = SimpleNamespace(**train_config)
            log_checkpoint_eval_metrics(
                run,
                args=args,
                metrics=dict(decision["raw_metrics"]),
                checkpoint_path=checkpoint_uri,
                checkpoint_step_value=int(payload["checkpoint_step"]),
                artifact_ref=checkpoint_uri,
                eval_source="modal",
                config=resolve_env_config(config),
                update_leader=bool(payload.get("canonical_promotion", False)),
                force_leader=bool(payload.get("canonical_promotion", False)),
            )
        else:
            run.log(dict(decision["metrics"]))
    finally:
        run.finish()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Project accepted Modal eval evidence to W&B")
    parser.add_argument("--payload", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = json.loads(args.payload.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("projection payload must be a mapping")
    project_payload(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
