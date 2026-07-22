from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from rlab.artifacts import load_model_metadata
from rlab.trusted_inputs import (
    ApprovedModelInput,
    approve_internal_model,
    stage_and_approve_model,
)


PolicyAlgorithmId = Literal["ppo", "a2c", "jerk"]

_BACKEND_ALGORITHMS: dict[str, PolicyAlgorithmId] = {
    "sb3.ppo": "ppo",
    "sb3.a2c": "a2c",
    "rlab.jerk": "jerk",
}
_MODEL_CLASS_ALGORITHMS: dict[str, PolicyAlgorithmId] = {
    "stable_baselines3.ppo.ppo.PPO": "ppo",
    "rlab.task_advantage.PerTaskAdvantagePPO": "ppo",
    "stable_baselines3.a2c.a2c.A2C": "a2c",
    "rlab.jerk.JerkPolicy": "jerk",
}


def resolve_policy_algorithm(metadata: Mapping[str, Any] | None) -> PolicyAlgorithmId:
    metadata = metadata or {}
    resolved: set[PolicyAlgorithmId] = set()
    backend_id = str(metadata.get("training_backend_id") or "").strip()
    if backend_id:
        algorithm = _BACKEND_ALGORITHMS.get(backend_id)
        if algorithm is None:
            raise ValueError(f"unsupported checkpoint training backend: {backend_id}")
        resolved.add(algorithm)
    algorithm_id = str(metadata.get("algorithm_id") or "").strip()
    if algorithm_id:
        if algorithm_id not in {"ppo", "a2c", "jerk"}:
            raise ValueError(f"unsupported checkpoint algorithm: {algorithm_id}")
        resolved.add(algorithm_id)  # type: ignore[arg-type]
    model_class = str(metadata.get("model_class") or "").strip()
    if model_class:
        algorithm = _MODEL_CLASS_ALGORITHMS.get(model_class)
        if algorithm is None:
            raise ValueError(f"unsupported checkpoint model class: {model_class}")
        resolved.add(algorithm)
    if len(resolved) > 1:
        raise ValueError("checkpoint backend, algorithm, and model class metadata disagree")
    return next(iter(resolved), "ppo")


def load_policy_model(
    model_input: ApprovedModelInput,
    *,
    device: str,
    env: Any | None = None,
    tensorboard_log: str | None = None,
    metadata: Mapping[str, Any] | None = None,
):
    if not isinstance(model_input, ApprovedModelInput):
        raise TypeError("load_policy_model requires an ApprovedModelInput")
    model_input.verify()
    path = model_input.model_path
    resolved_metadata = load_model_metadata(path) if metadata is None else dict(metadata)
    algorithm_id = resolve_policy_algorithm(resolved_metadata)
    if algorithm_id == "jerk":
        from rlab.jerk import JerkPolicy

        return JerkPolicy.load(path)
    from rlab.sb3_models import load_sb3_model

    return load_sb3_model(
        model_input,
        device=device,
        env=env,
        tensorboard_log=tensorboard_log,
        metadata=resolved_metadata,
    )


def load_external_policy_model(
    model_path: str | Path,
    *,
    device: str,
    env: Any | None = None,
    tensorboard_log: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    source_identity: str | None = None,
    approval_hash: str | None = None,
):
    with stage_and_approve_model(
        model_path,
        source_identity=source_identity,
        expected_hash=approval_hash,
    ) as approved:
        return load_policy_model(
            approved,
            device=device,
            env=env,
            tensorboard_log=tensorboard_log,
            metadata=metadata,
        )


def load_internal_policy_model(
    model_path: str | Path,
    *,
    execution_id: str,
    device: str,
    env: Any | None = None,
    tensorboard_log: str | None = None,
    metadata: Mapping[str, Any] | None = None,
):
    with approve_internal_model(model_path, execution_id=execution_id) as approved:
        return load_policy_model(
            approved,
            device=device,
            env=env,
            tensorboard_log=tensorboard_log,
            metadata=metadata,
        )


def load_pinned_remote_policy_model(
    source: str,
    *,
    download_root: Path,
    approval_hash: str,
    manifest: Any,
    device: str,
    env: Any | None = None,
    tensorboard_log: str | None = None,
    metadata: Mapping[str, Any] | None = None,
):
    from rlab.model_sources import download_remote_model_source
    from rlab.trusted_inputs import approve_staged_model, stage_model_input

    resolved = download_remote_model_source(source, root=download_root, require_pinned=True)
    staged = stage_model_input(resolved.model_path, source_identity=source)
    try:
        expected_manifest = list(manifest) if isinstance(manifest, list | tuple) else None
        actual_manifest = [entry.as_dict() for entry in staged.manifest]
        if expected_manifest is None or actual_manifest != expected_manifest:
            raise ValueError("pinned remote model byte manifest does not match the queued job")
        with approve_staged_model(
            staged,
            expected_hash=approval_hash,
            interactive=False,
        ) as approved:
            return load_policy_model(
                approved,
                device=device,
                env=env,
                tensorboard_log=tensorboard_log,
                metadata=metadata,
            )
    except Exception:
        staged.cleanup()
        raise
