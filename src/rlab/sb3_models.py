from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from rlab.artifacts import load_model_metadata


Sb3AlgorithmId = Literal["ppo", "a2c"]

_BACKEND_ALGORITHMS: dict[str, Sb3AlgorithmId] = {
    "sb3.ppo": "ppo",
    "sb3.a2c": "a2c",
}
_MODEL_CLASS_ALGORITHMS: dict[str, Sb3AlgorithmId] = {
    "stable_baselines3.ppo.ppo.PPO": "ppo",
    "rlab.task_advantage.PerTaskAdvantagePPO": "ppo",
    "stable_baselines3.a2c.a2c.A2C": "a2c",
}


def resolve_sb3_algorithm(metadata: Mapping[str, Any] | None) -> Sb3AlgorithmId:
    metadata = metadata or {}
    resolved: set[Sb3AlgorithmId] = set()

    backend_id = str(metadata.get("training_backend_id") or "").strip()
    if backend_id:
        algorithm_id = _BACKEND_ALGORITHMS.get(backend_id)
        if algorithm_id is None:
            raise ValueError(f"unsupported checkpoint training backend: {backend_id}")
        resolved.add(algorithm_id)

    algorithm_value = str(metadata.get("algorithm_id") or "").strip()
    if algorithm_value:
        if algorithm_value not in {"ppo", "a2c"}:
            raise ValueError(f"unsupported checkpoint algorithm: {algorithm_value}")
        resolved.add(algorithm_value)

    model_class = str(metadata.get("model_class") or "").strip()
    if model_class:
        algorithm_id = _MODEL_CLASS_ALGORITHMS.get(model_class)
        if algorithm_id is None:
            raise ValueError(f"unsupported checkpoint model class: {model_class}")
        resolved.add(algorithm_id)

    if len(resolved) > 1:
        raise ValueError("checkpoint backend, algorithm, and model class metadata disagree")
    return next(iter(resolved), "ppo")


def load_sb3_model(
    model_path: str | Path,
    *,
    device: str,
    env: Any | None = None,
    tensorboard_log: str | None = None,
    metadata: Mapping[str, Any] | None = None,
):
    path = Path(model_path)
    resolved_metadata = load_model_metadata(path) if metadata is None else dict(metadata)
    algorithm_id = resolve_sb3_algorithm(resolved_metadata)
    if algorithm_id == "a2c":
        from stable_baselines3 import A2C

        model_class = A2C
    else:
        from stable_baselines3 import PPO

        model_class = PPO
    kwargs: dict[str, Any] = {"device": device}
    if env is not None:
        kwargs["env"] = env
    if tensorboard_log is not None:
        kwargs["tensorboard_log"] = tensorboard_log
    return model_class.load(str(path), **kwargs)
