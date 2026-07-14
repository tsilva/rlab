from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rlab.training.sb3_on_policy import (
    policy_kwargs_from_args,
    policy_name_for_observation_space,
    run_sb3_on_policy,
)
from rlab.training_backend import BackendContext


DEFAULT_CONFIG: dict[str, Any] = {
    "learning_rate": 7e-4,
    "learning_rate_final": None,
    "learning_rate_schedule_timesteps": 0,
    "n_steps": 5,
    "device": "auto",
    "gamma": 0.99,
    "gae_lambda": 1.0,
    "ent_coef": 0.0,
    "ent_coef_final": None,
    "ent_coef_schedule_timesteps": 0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "rms_prop_eps": 1e-5,
    "use_rms_prop": True,
    "policy_net_arch": "",
    "value_net_arch": "",
    "normalize_advantage": False,
    "resume": None,
}

_INTEGER_FIELDS = {
    "learning_rate_schedule_timesteps",
    "n_steps",
    "ent_coef_schedule_timesteps",
}
_NUMBER_FIELDS = {
    "learning_rate",
    "learning_rate_final",
    "gamma",
    "gae_lambda",
    "ent_coef",
    "ent_coef_final",
    "vf_coef",
    "max_grad_norm",
    "rms_prop_eps",
}


def normalize_config(config: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    unexpected = sorted(set(config) - set(DEFAULT_CONFIG))
    if unexpected:
        raise ValueError(f"{label} has unexpected fields: {unexpected}")
    normalized = {**DEFAULT_CONFIG, **dict(config)}
    for key in _INTEGER_FIELDS:
        value = normalized[key]
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{label}.{key} must be an integer")
    if normalized["n_steps"] <= 0:
        raise ValueError(f"{label}.n_steps must be positive")
    for key in ("learning_rate_schedule_timesteps", "ent_coef_schedule_timesteps"):
        if normalized[key] < 0:
            raise ValueError(f"{label}.{key} must be non-negative")
    for key in _NUMBER_FIELDS:
        value = normalized[key]
        if value is None:
            continue
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise ValueError(f"{label}.{key} must be a number or null")
    if normalized["device"] not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError(f"{label}.device must be one of auto, cpu, cuda, mps")
    for key in ("policy_net_arch", "value_net_arch"):
        if not isinstance(normalized[key], str):
            raise ValueError(f"{label}.{key} must be a string")
    for key in ("normalize_advantage", "use_rms_prop"):
        if not isinstance(normalized[key], bool):
            raise ValueError(f"{label}.{key} must be a boolean")
    resume = normalized["resume"]
    if resume is not None and not isinstance(resume, str):
        raise ValueError(f"{label}.resume must be a string or null")
    return normalized


def _model_factory(context: BackendContext, env: Any, config: Any, device: str):
    del config
    from stable_baselines3 import A2C

    from rlab.schedules import apply_a2c_resume_hyperparameters, learning_rate_schedule

    args = context.args
    if args.resume:
        model = A2C.load(
            args.resume,
            env=env,
            tensorboard_log=str(context.run_dir),
            device=device,
        )
        apply_a2c_resume_hyperparameters(model, args)
        return model

    return A2C(
        policy_name_for_observation_space(env.observation_space),
        env,
        learning_rate=learning_rate_schedule(args),
        n_steps=args.n_steps,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        rms_prop_eps=args.rms_prop_eps,
        use_rms_prop=args.use_rms_prop,
        normalize_advantage=args.normalize_advantage,
        policy_kwargs=policy_kwargs_from_args(args),
        tensorboard_log=str(context.run_dir),
        device=device,
        verbose=1,
    )


class Sb3A2cBackend:
    def validate(
        self,
        common_config: Mapping[str, Any],
        backend_config: Mapping[str, Any],
    ) -> None:
        del common_config
        normalize_config(backend_config, label="training_backend.config")

    def run(self, context: BackendContext) -> None:
        run_sb3_on_policy(context, algorithm_id="a2c", model_factory=_model_factory)


_BACKEND = Sb3A2cBackend()


def backend_for_id(backend_id: str) -> Sb3A2cBackend:
    if backend_id != "sb3.a2c":
        raise ValueError(f"SB3 A2C backend module does not define {backend_id!r}")
    return _BACKEND


def contract_payload(backend_id: str) -> dict[str, Any]:
    if backend_id != "sb3.a2c":
        raise ValueError(f"SB3 A2C backend module does not define {backend_id!r}")
    return {"schema_version": 1, "status": "available", "defaults": DEFAULT_CONFIG}


def runtime_metadata(
    backend_id: str,
    backend_config: Mapping[str, Any],
) -> Mapping[str, str]:
    del backend_config
    if backend_id != "sb3.a2c":
        raise ValueError(f"SB3 A2C backend module does not define {backend_id!r}")
    return {
        "training_backend_id": backend_id,
        "algorithm_id": "a2c",
        "model_class": "stable_baselines3.a2c.a2c.A2C",
    }
