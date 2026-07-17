from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rlab.training.sb3_on_policy import (
    checkpoint_save_frequency as checkpoint_save_frequency,
    normalize_on_policy_config,
    policy_kwargs_from_args,
    policy_name_for_observation_space,
    run_sb3_on_policy,
    validate_action_space as validate_on_policy_action_space,
)
from rlab.training_backend import BackendContext


DEFAULT_CONFIG: dict[str, Any] = {
    "learning_rate": 1e-4,
    "learning_rate_final": None,
    "learning_rate_schedule_timesteps": 0,
    "n_steps": 512,
    "batch_size": 256,
    "n_epochs": 10,
    "device": "auto",
    "gamma": 0.9,
    "gae_lambda": 1.0,
    "ent_coef": 0.01,
    "ent_coef_final": None,
    "ent_coef_schedule_timesteps": 0,
    "vf_coef": 1.0,
    "clip_range": 0.2,
    "clip_range_vf": None,
    "policy_net_arch": "",
    "value_net_arch": "",
    "normalize_advantage": False,
    "advantage_normalization": "auto",
    "adam_eps": 1e-8,
    "target_kl": None,
    "resume": None,
}

_INTEGER_FIELDS = {
    "batch_size",
    "n_epochs",
}
_NUMBER_FIELDS = {
    "clip_range",
    "clip_range_vf",
    "adam_eps",
    "target_kl",
}


def normalize_config(config: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    normalized = normalize_on_policy_config(config, defaults=DEFAULT_CONFIG, label=label)
    for key in _INTEGER_FIELDS:
        value = normalized[key]
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{label}.{key} must be an integer")
    for key in ("batch_size", "n_epochs"):
        if normalized[key] <= 0:
            raise ValueError(f"{label}.{key} must be positive")
    for key in _NUMBER_FIELDS:
        value = normalized[key]
        if value is None:
            continue
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise ValueError(f"{label}.{key} must be a number or null")
    if normalized["advantage_normalization"] not in {"auto", "none", "global", "per-task"}:
        raise ValueError(
            f"{label}.advantage_normalization must be one of auto, none, global, per-task"
        )
    return normalized


def validate_action_space(action_space) -> None:
    validate_on_policy_action_space(action_space, algorithm_id="ppo")


def _model_factory(context: BackendContext, env: Any, config: Any, device: str):
    from stable_baselines3 import PPO

    from rlab.env import task_conditioning
    from rlab.schedules import apply_resume_hyperparameters, learning_rate_schedule
    from rlab.task_advantage import PerTaskAdvantagePPO, resolve_advantage_normalization_mode

    args = context.args
    advantage_normalization = resolve_advantage_normalization_mode(args)
    if advantage_normalization == "per-task" and not task_conditioning(config).get("enabled"):
        raise ValueError("per-task advantage normalization requires task conditioning")
    sb3_normalize_advantage = advantage_normalization == "global"
    if args.resume:
        model = PPO.load(
            args.resume,
            env=env,
            tensorboard_log=str(context.run_dir),
            device=device,
        )
        if advantage_normalization == "per-task":
            raise ValueError("per-task advantage normalization is not supported with resume")
        apply_resume_hyperparameters(model, args)
        model.normalize_advantage = sb3_normalize_advantage
        return model

    model_cls = PerTaskAdvantagePPO if advantage_normalization == "per-task" else PPO
    return model_cls(
        policy_name_for_observation_space(env.observation_space),
        env,
        learning_rate=learning_rate_schedule(args),
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        clip_range=args.clip_range,
        clip_range_vf=args.clip_range_vf,
        normalize_advantage=sb3_normalize_advantage,
        target_kl=args.target_kl,
        policy_kwargs=policy_kwargs_from_args(args, optimizer_eps=args.adam_eps),
        tensorboard_log=str(context.run_dir),
        device=device,
        verbose=1,
    )


class Sb3PpoBackend:
    def validate(
        self,
        common_config: Mapping[str, Any],
        backend_config: Mapping[str, Any],
    ) -> None:
        del common_config
        normalize_config(backend_config, label="training_backend.config")

    def run(self, context: BackendContext) -> None:
        run_sb3_on_policy(context, algorithm_id="ppo", model_factory=_model_factory)


_BACKEND = Sb3PpoBackend()


def backend_for_id(backend_id: str) -> Sb3PpoBackend:
    if backend_id != "sb3.ppo":
        raise ValueError(f"SB3 PPO backend module does not define {backend_id!r}")
    return _BACKEND


def contract_payload(backend_id: str) -> dict[str, Any]:
    if backend_id != "sb3.ppo":
        raise ValueError(f"SB3 PPO backend module does not define {backend_id!r}")
    return {"schema_version": 1, "status": "available", "defaults": DEFAULT_CONFIG}


def runtime_metadata(
    backend_id: str,
    backend_config: Mapping[str, Any],
) -> Mapping[str, str]:
    if backend_id != "sb3.ppo":
        raise ValueError(f"SB3 PPO backend module does not define {backend_id!r}")
    model_class = (
        "rlab.task_advantage.PerTaskAdvantagePPO"
        if backend_config.get("advantage_normalization") == "per-task"
        else "stable_baselines3.ppo.ppo.PPO"
    )
    return {
        "training_backend_id": backend_id,
        "algorithm_id": "ppo",
        "model_class": model_class,
    }
