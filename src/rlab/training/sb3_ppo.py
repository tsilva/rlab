from __future__ import annotations

import re
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from gymnasium import spaces

from rlab.artifacts import write_model_metadata
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
    "learning_rate_schedule_timesteps",
    "n_steps",
    "batch_size",
    "n_epochs",
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
    "clip_range",
    "clip_range_vf",
    "adam_eps",
    "target_kl",
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
    for key in ("n_steps", "batch_size", "n_epochs"):
        if normalized[key] <= 0:
            raise ValueError(f"{label}.{key} must be positive")
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
    if normalized["advantage_normalization"] not in {"auto", "none", "global", "per-task"}:
        raise ValueError(
            f"{label}.advantage_normalization must be one of auto, none, global, per-task"
        )
    for key in ("policy_net_arch", "value_net_arch"):
        if not isinstance(normalized[key], str):
            raise ValueError(f"{label}.{key} must be a string")
    if not isinstance(normalized["normalize_advantage"], bool):
        raise ValueError(f"{label}.normalize_advantage must be a boolean")
    resume = normalized["resume"]
    if resume is not None and not isinstance(resume, str):
        raise ValueError(f"{label}.resume must be a string or null")
    return normalized


def active_reward_components(task: Mapping[str, object]) -> tuple[str, ...]:
    reward = task.get("reward")
    if not isinstance(reward, Mapping):
        return ()
    components: list[str] = []
    reward_mode = str(reward.get("reward_mode") or "")
    if reward_mode == "native" or bool(reward.get("use_native_reward")):
        components.append("native")
    if float(reward.get("progress_reward_scale") or 0.0) != 0.0:
        components.append("progress")
    if reward_mode == "score":
        components.append("score")
    if (
        float(reward.get("terminal_reward") or 0.0) != 0.0
        or float(reward.get("completion_reward") or 0.0) != 0.0
    ):
        components.append("completion")
    if float(reward.get("death_penalty") or 0.0) != 0.0:
        components.append("death")
    if float(reward.get("time_penalty") or 0.0) != 0.0:
        components.append("time")
    return tuple(components)


def active_reward_signals(task: Mapping[str, object]) -> tuple[str, ...]:
    components = set(active_reward_components(task))
    return tuple(name for name in ("progress", "score") if name in components)


def policy_name_for_observation_space(observation_space) -> str:
    if isinstance(observation_space, spaces.Dict):
        return "MultiInputPolicy"
    if isinstance(observation_space, spaces.Box) and len(observation_space.shape) == 3:
        return "CnnPolicy"
    return "MlpPolicy"


def validate_action_space(action_space) -> None:
    supported = (spaces.Box, spaces.Discrete, spaces.MultiBinary, spaces.MultiDiscrete)
    if not isinstance(action_space, supported):
        raise ValueError(
            f"SB3 PPO does not support action space {type(action_space).__name__}; "
            "configure a task action codec or choose another backend"
        )


def checkpoint_prefix(game: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", game).strip("_").lower()
    return f"ppo_{slug or 'retro'}"


def parse_net_arch(value: str) -> list[int]:
    layers = [int(part.strip()) for part in str(value).split(",") if part.strip()]
    if any(size <= 0 for size in layers):
        raise ValueError("policy/value network layer sizes must be positive")
    return layers


def policy_kwargs_from_args(args) -> dict[str, object]:
    policy_kwargs: dict[str, object] = {"optimizer_kwargs": {"eps": args.adam_eps}}
    pi_arch = parse_net_arch(args.policy_net_arch)
    vf_arch = parse_net_arch(args.value_net_arch)
    if pi_arch or vf_arch:
        policy_kwargs["net_arch"] = {"pi": pi_arch, "vf": vf_arch}
    return policy_kwargs


def checkpoint_save_frequency(checkpoint_freq: int, n_envs: int) -> int | None:
    if checkpoint_freq <= 0:
        return None
    return max(checkpoint_freq // max(n_envs, 1), 1)


def save_model_bundle(*, model, context: BackendContext, model_path: Path, kind: str, step: int | None) -> Path:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = model_path.parent / f".{model_path.stem}.{uuid.uuid4().hex}.zip"
    model.save(str(temp_path))
    temp_path.replace(model_path)
    metadata_path = write_model_metadata(
        model_path,
        context.args,
        context.environment,
        kind,
        checkpoint_step_value=step,
    )
    checkpoint_id = context.metric_store.record_checkpoint(
        run_name=str(context.args.run_name),
        kind=kind,
        step=step,
        path=model_path,
        metadata_path=metadata_path,
        sha256=None,
        eval_required=context.args.checkpoint_eval_backend != "none",
    )
    print(f"{kind} model ready: id={checkpoint_id} step={step} path={model_path}", flush=True)
    return model_path


class Sb3PpoBackend:
    def validate(
        self,
        common_config: Mapping[str, Any],
        backend_config: Mapping[str, Any],
    ) -> None:
        del common_config
        normalize_config(backend_config, label="training_backend.config")

    def run(self, context: BackendContext) -> None:
        from rlab.env import make_training_vec_env

        args = context.args
        env = make_training_vec_env(
            config=context.environment,
            n_envs=int(args.resolved_n_envs),
            seed=args.seed,
        )
        try:
            self._run_with_env(context, env)
        finally:
            env.close()

    def _run_with_env(self, context: BackendContext, env: Any) -> None:
        from stable_baselines3 import PPO
        from stable_baselines3.common.utils import set_random_seed

        from rlab.callbacks import (
            LedgerCheckpointHelper,
            MetricStoreLoggerHelper,
            MetricThresholdStopHelper,
            RlabCallback,
            RolloutDiagnosticsHelper,
            RuntimeMetricsHelper,
            ThroughputHelper,
        )
        from rlab.device import resolve_sb3_device
        from rlab.env import task_conditioning, task_termination
        from rlab.metric_store import metric_store_path
        from rlab.schedules import (
            EntropyCoefficientScheduleHelper,
            apply_resume_hyperparameters,
            learning_rate_schedule,
        )
        from rlab.task_advantage import (
            PerTaskAdvantagePPO,
            resolve_advantage_normalization_mode,
        )
        from rlab.training.sb3_helpers import (
            GracefulStopHelper,
            Sb3HumanOutputFormatHelper,
        )

        args = context.args
        config = context.environment
        n_envs = int(args.resolved_n_envs)
        store_path = metric_store_path(context.run_dir)
        set_random_seed(args.seed)
        validate_action_space(env.action_space)
        device = resolve_sb3_device(args.device)
        print(f"Using torch device: {device}", flush=True)

        advantage_normalization = resolve_advantage_normalization_mode(args)
        if advantage_normalization == "per-task" and not task_conditioning(config).get("enabled"):
            env.close()
            raise ValueError("per-task advantage normalization requires task conditioning")
        sb3_normalize_advantage = advantage_normalization == "global"
        if args.resume:
            model = PPO.load(args.resume, env=env, tensorboard_log=str(context.run_dir), device=device)
            if advantage_normalization == "per-task":
                env.close()
                raise ValueError("per-task advantage normalization is not supported with resume")
            apply_resume_hyperparameters(model, args)
            model.normalize_advantage = sb3_normalize_advantage
        else:
            model_cls = PerTaskAdvantagePPO if advantage_normalization == "per-task" else PPO
            model = model_cls(
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
                policy_kwargs=policy_kwargs_from_args(args),
                tensorboard_log=str(context.run_dir),
                device=device,
                verbose=1,
            )

        components: list[Any] = [
            GracefulStopHelper(context.stop_flag),
            Sb3HumanOutputFormatHelper(),
            ThroughputHelper(metric_store_path=store_path, wandb_run=context.wandb_run),
            RuntimeMetricsHelper(
                event_names=tuple(task_termination(config).get("failure", ())),
                active_reward_components=active_reward_components(config.task),
                active_reward_signals=active_reward_signals(config.task),
                configured_starts=tuple(config.states or ((config.state,) if config.state else ())),
                track_success=bool(
                    isinstance(config.task.get("termination"), Mapping)
                    and config.task["termination"].get("success")
                ),
            ),
        ]
        if args.early_stop:
            components.append(
                MetricThresholdStopHelper(
                    marker_path=context.run_dir / "early_stop.txt",
                    detector=args.early_stop,
                    metric_store_path=store_path,
                )
            )
        components.extend(
            [
                RolloutDiagnosticsHelper(
                    wandb_run=context.wandb_run,
                    metric_store_path=store_path if context.external_wandb_publisher else None,
                    histogram_interval=64,
                ),
                MetricStoreLoggerHelper(store_path, wandb_run=context.wandb_run),
            ]
        )
        checkpoint_save_freq = checkpoint_save_frequency(args.checkpoint_freq, n_envs)
        if checkpoint_save_freq is not None:
            components.append(
                LedgerCheckpointHelper(
                    args=args,
                    config=config,
                    save_freq=checkpoint_save_freq,
                    save_path=str(context.checkpoint_dir),
                    name_prefix=checkpoint_prefix(config.game),
                    metric_store_path=store_path,
                    eval_required=args.checkpoint_eval_backend != "none",
                )
            )
        if args.ent_coef_final is not None:
            components.append(
                EntropyCoefficientScheduleHelper(
                    initial_value=args.ent_coef,
                    final_value=args.ent_coef_final,
                    schedule_timesteps=(
                        args.ent_coef_schedule_timesteps
                        if args.ent_coef_schedule_timesteps > 0
                        else args.timesteps
                    ),
                )
            )
        if args.checkpoint_eval_backend == "none":
            print("checkpoint evaluation disabled; this run cannot establish promotion or acceptance")
        else:
            print("training-loop eval disabled; async checkpoint eval handles promotion metrics")
        callback = RlabCallback(components)
        context.mark_ready()

        final_model_path = context.run_dir / "final_model.zip"
        try:
            model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=True)
            if context.stop_flag.requested and checkpoint_save_freq is not None:
                interrupted = context.checkpoint_dir / (
                    f"{checkpoint_prefix(config.game)}_interrupted_"
                    f"{model.num_timesteps}_steps.zip"
                )
                save_model_bundle(
                    model=model,
                    context=context,
                    model_path=interrupted,
                    kind="interrupted",
                    step=model.num_timesteps,
                )
            save_model_bundle(
                model=model,
                context=context,
                model_path=final_model_path,
                kind="final",
                step=model.num_timesteps,
            )
        finally:
            env.close()
        print(f"saved {final_model_path}")


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
