from __future__ import annotations

import re
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from gymnasium import spaces

from rlab.artifacts import write_model_metadata
from rlab.snapshot_curriculum import snapshot_curriculum_artifact_summary
from rlab.training_backend import BackendContext


ModelFactory = Callable[[BackendContext, Any, Any, str], Any]

_INTEGER_FIELDS = (
    "learning_rate_schedule_timesteps",
    "n_steps",
    "ent_coef_schedule_timesteps",
)
_NON_NEGATIVE_INTEGER_FIELDS = (
    "learning_rate_schedule_timesteps",
    "ent_coef_schedule_timesteps",
)
_NUMBER_FIELDS = (
    "learning_rate",
    "learning_rate_final",
    "gamma",
    "gae_lambda",
    "ent_coef",
    "ent_coef_final",
    "vf_coef",
)


def normalize_on_policy_config(
    config: Mapping[str, Any],
    *,
    defaults: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    unexpected = sorted(set(config) - set(defaults))
    if unexpected:
        raise ValueError(f"{label} has unexpected fields: {unexpected}")
    normalized = {**defaults, **dict(config)}
    for key in _INTEGER_FIELDS:
        value = normalized[key]
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{label}.{key} must be an integer")
    if normalized["n_steps"] <= 0:
        raise ValueError(f"{label}.n_steps must be positive")
    for key in _NON_NEGATIVE_INTEGER_FIELDS:
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
    if not isinstance(normalized["normalize_advantage"], bool):
        raise ValueError(f"{label}.normalize_advantage must be a boolean")
    resume = normalized["resume"]
    if resume is not None and not isinstance(resume, str):
        raise ValueError(f"{label}.resume must be a string or null")
    approval = normalized["resume_approval_hash"]
    manifest = normalized["resume_manifest"]
    if resume is None:
        if approval is not None or manifest is not None:
            raise ValueError(f"{label} resume approval fields require resume")
    elif (
        not isinstance(approval, str)
        or not re.fullmatch(r"[0-9a-f]{64}", approval)
        or not isinstance(manifest, list)
        or not manifest
    ):
        raise ValueError(f"{label}.resume requires a pinned approval hash and byte manifest")
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


def validate_action_space(action_space, *, algorithm_id: str) -> None:
    supported = (spaces.Box, spaces.Discrete, spaces.MultiBinary, spaces.MultiDiscrete)
    if not isinstance(action_space, supported):
        raise ValueError(
            f"SB3 {algorithm_id.upper()} does not support action space "
            f"{type(action_space).__name__}; configure a task action codec or choose another backend"
        )


def checkpoint_prefix(game: str, *, algorithm_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", game).strip("_").lower()
    return f"{algorithm_id}_{slug or 'retro'}"


def parse_net_arch(value: str) -> list[int]:
    layers = [int(part.strip()) for part in str(value).split(",") if part.strip()]
    if any(size <= 0 for size in layers):
        raise ValueError("policy/value network layer sizes must be positive")
    return layers


def policy_kwargs_from_args(args, *, optimizer_eps: float | None = None) -> dict[str, object]:
    policy_kwargs: dict[str, object] = {}
    if optimizer_eps is not None:
        policy_kwargs["optimizer_kwargs"] = {"eps": optimizer_eps}
    pi_arch = parse_net_arch(args.policy_net_arch)
    vf_arch = parse_net_arch(args.value_net_arch)
    if pi_arch or vf_arch:
        policy_kwargs["net_arch"] = {"pi": pi_arch, "vf": vf_arch}
    return policy_kwargs


def checkpoint_save_frequency(checkpoint_freq: int, n_envs: int) -> int | None:
    if checkpoint_freq <= 0:
        return None
    return max(checkpoint_freq // max(n_envs, 1), 1)


def save_model_bundle(
    *,
    model,
    context: BackendContext,
    model_path: Path,
    kind: str,
    step: int | None,
) -> Path:
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
        snapshot_curriculum_session=snapshot_curriculum_artifact_summary(
            getattr(model, "env", None)
        ),
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


def run_sb3_on_policy(
    context: BackendContext,
    *,
    algorithm_id: str,
    model_factory: ModelFactory,
) -> None:
    from stable_baselines3.common.utils import set_random_seed

    from rlab.callbacks import (
        LedgerCheckpointHelper,
        MetricStoreLoggerHelper,
        MetricThresholdStopHelper,
        RlabCallback,
        RolloutDiagnosticsHelper,
        RuntimeMetricsHelper,
        SnapshotCurriculumFeedbackHelper,
        ThroughputHelper,
    )
    from rlab.device import resolve_sb3_device
    from rlab.env import (
        make_training_vec_env,
        preflight_snapshot_curriculum_provider,
        task_termination,
    )
    from rlab.file_utils import file_sha256
    from rlab.metric_store import metric_store_path
    from rlab.policy_bundle import write_canonical_json
    from rlab.schedules import EntropyCoefficientScheduleHelper
    from rlab.training.sb3_helpers import GracefulStopHelper, Sb3HumanOutputFormatHelper

    args = context.args
    config = context.environment
    n_envs = int(args.resolved_n_envs)
    preflight = preflight_snapshot_curriculum_provider(
        config=config,
        n_envs=n_envs,
        seed=args.seed,
        rom_binding=getattr(context, "rom_binding", None),
        snapshot_curriculum=getattr(args, "snapshot_curriculum", None),
    )
    if preflight is not None:
        preflight_path = context.run_dir / "snapshot_curriculum_preflight.json"
        write_canonical_json(preflight_path, preflight)
        args.snapshot_curriculum_preflight_sha256 = file_sha256(preflight_path)
        print(
            "snapshot curriculum provider preflight passed: "
            f"provider={preflight['provider_id']} cell={preflight['cell_id']} "
            f"lanes={preflight['preflight_lanes']}",
            flush=True,
        )
    env = make_training_vec_env(
        config=config,
        n_envs=n_envs,
        seed=args.seed,
        rom_binding=getattr(context, "rom_binding", None),
        snapshot_curriculum=getattr(args, "snapshot_curriculum", None),
    )
    try:
        store_path = metric_store_path(context.run_dir)
        set_random_seed(args.seed)
        validate_action_space(env.action_space, algorithm_id=algorithm_id)
        device = resolve_sb3_device(args.device)
        print(f"Using torch device: {device}", flush=True)
        model = model_factory(context, env, config, device)

        components: list[Any] = [
            GracefulStopHelper(
                context.stop_flag,
                marker_path=context.run_dir / "learner_stop_observed.json",
            ),
            Sb3HumanOutputFormatHelper(),
            ThroughputHelper(
                metric_store_path=store_path,
                wandb_enabled=context.wandb_enabled,
            ),
            RuntimeMetricsHelper(
                event_names=tuple(task_termination(config).get("failure", ())),
                active_reward_components=active_reward_components(config.task),
                active_reward_signals=active_reward_signals(config.task),
                configured_starts=tuple(config.states or ((config.state,) if config.state else ())),
                track_success=bool(
                    isinstance(config.task.get("termination"), Mapping)
                    and config.task["termination"].get("success")
                ),
                metrics_schema_version=int(args.metrics_schema_version),
            ),
        ]
        if getattr(args, "snapshot_curriculum", None) is not None:
            components.append(SnapshotCurriculumFeedbackHelper())
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
                    algorithm_id=algorithm_id,
                    metric_store_path=store_path,
                    wandb_enabled=context.wandb_enabled,
                    histogram_interval=64,
                ),
                MetricStoreLoggerHelper(
                    store_path,
                    algorithm_id=algorithm_id,
                    wandb_enabled=context.wandb_enabled,
                ),
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
                    name_prefix=checkpoint_prefix(config.game, algorithm_id=algorithm_id),
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
                    algorithm_id=algorithm_id,
                )
            )
        if args.checkpoint_eval_backend == "none":
            print(
                "checkpoint evaluation disabled; this run cannot establish promotion or acceptance"
            )
        else:
            print("training-loop eval disabled; async checkpoint eval handles promotion metrics")
        callback = RlabCallback(components)
        context.mark_ready()

        final_model_path = context.run_dir / "final_model.zip"
        model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=True)
        save_model_bundle(
            model=model,
            context=context,
            model_path=final_model_path,
            kind="final",
            step=model.num_timesteps,
        )
        print(f"saved {final_model_path}")
    finally:
        env.close()
