from __future__ import annotations

import os
import multiprocessing
import time
import traceback
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any, Mapping

import numpy as np
import stable_retro as retro

from rlab import env_providers as provider_runtime
from rlab.action_contract import configured_action_values, normalize_action_configuration
from rlab.batch_runtime import BatchRuntime, ProviderDescriptor
from rlab.env_providers import (
    DEFAULT_RETRO_VEC_ENV as RetroVecEnv,
    ale_py_atari_vector_env_type as _ale_py_atari_vector_env_type,
    provider_descriptor,
    provider_native_vec_kwargs,
    super_mario_bros_nes_turbo_vec_env_type as _super_mario_bros_nes_turbo_vec_env_type,
)
from rlab.env_registry import (
    ALE_PY_PROVIDER,
    STABLE_RETRO_TURBO_PROVIDER,
    env_supports_states,
    qualify_env_id,
    resolve_env_provider,
)
from rlab.env_identity import task_config_from_train_config, validate_task_config
from rlab.targets import target_for_game
from rlab.task_kernels import IdentityTaskDefinition, MarioTaskConfig, MarioTaskDefinition
from rlab.validation import normalize_obs_crop as validate_obs_crop
from rlab.rom_runtime import RomRuntimeBinding

os.environ.setdefault("MPLCONFIGDIR", os.path.abspath(".matplotlib"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

GAME = os.environ.get("RETRO_GAME", "")
DEFAULT_OBS_RESIZE_ALGORITHM = "area"


@dataclass(frozen=True)
class EnvConfig:
    env_provider: str = STABLE_RETRO_TURBO_PROVIDER.provider_id
    game: str = GAME
    env_args: dict[str, Any] = field(default_factory=dict)
    task: dict[str, Any] = field(default_factory=dict)
    state: str = ""
    states: tuple[str, ...] = ()
    state_probs: tuple[float, ...] = ()
    frame_skip: int = 4
    max_pool_frames: bool = True
    sticky_action_prob: float = 0.0
    observation_size: int = 84
    hud_crop_top: int = -1
    obs_crop: tuple[int, int, int, int] | None = None
    obs_crop_mode: str = "remove"
    obs_crop_fill: int = 0
    obs_resize_algorithm: str = DEFAULT_OBS_RESIZE_ALGORITHM


def validate_obs_crop_mode(value: str) -> str:
    if value not in {"remove", "mask"}:
        raise ValueError("obs_crop_mode must be 'remove' or 'mask'")
    return value


def validate_obs_crop_fill(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 255:
        raise ValueError("obs_crop_fill must be an integer in [0, 255]")
    return int(value)


def native_obs_crop(config: EnvConfig) -> tuple[int, int, int, int] | None:
    obs_crop = validate_obs_crop(config.obs_crop)
    if obs_crop is not None:
        return obs_crop if any(obs_crop) else None
    if config.hud_crop_top > 0:
        return (config.hud_crop_top, 0, 0, 0)
    return None


def _validate_sticky_action_prob(value: float) -> float:
    if not 0.0 <= value <= 1.0:
        raise ValueError("sticky_action_prob must be in [0, 1]")
    return float(value)


def resolve_env_config(config: EnvConfig) -> EnvConfig:
    if not config.game and isinstance(config.env_args, Mapping) and config.env_args.get("game"):
        config = replace(config, game=str(config.env_args["game"]))
    if not config.game:
        raise ValueError("game is required; pass --game or set RETRO_GAME")
    normalized_args, normalized_task = normalize_action_configuration(
        provider_id=config.env_provider,
        game=config.game,
        env_args=config.env_args,
        task=config.task,
    )
    config = replace(config, env_args=normalized_args, task=normalized_task)
    qualify_env_id(config.env_provider, config.game)
    _validate_sticky_action_prob(config.sticky_action_prob)
    validate_obs_crop_mode(config.obs_crop_mode)
    validate_obs_crop_fill(config.obs_crop_fill)
    target = target_for_game(config.game)
    updates: dict[str, Any] = {}
    if not config.state and target.default_state:
        updates["state"] = target.default_state
    if config.obs_crop is None and config.hud_crop_top < 0:
        updates["hud_crop_top"] = target.default_hud_crop_top
    config = replace(config, **updates) if updates else config
    if config.task:
        validate_task_config(config.task)
        canonical_task = config.task
    else:
        canonical_task = task_config_from_train_config(
            {"env_provider": config.env_provider, "game": config.game}
        )
    return replace(config, task=canonical_task)


def _validate_state_names(game: str, states: tuple[str, ...]) -> None:
    if any(not state for state in states):
        raise ValueError("--states must not contain empty state names")
    valid_states = set(retro.data.list_states(game))
    unknown = [state for state in states if state not in valid_states]
    if unknown:
        valid_preview = ", ".join(sorted(valid_states)[:12])
        raise ValueError(
            "unknown stable-retro state(s) for "
            f"{game}: {', '.join(unknown)}. Known examples: {valid_preview}"
        )


def resolve_mixed_state_config(config: EnvConfig, n_envs: int) -> EnvConfig:
    config = resolve_env_config(config)
    if n_envs < 1:
        raise ValueError("n_envs must be >= 1")
    provider = resolve_env_provider(config.env_provider)
    if not env_supports_states(provider.provider_id, config.game) and (
        config.state or config.states or config.state_probs
    ):
        raise ValueError(
            f"environment provider {provider.provider_id!r} does not support "
            "state, states, or state_probs"
        )
    if not config.states:
        if config.state_probs:
            raise ValueError("--state-probs requires --states")
        return config
    if provider.provider_id == STABLE_RETRO_TURBO_PROVIDER.provider_id:
        _validate_state_names(config.game, config.states)
    if config.state_probs:
        if len(config.state_probs) != len(config.states):
            raise ValueError("--state-probs count must match --states count")
        probs = np.asarray(config.state_probs, dtype=np.float64)
        if not np.all(np.isfinite(probs)) or np.any(probs < 0.0) or probs.sum() <= 0.0:
            raise ValueError("--state-probs must be non-negative finite values with a positive sum")
        return config
    if len(config.states) != n_envs:
        raise ValueError(
            "--states without --state-probs must provide exactly one state per env slot: "
            f"got {len(config.states)} states for n_envs={n_envs}"
        )
    return config


def state_distribution_metadata(config: EnvConfig) -> list[dict[str, float | str]]:
    if not config.states:
        return []
    if config.state_probs:
        distribution: dict[str, float] = {}
        for state, prob in zip(config.states, config.state_probs, strict=True):
            distribution[state] = distribution.get(state, 0.0) + float(prob)
        total = sum(distribution.values())
        return [
            {"state": state, "probability": probability / total}
            for state, probability in distribution.items()
        ]
    probability = 1.0 / len(config.states)
    return [{"state": state, "probability": probability} for state in config.states]


def state_weight_mapping(config: EnvConfig) -> dict[str, float]:
    weights: dict[str, float] = {}
    for state, weight in zip(config.states, config.state_probs, strict=True):
        weights[state] = weights.get(state, 0.0) + float(weight)
    return weights


def state_name_candidates_from_level_id(level_id: str) -> tuple[str, ...]:
    candidates = [f"Level{level_id}"]
    parts = level_id.split("-", 1)
    if len(parts) == 2:
        try:
            candidates.append(f"Level{int(parts[0]) + 1}-{int(parts[1]) + 1}")
        except ValueError:
            pass
    return tuple(dict.fromkeys(candidates))


def info_value_from_state_name(
    state_name: str,
    info_vars: tuple[str, ...],
) -> tuple[int | str, ...] | None:
    if tuple(info_vars) == ("levelHi", "levelLo") and state_name.startswith("Level"):
        level = state_name.removeprefix("Level").split("-", 2)
        if len(level) >= 2:
            try:
                return (int(level[0]) - 1, int(level[1]) - 1)
            except ValueError:
                return None
    return None


def task_conditioning_info_values(config: EnvConfig) -> tuple[tuple[int | str, ...], ...]:
    conditioning = config.task.get("conditioning", {})
    if not isinstance(conditioning, Mapping) or not conditioning.get("enabled"):
        return ()
    configured = conditioning.get("values", ())
    if configured:
        return tuple(tuple(value) for value in configured)
    signal_name = conditioning.get("signal")
    signals = config.task.get("signals", {})
    source = signals.get(signal_name) if isinstance(signals, Mapping) else None
    info_vars = (source,) if isinstance(source, str) else tuple(source or ())
    values = []
    for state_name in dict.fromkeys(config.states or ((config.state,) if config.state else ())):
        value = info_value_from_state_name(state_name, info_vars)
        if value is not None:
            values.append(value)
    return tuple(values)


def task_action_set(config: EnvConfig) -> str:
    action = config.task.get("action", {})
    return str(action.get("set", "native")) if isinstance(action, Mapping) else "native"


def task_action_values(config: EnvConfig) -> tuple[Any, ...] | None:
    action = config.task.get("action", {})
    if not isinstance(action, Mapping):
        return None
    codec = action.get("codec")
    if not isinstance(codec, Mapping):
        return None
    values = codec.get("values")
    return tuple(values) if isinstance(values, list | tuple) else None


def task_termination(config: EnvConfig) -> Mapping[str, Any]:
    value = config.task.get("termination", {})
    return value if isinstance(value, Mapping) else {}


def task_reward(config: EnvConfig) -> Mapping[str, Any]:
    value = config.task.get("reward", {})
    return value if isinstance(value, Mapping) else {}


def task_max_episode_steps(config: EnvConfig) -> int:
    return int(task_termination(config).get("max_episode_steps", 0))


def task_conditioning(config: EnvConfig) -> Mapping[str, Any]:
    value = config.task.get("conditioning", {})
    return value if isinstance(value, Mapping) else {}


def with_task_termination(config: EnvConfig, **updates: Any) -> EnvConfig:
    task = deepcopy(config.task)
    termination = dict(task.get("termination", {}))
    termination.update(updates)
    task["termination"] = termination
    return replace(config, task=task)


def make_provider_vec_env(config: EnvConfig, *, native_kwargs: Mapping[str, Any]):
    return provider_runtime.make_provider_vec_env(
        config,
        native_kwargs=native_kwargs,
        retro_vec_env_type=RetroVecEnv,
        super_mario_vec_env_type=_super_mario_bros_nes_turbo_vec_env_type,
        ale_py_vec_env_type=_ale_py_atari_vector_env_type,
    )


def _provider_descriptor(config: EnvConfig, native_env: Any) -> ProviderDescriptor:
    return provider_descriptor(
        config,
        native_env,
        state_weight_mapping=state_weight_mapping,
    )


def make_native_provider(
    config: EnvConfig,
    n_envs: int,
    *,
    rom_binding: RomRuntimeBinding | None = None,
) -> tuple[Any, ProviderDescriptor]:
    """Construct and describe one provider, closing it if description fails."""

    provider = resolve_env_provider(config.env_provider)
    if provider.requires_external_rom_asset and rom_binding is None:
        raise FileNotFoundError(f"{provider.provider_id} requires a verified runtime ROM binding")

    native_kwargs = provider_native_vec_kwargs(
        config,
        n_envs=n_envs,
        native_obs_crop=native_obs_crop,
        state_weight_mapping=state_weight_mapping,
        runtime_rom_path=rom_binding.rom_path if rom_binding is not None else None,
    )
    native_env = make_provider_vec_env(config, native_kwargs=native_kwargs)
    try:
        descriptor = _provider_descriptor(config, native_env)
    except BaseException:
        native_env.close()
        raise
    return native_env, descriptor


def bind_native_provider(
    config: EnvConfig,
    *,
    n_envs: int,
    seed: int,
    native_env: Any,
    descriptor: ProviderDescriptor,
    global_lane_ids: tuple[int, ...] | None = None,
    capture_step_diagnostics: bool = False,
    snapshot_curriculum: Mapping[str, Any] | None = None,
) -> BatchRuntime:
    """Transfer a constructed provider into the task runtime or close it on failure."""

    runtime: BatchRuntime | None = None
    try:
        kernel = _bound_task_kernel(config, descriptor, n_envs)
        runtime = BatchRuntime(
            native_env,
            descriptor,
            kernel,
            run_seed=seed,
            global_lane_ids=global_lane_ids,
            capture_step_diagnostics=capture_step_diagnostics,
            snapshot_curriculum=snapshot_curriculum,
        )
        return runtime
    except BaseException:
        if runtime is None:
            native_env.close()
        else:
            runtime.close()
        raise


def _bound_task_kernel(config: EnvConfig, descriptor: ProviderDescriptor, n_envs: int):
    task_id = config.task.get("id")
    if task_id == "mario":
        return MarioTaskDefinition(MarioTaskConfig.from_env_config(config)).bind(descriptor, n_envs)
    if task_id != "identity":
        raise ValueError(f"unknown task kernel {task_id!r}")
    action_values = task_action_values(config)
    if action_values is None and config.env_provider == STABLE_RETRO_TURBO_PROVIDER.provider_id:
        action_values = configured_action_values(config)
    if task_action_set(config) != "native" and action_values is None:
        raise ValueError(
            "generic native-vector tasks require native actions or a discrete lookup codec"
        )
    if task_reward(config).get("reward_mode") != "native":
        raise ValueError("generic native-vector tasks require native rewards")
    if task_conditioning(config).get("enabled"):
        raise ValueError("generic native-vector tasks do not support task conditioning")
    # Stable Retro applies obs_crop natively. Only ale-py needs the task kernel
    # to mask its already-resized observations.
    observation_mask = (
        native_obs_crop(config) if config.env_provider == ALE_PY_PROVIDER.provider_id else None
    )
    source_shape = (210, 160) if observation_mask is not None else None
    return IdentityTaskDefinition(
        observation_mask=observation_mask,
        observation_mask_fill=config.obs_crop_fill,
        observation_source_shape=source_shape,
        max_episode_steps=task_max_episode_steps(config),
        action_values=action_values,
        signals=config.task.get("signals", {}),
        events=config.task.get("events", {}),
        termination=task_termination(config),
    ).bind(descriptor, n_envs)


def make_vec_envs(
    config: EnvConfig,
    n_envs: int,
    seed: int,
    *,
    capture_step_diagnostics: bool = False,
    rom_binding: RomRuntimeBinding | None = None,
    snapshot_curriculum: Mapping[str, Any] | None = None,
) -> Any:
    from rlab.training.sb3_vec_env import RlabVecEnv

    runtime = make_training_batch_runtime(
        config,
        n_envs,
        seed,
        capture_step_diagnostics=capture_step_diagnostics,
        rom_binding=rom_binding,
        snapshot_curriculum=snapshot_curriculum,
    )
    vec_env = RlabVecEnv(runtime)
    vec_env.seed(seed)
    return vec_env


def make_training_batch_runtime(
    config: EnvConfig,
    n_envs: int,
    seed: int,
    *,
    global_lane_ids: tuple[int, ...] | None = None,
    capture_step_diagnostics: bool = False,
    rom_binding: RomRuntimeBinding | None = None,
    snapshot_curriculum: Mapping[str, Any] | None = None,
) -> BatchRuntime:
    os.environ.setdefault("STABLE_RETRO_DISABLE_AUDIO", "1")
    config = resolve_mixed_state_config(config, n_envs=n_envs)
    native_env, descriptor = make_native_provider(config, n_envs, rom_binding=rom_binding)
    return bind_native_provider(
        config,
        n_envs=n_envs,
        seed=seed,
        native_env=native_env,
        descriptor=descriptor,
        global_lane_ids=global_lane_ids,
        capture_step_diagnostics=capture_step_diagnostics,
        snapshot_curriculum=snapshot_curriculum,
    )


def _snapshot_preflight_lane_count(value: Mapping[str, Any], configured_n_envs: int) -> int:
    from rlab.snapshot_curriculum import normalize_snapshot_curriculum_config

    if configured_n_envs < 2:
        raise ValueError("snapshot curriculum preflight requires at least two configured lanes")
    for lanes in range(2, min(configured_n_envs, 32) + 1):
        try:
            normalize_snapshot_curriculum_config(value, n_envs=lanes)
        except ValueError as exc:
            if "resolves to" not in str(exc):
                raise
        else:
            return lanes
    return configured_n_envs


def _snapshot_curriculum_preflight_child(
    connection: Any,
    config: EnvConfig,
    configured_n_envs: int,
    seed: int,
    rom_binding: RomRuntimeBinding | None,
    snapshot_curriculum: Mapping[str, Any],
) -> None:
    runtime: BatchRuntime | None = None
    try:
        preflight_lanes = _snapshot_preflight_lane_count(
            snapshot_curriculum,
            configured_n_envs,
        )
        runtime = make_training_batch_runtime(
            config,
            preflight_lanes,
            seed,
            rom_binding=rom_binding,
            snapshot_curriculum=snapshot_curriculum,
        )
        if runtime.snapshot_curriculum is None:
            raise RuntimeError("snapshot curriculum preflight runtime is disabled")
        if runtime.snapshot_curriculum.config.restore_snapshots:
            payload = runtime.preflight_snapshot_round_trip(seed=seed)
        else:
            payload = runtime.preflight_snapshot_capture(seed=seed)
        connection.send(("ok", payload))
    except BaseException as exc:
        connection.send(
            (
                "error",
                {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(limit=20),
                },
            )
        )
    finally:
        if runtime is not None:
            runtime.close()
        connection.close()


def preflight_snapshot_curriculum_provider(
    *,
    config: EnvConfig,
    n_envs: int,
    seed: int,
    rom_binding: RomRuntimeBinding | None,
    snapshot_curriculum: Mapping[str, Any] | None,
    timeout_seconds: float = 60.0,
) -> dict[str, Any] | None:
    """Run the live snapshot conformance probe in an isolated, disposable process."""

    if snapshot_curriculum is None:
        return None
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_snapshot_curriculum_preflight_child,
        args=(sender, config, int(n_envs), int(seed), rom_binding, dict(snapshot_curriculum)),
        name="rlab-snapshot-preflight",
    )
    started_at = time.perf_counter()
    process.start()
    sender.close()
    try:
        if not receiver.poll(timeout_seconds):
            process.terminate()
            process.join(timeout=10.0)
            raise TimeoutError(
                f"snapshot curriculum provider preflight exceeded {timeout_seconds:g} seconds"
            )
        status, payload = receiver.recv()
    finally:
        receiver.close()
    process.join(timeout=10.0)
    if process.is_alive():
        process.terminate()
        process.join(timeout=10.0)
        raise RuntimeError("snapshot curriculum provider preflight child did not exit")
    if status != "ok":
        raise RuntimeError(
            "snapshot curriculum provider preflight failed: "
            f"{payload['type']}: {payload['message']}\n{payload['traceback']}"
        )
    if process.exitcode != 0:
        raise RuntimeError(
            f"snapshot curriculum provider preflight exited with code {process.exitcode}"
        )
    return {
        **dict(payload),
        "elapsed_seconds": time.perf_counter() - started_at,
        "isolation": "spawned_process",
    }


def make_training_vec_env(
    config: EnvConfig,
    n_envs: int,
    seed: int,
    *,
    rom_binding: RomRuntimeBinding | None = None,
    snapshot_curriculum: Mapping[str, Any] | None = None,
) -> Any:
    return make_vec_envs(
        config=config,
        n_envs=n_envs,
        seed=seed,
        rom_binding=rom_binding,
        snapshot_curriculum=snapshot_curriculum,
    )


def make_eval_vec_env(
    config: EnvConfig,
    n_envs: int,
    seed: int,
    *,
    capture_step_diagnostics: bool = False,
    rom_binding: RomRuntimeBinding | None = None,
) -> Any:
    return make_vec_envs(
        config=resolve_env_config(config),
        n_envs=n_envs,
        seed=seed,
        capture_step_diagnostics=capture_step_diagnostics,
        rom_binding=rom_binding,
    )


def assert_rom_imported(game: str) -> str:
    if not game:
        raise ValueError("game is required; pass --game or set RETRO_GAME")
    try:
        return retro.data.get_romfile_path(game)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"{game} is not imported in this rlab runtime. "
            f"Run: rlab import-roms ~/Desktop/roms --game {game}"
        ) from exc


def assert_provider_runtime_available(
    config: EnvConfig,
    *,
    rom_binding: RomRuntimeBinding | None = None,
) -> None:
    provider = resolve_env_provider(config.env_provider)
    if provider.requires_external_rom_asset:
        if rom_binding is None:
            raise FileNotFoundError(f"{config.game} requires a verified external ROM asset binding")
        if rom_binding.manifest.get("game") != config.game:
            raise ValueError("runtime ROM binding game mismatch")
    elif provider.provider_id == ALE_PY_PROVIDER.provider_id:
        from ale_py import roms

        if roms.get_rom_path(config.game) is None:
            raise FileNotFoundError(
                f"{config.game} is not available to ale-py. "
                "Install an ALE ROM package or import ROMs with ale-import-roms."
            )


def default_run_dir(run_name: str, runs_dir: str = "runs") -> str:
    return os.path.join(runs_dir, run_name)
