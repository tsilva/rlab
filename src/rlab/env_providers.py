from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any

from gymnasium.vector import AutoresetMode
from stable_retro import RetroVecEnv as DEFAULT_RETRO_VEC_ENV

from rlab.env_registry import (
    ALE_PY_PROVIDER,
    STABLE_RETRO_TURBO_PROVIDER,
    SUPERMARIOBROS_NES_TURBO_PROVIDER,
    resolve_env_provider,
)


def _is_same_step_autoreset_mode(value: Any) -> bool:
    name = getattr(value, "name", None)
    if isinstance(name, str) and name == "SAME_STEP":
        return True
    raw_value = getattr(value, "value", value)
    normalized = "".join(char for char in str(raw_value).lower() if char.isalnum())
    return normalized == "samestep"


def _force_same_step_autoreset_kwargs(
    env_type: type,
    native_kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    kwargs = dict(native_kwargs)
    try:
        signature = inspect.signature(env_type.__init__)
    except (OSError, TypeError, ValueError):
        return kwargs
    if "autoreset_mode" in signature.parameters:
        kwargs["autoreset_mode"] = AutoresetMode.SAME_STEP
    return kwargs


def _require_same_step_autoreset_mode(env: Any, provider_id: str):
    mode = getattr(env, "autoreset_mode", None)
    metadata = getattr(env, "metadata", None)
    if mode is None and isinstance(metadata, Mapping):
        mode = metadata.get("autoreset_mode")
    if mode is None:
        return env
    if not _is_same_step_autoreset_mode(mode):
        raise RuntimeError(
            f"{provider_id} vector env must use same-step autoreset mode; "
            f"got {mode!r}",
        )
    return env


def super_mario_bros_nes_turbo_vec_env_type():
    try:
        from supermariobrosnes_turbo import SuperMarioBrosNesTurboVecEnv
    except ImportError as exc:
        raise ImportError(
            "supermariobrosnes-turbo provider requires "
            "supermariobrosnes-turbo",
        ) from exc
    return SuperMarioBrosNesTurboVecEnv


def ale_py_atari_vector_env_type():
    try:
        from ale_py.vector_env import AtariVectorEnv
    except ImportError as exc:
        raise ImportError(
            "ale-py provider requires ale-py with native vector env support",
        ) from exc
    return AtariVectorEnv


def provider_vec_env_type(
    config: Any | None = None,
    *,
    retro_vec_env_type=DEFAULT_RETRO_VEC_ENV,
    super_mario_vec_env_type=super_mario_bros_nes_turbo_vec_env_type,
    ale_py_vec_env_type=ale_py_atari_vector_env_type,
):
    provider_id = getattr(config, "env_provider", STABLE_RETRO_TURBO_PROVIDER.provider_id)
    provider = resolve_env_provider(provider_id)
    if provider.provider_id == STABLE_RETRO_TURBO_PROVIDER.provider_id:
        return retro_vec_env_type
    if provider.provider_id == SUPERMARIOBROS_NES_TURBO_PROVIDER.provider_id:
        return super_mario_vec_env_type()
    if provider.provider_id == ALE_PY_PROVIDER.provider_id:
        return ale_py_vec_env_type()
    raise ValueError(f"unsupported environment provider {provider.provider_id!r}")


def native_vec_env_supports_done_on(
    config: Any | None = None,
    *,
    retro_vec_env_type=DEFAULT_RETRO_VEC_ENV,
    super_mario_vec_env_type=super_mario_bros_nes_turbo_vec_env_type,
    ale_py_vec_env_type=ale_py_atari_vector_env_type,
) -> bool:
    try:
        signature = inspect.signature(
            provider_vec_env_type(
                config,
                retro_vec_env_type=retro_vec_env_type,
                super_mario_vec_env_type=super_mario_vec_env_type,
                ale_py_vec_env_type=ale_py_vec_env_type,
            ).__init__
        )
    except (OSError, TypeError):
        return False
    return "done_on" in signature.parameters


def native_vec_env_supports_named_done_on(
    config: Any | None = None,
    *,
    retro_vec_env_type=DEFAULT_RETRO_VEC_ENV,
    super_mario_vec_env_type=super_mario_bros_nes_turbo_vec_env_type,
    ale_py_vec_env_type=ale_py_atari_vector_env_type,
) -> bool:
    provider_id = getattr(config, "env_provider", STABLE_RETRO_TURBO_PROVIDER.provider_id)
    provider = resolve_env_provider(provider_id)
    if provider.provider_id == SUPERMARIOBROS_NES_TURBO_PROVIDER.provider_id:
        return True
    return callable(
        getattr(
            provider_vec_env_type(
                config,
                retro_vec_env_type=retro_vec_env_type,
                super_mario_vec_env_type=super_mario_vec_env_type,
                ale_py_vec_env_type=ale_py_vec_env_type,
            ),
            "resolve_info_event_rules",
            None,
        )
    )


def native_vec_env_supports_rgb_render(
    config: Any | None = None,
    *,
    retro_vec_env_type=DEFAULT_RETRO_VEC_ENV,
    super_mario_vec_env_type=super_mario_bros_nes_turbo_vec_env_type,
    ale_py_vec_env_type=ale_py_atari_vector_env_type,
) -> bool:
    provider_id = getattr(config, "env_provider", STABLE_RETRO_TURBO_PROVIDER.provider_id)
    provider = resolve_env_provider(provider_id)
    if provider.provider_id == STABLE_RETRO_TURBO_PROVIDER.provider_id:
        return True
    try:
        env_type = provider_vec_env_type(
            config,
            retro_vec_env_type=retro_vec_env_type,
            super_mario_vec_env_type=super_mario_vec_env_type,
            ale_py_vec_env_type=ale_py_vec_env_type,
        )
    except (ImportError, ValueError):
        return False
    metadata = getattr(env_type, "metadata", {})
    render_modes = metadata.get("render_modes", ()) if isinstance(metadata, Mapping) else ()
    return "rgb_array" in render_modes


def _require_provider(config: Any, expected_provider_id: str):
    provider = resolve_env_provider(config.env_provider)
    if provider.provider_id != expected_provider_id:
        raise ValueError(
            f"unsupported environment provider {provider.provider_id!r}; "
            f"expected {expected_provider_id}",
        )
    return provider


def _stable_retro_turbo_make_vec_env(
    config: Any,
    *,
    native_kwargs: Mapping[str, Any],
    retro_vec_env_type=DEFAULT_RETRO_VEC_ENV,
):
    _require_provider(config, STABLE_RETRO_TURBO_PROVIDER.provider_id)
    env_type = retro_vec_env_type
    env = env_type(
        config.game,
        **_force_same_step_autoreset_kwargs(env_type, native_kwargs),
    )
    return _require_same_step_autoreset_mode(env, STABLE_RETRO_TURBO_PROVIDER.provider_id)


def _super_mario_bros_nes_turbo_make_vec_env(
    config: Any,
    *,
    native_kwargs: Mapping[str, Any],
    super_mario_vec_env_type=super_mario_bros_nes_turbo_vec_env_type,
):
    _require_provider(config, SUPERMARIOBROS_NES_TURBO_PROVIDER.provider_id)
    env_type = super_mario_vec_env_type()
    env = env_type(
        config.game,
        **_force_same_step_autoreset_kwargs(env_type, native_kwargs),
    )
    return _require_same_step_autoreset_mode(env, SUPERMARIOBROS_NES_TURBO_PROVIDER.provider_id)


def _ale_py_make_vec_env(
    config: Any,
    *,
    native_kwargs: Mapping[str, Any],
    ale_py_vec_env_type=ale_py_atari_vector_env_type,
):
    _require_provider(config, ALE_PY_PROVIDER.provider_id)
    env_type = ale_py_vec_env_type()
    env = env_type(
        config.game,
        **_force_same_step_autoreset_kwargs(env_type, native_kwargs),
    )
    return _require_same_step_autoreset_mode(env, ALE_PY_PROVIDER.provider_id)


def make_provider_vec_env(
    config: Any,
    *,
    native_kwargs: Mapping[str, Any],
    retro_vec_env_type=DEFAULT_RETRO_VEC_ENV,
    super_mario_vec_env_type=super_mario_bros_nes_turbo_vec_env_type,
    ale_py_vec_env_type=ale_py_atari_vector_env_type,
):
    provider = resolve_env_provider(config.env_provider)
    if provider.provider_id == STABLE_RETRO_TURBO_PROVIDER.provider_id:
        return _stable_retro_turbo_make_vec_env(
            config,
            native_kwargs=native_kwargs,
            retro_vec_env_type=retro_vec_env_type,
        )
    if provider.provider_id == SUPERMARIOBROS_NES_TURBO_PROVIDER.provider_id:
        return _super_mario_bros_nes_turbo_make_vec_env(
            config,
            native_kwargs=native_kwargs,
            super_mario_vec_env_type=super_mario_vec_env_type,
        )
    if provider.provider_id == ALE_PY_PROVIDER.provider_id:
        return _ale_py_make_vec_env(
            config,
            native_kwargs=native_kwargs,
            ale_py_vec_env_type=ale_py_vec_env_type,
        )
    raise ValueError(f"unsupported environment provider {provider.provider_id!r}")
