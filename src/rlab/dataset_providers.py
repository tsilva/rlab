"""Gymrec v3 provider adapters used by Rlab dataset recording.

This module is adapted from Gymrec 0.1.1's MIT-licensed provider boundary.
See THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import inspect
import json
import os
import stat
import tempfile
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from rlab.action_contract import MARIO_ACTION_TABLES
from rlab.dataset_contract import canonical_json_bytes


PROVIDER_CONTRACT_VERSION = 1
STABLE_RETRO_PROVIDER_ID = "stable-retro-turbo"
MARIO_TURBO_PROVIDER_ID = "supermariobrosnes-turbo"
SUPPORTED_PROVIDER_IDS = frozenset({STABLE_RETRO_PROVIDER_ID, MARIO_TURBO_PROVIDER_ID})
BUILTIN_ACTION_SETS = MARIO_ACTION_TABLES
MANAGED_CONFIG_KEYS = frozenset(
    {"game", "num_envs", "num_threads", "render_mode", "autoreset_mode"}
)


def _json_value(value: Any) -> Any:
    if hasattr(value, "name") and hasattr(value, "value"):
        return str(value.name)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, memoryview)):
        return [_json_value(item) for item in value]
    return value


def _lane_info(infos: Any) -> dict[str, Any]:
    if not isinstance(infos, Mapping):
        return {}
    result: dict[str, Any] = {}
    for key, value in infos.items():
        if str(key).startswith("_"):
            continue
        mask = infos.get(f"_{key}")
        if mask is not None and not bool(np.asarray(mask).reshape(-1)[0]):
            continue
        if isinstance(value, np.ndarray) and value.shape[:1] == (1,):
            value = value[0]
        elif (
            isinstance(value, Sequence)
            and not isinstance(value, (str, bytes, bytearray, memoryview))
            and len(value) == 1
        ):
            value = value[0]
        result[str(key)] = _json_value(value)
    return result


class SingleLaneEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, vector_env: Any, asset_stage: _AssetStage | None = None) -> None:
        if int(vector_env.num_envs) != 1:
            raise ValueError("dataset providers require exactly one environment lane")
        self.vector_env = vector_env
        self.action_space = vector_env.single_action_space
        self.observation_space = vector_env.single_observation_space
        self.render_mode = "rgb_array"
        self._needs_reset = True
        self._asset_stage = asset_stage

    def reset(self, *, seed: int | None = None, options: Mapping[str, Any] | None = None):
        super().reset(seed=seed)
        observations, infos = self.vector_env.reset(seed=seed, options=options)
        self._needs_reset = False
        return observations[0], _lane_info(infos)

    def step(self, action: Any):
        if self._needs_reset:
            raise RuntimeError("reset() is required after a terminal step")
        if isinstance(self.action_space, gym.spaces.Discrete):
            scalar = int(np.asarray(action).reshape(-1)[0])
            if not self.action_space.contains(scalar):
                raise ValueError(f"action {scalar!r} is not in {self.action_space}")
            batched = np.asarray([scalar], dtype=self.action_space.dtype)
        else:
            scalar = np.asarray(action, dtype=self.action_space.dtype)
            if not self.action_space.contains(scalar):
                raise ValueError(f"action {action!r} is not in {self.action_space}")
            batched = scalar[np.newaxis, ...]
        observations, rewards, terminated, truncated, infos = self.vector_env.step(batched)
        is_terminated = bool(terminated[0])
        is_truncated = bool(truncated[0])
        self._needs_reset = is_terminated or is_truncated
        return (
            observations[0],
            float(rewards[0]),
            is_terminated,
            is_truncated,
            _lane_info(infos),
        )

    def render(self):
        return self.vector_env.render()

    def close(self) -> None:
        try:
            self.vector_env.close()
        finally:
            if self._asset_stage is not None:
                self._asset_stage.cleanup()
                self._asset_stage = None


class _AssetStage:
    def __init__(self) -> None:
        self._temporary = tempfile.TemporaryDirectory(prefix="rlab-runtime-assets-")
        self.root = Path(self._temporary.name)
        os.chmod(self.root, 0o700)

    def copy(self, source: str | Path, label: str) -> tuple[Path, str]:
        path = Path(source).expanduser()
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        destination = self.root / f"{len(tuple(self.root.iterdir())):02d}-{label}{path.suffix}"
        digest = hashlib.sha256()
        written = 0
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise ValueError(f"runtime asset is not a regular file: {path}")
            with destination.open("xb") as output:
                os.chmod(destination, 0o600)
                while chunk := os.read(descriptor, 1024 * 1024):
                    output.write(chunk)
                    digest.update(chunk)
                    written += len(chunk)
                output.flush()
                os.fsync(output.fileno())
            after = os.fstat(descriptor)
            if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            ):
                raise ValueError(f"runtime asset changed while staging: {path}")
            if written != before.st_size:
                raise ValueError(f"runtime asset was truncated while staging: {path}")
        finally:
            os.close(descriptor)
        return destination, digest.hexdigest()

    def cleanup(self) -> None:
        self._temporary.cleanup()


def _retro_data_path(data: Any, env_id: str, value: Any, default: str, inttype: Any) -> Path:
    token = default if value is None else str(value)
    explicit = Path(token).expanduser()
    if explicit.is_file():
        return explicit
    filename = token if token.endswith(".json") else f"{token}.json"
    resolved = data.get_file_path(env_id, filename, inttype)
    if not resolved:
        raise FileNotFoundError(f"could not resolve {filename!r} for {env_id}")
    return Path(resolved)


def _retro_state_path(data: Any, env_id: str, state: Any, inttype: Any) -> Path:
    explicit = Path(str(state)).expanduser()
    if explicit.is_file():
        return explicit
    filename = str(state) if str(state).endswith(".state") else f"{state}.state"
    resolved = data.get_file_path(env_id, filename, inttype)
    if not resolved:
        raise FileNotFoundError(f"could not resolve state {state!r} for {env_id}")
    return Path(resolved)


def _file_sha256(path: str | Path | None) -> str | None:
    if not path:
        return None
    try:
        with Path(path).open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    except FileNotFoundError, OSError, TypeError:
        return None


def _normalize_policy_actions(task: Any) -> tuple[tuple[tuple[str, ...], ...] | None, Any]:
    if task is None:
        return None, None
    if not isinstance(task, Mapping):
        raise ValueError("env config task must be an object")
    extra_task = sorted(set(task) - {"action"})
    if extra_task:
        warnings.warn(
            "dataset recording ignores task sections other than task.action: "
            + ", ".join(extra_task),
            stacklevel=2,
        )
    action = task.get("action")
    if action is None:
        return None, None
    if not isinstance(action, Mapping) or set(action) - {"set", "actions"}:
        raise ValueError("env config task.action has an invalid schema")
    explicit = action.get("actions")
    name = action.get("set")
    if explicit is not None and name is not None:
        raise ValueError("task.action cannot define both set and actions")
    if explicit is not None:
        if not isinstance(explicit, Sequence) or isinstance(explicit, (str, bytes)):
            raise ValueError("task.action.actions must be an array")
        if any(
            not isinstance(labels, Sequence) or isinstance(labels, (str, bytes))
            for labels in explicit
        ):
            raise ValueError("each task.action.actions entry must be an array of control labels")
        normalized = tuple(tuple(str(label).upper() for label in labels) for labels in explicit)
        if not normalized:
            raise ValueError("task.action.actions must not be empty")
        return normalized, {"actions": [list(labels) for labels in normalized]}
    if name in (None, "native"):
        return None, None
    normalized_name = str(name).lower()
    if normalized_name not in BUILTIN_ACTION_SETS:
        raise ValueError(f"unknown dataset policy action set {name!r}")
    return BUILTIN_ACTION_SETS[normalized_name], {"set": normalized_name}


def _prepare_config(config: Mapping[str, Any]) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    kwargs = copy.deepcopy(dict(config))
    policy_actions, effective_action = _normalize_policy_actions(kwargs.pop("task", None))
    managed = sorted(MANAGED_CONFIG_KEYS.intersection(kwargs))
    if managed:
        raise ValueError(f"managed environment config key(s): {', '.join(managed)}")
    effective = copy.deepcopy(kwargs)
    if effective_action is not None:
        effective["task"] = {"action": effective_action}
    return kwargs, policy_actions, effective


def validate_provider_request(config: Mapping[str, Any]) -> None:
    """Validate the provider-independent recording boundary before any runtime is built."""
    _prepare_config(config)
    state = config.get("state")
    if isinstance(state, Mapping) or (
        isinstance(state, Sequence) and not isinstance(state, (str, bytes, bytearray, memoryview))
    ):
        raise ValueError("dataset recording accepts only a default, named, or custom scalar state")


class ProviderSession:
    def __init__(
        self,
        *,
        provider_id: str,
        environment_id: str,
        effective_config: Mapping[str, Any],
        vector_env: Any,
        system: str,
        buttons: Sequence[str],
        policy_actions: tuple[tuple[str, ...], ...] | None,
        fps: float,
        assets: Mapping[str, Any],
        asset_stage: _AssetStage | None = None,
    ) -> None:
        self.provider_id = provider_id
        self.environment_id = environment_id
        self.effective_config = copy.deepcopy(dict(effective_config))
        self.env = SingleLaneEnv(vector_env, asset_stage)
        self.control_profile = f"stable_retro.{system}"
        self.fps = max(float(fps), 1.0)
        self._buttons = tuple(str(button).upper() for button in buttons)
        self._policy_actions = policy_actions
        self.provenance = {
            "distribution": provider_id,
            "version": importlib.metadata.version(provider_id),
            "assets": _json_value(assets),
        }

    def policy_observation(self, observation: Any) -> Any:
        return observation

    def recording_observation(self, observation: Any) -> Any:
        frame = self.env.render()
        return observation if frame is None else frame

    def adapt_policy_action(self, action: Any) -> Any:
        if self._policy_actions is None:
            if isinstance(self.env.action_space, gym.spaces.Discrete):
                return int(np.asarray(action).reshape(-1)[0])
            return action
        index = int(np.asarray(action).reshape(-1)[0])
        if not 0 <= index < len(self._policy_actions):
            raise ValueError(f"policy action {index} is outside the configured action set")
        return self.action_from_labels(self._policy_actions[index])

    def validate_policy(self, policy: Any) -> None:
        policy_action = getattr(policy, "action_space", None)
        if self._policy_actions is not None:
            if getattr(policy_action, "n", None) != len(self._policy_actions):
                raise ValueError("policy action space does not match the configured action set")
        elif policy_action is not None:
            expected_n = getattr(self.env.action_space, "n", None)
            actual_n = getattr(policy_action, "n", None)
            if expected_n is not None and actual_n != expected_n:
                raise ValueError("policy action space does not match the native environment")
            if expected_n is None and getattr(policy_action, "shape", None) != getattr(
                self.env.action_space, "shape", None
            ):
                raise ValueError("policy action shape does not match the native environment")
        policy_observation = getattr(policy, "observation_space", None)
        if policy_observation is not None and getattr(policy_observation, "shape", None) != getattr(
            self.env.observation_space, "shape", None
        ):
            raise ValueError("policy observation space does not match the provider")

    def action_from_labels(self, labels: Sequence[str]) -> Any:
        requested = {str(label).upper() for label in labels}
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            meanings = getattr(self.env.vector_env, "action_meanings", ())
            action_buttons = getattr(self.env.vector_env, "ACTION_BUTTONS", None)
            if action_buttons is None:
                try:
                    from supermariobrosnes_turbo import ACTION_BUTTONS

                    action_buttons = ACTION_BUTTONS
                except ImportError:
                    action_buttons = {}
            for index, meaning in enumerate(meanings):
                actual = {str(label).upper() for label in action_buttons.get(str(meaning), ())}
                if actual == requested:
                    return index
            raise ValueError(f"no configured action matches controls {sorted(requested)!r}")
        if not isinstance(self.env.action_space, gym.spaces.MultiBinary):
            raise ValueError("named controls require MultiBinary or named Discrete actions")
        action = np.zeros(self.env.action_space.n, dtype=self.env.action_space.dtype)
        for label in requested:
            try:
                action[self._buttons.index(label)] = 1
            except ValueError as exc:
                raise ValueError(f"control label {label!r} is unavailable") from exc
        return action


def _resolve_enum(value: Any, enum_type: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return enum_type[value.split(".")[-1].upper()]
    except KeyError as exc:
        raise ValueError(f"unknown {enum_type.__name__} value {value!r}") from exc


def _state_hashes(data: Any, stable_retro: Any, env_id: str, state: Any, inttype: Any):
    if state == stable_retro.State.NONE or state == stable_retro.State.DEFAULT:
        return {}
    values = state.keys() if isinstance(state, Mapping) else state
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        values = (values,)
    result = {}
    for value in values:
        raw = str(value)
        path = Path(raw).expanduser()
        if not path.exists():
            path = Path(
                data.get_file_path(
                    env_id, raw if raw.endswith(".state") else f"{raw}.state", inttype
                )
            )
        result[raw] = _file_sha256(path)
    return result


def _stable_retro_session(environment_id: str, config: Mapping[str, Any]) -> ProviderSession:
    import stable_retro
    from stable_retro import data

    kwargs, policy_actions, effective = _prepare_config(config)
    state = kwargs.pop("state", stable_retro.State.DEFAULT)
    if "use_restricted_actions" in kwargs:
        kwargs["use_restricted_actions"] = _resolve_enum(
            kwargs["use_restricted_actions"], stable_retro.Actions
        )
    if "obs_type" in kwargs:
        kwargs["obs_type"] = _resolve_enum(kwargs["obs_type"], stable_retro.Observations)
    inttype = _resolve_enum(kwargs.get("inttype", data.Integrations.STABLE), data.Integrations)
    if "inttype" in kwargs:
        kwargs["inttype"] = inttype
    if "autoreset_mode" in inspect.signature(stable_retro.RetroVecEnv).parameters:
        kwargs["autoreset_mode"] = "Disabled"
    stage = _AssetStage()
    try:
        rom_source = kwargs.get("rom_path") or data.get_original_romfile_path(
            environment_id, inttype
        )
        rom_path, rom_hash = stage.copy(rom_source, "rom")
        kwargs["rom_path"] = str(rom_path)
        info_path, info_hash = stage.copy(
            _retro_data_path(data, environment_id, kwargs.pop("info", None), "data", inttype),
            "info",
        )
        scenario_path, scenario_hash = stage.copy(
            _retro_data_path(
                data, environment_id, kwargs.pop("scenario", None), "scenario", inttype
            ),
            "scenario",
        )
        kwargs["info"] = str(info_path)
        kwargs["scenario"] = str(scenario_path)
        state_hashes = {}
        if state == stable_retro.State.DEFAULT:
            metadata_path, metadata_hash = stage.copy(
                _retro_data_path(data, environment_id, "metadata", "metadata", inttype),
                "metadata",
            )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            player_states = metadata.get("default_player_state")
            state = (
                player_states[0]
                if isinstance(player_states, list) and player_states
                else metadata.get("default_state")
            )
            state_hashes["metadata"] = metadata_hash
        if isinstance(state, (bytes, bytearray, memoryview)):
            payload = bytes(state)
            state_hashes["inline"] = hashlib.sha256(payload).hexdigest()
            state = payload
        elif state is not None and state != stable_retro.State.NONE:
            staged_state, state_hash = stage.copy(
                _retro_state_path(data, environment_id, state, inttype), "state"
            )
            state_hashes[str(state)] = state_hash
            state = str(staged_state)
        vector_env = stable_retro.RetroVecEnv(
            environment_id,
            state=state,
            render_mode="rgb_array",
            num_envs=1,
            num_threads=1,
            **kwargs,
        )
    except Exception:
        stage.cleanup()
        raise
    system = getattr(vector_env, "system", None) or stable_retro.get_romfile_system(rom_path)
    buttons = getattr(vector_env, "buttons", None) or data.EMU_INFO[system]["buttons"]
    frame_skip = max(int(kwargs.get("frame_skip", 1)), 1)
    return ProviderSession(
        provider_id=STABLE_RETRO_PROVIDER_ID,
        environment_id=environment_id,
        effective_config=effective,
        vector_env=vector_env,
        system=system,
        buttons=buttons,
        policy_actions=policy_actions,
        fps=60.0 / frame_skip,
        assets={
            "rom_sha256": rom_hash,
            "info_sha256": info_hash,
            "scenario_sha256": scenario_hash,
            "state_sha256": state_hashes,
        },
        asset_stage=stage,
    )


def _mario_session(environment_id: str, config: Mapping[str, Any]) -> ProviderSession:
    from stable_retro import data
    from supermariobrosnes_turbo import (
        NES_BUTTONS,
        SuperMarioBrosNesTurboVecEnv,
        resolve_required_rom_path,
    )
    from supermariobrosnes_turbo.env import _resolve_state_path

    kwargs, policy_actions, effective = _prepare_config(config)
    state = kwargs.pop("state", "Level1-1")
    state_dir = kwargs.pop("state_dir", None)
    stage = _AssetStage()
    try:
        rom_source = resolve_required_rom_path(kwargs.get("rom_path"), environment_id)
        rom_path, rom_hash = stage.copy(rom_source, "rom")
        kwargs["rom_path"] = str(rom_path)
        inttype = _resolve_enum(kwargs.get("inttype", data.Integrations.STABLE), data.Integrations)
        info_path, info_hash = stage.copy(
            _retro_data_path(data, environment_id, kwargs.pop("info", None), "data", inttype),
            "info",
        )
        scenario_path, scenario_hash = stage.copy(
            _retro_data_path(
                data, environment_id, kwargs.pop("scenario", None), "scenario", inttype
            ),
            "scenario",
        )
        kwargs["info"] = str(info_path)
        kwargs["scenario"] = str(scenario_path)
        if state is None or isinstance(state, (bytes, bytearray, memoryview)):
            resolved_state = state
            state_hashes = {}
        else:
            state_source = _resolve_state_path(state, state_dir)
            staged_state, state_hash = stage.copy(state_source, "state")
            resolved_state = str(staged_state)
            state_hashes = {str(state): state_hash}
        vector_env = SuperMarioBrosNesTurboVecEnv(
            environment_id,
            state=resolved_state,
            render_mode="rgb_array",
            num_envs=1,
            num_threads=1,
            **kwargs,
        )
    except Exception:
        stage.cleanup()
        raise
    frame_skip = max(int(kwargs.get("frame_skip", 1)), 1)
    return ProviderSession(
        provider_id=MARIO_TURBO_PROVIDER_ID,
        environment_id=environment_id,
        effective_config=effective,
        vector_env=vector_env,
        system="Nes",
        buttons=NES_BUTTONS,
        policy_actions=policy_actions,
        fps=60.0 / frame_skip,
        assets={
            "rom_sha256": rom_hash,
            "info_sha256": info_hash,
            "scenario_sha256": scenario_hash,
            "state_sha256": state_hashes,
        },
        asset_stage=stage,
    )


def create_provider_session(
    provider_id: str, environment_id: str, config: Mapping[str, Any]
) -> ProviderSession:
    validate_provider_request(config)
    if provider_id == STABLE_RETRO_PROVIDER_ID:
        return _stable_retro_session(environment_id, config)
    if provider_id == MARIO_TURBO_PROVIDER_ID:
        return _mario_session(environment_id, config)
    known = ", ".join(sorted(SUPPORTED_PROVIDER_IDS))
    raise ValueError(f"dataset recording supports providers: {known}")


def space_contract(space: Any) -> dict[str, Any]:
    document: dict[str, Any] = {"type": type(space).__name__, "repr": str(space)}
    for name in ("shape", "dtype", "n", "start"):
        value = getattr(space, name, None)
        if value is None:
            continue
        if name == "shape":
            document[name] = [int(item) for item in value]
        elif name in {"n", "start"}:
            document[name] = int(value)
        else:
            document[name] = str(value)
    for name in ("low", "high"):
        value = getattr(space, name, None)
        if value is not None:
            document[name] = _json_value(value)
    nvec = getattr(space, "nvec", None)
    if nvec is not None:
        document["nvec"] = _json_value(nvec)
    return document


@dataclass(frozen=True)
class EnvironmentArtifact:
    contract_id: str
    document: Mapping[str, Any]


def build_environment_artifact(
    *,
    provider_id: str,
    environment_id: str,
    declared_config: Mapping[str, Any],
    session: ProviderSession,
) -> EnvironmentArtifact:
    document = {
        "document_type": "gymrec.environment",
        "format_version": 1,
        "provider_id": provider_id,
        "provider_contract_version": PROVIDER_CONTRACT_VERSION,
        "environment_id": environment_id,
        "declared_config": copy.deepcopy(dict(declared_config)),
        "effective_config": copy.deepcopy(session.effective_config),
        "provenance": copy.deepcopy(session.provenance),
        "action_space": space_contract(session.env.action_space),
        "observation_space": space_contract(session.env.observation_space),
        "control_profile": session.control_profile,
        "fps": float(session.fps),
    }
    contract_id = hashlib.sha256(canonical_json_bytes(document)).hexdigest()
    return EnvironmentArtifact(contract_id, document)
