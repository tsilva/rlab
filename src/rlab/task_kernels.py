from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import IntEnum
import re
import math
from typing import Any, Protocol, TYPE_CHECKING

import gymnasium as gym
import numpy as np
from numba import njit

if TYPE_CHECKING:
    from rlab.batch_runtime import ProviderDescriptor


SignalSource = str | tuple[str, ...]
IMAGE_CHANNEL_COUNTS = frozenset({1, 3, 4})


def default_task_document(task_id: str) -> dict[str, Any]:
    if task_id == "identity":
        return {
            "id": "identity",
            "action": {"set": "native"},
            "signals": {},
            "events": {},
            "termination": {"max_episode_steps": 4500},
            "reward": {"reward_mode": "native"},
        }
    if task_id != "mario":
        raise ValueError(f"unknown task definition: {task_id!r}")
    return {
        "id": "mario",
        "action": {"set": "native"},
        "signals": {
            "x": ["xscrollHi", "xscrollLo"],
            "score": "score",
            "lives": "lives",
            "level": ["levelHi", "levelLo"],
        },
        "events": {
            "life_loss": {"signal": "lives", "operation": "decrease"},
            "level_change": {"signal": "level", "operation": "change"},
        },
        "termination": {
            "failure": ["life_loss"],
            "success": ["level_change"],
            "max_episode_steps": 4500,
        },
        "reward": {
            "reward_mode": "baseline",
            "use_native_reward": False,
            "clip_rewards": False,
            "progress_reward_cap": 30.0,
            "progress_reward_scale": 1.0,
            "terminal_reward": 50.0,
            "reward_scale": 10.0,
            "time_penalty": 0.0,
            "death_penalty": 25.0,
            "completion_reward": 0.0,
            "score_progress_clipped": False,
        },
    }


def _policy_observation_space(space: gym.Space) -> gym.Space:
    if isinstance(space, gym.spaces.Box) and len(space.shape) == 3:
        channels_first = (
            space.shape[0] in IMAGE_CHANNEL_COUNTS and space.shape[-1] not in IMAGE_CHANNEL_COUNTS
        )
        channels_last = (
            space.shape[-1] in IMAGE_CHANNEL_COUNTS and space.shape[0] not in IMAGE_CHANNEL_COUNTS
        )
        if channels_last:
            return gym.spaces.Box(
                low=np.moveaxis(space.low, -1, 0),
                high=np.moveaxis(space.high, -1, 0),
                dtype=space.dtype,
            )
        if not channels_first:
            raise ValueError(
                "three-dimensional image observations must have an unambiguous channel axis"
            )
    if isinstance(space, gym.spaces.Dict):
        return gym.spaces.Dict(
            {key: _policy_observation_space(value) for key, value in space.spaces.items()}
        )
    if isinstance(space, gym.spaces.Tuple):
        return gym.spaces.Tuple(tuple(_policy_observation_space(value) for value in space.spaces))
    return space


def _encode_policy_observation(observation: Any, native_space: gym.Space) -> Any:
    if isinstance(native_space, gym.spaces.Box) and len(native_space.shape) == 3:
        if (
            native_space.shape[-1] in IMAGE_CHANNEL_COUNTS
            and native_space.shape[0] not in IMAGE_CHANNEL_COUNTS
        ):
            return np.moveaxis(np.asarray(observation), -1, 1)
        return observation
    if isinstance(native_space, gym.spaces.Dict):
        return {
            key: _encode_policy_observation(observation[key], value)
            for key, value in native_space.spaces.items()
        }
    if isinstance(native_space, gym.spaces.Tuple):
        return tuple(
            _encode_policy_observation(item, value)
            for item, value in zip(observation, native_space.spaces, strict=True)
        )
    return observation


def _compile_action_lookup(space: gym.Space, values: Sequence[Any]) -> Any:
    for index, value in enumerate(values):
        if not space.contains(value):
            raise ValueError(
                f"task action codec value {index} is outside native action space {space}"
            )
    if isinstance(space, gym.spaces.Dict):
        return {
            key: _compile_action_lookup(
                child_space,
                [value[key] for value in values],
            )
            for key, child_space in space.spaces.items()
        }
    if isinstance(space, gym.spaces.Tuple):
        return tuple(
            _compile_action_lookup(
                child_space,
                [value[index] for value in values],
            )
            for index, child_space in enumerate(space.spaces)
        )
    return np.asarray(values, dtype=space.dtype)


def _empty_batched_action_lookup(lookup: Any, num_envs: int) -> Any:
    if isinstance(lookup, Mapping):
        return {
            key: _empty_batched_action_lookup(value, num_envs)
            for key, value in lookup.items()
        }
    if isinstance(lookup, tuple):
        return tuple(_empty_batched_action_lookup(value, num_envs) for value in lookup)
    return np.empty((num_envs, *lookup.shape[1:]), dtype=lookup.dtype)


def _map_action_lookup(lookup: Any, indices: np.ndarray, output: Any) -> None:
    if isinstance(lookup, Mapping):
        for key in lookup:
            _map_action_lookup(lookup[key], indices, output[key])
        return
    if isinstance(lookup, tuple):
        for table, target in zip(lookup, output, strict=True):
            _map_action_lookup(table, indices, target)
        return
    np.take(lookup, indices, axis=0, out=output)


@njit(cache=True, nogil=True)
def _identity_step_kernel(
    episode_steps,
    max_episode_steps,
    terminated,
    truncated,
    outcomes,
    event_bits,
):
    for lane in range(episode_steps.shape[0]):
        episode_steps[lane] += 1
        timed_out = max_episode_steps > 0 and episode_steps[lane] >= max_episode_steps
        terminated[lane] = False
        truncated[lane] = timed_out
        outcomes[lane] = 3 if timed_out else 0
        event_bits[lane] = 0


@njit(cache=True, nogil=True)
def _identity_equals_for_event_kernel(
    values,
    expected_value,
    required_steps,
    consecutive_steps,
    event_bit,
    event_outcome,
    terminated,
    truncated,
    outcomes,
    event_bits,
):
    for lane in range(values.shape[0]):
        if values[lane] == expected_value:
            consecutive_steps[lane] += 1
        else:
            consecutive_steps[lane] = 0
        if consecutive_steps[lane] != required_steps:
            continue

        event_bits[lane] |= event_bit
        if event_outcome == 2:
            terminated[lane] = True
            truncated[lane] = False
            outcomes[lane] = 2
        elif event_outcome == 1 and outcomes[lane] != 2:
            terminated[lane] = True
            truncated[lane] = False
            outcomes[lane] = 1
        elif event_outcome == 3 and not terminated[lane]:
            truncated[lane] = True
            outcomes[lane] = 3


@njit(cache=True, nogil=True)
def _identity_decrease_event_kernel(
    values,
    previous_values,
    previous_valid,
    transition_sources,
    transition_targets,
    event_bit,
    event_outcome,
    terminated,
    truncated,
    outcomes,
    event_bits,
):
    for lane in range(values.shape[0]):
        current_value = values[lane]
        transition_sources[lane] = previous_values[lane] if previous_valid[lane] else current_value
        transition_targets[lane] = current_value
        decreased = previous_valid[lane] and current_value < previous_values[lane]
        previous_values[lane] = current_value
        previous_valid[lane] = True
        if not decreased:
            continue

        event_bits[lane] |= event_bit
        if event_outcome == 2:
            terminated[lane] = True
            truncated[lane] = False
            outcomes[lane] = 2
        elif event_outcome == 1 and outcomes[lane] != 2:
            terminated[lane] = True
            truncated[lane] = False
            outcomes[lane] = 1
        elif event_outcome == 3 and not terminated[lane]:
            truncated[lane] = True
            outcomes[lane] = 3


@njit(cache=True, nogil=True)
def _mario_step_kernel(
    x_hi,
    x_lo,
    x_uses_pair,
    score,
    lives,
    level_hi,
    level_lo,
    game_mode,
    native_rewards,
    provider_terminated,
    provider_truncated,
    state_x,
    state_max_x,
    state_level_max_x,
    state_completed_base,
    state_completed_count,
    state_global_x,
    state_max_global_x,
    state_score,
    state_lives,
    state_level_hi,
    state_level_lo,
    episode_steps,
    last_progress_step,
    rewards,
    task_terminated,
    task_truncated,
    outcomes,
    event_bits,
    progress_delta,
    score_delta,
    life_lost,
    level_changed,
    completion,
    game_complete,
    stalled,
    raw_reward,
    progress_component,
    native_component,
    progress_reward_component,
    score_reward_component,
    completion_reward_component,
    death_penalty_component,
    time_penalty_component,
    life_transition_source,
    life_transition_target,
    level_transition_source,
    level_transition_target,
    reward_mode,
    use_native_reward,
    clip_rewards,
    progress_reward_cap,
    progress_reward_scale,
    terminal_reward,
    reward_scale,
    time_penalty,
    death_penalty,
    completion_reward,
    score_progress_clipped,
    max_episode_steps,
    no_progress_timeout_steps,
    no_progress_min_delta,
    final_level_hi,
    final_level_lo,
    game_complete_mode,
    terminate_on_life_loss,
    terminate_on_level_change,
    terminate_on_game_complete,
    stall_is_failure,
    emit_life_loss,
    emit_level_change,
    emit_game_complete,
    emit_stalled,
):
    any_events = False
    for lane in range(state_x.shape[0]):
        current_x = int(x_hi[lane])
        if x_uses_pair:
            current_x = current_x * 256 + int(x_lo[lane])
        current_score = int(score[lane])
        current_lives = int(lives[lane])
        current_level_hi = int(level_hi[lane])
        current_level_lo = int(level_lo[lane])
        current_game_mode = int(game_mode[lane])

        previous_lives = state_lives[lane]
        previous_level_hi = state_level_hi[lane]
        previous_level_lo = state_level_lo[lane]
        lost_life = current_lives < previous_lives
        changed_level = (
            current_level_hi != previous_level_hi or current_level_lo != previous_level_lo
        )
        completed_level = changed_level and not lost_life
        completed_game = (
            not lost_life
            and current_level_hi == final_level_hi
            and current_level_lo == final_level_lo
            and current_game_mode == game_complete_mode
        )
        completed = completed_level or completed_game

        life_transition_source[lane] = previous_lives
        life_transition_target[lane] = current_lives
        level_transition_source[lane, 0] = previous_level_hi
        level_transition_source[lane, 1] = previous_level_lo
        level_transition_target[lane, 0] = current_level_hi
        level_transition_target[lane, 1] = current_level_lo

        if completed:
            state_completed_base[lane] += state_level_max_x[lane]
            state_completed_count[lane] += 1
            state_level_max_x[lane] = 0
        effective_x = 0 if changed_level or completed_game else current_x
        if effective_x > state_level_max_x[lane]:
            state_level_max_x[lane] = effective_x
        global_x = state_completed_base[lane] + effective_x
        global_max = state_completed_base[lane] + state_level_max_x[lane]
        delta = global_max - state_max_global_x[lane]
        if delta < 0:
            delta = 0
        if global_max > state_max_global_x[lane]:
            state_max_global_x[lane] = global_max
        current_score_delta = current_score - state_score[lane]
        if current_score_delta < 0:
            current_score_delta = 0

        if delta > no_progress_min_delta:
            last_progress_step[lane] = episode_steps[lane]
        episode_steps[lane] += 1
        is_stalled = (
            no_progress_timeout_steps > 0
            and episode_steps[lane] - last_progress_step[lane] >= no_progress_timeout_steps
        )

        terminated = (terminate_on_life_loss and lost_life) or (
            terminate_on_level_change and completed_level
        ) or (
            terminate_on_game_complete and completed_game
        )
        if stall_is_failure and is_stalled:
            terminated = True
        truncated = (is_stalled and not stall_is_failure) or (
            max_episode_steps > 0 and episode_steps[lane] >= max_episode_steps
        )
        if terminated:
            truncated = False

        bits = 0
        if lost_life and emit_life_loss:
            bits |= 1
        if changed_level and emit_level_change:
            bits |= 2
        if is_stalled and emit_stalled:
            bits |= 4
        if completed_game and emit_game_complete:
            bits |= 16
        timed_out = truncated or provider_truncated[lane]
        any_events = any_events or bits != 0

        outcome = 0
        if timed_out:
            outcome = 3
        if completed and not lost_life:
            outcome = 1
        if (
            lost_life
            or (is_stalled and stall_is_failure)
            or (provider_terminated[lane] and not completed)
        ):
            outcome = 2

        capped_progress = float(delta)
        if capped_progress > progress_reward_cap:
            capped_progress = progress_reward_cap
        component_progress = capped_progress
        native_part = 0.0
        progress_part = 0.0
        score_part = 0.0
        completion_part = 0.0
        death_part = 0.0
        raw = float(native_rewards[lane])
        done = provider_terminated[lane] or provider_truncated[lane] or terminated or truncated

        if reward_mode == 0:  # native
            native_part = float(native_rewards[lane])
        elif reward_mode == 1:  # bounded
            raw = capped_progress
            if completed:
                raw = terminal_reward
            if lost_life:
                raw = -terminal_reward
            divisor = reward_scale if reward_scale else 1.0
            if completed:
                completion_part = terminal_reward / divisor
            elif lost_life:
                death_part = -terminal_reward / divisor
            else:
                progress_part = capped_progress / divisor
        elif reward_mode == 2:  # baseline
            raw = float(native_rewards[lane]) + float(current_score_delta) / 40.0
            if completed:
                raw += terminal_reward
            elif done:
                raw -= terminal_reward
            divisor = reward_scale if reward_scale else 1.0
            native_part = float(native_rewards[lane]) / divisor
            score_part = (float(current_score_delta) / 40.0) / divisor
            if completed:
                completion_part = terminal_reward / divisor
            elif done:
                death_part = -terminal_reward / divisor
        else:
            if use_native_reward:
                native_part = float(native_rewards[lane])
            if reward_mode == 3:  # score
                component_progress = capped_progress if score_progress_clipped else float(delta)
                score_part = 0.01 * float(current_score_delta)
            else:  # additive
                component_progress = float(delta)
            progress_part = progress_reward_scale * component_progress
            if completed:
                completion_part = completion_reward
            if lost_life:
                death_part = -death_penalty
            raw = native_part + progress_part + score_part + completion_part + death_part

        if reward_mode == 1 or reward_mode == 2:
            shaped = raw / (reward_scale if reward_scale else 1.0)
        else:
            shaped = raw
        shaped -= time_penalty
        if clip_rewards:
            shaped = 1.0 if shaped > 0 else -1.0 if shaped < 0 else 0.0

        rewards[lane] = shaped
        task_terminated[lane] = terminated
        task_truncated[lane] = truncated
        outcomes[lane] = outcome
        event_bits[lane] = bits
        progress_delta[lane] = delta
        score_delta[lane] = current_score_delta
        life_lost[lane] = lost_life
        level_changed[lane] = changed_level
        completion[lane] = completed
        game_complete[lane] = completed_game
        stalled[lane] = is_stalled
        raw_reward[lane] = raw
        progress_component[lane] = component_progress
        native_component[lane] = native_part
        progress_reward_component[lane] = progress_part
        score_reward_component[lane] = score_part
        completion_reward_component[lane] = completion_part
        death_penalty_component[lane] = death_part
        time_penalty_component[lane] = -time_penalty

        state_x[lane] = effective_x
        state_max_x[lane] = state_level_max_x[lane]
        state_global_x[lane] = global_x
        state_score[lane] = current_score
        state_lives[lane] = current_lives
        state_level_hi[lane] = current_level_hi
        state_level_lo[lane] = current_level_lo
    return any_events


@njit(cache=True, nogil=True)
def _mario_step_kernel_packed(signals, state, outputs, parameters):
    return _mario_step_kernel(*signals, *state, *outputs, *parameters)


class Outcome(IntEnum):
    NEUTRAL = 0
    SUCCESS = 1
    FAILURE = 2
    TIMEOUT = 3


@dataclass(frozen=True)
class TaskStep:
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    outcomes: np.ndarray
    event_bits: np.ndarray
    metrics: Mapping[str, np.ndarray]
    event_transitions: Mapping[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)


class BoundTaskKernel(Protocol):
    num_envs: int
    observation_space: gym.Space
    action_space: gym.Space
    event_names: tuple[str, ...]
    observation_encoding_is_view: bool

    def map_actions(self, actions: Any) -> Any: ...

    def encode_observations(self, observations: Any) -> Any: ...

    def process(
        self,
        native_rewards: np.ndarray,
        provider_terminated: np.ndarray,
        provider_truncated: np.ndarray,
        signals: Mapping[str, Any],
    ) -> TaskStep: ...

    def on_reset(
        self,
        reset_observations: Any,
        reset_signals: Mapping[str, Any],
        mask: np.ndarray,
    ) -> None: ...


class SignalBindings:
    """Resolve semantic task signals to dense provider info columns once."""

    def __init__(
        self,
        descriptor: ProviderDescriptor,
        bindings: Mapping[str, SignalSource],
        num_envs: int,
    ):
        self._bindings = dict(bindings)
        self._specs = dict(descriptor.signal_schema)
        self.num_envs = int(num_envs)
        available = set(descriptor.signal_schema)
        missing = sorted(
            {
                name
                for source in self._bindings.values()
                for name in ((source,) if isinstance(source, str) else source)
                if name not in available
            }
        )
        if missing:
            raise ValueError(
                f"provider {descriptor.provider_id!r} does not expose task signals: "
                + ", ".join(missing)
            )
        unavailable = sorted(
            {
                name
                for source in self._bindings.values()
                for name in ((source,) if isinstance(source, str) else source)
                if not self._specs[name].available_on_step
            }
        )
        if unavailable:
            raise ValueError(
                "task signals must be available on step: " + ", ".join(unavailable)
            )

    def available_on_reset(self, semantic_name: str) -> bool:
        source = self.source(semantic_name)
        names = (source,) if isinstance(source, str) else source
        return all(self._specs[name].available_on_reset for name in names)

    def source(self, semantic_name: str) -> SignalSource:
        try:
            return self._bindings[semantic_name]
        except KeyError as exc:
            raise KeyError(f"unknown semantic signal {semantic_name!r}") from exc

    def columns(
        self,
        semantic_name: str,
        signals: Mapping[str, Any],
        *,
        mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, ...]:
        source = self.source(semantic_name)
        names = (source,) if isinstance(source, str) else source
        columns: list[np.ndarray] = []
        for name in names:
            if name not in signals:
                raise ValueError(f"provider did not return required signal {name!r}")
            values = np.asarray(signals[name])
            spec = self._specs[name]
            expected_shape = (self.num_envs, *spec.shape)
            if values.shape != expected_shape:
                raise ValueError(
                    f"signal {name!r} must have shape {expected_shape}, got {values.shape}"
                )
            if values.dtype != spec.dtype:
                raise ValueError(
                    f"signal {name!r} must have dtype {spec.dtype}, got {values.dtype}"
                )
            presence = signals.get(f"_{name}")
            if presence is not None:
                present = np.asarray(presence, dtype=bool)
                required = np.ones(self.num_envs, dtype=bool) if mask is None else mask
                if present.shape != (self.num_envs,) or np.any(required & ~present):
                    raise ValueError(f"required signal {name!r} is absent for active lanes")
            columns.append(values)
        return tuple(columns)

    def unchecked_columns(
        self, semantic_name: str, signals: Mapping[str, Any]
    ) -> tuple[np.ndarray, ...]:
        source = self.source(semantic_name)
        names = (source,) if isinstance(source, str) else source
        return tuple(signals[name] for name in names)

    def scalar(
        self,
        semantic_name: str,
        signals: Mapping[str, Any],
        *,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        columns = self.columns(semantic_name, signals, mask=mask)
        if len(columns) != 1:
            raise ValueError(f"semantic signal {semantic_name!r} is not scalar")
        values = columns[0]
        if values.ndim != 1:
            raise ValueError(f"semantic signal {semantic_name!r} must be scalar per lane")
        return values

    def scalar_dtype(self, semantic_name: str) -> np.dtype:
        source = self.source(semantic_name)
        names = (source,) if isinstance(source, str) else source
        if len(names) != 1 or self._specs[names[0]].shape:
            raise ValueError(f"semantic signal {semantic_name!r} must be scalar per lane")
        return self._specs[names[0]].dtype


@dataclass(frozen=True)
class IdentityEvent:
    name: str
    signal: str
    operation: str
    outcome: Outcome = Outcome.NEUTRAL
    value: int | float | None = None
    steps: int = 0


class IdentityTaskDefinition:
    def __init__(
        self,
        *,
        observation_mask: tuple[int, int, int, int] | None = None,
        observation_mask_fill: int = 0,
        observation_source_shape: tuple[int, int] | None = None,
        max_episode_steps: int = 0,
        action_values: Sequence[Any] | None = None,
        signals: Mapping[str, SignalSource] | None = None,
        events: Mapping[str, Mapping[str, Any]] | None = None,
        termination: Mapping[str, Any] | None = None,
    ):
        self.observation_mask = observation_mask
        self.observation_mask_fill = int(observation_mask_fill)
        self.observation_source_shape = observation_source_shape
        self.max_episode_steps = int(max_episode_steps)
        self.action_values = None if action_values is None else tuple(action_values)
        self.signals = {
            str(name): source if isinstance(source, str) else tuple(source)
            for name, source in (signals or {}).items()
        }
        raw_events = dict(events or {})
        if len(raw_events) > 64:
            raise ValueError("identity task supports at most 64 events")
        event_outcomes: dict[str, Outcome] = {}
        termination = dict(termination or {})
        for outcome_name, outcome in (
            ("success", Outcome.SUCCESS),
            ("failure", Outcome.FAILURE),
            ("timeout", Outcome.TIMEOUT),
            ("neutral", Outcome.NEUTRAL),
        ):
            for name in termination.get(outcome_name, ()):
                event_name = str(name)
                if event_name in event_outcomes:
                    raise ValueError(f"identity event {event_name!r} has multiple outcomes")
                event_outcomes[event_name] = outcome
        compiled_events: list[IdentityEvent] = []
        for name, rule in raw_events.items():
            operation = str(rule.get("operation", ""))
            if operation not in {"decrease", "equals_for"}:
                raise ValueError(
                    f"identity event {name!r} supports only operations 'decrease' and 'equals_for'"
                )
            if operation == "equals_for":
                value = rule.get("value")
                steps = rule.get("steps")
                if not isinstance(value, int | float) or isinstance(value, bool):
                    raise ValueError(
                        f"identity equals_for event {name!r} requires a numeric value"
                    )
                if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
                    raise ValueError(
                        f"identity equals_for event {name!r} requires positive steps"
                    )
            compiled_events.append(
                IdentityEvent(
                    name=str(name),
                    signal=str(rule["signal"]),
                    operation=operation,
                    outcome=event_outcomes.get(str(name), Outcome.NEUTRAL),
                    value=rule.get("value"),
                    steps=int(rule.get("steps", 0)),
                )
            )
        self.events = tuple(compiled_events)
        if self.max_episode_steps < 0:
            raise ValueError("max_episode_steps must be non-negative")

    def bind(self, descriptor: ProviderDescriptor, num_envs: int) -> IdentityTaskKernel:
        return IdentityTaskKernel(
            descriptor,
            num_envs,
            observation_mask=self.observation_mask,
            observation_mask_fill=self.observation_mask_fill,
            observation_source_shape=self.observation_source_shape,
            max_episode_steps=self.max_episode_steps,
            action_values=self.action_values,
            signals=self.signals,
            events=self.events,
        )


class IdentityTaskKernel:
    def __init__(
        self,
        descriptor: ProviderDescriptor,
        num_envs: int,
        *,
        observation_mask: tuple[int, int, int, int] | None = None,
        observation_mask_fill: int = 0,
        observation_source_shape: tuple[int, int] | None = None,
        max_episode_steps: int = 0,
        action_values: Sequence[Any] | None = None,
        signals: Mapping[str, SignalSource] | None = None,
        events: Sequence[IdentityEvent] = (),
    ):
        self.num_envs = int(num_envs)
        self._native_observation_space = descriptor.native_observation_space
        self.observation_space = _policy_observation_space(self._native_observation_space)
        self._native_action_space = descriptor.native_action_space
        self._action_lookup = None
        self._action_buffer = None
        if action_values is None:
            self.action_space = self._native_action_space
        else:
            if not action_values:
                raise ValueError("task action codec values must not be empty")
            self._action_lookup = _compile_action_lookup(
                self._native_action_space,
                action_values,
            )
            self._action_buffer = _empty_batched_action_lookup(
                self._action_lookup,
                self.num_envs,
            )
            self.action_space = gym.spaces.Discrete(len(action_values))
        self._rewards = np.empty(self.num_envs, dtype=np.float32)
        self._terminated = np.zeros(self.num_envs, dtype=bool)
        self._truncated = np.zeros(self.num_envs, dtype=bool)
        self._outcomes = np.zeros(self.num_envs, dtype=np.uint8)
        self._events = np.zeros(self.num_envs, dtype=np.uint64)
        self._episode_steps = np.zeros(self.num_envs, dtype=np.int64)
        self._max_episode_steps = int(max_episode_steps)
        self._event_configs = tuple(events)
        self.event_names = tuple(event.name for event in self._event_configs)
        self.has_events = bool(self._event_configs)
        self._signal_bindings = (
            SignalBindings(descriptor, signals or {}, self.num_envs)
            if self._event_configs
            else None
        )
        self._event_consecutive_steps = tuple(
            np.zeros(self.num_envs, dtype=np.int64) for _event in self._event_configs
        )
        event_dtypes = tuple(
            self._signal_bindings.scalar_dtype(event.signal) for event in self._event_configs
        )
        for event, dtype in zip(self._event_configs, event_dtypes, strict=True):
            if event.operation == "decrease" and (
                not np.issubdtype(dtype, np.number) or np.issubdtype(dtype, np.bool_)
            ):
                raise ValueError(
                    f"identity decrease event {event.name!r} requires a numeric signal"
                )
        self._event_previous_values = tuple(
            np.empty(self.num_envs, dtype=dtype) for dtype in event_dtypes
        )
        self._event_previous_valid = tuple(
            np.zeros(self.num_envs, dtype=np.bool_) for _event in self._event_configs
        )
        self._event_transition_sources = tuple(
            np.empty(self.num_envs, dtype=dtype) for dtype in event_dtypes
        )
        self._event_transition_targets = tuple(
            np.empty(self.num_envs, dtype=dtype) for dtype in event_dtypes
        )
        self._event_transitions = {
            event.name: (
                self._event_transition_sources[index],
                self._event_transition_targets[index],
            )
            for index, event in enumerate(self._event_configs)
            if event.operation == "decrease"
        }
        self._observation_mask = observation_mask
        self._observation_mask_fill = int(observation_mask_fill)
        self._observation_source_shape = observation_source_shape
        self._masked_observation_buffer: np.ndarray | None = None
        self.observation_encoding_is_view = observation_mask is None

    def map_actions(self, actions: Any) -> Any:
        if self._action_lookup is None:
            return actions
        indices = np.asarray(actions, dtype=np.int64).reshape(-1)
        if indices.shape != (self.num_envs,):
            raise ValueError(f"expected {self.num_envs} task action ids, got {indices.shape}")
        if np.any(indices < 0) or np.any(indices >= self.action_space.n):
            raise ValueError("task action id is outside the configured action codec")
        _map_action_lookup(self._action_lookup, indices, self._action_buffer)
        return self._action_buffer

    def encode_observations(self, observations: Any) -> Any:
        encoded = _encode_policy_observation(observations, self._native_observation_space)
        if self._observation_mask is None:
            return encoded
        if isinstance(encoded, Mapping):
            if "image" not in encoded:
                raise ValueError("observation masking requires an image observation")
            result = dict(encoded)
            result["image"] = self._mask_observations(encoded["image"])
            return result
        return self._mask_observations(encoded)

    def _mask_observations(self, observations: Any) -> np.ndarray:
        source = np.asarray(observations)
        if source.ndim != 4:
            raise ValueError("observation masking requires batched channel-first images")
        if self._masked_observation_buffer is None:
            self._masked_observation_buffer = np.empty_like(source)
        np.copyto(self._masked_observation_buffer, source)
        target = self._masked_observation_buffer
        height, width = int(target.shape[-2]), int(target.shape[-1])
        source_height, source_width = self._observation_source_shape or (height, width)
        crop = self._observation_mask or (0, 0, 0, 0)

        def scaled(value: int, source_size: int, target_size: int) -> int:
            return min(
                target_size,
                int(math.ceil(float(value) * target_size / source_size)),
            )

        top = scaled(crop[0], source_height, height)
        right = scaled(crop[1], source_width, width)
        bottom = scaled(crop[2], source_height, height)
        left = scaled(crop[3], source_width, width)
        if top:
            target[:, :, :top, :] = self._observation_mask_fill
        if right:
            target[:, :, :, -right:] = self._observation_mask_fill
        if bottom:
            target[:, :, -bottom:, :] = self._observation_mask_fill
        if left:
            target[:, :, :, :left] = self._observation_mask_fill
        return target

    def process(
        self,
        native_rewards: np.ndarray,
        provider_terminated: np.ndarray,
        provider_truncated: np.ndarray,
        signals: Mapping[str, Any],
    ) -> TaskStep:
        del provider_terminated, provider_truncated
        np.copyto(self._rewards, np.asarray(native_rewards, dtype=np.float32))
        _identity_step_kernel(
            self._episode_steps,
            self._max_episode_steps,
            self._terminated,
            self._truncated,
            self._outcomes,
            self._events,
        )
        if self._signal_bindings is not None:
            for index, event in enumerate(self._event_configs):
                values = self._signal_bindings.scalar(event.signal, signals)
                if event.operation == "equals_for":
                    _identity_equals_for_event_kernel(
                        values,
                        event.value,
                        event.steps,
                        self._event_consecutive_steps[index],
                        np.uint64(1 << index),
                        int(event.outcome),
                        self._terminated,
                        self._truncated,
                        self._outcomes,
                        self._events,
                    )
                else:
                    _identity_decrease_event_kernel(
                        values,
                        self._event_previous_values[index],
                        self._event_previous_valid[index],
                        self._event_transition_sources[index],
                        self._event_transition_targets[index],
                        np.uint64(1 << index),
                        int(event.outcome),
                        self._terminated,
                        self._truncated,
                        self._outcomes,
                        self._events,
                    )
        return TaskStep(
            self._rewards,
            self._terminated,
            self._truncated,
            self._outcomes,
            self._events,
            {},
            self._event_transitions,
        )

    def on_reset(
        self,
        reset_observations: Any,
        reset_signals: Mapping[str, Any],
        mask: np.ndarray,
    ) -> bool:
        del reset_observations
        mask = np.asarray(mask, dtype=bool)
        if self._signal_bindings is not None:
            for index, (event, consecutive_steps) in enumerate(
                zip(
                    self._event_configs,
                    self._event_consecutive_steps,
                    strict=True,
                )
            ):
                available_on_reset = self._signal_bindings.available_on_reset(event.signal)
                reset_values = (
                    self._signal_bindings.scalar(event.signal, reset_signals, mask=mask)
                    if available_on_reset
                    else None
                )
                consecutive_steps[mask] = 0
                if event.operation == "decrease":
                    if reset_values is None:
                        self._event_previous_valid[index][mask] = False
                    else:
                        self._event_previous_values[index][mask] = reset_values[mask]
                        self._event_previous_valid[index][mask] = True
        self._episode_steps[mask] = 0


@dataclass(frozen=True)
class MarioTaskConfig:
    x: SignalSource = ("xscrollHi", "xscrollLo")
    score: SignalSource = "score"
    lives: SignalSource = "lives"
    level: SignalSource = ("levelHi", "levelLo")
    game_mode: SignalSource | None = None
    action_masks: np.ndarray | None = None
    reward_mode: str = "baseline"
    use_native_reward: bool = False
    clip_rewards: bool = False
    progress_reward_cap: float = 30.0
    progress_reward_scale: float = 1.0
    terminal_reward: float = 50.0
    reward_scale: float = 10.0
    time_penalty: float = 0.0
    death_penalty: float = 25.0
    completion_reward: float = 0.0
    score_progress_clipped: bool = False
    max_episode_steps: int = 0
    no_progress_timeout_steps: int = 0
    no_progress_min_delta: int = 0
    terminate_on_life_loss: bool = True
    terminate_on_level_change: bool = True
    game_complete_level: tuple[int, int] = (-1, -1)
    game_complete_mode: int = -1
    terminate_on_game_complete: bool = False
    stall_is_failure: bool = False
    task_conditioning: bool = False
    task_values: tuple[tuple[int | str, ...], ...] = ()
    emit_life_loss: bool = True
    emit_level_change: bool = True
    emit_game_complete: bool = False
    emit_stalled: bool = True

    @classmethod
    def from_env_config(cls, config: Any) -> MarioTaskConfig:
        from rlab.targets import target_for_game

        task = config.task if isinstance(getattr(config, "task", None), Mapping) else {}
        action = task.get("action", {}) if isinstance(task.get("action"), Mapping) else {}
        action_set = str(action.get("set", "native"))
        target = target_for_game(config.game)
        masks = target.action_masks_for_set(action_set)
        if not masks and config.env_provider == "stable-retro-turbo":
            from rlab.action_contract import configured_action_values

            masks = configured_action_values(config) or ()
        action_masks = np.stack(masks).astype(np.int8) if masks else None
        signals = task.get("signals", {}) if isinstance(task.get("signals"), Mapping) else {}
        termination = (
            task.get("termination", {}) if isinstance(task.get("termination"), Mapping) else {}
        )
        events = task.get("events", {}) if isinstance(task.get("events"), Mapping) else {}
        supported_events = {"life_loss", "level_change", "game_complete", "stalled"}
        unknown_events = sorted(set(events) - supported_events)
        if unknown_events:
            raise ValueError(
                "Mario task does not implement event(s): " + ", ".join(unknown_events)
            )
        expected_events = {
            "life_loss": ("lives", "decrease"),
            "level_change": ("level", "change"),
            "game_complete": ("game_mode", "equals"),
            "stalled": ("x", "unchanged_for"),
        }
        for event_name, (expected_signal, expected_operation) in expected_events.items():
            rule = events.get(event_name)
            if rule is None:
                continue
            if not isinstance(rule, Mapping) or (
                rule.get("signal"), rule.get("operation")
            ) != (expected_signal, expected_operation):
                raise ValueError(
                    f"Mario event {event_name!r} requires signal={expected_signal!r} "
                    f"and operation={expected_operation!r}"
                )
        game_complete_rule = events.get("game_complete", {})
        game_complete_when = (
            game_complete_rule.get("when", {})
            if isinstance(game_complete_rule, Mapping)
            else {}
        )
        raw_game_complete_level = (
            game_complete_when.get("value", (-1, -1))
            if isinstance(game_complete_when, Mapping)
            else (-1, -1)
        )
        if game_complete_rule and (
            not isinstance(game_complete_when, Mapping)
            or game_complete_when.get("signal") != "level"
        ):
            raise ValueError("Mario game_complete event requires a level guard")
        if (
            not isinstance(raw_game_complete_level, Sequence)
            or isinstance(raw_game_complete_level, str | bytes)
            or len(raw_game_complete_level) != 2
            or any(
                not isinstance(value, int) or isinstance(value, bool)
                for value in raw_game_complete_level
            )
        ):
            raise ValueError("Mario game_complete event value must be a pair of integers")
        game_complete_level = (
            int(raw_game_complete_level[0]),
            int(raw_game_complete_level[1]),
        )
        game_complete_mode = (
            game_complete_rule.get("value", -1)
            if isinstance(game_complete_rule, Mapping)
            else -1
        )
        if not isinstance(game_complete_mode, int) or isinstance(game_complete_mode, bool):
            raise ValueError("Mario game_complete event value must be an integer")
        game_mode_source = signals.get("game_mode")
        if game_complete_rule and not (
            isinstance(game_mode_source, str) and game_mode_source.strip()
        ):
            raise ValueError("Mario game_complete event requires a scalar game_mode signal")
        reward = task.get("reward", {}) if isinstance(task.get("reward"), Mapping) else {}
        conditioning = (
            task.get("conditioning", {}) if isinstance(task.get("conditioning"), Mapping) else {}
        )
        task_conditioning = bool(conditioning.get("enabled", False))
        raw_task_values = conditioning.get("values", ())
        task_values = tuple(tuple(value) for value in raw_task_values)
        if task_conditioning and not task_values:
            parsed_values: list[tuple[int, int]] = []
            for state in config.states or ((config.state,) if config.state else ()):
                match = re.fullmatch(r"Level(\d+)-(\d+)", str(state))
                if match is None:
                    raise ValueError(
                        "Mario task conditioning requires task_conditioning_info_values "
                        f"for non-level state {state!r}"
                    )
                parsed_values.append((int(match.group(1)) - 1, int(match.group(2)) - 1))
            task_values = tuple(parsed_values)
        failure_events = set(termination.get("failure", ()))
        success_events = set(termination.get("success", ()))
        stalled_rule = events.get("stalled", {})
        stalled_steps = (
            int(stalled_rule.get("steps", 0)) if isinstance(stalled_rule, Mapping) else 0
        )

        def reward_value(name: str, default: Any) -> Any:
            return reward.get(name, default)

        def signal_value(name: str, default: SignalSource) -> SignalSource:
            value = signals.get(name, default)
            return value if isinstance(value, str) else tuple(value)

        reward_mode = str(reward_value("reward_mode", "baseline"))
        if reward_mode not in {"native", "bounded", "baseline", "score", "additive"}:
            raise ValueError(f"unsupported Mario reward mode {reward_mode!r}")

        return cls(
            x=signal_value("x", cls.x),
            score=signal_value("score", cls.score),
            lives=signal_value("lives", cls.lives),
            level=signal_value("level", cls.level),
            game_mode=game_mode_source if game_complete_rule else None,
            action_masks=action_masks,
            reward_mode=reward_mode,
            use_native_reward=reward.get("use_native_reward", False),
            clip_rewards=reward_value("clip_rewards", False),
            progress_reward_cap=reward_value("progress_reward_cap", 30.0),
            progress_reward_scale=reward_value("progress_reward_scale", 1.0),
            terminal_reward=reward_value("terminal_reward", 50.0),
            reward_scale=reward_value("reward_scale", 10.0),
            time_penalty=reward_value("time_penalty", 0.0),
            death_penalty=reward_value("death_penalty", 25.0),
            completion_reward=reward_value("completion_reward", 0.0),
            score_progress_clipped=reward_value("score_progress_clipped", False),
            max_episode_steps=int(termination.get("max_episode_steps", 0)),
            no_progress_timeout_steps=stalled_steps,
            no_progress_min_delta=int(termination.get("no_progress_min_delta", 0)),
            game_complete_level=game_complete_level,
            game_complete_mode=int(game_complete_mode),
            terminate_on_life_loss="life_loss" in failure_events,
            terminate_on_level_change="level_change" in success_events,
            terminate_on_game_complete="game_complete" in success_events,
            stall_is_failure="stalled" in failure_events,
            task_conditioning=task_conditioning,
            task_values=task_values,
            emit_life_loss="life_loss" in events,
            emit_level_change="level_change" in events,
            emit_game_complete="game_complete" in events,
            emit_stalled="stalled" in events,
        )


class MarioTaskDefinition:
    def __init__(self, config: MarioTaskConfig):
        self.config = config

    def bind(self, descriptor: ProviderDescriptor, num_envs: int) -> MarioTaskKernel:
        return MarioTaskKernel(descriptor, self.config, num_envs)


class MarioTaskKernel:
    LIFE_LOSS = np.uint64(1 << 0)
    LEVEL_CHANGE = np.uint64(1 << 1)
    STALLED = np.uint64(1 << 2)
    TIMEOUT = np.uint64(1 << 3)
    GAME_COMPLETE = np.uint64(1 << 4)
    event_names = ("life_loss", "level_change", "stalled", "timeout", "game_complete")

    def __init__(
        self,
        descriptor: ProviderDescriptor,
        config: MarioTaskConfig,
        num_envs: int,
    ):
        self.config = config
        self.num_envs = int(num_envs)
        self._native_observation_space = descriptor.native_observation_space
        self.observation_space = _policy_observation_space(self._native_observation_space)
        signal_sources: dict[str, SignalSource] = {
            "x": config.x,
            "score": config.score,
            "lives": config.lives,
            "level": config.level,
        }
        if config.game_mode is not None:
            signal_sources["game_mode"] = config.game_mode
        self.bindings = SignalBindings(descriptor, signal_sources, self.num_envs)
        self._x_signal_names = (config.x,) if isinstance(config.x, str) else tuple(config.x)
        self._score_signal_name = str(config.score)
        self._lives_signal_name = str(config.lives)
        self._level_signal_names = (
            (config.level,) if isinstance(config.level, str) else tuple(config.level)
        )
        self._game_mode_signal_name = (
            str(config.game_mode) if config.game_mode is not None else None
        )
        self._action_masks = None
        if config.action_masks is None:
            self.action_space = descriptor.native_action_space
            self._action_buffer = None
        else:
            masks = np.asarray(config.action_masks)
            if masks.ndim < 2 or len(masks) == 0:
                raise ValueError("Mario action_masks must be a non-empty action table")
            self._action_masks = masks.copy()
            self._action_buffer = np.empty((self.num_envs, *masks.shape[1:]), dtype=masks.dtype)
            self.action_space = gym.spaces.Discrete(len(masks))

        self._task_observation: np.ndarray | None = None
        self._task_values: np.ndarray | None = None
        if config.task_conditioning:
            if not config.task_values:
                raise ValueError("Mario task conditioning requires task_values")
            task_values = np.asarray(config.task_values)
            if task_values.ndim != 2 or task_values.shape[1] != 2:
                raise ValueError("Mario task_values must contain level high/low pairs")
            self._task_values = task_values.astype(np.int64)
            self._task_observation = np.zeros(
                (self.num_envs, len(config.task_values)), dtype=np.float32
            )
            self.observation_space = gym.spaces.Dict(
                {
                    "image": _policy_observation_space(self._native_observation_space),
                    "task": gym.spaces.Box(
                        low=0.0,
                        high=1.0,
                        shape=(len(config.task_values),),
                        dtype=np.float32,
                    ),
                }
            )
        self.observation_encoding_is_view = self._task_observation is None

        self._rewards = np.zeros(self.num_envs, dtype=np.float32)
        self._terminated = np.zeros(self.num_envs, dtype=bool)
        self._truncated = np.zeros(self.num_envs, dtype=bool)
        self._outcomes = np.zeros(self.num_envs, dtype=np.uint8)
        self._events = np.zeros(self.num_envs, dtype=np.uint64)
        self._raw_reward = np.zeros(self.num_envs, dtype=np.float32)
        self._progress_component = np.zeros(self.num_envs, dtype=np.float32)
        self._native_reward_component = np.zeros(self.num_envs, dtype=np.float32)
        self._progress_reward_component = np.zeros(self.num_envs, dtype=np.float32)
        self._score_reward_component = np.zeros(self.num_envs, dtype=np.float32)
        self._completion_reward_component = np.zeros(self.num_envs, dtype=np.float32)
        self._death_penalty_component = np.zeros(self.num_envs, dtype=np.float32)
        self._time_penalty_component = np.zeros(self.num_envs, dtype=np.float32)

        self._x = np.zeros(self.num_envs, dtype=np.int64)
        self._max_x = np.zeros(self.num_envs, dtype=np.int64)
        self._level_max_x = np.zeros(self.num_envs, dtype=np.int64)
        self._completed_level_base = np.zeros(self.num_envs, dtype=np.int64)
        self._completed_level_count = np.zeros(self.num_envs, dtype=np.int64)
        self._global_x = np.zeros(self.num_envs, dtype=np.int64)
        self._max_global_x = np.zeros(self.num_envs, dtype=np.int64)
        self._score = np.zeros(self.num_envs, dtype=np.int64)
        self._lives = np.zeros(self.num_envs, dtype=np.int64)
        self._level_hi = np.zeros(self.num_envs, dtype=np.int64)
        self._level_lo = np.zeros(self.num_envs, dtype=np.int64)
        self._episode_steps = np.zeros(self.num_envs, dtype=np.int64)
        self._last_progress_step = np.zeros(self.num_envs, dtype=np.int64)

        self._progress_delta = np.zeros(self.num_envs, dtype=np.int64)
        self._score_delta = np.zeros(self.num_envs, dtype=np.int64)
        self._life_lost = np.zeros(self.num_envs, dtype=bool)
        self._level_changed = np.zeros(self.num_envs, dtype=bool)
        self._completion = np.zeros(self.num_envs, dtype=bool)
        self._game_complete = np.zeros(self.num_envs, dtype=bool)
        self._stalled = np.zeros(self.num_envs, dtype=bool)
        self._life_transition_source = np.zeros(self.num_envs, dtype=np.int64)
        self._life_transition_target = np.zeros(self.num_envs, dtype=np.int64)
        self._level_transition_source = np.zeros((self.num_envs, 2), dtype=np.int64)
        self._level_transition_target = np.zeros((self.num_envs, 2), dtype=np.int64)
        self._zero_signal = np.zeros(self.num_envs, dtype=np.int64)
        self._provider_false = np.zeros(self.num_envs, dtype=bool)
        self._zero_native_reward = np.zeros(self.num_envs, dtype=np.float32)
        self._reward_mode_code = {
            "native": 0,
            "bounded": 1,
            "baseline": 2,
            "score": 3,
            "additive": 4,
        }.get(config.reward_mode, 4)
        self._compiled_state = (
            self._x,
            self._max_x,
            self._level_max_x,
            self._completed_level_base,
            self._completed_level_count,
            self._global_x,
            self._max_global_x,
            self._score,
            self._lives,
            self._level_hi,
            self._level_lo,
            self._episode_steps,
            self._last_progress_step,
        )
        self._compiled_outputs = (
            self._rewards,
            self._terminated,
            self._truncated,
            self._outcomes,
            self._events,
            self._progress_delta,
            self._score_delta,
            self._life_lost,
            self._level_changed,
            self._completion,
            self._game_complete,
            self._stalled,
            self._raw_reward,
            self._progress_component,
            self._native_reward_component,
            self._progress_reward_component,
            self._score_reward_component,
            self._completion_reward_component,
            self._death_penalty_component,
            self._time_penalty_component,
            self._life_transition_source,
            self._life_transition_target,
            self._level_transition_source,
            self._level_transition_target,
        )
        self._compiled_parameters = (
            self._reward_mode_code,
            config.use_native_reward,
            config.clip_rewards,
            config.progress_reward_cap,
            config.progress_reward_scale,
            config.terminal_reward,
            config.reward_scale,
            config.time_penalty,
            config.death_penalty,
            config.completion_reward,
            config.score_progress_clipped,
            config.max_episode_steps,
            config.no_progress_timeout_steps,
            config.no_progress_min_delta,
            config.game_complete_level[0],
            config.game_complete_level[1],
            config.game_complete_mode,
            config.terminate_on_life_loss,
            config.terminate_on_level_change,
            config.terminate_on_game_complete,
            config.stall_is_failure,
            config.emit_life_loss,
            config.emit_level_change,
            config.emit_game_complete,
            config.emit_stalled,
        )
        # Compile once at bind time. The first real reset initializes all mutated state.
        self._run_compiled(
            self._zero_signal,
            self._zero_signal,
            self._zero_signal,
            self._zero_signal,
            self._zero_signal,
            self._zero_signal,
            self._zero_signal,
            self._provider_false,
            self._provider_false,
        )
        self.has_events = False
        self._metrics = {
            "x_pos": self._global_x,
            "max_x_pos": self._max_global_x,
            "level_x_pos": self._x,
            "level_max_x_pos": self._level_max_x,
            "completed_level_base": self._completed_level_base,
            "completed_level_count": self._completed_level_count,
            "global_x_pos": self._global_x,
            "global_max_x_pos": self._max_global_x,
            "progress_delta": self._progress_delta,
            "score_delta": self._score_delta,
            "score": self._score,
            "lives": self._lives,
            "level_hi": self._level_hi,
            "level_lo": self._level_lo,
            "level_changed": self._level_changed,
            "completion_event": self._completion,
            "level_complete": self._completion,
            "game_complete": self._game_complete,
            "died": self._life_lost,
            "life_loss": self._life_lost,
            "death_x_pos": self._max_global_x,
            "death_level_x_pos": self._level_max_x,
            "stalled": self._stalled,
            "raw_reward": self._raw_reward,
            "progress_component": self._progress_component,
            "native_reward_component": self._native_reward_component,
            "progress_reward_component": self._progress_reward_component,
            "score_reward_component": self._score_reward_component,
            "completion_reward_component": self._completion_reward_component,
            "death_penalty_component": self._death_penalty_component,
            "time_penalty_component": self._time_penalty_component,
            "shaped_reward": self._rewards,
        }
        self._event_transitions = {
            "life_loss": (
                self._life_transition_source,
                self._life_transition_target,
            ),
            "level_change": (
                self._level_transition_source,
                self._level_transition_target,
            ),
            "game_complete": (
                self._level_transition_source,
                self._level_transition_target,
            ),
        }
        self._task_step = TaskStep(
            self._rewards,
            self._terminated,
            self._truncated,
            self._outcomes,
            self._events,
            self._metrics,
            self._event_transitions,
        )

    def _run_compiled(
        self,
        x_hi: np.ndarray,
        x_lo: np.ndarray,
        score: np.ndarray,
        lives: np.ndarray,
        level_hi: np.ndarray,
        level_lo: np.ndarray,
        game_mode: np.ndarray,
        provider_terminated: np.ndarray,
        provider_truncated: np.ndarray,
        native_rewards: np.ndarray | None = None,
    ) -> bool:
        if native_rewards is None:
            native_rewards = self._zero_native_reward
        return _mario_step_kernel_packed(
            (
                x_hi,
                x_lo,
                len(self._x_signal_names) == 2,
                score,
                lives,
                level_hi,
                level_lo,
                game_mode,
                native_rewards,
                provider_terminated,
                provider_truncated,
            ),
            self._compiled_state,
            self._compiled_outputs,
            self._compiled_parameters,
        )

    def map_actions(self, actions: Any) -> Any:
        if self._action_masks is None:
            return actions
        indices = actions
        if not isinstance(indices, np.ndarray) or indices.shape != (self.num_envs,):
            indices = np.asarray(actions, dtype=np.int64).reshape(-1)
            if indices.shape != (self.num_envs,):
                raise ValueError(f"expected {self.num_envs} Mario action ids, got {indices.shape}")
        self._action_buffer[...] = self._action_masks[indices]
        return self._action_buffer

    def encode_observations(self, observations: Any) -> Any:
        image = _encode_policy_observation(observations, self._native_observation_space)
        if self._task_observation is not None:
            return {"image": image, "task": self._task_observation}
        return image

    def _update_task_observation(
        self,
        level_hi: np.ndarray,
        level_lo: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> None:
        if self._task_observation is None or self._task_values is None:
            return
        active = np.ones(self.num_envs, dtype=bool) if mask is None else mask
        pairs = np.stack((level_hi, level_lo), axis=1)
        matches = np.all(pairs[:, None, :] == self._task_values[None, :, :], axis=2)
        matched = active & np.any(matches, axis=1)
        if np.any(active & ~matched):
            raise ValueError("Mario level signal does not match a configured task value")
        indices = np.argmax(matches, axis=1)
        self._task_observation[active] = 0.0
        self._task_observation[np.flatnonzero(active), indices[active]] = 1.0

    def _x_values(
        self,
        signals: Mapping[str, Any],
        *,
        mask: np.ndarray | None = None,
        validate: bool = True,
    ) -> np.ndarray:
        columns = (
            self.bindings.columns("x", signals, mask=mask)
            if validate
            else self.bindings.unchecked_columns("x", signals)
        )
        if len(columns) == 1:
            return np.asarray(columns[0], dtype=np.int64)
        if len(columns) == 2:
            return np.asarray(columns[0], dtype=np.int64) * 256 + np.asarray(
                columns[1], dtype=np.int64
            )
        raise ValueError("Mario x signal must bind one value or a high/low pair")

    def _level_values(
        self,
        signals: Mapping[str, Any],
        *,
        mask: np.ndarray | None = None,
        validate: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        columns = (
            self.bindings.columns("level", signals, mask=mask)
            if validate
            else self.bindings.unchecked_columns("level", signals)
        )
        if len(columns) == 1:
            values = np.asarray(columns[0])
            if values.ndim != 2 or values.shape[1] != 2:
                raise ValueError("Mario level signal must contain two values per lane")
            return values[:, 0].astype(np.int64), values[:, 1].astype(np.int64)
        if len(columns) == 2:
            return (
                np.asarray(columns[0], dtype=np.int64),
                np.asarray(columns[1], dtype=np.int64),
            )
        raise ValueError("Mario level signal must bind a pair of values")

    def process(
        self,
        native_rewards: np.ndarray,
        provider_terminated: np.ndarray,
        provider_truncated: np.ndarray,
        signals: Mapping[str, Any],
    ) -> TaskStep:
        if len(self._x_signal_names) == 2:
            x_hi = signals[self._x_signal_names[0]]
            x_lo = signals[self._x_signal_names[1]]
        else:
            x_hi = signals[self._x_signal_names[0]]
            x_lo = self._zero_signal
        score = signals[self._score_signal_name]
        lives = signals[self._lives_signal_name]
        if len(self._level_signal_names) == 2:
            level_hi = signals[self._level_signal_names[0]]
            level_lo = signals[self._level_signal_names[1]]
        else:
            level_values = signals[self._level_signal_names[0]]
            level_hi, level_lo = level_values[:, 0], level_values[:, 1]
        game_mode = (
            signals[self._game_mode_signal_name]
            if self._game_mode_signal_name is not None
            else self._zero_signal
        )
        self._update_task_observation(level_hi, level_lo)
        self.has_events = self._run_compiled(
            x_hi,
            x_lo,
            score,
            lives,
            level_hi,
            level_lo,
            game_mode,
            provider_terminated,
            provider_truncated,
            np.asarray(native_rewards, dtype=np.float32),
        )

        return self._task_step

    def on_reset(
        self,
        reset_observations: Any,
        reset_signals: Mapping[str, Any],
        mask: np.ndarray,
    ) -> None:
        del reset_observations
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (self.num_envs,):
            raise ValueError(f"reset mask must have shape ({self.num_envs},)")
        x = self._x_values(reset_signals, mask=mask)
        score = self.bindings.scalar("score", reset_signals, mask=mask).astype(np.int64)
        lives = self.bindings.scalar("lives", reset_signals, mask=mask).astype(np.int64)
        level_hi, level_lo = self._level_values(reset_signals, mask=mask)
        self._update_task_observation(level_hi, level_lo, mask)
        self._x[mask] = x[mask]
        self._max_x[mask] = x[mask]
        self._level_max_x[mask] = x[mask]
        self._completed_level_base[mask] = 0
        self._completed_level_count[mask] = 0
        self._global_x[mask] = x[mask]
        self._max_global_x[mask] = x[mask]
        self._score[mask] = score[mask]
        self._lives[mask] = lives[mask]
        self._level_hi[mask] = level_hi[mask]
        self._level_lo[mask] = level_lo[mask]
        self._episode_steps[mask] = 0
        self._last_progress_step[mask] = 0
        self._life_lost[mask] = False
        self._level_changed[mask] = False
        self._completion[mask] = False
        self._game_complete[mask] = False
        self._stalled[mask] = False


def event_names_from_bits(bits: int, event_names: Sequence[str]) -> tuple[str, ...]:
    return tuple(name for index, name in enumerate(event_names) if bits & (1 << index))
