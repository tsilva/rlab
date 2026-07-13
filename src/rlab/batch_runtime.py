from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import gymnasium as gym
import numpy as np
from numba import njit
from stable_baselines3.common.vec_env import VecEnv

from rlab.task_kernels import BoundTaskKernel, Outcome, event_names_from_bits


@njit(cache=True, nogil=True)
def _combine_step_outputs(
    provider_terminated,
    provider_truncated,
    task_terminated,
    task_truncated,
    task_rewards,
    terminated,
    truncated,
    dones,
    rewards,
    episode_returns,
    episode_lengths,
):
    any_done = False
    for lane in range(terminated.shape[0]):
        lane_terminated = provider_terminated[lane] or task_terminated[lane]
        lane_truncated = (provider_truncated[lane] or task_truncated[lane]) and not lane_terminated
        lane_done = lane_terminated or lane_truncated
        terminated[lane] = lane_terminated
        truncated[lane] = lane_truncated
        dones[lane] = lane_done
        rewards[lane] = task_rewards[lane]
        episode_returns[lane] += rewards[lane]
        episode_lengths[lane] += 1
        any_done = any_done or lane_done
    return any_done


@dataclass(frozen=True)
class SignalSpec:
    name: str
    dtype: np.dtype | str | type = np.float32
    shape: tuple[int, ...] = ()
    available_on_reset: bool = True
    available_on_step: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("signal name must not be empty")
        object.__setattr__(self, "dtype", np.dtype(self.dtype))
        object.__setattr__(self, "shape", tuple(int(value) for value in self.shape))


@dataclass(frozen=True)
class ProviderDescriptor:
    provider_id: str
    native_observation_space: gym.Space
    native_action_space: gym.Space
    signal_schema: Mapping[str, SignalSpec] = field(default_factory=dict)
    start_catalog: tuple[str, ...] = ()
    start_probabilities: tuple[float, ...] = ()
    lane_start_ids: tuple[str, ...] = ()
    render_support: tuple[str, ...] = ()
    autoreset_mode: str = "disabled"
    observation_buffer_depth: int = 1

    def __post_init__(self) -> None:
        if not self.provider_id:
            raise ValueError("provider_id must not be empty")
        if self.autoreset_mode != "disabled":
            raise ValueError("the batch runtime requires disabled provider autoreset")
        if int(self.observation_buffer_depth) < 1:
            raise ValueError("provider observation_buffer_depth must be positive")
        object.__setattr__(self, "observation_buffer_depth", int(self.observation_buffer_depth))
        normalized: dict[str, SignalSpec] = {}
        for key, spec in self.signal_schema.items():
            if key != spec.name:
                raise ValueError(f"signal schema key {key!r} does not match {spec.name!r}")
            normalized[key] = spec
        object.__setattr__(self, "signal_schema", normalized)
        object.__setattr__(self, "start_catalog", tuple(self.start_catalog))
        probabilities = tuple(float(value) for value in self.start_probabilities)
        if probabilities:
            if len(probabilities) != len(self.start_catalog):
                raise ValueError("start probabilities must match the start catalog")
            if any(not np.isfinite(value) or value < 0.0 for value in probabilities):
                raise ValueError("start probabilities must be finite and non-negative")
            total = float(sum(probabilities))
            if total <= 0.0:
                raise ValueError("start probabilities must have a positive sum")
            probabilities = tuple(value / total for value in probabilities)
        lane_start_ids = tuple(self.lane_start_ids)
        unknown = sorted(set(lane_start_ids) - set(self.start_catalog))
        if unknown:
            raise ValueError(f"lane starts are absent from the start catalog: {unknown}")
        object.__setattr__(self, "start_probabilities", probabilities)
        object.__setattr__(self, "lane_start_ids", lane_start_ids)
        object.__setattr__(self, "render_support", tuple(self.render_support))


class NativeVectorEnv(Protocol):
    num_envs: int
    single_observation_space: gym.Space
    single_action_space: gym.Space

    def reset(
        self,
        *,
        seed: int | Sequence[int | None] | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[Any, Mapping[str, Any]]: ...

    def step(
        self, actions: Any
    ) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, Mapping[str, Any]]: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class EpisodeRecord:
    lane: int
    episode_index: int
    start_id: str | None
    episode_return: float
    episode_length: int
    terminated: bool
    truncated: bool
    outcome: Outcome
    events: tuple[str, ...]
    metrics: Mapping[str, Any]


@dataclass(frozen=True)
class TaskEventRecord:
    lane: int
    episode_index: int
    start_id: str | None
    events: tuple[str, ...]
    transitions: Mapping[str, tuple[Any, Any]]
    metrics: Mapping[str, Any]


@dataclass(frozen=True)
class BatchMetricRecord:
    num_envs: int
    metrics: Mapping[str, np.ndarray]


@dataclass
class BatchStep:
    observations: Any
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    dones: np.ndarray
    infos: list[dict[str, Any]]


def _copy_tree(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, Mapping):
        return type(value)((key, _copy_tree(item)) for key, item in value.items())
    if isinstance(value, tuple):
        return tuple(_copy_tree(item) for item in value)
    if isinstance(value, list):
        return [_copy_tree(item) for item in value]
    return value


def _empty_tree_like(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return np.empty_like(value)
    if isinstance(value, Mapping):
        return type(value)((key, _empty_tree_like(item)) for key, item in value.items())
    if isinstance(value, tuple):
        return tuple(_empty_tree_like(item) for item in value)
    if isinstance(value, list):
        return [_empty_tree_like(item) for item in value]
    raise TypeError(f"unsupported batched observation leaf {type(value).__name__}")


def _copy_tree_into(destination: Any, source: Any) -> None:
    if isinstance(destination, np.ndarray) and isinstance(source, np.ndarray):
        if destination.shape != source.shape:
            raise ValueError(
                f"observation shape changed from {destination.shape} to {source.shape}"
            )
        np.copyto(destination, source)
        return
    if isinstance(destination, Mapping) and isinstance(source, Mapping):
        if destination.keys() != source.keys():
            raise ValueError("observation mapping keys changed during the run")
        for key in destination:
            _copy_tree_into(destination[key], source[key])
        return
    if isinstance(destination, tuple) and isinstance(source, tuple):
        if len(destination) != len(source):
            raise ValueError("observation tuple length changed during the run")
        for target, item in zip(destination, source, strict=True):
            _copy_tree_into(target, item)
        return
    if isinstance(destination, list) and isinstance(source, list):
        if len(destination) != len(source):
            raise ValueError("observation list length changed during the run")
        for target, item in zip(destination, source, strict=True):
            _copy_tree_into(target, item)
        return
    raise TypeError("observation structure changed during the run")


def _copy_tree_lanes(destination: Any, source: Any, mask: np.ndarray) -> None:
    if isinstance(destination, np.ndarray) and isinstance(source, np.ndarray):
        destination[mask] = source[mask]
        return
    if isinstance(destination, Mapping) and isinstance(source, Mapping):
        if destination.keys() != source.keys():
            raise ValueError("reset observation mapping keys differ from step observations")
        for key in destination:
            _copy_tree_lanes(destination[key], source[key], mask)
        return
    if isinstance(destination, tuple) and isinstance(source, tuple):
        if len(destination) != len(source):
            raise ValueError("reset observation tuple differs from step observations")
        for target, item in zip(destination, source, strict=True):
            _copy_tree_lanes(target, item, mask)
        return
    if isinstance(destination, list) and isinstance(source, list):
        if len(destination) != len(source):
            raise ValueError("reset observation list differs from step observations")
        for target, item in zip(destination, source, strict=True):
            _copy_tree_lanes(target, item, mask)
        return
    raise TypeError("reset observation structure differs from step observations")


def _copy_tree_lane(value: Any, lane: int) -> Any:
    if isinstance(value, np.ndarray):
        return value[lane].copy()
    if isinstance(value, Mapping):
        return type(value)((key, _copy_tree_lane(item, lane)) for key, item in value.items())
    if isinstance(value, tuple):
        return tuple(_copy_tree_lane(item, lane) for item in value)
    if isinstance(value, list):
        return [_copy_tree_lane(item, lane) for item in value]
    raise TypeError(f"unsupported batched observation leaf {type(value).__name__}")


class _InfoColumns:
    def __init__(self, infos: Any, num_envs: int):
        self.infos = infos
        self.num_envs = num_envs

    def lane(self, lane: int) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in self.infos.items():
            if not isinstance(key, str) or key.startswith("_"):
                continue
            presence = self.infos.get(f"_{key}")
            if presence is not None and not bool(np.asarray(presence, dtype=bool)[lane]):
                continue
            result[key] = self._lane_value(value, lane)
        return result

    def _lane_value(self, value: Any, lane: int) -> Any:
        if isinstance(value, Mapping):
            return _InfoColumns(value, self.num_envs).lane(lane)
        if isinstance(value, np.ndarray):
            item = value[lane]
            if isinstance(item, np.generic):
                return item.item()
            return item.copy() if isinstance(item, np.ndarray) else item
        if isinstance(value, (list, tuple)) and len(value) == self.num_envs:
            return _copy_tree(value[lane])
        return _copy_tree(value)


class BatchRuntime:
    """Own the manual-reset lifecycle between a native provider and task kernel."""

    def __init__(
        self,
        provider: NativeVectorEnv,
        descriptor: ProviderDescriptor,
        kernel: BoundTaskKernel,
        *,
        run_seed: int = 0,
    ):
        self.provider = provider
        self.descriptor = descriptor
        self.kernel = kernel
        self.num_envs = int(provider.num_envs)
        if self.num_envs <= 0:
            raise ValueError("provider num_envs must be positive")
        if int(kernel.num_envs) != self.num_envs:
            raise ValueError("provider and task kernel num_envs differ")
        provider_obs_space = getattr(
            provider, "single_observation_space", descriptor.native_observation_space
        )
        provider_action_space = getattr(
            provider, "single_action_space", descriptor.native_action_space
        )
        if provider_obs_space != descriptor.native_observation_space:
            raise ValueError("provider observation space differs from its descriptor")
        if provider_action_space != descriptor.native_action_space:
            raise ValueError("provider action space differs from its descriptor")

        self.observation_space = kernel.observation_space
        self.action_space = kernel.action_space
        self.run_seed = int(run_seed)
        self.reset_infos: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]
        self._episode_returns = np.zeros(self.num_envs, dtype=np.float64)
        self._episode_lengths = np.zeros(self.num_envs, dtype=np.int64)
        self._episode_indices = np.zeros(self.num_envs, dtype=np.int64)
        self._start_ids: list[str | None] = [None for _ in range(self.num_envs)]
        self._records: list[EpisodeRecord | TaskEventRecord] = []
        self._latest_metric_record: BatchMetricRecord | None = None
        self._empty_sb3_infos: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]
        self._combined_terminated = [np.zeros(self.num_envs, dtype=bool) for _ in range(2)]
        self._combined_truncated = [np.zeros(self.num_envs, dtype=bool) for _ in range(2)]
        self._combined_dones = [np.zeros(self.num_envs, dtype=bool) for _ in range(2)]
        self._owned_rewards = [np.zeros(self.num_envs, dtype=np.float32) for _ in range(2)]
        self._empty_indices = np.empty(0, dtype=np.int64)
        self._batch_steps = [
            BatchStep(
                None,
                self._owned_rewards[index],
                self._combined_terminated[index],
                self._combined_truncated[index],
                self._combined_dones[index],
                self._empty_sb3_infos,
            )
            for index in range(2)
        ]
        self._metric_record: BatchMetricRecord | None = None
        _combine_step_outputs(
            self._combined_terminated[0],
            self._combined_truncated[0],
            self._combined_terminated[0],
            self._combined_truncated[0],
            self._owned_rewards[0],
            self._combined_terminated[1],
            self._combined_truncated[1],
            self._combined_dones[1],
            self._owned_rewards[1],
            self._episode_returns,
            self._episode_lengths,
        )
        self._observation_buffers: list[Any] = []
        self._current_observation_buffer = 0
        self._reuse_provider_observations = (
            descriptor.observation_buffer_depth >= 2
            and bool(getattr(kernel, "observation_encoding_is_view", False))
        )
        self._started_at = time.monotonic()
        self._native_step_seconds_total = 0.0
        self._native_step_calls_total = 0
        self._closed = False

    def _seed_for(self, lane: int, episode_index: int) -> int:
        sequence = np.random.SeedSequence([self.run_seed, lane, episode_index])
        return int(sequence.generate_state(1, dtype=np.uint32)[0])

    def _start_for(self, lane: int, episode_index: int) -> str | None:
        catalog = self.descriptor.start_catalog
        if not catalog:
            return None
        if self.descriptor.lane_start_ids:
            if len(self.descriptor.lane_start_ids) != self.num_envs:
                raise ValueError("lane_start_ids must contain one start for every lane")
            return self.descriptor.lane_start_ids[lane]
        probabilities = self.descriptor.start_probabilities
        if not probabilities:
            probabilities = tuple(1.0 / len(catalog) for _ in catalog)
        sequence = np.random.SeedSequence([self.run_seed, lane, episode_index, 0x53544152])
        generator = np.random.default_rng(sequence)
        return catalog[int(generator.choice(len(catalog), p=probabilities))]

    def _normalize_seeds(self, seed: int | Sequence[int | None] | None) -> list[int | None]:
        if seed is None:
            return [self._seed_for(lane, 0) for lane in range(self.num_envs)]
        if isinstance(seed, int):
            return [seed + lane for lane in range(self.num_envs)]
        seeds = list(seed)
        if len(seeds) != self.num_envs:
            raise ValueError(f"expected {self.num_envs} reset seeds, got {len(seeds)}")
        if all(value is None for value in seeds):
            return [self._seed_for(lane, 0) for lane in range(self.num_envs)]
        return [None if value is None else int(value) for value in seeds]

    def reset(
        self,
        *,
        seed: int | Sequence[int | None] | None = None,
        options_by_lane: Sequence[Mapping[str, Any]] | None = None,
    ) -> Any:
        self._episode_returns.fill(0.0)
        self._episode_lengths.fill(0)
        self._episode_indices.fill(0)
        mask = np.ones(self.num_envs, dtype=bool)
        starts = [self._start_for(lane, 0) for lane in range(self.num_envs)]
        options = self._reset_options(mask, starts, options_by_lane)
        observations, infos = self.provider.reset(seed=self._normalize_seeds(seed), options=options)
        if not isinstance(infos, Mapping):
            raise TypeError("native provider reset infos must be a columnar mapping")
        self.kernel.on_reset(observations, infos, mask)
        encoded = self.kernel.encode_observations(observations)
        if self._reuse_provider_observations:
            self._observation_buffers = [
                _empty_tree_like(encoded),
                _empty_tree_like(encoded),
            ]
            initial_observations = encoded
        else:
            self._observation_buffers = [_copy_tree(encoded), _empty_tree_like(encoded)]
            initial_observations = self._observation_buffers[0]
        self._current_observation_buffer = 0
        info_columns = _InfoColumns(infos, self.num_envs)
        self.reset_infos = [info_columns.lane(lane) for lane in range(self.num_envs)]
        self._start_ids = self._actual_start_ids(infos, starts, mask)
        return initial_observations

    def _reset_options(
        self,
        mask: np.ndarray,
        starts: Sequence[str | None],
        options_by_lane: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if options_by_lane is None:
            lane_options: tuple[dict[str, Any], ...] = tuple({} for _ in range(self.num_envs))
        else:
            if len(options_by_lane) != self.num_envs:
                raise ValueError(
                    f"expected {self.num_envs} lane reset options, got {len(options_by_lane)}"
                )
            lane_options = tuple(dict(value) for value in options_by_lane)
        if any(lane_options):
            raise ValueError(
                "native providers do not support arbitrary per-lane reset options; "
                "use configured start indices"
            )
        options: dict[str, Any] = {"reset_mask": mask.copy()}
        if self.descriptor.start_catalog:
            options["start_ids"] = np.asarray(starts, dtype=object)
        return options

    def _actual_start_ids(
        self,
        reset_infos: Mapping[str, Any],
        requested: Sequence[str | None],
        mask: np.ndarray,
    ) -> list[str | None]:
        values = reset_infos.get("start_id")
        if values is None:
            values = reset_infos.get("start_state")
        if values is None:
            values = reset_infos.get("state")
        if values is None:
            if self.descriptor.start_catalog and any(
                bool(mask[lane]) and requested[lane] is not None
                for lane in range(self.num_envs)
            ):
                raise ValueError(
                    "provider reset infos must report actual start_id, start_state, or state"
                )
            return list(requested)
        values = np.asarray(values, dtype=object)
        if values.shape != (self.num_envs,):
            raise ValueError("reset start_id must contain one value per lane")
        presence = None
        for key in ("start_id", "start_state", "state"):
            if key in reset_infos:
                presence = reset_infos.get(f"_{key}")
                break
        if presence is not None:
            present = np.asarray(presence, dtype=bool)
            if present.shape != (self.num_envs,) or np.any(mask & ~present):
                raise ValueError("provider reset infos omit the actual start for a reset lane")
        return [None if value is None else str(value) for value in values]

    @staticmethod
    def _bool_batch(value: Any, name: str, num_envs: int) -> np.ndarray:
        result = np.asarray(value, dtype=bool)
        if result.shape != (num_envs,):
            raise ValueError(f"{name} must have shape ({num_envs},), got {result.shape}")
        return result

    def step(self, actions: Any) -> BatchStep:
        if not self._observation_buffers:
            raise RuntimeError("BatchRuntime.reset() must be called before step()")
        native_actions = self.kernel.map_actions(actions)
        started_at = time.perf_counter()
        observations, native_rewards, provider_terminated, provider_truncated, infos = (
            self.provider.step(native_actions)
        )
        self._native_step_seconds_total += time.perf_counter() - started_at
        self._native_step_calls_total += 1
        if not isinstance(infos, Mapping):
            raise TypeError("native provider step infos must be a columnar mapping")
        native_rewards = np.asarray(native_rewards)
        if native_rewards.shape != (self.num_envs,):
            raise ValueError(
                f"provider rewards must have shape ({self.num_envs},), "
                f"got {native_rewards.shape}"
            )
        provider_terminated = self._bool_batch(
            provider_terminated,
            "provider terminated",
            self.num_envs,
        )
        provider_truncated = self._bool_batch(
            provider_truncated,
            "provider truncated",
            self.num_envs,
        )
        task_step = self.kernel.process(
            native_rewards, provider_terminated, provider_truncated, infos
        )
        if task_step.metrics:
            if self._metric_record is None:
                self._metric_record = BatchMetricRecord(
                    num_envs=self.num_envs,
                    metrics=task_step.metrics,
                )
            self._latest_metric_record = self._metric_record
        else:
            self._latest_metric_record = None
        task_terminated = task_step.terminated
        task_truncated = task_step.truncated
        next_buffer_index = 1 - self._current_observation_buffer
        terminated = self._combined_terminated[next_buffer_index]
        truncated = self._combined_truncated[next_buffer_index]
        dones = self._combined_dones[next_buffer_index]
        rewards = self._owned_rewards[next_buffer_index]
        any_done = _combine_step_outputs(
            provider_terminated,
            provider_truncated,
            task_terminated,
            task_truncated,
            task_step.rewards,
            terminated,
            truncated,
            dones,
            rewards,
            self._episode_returns,
            self._episode_lengths,
        )
        done_indices = np.flatnonzero(dones) if any_done else self._empty_indices

        encoded_transition = self.kernel.encode_observations(observations)
        if self._reuse_provider_observations and not any_done:
            next_observations = encoded_transition
        else:
            next_observations = self._observation_buffers[next_buffer_index]
            _copy_tree_into(next_observations, encoded_transition)

        if getattr(self.kernel, "has_events", True):
            self._append_event_records(task_step)
        sb3_infos = self._empty_sb3_infos
        if done_indices.size:
            sb3_infos = [{} for _ in range(self.num_envs)]
            terminal_info_columns = _InfoColumns(infos, self.num_envs)
            for lane in done_indices:
                lane_index = int(lane)
                terminal_observation = _copy_tree_lane(next_observations, lane_index)
                info = terminal_info_columns.lane(lane_index)
                info["terminal_observation"] = terminal_observation
                info["TimeLimit.truncated"] = bool(truncated[lane_index])
                sb3_infos[lane_index] = info
                self._append_record(
                    lane_index,
                    bool(terminated[lane_index]),
                    bool(truncated[lane_index]),
                    task_step,
                )

        if sb3_infos is not self._empty_sb3_infos:
            self._episode_indices[dones] += 1
            starts = [
                self._start_for(lane, int(self._episode_indices[lane]))
                for lane in range(self.num_envs)
            ]
            seeds: list[int | None] = [
                self._seed_for(lane, int(self._episode_indices[lane]))
                if bool(dones[lane])
                else None
                for lane in range(self.num_envs)
            ]
            reset_observations, reset_infos = self.provider.reset(
                seed=seeds,
                options=self._reset_options(dones, starts),
            )
            if not isinstance(reset_infos, Mapping):
                raise TypeError("native provider reset infos must be a columnar mapping")
            self.kernel.on_reset(reset_observations, reset_infos, dones)
            encoded_reset = self.kernel.encode_observations(reset_observations)
            _copy_tree_lanes(next_observations, encoded_reset, dones)
            reset_info_columns = _InfoColumns(reset_infos, self.num_envs)
            actual_starts = self._actual_start_ids(reset_infos, starts, dones)
            for lane in done_indices:
                lane_index = int(lane)
                reset_info = reset_info_columns.lane(lane_index)
                self.reset_infos[lane_index] = reset_info
                self._start_ids[lane_index] = actual_starts[lane_index]
                sb3_infos[lane_index]["reset_info"] = reset_info
                sb3_infos[lane_index]["episode"] = {
                    "r": float(self._episode_returns[lane_index]),
                    "l": int(self._episode_lengths[lane_index]),
                    "t": round(time.monotonic() - self._started_at, 6),
                }
                self._episode_returns[lane_index] = 0.0
                self._episode_lengths[lane_index] = 0

        self._current_observation_buffer = next_buffer_index
        batch_step = self._batch_steps[next_buffer_index]
        batch_step.observations = next_observations
        batch_step.infos = sb3_infos
        return batch_step

    def _append_record(
        self,
        lane: int,
        terminated: bool,
        truncated: bool,
        task_step: Any,
    ) -> None:
        outcome = Outcome(int(np.asarray(task_step.outcomes)[lane]))
        if outcome == Outcome.NEUTRAL and truncated:
            outcome = Outcome.TIMEOUT
        event_bits = int(np.asarray(task_step.event_bits, dtype=np.uint64)[lane])
        metrics = self._lane_metrics(task_step.metrics, lane)
        self._records.append(
            EpisodeRecord(
                lane=lane,
                episode_index=int(self._episode_indices[lane]),
                start_id=self._start_ids[lane],
                episode_return=float(self._episode_returns[lane]),
                episode_length=int(self._episode_lengths[lane]),
                terminated=terminated,
                truncated=truncated,
                outcome=outcome,
                events=event_names_from_bits(event_bits, self.kernel.event_names),
                metrics=metrics,
            )
        )

    @staticmethod
    def _lane_metrics(metrics: Mapping[str, Any], lane: int) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, values in metrics.items():
            value = np.asarray(values)[lane]
            if isinstance(value, np.generic):
                value = value.item()
            elif isinstance(value, np.ndarray):
                value = value.copy()
            result[name] = value
        return result

    def _append_event_records(self, task_step: Any) -> None:
        bits = np.asarray(task_step.event_bits, dtype=np.uint64)
        if not np.bitwise_or.reduce(bits):
            return
        for lane in np.flatnonzero(bits):
            lane_index = int(lane)
            event_names = event_names_from_bits(int(bits[lane_index]), self.kernel.event_names)
            transitions: dict[str, tuple[Any, Any]] = {}
            for name in event_names:
                transition = task_step.event_transitions.get(name)
                if transition is None:
                    continue
                source, target = transition
                source_value = _copy_tree_lane(np.asarray(source), lane_index)
                target_value = _copy_tree_lane(np.asarray(target), lane_index)
                transitions[name] = (source_value, target_value)
            self._records.append(
                TaskEventRecord(
                    lane=lane_index,
                    episode_index=int(self._episode_indices[lane_index]),
                    start_id=self._start_ids[lane_index],
                    events=event_names,
                    transitions=transitions,
                    metrics=self._lane_metrics(task_step.metrics, lane_index),
                )
            )

    def drain_records(
        self,
    ) -> list[BatchMetricRecord | EpisodeRecord | TaskEventRecord]:
        records = self._records
        self._records = []
        if self._latest_metric_record is not None:
            records.insert(0, self._latest_metric_record)
            self._latest_metric_record = None
        return records

    def native_step_stats(self) -> dict[str, float | int]:
        return {
            "seconds_total": self._native_step_seconds_total,
            "calls_total": self._native_step_calls_total,
            "num_envs": self.num_envs,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.provider.close()


class RlabVecEnv(VecEnv):
    """SB3 facade that exposes runtime-owned same-step reset observations."""

    def __init__(self, runtime: BatchRuntime):
        self.runtime = runtime
        self.env = runtime.provider
        self.waiting = False
        self._actions: Any = None
        super().__init__(
            runtime.num_envs,
            runtime.observation_space,
            runtime.action_space,
        )

    def __getattr__(self, name: str) -> Any:
        if name in {"runtime", "env"}:
            raise AttributeError(name)
        return getattr(self.env, name)

    def reset(self) -> Any:
        observations = self.runtime.reset(
            seed=list(self._seeds),
            options_by_lane=list(self._options),
        )
        self.reset_infos = self.runtime.reset_infos
        self._reset_seeds()
        self._reset_options()
        self.waiting = False
        self._actions = None
        return observations

    def step_async(self, actions: Any) -> None:
        if self.waiting:
            raise RuntimeError("step_async() called while another step is pending")
        self._actions = actions
        self.waiting = True

    def step_wait(self):
        if not self.waiting:
            raise RuntimeError("step_wait() called without step_async()")
        step = self.runtime.step(self._actions)
        self._actions = None
        self.waiting = False
        self.reset_infos = self.runtime.reset_infos
        return step.observations, step.rewards, step.dones, step.infos

    def drain_records(
        self,
    ) -> list[BatchMetricRecord | EpisodeRecord | TaskEventRecord]:
        return self.runtime.drain_records()

    def native_step_stats(self) -> dict[str, float | int]:
        return self.runtime.native_step_stats()

    def close(self) -> None:
        self.runtime.close()

    def get_images(self):
        if hasattr(self.env, "get_images"):
            images = self.env.get_images()
        else:
            images = self.env.render()
        if images is None or (isinstance(images, Sequence) and len(images) == 0):
            images = self.env.render()
        if images is None:
            return [None for _ in range(self.num_envs)]
        if isinstance(images, np.ndarray) and images.shape[0] == self.num_envs:
            return [images[index] for index in range(self.num_envs)]
        if isinstance(images, Sequence) and len(images) == self.num_envs:
            return list(images)
        if self.num_envs == 1:
            return [images]
        raise ValueError("provider render output must contain one frame per lane")

    def render(self, mode: str | None = None):
        if mode is None:
            return self.env.render()
        return self.env.render()

    def get_attr(self, attr_name: str, indices=None) -> list[Any]:
        value = getattr(self.env, attr_name)
        return [value for _ in self._get_indices(indices)]

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        del indices
        setattr(self.env, attr_name, value)

    def env_method(
        self,
        method_name: str,
        *method_args: Any,
        indices=None,
        **method_kwargs: Any,
    ) -> list[Any]:
        method = getattr(self.env, method_name)
        return [method(*method_args, **method_kwargs) for _ in self._get_indices(indices)]

    def env_is_wrapped(self, wrapper_class, indices=None) -> list[bool]:
        del wrapper_class
        return [False for _ in self._get_indices(indices)]
