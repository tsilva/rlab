from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import gymnasium as gym
import numpy as np
from numba import njit
from rlab.task_kernels import BoundTaskKernel, Outcome, event_names_from_bits


def __getattr__(name: str) -> Any:
    if name == "RlabVecEnv":
        from rlab.training.sb3_vec_env import RlabVecEnv

        return RlabVecEnv
    raise AttributeError(name)


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
    # Number of rotating provider-owned observation batches.
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
    boundary_reason: str = "natural"
    reset_reason: str | None = None


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
    final_observations: Any | None
    transition_info: Mapping[str, Any]
    reset_info: Mapping[str, Any] | None
    diagnostics: StepDiagnostics | None = None


@dataclass(frozen=True)
class StepDiagnostics:
    """Owned lane-zero facts for an opt-in interactive debugger step."""

    episode_index: int
    episode_seed: int | None
    start_id: str | None
    policy_action: Any
    native_action: Any
    provider_reward: float
    provider_terminated: bool
    provider_truncated: bool
    provider_info: Mapping[str, Any]
    task_reward: float
    task_terminated: bool
    task_truncated: bool
    outcome: Outcome
    events: tuple[str, ...]
    task_metrics: Mapping[str, Any]
    event_transitions: Mapping[str, tuple[Any, Any]]
    reward: float
    terminated: bool
    truncated: bool
    next_episode_seed: int | None


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
        return _copy_tree(value[lane])
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
        global_lane_ids: Sequence[int] | None = None,
        capture_step_diagnostics: bool = False,
    ):
        self.provider = provider
        self.descriptor = descriptor
        self.kernel = kernel
        self.num_envs = int(provider.num_envs)
        if self.num_envs <= 0:
            raise ValueError("provider num_envs must be positive")
        if capture_step_diagnostics and self.num_envs != 1:
            raise ValueError("step diagnostics require exactly one environment")
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
        self.global_lane_ids = tuple(
            range(self.num_envs) if global_lane_ids is None else global_lane_ids
        )
        if len(self.global_lane_ids) != self.num_envs:
            raise ValueError("global_lane_ids must contain one id per provider lane")
        if len(set(self.global_lane_ids)) != self.num_envs or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in self.global_lane_ids
        ):
            raise ValueError("global_lane_ids must be unique non-negative integers")
        self.capture_step_diagnostics = bool(capture_step_diagnostics)
        self.reset_infos: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]
        self._episode_returns = np.zeros(self.num_envs, dtype=np.float64)
        self._episode_lengths = np.zeros(self.num_envs, dtype=np.int64)
        self._episode_indices = np.zeros(self.num_envs, dtype=np.int64)
        self._episode_seeds: list[int | None] = [None for _ in range(self.num_envs)]
        self._start_ids: list[str | None] = [None for _ in range(self.num_envs)]
        self._records: list[EpisodeRecord | TaskEventRecord] = []
        self._pending_reset_mask = np.zeros(self.num_envs, dtype=np.bool_)
        self._pending_start_ids = np.full(self.num_envs, None, dtype=object)
        self._pending_reset_reasons = np.full(self.num_envs, None, dtype=object)
        self._has_pending_resets = False
        self._latest_metric_record: BatchMetricRecord | None = None
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
                None,
                {},
                None,
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
        self._final_observation_buffers: list[Any] = []
        self._current_observation_buffer = 0
        self._reuse_provider_observations = descriptor.observation_buffer_depth >= 2 and bool(
            getattr(kernel, "observation_encoding_is_view", False)
        )
        self._started_at = time.monotonic()
        self._native_step_seconds_total = 0.0
        self._native_step_calls_total = 0
        self._closed = False

    def request_resets(
        self,
        mask: np.ndarray,
        *,
        start_ids: Sequence[str | None] | np.ndarray | None = None,
        reason: str = "external",
    ) -> None:
        """Queue lane resets for the boundary after the next completed vector step."""

        if not self._observation_buffers:
            raise RuntimeError("BatchRuntime.reset() must be called before request_resets()")
        if not isinstance(mask, np.ndarray):
            raise TypeError("reset request mask must be a NumPy array")
        if mask.shape != (self.num_envs,):
            raise ValueError(f"reset request mask must have shape ({self.num_envs},)")
        if mask.dtype != np.bool_:
            raise TypeError("reset request mask must have dtype np.bool_")
        if not np.any(mask):
            raise ValueError("reset request mask must select at least one lane")
        if np.any(mask & self._pending_reset_mask):
            lanes = np.flatnonzero(mask & self._pending_reset_mask).tolist()
            raise ValueError(f"lanes already have pending reset requests: {lanes}")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reset request reason must be a non-empty string")

        requested_starts = np.full(self.num_envs, None, dtype=object)
        if start_ids is not None:
            requested_starts = np.asarray(start_ids, dtype=object)
            if requested_starts.shape != (self.num_envs,):
                raise ValueError(
                    f"reset request start_ids must have shape ({self.num_envs},)"
                )
            catalog = set(self.descriptor.start_catalog)
            for lane in np.flatnonzero(mask):
                start_id = requested_starts[int(lane)]
                if start_id is None or str(start_id) not in catalog:
                    raise ValueError(
                        f"unknown requested start id {start_id!r} for lane {int(lane)}"
                    )
                requested_starts[int(lane)] = str(start_id)

        self._pending_reset_mask[mask] = True
        self._pending_start_ids[mask] = requested_starts[mask]
        self._pending_reset_reasons[mask] = reason.strip()
        self._has_pending_resets = True

    def _seed_for(self, lane: int, episode_index: int) -> int:
        sequence = np.random.SeedSequence(
            [self.run_seed, self.global_lane_ids[lane], episode_index]
        )
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
        sequence = np.random.SeedSequence(
            [self.run_seed, self.global_lane_ids[lane], episode_index, 0x53544152]
        )
        generator = np.random.default_rng(sequence)
        return catalog[int(generator.choice(len(catalog), p=probabilities))]

    def _normalize_seeds(self, seed: int | Sequence[int | None] | None) -> list[int | None]:
        if seed is None:
            return [self._seed_for(lane, 0) for lane in range(self.num_envs)]
        if isinstance(seed, int):
            return [seed + lane_id for lane_id in self.global_lane_ids]
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
        normalized_seeds = self._normalize_seeds(seed)
        observations, infos = self.provider.reset(seed=normalized_seeds, options=options)
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
        self._final_observation_buffers = [
            _empty_tree_like(encoded),
            _empty_tree_like(encoded),
        ]
        self._current_observation_buffer = 0
        info_columns = _InfoColumns(infos, self.num_envs)
        self.reset_infos = [info_columns.lane(lane) for lane in range(self.num_envs)]
        self._start_ids = self._actual_start_ids(infos, starts, mask)
        self._episode_seeds = list(normalized_seeds)
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
                bool(mask[lane]) and requested[lane] is not None for lane in range(self.num_envs)
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
                f"provider rewards must have shape ({self.num_envs},), got {native_rewards.shape}"
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
        forced_reset_mask = self._pending_reset_mask
        forced_only_mask = self._pending_reset_mask
        if self._has_pending_resets:
            forced_reset_mask = self._pending_reset_mask.copy()
            forced_only_mask = forced_reset_mask & ~dones
            if np.any(forced_only_mask):
                terminated[forced_only_mask] = False
                truncated[forced_only_mask] = True
                dones[forced_only_mask] = True
                any_done = True
        done_indices = np.flatnonzero(dones) if any_done else self._empty_indices

        diagnostics = None
        if self.capture_step_diagnostics:
            lane = 0
            outcome = Outcome(int(np.asarray(task_step.outcomes)[lane]))
            if (
                outcome == Outcome.NEUTRAL
                and bool(truncated[lane])
                and not bool(forced_only_mask[lane])
            ):
                outcome = Outcome.TIMEOUT
            event_bits = int(np.asarray(task_step.event_bits, dtype=np.uint64)[lane])
            events = event_names_from_bits(event_bits, self.kernel.event_names)
            event_transitions: dict[str, tuple[Any, Any]] = {}
            for name in events:
                transition = task_step.event_transitions.get(name)
                if transition is None:
                    continue
                source, target = transition
                event_transitions[name] = (
                    _copy_tree_lane(np.asarray(source), lane),
                    _copy_tree_lane(np.asarray(target), lane),
                )
            next_episode_seed = (
                self._seed_for(lane, int(self._episode_indices[lane]) + 1)
                if bool(dones[lane])
                else None
            )
            diagnostics = StepDiagnostics(
                episode_index=int(self._episode_indices[lane]),
                episode_seed=self._episode_seeds[lane],
                start_id=self._start_ids[lane],
                policy_action=_copy_tree_lane(actions, lane),
                native_action=_copy_tree_lane(native_actions, lane),
                provider_reward=float(native_rewards[lane]),
                provider_terminated=bool(provider_terminated[lane]),
                provider_truncated=bool(provider_truncated[lane]),
                provider_info=_InfoColumns(infos, self.num_envs).lane(lane),
                task_reward=float(task_step.rewards[lane]),
                task_terminated=bool(task_terminated[lane]),
                task_truncated=bool(task_truncated[lane]),
                outcome=outcome,
                events=events,
                task_metrics=self._lane_metrics(task_step.metrics, lane),
                event_transitions=event_transitions,
                reward=float(rewards[lane]),
                terminated=bool(terminated[lane]),
                truncated=bool(truncated[lane]),
                next_episode_seed=next_episode_seed,
            )

        encoded_transition = self.kernel.encode_observations(observations)
        if self._reuse_provider_observations and not any_done:
            next_observations = encoded_transition
        else:
            # Terminal observations must survive the provider's masked reset.
            next_observations = self._observation_buffers[next_buffer_index]
            _copy_tree_into(next_observations, encoded_transition)

        if getattr(self.kernel, "has_events", True):
            self._append_event_records(task_step)
        transition_info: Mapping[str, Any] = infos
        reset_info: Mapping[str, Any] | None = None
        final_observations = None
        if done_indices.size:
            final_observations = self._final_observation_buffers[next_buffer_index]
            _copy_tree_lanes(final_observations, next_observations, dones)
            owned_transition_info = _copy_tree(infos)
            for name, values in (
                ("rlab_episode_return", self._episode_returns.copy()),
                ("rlab_episode_length", self._episode_lengths.copy()),
                (
                    "rlab_episode_elapsed",
                    np.full(
                        self.num_envs,
                        time.monotonic() - self._started_at,
                        dtype=np.float64,
                    ),
                ),
            ):
                owned_transition_info[name] = values
                owned_transition_info[f"_{name}"] = dones.copy()
            boundary_reasons = np.full(self.num_envs, None, dtype=object)
            reset_reasons = np.full(self.num_envs, None, dtype=object)
            for lane in done_indices:
                lane_index = int(lane)
                if bool(forced_only_mask[lane_index]):
                    boundary_reasons[lane_index] = "forced_reset"
                    reset_reasons[lane_index] = self._pending_reset_reasons[lane_index]
                elif bool(terminated[lane_index]):
                    boundary_reasons[lane_index] = "terminated"
                else:
                    boundary_reasons[lane_index] = "truncated"
            owned_transition_info["rlab_boundary_reason"] = boundary_reasons
            owned_transition_info["_rlab_boundary_reason"] = dones.copy()
            owned_transition_info["rlab_reset_reason"] = reset_reasons
            owned_transition_info["_rlab_reset_reason"] = forced_only_mask.copy()
            transition_info = owned_transition_info
            for lane in done_indices:
                lane_index = int(lane)
                self._append_record(
                    lane_index,
                    bool(terminated[lane_index]),
                    bool(truncated[lane_index]),
                    task_step,
                    forced_reset=bool(forced_only_mask[lane_index]),
                    reset_reason=(
                        str(self._pending_reset_reasons[lane_index])
                        if bool(forced_only_mask[lane_index])
                        else None
                    ),
                )

        if done_indices.size:
            self._episode_indices[dones] += 1
            starts = [
                self._start_for(lane, int(self._episode_indices[lane]))
                for lane in range(self.num_envs)
            ]
            for lane in np.flatnonzero(forced_reset_mask):
                lane_index = int(lane)
                requested_start = self._pending_start_ids[lane_index]
                if requested_start is not None:
                    starts[lane_index] = str(requested_start)
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
            reset_info = reset_infos
            self.kernel.on_reset(reset_observations, reset_infos, dones)
            encoded_reset = self.kernel.encode_observations(reset_observations)
            _copy_tree_lanes(next_observations, encoded_reset, dones)
            reset_info_columns = _InfoColumns(reset_infos, self.num_envs)
            actual_starts = self._actual_start_ids(reset_infos, starts, dones)
            for lane in done_indices:
                lane_index = int(lane)
                lane_reset_info = reset_info_columns.lane(lane_index)
                self.reset_infos[lane_index] = lane_reset_info
                self._start_ids[lane_index] = actual_starts[lane_index]
                self._episode_seeds[lane_index] = seeds[lane_index]
                self._episode_returns[lane_index] = 0.0
                self._episode_lengths[lane_index] = 0
            self._pending_reset_mask[dones] = False
            self._pending_start_ids[dones] = None
            self._pending_reset_reasons[dones] = None
            self._has_pending_resets = bool(np.any(self._pending_reset_mask))

        self._current_observation_buffer = next_buffer_index
        batch_step = self._batch_steps[next_buffer_index]
        batch_step.observations = next_observations
        batch_step.final_observations = final_observations
        batch_step.transition_info = transition_info
        batch_step.reset_info = reset_info
        batch_step.diagnostics = diagnostics
        return batch_step

    def _append_record(
        self,
        lane: int,
        terminated: bool,
        truncated: bool,
        task_step: Any,
        *,
        forced_reset: bool = False,
        reset_reason: str | None = None,
    ) -> None:
        outcome = Outcome(int(np.asarray(task_step.outcomes)[lane]))
        if forced_reset:
            outcome = Outcome.NEUTRAL
        elif outcome == Outcome.NEUTRAL and truncated:
            outcome = Outcome.TIMEOUT
        event_bits = int(np.asarray(task_step.event_bits, dtype=np.uint64)[lane])
        metrics = self._lane_metrics(task_step.metrics, lane)
        self._records.append(
            EpisodeRecord(
                lane=self.global_lane_ids[lane],
                episode_index=int(self._episode_indices[lane]),
                start_id=self._start_ids[lane],
                episode_return=float(self._episode_returns[lane]),
                episode_length=int(self._episode_lengths[lane]),
                terminated=terminated,
                truncated=truncated,
                outcome=outcome,
                events=event_names_from_bits(event_bits, self.kernel.event_names),
                metrics=metrics,
                boundary_reason=(
                    "forced_reset" if forced_reset else "terminated" if terminated else "truncated"
                ),
                reset_reason=reset_reason,
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
                    lane=self.global_lane_ids[lane_index],
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
