from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from stable_baselines3.common.vec_env import VecEnv

from rlab.batch_runtime import (
    BatchMetricRecord,
    BatchRuntime,
    CurriculumStepAttribution,
    EpisodeRecord,
    StepDiagnostics,
    TaskEventRecord,
    _InfoColumns,
    _copy_tree_lane,
)


class RlabVecEnv(VecEnv):
    """Translate the neutral batch runtime into SB3's VecEnv contract."""

    def __init__(self, runtime: BatchRuntime):
        self.runtime = runtime
        self.env = runtime.provider
        self.waiting = False
        self._actions: Any = None
        self._step_diagnostics: StepDiagnostics | None = None
        self._curriculum_step: Any | None = None
        self._snapshot_curriculum_enabled = (
            getattr(runtime, "snapshot_curriculum", None) is not None
        )
        self._seed_base: int | None = None
        self._dones = np.zeros(runtime.num_envs, dtype=bool)
        self._empty_infos: list[dict[str, Any]] = [{} for _ in range(runtime.num_envs)]
        super().__init__(runtime.num_envs, runtime.observation_space, runtime.action_space)

    def __getattr__(self, name: str) -> Any:
        if name in {"runtime", "env"}:
            raise AttributeError(name)
        return getattr(self.env, name)

    def seed(self, seed: int | None = None) -> list[int | None]:
        self._seed_base = seed
        if seed is None:
            return [None for _ in range(self.num_envs)]
        return [seed + lane_id for lane_id in self.runtime.global_lane_ids]

    def reset(self) -> Any:
        seed: int | list[int | None] | None = self._seed_base
        if seed is None and any(value is not None for value in self._seeds):
            seed = list(self._seeds)
        observations = self.runtime.reset(seed=seed, options_by_lane=list(self._options))
        self.reset_infos = self.runtime.reset_infos
        self._seed_base = None
        self._reset_seeds()
        self._reset_options()
        self.waiting = False
        self._actions = None
        self._step_diagnostics = None
        self._curriculum_step = None
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
        self._step_diagnostics = step.diagnostics
        if self._snapshot_curriculum_enabled:
            self._curriculum_step = CurriculumStepAttribution(
                curriculum_cell_ids=np.asarray(step.curriculum_cell_ids, dtype=object).copy(),
                curriculum_generations=np.asarray(
                    step.curriculum_generations, dtype=np.int64
                ).copy(),
                curriculum_episode_indices=np.asarray(
                    step.curriculum_episode_indices, dtype=np.int64
                ).copy(),
                curriculum_feedback_dones=np.asarray(
                    step.curriculum_feedback_dones, dtype=np.bool_
                ).copy(),
                control_boundaries=np.asarray(step.control_boundaries, dtype=np.bool_).copy(),
            )
        else:
            self._curriculum_step = None
        np.logical_or(step.terminated, step.truncated, out=self._dones)
        infos = self._empty_infos
        if np.any(self._dones):
            infos = [{} for _ in range(self.num_envs)]
            transition_columns = _InfoColumns(step.transition_info, self.num_envs)
            reset_columns = (
                _InfoColumns(step.reset_info, self.num_envs)
                if step.reset_info is not None
                else None
            )
            if step.final_observations is None:
                raise RuntimeError("done batch is missing final observations")
            for lane in np.flatnonzero(self._dones):
                lane_index = int(lane)
                info = transition_columns.lane(lane_index)
                episode_return = float(info.pop("rlab_episode_return"))
                episode_length = int(info.pop("rlab_episode_length"))
                episode_elapsed = float(info.pop("rlab_episode_elapsed"))
                control_boundary = bool(info.pop("rlab_control_boundary", False))
                info["terminal_observation"] = _copy_tree_lane(step.final_observations, lane_index)
                info["TimeLimit.truncated"] = bool(step.truncated[lane_index])
                if reset_columns is not None:
                    info["reset_info"] = reset_columns.lane(lane_index)
                if not control_boundary:
                    info["episode"] = {
                        "r": episode_return,
                        "l": episode_length,
                        "t": round(episode_elapsed, 6),
                    }
                infos[lane_index] = info
        return step.observations, step.rewards, self._dones, infos

    def take_step_diagnostics(self) -> StepDiagnostics | None:
        diagnostics = self._step_diagnostics
        self._step_diagnostics = None
        return diagnostics

    def take_curriculum_step(self) -> Any | None:
        step = self._curriculum_step
        self._curriculum_step = None
        return step

    def curriculum_begin_rollout(self) -> None:
        self.runtime.curriculum_begin_rollout()

    def curriculum_complete_rollout(self) -> dict[str, float]:
        return self.runtime.curriculum_complete_rollout()

    def submit_curriculum_feedback(self, cell_id: str, value_error: float) -> None:
        self.runtime.submit_curriculum_feedback(cell_id, value_error)

    def snapshot_curriculum_summary(self) -> Mapping[str, Any] | None:
        return self.runtime.snapshot_curriculum_summary()

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
        del mode
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
