from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from stable_baselines3 import PPO

from rlab.metric_names import TRAIN_ADV_NORM_MODE, train_adv_task_metric


ADVANTAGE_NORMALIZATION_CHOICES = ("auto", "none", "global", "per-task")


def resolve_advantage_normalization_mode(args: Any) -> str:
    mode = getattr(args, "advantage_normalization", "auto")
    if mode == "auto":
        return "global" if getattr(args, "normalize_advantage", False) else "none"
    if mode not in ADVANTAGE_NORMALIZATION_CHOICES:
        raise ValueError(f"unknown advantage normalization mode: {mode!r}")
    return mode


def normalize_advantages_by_task(
    advantages: np.ndarray,
    observations: Mapping[str, np.ndarray],
    *,
    eps: float = 1e-8,
) -> dict[int, dict[str, float]]:
    if "task" not in observations:
        raise ValueError("per-task advantage normalization requires dict observations with a 'task' key")

    task_vectors = np.asarray(observations["task"])
    if task_vectors.ndim < 2:
        raise ValueError(f"expected one-hot task observations, got shape {task_vectors.shape}")
    if task_vectors.shape[:-1] != advantages.shape:
        raise ValueError(
            "task observation shape must match advantages except for task dimension: "
            f"task={task_vectors.shape}, advantages={advantages.shape}"
        )

    task_ids = np.argmax(task_vectors, axis=-1)
    task_count = int(task_vectors.shape[-1])
    stats: dict[int, dict[str, float]] = {}
    for task_id in range(task_count):
        mask = task_ids == task_id
        count = int(np.count_nonzero(mask))
        if count == 0:
            continue
        task_advantages = advantages[mask]
        mean = float(np.mean(task_advantages))
        std = float(np.std(task_advantages))
        stats[task_id] = {
            "count": float(count),
            "mean_pre": mean,
            "std_pre": std,
        }
        if count > 1:
            advantages[mask] = (task_advantages - mean) / (std + eps)
        stats[task_id]["mean_post"] = float(np.mean(advantages[mask]))
        stats[task_id]["std_post"] = float(np.std(advantages[mask]))
    return stats


class PerTaskAdvantagePPO(PPO):
    """PPO variant that normalizes rollout advantages per task once per update."""

    def train(self) -> None:
        stats = normalize_advantages_by_task(
            self.rollout_buffer.advantages,
            self.rollout_buffer.observations,
        )
        self.logger.record(TRAIN_ADV_NORM_MODE, 1.0)
        for task_id, task_stats in stats.items():
            for key, value in task_stats.items():
                self.logger.record(train_adv_task_metric(task_id, key), value)
        super().train()
