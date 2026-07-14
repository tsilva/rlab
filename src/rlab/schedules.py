from __future__ import annotations

import argparse
from collections.abc import Callable

from stable_baselines3 import PPO
from stable_baselines3.common.utils import get_schedule_fn

from rlab.callbacks import CallbackHelper
from rlab.metric_names import TRAIN_PPO_ENTROPY_COEFFICIENT


def linear_decay_schedule(
    initial_value: float,
    final_value: float,
    total_timesteps: int,
    schedule_timesteps: int = 0,
) -> Callable[[float], float]:
    if schedule_timesteps <= 0:
        schedule_timesteps = total_timesteps
    if schedule_timesteps <= 0:
        raise ValueError("schedule_timesteps must be positive")

    def schedule(progress_remaining: float) -> float:
        progress_remaining = min(max(progress_remaining, 0.0), 1.0)
        elapsed_timesteps = (1.0 - progress_remaining) * total_timesteps
        progress = min(max(elapsed_timesteps / schedule_timesteps, 0.0), 1.0)
        return initial_value + (final_value - initial_value) * progress

    return schedule


def learning_rate_schedule(args: argparse.Namespace) -> float | Callable[[float], float]:
    if args.learning_rate_final is None:
        return args.learning_rate
    return linear_decay_schedule(
        args.learning_rate,
        args.learning_rate_final,
        args.timesteps,
        args.learning_rate_schedule_timesteps,
    )


class EntropyCoefficientScheduleHelper(CallbackHelper):
    def __init__(
        self,
        initial_value: float,
        final_value: float,
        schedule_timesteps: int,
    ):
        super().__init__()
        if schedule_timesteps <= 0:
            raise ValueError("schedule_timesteps must be positive")
        self.initial_value = initial_value
        self.final_value = final_value
        self.schedule_timesteps = schedule_timesteps

    def _current_value(self) -> float:
        progress = min(max(self.num_timesteps / self.schedule_timesteps, 0.0), 1.0)
        return self.initial_value + (self.final_value - self.initial_value) * progress

    def _on_training_start(self) -> None:
        self.model.ent_coef = self._current_value()

    def _on_step(self) -> bool:
        ent_coef = self._current_value()
        self.model.ent_coef = ent_coef
        self.logger.record(TRAIN_PPO_ENTROPY_COEFFICIENT, ent_coef)
        return True


def apply_resume_hyperparameters(model: PPO, args: argparse.Namespace) -> None:
    lr_schedule = learning_rate_schedule(args)
    model.learning_rate = lr_schedule
    model.lr_schedule = get_schedule_fn(lr_schedule)
    model.ent_coef = args.ent_coef
    model.vf_coef = args.vf_coef
    model.n_epochs = args.n_epochs
    model.batch_size = args.batch_size
    model.clip_range = get_schedule_fn(args.clip_range)
    model.normalize_advantage = args.normalize_advantage
    model.target_kl = args.target_kl
    model.policy.optimizer.defaults["eps"] = args.adam_eps
    for param_group in model.policy.optimizer.param_groups:
        param_group["eps"] = args.adam_eps
