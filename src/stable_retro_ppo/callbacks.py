from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from stable_retro_ppo.artifacts import checkpoint_step, log_wandb_model_artifact
from stable_retro_ppo.env import EnvConfig
from stable_retro_ppo.metric_names import metric_path_segment


class WandbCheckpointArtifactCallback(BaseCallback):
    def __init__(
        self,
        wandb_run,
        args: argparse.Namespace,
        config: EnvConfig,
        checkpoint_dir: str,
        scan_freq: int,
    ):
        super().__init__()
        self.wandb_run = wandb_run
        self.args = args
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.scan_freq = scan_freq
        self.logged_paths: set[Path] = set()

    def _on_step(self) -> bool:
        if self.scan_freq <= 1 or self.n_calls % self.scan_freq == 0:
            self.log_new_checkpoints()
        return True

    def log_new_checkpoints(self) -> None:
        for checkpoint_path in sorted(self.checkpoint_dir.glob("*.zip")):
            resolved_path = checkpoint_path.resolve()
            if resolved_path in self.logged_paths:
                continue
            step = checkpoint_step(checkpoint_path)
            aliases = ["latest"]
            if step is not None:
                aliases.append(f"step-{step}")
            log_wandb_model_artifact(
                self.wandb_run,
                self.args,
                self.config,
                checkpoint_path,
                kind="checkpoint",
                aliases=aliases,
            )
            self.logged_paths.add(resolved_path)


class ThroughputCallback(BaseCallback):
    """Log rollout-only and full-loop instantaneous throughput."""

    def __init__(self, clock: Callable[[], float] | None = None):
        super().__init__()
        self.clock = clock or time.perf_counter
        self.rollout_start_time: float | None = None
        self.rollout_start_timesteps: int | None = None
        self.previous_rollout_start_time: float | None = None
        self.previous_rollout_start_timesteps: int | None = None
        self.pending_fps_instant: float | None = None

    def _on_rollout_start(self) -> None:
        now = self.clock()
        if (
            self.previous_rollout_start_time is not None
            and self.previous_rollout_start_timesteps is not None
        ):
            elapsed = now - self.previous_rollout_start_time
            steps = self.num_timesteps - self.previous_rollout_start_timesteps
            if elapsed > 0 and steps > 0:
                self.pending_fps_instant = steps / elapsed

        self.rollout_start_time = now
        self.rollout_start_timesteps = self.num_timesteps
        self.previous_rollout_start_time = now
        self.previous_rollout_start_timesteps = self.num_timesteps

    def _on_rollout_end(self) -> None:
        now = self.clock()
        if self.rollout_start_time is not None and self.rollout_start_timesteps is not None:
            elapsed = now - self.rollout_start_time
            steps = self.num_timesteps - self.rollout_start_timesteps
            if elapsed > 0 and steps > 0:
                self.logger.record("time/rollout_fps", steps / elapsed)

        if self.pending_fps_instant is not None:
            self.logger.record("time/fps_instant", self.pending_fps_instant)
            self.pending_fps_instant = None

    def _on_step(self) -> bool:
        return True


class RolloutDiagnosticsCallback(BaseCallback):
    """Log rollout-buffer value and advantage distributions."""

    def __init__(self, wandb_run=None, log_histograms: bool = True):
        super().__init__()
        self.wandb_run = wandb_run
        self.log_histograms = log_histograms

    def _on_rollout_end(self) -> None:
        rollout_buffer = getattr(self.model, "rollout_buffer", None)
        if rollout_buffer is None:
            return

        value_predictions = self._finite_values(getattr(rollout_buffer, "values", None))
        advantages = self._finite_values(getattr(rollout_buffer, "advantages", None))
        self._record_stats("train/value_pred", value_predictions)
        self._record_stats("train/advantage", advantages)
        self._log_wandb_histograms(value_predictions, advantages)

    def _on_step(self) -> bool:
        return True

    @staticmethod
    def _finite_values(values: Any) -> np.ndarray:
        if values is None:
            return np.array([], dtype=np.float64)
        flattened = np.asarray(values, dtype=np.float64).reshape(-1)
        return flattened[np.isfinite(flattened)]

    def _record_stats(self, prefix: str, values: np.ndarray) -> None:
        if values.size == 0:
            return
        self.logger.record(f"{prefix}_mean", float(np.mean(values)))
        self.logger.record(f"{prefix}_std", float(np.std(values)))
        self.logger.record(f"{prefix}_min", float(np.min(values)))
        self.logger.record(f"{prefix}_max", float(np.max(values)))
        self.logger.record(f"{prefix}_abs_mean", float(np.mean(np.abs(values))))

    def _log_wandb_histograms(self, value_predictions: np.ndarray, advantages: np.ndarray) -> None:
        if self.wandb_run is None or not self.log_histograms:
            return

        import wandb

        payload: dict[str, object] = {"global_step": self.num_timesteps}
        if value_predictions.size > 0:
            payload["train/value_pred_histogram"] = wandb.Histogram(value_predictions)
        if advantages.size > 0:
            payload["train/advantage_histogram"] = wandb.Histogram(advantages)
        if len(payload) > 1:
            self.wandb_run.log(payload, step=self.num_timesteps)


class RollingCompletionStopCallback(BaseCallback):
    def __init__(
        self,
        rolling_window: int,
        threshold: float,
        run_dir: str,
        wandb_run=None,
    ):
        super().__init__()
        self.rolling_window = rolling_window
        self.threshold = threshold
        self.run_dir = Path(run_dir)
        self.wandb_run = wandb_run
        self.rollout_completion_count = 0
        self.total_completion_count = 0
        self.rollout_counts: deque[int] = deque(maxlen=rolling_window)
        self.rolling_mean = 0.0
        self.stop_requested = False

    def _on_step(self) -> bool:
        if self.stop_requested:
            return False

        infos = self.locals.get("infos", [])
        for info in infos:
            if bool(info.get("completion_event", info.get("level_complete", False))):
                self.rollout_completion_count += 1
                self.total_completion_count += 1

        return True

    def _on_rollout_end(self) -> None:
        self.rollout_counts.append(self.rollout_completion_count)
        self.rolling_mean = sum(self.rollout_counts) / len(self.rollout_counts)
        window_full = len(self.rollout_counts) >= self.rolling_window

        self.logger.record("train/completion_events_rollout", self.rollout_completion_count)
        self.logger.record("train/completion_events_rolling_mean", self.rolling_mean)
        self.logger.record("train/completion_events_total", self.total_completion_count)

        if self.wandb_run is not None:
            self.wandb_run.log(
                {
                    "train/completion_events_rollout": self.rollout_completion_count,
                    "train/completion_events_rolling_mean": self.rolling_mean,
                    "train/completion_events_total": self.total_completion_count,
                    "global_step": self.num_timesteps,
                },
                step=self.num_timesteps,
            )

        print(
            "completion rolling: "
            f"rollout={self.rollout_completion_count} "
            f"mean={self.rolling_mean:.3f}/{self.threshold:g} "
            f"window={len(self.rollout_counts)}/{self.rolling_window} "
            f"total={self.total_completion_count}",
            flush=True,
        )

        if window_full and self.rolling_mean >= self.threshold:
            self.stop_requested = True
            stop_path = self.run_dir / "early_stop.txt"
            stop_path.write_text(
                "\n".join(
                    [
                        "reason=rolling_completion_threshold",
                        f"timesteps={self.num_timesteps}",
                        f"rolling_window={self.rolling_window}",
                        f"rolling_mean={self.rolling_mean:.6f}",
                        f"threshold={self.threshold:.6f}",
                        f"total_completion_count={self.total_completion_count}",
                    ],
                )
                + "\n",
                encoding="utf-8",
            )
            print(
                "early stop requested: "
                f"rolling completion mean {self.rolling_mean:.3f} >= {self.threshold:g}",
                flush=True,
            )

        self.rollout_completion_count = 0


class TrainingCompletionRateStopCallback(BaseCallback):
    def __init__(
        self,
        episode_window: int,
        rate_threshold: float,
        run_dir: str,
        wandb_run=None,
        default_state: str | None = None,
    ):
        super().__init__()
        if not 0.0 < rate_threshold <= 1.0:
            raise ValueError("rate_threshold must be in (0, 1]")
        self.episode_window = episode_window
        self.rate_threshold = rate_threshold
        self.run_dir = Path(run_dir)
        self.wandb_run = wandb_run
        self.completed_episode_outcomes: deque[int] = deque(maxlen=episode_window)
        self.total_terminal_episodes = 0
        self.total_completed_episodes = 0
        self.default_state = default_state
        self.state_completed_episode_outcomes: dict[str, deque[int]] = {}
        self.state_total_terminal_episodes: dict[str, int] = {}
        self.state_total_completed_episodes: dict[str, int] = {}
        self.stop_requested = False

    def _on_step(self) -> bool:
        if self.stop_requested:
            return False

        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for done, info in zip(dones, infos, strict=False):
            if not bool(done) or bool(info.get("global_reset", False)):
                continue

            completed = bool(info.get("completion_event", info.get("level_complete", False)))
            self.completed_episode_outcomes.append(int(completed))
            self.total_terminal_episodes += 1
            if completed:
                self.total_completed_episodes += 1
            state_metric_payload = self.record_state_episode(info, completed)

            completion_rate = self.completion_rate
            self.logger.record("train/completion_episode_rate", completion_rate)
            self.logger.record(
                "train/completion_episode_window_size", len(self.completed_episode_outcomes)
            )
            self.logger.record("train/completion_episodes_total", self.total_completed_episodes)
            self.logger.record("train/terminal_episodes_total", self.total_terminal_episodes)

            if self.wandb_run is not None:
                self.wandb_run.log(
                    {
                        "train/completion_episode_rate": completion_rate,
                        "train/completion_episode_window_size": len(
                            self.completed_episode_outcomes,
                        ),
                        "train/completion_episodes_total": self.total_completed_episodes,
                        "train/terminal_episodes_total": self.total_terminal_episodes,
                        "global_step": self.num_timesteps,
                        **state_metric_payload,
                    },
                    step=self.num_timesteps,
                )

            if (
                len(self.completed_episode_outcomes) >= self.episode_window
                and completion_rate >= self.rate_threshold
            ):
                self.request_stop(completion_rate)
                return False

        return True

    @property
    def completion_rate(self) -> float:
        if not self.completed_episode_outcomes:
            return 0.0
        return sum(self.completed_episode_outcomes) / len(self.completed_episode_outcomes)

    def record_state_episode(self, info: dict[str, Any], completed: bool) -> dict[str, int | float]:
        state = info.get("start_state") or info.get("state") or self.default_state
        if not state:
            return {}

        state_key = str(state)
        outcomes = self.state_completed_episode_outcomes.setdefault(
            state_key,
            deque(maxlen=self.episode_window),
        )
        outcomes.append(int(completed))
        self.state_total_terminal_episodes[state_key] = (
            self.state_total_terminal_episodes.get(state_key, 0) + 1
        )
        if completed:
            self.state_total_completed_episodes[state_key] = (
                self.state_total_completed_episodes.get(state_key, 0) + 1
            )
        else:
            self.state_total_completed_episodes.setdefault(state_key, 0)

        segment = metric_path_segment(state_key)
        prefix = f"train/{segment}"
        state_rate = sum(outcomes) / len(outcomes)
        payload: dict[str, int | float] = {
            f"{prefix}/completion_episode_rate": state_rate,
            f"{prefix}/completion_episode_window_size": len(outcomes),
            f"{prefix}/completion_episodes_total": self.state_total_completed_episodes[state_key],
            f"{prefix}/terminal_episodes_total": self.state_total_terminal_episodes[state_key],
        }
        for key, value in payload.items():
            self.logger.record(key, value)
        return payload

    def request_stop(self, completion_rate: float) -> None:
        self.stop_requested = True
        stop_path = self.run_dir / "early_stop.txt"
        stop_path.write_text(
            "\n".join(
                [
                    "reason=training_completion_rate_threshold",
                    f"timesteps={self.num_timesteps}",
                    f"episode_window={self.episode_window}",
                    f"completion_rate={completion_rate:.6f}",
                    f"threshold={self.rate_threshold:.6f}",
                    f"total_terminal_episodes={self.total_terminal_episodes}",
                    f"total_completed_episodes={self.total_completed_episodes}",
                ],
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            "early stop requested: "
            f"training completion rate {completion_rate:.3f} >= {self.rate_threshold:g} "
            f"over last {self.episode_window} completed episodes",
            flush=True,
        )
