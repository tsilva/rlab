from __future__ import annotations

import json
import math
import time
import uuid
from collections import deque
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import KVWriter

from rlab.artifacts import write_model_metadata
from rlab.early_stop import (
    evaluate_early_stop_config,
    flat_metric_rule_from_early_stop,
    normalize_early_stop_config,
)
from rlab.env import EnvConfig
from rlab.eval_metrics import episode_reason_names
from rlab.metric_names import (
    canonical_training_scalars,
    GLOBAL_STEP,
    TRAIN_EPISODE_COUNT,
    TRAIN_OUTCOME_SUCCESS_CURRENT_RATE_MEAN,
    TRAIN_OUTCOME_SUCCESS_CURRENT_RATE_MIN,
    TRAIN_OUTCOME_SUCCESS_WINDOW_100_RATE_MEAN,
    TRAIN_OUTCOME_SUCCESS_WINDOW_100_RATE_MIN,
    TRAIN_OUTCOME_SUCCESS_START_COVERAGE_RATE,
    TRAIN_OUTCOME_TERMINAL_COUNT,
    TRAIN_PPO_ADVANTAGE_HIST,
    TRAIN_PPO_ADVANTAGE_ROOT,
    TRAIN_PPO_POLICY_ACTION_HIST,
    TRAIN_PPO_POLICY_DOMINANT_ACTION_RATE,
    TRAIN_PPO_VALUE_PREDICTION_HIST,
    TRAIN_PPO_VALUE_PREDICTION_ROOT,
    TRAIN_REWARD_ROOT,
    TRAIN_THROUGHPUT_BETWEEN_ROLLOUTS_SECONDS,
    TRAIN_THROUGHPUT_ENV_STEP_FPS,
    TRAIN_THROUGHPUT_ENV_STEP_SECONDS,
    TRAIN_THROUGHPUT_LOOP_FPS,
    TRAIN_THROUGHPUT_LOOP_SECONDS,
    TRAIN_THROUGHPUT_ROLLOUT_FPS,
    TRAIN_THROUGHPUT_ROLLOUT_OVERHEAD_SECONDS,
    TRAIN_THROUGHPUT_ROLLOUT_SECONDS,
    stat_metric,
    train_outcome_reason_count_metric,
    train_outcome_reason_window_rate_metric,
    train_reward_component_metric,
    train_reward_signal_metric,
    metric_value_segment,
    train_success_attempts_metric,
    train_success_count_metric,
    train_success_current_rate_metric,
    train_success_window_rate_metric,
    validate_metric_payload,
)
from rlab.metric_store import MetricStore


def task_metric_source(start_id: Any) -> Any:
    """Keep the same readable start identifier in training and evaluation."""
    return start_id


class CallbackHelper:
    """Plain lifecycle component driven by the single SB3 callback."""

    def __init__(self) -> None:
        self.model: Any = None
        self.locals: dict[str, Any] = {}
        self.globals: dict[str, Any] = {}
        self.num_timesteps = 0
        self.n_calls = 0

    @property
    def logger(self) -> Any:
        return self.model.logger

    def bind(self, callback: BaseCallback) -> None:
        self.model = callback.model
        self.locals = callback.locals
        self.globals = callback.globals
        self.num_timesteps = callback.num_timesteps
        self.n_calls = callback.n_calls


class LedgerCheckpointHelper(CallbackHelper):
    def __init__(
        self,
        *,
        args: Any,
        config: EnvConfig,
        save_freq: int,
        save_path: str | Path,
        name_prefix: str,
        metric_store_path: Path | str,
        eval_required: bool = True,
    ) -> None:
        super().__init__()
        self.args = args
        self.config = config
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.name_prefix = name_prefix
        self.metric_store = MetricStore(metric_store_path)
        self.eval_required = bool(eval_required)

    def _init_callback(self) -> None:
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.metric_store.init()

    def _on_step(self) -> bool:
        if self.save_freq <= 0 or self.n_calls % self.save_freq != 0:
            return True
        self.save_checkpoint(self.num_timesteps, kind="checkpoint")
        return True

    def save_checkpoint(self, step: int, *, kind: str) -> Path:
        final_path = self.save_path / f"{self.name_prefix}_{step}_steps.zip"
        temp_path = self.save_path / f".{final_path.stem}.{uuid.uuid4().hex}.zip"
        self.model.save(str(temp_path))
        temp_path.replace(final_path)
        metadata_path = write_model_metadata(
            final_path,
            self.args,
            self.config,
            kind,
            checkpoint_step_value=step,
        )
        checkpoint_id = self.metric_store.record_checkpoint(
            run_name=str(getattr(self.args, "run_name", "")),
            kind=kind,
            step=step,
            path=final_path,
            metadata_path=metadata_path,
            sha256=None,
            eval_required=self.eval_required,
        )
        print(
            f"checkpoint ready: id={checkpoint_id} step={step} path={final_path}",
            flush=True,
        )
        return final_path


@dataclass(frozen=True)
class _CompletedRollout:
    step: int
    steps: int
    start_time: float
    end_time: float
    rollout_seconds: float
    env_step_seconds: float | None


class ThroughputHelper(CallbackHelper):
    """Publish one temporally aligned frame for each completed training iteration."""

    def __init__(
        self,
        clock: Callable[[], float] | None = None,
        *,
        metric_store_path: Path | str | None = None,
        wandb_run=None,
    ):
        super().__init__()
        self.clock = clock or time.perf_counter
        self.metric_store = MetricStore(metric_store_path) if metric_store_path else None
        if self.metric_store is not None:
            self.metric_store.init()
        self.wandb_run = wandb_run
        self.rollout_start_time: float | None = None
        self.rollout_start_timesteps: int | None = None
        self.completed_rollout: _CompletedRollout | None = None
        self.native_step_stats_start: Mapping[str, float | int] | None = None

    @staticmethod
    def _native_step_stats_source(env: Any) -> Any | None:
        seen: set[int] = set()
        current = env
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            native_step_stats = getattr(current, "native_step_stats", None)
            if callable(native_step_stats):
                return current
            current = getattr(current, "venv", None) or getattr(current, "env", None)
        return None

    @classmethod
    def _native_step_stats(cls, env: Any) -> Mapping[str, float | int] | None:
        source = cls._native_step_stats_source(env)
        if source is None:
            return None
        stats = source.native_step_stats()
        return stats if isinstance(stats, Mapping) else None

    def _on_rollout_start(self) -> None:
        now = self.clock()
        if self.completed_rollout is not None:
            self._publish_completed_iteration(self.completed_rollout, next_start_time=now)
            self.completed_rollout = None

        self.rollout_start_time = now
        self.rollout_start_timesteps = self.num_timesteps
        self.native_step_stats_start = self._native_step_stats(getattr(self.model, "env", None))

    def _on_rollout_end(self) -> None:
        now = self.clock()
        if self.rollout_start_time is not None and self.rollout_start_timesteps is not None:
            elapsed = now - self.rollout_start_time
            steps = self.num_timesteps - self.rollout_start_timesteps
            if elapsed > 0 and steps > 0:
                self.completed_rollout = _CompletedRollout(
                    step=self.num_timesteps,
                    steps=steps,
                    start_time=self.rollout_start_time,
                    end_time=now,
                    rollout_seconds=elapsed,
                    env_step_seconds=self._native_step_seconds(),
                )

    def _on_step(self) -> bool:
        return True

    def _native_step_seconds(self) -> float | None:
        start = self.native_step_stats_start
        end = self._native_step_stats(getattr(self.model, "env", None))
        self.native_step_stats_start = None
        if start is None or end is None:
            return None
        native_seconds = float(end.get("seconds_total", 0.0)) - float(
            start.get("seconds_total", 0.0)
        )
        native_calls = int(end.get("calls_total", 0)) - int(start.get("calls_total", 0))
        if native_seconds <= 0 or native_calls <= 0:
            return None
        return native_seconds

    def _publish_completed_iteration(
        self,
        rollout: _CompletedRollout,
        *,
        next_start_time: float,
    ) -> None:
        between_seconds = next_start_time - rollout.end_time
        loop_seconds = next_start_time - rollout.start_time
        if between_seconds < 0 or loop_seconds <= 0:
            return
        payload: dict[str, float] = {
            TRAIN_THROUGHPUT_LOOP_FPS: rollout.steps / loop_seconds,
            TRAIN_THROUGHPUT_ROLLOUT_FPS: rollout.steps / rollout.rollout_seconds,
            TRAIN_THROUGHPUT_LOOP_SECONDS: loop_seconds,
            TRAIN_THROUGHPUT_ROLLOUT_SECONDS: rollout.rollout_seconds,
            TRAIN_THROUGHPUT_BETWEEN_ROLLOUTS_SECONDS: between_seconds,
        }
        if rollout.env_step_seconds is not None:
            payload.update(
                {
                    TRAIN_THROUGHPUT_ENV_STEP_FPS: rollout.steps / rollout.env_step_seconds,
                    TRAIN_THROUGHPUT_ENV_STEP_SECONDS: rollout.env_step_seconds,
                    TRAIN_THROUGHPUT_ROLLOUT_OVERHEAD_SECONDS: max(
                        rollout.rollout_seconds - rollout.env_step_seconds,
                        0.0,
                    ),
                }
            )
        validate_metric_payload(payload)
        if self.metric_store is not None:
            self.metric_store.append_metrics(
                payload,
                step=rollout.step,
                source="train",
                publish=self.wandb_run is None,
            )
            if self.wandb_run is not None:
                self.wandb_run.log({GLOBAL_STEP: rollout.step, **payload})
            return
        for name, value in payload.items():
            self.logger.record(name, value)


class MetricThresholdStopHelper(CallbackHelper):
    def __init__(
        self,
        *,
        marker_path: Path,
        detector: Any,
        metric_store_path: Path | None = None,
        poll_seconds: float = 30.0,
        clock: Callable[[], float] | None = None,
    ) -> None:
        super().__init__()
        self.detector = normalize_early_stop_config(detector, label="early_stop")
        self.flat_rule = flat_metric_rule_from_early_stop(self.detector)
        self.metric_name = str(self.flat_rule["metric"]) if self.flat_rule else ""
        self.threshold = float(self.flat_rule["threshold"]) if self.flat_rule else None
        self.operator = str(self.flat_rule["operator"]) if self.flat_rule else ""
        self.marker_path = marker_path
        self.triggered = False
        self.stop_requested = False
        self.metric_store = (
            MetricStore(metric_store_path, timeout=0.05) if metric_store_path else None
        )
        self.poll_seconds = poll_seconds
        self.clock = clock or time.perf_counter
        self.last_poll: float | None = None
        self.last_lookup_warning: float | None = None

    def _on_step(self) -> bool:
        if self.metric_store is not None:
            return not self.stop_requested
        return self.evaluate_now()

    def _on_rollout_end(self) -> None:
        if self.metric_store is None or self.stop_requested:
            return
        now = self.clock()
        if self.last_poll is not None and now - self.last_poll < self.poll_seconds:
            return
        self.last_poll = now
        self.evaluate_now()

    def evaluate_now(self) -> bool:
        result, values = evaluate_early_stop_config(self.detector, self.current_metric_value)
        if result is not True:
            return True
        self.triggered = True
        self.stop_requested = True
        self.write_marker(values)
        print(
            "early stop: "
            f"{self.describe_trigger(values)}; "
            f"stopping at num_timesteps={self.num_timesteps}",
            flush=True,
        )
        return False

    def current_metric_value(self, metric_name: str | None = None) -> float | None:
        metric_name = self.metric_name if metric_name is None else str(metric_name)
        logger = getattr(self.model, "logger", None)
        for attr in ("name_to_value", "records"):
            values = getattr(logger, attr, None)
            if not isinstance(values, Mapping) or metric_name not in values:
                continue
            try:
                value = float(values[metric_name])
            except TypeError, ValueError:
                return None
            return value if math.isfinite(value) else None
        if self.metric_store is not None:
            try:
                return self.metric_store.latest_metric(metric_name)
            except Exception as exc:
                now = self.clock()
                if self.last_lookup_warning is None or now - self.last_lookup_warning >= 60:
                    print(
                        f"warning: metric store lookup failed for early stop metric "
                        f"{metric_name}: {exc}",
                        flush=True,
                    )
                    self.last_lookup_warning = now
        return None

    def describe_trigger(self, values: Mapping[str, float]) -> str:
        if self.flat_rule:
            value = values.get(self.metric_name)
            if value is not None:
                return (
                    f"{self.metric_name} {value:.12g} {self.operator} {float(self.threshold):.12g}"
                )
        return "early_stop metrics matched"

    def write_marker(self, values: Mapping[str, float]) -> None:
        self.marker_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "early_stop=metric_threshold",
            f"early_stop_timesteps={self.num_timesteps}",
            f"early_stop_detector_json={json.dumps(self.detector, sort_keys=True, separators=(',', ':'))}",
            f"timesteps={self.num_timesteps}",
        ]
        if self.flat_rule:
            value = values.get(self.metric_name)
            if value is not None:
                lines.extend(
                    [
                        f"early_stop_metric={self.metric_name}",
                        f"early_stop_operator={self.operator}",
                        f"early_stop_threshold={float(self.threshold):.12g}",
                        f"early_stop_value={value:.12g}",
                        f"{self.metric_name}={value:.12g}",
                    ]
                )
        else:
            lines.extend(
                f"early_stop_value/{metric}={value:.12g}"
                for metric, value in sorted(values.items())
            )
        self.marker_path.write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )


class MetricStoreOutputFormat(KVWriter):
    """Persist the complete scalar payload received by SB3's logger dump."""

    def __init__(
        self,
        metric_store_path: Path | str,
        *,
        source: str = "train",
        wandb_run=None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        # Training metrics are durable evidence. Let SQLite's bounded busy
        # handler absorb brief publisher/coordinator transactions instead of
        # terminating a multi-hour run after 50 ms of contention.
        self.metric_store = MetricStore(metric_store_path)
        self.source = source
        self.wandb_run = wandb_run
        self.clock = clock or time.perf_counter
        self.last_warning: float | None = None
        self.metric_store.init()

    def write(
        self,
        key_values: dict[str, Any],
        key_excluded: dict[str, tuple[str, ...]],
        step: int = 0,
    ) -> None:
        del key_excluded
        payload = {
            key: value
            for key, value in canonical_training_scalars(key_values).items()
            if math.isfinite(value)
        }
        if not payload:
            return
        try:
            self.metric_store.append_metrics(
                payload,
                step=step,
                source=self.source,
                publish=self.wandb_run is None,
            )
            if self.wandb_run is not None:
                self.wandb_run.log({**payload, GLOBAL_STEP: step})
        except Exception as exc:
            now = self.clock()
            if self.last_warning is None or now - self.last_warning >= 60:
                print(f"metric store write failed: {exc}", flush=True)
                self.last_warning = now
            raise RuntimeError("durable metric frame write failed") from exc

    def close(self) -> None:
        """MetricStore owns no persistent connection and the W&B run is shared."""


class MetricStoreLoggerHelper(CallbackHelper):
    """Install the durable metric writer at SB3's authoritative dump boundary."""

    def __init__(
        self,
        metric_store_path: Path | str,
        *,
        source: str = "train",
        wandb_run=None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        super().__init__()
        self.output_format = MetricStoreOutputFormat(
            metric_store_path,
            source=source,
            wandb_run=wandb_run,
            clock=clock,
        )

    def _on_training_start(self) -> None:
        output_formats = self.logger.output_formats
        if self.output_format not in output_formats:
            output_formats.append(self.output_format)

    def _on_step(self) -> bool:
        return True

    def _on_training_end(self) -> None:
        pending = getattr(self.logger, "name_to_value", None)
        if isinstance(pending, Mapping) and pending:
            self.logger.dump(step=self.num_timesteps)


class RolloutDiagnosticsHelper(CallbackHelper):
    """Log compact PPO rollout and discrete-policy collapse diagnostics."""

    def __init__(
        self,
        wandb_run=None,
        log_histograms: bool = True,
        *,
        metric_store_path: Path | str | None = None,
        histogram_interval: int = 64,
    ):
        super().__init__()
        self.wandb_run = wandb_run
        self.log_histograms = log_histograms
        self.metric_store = MetricStore(metric_store_path) if metric_store_path else None
        self.histogram_interval = max(int(histogram_interval), 1)
        self.rollout_count = 0

    def _on_rollout_end(self) -> None:
        self.rollout_count += 1
        rollout_buffer = getattr(self.model, "rollout_buffer", None)
        if rollout_buffer is None:
            return

        value_predictions = self._finite_values(getattr(rollout_buffer, "values", None))
        advantages = self._finite_values(getattr(rollout_buffer, "advantages", None))
        discrete_actions = self._discrete_actions(
            getattr(rollout_buffer, "actions", None),
            getattr(self.model, "action_space", None),
        )
        self._record_stats(TRAIN_PPO_VALUE_PREDICTION_ROOT, value_predictions)
        self._record_stats(TRAIN_PPO_ADVANTAGE_ROOT, advantages)
        if discrete_actions.size > 0:
            _actions, counts = np.unique(discrete_actions, return_counts=True)
            self.logger.record(
                TRAIN_PPO_POLICY_DOMINANT_ACTION_RATE,
                float(np.max(counts) / discrete_actions.size),
            )
        if self.rollout_count % self.histogram_interval == 0:
            self._log_histograms(value_predictions, advantages, discrete_actions)

    def _on_step(self) -> bool:
        return True

    @staticmethod
    def _finite_values(values: Any) -> np.ndarray:
        if values is None:
            return np.array([], dtype=np.float64)
        flattened = np.asarray(values, dtype=np.float64).reshape(-1)
        return flattened[np.isfinite(flattened)]

    @staticmethod
    def _discrete_actions(actions: Any, action_space: Any) -> np.ndarray:
        if actions is None or not isinstance(getattr(action_space, "n", None), Integral):
            return np.array([], dtype=np.int64)
        values = np.asarray(actions)
        if values.size == 0 or (values.ndim > 1 and values.shape[-1] != 1):
            return np.array([], dtype=np.int64)
        flattened = values.reshape(-1)
        if not np.issubdtype(flattened.dtype, np.number):
            return np.array([], dtype=np.int64)
        finite = flattened[np.isfinite(flattened)]
        integers = finite.astype(np.int64)
        if not np.allclose(finite, integers):
            return np.array([], dtype=np.int64)
        return integers

    def _record_stats(self, prefix: str, values: np.ndarray) -> None:
        if values.size == 0:
            return
        self.logger.record(stat_metric(prefix, "mean"), float(np.mean(values)))
        self.logger.record(stat_metric(prefix, "std"), float(np.std(values)))
        self.logger.record(stat_metric(prefix, "min"), float(np.min(values)))
        self.logger.record(stat_metric(prefix, "max"), float(np.max(values)))

    def _log_histograms(
        self,
        value_predictions: np.ndarray,
        advantages: np.ndarray,
        discrete_actions: np.ndarray,
    ) -> None:
        if not self.log_histograms:
            return
        values: dict[str, list[float]] = {}
        if value_predictions.size > 0:
            values[TRAIN_PPO_VALUE_PREDICTION_HIST] = value_predictions.tolist()
        if advantages.size > 0:
            values[TRAIN_PPO_ADVANTAGE_HIST] = advantages.tolist()
        if discrete_actions.size > 0:
            values[TRAIN_PPO_POLICY_ACTION_HIST] = discrete_actions.astype(float).tolist()
        if not values:
            return
        validate_metric_payload(values)
        if self.metric_store is not None:
            self.metric_store.enqueue_event(
                kind="histogram",
                payload={"histograms": values},
                step=self.num_timesteps,
                source="train",
            )
        if self.wandb_run is not None:
            import wandb

            self.wandb_run.log(
                {
                    GLOBAL_STEP: self.num_timesteps,
                    **{name: wandb.Histogram(data) for name, data in values.items()},
                }
            )


class _BufferedStats:
    """Reusable contiguous storage for one rollout's vector batches."""

    __slots__ = ("buffer", "size")

    def __init__(self) -> None:
        self.buffer = np.empty(0, dtype=np.float64)
        self.size = 0

    def reset(self) -> None:
        self.size = 0

    def update(self, value: Any, *, reserve: int) -> None:
        values = np.asarray(value).reshape(-1)
        if values.size == 0:
            return
        end = self.size + values.size
        if end > self.buffer.size:
            capacity = max(end, reserve, max(64, self.buffer.size * 2))
            grown = np.empty(capacity, dtype=np.float64)
            grown[: self.size] = self.buffer[: self.size]
            self.buffer = grown
        self.buffer[self.size : end] = values
        self.size = end

    def flush(self) -> np.ndarray:
        values = self.buffer[: self.size]
        values = values[np.isfinite(values)]
        self.reset()
        return values


class _RewardStatsAccumulator:
    component_info_keys = {
        "native": "native_reward_component",
        "progress": "progress_reward_component",
        "score": "score_reward_component",
        "completion": "completion_reward_component",
        "death": "death_penalty_component",
        "time": "time_penalty_component",
    }
    signal_info_keys = {
        "progress": "progress_component",
        "score": "score_delta",
    }

    def __init__(
        self,
        *,
        active_components: Sequence[str] = (),
        active_signals: Sequence[str] = (),
    ) -> None:
        self.shaped = _BufferedStats()
        self.raw = _BufferedStats()
        self.active_components = tuple(
            component for component in active_components if component in self.component_info_keys
        )
        self.components = {component: _BufferedStats() for component in self.active_components}
        self.signals = {
            signal: _BufferedStats() for signal in active_signals if signal in self.signal_info_keys
        }

    def consume(self, metrics: Mapping[str, Any], *, reserve: int) -> None:
        if (value := metrics.get("shaped_reward")) is not None:
            self.shaped.update(value, reserve=reserve)
        if (value := metrics.get("raw_reward")) is not None:
            self.raw.update(value, reserve=reserve)
        for component, accumulator in self.components.items():
            info_key = self.component_info_keys[component]
            value = metrics.get(info_key)
            if value is not None:
                accumulator.update(value, reserve=reserve)
        for signal, accumulator in self.signals.items():
            value = metrics.get(self.signal_info_keys[signal])
            if value is not None:
                accumulator.update(value, reserve=reserve)

    @staticmethod
    def _distribution(prefix: str, values: np.ndarray, stats: Sequence[str]) -> dict[str, float]:
        if values.size == 0:
            return {}
        calculations = {
            "mean": lambda: float(np.mean(values)),
            "std": lambda: float(np.std(values)),
            "min": lambda: float(np.min(values)),
            "max": lambda: float(np.max(values)),
            "nonzero_rate": lambda: float(np.mean(values != 0.0)),
        }
        return {stat_metric(prefix, stat): calculations[stat]() for stat in stats}

    def flush(self) -> dict[str, float]:
        shaped = self.shaped.flush()
        raw = self.raw.flush()
        payload = self._distribution(
            f"{TRAIN_REWARD_ROOT}/shaped",
            shaped,
            ("mean", "std", "min", "max", "nonzero_rate"),
        )
        if raw.size > 0 and (shaped.size != raw.size or not np.array_equal(shaped, raw)):
            payload.update(self._distribution(f"{TRAIN_REWARD_ROOT}/raw", raw, ("mean", "std")))
        abs_sums: dict[str, float] = {}
        for component, accumulator in self.components.items():
            values = accumulator.flush()
            if values.size == 0:
                continue
            payload[train_reward_component_metric(component, "mean")] = float(np.mean(values))
            payload[train_reward_component_metric(component, "nonzero_rate")] = float(
                np.mean(values != 0.0)
            )
            abs_sums[component] = float(np.sum(np.abs(values)))
        total_abs_sum = sum(abs_sums.values())
        for component, abs_sum in abs_sums.items():
            payload[train_reward_component_metric(component, "share")] = (
                abs_sum / total_abs_sum if total_abs_sum > 0.0 else 0.0
            )
        for signal, accumulator in self.signals.items():
            values = accumulator.flush()
            if values.size == 0:
                continue
            payload.update(
                {
                    train_reward_signal_metric(signal, "mean"): float(np.mean(values)),
                    train_reward_signal_metric(signal, "max"): float(np.max(values)),
                    train_reward_signal_metric(signal, "nonzero_rate"): float(
                        np.mean(values != 0.0)
                    ),
                }
            )
        return payload


class _DoneMetricsReducer:
    ep_window_size = 100

    def __init__(
        self,
        event_names: Sequence[str] = (),
    ) -> None:
        self.event_names = tuple(str(name) for name in event_names)
        self.done_count = 0
        self.reason_counts: dict[str, int] = {reason: 0 for reason in self.event_names}
        self.reason_windows: dict[str, deque[bool]] = {
            reason: deque(maxlen=self.ep_window_size) for reason in self.event_names
        }

    def consume(self, record: Any) -> dict[str, int | float] | None:
        if not hasattr(record, "episode_return"):
            return None
        outcome = getattr(record, "outcome", None)
        outcome_name = str(getattr(outcome, "name", outcome)).lower()
        reasons = (
            set()
            if outcome_name == "success"
            else episode_reason_names(
                getattr(record, "events", ()) or (),
                terminated=bool(getattr(record, "terminated", False)),
                truncated=bool(getattr(record, "truncated", False)),
            )
        )
        return self.record_done(reasons)

    def record_done(self, reasons: set[str] | Mapping[str, Any]) -> dict[str, int | float]:
        active_reasons = {str(reason) for reason in reasons}
        prior_episode_count = self.done_count
        self.done_count += 1
        for reason in active_reasons:
            if reason not in self.reason_windows:
                prior = min(prior_episode_count, self.ep_window_size - 1)
                self.reason_windows[reason] = deque(
                    [False] * prior,
                    maxlen=self.ep_window_size,
                )
                self.reason_counts[reason] = 0
            self.reason_counts[reason] = self.reason_counts.get(reason, 0) + 1
        for reason, window in self.reason_windows.items():
            window.append(reason in active_reasons)
        return self.record_metrics()

    def record_metrics(self) -> dict[str, int | float]:
        payload: dict[str, int | float] = {
            TRAIN_OUTCOME_TERMINAL_COUNT: self.done_count,
            TRAIN_EPISODE_COUNT: self.done_count,
        }
        for reason, count in sorted(self.reason_counts.items()):
            window = self.reason_windows[reason]
            payload[train_outcome_reason_count_metric(reason)] = count
            payload[train_outcome_reason_window_rate_metric(reason)] = sum(window) / len(window)
        return payload


class _SuccessMetricsReducer:
    ep_window_size = 100

    def __init__(self, *, configured_starts: Sequence[str] = ()) -> None:
        self.configured_starts = tuple(dict.fromkeys(str(start) for start in configured_starts))
        self.success_counts: dict[str, int] = {}
        self.attempt_counts: dict[str, int] = {}
        self.attempt_windows: dict[str, deque[bool]] = {}

    def consume(self, record: Any) -> dict[str, int | float]:
        if not hasattr(record, "episode_return"):
            return {}
        start_id = getattr(record, "start_id", None)
        if start_id is None:
            return {}
        outcome = getattr(record, "outcome", None)
        outcome_name = str(getattr(outcome, "name", outcome)).lower()
        return self.record_attempt(
            task_metric_source(start_id), completed=outcome_name == "success"
        )

    def record_attempt(self, source: Any, *, completed: bool) -> dict[str, int | float]:
        source_key = metric_value_segment(source)
        count_metric = train_success_count_metric(source_key)
        attempts_metric = train_success_attempts_metric(source_key)
        current_rate_metric = train_success_current_rate_metric(source_key)
        window_rate_metric = train_success_window_rate_metric(source_key)
        window = self.attempt_windows.setdefault(source_key, deque(maxlen=self.ep_window_size))
        window.append(completed)
        self.attempt_counts[source_key] = self.attempt_counts.get(source_key, 0) + 1
        if completed:
            self.success_counts[source_key] = self.success_counts.get(source_key, 0) + 1

        current_rates = {
            start: self.success_counts.get(start, 0) / attempts
            for start, attempts in self.attempt_counts.items()
        }
        current_rate = current_rates[source_key]
        expected_starts = self.configured_starts or tuple(self.attempt_counts)
        coverage = sum(start in self.attempt_counts for start in expected_starts) / len(
            expected_starts
        )
        payload: dict[str, int | float] = {
            count_metric: self.success_counts.get(source_key, 0),
            attempts_metric: self.attempt_counts[source_key],
            current_rate_metric: current_rate,
            TRAIN_OUTCOME_SUCCESS_CURRENT_RATE_MIN: min(current_rates.values()),
            TRAIN_OUTCOME_SUCCESS_CURRENT_RATE_MEAN: float(np.mean(tuple(current_rates.values()))),
            TRAIN_OUTCOME_SUCCESS_START_COVERAGE_RATE: coverage,
        }
        if len(window) >= self.ep_window_size:
            payload[window_rate_metric] = sum(window) / len(window)
        if expected_starts and all(
            len(self.attempt_windows.get(start, ())) >= self.ep_window_size
            for start in expected_starts
        ):
            rates = [
                sum(self.attempt_windows[start]) / self.ep_window_size for start in expected_starts
            ]
            payload[TRAIN_OUTCOME_SUCCESS_WINDOW_100_RATE_MIN] = min(rates)
            payload[TRAIN_OUTCOME_SUCCESS_WINDOW_100_RATE_MEAN] = float(np.mean(rates))
        return payload


class RuntimeMetricsHelper(CallbackHelper):
    """Reduce runtime records and publish one scalar payload per rollout."""

    def __init__(
        self,
        *,
        wandb_run=None,
        event_names: Sequence[str] = (),
        active_reward_components: Sequence[str] = (),
        active_reward_signals: Sequence[str] = (),
        configured_starts: Sequence[str] = (),
        track_success: bool = False,
    ) -> None:
        super().__init__()
        self.wandb_run = wandb_run
        self.reward_stats = _RewardStatsAccumulator(
            active_components=active_reward_components,
            active_signals=active_reward_signals,
        )
        self.done_metrics = _DoneMetricsReducer(event_names=event_names)
        self.success_metrics = (
            _SuccessMetricsReducer(configured_starts=configured_starts) if track_success else None
        )
        self.pending_metrics: dict[str, int | float] = {}

    def _on_records(self, records: Iterable[Any]) -> bool:
        for record in records:
            if hasattr(record, "num_envs") and not hasattr(record, "lane"):
                num_envs = int(record.num_envs)
                rollout_steps = int(getattr(self.model, "n_steps", 1))
                self.reward_stats.consume(
                    getattr(record, "metrics", {}) or {},
                    reserve=num_envs * rollout_steps,
                )
                continue
            done_payload = self.done_metrics.consume(record)
            if done_payload:
                self.pending_metrics.update(done_payload)
            if self.success_metrics is not None:
                self.pending_metrics.update(self.success_metrics.consume(record))
        return True

    def _on_rollout_end(self) -> None:
        payload = self.pending_metrics
        payload.update(self.reward_stats.flush())
        self.pending_metrics = {}
        if not payload:
            return
        for key, value in payload.items():
            self.logger.record(key, value)


class RlabCallback(BaseCallback):
    """The sole SB3 callback; delegates lifecycle work to plain components."""

    def __init__(self, components: Sequence[CallbackHelper]) -> None:
        super().__init__()
        self.components = tuple(components)
        self._record_source: Any | None = None
        self._record_source_searched = False
        hook_names = (
            "_init_callback",
            "_on_training_start",
            "_on_rollout_start",
            "_on_rollout_end",
            "_on_training_end",
        )
        self._hooks = {
            hook: tuple(
                (component, method)
                for component in self.components
                if callable(method := getattr(component, hook, None))
            )
            for hook in hook_names
        }
        self._step_operations = tuple(
            (
                component,
                record_hook if callable(record_hook) else getattr(component, "_on_step", None),
                callable(record_hook),
            )
            for component in self.components
            if callable(record_hook := getattr(component, "_on_records", None))
            or callable(getattr(component, "_on_step", None))
        )

    def _bind(self, component: CallbackHelper) -> None:
        component.bind(self)

    def _call(self, hook: str) -> bool:
        for component, method in self._hooks[hook]:
            self._bind(component)
            if method() is False:
                return False
        return True

    def _call_step(self) -> bool:
        records: Iterable[Any] | None = None
        for component, method, consumes_records in self._step_operations:
            self._bind(component)
            if consumes_records:
                if records is None:
                    source = self._find_record_source()
                    if source is None:
                        raise RuntimeError("RlabCallback requires RlabVecEnv.drain_records()")
                    records = source.drain_records()
                result = method(records)
            else:
                result = method()
            if result is False:
                return False
        return True

    def _find_record_source(self) -> Any | None:
        if self._record_source_searched:
            return self._record_source
        self._record_source_searched = True
        seen: set[int] = set()
        current = getattr(self.model, "env", None)
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if callable(getattr(current, "drain_records", None)):
                self._record_source = current
                break
            current = getattr(current, "venv", None) or getattr(current, "env", None)
        return self._record_source

    def _init_callback(self) -> None:
        self._call("_init_callback")

    def _on_training_start(self) -> None:
        self._call("_on_training_start")

    def _on_rollout_start(self) -> None:
        self._call("_on_rollout_start")

    def _on_step(self) -> bool:
        return self._call_step()

    def _on_rollout_end(self) -> None:
        self._call("_on_rollout_end")

    def _on_training_end(self) -> None:
        self._call("_on_training_end")
