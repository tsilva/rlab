from __future__ import annotations

import json
import math
import re
import time
import uuid
from collections import deque
from numbers import Real
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
from rlab.metric_names import (
    GLOBAL_STEP,
    ROLLOUT_ADVANTAGE,
    ROLLOUT_ADVANTAGE_HIST,
    ROLLOUT_VALUE_PRED,
    ROLLOUT_VALUE_PRED_HIST,
    TIME_TIME_ELAPSED,
    THROUGHPUT_LOOP_FPS,
    THROUGHPUT_NATIVE_ENV_STEP_BATCH_FPS,
    THROUGHPUT_NATIVE_ENV_STEP_FPS,
    THROUGHPUT_NATIVE_ENV_STEP_FRACTION,
    THROUGHPUT_NATIVE_ENV_STEP_SECONDS,
    THROUGHPUT_ROLLOUT_FPS,
    TRAIN_DONE_ALL,
    TRAIN_DONE_MAX_STEPS,
    TRAIN_DONE_ROOT,
    TRAIN_DONE_UNCLASSIFIED,
    TRAIN_INFO_LEVEL_COMPLETE_RATE_MEAN_CURRENT,
    TRAIN_INFO_LEVEL_COMPLETE_RATE_MEAN_LAST,
    TRAIN_INFO_LEVEL_COMPLETE_RATE_MIN_CURRENT,
    TRAIN_INFO_LEVEL_COMPLETE_RATE_MIN_LAST,
    TRAIN_REWARD_COMPONENT_ROOT,
    TRAIN_REWARD_SHARE_ROOT,
    stat_metric,
    train_info_level_complete_attempts_metric,
    train_info_level_complete_count_metric,
    train_info_level_complete_current_rate_metric,
    train_info_level_complete_from_metric,
    train_info_level_complete_rate_metric,
    train_done_from_rate_metric,
    train_done_value_metric,
    train_done_reason_metric,
)
from rlab.metric_store import MetricStore, file_sha256


def task_metric_source(start_id: Any) -> Any:
    """Map readable Mario starts to the native coordinate used by existing metrics."""
    match = re.fullmatch(r"Level(\d+)-(\d+)", str(start_id))
    if match is None:
        return start_id
    return int(match.group(1)) - 1, int(match.group(2)) - 1


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
    ) -> None:
        super().__init__()
        self.args = args
        self.config = config
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.name_prefix = name_prefix
        self.metric_store = MetricStore(metric_store_path)

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
            sha256=file_sha256(final_path),
        )
        print(
            f"checkpoint ready: id={checkpoint_id} step={step} path={final_path}",
            flush=True,
        )
        return final_path


class ThroughputHelper(CallbackHelper):
    """Log rollout-only and full-loop instantaneous throughput."""

    def __init__(self, clock: Callable[[], float] | None = None):
        super().__init__()
        self.clock = clock or time.perf_counter
        self.rollout_start_time: float | None = None
        self.rollout_start_timesteps: int | None = None
        self.previous_rollout_start_time: float | None = None
        self.previous_rollout_start_timesteps: int | None = None
        self.pending_fps_instant: float | None = None
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
        self.native_step_stats_start = self._native_step_stats(getattr(self.model, "env", None))
        self.previous_rollout_start_time = now
        self.previous_rollout_start_timesteps = self.num_timesteps

    def _on_rollout_end(self) -> None:
        now = self.clock()
        if self.rollout_start_time is not None and self.rollout_start_timesteps is not None:
            elapsed = now - self.rollout_start_time
            steps = self.num_timesteps - self.rollout_start_timesteps
            if elapsed > 0 and steps > 0:
                self.logger.record(THROUGHPUT_ROLLOUT_FPS, steps / elapsed)
                self._record_native_step_throughput(
                    steps=steps,
                    rollout_elapsed=elapsed,
                )

        if self.pending_fps_instant is not None:
            self.logger.record(THROUGHPUT_LOOP_FPS, self.pending_fps_instant)
            self.pending_fps_instant = None

    def _on_step(self) -> bool:
        return True

    def _record_native_step_throughput(self, *, steps: int, rollout_elapsed: float) -> None:
        start = self.native_step_stats_start
        end = self._native_step_stats(getattr(self.model, "env", None))
        self.native_step_stats_start = None
        if start is None or end is None:
            return
        native_seconds = float(end.get("seconds_total", 0.0)) - float(
            start.get("seconds_total", 0.0)
        )
        native_calls = int(end.get("calls_total", 0)) - int(start.get("calls_total", 0))
        if native_seconds <= 0 or native_calls <= 0:
            return
        self.logger.record(THROUGHPUT_NATIVE_ENV_STEP_SECONDS, native_seconds)
        self.logger.record(THROUGHPUT_NATIVE_ENV_STEP_FPS, steps / native_seconds)
        self.logger.record(THROUGHPUT_NATIVE_ENV_STEP_BATCH_FPS, native_calls / native_seconds)
        if rollout_elapsed > 0:
            self.logger.record(
                THROUGHPUT_NATIVE_ENV_STEP_FRACTION,
                min(native_seconds / rollout_elapsed, 1.0),
            )


class TimeElapsedHelper(CallbackHelper):
    """Record elapsed SB3 learn-loop time for the rollout metric frame."""

    def __init__(self, wandb_run=None, clock: Callable[[], float] | None = None):
        super().__init__()
        self.wandb_run = wandb_run
        self.clock = clock or time.perf_counter
        self.started_at: float | None = None

    def _on_training_start(self) -> None:
        self.started_at = self.clock()

    def _on_rollout_end(self) -> None:
        elapsed = self.elapsed_seconds()
        if elapsed is None:
            return

        self.logger.record(TIME_TIME_ELAPSED, elapsed)

    def _on_step(self) -> bool:
        return True

    def elapsed_seconds(self) -> float | None:
        if self.started_at is None:
            return None
        elapsed = self.clock() - self.started_at
        return elapsed if elapsed >= 0 else None


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
        self.metric_store = MetricStore(metric_store_path, timeout=0.05)
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
        payload: dict[str, float] = {}
        for key, value in key_values.items():
            if isinstance(value, bool) or not isinstance(value, Real):
                continue
            numeric = float(value)
            if math.isfinite(numeric):
                payload[str(key)] = numeric
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
    """Log rollout-buffer value and advantage distributions."""

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
        self.metric_store = (
            MetricStore(metric_store_path, timeout=0.05) if metric_store_path else None
        )
        self.histogram_interval = max(int(histogram_interval), 1)
        self.rollout_count = 0

    def _on_rollout_end(self) -> None:
        self.rollout_count += 1
        rollout_buffer = getattr(self.model, "rollout_buffer", None)
        if rollout_buffer is None:
            return

        value_predictions = self._finite_values(getattr(rollout_buffer, "values", None))
        advantages = self._finite_values(getattr(rollout_buffer, "advantages", None))
        self._record_stats(ROLLOUT_VALUE_PRED, value_predictions)
        self._record_stats(ROLLOUT_ADVANTAGE, advantages)
        if self.rollout_count % self.histogram_interval == 0:
            self._log_histograms(value_predictions, advantages)

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
        self.logger.record(stat_metric(prefix, "mean"), float(np.mean(values)))
        self.logger.record(stat_metric(prefix, "std"), float(np.std(values)))
        self.logger.record(stat_metric(prefix, "min"), float(np.min(values)))
        self.logger.record(stat_metric(prefix, "max"), float(np.max(values)))
        self.logger.record(stat_metric(prefix, "abs_mean"), float(np.mean(np.abs(values))))

    def _log_histograms(self, value_predictions: np.ndarray, advantages: np.ndarray) -> None:
        if not self.log_histograms:
            return
        values: dict[str, list[float]] = {}
        if value_predictions.size > 0:
            values[ROLLOUT_VALUE_PRED_HIST] = value_predictions.tolist()
        if advantages.size > 0:
            values[ROLLOUT_ADVANTAGE_HIST] = advantages.tolist()
        if not values:
            return
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

    def flush(self, prefix: str) -> tuple[dict[str, float], float]:
        values = self.buffer[: self.size]
        values = values[np.isfinite(values)]
        self.reset()
        if values.size == 0:
            return {}, 0.0
        abs_sum = float(np.sum(np.abs(values)))
        return (
            {
                stat_metric(prefix, "mean"): float(np.mean(values)),
                stat_metric(prefix, "std"): float(np.std(values)),
                stat_metric(prefix, "min"): float(np.min(values)),
                stat_metric(prefix, "max"): float(np.max(values)),
                stat_metric(prefix, "abs_mean"): abs_sum / values.size,
                stat_metric(prefix, "nonzero_rate"): float(np.mean(values != 0.0)),
            },
            abs_sum,
        )


class _RewardStatsAccumulator:
    component_info_keys = {
        "shaped": "shaped_reward",
        "raw": "raw_reward",
        "native": "native_reward_component",
        "prog": "progress_component",
        "prog_x": "progress_reward_component",
        "score": "score_reward_component",
        "score_d": "score_delta",
        "done": "completion_reward_component",
        "death": "death_penalty_component",
        "time": "time_penalty_component",
    }
    reward_share_components = ("prog_x", "score", "death", "done", "time", "native")

    def __init__(self) -> None:
        self.components = {component: _BufferedStats() for component in self.component_info_keys}

    def consume(self, metrics: Mapping[str, Any], *, reserve: int) -> None:
        for component, info_key in self.component_info_keys.items():
            value = metrics.get(info_key)
            if value is not None:
                self.components[component].update(value, reserve=reserve)

    def flush(self) -> dict[str, float]:
        payload: dict[str, float] = {}
        abs_sums: dict[str, float] = {}
        has_reward_data = any(accumulator.size > 0 for accumulator in self.components.values())
        for component, accumulator in self.components.items():
            prefix = f"{TRAIN_REWARD_COMPONENT_ROOT}/{component}"
            metrics, abs_sum = accumulator.flush(prefix)
            payload.update(metrics)
            if component in self.reward_share_components:
                abs_sums[component] = abs_sum
        if has_reward_data:
            total_abs_sum = sum(abs_sums.values())
            payload.update(
                {
                    f"{TRAIN_REWARD_SHARE_ROOT}/{component}": (
                        abs_sum / total_abs_sum if total_abs_sum > 0.0 else 0.0
                    )
                    for component, abs_sum in abs_sums.items()
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
        self.reason_counts: dict[str, int] = {}
        self.detail_counts: dict[str, int] = {}
        self.detail_episode_windows: dict[str, deque[bool]] = {}

    def consume(self, record: Any) -> dict[str, int | float] | None:
        if not hasattr(record, "episode_return"):
            return None
        events = set(getattr(record, "events", ()))
        start_id = getattr(record, "start_id", None)
        metric_source = task_metric_source(start_id) if start_id is not None else None
        reason_payloads = {
            event: ({"prev": metric_source} if metric_source is not None else {})
            for event in sorted(events - {"timeout"})
        }
        if bool(getattr(record, "truncated", False)):
            reason_payloads["max_steps"] = {}
        if not reason_payloads:
            reason_payloads["unclassified"] = {}
        return self.record_done(reason_payloads, source_value=metric_source)

    def record_done(
        self,
        reason_payloads: dict[str, Any],
        *,
        source_value: Any | None = None,
    ) -> dict[str, int | float]:
        self.done_count += 1
        episode_detail_metrics: set[str] = set()
        for reason, payload in reason_payloads.items():
            self.reason_counts[reason] = self.reason_counts.get(reason, 0) + 1
            for metric in self.done_detail_metrics(reason, payload):
                self.detail_counts[metric] = self.detail_counts.get(metric, 0) + 1
                episode_detail_metrics.add(metric)
        self.record_detail_episode_windows(
            reason_payloads,
            episode_detail_metrics,
            source_value=source_value,
        )
        return self.record_metrics()

    @staticmethod
    def done_detail_metrics(reason: str, payload: Any) -> tuple[str, ...]:
        if not isinstance(payload, dict):
            return ()
        has_prev = "prev" in payload and payload["prev"] is not None
        if has_prev:
            return (train_done_value_metric(reason, "from", payload["prev"]),)
        return ()

    @staticmethod
    def done_ep_window_rate_metric(metric: str) -> str:
        return f"{metric}/ep_window/rate"

    def record_detail_episode_windows(
        self,
        reason_payloads: dict[str, Any],
        fired_detail_metrics: set[str],
        *,
        source_value: Any | None,
    ) -> None:
        source_reasons = set(self.event_names)
        source_reasons.update(reason_payloads)
        for reason in sorted(source_reasons):
            payload = reason_payloads.get(reason)
            event_source = (
                payload.get("prev")
                if isinstance(payload, Mapping) and payload.get("prev") is not None
                else source_value
            )
            if event_source is None:
                continue
            metric = train_done_value_metric(reason, "from", event_source)
            window = self.detail_episode_windows.setdefault(
                metric,
                deque(maxlen=self.ep_window_size),
            )
            window.append(metric in fired_detail_metrics)

    def record_ep_window_rates(self) -> dict[str, float]:
        detail_rates = {
            metric: sum(window) / len(window)
            for metric, window in sorted(self.detail_episode_windows.items())
            if len(window) >= self.ep_window_size
        }
        payload = {
            self.done_ep_window_rate_metric(metric): rate for metric, rate in detail_rates.items()
        }

        rates_by_reason: dict[str, list[float]] = {}
        for metric, rate in detail_rates.items():
            reason = self.done_detail_metric_reason(metric)
            if reason is not None:
                rates_by_reason.setdefault(reason, []).append(rate)

        for reason, rates in sorted(rates_by_reason.items()):
            if len(rates) < 2:
                continue
            payload[train_done_from_rate_metric(reason, "min")] = min(rates)
            payload[train_done_from_rate_metric(reason, "mean")] = float(np.mean(rates))
        return payload

    @staticmethod
    def done_detail_metric_reason(metric: str) -> str | None:
        prefix = TRAIN_DONE_ROOT
        marker = "/from/"
        if not metric.startswith(prefix) or marker not in metric:
            return None
        reason, _value = metric.removeprefix(prefix).split(marker, 1)
        return reason or None

    def record_metrics(self) -> dict[str, int | float]:
        payload: dict[str, int | float] = {TRAIN_DONE_ALL: self.done_count}
        payload.update(
            {
                train_done_reason_metric(reason): count
                for reason, count in self.reason_counts.items()
            },
        )
        payload.update(self.detail_counts)
        payload.update(self.record_ep_window_rates())
        payload.setdefault(TRAIN_DONE_MAX_STEPS, self.reason_counts.get("max_steps", 0))
        payload.setdefault(TRAIN_DONE_UNCLASSIFIED, self.reason_counts.get("unclassified", 0))
        return payload


class _LevelCompletionMetricsReducer:
    ep_window_size = 100
    completion_source_event = "level_change"

    def __init__(self) -> None:
        self.complete_counts: dict[str, int] = {}
        self.attempt_windows: dict[str, deque[bool]] = {}
        self.latest_current_rates: dict[str, float] = {}
        self.latest_rates: dict[str, float] = {}
        self.current_sources: list[Any | None] = []

    def consume(self, record: Any) -> dict[str, int | float]:
        payload: dict[str, int | float] = {}
        events = set(getattr(record, "events", ()))
        lane = int(getattr(record, "lane", 0))
        self.ensure_slots(lane + 1)
        start_id = getattr(record, "start_id", None)
        if self.current_sources[lane] is None and start_id is not None:
            self.current_sources[lane] = task_metric_source(start_id)

        if not hasattr(record, "episode_return"):
            transition = (getattr(record, "transitions", {}) or {}).get(
                self.completion_source_event
            )
            if self.completion_source_event not in events or transition is None:
                return payload
            source, target = transition
            source = tuple(np.asarray(source).tolist())
            target = tuple(np.asarray(target).tolist())
            if "life_loss" not in events:
                payload.update(self.record_attempt(source, completed=True))
                self.current_sources[lane] = target
            return payload

        outcome = getattr(record, "outcome", None)
        outcome_name = str(getattr(outcome, "name", outcome)).lower()
        if outcome_name == "success" and "life_loss" not in events:
            self.current_sources[lane] = None
            return payload
        source = self.current_sources[lane]
        if source is None:
            metrics = getattr(record, "metrics", {}) or {}
            if "level_hi" in metrics and "level_lo" in metrics:
                source = (int(metrics["level_hi"]), int(metrics["level_lo"]))
        if source is None:
            return payload
        payload.update(self.record_attempt(source, completed=False))
        self.current_sources[lane] = None
        return payload

    def ensure_slots(self, count: int) -> None:
        while len(self.current_sources) < count:
            self.current_sources.append(None)

    def record_attempt(self, source: Any, *, completed: bool) -> dict[str, int | float]:
        metric = train_info_level_complete_from_metric(source)
        count_metric = train_info_level_complete_count_metric(source)
        attempts_metric = train_info_level_complete_attempts_metric(source)
        current_rate_metric = train_info_level_complete_current_rate_metric(source)
        window = self.attempt_windows.setdefault(metric, deque(maxlen=self.ep_window_size))
        window.append(completed)
        if completed:
            self.complete_counts[count_metric] = self.complete_counts.get(count_metric, 0) + 1

        current_rate = sum(window) / len(window)
        self.latest_current_rates[current_rate_metric] = current_rate
        payload: dict[str, int | float] = {
            count_metric: self.complete_counts.get(count_metric, 0),
            attempts_metric: len(window),
            current_rate_metric: current_rate,
            TRAIN_INFO_LEVEL_COMPLETE_RATE_MIN_CURRENT: min(self.latest_current_rates.values()),
            TRAIN_INFO_LEVEL_COMPLETE_RATE_MEAN_CURRENT: sum(self.latest_current_rates.values())
            / len(self.latest_current_rates),
        }
        if len(window) >= self.ep_window_size:
            rate_metric = train_info_level_complete_rate_metric(source)
            rate = sum(window) / len(window)
            payload[rate_metric] = rate
            self.latest_rates[rate_metric] = rate
            payload[TRAIN_INFO_LEVEL_COMPLETE_RATE_MIN_LAST] = min(self.latest_rates.values())
            payload[TRAIN_INFO_LEVEL_COMPLETE_RATE_MEAN_LAST] = sum(
                self.latest_rates.values()
            ) / len(self.latest_rates)
        return payload


class RuntimeMetricsHelper(CallbackHelper):
    """Reduce runtime records and publish one scalar payload per rollout."""

    def __init__(self, *, wandb_run=None, event_names: Sequence[str] = ()) -> None:
        super().__init__()
        self.wandb_run = wandb_run
        self.reward_stats = _RewardStatsAccumulator()
        self.done_metrics = _DoneMetricsReducer(event_names=event_names)
        self.completion_metrics = (
            _LevelCompletionMetricsReducer() if "level_change" in event_names else None
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
            if self.completion_metrics is not None:
                self.pending_metrics.update(self.completion_metrics.consume(record))
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
