from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ProjectionBenchmarkResult:
    architecture: str
    producer_steps_per_second: float
    canonical_events_per_second: float
    raw_normalized_frames_per_second: float
    selected_rows_per_second: float
    publisher_rows_per_second: float
    maximum_backlog_rows: int
    maximum_backlog_age_seconds: float
    maximum_wandb_freshness_seconds: float
    terminal_drain_seconds: float


def _simulate(
    *,
    architecture: str,
    duration_seconds: int,
    learner_steps_per_second: int,
    batch_interval_seconds: int,
    frames_per_batch: int,
    sdk_enqueue_seconds: float,
    verification_seconds_per_row: float,
) -> ProjectionBenchmarkResult:
    batch_count = max(1, duration_seconds // batch_interval_seconds)
    finish_at = 0.0
    maximum_age = 0.0
    maximum_backlog = 0
    published_rows = 0
    for batch_index in range(batch_count):
        arrival = float(batch_index * batch_interval_seconds)
        if architecture == "legacy":
            selected = frames_per_batch
            verification_rows = (batch_index + 1) * frames_per_batch
        else:
            selected = 1
            verification_rows = min(batch_index + 1, 256)
        service_seconds = (
            selected * sdk_enqueue_seconds + verification_rows * verification_seconds_per_row
        )
        start_at = max(arrival, finish_at)
        finish_at = start_at + service_seconds
        age = finish_at - arrival
        maximum_age = max(maximum_age, age)
        maximum_backlog = max(
            maximum_backlog,
            int(max(0.0, finish_at - arrival) / batch_interval_seconds + 0.999) * selected,
        )
        published_rows += selected
    production_ended_at = float((batch_count - 1) * batch_interval_seconds)
    wall_seconds = max(finish_at, float(duration_seconds), 1.0)
    return ProjectionBenchmarkResult(
        architecture=architecture,
        producer_steps_per_second=float(learner_steps_per_second),
        canonical_events_per_second=batch_count / max(float(duration_seconds), 1.0),
        raw_normalized_frames_per_second=(
            batch_count * frames_per_batch / max(float(duration_seconds), 1.0)
        ),
        selected_rows_per_second=(
            batch_count
            * (frames_per_batch if architecture == "legacy" else 1)
            / max(float(duration_seconds), 1.0)
        ),
        publisher_rows_per_second=published_rows / wall_seconds,
        maximum_backlog_rows=maximum_backlog,
        maximum_backlog_age_seconds=maximum_age,
        maximum_wandb_freshness_seconds=maximum_age,
        terminal_drain_seconds=max(0.0, finish_at - production_ended_at),
    )


def benchmark_pair(
    *,
    duration_seconds: int = 1_800,
    learner_steps_per_second: int = 20_000,
    batch_interval_seconds: int = 30,
    frames_per_batch: int = 100,
    delayed_sdk_seconds: float = 0.35,
) -> dict[str, ProjectionBenchmarkResult]:
    """Reproduce the old overload and bounded-v2 behavior with virtual time."""

    return {
        "legacy": _simulate(
            architecture="legacy",
            duration_seconds=duration_seconds,
            learner_steps_per_second=learner_steps_per_second,
            batch_interval_seconds=batch_interval_seconds,
            frames_per_batch=frames_per_batch,
            sdk_enqueue_seconds=delayed_sdk_seconds,
            verification_seconds_per_row=0.0005,
        ),
        "bounded_v2": _simulate(
            architecture="bounded_v2",
            duration_seconds=duration_seconds,
            learner_steps_per_second=learner_steps_per_second,
            batch_interval_seconds=batch_interval_seconds,
            frames_per_batch=frames_per_batch,
            sdk_enqueue_seconds=0.2,
            verification_seconds_per_row=0.0005,
        ),
    }


def realtime_enqueue_measurement(*, rows: int = 20, delay_seconds: float = 0.001) -> dict:
    """Non-gating wall-clock record for the delayed fake asynchronous SDK."""

    started = time.perf_counter()
    for _ in range(max(1, int(rows))):
        time.sleep(max(0.0, float(delay_seconds)))
    elapsed = time.perf_counter() - started
    return {
        "rows": max(1, int(rows)),
        "elapsed_seconds": elapsed,
        "rows_per_second": max(1, int(rows)) / max(elapsed, 1e-9),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark W&B projection backpressure.")
    parser.add_argument("--duration-seconds", type=int, default=1_800)
    parser.add_argument("--real-time", action="store_true")
    args = parser.parse_args(argv)
    result = {
        name: asdict(value)
        for name, value in benchmark_pair(
            duration_seconds=max(30, int(args.duration_seconds))
        ).items()
    }
    if args.real_time:
        result["real_time"] = realtime_enqueue_measurement()
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
