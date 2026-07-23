from __future__ import annotations

import concurrent.futures
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RecoveryLimits:
    archive_workers: int = 4
    wandb_workers: int = 4
    artifact_workers: int = 2
    global_inflight: int = 32
    per_run_inflight: int = 2
    poison_failures: int = 5
    watchdog_seconds: float = 120.0


@dataclass
class RecoveryWatermark:
    submitted: int = 0
    completed: int = 0
    failed: int = 0
    poisoned: int = 0
    last_progress_monotonic: float = field(default_factory=time.monotonic)


class FairRunQueue:
    """Bounded round-robin queue; one poison stream cannot starve other runs."""

    def __init__(self, *, per_run_limit: int) -> None:
        self.per_run_limit = max(1, int(per_run_limit))
        self._queues: dict[int, deque[Any]] = {}
        self._runs: deque[int] = deque()

    def extend(self, run_id: int, items: Iterable[Any]) -> None:
        queue = self._queues.setdefault(int(run_id), deque())
        was_empty = not queue
        for item in items:
            queue.append(item)
        if was_empty and queue and int(run_id) not in self._runs:
            self._runs.append(int(run_id))

    def pop(self) -> tuple[int, Any] | None:
        if not self._runs:
            return None
        run_id = self._runs.popleft()
        queue = self._queues[run_id]
        item = queue.popleft()
        if queue:
            self._runs.append(run_id)
        else:
            del self._queues[run_id]
        return run_id, item


class TelemetryRecoveryExecutors:
    """Separate bounded executor pools for independent remote failure domains."""

    def __init__(self, limits: RecoveryLimits = RecoveryLimits()) -> None:
        self.limits = limits
        self.archive = concurrent.futures.ThreadPoolExecutor(
            max_workers=limits.archive_workers,
            thread_name_prefix="telemetry-archive",
        )
        self.wandb = concurrent.futures.ThreadPoolExecutor(
            max_workers=limits.wandb_workers,
            thread_name_prefix="telemetry-wandb",
        )
        self.artifact = concurrent.futures.ThreadPoolExecutor(
            max_workers=limits.artifact_workers,
            thread_name_prefix="telemetry-artifact",
        )
        self.watermarks = {
            "archive": RecoveryWatermark(),
            "wandb": RecoveryWatermark(),
            "artifact": RecoveryWatermark(),
        }
        self._global = threading.BoundedSemaphore(limits.global_inflight)
        self._run_locks: dict[int, threading.BoundedSemaphore] = {}
        self._failures: dict[tuple[str, int, str], int] = {}
        self._guard = threading.Lock()

    def submit(
        self,
        lane: str,
        *,
        run_id: int,
        poison_key: str,
        work: Callable[[], Any],
    ) -> concurrent.futures.Future[Any]:
        if lane not in self.watermarks:
            raise ValueError(f"unknown recovery executor lane: {lane}")
        key = (lane, int(run_id), str(poison_key))
        if self._failures.get(key, 0) >= self.limits.poison_failures:
            self.watermarks[lane].poisoned += 1
            raise RuntimeError(f"recovery item is poison-isolated: {key}")
        with self._guard:
            run_limit = self._run_locks.setdefault(
                int(run_id),
                threading.BoundedSemaphore(self.limits.per_run_inflight),
            )
        if not self._global.acquire(blocking=False) or not run_limit.acquire(blocking=False):
            try:
                self._global.release()
            except ValueError:
                pass
            raise RuntimeError("recovery executor budget is full")
        watermark = self.watermarks[lane]
        watermark.submitted += 1

        def wrapped() -> Any:
            try:
                result = work()
            except Exception:
                with self._guard:
                    self._failures[key] = self._failures.get(key, 0) + 1
                    watermark.failed += 1
                    watermark.last_progress_monotonic = time.monotonic()
                raise
            else:
                with self._guard:
                    self._failures.pop(key, None)
                    watermark.completed += 1
                    watermark.last_progress_monotonic = time.monotonic()
                return result
            finally:
                run_limit.release()
                self._global.release()

        executor = getattr(self, lane)
        return executor.submit(wrapped)

    def health(self) -> dict[str, Any]:
        now = time.monotonic()
        return {
            lane: {
                **watermark.__dict__,
                "watchdog_stalled": (
                    watermark.submitted > watermark.completed + watermark.failed
                    and now - watermark.last_progress_monotonic
                    > self.limits.watchdog_seconds
                ),
            }
            for lane, watermark in self.watermarks.items()
        }

    def close(self) -> None:
        for executor in (self.archive, self.wandb, self.artifact):
            executor.shutdown(wait=True, cancel_futures=False)
