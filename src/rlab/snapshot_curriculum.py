from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


SNAPSHOT_CURRICULUM_SEMANTIC_ID = "snapshot_curriculum_v1"
_DEFAULTS: dict[str, Any] = {
    "representatives_per_cell": 4,
    "max_snapshots": 1024,
    "feedback_ema_alpha": 0.10,
    "staleness_weight": 0.30,
    "rank_temperature": 1.0,
    "max_cell_probability": 0.25,
}
_TOP_LEVEL_KEYS = frozenset(
    {
        "cell",
        "snapshot_share",
        "priority_metric",
        "semantic_id",
        "resolved_snapshot_lanes",
        *_DEFAULTS,
    }
)
_CELL_KEYS = frozenset({"signal", "bucket_size"})


def snapshot_lane_count(snapshot_share: float, n_envs: int) -> int:
    return int(math.floor(float(snapshot_share) * int(n_envs) + 0.5))


def normalize_snapshot_curriculum_config(
    value: Any,
    *,
    label: str = "snapshot_curriculum",
    n_envs: int | None = None,
) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object or null")
    unexpected = sorted(set(value) - _TOP_LEVEL_KEYS)
    if unexpected:
        raise ValueError(f"{label} has unexpected fields: {unexpected}")
    semantic_id = value.get("semantic_id")
    if semantic_id is not None and semantic_id != SNAPSHOT_CURRICULUM_SEMANTIC_ID:
        raise ValueError(f"{label}.semantic_id must be {SNAPSHOT_CURRICULUM_SEMANTIC_ID!r}")
    cell = value.get("cell")
    if not isinstance(cell, Mapping):
        raise ValueError(f"{label}.cell must be an object")
    unexpected_cell = sorted(set(cell) - _CELL_KEYS)
    if unexpected_cell:
        raise ValueError(f"{label}.cell has unexpected fields: {unexpected_cell}")
    signal = str(cell.get("signal") or "").strip()
    if not signal:
        raise ValueError(f"{label}.cell.signal must be a non-empty string")
    bucket_size = cell.get("bucket_size")
    if (
        not isinstance(bucket_size, int | float)
        or isinstance(bucket_size, bool)
        or not math.isfinite(float(bucket_size))
        or float(bucket_size) <= 0.0
    ):
        raise ValueError(f"{label}.cell.bucket_size must be a positive finite number")
    snapshot_share = value.get("snapshot_share")
    if (
        not isinstance(snapshot_share, int | float)
        or isinstance(snapshot_share, bool)
        or not math.isfinite(float(snapshot_share))
        or not 0.0 < float(snapshot_share) < 1.0
    ):
        raise ValueError(f"{label}.snapshot_share must be a finite number in (0, 1)")
    priority_metric = str(value.get("priority_metric") or "").strip()
    if not priority_metric:
        raise ValueError(f"{label}.priority_metric must be a non-empty string")
    normalized = {
        "semantic_id": SNAPSHOT_CURRICULUM_SEMANTIC_ID,
        "cell": {"signal": signal, "bucket_size": float(bucket_size)},
        "snapshot_share": float(snapshot_share),
        "priority_metric": priority_metric,
        **_DEFAULTS,
    }
    normalized.update({key: value[key] for key in _DEFAULTS if key in value})
    integer_fields = ("representatives_per_cell", "max_snapshots")
    for key in integer_fields:
        item = normalized[key]
        if not isinstance(item, int) or isinstance(item, bool) or item < 1:
            raise ValueError(f"{label}.{key} must be a positive integer")
        normalized[key] = int(item)
    if normalized["max_snapshots"] < normalized["representatives_per_cell"]:
        raise ValueError(f"{label}.max_snapshots must be >= representatives_per_cell")
    if normalized["max_snapshots"] > 16384:
        raise ValueError(f"{label}.max_snapshots must be <= 16384")
    ranges = {
        "feedback_ema_alpha": (0.0, 1.0, False),
        "staleness_weight": (0.0, 1.0, True),
        "rank_temperature": (0.0, math.inf, False),
        "max_cell_probability": (0.0, 1.0, False),
    }
    for key, (minimum, maximum, include_minimum) in ranges.items():
        item = normalized[key]
        if not isinstance(item, int | float) or isinstance(item, bool):
            raise ValueError(f"{label}.{key} must be a number")
        numeric = float(item)
        lower_ok = numeric >= minimum if include_minimum else numeric > minimum
        if not math.isfinite(numeric) or not lower_ok or numeric > maximum:
            left = "[" if include_minimum else "("
            right = "]" if math.isfinite(maximum) else ")"
            upper = f"{maximum:g}" if math.isfinite(maximum) else "inf"
            raise ValueError(f"{label}.{key} must be in {left}{minimum:g}, {upper}{right}")
        normalized[key] = numeric
    if n_envs is not None:
        if not isinstance(n_envs, int) or isinstance(n_envs, bool) or n_envs < 2:
            raise ValueError(f"{label} requires n_envs >= 2")
        lanes = snapshot_lane_count(normalized["snapshot_share"], n_envs)
        if lanes < 1 or lanes >= n_envs:
            raise ValueError(
                f"{label}.snapshot_share resolves to {lanes} snapshot lanes for n_envs={n_envs}; "
                "it must resolve to at least one snapshot lane and one target lane"
            )
        normalized["resolved_snapshot_lanes"] = lanes
    return normalized


def validate_snapshot_curriculum_runtime_contract(
    common_config: Mapping[str, Any],
    *,
    backend_id: str,
    supported_priority_metrics: Sequence[str],
) -> None:
    value = common_config.get("snapshot_curriculum")
    if value is None:
        return
    n_envs = int(common_config.get("n_envs", 0))
    normalized = normalize_snapshot_curriculum_config(value, n_envs=n_envs)
    assert normalized is not None
    supported = frozenset(str(metric).strip() for metric in supported_priority_metrics)
    if normalized["priority_metric"] not in supported:
        raise ValueError(
            f"training backend {backend_id!r} does not provide snapshot priority "
            f"{normalized['priority_metric']!r}; supported priorities: {sorted(supported)}"
        )
    env_args = common_config.get("env_args")
    if isinstance(env_args, Mapping) and (
        env_args.get("snapshot_bank_uri") or env_args.get("snapshot_bank_sha256")
    ):
        raise ValueError("snapshot_curriculum_v1 cannot be combined with a persisted snapshot bank")
    task = common_config.get("task")
    signals = task.get("signals") if isinstance(task, Mapping) else None
    signal = str(normalized["cell"]["signal"])
    if not isinstance(signals, Mapping) or signal not in signals:
        raise ValueError(f"snapshot_curriculum_v1 requires task.signals.{signal} to be declared")


@dataclass(frozen=True)
class SnapshotCurriculumConfig:
    signal: str
    bucket_size: float
    snapshot_share: float
    priority_metric: str
    representatives_per_cell: int
    max_snapshots: int
    feedback_ema_alpha: float
    staleness_weight: float
    rank_temperature: float
    max_cell_probability: float
    resolved_snapshot_lanes: int
    semantic_id: str = SNAPSHOT_CURRICULUM_SEMANTIC_ID

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, n_envs: int) -> "SnapshotCurriculumConfig":
        normalized = normalize_snapshot_curriculum_config(value, n_envs=n_envs)
        assert normalized is not None
        return cls(
            signal=str(normalized["cell"]["signal"]),
            bucket_size=float(normalized["cell"]["bucket_size"]),
            snapshot_share=float(normalized["snapshot_share"]),
            priority_metric=str(normalized["priority_metric"]),
            representatives_per_cell=int(normalized["representatives_per_cell"]),
            max_snapshots=int(normalized["max_snapshots"]),
            feedback_ema_alpha=float(normalized["feedback_ema_alpha"]),
            staleness_weight=float(normalized["staleness_weight"]),
            rank_temperature=float(normalized["rank_temperature"]),
            max_cell_probability=float(normalized["max_cell_probability"]),
            resolved_snapshot_lanes=int(normalized["resolved_snapshot_lanes"]),
        )


@dataclass(frozen=True)
class SnapshotSelection:
    cell_id: str
    handle: Any
    generation: int


@dataclass
class _Cell:
    cell_id: str
    admission_index: int
    representatives: list[Any] = field(default_factory=list)
    seen: int = 0
    feedback_score: float | None = None
    last_sample_rollout: int = 0
    cold_dispatched: bool = False
    active_count: int = 0
    pending_feedback: int = 0


def _counter_uint64(*parts: object) -> int:
    encoded = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(encoded, digest_size=8).digest(), "little")


class SnapshotCurriculum:
    """Bounded, provider-handle archive with deterministic cell-level sampling."""

    def __init__(
        self,
        config: SnapshotCurriculumConfig,
        *,
        n_envs: int,
        run_seed: int,
        global_lane_ids: Sequence[int],
    ) -> None:
        self.config = config
        self.n_envs = int(n_envs)
        self.run_seed = int(run_seed)
        self.global_lane_ids = tuple(int(value) for value in global_lane_ids)
        self.generation = 1
        self.completed_rollout = 0
        self.activated = False
        self.activation_scheduled = False
        self._cells: dict[str, _Cell] = {}
        self._admission_counter = 0
        self._probabilities: dict[str, float] = {}
        self._sampled_this_rollout: set[str] = set()
        self._metrics: dict[str, float] = {}
        self.begin_rollout()

    @property
    def snapshot_lane_mask(self) -> np.ndarray:
        mask = np.zeros(self.n_envs, dtype=np.bool_)
        mask[: self.config.resolved_snapshot_lanes] = True
        return mask

    @property
    def cell_count(self) -> int:
        return len(self._cells)

    @property
    def snapshot_count(self) -> int:
        return sum(len(cell.representatives) for cell in self._cells.values())

    @property
    def ready(self) -> bool:
        return self.snapshot_count > 0

    def cell_id(self, value: Any) -> str:
        if isinstance(value, bool) or not isinstance(value, int | float | np.number):
            raise ValueError(f"snapshot curriculum signal {self.config.signal!r} must be numeric")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"snapshot curriculum signal {self.config.signal!r} must be finite")
        quotient = math.floor(numeric / self.config.bucket_size)
        if quotient < np.iinfo(np.int64).min or quotient > np.iinfo(np.int64).max:
            raise ValueError("snapshot curriculum cell index exceeds signed int64")
        return f"{self.config.signal}:{quotient}"

    def begin_rollout(self) -> None:
        self._metrics = {
            "admission_candidate_count": 0.0,
            "admission_accepted_count": 0.0,
            "evicted_count": 0.0,
            "capture_call_count": 0.0,
            "snapshot_reset_count": 0.0,
            "forced_boundary_count": 0.0,
            "feedback_trajectory_count": 0.0,
            "curriculum_transition_count": 0.0,
            "transition_count": 0.0,
            "capture_seconds": 0.0,
            "reset_seconds": 0.0,
        }
        self._rebuild_probabilities()

    def note_transition_batch(self, curriculum_count: int) -> None:
        self._metrics["transition_count"] += float(self.n_envs)
        self._metrics["curriculum_transition_count"] += float(curriculum_count)

    def note_candidates(self, count: int) -> None:
        self._metrics["admission_candidate_count"] += float(count)

    def note_capture(self, seconds: float) -> None:
        self._metrics["capture_call_count"] += 1.0
        self._metrics["capture_seconds"] += float(seconds)

    def note_reset(self, count: int, seconds: float, *, forced_boundaries: int = 0) -> None:
        self._metrics["snapshot_reset_count"] += float(count)
        self._metrics["forced_boundary_count"] += float(forced_boundaries)
        self._metrics["reset_seconds"] += float(seconds)

    def _representative_index(self, cell_id: str, seen: int) -> int:
        return (
            _counter_uint64(
                self.run_seed,
                self.generation,
                "reservoir",
                cell_id,
                seen,
            )
            % seen
        )

    def _evictable_cell(self, *, excluding: str | None = None) -> _Cell | None:
        candidates = [
            cell
            for cell in self._cells.values()
            if cell.cell_id != excluding
            and cell.feedback_score is not None
            and cell.active_count == 0
            and cell.pending_feedback == 0
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda cell: (self._probabilities.get(cell.cell_id, 0.0), cell.cell_id),
        )

    def _make_room(self, *, excluding: str | None = None) -> bool:
        if self.snapshot_count < self.config.max_snapshots:
            return True
        evicted = self._evictable_cell(excluding=excluding)
        if evicted is None:
            return False
        del self._cells[evicted.cell_id]
        self._probabilities.pop(evicted.cell_id, None)
        self._metrics["evicted_count"] += 1.0
        return self.snapshot_count < self.config.max_snapshots

    def admit(self, cell_id: str, handle: Any) -> bool:
        if handle is None:
            raise ValueError("provider returned no snapshot for an admitted lane")
        cell = self._cells.get(cell_id)
        if cell is None:
            if not self._make_room():
                return False
            self._admission_counter += 1
            cell = _Cell(cell_id=cell_id, admission_index=self._admission_counter)
            self._cells[cell_id] = cell
        cell.seen += 1
        if len(cell.representatives) < self.config.representatives_per_cell:
            if not self._make_room(excluding=cell_id):
                return False
            cell.representatives.append(handle)
            self._metrics["admission_accepted_count"] += 1.0
            self._rebuild_probabilities()
            return True
        replacement = self._representative_index(cell_id, cell.seen)
        if replacement >= self.config.representatives_per_cell:
            return False
        cell.representatives[int(replacement)] = handle
        self._metrics["admission_accepted_count"] += 1.0
        return True

    def schedule_activation(self) -> bool:
        if self.activated or self.activation_scheduled or not self.ready:
            return False
        self.activation_scheduled = True
        return True

    def activate(self) -> None:
        if not self.ready:
            raise RuntimeError("snapshot curriculum cannot activate with an empty archive")
        self.activated = True
        self.activation_scheduled = False

    def _draw_index(self, size: int, *, lane: int, episode_index: int, domain: str) -> int:
        if size <= 0:
            raise RuntimeError("cannot sample an empty snapshot set")
        value = _counter_uint64(
            self.run_seed,
            self.generation,
            self.completed_rollout,
            self.global_lane_ids[lane],
            episode_index,
            domain,
        )
        return int(value % size)

    def _draw_probability_cell(self, *, lane: int, episode_index: int) -> _Cell:
        ordered = sorted(self._probabilities)
        if not ordered:
            ordered = sorted(self._cells)
            return self._cells[
                ordered[
                    self._draw_index(
                        len(ordered), lane=lane, episode_index=episode_index, domain="unscored"
                    )
                ]
            ]
        raw = _counter_uint64(
            self.run_seed,
            self.generation,
            self.completed_rollout,
            self.global_lane_ids[lane],
            episode_index,
            "cell",
        )
        point = (raw + 0.5) / float(2**64)
        cumulative = 0.0
        for cell_id in ordered:
            cumulative += self._probabilities[cell_id]
            if point <= cumulative:
                return self._cells[cell_id]
        return self._cells[ordered[-1]]

    def sample(self, *, lane: int, episode_index: int) -> SnapshotSelection:
        cold = sorted(
            (cell for cell in self._cells.values() if not cell.cold_dispatched),
            key=lambda cell: (cell.admission_index, cell.cell_id),
        )
        if cold:
            cell = cold[0]
            cell.cold_dispatched = True
        else:
            cell = self._draw_probability_cell(lane=lane, episode_index=episode_index)
        representative_index = self._draw_index(
            len(cell.representatives),
            lane=lane,
            episode_index=episode_index,
            domain=f"representative:{cell.cell_id}",
        )
        self._sampled_this_rollout.add(cell.cell_id)
        cell.active_count += 1
        return SnapshotSelection(
            cell_id=cell.cell_id,
            handle=cell.representatives[representative_index],
            generation=self.generation,
        )

    def close_episode(self, cell_id: str) -> None:
        cell = self._cells.get(cell_id)
        if cell is None:
            return
        if cell.active_count <= 0:
            raise RuntimeError(f"snapshot cell {cell_id!r} has no active episode to close")
        cell.active_count -= 1
        cell.pending_feedback += 1

    def submit_feedback(self, cell_id: str, value_error: float) -> None:
        cell = self._cells.get(cell_id)
        if cell is None:
            return
        if cell.pending_feedback <= 0:
            raise RuntimeError(f"snapshot cell {cell_id!r} has no pending feedback")
        value = float(value_error)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("snapshot value_error feedback must be finite and non-negative")
        cell.pending_feedback -= 1
        alpha = self.config.feedback_ema_alpha
        cell.feedback_score = (
            value
            if cell.feedback_score is None
            else (1.0 - alpha) * cell.feedback_score + alpha * value
        )
        self._metrics["feedback_trajectory_count"] += 1.0
        self._rebuild_probabilities()

    @staticmethod
    def _rank_weights(
        values: Mapping[str, float], *, largest_first: bool, temperature: float
    ) -> dict[str, float]:
        ordered = sorted(
            values,
            key=lambda cell_id: (
                -values[cell_id] if largest_first else values[cell_id],
                cell_id,
            ),
        )
        weights = {
            cell_id: float((rank + 1) ** (-1.0 / temperature))
            for rank, cell_id in enumerate(ordered)
        }
        total = sum(weights.values())
        return {cell_id: weight / total for cell_id, weight in weights.items()}

    @staticmethod
    def _cap_probabilities(probabilities: Mapping[str, float], cap: float) -> dict[str, float]:
        if not probabilities:
            return {}
        result = {key: float(value) for key, value in probabilities.items()}
        active = set(result)
        fixed_mass = 0.0
        original = dict(result)
        while active:
            active_weight = sum(original[key] for key in active)
            if active_weight <= 0.0:
                share = (1.0 - fixed_mass) / len(active)
                for key in active:
                    result[key] = share
                break
            changed = False
            for key in sorted(tuple(active)):
                projected = (1.0 - fixed_mass) * original[key] / active_weight
                if projected > cap:
                    result[key] = cap
                    fixed_mass += cap
                    active.remove(key)
                    changed = True
            if not changed:
                active_weight = sum(original[key] for key in active)
                for key in active:
                    result[key] = (1.0 - fixed_mass) * original[key] / active_weight
                break
        residual = 1.0 - sum(result.values())
        if abs(residual) > 1e-15:
            for key in sorted(result):
                candidate = result[key] + residual
                if 0.0 <= candidate <= cap + 1e-12:
                    result[key] = candidate
                    break
        return result

    def _rebuild_probabilities(self) -> None:
        scored = {
            cell.cell_id: float(cell.feedback_score)
            for cell in self._cells.values()
            if cell.feedback_score is not None and cell.representatives
        }
        if not scored:
            self._probabilities = {}
            return
        score_weights = self._rank_weights(
            scored,
            largest_first=True,
            temperature=self.config.rank_temperature,
        )
        ages = {
            cell_id: float(self.completed_rollout - self._cells[cell_id].last_sample_rollout)
            for cell_id in scored
        }
        stale_weights = self._rank_weights(
            ages,
            largest_first=True,
            temperature=self.config.rank_temperature,
        )
        rho = self.config.staleness_weight
        mixed = {
            cell_id: (1.0 - rho) * score_weights[cell_id] + rho * stale_weights[cell_id]
            for cell_id in scored
        }
        effective_cap = max(self.config.max_cell_probability, 1.0 / len(mixed))
        self._probabilities = self._cap_probabilities(mixed, effective_cap)

    def complete_rollout(self) -> dict[str, float]:
        for cell_id in self._sampled_this_rollout:
            cell = self._cells.get(cell_id)
            if cell is not None:
                cell.last_sample_rollout = self.completed_rollout
        self._sampled_this_rollout.clear()
        transition_count = self._metrics["transition_count"]
        probabilities = tuple(self._probabilities.values())
        payload = {
            **self._metrics,
            "archive_cell_count": float(self.cell_count),
            "archive_snapshot_count": float(self.snapshot_count),
            "transition_share": (
                self._metrics["curriculum_transition_count"] / transition_count
                if transition_count > 0.0
                else 0.0
            ),
            "sampling_probability_max": max(probabilities, default=0.0),
            "sampling_effective_cell_count": (
                1.0 / sum(value * value for value in probabilities) if probabilities else 0.0
            ),
        }
        self.completed_rollout += 1
        self._rebuild_probabilities()
        return payload

    def artifact_summary(self) -> dict[str, Any]:
        return {
            "semantic_id": self.config.semantic_id,
            "generation": self.generation,
            "persistence": "session_local",
            "resume_behavior": "cold_archive",
            "archive_cell_count": self.cell_count,
            "archive_snapshot_count": self.snapshot_count,
            "completed_rollout": self.completed_rollout,
        }

    def close(self) -> None:
        self._cells.clear()
        self._probabilities.clear()
        self._sampled_this_rollout.clear()


def snapshot_curriculum_artifact_summary(source: Any) -> Mapping[str, Any] | None:
    """Find the neutral runtime summary through common environment wrappers."""

    seen: set[int] = set()
    current = source
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        summary = getattr(current, "snapshot_curriculum_summary", None)
        if callable(summary):
            value = summary()
            return dict(value) if value is not None else None
        current = getattr(current, "venv", None) or getattr(current, "env", None)
    return None
