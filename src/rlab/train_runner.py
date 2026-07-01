from __future__ import annotations

import argparse
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import json
import os
import queue
import re
import signal
import shutil
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from rlab.job_queue import (
    claim_train_job,
    connect,
    database_url,
    enqueue_eval_job,
    eval_protocol_hash,
    finish_train_job,
    heartbeat_train_job,
    load_goal_eval_spec,
    new_worker_id,
    print_status,
    queue_status,
)
from rlab.runtime_refs import normalize_runtime_image_ref
from rlab.seeds import validate_training_seed
from rlab.wandb_artifacts import (
    artifact_download_dir,
    checkpoint_step_from_name,
    download_model_artifact,
)
from rlab.wandb_utils import DEFAULT_WANDB_ENTITY, DEFAULT_WANDB_PROJECT


ARTIFACT_RE = re.compile(r"wandb artifact logged: (?P<name>[^ ]+) \((?P<location>[^)]+)\)")
METRIC_ROW_RE = re.compile(r"\|\s+(?P<key>[A-Za-z0-9_./-]+)\s+\|\s+(?P<value>[^|]+?)\s+\|")
WANDB_RUN_URL_RE = re.compile(r"https://wandb\.ai/\S+/runs/[A-Za-z0-9_-]+")
LEVEL_STATE_RE = re.compile(r"^Level\d+-\d+$")
RESUME_ARTIFACT_ROOT = Path("artifacts/train_resumes")
GRACEFUL_STOP_SIGNAL = getattr(signal, "SIGUSR1", None)
DEFAULT_CANCEL_GRACE_SECONDS = 30 * 60
DEFAULT_AUTOSCALE_SAMPLE_SECONDS = 30.0
DEFAULT_AUTOSCALE_WINDOW_SIZE = 5
DEFAULT_AUTOSCALE_COOLDOWN_SECONDS = 180.0
DEFAULT_WORKERS = 4
DEFAULT_MIN_WORKERS = 1
DEFAULT_MAX_WORKERS = 16
AUTOSCALE_SCALE_UP_THRESHOLDS = {
    "cpu_percent": 80.0,
    "memory_percent": 80.0,
    "gpu_percent": 85.0,
    "vram_percent": 85.0,
}
AUTOSCALE_SCALE_DOWN_THRESHOLDS = {
    "cpu_percent": 90.0,
    "memory_percent": 90.0,
    "gpu_percent": 95.0,
    "vram_percent": 95.0,
}
WORKER_IDLE = "idle"
WORKER_CLAIMING = "claiming"
WORKER_RUNNING = "running"
WORKER_RETIRING = "retiring"
WORKER_EXITED = "exited"


@dataclass(frozen=True)
class ResourceSample:
    cpu_percent: float | None = None
    memory_percent: float | None = None
    gpu_percent: float | None = None
    vram_percent: float | None = None
    error: str | None = None

    def values(self) -> dict[str, float]:
        return {
            key: float(value)
            for key, value in {
                "cpu_percent": self.cpu_percent,
                "memory_percent": self.memory_percent,
                "gpu_percent": self.gpu_percent,
                "vram_percent": self.vram_percent,
            }.items()
            if value is not None
        }

    def missing_resources(self, resources: Mapping[str, float]) -> tuple[str, ...]:
        return tuple(key for key in resources if getattr(self, key) is None)


@dataclass(frozen=True)
class WorkerBounds:
    starter_workers: int
    min_workers: int
    max_workers: int


@dataclass(frozen=True)
class AutoscaleConfig:
    starter_workers: int
    min_workers: int
    max_workers: int
    window_size: int = DEFAULT_AUTOSCALE_WINDOW_SIZE
    cooldown_seconds: float = DEFAULT_AUTOSCALE_COOLDOWN_SECONDS
    scale_up_thresholds: Mapping[str, float] = field(
        default_factory=lambda: dict(AUTOSCALE_SCALE_UP_THRESHOLDS)
    )
    scale_down_thresholds: Mapping[str, float] = field(
        default_factory=lambda: dict(AUTOSCALE_SCALE_DOWN_THRESHOLDS)
    )


@dataclass(frozen=True)
class AutoscaleDecision:
    action: str
    target_workers: int
    reason: str
    averages: dict[str, float]
    missing_resources: tuple[str, ...] = ()


class AutoscaleController:
    def __init__(self, config: AutoscaleConfig) -> None:
        if config.window_size < 1:
            raise ValueError("autoscale window size must be at least 1")
        self.config = config
        self.target_workers = config.starter_workers
        self.samples: deque[ResourceSample] = deque(maxlen=config.window_size)
        self.last_scale_at: float | None = None
        self.last_decision = AutoscaleDecision(
            action="hold",
            target_workers=self.target_workers,
            reason="starting",
            averages={},
        )

    @property
    def window_ready(self) -> bool:
        return len(self.samples) >= self.config.window_size

    def observe(self, sample: ResourceSample) -> None:
        self.samples.append(sample)

    def averages(self) -> dict[str, float]:
        sums: dict[str, float] = {}
        counts: dict[str, int] = {}
        for sample in self.samples:
            for key, value in sample.values().items():
                sums[key] = sums.get(key, 0.0) + value
                counts[key] = counts.get(key, 0) + 1
        return {key: sums[key] / counts[key] for key in sorted(sums)}

    def decide(
        self,
        *,
        pending_jobs: bool,
        active_workers: int,
        now: float,
    ) -> AutoscaleDecision:
        averages = self.averages()
        missing_for_up = tuple(
            key for key in self.config.scale_up_thresholds if key not in averages
        )
        if not self.samples:
            return self._remember("hold", "no resource samples", averages, missing_for_up)
        latest = self.samples[-1]
        if latest.error:
            return self._remember("hold", f"resource sample failed: {latest.error}", averages, missing_for_up)
        if not self.window_ready:
            return self._remember("hold", "warming up resource window", averages, missing_for_up)
        if self.last_scale_at is not None and now - self.last_scale_at < self.config.cooldown_seconds:
            return self._remember("hold", "autoscale cooldown active", averages, missing_for_up)

        saturated = [
            key
            for key, threshold in self.config.scale_down_thresholds.items()
            if averages.get(key) is not None and averages[key] >= threshold
        ]
        if saturated and self.target_workers > self.config.min_workers:
            self.target_workers -= 1
            self.last_scale_at = now
            return self._remember(
                "scale_down",
                "saturated: " + ",".join(saturated),
                averages,
                missing_for_up,
            )

        if self.target_workers >= self.config.max_workers:
            return self._remember("hold", "at max workers", averages, missing_for_up)
        if not pending_jobs:
            return self._remember("hold", "no pending queue demand", averages, missing_for_up)
        if active_workers < self.target_workers:
            return self._remember("hold", "active workers below target", averages, missing_for_up)
        if missing_for_up:
            return self._remember(
                "hold",
                "missing resources for scale up: " + ",".join(missing_for_up),
                averages,
                missing_for_up,
            )
        headroom = all(
            averages[key] < threshold
            for key, threshold in self.config.scale_up_thresholds.items()
        )
        if headroom:
            self.target_workers += 1
            self.last_scale_at = now
            return self._remember("scale_up", "resource headroom and pending demand", averages)
        return self._remember("hold", "resource headroom insufficient", averages, missing_for_up)

    def _remember(
        self,
        action: str,
        reason: str,
        averages: dict[str, float],
        missing_resources: tuple[str, ...] = (),
    ) -> AutoscaleDecision:
        self.last_decision = AutoscaleDecision(
            action=action,
            target_workers=self.target_workers,
            reason=reason,
            averages=averages,
            missing_resources=missing_resources,
        )
        return self.last_decision


@dataclass
class WorkerSlot:
    index: int
    worker_id: str
    state: str = WORKER_IDLE
    retire_requested: bool = False
    exit_reason: str | None = None
    exception: str | None = None
    thread: threading.Thread | None = None
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def set_state(self, state: str) -> None:
        with self.lock:
            if self.retire_requested and state == WORKER_IDLE:
                self.state = WORKER_RETIRING
            else:
                self.state = state

    def request_retire(self) -> None:
        with self.lock:
            self.retire_requested = True
            if self.state in {WORKER_IDLE, WORKER_CLAIMING}:
                self.state = WORKER_RETIRING

    def should_retire(self) -> bool:
        with self.lock:
            return self.retire_requested

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "index": self.index,
                "worker_id": self.worker_id,
                "state": self.state,
                "retire_requested": self.retire_requested,
                "exit_reason": self.exit_reason,
                "exception": self.exception,
            }

    def mark_exited(self, reason: str, *, exception: str | None = None) -> None:
        with self.lock:
            self.state = WORKER_EXITED
            self.exit_reason = reason
            if exception:
                self.exception = exception


def live_worker_slots(slots: list[WorkerSlot]) -> list[WorkerSlot]:
    return [slot for slot in slots if slot.snapshot()["state"] != WORKER_EXITED]


def mark_surplus_workers_for_retirement(slots: list[WorkerSlot], *, target_workers: int) -> tuple[str, ...]:
    live_slots = live_worker_slots(slots)
    surplus = max(0, len(live_slots) - target_workers)
    if surplus <= 0:
        return ()
    idle_slots = [
        slot
        for slot in live_slots
        if not slot.should_retire() and slot.snapshot()["state"] in {WORKER_IDLE, WORKER_CLAIMING, WORKER_RETIRING}
    ]
    busy_slots = [
        slot
        for slot in live_slots
        if not slot.should_retire() and slot.snapshot()["state"] == WORKER_RUNNING
    ]
    retired: list[str] = []
    for slot in [*idle_slots, *busy_slots]:
        if len(retired) >= surplus:
            break
        slot.request_retire()
        retired.append(slot.worker_id)
    return tuple(retired)


def strip_env_file_quotes(value: str) -> str:
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        return text[1:-1]
    return text


def normalize_wandb_tags(value: Any) -> list[str]:
    if isinstance(value, str):
        raw_tags = value.split(",")
    elif isinstance(value, list | tuple):
        raw_tags = value
    else:
        raw_tags = []

    tags: list[str] = []
    seen: set[str] = set()
    for raw_tag in raw_tags:
        tag = str(raw_tag).strip()
        if not tag or tag in seen:
            continue
        seen.add(tag)
        tags.append(tag)
    return tags


def append_unique_wandb_tag(tags: list[str], tag: str) -> list[str]:
    tag = tag.strip()
    if tag and tag not in set(tags):
        tags.append(tag)
    return tags


def normalize_level_states(config: Mapping[str, Any]) -> list[str]:
    raw_states = config.get("states") or []
    if isinstance(raw_states, str):
        candidates = [raw_states]
    elif isinstance(raw_states, list | tuple):
        candidates = [str(state) for state in raw_states]
    else:
        candidates = []

    state = str(config.get("state") or "").strip()
    if state:
        candidates.append(state)

    levels: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        level = candidate.strip()
        if not LEVEL_STATE_RE.match(level) or level in seen:
            continue
        seen.add(level)
        levels.append(level)
    return levels


def normalize_train_config(
    job: dict[str, Any], *, resolve_resume_artifact: bool = True
) -> dict[str, Any]:
    config = dict(job.get("train_config") or {})
    run_name = job.get("run_name") or config.get("run_name") or f"train_job_{job['id']}"
    config["run_name"] = run_name
    if job.get("run_description"):
        config["run_description"] = job["run_description"]
    if job.get("wandb_group"):
        config["wandb_group"] = job["wandb_group"]
    tags = normalize_wandb_tags(config.get("wandb_tags") or job.get("wandb_tags"))
    goal_slug = str(job.get("goal_slug") or "").strip()
    if goal_slug:
        config["goal_slug"] = goal_slug
        append_unique_wandb_tag(tags, f"goal:{goal_slug}")
    spec_slug = str(job.get("spec_slug") or "").strip()
    if spec_slug:
        config["spec_slug"] = spec_slug
        append_unique_wandb_tag(tags, f"spec:{spec_slug}")
    spec_path = str(job.get("spec_path") or "").strip()
    if spec_path:
        config["spec_path"] = spec_path
    if job.get("id") is not None:
        config["queue_train_job_id"] = int(job["id"])
    for level in normalize_level_states(config):
        append_unique_wandb_tag(tags, f"level:{level}")
    if tags:
        config["wandb_tags"] = ",".join(tags)
    if "seed" in config and config["seed"] is not None:
        validate_training_seed(
            config["seed"],
            label="train_config.seed",
            seed_span=config.get("n_envs", 1),
        )
    if job.get("runtime_image_ref"):
        config["runtime_image_ref"] = job["runtime_image_ref"]
    if job.get("run_target"):
        config["run_target"] = job["run_target"]
    if config.get("wandb_artifact_storage_uri") == "${CHECKPOINT_BUCKET_URI}":
        config["wandb_artifact_storage_uri"] = strip_env_file_quotes(os.environ.get("CHECKPOINT_BUCKET_URI", ""))
    resume_artifact = config.pop("resume_artifact", None)
    if resume_artifact:
        if config.get("resume"):
            raise ValueError("Use only one of resume or resume_artifact in train_config")
        if resolve_resume_artifact:
            resume_ref = str(resume_artifact)
            config["resume"] = str(
                download_model_artifact(
                    resume_ref,
                    artifact_download_dir(RESUME_ARTIFACT_ROOT, resume_ref),
                )
            )
    return config


def write_train_config_file(job: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(normalize_train_config(job), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def train_command_for_job(config_path: Path) -> list[str]:
    return [sys.executable, "-m", "rlab.train", "--train-config-json", str(config_path)]


def proc_stat_cpu_values(line: str) -> list[int]:
    parts = line.split()
    if parts and parts[0] == "cpu":
        parts = parts[1:]
    return [int(float(part)) for part in parts]


def cpu_percent_from_proc_stat(first: str, second: str) -> float | None:
    try:
        before = proc_stat_cpu_values(first)
        after = proc_stat_cpu_values(second)
    except ValueError:
        return None
    if len(before) < 8 or len(after) < 8:
        return None
    idle_before = before[3] + before[4]
    idle_after = after[3] + after[4]
    total_before = sum(before[:8])
    total_after = sum(after[:8])
    total_delta = total_after - total_before
    idle_delta = idle_after - idle_before
    if total_delta <= 0:
        return None
    return max(0.0, min(100.0, (total_delta - idle_delta) * 100.0 / total_delta))


def memory_percent_from_meminfo(text: str) -> float | None:
    values: dict[str, float] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        key = parts[0].rstrip(":")
        try:
            values[key] = float(parts[1])
        except ValueError:
            continue
    total = values.get("MemTotal")
    available = values.get("MemAvailable")
    if not total or available is None:
        return None
    return max(0.0, min(100.0, (total - available) * 100.0 / total))


def parse_nvidia_smi_resource_output(output: str) -> tuple[float | None, float | None]:
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            gpu_util = float(parts[0])
            memory_used = float(parts[1])
            memory_total = float(parts[2])
        except ValueError:
            continue
        if memory_total <= 0:
            return gpu_util, None
        return gpu_util, max(0.0, min(100.0, memory_used * 100.0 / memory_total))
    return None, None


def local_resource_sample() -> ResourceSample:
    errors: list[str] = []
    cpu_percent = None
    memory_percent = None
    gpu_percent = None
    vram_percent = None
    try:
        with Path("/proc/stat").open(encoding="utf-8") as handle:
            first_cpu = next((line for line in handle if line.startswith("cpu ")), "")
        time.sleep(0.2)
        with Path("/proc/stat").open(encoding="utf-8") as handle:
            second_cpu = next((line for line in handle if line.startswith("cpu ")), "")
        cpu_percent = cpu_percent_from_proc_stat(first_cpu, second_cpu)
        if cpu_percent is None:
            errors.append("cpu")
    except OSError as exc:
        errors.append(f"cpu:{exc}")
    try:
        memory_percent = memory_percent_from_meminfo(Path("/proc/meminfo").read_text(encoding="utf-8"))
        if memory_percent is None:
            errors.append("memory")
    except OSError as exc:
        errors.append(f"memory:{exc}")
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=3.0,
        )
        if result.returncode == 0:
            gpu_percent, vram_percent = parse_nvidia_smi_resource_output(result.stdout)
            if gpu_percent is None:
                errors.append("gpu")
            if vram_percent is None:
                errors.append("vram")
        else:
            errors.append("nvidia-smi")
    except (OSError, subprocess.TimeoutExpired) as exc:
        errors.append(f"nvidia-smi:{exc}")
    return ResourceSample(
        cpu_percent=cpu_percent,
        memory_percent=memory_percent,
        gpu_percent=gpu_percent,
        vram_percent=vram_percent,
        error="; ".join(errors) if errors else None,
    )


def matching_pending_train_job_exists(conn, args: argparse.Namespace) -> bool:
    runtime_image_ref = normalize_runtime_image_ref(args.runtime_image_ref)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT EXISTS (
              SELECT 1
              FROM train_jobs
              WHERE
                runtime_image_ref = %(runtime_image_ref)s
                AND cancel_requested = FALSE
                AND status = 'pending'
              LIMIT 1
            ) AS has_pending
            """,
            {
                "runtime_image_ref": runtime_image_ref,
            },
        )
        row = cur.fetchone()
    return bool(row and row.get("has_pending"))


def read_text_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8").strip() or None


def parse_key_value_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def parse_metric_value(value: str) -> int | float | str:
    text = value.strip().replace(",", "")
    try:
        number = float(text)
    except ValueError:
        return value.strip()
    if number.is_integer() and not any(marker in text.lower() for marker in (".", "e")):
        return int(number)
    return number


def parse_log_metrics(log_text: str) -> dict[str, int | float | str]:
    metrics: dict[str, int | float | str] = {}
    section = ""
    for line in log_text.splitlines():
        match = METRIC_ROW_RE.search(line)
        if not match:
            continue
        key = match.group("key").strip()
        value = match.group("value").strip()
        if key.endswith("/") and not value:
            section = key.rstrip("/")
            continue
        metric_key = key if key == "total_timesteps" or "/" in key or not section else f"{section}/{key}"
        parsed = parse_metric_value(value)
        metrics[metric_key] = parsed
        if metric_key in {"total_timesteps", "time/total_timesteps"}:
            metrics[key] = parsed
    return metrics


def parse_wandb_run_url(log_text: str) -> str | None:
    matches = WANDB_RUN_URL_RE.findall(log_text)
    return matches[-1] if matches else None


def collect_result_metadata(job: dict[str, Any], log_path: Path) -> dict[str, Any]:
    config = normalize_train_config(job, resolve_resume_artifact=False)
    run_name = str(config["run_name"])
    runs_dir = str(config.get("runs_dir") or "runs")
    run_dir = Path(runs_dir) / run_name
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    artifact_refs = [
        {"name": match.group("name"), "location": match.group("location")}
        for match in ARTIFACT_RE.finditer(log_text)
    ]
    metrics = parse_log_metrics(log_text)
    metrics.update(parse_key_value_file(run_dir / "early_stop.txt"))
    return {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "final_model_path": str(run_dir / "final_model.zip")
        if (run_dir / "final_model.zip").is_file()
        else None,
        "wandb_run_id": read_text_file(run_dir / "wandb_run_id.txt"),
        "wandb_url": read_text_file(run_dir / "wandb_url.txt") or parse_wandb_run_url(log_text),
        "artifact_refs": artifact_refs,
        "metrics_json": metrics,
    }


def artifact_checkpoint_step(ref: Mapping[str, Any]) -> int | None:
    location = str(ref.get("location") or "")
    name = str(ref.get("name") or "")
    for candidate in (location.rsplit("/", 1)[-1], name):
        step = checkpoint_step_from_name(candidate)
        if step is not None:
            return step
    return None


def checkpoint_artifact_ref(job: dict[str, Any], ref: Mapping[str, Any], step: int) -> str:
    config = normalize_train_config(job, resolve_resume_artifact=False)
    project = str(config.get("wandb_project") or DEFAULT_WANDB_PROJECT)
    entity = str(config.get("wandb_entity") or DEFAULT_WANDB_ENTITY)
    name = str(ref.get("name") or "").strip()
    if not name:
        raise ValueError("artifact name is required")
    return f"{entity}/{project}/{name}:step-{step}"


def enqueue_eval_jobs_for_checkpoints(conn, job: dict[str, Any], result: Mapping[str, Any]) -> int:
    refs = [dict(item) for item in result.get("artifact_refs") or []]
    checkpoint_refs = [ref for ref in refs if str(ref.get("name") or "").endswith("-checkpoint")]
    if not checkpoint_refs:
        return 0
    eval_spec = load_goal_eval_spec(str(job["goal_slug"]))
    protocol_hash = eval_protocol_hash(eval_spec)
    base_eval_config = dict(eval_spec["eval_config"])
    count = 0
    for ref in checkpoint_refs:
        step = artifact_checkpoint_step(ref)
        if step is None:
            print(f"warning: could not infer checkpoint step for artifact {ref}", flush=True)
            continue
        artifact_ref = checkpoint_artifact_ref(job, ref, step)
        eval_config = {
            **base_eval_config,
            "artifact_ref": artifact_ref,
            "checkpoint_step": step,
            "eval_protocol_hash": protocol_hash,
        }
        row = enqueue_eval_job(
            conn,
            goal_slug=str(job["goal_slug"]),
            spec_slug=job.get("spec_slug"),
            spec_path=job.get("spec_path"),
            train_job_id=int(job["id"]),
            runtime_image_ref=str(job["runtime_image_ref"]),
            artifact_ref=artifact_ref,
            checkpoint_step=step,
            eval_protocol_hash=protocol_hash,
            eval_config=eval_config,
            max_attempts=1,
            candidate_label=str(job.get("run_name") or ""),
        )
        count += 1
        print(
            f"eval enqueued: eval_job_id={row['id']} checkpoint_step={step} "
            f"artifact={artifact_ref}",
            flush=True,
        )
    return count


def should_purge_successful_run_data(job: dict[str, Any], result: Mapping[str, Any]) -> bool:
    config = normalize_train_config(job, resolve_resume_artifact=False)
    if not bool(config.get("wandb")):
        return False
    if str(config.get("wandb_mode") or "online") != "online":
        return False
    if bool(config.get("no_wandb_artifacts")):
        return False
    return bool(result.get("artifact_refs"))


def purge_successful_run_data(job: dict[str, Any], result: Mapping[str, Any]) -> bool:
    config = normalize_train_config(job, resolve_resume_artifact=False)
    runs_dir = Path(str(config.get("runs_dir") or "runs"))
    raw_run_dir = str(result.get("run_dir") or "").strip()
    if not raw_run_dir:
        return False
    run_dir = Path(raw_run_dir)
    try:
        runs_root = runs_dir.resolve()
        target = run_dir.resolve()
    except OSError as exc:
        print(f"warning: could not resolve run cleanup paths run_dir={run_dir}: {exc}", flush=True)
        return False
    if target == runs_root or not target.is_relative_to(runs_root):
        print(
            f"warning: refusing to purge run_dir outside runs_dir: run_dir={target} runs_dir={runs_root}",
            flush=True,
        )
        return False
    if not target.exists():
        return False
    try:
        shutil.rmtree(target)
    except OSError as exc:
        print(f"warning: could not purge successful run data {target}: {exc}", flush=True)
        return False
    print(f"purged successful run data: {target}", flush=True)
    return True


def signal_label(signum: int) -> str:
    try:
        return signal.Signals(signum).name
    except ValueError:
        return f"signal-{signum}"


def request_graceful_stop(process: subprocess.Popen[str]) -> bool:
    if GRACEFUL_STOP_SIGNAL is None or process.poll() is not None:
        return False
    try:
        process.send_signal(GRACEFUL_STOP_SIGNAL)
    except ProcessLookupError:
        return False
    return True


def terminate_process(process: subprocess.Popen[str], *, grace_seconds: float = 20.0) -> None:
    if process.poll() is not None:
        return
    try:
        process.send_signal(signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace_seconds
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.5)
    if process.poll() is None:
        process.kill()


def run_training_job(
    conn,
    *,
    job: dict[str, Any],
    worker_id: str,
    lease_seconds: int,
    heartbeat_interval: float,
    log_dir: Path,
    dry_run: bool = False,
    cancel_grace_seconds: float = DEFAULT_CANCEL_GRACE_SECONDS,
    purge_successful_run_data_enabled: bool = True,
) -> str:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_stem = f"train_job_{job['id']}_{uuid.uuid4().hex[:8]}"
    config_path = write_train_config_file(job, log_dir / f"{log_stem}.config.json")
    command = train_command_for_job(config_path)
    log_path = log_dir / f"{log_stem}.log"
    print(f"train_job={job['id']} command={' '.join(command)}", flush=True)

    if dry_run:
        finish_train_job(
            conn,
            job=job,
            worker_id=worker_id,
            status="succeeded",
            exit_code=0,
            result={
                "run_name": normalize_train_config(job)["run_name"],
                "metrics_json": {"dry_run": True},
            },
        )
        return "succeeded"

    canceled = False
    graceful_cancel_started_at = None
    drain_after_job = bool(job.get("drain_requested"))
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=os.environ.copy(),
        )
        assert process.stdout is not None
        output_queue: queue.Queue[str | None] = queue.Queue()

        def read_output() -> None:
            assert process.stdout is not None
            for output_line in process.stdout:
                output_queue.put(output_line)
            output_queue.put(None)

        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()
        last_heartbeat = 0.0
        saw_eof = False
        try:
            while True:
                try:
                    line = output_queue.get(timeout=0.5)
                except queue.Empty:
                    line = ""
                if line is None:
                    saw_eof = True
                    line = ""
                elif line:
                    log_file.write(line)
                    log_file.flush()
                    if (
                        "wandb.ai/" in line
                        or "wandb artifact logged:" in line
                        or "early stop" in line.lower()
                    ):
                        print(line.rstrip(), flush=True)
                if process.poll() is not None and saw_eof:
                    break
                now = time.monotonic()
                if now - last_heartbeat >= heartbeat_interval:
                    heartbeat = heartbeat_train_job(
                        conn,
                        job_id=int(job["id"]),
                        worker_id=worker_id,
                        lease_seconds=lease_seconds,
                    )
                    last_heartbeat = now
                    if heartbeat is None:
                        terminate_process(process)
                        canceled = True
                        break
                    if heartbeat.get("cancel_requested"):
                        canceled = True
                        if graceful_cancel_started_at is None:
                            graceful_cancel_started_at = now
                            if request_graceful_stop(process):
                                signal_name = signal_label(int(GRACEFUL_STOP_SIGNAL))
                                print(
                                    f"train_job={job['id']} graceful_cancel_signal={signal_name} "
                                    f"cancel_grace_seconds={cancel_grace_seconds:g}",
                                    flush=True,
                                )
                            else:
                                terminate_process(process)
                                break
                    if heartbeat.get("drain_requested"):
                        drain_after_job = True
                    try:
                        running_result = collect_result_metadata(job, log_path)
                        enqueue_eval_jobs_for_checkpoints(conn, job, running_result)
                    except Exception as exc:
                        if hasattr(conn, "rollback"):
                            conn.rollback()
                        print(
                            f"warning: failed to record running train metadata or enqueue eval "
                            f"job={job['id']}: {exc}",
                            flush=True,
                        )
                if (
                    graceful_cancel_started_at is not None
                    and process.poll() is None
                    and now - graceful_cancel_started_at >= cancel_grace_seconds
                ):
                    print(
                        f"train_job={job['id']} graceful_cancel_timeout="
                        f"{cancel_grace_seconds:g}; sending SIGTERM",
                        flush=True,
                    )
                    terminate_process(process)
                    break
        finally:
            if process.poll() is None:
                terminate_process(process)
            reader.join(timeout=5.0)

    exit_code = process.returncode
    result = collect_result_metadata(job, log_path)
    try:
        enqueue_eval_jobs_for_checkpoints(conn, job, result)
    except Exception as exc:
        if hasattr(conn, "rollback"):
            conn.rollback()
        print(f"warning: failed to enqueue eval jobs job={job['id']}: {exc}", flush=True)
    if canceled:
        finish_train_job(
            conn,
            job=job,
            worker_id=worker_id,
            status="canceled",
            exit_code=exit_code,
            result=result,
            error="cancel requested or lease lost",
        )
        return "canceled"
    if exit_code == 0:
        finish_train_job(
            conn,
            job=job,
            worker_id=worker_id,
            status="succeeded",
            exit_code=exit_code,
            result=result,
        )
        if purge_successful_run_data_enabled and should_purge_successful_run_data(job, result):
            purge_successful_run_data(job, result)
        if drain_after_job:
            return "succeeded_drained"
        return "succeeded"
    finish_train_job(
        conn,
        job=job,
        worker_id=worker_id,
        status="failed",
        exit_code=exit_code,
        result=result,
        error=f"train process exited {exit_code}",
    )
    return "failed"


def sleep_with_retire_check(seconds: float, slot: WorkerSlot | None = None) -> None:
    deadline = time.monotonic() + max(0.0, seconds)
    while time.monotonic() < deadline:
        if slot is not None and slot.should_retire():
            return
        time.sleep(min(0.5, max(0.0, deadline - time.monotonic())))


def worker_loop(args: argparse.Namespace, *, worker_id: str, slot: WorkerSlot | None = None) -> str:
    conn = connect(database_url(args.direct))
    completed = 0
    exit_reason = "closed"
    try:
        while args.max_jobs <= 0 or completed < args.max_jobs:
            if slot is not None and slot.should_retire():
                exit_reason = "retired"
                return "retired"
            if slot is not None:
                slot.set_state(WORKER_CLAIMING)
            job = claim_train_job(
                conn,
                runtime_image_ref=args.runtime_image_ref,
                worker_id=worker_id,
                lease_seconds=args.lease_seconds,
            )
            if job is None:
                if slot is not None:
                    slot.set_state(WORKER_IDLE)
                if args.once:
                    exit_reason = "once_empty"
                    return "once_empty"
                sleep_with_retire_check(args.poll_seconds, slot)
                continue
            if slot is not None:
                slot.set_state(WORKER_RUNNING)
            status = run_training_job(
                conn,
                job=job,
                worker_id=worker_id,
                lease_seconds=args.lease_seconds,
                heartbeat_interval=args.heartbeat_seconds,
                log_dir=Path(args.log_dir),
                dry_run=args.dry_run,
                cancel_grace_seconds=args.cancel_grace_seconds,
                purge_successful_run_data_enabled=not args.keep_successful_run_data,
            )
            completed += 1
            print(f"worker={worker_id} train_job={job['id']} status={status}", flush=True)
            if slot is not None:
                slot.set_state(WORKER_IDLE)
            if job.get("drain_requested") or status.endswith("_drained"):
                exit_reason = "drained"
                return "drained"
            if slot is not None and slot.should_retire():
                exit_reason = "retired"
                return "retired"
        exit_reason = "max_jobs"
        return "max_jobs"
    finally:
        if slot is not None:
            slot.mark_exited(exit_reason)
        conn.close()


def resolve_worker_bounds(args: argparse.Namespace) -> WorkerBounds:
    workers = int(args.workers)
    if workers < 1:
        raise SystemExit("--workers must be at least 1")
    if not getattr(args, "autoscale", False):
        return WorkerBounds(
            starter_workers=workers,
            min_workers=workers,
            max_workers=workers,
        )
    min_workers = int(args.min_workers)
    max_workers = int(args.max_workers)
    if min_workers < 1:
        raise SystemExit("--min-workers must be at least 1")
    if not min_workers <= workers <= max_workers:
        raise SystemExit("autoscale requires 1 <= --min-workers <= --workers <= --max-workers")
    if int(args.autoscale_window_size) < 1:
        raise SystemExit("--autoscale-window-size must be at least 1")
    if float(args.autoscale_sample_seconds) <= 0:
        raise SystemExit("--autoscale-sample-seconds must be positive")
    if float(args.autoscale_cooldown_seconds) < 0:
        raise SystemExit("--autoscale-cooldown-seconds must be non-negative")
    return WorkerBounds(
        starter_workers=workers,
        min_workers=min_workers,
        max_workers=max_workers,
    )


def worker_state_counts(slots: list[WorkerSlot]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for slot in slots:
        state = str(slot.snapshot()["state"])
        counts[state] = counts.get(state, 0) + 1
    return counts


def autoscale_status_payload(
    *,
    bounds: WorkerBounds,
    controller: AutoscaleController,
    slots: list[WorkerSlot],
    pending_jobs: bool,
    decision: AutoscaleDecision,
    retired_workers: tuple[str, ...] = (),
) -> dict[str, Any]:
    live_slots = live_worker_slots(slots)
    return {
        "updated_at_unix": time.time(),
        "starter_workers": bounds.starter_workers,
        "min_workers": bounds.min_workers,
        "max_workers": bounds.max_workers,
        "target_workers": controller.target_workers,
        "active_workers": len(live_slots),
        "running_workers": sum(1 for slot in live_slots if slot.snapshot()["state"] == WORKER_RUNNING),
        "retiring_workers": sum(1 for slot in live_slots if slot.snapshot()["retire_requested"]),
        "pending_jobs": pending_jobs,
        "window_ready": controller.window_ready,
        "sample_count": len(controller.samples),
        "rolling_averages": decision.averages,
        "last_decision": decision.action,
        "last_reason": decision.reason,
        "missing_resources": list(decision.missing_resources),
        "retired_workers": list(retired_workers),
        "state_counts": worker_state_counts(slots),
        "workers": [slot.snapshot() for slot in slots],
    }


def write_autoscale_status(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def start_autoscale_worker(
    args: argparse.Namespace,
    *,
    slot: WorkerSlot,
) -> None:
    def run() -> None:
        try:
            worker_loop(args, worker_id=slot.worker_id, slot=slot)
        except Exception as exc:
            slot.mark_exited("error", exception=str(exc))
            print(f"worker={slot.worker_id} autoscale_worker_error={exc}", flush=True)

    thread = threading.Thread(target=run, name=slot.worker_id)
    slot.thread = thread
    thread.start()


def spawn_autoscale_worker(
    args: argparse.Namespace,
    slots: list[WorkerSlot],
    *,
    index: int,
) -> WorkerSlot:
    worker_prefix = args.worker_id or "train-runner"
    slot = WorkerSlot(
        index=index,
        worker_id=f"{worker_prefix}-{index}-{uuid.uuid4().hex[:8]}",
    )
    slots.append(slot)
    start_autoscale_worker(args, slot=slot)
    return slot


def run_autoscale_pool(args: argparse.Namespace, bounds: WorkerBounds) -> int:
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    status_path = log_dir / "autoscale_status.json"
    controller = AutoscaleController(
        AutoscaleConfig(
            starter_workers=bounds.starter_workers,
            min_workers=bounds.min_workers,
            max_workers=bounds.max_workers,
            window_size=int(args.autoscale_window_size),
            cooldown_seconds=float(args.autoscale_cooldown_seconds),
        )
    )
    resource_probe: Callable[[], ResourceSample] = getattr(args, "resource_probe", local_resource_sample)
    pending_probe: Callable[[Any, argparse.Namespace], bool] = getattr(
        args,
        "pending_probe",
        matching_pending_train_job_exists,
    )
    slots: list[WorkerSlot] = []
    next_worker_index = 0
    for _ in range(bounds.starter_workers):
        spawn_autoscale_worker(args, slots, index=next_worker_index)
        next_worker_index += 1

    conn = connect(database_url(args.direct))
    next_sample_at = 0.0
    exit_status = 0
    try:
        while live_worker_slots(slots):
            now = time.monotonic()
            for slot in slots:
                if slot.thread is not None and not slot.thread.is_alive():
                    slot.thread.join(timeout=0)
            errors = [slot.snapshot() for slot in slots if slot.snapshot().get("exception")]
            if errors:
                exit_status = 1
            if now >= next_sample_at:
                sample = resource_probe()
                controller.observe(sample)
                try:
                    pending_jobs = pending_probe(conn, args)
                except Exception as exc:
                    pending_jobs = False
                    sample = ResourceSample(error=f"pending queue check failed: {exc}")
                    controller.observe(sample)
                    print(f"warning: autoscale pending queue check failed: {exc}", flush=True)
                live_count = len(live_worker_slots(slots))
                decision = controller.decide(
                    pending_jobs=pending_jobs,
                    active_workers=live_count,
                    now=now,
                )
                retired_workers: tuple[str, ...] = ()
                if decision.action == "scale_up":
                    spawn_autoscale_worker(args, slots, index=next_worker_index)
                    next_worker_index += 1
                    print(
                        f"autoscale action=scale_up target_workers={decision.target_workers} "
                        f"reason={decision.reason}",
                        flush=True,
                    )
                elif decision.action == "scale_down":
                    retired_workers = mark_surplus_workers_for_retirement(
                        slots,
                        target_workers=decision.target_workers,
                    )
                    print(
                        f"autoscale action=scale_down target_workers={decision.target_workers} "
                        f"retiring={len(retired_workers)} reason={decision.reason}",
                        flush=True,
                    )
                elif decision.reason.startswith("resource sample failed"):
                    print(f"warning: autoscale {decision.reason}", flush=True)
                write_autoscale_status(
                    status_path,
                    autoscale_status_payload(
                        bounds=bounds,
                        controller=controller,
                        slots=slots,
                        pending_jobs=pending_jobs,
                        decision=decision,
                        retired_workers=retired_workers,
                    ),
                )
                next_sample_at = now + float(args.autoscale_sample_seconds)
            sleep_with_retire_check(min(1.0, float(args.autoscale_sample_seconds)))
    finally:
        conn.close()
        for slot in live_worker_slots(slots):
            slot.request_retire()
        for slot in slots:
            if slot.thread is not None:
                slot.thread.join(timeout=5.0)
    return exit_status


def run_pool(args: argparse.Namespace) -> int:
    bounds = resolve_worker_bounds(args)
    if args.autoscale:
        return run_autoscale_pool(args, bounds)
    workers = bounds.starter_workers
    if workers == 1:
        worker_loop(args, worker_id=args.worker_id or new_worker_id("train-runner"))
        return 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                worker_loop,
                args,
                worker_id=f"{args.worker_id or 'train-runner'}-{index}-{uuid.uuid4().hex[:8]}",
            )
            for index in range(workers)
        ]
        for future in futures:
            future.result()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Drain Codex-authored PPO train jobs.")
    parser.add_argument("--profile", help=argparse.SUPPRESS)
    parser.add_argument(
        "--runtime-image-ref",
        required=True,
        help="Exact immutable runtime image ref that claimed train_jobs must require.",
    )
    parser.add_argument("--run-target", help=argparse.SUPPRESS)
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Fixed worker count, or autoscale starter count.",
    )
    parser.add_argument(
        "--autoscale",
        action="store_true",
        help="Dynamically adjust desired workers between --min-workers and --max-workers.",
    )
    parser.add_argument(
        "--min-workers",
        type=int,
        default=DEFAULT_MIN_WORKERS,
        help="Minimum workers when --autoscale is enabled.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help="Maximum workers when --autoscale is enabled.",
    )
    parser.add_argument(
        "--autoscale-sample-seconds",
        type=float,
        default=DEFAULT_AUTOSCALE_SAMPLE_SECONDS,
        help="Seconds between autoscale resource samples.",
    )
    parser.add_argument(
        "--autoscale-window-size",
        type=int,
        default=DEFAULT_AUTOSCALE_WINDOW_SIZE,
        help="Number of resource samples in the rolling autoscale window.",
    )
    parser.add_argument(
        "--autoscale-cooldown-seconds",
        type=float,
        default=DEFAULT_AUTOSCALE_COOLDOWN_SECONDS,
        help="Minimum seconds between autoscale target changes.",
    )
    parser.add_argument("--worker-id")
    parser.add_argument("--lease-seconds", type=int, default=1800)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument(
        "--cancel-grace-seconds",
        type=float,
        default=DEFAULT_CANCEL_GRACE_SECONDS,
        help="Seconds to wait after graceful cancel signal before SIGTERM.",
    )
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--max-jobs", type=int, default=0, help="0 means unlimited.")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Exit when no matching job is available.",
    )
    parser.add_argument("--log-dir", default="logs/train_runner")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Claim and complete jobs without training.",
    )
    parser.add_argument(
        "--keep-successful-run-data",
        action="store_true",
        help="Do not purge local run directories after successful online artifact uploads.",
    )
    parser.add_argument("--direct", action="store_true", help="Use DIRECT_DATABASE_URL.")
    parser.add_argument(
        "--status-goal",
        help="Print compact queue status for this goal before starting workers.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.status_goal:
        conn = connect(database_url(args.direct))
        try:
            print_status(queue_status(conn, goal_slug=args.status_goal))
        finally:
            conn.close()
    raise SystemExit(run_pool(args))


if __name__ == "__main__":
    main()
