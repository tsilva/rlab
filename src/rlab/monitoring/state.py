from __future__ import annotations

import json
import os
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from rlab.campaign import connect, database_url
from rlab.compute_targets import instance_label, target_kind
from rlab.json_utils import json_safe
from rlab.metric_names import (
    THROUGHPUT_LOOP_FPS,
    TRAIN_DONE_ALL,
)


RUNNING_STATES = {"running"}
QUEUED_STATES = {"pending"}
PROBE_TIMEOUT_SECONDS = 3.0


@dataclass(frozen=True)
class MonitorOptions:
    repo_root: Path
    goal: str | None = None
    direct: bool = False
    sample: bool = False
    limit: int = 40


@dataclass(frozen=True)
class DeviceProbe:
    ok: bool
    label: str
    detail: str
    metrics: Mapping[str, Any] | None = None


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def short_age(value: Any) -> str:
    if not value:
        return ""
    if isinstance(value, str):
        try:
            timestamp = datetime.fromisoformat(value)
        except ValueError:
            return value
    elif isinstance(value, datetime):
        timestamp = value
    else:
        return str(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    delta = datetime.now(UTC) - timestamp.astimezone(UTC)
    seconds = max(int(delta.total_seconds()), 0)
    if seconds < 60:
        return f"{seconds}s ago"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m ago"
    hours = minutes // 60
    if hours < 48:
        return f"{hours}h ago"
    return f"{hours // 24}d ago"


def minutes_until(value: Any) -> str:
    if not value:
        return ""
    timestamp = value
    if isinstance(value, str):
        try:
            timestamp = datetime.fromisoformat(value)
        except ValueError:
            return value
    if not isinstance(timestamp, datetime):
        return str(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    delta = timestamp.astimezone(UTC) - datetime.now(UTC)
    minutes = int(delta.total_seconds() // 60)
    if minutes < 0:
        return "expired"
    if minutes == 0:
        return "<1m"
    return f"{minutes}m"


def load_instances(repo_root: Path) -> dict[str, Any]:
    path = repo_root / "experiments" / "instances.json"
    if not path.is_file():
        return {"instances": {}}
    return json.loads(path.read_text(encoding="utf-8"))


REMOTE_METRICS_SCRIPT = r"""
printf 'host='; hostname
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); gsub(/ /,"",$3); print "gpu_util_pct="$1; print "vram_used_mib="$2; print "vram_total_mib="$3; exit}'
awk '/^MemTotal:/ {total=$2} /^MemAvailable:/ {avail=$2} END {if (total > 0) {printf "ram_used_mib=%d\nram_total_mib=%d\n", (total - avail) / 1024, total / 1024}}' /proc/meminfo
awk '/^cpu / {print "cpu1="$0}' /proc/stat
sleep 0.2
awk '/^cpu / {print "cpu2="$0}' /proc/stat
""".strip()


def probe_command(device_key: str) -> list[str] | None:
    if device_key == "rtx4090":
        return [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={int(PROBE_TIMEOUT_SECONDS)}",
            "tsilva@beast-3",
            REMOTE_METRICS_SCRIPT,
        ]
    if device_key == "rtx2060":
        return [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={int(PROBE_TIMEOUT_SECONDS)}",
            "-o",
            "HostKeyAlias=beast-2",
            "tsilva@192.168.133.26",
            REMOTE_METRICS_SCRIPT,
        ]
    return None


def cpu_percent_from_proc_stat(first: str, second: str) -> float | None:
    def values(line: str) -> list[int]:
        parts = line.split()
        if parts and parts[0] == "cpu":
            parts = parts[1:]
        return [int(float(part)) for part in parts]

    try:
        before = values(first)
        after = values(second)
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


def parse_probe_metrics(output: str) -> tuple[str, dict[str, Any]]:
    raw: dict[str, str] = {}
    for line in output.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        raw[key.strip()] = value.strip()

    metrics: dict[str, Any] = {}
    for key in ("gpu_util_pct", "vram_used_mib", "vram_total_mib", "ram_used_mib", "ram_total_mib"):
        value = raw.get(key)
        if value is None:
            continue
        try:
            metrics[key] = float(value)
        except ValueError:
            continue
    cpu = cpu_percent_from_proc_stat(raw.get("cpu1", ""), raw.get("cpu2", ""))
    if cpu is not None:
        metrics["cpu_util_pct"] = cpu
    return raw.get("host") or "reachable", metrics


def percent_text(value: Any) -> str:
    if not isinstance(value, int | float):
        return ""
    return f"{value:.0f}%"


def mib_text(used: Any, total: Any) -> str:
    if not isinstance(used, int | float) or not isinstance(total, int | float) or total <= 0:
        return ""
    return f"{used / 1024:.1f}/{total / 1024:.1f} GB"


def usage_summary(metrics: Mapping[str, Any] | None) -> str:
    if not metrics:
        return ""
    parts = []
    gpu = percent_text(metrics.get("gpu_util_pct"))
    cpu = percent_text(metrics.get("cpu_util_pct"))
    ram = percent_text(
        float(metrics["ram_used_mib"]) * 100.0 / float(metrics["ram_total_mib"])
        if metrics.get("ram_used_mib") is not None and metrics.get("ram_total_mib")
        else None
    )
    if gpu:
        parts.append(f"gpu {gpu}")
    if cpu:
        parts.append(f"cpu {cpu}")
    if ram:
        parts.append(f"mem {ram}")
    return " / ".join(parts)


def percent_value(value: Any) -> float | None:
    if not isinstance(value, int | float):
        return None
    return max(0.0, min(100.0, float(value)))


def resource_metrics(metrics: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not metrics:
        return {}
    resources: dict[str, dict[str, Any]] = {}
    gpu = percent_value(metrics.get("gpu_util_pct"))
    cpu = percent_value(metrics.get("cpu_util_pct"))
    if gpu is not None:
        resources["gpu"] = {"percent": gpu, "label": percent_text(gpu)}
    if cpu is not None:
        resources["cpu"] = {"percent": cpu, "label": percent_text(cpu)}
    if metrics.get("ram_used_mib") is not None and metrics.get("ram_total_mib"):
        ram_percent = percent_value(
            float(metrics["ram_used_mib"]) * 100.0 / float(metrics["ram_total_mib"])
        )
        if ram_percent is not None:
            resources["memory"] = {
                "percent": ram_percent,
                "label": mib_text(metrics.get("ram_used_mib"), metrics.get("ram_total_mib")),
            }
    if metrics.get("vram_used_mib") is not None and metrics.get("vram_total_mib"):
        vram_percent = percent_value(
            float(metrics["vram_used_mib"]) * 100.0 / float(metrics["vram_total_mib"])
        )
        if vram_percent is not None:
            resources["vram"] = {
                "percent": vram_percent,
                "label": mib_text(metrics.get("vram_used_mib"), metrics.get("vram_total_mib")),
            }
    return resources


def probe_device(device_key: str) -> DeviceProbe:
    command = probe_command(device_key)
    if command is None:
        return DeviceProbe(ok=True, label="config", detail="no live probe configured")
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            check=False,
            text=True,
            timeout=PROBE_TIMEOUT_SECONDS + 1,
        )
    except subprocess.TimeoutExpired:
        return DeviceProbe(ok=False, label="SSH timeout", detail="connection timed out")
    except OSError as exc:
        return DeviceProbe(ok=False, label="probe failed", detail=str(exc))

    output = (result.stdout or result.stderr).strip()
    if result.returncode == 0:
        host, metrics = parse_probe_metrics(output)
        return DeviceProbe(ok=True, label="SSH", detail=host, metrics=metrics)
    return DeviceProbe(
        ok=False,
        label="SSH failed",
        detail=output.replace("\n", " ") or f"exit {result.returncode}",
    )


def live_device_probes(device_keys: list[str]) -> dict[str, DeviceProbe]:
    probeable = [key for key in device_keys if probe_command(key) is not None]
    if not probeable:
        return {}
    probes: dict[str, DeviceProbe] = {}
    with ThreadPoolExecutor(max_workers=len(probeable)) as executor:
        futures = {executor.submit(probe_device, key): key for key in probeable}
        for future in as_completed(futures):
            probes[futures[future]] = future.result()
    return probes


def game_short_name(game: str) -> str:
    if "Mario" in game:
        return "Mario"
    if game.endswith("-v0"):
        return game[:-3]
    return game or "target"


def target_label(config: dict[str, Any], *, fallback: str = "") -> str:
    game = game_short_name(str(config.get("game") or fallback or "target"))
    states = config.get("states") or ()
    if isinstance(states, str):
        states = [part.strip() for part in states.split(",") if part.strip()]
    state = str(config.get("state") or "").strip()
    state_probs = config.get("state_probs") or ()
    if isinstance(state_probs, str):
        state_probs = [part.strip() for part in state_probs.split(",") if part.strip()]

    if states:
        compact_states = [str(item).replace("Level", "L") for item in states]
        if len(compact_states) > 1:
            if state_probs and len(state_probs) == len(compact_states):
                probs = " / ".join(str(prob) for prob in state_probs)
                return f"{game} mixed {probs}"
            counts = Counter(compact_states)
            if len(counts) < len(compact_states):
                parts = [
                    f"{label} x{count}" if count > 1 else label
                    for label, count in counts.items()
                ]
                return f"{game} {' + '.join(parts)}"
            return f"{game} {' + '.join(compact_states)}"
        return f"{game} {compact_states[0]}"
    if state:
        return f"{game} {state.replace('Level', 'L')}"
    candidate = config.get("artifact_ref") or config.get("model_artifact") or config.get("model_path")
    if candidate:
        return Path(str(candidate)).name[:32]
    return game


def completion_progress(metrics: dict[str, Any]) -> str:
    done_all = metrics.get(TRAIN_DONE_ALL)
    if isinstance(done_all, int | float):
        return f"done:{int(done_all)}"
    value = metrics.get("completion_rate")
    if isinstance(value, int | float):
        return f"{value:.2f}"
    episodes = metrics.get("episodes")
    completed = metrics.get("completed_episodes") or metrics.get("completion_count")
    if episodes and completed:
        return f"{completed}/{episodes}"
    return ""


def metric_value(metrics: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in metrics:
            return metrics[key]
    return None


def payload_from_row(
    *,
    table: str,
    row: dict[str, Any],
    config_key: str,
    result_table: str,
) -> dict[str, Any]:
    payload = row.get("job_payload")
    if not isinstance(payload, dict):
        payload = {key: value for key, value in row.items() if key not in {"job_payload", "result_payload"}}
    result_payload = row.get("result_payload")
    context = {
        "goal_slug": row.get("goal_slug"),
        "spec_slug": row.get("spec_slug"),
    }
    return {
        "table": table,
        "schema": list(payload.keys()),
        "config_key": config_key,
        "job": payload,
        "context": {key: value for key, value in context.items() if value},
        result_table: result_payload if isinstance(result_payload, dict) else None,
    }


def infer_device_key(kind: str, profile: str, worker: str, config: dict[str, Any]) -> str:
    text = " ".join(
        [
            kind,
            profile.lower(),
            worker.lower(),
            str(config.get("device") or "").lower(),
            str(config.get("runner") or "").lower(),
            str(config.get("target") or "").lower(),
        ]
    )
    if "4090" in text or "beast-3" in text or "k8s/rtx4090" in text or "k8s/beast-3" in text:
        if "runpod" in text:
            return "runpod-rtx4090"
        return "rtx4090"
    if "2060" in text or "beast2" in text or "beast-2" in text:
        return "rtx2060"
    if "runpod-l4" in text or "runpod_l4" in text or "runpod l4" in text:
        return "runpod-l4"
    if "runpod-t4" in text or "runpod_t4" in text or "runpod t4" in text:
        return "runpod-t4"
    if "runpod" in text:
        return "runpod-rtx4090"
    if "modal-t4" in text or "t4-modal" in text:
        return "modal-t4"
    if "modal" in text:
        return "modal"
    return "local"


def where_label(device_key: str, *, kind: str) -> str:
    if device_key == "rtx4090":
        return "beast-3 / RTX4090"
    if device_key == "rtx2060":
        return "beast-2"
    if device_key.startswith("runpod"):
        return device_key
    if device_key == "modal-t4":
        return "Modal T4"
    if device_key == "modal":
        return "Modal CPU"
    return "local" if kind == "eval" else "Local Mac"


def attention_for_row(
    *,
    status: str,
    error: str | None,
    heartbeat_at: Any,
    lease_expires_at: Any,
    metrics: dict[str, Any],
) -> str:
    if error:
        return "check logs"
    if status == "failed":
        return "check logs"
    if status == "running":
        lease = minutes_until(lease_expires_at)
        if lease == "expired":
            return "lease expired"
        stale_hint = metrics.get("wandb_state") or metrics.get("state")
        if stale_hint == "crashed":
            return "W&B stale"
        if heartbeat_at:
            age = short_age(heartbeat_at)
            if age.endswith("m ago"):
                try:
                    if int(age.split("m", 1)[0]) >= 5:
                        return "heartbeat stale"
                except ValueError:
                    return ""
    return ""


def job_from_train_row(row: dict[str, Any]) -> dict[str, Any]:
    config = dict(row.get("train_config") or {})
    metrics = dict(row.get("metrics_json") or {})
    status = str(row.get("status") or "")
    worker = str(row.get("lease_owner") or "")
    profile = str(row.get("profile_id") or "")
    device_key = infer_device_key("train", profile, worker, config)
    artifact_refs = row.get("artifact_refs") or []
    artifact = ""
    if artifact_refs:
        latest = artifact_refs[-1]
        if isinstance(latest, dict):
            location = str(latest.get("location") or "")
            artifact = "R2 ref" if location.startswith("s3://") else "W&B"
    return {
        "id": f"train-{row['id']}",
        "kind": "train",
        "target": target_label(config, fallback=str(row.get("goal_slug") or "")),
        "where": where_label(device_key, kind="train"),
        "device_key": device_key,
        "state": status,
        "progress": completion_progress(metrics),
        "attention": attention_for_row(
            status=status,
            error=row.get("error"),
            heartbeat_at=row.get("heartbeat_at"),
            lease_expires_at=row.get("lease_expires_at"),
            metrics=metrics,
        ),
        "details": {
            "goal": row.get("goal_slug") or "",
            "spec": row.get("spec_slug") or "",
            "profile": profile,
            "run": row.get("run_name") or "",
            "worker": worker,
            "lease": minutes_until(row.get("lease_expires_at")),
            "heartbeat": short_age(row.get("heartbeat_at")),
            "wandb": row.get("wandb_url") or "",
            "artifact": artifact,
            "fps": metric_value(metrics, "time/fps", THROUGHPUT_LOOP_FPS) or "",
            "completion": completion_progress(metrics),
        },
        "payload": payload_from_row(
            table="train_jobs",
            row=row,
            config_key="train_config",
            result_table="train_results",
        ),
    }


def job_from_eval_row(row: dict[str, Any]) -> dict[str, Any]:
    config = dict(row.get("eval_config") or {})
    metrics = dict(row.get("metrics_json") or {})
    status = str(row.get("status") or "")
    worker = str(row.get("lease_owner") or "")
    profile = str(row.get("profile_id") or "")
    device_key = infer_device_key("eval", profile, worker, config)
    progress = ""
    if status == "running" and config.get("episodes"):
        progress = f"0/{config['episodes']}"
    if metrics:
        progress = completion_progress(metrics) or progress
    return {
        "id": f"eval-{row['id']}",
        "kind": "eval",
        "target": row.get("candidate_label") or target_label(config),
        "where": where_label(device_key, kind="eval"),
        "device_key": device_key,
        "state": status,
        "progress": progress,
        "attention": attention_for_row(
            status=status,
            error=row.get("error"),
            heartbeat_at=row.get("heartbeat_at"),
            lease_expires_at=row.get("lease_expires_at"),
            metrics=metrics,
        ),
        "details": {
            "goal": row.get("goal_slug") or "",
            "profile": profile,
            "worker": worker,
            "lease": minutes_until(row.get("lease_expires_at")),
            "heartbeat": short_age(row.get("heartbeat_at")),
            "episodes": config.get("episodes") or "",
            "seed": config.get("seed") or "",
            "n_envs": config.get("n_envs") or "",
            "artifact": config.get("artifact_ref") or config.get("model_artifact") or "",
            "completion": completion_progress(metrics),
            "reward": metric_value(metrics, "reward_mean", "mean_reward") or "",
            "max_x": metric_value(metrics, "max_x_position_mean", "max_x") or "",
        },
        "payload": payload_from_row(
            table="eval_jobs",
            row=row,
            config_key="eval_config",
            result_table="eval_results",
        ),
    }


def campaign_jobs(options: MonitorOptions) -> tuple[list[dict[str, Any]], dict[str, str]]:
    if options.sample:
        return sample_jobs(), {"campaign": "sample", "message": "sample mode"}

    try:
        url = database_url(options.direct)
    except SystemExit as exc:
        return sample_jobs(), {"campaign": "sample", "message": str(exc)}

    goal_filter = "AND (%(goal)s IS NULL OR g.slug = %(goal)s)"
    params = {"goal": options.goal, "limit": options.limit}
    try:
        conn = connect(url)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT
                      j.*, g.slug AS goal_slug, s.slug AS spec_slug,
                      r.wandb_url, r.metrics_json, r.artifact_refs,
                      to_jsonb(j) AS job_payload,
                      to_jsonb(r) AS result_payload
                    FROM train_jobs j
                    JOIN research_goals g ON g.id = j.goal_id
                    LEFT JOIN experiment_specs s ON s.id = j.experiment_spec_id
                    LEFT JOIN train_results r ON r.train_job_id = j.id
                    WHERE j.status IN ('running', 'pending', 'failed')
                    {goal_filter}
                    ORDER BY
                      CASE j.status
                        WHEN 'running' THEN 0
                        WHEN 'pending' THEN 1
                        WHEN 'failed' THEN 2
                        ELSE 3
                      END,
                      j.priority DESC,
                      j.id DESC
                    LIMIT %(limit)s
                    """,
                    params,
                )
                train_rows = [dict(row) for row in cur.fetchall()]
                cur.execute(
                    f"""
                    SELECT
                      j.*, g.slug AS goal_slug,
                      r.metrics_json, r.model_ref, r.output_path, r.video_path,
                      to_jsonb(j) AS job_payload,
                      to_jsonb(r) AS result_payload
                    FROM eval_jobs j
                    JOIN research_goals g ON g.id = j.goal_id
                    LEFT JOIN eval_results r ON r.eval_job_id = j.id
                    WHERE j.status IN ('running', 'pending', 'failed')
                    {goal_filter}
                    ORDER BY
                      CASE j.status
                        WHEN 'running' THEN 0
                        WHEN 'pending' THEN 1
                        WHEN 'failed' THEN 2
                        ELSE 3
                      END,
                      j.priority DESC,
                      j.id DESC
                    LIMIT %(limit)s
                    """,
                    params,
                )
                eval_rows = [dict(row) for row in cur.fetchall()]
        finally:
            conn.close()
    except Exception as exc:
        return sample_jobs(), {"campaign": "sample", "message": f"DB unavailable: {exc}"}

    jobs = [job_from_train_row(row) for row in train_rows]
    jobs.extend(job_from_eval_row(row) for row in eval_rows)
    return jobs, {"campaign": "live", "message": f"{len(jobs)} active jobs"}


def sample_jobs() -> list[dict[str, Any]]:
    return [
        {
            "id": "train-184",
            "kind": "train",
            "target": "Mario L1 mixed",
            "where": "beast-3 / RTX4090",
            "device_key": "rtx4090",
            "state": "running",
            "progress": "42%",
            "attention": "W&B stale",
            "details": {
                "cluster": "beast-3",
                "pod": "rtx4090-head",
                "worker": "train-runner-2",
                "lease": "11m",
                "wandb": "crashed",
                "k8s": "alive",
                "artifact": "R2 ref",
                "fps": "1620",
            },
            "payload": {
                "table": "train_jobs",
                "schema": ["id", "profile_id", "train_config", "status"],
                "config_key": "train_config",
                "job": {
                    "id": 184,
                    "profile_id": "rtx4090-screening",
                    "train_config": {"game": "SuperMarioBros-Nes-v0"},
                    "status": "running",
                },
                "context": {"goal_slug": "sample"},
                "train_results": None,
            },
        },
        {
            "id": "eval-77",
            "kind": "eval",
            "target": "checkpoint v47",
            "where": "Modal CPU",
            "device_key": "modal",
            "state": "running",
            "progress": "68/100",
            "attention": "",
            "details": {"worker": "modal-eval", "episodes": 100, "seed": 10007, "n_envs": 20},
            "payload": {
                "table": "eval_jobs",
                "schema": ["id", "profile_id", "eval_config", "status"],
                "config_key": "eval_config",
                "job": {
                    "id": 77,
                    "profile_id": "mario-level1-quick",
                    "eval_config": {"episodes": 100},
                    "status": "running",
                },
                "context": {"goal_slug": "sample"},
                "eval_results": None,
            },
        },
        {
            "id": "train-185",
            "kind": "train",
            "target": "Mario L1-2",
            "where": "beast-3 / RTX4090",
            "device_key": "rtx4090",
            "state": "pending",
            "progress": "",
            "attention": "",
            "details": {"profile": "rtx4090-screening"},
        },
        {
            "id": "eval-78",
            "kind": "eval",
            "target": "seed81 best",
            "where": "local",
            "device_key": "local",
            "state": "pending",
            "progress": "",
            "attention": "",
            "details": {"profile": "mario-level1-quick"},
        },
        {
            "id": "train-181",
            "kind": "train",
            "target": "Mario L1-1",
            "where": "beast-2",
            "device_key": "rtx2060",
            "state": "failed",
            "progress": "",
            "attention": "check logs",
            "details": {"error": "train process exited 1"},
        },
    ]


def target_display(instance_name: str, instance: dict[str, Any]) -> str:
    kind = target_kind(instance)
    if kind == "skypilot":
        return str(instance.get("infra") or instance_name)
    if kind == "modal":
        return f"modal:{instance.get('modal_gpu') or instance.get('accelerator') or instance_name}"
    if kind == "local":
        return "local"
    return kind or instance_name


def capacity_label(instance: dict[str, Any]) -> str:
    if instance.get("available") is False:
        return "unavailable"
    kind = target_kind(instance)
    slots = instance.get("max_children") or instance.get("children")
    if kind == "modal":
        return f"{slots} launch" if slots else "on demand"
    if slots:
        return f"{slots} slots"
    return ""


def instance_details(instance_name: str, instance: dict[str, Any]) -> dict[str, Any]:
    aliases = instance.get("aliases") if isinstance(instance.get("aliases"), list) else []
    details: dict[str, Any] = {
        "target": instance_name,
        "provider": target_kind(instance),
        "aliases": ", ".join(str(alias) for alias in aliases),
        "accelerator": instance.get("accelerator") or "",
        "infra": instance.get("infra") or "",
        "modal_gpu": instance.get("modal_gpu") or "",
        "image": instance.get("image_id") or "",
        "cpu_shape": instance.get("cpu") or instance.get("cpus") or "",
        "memory_shape": instance.get("memory_mib") or instance.get("memory") or "",
        "n_envs": instance.get("n_envs") or "",
        "env_threads": instance.get("env_threads") if instance.get("env_threads") is not None else "",
        "torch_num_threads": instance.get("torch_num_threads")
        if instance.get("torch_num_threads") is not None
        else "",
        "expected_cost": instance.get("expected_hourly_cost") or "",
        "expected_fps": instance.get("expected_aggregate_wall_fps") or "",
    }
    if instance.get("available") is False:
        details["availability"] = instance.get("disabled_reason") or "unavailable"
    notes = instance.get("notes")
    if isinstance(notes, list) and notes:
        details["notes"] = " | ".join(str(note) for note in notes[:3])
    return details


def base_devices(repo_root: Path) -> list[dict[str, Any]]:
    instances = load_instances(repo_root).get("instances", {})
    if not isinstance(instances, dict):
        return []
    devices: list[dict[str, Any]] = []
    for instance_name, raw in instances.items():
        if not isinstance(raw, dict):
            continue
        instance = dict(raw)
        instance.setdefault("name", str(instance_name))
        aliases = instance.get("aliases") if isinstance(instance.get("aliases"), list) else []
        devices.append(
            {
                "id": str(instance_name),
                "aliases": [str(alias) for alias in aliases],
                "device": instance_label(instance),
                "target": target_display(str(instance_name), instance),
                "capacity": capacity_label(instance),
                "available": instance.get("available") is not False,
                "details": instance_details(str(instance_name), instance),
            }
        )
    return devices


def devices_from_jobs(
    repo_root: Path,
    jobs: list[dict[str, Any]],
    probes: Mapping[str, DeviceProbe] | None = None,
) -> list[dict[str, Any]]:
    devices = base_devices(repo_root)
    probes = probes or {}
    by_key = {device["id"]: device for device in devices}
    for device in devices:
        for alias in device.get("aliases") or []:
            by_key.setdefault(str(alias), device)
    for device in devices:
        device["current_jobs"] = []
        device["queued_jobs"] = []
        device["attention"] = ""
        device["usage"] = ""
        device["metrics"] = {}
        kind = str((device.get("details") or {}).get("provider") or "")
        if kind == "modal":
            device["last_check"] = "on demand"
        elif kind == "local":
            device["last_check"] = "local"
        else:
            device["last_check"] = "not probed"
        if not device.get("available", True):
            device["last_check"] = "unavailable"

    for job in jobs:
        key = job.get("device_key") or "local"
        device = by_key.get(str(key))
        if device is None:
            continue
        if job.get("state") in RUNNING_STATES:
            device["current_jobs"].append(job["id"])
            if job.get("attention") and not device["attention"]:
                device["attention"] = str(job["attention"])
        elif job.get("state") in QUEUED_STATES:
            device["queued_jobs"].append(job["id"])

    for device in devices:
        active_count = len(device["current_jobs"])
        state = "available"
        if not device.get("available", True):
            state = "unavailable"
        if active_count:
            state = "busy"
        if device["attention"]:
            state = "warning" if state == "busy" else state
        device["state"] = state
        device["current_job"] = ", ".join(device["current_jobs"])
        device["queued_job"] = ", ".join(device["queued_jobs"])
        if not device["attention"] and device["queued_jobs"]:
            device["attention"] = f"{len(device['queued_jobs'])} queued"
        if not device["attention"] and not device.get("available", True):
            device["attention"] = "unavailable"
        probe = probes.get(str(device["id"]))
        if probe is not None:
            if not probe.ok:
                if device["current_jobs"]:
                    state = "warning"
                else:
                    state = "offline"
                device["state"] = state
                device["attention"] = "unreachable"
                device["last_check"] = "unreachable"
            else:
                device["last_check"] = "reachable"
            metrics = probe.metrics or {}
            device["usage"] = usage_summary(metrics)
            device["metrics"] = resource_metrics(metrics)
        details = dict(device.get("details") or {})
        details.update(
            {
                "state": device["state"],
                "slots": device["capacity"],
                "running jobs": device["current_job"],
                "queued jobs": device["queued_job"],
                "attention": device["attention"],
            }
        )
        if probe is not None:
            details["reachability"] = "reachable" if probe.ok else "unreachable"
            details["health check"] = probe.label
            details["probe"] = probe.detail
            metrics = probe.metrics or {}
            if metrics:
                details["gpu"] = percent_text(metrics.get("gpu_util_pct"))
                details["vram"] = mib_text(
                    metrics.get("vram_used_mib"),
                    metrics.get("vram_total_mib"),
                )
                details["cpu"] = percent_text(metrics.get("cpu_util_pct"))
                details["memory"] = mib_text(
                    metrics.get("ram_used_mib"),
                    metrics.get("ram_total_mib"),
                )
        device["details"] = details
    return devices


def collect_state(options: MonitorOptions) -> dict[str, Any]:
    jobs, source = campaign_jobs(options)
    probe_keys = [str(device["id"]) for device in base_devices(options.repo_root)]
    probes = {} if options.sample else live_device_probes(probe_keys)
    devices = devices_from_jobs(options.repo_root, jobs, probes)
    return json_safe(
        {
            "refreshed_at": utc_now_iso(),
            "goal": options.goal or "all goals",
            "source": source,
            "jobs": jobs,
            "devices": devices,
        }
    )


def state_from_env(
    repo_root: Path,
    *,
    goal: str | None = None,
    direct: bool = False,
    sample: bool | None = None,
) -> dict[str, Any]:
    if sample is None:
        sample = os.environ.get("STABLE_RETRO_MONITOR_SAMPLE") == "1"
    return collect_state(
        MonitorOptions(repo_root=repo_root, goal=goal, direct=direct, sample=sample)
    )
