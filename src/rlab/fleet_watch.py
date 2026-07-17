from __future__ import annotations

import copy
import json
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Group
from rich.padding import Padding
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Static

from rlab.fleet_service import ServicePaths, controller_services_status, service_status


WATCH_SCHEMA_VERSION = 1
PLAIN_HEARTBEAT_SECONDS = 300.0
RECENT_ACTIVITY_HOURS = 24


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _iso_utc(value: datetime | None = None) -> str:
    return (value or _utc_now()).isoformat().replace("+00:00", "Z")


def _parse_time(value: object) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone(UTC)


def _age_seconds(value: object, *, now: datetime) -> float | None:
    parsed = _parse_time(value)
    return None if parsed is None else max(0.0, (now - parsed).total_seconds())


def _load_json(path: Path, errors: list[str]) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{path.name}: {type(exc).__name__}")
        return None
    if not isinstance(value, dict):
        errors.append(f"{path.name}: expected a JSON object")
        return None
    return value


def _recent_activity(path: Path, *, now: datetime, errors: list[str]) -> list[dict[str, Any]]:
    cutoff = now - timedelta(hours=RECENT_ACTIVITY_HOURS)
    records: list[dict[str, Any]] = []
    paths = [path.with_name(f"{path.name}.{index}") for index in range(3, 0, -1)] + [path]
    for candidate in paths:
        try:
            lines = candidate.read_text(encoding="utf-8", errors="replace").splitlines()
        except FileNotFoundError:
            continue
        except OSError as exc:
            errors.append(f"{candidate.name}: {type(exc).__name__}")
            continue
        for line in lines[-500:]:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            timestamp = _parse_time(record.get("timestamp"))
            if timestamp is None or timestamp < cutoff:
                continue
            event = str(record.get("event") or "")
            if event == "work_changed":
                for change in record.get("changes") or []:
                    if isinstance(change, Mapping):
                        records.append(
                            {
                                "timestamp": record.get("timestamp"),
                                "entity": str(change.get("entity") or "scheduler"),
                                "title": str(change.get("title") or "Work state changed"),
                                "detail": str(change.get("detail") or ""),
                            }
                        )
            elif event == "pass_finished" and str(record.get("status")) not in {"idle", "ok"}:
                records.append(
                    {
                        "timestamp": record.get("timestamp"),
                        "entity": "scheduler",
                        "title": f"Pass finished {record.get('status')}",
                        "detail": str(record.get("error") or "One or more lanes did not complete"),
                    }
                )
    records.sort(key=lambda row: str(row.get("timestamp") or ""), reverse=True)
    return records[:5]


def _lane_items(current_pass: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not current_pass:
        return []
    lanes = current_pass.get("lanes") or {}
    if isinstance(lanes, Mapping):
        values = lanes.values()
    elif isinstance(lanes, Sequence) and not isinstance(lanes, str | bytes):
        values = lanes
    else:
        values = ()
    items: list[dict[str, Any]] = []
    for lane in values:
        if not isinstance(lane, Mapping):
            continue
        entity_kind = str(lane.get("entity_kind") or "")
        entity_id = str(lane.get("entity_id") or "")
        entity = (
            f"{entity_kind}/{entity_id}"
            if entity_kind and entity_id
            else str(lane.get("lane") or "scheduler")
        )
        items.append(
            {
                "id": str(lane.get("lane") or entity),
                "entity": entity,
                "title": str(lane.get("action") or "RECONCILING").upper(),
                "detail": str(lane.get("reason") or "Scheduler pass in progress"),
                "started_at": lane.get("started_at"),
                "deadline_at": lane.get("deadline_at"),
                "resolution": "automatic",
                "blast_radius": str(lane.get("blast_radius") or entity),
                "command": str(lane.get("command") or ""),
            }
        )
    return sorted(items, key=lambda row: (str(row["entity"]), str(row["id"])))


def collect_watch_snapshot(
    paths: ServicePaths,
    *,
    runner: Callable[..., Any] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    from rlab.fleet_service import _run_command

    captured = now or _utc_now()
    controller_status = controller_services_status(paths, runner=runner or _run_command)
    if controller_status["installed"]:
        return _collect_authoritative_snapshot(paths, controller_status, captured=captured)
    errors: list[str] = []
    status = service_status(paths, runner=runner or _run_command)
    current_pass = _load_json(paths.state_dir / "current-pass.json", errors)
    last_pass = _load_json(paths.last_pass, errors) or status.get("last_pass") or {}
    workload = dict((current_pass or {}).get("workload") or last_pass.get("workload") or {})
    needs_action = list(workload.get("needs_action") or [])
    retrying = list(workload.get("retrying") or [])
    waiting = list(workload.get("waiting") or [])
    now_items = _lane_items(current_pass)
    active_work = [dict(item) for item in workload.get("active") or []]
    now_items = list(
        {
            str(item.get("id") or f"{item.get('entity')}:{item.get('title')}"): item
            for item in [*now_items, *active_work]
        }.values()
    )

    deadline = _parse_time((current_pass or {}).get("deadline_at"))
    updated_at = (current_pass or {}).get("updated_at") or last_pass.get("finished_at")
    freshness_age = _age_seconds(updated_at, now=captured)
    if current_pass:
        if not status.get("loaded") or not status.get("running"):
            scheduler_state = "INTERRUPTED"
        elif deadline is not None and captured > deadline:
            scheduler_state = "STUCK"
        else:
            scheduler_state = "RECONCILING"
    elif status.get("last_pass_stale"):
        scheduler_state = "LAST KNOWN"
    else:
        scheduler_state = "IDLE"

    if not status.get("installed"):
        service_state = "UNINSTALLED"
    elif not status.get("loaded"):
        service_state = "STOPPED"
    elif scheduler_state in {"STUCK", "INTERRUPTED", "LAST KNOWN"}:
        service_state = "DEGRADED"
    elif (status.get("health") or {}).get("alert_active") or str(last_pass.get("status")) in {
        "degraded",
        "error",
    }:
        service_state = "DEGRADED"
    else:
        service_state = "HEALTHY"

    source = dict((current_pass or {}).get("source") or last_pass.get("source") or {})
    recent = _recent_activity(paths.service_log, now=captured, errors=errors)
    if not recent:
        recent = list(last_pass.get("meaningful_changes") or [])[:5]

    in_progress = int(workload.get("in_progress") or len(now_items))
    work = {
        "in_progress": in_progress,
        "waiting": len(waiting),
        "retrying": len(retrying),
        "needs_action": len(needs_action),
        "counts": dict(workload.get("counts") or {}),
    }
    return {
        "schema_version": WATCH_SCHEMA_VERSION,
        "captured_at": _iso_utc(captured),
        "data_freshness": {
            "source_at": updated_at,
            "age_seconds": freshness_age,
            "errors": errors,
            "stale": bool(status.get("last_pass_stale") and not current_pass),
        },
        "service": {
            "label": paths.label,
            "state": service_state,
            "installed": bool(status.get("installed")),
            "loaded": bool(status.get("loaded")),
            "running": bool(status.get("running")),
            "interval_seconds": status.get("interval_seconds"),
            "last_pass_status": status.get("last_pass_status"),
            "consecutive_failures": int(
                (status.get("health") or {}).get("consecutive_failures") or 0
            ),
        },
        "scheduler": {
            "state": scheduler_state,
            "pass_id": (current_pass or last_pass).get("pass_id"),
            "started_at": (current_pass or last_pass).get("started_at"),
            "deadline_at": (current_pass or {}).get("deadline_at"),
            "triggers": list((current_pass or last_pass).get("triggers") or []),
            "source": source,
        },
        "work": work,
        "needs_action": needs_action,
        "retrying": retrying,
        "now": now_items,
        "waiting": waiting,
        "recent_changes": recent,
    }


def _collect_authoritative_snapshot(
    paths: ServicePaths,
    controller_status: Mapping[str, Any],
    *,
    captured: datetime,
) -> dict[str, Any]:
    """Build the dashboard from read-only PostgreSQL state under the split controllers."""

    from rlab.fleet_service import _connect_queue

    errors: list[str] = []
    now_items: list[dict[str, Any]] = []
    waiting: list[dict[str, Any]] = []
    retrying: list[dict[str, Any]] = []
    needs_action: list[dict[str, Any]] = []
    recent: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    source_at: object = None
    projections: dict[int, dict[str, Any]] = {}
    try:
        conn = _connect_queue(paths.repo_root)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT t.id, t.status, t.machine, t.created_at, t.started_at,
                      t.finished_at, t.error,
                      r.status AS eval_status, r.outcome AS eval_outcome,
                      r.updated_at
                    FROM train_jobs t
                    LEFT JOIN eval_runs r ON r.train_job_id = t.id
                    WHERE t.status IN ('pending', 'launching', 'starting', 'running', 'finalizing')
                       OR (
                         t.status = 'finalization_failed'
                         AND coalesce(
                           t.finished_at, r.updated_at, t.process_exited_at,
                           t.started_at, t.created_at
                         ) >= now() - interval '24 hours'
                       )
                    ORDER BY t.id
                    """
                )
                jobs = [dict(row) for row in cur.fetchall()]
                from rlab.run_observability import run_projection

                projections = {int(row["id"]): run_projection(conn, int(row["id"])) for row in jobs}
                cur.execute(
                    """
                    SELECT j.id, j.train_job_id, j.status, j.purpose, j.checkpoint_step,
                      j.created_at, j.updated_at, j.error,
                      count(a.id) FILTER (WHERE a.status IN ('failed', 'expired')) AS failures
                    FROM eval_jobs j
                    JOIN train_jobs t ON t.id = j.train_job_id
                    LEFT JOIN eval_runs r ON r.train_job_id = j.train_job_id
                    LEFT JOIN eval_attempts a ON a.eval_job_id = j.id
                    WHERE (
                      j.status IN ('pending', 'dispatching', 'submitted', 'blocked_budget')
                      AND t.status IN ('pending', 'launching', 'starting', 'running', 'finalizing')
                    ) OR (
                      j.status = 'failed'
                      AND t.status = 'finalizing'
                      AND r.outcome IS NULL
                    )
                    GROUP BY j.id, t.id, r.train_job_id
                    ORDER BY j.updated_at, j.id
                    """
                )
                eval_jobs = [dict(row) for row in cur.fetchall()]
                cur.execute(
                    """
                    SELECT e.job_id, e.event_type, e.message, e.created_at
                    FROM job_events e
                    ORDER BY e.id DESC
                    LIMIT 5
                    """
                )
                events = [dict(row) for row in cur.fetchall()]
        finally:
            conn.close()
    except Exception as exc:
        errors.append(f"postgres: {type(exc).__name__}: {exc}")
        jobs = []
        eval_jobs = []
        events = []

    for row in jobs:
        projection = projections.get(int(row["id"]), {})
        status = str(projection.get("status") or row["status"])
        evaluation = dict(projection.get("evaluation") or {})
        source_at = max(
            [
                value
                for value in (
                    source_at,
                    row.get("updated_at"),
                    row.get("finished_at"),
                    row.get("started_at"),
                    row.get("created_at"),
                )
                if value is not None
            ],
            default=source_at,
        )
        counts[status] = counts.get(status, 0) + 1
        item = {
            "id": f"train/{row['id']}",
            "entity": f"train/{row['id']}",
            "title": status.replace("_", " ").upper(),
            "detail": (
                f"machine={row['machine']} · eval={evaluation.get('status') or row.get('eval_status') or 'not created'}"
                + (
                    f" · outcome={evaluation.get('outcome') or row.get('eval_outcome')}"
                    if evaluation.get("outcome") or row.get("eval_outcome")
                    else ""
                )
            ),
            "started_at": row.get("started_at") or row.get("created_at"),
            "resolution": "automatic" if status != "finalization_failed" else "manual inspection",
            "blast_radius": f"train/{row['id']}",
        }
        if status == "pending":
            waiting.append(item)
        elif status != "finalization_failed":
            now_items.append(item)
        current_incidents = list((projection.get("incidents") or {}).get("current") or [])
        if status == "finalization_failed" and not current_incidents:
            current_incidents = [
                {
                    "fingerprint": f"terminal:{row['id']}",
                    "category": "terminal_operational_failure",
                    "detail": str(row.get("error") or item["detail"]),
                }
            ]
        for incident in current_incidents:
            category = str(incident.get("category") or "potential_bug")
            incident_item = {
                **item,
                "id": str(incident.get("fingerprint") or item["id"]),
                "title": category.replace("_", " ").title(),
                "detail": str(incident.get("detail") or item["detail"]),
                "resolution": "manual inspection",
            }
            if category in {
                "wandb_publication_retry",
                "artifact_projection_retry",
            } and status not in {"failed", "finalization_failed", "canceled"}:
                incident_item["resolution"] = "automatic"
                retrying.append(incident_item)
            else:
                if category in {
                    "wandb_publication_retry",
                    "artifact_projection_retry",
                    "terminal_operational_failure",
                }:
                    from rlab.cli_commands import (
                        experiment_retry_finalization_command,
                        render_command,
                    )

                    incident_item["command"] = render_command(
                        experiment_retry_finalization_command(int(row["id"]))
                    )
                needs_action.append(incident_item)

    for row in eval_jobs:
        status = str(row["status"])
        source_at = max(
            [
                value
                for value in (source_at, row.get("updated_at"), row.get("created_at"))
                if value is not None
            ],
            default=source_at,
        )
        item = {
            "id": f"eval/{row['id']}",
            "entity": f"eval/{row['id']}",
            "title": f"{row['purpose']} {status}".upper(),
            "detail": f"train/{row['train_job_id']} · checkpoint {int(row['checkpoint_step']):,}",
            "started_at": row.get("created_at"),
            "resolution": "automatic" if status != "failed" else "manual inspection",
            "blast_radius": f"train/{row['train_job_id']}",
        }
        if status == "failed":
            item["detail"] = str(row.get("error") or item["detail"])
            needs_action.append(item)
        elif int(row.get("failures") or 0) > 0:
            retrying.append(item)
        elif status in {"pending", "blocked_budget"}:
            waiting.append(item)
        else:
            now_items.append(item)

    for row in events:
        recent.append(
            {
                "id": f"event/{row['job_id']}/{row['event_type']}",
                "entity": f"train/{row['job_id']}",
                "title": str(row["event_type"]).replace("_", " ").title(),
                "detail": str(row.get("message") or ""),
                "timestamp": row.get("created_at"),
            }
        )
    loaded = bool(controller_status.get("loaded"))
    running = bool(controller_status.get("running"))
    healthy = bool(controller_status.get("healthy")) and not errors
    scheduler_state = "RECONCILING" if now_items else "IDLE"
    service_state = "HEALTHY" if healthy else "DEGRADED"
    return {
        "schema_version": WATCH_SCHEMA_VERSION,
        "captured_at": _iso_utc(captured),
        "data_freshness": {
            "source_at": source_at,
            "age_seconds": _age_seconds(source_at, now=captured),
            "errors": errors,
            "stale": bool(errors),
        },
        "service": {
            "label": paths.label,
            "state": service_state,
            "installed": True,
            "loaded": loaded,
            "running": running,
            "interval_seconds": controller_status.get("poll_seconds"),
            "last_pass_status": None,
            "consecutive_failures": 0,
        },
        "scheduler": {
            "state": scheduler_state,
            "pass_id": None,
            "started_at": None,
            "deadline_at": None,
            "triggers": [],
            "source": {},
        },
        "work": {
            "in_progress": len(now_items),
            "waiting": len(waiting),
            "retrying": len(retrying),
            "needs_action": len(needs_action),
            "counts": counts,
        },
        "needs_action": needs_action,
        "retrying": retrying,
        "now": now_items,
        "waiting": waiting,
        "recent_changes": recent,
    }


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, remainder = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {remainder:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def _item_age(item: Mapping[str, Any], now: datetime) -> str:
    value = item.get("started_at") or item.get("timestamp")
    parsed = _parse_time(value)
    if parsed is not None and parsed > now:
        return f"in {_duration((parsed - now).total_seconds())}"
    return _duration(_age_seconds(value, now=now))


def _flatten_items(snapshot: Mapping[str, Any]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for category in ("needs_action", "now", "retrying", "waiting", "recent_changes"):
        for item in snapshot.get(category) or []:
            if isinstance(item, Mapping):
                flattened.append({"category": category, **dict(item)})
    return flattened


def render_plain_snapshot(snapshot: Mapping[str, Any]) -> str:
    service = snapshot["service"]
    scheduler = snapshot["scheduler"]
    work = snapshot["work"]
    lines = [
        f"FLEET SCHEDULER {snapshot['captured_at']} READ ONLY",
        f"SERVICE {service['state']} installed={str(service['installed']).lower()} loaded={str(service['loaded']).lower()}",
        f"SCHEDULER {scheduler['state']} pass={scheduler.get('pass_id') or 'none'}",
        (
            f"WORK {work['in_progress']} in progress · {work['waiting']} waiting · "
            f"{work['retrying']} retrying · {work['needs_action']} need action"
        ),
    ]
    headings = (
        ("needs_action", "NEEDS ATTENTION"),
        ("now", "ACTIVE NOW"),
        ("retrying", "RETRYING"),
        ("waiting", "WAITING"),
        ("recent_changes", "RECENT ACTIVITY"),
    )
    for key, title in headings:
        items = snapshot.get(key) or []
        if not items:
            continue
        lines.append("")
        lines.append(title)
        for item in items:
            entity = str(item.get("entity") or item.get("id") or "scheduler")
            detail = str(item.get("detail") or "")
            resolution = str(item.get("resolution") or "")
            suffix = f" · {resolution}" if resolution else ""
            lines.append(f"  {entity}  {item.get('title') or 'State changed'}{suffix}")
            if detail:
                lines.append(f"    {detail}")
    errors = snapshot.get("data_freshness", {}).get("errors") or []
    if errors:
        lines.extend(("", "STATE READ ERRORS", *(f"  {error}" for error in errors)))
    return "\n".join(lines)


def _semantic_fingerprint(snapshot: Mapping[str, Any]) -> str:
    value = copy.deepcopy(dict(snapshot))
    value.pop("captured_at", None)
    freshness = value.get("data_freshness") or {}
    freshness.pop("age_seconds", None)
    return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))


def stream_plain_watch(
    loader: Callable[[], dict[str, Any]],
    *,
    heartbeat_seconds: float = PLAIN_HEARTBEAT_SECONDS,
    poll_seconds: float = 0.25,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    max_iterations: int | None = None,
) -> int:
    previous = ""
    last_printed = float("-inf")
    iterations = 0
    try:
        while max_iterations is None or iterations < max_iterations:
            snapshot = loader()
            fingerprint = _semantic_fingerprint(snapshot)
            now = clock()
            if fingerprint != previous or now - last_printed >= heartbeat_seconds:
                if previous:
                    print()
                print(render_plain_snapshot(snapshot), flush=True)
                previous = fingerprint
                last_printed = now
            iterations += 1
            if max_iterations is None or iterations < max_iterations:
                sleep(poll_seconds)
    except KeyboardInterrupt:
        return 130
    return 0


@contextmanager
def _temporary_no_color(enabled: bool):
    previous = os.environ.get("NO_COLOR")
    if enabled:
        os.environ["NO_COLOR"] = "1"
    try:
        yield
    finally:
        if enabled:
            if previous is None:
                os.environ.pop("NO_COLOR", None)
            else:
                os.environ["NO_COLOR"] = previous


def _status_color(state: str, monochrome: bool) -> str:
    if monochrome:
        return "bold"
    if state in {"HEALTHY", "IDLE"}:
        return "bold #64d98b"
    if state in {"RECONCILING"}:
        return "bold #59d5e0"
    if state in {"LAST KNOWN"}:
        return "bold #e5b567"
    return "bold #ef6f6c"


def _summary_renderable(snapshot: Mapping[str, Any], width: int, monochrome: bool) -> Any:
    service = snapshot["service"]
    scheduler = snapshot["scheduler"]
    work = snapshot["work"]
    source = scheduler.get("source") or {}
    badges: list[str] = ["READ ONLY"]
    if source.get("dirty"):
        badges.append("SOURCE DIRTY")
    if snapshot.get("data_freshness", {}).get("stale"):
        badges.append("STALE DATA")
    title = Text("FLEET SCHEDULER", style="bold #e7edf2" if not monochrome else "bold")
    title.append("  ")
    title.append(datetime.now().astimezone().strftime("%H:%M:%S"), style="dim")
    title.append("  ·  ")
    title.append(
        " · ".join(badges), style="bold #e5b567" if len(badges) > 1 and not monochrome else "dim"
    )

    if width < 60:
        compact = Text()
        compact.append(
            f"SERVICE {service['state']}  ", style=_status_color(service["state"], monochrome)
        )
        compact.append(
            f"SCHEDULER {scheduler['state']}\n", style=_status_color(scheduler["state"], monochrome)
        )
        compact.append(
            f"{work['in_progress']} active · {work['waiting']} wait · "
            f"{work['retrying']} retry · {work['needs_action']} alert"
        )
        return Group(title, compact)

    table = Table.grid(expand=True, padding=(0, 2))
    for _ in range(3):
        table.add_column(ratio=1)
    service_text = Text("SERVICE\n", style="dim")
    service_text.append(
        str(service["state"]), style=_status_color(str(service["state"]), monochrome)
    )
    service_text.append(
        f"\nlaunchd {'loaded' if service['loaded'] else 'not loaded'}",
        style="dim",
    )
    scheduler_text = Text("SCHEDULER\n", style="dim")
    scheduler_text.append(
        str(scheduler["state"]), style=_status_color(str(scheduler["state"]), monochrome)
    )
    scheduler_text.append(f"\npass {scheduler.get('pass_id') or '—'}", style="dim")
    work_text = Text("WORK\n", style="dim")
    work_text.append(f"{work['in_progress']} in progress", style="bold")
    work_text.append(
        f"\n{work['waiting']} waiting · {work['retrying']} retrying · {work['needs_action']} alert",
        style="dim",
    )
    table.add_row(service_text, scheduler_text, work_text)
    return Panel(
        table,
        title=title,
        title_align="left",
        box=box.ROUNDED,
        border_style="grey35",
        padding=(0, 1),
    )


def _section_panel(
    title: str,
    items: Sequence[Mapping[str, Any]],
    *,
    category: str,
    selected_index: int,
    offset: int,
    now: datetime,
    monochrome: bool,
    width: int,
    maximum: int,
) -> tuple[Any | None, int]:
    if not items:
        return None, offset
    visible = list(items[:maximum])
    table = Table.grid(expand=True, padding=(0, 1))
    compact_columns = width < 55
    table.add_column(
        width=min(18, max(10, width // 5 if compact_columns else width // 7)), no_wrap=True
    )
    table.add_column(ratio=1)
    if not compact_columns:
        table.add_column(width=9, justify="right", no_wrap=True)
    accent = {
        "needs_action": "#ef6f6c",
        "now": "#59d5e0",
        "retrying": "#e5b567",
        "waiting": "#a7b0b8",
        "recent_changes": "#7aa2c7",
    }.get(category, "grey70")
    for index, item in enumerate(visible):
        selected = offset + index == selected_index
        selection_style = "bold underline" if selected else "bold"
        marker_style = "bold" if monochrome else accent
        entity = Text("› " if selected else "  ", style=marker_style)
        entity.append(
            str(item.get("entity") or item.get("id") or "scheduler"),
            style=selection_style,
        )
        body = Text(str(item.get("title") or "State changed"), style=selection_style)
        detail = str(item.get("detail") or "")
        resolution = str(item.get("resolution") or "")
        if detail:
            body.append(f"\n{detail}", style="dim underline" if selected else "dim")
        if resolution:
            body.append(f" · {resolution}", style="dim underline" if selected else "dim")
        age = Text(_item_age(item, now), style="underline" if selected else "dim")
        if compact_columns:
            table.add_row(entity, body)
        else:
            table.add_row(entity, body, age)
    if len(items) > len(visible):
        more = ("", Text(f"+{len(items) - len(visible)} more", style="dim"))
        table.add_row(*more) if compact_columns else table.add_row(*more, "")
    heading = f"{title} · {len(items)}" if category == "needs_action" else title
    if width < 80:
        result: Any = Group(Text(heading, style="bold"), Padding(table, (0, 0, 1, 0)))
    else:
        result = Panel(
            table,
            title=Text(heading, style="bold"),
            title_align="left",
            box=box.ROUNDED,
            border_style="grey35" if monochrome else accent,
            padding=(0, 1),
        )
    return result, offset + len(items)


def _content_renderable(
    snapshot: Mapping[str, Any], width: int, selected_index: int, monochrome: bool
) -> Any:
    now = _utc_now()
    offset = 0
    side_width = max(40, int(width * 0.35) - 3) if width >= 120 else width
    attention, offset = _section_panel(
        "NEEDS ATTENTION",
        snapshot.get("needs_action") or [],
        category="needs_action",
        selected_index=selected_index,
        offset=offset,
        now=now,
        monochrome=monochrome,
        width=width,
        maximum=4,
    )
    active, offset = _section_panel(
        "ACTIVE NOW",
        snapshot.get("now") or [],
        category="now",
        selected_index=selected_index,
        offset=offset,
        now=now,
        monochrome=monochrome,
        width=width,
        maximum=6,
    )
    retrying, offset = _section_panel(
        "RETRYING",
        snapshot.get("retrying") or [],
        category="retrying",
        selected_index=selected_index,
        offset=offset,
        now=now,
        monochrome=monochrome,
        width=side_width,
        maximum=5,
    )
    waiting, offset = _section_panel(
        "WAITING",
        snapshot.get("waiting") or [],
        category="waiting",
        selected_index=selected_index,
        offset=offset,
        now=now,
        monochrome=monochrome,
        width=side_width,
        maximum=5,
    )
    recent, _offset = _section_panel(
        "RECENT ACTIVITY",
        snapshot.get("recent_changes") or [],
        category="recent_changes",
        selected_index=selected_index,
        offset=offset,
        now=now,
        monochrome=monochrome,
        width=width,
        maximum=5,
    )
    parts: list[Any] = []
    if attention is not None:
        parts.append(attention)
    if width >= 120:
        side_parts = [part for part in (retrying, waiting) if part is not None]
        if active is not None or side_parts:
            grid = Table.grid(expand=True, padding=(0, 1))
            grid.add_column(ratio=65)
            grid.add_column(ratio=35)
            side = Panel(
                Group(*side_parts) if side_parts else Text("Queue clear", style="dim"),
                title=Text("QUEUE & RETRIES", style="bold"),
                title_align="left",
                box=box.ROUNDED,
                border_style="grey35" if monochrome else "#e5b567",
                padding=(0, 1),
            )
            grid.add_row(
                active or Text("No active scheduler phases", style="dim"),
                side,
            )
            parts.append(grid)
    else:
        parts.extend(part for part in (active, retrying, waiting) if part is not None)
    if recent is not None:
        parts.append(recent)
    if not parts:
        parts.append(
            Panel(
                Text("No queued or active work", style="dim"),
                box=box.ROUNDED,
                border_style="grey35",
            )
        )
    return Group(*parts)


class _InfoScreen(ModalScreen[None]):
    BINDINGS = [Binding("escape", "dismiss", "Close"), Binding("q", "dismiss", "Close")]

    DEFAULT_CSS = """
    _InfoScreen { align: center middle; background: rgba(8, 12, 16, 0.75); }
    _InfoScreen > VerticalScroll {
        width: 80%; max-width: 100; height: 80%;
        background: #111820; border: round #52606d; padding: 1 2;
    }
    """

    def __init__(self, title: str, body: Any) -> None:
        super().__init__()
        self.title_text = title
        self.body = body

    def compose(self) -> ComposeResult:
        content = Group(Text(self.title_text, style="bold"), Text(""), self.body)
        with VerticalScroll():
            yield Static(content)

    def action_dismiss(self) -> None:
        self.dismiss()


class FleetWatchApp(App[int]):
    TITLE = "rlab fleet scheduler"
    ENABLE_COMMAND_PALETTE = False
    BINDINGS = [
        Binding("q", "quit_watch", "Quit"),
        Binding("up,k", "previous_item", "Previous", show=False),
        Binding("down,j", "next_item", "Next", show=False),
        Binding("enter,d", "details", "Details"),
        Binding("space", "freeze", "Freeze"),
        Binding("?", "help", "Help"),
    ]
    CSS = """
    Screen { background: #0b1015; color: #dce4eb; }
    #summary { height: auto; padding: 0 1; }
    #frozen { display: none; height: 1; text-align: center; background: #7a581d; color: #fff4d6; text-style: bold; }
    #frozen.visible { display: block; }
    #viewport { height: 1fr; padding: 0 1; scrollbar-color: #52606d; scrollbar-background: #111820; }
    #content { width: 100%; height: auto; }
    #keys { dock: bottom; height: 1; color: #94a3af; background: #0f161d; text-align: center; }
    """

    def __init__(
        self,
        loader: Callable[[], dict[str, Any]],
        *,
        initial: dict[str, Any] | None = None,
        monochrome: bool = False,
    ) -> None:
        super().__init__()
        self.loader = loader
        self.snapshot = initial or loader()
        self.monochrome = monochrome
        self.frozen = False
        self.selected_index = 0

    def compose(self) -> ComposeResult:
        yield Static(id="summary")
        yield Static("DISPLAY FROZEN · scheduler still running", id="frozen")
        with VerticalScroll(id="viewport"):
            yield Static(id="content")
        yield Static("↑↓/jk navigate   enter/d details   space freeze   ? help   q quit", id="keys")

    def on_mount(self) -> None:
        self.set_interval(1.0, self._tick)
        self._render()

    def on_resize(self, _event: events.Resize) -> None:
        self._render()

    def _tick(self) -> None:
        if not self.frozen:
            self.snapshot = self.loader()
            self._clamp_selection()
        self._render()

    def _clamp_selection(self) -> None:
        count = len(_flatten_items(self.snapshot))
        self.selected_index = min(max(0, self.selected_index), max(0, count - 1))

    def _render(self) -> None:
        if not self.is_mounted:
            return
        width = max(20, self.size.width)
        self.query_one("#summary", Static).update(
            _summary_renderable(self.snapshot, width, self.monochrome)
        )
        self.query_one("#content", Static).update(
            _content_renderable(self.snapshot, width, self.selected_index, self.monochrome)
        )
        self.query_one("#frozen", Static).set_class(self.frozen, "visible")

    def action_quit_watch(self) -> None:
        self.exit(0)

    def action_previous_item(self) -> None:
        self.selected_index = max(0, self.selected_index - 1)
        self._render()

    def action_next_item(self) -> None:
        count = len(_flatten_items(self.snapshot))
        if count:
            self.selected_index = min(count - 1, self.selected_index + 1)
        self._render()

    def action_freeze(self) -> None:
        self.frozen = not self.frozen
        self._render()

    def action_details(self) -> None:
        items = _flatten_items(self.snapshot)
        if not items:
            return
        item = items[self.selected_index]
        table = Table.grid(padding=(0, 1))
        table.add_column(style="dim", no_wrap=True)
        table.add_column(ratio=1)
        for key in (
            "entity",
            "title",
            "detail",
            "resolution",
            "blast_radius",
            "started_at",
            "deadline_at",
            "timestamp",
            "command",
        ):
            if item.get(key):
                table.add_row(key.replace("_", " ").title(), str(item[key]))
        self.push_screen(_InfoScreen("DETAILS · READ ONLY", table))

    def action_help(self) -> None:
        body = Text(
            "↑/↓ or j/k  Select a row\n"
            "Enter or d  Open full details\n"
            "Space       Freeze or resume display updates\n"
            "Esc         Close an overlay\n"
            "q           Quit\n\n"
            "This watcher never changes scheduler, queue, machine, or evaluation state."
        )
        self.push_screen(_InfoScreen("FLEET WATCH HELP", body))


def run_watch_tui(loader: Callable[[], dict[str, Any]], *, monochrome: bool = False) -> int:
    try:
        with _temporary_no_color(monochrome):
            result = FleetWatchApp(loader, monochrome=monochrome).run()
    except KeyboardInterrupt:
        return 130
    return int(result or 0)


def run_watch_command(args: Any) -> int:
    from rlab.fleet_service import _paths_from_args

    paths = _paths_from_args(args)

    def loader() -> dict[str, Any]:
        return collect_watch_snapshot(paths)

    if bool(args.json):
        snapshot = loader()
        print(json.dumps(snapshot, sort_keys=True, default=str))
        return 1 if snapshot["data_freshness"]["errors"] else 0
    if bool(args.once):
        snapshot = loader()
        print(render_plain_snapshot(snapshot))
        return 1 if snapshot["data_freshness"]["errors"] else 0
    if bool(args.plain):
        return stream_plain_watch(loader)
    if not sys.stdout.isatty() or os.environ.get("TERM", "").lower() == "dumb":
        snapshot = loader()
        print(render_plain_snapshot(snapshot))
        return 1 if snapshot["data_freshness"]["errors"] else 0
    return run_watch_tui(loader, monochrome=bool(args.no_color or os.environ.get("NO_COLOR")))
