from __future__ import annotations

import re
import shutil
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from rlab.fleet_labels import JOB_ID_LABEL
from rlab.machines import MachineConfig


@dataclass(frozen=True)
class MachineWatchSnapshot:
    captured_at: datetime
    machine: MachineConfig
    containers: tuple[Any, ...]
    launches: tuple[dict[str, Any], ...]
    queue_counts: Mapping[str, Mapping[str, int]]
    result_present: Mapping[str, bool]
    warnings: tuple[str, ...] = ()


ANSI_STYLES = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "gray": "\033[90m",
    "red": "\033[31m",
    "bright_red": "\033[1;31m",
    "green": "\033[32m",
    "bright_green": "\033[1;32m",
    "yellow": "\033[33m",
    "bright_yellow": "\033[1;33m",
    "blue": "\033[34m",
    "bright_blue": "\033[1;34m",
    "magenta": "\033[35m",
    "bright_magenta": "\033[1;35m",
    "cyan": "\033[36m",
    "bright_cyan": "\033[1;36m",
    "white": "\033[37m",
    "orange": "\033[38;5;208m",
}


def format_utc_second(value: Any | None = None) -> str:
    if value is None:
        value = datetime.now(UTC)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return "unknown"
        try:
            value = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return text
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    return str(value)


def colorize(text: str, style: str, *, enabled: bool) -> str:
    if not enabled:
        return text
    return f"{ANSI_STYLES[style]}{text}{ANSI_STYLES['reset']}"


def dashboard_divider(width: int, *, color: bool) -> str:
    return colorize("-" * min(width, 120), "gray", enabled=color)


def dashboard_chip(label: str, value: str, style: str, *, color: bool) -> str:
    return f"{label}={colorize(value, style, enabled=color)}"


def section_label(text: str, style: str, *, color: bool) -> str:
    return colorize(text, style, enabled=color)


def numbered_section(number: int, text: str, style: str, *, color: bool) -> str:
    prefix = colorize(f"{number}", "bright_red", enabled=color)
    return f"{prefix}{section_label(text, style, color=color)}"


def heat_style(ratio: float) -> str:
    if ratio >= 0.9:
        return "bright_red"
    if ratio >= 0.75:
        return "orange"
    if ratio >= 0.55:
        return "bright_yellow"
    return "bright_green"


def percent_ratio(value: str) -> float | None:
    match = re.search(r"(-?\d+(?:\.\d+)?)\s*%", value)
    if not match:
        return None
    return max(0.0, min(1.0, float(match.group(1)) / 100.0))


def used_total_ratio(value: str) -> float | None:
    match = re.search(r"(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)", value)
    if not match:
        return None
    total = float(match.group(2))
    if total <= 0:
        return None
    return max(0.0, min(1.0, float(match.group(1)) / total))


def highlight_dashboard_text(text: str, *, color: bool) -> str:
    if not color:
        return text
    styles = [
        (r"\bwould_fail\b", "bright_yellow"),
        (r"\bneeds_shepherd_(?:finalize|release)\b", "bright_yellow"),
        (r"\bneeds_shepherd_[a-z_]+\b", "bright_red"),
        (r"\borphaned_container\b", "bright_red"),
        (r"\bfailed\b", "bright_red"),
        (r"\bexit=\d+\b", "bright_red"),
        (r"\bdown\b", "bright_red"),
        (r"\bmissing\b", "yellow"),
        (r"\bbusy\b", "bright_green"),
        (r"\bwarning\b", "bright_yellow"),
        (r"\boffline\b", "bright_red"),
        (r"\bunreachable\b", "bright_red"),
        (r"\breachable\b", "bright_green"),
        (r"\blive\b", "bright_green"),
        (r"\bok\b", "bright_green"),
        (r"\bsteady\b", "bright_green"),
        (r"\bstart\b", "bright_cyan"),
        (r"\brestart\b", "bright_yellow"),
        (r"\bremove\b", "bright_yellow"),
        (r"\bplanned\b", "bright_cyan"),
        (r"\bnone\b", "dim"),
        (r"\bunknown\b", "dim"),
    ]
    highlighted = text
    for pattern, style in styles:
        highlighted = re.sub(
            pattern,
            lambda match, style=style: colorize(match.group(0), style, enabled=True),
            highlighted,
        )
    highlighted = re.sub(
        r"\[[#-]{1,10}\]\s+(?:\d+(?:\.\d+)?%|\d+(?:\.\d+)?/\d+(?:\.\d+)?\s+[A-Za-z]+)",
        lambda match: colorize(
            match.group(0),
            heat_style(percent_ratio(match.group(0)) or used_total_ratio(match.group(0)) or 0.0),
            enabled=True,
        ),
        highlighted,
    )
    return re.sub(
        r"\b[0-9a-f]{12}\b",
        lambda match: colorize(match.group(0), "cyan", enabled=True),
        highlighted,
    )


def style_table(table: str, *, color: bool) -> str:
    if not color or not table:
        return table
    lines = table.splitlines()
    if lines:
        lines[0] = colorize(lines[0], "white", enabled=True)
    if len(lines) > 1:
        lines[1] = colorize(lines[1], "dim", enabled=True)
    for index in range(2, len(lines)):
        lines[index] = highlight_dashboard_text(lines[index], color=True)
    return "\n".join(lines)


def truncate_cell(value: Any, width: int) -> str:
    text = str(value)
    if width < 4 or len(text) <= width:
        return text
    return f"{text[: width - 3]}..."


def format_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    *,
    max_width: int,
) -> str:
    if not headers:
        return ""
    string_rows = [[str(cell) for cell in row] for row in rows]
    widths = [
        max(
            len(str(headers[index])),
            *(len(row[index]) for row in string_rows),
        )
        for index in range(len(headers))
    ]
    min_widths = [min(len(str(header)), 10) for header in headers]
    while sum(widths) + (3 * (len(widths) - 1)) > max_width and max(widths) > 10:
        widest = max(range(len(widths)), key=lambda index: widths[index])
        if widths[widest] <= min_widths[widest]:
            break
        widths[widest] -= 1
    line_parts = [str(header).ljust(widths[index]) for index, header in enumerate(headers)]
    lines = [" | ".join(line_parts)]
    lines.append("-+-".join("-" * width for width in widths))
    for row in string_rows:
        lines.append(
            " | ".join(
                truncate_cell(row[index], widths[index]).ljust(widths[index])
                for index in range(len(headers))
            )
        )
    return "\n".join(lines)


def _job_container_active(container: Any) -> bool:
    return container.state in {"running", "created", "restarting"}


def _watch_hint(
    *,
    launch: Mapping[str, Any] | None,
    container: Any | None,
    result_present: bool,
) -> str:
    if launch is None:
        return "orphaned_container"
    if container is None:
        if result_present:
            return "needs_shepherd_finalize"
        if launch.get("state") == "launching":
            return "needs_shepherd_release"
        return "needs_shepherd_fail_or_retry"
    if container.state == "running":
        return "ok"
    if result_present:
        return "needs_shepherd_finalize"
    return "needs_shepherd_mark_failed"


def watch_hint_icon(hint: str) -> str:
    if hint == "ok":
        return "ok"
    if hint == "orphaned_container":
        return "!"
    if hint.startswith("needs_shepherd"):
        return "->"
    return "?"


def render_machine_watch_dashboard(snapshot: MachineWatchSnapshot, *, color: bool = False) -> str:
    machine = snapshot.machine
    active_containers = [container for container in snapshot.containers if _job_container_active(container)]
    active_by_kind: dict[str, int] = {"train": 0}
    for container in active_containers:
        if container.job_kind in active_by_kind:
            active_by_kind[str(container.job_kind)] += 1
    launches_by_id = {str(launch["launch_id"]): launch for launch in snapshot.launches}
    containers_by_launch = {
        container.launch_id: container for container in snapshot.containers if container.launch_id
    }
    train_counts = snapshot.queue_counts.get("train", {})
    width = max(shutil.get_terminal_size((120, 30)).columns, 72)
    captured = format_utc_second(snapshot.captured_at)
    clock = colorize(captured, "white", enabled=color)
    title = colorize("rlab fleet watch", "bright_cyan", enabled=color)
    capacity = f"{len(active_containers)}/{machine.limits.max_parallel_containers}"
    train_capacity = f"{active_by_kind['train']}/{machine.limits.max_parallel_containers}"
    header = [
        f"{title} {colorize('time', 'gray', enabled=color)} {clock}",
        (
            f"machine={colorize(machine.name, 'cyan', enabled=color)} "
            f"{dashboard_chip('mode', 'read-only', 'blue', color=color)} "
            f"{dashboard_chip('capacity', capacity, heat_style(used_total_ratio(capacity) or 0.0), color=color)} "
            f"{dashboard_chip('train', train_capacity, heat_style(used_total_ratio(train_capacity) or 0.0), color=color)}"
        ),
        (
            "queue "
            f"{dashboard_chip('train_pending', str(int(train_counts.get('pending', 0))), 'bright_cyan', color=color)} "
            f"{dashboard_chip('train_launching', str(int(train_counts.get('launching', 0))), 'bright_yellow', color=color)} "
            f"{dashboard_chip('train_running', str(int(train_counts.get('running', 0))), 'bright_green', color=color)}"
        ),
        dashboard_divider(width, color=color),
    ]
    sections = ["\n".join(header)]

    launch_rows: list[list[str]] = []
    for launch in snapshot.launches:
        launch_id = str(launch["launch_id"])
        container = containers_by_launch.get(launch_id)
        result_present = bool(snapshot.result_present.get(launch_id, False))
        hint = _watch_hint(launch=launch, container=container, result_present=result_present)
        launch_rows.append(
            [
                f"{watch_hint_icon(hint)} {hint}",
                launch_id,
                f"{launch['job_kind']}/{launch['job_id']}",
                str(launch["state"]),
                container.name if container else "missing",
                container.state if container else "missing",
                "yes" if result_present else "no",
            ]
        )
    sections.append(
        numbered_section(1, " launches:", "cyan", color=color)
        + "\n"
        + (
            style_table(
                format_table(
                    ["hint", "launch_id", "job", "launch", "container", "state", "result"],
                    launch_rows,
                    max_width=width,
                ),
                color=color,
            )
            if launch_rows
            else highlight_dashboard_text("none", color=color)
        )
    )

    orphaned = [
        container
        for container in snapshot.containers
        if container.launch_id and container.launch_id not in launches_by_id
    ]
    if orphaned:
        orphan_rows = []
        for container in orphaned:
            result_present = bool(snapshot.result_present.get(str(container.launch_id), False))
            orphan_rows.append(
                [
                    "! orphaned_container",
                    container.name,
                    container.launch_id or "unknown",
                    f"{container.job_kind or 'unknown'}/{container.labels.get(JOB_ID_LABEL, 'unknown')}",
                    container.state,
                    "yes" if result_present else "no",
                ]
            )
        sections.append(
            numbered_section(2, " orphaned containers:", "bright_yellow", color=color)
            + "\n"
            + style_table(
                format_table(
                    ["hint", "container", "launch_id", "job", "state", "result"],
                    orphan_rows,
                    max_width=width,
                ),
                color=color,
            )
        )
    if snapshot.warnings:
        sections.append(
            numbered_section(3, " warnings:", "yellow", color=color)
            + "\n"
            + "\n".join(
                highlight_dashboard_text(f"  ! {warning}", color=color)
                for warning in snapshot.warnings
            )
        )
    return "\n\n".join(sections)


def write_tui_frame(text: str, *, enabled: bool) -> None:
    if enabled and sys.stdout.isatty():
        sys.stdout.write("\033[2J\033[H")
    print(text, flush=True)
