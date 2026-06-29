from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rlab.monitoring.state import MonitorOptions, collect_state


JOBS_COLUMNS = (
    ("Job", "id"),
    ("Workload", "target"),
    ("Device", "device"),
    ("Container", "container"),
    ("State", "state"),
    ("Progress", "progress"),
    ("Attention", "attention"),
)
DEVICES_COLUMNS = (
    ("Host", "device"),
    ("Runner", "target"),
    ("State", "state"),
    ("Capacity", "capacity"),
    ("Usage", "usage"),
    ("Running", "current_job"),
    ("Queued", "queued_job"),
    ("Health", "last_check"),
)


def compact_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, list | tuple):
        return ", ".join(str(item) for item in value)
    if isinstance(value, Mapping):
        return json.dumps(value, sort_keys=True)
    return str(value)


def truncate(value: str, width: int) -> str:
    if len(value) <= width:
        return value
    if width <= 1:
        return value[:width]
    return value[: width - 1] + "."


def table_lines(
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[tuple[str, str]],
    *,
    max_width: int = 36,
) -> list[str]:
    if not rows:
        return ["  no rows"]

    rendered = [
        [compact_value(row.get(key)) for _label, key in columns]
        for row in rows
    ]
    widths = [
        min(
            max_width,
            max(len(label), *(len(row[index]) for row in rendered)),
        )
        for index, (label, _key) in enumerate(columns)
    ]
    header = "  " + "  ".join(
        label.ljust(widths[index]) for index, (label, _key) in enumerate(columns)
    )
    divider = "  " + "  ".join("-" * width for width in widths)
    body = [
        "  "
        + "  ".join(
            truncate(value, widths[index]).ljust(widths[index])
            for index, value in enumerate(row)
        )
        for row in rendered
    ]
    return [header, divider, *body]


def format_monitor_state(state: Mapping[str, Any], *, view: str = "jobs") -> str:
    source = state.get("source", {})
    if not isinstance(source, Mapping):
        source = {}
    lines = [
        f"rlab monitor: {source.get('queue', 'unknown')} - {source.get('message', '')}",
        f"refreshed_at: {state.get('refreshed_at', '')}",
    ]
    if view in {"jobs", "all"}:
        jobs = state.get("jobs", [])
        if not isinstance(jobs, list):
            jobs = []
        lines.extend(["", f"jobs ({len(jobs)})", *table_lines(jobs, JOBS_COLUMNS)])
    if view in {"devices", "all"}:
        devices = state.get("devices", [])
        if not isinstance(devices, list):
            devices = []
        lines.extend(["", f"fleet ({len(devices)})", *table_lines(devices, DEVICES_COLUMNS)])
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Print read-only rlab queue and fleet state.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--goal", help="Optional queue goal_slug filter.")
    parser.add_argument("--direct", action="store_true", help="Use DIRECT_DATABASE_URL.")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use sample rows instead of connecting to the queue database.",
    )
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--view", choices=("jobs", "devices", "all"), default="jobs")
    parser.add_argument("--json", action="store_true", help="Print raw monitor state as JSON.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    options = MonitorOptions(
        repo_root=args.repo_root.expanduser().resolve(),
        goal=args.goal,
        direct=args.direct,
        sample=args.sample,
        limit=args.limit,
    )
    state = collect_state(options)
    if args.json:
        print(json.dumps(state, indent=2, sort_keys=True))
        return
    print(format_monitor_state(state, view=args.view))


if __name__ == "__main__":
    main()
