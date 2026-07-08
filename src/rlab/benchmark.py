from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from rlab.benchmark_profiles import (
    DEFAULT_PROFILE_DIR,
    DEFAULT_RESULT_DIR,
    BenchmarkCommand,
    build_benchmark_commands,
    find_benchmark_profile,
    load_benchmark_profiles,
)


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _json(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True)


def _command_plan(commands: list[BenchmarkCommand]) -> list[dict[str, Any]]:
    return [command.to_json() for command in commands]


def list_profiles(args: argparse.Namespace) -> int:
    profiles = load_benchmark_profiles(args.profile_dir)
    rows = [
        {
            "name": profile.name,
            "kind": profile.kind,
            "description": profile.description,
            "path": str(profile.path),
        }
        for profile in profiles
    ]
    if args.json:
        print(_json(rows))
        return 0
    for row in rows:
        suffix = f" - {row['description']}" if row["description"] else ""
        print(f"{row['name']} ({row['kind']}){suffix}")
    return 0


def show_profile(args: argparse.Namespace) -> int:
    profile = find_benchmark_profile(args.profile, profile_dir=args.profile_dir)
    commands = build_benchmark_commands(profile)
    payload = {
        "profile": profile.payload,
        "path": str(profile.path),
        "commands": _command_plan(commands),
    }
    print(_json(payload))
    return 0


def run_command(command: BenchmarkCommand) -> dict[str, Any]:
    env = os.environ.copy()
    if command.env:
        env.update(command.env)
    started_at = datetime.now(UTC)
    result = subprocess.run(
        command.argv,
        check=False,
        cwd=command.cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    finished_at = datetime.now(UTC)
    return {
        "label": command.label,
        "argv": list(command.argv),
        "cwd": str(command.cwd) if command.cwd else None,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def run_profile(args: argparse.Namespace) -> int:
    profile = find_benchmark_profile(args.profile, profile_dir=args.profile_dir)
    commands = build_benchmark_commands(profile)
    plan = _command_plan(commands)
    if args.dry_run:
        print(_json({"profile": profile.name, "dry_run": True, "commands": plan}))
        return 0

    results = []
    failed = False
    for command in commands:
        print(f"running benchmark command: {command.label}", flush=True)
        result = run_command(command)
        results.append(result)
        if result["returncode"] != 0:
            failed = True
            if not args.keep_going:
                break

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{_timestamp()}-{profile.name}.json"
    output = {
        "profile": profile.payload,
        "profile_path": str(profile.path),
        "commands": plan,
        "results": results,
        "status": "failed" if failed else "passed",
    }
    output_path.write_text(_json(output) + "\n", encoding="utf-8")
    print(f"wrote benchmark result: {output_path}")
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run named rlab benchmark profiles.")
    parser.add_argument("--profile-dir", type=Path, default=DEFAULT_PROFILE_DIR)
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List available benchmark profiles.")
    list_parser.add_argument("--json", action="store_true")
    list_parser.set_defaults(func=list_profiles)

    show_parser = subparsers.add_parser("show", help="Show a profile and its command plan.")
    show_parser.add_argument("profile")
    show_parser.set_defaults(func=show_profile)

    run_parser = subparsers.add_parser("run", help="Run a benchmark profile.")
    run_parser.add_argument("profile")
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument("--keep-going", action="store_true")
    run_parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULT_DIR)
    run_parser.set_defaults(func=run_profile)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
