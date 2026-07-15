from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import UTC, datetime
from importlib import metadata
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

from rlab.benchmark_profiles import (
    DEFAULT_PROFILE_DIR,
    DEFAULT_RESULT_DIR,
    BenchmarkCommand,
    BenchmarkProfile,
    build_benchmark_commands,
    find_benchmark_profile,
    load_benchmark_profiles,
)
from rlab.env import default_run_dir
from rlab.metric_store import MetricStore, metric_store_path


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _json(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True)


def _command_plan(commands: list[BenchmarkCommand]) -> list[dict[str, Any]]:
    return [command.to_json() for command in commands]


def _execution_commands(
    profile: BenchmarkProfile,
    commands: list[BenchmarkCommand],
    *,
    execution_id: str,
    output_dir: Path,
) -> list[BenchmarkCommand]:
    if profile.kind != "train_loop_throughput":
        return commands
    config = json.loads(commands[0].stdin or "{}")
    config["runs_dir"] = str(output_dir / "runs" / f"{execution_id}-{profile.name}")
    return [replace(commands[0], stdin=json.dumps(config, sort_keys=True))]


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _git_dirty() -> bool | None:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return bool(result.stdout.strip()) if result.returncode == 0 else None


def _load_average() -> list[float] | None:
    try:
        return [round(float(value), 3) for value in os.getloadavg()]
    except OSError:
        return None


def _environment_snapshot(profile: BenchmarkProfile) -> dict[str, Any]:
    versions = {"python": platform.python_version()}
    distributions = {"rlab", "stable-retro-turbo", "supermariobrosnes-turbo"}
    provider = str(profile.payload.get("env_provider") or "").strip()
    if provider:
        distributions.add(provider)
    for distribution in sorted(distributions):
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[distribution] = None
    return {
        "captured_at": datetime.now(UTC).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "load_average_1m_5m_15m": _load_average(),
        "source_commit": _git_commit(),
        "source_dirty": _git_dirty(),
        "versions": versions,
    }


def _stdout_json(result: dict[str, Any], *, label: str, issues: list[str]) -> Any:
    try:
        return json.loads(str(result.get("stdout") or ""))
    except json.JSONDecodeError as exc:
        issues.append(f"{label} did not emit valid JSON: {exc}")
        return None


def validate_benchmark_results(
    profile: BenchmarkProfile,
    commands: list[BenchmarkCommand],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    issues = [
        f"{result['label']} exited with {result['returncode']}"
        for result in results
        if int(result["returncode"]) != 0
    ]
    evidence: dict[str, Any] = {}

    if len(results) != len(commands):
        issues.append(f"executed {len(results)} of {len(commands)} benchmark commands")

    successful = [result for result in results if int(result["returncode"]) == 0]
    if profile.kind == "local_smoke" and successful:
        payload = _stdout_json(successful[0], label="local smoke", issues=issues)
        if isinstance(payload, dict):
            wait = payload.get("wait")
            jobs = payload.get("jobs")
            if not isinstance(wait, dict) or not wait.get("reached"):
                issues.append("local smoke did not reach terminal state")
            if not isinstance(jobs, list) or not jobs:
                issues.append("local smoke returned no jobs")
            else:
                statuses = [str(job.get("status")) for job in jobs if isinstance(job, dict)]
                evidence["job_statuses"] = statuses
                if len(statuses) != len(jobs) or statuses != ["succeeded"] * len(statuses):
                    issues.append(f"local smoke jobs were not all successful: {statuses}")

    if profile.kind == "env_throughput":
        samples = []
        for result in successful:
            payload = _stdout_json(result, label=str(result["label"]), issues=issues)
            if not isinstance(payload, dict):
                continue
            sample = {
                "label": result["label"],
                "envs": payload.get("envs"),
                "results": payload.get("results"),
                "runtime_overhead_fraction": payload.get("runtime_overhead_fraction"),
                "overhead_gate_passed": payload.get("overhead_gate_passed"),
            }
            samples.append(sample)
            if payload.get("mode") == "compare" and payload.get("overhead_gate_passed") is not True:
                issues.append(f"{result['label']} exceeded the runtime-overhead gate")
        evidence["samples"] = samples

    if profile.kind == "train_loop_throughput" and successful:
        config = json.loads(commands[0].stdin or "{}")
        base_dir = commands[0].cwd or Path.cwd()
        run_dir = base_dir / default_run_dir(
            str(config["run_name"]), str(config.get("runs_dir") or "runs")
        )
        store_path = metric_store_path(run_dir)
        evidence["metric_store"] = str(store_path)
        if not store_path.is_file():
            issues.append(f"training-loop metric store is missing: {store_path}")
        else:
            store = MetricStore(store_path)
            metrics = {
                name: store.latest_metric(name)
                for name in profile.payload.get("required_metrics", ())
            }
            evidence["metrics"] = metrics
            missing = sorted(name for name, value in metrics.items() if value is None)
            if missing:
                issues.append(f"training-loop metrics are missing: {', '.join(missing)}")

    return {"passed": not issues, "issues": issues, "evidence": evidence}


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
        input=command.stdin,
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
    if args.dry_run:
        print(
            _json({"profile": profile.name, "dry_run": True, "commands": _command_plan(commands)})
        )
        return 0

    execution_id = _timestamp()
    commands = _execution_commands(
        profile,
        commands,
        execution_id=execution_id,
        output_dir=args.output_dir,
    )
    plan = _command_plan(commands)
    environment_before = _environment_snapshot(profile)
    results = []
    for command in commands:
        print(f"running benchmark command: {command.label}", flush=True)
        result = run_command(command)
        results.append(result)
        if result["returncode"] != 0 and not args.keep_going:
            break

    validation = validate_benchmark_results(profile, commands, results)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{execution_id}-{profile.name}.json"
    output = {
        "profile": profile.payload,
        "profile_path": str(profile.path),
        "commands": plan,
        "environment_before": environment_before,
        "environment_after": _environment_snapshot(profile),
        "results": results,
        "validation": validation,
        "status": "passed" if validation["passed"] else "failed",
    }
    output_path.write_text(_json(output) + "\n", encoding="utf-8")
    print(f"wrote benchmark result: {output_path}")
    return 0 if validation["passed"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rlab benchmark",
        description="Run named rlab benchmark profiles.",
    )
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
