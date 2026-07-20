from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

from rlab.cli_args import add_direct_database_arg
from rlab.json_utils import json_safe
from rlab.machines import DEFAULT_MACHINE_REGISTRY


def _git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and result.returncode:
        raise RuntimeError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result


def repository_root() -> Path:
    result = _git(Path.cwd(), "rev-parse", "--show-toplevel")
    return Path(result.stdout.strip()).resolve()


def _tracked_committed_path(root: Path, path: Path, *, label: str) -> Path:
    root = root.resolve()
    resolved = (root / path).resolve() if not path.is_absolute() else path.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must be inside the repository: {resolved}") from exc
    if not resolved.is_file():
        raise ValueError(f"{label} does not exist: {relative}")
    if _git(root, "ls-files", "--error-unmatch", str(relative), check=False).returncode:
        raise ValueError(f"{label} must be checked in: {relative}")
    if _git(root, "diff", "--quiet", "HEAD", "--", str(relative), check=False).returncode:
        raise ValueError(f"{label} has uncommitted changes: {relative}")
    return relative


def _source(root: Path) -> tuple[str, str, bool]:
    from rlab.runtime_refs import current_git_branch

    source_sha = _git(root, "rev-parse", "HEAD").stdout.strip()
    branch = current_git_branch(root)
    dirty = bool(_git(root, "status", "--porcelain", check=True).stdout.strip())
    return source_sha, branch, dirty


def _link_if_present(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source)


@contextmanager
def isolated_head_worktree(root: Path, source_sha: str) -> Iterator[Path]:
    path = Path(tempfile.gettempdir()) / f"rlab-head-{source_sha[:12]}-{uuid.uuid4().hex[:8]}"
    _git(root, "worktree", "add", "--detach", "--quiet", str(path), source_sha)
    try:
        _link_if_present(root / ".env", path / ".env")
        _link_if_present(
            root / "logs" / "fleet" / "modal-eval-assets.json",
            path / "logs" / "fleet" / "modal-eval-assets.json",
        )
        yield path
    finally:
        _git(root, "worktree", "remove", "--force", str(path), check=False)
        _git(root, "worktree", "prune", check=False)


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _load_environment(root: Path) -> None:
    from rlab.dotenv import load_env_file

    load_env_file(root / ".env")


def _connect(args: argparse.Namespace, root: Path):
    from rlab.job_queue import connect, database_url

    _load_environment(root)
    conn = connect(database_url(bool(getattr(args, "direct", False))))
    conn.autocommit = True
    return conn


def _parse_captured_json(output: str) -> dict[str, Any]:
    for line in reversed(output.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise RuntimeError("internal queue command did not produce a JSON object")


def _canonical_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    if "job_id" in value:
        value["run_id"] = value.pop("job_id")
    if "job_ids" in value:
        value["run_ids"] = value.pop("job_ids")
    if "jobs" in value:
        value["runs"] = value.pop("jobs")
    if "retried_from" in value:
        value["retried_from_run_id"] = value.pop("retried_from")
    return value


def _call_json_command(function, args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        code = int(function(args))
    return code, _canonical_payload(_parse_captured_json(stdout.getvalue()))


def cmd_launch(args: argparse.Namespace) -> int:
    from rlab.fleet_service import require_compatible_controller_services
    from rlab.job_queue import cmd_enqueue_train
    from rlab.runtime_refs import clean_git_source_sha

    root = repository_root()
    goal = _tracked_committed_path(root, args.goal_file, label="goal")
    recipe = _tracked_committed_path(root, args.recipe_file, label="recipe")
    source_sha, source_branch, caller_dirty = _source(root)
    if args.runtime_image_ref_file is not None:
        args.runtime_image_ref_file = args.runtime_image_ref_file.expanduser().resolve()
    require_compatible_controller_services()
    args.goal_file = goal
    args.recipe_file = recipe
    args.image_branch = args.image_branch or source_branch
    args.json = True
    if args.from_head:
        context = isolated_head_worktree(root, source_sha)
    else:
        clean_git_source_sha(root)
        context = contextlib.nullcontext(root)
    with context as launch_root, _working_directory(launch_root):
        code, payload = _call_json_command(cmd_enqueue_train, args)
    payload.update(
        {
            "schema_version": 1,
            "source_sha": source_sha,
            "source_branch": source_branch,
            "from_head": bool(args.from_head),
            "local_changes_excluded": bool(args.from_head and caller_dirty),
            "goal_file": str(goal),
            "recipe_file": str(recipe),
        }
    )
    if args.output_json:
        print(json.dumps(json_safe(payload), sort_keys=True))
    else:
        print(
            f"runs={','.join(str(value) for value in payload.get('run_ids') or [])} "
            f"batch={payload.get('batch_id')} machine={payload.get('machine')} "
            f"source={source_sha[:12]}"
        )
    return code


def _selector(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: getattr(args, key)
        for key in ("run_id", "batch_id", "machine", "goal")
        if getattr(args, key, None) is not None
    }


def cmd_status(args: argparse.Namespace) -> int:
    from rlab.run_observability import projections_for_selector

    root = repository_root()
    conn = _connect(args, root)
    try:
        report = projections_for_selector(conn, **_selector(args))
    finally:
        conn.close()
    print(json.dumps(json_safe(report), indent=None if args.json else 2, sort_keys=True))
    return 0


def cmd_follow(args: argparse.Namespace) -> int:
    from rlab.run_observability import follow_run

    root = repository_root()
    conn = _connect(args, root)

    def emit(value: Mapping[str, Any]) -> None:
        print(json.dumps(json_safe(value), sort_keys=True, separators=(",", ":")), flush=True)

    return follow_run(
        conn,
        int(args.run_id),
        emit=emit,
        poll_seconds=float(args.poll_seconds),
        heartbeat_seconds=float(args.heartbeat_seconds),
        connection_factory=lambda: _connect(args, root),
    )


def cmd_wait(args: argparse.Namespace) -> int:
    from rlab.job_queue import _job_ids_for_selector, wait_for_job_ids

    root = repository_root()
    conn = _connect(args, root)
    try:
        ids = _job_ids_for_selector(conn, job_id=args.run_id, batch_id=args.batch_id)
        result = wait_for_job_ids(
            conn,
            ids,
            until=args.until,
            timeout=float(args.timeout),
        )
    finally:
        conn.close()
    value = _canonical_payload(result)
    value["run_ids"] = ids
    print(json.dumps(json_safe(value), sort_keys=True))
    return 0 if result["reached"] else 1


def _queue_namespace(args: argparse.Namespace) -> argparse.Namespace:
    args.job_id = getattr(args, "run_id", None)
    args.goal_slug = getattr(args, "goal", None)
    args.json = True
    return args


def cmd_cancel(args: argparse.Namespace) -> int:
    from rlab.fleet_service import require_compatible_controller_services
    from rlab.job_queue import cmd_cancel as queue_cancel

    require_compatible_controller_services(require_source_current=False)
    code, payload = _call_json_command(queue_cancel, _queue_namespace(args))
    print(json.dumps(json_safe(payload), sort_keys=True))
    return code


def cmd_retry(args: argparse.Namespace) -> int:
    from rlab.fleet_service import require_compatible_controller_services
    from rlab.job_queue import cmd_retry as queue_retry

    require_compatible_controller_services()
    args.job_id = args.run_id
    args.json = True
    code, payload = _call_json_command(queue_retry, args)
    print(json.dumps(json_safe(payload), sort_keys=True))
    return code


def cmd_retry_finalization(args: argparse.Namespace) -> int:
    from rlab.fleet_service import require_compatible_controller_services
    from rlab.job_queue import cmd_retry_finalization as queue_retry_finalization

    require_compatible_controller_services(require_source_current=False)
    args.job_id = args.run_id
    args.json = True
    code, payload = _call_json_command(queue_retry_finalization, args)
    print(json.dumps(json_safe(payload), sort_keys=True))
    return code


def cmd_logs(args: argparse.Namespace) -> int:
    from rlab.job_queue import cmd_logs as queue_logs

    args.job_id = args.run_id
    return int(queue_logs(args))


def _add_selector(
    parser: argparse.ArgumentParser,
    *,
    machine: bool,
    goal: bool,
) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", dest="run_id", type=int)
    group.add_argument("--batch", dest="batch_id")
    if machine:
        group.add_argument("--machine")
    if goal:
        group.add_argument("--goal")


def build_parser() -> argparse.ArgumentParser:
    from rlab.job_queue import build_train_enqueue_parser, parse_duration_seconds
    from rlab.runtime_refs import (
        DEFAULT_IMAGE_ARTIFACT,
        DEFAULT_IMAGE_WORKFLOW,
        DEFAULT_RUNTIME_READINESS_TIMEOUT_SECONDS,
    )

    parser = argparse.ArgumentParser(
        prog="rlab experiment",
        description="Launch and observe queue-backed training experiments.",
        allow_abbrev=False,
    )
    commands = parser.add_subparsers(dest="command", required=True)

    launch_parent = build_train_enqueue_parser()
    launch = commands.add_parser(
        "launch",
        parents=[launch_parent],
        add_help=False,
        help="Launch a checked-in goal and recipe.",
    )
    launch.add_argument(
        "--from-head",
        action="store_true",
        help="Launch committed HEAD from an isolated detached worktree.",
    )
    launch.set_defaults(func=cmd_launch, output_json=False)
    for action in launch._actions:
        if action.dest == "json":
            action.dest = "output_json"

    status = commands.add_parser("status", help="Inspect authoritative run state.")
    add_direct_database_arg(status)
    _add_selector(status, machine=True, goal=True)
    status.add_argument("--json", action="store_true")
    status.set_defaults(func=cmd_status)

    follow = commands.add_parser("follow", help="Stream semantic run events as JSONL.")
    add_direct_database_arg(follow)
    follow.add_argument("--run", dest="run_id", type=int, required=True)
    follow.add_argument("--jsonl", action="store_true", help="Emit the stable JSONL event schema.")
    follow.add_argument("--poll-seconds", type=float, default=2.0, help=argparse.SUPPRESS)
    follow.add_argument("--heartbeat-seconds", type=float, default=60.0, help=argparse.SUPPRESS)
    follow.set_defaults(func=cmd_follow)

    wait = commands.add_parser("wait", help="Wait for one run or batch state.")
    add_direct_database_arg(wait)
    _add_selector(wait, machine=False, goal=False)
    wait.add_argument("--until", choices=("running", "terminal"), required=True)
    wait.add_argument("--timeout", type=parse_duration_seconds, default=12 * 60 * 60)
    wait.set_defaults(func=cmd_wait)

    cancel = commands.add_parser("cancel", help="Request idempotent cancellation.")
    add_direct_database_arg(cancel)
    _add_selector(cancel, machine=True, goal=False)
    cancel.add_argument("--machines", type=Path, default=DEFAULT_MACHINE_REGISTRY)
    cancel.add_argument("--all-active", action="store_true")
    cancel.add_argument("--drain", action="store_true")
    cancel.add_argument("--wait", action="store_true")
    cancel.add_argument("--timeout", type=parse_duration_seconds, default=10 * 60)
    cancel.set_defaults(func=cmd_cancel)

    retry = commands.add_parser("retry", help="Create a new run from a terminal run.")
    add_direct_database_arg(retry)
    retry.add_argument("--run", dest="run_id", type=int, required=True)
    retry.add_argument("--request-id", dest="submission_key")
    retry.add_argument("--machines", type=Path, default=DEFAULT_MACHINE_REGISTRY)
    retry.add_argument("--runtime-image-ref-file", type=Path)
    retry.add_argument("--image-workflow", default=DEFAULT_IMAGE_WORKFLOW)
    retry.add_argument("--image-branch")
    retry.add_argument("--image-artifact", default=DEFAULT_IMAGE_ARTIFACT)
    retry.add_argument(
        "--runtime-readiness-timeout",
        type=parse_duration_seconds,
        default=DEFAULT_RUNTIME_READINESS_TIMEOUT_SECONDS,
    )
    retry.add_argument("--wait", choices=("running", "terminal"))
    retry.add_argument("--timeout", type=parse_duration_seconds, default=12 * 60 * 60)
    retry.set_defaults(func=cmd_retry)

    retry_finalization = commands.add_parser(
        "retry-finalization",
        help="Retry publication/evaluation finalization without retraining.",
    )
    add_direct_database_arg(retry_finalization)
    retry_finalization.add_argument("--run", dest="run_id", type=int, required=True)
    retry_finalization.set_defaults(func=cmd_retry_finalization)

    logs = commands.add_parser("logs", help="Read durable output logs for one run.")
    add_direct_database_arg(logs)
    logs.add_argument("--run", dest="run_id", type=int, required=True)
    logs.add_argument("--machines", type=Path, default=DEFAULT_MACHINE_REGISTRY)
    logs.add_argument("--tail", type=int, default=100)
    logs.add_argument("--follow", action="store_true")
    logs.set_defaults(func=cmd_logs)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
