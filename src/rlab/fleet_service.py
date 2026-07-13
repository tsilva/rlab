from __future__ import annotations

import argparse
import json
import os
import plistlib
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from logging import Formatter, Logger
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


SERVICE_LABEL = "com.rlab.fleet-service"
SERVICE_INTERVAL_SECONDS = 30
DEFAULT_LANE_TIMEOUT_SECONDS = 120.0
DEFAULT_PASS_TIMEOUT_SECONDS = 300.0
DEFAULT_MAX_MACHINE_LANES = 4
SCHEMA_MAINTENANCE_LOCK = "rlab-fleet-schema-maintenance"
TERMINAL_JOB_STATES = frozenset({"succeeded", "failed", "canceled"})

CommandRunner = Callable[..., subprocess.CompletedProcess[str]]
DiscoverMachines = Callable[[Path], Sequence[str]]
ReconcileMachine = Callable[[Path, str, float], Mapping[str, Any] | None]
CountNonterminalJobs = Callable[[Path], int]


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _iso_utc(value: datetime | None = None) -> str:
    return (value or _utc_now()).isoformat().replace("+00:00", "Z")


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ServicePaths:
    repo_root: Path
    python: Path
    state_dir: Path
    launch_agents_dir: Path
    label: str = SERVICE_LABEL

    @property
    def plist(self) -> Path:
        return self.launch_agents_dir / f"{self.label}.plist"

    @property
    def bootstrap_stdout(self) -> Path:
        return self.state_dir / "bootstrap.stdout.log"

    @property
    def bootstrap_stderr(self) -> Path:
        return self.state_dir / "bootstrap.stderr.log"

    @property
    def service_log(self) -> Path:
        return self.state_dir / "service.jsonl"

    @property
    def last_pass(self) -> Path:
        return self.state_dir / "last-pass.json"


def default_service_paths(
    *,
    repo_root: Path | str | None = None,
    state_dir: Path | str | None = None,
    launch_agents_dir: Path | str | None = None,
    label: str = SERVICE_LABEL,
) -> ServicePaths:
    root = Path(repo_root or _default_repo_root()).expanduser().resolve()
    state = (
        Path(
            state_dir
            or (Path.home() / "Library" / "Application Support" / "rlab" / "fleet-service")
        )
        .expanduser()
        .resolve()
    )
    agents = Path(launch_agents_dir or (Path.home() / "Library" / "LaunchAgents"))
    return ServicePaths(
        repo_root=root,
        # Keep the stable repo-local path in the plist even when it is a symlink
        # to a managed interpreter. Replacing .venv must be an explicit service update.
        python=root / ".venv" / "bin" / "python",
        state_dir=state,
        launch_agents_dir=agents.expanduser().resolve(),
        label=label,
    )


def launch_agent_payload(paths: ServicePaths) -> dict[str, Any]:
    return {
        "Label": paths.label,
        "ProgramArguments": [
            str(paths.python),
            "-m",
            "rlab.fleet_service",
            "run-once",
            "--repo-root",
            str(paths.repo_root),
            "--state-dir",
            str(paths.state_dir),
        ],
        "WorkingDirectory": str(paths.repo_root),
        "StartInterval": SERVICE_INTERVAL_SECONDS,
        "KeepAlive": False,
        "ProcessType": "Background",
        "StandardOutPath": str(paths.bootstrap_stdout),
        "StandardErrorPath": str(paths.bootstrap_stderr),
    }


def validate_launch_agent_payload(payload: Mapping[str, Any], paths: ServicePaths) -> None:
    expected = launch_agent_payload(paths)
    forbidden = {"RunAtLoad", "EnvironmentVariables", "UserName", "GroupName"}
    found_forbidden = forbidden.intersection(payload)
    if found_forbidden:
        raise ValueError(f"launch agent contains forbidden keys: {sorted(found_forbidden)}")
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"launch agent {key} must be {value!r}")
    if payload.get("KeepAlive") is not False:
        raise ValueError("launch agent KeepAlive must be false")
    if payload.get("StartInterval") != SERVICE_INTERVAL_SECONDS:
        raise ValueError(f"launch agent StartInterval must be {SERVICE_INTERVAL_SECONDS}")
    for value in (
        paths.repo_root,
        paths.python,
        paths.state_dir,
        paths.launch_agents_dir,
        paths.bootstrap_stdout,
        paths.bootstrap_stderr,
    ):
        if not value.is_absolute():
            raise ValueError(f"launch agent path must be absolute: {value}")


def render_launch_agent_plist(paths: ServicePaths) -> bytes:
    payload = launch_agent_payload(paths)
    validate_launch_agent_payload(payload, paths)
    data = plistlib.dumps(payload, fmt=plistlib.FMT_XML, sort_keys=True)
    parsed = plistlib.loads(data)
    validate_launch_agent_payload(parsed, paths)
    return data


def _atomic_write(path: Path, data: bytes, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write(
        path,
        (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8"),
    )


_SENSITIVE_KEY = re.compile(
    r"(?:password|passwd|secret|token|api[_-]?key|access[_-]?key|database[_-]?url|dsn)",
    re.IGNORECASE,
)
_URL_CREDENTIALS = re.compile(r"(?P<scheme>[a-z][a-z0-9+.-]*://)[^/@\s:]+:[^/@\s]+@", re.I)
_ASSIGNMENT_SECRET = re.compile(
    r"(?i)\b(password|passwd|secret|token|api[_-]?key|access[_-]?key)=([^\s,;]+)"
)


def redact(value: Any, *, key: str | None = None) -> Any:
    if key and _SENSITIVE_KEY.search(key):
        return "[REDACTED]"
    if isinstance(value, Mapping):
        return {str(item_key): redact(item, key=str(item_key)) for item_key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [redact(item) for item in value]
    if isinstance(value, str):
        text = _URL_CREDENTIALS.sub(r"\g<scheme>[REDACTED]@", value)
        return _ASSIGNMENT_SECRET.sub(lambda match: f"{match.group(1)}=[REDACTED]", text)
    return value


class ServiceEventLog:
    def __init__(
        self,
        path: Path,
        *,
        max_bytes: int = 5 * 1024 * 1024,
        backup_count: int = 3,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._logger = Logger(f"rlab.fleet-service.{id(self)}")
        self._logger.propagate = False
        handler = RotatingFileHandler(
            path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        handler.setFormatter(Formatter("%(message)s"))
        self._logger.addHandler(handler)

    def emit(self, event: str, **fields: Any) -> None:
        record = redact({"timestamp": _iso_utc(), "event": event, **fields})
        self._logger.info(json.dumps(record, sort_keys=True, separators=(",", ":")))

    def close(self) -> None:
        for handler in list(self._logger.handlers):
            handler.close()
            self._logger.removeHandler(handler)


def _run_command(
    argv: Sequence[str],
    *,
    check: bool = False,
    capture_output: bool = True,
    text: bool = True,
    cwd: str | None = None,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv),
        check=check,
        capture_output=capture_output,
        text=text,
        cwd=cwd,
        env=dict(env) if env is not None else None,
    )


def _domain(uid: int | None = None) -> str:
    return f"gui/{os.getuid() if uid is None else uid}"


def _target(label: str, uid: int | None = None) -> str:
    return f"{_domain(uid)}/{label}"


def service_is_loaded(
    label: str = SERVICE_LABEL,
    *,
    uid: int | None = None,
    runner: CommandRunner = _run_command,
) -> bool:
    result = runner(
        ["launchctl", "print", _target(label, uid)],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def kick_service(
    label: str = SERVICE_LABEL,
    *,
    uid: int | None = None,
    runner: CommandRunner = _run_command,
) -> bool:
    command = ["launchctl", "kickstart", _target(label, uid)]
    if "-k" in command:
        raise AssertionError("fleet service kick must never terminate an active pass")
    result = runner(command, check=False, capture_output=True, text=True)
    return result.returncode == 0


def _validate_candidate_plist(path: Path, *, runner: CommandRunner) -> None:
    payload = plistlib.loads(path.read_bytes())
    if not isinstance(payload, Mapping):
        raise ValueError("launch agent plist must contain a dictionary")
    plutil = shutil.which("plutil")
    if plutil:
        runner([plutil, "-lint", str(path)], check=True, capture_output=True, text=True)


def _validate_service_entrypoint(paths: ServicePaths, *, runner: CommandRunner) -> None:
    environment = {
        "HOME": str(Path.home()),
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "TMPDIR": tempfile.gettempdir(),
    }
    result = runner(
        [str(paths.python), "-m", "rlab.fleet_service", "--help"],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(paths.repo_root),
        env=environment,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "fleet service entrypoint validation failed: "
            + str(redact(result.stderr or result.stdout or f"exit={result.returncode}"))
        )


@dataclass(frozen=True)
class InstallResult:
    installed: bool
    replaced: bool
    kicked: bool


def install_service(
    paths: ServicePaths,
    *,
    replace: bool = False,
    runner: CommandRunner = _run_command,
) -> InstallResult:
    if not paths.repo_root.is_dir():
        raise FileNotFoundError(f"repository root does not exist: {paths.repo_root}")
    if not paths.python.is_file() or not os.access(paths.python, os.X_OK):
        raise FileNotFoundError(f"service interpreter is not executable: {paths.python}")

    paths.state_dir.mkdir(parents=True, exist_ok=True)
    paths.launch_agents_dir.mkdir(parents=True, exist_ok=True)
    data = render_launch_agent_plist(paths)
    old_data = paths.plist.read_bytes() if paths.plist.exists() else None
    if old_data is not None and not replace:
        raise FileExistsError(f"launch agent already exists: {paths.plist}; use --replace")

    was_loaded = service_is_loaded(paths.label, runner=runner)
    candidate = paths.plist.with_name(f".{paths.plist.name}.candidate")
    _atomic_write(candidate, data)
    try:
        _validate_candidate_plist(candidate, runner=runner)
        _validate_service_entrypoint(paths, runner=runner)
        if was_loaded:
            runner(
                ["launchctl", "bootout", _target(paths.label)],
                check=True,
                capture_output=True,
                text=True,
            )
        os.replace(candidate, paths.plist)
        try:
            runner(
                ["launchctl", "bootstrap", _domain(), str(paths.plist)],
                check=True,
                capture_output=True,
                text=True,
            )
        except Exception:
            if old_data is None:
                paths.plist.unlink(missing_ok=True)
            else:
                _atomic_write(paths.plist, old_data)
                if was_loaded:
                    runner(
                        ["launchctl", "bootstrap", _domain(), str(paths.plist)],
                        check=False,
                        capture_output=True,
                        text=True,
                    )
            raise
    finally:
        candidate.unlink(missing_ok=True)

    return InstallResult(
        installed=True,
        replaced=old_data is not None,
        kicked=kick_service(paths.label, runner=runner),
    )


def _load_last_pass(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _last_pass_age_seconds(last_pass: Mapping[str, Any] | None) -> float | None:
    if not last_pass or not last_pass.get("finished_at"):
        return None
    try:
        finished = datetime.fromisoformat(str(last_pass["finished_at"]).replace("Z", "+00:00"))
    except ValueError:
        return None
    return max(0.0, (_utc_now() - finished.astimezone(UTC)).total_seconds())


def service_status(
    paths: ServicePaths,
    *,
    runner: CommandRunner = _run_command,
) -> dict[str, Any]:
    loaded = service_is_loaded(paths.label, runner=runner)
    last_pass = _load_last_pass(paths.last_pass)
    last_pass_age = _last_pass_age_seconds(last_pass)
    return {
        "label": paths.label,
        "installed": paths.plist.is_file(),
        "loaded": loaded,
        "plist": str(paths.plist),
        "repo_root": str(paths.repo_root),
        "python": str(paths.python),
        "interval_seconds": SERVICE_INTERVAL_SECONDS,
        "last_pass": last_pass,
        "last_pass_age_seconds": last_pass_age,
        "last_pass_stale": last_pass_age is None or last_pass_age > DEFAULT_PASS_TIMEOUT_SECONDS * 2,
    }


def service_doctor(
    paths: ServicePaths,
    *,
    runner: CommandRunner = _run_command,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": ok, "detail": detail})

    add("repo_root", paths.repo_root.is_dir(), str(paths.repo_root))
    add(
        "python",
        paths.python.is_file() and os.access(paths.python, os.X_OK),
        str(paths.python),
    )
    payload_ok = False
    payload_detail = str(paths.plist)
    if paths.plist.is_file():
        try:
            payload = plistlib.loads(paths.plist.read_bytes())
            validate_launch_agent_payload(payload, paths)
            payload_ok = True
        except Exception as exc:
            payload_detail = str(redact(str(exc)))
    add("plist", payload_ok, payload_detail)
    add("launchd_loaded", service_is_loaded(paths.label, runner=runner), paths.label)
    last_pass = _load_last_pass(paths.last_pass)
    last_pass_age = _last_pass_age_seconds(last_pass)
    add(
        "last_pass",
        last_pass_age is not None and last_pass_age <= DEFAULT_PASS_TIMEOUT_SECONDS * 2,
        str(paths.last_pass),
    )
    return {"ok": all(check["ok"] for check in checks), "checks": checks}


def _load_repo_environment(repo_root: Path) -> None:
    from rlab.dotenv import load_env_file

    load_env_file(repo_root / ".env")


def _connect_queue(repo_root: Path):
    from rlab.job_queue import connect, database_url

    _load_repo_environment(repo_root)
    return connect(database_url())


@contextmanager
def schema_read_lock(repo_root: Path):
    conn = _connect_queue(repo_root)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT pg_advisory_lock_shared(hashtextextended(%(key)s, 0))",
                {"key": SCHEMA_MAINTENANCE_LOCK},
            )
        yield
    finally:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT pg_advisory_unlock_shared(hashtextextended(%(key)s, 0))",
                    {"key": SCHEMA_MAINTENANCE_LOCK},
                )
        finally:
            conn.close()


def service_is_running(
    label: str = SERVICE_LABEL,
    *,
    uid: int | None = None,
    runner: CommandRunner = _run_command,
) -> bool:
    result = runner(
        ["launchctl", "print", _target(label, uid)],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and "state = running" in (result.stdout or "")


@contextmanager
def schema_change_service_guard(
    paths: ServicePaths,
    *,
    runner: CommandRunner = _run_command,
    timeout_seconds: float = DEFAULT_PASS_TIMEOUT_SECONDS + 30,
):
    loaded = service_is_loaded(paths.label, runner=runner)
    if not loaded:
        yield
        return
    target = _target(paths.label)
    runner(["launchctl", "disable", target], check=True, capture_output=True, text=True)
    try:
        deadline = time.monotonic() + timeout_seconds
        while service_is_running(paths.label, runner=runner):
            if time.monotonic() >= deadline:
                raise TimeoutError("timed out waiting for the active fleet service pass")
            time.sleep(0.25)
        runner(["launchctl", "bootout", target], check=True, capture_output=True, text=True)
        yield
    finally:
        runner(["launchctl", "enable", target], check=False, capture_output=True, text=True)
        if paths.plist.is_file():
            runner(
                ["launchctl", "bootstrap", _domain(), str(paths.plist)],
                check=False,
                capture_output=True,
                text=True,
            )
            kick_service(paths.label, runner=runner)


def _default_count_nonterminal_jobs(repo_root: Path) -> int:
    from rlab import job_queue

    hook = getattr(job_queue, "count_nonterminal_jobs", None)
    if callable(hook):
        return int(hook())
    conn = _connect_queue(repo_root)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) AS count FROM train_jobs WHERE status <> ALL(%(terminal)s)",
                {"terminal": list(TERMINAL_JOB_STATES)},
            )
            row = cur.fetchone()
        return int(row["count"] if isinstance(row, Mapping) else row[0])
    finally:
        conn.close()


def uninstall_service(
    paths: ServicePaths,
    *,
    force: bool = False,
    count_nonterminal_jobs: CountNonterminalJobs = _default_count_nonterminal_jobs,
    runner: CommandRunner = _run_command,
) -> bool:
    if not force:
        count = int(count_nonterminal_jobs(paths.repo_root))
        if count:
            raise RuntimeError(
                f"refusing to uninstall fleet service with {count} nonterminal job(s); use --force"
            )
    loaded = service_is_loaded(paths.label, runner=runner)
    if loaded:
        runner(
            ["launchctl", "bootout", _target(paths.label)],
            check=True,
            capture_output=True,
            text=True,
        )
    existed = paths.plist.exists()
    paths.plist.unlink(missing_ok=True)
    return existed or loaded


def _default_discover_machines(repo_root: Path) -> Sequence[str]:
    from rlab import job_queue
    from rlab.machines import load_machine_registry

    hook = getattr(job_queue, "machines_with_service_work", None)
    if callable(hook):
        work = {str(name) for name in hook()}
    else:
        conn = _connect_queue(repo_root)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT DISTINCT machine
                    FROM train_jobs
                    WHERE machine IS NOT NULL
                      AND (status IN ('pending', 'launching', 'running') OR cancel_requested = TRUE)
                    ORDER BY machine
                    """
                )
                rows = cur.fetchall()
            work = {
                str(row["machine"] if isinstance(row, Mapping) else row[0]) for row in rows
            }
        finally:
            conn.close()
    registry = load_machine_registry(repo_root / "experiments" / "machines.yaml")
    now = time.time()
    for machine_name in registry.machines:
        marker = repo_root / "logs" / "fleet" / f"maintenance-{machine_name}.stamp"
        try:
            maintenance_due = now - marker.stat().st_mtime >= 3600
        except FileNotFoundError:
            maintenance_due = True
        if maintenance_due:
            work.add(machine_name)
    return tuple(sorted(work))


def _default_reconcile_machine(
    repo_root: Path,
    machine_name: str,
    deadline_monotonic: float,
) -> Mapping[str, Any] | None:
    from rlab import fleet

    hook = getattr(fleet, "run_service_machine_pass", None)
    if not callable(hook):
        raise RuntimeError("rlab.fleet.run_service_machine_pass is required by the fleet service")
    return hook(
        machine_name=machine_name,
        machines_path=repo_root / "experiments" / "machines.yaml",
        repo_root=repo_root,
        deadline_monotonic=deadline_monotonic,
    )


def run_service_pass(
    *,
    repo_root: Path,
    state_dir: Path,
    discover_machines: DiscoverMachines = _default_discover_machines,
    reconcile_machine: ReconcileMachine = _default_reconcile_machine,
    max_machine_lanes: int = DEFAULT_MAX_MACHINE_LANES,
    lane_timeout_seconds: float = DEFAULT_LANE_TIMEOUT_SECONDS,
    pass_timeout_seconds: float = DEFAULT_PASS_TIMEOUT_SECONDS,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    started_at = _utc_now()
    started_monotonic = clock()
    pass_deadline = started_monotonic + pass_timeout_seconds
    event_log = ServiceEventLog(state_dir / "service.jsonl")
    summary: dict[str, Any] = {
        "schema_version": 1,
        "started_at": _iso_utc(started_at),
        "repo_root": str(repo_root),
        "status": "running",
        "machines": [],
    }
    event_log.emit("pass_started", repo_root=str(repo_root))
    try:
        machine_names = sorted(set(str(name) for name in discover_machines(repo_root)))
        if not machine_names:
            summary["status"] = "idle"
        else:
            workers = max(1, min(int(max_machine_lanes), len(machine_names)))
            executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="rlab-fleet")
            futures: dict[Future[Mapping[str, Any] | None], tuple[str, float]] = {}
            try:
                for machine_name in machine_names:
                    lane_deadline = min(pass_deadline, clock() + lane_timeout_seconds)
                    future = executor.submit(
                        reconcile_machine,
                        repo_root,
                        machine_name,
                        lane_deadline,
                    )
                    futures[future] = (machine_name, lane_deadline)
                remaining = max(0.0, pass_deadline - clock())
                done, pending = wait(futures, timeout=remaining)
                machine_results: list[dict[str, Any]] = []
                for future in done:
                    machine_name, _deadline = futures[future]
                    try:
                        detail = future.result()
                    except Exception as exc:
                        machine_results.append(
                            {
                                "machine": machine_name,
                                "status": "error",
                                "error": redact(str(exc)),
                            }
                        )
                    else:
                        machine_results.append(
                            {
                                "machine": machine_name,
                                "status": "ok",
                                "detail": redact(dict(detail or {})),
                            }
                        )
                for future in pending:
                    machine_name, _deadline = futures[future]
                    future.cancel()
                    machine_results.append({"machine": machine_name, "status": "timeout"})
                summary["machines"] = sorted(machine_results, key=lambda row: row["machine"])
                summary["status"] = (
                    "ok" if all(row["status"] == "ok" for row in machine_results) else "degraded"
                )
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
    except Exception as exc:
        summary["status"] = "error"
        summary["error"] = redact(str(exc))
    finally:
        summary["finished_at"] = _iso_utc()
        summary["duration_seconds"] = round(max(0.0, clock() - started_monotonic), 6)
        _atomic_write_json(state_dir / "last-pass.json", redact(summary))
        event_log.emit(
            "pass_finished",
            status=summary["status"],
            duration_seconds=summary["duration_seconds"],
            machines=summary["machines"],
            error=summary.get("error"),
        )
        event_log.close()
    return summary


def _tail(path: Path, count: int) -> list[str]:
    if count <= 0:
        return []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            return stream.readlines()[-count:]
    except FileNotFoundError:
        return []


def stream_service_logs(path: Path, *, tail: int = 100, follow: bool = False) -> int:
    for line in _tail(path, tail):
        print(line, end="" if line.endswith("\n") else "\n")
    if not follow:
        return 0
    try:
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            stream.seek(0, os.SEEK_END)
            while True:
                line = stream.readline()
                if line:
                    print(line, end="" if line.endswith("\n") else "\n", flush=True)
                else:
                    time.sleep(0.25)
    except FileNotFoundError:
        raise FileNotFoundError(f"fleet service log does not exist: {path}") from None
    except KeyboardInterrupt:
        return 130


def _paths_from_args(args: argparse.Namespace) -> ServicePaths:
    return default_service_paths(
        repo_root=getattr(args, "repo_root", None),
        state_dir=getattr(args, "state_dir", None),
        launch_agents_dir=getattr(args, "launch_agents_dir", None),
        label=getattr(args, "label", SERVICE_LABEL),
    )


def cmd_install(args: argparse.Namespace) -> int:
    result = install_service(_paths_from_args(args), replace=bool(args.replace))
    print(
        json.dumps(
            {
                "installed": result.installed,
                "replaced": result.replaced,
                "dispatch": "kicked" if result.kicked else "degraded",
            },
            sort_keys=True,
        )
    )
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    status = service_status(_paths_from_args(args))
    if args.json:
        print(json.dumps(status, sort_keys=True))
    else:
        print(
            f"fleet service installed={status['installed']} loaded={status['loaded']} "
            f"interval={status['interval_seconds']}s"
        )
        if status["last_pass"]:
            print(json.dumps(status["last_pass"], sort_keys=True))
    return 0 if status["installed"] and status["loaded"] else 1


def cmd_doctor(args: argparse.Namespace) -> int:
    report = service_doctor(_paths_from_args(args))
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        for check in report["checks"]:
            print(f"{'ok' if check['ok'] else 'FAIL'} {check['name']}: {check['detail']}")
    return 0 if report["ok"] else 1


def cmd_logs(args: argparse.Namespace) -> int:
    return stream_service_logs(
        _paths_from_args(args).service_log,
        tail=int(args.tail),
        follow=bool(args.follow),
    )


def cmd_uninstall(args: argparse.Namespace) -> int:
    changed = uninstall_service(_paths_from_args(args), force=bool(args.force))
    print(json.dumps({"uninstalled": changed}, sort_keys=True))
    return 0


def cmd_run_once(args: argparse.Namespace) -> int:
    paths = _paths_from_args(args)
    with schema_read_lock(paths.repo_root):
        summary = run_service_pass(
            repo_root=paths.repo_root,
            state_dir=paths.state_dir,
            max_machine_lanes=int(args.max_machine_lanes),
            lane_timeout_seconds=float(args.lane_timeout),
            pass_timeout_seconds=float(args.pass_timeout),
        )
    return 0 if summary["status"] in {"idle", "ok"} else 1


def _add_path_arguments(
    parser: argparse.ArgumentParser, *, include_launch_agents: bool = True
) -> None:
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--state-dir", type=Path, default=None)
    if include_launch_agents:
        parser.add_argument("--launch-agents-dir", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--label", default=SERVICE_LABEL, help=argparse.SUPPRESS)


def add_service_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    service = subparsers.add_parser("service", help="Manage the short-lived launchd fleet service.")
    commands = service.add_subparsers(dest="service_command", required=True)

    install = commands.add_parser("install", help="Install and start the fleet LaunchAgent.")
    _add_path_arguments(install)
    install.add_argument("--replace", action="store_true")
    install.set_defaults(func=cmd_install)

    status = commands.add_parser("status", help="Show launchd and last-pass status.")
    _add_path_arguments(status)
    status.add_argument("--json", action="store_true")
    status.set_defaults(func=cmd_status)

    doctor = commands.add_parser("doctor", help="Validate the service installation.")
    _add_path_arguments(doctor)
    doctor.add_argument("--json", action="store_true")
    doctor.set_defaults(func=cmd_doctor)

    logs = commands.add_parser("logs", help="Read rotating service logs.")
    _add_path_arguments(logs)
    logs.add_argument("--tail", type=int, default=100)
    logs.add_argument("--follow", action="store_true")
    logs.set_defaults(func=cmd_logs)

    uninstall = commands.add_parser("uninstall", help="Uninstall the fleet LaunchAgent.")
    _add_path_arguments(uninstall)
    uninstall.add_argument("--force", action="store_true")
    uninstall.set_defaults(func=cmd_uninstall)
    return service


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run and administer the rlab launchd fleet service."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_service_parser(subparsers)

    run_once = subparsers.add_parser("run-once", help=argparse.SUPPRESS)
    _add_path_arguments(run_once, include_launch_agents=False)
    run_once.add_argument("--max-machine-lanes", type=int, default=DEFAULT_MAX_MACHINE_LANES)
    run_once.add_argument("--lane-timeout", type=float, default=DEFAULT_LANE_TIMEOUT_SECONDS)
    run_once.add_argument("--pass-timeout", type=float, default=DEFAULT_PASS_TIMEOUT_SECONDS)
    run_once.set_defaults(func=cmd_run_once)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
