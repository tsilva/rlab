from __future__ import annotations

import argparse
import hashlib
import json
import os
import plistlib
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from logging import Formatter, Logger
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit


SERVICE_LABEL = "com.rlab.fleet-service"
SERVICE_INTERVAL_SECONDS = 30
DEFAULT_LANE_TIMEOUT_SECONDS = 120.0
DEFAULT_PASS_TIMEOUT_SECONDS = 300.0
DEFAULT_MAX_MACHINE_LANES = 4
SCHEMA_MAINTENANCE_LOCK = "rlab-fleet-schema-maintenance"

CommandRunner = Callable[..., subprocess.CompletedProcess[str]]
DiscoverMachines = Callable[[Path], Sequence[str]]
ReconcileMachine = Callable[[Path, str, float], Mapping[str, Any] | None]
ReconcileEval = Callable[[Path, float], Mapping[str, Any] | None]
CountNonterminalJobs = Callable[[Path], int]
ServiceNotifier = Callable[[str, str], None]

SERVICE_ALERT_AFTER_FAILURES = 2
SERVICE_ALERT_REPEAT_SECONDS = 3600


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

    @property
    def health(self) -> Path:
        return self.state_dir / "health.json"


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
            "rlab.fleet_service_entrypoint",
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
    r"(?:password|passwd|secret|token|api[_-]?key|access[_-]?key|database[_-]?url|dsn|"
    r"presigned|signed[_-]?url|(?:get|put)[_-]?url)",
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
        text = _ASSIGNMENT_SECRET.sub(lambda match: f"{match.group(1)}=[REDACTED]", text)
        if text.startswith(("http://", "https://")):
            parts = urlsplit(text)
            if parts.query:
                text = urlunsplit(
                    (parts.scheme, parts.netloc, parts.path, "[REDACTED]", parts.fragment)
                )
        return text
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
        [str(paths.python), "-m", "rlab.fleet_service_entrypoint", "--help"],
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
    except FileNotFoundError, OSError, json.JSONDecodeError:
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
    last_pass_stale = (
        last_pass_age is None or last_pass_age > DEFAULT_PASS_TIMEOUT_SECONDS * 2
    )
    last_pass_status = str((last_pass or {}).get("status") or "missing")
    health = _load_last_pass(paths.health)
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
        "last_pass_stale": last_pass_stale,
        "last_pass_status": last_pass_status,
        "healthy": bool(
            loaded
            and not last_pass_stale
            and last_pass_status in {"idle", "ok"}
        ),
        "health": health,
    }


def _default_service_notifier(title: str, message: str) -> None:
    script = (
        f"display notification {json.dumps(message)} "
        f"with title {json.dumps(title)}"
    )
    subprocess.run(
        ["/usr/bin/osascript", "-e", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )


def _service_failure_summary(summary: Mapping[str, Any]) -> str:
    details: list[str] = []
    if summary.get("error"):
        details.append(str(summary["error"]))
    eval_lane = summary.get("eval") or {}
    if isinstance(eval_lane, Mapping) and eval_lane.get("status") not in {None, "ok"}:
        details.append(f"eval: {eval_lane.get('error') or eval_lane.get('status')}")
    for row in summary.get("machines") or []:
        if isinstance(row, Mapping) and row.get("status") != "ok":
            details.append(f"{row.get('machine')}: {row.get('error') or row.get('status')}")
    return "; ".join(details)[:500] or str(summary.get("status") or "unknown failure")


def record_service_health(
    paths: ServicePaths,
    summary: Mapping[str, Any],
    *,
    notifier: ServiceNotifier = _default_service_notifier,
) -> dict[str, Any]:
    previous = _load_last_pass(paths.health) or {}
    status = str(summary.get("status") or "error")
    healthy = status in {"idle", "ok"}
    now = _utc_now()
    prior_failures = int(previous.get("consecutive_failures") or 0)
    prior_alert_active = bool(previous.get("alert_active"))
    detail = "" if healthy else _service_failure_summary(summary)
    fingerprint = hashlib.sha256(detail.encode("utf-8")).hexdigest() if detail else ""
    last_notified_at = previous.get("last_notified_at")
    last_notified_age = None
    if last_notified_at:
        try:
            last_notified = datetime.fromisoformat(
                str(last_notified_at).replace("Z", "+00:00")
            )
            last_notified_age = (now - last_notified.astimezone(UTC)).total_seconds()
        except ValueError:
            last_notified_age = None

    notified_at = last_notified_at
    if healthy:
        if prior_alert_active:
            notifier("rlab fleet service recovered", "The fleet reconciler is healthy again.")
            notified_at = _iso_utc(now)
        state = {
            "healthy": True,
            "status": status,
            "consecutive_failures": 0,
            "alert_active": False,
            "failure_fingerprint": "",
            "last_failure": None,
            "last_notified_at": notified_at,
            "updated_at": _iso_utc(now),
        }
    else:
        consecutive = prior_failures + 1
        should_notify = consecutive >= SERVICE_ALERT_AFTER_FAILURES and (
            not prior_alert_active
            or fingerprint != str(previous.get("failure_fingerprint") or "")
            or last_notified_age is None
            or last_notified_age >= SERVICE_ALERT_REPEAT_SECONDS
        )
        if should_notify:
            notifier("rlab fleet service failure", detail)
            notified_at = _iso_utc(now)
        state = {
            "healthy": False,
            "status": status,
            "consecutive_failures": consecutive,
            "alert_active": prior_alert_active or should_notify,
            "failure_fingerprint": fingerprint,
            "last_failure": detail,
            "last_notified_at": notified_at,
            "updated_at": _iso_utc(now),
        }
    _atomic_write_json(paths.health, state)
    return state


def eval_service_health(
    paths: ServicePaths,
    *,
    runner: CommandRunner = _run_command,
) -> dict[str, Any]:
    """Return whether a recent fleet pass successfully reconciled evaluation."""
    status = service_status(paths, runner=runner)
    last_pass = status.get("last_pass") or {}
    eval_lane = last_pass.get("eval") or {}
    eval_detail = eval_lane.get("detail") or {}
    eval_status = str(eval_lane.get("status") or "missing")
    detail_status = str(eval_detail.get("status") or "missing")
    ready = bool(
        status.get("loaded")
        and not status.get("last_pass_stale")
        and eval_status == "ok"
        and detail_status == "ok"
    )
    return {
        "ready": ready,
        "loaded": bool(status.get("loaded")),
        "last_pass_stale": bool(status.get("last_pass_stale")),
        "last_pass_age_seconds": status.get("last_pass_age_seconds"),
        "eval_status": eval_status,
        "eval_detail_status": detail_status,
        "error": eval_lane.get("error") or eval_detail.get("error"),
        "app_cleanup": eval_detail.get("app_cleanup"),
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
    add(
        "last_pass_status",
        str((last_pass or {}).get("status") or "missing") in {"idle", "ok"},
        str((last_pass or {}).get("status") or "missing"),
    )
    eval_health = eval_service_health(paths, runner=runner)
    add(
        "last_eval_pass",
        bool(eval_health["ready"]),
        json.dumps(eval_health, sort_keys=True, default=str),
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
    from rlab.job_queue import count_nonterminal_jobs

    conn = _connect_queue(repo_root)
    try:
        return int(count_nonterminal_jobs(conn))
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
    from rlab.job_queue import machines_with_service_work
    from rlab.machines import load_machine_registry

    conn = _connect_queue(repo_root)
    try:
        work = {str(name) for name in machines_with_service_work(conn)}
    finally:
        conn.close()
    registry = load_machine_registry(repo_root / "experiments" / "machines.yaml")
    now = time.time()
    for machine_name, machine in registry.machines.items():
        if machine.prewarm_latest_runtime:
            work.add(machine_name)
        marker = repo_root / "logs" / "fleet" / f"maintenance-attempt-{machine_name}.stamp"
        try:
            maintenance_due = now - marker.stat().st_mtime >= 3600
        except FileNotFoundError:
            maintenance_due = True
        if maintenance_due:
            work.add(machine_name)
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.touch()
    return tuple(sorted(work))


def _default_reconcile_machine(
    repo_root: Path,
    machine_name: str,
    deadline_monotonic: float,
) -> Mapping[str, Any] | None:
    from rlab.fleet import run_service_machine_pass

    return run_service_machine_pass(
        machine_name=machine_name,
        machines_path=repo_root / "experiments" / "machines.yaml",
        repo_root=repo_root,
        deadline_monotonic=deadline_monotonic,
    )


def _default_reconcile_eval(
    repo_root: Path,
    deadline_monotonic: float,
) -> Mapping[str, Any] | None:
    config_path = repo_root / "experiments" / "modal_eval.yaml"
    if not config_path.is_file() and not (repo_root / ".env").is_file():
        return {"status": "unconfigured"}
    if config_path.is_file():
        from rlab.modal_eval_orchestrator import run_service_eval_pass

        detail = dict(
            run_service_eval_pass(
                repo_root=repo_root,
                deadline_monotonic=deadline_monotonic,
            )
            or {}
        )
    else:
        detail = {"status": "unconfigured"}
    from rlab.fleet_wandb_publisher import drain_cycle
    from rlab.telemetry_mailbox import (
        consume_attempt_events,
        discard_disabled_metric_batches,
        finalize_mailbox_runs_without_eval,
        mailbox_storage_bytes,
    )

    conn = _connect_queue(repo_root)
    try:
        detail["consumed_attempt_events"] = consume_attempt_events(conn)
        detail["discarded_disabled_metric_streams"] = discard_disabled_metric_batches(conn)
        detail["wandb_publication"] = drain_cycle(conn)
        detail["finalized_mailbox_runs_without_eval"] = finalize_mailbox_runs_without_eval(conn)
        storage_bytes = mailbox_storage_bytes(conn)
        detail["metric_mailbox_bytes"] = storage_bytes
        detail["metric_mailbox_pressure"] = (
            "hard" if storage_bytes >= 5 * 1024**3
            else "soft" if storage_bytes >= 1024**3
            else "ok"
        )
    finally:
        conn.close()
    return detail


def run_service_pass(
    *,
    repo_root: Path,
    state_dir: Path,
    discover_machines: DiscoverMachines = _default_discover_machines,
    reconcile_machine: ReconcileMachine = _default_reconcile_machine,
    reconcile_eval: ReconcileEval = _default_reconcile_eval,
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
        "eval": {"status": "running"},
    }
    event_log.emit("pass_started", repo_root=str(repo_root))
    try:
        machine_names = sorted(set(str(name) for name in discover_machines(repo_root)))
        machine_workers = min(int(max_machine_lanes), len(machine_names))
        workers = max(1, machine_workers + 1)
        if workers:
            executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="rlab-fleet")
            futures: dict[Future[Mapping[str, Any] | None], tuple[str, str, float]] = {}
            try:
                for machine_name in machine_names:
                    lane_deadline = min(pass_deadline, clock() + lane_timeout_seconds)
                    future = executor.submit(
                        reconcile_machine,
                        repo_root,
                        machine_name,
                        lane_deadline,
                    )
                    futures[future] = ("machine", machine_name, lane_deadline)
                eval_deadline = min(pass_deadline, clock() + lane_timeout_seconds)
                eval_future = executor.submit(reconcile_eval, repo_root, eval_deadline)
                futures[eval_future] = ("eval", "modal", eval_deadline)
                remaining = max(0.0, pass_deadline - clock())
                done, pending = wait(futures, timeout=remaining)
                machine_results: list[dict[str, Any]] = []
                for future in done:
                    lane_kind, lane_name, _deadline = futures[future]
                    try:
                        detail = future.result()
                    except Exception as exc:
                        row = {"status": "error", "error": redact(str(exc))}
                        if lane_kind == "machine":
                            machine_results.append({"machine": lane_name, **row})
                        else:
                            summary["eval"] = row
                    else:
                        row = {"status": "ok", "detail": redact(dict(detail or {}))}
                        if lane_kind == "machine":
                            machine_results.append({"machine": lane_name, **row})
                        else:
                            summary["eval"] = row
                for future in pending:
                    lane_kind, lane_name, _deadline = futures[future]
                    future.cancel()
                    if lane_kind == "machine":
                        machine_results.append({"machine": lane_name, "status": "timeout"})
                    else:
                        summary["eval"] = {"status": "timeout"}
                summary["machines"] = sorted(machine_results, key=lambda row: row["machine"])
                all_ok = (
                    all(row["status"] == "ok" for row in machine_results)
                    and summary["eval"]["status"] == "ok"
                )
                eval_detail_status = summary["eval"].get("detail", {}).get("status")
                summary["status"] = (
                    "idle"
                    if not machine_results and eval_detail_status in {"disabled", "unconfigured"}
                    else "ok"
                    if all_ok
                    else "degraded"
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
            eval=summary["eval"],
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
            f"healthy={status['healthy']} interval={status['interval_seconds']}s"
        )
        if status["last_pass"]:
            print(json.dumps(status["last_pass"], sort_keys=True))
    return 0 if status["installed"] and status["loaded"] and status["healthy"] else 1


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
    try:
        with schema_read_lock(paths.repo_root):
            summary = run_service_pass(
                repo_root=paths.repo_root,
                state_dir=paths.state_dir,
                max_machine_lanes=int(args.max_machine_lanes),
                lane_timeout_seconds=float(args.lane_timeout),
                pass_timeout_seconds=float(args.pass_timeout),
            )
    except Exception as exc:
        traceback.print_exc()
        now = _iso_utc()
        summary = {
            "schema_version": 1,
            "started_at": now,
            "finished_at": now,
            "repo_root": str(paths.repo_root),
            "status": "error",
            "machines": [],
            "eval": {"status": "error"},
            "error": redact(f"{type(exc).__name__}: {exc}"),
        }
        _atomic_write_json(paths.last_pass, summary)
    record_service_health(paths, summary)
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
