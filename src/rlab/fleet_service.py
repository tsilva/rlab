from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import plistlib
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from logging import Formatter, Logger
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from rlab.cli_commands import (
    eval_modal_abandon_command,
    eval_modal_recover_command,
    experiment_retry_finalization_command,
    render_command,
)


SERVICE_LABEL = "com.rlab.fleet-service"
SERVICE_INTERVAL_SECONDS = 30
CONTROLLER_NAMES = ("machine", "evaluation", "wandb", "workspace")
CONTROLLER_POLL_SECONDS = 2
CONTROL_PLANE_PROTOCOL_VERSION = 4
CONTROLLER_READINESS_TIMEOUT_SECONDS = 120.0
CONTROLLER_INSTALL_HEARTBEAT_TIMEOUT_SECONDS = 180.0
CONTROLLER_HEARTBEAT_MAX_AGE_SECONDS = 70.0
CONTROLLER_HEARTBEAT_PROTOCOL_VERSION = 2
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
WorkloadSnapshot = Callable[[Path], Mapping[str, Any]]

SERVICE_ALERT_AFTER_FAILURES = 2
SERVICE_ALERT_REPEAT_SECONDS = 3600
SERVICE_BOOTSTRAP_ATTEMPTS = 10
SERVICE_BOOTSTRAP_RETRY_SECONDS = 0.25
SERVICE_BOOTOUT_ATTEMPTS = 150
SERVICE_BOOTOUT_RETRY_SECONDS = 0.1


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
    def current_pass(self) -> Path:
        return self.state_dir / "current-pass.json"

    @property
    def wake_requests(self) -> Path:
        return self.state_dir / "wake-requests.json"

    @property
    def health(self) -> Path:
        return self.state_dir / "health.json"

    @property
    def readiness(self) -> Path:
        return self.state_dir / "readiness.json"


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


def controller_source_fingerprint(repo_root: Path | str) -> str:
    """Hash the local runtime inputs loaded by persistent fleet controllers."""

    root = Path(repo_root).expanduser().resolve()
    candidates = sorted(
        path
        for path in (root / "src" / "rlab").rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    )
    candidates.extend(path for path in (root / "pyproject.toml", root / "uv.lock") if path.is_file())
    digest = hashlib.sha256()
    for path in candidates:
        if not path.is_file():
            continue
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


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


def controller_service_paths(paths: ServicePaths, controller: str) -> ServicePaths:
    if controller not in CONTROLLER_NAMES:
        raise ValueError(f"unknown fleet controller: {controller}")
    return ServicePaths(
        repo_root=paths.repo_root,
        python=paths.python,
        state_dir=paths.state_dir / controller,
        launch_agents_dir=paths.launch_agents_dir,
        label=f"{paths.label}.{controller}",
    )


def controller_launch_agent_payload(paths: ServicePaths, controller: str) -> dict[str, Any]:
    controller_paths = controller_service_paths(paths, controller)
    return {
        "Label": controller_paths.label,
        "ProgramArguments": [
            str(paths.python),
            "-m",
            "rlab.fleet_controllers",
            controller,
            "--repo-root",
            str(paths.repo_root),
            "--protocol-version",
            str(CONTROL_PLANE_PROTOCOL_VERSION),
        ],
        "WorkingDirectory": str(paths.repo_root),
        "KeepAlive": True,
        "ThrottleInterval": CONTROLLER_POLL_SECONDS,
        "ProcessType": "Background",
        "StandardOutPath": str(controller_paths.bootstrap_stdout),
        "StandardErrorPath": str(controller_paths.bootstrap_stderr),
    }


def validate_controller_launch_agent_payload(
    payload: Mapping[str, Any], paths: ServicePaths, controller: str
) -> None:
    expected = controller_launch_agent_payload(paths, controller)
    forbidden = {
        "RunAtLoad",
        "StartInterval",
        "EnvironmentVariables",
        "UserName",
        "GroupName",
    }
    found_forbidden = forbidden.intersection(payload)
    if found_forbidden:
        raise ValueError(
            f"controller launch agent contains forbidden keys: {sorted(found_forbidden)}"
        )
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"controller launch agent {key} must be {value!r}")
    if payload.get("KeepAlive") is not True:
        raise ValueError("controller launch agent KeepAlive must be true")
    if payload.get("ThrottleInterval") != CONTROLLER_POLL_SECONDS:
        raise ValueError(
            f"controller launch agent ThrottleInterval must be {CONTROLLER_POLL_SECONDS}"
        )
    for value in (
        paths.repo_root,
        paths.python,
        paths.state_dir,
        paths.launch_agents_dir,
    ):
        if not value.is_absolute():
            raise ValueError(f"controller launch agent path must be absolute: {value}")


def render_controller_launch_agent_plist(paths: ServicePaths, controller: str) -> bytes:
    payload = controller_launch_agent_payload(paths, controller)
    validate_controller_launch_agent_payload(payload, paths, controller)
    data = plistlib.dumps(payload, fmt=plistlib.FMT_XML, sort_keys=True)
    parsed = plistlib.loads(data)
    validate_controller_launch_agent_payload(parsed, paths, controller)
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


@contextmanager
def _state_file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def record_wake_request(
    reason: str,
    *,
    entity_kind: str | None = None,
    entity_id: str | int | None = None,
    state_dir: Path | str | None = None,
) -> None:
    """Persist why a mutating command is waking launchd without exposing command data."""

    paths = default_service_paths(state_dir=state_dir)
    request = {
        "requested_at": _iso_utc(),
        "reason": str(reason).strip() or "unknown",
        "entity_kind": str(entity_kind or ""),
        "entity_id": str(entity_id or ""),
    }
    lock_path = paths.wake_requests.with_suffix(".lock")
    with _state_file_lock(lock_path):
        existing = _load_last_pass(paths.wake_requests) or {}
        requests = [row for row in existing.get("requests") or [] if isinstance(row, Mapping)]
        requests.append(request)
        _atomic_write_json(paths.wake_requests, {"schema_version": 1, "requests": requests[-32:]})


def consume_wake_requests(state_dir: Path) -> list[dict[str, str]]:
    path = state_dir / "wake-requests.json"
    lock_path = path.with_suffix(".lock")
    with _state_file_lock(lock_path):
        existing = _load_last_pass(path) or {}
        path.unlink(missing_ok=True)
    requests: list[dict[str, str]] = []
    for row in existing.get("requests") or []:
        if not isinstance(row, Mapping):
            continue
        requests.append(
            {
                "requested_at": str(row.get("requested_at") or ""),
                "reason": str(row.get("reason") or "unknown"),
                "entity_kind": str(row.get("entity_kind") or ""),
                "entity_id": str(row.get("entity_id") or ""),
            }
        )
    return requests


def _source_state(repo_root: Path) -> dict[str, Any]:
    commit = "unknown"
    dirty = True
    try:
        result = _run_command(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            commit = result.stdout.strip()
        dirty_result = _run_command(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=str(repo_root),
            check=False,
        )
        dirty = dirty_result.returncode != 0 or bool(dirty_result.stdout.strip())
    except OSError:
        pass
    return {"sha": commit, "dirty": dirty}


class PassProgress:
    """Thread-safe, atomic live state for one bounded scheduler pass."""

    def __init__(self, path: Path, payload: Mapping[str, Any]) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._closed = False
        self._payload = dict(payload)
        self._payload.setdefault("lanes", {})
        self._write()

    def _write(self) -> None:
        if self._closed:
            return
        self._payload["updated_at"] = _iso_utc()
        _atomic_write_json(self.path, redact(self._payload))

    def set_workload(self, workload: Mapping[str, Any]) -> None:
        with self._lock:
            self._payload["workload"] = dict(workload)
            self._write()

    def start_lane(
        self,
        lane: str,
        *,
        action: str,
        reason: str,
        deadline_at: str,
        entity_kind: str = "",
        entity_id: str | int = "",
        blast_radius: str = "",
        command: str = "",
    ) -> None:
        with self._lock:
            lanes = dict(self._payload.get("lanes") or {})
            lanes[lane] = {
                "lane": lane,
                "action": action,
                "reason": reason,
                "started_at": _iso_utc(),
                "deadline_at": deadline_at,
                "entity_kind": entity_kind,
                "entity_id": str(entity_id),
                "blast_radius": blast_radius,
                "command": command,
            }
            self._payload["lanes"] = lanes
            self._write()

    def finish_lane(self, lane: str) -> None:
        with self._lock:
            lanes = dict(self._payload.get("lanes") or {})
            lanes.pop(lane, None)
            self._payload["lanes"] = lanes
            self._write()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self.path.unlink(missing_ok=True)


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


def _wait_for_service_unloaded(
    label: str,
    *,
    runner: CommandRunner,
) -> None:
    """Wait for launchd's asynchronous bootout transaction to release a label."""

    for attempt in range(SERVICE_BOOTOUT_ATTEMPTS):
        if not service_is_loaded(label, runner=runner):
            return
        if attempt + 1 < SERVICE_BOOTOUT_ATTEMPTS:
            time.sleep(SERVICE_BOOTOUT_RETRY_SECONDS)
    raise TimeoutError(
        f"launchd did not unload {_target(label)} within "
        f"{SERVICE_BOOTOUT_ATTEMPTS * SERVICE_BOOTOUT_RETRY_SECONDS:.1f}s"
    )


def kick_service(
    label: str = SERVICE_LABEL,
    *,
    uid: int | None = None,
    reason: str | None = None,
    entity_kind: str | None = None,
    entity_id: str | int | None = None,
    state_dir: Path | str | None = None,
    runner: CommandRunner = _run_command,
) -> bool:
    if reason:
        record_wake_request(
            reason,
            entity_kind=entity_kind,
            entity_id=entity_id,
            state_dir=state_dir,
        )
    controller_by_entity = {
        "batch": "machine",
        "machine": "machine",
        "train": "machine",
        "eval": "evaluation",
        "evaluation": "evaluation",
        "wandb": "wandb",
    }
    controller = controller_by_entity.get(str(entity_kind or "").strip().lower())
    target_label = f"{label}.{controller}" if label == SERVICE_LABEL and controller else label
    command = ["launchctl", "kickstart", _target(target_label, uid)]
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


def _validate_controller_entrypoint(paths: ServicePaths, *, runner: CommandRunner) -> None:
    environment = {
        "HOME": str(Path.home()),
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "TMPDIR": tempfile.gettempdir(),
    }
    result = runner(
        [str(paths.python), "-m", "rlab.fleet_controllers", "--help"],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(paths.repo_root),
        env=environment,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "fleet controller entrypoint validation failed: "
            + str(redact(result.stderr or result.stdout or f"exit={result.returncode}"))
        )


@dataclass(frozen=True)
class InstallResult:
    installed: bool
    replaced: bool
    kicked: bool


def _bootstrap_launch_agent(
    paths: ServicePaths,
    *,
    runner: CommandRunner,
    retry_busy: bool,
) -> None:
    attempts = SERVICE_BOOTSTRAP_ATTEMPTS if retry_busy else 1
    command = ["launchctl", "bootstrap", _domain(), str(paths.plist)]
    for attempt in range(attempts):
        try:
            runner(command, check=True, capture_output=True, text=True)
            return
        except subprocess.CalledProcessError as exc:
            if exc.returncode == 5 and attempt + 1 < attempts:
                time.sleep(SERVICE_BOOTSTRAP_RETRY_SECONDS)
                continue
            detail = redact(exc.stderr or exc.stdout or "no launchctl diagnostic output")
            raise RuntimeError(
                f"launchctl bootstrap failed for {paths.label} with exit "
                f"{exc.returncode}: {detail}"
            ) from exc


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
            _wait_for_service_unloaded(paths.label, runner=runner)
        os.replace(candidate, paths.plist)
        try:
            _bootstrap_launch_agent(
                paths,
                runner=runner,
                retry_busy=was_loaded,
            )
        except Exception as install_error:
            if old_data is None:
                paths.plist.unlink(missing_ok=True)
            else:
                _atomic_write(paths.plist, old_data)
                if was_loaded:
                    try:
                        _bootstrap_launch_agent(paths, runner=runner, retry_busy=True)
                    except Exception as rollback_error:
                        raise RuntimeError(
                            "fleet service install failed "
                            f"({install_error}); rollback also failed: {rollback_error}"
                        ) from install_error
            raise
    finally:
        candidate.unlink(missing_ok=True)

    return InstallResult(
        installed=True,
        replaced=old_data is not None,
        kicked=kick_service(
            paths.label,
            reason="service_install",
            state_dir=paths.state_dir,
            runner=runner,
        ),
    )


def install_controller_services(
    paths: ServicePaths,
    *,
    replace: bool = False,
    runner: CommandRunner = _run_command,
) -> InstallResult:
    """Install the independent persistent fleet controllers atomically enough for launchd.

    Each controller has its own label and process. Existing controller plists are restored if a
    later bootstrap fails. The retired combined pass is removed only after all controllers
    are loaded successfully.
    """

    install_started_at = time.time()
    if not paths.repo_root.is_dir():
        raise FileNotFoundError(f"repository root does not exist: {paths.repo_root}")
    if not paths.python.is_file() or not os.access(paths.python, os.X_OK):
        raise FileNotFoundError(f"service interpreter is not executable: {paths.python}")

    controller_paths = [controller_service_paths(paths, name) for name in CONTROLLER_NAMES]
    previous: dict[str, tuple[bytes | None, bool]] = {}
    for name, item in zip(CONTROLLER_NAMES, controller_paths, strict=True):
        old_data = item.plist.read_bytes() if item.plist.exists() else None
        if old_data is not None and not replace:
            raise FileExistsError(f"launch agent already exists: {item.plist}; use --replace")
        previous[name] = (old_data, service_is_loaded(item.label, runner=runner))

    legacy_loaded = service_is_loaded(paths.label, runner=runner)
    replacing_loaded_control_plane = legacy_loaded or any(
        was_loaded for _old_data, was_loaded in previous.values()
    )
    if replace and replacing_loaded_control_plane:
        count = _default_count_nonterminal_jobs(paths.repo_root)
        if count:
            raise RuntimeError(
                "refusing to replace machine/evaluation controllers with "
                f"{count} nonterminal job(s); wait for quiescence"
            )

    paths.launch_agents_dir.mkdir(parents=True, exist_ok=True)
    paths.state_dir.mkdir(parents=True, exist_ok=True)
    _validate_controller_entrypoint(paths, runner=runner)
    installed: list[tuple[str, ServicePaths]] = []
    try:
        for name, item in zip(CONTROLLER_NAMES, controller_paths, strict=True):
            item.state_dir.mkdir(parents=True, exist_ok=True)
            data = render_controller_launch_agent_plist(paths, name)
            candidate = item.plist.with_name(f".{item.plist.name}.candidate")
            _atomic_write(candidate, data)
            try:
                _validate_candidate_plist(candidate, runner=runner)
                old_data, was_loaded = previous[name]
                if was_loaded:
                    runner(
                        ["launchctl", "bootout", _target(item.label)],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    _wait_for_service_unloaded(item.label, runner=runner)
                os.replace(candidate, item.plist)
                installed.append((name, item))
                _bootstrap_launch_agent(item, runner=runner, retry_busy=was_loaded)
            finally:
                candidate.unlink(missing_ok=True)
        kicked = True
        for item in controller_paths:
            result = runner(
                ["launchctl", "kickstart", _target(item.label)],
                check=False,
                capture_output=True,
                text=True,
            )
            kicked = kicked and result.returncode == 0
        if CONTROL_PLANE_PROTOCOL_VERSION >= CONTROLLER_HEARTBEAT_PROTOCOL_VERSION:
            deadline = time.monotonic() + CONTROLLER_INSTALL_HEARTBEAT_TIMEOUT_SECONDS
            expected_fingerprint = controller_source_fingerprint(paths.repo_root)
            while time.monotonic() < deadline:
                ready = True
                for name, item in zip(CONTROLLER_NAMES, controller_paths, strict=True):
                    heartbeat = _load_last_pass(item.state_dir / "heartbeat.json") or {}
                    pid = _service_pid(item.label, runner=runner)
                    matches, _age = _controller_heartbeat_matches(
                        heartbeat,
                        controller=name,
                        pid=pid,
                        source_fingerprint=expected_fingerprint,
                        updated_after=install_started_at,
                    )
                    ready = ready and matches
                if ready:
                    break
                time.sleep(0.05)
            else:
                raise RuntimeError(
                    "fleet controllers did not publish matching heartbeat evidence"
                )
    except Exception as install_error:
        rollback_errors: list[str] = []
        for name, item in reversed(installed):
            if service_is_loaded(item.label, runner=runner):
                runner(
                    ["launchctl", "bootout", _target(item.label)],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                try:
                    _wait_for_service_unloaded(item.label, runner=runner)
                except Exception as exc:
                    rollback_errors.append(f"{name} unload: {exc}")
            old_data, was_loaded = previous[name]
            if old_data is None:
                item.plist.unlink(missing_ok=True)
            else:
                _atomic_write(item.plist, old_data)
                if was_loaded:
                    try:
                        _bootstrap_launch_agent(item, runner=runner, retry_busy=True)
                    except Exception as exc:
                        rollback_errors.append(f"{name} restore: {exc}")
        if rollback_errors:
            raise RuntimeError(
                f"fleet controller install failed ({install_error}); rollback also failed: "
                + "; ".join(rollback_errors)
            ) from install_error
        raise

    if legacy_loaded:
        runner(
            ["launchctl", "bootout", _target(paths.label)],
            check=True,
            capture_output=True,
            text=True,
        )
        _wait_for_service_unloaded(paths.label, runner=runner)
    paths.plist.unlink(missing_ok=True)
    return InstallResult(
        installed=True,
        replaced=replace and any(value[0] is not None for value in previous.values()),
        kicked=kicked,
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


def _installed_controller_protocol_version(item: ServicePaths) -> int | None:
    if not item.plist.is_file():
        return None
    try:
        payload = plistlib.loads(item.plist.read_bytes())
        arguments = list(payload.get("ProgramArguments") or [])
        index = arguments.index("--protocol-version")
        return int(arguments[index + 1])
    except OSError, ValueError, TypeError, IndexError, plistlib.InvalidFileException:
        return None


def _controller_heartbeat_runtime_matches(
    heartbeat: Mapping[str, Any],
    *,
    controller: str,
    pid: int | None,
    updated_after: float | None = None,
) -> tuple[bool, float | None]:
    updated_at = heartbeat.get("updated_at")
    try:
        heartbeat_age = max(0.0, time.time() - float(updated_at))
    except (TypeError, ValueError):
        return False, None
    try:
        matches = bool(
            heartbeat_age <= CONTROLLER_HEARTBEAT_MAX_AGE_SECONDS
            and str(heartbeat.get("controller") or "") == controller
            and int(heartbeat.get("protocol_version") or 0)
            == CONTROL_PLANE_PROTOCOL_VERSION
            and int(heartbeat.get("pid") or 0) == int(pid or 0)
            and int(pid or 0) > 0
            and heartbeat.get("last_success_at") is not None
            and str(heartbeat.get("phase") or "") in {"idle", "reconciling"}
            and not heartbeat.get("last_error")
            and (
                updated_after is None
                or float(heartbeat.get("updated_at") or 0.0) >= float(updated_after)
            )
        )
    except (TypeError, ValueError):
        matches = False
    return matches, heartbeat_age


def _controller_heartbeat_matches(
    heartbeat: Mapping[str, Any],
    *,
    controller: str,
    pid: int | None,
    source_fingerprint: str,
    updated_after: float | None = None,
) -> tuple[bool, float | None]:
    matches, heartbeat_age = _controller_heartbeat_runtime_matches(
        heartbeat,
        controller=controller,
        pid=pid,
        updated_after=updated_after,
    )
    return (
        matches
        and str(heartbeat.get("source_fingerprint") or "") == source_fingerprint,
        heartbeat_age,
    )


def service_status(
    paths: ServicePaths,
    *,
    runner: CommandRunner = _run_command,
) -> dict[str, Any]:
    loaded = service_is_loaded(paths.label, runner=runner)
    try:
        running = service_is_running(paths.label, runner=runner) if loaded else False
    except OSError:
        running = False
    last_pass = _load_last_pass(paths.last_pass)
    last_pass_age = _last_pass_age_seconds(last_pass)
    last_pass_stale = last_pass_age is None or last_pass_age > DEFAULT_PASS_TIMEOUT_SECONDS * 2
    last_pass_status = str((last_pass or {}).get("status") or "missing")
    health = _load_last_pass(paths.health)
    return {
        "label": paths.label,
        "installed": paths.plist.is_file(),
        "loaded": loaded,
        "running": running,
        "plist": str(paths.plist),
        "repo_root": str(paths.repo_root),
        "python": str(paths.python),
        "interval_seconds": SERVICE_INTERVAL_SECONDS,
        "last_pass": last_pass,
        "last_pass_age_seconds": last_pass_age,
        "last_pass_stale": last_pass_stale,
        "last_pass_status": last_pass_status,
        "healthy": bool(loaded and not last_pass_stale and last_pass_status in {"idle", "ok"}),
        "health": health,
    }


def controller_services_status(
    paths: ServicePaths,
    *,
    runner: CommandRunner = _run_command,
) -> dict[str, Any]:
    controllers: dict[str, dict[str, Any]] = {}
    expected_fingerprint = controller_source_fingerprint(paths.repo_root)
    heartbeat_required = CONTROL_PLANE_PROTOCOL_VERSION >= CONTROLLER_HEARTBEAT_PROTOCOL_VERSION
    for name in CONTROLLER_NAMES:
        item = controller_service_paths(paths, name)
        installed_protocol_version = _installed_controller_protocol_version(item)
        loaded = service_is_loaded(item.label, runner=runner)
        try:
            running = service_is_running(item.label, runner=runner) if loaded else False
        except OSError:
            running = False
        controller_pid = _service_pid(item.label, runner=runner) if loaded else None
        heartbeat = _load_last_pass(item.state_dir / "heartbeat.json") or {}
        heartbeat_compatible, heartbeat_age = _controller_heartbeat_matches(
            heartbeat,
            controller=name,
            pid=controller_pid,
            source_fingerprint=expected_fingerprint,
        )
        controllers[name] = {
            "label": item.label,
            "installed": item.plist.is_file(),
            "loaded": loaded,
            "running": running,
            "pid": controller_pid,
            "plist": str(item.plist),
            "stdout": str(item.bootstrap_stdout),
            "stderr": str(item.bootstrap_stderr),
            "installed_protocol_version": installed_protocol_version,
            "expected_protocol_version": CONTROL_PLANE_PROTOCOL_VERSION,
            "protocol_compatible": (installed_protocol_version == CONTROL_PLANE_PROTOCOL_VERSION),
            "heartbeat": heartbeat or None,
            "heartbeat_age_seconds": heartbeat_age,
            "heartbeat_required": heartbeat_required,
            "heartbeat_compatible": heartbeat_compatible,
        }
    installed = all(row["installed"] for row in controllers.values())
    loaded = all(row["loaded"] for row in controllers.values())
    running = all(row["running"] for row in controllers.values())
    protocol_compatible = all(row["protocol_compatible"] for row in controllers.values())
    heartbeat_compatible = all(
        row["heartbeat_compatible"] for row in controllers.values()
    )
    return {
        "label": paths.label,
        "installed": installed,
        "loaded": loaded,
        "running": running,
        "healthy": (
            installed
            and loaded
            and running
            and protocol_compatible
            and (heartbeat_compatible or not heartbeat_required)
        ),
        "protocol_compatible": protocol_compatible,
        "heartbeat_compatible": heartbeat_compatible,
        "heartbeat_required": heartbeat_required,
        "expected_protocol_version": CONTROL_PLANE_PROTOCOL_VERSION,
        "poll_seconds": CONTROLLER_POLL_SECONDS,
        "repo_root": str(paths.repo_root),
        "python": str(paths.python),
        "controllers": controllers,
    }


def require_compatible_controller_services(
    paths: ServicePaths | None = None,
    *,
    runner: CommandRunner = _run_command,
    require_source_current: bool = True,
) -> dict[str, Any]:
    """Fail closed before queue mutation when the persistent control plane is stale."""

    selected = paths or default_service_paths()
    try:
        status = controller_services_status(selected, runner=runner)
    except OSError as exc:
        raise RuntimeError(
            "fleet controllers are unavailable; run `rlab fleet service install --replace`"
        ) from exc
    def row_compatible(name: str, row: Mapping[str, Any]) -> bool:
        heartbeat_ok = bool(row.get("heartbeat_compatible"))
        if row.get("heartbeat_required") and not require_source_current:
            heartbeat_ok, _age = _controller_heartbeat_runtime_matches(
                row.get("heartbeat") or {},
                controller=name,
                pid=row.get("pid"),
            )
        return bool(
            row.get("installed")
            and row.get("loaded")
            and row.get("running")
            and row.get("protocol_compatible")
            and (heartbeat_ok or not row.get("heartbeat_required"))
        )

    compatible = all(
        row_compatible(name, row)
        for name, row in status["controllers"].items()
    )
    if not compatible:
        incompatible = [
            name
            for name, row in status["controllers"].items()
            if not row_compatible(name, row)
        ]
        raise RuntimeError(
            "fleet controllers are not installed, running, protocol-compatible, and healthy "
            f"({', '.join(incompatible) or 'unknown'}); run "
            "`rlab fleet service install --replace`"
        )
    return status


@dataclass(frozen=True)
class _PublisherProcess:
    pid: int
    ppid: int
    pgid: int
    command: str


def _service_pid(label: str, *, runner: CommandRunner = _run_command) -> int | None:
    result = runner(
        ["launchctl", "print", _target(label)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        return None
    matched = re.search(r"(?m)^\s*pid\s*=\s*(\d+)\s*$", result.stdout or "")
    return int(matched.group(1)) if matched else None


def _process_cwd(pid: int, *, runner: CommandRunner = _run_command) -> Path | None:
    executable = shutil.which("lsof") or "/usr/sbin/lsof"
    result = runner(
        [executable, "-a", "-p", str(int(pid)), "-d", "cwd", "-Fn"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        return None
    for line in (result.stdout or "").splitlines():
        if line.startswith("n") and len(line) > 1:
            return Path(line[1:]).expanduser().resolve()
    return None


def _publisher_processes(
    repo_root: Path,
    *,
    parent_pid: int | None = None,
    expected_process_groups: Sequence[int] = (),
    runner: CommandRunner = _run_command,
) -> list[_PublisherProcess]:
    result = runner(
        ["ps", "-axo", "pid=,ppid=,pgid=,command="],
        check=True,
        capture_output=True,
        text=True,
    )
    found: list[_PublisherProcess] = []
    for raw in (result.stdout or "").splitlines():
        parts = raw.strip().split(maxsplit=3)
        if len(parts) != 4:
            continue
        try:
            pid, ppid, pgid = (int(parts[index]) for index in range(3))
            argv = shlex.split(parts[3])
        except (ValueError, IndexError):
            continue
        if parent_pid is not None and ppid != int(parent_pid):
            continue
        try:
            module_index = argv.index("-m")
        except ValueError:
            continue
        if argv[module_index + 1 : module_index + 2] != ["rlab.fleet_wandb_publisher"]:
            continue
        module_args = argv[module_index + 1 :]
        if len(module_args) != 3 or module_args[1] != "--train-job-id":
            raise RuntimeError(
                f"refusing to stop W&B publisher pid {pid}: unverified module arguments"
            )
        try:
            if int(module_args[2]) <= 0:
                raise ValueError
        except ValueError as exc:
            raise RuntimeError(
                f"refusing to stop W&B publisher pid {pid}: invalid train job id"
            ) from exc
        cwd = _process_cwd(pid, runner=runner)
        if cwd is None:
            still_running = runner(
                ["ps", "-p", str(pid), "-o", "pid="],
                check=False,
                capture_output=True,
                text=True,
            )
            if still_running.returncode or not (still_running.stdout or "").strip():
                # The actor exited after the initial process snapshot. It no
                # longer needs cleanup and must not turn a safe reload into a
                # false verification failure.
                continue
        if cwd != repo_root.resolve():
            raise RuntimeError(
                f"refusing to stop unverified W&B publisher pid {pid}: cwd={cwd}"
            )
        expected_groups = {int(value) for value in expected_process_groups}
        actor_owned_group = pgid == pid
        manager_owned_group = (
            parent_pid is not None and ppid == int(parent_pid) and pgid == int(parent_pid)
        )
        previously_verified_group = pgid in expected_groups
        if not (actor_owned_group or manager_owned_group or previously_verified_group):
            raise RuntimeError(
                f"refusing to stop W&B publisher pid {pid}: unverified process group {pgid}"
            )
        found.append(_PublisherProcess(pid=pid, ppid=ppid, pgid=pgid, command=parts[3]))
    return found


def _stop_legacy_wandb_publishers(
    repo_root: Path,
    captured: Sequence[_PublisherProcess],
    *,
    expected_process_groups: Sequence[int] = (),
    runner: CommandRunner = _run_command,
    timeout_seconds: float = 5.0,
) -> None:
    processes = {item.pid: item for item in captured}
    verified_groups = {
        *(int(item.pgid) for item in captured),
        *(int(value) for value in expected_process_groups),
    }
    for item in _publisher_processes(
        repo_root,
        expected_process_groups=tuple(sorted(verified_groups)),
        runner=runner,
    ):
        processes[item.pid] = item
    for pgid in {item.pgid for item in processes.values()}:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + max(0.0, float(timeout_seconds))
    remaining = set(processes)
    while remaining and time.monotonic() < deadline:
        for pid in list(remaining):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                remaining.discard(pid)
        if remaining:
            time.sleep(0.05)
    for pgid in {processes[pid].pgid for pid in remaining}:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    lingering = _publisher_processes(
        repo_root,
        expected_process_groups=tuple(sorted(verified_groups)),
        runner=runner,
    )
    if lingering:
        raise RuntimeError(
            "W&B publisher cleanup did not complete: "
            + ", ".join(str(item.pid) for item in lingering)
        )


def _wandb_readiness_matches(
    item: ServicePaths,
    *,
    source_fingerprint: str,
    started_after: float,
) -> bool:
    payload = _load_last_pass(item.readiness)
    if not payload:
        return False
    try:
        return (
            int(payload.get("pid") or 0) > 0
            and float(payload.get("started_at") or 0.0) >= float(started_after)
            and float(payload.get("last_reconciled_at") or 0.0) >= float(started_after)
            and str(payload.get("source_fingerprint") or "") == source_fingerprint
            and int(payload.get("protocol_version") or 0) == CONTROL_PLANE_PROTOCOL_VERSION
            and str(payload.get("phase") or "") in {"idle", "reconciling"}
        )
    except TypeError, ValueError:
        return False


def reload_controller_service(
    paths: ServicePaths,
    controller: str,
    *,
    count_nonterminal_jobs: CountNonterminalJobs | None = None,
    runner: CommandRunner = _run_command,
) -> bool:
    """Reload one persistent controller, allowing the durable W&B lane to stay active."""

    if controller not in CONTROLLER_NAMES:
        raise ValueError(f"unknown fleet controller: {controller}")
    if controller != "wandb":
        counter = count_nonterminal_jobs or (
            _default_count_machine_reload_blockers
            if controller == "machine"
            else _default_count_nonterminal_jobs
        )
        nonterminal = int(counter(paths.repo_root))
        if nonterminal:
            description = (
                "active execution blocker"
                if controller == "machine"
                else "nonterminal job"
            )
            raise RuntimeError(
                f"refusing to reload the {controller} controller with "
                f"{nonterminal} {description}(s)"
            )
    item = controller_service_paths(paths, controller)
    if not item.plist.is_file():
        raise FileNotFoundError(f"fleet controller is not installed: {item.plist}")
    old_plist = item.plist.read_bytes()
    replacement = render_controller_launch_agent_plist(paths, controller)
    candidate = item.plist.with_name(f".{item.plist.name}.candidate")
    _atomic_write(candidate, replacement)
    try:
        _validate_candidate_plist(candidate, runner=runner)
    except Exception:
        candidate.unlink(missing_ok=True)
        raise
    was_loaded = service_is_loaded(item.label, runner=runner)
    captured: list[_PublisherProcess] = []
    manager_pid: int | None = None
    if controller == "wandb" and was_loaded:
        manager_pid = _service_pid(item.label, runner=runner)
        if manager_pid is None:
            raise RuntimeError("cannot verify the active W&B manager pid before reload")
        captured = _publisher_processes(
            paths.repo_root,
            parent_pid=manager_pid,
            runner=runner,
        )
        item.readiness.unlink(missing_ok=True)
    replaced = False
    try:
        if was_loaded:
            runner(
                ["launchctl", "bootout", _target(item.label)],
                check=True,
                capture_output=True,
                text=True,
            )
        if controller == "wandb":
            _stop_legacy_wandb_publishers(
                paths.repo_root,
                captured,
                expected_process_groups=(manager_pid,) if manager_pid is not None else (),
                runner=runner,
            )
        os.replace(candidate, item.plist)
        replaced = True
        started_after = time.time()
        _bootstrap_launch_agent(item, runner=runner, retry_busy=True)
        expected_fingerprint = controller_source_fingerprint(paths.repo_root)
        deadline = time.monotonic() + CONTROLLER_READINESS_TIMEOUT_SECONDS
        while not service_is_running(item.label, runner=runner):
            if time.monotonic() >= deadline:
                raise RuntimeError(f"reloaded {controller} controller did not start")
            time.sleep(0.1)
        if controller == "wandb":
            while not _wandb_readiness_matches(
                item,
                source_fingerprint=expected_fingerprint,
                started_after=started_after,
            ):
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        "reloaded W&B controller did not publish matching readiness evidence"
                    )
                time.sleep(0.1)
    except Exception as reload_error:
        rollback_errors: list[str] = []
        if service_is_loaded(item.label, runner=runner):
            result = runner(
                ["launchctl", "bootout", _target(item.label)],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode:
                rollback_errors.append(
                    "unload: "
                    + str(redact(result.stderr or result.stdout or result.returncode))
                )
            else:
                try:
                    _wait_for_service_unloaded(item.label, runner=runner)
                except Exception as exc:
                    rollback_errors.append(f"unload wait: {exc}")
        if replaced:
            _atomic_write(item.plist, old_plist)
        if was_loaded:
            try:
                _bootstrap_launch_agent(item, runner=runner, retry_busy=True)
            except Exception as exc:
                rollback_errors.append(f"restore: {exc}")
        if rollback_errors:
            raise RuntimeError(
                f"{controller} controller reload failed ({reload_error}); "
                "rollback also failed: " + "; ".join(rollback_errors)
            ) from reload_error
        raise
    finally:
        candidate.unlink(missing_ok=True)
    return True


def _default_service_notifier(title: str, message: str) -> None:
    script = f"display notification {json.dumps(message)} with title {json.dumps(title)}"
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
            last_notified = datetime.fromisoformat(str(last_notified_at).replace("Z", "+00:00"))
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
    controllers = controller_services_status(paths, runner=runner)
    if controllers["installed"]:
        evaluation = dict(controllers["controllers"]["evaluation"])
        ready = bool(
            evaluation.get("loaded")
            and evaluation.get("running")
            and (
                not evaluation.get("heartbeat_required")
                or (
                    evaluation.get("protocol_compatible")
                    and evaluation.get("heartbeat_compatible")
                )
            )
        )
        return {
            "ready": ready,
            "loaded": bool(evaluation.get("loaded")),
            "last_pass_stale": False,
            "last_pass_age_seconds": None,
            "eval_status": "ok" if ready else "error",
            "eval_detail_status": "ok" if ready else "unavailable",
            "error": None if ready else "evaluation controller is not running",
            "app_cleanup": None,
        }
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


def controller_services_doctor(
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
    expected_fingerprint = controller_source_fingerprint(paths.repo_root)
    for name in CONTROLLER_NAMES:
        item = controller_service_paths(paths, name)
        payload_ok = False
        detail = str(item.plist)
        if item.plist.is_file():
            try:
                payload = plistlib.loads(item.plist.read_bytes())
                validate_controller_launch_agent_payload(payload, paths, name)
                payload_ok = True
            except Exception as exc:
                detail = str(redact(str(exc)))
        add(f"{name}_plist", payload_ok, detail)
        loaded = service_is_loaded(item.label, runner=runner)
        add(f"{name}_loaded", loaded, item.label)
        running = service_is_running(item.label, runner=runner) if loaded else False
        add(f"{name}_running", running, item.label)
        installed_protocol_version = _installed_controller_protocol_version(item)
        add(
            f"{name}_protocol",
            installed_protocol_version == CONTROL_PLANE_PROTOCOL_VERSION,
            (f"installed={installed_protocol_version} expected={CONTROL_PLANE_PROTOCOL_VERSION}"),
        )
        if CONTROL_PLANE_PROTOCOL_VERSION >= CONTROLLER_HEARTBEAT_PROTOCOL_VERSION:
            heartbeat = _load_last_pass(item.state_dir / "heartbeat.json") or {}
            controller_pid = _service_pid(item.label, runner=runner) if loaded else None
            compatible, age = _controller_heartbeat_matches(
                heartbeat,
                controller=name,
                pid=controller_pid,
                source_fingerprint=expected_fingerprint,
            )
            add(
                f"{name}_heartbeat",
                compatible,
                f"age={age} path={item.state_dir / 'heartbeat.json'}",
            )
    return {"ok": all(check["ok"] for check in checks), "checks": checks}


def _load_repo_environment(repo_root: Path) -> None:
    from rlab.dotenv import load_env_file

    load_env_file(repo_root / ".env")


def _connect_queue(repo_root: Path):
    from rlab.job_queue import connect, database_url

    _load_repo_environment(repo_root)
    # Fleet holds session-scoped schema and W&B publication advisory locks.
    # Those locks are unsafe through PgBouncer because a killed process can
    # return its locked backend session to the pool.
    return connect(database_url(use_direct=True))


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
    candidates = [paths, *(controller_service_paths(paths, name) for name in CONTROLLER_NAMES)]
    loaded = [item for item in candidates if service_is_loaded(item.label, runner=runner)]
    count = _default_count_nonterminal_jobs(paths.repo_root)
    if count:
        raise RuntimeError(
            f"refusing schema reset with {count} nonterminal job(s); wait for quiescence"
        )
    if not loaded:
        yield
        return
    stopped: list[ServicePaths] = []
    captured_publishers: list[_PublisherProcess] = []
    publisher_manager_groups: tuple[int, ...] = ()
    try:
        wandb_item = controller_service_paths(paths, "wandb")
        if any(item.label == wandb_item.label for item in loaded):
            manager_pid = _service_pid(wandb_item.label, runner=runner)
            if manager_pid is None:
                raise RuntimeError("cannot verify the active W&B manager pid before schema reset")
            captured_publishers = _publisher_processes(
                paths.repo_root,
                parent_pid=manager_pid,
                runner=runner,
            )
            publisher_manager_groups = (manager_pid,)
        for item in loaded:
            runner(
                ["launchctl", "bootout", _target(item.label)],
                check=True,
                capture_output=True,
                text=True,
            )
            stopped.append(item)
        deadline = time.monotonic() + timeout_seconds
        while any(service_is_running(item.label, runner=runner) for item in stopped):
            if time.monotonic() >= deadline:
                raise TimeoutError("timed out stopping fleet controllers for schema reset")
            time.sleep(0.25)
        _stop_legacy_wandb_publishers(
            paths.repo_root,
            captured_publishers,
            expected_process_groups=publisher_manager_groups,
            runner=runner,
        )
        if _publisher_processes(
            paths.repo_root,
            expected_process_groups=publisher_manager_groups,
            runner=runner,
        ):
            raise RuntimeError("publisher processes remain after controller shutdown")
        yield
    finally:
        for item in stopped:
            if not item.plist.is_file():
                continue
            runner(
                ["launchctl", "bootstrap", _domain(), str(item.plist)],
                check=True,
                capture_output=True,
                text=True,
            )


def _default_count_nonterminal_jobs(repo_root: Path) -> int:
    from rlab.job_queue import count_nonterminal_jobs

    conn = _connect_queue(repo_root)
    try:
        return int(count_nonterminal_jobs(conn))
    finally:
        conn.close()


def _default_count_machine_reload_blockers(repo_root: Path) -> int:
    from rlab.job_queue import count_machine_reload_blockers

    conn = _connect_queue(repo_root)
    try:
        return int(count_machine_reload_blockers(conn))
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


def uninstall_controller_services(
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
                f"refusing to uninstall fleet controllers with {count} nonterminal job(s); "
                "use --force"
            )
    changed = False
    for name in CONTROLLER_NAMES:
        item = controller_service_paths(paths, name)
        loaded = service_is_loaded(item.label, runner=runner)
        if loaded:
            runner(
                ["launchctl", "bootout", _target(item.label)],
                check=True,
                capture_output=True,
                text=True,
            )
        existed = item.plist.exists()
        item.plist.unlink(missing_ok=True)
        changed = changed or loaded or existed
    legacy_loaded = service_is_loaded(paths.label, runner=runner)
    if legacy_loaded:
        runner(
            ["launchctl", "bootout", _target(paths.label)],
            check=True,
            capture_output=True,
            text=True,
        )
    legacy_existed = paths.plist.exists()
    paths.plist.unlink(missing_ok=True)
    return changed or legacy_loaded or legacy_existed


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
    *,
    progress: Callable[[str, str], None] | None = None,
) -> Mapping[str, Any] | None:
    from rlab.fleet import run_service_machine_pass

    kwargs: dict[str, Any] = {
        "machine_name": machine_name,
        "machines_path": repo_root / "experiments" / "machines.yaml",
        "repo_root": repo_root,
        "deadline_monotonic": deadline_monotonic,
    }
    if progress is not None:
        kwargs["progress"] = progress
    return run_service_machine_pass(**kwargs)


def _default_reconcile_eval(
    repo_root: Path,
    deadline_monotonic: float,
    *,
    progress: Callable[[str, str], None] | None = None,
) -> Mapping[str, Any] | None:
    config_path = repo_root / "experiments" / "modal_eval.yaml"
    if not config_path.is_file() and not (repo_root / ".env").is_file():
        return {"status": "unconfigured"}
    if config_path.is_file():
        from rlab.modal_eval_orchestrator import run_service_eval_pass

        kwargs: dict[str, Any] = {
            "repo_root": repo_root,
            "deadline_monotonic": deadline_monotonic,
        }
        if progress is not None:
            kwargs["progress"] = progress
        detail = dict(run_service_eval_pass(**kwargs) or {})
    else:
        detail = {"status": "unconfigured"}
    from rlab.fleet_wandb_publisher import drain_cycle_parallel
    from rlab.telemetry_v2_controller import (
        archive_once as archive_v2_once,
        finalize_roots_once,
        project_once as project_v2_once,
    )
    from rlab.telemetry_mailbox import (
        consume_attempt_events,
        discard_disabled_metric_batches,
        mailbox_storage_bytes,
    )

    if progress:
        progress("PUBLISHING TELEMETRY", "Draining mailbox and W&B publication state")
    conn = _connect_queue(repo_root)
    try:
        detail["consumed_attempt_events"] = consume_attempt_events(conn)
        detail["discarded_disabled_metric_streams"] = discard_disabled_metric_batches(conn)
        detail["wandb_publication"] = drain_cycle_parallel(
            conn,
            repo_root=repo_root,
            max_runs=100,
            deadline_monotonic=deadline_monotonic,
        )
        if (
            os.environ.get("TELEMETRY_ARCHIVE_PRIMARY_URI")
            and os.environ.get("TELEMETRY_ARCHIVE_BACKUP_URI")
        ):
            detail["telemetry_v2_archived_segments"] = archive_v2_once(conn, limit=100)
            detail["telemetry_v2_finalized_roots"] = finalize_roots_once(conn, limit=100)
        else:
            detail["telemetry_v2_archive_status"] = "unconfigured_fail_closed"
        if (
            os.environ.get("RLAB_WANDB_SERVICE_IDENTITY")
            and os.environ.get("RLAB_WANDB_SERVICE_CREDENTIAL_GENERATION")
        ):
            detail["telemetry_v2_projected_rows"] = project_v2_once(conn, limit=100)
        else:
            detail["telemetry_v2_projection_status"] = "unconfigured_fail_closed"
        storage_bytes = mailbox_storage_bytes(conn)
        detail["metric_mailbox_bytes"] = storage_bytes
        detail["metric_mailbox_pressure"] = (
            "hard" if storage_bytes >= 5 * 1024**3 else "soft" if storage_bytes >= 1024**3 else "ok"
        )
    finally:
        conn.close()
    return detail


def _watch_item(
    entity: str,
    title: str,
    detail: str,
    *,
    reason_code: str,
    resolution: str,
    blast_radius: str,
    command: str = "",
    timestamp: object = None,
) -> dict[str, Any]:
    return {
        "id": f"{reason_code}:{entity}",
        "entity": entity,
        "title": title,
        "detail": str(redact(detail)),
        "reason_code": reason_code,
        "resolution": resolution,
        "blast_radius": blast_radius,
        "command": command,
        "timestamp": str(timestamp or ""),
    }


def _default_workload_snapshot(repo_root: Path) -> dict[str, Any]:
    """Collect the queue state once inside the service; watchers never access PostgreSQL."""

    from rlab.machines import load_machine_registry

    captured_at = _iso_utc()
    needs_action: list[dict[str, Any]] = []
    retrying: list[dict[str, Any]] = []
    waiting: list[dict[str, Any]] = []
    active: list[dict[str, Any]] = []
    counts: dict[str, dict[str, int]] = {"train": {}, "eval_jobs": {}, "eval_runs": {}}
    in_progress = 0
    try:
        registry = load_machine_registry(repo_root / "experiments" / "machines.yaml")
        hard_capacity = {
            name: machine.limits.max_parallel_containers
            for name, machine in registry.machines.items()
        }
        conn = _connect_queue(repo_root)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT t.id, t.machine, t.status, t.cancel_requested, t.error,
                      t.live_publication_status, t.live_publication_attempts,
                      t.live_publication_next_retry_at, t.live_publication_error,
                      t.created_at, t.started_at,
                      r.status AS eval_status, r.error AS eval_error,
                      r.artifact_projection_attempts,
                      r.artifact_projection_next_retry_at,
                      control.drained AS machine_drained,
                      control.effective_capacity, control.reason AS control_reason,
                      launch.state AS launch_state, launch.error AS launch_error,
                      launch.next_retry_at,
                      (SELECT COUNT(*) FROM job_launches active
                       WHERE active.machine = t.machine
                         AND active.state IN ('launching', 'running')) AS active_reservations
                    FROM train_jobs t
                    LEFT JOIN eval_runs r ON r.train_job_id = t.id
                    LEFT JOIN machine_controls control ON control.machine = t.machine
                    LEFT JOIN LATERAL (
                      SELECT state, error, next_retry_at FROM job_launches
                      WHERE job_id = t.id ORDER BY id DESC LIMIT 1
                    ) launch ON TRUE
                    WHERE t.status IN ('pending', 'launching', 'starting', 'running', 'finalizing')
                       OR r.status IN ('active', 'awaiting_artifact_recovery', 'finalizing')
                    ORDER BY t.id
                    LIMIT 500
                    """
                )
                trains = [dict(row) for row in cur.fetchall()]
                cur.execute(
                    """
                    SELECT j.id, j.train_job_id, j.checkpoint_step, j.status,
                      j.purpose, j.stage_name,
                      j.projection_attempts, j.projection_next_retry_at,
                      j.projection_error, j.error, j.created_at,
                      t.status AS train_status, r.status AS eval_run_status,
                      backend.drained AS backend_drained,
                      (SELECT COUNT(*) FROM eval_attempts a
                       WHERE a.eval_job_id = j.id
                         AND a.status IN ('dispatching', 'submitted')) AS active_attempts
                    FROM eval_jobs j
                    JOIN train_jobs t ON t.id = j.train_job_id
                    JOIN eval_runs r ON r.train_job_id = j.train_job_id
                    LEFT JOIN eval_backend_state backend ON backend.backend = 'modal'
                    WHERE j.status IN ('pending', 'dispatching', 'submitted', 'blocked_budget')
                       OR (j.status = 'succeeded' AND j.projected_at IS NULL
                           AND j.projection_error IS NOT NULL)
                    ORDER BY j.id
                    LIMIT 500
                    """
                )
                eval_jobs = [dict(row) for row in cur.fetchall()]
                cur.execute("SELECT status, count(*) AS count FROM train_jobs GROUP BY status")
                counts["train"] = {str(row["status"]): int(row["count"]) for row in cur.fetchall()}
                cur.execute("SELECT status, count(*) AS count FROM eval_jobs GROUP BY status")
                counts["eval_jobs"] = {
                    str(row["status"]): int(row["count"]) for row in cur.fetchall()
                }
                cur.execute("SELECT status, count(*) AS count FROM eval_runs GROUP BY status")
                counts["eval_runs"] = {
                    str(row["status"]): int(row["count"]) for row in cur.fetchall()
                }
        finally:
            conn.close()

        terminal_train = {"succeeded", "failed", "finalization_failed", "canceled"}
        for row in trains:
            job_id = int(row["id"])
            entity = f"train/{job_id}"
            status = str(row.get("status") or "unknown")
            eval_status = str(row.get("eval_status") or "")
            if status in {"launching", "starting", "running", "finalizing"}:
                in_progress += 1
                title = {
                    "launching": "Launching training",
                    "starting": "Starting training",
                    "running": "Training",
                    "finalizing": "Finalizing training",
                }[status]
                active.append(
                    _watch_item(
                        entity,
                        title,
                        f"{row['machine']} · evaluation {eval_status or 'not started'}",
                        reason_code=f"train_{status}",
                        resolution="automatic",
                        blast_radius=entity,
                        timestamp=row.get("started_at") or row.get("created_at"),
                    )
                )
            if eval_status == "awaiting_artifact_recovery" and status in {
                "failed",
                "finalization_failed",
                "canceled",
            }:
                needs_action.append(
                    _watch_item(
                        entity,
                        "Terminal evaluation cleanup required",
                        f"Training is {status}; incomplete evaluation cannot establish acceptance",
                        reason_code="orphaned_eval",
                        resolution="manual action",
                        blast_radius="evaluation and promotion for this train job",
                        command=render_command(eval_modal_abandon_command(job_id)),
                        timestamp=row.get("started_at"),
                    )
                )
            elif eval_status == "awaiting_artifact_recovery":
                needs_action.append(
                    _watch_item(
                        entity,
                        "Artifact recovery required",
                        str(row.get("eval_error") or "Evaluation cannot continue"),
                        reason_code="artifact_recovery",
                        resolution="manual action",
                        blast_radius="checkpoint promotion and publication",
                        command=render_command(eval_modal_recover_command(job_id)),
                        timestamp=row.get("started_at"),
                    )
                )
            if status in terminal_train and eval_status in {"active", "finalizing"}:
                needs_action.append(
                    _watch_item(
                        entity,
                        "Inconsistent terminal state",
                        f"Training is {status} while evaluation remains {eval_status}",
                        reason_code="orphaned_eval",
                        resolution="manual action",
                        blast_radius="evaluation and promotion for this train job",
                        command=render_command(eval_modal_abandon_command(job_id)),
                    )
                )
            if str(row.get("live_publication_status") or "") == "failed":
                needs_action.append(
                    _watch_item(
                        entity,
                        "Publication retries exhausted",
                        str(row.get("live_publication_error") or "Live publication failed"),
                        reason_code="publication_failed",
                        resolution="manual action",
                        blast_radius="W&B publication for this train job",
                        command=render_command(experiment_retry_finalization_command(job_id)),
                    )
                )
            elif row.get("live_publication_next_retry_at"):
                retrying.append(
                    _watch_item(
                        entity,
                        "Retrying publication",
                        "The service will retry the durable publication outbox",
                        reason_code="publication_retry",
                        resolution="automatic",
                        blast_radius="publication only",
                        timestamp=row.get("live_publication_next_retry_at"),
                    )
                )
            launch_error = str(row.get("launch_error") or "")
            if row.get("next_retry_at"):
                retrying.append(
                    _watch_item(
                        entity,
                        "Retrying machine launch",
                        launch_error or "The launch has a scheduled retry",
                        reason_code="launch_retry",
                        resolution="automatic",
                        blast_radius=entity,
                        timestamp=row.get("next_retry_at"),
                    )
                )
            if status != "pending":
                continue
            if bool(row.get("machine_drained")):
                waiting.append(
                    _watch_item(
                        entity,
                        f"Waiting for {row['machine']} resume",
                        str(row.get("control_reason") or "Machine is operator-drained"),
                        reason_code="machine_drained",
                        resolution="intentionally paused",
                        blast_radius=f"pending jobs for {row['machine']}",
                        command=f"rlab fleet resume --machine {row['machine']}",
                    )
                )
                continue
            configured = hard_capacity.get(str(row["machine"]))
            override = row.get("effective_capacity")
            capacity = (
                min(configured, int(override))
                if configured is not None and override is not None
                else int(override)
                if override is not None
                else configured
            )
            at_capacity = (
                capacity is not None and int(row.get("active_reservations") or 0) >= capacity
            )
            waiting.append(
                _watch_item(
                    entity,
                    f"Waiting for {row['machine']} capacity"
                    if at_capacity
                    else "Queued for next pass",
                    (
                        f"{row.get('active_reservations') or 0}/{capacity} train slots reserved"
                        if at_capacity
                        else "The scheduler will claim this job automatically"
                    ),
                    reason_code="machine_capacity" if at_capacity else "queued",
                    resolution="automatic",
                    blast_radius=entity,
                )
            )

        for row in eval_jobs:
            eval_job_id = int(row["id"])
            train_job_id = int(row["train_job_id"])
            entity = f"eval/{eval_job_id}"
            status = str(row.get("status") or "unknown")
            purpose = str(row.get("purpose") or "evaluation")
            if status in {"dispatching", "submitted"}:
                in_progress += 1
                active.append(
                    _watch_item(
                        entity,
                        f"{purpose.title()} evaluation",
                        f"train/{train_job_id} checkpoint {int(row['checkpoint_step']):,} · Modal {status}",
                        reason_code=f"eval_{status}",
                        resolution="automatic",
                        blast_radius=entity,
                        timestamp=row.get("created_at"),
                    )
                )
            train_status = str(row.get("train_status") or "")
            if (
                purpose == "promotion"
                and status == "pending"
                and train_status
                in {
                    "failed",
                    "finalization_failed",
                    "canceled",
                }
            ):
                needs_action.append(
                    _watch_item(
                        entity,
                        "Promotion cannot become dispatchable",
                        f"train/{train_job_id} is {train_status}; promotion requires finalizing",
                        reason_code="ineligible_promotion",
                        resolution="manual action",
                        blast_radius=f"promotion for train/{train_job_id}",
                        command=render_command(eval_modal_abandon_command(train_job_id)),
                    )
                )
                continue
            if purpose == "promotion" and status == "pending" and train_status != "finalizing":
                waiting.append(
                    _watch_item(
                        entity,
                        "Waiting for training finalization",
                        f"train/{train_job_id} is {train_status or 'unknown'}",
                        reason_code="promotion_wait",
                        resolution="automatic",
                        blast_radius=f"promotion for train/{train_job_id}",
                    )
                )
                continue
            if status == "blocked_budget":
                needs_action.append(
                    _watch_item(
                        entity,
                        "Evaluation budget blocked",
                        f"{purpose} evaluation cannot reserve its configured budget",
                        reason_code="budget_blocked",
                        resolution="manual action",
                        blast_radius=f"evaluation for train/{train_job_id}",
                        command="rlab eval modal status",
                    )
                )
                continue
            if row.get("projection_next_retry_at"):
                retrying.append(
                    _watch_item(
                        entity,
                        "Retrying evaluation projection",
                        str(row.get("projection_error") or "Projection retry is scheduled"),
                        reason_code="projection_retry",
                        resolution="automatic",
                        blast_radius=f"evaluation evidence for train/{train_job_id}",
                        timestamp=row.get("projection_next_retry_at"),
                    )
                )
            if status == "pending":
                waiting.append(
                    _watch_item(
                        entity,
                        "Waiting for Modal resume"
                        if row.get("backend_drained")
                        else "Waiting for evaluation capacity",
                        f"{purpose} evaluation for train/{train_job_id}",
                        reason_code="eval_drained"
                        if row.get("backend_drained")
                        else "eval_capacity",
                        resolution="intentionally paused"
                        if row.get("backend_drained")
                        else "automatic",
                        blast_radius=entity,
                        command="rlab eval modal resume" if row.get("backend_drained") else "",
                    )
                )
    except Exception as exc:
        needs_action.append(
            _watch_item(
                "scheduler",
                "Work snapshot unavailable",
                f"{type(exc).__name__}: {exc}",
                reason_code="snapshot_error",
                resolution="manual inspection",
                blast_radius="watcher queue visibility only",
                command="rlab fleet service logs --tail 100",
            )
        )

    def deduplicate(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return list({str(item["id"]): item for item in items}.values())

    return redact(
        {
            "captured_at": captured_at,
            "in_progress": in_progress,
            "counts": counts,
            "active": deduplicate(active),
            "needs_action": deduplicate(needs_action),
            "retrying": deduplicate(retrying),
            "waiting": deduplicate(waiting),
        }
    )


def _meaningful_work_changes(
    previous: Mapping[str, Any] | None,
    current: Mapping[str, Any],
) -> list[dict[str, str]]:
    before: dict[str, tuple[str, str]] = {}
    after: dict[str, tuple[str, str]] = {}
    for bucket in ("needs_action", "retrying", "waiting"):
        for item in (previous or {}).get(bucket) or []:
            if isinstance(item, Mapping):
                before[str(item.get("id"))] = (bucket, str(item.get("title") or ""))
        for item in current.get(bucket) or []:
            if isinstance(item, Mapping):
                after[str(item.get("id"))] = (bucket, str(item.get("title") or ""))
    changes: list[dict[str, str]] = []
    for item_id, (bucket, title) in after.items():
        if before.get(item_id) == (bucket, title):
            continue
        item = next(
            (
                row
                for row in current.get(bucket) or []
                if isinstance(row, Mapping) and str(row.get("id")) == item_id
            ),
            {},
        )
        changes.append(
            {
                "entity": str(item.get("entity") or "scheduler"),
                "title": title,
                "detail": str(item.get("detail") or ""),
                "timestamp": str(current.get("captured_at") or _iso_utc()),
            }
        )
    for item_id, (_bucket, title) in before.items():
        if item_id not in after:
            changes.append(
                {
                    "entity": item_id.rsplit(":", 1)[-1],
                    "title": f"Resolved: {title}",
                    "detail": "The condition is no longer present",
                    "timestamp": str(current.get("captured_at") or _iso_utc()),
                }
            )
    return changes[:20]


def run_service_pass(
    *,
    repo_root: Path,
    state_dir: Path,
    discover_machines: DiscoverMachines = _default_discover_machines,
    reconcile_machine: ReconcileMachine = _default_reconcile_machine,
    reconcile_eval: ReconcileEval = _default_reconcile_eval,
    workload_snapshot: WorkloadSnapshot = _default_workload_snapshot,
    max_machine_lanes: int = DEFAULT_MAX_MACHINE_LANES,
    lane_timeout_seconds: float = DEFAULT_LANE_TIMEOUT_SECONDS,
    pass_timeout_seconds: float = DEFAULT_PASS_TIMEOUT_SECONDS,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    started_at = _utc_now()
    started_monotonic = clock()
    pass_deadline = started_monotonic + pass_timeout_seconds
    deadline_at = _iso_utc(started_at + timedelta(seconds=pass_timeout_seconds))
    pass_id = uuid.uuid4().hex[:12]
    previous_pass = _load_last_pass(state_dir / "last-pass.json") or {}
    wake_requests = consume_wake_requests(state_dir)
    triggers = wake_requests or [
        {
            "requested_at": _iso_utc(started_at),
            "reason": "interval",
            "entity_kind": "service",
            "entity_id": "",
        }
    ]
    source = _source_state(repo_root)
    event_log = ServiceEventLog(state_dir / "service.jsonl")
    summary: dict[str, Any] = {
        "schema_version": 2,
        "pass_id": pass_id,
        "started_at": _iso_utc(started_at),
        "repo_root": str(repo_root),
        "deadline_at": deadline_at,
        "triggers": triggers,
        "source": source,
        "status": "running",
        "machines": [],
        "eval": {"status": "running"},
        "workload": {},
        "meaningful_changes": [],
    }
    progress = PassProgress(
        state_dir / "current-pass.json",
        {
            "schema_version": 1,
            "pass_id": pass_id,
            "pid": os.getpid(),
            "started_at": summary["started_at"],
            "deadline_at": deadline_at,
            "repo_root": str(repo_root),
            "triggers": triggers,
            "source": source,
            "workload": {},
            "lanes": {},
        },
    )
    event_log.emit(
        "pass_started",
        pass_id=pass_id,
        repo_root=str(repo_root),
        triggers=triggers,
        source=source,
    )
    try:
        starting_workload = dict(workload_snapshot(repo_root) or {})
        summary["workload"] = starting_workload
        progress.set_workload(starting_workload)
        machine_names = sorted(set(str(name) for name in discover_machines(repo_root)))
        machine_workers = min(int(max_machine_lanes), len(machine_names))
        workers = max(1, machine_workers + 1)
        if workers:
            executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="rlab-fleet")
            futures: dict[Future[Mapping[str, Any] | None], tuple[str, str, float]] = {}
            try:
                trigger_reason = ", ".join(
                    sorted({str(row.get("reason") or "unknown") for row in triggers})
                )
                for machine_name in machine_names:
                    lane_deadline = min(pass_deadline, clock() + lane_timeout_seconds)

                    def run_machine_lane(
                        name: str = machine_name,
                        deadline: float = lane_deadline,
                    ) -> Mapping[str, Any] | None:
                        lane_deadline_at = _iso_utc(
                            _utc_now() + timedelta(seconds=max(0.0, deadline - clock()))
                        )
                        progress.start_lane(
                            name,
                            action="RECONCILING TRAIN",
                            reason=trigger_reason,
                            deadline_at=lane_deadline_at,
                            blast_radius=f"train work assigned to {name}",
                        )
                        try:
                            if reconcile_machine is _default_reconcile_machine:
                                return _default_reconcile_machine(
                                    repo_root,
                                    name,
                                    deadline,
                                    progress=lambda action, reason: progress.start_lane(
                                        name,
                                        action=action,
                                        reason=reason,
                                        deadline_at=lane_deadline_at,
                                        blast_radius=f"train work assigned to {name}",
                                    ),
                                )
                            return reconcile_machine(repo_root, name, deadline)
                        finally:
                            progress.finish_lane(name)

                    future = executor.submit(run_machine_lane)
                    futures[future] = ("machine", machine_name, lane_deadline)
                eval_deadline = min(pass_deadline, clock() + lane_timeout_seconds)

                def run_eval_lane() -> Mapping[str, Any] | None:
                    lane_deadline_at = _iso_utc(
                        _utc_now() + timedelta(seconds=max(0.0, eval_deadline - clock()))
                    )
                    progress.start_lane(
                        "eval",
                        action="RECONCILING EVALUATION",
                        reason=trigger_reason,
                        deadline_at=lane_deadline_at,
                        blast_radius="checkpoint evaluation, promotion, and publication",
                    )
                    try:
                        if reconcile_eval is _default_reconcile_eval:
                            return _default_reconcile_eval(
                                repo_root,
                                eval_deadline,
                                progress=lambda action, reason: progress.start_lane(
                                    "eval",
                                    action=action,
                                    reason=reason,
                                    deadline_at=lane_deadline_at,
                                    blast_radius=(
                                        "checkpoint evaluation, promotion, and publication"
                                    ),
                                ),
                            )
                        return reconcile_eval(repo_root, eval_deadline)
                    finally:
                        progress.finish_lane("eval")

                eval_future = executor.submit(run_eval_lane)
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
        try:
            final_workload = dict(workload_snapshot(repo_root) or {})
        except Exception as exc:
            final_workload = dict(summary.get("workload") or {})
            final_workload.setdefault("needs_action", []).append(
                _watch_item(
                    "scheduler",
                    "Final work snapshot unavailable",
                    f"{type(exc).__name__}: {exc}",
                    reason_code="snapshot_error",
                    resolution="manual inspection",
                    blast_radius="watcher queue visibility only",
                    command="rlab fleet service logs --tail 100",
                )
            )
        summary["workload"] = final_workload
        summary["meaningful_changes"] = _meaningful_work_changes(
            previous_pass.get("workload") if isinstance(previous_pass, Mapping) else None,
            final_workload,
        )
        summary["finished_at"] = _iso_utc()
        summary["duration_seconds"] = round(max(0.0, clock() - started_monotonic), 6)
        _atomic_write_json(state_dir / "last-pass.json", redact(summary))
        progress.close()
        if summary["meaningful_changes"]:
            event_log.emit(
                "work_changed",
                pass_id=pass_id,
                changes=summary["meaningful_changes"],
            )
        event_log.emit(
            "pass_finished",
            pass_id=pass_id,
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
    result = install_controller_services(_paths_from_args(args), replace=bool(args.replace))
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
    status = controller_services_status(_paths_from_args(args))
    if args.json:
        print(json.dumps(status, sort_keys=True))
    else:
        print(
            f"fleet service installed={status['installed']} loaded={status['loaded']} "
            f"healthy={status['healthy']} poll={status['poll_seconds']}s"
        )
        for name, row in status["controllers"].items():
            print(
                f"  {name}: loaded={row['loaded']} running={row['running']} "
                f"protocol={row['installed_protocol_version']}/"
                f"{row['expected_protocol_version']} label={row['label']}"
            )
    return 0 if status["installed"] and status["loaded"] and status["healthy"] else 1


def cmd_reload(args: argparse.Namespace) -> int:
    controller = str(args.controller)
    reloaded = reload_controller_service(_paths_from_args(args), controller)
    print(json.dumps({"controller": controller, "reloaded": reloaded}, sort_keys=True))
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    report = controller_services_doctor(_paths_from_args(args))
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


def cmd_watch(args: argparse.Namespace) -> int:
    from rlab.fleet_watch import run_watch_command

    return run_watch_command(args)


def cmd_uninstall(args: argparse.Namespace) -> int:
    changed = uninstall_controller_services(_paths_from_args(args), force=bool(args.force))
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

    reload_controller = commands.add_parser(
        "reload", help="Reload one idle persistent fleet controller."
    )
    _add_path_arguments(reload_controller)
    reload_controller.add_argument("--controller", choices=CONTROLLER_NAMES, required=True)
    reload_controller.set_defaults(func=cmd_reload)

    doctor = commands.add_parser("doctor", help="Validate the service installation.")
    _add_path_arguments(doctor)
    doctor.add_argument("--json", action="store_true")
    doctor.set_defaults(func=cmd_doctor)

    logs = commands.add_parser("logs", help="Read rotating service logs.")
    _add_path_arguments(logs)
    logs.add_argument("--tail", type=int, default=100)
    logs.add_argument("--follow", action="store_true")
    logs.set_defaults(func=cmd_logs)

    watch = commands.add_parser("watch", help="Watch scheduler health and current work.")
    _add_path_arguments(watch)
    output = watch.add_mutually_exclusive_group()
    output.add_argument("--once", action="store_true", help="Print one human snapshot and exit.")
    output.add_argument("--plain", action="store_true", help="Stream append-only human updates.")
    output.add_argument("--json", action="store_true", help="Print one JSON snapshot and exit.")
    watch.add_argument("--no-color", action="store_true", help="Use the monochrome TUI theme.")
    watch.set_defaults(func=cmd_watch)

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
