from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from rlab.modal_eval_config import ModalEvalConfig, modal_app_name


MIN_CLEANUP_REMAINING_SECONDS = 15.0
MAX_MODAL_CLI_SECONDS = 15.0
APP_ID_PATTERN = re.compile(r"ap-[A-Za-z0-9]{22}")


@dataclass(frozen=True)
class ModalAppInfo:
    app_id: str
    name: str
    state: str
    running_tasks: int
    created_at: datetime


class ModalAppClient(Protocol):
    def list_apps(self, *, deadline_monotonic: float) -> Sequence[ModalAppInfo]: ...
    def stop_app(self, app_id: str, *, deadline_monotonic: float) -> None: ...


def _command_timeout(deadline_monotonic: float) -> float:
    remaining = deadline_monotonic - time.monotonic()
    if remaining <= 1.0:
        raise TimeoutError("Modal app cleanup deadline expired")
    return min(MAX_MODAL_CLI_SECONDS, remaining - 1.0)


class DefaultModalAppClient:
    def _run(self, arguments: Sequence[str], *, deadline_monotonic: float) -> str:
        completed = subprocess.run(
            [sys.executable, "-m", "modal", *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=_command_timeout(deadline_monotonic),
        )
        if completed.returncode != 0:
            command = " ".join(arguments[:2])
            raise RuntimeError(f"Modal {command} failed with exit code {completed.returncode}")
        return completed.stdout

    def list_apps(self, *, deadline_monotonic: float) -> Sequence[ModalAppInfo]:
        raw = json.loads(
            self._run(("app", "list", "--json"), deadline_monotonic=deadline_monotonic)
        )
        if not isinstance(raw, list):
            raise ValueError("Modal app inventory must be a list")
        apps: list[ModalAppInfo] = []
        for value in raw:
            if not isinstance(value, Mapping):
                raise ValueError("Modal app inventory rows must be mappings")
            app_id = str(value.get("app_id") or "").strip()
            name = str(value.get("description") or "").strip()
            state = str(value.get("state") or "").strip()
            try:
                running_tasks = int(value.get("tasks"))
                created_at = datetime.fromisoformat(str(value.get("created_at") or ""))
            except (TypeError, ValueError) as exc:
                raise ValueError("Modal app inventory row has invalid fields") from exc
            if not app_id or not name or not state or running_tasks < 0 or created_at.tzinfo is None:
                raise ValueError("Modal app inventory row has invalid fields")
            apps.append(
                ModalAppInfo(
                    app_id=app_id,
                    name=name,
                    state=state,
                    running_tasks=running_tasks,
                    created_at=created_at.astimezone(UTC),
                )
            )
        return tuple(apps)

    def stop_app(self, app_id: str, *, deadline_monotonic: float) -> None:
        self._run(
            ("app", "stop", app_id, "--yes"),
            deadline_monotonic=deadline_monotonic,
        )


def latest_runtime_image_ref() -> str:
    from rlab.runtime_refs import recent_runtime_images

    releases = recent_runtime_images(limit=1)
    if not releases:
        raise RuntimeError("latest runtime image receipt is unavailable")
    return str(releases[0].runtime_image_ref)


def protected_modal_app_names(
    conn,
    config: ModalEvalConfig,
    *,
    latest_runtime_ref: Callable[[], str] = latest_runtime_image_ref,
) -> set[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT t.runtime_image_ref
            FROM train_jobs t
            LEFT JOIN eval_runs r ON r.train_job_id = t.id
            WHERE t.train_config->>'checkpoint_eval_backend' = 'modal'
              AND (
                t.status IN ('pending', 'launching', 'starting', 'running')
                OR r.status IN ('active', 'awaiting_artifact_recovery', 'finalizing')
                OR EXISTS (
                  SELECT 1 FROM eval_jobs j
                  WHERE j.train_job_id = t.id
                    AND j.status IN ('pending', 'dispatching', 'submitted', 'blocked_budget')
                )
                OR EXISTS (
                  SELECT 1 FROM eval_attempts a
                  JOIN eval_jobs j ON j.id = a.eval_job_id
                  WHERE j.train_job_id = t.id
                    AND a.status IN ('dispatching', 'submitted')
                )
              )
            """
        )
        runtime_refs = [str(row["runtime_image_ref"]) for row in cur.fetchall()]
        cur.execute(
            """
            SELECT DISTINCT modal_app_name FROM eval_attempts
            WHERE status IN ('dispatching', 'submitted')
            """
        )
        attempt_app_names = {
            str(row["modal_app_name"]).strip()
            for row in cur.fetchall()
            if str(row.get("modal_app_name") or "").strip()
        }
    names = {modal_app_name(config.app_name_prefix, value) for value in runtime_refs}
    names.update(attempt_app_names)
    names.add(modal_app_name(config.app_name_prefix, latest_runtime_ref()))
    return names


def _result(status: str, **updates: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": status,
        "owned_deployed": 0,
        "protected": 0,
        "candidates": 0,
        "stopped": 0,
        "stopped_apps": [],
        "errors": [],
    }
    result.update(updates)
    return result


def _maintenance_due(marker: Path, *, now: datetime, interval_seconds: int) -> bool:
    try:
        age = now.timestamp() - marker.stat().st_mtime
    except FileNotFoundError:
        return True
    return age >= interval_seconds


def run_modal_app_cleanup(
    conn,
    config: ModalEvalConfig,
    *,
    repo_root: Path,
    deadline_monotonic: float,
    client: ModalAppClient | None = None,
    latest_runtime_ref: Callable[[], str] = latest_runtime_image_ref,
    now: datetime | None = None,
) -> dict[str, Any]:
    if not config.cleanup_enabled:
        return _result("disabled")
    now = (now or datetime.now(UTC)).astimezone(UTC)
    marker = repo_root / "logs" / "fleet" / "modal-app-cleanup.stamp"
    if not _maintenance_due(
        marker,
        now=now,
        interval_seconds=config.cleanup_interval_seconds,
    ):
        return _result("not_due")
    if deadline_monotonic - time.monotonic() < MIN_CLEANUP_REMAINING_SECONDS:
        return _result("skipped_deadline")
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
    except OSError as exc:
        return _result("error", errors=[f"maintenance marker: {type(exc).__name__}"])
    try:
        protected = protected_modal_app_names(
            conn,
            config,
            latest_runtime_ref=latest_runtime_ref,
        )
        inventory = tuple((client or DefaultModalAppClient()).list_apps(
            deadline_monotonic=deadline_monotonic
        ))
    except Exception as exc:
        return _result("error", errors=[type(exc).__name__])

    owned_pattern = re.compile(rf"{re.escape(config.app_name_prefix)}-[0-9a-f]{{12}}")
    owned = tuple(
        app
        for app in inventory
        if app.state == "deployed" and owned_pattern.fullmatch(app.name)
    )
    if any(not APP_ID_PATTERN.fullmatch(app.app_id) for app in owned):
        return _result(
            "error",
            owned_deployed=len(owned),
            protected=len({app.name for app in owned} & protected),
            errors=["invalid owned Modal app id"],
        )
    grace_seconds = config.cleanup_grace_seconds
    candidates = sorted(
        (
            app
            for app in owned
            if app.name not in protected
            and app.running_tasks == 0
            and (now - app.created_at).total_seconds() >= grace_seconds
        ),
        key=lambda app: (app.created_at, app.app_id),
    )
    stopped: list[str] = []
    errors: list[str] = []
    app_client = client or DefaultModalAppClient()
    for app in candidates[: config.cleanup_max_stops_per_pass]:
        if deadline_monotonic - time.monotonic() <= 1.0:
            errors.append("cleanup deadline expired")
            break
        try:
            app_client.stop_app(app.app_id, deadline_monotonic=deadline_monotonic)
        except Exception as exc:
            errors.append(f"{app.name}: {type(exc).__name__}")
        else:
            stopped.append(app.name)
    return _result(
        "partial" if errors else "ok",
        owned_deployed=len(owned),
        protected=len({app.name for app in owned} & protected),
        candidates=len(candidates),
        stopped=len(stopped),
        stopped_apps=stopped,
        errors=errors,
    )
