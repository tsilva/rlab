from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping


def _state_dir(argv: list[str]) -> Path:
    try:
        value = argv[argv.index("--state-dir") + 1]
    except (ValueError, IndexError):
        value = str(
            Path.home() / "Library" / "Application Support" / "rlab" / "fleet-service"
        )
    return Path(value).expanduser().resolve()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _notify(message: str) -> None:
    script = (
        f"display notification {json.dumps(message)} "
        f"with title {json.dumps('rlab fleet service bootstrap failure')}"
    )
    subprocess.run(
        ["/usr/bin/osascript", "-e", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )


def _record_bootstrap_failure(argv: list[str], exc: Exception) -> None:
    state_dir = _state_dir(argv)
    now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    detail = f"{type(exc).__name__} while loading or starting the reconciler"
    _atomic_json(
        state_dir / "last-pass.json",
        {
            "schema_version": 1,
            "started_at": now,
            "finished_at": now,
            "status": "error",
            "machines": [],
            "eval": {"status": "error"},
            "error": detail,
        },
    )
    _atomic_json(
        state_dir / "health.json",
        {
            "healthy": False,
            "status": "error",
            "consecutive_failures": 1,
            "alert_active": True,
            "failure_fingerprint": type(exc).__name__,
            "last_failure": detail,
            "last_notified_at": now,
            "updated_at": now,
        },
    )
    _notify(detail)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        from rlab.fleet_service import main as fleet_main

        return int(fleet_main(args))
    except Exception as exc:
        traceback.print_exc()
        try:
            _record_bootstrap_failure(args, exc)
        except Exception:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
