from __future__ import annotations

import argparse
import json
import os
import subprocess
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rlab.json_utils import json_safe
from rlab.job_execution import (
    collect_result_metadata,
    train_command_for_job,
    write_train_config_file,
)


RESULT_SCHEMA_VERSION = 1


def load_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("job payload must be a JSON object")
    if payload.get("job_kind") != "train":
        raise ValueError("job payload must define job_kind=train")
    if not isinstance(payload.get("job"), dict):
        raise ValueError("job payload must define a job object")
    return payload


def write_result(output_dir: Path, result: Mapping[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "result.json"
    path.write_text(json.dumps(json_safe(dict(result)), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def base_result(payload: Mapping[str, Any]) -> dict[str, Any]:
    job = dict(payload["job"])
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "job_kind": payload.get("job_kind"),
        "job_id": int(job["id"]),
        "launch_id": payload.get("launch_id"),
        "machine": payload.get("machine"),
        "runtime_image_ref": payload.get("runtime_image_ref") or job.get("runtime_image_ref"),
    }


def run_train_payload(payload: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    job = dict(payload["job"])
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    config_path = write_train_config_file(job, output_dir / "train_config.json")
    command = train_command_for_job(config_path)
    log_path = log_dir / f"train_job_{job['id']}_{uuid.uuid4().hex[:8]}.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.run(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            env=os.environ.copy(),
        )
    metadata = collect_result_metadata(job, log_path)
    status = "succeeded" if process.returncode == 0 else "failed"
    result = {
        **base_result(payload),
        "status": status,
        "exit_code": process.returncode,
        "train": {
            "status": status,
            "result": metadata,
            "log_path": str(log_path),
        },
    }
    if process.returncode != 0:
        result["error"] = f"train process exited {process.returncode}"
    return result


def run_payload(payload: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    if payload.get("job_kind") == "train":
        return run_train_payload(payload, output_dir)
    raise ValueError(f"unsupported job_kind: {payload.get('job_kind')}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one claimed rlab job payload and write result.json.")
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = load_payload(args.payload)
    try:
        result = run_payload(payload, args.output_dir)
    except Exception as exc:
        result = {
            **base_result(payload),
            "status": "failed",
            "exit_code": 1,
            "error": repr(exc),
        }
        write_result(args.output_dir, result)
        raise
    write_result(args.output_dir, result)
    return int(result.get("exit_code") or 0)


if __name__ == "__main__":
    raise SystemExit(main())
