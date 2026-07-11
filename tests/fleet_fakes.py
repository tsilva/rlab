from __future__ import annotations

import json
from pathlib import Path


class FakeCursor:
    def __init__(self, row=None, rows=None) -> None:
        self.row = row
        self.rows = [] if rows is None else rows
        self.executed_sql = ""
        self.executed_params = {}
        self.executed_sqls = []
        self.executed_params_list = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def execute(self, sql, params=None) -> None:
        self.executed_sql = sql
        self.executed_params = params or {}
        self.executed_sqls.append(sql)
        self.executed_params_list.append(params or {})

    def fetchone(self):
        return self.row if self.row is not None else (self.rows[0] if self.rows else None)

    def fetchall(self):
        return self.rows


class FakeConnection:
    def __init__(self, row=None, rows=None) -> None:
        self.cursor_obj = FakeCursor(row=row, rows=rows)
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def cursor(self):
        return self.cursor_obj

    def close(self) -> None:
        self.closed = True


def write_machine_registry(
    path: Path,
    *,
    backend: str,
    max_parallel_containers: int,
    host_root: str,
    machine_name: str = "beast-test",
    run_target: str = "rtx4090",
) -> None:
    machine = {
        "backend": backend,
        "run_target": run_target,
        "pull_policy": "never",
        "limits": {"max_parallel_containers": max_parallel_containers},
        "paths": {
            "host_root": host_root,
            "payloads_dir": f"{host_root}/payloads",
            "outputs_dir": f"{host_root}/outputs",
            "logs_dir": f"{host_root}/logs",
            "roms_dir": "/host/roms" if backend == "docker_ssh" else "/roms",
            "env_file": f"{host_root}/.env.runner",
        },
    }
    if backend == "docker_ssh":
        machine.update(
            {
                "ssh_target": f"tsilva@{machine_name}",
                "docker": {"command": ["sudo", "-n", "docker"]},
            }
        )
    path.write_text(json.dumps({"machines": {machine_name: machine}}), encoding="utf-8")
