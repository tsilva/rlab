from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

import yaml

from rlab.run_contracts import RUN_ID_PATTERN


DSTACK_VERSION = "0.20.28"
DSTACK_SERVER_IMAGE = (
    "dstackai/dstack@sha256:86b820cf5f6e0cfc54dd387527493168a4045b362ca9459265ea9828eef0b4af"
)
TERMINAL_DSTACK_STATUSES = {
    "done",
    "failed",
    "stopped",
    "terminated",
    "aborted",
}


@dataclass(frozen=True)
class ComputeRequest:
    kind: Literal["auto", "local", "spot", "on-demand"]
    target: str | None
    max_price: float | None
    max_cost_usd: float | None
    allow_on_demand: bool
    max_duration_seconds: int

    def validate(self) -> None:
        if self.kind not in {"auto", "local", "spot", "on-demand"}:
            raise ValueError(f"unsupported compute policy: {self.kind}")
        if int(self.max_duration_seconds) <= 0:
            raise ValueError("max_duration_seconds must be finite and positive")
        for label, value in (
            ("max_price", self.max_price),
            ("max_cost_usd", self.max_cost_usd),
        ):
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{label} must be finite and positive")
        if (self.max_price is None) != (self.max_cost_usd is None):
            raise ValueError("--max-price and --max-cost-usd must be supplied together")
        if self.kind == "spot" and self.max_price is None:
            raise ValueError("spot compute requires --max-price and --max-cost-usd")
        if self.kind == "on-demand":
            if not self.allow_on_demand:
                raise ValueError("on-demand compute requires --allow-on-demand")
            if self.max_price is None:
                raise ValueError("on-demand compute requires a finite cloud budget")
        if self.kind != "on-demand" and self.allow_on_demand:
            raise ValueError("--allow-on-demand is valid only with --compute on-demand")

    @property
    def bounded_duration_seconds(self) -> int:
        self.validate()
        if self.max_price is None or self.max_cost_usd is None:
            return int(self.max_duration_seconds)
        cost_seconds = math.floor(self.max_cost_usd / self.max_price * 3600)
        bounded = min(int(self.max_duration_seconds), cost_seconds)
        if bounded < 60:
            raise ValueError("cloud budget allows less than 60 seconds of runtime")
        return bounded

    def as_manifest(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "target": self.target,
            "max_price": self.max_price,
            "max_cost_usd": self.max_cost_usd,
            "allow_on_demand": self.allow_on_demand,
            "max_duration_seconds": self.bounded_duration_seconds,
        }


@dataclass(frozen=True)
class TaskRequest:
    run_id: str
    task_name: str
    image: str
    manifest_uri: str
    compute: ComputeRequest
    secret_env: Sequence[str]
    cpu: int = 12
    memory: str = "40GB"
    gpu: str = "1"
    disk: str = "50GB"
    retry_duration: str = "24h"
    rom_mount: str | None = None

    def validate(self) -> None:
        if RUN_ID_PATTERN.fullmatch(self.run_id) is None:
            raise ValueError("run_id must be the immutable rlab run id")
        if (
            not self.task_name
            or len(self.task_name) > 63
            or any(
                character not in "abcdefghijklmnopqrstuvwxyz0123456789-"
                for character in self.task_name
            )
        ):
            raise ValueError("dstack task_name must be a lowercase DNS-style name")
        if "@sha256:" not in self.image:
            raise ValueError("dstack task image must be immutable")
        if not self.manifest_uri:
            raise ValueError("manifest URI must not be empty")
        if int(self.cpu) <= 0:
            raise ValueError("cpu must be positive")
        if not self.memory or not self.gpu or not self.disk:
            raise ValueError("memory, gpu, and disk requirements must not be empty")
        if self.rom_mount is not None:
            source, separator, destination = self.rom_mount.partition(":")
            if (
                separator != ":"
                or not source.startswith("/")
                or not destination.startswith("/")
            ):
                raise ValueError("ROM mount must be /host/path:/container/path")
        self.compute.validate()


@dataclass(frozen=True)
class DstackTask:
    project: str
    name: str
    status: str
    submitted_config: Mapping[str, Any] | None = None
    raw: Mapping[str, Any] | None = None

    @property
    def terminal(self) -> bool:
        return self.status.lower().replace("_", "-") in TERMINAL_DSTACK_STATUSES


def _duration_text(seconds: int) -> str:
    if seconds % 86400 == 0:
        return f"{seconds // 86400}d"
    if seconds % 3600 == 0:
        return f"{seconds // 3600}h"
    if seconds % 60 == 0:
        return f"{seconds // 60}m"
    return f"{seconds}s"


def render_task_config(request: TaskRequest) -> dict[str, Any]:
    request.validate()
    compute = request.compute
    config: dict[str, Any] = {
        "type": "task",
        "name": request.task_name,
        "image": request.image.removeprefix("docker:"),
        "working_dir": "/workspace",
        "env": [
            f"RLAB_RUN_MANIFEST_URI={request.manifest_uri}",
            "RLAB_ORCHESTRATOR=dstack",
            *sorted(set(request.secret_env)),
        ],
        "commands": [
            "python -m rlab.run_supervisor --manifest-uri "
            '"$RLAB_RUN_MANIFEST_URI"'
        ],
        "resources": {
            # An unsplit SSH fleet contributes the whole machine as one block.
            # Minimum ranges admit that block while retaining the resource floor
            # for cloud offers; exact CPU/RAM values reject larger local hosts.
            "cpu": f"{int(request.cpu)}..",
            "memory": f"{request.memory}..",
            "gpu": request.gpu,
            "disk": f"{request.disk}..",
        },
        "max_duration": _duration_text(compute.bounded_duration_seconds),
        "stop_duration": "10m",
        "retry": {
            "on_events": ["no-capacity", "interruption"],
            "duration": request.retry_duration,
        },
        "tags": {
            "managed-by": "rlab",
            "rlab-run-id": request.run_id,
        },
    }
    if request.rom_mount:
        config["volumes"] = [request.rom_mount]
        config["env"].append("RLAB_ROM_CACHE_READ_ONLY=1")
    target = str(compute.target or "").strip()
    if compute.kind == "auto" and compute.max_price is not None:
        config["creation_policy"] = "reuse-or-create"
        # With no explicit fleet filter dstack considers reusable fleet instances
        # before provisionable backend offers. That gives idle local hosts priority
        # while retaining the bounded spot fallback.
        if target:
            config["fleets"] = [target]
        config["spot_policy"] = "spot"
        config["max_price"] = float(compute.max_price)
        config["idle_duration"] = "0s"
    elif compute.kind in {"auto", "local"}:
        config["creation_policy"] = "reuse"
        config["fleets"] = [target or "b3"]
    else:
        config["creation_policy"] = "create"
        config["spot_policy"] = "spot" if compute.kind == "spot" else "on-demand"
        config["max_price"] = float(compute.max_price)
        if target:
            config["backends"] = [target]
        config["idle_duration"] = "0s"
    return config


def render_fleet_config(
    *,
    name: str,
    hostname: str,
    user: str,
    identity_file: str,
) -> dict[str, Any]:
    if not all(str(value).strip() for value in (name, hostname, user, identity_file)):
        raise ValueError("fleet name, hostname, user, and identity file are required")
    return {
        "type": "fleet",
        "name": name,
        "placement": "any",
        "blocks": 1,
        "ssh_config": {
            "user": user,
            "identity_file": identity_file,
            "hosts": [hostname],
        },
    }


class DstackBackend:
    def __init__(
        self,
        *,
        project: str = "main",
        executable: str = "dstack",
        environment: Mapping[str, str] | None = None,
    ):
        self.project = str(project).strip() or "main"
        self.executable = executable
        self.environment = dict(os.environ if environment is None else environment)

    def _command(
        self,
        arguments: Sequence[str],
        *,
        input_text: str | None = None,
        timeout: float = 60,
    ) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            [self.executable, *arguments],
            input=input_text,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            env=self.environment,
            timeout=timeout,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(
                f"dstack {' '.join(arguments[:2])} failed with exit {result.returncode}: "
                f"{detail}"
            )
        return result

    def preflight(self) -> None:
        if shutil.which(self.executable, path=self.environment.get("PATH")) is None:
            raise RuntimeError(f"{self.executable} is not installed")
        result = self._command(["-v"])
        if result.stdout.strip() != DSTACK_VERSION:
            raise RuntimeError(
                f"dstack CLI must be {DSTACK_VERSION}, got {result.stdout.strip() or 'unknown'}"
            )
        if not self.environment.get("DSTACK_SERVER_URL", "").strip():
            raise RuntimeError("DSTACK_SERVER_URL must identify the pinned private server")
        if not self.environment.get("DSTACK_TOKEN", "").strip():
            raise RuntimeError("DSTACK_TOKEN must be set outside source control")

    def select_compute(
        self,
        request: ComputeRequest,
        *,
        local_fleet: str = "b3",
    ) -> tuple[ComputeRequest, Mapping[str, Any] | None]:
        request.validate()
        if request.kind != "auto":
            return request, None
        local = replace(
            request,
            kind="local",
            target=str(request.target or local_fleet),
            max_price=None,
            max_cost_usd=None,
        )
        if request.max_price is None:
            return local, None
        self.preflight()
        result = self._command(
            [
                "offer",
                "--project",
                self.project,
                "--json",
                "--fleet",
                str(request.target or local_fleet),
                "--reuse",
                "--cpu",
                "12..",
                "--memory",
                "40GB..",
                "--gpu",
                "1",
                "--disk",
                "50GB..",
            ],
            timeout=60,
        )
        document = json.loads(result.stdout or "{}")
        offers = document.get("offers") if isinstance(document, Mapping) else None
        for offer in offers if isinstance(offers, list) else []:
            if not isinstance(offer, Mapping):
                continue
            if str(offer.get("availability") or "") != "idle":
                continue
            return local, dict(offer)
        cloud = replace(
            request,
            kind="spot",
            target=None,
            allow_on_demand=False,
        )
        cloud.validate()
        return cloud, None

    def submit(self, request: TaskRequest) -> DstackTask:
        self.preflight()
        config = render_task_config(request)
        self._command(
            [
                "apply",
                "--project",
                self.project,
                "-f",
                "-",
                "-y",
                "-d",
            ],
            input_text=yaml.safe_dump(config, sort_keys=False),
            timeout=180,
        )
        return DstackTask(
            project=self.project,
            name=request.task_name,
            status="submitted",
            submitted_config=config,
        )

    def status(self, name: str) -> DstackTask:
        result = self._command(
            ["ps", "--project", self.project, "--all", "--json"],
            timeout=30,
        )
        parsed = json.loads(result.stdout or "[]")
        rows = parsed if isinstance(parsed, list) else parsed.get("runs") or []
        ordered_rows = sorted(
            (row for row in rows if isinstance(row, Mapping)),
            key=lambda row: str(row.get("submitted_at") or ""),
            reverse=True,
        )
        for row in ordered_rows:
            if not isinstance(row, Mapping):
                continue
            run_spec = row.get("run_spec")
            configuration = (
                run_spec.get("configuration")
                if isinstance(run_spec, Mapping)
                else None
            )
            configured_name = (
                configuration.get("name")
                if isinstance(configuration, Mapping)
                else None
            )
            if str(row.get("name") or row.get("run_name") or configured_name or "") != name:
                continue
            status = str(row.get("status") or row.get("state") or "unknown")
            return DstackTask(
                project=self.project,
                name=name,
                status=status,
                raw=dict(row),
            )
        raise KeyError(f"dstack task not found: {self.project}/{name}")

    def logs(self, name: str, *, since: str | None = None) -> str:
        arguments = ["logs", "--project", self.project]
        if since:
            arguments.extend(["--since", since])
        arguments.append(name)
        return self._command(arguments, timeout=300).stdout

    def cancel(self, name: str, *, abort: bool = False) -> None:
        arguments = ["stop", "--project", self.project, "-y"]
        if abort:
            arguments.append("-x")
        arguments.append(name)
        self._command(arguments, timeout=60)
