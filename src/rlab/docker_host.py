from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import shlex
import subprocess
import sys
import time
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rlab.fleet_labels import (
    DEFAULT_RUNTIME_IMAGE_REPOSITORIES,
    JOB_CONTAINER_LABEL,
    JOB_KIND_LABEL,
    LABEL_PREFIX,
    LAUNCH_ID_LABEL,
    MANAGED_LABEL,
)
from rlab.json_utils import json_safe
from rlab.machines import MachineConfig
from rlab.runtime_refs import docker_image_ref, normalize_runtime_image_ref
from rlab.rom_assets import validate_rom_asset_manifest
from rlab.workspace_contract import CleanupBatchEnvelope, WorkspaceManifest, exact_bind_mount


GPU_TEST_IMAGE = "nvidia/cuda:12.9.1-base-ubuntu22.04"
SSH_CONNECT_TIMEOUT_SECONDS = 10
MACHINE_COMMAND_TIMEOUT_SECONDS = 120.0
DOCKER_PULL_TIMEOUT_SECONDS = 900.0
DOCKER_STOP_TIMEOUT_SECONDS = 150.0
TRAIN_CONTAINER_STOP_TIMEOUT_SECONDS = 300
SSH_WORKSPACE_HELPER_PATH = "/usr/local/libexec/rlab-workspace-helper"


@dataclass(frozen=True)
class HostOperationResult:
    ok: bool
    detail: str = ""


@dataclass(frozen=True)
class JobContainer:
    machine: str
    name: str
    state: str
    status: str
    labels: dict[str, str]

    @property
    def launch_id(self) -> str | None:
        return self.labels.get(LAUNCH_ID_LABEL)

    @property
    def job_kind(self) -> str | None:
        return self.labels.get(JOB_KIND_LABEL)


@dataclass(frozen=True)
class RuntimeHostImage:
    machine: str
    repository: str
    digest: str
    image_id: str

    @property
    def image_ref(self) -> str:
        return f"{self.repository}@{self.digest}"

    @property
    def runtime_image_ref(self) -> str:
        return f"docker:{self.image_ref}"


@dataclass(frozen=True)
class ResultObservation:
    state: str
    payload: dict[str, Any] | None = None
    error: str | None = None


class MachineCommandTimeout(RuntimeError):
    def __init__(self, machine: str, timeout: float) -> None:
        super().__init__(f"machine command timed out machine={machine} timeout={timeout:g}s")
        self.machine = machine
        self.timeout = timeout


def _shell_join(parts: Sequence[str]) -> str:
    return shlex.join([str(part) for part in parts])


def _machine_docker_command(machine: MachineConfig, args: Sequence[str]) -> list[str]:
    return [*machine.docker_command, *args]


def _machine_ssh_prefix(machine: MachineConfig) -> list[str]:
    if machine.backend != "docker_ssh":
        return []
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={SSH_CONNECT_TIMEOUT_SECONDS}",
        "-o",
        "ServerAliveInterval=10",
        "-o",
        "ServerAliveCountMax=3",
        *machine.ssh_options,
        machine.ssh_target,
    ]


def _run_machine_shell(
    machine: MachineConfig,
    script: str,
    *,
    input_text: str | None = None,
    capture: bool = False,
    timeout: float | None = MACHINE_COMMAND_TIMEOUT_SECONDS,
    deadline_monotonic: float | None = None,
) -> subprocess.CompletedProcess[str]:
    effective_timeout = timeout
    if timeout is not None and deadline_monotonic is not None:
        remaining = deadline_monotonic - time.monotonic()
        if remaining <= 0:
            raise MachineCommandTimeout(machine.name, 0.0)
        effective_timeout = min(float(timeout), max(0.1, remaining))
    if machine.backend == "local_docker":
        command = ["sh", "-lc", script]
    else:
        remote_command = ["sh", "-lc", shlex.quote(script)]
        if effective_timeout is not None:
            remote_timeout = max(1.0, float(effective_timeout) - 5.0)
            remote_command = [
                "timeout",
                "--signal=TERM",
                "--kill-after=5s",
                f"{remote_timeout:g}s",
                *remote_command,
            ]
        command = [*_machine_ssh_prefix(machine), *remote_command]
    try:
        return subprocess.run(
            command,
            input=input_text,
            capture_output=capture,
            text=True,
            check=False,
            timeout=effective_timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise MachineCommandTimeout(machine.name, float(effective_timeout or 0.0)) from exc


def _run_machine_docker(
    machine: MachineConfig,
    docker_args: Sequence[str],
    *,
    input_text: str | None = None,
    capture: bool = False,
    timeout: float | None = MACHINE_COMMAND_TIMEOUT_SECONDS,
    deadline_monotonic: float | None = None,
) -> subprocess.CompletedProcess[str]:
    return _run_machine_shell(
        machine,
        _shell_join(_machine_docker_command(machine, docker_args)),
        input_text=input_text,
        capture=capture,
        timeout=timeout,
        deadline_monotonic=deadline_monotonic,
    )


def _operation_result(result: subprocess.CompletedProcess[str]) -> HostOperationResult:
    detail = (result.stderr or result.stdout or "").strip()
    if result.returncode != 0 and not detail:
        detail = f"exit={result.returncode}"
    return HostOperationResult(ok=result.returncode == 0, detail=detail)


def _parse_docker_labels(value: str) -> dict[str, str]:
    labels: dict[str, str] = {}
    for part in str(value or "").split(","):
        if "=" not in part:
            continue
        key, label_value = part.split("=", 1)
        labels[key.strip()] = label_value.strip()
    return labels


def _parse_containers(
    machine: MachineConfig,
    output: str,
    *,
    required_label: str,
    required_value: str | None = None,
) -> list[JobContainer]:
    containers: list[JobContainer] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        labels = _parse_docker_labels(str(row.get("Labels") or ""))
        if required_label not in labels or (
            required_value is not None and labels[required_label] != required_value
        ):
            continue
        containers.append(
            JobContainer(
                machine=machine.name,
                name=str(row.get("Names") or row.get("Name") or ""),
                state=str(row.get("State") or "").lower(),
                status=str(row.get("Status") or ""),
                labels=labels,
            )
        )
    return containers


def _parse_runtime_host_images(
    machine: MachineConfig,
    output: str,
    *,
    repositories: Sequence[str] = DEFAULT_RUNTIME_IMAGE_REPOSITORIES,
) -> tuple[RuntimeHostImage, ...]:
    allowed = set(repositories)
    images: list[RuntimeHostImage] = []
    seen: set[str] = set()
    for line in output.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        repository = str(row.get("Repository") or "").strip()
        digest = str(row.get("Digest") or "").strip()
        if repository not in allowed or not digest.startswith("sha256:"):
            continue
        image = RuntimeHostImage(
            machine=machine.name,
            repository=repository,
            digest=digest,
            image_id=str(row.get("ID") or "").strip(),
        )
        try:
            normalize_runtime_image_ref(image.runtime_image_ref)
        except ValueError:
            continue
        if image.runtime_image_ref in seen:
            continue
        seen.add(image.runtime_image_ref)
        images.append(image)
    return tuple(images)


class DockerRunnerHost:
    def __init__(
        self,
        machine: MachineConfig,
        *,
        deadline_monotonic: float | None = None,
    ) -> None:
        self.machine = machine
        self.deadline_monotonic = deadline_monotonic

    def payload_host_path(self, launch_id: str) -> str:
        return f"{self.machine.paths.payloads_dir.rstrip('/')}/{launch_id}.json"

    def output_host_path(self, launch_id: str) -> str:
        return f"{self.machine.paths.outputs_dir.rstrip('/')}/{launch_id}"

    def attempt_env_host_path(self, launch_id: str) -> str:
        return f"{self.machine.paths.payloads_dir.rstrip('/')}/{launch_id}.env"

    def payload_container_path(self, launch_id: str) -> str:
        return f"{self.machine.paths.container_payloads_dir.rstrip('/')}/{launch_id}.json"

    def output_container_path(self, launch_id: str) -> str:
        return f"{self.machine.paths.container_outputs_dir.rstrip('/')}/{launch_id}"

    def run_workspace_helper(
        self,
        operation: str,
        request: Mapping[str, Any],
        *,
        timeout: float | None = MACHINE_COMMAND_TIMEOUT_SECONDS,
    ) -> dict[str, Any]:
        if self.machine.backend == "local_docker":
            command = [sys.executable, "-m", "rlab.workspace_helper", operation]
            try:
                result = subprocess.run(
                    command,
                    input=json.dumps(dict(request), sort_keys=True),
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired as exc:
                raise MachineCommandTimeout(self.machine.name, float(timeout or 0.0)) from exc
        else:
            script = _shell_join(["sudo", "-n", SSH_WORKSPACE_HELPER_PATH, operation])
            result = _run_machine_shell(
                self.machine,
                script,
                input_text=json.dumps(dict(request), sort_keys=True),
                capture=True,
                timeout=timeout,
                deadline_monotonic=self.deadline_monotonic,
            )
        if result.returncode != 0:
            raise RuntimeError(
                f"workspace helper {operation} failed on {self.machine.name}: "
                f"{(result.stderr or result.stdout or '').strip()}"
            )
        try:
            response = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"workspace helper {operation} returned invalid JSON") from exc
        if not isinstance(response, dict):
            raise RuntimeError(f"workspace helper {operation} response must be an object")
        return response

    def workspace_helper_doctor(self) -> dict[str, Any]:
        return self.run_workspace_helper("doctor", {}, timeout=5)

    def workspace_drain_zero_receipt(
        self,
        *,
        manifests: Sequence[WorkspaceManifest],
        exact_container_names: Sequence[str],
        exact_protected_paths: Sequence[str],
        receipt_nonce: str,
    ) -> dict[str, Any]:
        return self.run_workspace_helper(
            "drain-zero",
            {
                "protected_metadata_root": self.machine.paths.protected_metadata_dir,
                "manifests": [manifest.as_dict() for manifest in manifests],
                "exact_container_names": list(exact_container_names),
                "exact_protected_paths": list(exact_protected_paths),
                "receipt_nonce": receipt_nonce,
            },
            timeout=15,
        )

    def install_workspace_key_policy(
        self, *, key_revision: str, public_key: bytes
    ) -> dict[str, Any]:
        return self.run_workspace_helper(
            "install-key-policy",
            {
                "protected_metadata_root": self.machine.paths.protected_metadata_dir,
                "key_revision": key_revision,
                "public_key_base64": base64.b64encode(public_key).decode("ascii"),
            },
            timeout=10,
        )

    def reserve_workspace(
        self,
        manifest: WorkspaceManifest,
        *,
        payload: bytes,
        attempt_env: Mapping[str, str],
    ) -> dict[str, Any]:
        return self.run_workspace_helper(
            "reserve",
            {
                "manifest": manifest.as_dict(),
                "payload_base64": base64.b64encode(payload).decode("ascii"),
                "attempt_env": dict(attempt_env),
            },
        )

    def release_reservation_intent(
        self, manifest: WorkspaceManifest, *, receipt_sha256: str
    ) -> None:
        self.run_workspace_helper(
            "release-reservation-intent",
            {"manifest": manifest.as_dict(), "receipt_sha256": receipt_sha256},
        )

    def unlink_reserved_attempt_env(
        self, manifest: WorkspaceManifest, *, expected_identity: Mapping[str, Any]
    ) -> dict[str, Any]:
        return self.run_workspace_helper(
            "unlink-env",
            {"manifest": manifest.as_dict(), "expected_identity": dict(expected_identity)},
        )

    def reserve_recovery_env(
        self,
        manifest: WorkspaceManifest,
        *,
        container_generation: int,
        attempt_env: Mapping[str, str],
    ) -> dict[str, Any]:
        return self.run_workspace_helper(
            "reserve-generation-env",
            {
                "manifest": manifest.as_dict(),
                "container_generation": int(container_generation),
                "attempt_env": dict(attempt_env),
            },
        )

    def release_recovery_env_intent(
        self,
        manifest: WorkspaceManifest,
        *,
        container_generation: int,
        receipt_sha256: str,
    ) -> dict[str, Any]:
        return self.run_workspace_helper(
            "release-generation-env-intent",
            {
                "manifest": manifest.as_dict(),
                "container_generation": int(container_generation),
                "receipt_sha256": receipt_sha256,
            },
        )

    def delete_prepare(
        self,
        manifest: WorkspaceManifest,
        *,
        cleanup_row_id: int,
        cleanup_attempt_id: str,
        control_revision: int,
        cursor_sha256: str,
    ) -> dict[str, Any]:
        return self.run_workspace_helper(
            "delete-prepare",
            {
                "manifest": manifest.as_dict(),
                "cleanup_row_id": int(cleanup_row_id),
                "cleanup_attempt_id": cleanup_attempt_id,
                "control_revision": int(control_revision),
                "cursor_sha256": cursor_sha256,
            },
            timeout=10,
        )

    def delete_commit(
        self,
        envelope: CleanupBatchEnvelope,
        *,
        signature: str,
        manifests: Mapping[int, WorkspaceManifest],
    ) -> dict[str, Any]:
        return self.run_workspace_helper(
            "delete-commit",
            {
                "envelope": envelope.as_dict(),
                "signature": signature,
                "manifests": {
                    str(row_id): manifest.as_dict() for row_id, manifest in manifests.items()
                },
            },
            timeout=5,
        )

    def remove_prepare_receipt(
        self,
        manifest: WorkspaceManifest,
        *,
        cleanup_attempt_id: str,
        prepare_receipt_sha256: str,
    ) -> dict[str, Any]:
        return self.run_workspace_helper(
            "remove-prepare",
            {
                "manifest": manifest.as_dict(),
                "cleanup_attempt_id": cleanup_attempt_id,
                "prepare_receipt_sha256": prepare_receipt_sha256,
            },
            timeout=5,
        )

    def remove_host_deleted_journal(
        self,
        manifest: WorkspaceManifest,
        *,
        host_deleted_journal_sha256: str,
    ) -> dict[str, Any]:
        return self.run_workspace_helper(
            "remove-host-deleted-journal",
            {
                "manifest": manifest.as_dict(),
                "host_deleted_journal_sha256": host_deleted_journal_sha256,
            },
            timeout=5,
        )

    def sync_shared_env(self, values: Mapping[str, str]) -> None:
        if self.machine.backend != "docker_ssh":
            return
        keys = tuple(values)
        env_file = shlex.quote(self.machine.paths.env_file)
        shared_keys = " ".join(keys)
        awk_program = (
            f'BEGIN {{ split("{shared_keys}", keys, " "); '
            "for (i in keys) shared[keys[i]] = 1 } !($1 in shared)"
        )
        verify_keys = " ".join(shlex.quote(key) for key in keys)
        script = "\n".join(
            [
                "set -eu",
                "shared_tmp=$(mktemp)",
                "merged_tmp=$(mktemp)",
                'trap \'rm -f "$shared_tmp" "$merged_tmp"\' EXIT',
                "umask 077",
                'cat > "$shared_tmp"',
                f"if [ -f {env_file} ]; then",
                f'  awk -F= {shlex.quote(awk_program)} {env_file} > "$merged_tmp"',
                "else",
                '  : > "$merged_tmp"',
                "fi",
                'cat "$shared_tmp" >> "$merged_tmp"',
                "owner=$(id -un)",
                "group=$(id -gn)",
                f'sudo -n install -o "$owner" -g "$group" -m 0600 "$merged_tmp" {env_file}',
                f"for key in {verify_keys}; do",
                f'  count=$(grep -c "^${{key}}=" {env_file} || true)',
                '  test "$count" -eq 1',
                "done",
            ]
        )
        input_text = "".join(f"{key}={values[key]}\n" for key in keys)
        result = _run_machine_shell(
            self.machine,
            script,
            input_text=input_text,
            capture=True,
            deadline_monotonic=self.deadline_monotonic,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"failed to sync shared runner env to machine={self.machine.name} "
                f"exit={result.returncode}"
            )

    def write_payload(self, launch_id: str, payload: Mapping[str, Any]) -> None:
        path = self.payload_host_path(launch_id)
        payload_text = json.dumps(json_safe(dict(payload)), indent=2, sort_keys=True) + "\n"
        script = f"mkdir -p {shlex.quote(str(Path(path).parent))} && cat > {shlex.quote(path)}"
        result = _run_machine_shell(
            self.machine,
            script,
            input_text=payload_text,
            capture=True,
            deadline_monotonic=self.deadline_monotonic,
        )
        if result.returncode != 0:
            raise RuntimeError(f"failed to write payload {path}: {result.stderr or result.stdout}")

    def prepare_legacy_workspace(self, launch_id: str, payload: Mapping[str, Any]) -> None:
        """Prepare exact bind sources when workspace layout v1 is dormant."""

        output_path = self.output_host_path(launch_id)
        script = (
            "set -eu; umask 077; "
            f"if [ -L {shlex.quote(output_path)} ]; then exit 72; fi; "
            f"if [ -e {shlex.quote(output_path)} ]; then "
            f"test -d {shlex.quote(output_path)}; "
            f"else mkdir {shlex.quote(output_path)}; fi"
        )
        result = _run_machine_shell(
            self.machine,
            script,
            capture=True,
            deadline_monotonic=self.deadline_monotonic,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"failed to prepare legacy output {output_path}: "
                f"{result.stderr or result.stdout}"
            )
        self.write_payload(launch_id, payload)

    def write_attempt_env(self, launch_id: str, values: Mapping[str, str]) -> str:
        if not values:
            raise ValueError("attempt environment must not be empty")
        for key, value in values.items():
            if not key or any(char not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_" for char in key):
                raise ValueError(f"invalid attempt environment key: {key!r}")
            if not value or "\n" in value or "\r" in value or "\x00" in value:
                raise ValueError(f"invalid attempt environment value for {key}")
        path = self.attempt_env_host_path(launch_id)
        text = "".join(f"{key}={values[key]}\n" for key in sorted(values))
        script = (
            "set -eu; umask 077; temporary=$(mktemp); "
            'trap \'rm -f "$temporary"\' EXIT; cat > "$temporary"; '
            f'install -m 0600 "$temporary" {shlex.quote(path)}'
        )
        result = _run_machine_shell(
            self.machine,
            script,
            input_text=text,
            capture=True,
            deadline_monotonic=self.deadline_monotonic,
        )
        if result.returncode != 0:
            raise RuntimeError(f"failed to write attempt environment {path}")
        return path

    def remove_attempt_env(self, launch_id: str) -> None:
        path = self.attempt_env_host_path(launch_id)
        _run_machine_shell(
            self.machine,
            f"rm -f {shlex.quote(path)}",
            capture=True,
            deadline_monotonic=self.deadline_monotonic,
        )

    def _list_containers(
        self,
        *,
        required_label: str,
        required_value: str | None = None,
    ) -> list[JobContainer]:
        label_filter = (
            f"label={required_label}={required_value}"
            if required_value is not None
            else f"label={required_label}"
        )
        result = _run_machine_docker(
            self.machine,
            ["ps", "-a", "--filter", label_filter, "--format", "{{json .}}"],
            capture=True,
            deadline_monotonic=self.deadline_monotonic,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"docker ps failed on {self.machine.name}: {result.stderr or result.stdout}"
            )
        return _parse_containers(
            self.machine,
            result.stdout,
            required_label=required_label,
            required_value=required_value,
        )

    def list_job_containers(self) -> list[JobContainer]:
        return self._list_containers(
            required_label=JOB_CONTAINER_LABEL,
            required_value="true",
        )

    def list_runtime_image_containers(self) -> list[JobContainer]:
        return self._list_containers(required_label=f"{LABEL_PREFIX}runtime-image-ref")

    def list_runtime_images(self, repositories: Sequence[str]) -> tuple[RuntimeHostImage, ...]:
        result = _run_machine_docker(
            self.machine,
            ["image", "ls", "--digests", "--format", "{{json .}}"],
            capture=True,
            deadline_monotonic=self.deadline_monotonic,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"docker image ls failed on {self.machine.name}: {result.stderr or result.stdout}"
            )
        return _parse_runtime_host_images(
            self.machine,
            result.stdout,
            repositories=repositories,
        )

    def runtime_image_present(self, runtime_image_ref: str) -> bool:
        image = docker_image_ref(runtime_image_ref)
        inspected = _run_machine_docker(
            self.machine,
            ["image", "inspect", image],
            capture=True,
            deadline_monotonic=self.deadline_monotonic,
        )
        return inspected.returncode == 0

    def ensure_runtime_image(
        self,
        runtime_image_ref: str,
        *,
        timings: dict[str, float] | None = None,
    ) -> bool:
        inspect_started = time.perf_counter()
        if self.runtime_image_present(runtime_image_ref):
            if timings is not None:
                timings["host_image_inspect_seconds"] = time.perf_counter() - inspect_started
                timings["host_image_pull_seconds"] = 0.0
            return True
        if timings is not None:
            timings["host_image_inspect_seconds"] = time.perf_counter() - inspect_started
        image = docker_image_ref(runtime_image_ref)
        pull_started = time.perf_counter()
        if self.machine.pull_policy != "never":
            pulled = _run_machine_docker(
                self.machine,
                ["pull", image],
                capture=True,
                timeout=DOCKER_PULL_TIMEOUT_SECONDS,
                deadline_monotonic=self.deadline_monotonic,
            )
            if pulled.returncode != 0:
                if timings is not None:
                    timings["host_image_pull_seconds"] = time.perf_counter() - pull_started
                return False
        if timings is not None:
            timings["host_image_pull_seconds"] = time.perf_counter() - pull_started
        return self.runtime_image_present(runtime_image_ref)

    def validate_runtime_train_config(
        self,
        *,
        runtime_image_ref: str,
        train_config: Mapping[str, Any],
        expected_source_sha: str,
        expected_contract_sha256: str,
        expected_runtime_input_sha256: str = "",
    ) -> dict[str, Any]:
        timings: dict[str, float] = {}
        if not self.ensure_runtime_image(runtime_image_ref, timings=timings):
            raise RuntimeError(
                f"runtime preflight could not pull {runtime_image_ref} on {self.machine.name}"
            )
        validation_started = time.perf_counter()
        result = _run_machine_docker(
            self.machine,
            [
                "run",
                "--rm",
                "-i",
                docker_image_ref(runtime_image_ref),
                "python",
                "-m",
                "rlab.runtime_contract",
                "--validate-config-stdin",
            ],
            input_text=json.dumps(dict(train_config), sort_keys=True),
            capture=True,
            timeout=DOCKER_PULL_TIMEOUT_SECONDS,
            deadline_monotonic=self.deadline_monotonic,
        )
        timings["host_config_validation_seconds"] = time.perf_counter() - validation_started
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "runtime config validation failed").strip()
            raise RuntimeError(
                f"runtime preflight rejected the materialized train config on "
                f"{self.machine.name}: {detail}"
            )
        try:
            receipt = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"runtime preflight returned invalid JSON: {exc}") from exc
        expected = {
            "source_sha": expected_source_sha,
            "train_config_contract_sha256": expected_contract_sha256,
            "validated": True,
        }
        if expected_runtime_input_sha256:
            expected.update(
                {
                    "runtime_build_source_sha": expected_source_sha,
                    "runtime_input_sha256": expected_runtime_input_sha256,
                }
            )
        for key, value in expected.items():
            if receipt.get(key) != value:
                raise RuntimeError(
                    f"runtime preflight receipt mismatch for {key}: "
                    f"expected {value!r}, got {receipt.get(key)!r}"
                )
        receipt = dict(receipt)
        receipt["preflight_timings"] = timings
        return receipt

    def probe_runtime_image(
        self,
        *,
        runtime_image_ref: str,
        expected_source_sha: str,
        expected_runtime_input_sha256: str = "",
    ) -> dict[str, Any]:
        result = _run_machine_docker(
            self.machine,
            [
                "run",
                "--rm",
                docker_image_ref(runtime_image_ref),
                "python",
                "-m",
                "rlab.runtime_contract",
            ],
            capture=True,
            timeout=DOCKER_PULL_TIMEOUT_SECONDS,
            deadline_monotonic=self.deadline_monotonic,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "runtime probe failed").strip()
            raise RuntimeError(f"runtime probe failed on {self.machine.name}: {detail}")
        try:
            receipt = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"runtime probe returned invalid JSON: {exc}") from exc
        expected = {"source_sha": expected_source_sha}
        if expected_runtime_input_sha256:
            expected.update(
                {
                    "runtime_build_source_sha": expected_source_sha,
                    "runtime_input_sha256": expected_runtime_input_sha256,
                }
            )
        for key, value in expected.items():
            if receipt.get(key) != value:
                raise RuntimeError(
                    f"runtime probe receipt mismatch for {key}: "
                    f"expected {value!r}, got {receipt.get(key)!r}"
                )
        return dict(receipt)

    def create_train_container(
        self,
        *,
        launch_id: str,
        container_name: str,
        runtime_image_ref: str,
        labels: Mapping[str, str],
        attempt_env_path: str | None = None,
        rom_asset_manifest: Mapping[str, Any] | None = None,
    ) -> HostOperationResult:
        args = [
            "create",
            "--name",
            container_name,
            "--restart",
            "no",
            "--stop-timeout",
            str(TRAIN_CONTAINER_STOP_TIMEOUT_SECONDS),
            *self.machine.docker_gpu_args,
            "--env-file",
            attempt_env_path or self.machine.paths.env_file,
            "--mount",
            exact_bind_mount(
                self.payload_host_path(launch_id),
                self.payload_container_path(launch_id),
                readonly=True,
            ),
            "--mount",
            exact_bind_mount(
                self.output_host_path(launch_id),
                self.output_container_path(launch_id),
                readonly=False,
            ),
        ]
        if rom_asset_manifest is not None:
            manifest = validate_rom_asset_manifest(rom_asset_manifest, allow_legacy=True)
            host_digest_dir = (
                Path(self.machine.paths.rom_cache_dir) / "sha256" / manifest["sha256"]
            )
            container_digest_dir = (
                Path(self.machine.paths.container_rom_cache_dir)
                / "sha256"
                / manifest["sha256"]
            )
            args.extend(
                [
                    "--mount",
                    exact_bind_mount(
                        str(host_digest_dir), str(container_digest_dir), readonly=True
                    ),
                    "-e",
                    f"RLAB_ROM_CACHE_DIR={self.machine.paths.container_rom_cache_dir}",
                ]
            )
        for key, value in sorted(labels.items()):
            args.extend(["--label", f"{key}={value}"])
        args.extend(
            [
                docker_image_ref(runtime_image_ref),
                "rlab-container-entrypoint",
                "rlab",
                "run-job",
                "--payload",
                self.payload_container_path(launch_id),
                "--output-dir",
                self.output_container_path(launch_id),
            ]
        )
        result = _run_machine_docker(
            self.machine,
            args,
            capture=True,
            deadline_monotonic=self.deadline_monotonic,
        )
        return _operation_result(result)

    def ensure_rom_cache(
        self,
        *,
        launch_id: str,
        runtime_image_ref: str,
        attempt_env_path: str | None = None,
        rom_asset_manifest: Mapping[str, Any],
        container_name: str | None = None,
    ) -> HostOperationResult:
        manifest = validate_rom_asset_manifest(rom_asset_manifest, allow_legacy=True)
        host_digest_dir = Path(self.machine.paths.rom_cache_dir) / "sha256" / manifest["sha256"]
        container_digest_dir = (
            Path(self.machine.paths.container_rom_cache_dir) / "sha256" / manifest["sha256"]
        )
        prepare = _run_machine_shell(
            self.machine,
            f"mkdir -p {shlex.quote(str(host_digest_dir))}",
            capture=True,
            deadline_monotonic=self.deadline_monotonic,
        )
        if prepare.returncode != 0:
            return _operation_result(prepare)
        result = _run_machine_docker(
            self.machine,
            [
                "run",
                "--rm",
                "--name",
                container_name or f"rlab-rom-{launch_id}",
                "--env-file",
                attempt_env_path or self.machine.paths.env_file,
                "-e",
                f"RLAB_ROM_CACHE_DIR={self.machine.paths.container_rom_cache_dir}",
                "--mount",
                exact_bind_mount(
                    self.payload_host_path(launch_id),
                    self.payload_container_path(launch_id),
                    readonly=True,
                ),
                "--mount",
                exact_bind_mount(
                    str(host_digest_dir),
                    str(container_digest_dir),
                    readonly=False,
                ),
                docker_image_ref(runtime_image_ref),
                "python",
                "-m",
                "rlab.rom_cache",
                "--payload",
                self.payload_container_path(launch_id),
                "--cache-root",
                self.machine.paths.container_rom_cache_dir,
            ],
            capture=True,
            timeout=DOCKER_PULL_TIMEOUT_SECONDS,
            deadline_monotonic=self.deadline_monotonic,
        )
        return _operation_result(result)

    def start_container(self, container_name: str) -> HostOperationResult:
        result = _run_machine_docker(
            self.machine,
            ["start", container_name],
            capture=True,
            deadline_monotonic=self.deadline_monotonic,
        )
        return _operation_result(result)

    def stop_container(self, container_name: str, *, grace_seconds: int) -> HostOperationResult:
        result = _run_machine_docker(
            self.machine,
            ["stop", "--time", str(grace_seconds), container_name],
            capture=True,
            timeout=DOCKER_STOP_TIMEOUT_SECONDS,
            deadline_monotonic=self.deadline_monotonic,
        )
        return _operation_result(result)

    def remove_container(self, container_name: str, *, force: bool = False) -> HostOperationResult:
        args = ["rm"]
        if force:
            args.append("--force")
        args.append(container_name)
        result = _run_machine_docker(
            self.machine,
            args,
            capture=True,
            timeout=DOCKER_STOP_TIMEOUT_SECONDS if force else MACHINE_COMMAND_TIMEOUT_SECONDS,
            deadline_monotonic=self.deadline_monotonic,
        )
        return _operation_result(result)

    def container_is_absent(self, container_name: str) -> bool:
        result = _run_machine_docker(
            self.machine,
            ["container", "inspect", container_name],
            capture=True,
            deadline_monotonic=self.deadline_monotonic,
        )
        if result.returncode == 0:
            return False
        detail = str(result.stderr or result.stdout or "").lower()
        if "no such" in detail or "not found" in detail:
            return True
        raise RuntimeError(
            f"cannot prove container absence on {self.machine.name}: "
            f"{result.stderr or result.stdout}"
        )

    def attest_train_container(
        self,
        *,
        container_name: str,
        launch_id: str,
        runtime_image_ref: str,
    ) -> dict[str, Any]:
        result = _run_machine_docker(
            self.machine,
            ["container", "inspect", container_name],
            capture=True,
            deadline_monotonic=self.deadline_monotonic,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"cannot inspect workspace container {container_name}: "
                f"{result.stderr or result.stdout}"
            )
        try:
            documents = json.loads(result.stdout)
            document = documents[0]
        except (json.JSONDecodeError, IndexError, KeyError, TypeError) as exc:
            raise RuntimeError("docker inspect returned an invalid container document") from exc
        expected_mounts = {
            (
                os.path.normpath(self.payload_host_path(launch_id)),
                os.path.normpath(self.payload_container_path(launch_id)),
                False,
            ),
            (
                os.path.normpath(self.output_host_path(launch_id)),
                os.path.normpath(self.output_container_path(launch_id)),
                True,
            ),
        }
        all_bind_mounts = {
            (
                os.path.normpath(str(mount.get("Source") or "")),
                os.path.normpath(str(mount.get("Destination") or "")),
                bool(mount.get("RW")),
            )
            for mount in document.get("Mounts") or ()
            if str(mount.get("Type") or "") == "bind"
        }
        actual_mounts = {
            mount
            for mount in all_bind_mounts
            if mount[1]
            in {self.payload_container_path(launch_id), self.output_container_path(launch_id)}
        }
        if actual_mounts != expected_mounts:
            raise RuntimeError(
                f"workspace container exact mount attestation failed: {sorted(actual_mounts)!r}"
            )
        exclusive_roots = tuple(
            os.path.normpath(value)
            for value in (
                self.machine.paths.host_root,
                self.machine.paths.payloads_dir,
                self.machine.paths.outputs_dir,
                self.machine.paths.protected_metadata_dir,
            )
        )
        for source, destination, writable in all_bind_mounts:
            rom_source_prefix = os.path.normpath(
                f"{self.machine.paths.rom_cache_dir.rstrip('/')}/sha256"
            )
            rom_destination_prefix = os.path.normpath(
                f"{self.machine.paths.container_rom_cache_dir.rstrip('/')}/sha256"
            )
            source_digest = os.path.relpath(source, rom_source_prefix)
            destination_digest = os.path.relpath(destination, rom_destination_prefix)
            exact_rom = (
                not writable
                and source_digest == destination_digest
                and bool(re.fullmatch(r"[0-9a-f]{64}", source_digest))
            )
            overlaps_exclusive = any(
                source == root
                or source.startswith(f"{root}{os.sep}")
                or root.startswith(f"{source}{os.sep}")
                for root in exclusive_roots
            )
            if (
                overlaps_exclusive
                and (source, destination, writable) not in expected_mounts
                and not exact_rom
            ):
                raise RuntimeError(
                    f"workspace container has an unexpected broad/exclusive bind: {source}"
                )
        config = dict(document.get("Config") or {})
        if str(config.get("Image") or "") != docker_image_ref(runtime_image_ref):
            raise RuntimeError("workspace container image attestation failed")
        labels = dict(config.get("Labels") or {})
        if labels.get(LAUNCH_ID_LABEL) != launch_id or labels.get(MANAGED_LABEL) != "true":
            raise RuntimeError("workspace container ownership-label attestation failed")
        def command_parts(value: object) -> list[str]:
            if isinstance(value, str):
                return [value]
            if isinstance(value, Sequence):
                return [str(part) for part in value]
            return []

        command = [
            *command_parts(config.get("Entrypoint")),
            *command_parts(config.get("Cmd")),
        ]
        if "rlab-container-entrypoint" not in command or "run-job" not in command:
            raise RuntimeError("workspace container entrypoint attestation failed")
        return {
            "container_id": str(document.get("Id") or ""),
            "container_name": container_name,
            "runtime_image_ref": runtime_image_ref,
            "mounts": [
                {"source": source, "destination": destination, "writable": writable}
                for source, destination, writable in sorted(actual_mounts)
            ],
            "entrypoint": command,
            "attested_at": datetime.now(UTC).isoformat(),
        }

    def attest_recovery_container(
        self,
        *,
        container_name: str,
        launch_id: str,
        runtime_image_ref: str,
        member_kind: str,
    ) -> dict[str, Any]:
        result = _run_machine_docker(
            self.machine,
            ["container", "inspect", container_name],
            capture=True,
            deadline_monotonic=self.deadline_monotonic,
        )
        if result.returncode != 0:
            raise RuntimeError(f"cannot inspect recovery container {container_name}")
        try:
            document = json.loads(result.stdout)[0]
        except (json.JSONDecodeError, IndexError, TypeError) as exc:
            raise RuntimeError("recovery container inspect returned invalid JSON") from exc
        mounts = list(document.get("Mounts") or ())
        expected_source = os.path.normpath(self.output_host_path(launch_id))
        expected_destination = os.path.normpath(self.output_container_path(launch_id))
        if len(mounts) != 1:
            raise RuntimeError("recovery container must have exactly one mount")
        mount = mounts[0]
        if (
            mount.get("Type") != "bind"
            or os.path.normpath(str(mount.get("Source") or "")) != expected_source
            or os.path.normpath(str(mount.get("Destination") or ""))
            != expected_destination
            or not bool(mount.get("RW"))
        ):
            raise RuntimeError("recovery container exact output mount attestation failed")
        config = dict(document.get("Config") or {})
        if str(config.get("Image") or "") != docker_image_ref(runtime_image_ref):
            raise RuntimeError("recovery container image attestation failed")
        return {
            "container_id": str(document.get("Id") or ""),
            "container_name": container_name,
            "member_kind": member_kind,
            "runtime_image_ref": runtime_image_ref,
            "mount": {
                "source": expected_source,
                "destination": expected_destination,
                "writable": True,
            },
            "attested_at": datetime.now(UTC).isoformat(),
        }

    def remove_runtime_image(self, image_ref: str) -> HostOperationResult:
        result = _run_machine_docker(
            self.machine,
            ["rmi", image_ref],
            capture=True,
            deadline_monotonic=self.deadline_monotonic,
        )
        return _operation_result(result)

    def observe_result(self, output_uri: str) -> ResultObservation:
        return self._observe_json_file(output_uri, "result.json")

    def observe_readiness(self, output_uri: str) -> ResultObservation:
        return self._observe_json_file(output_uri, "readiness.json")

    def _observe_json_file(self, output_uri: str, filename: str) -> ResultObservation:
        result_path = f"{str(output_uri).rstrip('/')}/{filename}"
        script = (
            f"if [ -s {shlex.quote(result_path)} ]; then cat {shlex.quote(result_path)}; "
            f"elif [ -e {shlex.quote(result_path)} ]; then exit 3; else exit 4; fi"
        )
        try:
            result = _run_machine_shell(
                self.machine,
                script,
                capture=True,
                deadline_monotonic=self.deadline_monotonic,
            )
        except MachineCommandTimeout as exc:
            return ResultObservation("error", error=str(exc))
        if result.returncode == 4:
            return ResultObservation("absent")
        if result.returncode != 0:
            return ResultObservation(
                "error",
                error=(result.stderr or result.stdout or "").strip() or f"exit={result.returncode}",
            )
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            return ResultObservation("error", error=f"invalid {filename} JSON: {exc}")
        if not isinstance(payload, dict):
            return ResultObservation("error", error=f"{filename} JSON must be an object")
        return ResultObservation("present", payload=payload)

    def stream_logs(self, output_uri: str, *, tail: int = 100, follow: bool = False) -> int:
        log_dir = f"{str(output_uri).rstrip('/')}/logs"
        follow_flag = "-F" if follow else ""
        script = (
            f"file=$(find {shlex.quote(log_dir)} -maxdepth 1 -type f "
            "-name 'train_job_*.log' -print 2>/dev/null | sort | tail -n 1); "
            'test -n "$file"; '
            f'tail {follow_flag} -n {max(0, int(tail))} "$file"'
        )
        result = _run_machine_shell(
            self.machine,
            script,
            capture=False,
            timeout=None if follow else MACHINE_COMMAND_TIMEOUT_SECONDS,
            deadline_monotonic=self.deadline_monotonic,
        )
        return int(result.returncode)


def _workspace_helper_zipapp() -> tuple[str, str]:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "__main__.py",
            "from rlab.workspace_helper import main\nraise SystemExit(main())\n",
        )
        archive.writestr("rlab/__init__.py", "")
        for name in ("workspace_helper.py", "workspace_contract.py"):
            archive.writestr(
                f"rlab/{name}",
                Path(__file__).with_name(name).read_bytes(),
            )
    executable = b"#!/usr/bin/env python3\n" + payload.getvalue()
    return base64.b64encode(executable).decode("ascii"), hashlib.sha256(executable).hexdigest()


def _setup_host_script(machine: MachineConfig, *, runtime_image_ref: str | None = None) -> str:
    if machine.backend != "docker_ssh":
        raise ValueError(f"setup-host requires a docker_ssh machine, got {machine.name!r}")
    host_root = machine.paths.host_root.rstrip("/")
    runs_dir = f"{host_root}/runs"
    state_dir = f"{host_root}/fleet"
    docker_info = _shell_join(_machine_docker_command(machine, ["info"]))
    gpu_test = _shell_join(
        _machine_docker_command(
            machine,
            ["run", "--rm", *machine.docker_gpu_args, GPU_TEST_IMAGE, "nvidia-smi"],
        )
    )
    helper_base64, helper_sha256 = _workspace_helper_zipapp()
    lines = [
        "set -euo pipefail",
        f"mkdir -p {shlex.quote(machine.paths.host_root)}",
        f"mkdir -p {shlex.quote(runs_dir)} {shlex.quote(machine.paths.logs_dir)} "
        f"{shlex.quote(state_dir)} {shlex.quote(machine.paths.payloads_dir)} "
        f"{shlex.quote(machine.paths.outputs_dir)}",
        f"sudo -n install -d -o root -g root -m 0700 "
        f"{shlex.quote(machine.paths.protected_metadata_dir)}",
        "helper_tmp=$(mktemp)",
        "trap 'rm -f \"$helper_tmp\"' EXIT",
        f"printf %s {shlex.quote(helper_base64)} | base64 --decode > \"$helper_tmp\"",
        f"printf '%s  %s\\n' {shlex.quote(helper_sha256)} \"$helper_tmp\" | sha256sum -c -",
        f"sudo -n install -o root -g root -m 0755 \"$helper_tmp\" "
        f"{shlex.quote(SSH_WORKSPACE_HELPER_PATH)}",
        f"test \"$(sudo -n stat -c %U {shlex.quote(SSH_WORKSPACE_HELPER_PATH)})\" = root",
        f"test \"$(sudo -n stat -c %a {shlex.quote(SSH_WORKSPACE_HELPER_PATH)})\" = 755",
        f"sudo -n {shlex.quote(SSH_WORKSPACE_HELPER_PATH)} --help >/dev/null",
        f"if ! mkdir -p {shlex.quote(machine.paths.rom_cache_dir)} 2>/dev/null; then",
        f"  sudo -n install -d -o \"$(id -u)\" -g \"$(id -g)\" "
        f"{shlex.quote(machine.paths.rom_cache_dir)}",
        "fi",
        "if ! command -v docker >/dev/null 2>&1; then",
        "  if command -v apt-get >/dev/null 2>&1; then",
        "    sudo -n apt-get update",
        "    sudo -n apt-get install -y docker.io",
        "  else",
        "    echo 'docker is missing and apt-get is unavailable' >&2",
        "    exit 1",
        "  fi",
        "fi",
        "sudo -n systemctl enable --now docker >/dev/null 2>&1 || true",
        f"{docker_info} >/dev/null",
        "if ! command -v nvidia-smi >/dev/null 2>&1; then",
        "  echo 'warning: nvidia-smi is not on PATH' >&2",
        "else",
        "  nvidia-smi >/dev/null",
        "fi",
        "if ! command -v nvidia-ctk >/dev/null 2>&1; then",
        "  if command -v apt-get >/dev/null 2>&1; then",
        "    sudo -n apt-get install -y --no-install-recommends ca-certificates curl gnupg2",
        "    sudo -n install -d -m 0755 /usr/share/keyrings",
        "    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | "
        "sudo -n gpg --batch --yes --dearmor "
        "-o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg",
        "    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/"
        "nvidia-container-toolkit.list | sed 's#deb https://#deb "
        "[signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | "
        "sudo -n tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null",
        "    sudo -n apt-get update",
        "    sudo -n apt-get install -y nvidia-container-toolkit",
        "  else",
        "    echo 'nvidia-ctk is missing and apt-get is unavailable' >&2",
        "    exit 1",
        "  fi",
        "fi",
        "if command -v nvidia-ctk >/dev/null 2>&1; then",
        "  sudo -n nvidia-ctk runtime configure --runtime=docker",
        "  sudo -n systemctl restart docker || true",
        "fi",
        f"if ! {gpu_test} >/dev/null; then",
        f"  {gpu_test} >/dev/null",
        "fi",
        f"if [ ! -f {shlex.quote(machine.paths.env_file)} ]; then",
        f"  umask 077; cat > {shlex.quote(machine.paths.env_file)} <<'EOF'",
        "# rlab job-container secrets live here; fill values on the host.",
        "TRAIN_QUEUE_DATABASE_URL=",
        "WORKER_MAILBOX_DATABASE_URL=",
        "WANDB_API_KEY=",
        "AWS_ACCESS_KEY_ID=",
        "AWS_SECRET_ACCESS_KEY=",
        "AWS_S3_ENDPOINT_URL=",
        "AWS_REGION=",
        "CHECKPOINT_BUCKET_URI=",
        "EOF",
        "fi",
        f"test -f {shlex.quote(machine.paths.env_file)}",
    ]
    if runtime_image_ref:
        image = docker_image_ref(runtime_image_ref)
        lines.extend(
            [
                _shell_join(_machine_docker_command(machine, ["pull", image])),
                _shell_join(
                    _machine_docker_command(
                        machine,
                        [
                            "run",
                            "--rm",
                            *machine.docker_gpu_args,
                            image,
                            "rlab-container-entrypoint",
                            "rlab-container-smoke",
                        ],
                    )
                ),
            ]
        )
    return "\n".join(lines) + "\n"


def setup_docker_host(
    machine: MachineConfig,
    runtime_image_ref: str | None = None,
    *,
    execute: bool,
) -> tuple[str, int | None]:
    script = _setup_host_script(machine, runtime_image_ref=runtime_image_ref)
    if not execute:
        return script, None
    timeout = DOCKER_PULL_TIMEOUT_SECONDS if runtime_image_ref else MACHINE_COMMAND_TIMEOUT_SECONDS
    result = _run_machine_shell(machine, script, timeout=timeout)
    return script, int(result.returncode)


def run_checkpoint_coordinator_container(
    host: DockerRunnerHost,
    *,
    launch_id: str,
    run_name: str,
    runtime_image_ref: str,
) -> None:
    container_output = host.output_container_path(launch_id)
    result = _run_machine_docker(
        host.machine,
        [
            "run",
            "--rm",
            "--name",
            f"rlab-checkpoint-recovery-{launch_id}",
            "--env-file",
            host.machine.paths.env_file,
            "--mount",
            exact_bind_mount(
                host.output_host_path(launch_id),
                host.output_container_path(launch_id),
                readonly=False,
            ),
            docker_image_ref(runtime_image_ref),
            "python",
            "-m",
            "rlab.checkpoint_coordinator",
            "--run-dir",
            f"{container_output}/runs/{run_name}",
            "--train-config-json",
            f"{container_output}/train_config.json",
            "--drain-and-exit",
        ],
        capture=True,
        timeout=900,
        deadline_monotonic=host.deadline_monotonic,
    )
    if result.returncode:
        raise RuntimeError(result.stderr or result.stdout)
    container_name = f"rlab-checkpoint-recovery-{launch_id}"
    if not host.container_is_absent(container_name):
        raise RuntimeError("checkpoint recovery container object remains after exit")


def run_wandb_publisher_recovery_container(
    host: DockerRunnerHost,
    *,
    launch_id: str,
    run_name: str,
    runtime_image_ref: str,
    attempt_env_path: str | None = None,
    on_attested: Callable[[Mapping[str, Any]], None] | None = None,
    on_absent: Callable[[Mapping[str, Any]], None] | None = None,
    container_name: str | None = None,
) -> None:
    """Drain the durable SQLite outbox in a CPU-only container on the launch host."""

    container_output = host.output_container_path(launch_id)
    container_name = container_name or f"rlab-wandb-recovery-{launch_id}"
    created = _run_machine_docker(
        host.machine,
        [
            "create",
            "--name",
            container_name,
            "--env-file",
            attempt_env_path or host.machine.paths.env_file,
            "--mount",
            exact_bind_mount(
                host.output_host_path(launch_id),
                host.output_container_path(launch_id),
                readonly=False,
            ),
            docker_image_ref(runtime_image_ref),
            "timeout",
            "--signal=TERM",
            "--kill-after=5s",
            "105s",
            "python",
            "-m",
            "rlab.wandb_publisher",
            "--run-dir",
            f"{container_output}/runs/{run_name}",
            "--train-config-json",
            f"{container_output}/train_config.json",
            "--stop-file",
            f"{container_output}/publisher.stop",
        ],
        capture=True,
        timeout=120,
        deadline_monotonic=host.deadline_monotonic,
    )
    if created.returncode:
        raise RuntimeError(created.stderr or created.stdout)
    attestation = host.attest_recovery_container(
        container_name=container_name,
        launch_id=launch_id,
        runtime_image_ref=runtime_image_ref,
        member_kind="wandb",
    )
    if on_attested:
        on_attested(attestation)
    error: str | None = None
    try:
        result = _run_machine_docker(
            host.machine,
            ["start", "--attach", container_name],
            capture=True,
            timeout=120,
            deadline_monotonic=host.deadline_monotonic,
        )
        if result.returncode:
            error = str(result.stderr or result.stdout)
    finally:
        removed = host.remove_container(container_name, force=True)
        if not removed.ok and not host.container_is_absent(container_name):
            raise RuntimeError(removed.detail or "W&B recovery container removal failed")
        if not host.container_is_absent(container_name):
            raise RuntimeError("W&B recovery container remains after removal")
        if on_absent:
            on_absent(attestation)
    if error:
        raise RuntimeError(error)
