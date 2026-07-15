from __future__ import annotations

import json
import shlex
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rlab.fleet_labels import (
    DEFAULT_RUNTIME_IMAGE_REPOSITORIES,
    JOB_CONTAINER_LABEL,
    JOB_KIND_LABEL,
    LABEL_PREFIX,
    LAUNCH_ID_LABEL,
)
from rlab.json_utils import json_safe
from rlab.machines import MachineConfig
from rlab.runtime_refs import docker_image_ref, normalize_runtime_image_ref


GPU_TEST_IMAGE = "nvidia/cuda:12.9.1-base-ubuntu22.04"
SSH_CONNECT_TIMEOUT_SECONDS = 10
MACHINE_COMMAND_TIMEOUT_SECONDS = 120.0
DOCKER_PULL_TIMEOUT_SECONDS = 900.0
DOCKER_STOP_TIMEOUT_SECONDS = 150.0


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
    command = (
        ["sh", "-lc", script]
        if machine.backend == "local_docker"
        else [*_machine_ssh_prefix(machine), "sh", "-lc", shlex.quote(script)]
    )
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
    ) -> HostOperationResult:
        args = [
            "create",
            "--name",
            container_name,
            "--restart",
            "no",
            *self.machine.docker_gpu_args,
            "--env-file",
            attempt_env_path or self.machine.paths.env_file,
            "-v",
            f"{self.machine.paths.payloads_dir}:{self.machine.paths.container_payloads_dir}:ro",
            "-v",
            f"{self.machine.paths.outputs_dir}:{self.machine.paths.container_outputs_dir}",
            "-v",
            f"{self.machine.paths.roms_dir}:{self.machine.paths.container_roms_dir}:ro",
            "-e",
            f"RLAB_ROM_DIR={self.machine.paths.container_roms_dir}",
        ]
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
    lines = [
        "set -euo pipefail",
        f"mkdir -p {shlex.quote(machine.paths.host_root)}",
        f"mkdir -p {shlex.quote(runs_dir)} {shlex.quote(machine.paths.logs_dir)} "
        f"{shlex.quote(machine.paths.roms_dir)} {shlex.quote(state_dir)}",
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
                            "--env-file",
                            machine.paths.env_file,
                            "-e",
                            f"RLAB_ROM_DIR={machine.paths.container_roms_dir}",
                            "-v",
                            f"{machine.paths.roms_dir}:{machine.paths.container_roms_dir}:ro",
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
            "--env-file",
            host.machine.paths.env_file,
            "-e",
            "RLAB_IMPORT_ROMS=0",
            "-v",
            f"{host.machine.paths.outputs_dir}:{host.machine.paths.container_outputs_dir}",
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


def run_wandb_publisher_recovery_container(
    host: DockerRunnerHost,
    *,
    launch_id: str,
    run_name: str,
    runtime_image_ref: str,
) -> None:
    """Drain the durable SQLite outbox in a CPU-only container on the launch host."""

    container_output = host.output_container_path(launch_id)
    result = _run_machine_docker(
        host.machine,
        [
            "run",
            "--rm",
            "--name",
            f"rlab-wandb-recovery-{launch_id}",
            "--env-file",
            host.machine.paths.env_file,
            "-e",
            "RLAB_IMPORT_ROMS=0",
            "-v",
            f"{host.machine.paths.outputs_dir}:{host.machine.paths.container_outputs_dir}",
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
    if result.returncode:
        raise RuntimeError(result.stderr or result.stdout)
