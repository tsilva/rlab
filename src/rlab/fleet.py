from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import shlex
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from rlab.cli_args import add_direct_database_arg, add_dry_run_arg
from rlab.job_queue import (
    QueueDemand,
    TRAIN_JOB_KIND,
    active_job_launches,
    claim_job_launch,
    connect,
    database_url,
    finish_job_launch_from_result,
    job_payload_for_launch,
    machine_control,
    mark_job_launch_running,
    new_train_launch_id,
    next_pending_train_job,
    queue_demands,
    record_job_launch_error,
    set_machine_control,
)
from rlab.json_utils import json_safe
from rlab.machines import (
    DEFAULT_MACHINE_REGISTRY,
    MachineConfig,
    MachineRegistry,
    load_machine_registry,
    resolve_machine,
)
from rlab.runtime_refs import (
    docker_image_ref,
    normalize_runtime_image_ref,
    runtime_image_ref_from_args,
)
from rlab.fleet_labels import (
    JOB_CONTAINER_LABEL,
    JOB_ID_LABEL,
    JOB_KIND_LABEL,
    DEFAULT_RUNTIME_IMAGE_REPOSITORIES,
    LABEL_PREFIX,
    LAUNCH_ID_LABEL,
    MACHINE_LABEL,
    MANAGED_LABEL,
    OUTPUT_URI_LABEL,
)
DEFAULT_SHARED_RUNNER_ENV_FILE = Path(".env")
GPU_TEST_IMAGE = "nvidia/cuda:12.9.1-base-ubuntu22.04"
SSH_CONNECT_TIMEOUT_SECONDS = 10
MACHINE_COMMAND_TIMEOUT_SECONDS = 120.0
DOCKER_PULL_TIMEOUT_SECONDS = 900.0
DOCKER_STOP_TIMEOUT_SECONDS = 150.0
SHARED_RUNNER_ENV_KEYS = (
    "WANDB_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_S3_ENDPOINT_URL",
    "AWS_REGION",
    "CHECKPOINT_BUCKET_URI",
)
_MACHINE_LANE_DEADLINE: ContextVar[float | None] = ContextVar(
    "rlab_machine_lane_deadline", default=None
)


@dataclass(frozen=True)
class MachineMutationLock:
    machine: str
    key: str


class MachineLockBusy(RuntimeError):
    def __init__(self, machine: str) -> None:
        super().__init__(f"another reconciler is already running for machine={machine}")
        self.machine = machine


class MachineCommandTimeout(RuntimeError):
    def __init__(self, machine: str, timeout: float) -> None:
        super().__init__(f"machine command timed out machine={machine} timeout={timeout:g}s")
        self.machine = machine
        self.timeout = timeout


def _candidate_repo_roots(start: Path) -> tuple[Path, ...]:
    base = start if start.is_dir() else start.parent
    return (base, *base.parents)


def _is_fleet_repo_root(path: Path) -> bool:
    return (path / DEFAULT_MACHINE_REGISTRY).is_file()


def default_repo_root() -> Path:
    for start in (Path.cwd(), Path(__file__).resolve()):
        for candidate in _candidate_repo_roots(start):
            if _is_fleet_repo_root(candidate):
                return candidate.resolve()
    return Path.cwd().resolve()


def sanitize_slug(value: str, *, limit: int = 40) -> str:
    chars = []
    for char in value.lower():
        if char.isalnum():
            chars.append(char)
        elif chars and chars[-1] != "-":
            chars.append("-")
    slug = "".join(chars).strip("-") or "value"
    return slug[:limit].strip("-") or "value"


def shell_join(parts: Sequence[str]) -> str:
    return shlex.join([str(part) for part in parts])


def setup_host_script(machine: MachineConfig, *, runtime_image_ref: str | None = None) -> str:
    if machine.backend != "docker_ssh":
        raise ValueError(f"setup-host requires a docker_ssh machine, got {machine.name!r}")
    host_root = machine.paths.host_root.rstrip("/")
    runs_dir = f"{host_root}/runs"
    state_dir = f"{host_root}/fleet"
    docker_info = shell_join(machine_docker_command(machine, ["info"]))
    gpu_test = shell_join(
        machine_docker_command(machine, ["run", "--rm", *machine.docker_gpu_args, GPU_TEST_IMAGE, "nvidia-smi"])
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
                shell_join(machine_docker_command(machine, ["pull", image])),
                shell_join(
                    machine_docker_command(
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


def _connect_from_args(args: argparse.Namespace):
    return connect(database_url(getattr(args, "direct", False)))


def repo_root_from_args(args: argparse.Namespace) -> Path:
    repo_root = getattr(args, "repo_root", None)
    if repo_root:
        return Path(repo_root).expanduser().resolve()
    return default_repo_root()


def shared_runner_env_file_from_args(args: argparse.Namespace) -> Path:
    return repo_root_from_args(args) / DEFAULT_SHARED_RUNNER_ENV_FILE


def load_shared_runner_env(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise RuntimeError(f"shared runner env file is missing: {path}")
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if key not in SHARED_RUNNER_ENV_KEYS:
            continue
        if key in values:
            raise RuntimeError(f"shared runner env file defines {key} more than once: {path}")
        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if not value:
            raise RuntimeError(f"shared runner env value is empty: {key} in {path}")
        if "\x00" in value or "\n" in value or "\r" in value:
            raise RuntimeError(f"shared runner env value contains an invalid control character: {key}")
        values[key] = value
    missing = [key for key in SHARED_RUNNER_ENV_KEYS if key not in values]
    if missing:
        raise RuntimeError(
            f"shared runner env file is missing required key(s): {', '.join(missing)} in {path}"
        )
    return values


def format_shared_runner_env(values: Mapping[str, str]) -> str:
    return "".join(f"{key}={values[key]}\n" for key in SHARED_RUNNER_ENV_KEYS)


def shared_runner_env_sync_script(machine: MachineConfig) -> str:
    if machine.backend != "docker_ssh":
        raise ValueError(f"shared runner env sync requires docker_ssh, got {machine.name!r}")
    env_file = shlex.quote(machine.paths.env_file)
    shared_keys = " ".join(SHARED_RUNNER_ENV_KEYS)
    awk_program = (
        f'BEGIN {{ split("{shared_keys}", keys, " "); '
        "for (i in keys) shared[keys[i]] = 1 } !($1 in shared)"
    )
    verify_keys = " ".join(shlex.quote(key) for key in SHARED_RUNNER_ENV_KEYS)
    return "\n".join(
        [
            "set -eu",
            "shared_tmp=$(mktemp)",
            "merged_tmp=$(mktemp)",
            "trap 'rm -f \"$shared_tmp\" \"$merged_tmp\"' EXIT",
            "umask 077",
            'cat > "$shared_tmp"',
            f"if [ -f {env_file} ]; then",
            f"  awk -F= {shlex.quote(awk_program)} {env_file} > \"$merged_tmp\"",
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


def sync_shared_runner_env(
    machine: MachineConfig,
    source_path: Path,
) -> None:
    if machine.backend != "docker_ssh":
        return
    values = load_shared_runner_env(source_path)
    result = run_machine_shell(
        machine,
        shared_runner_env_sync_script(machine),
        input_text=format_shared_runner_env(values),
        capture=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"failed to sync shared runner env to machine={machine.name} exit={result.returncode}"
        )


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


def load_registry_from_args(args: argparse.Namespace) -> MachineRegistry:
    return load_machine_registry(args.machines)


def sanitize_container_part(value: str, *, limit: int = 32) -> str:
    return sanitize_slug(value, limit=limit)


def job_container_name(machine: MachineConfig, *, launch_id: str) -> str:
    return (
        f"rlab-job-{sanitize_container_part(machine.name, limit=16)}-"
        f"{TRAIN_JOB_KIND}-"
        f"{sanitize_container_part(launch_id, limit=48)}"
    )[:120].strip("-")


def launch_payload_path(machine: MachineConfig, launch_id: str) -> str:
    return f"{machine.paths.payloads_dir.rstrip('/')}/{launch_id}.json"


def launch_output_path(machine: MachineConfig, launch_id: str) -> str:
    return f"{machine.paths.outputs_dir.rstrip('/')}/{launch_id}"


def container_payload_path(machine: MachineConfig, launch_id: str) -> str:
    return f"{machine.paths.container_payloads_dir.rstrip('/')}/{launch_id}.json"


def container_output_path(machine: MachineConfig, launch_id: str) -> str:
    return f"{machine.paths.container_outputs_dir.rstrip('/')}/{launch_id}"


def machine_docker_command(machine: MachineConfig, args: Sequence[str]) -> list[str]:
    return [*machine.docker_command, *args]


def job_container_create_command(
    machine: MachineConfig,
    *,
    job_id: int,
    launch_id: str,
    runtime_image_ref: str,
    container_name: str,
) -> list[str]:
    image = docker_image_ref(runtime_image_ref)
    cmd = machine_docker_command(
        machine,
        [
            "create",
            "--name",
            container_name,
            "--restart",
            "no",
            *machine.docker_gpu_args,
            "--env-file",
            machine.paths.env_file,
            "-v",
            f"{machine.paths.payloads_dir}:{machine.paths.container_payloads_dir}:ro",
            "-v",
            f"{machine.paths.outputs_dir}:{machine.paths.container_outputs_dir}",
            "-v",
            f"{machine.paths.roms_dir}:{machine.paths.container_roms_dir}:ro",
            "-e",
            f"RLAB_ROM_DIR={machine.paths.container_roms_dir}",
        ],
    )
    labels = {
        MANAGED_LABEL: "true",
        JOB_CONTAINER_LABEL: "true",
        MACHINE_LABEL: machine.name,
        JOB_KIND_LABEL: TRAIN_JOB_KIND,
        JOB_ID_LABEL: str(job_id),
        LAUNCH_ID_LABEL: launch_id,
        OUTPUT_URI_LABEL: launch_output_path(machine, launch_id),
        f"{LABEL_PREFIX}runtime-image-ref": runtime_image_ref,
    }
    for key, value in sorted(labels.items()):
        cmd.extend(["--label", f"{key}={value}"])
    cmd.extend(
        [
            image,
            "rlab-container-entrypoint",
            "rlab",
            "run-job",
            "--payload",
            container_payload_path(machine, launch_id),
            "--output-dir",
            container_output_path(machine, launch_id),
        ]
    )
    return cmd


def machine_ssh_prefix(machine: MachineConfig) -> list[str]:
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


def run_machine_shell(
    machine: MachineConfig,
    script: str,
    *,
    input_text: str | None = None,
    capture: bool = False,
    timeout: float | None = MACHINE_COMMAND_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[str]:
    effective_timeout = timeout
    lane_deadline = _MACHINE_LANE_DEADLINE.get()
    if timeout is not None and lane_deadline is not None:
        remaining = lane_deadline - time.monotonic()
        if remaining <= 0:
            raise MachineCommandTimeout(machine.name, 0.0)
        effective_timeout = min(float(timeout), max(0.1, remaining))
    command = (
        ["sh", "-lc", script]
        if machine.backend == "local_docker"
        else [*machine_ssh_prefix(machine), "sh", "-lc", shlex.quote(script)]
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


def run_machine_docker(
    machine: MachineConfig,
    docker_args: Sequence[str],
    *,
    capture: bool = False,
    timeout: float | None = MACHINE_COMMAND_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[str]:
    command = shell_join(machine_docker_command(machine, docker_args))
    return run_machine_shell(machine, command, capture=capture, timeout=timeout)


def write_remote_payload(machine: MachineConfig, path: str, payload: Mapping[str, Any]) -> None:
    payload_text = json.dumps(json_safe(dict(payload)), indent=2, sort_keys=True) + "\n"
    script = f"mkdir -p {shlex.quote(str(Path(path).parent))} && cat > {shlex.quote(path)}"
    result = run_machine_shell(machine, script, input_text=payload_text, capture=True)
    if result.returncode != 0:
        raise RuntimeError(f"failed to write payload {path}: {result.stderr or result.stdout}")


def parse_docker_labels(value: str) -> dict[str, str]:
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
        labels = parse_docker_labels(str(row.get("Labels") or ""))
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


def parse_job_containers(machine: MachineConfig, output: str) -> list[JobContainer]:
    return _parse_containers(
        machine,
        output,
        required_label=JOB_CONTAINER_LABEL,
        required_value="true",
    )


def _list_containers(
    machine: MachineConfig,
    *,
    required_label: str,
    required_value: str | None = None,
) -> list[JobContainer]:
    label_filter = (
        f"label={required_label}={required_value}"
        if required_value is not None
        else f"label={required_label}"
    )
    result = run_machine_docker(
        machine,
        [
            "ps",
            "-a",
            "--filter",
            label_filter,
            "--format",
            "{{json .}}",
        ],
        capture=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"docker ps failed on {machine.name}: {result.stderr or result.stdout}")
    return _parse_containers(
        machine,
        result.stdout,
        required_label=required_label,
        required_value=required_value,
    )


def list_job_containers(machine: MachineConfig) -> list[JobContainer]:
    return _list_containers(
        machine,
        required_label=JOB_CONTAINER_LABEL,
        required_value="true",
    )


def list_runtime_image_containers(machine: MachineConfig) -> list[JobContainer]:
    return _list_containers(
        machine,
        required_label=f"{LABEL_PREFIX}runtime-image-ref",
    )


def parse_runtime_host_images(
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


def runtime_image_repository(runtime_image_ref: str) -> str | None:
    try:
        image = docker_image_ref(runtime_image_ref)
    except ValueError:
        return None
    if "@" not in image:
        return None
    repository, _ = image.split("@", 1)
    return repository or None


def protected_runtime_image_refs(
    *,
    machine: MachineConfig,
    demands: Sequence[QueueDemand],
    containers: Sequence[JobContainer],
) -> set[str]:
    protected: set[str] = set()
    for demand in demands:
        if demand.total <= 0:
            continue
        if demand.machine != machine.name:
            continue
        protected.add(normalize_runtime_image_ref(demand.runtime_image_ref))
    for container in containers:
        if container.state in {"removing", "dead"}:
            continue
        runtime_image_ref = container.labels.get(f"{LABEL_PREFIX}runtime-image-ref")
        if not runtime_image_ref:
            continue
        try:
            protected.add(normalize_runtime_image_ref(runtime_image_ref))
        except ValueError:
            continue
    return protected


def repositories_for_runtime_images(protected_refs: set[str]) -> tuple[str, ...]:
    repositories = set(DEFAULT_RUNTIME_IMAGE_REPOSITORIES)
    for runtime_image_ref in protected_refs:
        repository = runtime_image_repository(runtime_image_ref)
        if repository:
            repositories.add(repository)
    return tuple(sorted(repositories))


def list_runtime_host_images(
    machine: MachineConfig,
    *,
    repositories: Sequence[str],
) -> tuple[RuntimeHostImage, ...]:
    result = run_machine_docker(
        machine,
        ["image", "ls", "--digests", "--format", "{{json .}}"],
        capture=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"docker image ls failed on {machine.name}: {result.stderr or result.stdout}")
    return parse_runtime_host_images(machine, result.stdout, repositories=repositories)


def stale_runtime_host_images(
    *,
    machine: MachineConfig,
    images: Sequence[RuntimeHostImage],
    demands: Sequence[QueueDemand],
    containers: Sequence[JobContainer],
) -> tuple[RuntimeHostImage, ...]:
    protected = protected_runtime_image_refs(
        machine=machine,
        demands=demands,
        containers=containers,
    )
    return tuple(image for image in images if image.runtime_image_ref not in protected)


def prune_stale_runtime_images(
    conn,
    machine: MachineConfig,
) -> int:
    demands = queue_demands(conn)
    containers = list_runtime_image_containers(machine)
    protected = protected_runtime_image_refs(
        machine=machine,
        demands=demands,
        containers=containers,
    )
    images = list_runtime_host_images(
        machine,
        repositories=repositories_for_runtime_images(protected),
    )
    stale_images = tuple(image for image in images if image.runtime_image_ref not in protected)
    pruned = 0
    for image in stale_images:
        result = run_machine_docker(machine, ["rmi", image.image_ref], capture=True)
        if result.returncode == 0:
            pruned += 1
    return pruned


def prune_inactive_job_containers(conn, machine: MachineConfig) -> int:
    active_launch_ids = {
        str(launch["launch_id"])
        for launch in active_job_launches(conn, machine=machine.name)
    }
    removed = 0
    for container in list_job_containers(machine):
        if container.state not in {"exited", "dead"}:
            continue
        if container.launch_id and container.launch_id in active_launch_ids:
            continue
        result = run_machine_docker(machine, ["rm", container.name], capture=True)
        if result.returncode == 0:
            removed += 1
    return removed


@dataclass(frozen=True)
class ResultObservation:
    state: str
    payload: dict[str, Any] | None = None
    error: str | None = None


def observe_remote_result(machine: MachineConfig, output_uri: str) -> ResultObservation:
    result_path = f"{str(output_uri).rstrip('/')}/result.json"
    script = (
        f"if [ -s {shlex.quote(result_path)} ]; then cat {shlex.quote(result_path)}; "
        f"elif [ -e {shlex.quote(result_path)} ]; then exit 3; else exit 4; fi"
    )
    try:
        result = run_machine_shell(machine, script, capture=True)
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
        return ResultObservation("error", error=f"invalid result JSON: {exc}")
    if not isinstance(payload, dict):
        return ResultObservation("error", error="result JSON must be an object")
    return ResultObservation("present", payload=payload)


def stream_job_logs(
    machine: MachineConfig,
    output_uri: str,
    *,
    tail: int = 100,
    follow: bool = False,
) -> int:
    log_dir = f"{str(output_uri).rstrip('/')}/logs"
    follow_flag = "-F" if follow else ""
    script = (
        f"file=$(find {shlex.quote(log_dir)} -maxdepth 1 -type f "
        "-name 'train_job_*.log' -print 2>/dev/null | sort | tail -n 1); "
        "test -n \"$file\"; "
        f"tail {follow_flag} -n {max(0, int(tail))} \"$file\""
    )
    result = run_machine_shell(
        machine,
        script,
        capture=False,
        timeout=None if follow else MACHINE_COMMAND_TIMEOUT_SECONDS,
    )
    return int(result.returncode)


def ensure_runtime_image_available(machine: MachineConfig, runtime_image_ref: str) -> bool:
    image = docker_image_ref(runtime_image_ref)
    if machine.pull_policy != "never":
        pulled = run_machine_docker(
            machine,
            ["pull", image],
            capture=True,
            timeout=DOCKER_PULL_TIMEOUT_SECONDS,
        )
        if pulled.returncode != 0:
            return False
    inspected = run_machine_docker(
        machine,
        ["image", "inspect", image],
        capture=True,
    )
    return inspected.returncode == 0


def _load_train_job(conn, job_id: int) -> dict[str, Any]:
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM train_jobs WHERE id = %(job_id)s", {"job_id": int(job_id)})
        row = cur.fetchone()
    if not row:
        raise RuntimeError(f"train job not found: {job_id}")
    return dict(row)


def _terminal_result(
    launch: Mapping[str, Any],
    *,
    status: str,
    error: str,
    exit_code: int | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "job_kind": launch["job_kind"],
        "job_id": int(launch["job_id"]),
        "launch_id": str(launch["launch_id"]),
        "machine": str(launch["machine"]),
        "runtime_image_ref": str(launch["runtime_image_ref"]),
        "status": status,
        "exit_code": exit_code,
        "error": error,
    }


def _record_launch_error(conn, launch_id: str, error: str) -> None:
    record_job_launch_error(conn, launch_id=launch_id, error=error, retry_after_seconds=30.0)


def _start_or_resume_launch(
    conn,
    machine: MachineConfig,
    *,
    launch: Mapping[str, Any],
    known_container: JobContainer | None = None,
    image_ready: bool = False,
) -> bool:
    launch_id = str(launch["launch_id"])
    container_name = str(launch["container_name"])
    job = _load_train_job(conn, int(launch["job_id"]))
    try:
        if not image_ready and not ensure_runtime_image_available(
            machine, str(launch["runtime_image_ref"])
        ):
            _record_launch_error(conn, launch_id, "runtime image is unavailable")
            return False
        write_remote_payload(
            machine,
            launch_payload_path(machine, launch_id),
            job_payload_for_launch(job, launch),
        )
        container = known_container
        if container is None:
            create_command = job_container_create_command(
                machine,
                job_id=int(job["id"]),
                launch_id=launch_id,
                runtime_image_ref=str(launch["runtime_image_ref"]),
                container_name=container_name,
            )
            created = run_machine_shell(
                machine,
                shell_join(create_command),
                capture=True,
                timeout=MACHINE_COMMAND_TIMEOUT_SECONDS,
            )
            containers = {item.launch_id: item for item in list_job_containers(machine)}
            container = containers.get(launch_id)
            if container is None:
                detail = (created.stderr or created.stdout or "").strip()
                _record_launch_error(
                    conn,
                    launch_id,
                    detail or f"docker create failed exit={created.returncode}",
                )
                return False
        if container.state != "running":
            started = run_machine_docker(machine, ["start", container_name], capture=True)
            containers = {item.launch_id: item for item in list_job_containers(machine)}
            container = containers.get(launch_id)
            if container is None or container.state != "running":
                detail = (started.stderr or started.stdout or "").strip()
                _record_launch_error(
                    conn,
                    launch_id,
                    detail or f"docker start failed exit={started.returncode}",
                )
                return False
        mark_job_launch_running(
            conn,
            launch_id=launch_id,
            container_name=container_name,
            provider_run_id=container_name,
        )
        return True
    except Exception as exc:
        _record_launch_error(conn, launch_id, str(exc))
        return False


def launch_cancel_requested(conn, launch: Mapping[str, Any]) -> bool:
    if str(launch.get("job_kind") or "") != "train":
        return False
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT cancel_requested
            FROM train_jobs
            WHERE id = %(job_id)s
            """,
            {"job_id": launch["job_id"]},
        )
        row = cur.fetchone()
    return bool(row and row.get("cancel_requested"))


def cancel_running_job_launch(
    conn,
    machine: MachineConfig,
    *,
    launch: Mapping[str, Any],
    container: JobContainer,
) -> bool:
    launch_id = str(launch["launch_id"])
    docker_args = (
        ["rm", "--force", container.name]
        if container.state == "created"
        else ["stop", "--time", "120", container.name]
    )
    result = run_machine_docker(
        machine,
        docker_args,
        capture=True,
        timeout=DOCKER_STOP_TIMEOUT_SECONDS,
    )
    if result.returncode != 0:
        _record_launch_error(
            conn,
            launch_id,
            (result.stderr or result.stdout or "").strip() or f"docker stop exit={result.returncode}",
        )
        return False
    finish_job_launch_from_result(
        conn,
        launch_id=launch_id,
        result=_terminal_result(
            launch,
            status="canceled",
            exit_code=130,
            error="cancel requested",
        ),
    )
    return True


def reconcile_machine_launches(conn, machine: MachineConfig) -> int:
    launches = active_job_launches(conn, machine=machine.name)
    containers = {container.launch_id: container for container in list_job_containers(machine)}
    reconciled = 0
    for launch in launches:
        launch_id = str(launch["launch_id"])
        retry_at = launch.get("next_retry_at")
        if retry_at is not None and retry_at > datetime.now(UTC):
            continue
        container = containers.get(launch_id)
        if container is None:
            observation = observe_remote_result(machine, str(launch["output_uri"]))
            if observation.state == "present":
                finish_job_launch_from_result(conn, launch_id=launch_id, result=observation.payload or {})
                reconciled += 1
            elif observation.state == "error":
                _record_launch_error(conn, launch_id, observation.error or "result observation failed")
            elif launch_cancel_requested(conn, launch):
                finish_job_launch_from_result(
                    conn,
                    launch_id=launch_id,
                    result=_terminal_result(
                        launch,
                        status="canceled",
                        exit_code=130,
                        error="cancel requested; container authoritatively absent",
                    ),
                )
                reconciled += 1
            elif launch["state"] == "launching":
                if _start_or_resume_launch(conn, machine, launch=launch):
                    reconciled += 1
            else:
                finish_job_launch_from_result(
                    conn,
                    launch_id=launch_id,
                    result=_terminal_result(
                        launch,
                        status="failed",
                        exit_code=None,
                        error="running container authoritatively absent without result.json",
                    ),
                )
                reconciled += 1
            continue
        if (
            container.state in {"created", "restarting", "running"}
            and launch_cancel_requested(conn, launch)
        ):
            if cancel_running_job_launch(
                conn,
                machine,
                launch=launch,
                container=container,
            ):
                reconciled += 1
                continue
        if container.state == "created":
            if _start_or_resume_launch(conn, machine, launch=launch, known_container=container):
                reconciled += 1
            continue
        if container.state == "running":
            if launch["state"] == "launching":
                mark_job_launch_running(
                    conn,
                    launch_id=launch_id,
                    container_name=container.name,
                    provider_run_id=container.name,
                )
                reconciled += 1
            continue
        observation = observe_remote_result(machine, str(launch["output_uri"]))
        if observation.state == "present":
            finish_job_launch_from_result(
                conn,
                launch_id=launch_id,
                result=observation.payload or {},
            )
            reconciled += 1
        elif observation.state == "error":
            _record_launch_error(conn, launch_id, observation.error or "result observation failed")
        else:
            finish_job_launch_from_result(
                conn,
                launch_id=launch_id,
                result=_terminal_result(
                    launch,
                    status="failed",
                    exit_code=None,
                    error=f"container exited without result.json: {container.status}",
                ),
            )
            reconciled += 1
    return reconciled


def train_container_slot_usage(conn, machine: MachineConfig) -> tuple[int, int, int]:
    containers = list_job_containers(machine)
    launches = active_job_launches(conn, machine=machine.name)
    reserved = {str(launch["launch_id"]) for launch in launches}
    orphan_count = 0
    for container in containers:
        if container.job_kind != TRAIN_JOB_KIND or container.state not in {"running", "created", "restarting"}:
            continue
        if container.launch_id:
            reserved.add(container.launch_id)
        else:
            orphan_count += 1
    control = machine_control(conn, machine=machine.name)
    configured = machine.limits.max_parallel_containers
    requested = control.get("effective_capacity")
    capacity = min(configured, int(requested)) if requested is not None else configured
    used = len(reserved) + orphan_count
    return used, capacity, max(0, capacity - used)


def launch_next_jobs(
    conn,
    *,
    machine: MachineConfig,
) -> int:
    launched = 0
    available_images: set[str] = set()
    control = machine_control(conn, machine=machine.name)
    if bool(control.get("drained")):
        return 0
    _used, _capacity, slots = train_container_slot_usage(conn, machine)
    for _ in range(slots):
        pending = next_pending_train_job(conn, machine=machine.name)
        if pending is None:
            break
        runtime_image_ref = str(pending["runtime_image_ref"])
        if runtime_image_ref not in available_images:
            if not ensure_runtime_image_available(machine, runtime_image_ref):
                break
            available_images.add(runtime_image_ref)
        job_id = int(pending["id"])
        launch_id = new_train_launch_id(job_id)
        container_name = job_container_name(machine, launch_id=launch_id)
        claimed = claim_job_launch(
            conn,
            machine=machine.name,
            backend=machine.backend,
            job_id=job_id,
            launch_id=launch_id,
            container_name=container_name,
            output_uri=launch_output_path(machine, launch_id),
        )
        if claimed is None:
            break
        _job, launch = claimed
        if _start_or_resume_launch(conn, machine, launch=launch, image_ready=True):
            launched += 1
    return launched


def run_reconcile_fill_pass(
    conn,
    *,
    machine: MachineConfig,
    shared_env_file: Path | None = None,
) -> tuple[int, int]:
    pending = next_pending_train_job(conn, machine=machine.name)
    launching = active_job_launches(conn, machine=machine.name, states=("launching",))
    if machine.backend == "docker_ssh" and (pending is not None or launching):
        sync_shared_runner_env(
            machine,
            shared_env_file or (default_repo_root() / DEFAULT_SHARED_RUNNER_ENV_FILE),
        )
    reconciled = reconcile_machine_launches(conn, machine)
    launched = launch_next_jobs(conn, machine=machine)
    return reconciled, launched


@contextmanager
def machine_mutation_lock(conn, machine_name: str):
    lock = acquire_machine_lock(conn, machine_name)
    try:
        yield lock
    finally:
        release_machine_lock(conn, lock)


def machine_lock_key(machine_name: str) -> str:
    return f"rlab-fleet-reconciler:{machine_name}"


def acquire_machine_lock(conn, machine_name: str) -> MachineMutationLock:
    key = machine_lock_key(machine_name)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT pg_try_advisory_lock(hashtextextended(%(key)s, 0)) AS acquired",
            {"key": key},
        )
        row = cur.fetchone()
    if not row or not row.get("acquired"):
        raise MachineLockBusy(machine_name)
    return MachineMutationLock(machine=machine_name, key=key)


def release_machine_lock(conn, lock: MachineMutationLock) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT pg_advisory_unlock(hashtextextended(%(key)s, 0)) AS released",
            {"key": lock.key},
        )


def _maintenance_due(path: Path, *, interval_seconds: float = 3600.0) -> bool:
    try:
        return time.time() - path.stat().st_mtime >= interval_seconds
    except FileNotFoundError:
        return True


def run_service_machine_pass(
    *,
    machine_name: str,
    machines_path: Path,
    repo_root: Path,
    deadline_monotonic: float,
) -> dict[str, Any]:
    if time.monotonic() >= deadline_monotonic:
        raise TimeoutError(f"machine lane deadline expired before start: {machine_name}")
    machine = resolve_machine(load_machine_registry(machines_path), machine_name)
    deadline_token = _MACHINE_LANE_DEADLINE.set(deadline_monotonic)
    conn = None
    try:
        conn = connect(database_url())
        with machine_mutation_lock(conn, machine.name):
            reconciled, launched = run_reconcile_fill_pass(
                conn,
                machine=machine,
                shared_env_file=repo_root / DEFAULT_SHARED_RUNNER_ENV_FILE,
            )
            if time.monotonic() >= deadline_monotonic:
                raise TimeoutError(f"machine lane deadline expired: {machine_name}")
            maintenance_marker = repo_root / "logs" / "fleet" / f"maintenance-{machine.name}.stamp"
            pruned = 0
            removed_containers = 0
            if reconciled or launched or _maintenance_due(maintenance_marker):
                removed_containers = prune_inactive_job_containers(conn, machine)
                pruned = prune_stale_runtime_images(conn, machine)
                maintenance_marker.parent.mkdir(parents=True, exist_ok=True)
                maintenance_marker.touch()
            return {
                "reconciled": reconciled,
                "launched": launched,
                "removed_containers": removed_containers,
                "pruned_images": pruned,
            }
    finally:
        if conn is not None:
            conn.close()
        _MACHINE_LANE_DEADLINE.reset(deadline_token)


def _kick_after_machine_control() -> str:
    from rlab.fleet_service import kick_service

    try:
        return "kicked" if kick_service() else "degraded"
    except Exception:
        return "degraded"


def cmd_drain(args: argparse.Namespace) -> int:
    resolve_machine(load_registry_from_args(args), args.machine)
    conn = _connect_from_args(args)
    try:
        control = set_machine_control(
            conn,
            machine=args.machine,
            drained=True,
            reason=args.reason or "operator drain",
        )
    finally:
        conn.close()
    print(json.dumps({"control": json_safe(control), "dispatch": _kick_after_machine_control()}, sort_keys=True))
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    resolve_machine(load_registry_from_args(args), args.machine)
    conn = _connect_from_args(args)
    try:
        control = set_machine_control(conn, machine=args.machine, drained=False, reason="resumed")
    finally:
        conn.close()
    print(json.dumps({"control": json_safe(control), "dispatch": _kick_after_machine_control()}, sort_keys=True))
    return 0


def cmd_capacity(args: argparse.Namespace) -> int:
    machine = resolve_machine(load_registry_from_args(args), args.machine)
    if args.capacity is not None and args.capacity > machine.limits.max_parallel_containers:
        raise SystemExit(
            f"capacity {args.capacity} exceeds configured maximum "
            f"{machine.limits.max_parallel_containers} for {machine.name}"
        )
    conn = _connect_from_args(args)
    try:
        control = set_machine_control(
            conn,
            machine=machine.name,
            effective_capacity=args.capacity,
            reset_capacity=bool(args.reset),
            reason="capacity reset" if args.reset else f"capacity set to {args.capacity}",
        )
    finally:
        conn.close()
    print(json.dumps({"control": json_safe(control), "dispatch": _kick_after_machine_control()}, sort_keys=True))
    return 0


def cmd_setup_host(args: argparse.Namespace) -> int:
    machine = resolve_machine(load_registry_from_args(args), args.host)
    runtime_image_ref = runtime_image_ref_from_args(args)
    script = setup_host_script(machine, runtime_image_ref=runtime_image_ref)
    print(f"host: {machine.name}")
    print(script.rstrip())
    if not args.execute:
        print("dry_run: rerun without --dry-run to run setup over SSH")
        return 0
    return int(run_machine_shell(machine, script).returncode)


def add_machine_registry_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--machines", type=Path, default=DEFAULT_MACHINE_REGISTRY)


def add_database_arg(parser: argparse.ArgumentParser) -> None:
    add_direct_database_arg(parser)


def add_runtime_image_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--runtime-image-ref")
    group.add_argument("--runtime-image-ref-file", type=Path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage one-job rlab containers from queue state.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    drain = subparsers.add_parser("drain", help="Block new claims for one machine.")
    add_machine_registry_arg(drain)
    add_database_arg(drain)
    drain.add_argument("--machine", required=True)
    drain.add_argument("--reason")
    drain.set_defaults(func=cmd_drain)

    resume = subparsers.add_parser("resume", help="Allow new claims for one machine.")
    add_machine_registry_arg(resume)
    add_database_arg(resume)
    resume.add_argument("--machine", required=True)
    resume.set_defaults(func=cmd_resume)

    capacity = subparsers.add_parser("capacity", help="Set temporary effective machine capacity.")
    add_machine_registry_arg(capacity)
    add_database_arg(capacity)
    capacity.add_argument("--machine", required=True)
    capacity_action = capacity.add_mutually_exclusive_group(required=True)
    capacity_action.add_argument("--set", dest="capacity", type=int)
    capacity_action.add_argument("--reset", action="store_true")
    capacity.set_defaults(func=cmd_capacity)

    setup = subparsers.add_parser("setup-host", help="Prepare SSH Docker hosts for job containers.")
    add_machine_registry_arg(setup)
    setup.add_argument("--host", required=True, help="Fleet host to set up.")
    add_dry_run_arg(setup)
    add_runtime_image_args(setup)
    setup.set_defaults(func=cmd_setup_host)

    from rlab.fleet_service import add_service_parser

    add_service_parser(subparsers)

    return parser


def main(argv: list[str] | None = None) -> int:
    argv_list = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv_list)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
