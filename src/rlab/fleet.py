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
from pathlib import Path
from typing import Any

from rlab.cli_args import add_direct_database_arg, add_dry_run_arg
from rlab.job_queue import (
    QueueDemand,
    TRAIN_JOB_KIND,
    adopt_terminal_train_launch,
    active_job_launches,
    claim_job_launch,
    connect,
    database_url,
    finish_job_launch_from_result,
    job_payload_for_launch,
    machine_queue_counts,
    mark_job_launch_running,
    new_train_launch_id,
    queue_demands,
    release_job_launch,
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
    runtime_image_digest_slug,
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
from rlab.fleet_rendering import (
    MachineWatchSnapshot,
    colorize,
    format_utc_second,
    highlight_dashboard_text,
    render_machine_watch_dashboard,
    write_tui_frame,
)


DEFAULT_SHARED_RUNNER_ENV_FILE = Path(".env")
DEFAULT_WATCH_LATEST_INTERVAL_SECONDS = 15.0
GPU_TEST_IMAGE = "nvidia/cuda:12.9.1-base-ubuntu22.04"
SHARED_RUNNER_ENV_KEYS = (
    "WANDB_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_S3_ENDPOINT_URL",
    "AWS_REGION",
    "CHECKPOINT_BUCKET_URI",
)


@dataclass(frozen=True)
class ShepherdLock:
    machine: str
    key: str


class ShepherdLockBusy(RuntimeError):
    def __init__(self, machine: str) -> None:
        super().__init__(f"another shepherd is already running for machine={machine}")
        self.machine = machine


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
    *,
    color: bool | None = None,
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
    log_shepherd_event(
        machine=machine.name,
        action="sync-env",
        result="ok",
        keys=len(SHARED_RUNNER_ENV_KEYS),
        color=color,
    )


def format_demands(demands: Sequence[QueueDemand]) -> str:
    if not demands:
        return "queue demand: none"
    lines = ["queue demand:"]
    for demand in demands:
        lines.append(
            "  "
            f"target={demand.run_target or 'any'} "
            f"pending={demand.pending_count} running={demand.running_count} "
            f"digest={runtime_image_digest_slug(demand.runtime_image_ref)}"
        )
    return "\n".join(lines)


def cmd_status(args: argparse.Namespace) -> int:
    conn = _connect_from_args(args)
    try:
        demands = queue_demands(conn)
    finally:
        conn.close()
    print(format_demands(demands))
    return 0


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


def job_container_run_command(
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
            "run",
            "-d",
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
    return ["ssh", *machine.ssh_options, machine.ssh_target]


def run_machine_shell(
    machine: MachineConfig,
    script: str,
    *,
    input_text: str | None = None,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    if machine.backend == "local_docker":
        return subprocess.run(
            ["sh", "-lc", script],
            input=input_text,
            capture_output=capture,
            text=True,
            check=False,
        )
    return subprocess.run(
        [*machine_ssh_prefix(machine), "sh", "-lc", shlex.quote(script)],
        input=input_text,
        capture_output=capture,
        text=True,
        check=False,
    )


def run_machine_docker(
    machine: MachineConfig,
    docker_args: Sequence[str],
    *,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    command = shell_join(machine_docker_command(machine, docker_args))
    return run_machine_shell(machine, command, capture=capture)


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
        if demand.run_target not in (None, machine.run_target):
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
    *,
    color: bool | None = None,
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
            log_shepherd_event(
                machine=machine.name,
                action="prune-image",
                result="ok",
                image=image.digest,
                repository=image.repository,
                color=color,
            )
            continue
        log_shepherd_event(
            machine=machine.name,
            action="prune-image",
            result="failed",
            image=image.digest,
            repository=image.repository,
            error=(result.stderr or result.stdout or "").strip() or f"exit={result.returncode}",
            color=color,
        )
    return pruned


def read_remote_result(machine: MachineConfig, output_uri: str) -> dict[str, Any] | None:
    result_path = f"{str(output_uri).rstrip('/')}/result.json"
    result = run_machine_shell(machine, f"cat {shlex.quote(result_path)}", capture=True)
    if result.returncode != 0 or not result.stdout.strip():
        return None
    payload = json.loads(result.stdout)
    if not isinstance(payload, dict):
        raise ValueError(f"{result_path} did not contain a JSON object")
    return payload


def remote_result_exists(machine: MachineConfig, output_uri: str) -> bool:
    result_path = f"{str(output_uri).rstrip('/')}/result.json"
    result = run_machine_shell(machine, f"test -s {shlex.quote(result_path)}", capture=True)
    return result.returncode == 0


def build_machine_watch_snapshot(args: argparse.Namespace) -> MachineWatchSnapshot:
    machine = resolve_machine(load_registry_from_args(args), args.machine)
    warnings: list[str] = []
    containers = tuple(sorted(list_job_containers(machine), key=lambda item: item.name))
    conn = _connect_from_args(args)
    try:
        launches = tuple(
            active_job_launches(
                conn,
                machine=machine.name,
                states=("launching", "running"),
            )
        )
        queue_counts = machine_queue_counts(conn)
    finally:
        conn.close()

    containers_by_launch = {container.launch_id: container for container in containers if container.launch_id}
    result_present: dict[str, bool] = {}
    for launch in launches:
        launch_id = str(launch["launch_id"])
        container = containers_by_launch.get(launch_id)
        if container is None or container.state != "running":
            try:
                result_present[launch_id] = remote_result_exists(machine, str(launch["output_uri"]))
            except Exception as exc:
                result_present[launch_id] = False
                warnings.append(f"result check failed launch_id={launch_id}: {exc}")
    for container in containers:
        launch_id = container.launch_id
        output_uri = container.labels.get(OUTPUT_URI_LABEL)
        if not launch_id or not output_uri or container.state == "running":
            continue
        if launch_id in result_present:
            continue
        try:
            result_present[launch_id] = remote_result_exists(machine, output_uri)
        except Exception as exc:
            result_present[launch_id] = False
            warnings.append(f"result check failed launch_id={launch_id}: {exc}")

    return MachineWatchSnapshot(
        captured_at=datetime.now(UTC),
        machine=machine,
        containers=containers,
        launches=launches,
        queue_counts=queue_counts,
        result_present=result_present,
        warnings=tuple(warnings),
    )


def shepherd_color_enabled(color: bool | None) -> bool:
    return sys.stdout.isatty() if color is None else color


def shepherd_event_style(action: str, result: str | None) -> str:
    if result in {"ok", "started"}:
        return "bright_green"
    if result in {"busy", "skip"} or action in {"launch-next", "reconcile"}:
        return "bright_yellow" if result == "start" else "bright_cyan"
    if result in {"failed", "error"} or action == "error":
        return "bright_red"
    if action == "stop":
        return "yellow"
    return "white"


def shepherd_event_icon(action: str, result: str | None) -> str:
    if result in {"ok", "started"}:
        return "✓"
    if result in {"busy", "skip"}:
        return "!"
    if result in {"failed", "error"} or action == "error":
        return "✕"
    if action in {"launch", "launch-next"}:
        return "→"
    if action == "lock":
        return "■"
    if action == "stop":
        return "■"
    return "•"


def format_shepherd_event(
    *,
    machine: str,
    action: str,
    result: str | None = None,
    color: bool | None = None,
    timestamp: datetime | None = None,
    **fields: Any,
) -> str:
    enabled = shepherd_color_enabled(color)
    style = shepherd_event_style(action, result)
    icon = colorize(shepherd_event_icon(action, result), style, enabled=enabled)
    parts = [
        colorize(format_utc_second(timestamp), "gray", enabled=enabled),
        icon,
        f"machine={colorize(machine, 'cyan', enabled=enabled)}",
        f"action={colorize(action, style, enabled=enabled)}",
    ]
    if result is not None:
        parts.append(f"result={colorize(result, style, enabled=enabled)}")
    for key, value in fields.items():
        if value is None:
            continue
        rendered = shlex.quote(str(value)) if isinstance(value, str) and any(char.isspace() for char in value) else str(value)
        parts.append(f"{key}={highlight_dashboard_text(rendered, color=enabled)}")
    return " ".join(parts)


def log_shepherd_event(*, color: bool | None = None, **fields: Any) -> None:
    print(format_shepherd_event(color=color, **fields), flush=True)


def docker_pull_for_job(machine: MachineConfig, runtime_image_ref: str) -> int:
    if machine.pull_policy == "never":
        return 0
    result = run_machine_docker(machine, ["pull", docker_image_ref(runtime_image_ref)], capture=False)
    return int(result.returncode)


def launch_claimed_job_container(
    conn,
    *,
    machine: MachineConfig,
    job_id: int | None = None,
    shared_env_file: Path | None = None,
    color: bool | None = None,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    if machine.backend == "docker_ssh":
        sync_shared_runner_env(
            machine,
            shared_env_file or (default_repo_root() / DEFAULT_SHARED_RUNNER_ENV_FILE),
            color=color,
        )
    launch_id = new_train_launch_id(job_id)
    claimed = claim_job_launch(
        conn,
        machine=machine.name,
        backend=machine.backend,
        job_id=job_id,
        run_target=machine.run_target,
        launch_id=launch_id,
        output_uri=launch_output_path(machine, launch_id),
    )
    if claimed is None:
        return None
    job, launch = claimed
    runtime_image_ref = str(job["runtime_image_ref"])
    container_name = job_container_name(machine, launch_id=launch_id)
    payload = job_payload_for_launch(job, {**launch, "container_name": container_name})
    try:
        write_remote_payload(machine, launch_payload_path(machine, launch_id), payload)
        pull_status = docker_pull_for_job(machine, runtime_image_ref)
        if pull_status != 0:
            release_job_launch(conn, launch_id=launch_id, error=f"docker pull failed {pull_status}")
            raise RuntimeError(f"docker pull failed exit={pull_status}")
        run_command = job_container_run_command(
            machine,
            job_id=int(job["id"]),
            launch_id=launch_id,
            runtime_image_ref=runtime_image_ref,
            container_name=container_name,
        )
        result = run_machine_shell(machine, shell_join(run_command), capture=True)
        if result.returncode != 0:
            release_job_launch(
                conn,
                launch_id=launch_id,
                error=f"docker run failed {result.returncode}: {result.stderr or result.stdout}",
            )
            raise RuntimeError(f"docker run failed exit={result.returncode}: {result.stderr or result.stdout}")
        provider_run_id = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else container_name
        mark_job_launch_running(
            conn,
            launch_id=launch_id,
            container_name=container_name,
            provider_run_id=provider_run_id,
        )
        log_shepherd_event(
            machine=machine.name,
            action="launch",
            result="started",
            job_kind=TRAIN_JOB_KIND,
            job_id=job["id"],
            launch_id=launch_id,
            container=container_name,
            color=color,
        )
        return job, launch
    except Exception:
        raise


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
    color: bool | None = None,
) -> bool:
    launch_id = str(launch["launch_id"])
    result = run_machine_docker(machine, ["stop", container.name], capture=True)
    if result.returncode != 0:
        log_shepherd_event(
            machine=machine.name,
            action="cancel",
            result="failed",
            launch_id=launch_id,
            container=container.name,
            error=(result.stderr or result.stdout or "").strip() or f"exit={result.returncode}",
            color=color,
        )
        return False
    synthetic = {
        "job_kind": launch["job_kind"],
        "job_id": launch["job_id"],
        "launch_id": launch_id,
        "status": "canceled",
        "exit_code": 130,
        "error": "cancel requested",
    }
    finish_job_launch_from_result(conn, launch_id=launch_id, result=synthetic)
    log_shepherd_event(
        machine=machine.name,
        action="cancel",
        result="ok",
        launch_id=launch_id,
        container=container.name,
        color=color,
    )
    return True


def reconcile_machine_launches(conn, machine: MachineConfig, *, color: bool | None = None) -> int:
    launches = active_job_launches(conn, machine=machine.name)
    containers = {container.launch_id: container for container in list_job_containers(machine)}
    reconciled = 0
    released_launch_ids: set[str] = set()
    for launch in launches:
        launch_id = str(launch["launch_id"])
        if launch_id in released_launch_ids:
            reconciled += 1
            continue
        container = containers.get(launch_id)
        if container is None:
            result = read_remote_result(machine, str(launch["output_uri"]))
            if result is not None:
                finish_job_launch_from_result(conn, launch_id=launch_id, result=result)
                log_shepherd_event(
                    machine=machine.name,
                    action="finalize",
                    result="ok",
                    launch_id=launch_id,
                    source="result.json",
                    color=color,
                )
            elif launch["state"] == "launching":
                release_job_launch(conn, launch_id=launch_id, error="launching container missing")
                log_shepherd_event(
                    machine=machine.name,
                    action="release",
                    result="ok",
                    launch_id=launch_id,
                    reason="launching container missing",
                    color=color,
                )
            else:
                synthetic = {
                    "job_kind": launch["job_kind"],
                    "job_id": launch["job_id"],
                    "launch_id": launch_id,
                    "status": "failed",
                    "exit_code": None,
                    "error": "running container missing and result.json not found",
                }
                finish_job_launch_from_result(conn, launch_id=launch_id, result=synthetic)
                log_shepherd_event(
                    machine=machine.name,
                    action="finalize",
                    result="failed",
                    launch_id=launch_id,
                    reason="running container missing",
                    color=color,
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
                color=color,
            ):
                reconciled += 1
                continue
        if container.state == "running":
            mark_job_launch_running(
                conn,
                launch_id=launch_id,
                container_name=container.name,
                provider_run_id=container.name,
            )
            reconciled += 1
            continue
        result = read_remote_result(machine, str(launch["output_uri"]))
        if result is not None:
            superseded = tuple(
                str(candidate["launch_id"])
                for candidate in launches
                if candidate["job_id"] == launch["job_id"]
                and candidate["launch_id"] != launch_id
                and (
                    (sibling := containers.get(str(candidate["launch_id"]))) is None
                    or sibling.state not in {"created", "restarting", "running"}
                )
            )
            if superseded:
                released = adopt_terminal_train_launch(
                    conn,
                    launch_id=launch_id,
                    superseded_launch_ids=superseded,
                )
                released_launch_ids.update(released)
                log_shepherd_event(
                    machine=machine.name,
                    action="supersede",
                    result="ok",
                    launch_id=launch_id,
                    released=",".join(released),
                    color=color,
                )
            finish_job_launch_from_result(conn, launch_id=launch_id, result=result)
            log_shepherd_event(
                machine=machine.name,
                action="finalize",
                result="ok",
                launch_id=launch_id,
                exit_state=container.state,
                color=color,
            )
        else:
            synthetic = {
                "job_kind": launch["job_kind"],
                "job_id": launch["job_id"],
                "launch_id": launch_id,
                "status": "failed",
                "exit_code": None,
                "error": f"container exited without result.json: {container.status}",
            }
            finish_job_launch_from_result(conn, launch_id=launch_id, result=synthetic)
            log_shepherd_event(
                machine=machine.name,
                action="finalize",
                result="failed",
                launch_id=launch_id,
                reason="missing result.json",
                color=color,
            )
        reconciled += 1
    return reconciled


def train_container_slot_usage(machine: MachineConfig) -> tuple[int, int, int]:
    containers = list_job_containers(machine)
    active = [
        container
        for container in containers
        if container.job_kind == TRAIN_JOB_KIND
        and container.state in {"running", "created", "restarting"}
    ]
    capacity = machine.limits.max_parallel_containers
    return len(active), capacity, max(0, capacity - len(active))


def machine_available_train_slots(machine: MachineConfig, *, limit: int | None = None) -> int:
    active, capacity, available = train_container_slot_usage(machine)
    if limit is None:
        return available
    if limit < 1:
        raise ValueError("fleet container limit must be at least 1")
    desired_capacity = min(limit, capacity)
    return max(0, desired_capacity - active)


def launch_next_jobs(
    conn,
    *,
    machine: MachineConfig,
    limit: int,
    shared_env_file: Path | None = None,
    color: bool | None = None,
) -> int:
    launched = 0
    active, capacity, _available = train_container_slot_usage(machine)
    slots = machine_available_train_slots(machine, limit=int(limit))
    if slots <= 0:
        log_shepherd_event(
            machine=machine.name,
            action="launch-next",
            result="skip",
            reason="no_available_slots",
            job_kind=TRAIN_JOB_KIND,
            used=active,
            max=capacity,
            color=color,
        )
        return 0
    for _ in range(slots):
        claimed = launch_claimed_job_container(
            conn,
            machine=machine,
            shared_env_file=shared_env_file,
            color=color,
        )
        if claimed is None:
            break
        launched += 1
    return launched


def run_reconcile_fill_pass(
    conn,
    *,
    machine: MachineConfig,
    limit: int,
    shared_env_file: Path | None = None,
    color: bool | None = None,
) -> tuple[int, int]:
    log_shepherd_event(machine=machine.name, action="reconcile", result="start", color=color)
    reconciled = reconcile_machine_launches(conn, machine, color=color)
    log_shepherd_event(
        machine=machine.name,
        action="reconcile",
        result="ok",
        reconciled=reconciled,
        color=color,
    )
    launched = launch_next_jobs(
        conn,
        machine=machine,
        limit=limit,
        shared_env_file=shared_env_file,
        color=color,
    )
    log_shepherd_event(
        machine=machine.name,
        action="launch-next",
        result="ok",
        job_kind=TRAIN_JOB_KIND,
        launched=launched,
        color=color,
    )
    return reconciled, launched


@contextmanager
def machine_mutation_lock(conn, machine_name: str):
    lock = acquire_shepherd_lock(conn, machine_name)
    try:
        yield lock
    finally:
        release_shepherd_lock(conn, lock)


def shepherd_lock_key(machine_name: str) -> str:
    return f"rlab-fleet-shepherd:{machine_name}"


def acquire_shepherd_lock(conn, machine_name: str) -> ShepherdLock:
    key = shepherd_lock_key(machine_name)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT pg_try_advisory_lock(hashtextextended(%(key)s, 0)) AS acquired",
            {"key": key},
        )
        row = cur.fetchone()
    if not row or not row.get("acquired"):
        raise ShepherdLockBusy(machine_name)
    return ShepherdLock(machine=machine_name, key=key)


def release_shepherd_lock(conn, lock: ShepherdLock) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT pg_advisory_unlock(hashtextextended(%(key)s, 0)) AS released",
            {"key": lock.key},
        )


def cmd_container_shepherd(args: argparse.Namespace) -> int:
    machine = resolve_machine(load_registry_from_args(args), args.machine)
    shared_env_file = shared_runner_env_file_from_args(args)
    color = not getattr(args, "no_color", False)
    conn = _connect_from_args(args)
    try:
        try:
            with machine_mutation_lock(conn, machine.name):
                while True:
                    try:
                        _reconciled, _launched = run_reconcile_fill_pass(
                            conn,
                            machine=machine,
                            limit=int(args.limit),
                            shared_env_file=shared_env_file,
                            color=color,
                        )
                        pruned = prune_stale_runtime_images(
                            conn,
                            machine,
                            color=color,
                        )
                        log_shepherd_event(
                            machine=machine.name,
                            action="prune-image",
                            result="ok",
                            pruned=pruned,
                            color=color,
                        )
                        if args.once:
                            return 0
                    except KeyboardInterrupt:
                        log_shepherd_event(
                            machine=machine.name,
                            action="stop",
                            reason="keyboard_interrupt",
                            color=color,
                        )
                        return 130
                    except Exception as exc:
                        log_shepherd_event(
                            machine=machine.name,
                            action="error",
                            result="error",
                            error=str(exc),
                            color=color,
                        )
                        if args.once or getattr(args, "fail_fast", False):
                            return 1
                    time.sleep(args.interval)
        except ShepherdLockBusy as exc:
            log_shepherd_event(machine=exc.machine, action="lock", result="busy", color=color)
            return 2
    finally:
        conn.close()


def cmd_container_watch_dashboard(args: argparse.Namespace) -> int:
    while True:
        try:
            snapshot = build_machine_watch_snapshot(args)
            write_tui_frame(
                render_machine_watch_dashboard(snapshot, color=not args.no_color),
                enabled=not args.no_tui,
            )
            if args.once:
                return 0
        except KeyboardInterrupt:
            print("\nwatch stopped")
            return 130
        except Exception as exc:
            message = (
                f"rlab fleet watch machine={args.machine} mode=read-only\n"
                f"snapshot failed: {exc}\n\nCtrl-C to stop."
            )
            write_tui_frame(message, enabled=not args.no_tui)
            if args.once or getattr(args, "fail_fast", False):
                return 1
        time.sleep(args.interval)


def cmd_ps(args: argparse.Namespace) -> int:
    registry = load_registry_from_args(args)
    machines = (
        [resolve_machine(registry, args.machine)]
        if getattr(args, "machine", None)
        else list(registry.machines.values())
    )
    lines: list[str] = []
    for machine in machines:
        try:
            containers = sorted(list_job_containers(machine), key=lambda item: item.name)
        except Exception as exc:
            lines.append(f"{machine.name}: failed to list job containers: {exc}")
            continue
        if not containers:
            lines.append(f"{machine.name}: job containers: none")
            continue
        lines.append(f"{machine.name}: job containers:")
        for container in containers:
            lines.append(
                "  "
                f"name={container.name} state={container.state or 'unknown'} "
                f"status={container.status or 'unknown'} "
                f"job={container.labels.get(JOB_ID_LABEL, 'unknown')} "
                f"launch={container.launch_id or 'unknown'}"
            )
    print("\n".join(lines) if lines else "job containers: none")
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

    status = subparsers.add_parser("status", help="Print train queue demand and active leases.")
    add_database_arg(status)
    status.set_defaults(func=cmd_status)

    ps = subparsers.add_parser("ps", help="List one-job containers across configured machines.")
    add_machine_registry_arg(ps)
    ps.add_argument("--machine", help="Limit listing to one machine.")
    ps.set_defaults(func=cmd_ps)

    shepherd = subparsers.add_parser(
        "shepherd",
        help="Run the mutating one-job-container orchestration loop for one machine.",
    )
    add_machine_registry_arg(shepherd)
    add_database_arg(shepherd)
    shepherd.add_argument("--repo-root", default=None)
    shepherd.add_argument("--machine", required=True)
    shepherd.add_argument("--limit", type=int, default=1)
    shepherd.add_argument("--interval", type=float, default=30.0, help="Polling interval in seconds.")
    shepherd.add_argument("--once", action="store_true", help="Run one reconcile/fill pass and exit.")
    shepherd.add_argument(
        "--fail-fast",
        action="store_true",
        help="Exit when a poll or action fails instead of retrying forever.",
    )
    shepherd.add_argument("--no-color", action="store_true", help="Disable ANSI color output.")
    shepherd.set_defaults(func=cmd_container_shepherd)

    watch_latest = subparsers.add_parser(
        "watch",
        help="Run a read-only one-job-container dashboard.",
    )
    add_machine_registry_arg(watch_latest)
    add_database_arg(watch_latest)
    watch_latest.add_argument(
        "--machine",
        required=True,
        help="Machine to monitor.",
    )
    watch_latest.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_WATCH_LATEST_INTERVAL_SECONDS,
        help="Polling interval in seconds; defaults to 15.",
    )
    watch_latest.add_argument("--once", action="store_true", help="Render/apply one poll and exit.")
    watch_latest.add_argument(
        "--fail-fast",
        action="store_true",
        help="Exit when a poll or action fails instead of retrying forever.",
    )
    watch_latest.add_argument("--no-tui", action="store_true", help="Do not clear/redraw the terminal.")
    watch_latest.add_argument("--no-color", action="store_true", help="Disable ANSI color output.")
    watch_latest.add_argument("--width", type=int, help="Override dashboard width.")
    watch_latest.set_defaults(func=cmd_container_watch_dashboard)

    setup = subparsers.add_parser("setup-host", help="Prepare SSH Docker hosts for job containers.")
    add_machine_registry_arg(setup)
    setup.add_argument("--host", required=True, help="Fleet host to set up.")
    add_dry_run_arg(setup)
    add_runtime_image_args(setup)
    setup.set_defaults(func=cmd_setup_host)

    return parser


def main(argv: list[str] | None = None) -> int:
    argv_list = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv_list)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
