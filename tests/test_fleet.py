from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from rlab import fleet
from rlab.job_queue import QueueDemand
from rlab.machines import load_machine_registry, resolve_machine
from rlab.runtime_refs import runtime_image_ref_from_file, runtime_image_ref_from_payload
from tests.db_fakes import FakeConnection, write_machine_registry


RUNTIME_IMAGE_REF = (
    "docker:ghcr.io/tsilva/rlab/rlab-train@sha256:"
    "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
)
OTHER_IMAGE_REF = (
    "docker:ghcr.io/tsilva/rlab/rlab-train@sha256:"
    "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"
)


def local_machine(*, capacity: int = 2):
    with tempfile.TemporaryDirectory() as temporary_dir:
        path = Path(temporary_dir) / "machines.yaml"
        write_machine_registry(
            path,
            backend="local_docker",
            max_parallel_containers=capacity,
            host_root="/tmp/rlab",
        )
        return resolve_machine(load_machine_registry(path), "beast-test")


def ssh_machine():
    with tempfile.TemporaryDirectory() as temporary_dir:
        path = Path(temporary_dir) / "machines.yaml"
        write_machine_registry(
            path,
            backend="docker_ssh",
            max_parallel_containers=6,
            host_root="/home/tsilva/rlab",
            machine_name="beast-3",
        )
        return resolve_machine(load_machine_registry(path), "beast-3")


class FleetHostTests(unittest.TestCase):
    def test_setup_host_uses_configured_sudo_docker_command(self) -> None:
        script = fleet.setup_host_script(ssh_machine())

        self.assertIn("sudo -n docker info", script)
        self.assertIn("/home/tsilva/rlab", script)

    def test_ssh_prefix_is_noninteractive_and_bounded(self) -> None:
        prefix = fleet.machine_ssh_prefix(ssh_machine())
        text = " ".join(prefix)

        self.assertIn("BatchMode=yes", text)
        self.assertIn("ConnectTimeout=10", text)
        self.assertIn("ServerAliveCountMax=3", text)

    def test_machine_command_timeout_is_typed(self) -> None:
        machine = local_machine()
        expired = subprocess.TimeoutExpired(["sh"], 0.01)
        with mock.patch.object(subprocess, "run", side_effect=expired):
            with self.assertRaises(fleet.MachineCommandTimeout):
                fleet.run_machine_shell(machine, "true", timeout=0.01)

    def test_shared_runner_env_is_allowlisted_and_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / ".env"
            path.write_text(
                "\n".join(
                    [
                        'WANDB_API_KEY="wandb-value"',
                        "AWS_ACCESS_KEY_ID=access-value",
                        "AWS_SECRET_ACCESS_KEY=secret-value",
                        "AWS_S3_ENDPOINT_URL=https://r2.invalid",
                        "AWS_REGION=auto",
                        "CHECKPOINT_BUCKET_URI=s3://bucket/prefix",
                        "DATABASE_URL=postgresql://must-not-sync",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            values = fleet.load_shared_runner_env(path)

        self.assertEqual(tuple(values), fleet.SHARED_RUNNER_ENV_KEYS)
        self.assertEqual(values["WANDB_API_KEY"], "wandb-value")
        self.assertNotIn("DATABASE_URL", values)

    def test_sync_sends_secrets_only_over_stdin(self) -> None:
        machine = ssh_machine()
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / ".env"
            path.write_text(
                "\n".join(
                    [
                        "WANDB_API_KEY=wandb-value",
                        "AWS_ACCESS_KEY_ID=access-value",
                        "AWS_SECRET_ACCESS_KEY=secret-value",
                        "AWS_S3_ENDPOINT_URL=https://r2.invalid",
                        "AWS_REGION=auto",
                        "CHECKPOINT_BUCKET_URI=s3://bucket/prefix",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            with mock.patch.object(
                fleet,
                "run_machine_shell",
                return_value=SimpleNamespace(returncode=0, stdout="", stderr=""),
            ) as run:
                fleet.sync_shared_runner_env(machine, path)

        self.assertIn("WANDB_API_KEY=wandb-value", run.call_args.kwargs["input_text"])
        self.assertNotIn("wandb-value", run.call_args.args[1])

    def test_runtime_image_ref_file_and_alias_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "image.json"
            path.write_text(json.dumps({"runtime_image_ref": RUNTIME_IMAGE_REF}), encoding="utf-8")
            self.assertEqual(runtime_image_ref_from_file(path), RUNTIME_IMAGE_REF)
        with self.assertRaisesRegex(ValueError, "must include runtime_image_ref"):
            runtime_image_ref_from_payload({"image_ref": RUNTIME_IMAGE_REF})


class ContainerContractTests(unittest.TestCase):
    def test_create_command_uses_stable_payload_contract(self) -> None:
        command = fleet.job_container_create_command(
            local_machine(),
            job_id=12,
            launch_id="train-12",
            runtime_image_ref=RUNTIME_IMAGE_REF,
            container_name="rlab-job-beast-test-train-train-12",
        )
        text = fleet.shell_join(command)

        self.assertIn("docker create", text)
        self.assertNotIn("docker run", text)
        self.assertIn("rlab-container-entrypoint rlab run-job", text)
        self.assertIn("--payload /input/payloads/train-12.json", text)
        self.assertIn("--label rlab.launch-id=train-12", text)

    def test_parse_job_containers_filters_managed_job_label(self) -> None:
        output = "\n".join(
            [
                json.dumps(
                    {
                        "Names": "current",
                        "State": "running",
                        "Status": "Up",
                        "Labels": "rlab.job-container=true,rlab.launch-id=train-12",
                    }
                ),
                json.dumps({"Names": "other", "State": "running", "Labels": "rlab.managed=true"}),
            ]
        )

        containers = fleet.parse_job_containers(local_machine(), output)

        self.assertEqual([container.name for container in containers], ["current"])
        self.assertEqual(containers[0].launch_id, "train-12")

    def test_result_observation_distinguishes_absent_and_unreachable(self) -> None:
        machine = local_machine()
        with mock.patch.object(
            fleet,
            "run_machine_shell",
            return_value=SimpleNamespace(returncode=4, stdout="", stderr=""),
        ):
            self.assertEqual(fleet.observe_remote_result(machine, "/output/train-1").state, "absent")
        with mock.patch.object(
            fleet,
            "run_machine_shell",
            return_value=SimpleNamespace(returncode=255, stdout="", stderr="unreachable"),
        ):
            observation = fleet.observe_remote_result(machine, "/output/train-1")
        self.assertEqual(observation.state, "error")
        self.assertIn("unreachable", observation.error or "")

    def test_exact_machine_demand_protects_only_that_hosts_image(self) -> None:
        demands = [
            QueueDemand("beast-3", RUNTIME_IMAGE_REF, 1, 0, 1),
            QueueDemand("beast-2", OTHER_IMAGE_REF, 1, 0, 2),
        ]

        protected = fleet.protected_runtime_image_refs(
            machine=SimpleNamespace(name="beast-3"),
            demands=demands,
            containers=[],
        )

        self.assertEqual(protected, {RUNTIME_IMAGE_REF})

    def test_capacity_counts_database_reservations_and_orphans(self) -> None:
        machine = local_machine(capacity=4)
        containers = [
            fleet.JobContainer(
                machine=machine.name,
                name="known",
                state="running",
                status="Up",
                labels={fleet.JOB_KIND_LABEL: "train", fleet.LAUNCH_ID_LABEL: "train-1"},
            ),
            fleet.JobContainer(
                machine=machine.name,
                name="orphan",
                state="running",
                status="Up",
                labels={fleet.JOB_KIND_LABEL: "train"},
            ),
        ]
        with (
            mock.patch.object(fleet, "list_job_containers", return_value=containers),
            mock.patch.object(
                fleet,
                "active_job_launches",
                return_value=[{"launch_id": "train-1"}, {"launch_id": "train-2"}],
            ),
            mock.patch.object(
                fleet,
                "machine_control",
                return_value={"drained": False, "effective_capacity": 3},
            ),
        ):
            used, capacity, available = fleet.train_container_slot_usage(FakeConnection(), machine)

        self.assertEqual((used, capacity, available), (3, 3, 0))

    def test_maintenance_removes_only_inactive_terminal_containers(self) -> None:
        target = local_machine()
        exited = fleet.JobContainer(
            machine=target.name,
            name="old",
            state="exited",
            status="Exited",
            labels={fleet.JOB_KIND_LABEL: "train", fleet.LAUNCH_ID_LABEL: "train-1"},
        )
        active = fleet.JobContainer(
            machine=target.name,
            name="active",
            state="running",
            status="Up",
            labels={fleet.JOB_KIND_LABEL: "train", fleet.LAUNCH_ID_LABEL: "train-2"},
        )
        with (
            mock.patch.object(fleet, "active_job_launches", return_value=[{"launch_id": "train-2"}]),
            mock.patch.object(fleet, "list_job_containers", return_value=[exited, active]),
            mock.patch.object(
                fleet,
                "run_machine_docker",
                return_value=SimpleNamespace(returncode=0, stdout="", stderr=""),
            ) as docker,
        ):
            removed = fleet.prune_inactive_job_containers(FakeConnection(), target)

        self.assertEqual(removed, 1)
        docker.assert_called_once_with(target, ["rm", "old"], capture=True)

    def test_missing_launching_container_reuses_same_launch(self) -> None:
        launch = {
            "launch_id": "train-12",
            "job_id": 12,
            "job_kind": "train",
            "machine": "beast-test",
            "runtime_image_ref": RUNTIME_IMAGE_REF,
            "output_uri": "/output/train-12",
            "state": "launching",
            "next_retry_at": None,
        }
        with (
            mock.patch.object(fleet, "active_job_launches", return_value=[launch]),
            mock.patch.object(fleet, "list_job_containers", return_value=[]),
            mock.patch.object(
                fleet,
                "observe_remote_result",
                return_value=fleet.ResultObservation("absent"),
            ),
            mock.patch.object(fleet, "launch_cancel_requested", return_value=False),
            mock.patch.object(fleet, "_start_or_resume_launch", return_value=True) as start,
        ):
            changed = fleet.reconcile_machine_launches(FakeConnection(), local_machine())

        self.assertEqual(changed, 1)
        self.assertEqual(start.call_args.kwargs["launch"]["launch_id"], "train-12")

    def test_public_parser_exposes_only_new_fleet_modes(self) -> None:
        help_text = fleet.build_parser().format_help()
        self.assertIn("service", help_text)
        self.assertIn("drain", help_text)
        self.assertIn("capacity", help_text)


if __name__ == "__main__":
    unittest.main()
