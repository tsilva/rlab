from __future__ import annotations

import json
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

    def test_runtime_image_ref_file_and_alias_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "image.json"
            path.write_text(json.dumps({"runtime_image_ref": RUNTIME_IMAGE_REF}), encoding="utf-8")
            self.assertEqual(runtime_image_ref_from_file(path), RUNTIME_IMAGE_REF)
        with self.assertRaisesRegex(ValueError, "must include runtime_image_ref"):
            runtime_image_ref_from_payload({"image_ref": RUNTIME_IMAGE_REF})


class ContainerContractTests(unittest.TestCase):
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
            host = SimpleNamespace(machine=machine, list_job_containers=lambda: containers)
            used, capacity, available = fleet.train_container_slot_usage(FakeConnection(), host)

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
        ):
            host = SimpleNamespace(
                machine=target,
                list_job_containers=lambda: [exited, active],
                remove_container=mock.Mock(return_value=SimpleNamespace(ok=True)),
            )
            removed = fleet.prune_inactive_job_containers(FakeConnection(), host)

        self.assertEqual(removed, 1)
        host.remove_container.assert_called_once_with("old")

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
            mock.patch.object(fleet, "launch_cancel_requested", return_value=False),
            mock.patch.object(fleet, "_start_or_resume_launch", return_value=True) as start,
        ):
            host = SimpleNamespace(
                machine=local_machine(),
                list_job_containers=lambda: [],
                observe_result=lambda _path: SimpleNamespace(state="absent"),
            )
            changed = fleet.reconcile_machine_launches(FakeConnection(), host)

        self.assertEqual(changed, 1)
        self.assertEqual(start.call_args.kwargs["launch"]["launch_id"], "train-12")

    def test_public_parser_exposes_only_new_fleet_modes(self) -> None:
        help_text = fleet.build_parser().format_help()
        self.assertIn("service", help_text)
        self.assertIn("drain", help_text)
        self.assertIn("capacity", help_text)


if __name__ == "__main__":
    unittest.main()
