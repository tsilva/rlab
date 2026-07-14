from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rlab import fleet
from rlab.docker_host import (
    HostOperationResult,
    JobContainer,
    MachineCommandTimeout,
    ResultObservation,
)
from rlab.machines import load_machine_registry, resolve_machine
from tests.db_fakes import FakeConnection, write_machine_registry


RUNTIME_IMAGE_REF = (
    "docker:ghcr.io/tsilva/rlab/rlab-train@sha256:"
    "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
)


def machine():
    with tempfile.TemporaryDirectory() as temporary_dir:
        path = Path(temporary_dir) / "machines.yaml"
        write_machine_registry(
            path,
            backend="local_docker",
            max_parallel_containers=2,
            host_root="/host/rlab",
        )
        return resolve_machine(load_machine_registry(path), "beast-test")


def launch(*, state: str = "launching") -> dict[str, object]:
    return {
        "launch_id": "train-12",
        "job_id": 12,
        "job_kind": "train",
        "machine": "beast-test",
        "runtime_image_ref": RUNTIME_IMAGE_REF,
        "container_name": "rlab-job-beast-test-train-train-12",
        "output_uri": "/host/rlab/runs/outputs/train-12",
        "state": state,
        "next_retry_at": None,
    }


def container(*, state: str = "running") -> JobContainer:
    return JobContainer(
        machine="beast-test",
        name="rlab-job-beast-test-train-train-12",
        state=state,
        status="Up" if state == "running" else "Created",
        labels={
            fleet.JOB_CONTAINER_LABEL: "true",
            fleet.JOB_ID_LABEL: "12",
            fleet.JOB_KIND_LABEL: "train",
            fleet.LAUNCH_ID_LABEL: "train-12",
        },
    )


class FakeHost:
    def __init__(
        self,
        target,
        *,
        container_snapshots: list[list[JobContainer]] | None = None,
        result: ResultObservation | None = None,
        readiness: ResultObservation | None = None,
        image_ready: bool = True,
        create_error: Exception | None = None,
    ) -> None:
        self.machine = target
        self.container_snapshots = list(container_snapshots or [[]])
        self.result = result or ResultObservation("absent")
        self.readiness = readiness or ResultObservation("absent")
        self.image_ready = image_ready
        self.create_error = create_error
        self.sync_calls: list[dict[str, str]] = []
        self.payload_calls: list[tuple[str, dict[str, object]]] = []
        self.create_calls: list[dict[str, object]] = []
        self.start_calls: list[str] = []
        self.stop_calls: list[tuple[str, int]] = []
        self.remove_calls: list[tuple[str, bool]] = []

    def payload_host_path(self, launch_id: str) -> str:
        return f"{self.machine.paths.payloads_dir}/{launch_id}.json"

    def output_host_path(self, launch_id: str) -> str:
        return f"{self.machine.paths.outputs_dir}/{launch_id}"

    def list_job_containers(self) -> list[JobContainer]:
        if len(self.container_snapshots) > 1:
            return self.container_snapshots.pop(0)
        return self.container_snapshots[0]

    def ensure_runtime_image(self, _runtime_image_ref: str) -> bool:
        return self.image_ready

    def write_payload(self, launch_id: str, payload) -> None:
        self.payload_calls.append((launch_id, dict(payload)))

    def create_train_container(self, **kwargs) -> HostOperationResult:
        if self.create_error is not None:
            raise self.create_error
        self.create_calls.append(dict(kwargs))
        return HostOperationResult(True)

    def start_container(self, container_name: str) -> HostOperationResult:
        self.start_calls.append(container_name)
        return HostOperationResult(True)

    def stop_container(self, container_name: str, *, grace_seconds: int) -> HostOperationResult:
        self.stop_calls.append((container_name, grace_seconds))
        return HostOperationResult(True)

    def remove_container(self, container_name: str, *, force: bool = False) -> HostOperationResult:
        self.remove_calls.append((container_name, force))
        return HostOperationResult(True)

    def observe_result(self, _output_uri: str) -> ResultObservation:
        return self.result

    def observe_readiness(self, _output_uri: str) -> ResultObservation:
        return self.readiness

    def sync_shared_env(self, values) -> None:
        self.sync_calls.append(dict(values))


class StableLaunchTests(unittest.TestCase):
    def test_launch_identity_is_deterministic(self) -> None:
        target = machine()
        host = FakeHost(target)
        self.assertEqual(fleet.job_container_name(target, launch_id="train-12"), "rlab-job-beast-test-train-train-12")
        self.assertEqual(host.payload_host_path("train-12"), "/host/rlab/payloads/train-12.json")
        self.assertEqual(host.output_host_path("train-12"), "/host/rlab/outputs/train-12")

    def test_create_then_inspect_then_start_same_container(self) -> None:
        target = machine()
        job = {
            "id": 12,
            "goal_slug": "goal",
            "runtime_image_ref": RUNTIME_IMAGE_REF,
            "machine": target.name,
            "train_config": {},
        }
        created = container(state="created")
        running = container(state="running")
        host = FakeHost(target, container_snapshots=[[created], [running]])
        with (
            mock.patch.object(fleet, "_load_train_job", return_value=job),
            mock.patch.object(fleet, "job_payload_for_launch", return_value={"job_id": 12}),
            mock.patch.object(fleet, "mark_job_launch_running") as mark_running,
        ):
            changed = fleet._start_or_resume_launch(FakeConnection(), host, launch=launch())

        self.assertTrue(changed)
        self.assertEqual(host.create_calls[0]["launch_id"], "train-12")
        self.assertEqual(host.start_calls, ["rlab-job-beast-test-train-train-12"])
        mark_running.assert_called_once()

    def test_timeout_after_create_records_error_without_new_identity(self) -> None:
        target = machine()
        job = {"id": 12, "runtime_image_ref": RUNTIME_IMAGE_REF, "machine": target.name}
        host = FakeHost(
            target,
            create_error=MachineCommandTimeout(target.name, 120),
        )
        with (
            mock.patch.object(fleet, "_load_train_job", return_value=job),
            mock.patch.object(fleet, "job_payload_for_launch", return_value={}),
            mock.patch.object(fleet, "record_job_launch_error") as record_error,
        ):
            changed = fleet._start_or_resume_launch(FakeConnection(), host, launch=launch())

        self.assertFalse(changed)
        self.assertEqual(record_error.call_args.kwargs["launch_id"], "train-12")


class ReconciliationTests(unittest.TestCase):
    def test_running_container_becomes_job_running_only_after_readiness(self) -> None:
        target = machine()
        host = FakeHost(
            target,
            container_snapshots=[[container(state="running")]],
            readiness=ResultObservation(
                "present",
                payload={
                    "schema_version": 1,
                    "ready": True,
                    "wandb_run_id": "abc123",
                    "wandb_url": "https://wandb.ai/entity/project/runs/abc123",
                },
            ),
        )
        with (
            mock.patch.object(fleet, "active_job_launches", return_value=[launch()]),
            mock.patch.object(fleet, "mark_job_launch_running") as mark_starting,
            mock.patch.object(fleet, "mark_train_job_ready", return_value={"id": 12}) as mark_ready,
        ):
            changed = fleet.reconcile_machine_launches(FakeConnection(), host)

        self.assertEqual(changed, 2)
        mark_starting.assert_called_once()
        mark_ready.assert_called_once()

    def test_unreachable_never_proves_absence_or_relaunches(self) -> None:
        target = machine()
        host = FakeHost(target, result=ResultObservation("error", error="ssh unreachable"))
        with (
            mock.patch.object(fleet, "active_job_launches", return_value=[launch()]),
            mock.patch.object(fleet, "record_job_launch_error") as record_error,
            mock.patch.object(fleet, "_start_or_resume_launch") as start,
            mock.patch.object(fleet, "finish_job_launch_from_result") as finish,
        ):
            changed = fleet.reconcile_machine_launches(FakeConnection(), host)

        self.assertEqual(changed, 0)
        record_error.assert_called_once()
        start.assert_not_called()
        finish.assert_not_called()

    def test_authoritatively_missing_running_container_fails(self) -> None:
        target = machine()
        host = FakeHost(target, result=ResultObservation("absent"))
        with (
            mock.patch.object(fleet, "active_job_launches", return_value=[launch(state="running")]),
            mock.patch.object(fleet, "launch_cancel_requested", return_value=False),
            mock.patch.object(fleet, "finish_job_launch_from_result") as finish,
        ):
            changed = fleet.reconcile_machine_launches(FakeConnection(), host)

        self.assertEqual(changed, 1)
        self.assertEqual(finish.call_args.kwargs["result"]["status"], "failed")

    def test_cancel_running_uses_grace_period_and_terminalizes_once(self) -> None:
        target = machine()
        host = FakeHost(target)
        with mock.patch.object(fleet, "finish_job_launch_from_result") as finish:
            changed = fleet.cancel_running_job_launch(
                FakeConnection(), host, launch=launch(state="running"), container=container()
            )

        self.assertTrue(changed)
        self.assertEqual(host.stop_calls, [("rlab-job-beast-test-train-train-12", 120)])
        self.assertEqual(finish.call_args.kwargs["result"]["status"], "canceled")


class ServicePassTests(unittest.TestCase):
    def test_busy_machine_lock_skips_mutation(self) -> None:
        conn = FakeConnection(row={"acquired": False})
        with self.assertRaises(fleet.MachineLockBusy):
            fleet.acquire_machine_lock(conn, "beast-test")

    def test_idle_machine_does_not_sync_runner_environment(self) -> None:
        target = machine()
        host = FakeHost(target)
        with (
            mock.patch.object(fleet, "next_pending_train_job", return_value=None),
            mock.patch.object(fleet, "active_job_launches", return_value=[]),
            mock.patch.object(fleet, "reconcile_machine_launches", return_value=0),
            mock.patch.object(fleet, "launch_next_jobs", return_value=0),
        ):
            result = fleet.run_reconcile_fill_pass(FakeConnection(), host=host)

        self.assertEqual(result, (0, 0))
        self.assertEqual(host.sync_calls, [])

    def test_unavailable_image_leaves_pending_job_unclaimed(self) -> None:
        target = machine()
        host = FakeHost(target, image_ready=False)
        pending = {"id": 12, "runtime_image_ref": RUNTIME_IMAGE_REF}
        with (
            mock.patch.object(fleet, "machine_control", return_value={"drained": False}),
            mock.patch.object(fleet, "train_container_slot_usage", return_value=(0, 2, 2)),
            mock.patch.object(fleet, "next_pending_train_job", return_value=pending),
            mock.patch.object(fleet, "claim_job_launch") as claim,
        ):
            launched = fleet.launch_next_jobs(FakeConnection(), host=host)

        self.assertEqual(launched, 0)
        claim.assert_not_called()

if __name__ == "__main__":
    unittest.main()
