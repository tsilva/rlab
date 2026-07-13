from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from rlab import fleet
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


def container(*, state: str = "running") -> fleet.JobContainer:
    return fleet.JobContainer(
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


class StableLaunchTests(unittest.TestCase):
    def test_launch_identity_is_deterministic(self) -> None:
        target = machine()
        self.assertEqual(fleet.job_container_name(target, launch_id="train-12"), "rlab-job-beast-test-train-train-12")
        self.assertEqual(fleet.launch_payload_path(target, "train-12"), "/host/rlab/payloads/train-12.json")
        self.assertEqual(fleet.launch_output_path(target, "train-12"), "/host/rlab/outputs/train-12")

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
        shell_result = SimpleNamespace(returncode=0, stdout="", stderr="")
        with (
            mock.patch.object(fleet, "_load_train_job", return_value=job),
            mock.patch.object(fleet, "job_payload_for_launch", return_value={"job_id": 12}),
            mock.patch.object(fleet, "ensure_runtime_image_available", return_value=True),
            mock.patch.object(fleet, "write_remote_payload"),
            mock.patch.object(fleet, "run_machine_shell", return_value=shell_result) as run_shell,
            mock.patch.object(fleet, "run_machine_docker", return_value=shell_result) as run_docker,
            mock.patch.object(fleet, "list_job_containers", side_effect=[[created], [running]]),
            mock.patch.object(fleet, "mark_job_launch_running") as mark_running,
        ):
            changed = fleet._start_or_resume_launch(FakeConnection(), target, launch=launch())

        self.assertTrue(changed)
        self.assertIn("docker create", run_shell.call_args.args[1])
        run_docker.assert_called_once_with(
            target,
            ["start", "rlab-job-beast-test-train-train-12"],
            capture=True,
        )
        mark_running.assert_called_once()

    def test_timeout_after_create_records_error_without_new_identity(self) -> None:
        target = machine()
        job = {"id": 12, "runtime_image_ref": RUNTIME_IMAGE_REF, "machine": target.name}
        with (
            mock.patch.object(fleet, "_load_train_job", return_value=job),
            mock.patch.object(fleet, "job_payload_for_launch", return_value={}),
            mock.patch.object(fleet, "ensure_runtime_image_available", return_value=True),
            mock.patch.object(fleet, "write_remote_payload"),
            mock.patch.object(
                fleet,
                "run_machine_shell",
                side_effect=fleet.MachineCommandTimeout(target.name, 120),
            ),
            mock.patch.object(fleet, "record_job_launch_error") as record_error,
        ):
            changed = fleet._start_or_resume_launch(FakeConnection(), target, launch=launch())

        self.assertFalse(changed)
        self.assertEqual(record_error.call_args.kwargs["launch_id"], "train-12")


class ReconciliationTests(unittest.TestCase):
    def test_unreachable_never_proves_absence_or_relaunches(self) -> None:
        target = machine()
        with (
            mock.patch.object(fleet, "active_job_launches", return_value=[launch()]),
            mock.patch.object(fleet, "list_job_containers", return_value=[]),
            mock.patch.object(
                fleet,
                "observe_remote_result",
                return_value=fleet.ResultObservation("error", error="ssh unreachable"),
            ),
            mock.patch.object(fleet, "record_job_launch_error") as record_error,
            mock.patch.object(fleet, "_start_or_resume_launch") as start,
            mock.patch.object(fleet, "finish_job_launch_from_result") as finish,
        ):
            changed = fleet.reconcile_machine_launches(FakeConnection(), target)

        self.assertEqual(changed, 0)
        record_error.assert_called_once()
        start.assert_not_called()
        finish.assert_not_called()

    def test_authoritatively_missing_running_container_fails(self) -> None:
        target = machine()
        with (
            mock.patch.object(fleet, "active_job_launches", return_value=[launch(state="running")]),
            mock.patch.object(fleet, "list_job_containers", return_value=[]),
            mock.patch.object(
                fleet,
                "observe_remote_result",
                return_value=fleet.ResultObservation("absent"),
            ),
            mock.patch.object(fleet, "launch_cancel_requested", return_value=False),
            mock.patch.object(fleet, "finish_job_launch_from_result") as finish,
        ):
            changed = fleet.reconcile_machine_launches(FakeConnection(), target)

        self.assertEqual(changed, 1)
        self.assertEqual(finish.call_args.kwargs["result"]["status"], "failed")

    def test_cancel_running_uses_grace_period_and_terminalizes_once(self) -> None:
        target = machine()
        shell_result = SimpleNamespace(returncode=0, stdout="", stderr="")
        with (
            mock.patch.object(fleet, "run_machine_docker", return_value=shell_result) as stop,
            mock.patch.object(fleet, "finish_job_launch_from_result") as finish,
        ):
            changed = fleet.cancel_running_job_launch(
                FakeConnection(), target, launch=launch(state="running"), container=container()
            )

        self.assertTrue(changed)
        stop.assert_called_once_with(
            target,
            ["stop", "--time", "120", "rlab-job-beast-test-train-train-12"],
            capture=True,
            timeout=fleet.DOCKER_STOP_TIMEOUT_SECONDS,
        )
        self.assertEqual(finish.call_args.kwargs["result"]["status"], "canceled")

    def test_partial_result_is_not_accepted(self) -> None:
        target = machine()
        result = SimpleNamespace(returncode=0, stdout=json.dumps({"status": "succeeded"}), stderr="")
        with mock.patch.object(fleet, "run_machine_shell", return_value=result):
            observation = fleet.observe_remote_result(target, "/output/train-12")

        self.assertEqual(observation.state, "present")
        # Identity is checked transactionally by finish_job_launch_from_result.
        self.assertEqual(observation.payload, {"status": "succeeded"})


class ServicePassTests(unittest.TestCase):
    def test_busy_machine_lock_skips_mutation(self) -> None:
        conn = FakeConnection(row={"acquired": False})
        with self.assertRaises(fleet.MachineLockBusy):
            fleet.acquire_machine_lock(conn, "beast-test")

    def test_idle_machine_does_not_sync_runner_environment(self) -> None:
        target = machine()
        with (
            mock.patch.object(fleet, "next_pending_train_job", return_value=None),
            mock.patch.object(fleet, "active_job_launches", return_value=[]),
            mock.patch.object(fleet, "sync_shared_runner_env") as sync,
            mock.patch.object(fleet, "reconcile_machine_launches", return_value=0),
            mock.patch.object(fleet, "launch_next_jobs", return_value=0),
        ):
            result = fleet.run_reconcile_fill_pass(FakeConnection(), machine=target)

        self.assertEqual(result, (0, 0))
        sync.assert_not_called()

    def test_unavailable_image_leaves_pending_job_unclaimed(self) -> None:
        target = machine()
        pending = {"id": 12, "runtime_image_ref": RUNTIME_IMAGE_REF}
        with (
            mock.patch.object(fleet, "machine_control", return_value={"drained": False}),
            mock.patch.object(fleet, "train_container_slot_usage", return_value=(0, 2, 2)),
            mock.patch.object(fleet, "next_pending_train_job", return_value=pending),
            mock.patch.object(fleet, "ensure_runtime_image_available", return_value=False),
            mock.patch.object(fleet, "claim_job_launch") as claim,
        ):
            launched = fleet.launch_next_jobs(FakeConnection(), machine=target)

        self.assertEqual(launched, 0)
        claim.assert_not_called()

    def test_lane_deadline_caps_remote_operation_timeout(self) -> None:
        target = machine()
        deadline = time.monotonic() + 2
        token = fleet._MACHINE_LANE_DEADLINE.set(deadline)
        try:
            with mock.patch.object(
                fleet.subprocess,
                "run",
                return_value=SimpleNamespace(returncode=0, stdout="", stderr=""),
            ) as run:
                fleet.run_machine_shell(target, "true", timeout=900)
        finally:
            fleet._MACHINE_LANE_DEADLINE.reset(token)

        self.assertLessEqual(run.call_args.kwargs["timeout"], 2)


if __name__ == "__main__":
    unittest.main()
