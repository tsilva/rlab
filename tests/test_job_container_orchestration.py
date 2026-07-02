from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from rlab import fleet, job_queue, run_job
from rlab.machines import load_machine_registry, resolve_machine


RUNTIME_IMAGE_REF = (
    "docker:ghcr.io/tsilva/rlab/rlab-train@sha256:"
    "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
)


class FakeCursor:
    def __init__(self, row=None, rows=None) -> None:
        self.row = row
        self.rows = rows or []
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
        return self.row

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


def write_registry(path: Path) -> None:
    path.write_text(
        """
machines:
  beast-test:
    backend: docker_ssh
    ssh_target: tsilva@beast-test
    run_target: rtx4090
    docker:
      command: ["sudo", "-n", "docker"]
    limits:
      max_parallel_containers: 5
      max_train_containers: 4
    paths:
      host_root: /host/rlab
      payloads_dir: /host/rlab/payloads
      outputs_dir: /host/rlab/outputs
      logs_dir: /host/rlab/logs
      roms_dir: /host/roms
      env_file: /host/rlab/.env.runner
""",
        encoding="utf-8",
    )


class MachineRegistryTests(unittest.TestCase):
    def test_machine_registry_parses_container_slot_caps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "machines.yaml"
            write_registry(path)

            machine = resolve_machine(load_machine_registry(path), "beast-test")

        self.assertEqual(machine.backend, "docker_ssh")
        self.assertEqual(machine.max_containers_for_kind("train"), 4)
        self.assertEqual(machine.max_containers_for_kind("other"), 5)
        self.assertEqual(machine.paths.container_payloads_dir, "/input/payloads")

    def test_job_container_run_command_uses_dumb_run_job_entrypoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "machines.yaml"
            write_registry(path)
            machine = resolve_machine(load_machine_registry(path), "beast-test")

        command = fleet.job_container_run_command(
            machine,
            job_kind="train",
            job_id=12,
            launch_id="train-12-abcdef",
            runtime_image_ref=RUNTIME_IMAGE_REF,
            container_name="rlab-job-test",
        )
        text = " ".join(command)

        self.assertIn("run-job", command)
        self.assertIn("--payload", command)
        self.assertIn("--output-dir", command)
        self.assertIn(f"{fleet.JOB_CONTAINER_LABEL}=true", text)
        self.assertIn(f"{fleet.JOB_ID_LABEL}=12", text)
        self.assertNotIn("DATABASE_URL", text)


class FleetShepherdSplitTests(unittest.TestCase):
    def test_parser_exposes_shepherd_command(self) -> None:
        args = fleet.build_parser().parse_args(
            ["shepherd", "--machine", "beast-test", "--limit", "5", "--once"]
        )

        self.assertIs(args.func, fleet.cmd_container_shepherd)
        self.assertEqual(args.machine, "beast-test")
        self.assertEqual(args.limit, 5)

    def test_machine_watch_is_read_only_dashboard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "machines.yaml"
            write_registry(path)
            machine = resolve_machine(load_machine_registry(path), "beast-test")
            snapshot = fleet.MachineWatchSnapshot(
                captured_at=fleet.datetime(2026, 7, 1, tzinfo=fleet.UTC),
                machine=machine,
                containers=(),
                launches=(),
                queue_counts={"train": {"pending": 2}},
                result_present={},
            )
            args = fleet.build_parser().parse_args(
                [
                    "watch",
                    "--machines",
                    str(path),
                    "--machine",
                    "beast-test",
                    "--once",
                    "--no-tui",
                    "--no-color",
                ]
            )

            stdout = io.StringIO()
            with (
                contextlib.redirect_stdout(stdout),
                mock.patch.object(fleet, "build_machine_watch_snapshot", return_value=snapshot),
                mock.patch.object(fleet, "reconcile_machine_launches") as reconcile,
                mock.patch.object(fleet, "launch_claimed_job_container") as launch,
                mock.patch.object(fleet, "release_job_launch") as release,
                mock.patch.object(fleet, "finish_job_launch_from_result") as finish,
                mock.patch.object(fleet, "mark_job_launch_running") as mark_running,
            ):
                status = fleet.cmd_watch_latest(args)

        self.assertEqual(status, 0)
        self.assertIn("mode=read-only", stdout.getvalue())
        reconcile.assert_not_called()
        launch.assert_not_called()
        release.assert_not_called()
        finish.assert_not_called()
        mark_running.assert_not_called()

    def test_machine_watch_snapshot_renderer_marks_recovery_hints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "machines.yaml"
            write_registry(path)
            machine = resolve_machine(load_machine_registry(path), "beast-test")
            containers = (
                fleet.JobContainer(
                    machine="beast-test",
                    name="rlab-job-live",
                    state="running",
                    status="Up",
                    labels={
                        fleet.JOB_CONTAINER_LABEL: "true",
                        fleet.LAUNCH_ID_LABEL: "launch-live",
                        fleet.JOB_KIND_LABEL: "train",
                        fleet.JOB_ID_LABEL: "1",
                    },
                ),
                fleet.JobContainer(
                    machine="beast-test",
                    name="rlab-job-orphan",
                    state="exited",
                    status="Exited",
                    labels={
                        fleet.JOB_CONTAINER_LABEL: "true",
                        fleet.LAUNCH_ID_LABEL: "launch-orphan",
                        fleet.JOB_KIND_LABEL: "train",
                        fleet.JOB_ID_LABEL: "2",
                        fleet.OUTPUT_URI_LABEL: "/out/launch-orphan",
                    },
                ),
            )
            snapshot = fleet.MachineWatchSnapshot(
                captured_at=fleet.datetime(2026, 7, 1, tzinfo=fleet.UTC),
                machine=machine,
                containers=containers,
                launches=(
                    {
                        "launch_id": "launch-live",
                        "job_kind": "train",
                        "job_id": 1,
                        "state": "running",
                        "output_uri": "/out/launch-live",
                    },
                    {
                        "launch_id": "launch-missing",
                        "job_kind": "train",
                        "job_id": 3,
                        "state": "launching",
                        "output_uri": "/out/launch-missing",
                    },
                ),
                queue_counts={"train": {"pending": 4, "running": 1}},
                result_present={"launch-missing": True, "launch-orphan": True},
            )

        output = fleet.render_machine_watch_dashboard(snapshot)

        self.assertIn("capacity total=1/5 train=1/4", output)
        self.assertIn("train_pending=4", output)
        self.assertNotIn("eval" + "_pending", output)
        self.assertIn("launch_id=launch-live", output)
        self.assertIn("hint=ok", output)
        self.assertIn("launch_id=launch-missing", output)
        self.assertIn("hint=needs_shepherd_finalize", output)
        self.assertIn("orphaned_containers:", output)
        self.assertIn("launch_id=launch-orphan", output)

    def test_shepherd_once_reconciles_then_fills_slots_under_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "machines.yaml"
            write_registry(path)
            args = fleet.build_parser().parse_args(
                [
                    "shepherd",
                    "--machines",
                    str(path),
                    "--machine",
                    "beast-test",
                    "--limit",
                    "3",
                    "--once",
                ]
            )
            conn = FakeConnection()
            calls: list[str] = []

            def reconcile(_conn, machine):
                calls.append(f"reconcile:{machine.name}")
                return 1

            def launch_next(_conn, *, machine, job_kind, limit, reconcile):
                calls.append(f"launch:{machine.name}:{job_kind}:{limit}:{reconcile}")
                return 2

            with (
                contextlib.redirect_stdout(io.StringIO()),
                mock.patch.object(fleet, "_connect_from_args", return_value=conn),
                mock.patch.object(
                    fleet,
                    "acquire_shepherd_lock",
                    return_value=fleet.ShepherdLock(machine="beast-test", key="lock"),
                ) as acquire,
                mock.patch.object(fleet, "release_shepherd_lock") as release,
                mock.patch.object(fleet, "reconcile_machine_launches", side_effect=reconcile),
                mock.patch.object(fleet, "launch_next_jobs", side_effect=launch_next),
            ):
                status = fleet.cmd_container_shepherd(args)

        self.assertEqual(status, 0)
        self.assertEqual(calls, ["reconcile:beast-test", "launch:beast-test:train:3:False"])
        acquire.assert_called_once()
        release.assert_called_once()
        self.assertTrue(conn.closed)

    def test_shepherd_busy_lock_exits_without_launching(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "machines.yaml"
            write_registry(path)
            args = fleet.build_parser().parse_args(
                ["shepherd", "--machines", str(path), "--machine", "beast-test", "--once"]
            )
            conn = FakeConnection()

            with (
                contextlib.redirect_stdout(io.StringIO()),
                mock.patch.object(fleet, "_connect_from_args", return_value=conn),
                mock.patch.object(
                    fleet,
                    "acquire_shepherd_lock",
                    side_effect=fleet.ShepherdLockBusy("beast-test"),
                ),
                mock.patch.object(fleet, "reconcile_machine_launches") as reconcile,
                mock.patch.object(fleet, "launch_next_jobs") as launch,
            ):
                status = fleet.cmd_container_shepherd(args)

        self.assertEqual(status, 2)
        reconcile.assert_not_called()
        launch.assert_not_called()
        self.assertTrue(conn.closed)


class LaunchLedgerTests(unittest.TestCase):
    def test_schema_defines_launch_ledger(self) -> None:
        self.assertIn("CREATE TABLE IF NOT EXISTS job_launches", job_queue.SCHEMA_SQL)
        self.assertIn("launch_id TEXT NOT NULL UNIQUE", job_queue.SCHEMA_SQL)
        self.assertIn("job_launches_machine_state_idx", job_queue.SCHEMA_SQL)
        self.assertIn("job_launches", job_queue.RESET_TABLES)

    def test_claim_job_launch_sets_launching_without_incrementing_attempts(self) -> None:
        conn = FakeConnection(
            row={
                "job_json": {
                    "id": 7,
                    "runtime_image_ref": RUNTIME_IMAGE_REF,
                    "status": "launching",
                },
                "launch_json": {
                    "launch_id": "train-7-test",
                    "job_kind": "train",
                    "job_id": 7,
                    "machine": "beast-test",
                    "backend": "docker_ssh",
                    "runtime_image_ref": RUNTIME_IMAGE_REF,
                    "output_uri": "/out/train-7-test",
                },
            }
        )

        claimed = job_queue.claim_job_launch(
            conn,
            job_kind="train",
            machine="beast-test",
            backend="docker_ssh",
            job_id=7,
            launch_id="train-7-test",
            output_uri="/out/train-7-test",
        )

        self.assertIsNotNone(claimed)
        claim_sql = conn.cursor_obj.executed_sqls[0]
        self.assertIn("status = 'launching'", claim_sql)
        self.assertNotIn("attempts = attempts + 1", claim_sql)
        self.assertEqual(conn.cursor_obj.executed_params_list[0]["launch_id"], "train-7-test")

    def test_job_payload_for_launch_excludes_db_credentials(self) -> None:
        with self.assertRaisesRegex(ValueError, "secret-like key"):
            job_queue.job_payload_for_launch(
                {"id": 1, "runtime_image_ref": RUNTIME_IMAGE_REF, "train_config": {"DATABASE_URL": "x"}},
                {
                    "launch_id": "train-1-x",
                    "job_kind": "train",
                    "machine": "beast-test",
                    "backend": "docker_ssh",
                    "runtime_image_ref": RUNTIME_IMAGE_REF,
                    "output_uri": "/out/train-1-x",
                },
            )


class RunJobCommandTests(unittest.TestCase):
    def test_train_payload_writes_result_envelope_without_db(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload_path = root / "payload.json"
            output_dir = root / "output"
            payload = {
                "schema_version": 1,
                "job_kind": "train",
                "launch_id": "train-3-test",
                "machine": "beast-test",
                "runtime_image_ref": RUNTIME_IMAGE_REF,
                "job": {
                    "id": 3,
                    "goal_slug": "Level1-1",
                    "spec_slug": "candidate",
                    "runtime_image_ref": RUNTIME_IMAGE_REF,
                    "train_config": {"game": "SuperMarioBros-Nes-v0", "run_name": "unit-run"},
                },
            }
            payload_path.write_text(json.dumps(payload), encoding="utf-8")

            with (
                mock.patch.object(
                    run_job.subprocess,
                    "run",
                    return_value=SimpleNamespace(returncode=0),
                ),
                mock.patch.object(
                    run_job,
                    "collect_result_metadata",
                    return_value={"run_name": "unit-run", "metrics_json": {"dry": True}},
                ),
            ):
                exit_code = run_job.main(["--payload", str(payload_path), "--output-dir", str(output_dir)])

            result = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(result["job_kind"], "train")
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["train"]["status"], "succeeded")
        self.assertNotIn("eval", result)


if __name__ == "__main__":
    unittest.main()
