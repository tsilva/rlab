from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

from rlab import fleet
from rlab.machines import load_machine_registry, resolve_machine
from rlab.runtime_refs import runtime_image_ref_from_file


RUNTIME_IMAGE_REF = (
    "docker:ghcr.io/tsilva/rlab/rlab-train@sha256:"
    "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
)
OTHER_IMAGE_REF = (
    "docker:ghcr.io/tsilva/rlab/rlab-train@sha256:"
    "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"
)


class FakeCursor:
    def __init__(self, rows=None, row=None) -> None:
        self.rows = rows if rows is not None else []
        self.row = row
        self.executed_sql = ""
        self.executed_params = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def execute(self, sql, params=None) -> None:
        self.executed_sql = sql
        self.executed_params = params or {}

    def fetchall(self):
        return self.rows

    def fetchone(self):
        return self.row if self.row is not None else (self.rows[0] if self.rows else None)


class FakeConnection:
    def __init__(self, rows=None, row=None) -> None:
        self.cursor_obj = FakeCursor(rows=rows, row=row)

    def cursor(self):
        return self.cursor_obj

    def close(self):
        pass


def write_registry(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "machines": {
                    "beast-test": {
                        "backend": "local_docker",
                        "run_target": "rtx4090",
                        "pull_policy": "never",
                        "limits": {"max_parallel_containers": 2},
                        "paths": {
                            "host_root": "/tmp/rlab",
                            "payloads_dir": "/tmp/rlab/payloads",
                            "outputs_dir": "/tmp/rlab/outputs",
                            "roms_dir": "/roms",
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )


def sample_registry_path(root: Path) -> Path:
    (root / "experiments").mkdir()
    path = root / "experiments" / "machines.yaml"
    path.write_text(
        json.dumps(
            {
                "machines": {
                    "beast-3": {
                        "backend": "docker_ssh",
                        "ssh_target": "tsilva@beast-3",
                        "run_target": "rtx4090",
                        "docker": {"command": ["sudo", "-n", "docker"]},
                        "limits": {"max_parallel_containers": 5, "max_train_containers": 5},
                        "paths": {"roms_dir": "/roms-host"},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def sample_registry():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        return load_machine_registry(sample_registry_path(root))


class FleetQueueTests(unittest.TestCase):
    def test_queue_demands_groups_by_runtime_digest_and_target(self) -> None:
        conn = FakeConnection(
            rows=[
                {
                    "runtime_image_ref": RUNTIME_IMAGE_REF,
                    "run_target": "rtx4090",
                    "pending_count": 2,
                    "running_count": 1,
                    "oldest_job_id": 7,
                }
            ]
        )

        rows = fleet.queue_demands(conn)

        self.assertEqual(rows[0].runtime_image_ref, RUNTIME_IMAGE_REF)
        self.assertEqual(rows[0].pending_count, 2)
        self.assertEqual(rows[0].running_count, 1)
        self.assertIn("GROUP BY runtime_image_ref, run_target", conn.cursor_obj.executed_sql)

    def test_format_demands_omits_profiles(self) -> None:
        output = fleet.format_demands(
            [
                fleet.QueueDemand(
                    runtime_image_ref=RUNTIME_IMAGE_REF,
                    run_target="rtx4090",
                    pending_count=1,
                    running_count=0,
                    oldest_job_id=7,
                )
            ]
        )

        self.assertIn("target=rtx4090", output)
        self.assertNotIn("profile=", output)


class FleetHostTests(unittest.TestCase):
    def test_setup_host_uses_configured_sudo_docker_command(self) -> None:
        registry = sample_registry()

        script = fleet.setup_host_script(resolve_machine(registry, "beast-3"))

        self.assertIn("sudo -n docker info", script)
        self.assertIn("/roms-host", script)

    def test_runtime_image_ref_file_accepts_ci_json_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rlab-train-image.json"
            path.write_text(json.dumps({"runtime_image_ref": RUNTIME_IMAGE_REF}), encoding="utf-8")

            self.assertEqual(runtime_image_ref_from_file(path), RUNTIME_IMAGE_REF)

    def test_capacity_policy_renders_missing_container_limit_plainly(self) -> None:
        registry = sample_registry()
        policy = {"lanes": [{"name": "beast", "manager": "rlab_fleet", "host": "beast-3"}]}

        fleet.validate_capacity_policy(policy, registry)
        rendered = fleet.format_capacity_policy(policy)

        self.assertIn("max_train_containers=None", rendered)

    def test_capacity_policy_accepts_local_docker_fleet_machine(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "machines.yaml"
            write_registry(path)
            registry = load_machine_registry(path)
        policy = {
            "lanes": [
                {
                    "name": "localhost-smoke",
                    "manager": "rlab_fleet",
                    "host": "beast-test",
                    "max_train_containers": 1,
                }
            ]
        }

        fleet.validate_capacity_policy(policy, registry)


class JobContainerTests(unittest.TestCase):
    def test_job_container_run_command_uses_run_job_payload_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "machines.yaml"
            write_registry(path)
            machine = resolve_machine(load_machine_registry(path), "beast-test")

        command = fleet.job_container_run_command(
            machine,
            job_kind="train",
            job_id=12,
            launch_id="train-12-abc",
            runtime_image_ref=RUNTIME_IMAGE_REF,
            container_name="rlab-job-beast-test-train-train-12-abc",
        )
        text = fleet.shell_join(command)

        self.assertIn("rlab-container-entrypoint rlab run-job", text)
        self.assertIn("--payload /input/payloads/train-12-abc.json", text)
        self.assertIn("--label rlab.job-container=true", text)
        self.assertNotIn("--gpus all", text)

    def test_launch_claims_only_machine_run_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "machines.yaml"
            write_registry(path)
            machine = resolve_machine(load_machine_registry(path), "beast-test")

        with mock.patch.object(fleet, "claim_job_launch", return_value=None) as claim:
            launched = fleet.launch_claimed_job_container(
                FakeConnection(),
                machine=machine,
                job_kind="train",
            )

        self.assertIsNone(launched)
        self.assertEqual(claim.call_args.kwargs["run_target"], "rtx4090")

    def test_parse_job_containers_filters_to_job_container_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "machines.yaml"
            write_registry(path)
            machine = resolve_machine(load_machine_registry(path), "beast-test")
        output = "\n".join(
            [
                json.dumps(
                    {
                        "Names": "current",
                        "State": "running",
                        "Status": "Up",
                        "Labels": "rlab.job-container=true,rlab.launch-id=train-12-abc",
                    }
                ),
                json.dumps({"Names": "other", "State": "running", "Labels": "rlab.managed=true"}),
            ]
        )

        containers = fleet.parse_job_containers(machine, output)

        self.assertEqual([container.name for container in containers], ["current"])
        self.assertEqual(containers[0].launch_id, "train-12-abc")

    def test_render_machine_watch_dashboard_shows_launch_hints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "machines.yaml"
            write_registry(path)
            machine = resolve_machine(load_machine_registry(path), "beast-test")
        snapshot = fleet.MachineWatchSnapshot(
            captured_at=fleet.datetime(2026, 7, 7, tzinfo=fleet.UTC),
            machine=machine,
            containers=(),
            launches=(
                {
                    "launch_id": "train-12-abc",
                    "job_kind": "train",
                    "job_id": 12,
                    "state": "launching",
                    "output_uri": "/tmp/rlab/outputs/train-12-abc",
                },
            ),
            queue_counts={"train": {"pending": 1, "launching": 1, "running": 0}},
            result_present={"train-12-abc": False},
        )

        output = fleet.render_machine_watch_dashboard(snapshot, color=False)

        self.assertIn("needs_shepherd_release", output)
        self.assertIn("train-12-abc", output)


class RuntimeImagePruneTests(unittest.TestCase):
    def test_stale_runtime_image_plan_preserves_active_containers_and_demand(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "machines.yaml"
            write_registry(path)
            machine = resolve_machine(load_machine_registry(path), "beast-test")
        images = (
            fleet.RuntimeHostImage(
                machine="beast-test",
                repository="ghcr.io/tsilva/rlab/rlab-train",
                digest=RUNTIME_IMAGE_REF.removeprefix("docker:ghcr.io/tsilva/rlab/rlab-train@"),
                image_id="keep-demand",
            ),
            fleet.RuntimeHostImage(
                machine="beast-test",
                repository="ghcr.io/tsilva/rlab/rlab-train",
                digest=OTHER_IMAGE_REF.removeprefix("docker:ghcr.io/tsilva/rlab/rlab-train@"),
                image_id="stale",
            ),
        )
        demands = [
            fleet.QueueDemand(
                runtime_image_ref=RUNTIME_IMAGE_REF,
                run_target="rtx4090",
                pending_count=1,
                running_count=0,
                oldest_job_id=7,
            )
        ]

        stale = fleet.stale_runtime_host_images(
            machine=machine,
            images=images,
            demands=demands,
            containers=(),
            job_kind="train",
        )

        self.assertEqual([image.runtime_image_ref for image in stale], [OTHER_IMAGE_REF])

    def test_shepherd_busy_lock_returns_distinct_status(self) -> None:
        args = Namespace(
            machine="beast-test",
            execute=True,
            no_color=True,
            once=True,
            interval=1,
            job_kind="train",
            limit=1,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "machines.yaml"
            write_registry(path)
            registry = load_machine_registry(path)
        conn = mock.Mock()

        with (
            mock.patch.object(fleet, "load_registry_from_args", return_value=registry),
            mock.patch.object(fleet, "_connect_from_args", return_value=conn),
            mock.patch.object(fleet, "acquire_shepherd_lock", side_effect=fleet.ShepherdLockBusy("beast-test")),
            mock.patch("builtins.print"),
        ):
            status = fleet.cmd_container_shepherd(args)

        self.assertEqual(status, 2)
        conn.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
