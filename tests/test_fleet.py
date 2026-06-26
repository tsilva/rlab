from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rlab import fleet
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
    def __init__(self, rows) -> None:
        self.rows = rows
        self.executed_sql = ""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def execute(self, sql, params=None) -> None:
        self.executed_sql = sql

    def fetchall(self):
        return self.rows


class FakeConnection:
    def __init__(self, rows) -> None:
        self.cursor_obj = FakeCursor(rows)

    def cursor(self):
        return self.cursor_obj


def sample_config() -> fleet.FleetConfig:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "experiments").mkdir()
        (root / "experiments" / "instances.json").write_text(
            json.dumps(
                {
                    "instances": {
                        "rtx4090": {"name": "rtx4090", "children": 5, "max_children": 5},
                        "rtx2060": {"name": "rtx2060", "children": 4, "max_children": 4},
                    }
                }
            ),
            encoding="utf-8",
        )
        (root / "experiments" / "fleet.json").write_text(
            json.dumps(
                {
                    "hosts": {
                        "beast-3": {
                            "ssh_target": "tsilva@beast-3",
                            "run_target": "rtx4090",
                            "max_workers": 5,
                            "rom_dir": "/roms-host",
                        },
                        "beast-2": {
                            "ssh_target": "tsilva@192.168.133.26",
                            "ssh_options": ["-o", "HostKeyAlias=beast-2"],
                            "run_target": "rtx2060",
                            "max_workers": 4,
                        },
                    },
                    "profile_policies": [{"profile_id": "*", "hosts": ["beast-3", "beast-2"]}],
                }
            ),
            encoding="utf-8",
        )
        return fleet.load_fleet_config(root)


def demand(
    *,
    profile: str = "mario-ppo/post21/rtx4090",
    image: str = RUNTIME_IMAGE_REF,
    target: str | None = "rtx4090",
    pending: int = 1,
    running: int = 0,
    priority: int = 0,
    oldest: int = 10,
) -> fleet.QueueDemand:
    return fleet.QueueDemand(
        profile_id=profile,
        runtime_image_ref=image,
        run_target=target,
        pending_count=pending,
        running_count=running,
        max_priority=priority,
        oldest_job_id=oldest,
    )


class FleetQueueTests(unittest.TestCase):
    def test_queue_demands_groups_by_profile_digest_and_target(self) -> None:
        conn = FakeConnection(
            [
                {
                    "profile_id": "profile-a",
                    "runtime_image_ref": RUNTIME_IMAGE_REF,
                    "run_target": "rtx4090",
                    "pending_count": 2,
                    "running_count": 1,
                    "max_priority": 3,
                    "oldest_job_id": 7,
                }
            ]
        )

        rows = fleet.queue_demands(conn)

        self.assertEqual(rows[0].profile_id, "profile-a")
        self.assertEqual(rows[0].pending_count, 2)
        self.assertEqual(rows[0].running_count, 1)
        self.assertIn("GROUP BY profile_id, runtime_image_ref, run_target", conn.cursor_obj.executed_sql)
        self.assertIn("status IN ('pending', 'running')", conn.cursor_obj.executed_sql)

    def test_runtime_image_ref_file_accepts_ci_json_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rlab-train-image.json"
            path.write_text(
                json.dumps({"runtime_image_ref": RUNTIME_IMAGE_REF}),
                encoding="utf-8",
            )

            self.assertEqual(runtime_image_ref_from_file(path), RUNTIME_IMAGE_REF)

    def test_runtime_image_ref_file_rejects_mutable_tags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "image.txt"
            path.write_text("docker:ghcr.io/tsilva/rlab/rlab-train:latest", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "immutable docker digest ref"):
                runtime_image_ref_from_file(path)


class FleetPlanTests(unittest.TestCase):
    def test_start_action_renders_digest_pinned_docker_runner(self) -> None:
        config = sample_config()
        plan = fleet.build_fleet_plan(config, [demand(pending=3)], [], [])

        start = next(action for action in plan.actions if action.kind == "start")
        command_text = "\n".join(start.commands)

        self.assertIn("docker pull ghcr.io/tsilva/rlab/rlab-train@sha256:", command_text)
        self.assertIn("docker run -d", command_text)
        self.assertIn("--gpus all", command_text)
        self.assertIn("--env-file /home/tsilva/rlab/.env.runner", command_text)
        self.assertIn("--label rlab.managed=true", command_text)
        self.assertIn("--label rlab.runtime-image-ref=", command_text)
        self.assertIn("rlab-container-entrypoint rlab-train-runner", command_text)
        self.assertIn("--runtime-image-ref", command_text)
        self.assertIn(RUNTIME_IMAGE_REF, command_text)
        self.assertIn("--worker-id rlab-beast-3-rtx4090", command_text)

    def test_running_matching_container_is_kept(self) -> None:
        config = sample_config()
        desired = fleet.allocate_desired_deployments(config, [demand()])[0][0]
        existing = fleet.ExistingContainer(
            host="beast-3",
            name=desired.name,
            state="running",
            status="Up 2 minutes",
            image="ghcr.io/tsilva/rlab/rlab-train",
            labels=desired.labels,
        )

        plan = fleet.build_fleet_plan(config, [demand()], [existing], [])

        self.assertTrue(any(action.kind == "keep" for action in plan.actions))
        self.assertFalse(any(action.kind == "restart" for action in plan.actions))

    def test_exited_desired_container_restarts_without_active_lease(self) -> None:
        config = sample_config()
        desired = fleet.allocate_desired_deployments(config, [demand()])[0][0]
        existing = fleet.ExistingContainer(
            host="beast-3",
            name=desired.name,
            state="exited",
            status="Exited",
            image="ghcr.io/tsilva/rlab/rlab-train",
            labels=desired.labels,
        )

        plan = fleet.build_fleet_plan(config, [demand()], [existing], [])

        restart = next(action for action in plan.actions if action.kind == "restart")
        self.assertIn("docker rm -f", "\n".join(restart.commands))

    def test_obsolete_container_is_removed_only_without_demand_or_lease(self) -> None:
        config = sample_config()
        desired = fleet.allocate_desired_deployments(config, [demand(image=OTHER_IMAGE_REF)])[0][0]
        obsolete = fleet.ExistingContainer(
            host="beast-3",
            name=desired.name.replace("dddddddddddd", "cccccccccccc"),
            state="running",
            status="Up",
            image="ghcr.io/tsilva/rlab/rlab-train",
            labels={
                **desired.labels,
                "rlab.runtime-image-ref": RUNTIME_IMAGE_REF,
                "rlab.runtime-digest": "cccccccccccc",
            },
        )

        plan = fleet.build_fleet_plan(config, [demand(image=OTHER_IMAGE_REF)], [obsolete], [])

        self.assertTrue(any(action.kind == "remove" for action in plan.actions))

    def test_active_lease_prevents_obsolete_container_removal(self) -> None:
        config = sample_config()
        desired = fleet.allocate_desired_deployments(config, [demand()])[0][0]
        obsolete = fleet.ExistingContainer(
            host="beast-3",
            name=desired.name,
            state="running",
            status="Up",
            image="ghcr.io/tsilva/rlab/rlab-train",
            labels=desired.labels,
        )
        lease = fleet.ActiveLease(
            lease_owner=f"{desired.worker_prefix}-0-deadbeef",
            profile_id=desired.key.profile_id,
            runtime_image_ref=desired.key.runtime_image_ref,
            run_target=desired.key.run_target,
            running_count=1,
        )

        plan = fleet.build_fleet_plan(config, [], [obsolete], [lease])

        self.assertFalse(any(action.kind == "remove" for action in plan.actions))
        self.assertTrue(any("active lease" in warning for warning in plan.warnings))

    def test_capacity_overflow_is_reported_without_queue_mutation(self) -> None:
        config = sample_config()
        demands = [
            demand(profile=f"profile-{index}", pending=3, priority=10 - index, oldest=index)
            for index in range(3)
        ]

        desired, warnings = fleet.allocate_desired_deployments(config, demands)

        self.assertLessEqual(sum(item.workers for item in desired), config.hosts["beast-3"].max_workers)
        self.assertTrue(any("capacity" in warning or "partially allocated" in warning for warning in warnings))

    def test_host_filter_limits_desired_state_to_one_host(self) -> None:
        config = fleet.filter_config_to_host(sample_config(), "beast-2")

        desired, warnings = fleet.allocate_desired_deployments(
            config,
            [demand(target=None, pending=1)],
        )

        self.assertEqual(warnings, ())
        self.assertEqual(len(desired), 1)
        self.assertEqual(desired[0].key.host, "beast-2")


class FleetHostSetupTests(unittest.TestCase):
    def test_default_fleet_config_encodes_beast_host_setup(self) -> None:
        config = fleet.load_fleet_config(Path(".").resolve())

        self.assertEqual(config.hosts["beast-3"].ssh_target, "tsilva@beast-3")
        self.assertEqual(config.hosts["beast-3"].run_target, "rtx4090")
        self.assertEqual(config.hosts["beast-3"].max_workers, 5)
        self.assertEqual(config.hosts["beast-2"].ssh_target, "tsilva@192.168.133.26")
        self.assertIn("HostKeyAlias=beast-2", config.hosts["beast-2"].ssh_options)
        self.assertEqual(config.hosts["beast-2"].run_target, "rtx2060")
        self.assertEqual(config.hosts["beast-2"].max_workers, 4)

    def test_setup_host_script_verifies_docker_nvidia_and_digest_smoke(self) -> None:
        config = sample_config()
        script = fleet.setup_host_script(
            config.hosts["beast-3"],
            runtime_image_ref=RUNTIME_IMAGE_REF,
        )

        self.assertIn("command -v docker", script)
        self.assertIn("nvidia-smi", script)
        self.assertIn("docker run --rm --gpus all", script)
        self.assertIn("nvidia-container-toolkit", script)
        self.assertIn("TRAIN_QUEUE_DATABASE_URL=", script)
        self.assertIn("docker pull ghcr.io/tsilva/rlab/rlab-train@sha256:", script)
        self.assertIn("rlab-container-entrypoint rlab-container-smoke", script)

    def test_systemd_script_runs_host_local_remote_reconcile_watch(self) -> None:
        config = sample_config()
        script = fleet.systemd_install_script(config.hosts["beast-3"], interval=30)

        self.assertIn("rlab-fleet remote-reconcile --host beast-3 --execute --watch", script)
        self.assertIn("Restart=always", script)
        self.assertIn("systemctl --user enable --now rlab-fleet-beast-3.service", script)


if __name__ == "__main__":
    unittest.main()
