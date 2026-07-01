from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stderr
from datetime import UTC, datetime, timedelta
from io import StringIO
from pathlib import Path
from unittest import mock

from rlab import fleet, runtime_refs
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
        (root / "experiments" / "instances.yaml").write_text(
            json.dumps(
                {
                    "instances": {
                        "rtx4090": {
                            "name": "rtx4090",
                            "default_workers": 5,
                            "hardware_max_workers": 5,
                        },
                        "rtx2060": {
                            "name": "rtx2060",
                            "default_workers": 4,
                            "hardware_max_workers": 4,
                        },
                    }
                }
            ),
            encoding="utf-8",
        )
        (root / "experiments" / "machines.yaml").write_text(
            json.dumps(
                {
                    "machines": {
                        "beast-3": {
                            "backend": "docker_ssh",
                            "ssh_target": "tsilva@beast-3",
                            "run_target": "rtx4090",
                            "docker": {"command": ["sudo", "-n", "docker"]},
                            "limits": {"max_parallel_containers": 5},
                            "paths": {"roms_dir": "/roms-host"},
                        },
                        "beast-2": {
                            "backend": "docker_ssh",
                            "ssh_target": "tsilva@192.168.133.26",
                            "ssh_options": ["-o", "HostKeyAlias=beast-2"],
                            "run_target": "rtx2060",
                            "docker": {"command": ["sudo", "-n", "docker"]},
                            "limits": {"max_parallel_containers": 4},
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
    oldest: int = 10,
) -> fleet.QueueDemand:
    return fleet.QueueDemand(
        profile_id=profile,
        runtime_image_ref=image,
        run_target=target,
        pending_count=pending,
        running_count=running,
        oldest_job_id=oldest,
    )


class GoalDiscoveryTests(unittest.TestCase):
    def test_discover_active_goal_slugs_uses_eval_runner_requirements(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            goals_dir = root / "experiments" / "goals"
            runnable_goal = goals_dir / "runnable-goal"
            inert_goal = goals_dir / "inert-goal"
            runnable_goal.mkdir(parents=True)
            inert_goal.mkdir(parents=True)
            runnable_goal.joinpath("goal.yaml").write_text(
                """
execution:
  eval_runner_requirements:
  - host: beast-3
    min_replicas: 1
    image: latest
""",
                encoding="utf-8",
            )
            inert_goal.joinpath("goal.yaml").write_text(
                """
execution: {}
""",
                encoding="utf-8",
            )

            slugs = fleet.discover_active_goal_slugs(root)

        self.assertEqual(slugs, ("runnable-goal",))


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

    def test_queue_demands_preserves_profileless_jobs_as_any_profile(self) -> None:
        conn = FakeConnection(
            [
                {
                    "profile_id": None,
                    "runtime_image_ref": RUNTIME_IMAGE_REF,
                    "run_target": "rtx4090",
                    "pending_count": 2,
                    "running_count": 0,
                    "oldest_job_id": 7,
                }
            ]
        )

        rows = fleet.queue_demands(conn)

        self.assertIsNone(rows[0].profile_id)
        self.assertIn("profile=any", fleet.format_demands(rows))

    def test_stale_lease_owner_prefix_for_host_uses_fleet_host_name(self) -> None:
        config = sample_config()

        prefix = fleet.stale_lease_owner_prefix_for_host(config.hosts["beast-2"])

        self.assertEqual(prefix, "rlab-beast-2-")

    def test_mark_stale_failed_scopes_to_host_target_and_owner_prefix(self) -> None:
        config = sample_config()
        conn = mock.Mock()
        args = Namespace(
            host="beast-2",
            lease_owner_prefix=None,
            execute=False,
            job_id=[],
            older_than_seconds=600,
            limit=0,
            error=None,
        )
        rows = [
            {
                "id": 12,
                "profile_id": None,
                "run_target": "rtx2060",
                "run_name": "candidate",
                "stale_lease_owner": "rlab-beast-2-rtx2060-any-profile-cccc-0-deadbeef",
                "stale_heartbeat_at": "2026-06-26T10:00:00Z",
            }
        ]

        with (
            mock.patch.object(fleet, "_load_config_from_args", return_value=config),
            mock.patch.object(fleet, "_connect_from_args", return_value=conn),
            mock.patch.object(fleet, "list_stale_train_jobs", return_value=rows) as list_stale,
            mock.patch("builtins.print"),
        ):
            status = fleet.cmd_mark_stale_failed(args)

        self.assertEqual(status, 0)
        list_stale.assert_called_once_with(
            conn,
            job_ids=[],
            run_target="rtx2060",
            lease_owner_prefix="rlab-beast-2-",
            older_than_seconds=600,
            limit=0,
        )
        conn.close.assert_called_once_with()

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

    def test_recent_runtime_images_reads_artifact_metadata(self) -> None:
        runs = [
            {
                "databaseId": 101,
                "headSha": "1111111111111111111111111111111111111111",
                "displayTitle": "Add fleet watcher provenance",
                "updatedAt": "2026-06-26T19:07:51Z",
            },
            {
                "databaseId": 100,
                "headSha": "2222222222222222222222222222222222222222",
                "displayTitle": "Tune train image",
                "updatedAt": "2026-06-26T18:41:12Z",
            },
        ]
        payloads = {
            "101": {
                "image": "ghcr.io/tsilva/rlab/rlab-train",
                "digest": "sha256:" + "a" * 64,
                "source_sha": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "workflow_run_id": "101",
            },
            "100": {
                "runtime_image_ref": "docker:ghcr.io/tsilva/rlab/rlab-train@sha256:" + "b" * 64,
                "source_sha": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                "workflow_run_id": "100",
            },
        }

        def fake_run(command, **kwargs):
            if command[:3] == ["gh", "run", "list"]:
                return mock.Mock(returncode=0, stdout=json.dumps(runs), stderr="")
            if command[:3] == ["gh", "run", "download"]:
                run_id = command[3]
                output_dir = Path(command[command.index("--dir") + 1])
                (output_dir / "rlab-train-image.json").write_text(
                    json.dumps(payloads[run_id]),
                    encoding="utf-8",
                )
                return mock.Mock(returncode=0, stdout="", stderr="")
            raise AssertionError(f"unexpected command: {command}")

        with mock.patch.object(runtime_refs.subprocess, "run", side_effect=fake_run):
            images = runtime_refs.recent_runtime_images(
                workflow="workflow",
                branch="main",
                artifact_name="artifact",
                limit=2,
            )

        self.assertEqual(len(images), 2)
        self.assertEqual(images[0].runtime_image_ref, "docker:ghcr.io/tsilva/rlab/rlab-train@sha256:" + "a" * 64)
        self.assertEqual(images[0].source_sha, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        self.assertEqual(images[0].commit_message, "Add fleet watcher provenance")
        self.assertEqual(images[0].published_at, "2026-06-26T19:07:51Z")

    def test_capacity_policy_formatter_lists_lanes_and_checks(self) -> None:
        text = fleet.format_capacity_policy(
            {
                "schema_version": 1,
                "updated_at": "2026-06-26",
                "purpose": "keep queue full",
                "defaults": {"runtime_image_ref": "digest"},
                "lanes": [
                    {
                        "name": "rtx4090-screening",
                        "target": "rtx4090",
                        "manager": "rlab-fleet",
                        "max_runner_workers": 5,
                        "env_threads": 4,
                    }
                ],
                "policy_checks": ["promote by eval"],
            }
        )

        self.assertIn("capacity_policy schema=1", text)
        self.assertIn("rtx4090-screening target=rtx4090", text)
        self.assertIn("max_runner_workers=5", text)
        self.assertIn("promote by eval", text)

    def test_capacity_policy_rejects_lane_above_host_cap(self) -> None:
        config = sample_config()

        with self.assertRaisesRegex(
            ValueError,
            "max_runner_workers=6 exceeds beast-3 max_workers=5",
        ):
            fleet.validate_capacity_policy(
                {
                    "lanes": [
                        {
                            "name": "rtx4090-too-wide",
                            "host": "beast-3",
                            "manager": "rlab-fleet",
                            "max_runner_workers": 6,
                        }
                    ]
                },
                config,
            )

    def test_capacity_policy_accepts_lane_at_host_cap(self) -> None:
        fleet.validate_capacity_policy(
            {
                "lanes": [
                    {
                        "name": "rtx4090-screening",
                        "host": "beast-3",
                        "manager": "rlab-fleet",
                        "max_runner_workers": 5,
                    }
                ]
            },
            sample_config(),
        )

    def test_ensure_runner_defaults_to_latest_image_ref(self) -> None:
        args = Namespace(
            image=None,
            image_file=None,
            runtime_image_ref=None,
            runtime_image_ref_file=None,
            latest_image=False,
            image_workflow="rlab train image",
            image_branch="main",
            image_artifact="rlab-train-image",
        )

        with mock.patch.object(fleet, "latest_runtime_image_ref", return_value=RUNTIME_IMAGE_REF) as latest:
            self.assertEqual(fleet.image_ref_from_args(args, default_latest=True), RUNTIME_IMAGE_REF)

        latest.assert_called_once_with(
            workflow="rlab train image",
            branch="main",
            artifact_name="rlab-train-image",
        )

    def test_ensure_runner_image_latest_uses_latest_ref(self) -> None:
        args = Namespace(
            image="latest",
            image_file=None,
            runtime_image_ref=None,
            runtime_image_ref_file=None,
            latest_image=False,
            image_workflow="rlab train image",
            image_branch="main",
            image_artifact="rlab-train-image",
        )

        with mock.patch.object(fleet, "latest_runtime_image_ref", return_value=RUNTIME_IMAGE_REF) as latest:
            self.assertEqual(fleet.image_ref_from_args(args, default_latest=True), RUNTIME_IMAGE_REF)

        latest.assert_called_once()

    def test_ensure_runner_image_digest_uses_explicit_ref(self) -> None:
        args = Namespace(
            image=RUNTIME_IMAGE_REF,
            image_file=None,
            runtime_image_ref=None,
            runtime_image_ref_file=None,
            latest_image=False,
        )

        self.assertEqual(fleet.image_ref_from_args(args, default_latest=True), RUNTIME_IMAGE_REF)


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
        self.assertIn("rlab-container-entrypoint rlab train worker", command_text)
        self.assertIn("--runtime-image-ref", command_text)
        self.assertIn(RUNTIME_IMAGE_REF, command_text)
        self.assertIn("--autoscale --min-workers 1 --max-workers 16", command_text)
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
            demand(profile=f"profile-{index}", pending=3, oldest=index)
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

    def test_ensure_runner_starts_without_queue_demand(self) -> None:
        config = fleet.filter_config_to_host(sample_config(), "beast-3")

        plan = fleet.build_ensure_runner_plan(
            config,
            host_name="beast-3",
            profile_id=None,
            runtime_image_ref=RUNTIME_IMAGE_REF,
            run_target=None,
            workers=None,
            existing=[],
            leases=[],
        )

        self.assertEqual(len(plan.desired), 1)
        self.assertEqual(plan.desired[0].workers, config.hosts["beast-3"].max_workers)
        self.assertIsNone(plan.desired[0].key.profile_id)
        self.assertEqual(plan.desired[0].key.run_target, "rtx4090")
        self.assertEqual(plan.actions[0].kind, "start")
        self.assertIn("explicit ensure-runner request", plan.actions[0].reason)
        command_text = "\n".join(plan.actions[0].commands)
        self.assertIn("docker pull ghcr.io/tsilva/rlab/rlab-train@sha256:", command_text)
        self.assertIn("--autoscale --min-workers 1 --max-workers 16", command_text)
        self.assertNotIn("--profile", command_text)

    def test_eval_deployment_command_uses_eval_worker(self) -> None:
        config = fleet.filter_config_to_host(sample_config(), "beast-3")
        desired = fleet.build_desired_deployment(
            host=config.hosts["beast-3"],
            key=fleet.DeploymentKey(
                host="beast-3",
                profile_id=None,
                runtime_image_ref=RUNTIME_IMAGE_REF,
                run_target=None,
                worker_kind=fleet.WORKER_KIND_EVAL,
                replica=0,
            ),
            workers=1,
            pending_count=0,
            running_count=0,
        )

        command_text = fleet.shell_join(fleet.docker_run_command(config.hosts["beast-3"], desired))

        self.assertIn("rlab-container-entrypoint rlab eval worker", command_text)
        self.assertNotIn("rlab train worker", command_text)
        self.assertIn("--artifact-root /root/rlab/runs/eval_artifacts", command_text)
        self.assertIn("--output-dir /root/rlab/logs/eval_runner", command_text)
        self.assertIn("--worker-id rlab-beast-3-eval-cccccccccccc-0", command_text)

    def test_goal_eval_runners_plan_starts_one_eval_container(self) -> None:
        config = fleet.filter_config_to_host(sample_config(), "beast-3")

        with mock.patch.object(
            fleet,
            "load_goal_eval_runner_requirements",
            return_value=(
                fleet.EvalRunnerRequirement(
                    goal_slug="Level1-1",
                    host="beast-3",
                    min_replicas=1,
                    image="latest",
                ),
            ),
        ):
            plan = fleet.build_goal_eval_runners_plan(
                config,
                repo_root=Path("."),
                goal_slugs=("Level1-1",),
                runtime_image_ref_for_latest=RUNTIME_IMAGE_REF,
                existing=[],
                leases=[],
            )

        self.assertEqual(len(plan.desired), 1)
        self.assertEqual(plan.desired[0].key.worker_kind, fleet.WORKER_KIND_EVAL)
        self.assertEqual(plan.desired[0].key.host, "beast-3")
        self.assertEqual(plan.desired[0].key.replica, 0)
        self.assertEqual(plan.actions[0].kind, "start")
        self.assertIn("goal-declared eval runner requirement", plan.actions[0].reason)

    def test_ensure_runner_does_not_remove_unrelated_obsolete_container(self) -> None:
        config = fleet.filter_config_to_host(sample_config(), "beast-3")
        old_desired = fleet.build_desired_deployment(
            host=config.hosts["beast-3"],
            key=fleet.DeploymentKey(
                host="beast-3",
                profile_id="mario-ppo/post21/rtx4090-prebuilt-l11-l12-50x50-v2",
                runtime_image_ref=OTHER_IMAGE_REF,
                run_target="rtx4090",
            ),
            workers=1,
            pending_count=0,
            running_count=0,
        )
        existing = fleet.ExistingContainer(
            host="beast-3",
            name=old_desired.name,
            state="running",
            status="Up",
            image="ghcr.io/tsilva/rlab/rlab-train",
            labels=old_desired.labels,
        )

        plan = fleet.build_ensure_runner_plan(
            config,
            host_name="beast-3",
            profile_id=old_desired.key.profile_id,
            runtime_image_ref=RUNTIME_IMAGE_REF,
            run_target="rtx4090",
            workers=1,
            existing=[existing],
            leases=[],
        )

        self.assertTrue(any(action.kind == "start" for action in plan.actions))
        self.assertFalse(any(action.kind == "remove" for action in plan.actions))

    def test_ensure_runner_keeps_matching_container(self) -> None:
        config = fleet.filter_config_to_host(sample_config(), "beast-3")
        desired = fleet.build_ensure_runner_plan(
            config,
            host_name="beast-3",
            profile_id="mario-ppo/post21/rtx4090-prebuilt-l11-l12-50x50-v2",
            runtime_image_ref=RUNTIME_IMAGE_REF,
            run_target="rtx4090",
            workers=2,
            existing=[],
            leases=[],
        ).desired[0]
        existing = fleet.ExistingContainer(
            host="beast-3",
            name=desired.name,
            state="running",
            status="Up",
            image="ghcr.io/tsilva/rlab/rlab-train",
            labels=desired.labels,
        )

        plan = fleet.build_ensure_runner_plan(
            config,
            host_name="beast-3",
            profile_id=desired.key.profile_id,
            runtime_image_ref=RUNTIME_IMAGE_REF,
            run_target="rtx4090",
            workers=2,
            existing=[existing],
            leases=[],
        )

        self.assertEqual(plan.actions[0].kind, "keep")

    def test_ensure_latest_starts_latest_on_each_host_and_removes_idle_old_runner(self) -> None:
        config = sample_config()
        old = fleet.build_desired_deployment(
            host=config.hosts["beast-3"],
            key=fleet.DeploymentKey(
                host="beast-3",
                profile_id=None,
                runtime_image_ref=OTHER_IMAGE_REF,
                run_target="rtx4090",
            ),
            workers=5,
            pending_count=0,
            running_count=0,
        )
        existing = fleet.ExistingContainer(
            host="beast-3",
            name=old.name,
            state="running",
            status="Up",
            image="ghcr.io/tsilva/rlab/rlab-train",
            labels=old.labels,
        )

        plan = fleet.build_ensure_latest_plan(
            config,
            runtime_image_ref=RUNTIME_IMAGE_REF,
            workers=None,
            existing=[existing],
            leases=[],
            demands=[],
        )

        starts = [action for action in plan.actions if action.kind == "start"]
        removes = [action for action in plan.actions if action.kind == "remove"]
        self.assertEqual({action.host for action in starts}, {"beast-2", "beast-3"})
        self.assertEqual(removes[0].container, old.name)
        self.assertIn("not latest baseline", removes[0].reason)

    def test_ensure_latest_keeps_old_runner_with_matching_demand(self) -> None:
        config = fleet.filter_config_to_host(sample_config(), "beast-3")
        old = fleet.build_desired_deployment(
            host=config.hosts["beast-3"],
            key=fleet.DeploymentKey(
                host="beast-3",
                profile_id=None,
                runtime_image_ref=OTHER_IMAGE_REF,
                run_target="rtx4090",
            ),
            workers=5,
            pending_count=0,
            running_count=0,
        )
        existing = fleet.ExistingContainer(
            host="beast-3",
            name=old.name,
            state="running",
            status="Up",
            image="ghcr.io/tsilva/rlab/rlab-train",
            labels=old.labels,
        )

        plan = fleet.build_ensure_latest_plan(
            config,
            runtime_image_ref=RUNTIME_IMAGE_REF,
            workers=None,
            existing=[existing],
            leases=[],
            demands=[demand(image=OTHER_IMAGE_REF, pending=1)],
        )

        self.assertFalse(any(action.kind == "remove" for action in plan.actions))
        self.assertTrue(any("matching pending/running demand" in warning for warning in plan.warnings))

    def test_ensure_latest_ignores_eval_containers(self) -> None:
        config = fleet.filter_config_to_host(sample_config(), "beast-3")
        eval_desired = fleet.build_desired_deployment(
            host=config.hosts["beast-3"],
            key=fleet.DeploymentKey(
                host="beast-3",
                profile_id=None,
                runtime_image_ref=OTHER_IMAGE_REF,
                run_target=None,
                worker_kind=fleet.WORKER_KIND_EVAL,
                replica=0,
            ),
            workers=1,
            pending_count=0,
            running_count=0,
        )
        existing = fleet.ExistingContainer(
            host="beast-3",
            name=eval_desired.name,
            state="running",
            status="Up",
            image="ghcr.io/tsilva/rlab/rlab-train",
            labels=eval_desired.labels,
        )

        plan = fleet.build_ensure_latest_plan(
            config,
            runtime_image_ref=RUNTIME_IMAGE_REF,
            workers=None,
            existing=[existing],
            leases=[],
            demands=[],
        )

        removes = [action for action in plan.actions if action.kind == "remove"]
        self.assertEqual(removes, [])

    def test_ensure_latest_keeps_old_runner_with_active_lease(self) -> None:
        config = fleet.filter_config_to_host(sample_config(), "beast-3")
        old = fleet.build_desired_deployment(
            host=config.hosts["beast-3"],
            key=fleet.DeploymentKey(
                host="beast-3",
                profile_id=None,
                runtime_image_ref=OTHER_IMAGE_REF,
                run_target="rtx4090",
            ),
            workers=5,
            pending_count=0,
            running_count=0,
        )
        existing = fleet.ExistingContainer(
            host="beast-3",
            name=old.name,
            state="running",
            status="Up",
            image="ghcr.io/tsilva/rlab/rlab-train",
            labels=old.labels,
        )
        lease = fleet.ActiveLease(
            lease_owner=f"{old.worker_prefix}-4-deadbeef",
            profile_id="mario-ppo/post21/rtx4090",
            runtime_image_ref=OTHER_IMAGE_REF,
            run_target="rtx4090",
            running_count=1,
        )

        plan = fleet.build_ensure_latest_plan(
            config,
            runtime_image_ref=RUNTIME_IMAGE_REF,
            workers=None,
            existing=[existing],
            leases=[lease],
            demands=[],
        )

        self.assertFalse(any(action.kind == "remove" for action in plan.actions))
        self.assertTrue(any("active lease" in warning for warning in plan.warnings))

    def test_watch_latest_dashboard_summarizes_hosts_actions_and_jobs(self) -> None:
        config = fleet.filter_config_to_host(sample_config(), "beast-3")
        captured_at = datetime(2026, 6, 26, 19, 15, tzinfo=UTC)
        desired = fleet.build_ensure_runner_plan(
            config,
            host_name="beast-3",
            profile_id=None,
            runtime_image_ref=RUNTIME_IMAGE_REF,
            run_target="rtx4090",
            workers=5,
            existing=[],
            leases=[],
        ).desired[0]
        container = fleet.ExistingContainer(
            host="beast-3",
            name=desired.name,
            state="running",
            status="Up 2 minutes",
            image="ghcr.io/tsilva/rlab/rlab-train",
            labels=desired.labels,
        )
        job = fleet.RunningJob(
            id=141,
            lease_owner=f"{desired.worker_prefix}-0-aabbccdd",
            profile_id=None,
            runtime_image_ref=RUNTIME_IMAGE_REF,
            run_target="rtx4090",
            run_name="b83_l11_b55post21_s23_20260626T190751Z",
            started_at=captured_at - timedelta(hours=1, minutes=5),
            heartbeat_at=captured_at - timedelta(seconds=12),
        )
        snapshot = fleet.LatestWatchSnapshot(
            captured_at=captured_at,
            config=config,
            runtime_image_ref=RUNTIME_IMAGE_REF,
            demands=(demand(profile=None, pending=0, running=1),),
            leases=(),
            jobs=(job,),
            plan=fleet.FleetPlan(
                desired=(desired,),
                existing=(container,),
                actions=(
                    fleet.FleetAction(
                        kind="keep",
                        host="beast-3",
                        container=desired.name,
                        reason="container already matches desired state",
                    ),
                ),
                warnings=(),
            ),
            recent_images=(
                fleet.RuntimeImageInfo(
                    runtime_image_ref=RUNTIME_IMAGE_REF,
                    source_sha="abcdef1234567890abcdef1234567890abcdef12",
                    commit_message="Add fleet watcher provenance",
                    published_at="2026-06-26T19:07:51Z",
                    workflow_run_id="101",
                ),
            ),
            devices=(
                {
                    "id": "rtx4090",
                    "device": "beast-3",
                    "target": "docker/beast-3",
                    "state": "busy",
                    "current_jobs": ["141"],
                    "current_job": "141",
                    "last_check": "reachable",
                    "details": {
                        "cpu": "33%",
                        "memory": "20.0/64.0 GB",
                        "gpu": "91%",
                        "vram": "15.0/24.0 GB",
                    },
                },
            ),
            execute=True,
            interval=30,
        )

        output = fleet.render_latest_watch_dashboard(snapshot, color=False, max_width=120)

        self.assertIn("rlab fleet watch", output)
        self.assertIn("mode=execute", output)
        self.assertIn("hosts live=1 down=0 queue pending=0 running=1 actions=0 stale=0", output)
        self.assertIn("latest=cccccccccccc", output)
        self.assertNotIn("sha256:cccccccccccc", output)
        self.assertNotIn("\033[", output)
        self.assertIn("recent images:", output)
        self.assertIn("abcdef123456", output)
        self.assertIn("2026-06-26 19:07Z", output)
        self.assertIn("Add fleet watcher provenance", output)
        self.assertIn("beast-3", output)
        self.assertIn("live", output)
        self.assertIn("active devices:", output)
        self.assertIn("91%", output)
        self.assertIn("15.0/24.0 GB", output)
        self.assertIn("20.0/64.0 GB", output)
        self.assertIn("b83_l11_b55post21_s23_20260626T190751Z", output)
        self.assertIn("runtime", output)
        self.assertIn("1h", output)
        self.assertIn("12s_ago", output)
        self.assertIn("actions:\nnone", output)

        color_output = fleet.render_latest_watch_dashboard(snapshot, color=True, max_width=120)

        self.assertIn("\033[", color_output)
        self.assertIn("recent images:", color_output)
        self.assertIn("Add fleet watcher provenance", color_output)

    def test_watch_latest_snapshot_probes_only_active_devices(self) -> None:
        args = Namespace(
            repo_root=".",
            instances=None,
            direct=False,
            host=None,
            workers=None,
            image="latest",
            image_file=None,
            runtime_image_ref=None,
            runtime_image_ref_file=None,
            image_workflow="rlab train image",
            image_branch="main",
            image_artifact="rlab-train-image",
            execute=False,
            interval=5,
            claim_stale_jobs=True,
            stale_older_than_seconds=300,
            stale_limit=50,
        )
        fake_conn = mock.Mock()
        job = fleet.RunningJob(
            id=141,
            lease_owner="rlab-beast-3-rtx4090-any-profile-cccccccccccc-0-aabbccdd",
            profile_id=None,
            runtime_image_ref=RUNTIME_IMAGE_REF,
            run_target="rtx4090",
            run_name="b83_l11_b55post21_s23_20260626T190751Z",
            started_at=datetime.now(UTC),
            heartbeat_at=datetime.now(UTC),
        )
        probe = fleet.DeviceProbe(
            ok=True,
            label="SSH",
            detail="BEAST-3",
            metrics={
                "gpu_util_pct": 75,
                "vram_used_mib": 12288,
                "vram_total_mib": 24576,
                "ram_used_mib": 32768,
                "ram_total_mib": 65536,
                "cpu_util_pct": 42,
            },
        )

        with (
            mock.patch.object(fleet, "_load_config_from_args", return_value=sample_config()),
            mock.patch.object(fleet, "image_ref_from_args", return_value=RUNTIME_IMAGE_REF),
            mock.patch.object(fleet, "recent_runtime_images", return_value=()),
            mock.patch.object(fleet, "_connect_from_args", return_value=fake_conn),
            mock.patch.object(fleet, "list_stale_train_jobs", return_value=[]),
            mock.patch.object(fleet, "queue_demands", return_value=[]),
            mock.patch.object(fleet, "active_leases", return_value=[]),
            mock.patch.object(fleet, "active_eval_leases", return_value=[]),
            mock.patch.object(fleet, "running_jobs", return_value=[job]),
            mock.patch.object(fleet, "live_device_probes", return_value={"rtx4090": probe}) as probes,
            mock.patch.object(fleet, "collect_existing_containers", return_value=([], [])),
        ):
            snapshot = fleet.build_latest_watch_snapshot(args)

        probes.assert_called_once_with(["rtx4090"])
        self.assertEqual(len(snapshot.devices), 1)
        device = snapshot.devices[0]
        self.assertEqual(device["id"], "rtx4090")
        self.assertEqual(device["state"], "busy")
        self.assertEqual(device["details"]["cpu"], "42%")
        self.assertEqual(device["details"]["memory"], "32.0/64.0 GB")
        self.assertEqual(device["details"]["gpu"], "75%")
        self.assertEqual(device["details"]["vram"], "12.0/24.0 GB")

    def test_watch_latest_treats_unreachable_host_as_down_not_failed_action(self) -> None:
        args = Namespace(
            repo_root=".",
            instances=None,
            direct=False,
            host=None,
            workers=None,
            image="latest",
            image_file=None,
            runtime_image_ref=None,
            runtime_image_ref_file=None,
            image_workflow="rlab train image",
            image_branch="main",
            image_artifact="rlab-train-image",
            execute=False,
            interval=5,
        )
        fake_conn = mock.Mock()

        with (
            mock.patch.object(fleet, "_load_config_from_args", return_value=sample_config()),
            mock.patch.object(fleet, "image_ref_from_args", return_value=RUNTIME_IMAGE_REF),
            mock.patch.object(fleet, "recent_runtime_images", return_value=()),
            mock.patch.object(fleet, "_connect_from_args", return_value=fake_conn),
            mock.patch.object(fleet, "list_stale_train_jobs", return_value=[]),
            mock.patch.object(fleet, "queue_demands", return_value=[]),
            mock.patch.object(fleet, "active_leases", return_value=[]),
            mock.patch.object(fleet, "active_eval_leases", return_value=[]),
            mock.patch.object(fleet, "running_jobs", return_value=[]),
            mock.patch.object(
                fleet,
                "collect_existing_containers",
                return_value=(
                    [],
                    [
                        "failed to list managed containers on beast-2: "
                        "ssh: connect to host 192.168.133.26 port 22: Operation timed out"
                    ],
                ),
            ),
        ):
            snapshot = fleet.build_latest_watch_snapshot(args)

        self.assertEqual(snapshot.down_hosts, ("beast-2",))
        self.assertFalse(any(action.host == "beast-2" for action in snapshot.plan.actions))
        self.assertFalse(any("beast-2" in warning for warning in snapshot.plan.warnings))

        output = fleet.render_latest_watch_dashboard(snapshot, color=False, max_width=120)

        beast2_line = next(line for line in output.splitlines() if line.startswith("beast-2"))
        self.assertIn("down", beast2_line)
        self.assertNotIn("start", beast2_line)
        self.assertNotIn("failed actions:", output)
        self.assertNotIn("failed to list managed containers on beast-2", output)

    def test_watch_latest_lists_stale_jobs_in_dry_run(self) -> None:
        args = Namespace(
            repo_root=".",
            instances=None,
            direct=False,
            host=None,
            workers=None,
            image="latest",
            image_file=None,
            runtime_image_ref=None,
            runtime_image_ref_file=None,
            image_workflow="rlab train image",
            image_branch="main",
            image_artifact="rlab-train-image",
            execute=False,
            interval=5,
            claim_stale_jobs=True,
            stale_older_than_seconds=600,
            stale_limit=7,
        )
        fake_conn = mock.Mock()
        stale_row = {
            "id": 132,
            "profile_id": None,
            "runtime_image_ref": OTHER_IMAGE_REF,
            "run_target": "rtx2060",
            "run_name": "b85_beast2_l11l12_b74current_s189_20260626T163035Z",
            "stale_lease_owner": "rlab-beast-2-rtx2060-any-profile-10b659be2346-0-deadbeef",
            "stale_heartbeat_at": datetime(2026, 6, 26, 16, 30, tzinfo=UTC),
        }

        with (
            mock.patch.object(fleet, "_load_config_from_args", return_value=sample_config()),
            mock.patch.object(fleet, "image_ref_from_args", return_value=RUNTIME_IMAGE_REF),
            mock.patch.object(fleet, "recent_runtime_images", return_value=()),
            mock.patch.object(fleet, "_connect_from_args", return_value=fake_conn),
            mock.patch.object(fleet, "list_stale_train_jobs", side_effect=([stale_row], [])) as list_stale,
            mock.patch.object(fleet, "mark_stale_train_jobs_failed") as mark_stale,
            mock.patch.object(fleet, "queue_demands", return_value=[]),
            mock.patch.object(fleet, "active_leases", return_value=[]),
            mock.patch.object(fleet, "active_eval_leases", return_value=[]),
            mock.patch.object(fleet, "running_jobs", return_value=[]),
            mock.patch.object(fleet, "collect_existing_containers", return_value=([], [])),
        ):
            snapshot = fleet.build_latest_watch_snapshot(args)

        list_stale.assert_any_call(
            fake_conn,
            run_target="rtx2060",
            lease_owner_prefix="rlab-beast-2-",
            older_than_seconds=600,
            limit=7,
        )
        mark_stale.assert_not_called()
        self.assertEqual(len(snapshot.stale_train_jobs), 1)

        output = fleet.render_latest_watch_dashboard(snapshot, color=False, max_width=120)

        self.assertIn("stale train jobs:", output)
        self.assertIn("would_fail", output)
        self.assertIn("132", output)
        self.assertIn("b85_beast2_l11l12_b74current_s", output)

    def test_watch_latest_marks_stale_jobs_failed_before_reading_queue(self) -> None:
        args = Namespace(
            repo_root=".",
            instances=None,
            direct=False,
            host=None,
            workers=None,
            image="latest",
            image_file=None,
            runtime_image_ref=None,
            runtime_image_ref_file=None,
            image_workflow="rlab train image",
            image_branch="main",
            image_artifact="rlab-train-image",
            execute=True,
            interval=5,
            claim_stale_jobs=True,
            stale_older_than_seconds=300,
            stale_limit=50,
        )
        fake_conn = mock.Mock()
        events = []
        stale_row = {
            "id": 132,
            "profile_id": None,
            "runtime_image_ref": OTHER_IMAGE_REF,
            "run_target": "rtx2060",
            "run_name": "b85_beast2_l11l12_b74current_s189_20260626T163035Z",
            "stale_lease_owner": "rlab-beast-2-rtx2060-any-profile-10b659be2346-0-deadbeef",
            "stale_heartbeat_at": datetime(2026, 6, 26, 16, 30, tzinfo=UTC),
        }

        def fake_mark_stale(conn, **kwargs):
            events.append(f"stale:{kwargs['run_target']}")
            return [stale_row] if kwargs["run_target"] == "rtx2060" else []

        def fake_queue_demands(conn):
            events.append("queue")
            return []

        with (
            mock.patch.object(fleet, "_load_config_from_args", return_value=sample_config()),
            mock.patch.object(fleet, "image_ref_from_args", return_value=RUNTIME_IMAGE_REF),
            mock.patch.object(fleet, "recent_runtime_images", return_value=()),
            mock.patch.object(fleet, "_connect_from_args", return_value=fake_conn),
            mock.patch.object(fleet, "list_stale_train_jobs") as list_stale,
            mock.patch.object(fleet, "mark_stale_train_jobs_failed", side_effect=fake_mark_stale) as mark_stale,
            mock.patch.object(fleet, "queue_demands", side_effect=fake_queue_demands),
            mock.patch.object(fleet, "active_leases", return_value=[]),
            mock.patch.object(fleet, "active_eval_leases", return_value=[]),
            mock.patch.object(fleet, "running_jobs", return_value=[]),
            mock.patch.object(fleet, "collect_existing_containers", return_value=([], [])),
        ):
            snapshot = fleet.build_latest_watch_snapshot(args)

        self.assertEqual(events, ["stale:rtx2060", "stale:rtx4090", "queue"])
        mark_stale.assert_any_call(
            fake_conn,
            run_target="rtx2060",
            lease_owner_prefix="rlab-beast-2-",
            older_than_seconds=300,
            limit=50,
            error="worker_lost: stale train job marked failed by rlab fleet watch host=beast-2",
        )
        list_stale.assert_not_called()
        self.assertEqual(len(snapshot.stale_train_jobs), 1)

        output = fleet.render_latest_watch_dashboard(snapshot, color=False, max_width=120)

        self.assertIn("stale train jobs:", output)
        self.assertIn("failed", output)
        self.assertIn("132", output)

    def test_watch_latest_prints_starting_frame_before_first_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = Namespace(
                repo_root=tmp,
                execute=False,
                interval=30,
                host=None,
                image="latest",
                image_file=None,
                runtime_image_ref=None,
                runtime_image_ref_file=None,
                no_color=True,
                no_tui=True,
                width=120,
                once=True,
                fail_fast=False,
            )
            frames = []

            def fake_write_frame(text, *, enabled):
                frames.append(text)

            with (
                mock.patch.object(fleet, "write_tui_frame", side_effect=fake_write_frame),
                mock.patch.object(fleet, "build_latest_watch_snapshot", side_effect=RuntimeError("boom")),
            ):
                status = fleet.cmd_watch_latest(args)

        self.assertEqual(status, 1)
        self.assertGreaterEqual(len(frames), 2)
        self.assertIn("status=starting", frames[0])
        self.assertIn("polling now", frames[0])
        self.assertIn("snapshot failed: boom", frames[1])

    def test_watch_latest_lock_rejects_second_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = Namespace(
                repo_root=tmp,
                execute=False,
                interval=5,
                host=None,
                image="latest",
                image_file=None,
                runtime_image_ref=None,
                runtime_image_ref_file=None,
                no_color=True,
                no_tui=True,
                width=120,
                once=True,
                fail_fast=False,
            )
            lock = fleet.acquire_watch_latest_lock(args)
            try:
                with self.assertRaises(fleet.WatchLatestLockBusy) as raised:
                    fleet.acquire_watch_latest_lock(args)
            finally:
                fleet.release_watch_latest_lock(lock)

        self.assertIn("watch.lock", str(raised.exception.path))
        self.assertIn('"pid"', raised.exception.owner)

    def test_watch_latest_busy_lock_exits_without_polling(self) -> None:
        args = Namespace(
            repo_root=".",
            execute=False,
            interval=5,
            host=None,
            image="latest",
            image_file=None,
            runtime_image_ref=None,
            runtime_image_ref_file=None,
            no_color=True,
            no_tui=True,
            width=120,
            once=True,
            fail_fast=False,
        )
        frames = []
        lock_error = fleet.WatchLatestLockBusy(Path("/tmp/watch.lock"), '{"pid": 123}')

        def fake_write_frame(text, *, enabled):
            frames.append(text)

        with (
            mock.patch.object(fleet, "acquire_watch_latest_lock", side_effect=lock_error),
            mock.patch.object(fleet, "build_latest_watch_snapshot") as build_snapshot,
            mock.patch.object(fleet, "write_tui_frame", side_effect=fake_write_frame),
        ):
            status = fleet.cmd_watch_latest(args)

        self.assertEqual(status, 2)
        build_snapshot.assert_not_called()
        self.assertIn("already owns the lock", frames[0])
        self.assertIn('"pid": 123', frames[0])

    def test_watch_latest_default_interval_is_fifteen_seconds(self) -> None:
        args = fleet.build_parser().parse_args(["watch"])

        self.assertEqual(args.interval, 15.0)
        self.assertTrue(args.execute)
        self.assertTrue(args.claim_stale_jobs)
        self.assertEqual(args.stale_older_than_seconds, 300)
        self.assertEqual(args.stale_limit, 50)

    def test_dry_run_replaces_execute_flag(self) -> None:
        args = fleet.build_parser().parse_args(["watch", "--dry-run"])

        self.assertFalse(args.execute)
        args = fleet.build_parser().parse_args(["watch", "--dry-run", "--execute"])
        self.assertTrue(args.execute)

    def test_watch_latest_can_disable_stale_job_claims(self) -> None:
        args = fleet.build_parser().parse_args(["watch", "--no-claim-stale-jobs"])

        self.assertFalse(args.claim_stale_jobs)

    def test_watch_latest_skips_later_actions_on_failed_host(self) -> None:
        config = sample_config()
        actions = (
            fleet.FleetAction(kind="start", host="beast-2", container="new-2", reason="latest"),
            fleet.FleetAction(kind="remove", host="beast-2", container="old-2", reason="old"),
            fleet.FleetAction(kind="start", host="beast-3", container="new-3", reason="latest"),
        )
        calls = []

        def fake_run_action_result(config_arg, action, *, local=False, capture=False):
            calls.append(action.container)
            return fleet.ActionResult(
                kind=action.kind,
                host=action.host,
                container=action.container,
                exit_code=1 if action.container == "new-2" else 0,
            )

        with mock.patch.object(fleet, "run_action_result", side_effect=fake_run_action_result):
            results = fleet.run_latest_watch_actions(
                config,
                fleet.FleetPlan(desired=(), existing=(), actions=actions, warnings=()),
            )

        self.assertEqual(calls, ["new-2", "new-3"])
        self.assertEqual([result.container for result in results], ["new-2", "new-3"])
        self.assertEqual(results[0].exit_code, 1)

    def test_reconcile_keeps_unprofiled_container_for_profile_demand(self) -> None:
        config = fleet.filter_config_to_host(sample_config(), "beast-3")
        any_profile = fleet.build_ensure_runner_plan(
            config,
            host_name="beast-3",
            profile_id=None,
            runtime_image_ref=RUNTIME_IMAGE_REF,
            run_target="rtx4090",
            workers=5,
            existing=[],
            leases=[],
        ).desired[0]
        existing = fleet.ExistingContainer(
            host="beast-3",
            name=any_profile.name,
            state="running",
            status="Up",
            image="ghcr.io/tsilva/rlab/rlab-train",
            labels=any_profile.labels,
        )

        plan = fleet.build_fleet_plan(config, [demand(pending=1)], [existing], [])

        self.assertTrue(any(action.kind == "keep" for action in plan.actions))
        self.assertFalse(any(action.kind == "start" for action in plan.actions))
        self.assertFalse(any(action.kind == "remove" for action in plan.actions))

    def test_format_containers_lists_managed_state_across_hosts(self) -> None:
        container = fleet.ExistingContainer(
            host="beast-3",
            name="rlab-beast-3-rtx4090-any-profile-cccccccccccc",
            state="running",
            status="Up 5 minutes",
            image="ghcr.io/tsilva/rlab/rlab-train",
            labels={
                "rlab.managed": "true",
                "rlab.host": "beast-3",
                "rlab.profile": "",
                "rlab.run-target": "rtx4090",
                "rlab.runtime-digest": "cccccccccccc",
                "rlab.runtime-image-ref": RUNTIME_IMAGE_REF,
                "rlab.worker-prefix": "rlab-beast-3-rtx4090-any-profile-cccccccccccc",
            },
        )

        output = fleet.format_containers([container])

        self.assertIn("managed containers:", output)
        self.assertIn("host=beast-3", output)
        self.assertIn("state=running", output)
        self.assertIn("profile=any", output)
        self.assertIn("target=rtx4090", output)
        self.assertIn("digest=cccccccccccc", output)

    def test_format_containers_includes_running_jobs_by_worker_prefix(self) -> None:
        container = fleet.ExistingContainer(
            host="beast-3",
            name="rlab-beast-3-rtx4090-any-profile-cccccccccccc",
            state="running",
            status="Up 5 minutes",
            image="ghcr.io/tsilva/rlab/rlab-train",
            labels={
                "rlab.managed": "true",
                "rlab.host": "beast-3",
                "rlab.profile": "",
                "rlab.run-target": "rtx4090",
                "rlab.runtime-digest": "cccccccccccc",
                "rlab.runtime-image-ref": RUNTIME_IMAGE_REF,
                "rlab.worker-prefix": "rlab-beast-3-rtx4090-any-profile-cccccccccccc",
            },
        )
        job = fleet.RunningJob(
            id=123,
            lease_owner="rlab-beast-3-rtx4090-any-profile-cccccccccccc-4-56ea2c67",
            profile_id="mario-ppo/post21/rtx4090-prebuilt-l11-lowkl-lrdecay-v1",
            runtime_image_ref=RUNTIME_IMAGE_REF,
            run_target="rtx4090",
            run_name="b83_l11_lowkl_lrdecay_image_s121_20260626T153508Z",
            started_at=None,
            heartbeat_at=None,
        )

        output = fleet.format_containers([container], [job])
        job_line = next(line for line in output.splitlines() if "job=123" in line)

        self.assertIn("job=123", job_line)
        self.assertIn("run=b83_l11_lowkl_lrdecay_image_s121_20260626T153508Z", job_line)
        self.assertIn("worker=4-56ea2c67", job_line)
        self.assertNotIn("target=rtx4090", job_line)
        self.assertNotIn("owner=rlab-beast-3-rtx4090-any-profile-cccccccccccc-4-56ea2c67", job_line)

    def test_format_containers_shows_heartbeat_as_elapsed_time(self) -> None:
        container = fleet.ExistingContainer(
            host="beast-3",
            name="rlab-beast-3-rtx4090-any-profile-cccccccccccc",
            state="running",
            status="Up 5 minutes",
            image="ghcr.io/tsilva/rlab/rlab-train",
            labels={
                "rlab.managed": "true",
                "rlab.host": "beast-3",
                "rlab.profile": "",
                "rlab.run-target": "rtx4090",
                "rlab.runtime-digest": "cccccccccccc",
                "rlab.runtime-image-ref": RUNTIME_IMAGE_REF,
                "rlab.worker-prefix": "rlab-beast-3-rtx4090-any-profile-cccccccccccc",
            },
        )
        job = fleet.RunningJob(
            id=123,
            lease_owner="rlab-beast-3-rtx4090-any-profile-cccccccccccc-4-56ea2c67",
            profile_id=None,
            runtime_image_ref=RUNTIME_IMAGE_REF,
            run_target="rtx4090",
            run_name="b83_l11_lowkl_lrdecay_image_s121_20260626T153508Z",
            started_at=None,
            heartbeat_at=datetime.now(UTC),
        )

        output = fleet.format_containers([container], [job])
        job_line = next(line for line in output.splitlines() if "job=123" in line)

        self.assertRegex(job_line, r"heartbeat=\d+s_ago")
        self.assertNotIn("+00:00", job_line)

    def test_format_elapsed_since_uses_compact_age_buckets(self) -> None:
        now = datetime(2026, 6, 26, 16, 30, tzinfo=UTC)

        self.assertEqual(
            fleet.format_elapsed_since(datetime(2026, 6, 26, 16, 29, 42, tzinfo=UTC), now=now),
            "18s_ago",
        )
        self.assertEqual(
            fleet.format_elapsed_since(datetime(2026, 6, 26, 16, 25, tzinfo=UTC), now=now),
            "5m_ago",
        )
        self.assertEqual(
            fleet.format_elapsed_since(datetime(2026, 6, 26, 14, 30, tzinfo=UTC), now=now),
            "2h_ago",
        )

    def test_running_job_dashboard_rows_show_runtime_and_heartbeat_elapsed(self) -> None:
        now = datetime(2026, 6, 26, 16, 30, tzinfo=UTC)
        job = fleet.RunningJob(
            id=123,
            lease_owner="rlab-beast-3-rtx4090-any-profile-cccccccccccc-4-56ea2c67",
            profile_id=None,
            runtime_image_ref=RUNTIME_IMAGE_REF,
            run_target="rtx4090",
            run_name="b83_l11_lowkl_lrdecay_image_s121_20260626T153508Z",
            started_at=now - timedelta(hours=2, minutes=5),
            heartbeat_at=now - timedelta(seconds=18),
        )

        rows = fleet.running_job_dashboard_rows([job], now=now)

        self.assertEqual(rows[0][-2:], ["2h", "18s_ago"])

    def test_format_containers_keeps_mismatched_job_target_visible(self) -> None:
        container = fleet.ExistingContainer(
            host="beast-3",
            name="rlab-beast-3-rtx4090-any-profile-cccccccccccc",
            state="running",
            status="Up 5 minutes",
            image="ghcr.io/tsilva/rlab/rlab-train",
            labels={
                "rlab.managed": "true",
                "rlab.host": "beast-3",
                "rlab.profile": "",
                "rlab.run-target": "rtx4090",
                "rlab.runtime-digest": "cccccccccccc",
                "rlab.runtime-image-ref": RUNTIME_IMAGE_REF,
                "rlab.worker-prefix": "rlab-beast-3-rtx4090-any-profile-cccccccccccc",
            },
        )
        job = fleet.RunningJob(
            id=123,
            lease_owner="rlab-beast-3-rtx4090-any-profile-cccccccccccc-4-56ea2c67",
            profile_id="mario-ppo/post21/rtx4090-prebuilt-l11-lowkl-lrdecay-v1",
            runtime_image_ref=RUNTIME_IMAGE_REF,
            run_target="rtx2060",
            run_name="b83_l11_lowkl_lrdecay_image_s121_20260626T153508Z",
            started_at=None,
            heartbeat_at=None,
        )

        output = fleet.format_containers([container], [job])
        job_line = next(line for line in output.splitlines() if "job=123" in line)

        self.assertIn("target=rtx2060", job_line)

    def test_format_containers_reports_empty_and_warnings(self) -> None:
        output = fleet.format_containers([], warnings=["failed to list managed containers on beast-3"])

        self.assertIn("managed containers: none", output)
        self.assertIn("warnings:", output)
        self.assertIn("failed to list managed containers on beast-3", output)


class FleetHostSetupTests(unittest.TestCase):
    def test_machine_registry_encodes_beast_host_setup(self) -> None:
        config = fleet.load_fleet_config(Path(".").resolve())

        self.assertEqual(config.hosts["beast-3"].ssh_target, "tsilva@beast-3")
        self.assertEqual(config.hosts["beast-3"].run_target, "rtx4090")
        self.assertEqual(config.hosts["beast-3"].max_workers, 5)
        self.assertEqual(config.hosts["beast-2"].ssh_target, "tsilva@192.168.133.26")
        self.assertIn("HostKeyAlias=beast-2", config.hosts["beast-2"].ssh_options)
        self.assertEqual(config.hosts["beast-2"].run_target, "rtx2060")
        self.assertEqual(config.hosts["beast-2"].max_workers, 4)
        self.assertEqual(config.hosts["beast-2"].docker_command, ("sudo", "-n", "docker"))

    def test_host_capacity_comes_from_machine_slot_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "experiments").mkdir()
            (root / "experiments" / "instances.yaml").write_text(
                json.dumps(
                    {
                        "instances": {
                            "rtx4090": {
                                "name": "rtx4090",
                                "default_workers": 3,
                                "hardware_max_workers": 5,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            (root / "experiments" / "machines.yaml").write_text(
                json.dumps(
                    {
                        "machines": {
                            "beast-3": {
                                "backend": "docker_ssh",
                                "ssh_target": "tsilva@beast-3",
                                "run_target": "rtx4090",
                                "limits": {"max_parallel_containers": 2},
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            config = fleet.load_fleet_config(root)

        self.assertEqual(config.hosts["beast-3"].max_workers, 2)

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
        self.assertIn("nvidia.github.io/libnvidia-container", script)
        self.assertIn("nvidia-ctk runtime configure --runtime=docker", script)
        self.assertIn("TRAIN_QUEUE_DATABASE_URL=", script)
        self.assertIn("docker pull ghcr.io/tsilva/rlab/rlab-train@sha256:", script)
        self.assertIn("rlab-container-entrypoint rlab-container-smoke", script)

    def test_beast_2_uses_configured_sudo_docker_command(self) -> None:
        config = fleet.filter_config_to_host(fleet.load_fleet_config(Path(".").resolve()), "beast-2")
        beast2_demand = fleet.QueueDemand(
            profile_id="mario-ppo/post21/rtx2060",
            runtime_image_ref=RUNTIME_IMAGE_REF,
            run_target="rtx2060",
            pending_count=1,
            running_count=0,
            oldest_job_id=1,
        )
        plan = fleet.build_fleet_plan(config, [beast2_demand], [], [])
        command_text = "\n".join(plan.actions[0].commands)
        setup_script = fleet.setup_host_script(config.hosts["beast-2"], runtime_image_ref=RUNTIME_IMAGE_REF)

        self.assertIn("sudo -n docker pull ghcr.io/tsilva/rlab/rlab-train@sha256:", command_text)
        self.assertIn("sudo -n docker run -d", command_text)
        self.assertIn("sudo -n docker info", setup_script)
        self.assertIn("sudo -n docker run --rm --gpus all", setup_script)

    def test_host_command_quotes_remote_shell_script(self) -> None:
        config = sample_config()
        command = fleet.host_command(
            config.hosts["beast-2"],
            ["bash", "-lc", "sudo -n docker ps -a --format '{{json .}}'"],
        )

        self.assertEqual(command[-2], "tsilva@192.168.133.26")
        self.assertIn("bash -lc", command[-1])
        self.assertIn("sudo -n docker ps", command[-1])
        self.assertIn("{{json .}}", command[-1])

    def test_cli_exposes_only_mac_managed_reconciliation(self) -> None:
        help_text = fleet.build_parser().format_help()

        self.assertIn("ps", help_text)
        self.assertIn("reconcile", help_text)
        self.assertIn("watch", help_text)
        self.assertNotIn("watch-latest", help_text)
        self.assertIn("setup-host", help_text)
        self.assertNotIn("remote-reconcile", help_text)
        self.assertNotIn("install-systemd", help_text)


if __name__ == "__main__":
    unittest.main()
