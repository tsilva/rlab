from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from rlab import fleet
from rlab.docker_host import RuntimeHostImage
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
    def test_drained_idle_machine_is_not_contacted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            machines_path = Path(temporary_dir) / "machines.yaml"
            write_machine_registry(
                machines_path,
                backend="docker_ssh",
                max_parallel_containers=4,
                host_root="/home/tsilva/rlab",
                machine_name="beast-2",
            )
            conn = mock.MagicMock()
            with (
                mock.patch.object(fleet, "connect", return_value=conn),
                mock.patch.object(fleet, "machine_mutation_lock", return_value=nullcontext()),
                mock.patch.object(fleet, "machine_control", return_value={"drained": True}),
                mock.patch.object(fleet, "active_job_launches", return_value=[]),
                mock.patch.object(fleet, "DockerRunnerHost") as host,
            ):
                result = fleet.run_service_machine_pass(
                    machine_name="beast-2",
                    machines_path=machines_path,
                    repo_root=Path(temporary_dir),
                    deadline_monotonic=10**12,
                )

        self.assertEqual(result["prewarm"]["status"], "skipped_drained")
        self.assertIn("drained", result["maintenance_skipped"])
        self.assertEqual(host.return_value.method_calls, [])

    def test_failed_prewarm_preserves_host_runtime_images(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            machines_path = root / "machines.yaml"
            write_machine_registry(
                machines_path,
                backend="docker_ssh",
                max_parallel_containers=4,
                host_root="/home/tsilva/rlab",
                machine_name="beast-3",
            )
            payload = json.loads(machines_path.read_text(encoding="utf-8"))
            payload["machines"]["beast-3"]["prewarm_latest_runtime"] = True
            machines_path.write_text(json.dumps(payload), encoding="utf-8")
            conn = mock.MagicMock()
            release = SimpleNamespace(runtime_image_ref=RUNTIME_IMAGE_REF)
            with (
                mock.patch.object(fleet, "connect", return_value=conn),
                mock.patch.object(fleet, "machine_mutation_lock", return_value=nullcontext()),
                mock.patch.object(fleet, "machine_control", return_value={"drained": False}),
                mock.patch.object(fleet, "run_reconcile_fill_pass", return_value=(0, 0)),
                mock.patch.object(fleet, "recover_live_publication", return_value=0),
                mock.patch.object(fleet, "recent_runtime_images", return_value=(release,)),
                mock.patch.object(
                    fleet, "prewarm_latest_runtime", side_effect=RuntimeError("pull failed")
                ),
                mock.patch.object(
                    fleet, "prune_inactive_job_containers", return_value=2
                ) as prune_containers,
                mock.patch.object(fleet, "prune_stale_runtime_images") as prune_images,
            ):
                result = fleet.run_service_machine_pass(
                    machine_name="beast-3",
                    machines_path=machines_path,
                    repo_root=root,
                    deadline_monotonic=10**12,
                )

        self.assertEqual(result["prewarm"]["status"], "error")
        self.assertEqual(result["removed_containers"], 2)
        self.assertIn("prewarm failed", result["image_pruning_skipped"])
        prune_containers.assert_called_once()
        prune_images.assert_not_called()

    def test_machine_registry_rejects_unknown_nested_and_root_fields(self) -> None:
        cases = {
            "root": "surprise: true\n",
            "docker": "      typo: true\n",
            "paths": "      typo: true\n",
        }
        for location, extra in cases.items():
            with self.subTest(location=location), tempfile.TemporaryDirectory() as temporary_dir:
                path = Path(temporary_dir) / "machines.yaml"
                if location == "root":
                    text = (
                        "machines:\n"
                        "  local:\n"
                        "    backend: local_docker\n"
                        "    limits: {max_parallel_containers: 1}\n"
                        f"{extra}"
                    )
                elif location == "docker":
                    text = (
                        "machines:\n"
                        "  local:\n"
                        "    backend: local_docker\n"
                        "    docker:\n"
                        f"{extra}"
                        "    limits: {max_parallel_containers: 1}\n"
                    )
                else:
                    text = (
                        "machines:\n"
                        "  local:\n"
                        "    backend: local_docker\n"
                        "    limits: {max_parallel_containers: 1}\n"
                        "    paths:\n"
                        f"{extra}"
                    )
                path.write_text(text, encoding="utf-8")

                with self.assertRaisesRegex(ValueError, "unknown.*typo|unknown root"):
                    load_machine_registry(path)

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

    def test_beast_three_enables_latest_runtime_prewarming(self) -> None:
        registry = load_machine_registry(Path("experiments/machines.yaml"))

        self.assertTrue(registry.machines["beast-3"].prewarm_latest_runtime)
        self.assertFalse(registry.machines["beast-2"].prewarm_latest_runtime)


class ContainerContractTests(unittest.TestCase):
    def test_prewarmed_runtime_is_protected_from_cleanup(self) -> None:
        target = local_machine()
        keep = RuntimeHostImage(
            machine=target.name,
            repository="ghcr.io/tsilva/rlab/rlab-train",
            digest="sha256:" + "c" * 64,
            image_id="sha256:keep",
        )
        remove = RuntimeHostImage(
            machine=target.name,
            repository="ghcr.io/tsilva/rlab/rlab-train",
            digest="sha256:" + "d" * 64,
            image_id="sha256:remove",
        )
        host = SimpleNamespace(
            machine=target,
            list_runtime_image_containers=lambda: [],
            list_runtime_images=lambda _repos: [keep, remove],
            remove_runtime_image=mock.Mock(return_value=SimpleNamespace(ok=True)),
        )
        with mock.patch.object(fleet, "queue_demands", return_value=[]):
            pruned = fleet.prune_stale_runtime_images(
                FakeConnection(),
                host,
                extra_protected_refs=(RUNTIME_IMAGE_REF,),
            )

        self.assertEqual(pruned, 1)
        host.remove_runtime_image.assert_called_once_with(remove.image_ref)

    def test_prewarm_pulls_and_probes_only_the_latest_new_runtime(self) -> None:
        target = local_machine()
        release = SimpleNamespace(
            runtime_image_ref=RUNTIME_IMAGE_REF,
            source_sha="source",
            runtime_build_source_sha="build-source",
            runtime_input_sha256="a" * 64,
        )
        host = SimpleNamespace(
            machine=target,
            runtime_image_present=mock.Mock(return_value=False),
            ensure_runtime_image=mock.Mock(return_value=True),
            probe_runtime_image=mock.Mock(return_value={}),
        )
        with (
            tempfile.TemporaryDirectory() as temporary,
            mock.patch.object(fleet, "recent_runtime_images", return_value=(release,)) as recent,
        ):
            state_path = Path(temporary) / "prewarm.json"
            detail, protected_ref = fleet.prewarm_latest_runtime(
                host,
                state_path=state_path,
            )
            host.runtime_image_present.return_value = True
            second_detail, second_ref = fleet.prewarm_latest_runtime(
                host,
                state_path=state_path,
            )

        self.assertEqual(recent.call_args_list[0].kwargs, {"limit": 1})
        self.assertEqual(protected_ref, RUNTIME_IMAGE_REF)
        self.assertEqual(second_ref, RUNTIME_IMAGE_REF)
        self.assertTrue(detail["pulled"])
        self.assertTrue(detail["probed"])
        self.assertFalse(second_detail["pulled"])
        self.assertFalse(second_detail["probed"])
        host.ensure_runtime_image.assert_called_once_with(RUNTIME_IMAGE_REF)
        host.probe_runtime_image.assert_called_once_with(
            runtime_image_ref=RUNTIME_IMAGE_REF,
            expected_source_sha="build-source",
            expected_runtime_input_sha256="a" * 64,
        )

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

    def test_runtime_image_backoff_does_not_block_unrelated_pending_job(self) -> None:
        target = local_machine(capacity=2)
        bad = {"id": 1, "runtime_image_ref": RUNTIME_IMAGE_REF}
        good = {"id": 2, "runtime_image_ref": OTHER_IMAGE_REF}
        host = SimpleNamespace(
            machine=target,
            ensure_runtime_image=mock.Mock(side_effect=[False, True]),
            output_host_path=lambda launch_id: f"/output/{launch_id}",
        )
        with (
            mock.patch.object(fleet, "machine_control", return_value={"drained": False}),
            mock.patch.object(fleet, "train_container_slot_usage", return_value=(0, 2, 2)),
            mock.patch.object(
                fleet, "next_pending_train_job", side_effect=[bad, good, None]
            ),
            mock.patch.object(fleet, "record_runtime_image_failure") as failed,
            mock.patch.object(fleet, "reset_runtime_image_retry") as reset,
            mock.patch.object(
                fleet,
                "claim_job_launch",
                return_value=(good, {"launch_id": "train-2"}),
            ) as claim,
            mock.patch.object(fleet, "_start_or_resume_launch", return_value=True),
        ):
            launched = fleet.launch_next_jobs(FakeConnection(), host=host)

        self.assertEqual(launched, 1)
        failed.assert_called_once_with(
            mock.ANY,
            machine=target.name,
            runtime_image_ref=RUNTIME_IMAGE_REF,
            error="runtime image is unavailable",
        )
        reset.assert_called_once_with(
            mock.ANY,
            machine=target.name,
            runtime_image_ref=OTHER_IMAGE_REF,
        )
        self.assertEqual(claim.call_args.kwargs["job_id"], 2)

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
