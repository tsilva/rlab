from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from rlab import fleet
from rlab.machines import load_machine_registry, resolve_machine
from rlab.runtime_refs import runtime_image_ref_from_file
from tests.fleet_fakes import FakeConnection, write_machine_registry


RUNTIME_IMAGE_REF = (
    "docker:ghcr.io/tsilva/rlab/rlab-train@sha256:"
    "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
)
OTHER_IMAGE_REF = (
    "docker:ghcr.io/tsilva/rlab/rlab-train@sha256:"
    "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"
)


def write_registry(path: Path) -> None:
    write_machine_registry(
        path,
        backend="local_docker",
        max_parallel_containers=2,
        host_root="/tmp/rlab",
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
                        "limits": {"max_parallel_containers": 5},
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

    def test_setup_host_uses_the_shared_machine_shell_transport(self) -> None:
        registry = sample_registry()
        args = fleet.build_parser().parse_args(["setup-host", "--host", "beast-3"])
        completed = SimpleNamespace(returncode=0)

        with (
            mock.patch.object(fleet, "load_registry_from_args", return_value=registry),
            mock.patch.object(fleet, "run_machine_shell", return_value=completed) as run,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            status = fleet.cmd_setup_host(args)

        self.assertEqual(status, 0)
        self.assertEqual(run.call_args.args[0].name, "beast-3")
        self.assertIn("sudo -n docker info", run.call_args.args[1])

    def test_load_shared_runner_env_normalizes_quotes_and_ignores_machine_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text(
                "\n".join(
                    [
                        'WANDB_API_KEY="wandb-value"',
                        'AWS_ACCESS_KEY_ID="access-value"',
                        'AWS_SECRET_ACCESS_KEY="secret-value"',
                        'AWS_S3_ENDPOINT_URL="https://r2.example.invalid"',
                        'AWS_REGION="auto"',
                        'CHECKPOINT_BUCKET_URI="s3://checkpoints/prefix"',
                        'ROM_PATH="/local/mario.nes"',
                        'DATABASE_URL="postgresql://local-only"',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            values = fleet.load_shared_runner_env(path)

        self.assertEqual(tuple(values), fleet.SHARED_RUNNER_ENV_KEYS)
        self.assertEqual(values["AWS_REGION"], "auto")
        self.assertNotIn("ROM_PATH", values)
        self.assertNotIn("DATABASE_URL", values)

    def test_load_shared_runner_env_fails_closed_when_required_value_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text("WANDB_API_KEY=wandb-value\n", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "missing required key"):
                fleet.load_shared_runner_env(path)

    def test_sync_shared_runner_env_sends_only_normalized_allowlist_over_stdin(self) -> None:
        registry = sample_registry()
        machine = resolve_machine(registry, "beast-3")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text(
                "\n".join(
                    [
                        'WANDB_API_KEY="wandb-value"',
                        'AWS_ACCESS_KEY_ID="access-value"',
                        'AWS_SECRET_ACCESS_KEY="secret-value"',
                        'AWS_S3_ENDPOINT_URL="https://r2.example.invalid"',
                        'AWS_REGION="auto"',
                        'CHECKPOINT_BUCKET_URI="s3://checkpoints/prefix"',
                        'ROM_PATH="/local/mario.nes"',
                        'DATABASE_URL="postgresql://local-only"',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            with (
                contextlib.redirect_stdout(io.StringIO()),
                mock.patch.object(
                    fleet,
                    "run_machine_shell",
                    return_value=SimpleNamespace(returncode=0, stdout="", stderr=""),
                ) as run,
            ):
                fleet.sync_shared_runner_env(machine, path, color=False)

        script = run.call_args.args[1]
        payload = run.call_args.kwargs["input_text"]
        self.assertIn("AWS_REGION=auto\n", payload)
        self.assertNotIn('AWS_REGION="auto"', payload)
        self.assertNotIn("ROM_PATH", payload)
        self.assertNotIn("DATABASE_URL", payload)
        self.assertNotIn("wandb-value", script)
        self.assertNotIn("secret-value", script)
        self.assertIn("sudo -n install", script)
        self.assertIn("for (i in keys)", script)
        self.assertIn("!($1 in shared)", script)

    def test_remote_launch_syncs_shared_env_before_claiming_queue_row(self) -> None:
        registry = sample_registry()
        machine = resolve_machine(registry, "beast-3")
        calls: list[str] = []

        def sync(*_args, **_kwargs) -> None:
            calls.append("sync")

        def claim(*_args, **_kwargs):
            calls.append("claim")
            return None

        with (
            mock.patch.object(fleet, "sync_shared_runner_env", side_effect=sync),
            mock.patch.object(fleet, "claim_job_launch", side_effect=claim),
        ):
            launched = fleet.launch_claimed_job_container(
                FakeConnection(),
                machine=machine,
                shared_env_file=Path("/repo/.env"),
            )

        self.assertIsNone(launched)
        self.assertEqual(calls, ["sync", "claim"])

    def test_remote_launch_does_not_claim_when_shared_env_sync_fails(self) -> None:
        registry = sample_registry()
        machine = resolve_machine(registry, "beast-3")

        with (
            mock.patch.object(
                fleet,
                "sync_shared_runner_env",
                side_effect=RuntimeError("sync failed"),
            ),
            mock.patch.object(fleet, "claim_job_launch") as claim,
        ):
            with self.assertRaisesRegex(RuntimeError, "sync failed"):
                fleet.launch_claimed_job_container(
                    FakeConnection(),
                    machine=machine,
                    shared_env_file=Path("/repo/.env"),
                )

        claim.assert_not_called()

    def test_runtime_image_ref_file_accepts_ci_json_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rlab-train-image.json"
            path.write_text(json.dumps({"runtime_image_ref": RUNTIME_IMAGE_REF}), encoding="utf-8")

            self.assertEqual(runtime_image_ref_from_file(path), RUNTIME_IMAGE_REF)

class JobContainerTests(unittest.TestCase):
    def test_shepherd_limit_caps_total_active_containers(self) -> None:
        registry = sample_registry()
        machine = resolve_machine(registry, "beast-3")
        active = [
            fleet.JobContainer(
                machine=machine.name,
                name=f"job-{index}",
                state="running",
                status="Up",
                labels={fleet.JOB_KIND_LABEL: fleet.TRAIN_JOB_KIND},
            )
            for index in range(3)
        ]

        with mock.patch.object(fleet, "list_job_containers", return_value=active):
            self.assertEqual(
                fleet.machine_available_train_slots(machine, limit=4),
                1,
            )
            self.assertEqual(
                fleet.machine_available_train_slots(machine, limit=3),
                0,
            )

    def test_job_container_run_command_uses_run_job_payload_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "machines.yaml"
            write_registry(path)
            machine = resolve_machine(load_machine_registry(path), "beast-test")

        command = fleet.job_container_run_command(
            machine,
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
