from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from rlab import docker_host
from rlab.docker_host import DockerRunnerHost, MachineCommandTimeout
from rlab.fleet_labels import JOB_CONTAINER_LABEL, LAUNCH_ID_LABEL
from rlab.machines import load_machine_registry, resolve_machine
from tests.db_fakes import write_machine_registry


RUNTIME_IMAGE_REF = (
    "docker:ghcr.io/tsilva/rlab/rlab-train@sha256:"
    "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
)
ROM_MANIFEST = {
    "schema_version": 2,
    "game": "SuperMarioBros-Nes-v0",
    "filename": "rom.nes",
    "size_bytes": 1,
    "sha256": "a" * 64,
    "object_uri": "s3://bucket/rom.nes",
    "provider_rom_identity": "b" * 40,
    "provider_rom_identity_algorithm": "sha1-provider-body-v1",
}


def machine(*, backend: str = "local_docker", host_root: str = "/tmp/rlab"):
    with tempfile.TemporaryDirectory() as temporary_dir:
        path = Path(temporary_dir) / "machines.yaml"
        write_machine_registry(
            path,
            backend=backend,
            max_parallel_containers=2,
            host_root=host_root,
            machine_name="beast-3" if backend == "docker_ssh" else "beast-test",
        )
        name = "beast-3" if backend == "docker_ssh" else "beast-test"
        return resolve_machine(load_machine_registry(path), name)


class DockerHostTests(unittest.TestCase):
    def test_ensure_runtime_image_skips_pull_when_digest_is_present(self) -> None:
        present = subprocess.CompletedProcess([], 0, "{}", "")
        with mock.patch.object(docker_host, "_run_machine_docker", return_value=present) as run:
            timings: dict[str, float] = {}
            ready = DockerRunnerHost(machine()).ensure_runtime_image(
                RUNTIME_IMAGE_REF,
                timings=timings,
            )

        self.assertTrue(ready)
        self.assertEqual(run.call_args.args[1][:2], ["image", "inspect"])
        self.assertEqual(run.call_count, 1)
        self.assertEqual(timings["host_image_pull_seconds"], 0.0)

    def test_ensure_runtime_image_pulls_only_after_missing_digest(self) -> None:
        missing = subprocess.CompletedProcess([], 1, "", "missing")
        pulled = subprocess.CompletedProcess([], 0, "", "")
        present = subprocess.CompletedProcess([], 0, "{}", "")
        with mock.patch.object(
            docker_host,
            "_run_machine_docker",
            side_effect=[missing, pulled, present],
        ) as run:
            ready = DockerRunnerHost(
                replace(machine(backend="docker_ssh"), pull_policy="always")
            ).ensure_runtime_image(RUNTIME_IMAGE_REF)

        self.assertTrue(ready)
        self.assertEqual(run.call_args_list[0].args[1][:2], ["image", "inspect"])
        self.assertEqual(run.call_args_list[1].args[1][0], "pull")
        self.assertEqual(run.call_args_list[2].args[1][:2], ["image", "inspect"])

    def test_setup_host_uses_configured_sudo_docker_command(self) -> None:
        script, returncode = docker_host.setup_docker_host(
            machine(backend="docker_ssh", host_root="/home/tsilva/rlab"),
            execute=False,
        )

        self.assertIsNone(returncode)
        self.assertIn("sudo -n docker info", script)
        self.assertIn("/home/tsilva/rlab", script)

    def test_setup_host_allows_a_cold_runtime_image_pull(self) -> None:
        result = subprocess.CompletedProcess([], 0, "", "")
        with mock.patch.object(docker_host, "_run_machine_shell", return_value=result) as run:
            _, returncode = docker_host.setup_docker_host(
                machine(backend="docker_ssh", host_root="/home/tsilva/rlab"),
                runtime_image_ref=RUNTIME_IMAGE_REF,
                execute=True,
            )

        self.assertEqual(returncode, 0)
        self.assertEqual(run.call_args.kwargs["timeout"], docker_host.DOCKER_PULL_TIMEOUT_SECONDS)

    def test_ssh_transport_is_noninteractive_and_bounded(self) -> None:
        target = machine(backend="docker_ssh", host_root="/home/tsilva/rlab")
        result = subprocess.CompletedProcess([], 0, "", "")
        with mock.patch.object(docker_host.subprocess, "run", return_value=result) as run:
            DockerRunnerHost(target).list_job_containers()

        command = run.call_args.args[0]
        text = " ".join(command)
        self.assertIn("BatchMode=yes", text)
        self.assertIn("ConnectTimeout=10", text)
        self.assertIn("ServerAliveCountMax=3", text)
        self.assertIn("timeout --signal=TERM --kill-after=5s 115s sh -lc", text)

    def test_machine_command_timeout_is_typed(self) -> None:
        expired = subprocess.TimeoutExpired(["sh"], 0.01)
        with mock.patch.object(docker_host.subprocess, "run", side_effect=expired):
            with self.assertRaises(MachineCommandTimeout):
                DockerRunnerHost(machine()).list_job_containers()

    def test_lane_deadline_caps_remote_operation_timeout(self) -> None:
        deadline = time.monotonic() + 2
        result = subprocess.CompletedProcess([], 0, "", "")
        with mock.patch.object(docker_host.subprocess, "run", return_value=result) as run:
            DockerRunnerHost(machine(), deadline_monotonic=deadline).list_job_containers()

        self.assertLessEqual(run.call_args.kwargs["timeout"], 2)

    def test_sync_sends_secrets_only_over_stdin(self) -> None:
        target = machine(backend="docker_ssh", host_root="/home/tsilva/rlab")
        values = {
            "WANDB_API_KEY": "wandb-value",
            "AWS_ACCESS_KEY_ID": "access-value",
        }
        result = subprocess.CompletedProcess([], 0, "", "")
        with mock.patch.object(docker_host.subprocess, "run", return_value=result) as run:
            DockerRunnerHost(target).sync_shared_env(values)

        self.assertIn("WANDB_API_KEY=wandb-value", run.call_args.kwargs["input"])
        self.assertNotIn("wandb-value", " ".join(run.call_args.args[0]))

    def test_create_train_container_preserves_payload_contract(self) -> None:
        result = subprocess.CompletedProcess([], 0, "", "")
        with mock.patch.object(docker_host, "_run_machine_docker", return_value=result) as run:
            outcome = DockerRunnerHost(machine()).create_train_container(
                launch_id="train-12",
                container_name="rlab-job-beast-test-train-train-12",
                runtime_image_ref=RUNTIME_IMAGE_REF,
                labels={LAUNCH_ID_LABEL: "train-12"},
            )

        self.assertTrue(outcome.ok)
        args = run.call_args.args[1]
        text = " ".join(args)
        self.assertEqual(args[0], "create")
        self.assertIn("--stop-timeout 300", text)
        self.assertIn("rlab-container-entrypoint rlab run-job", text)
        self.assertIn("--payload /input/payloads/train-12.json", text)
        self.assertIn("--label rlab.launch-id=train-12", text)
        self.assertIn(
            "src=/tmp/rlab/payloads/train-12.json,dst=/input/payloads/train-12.json",
            text,
        )
        self.assertIn("src=/tmp/rlab/outputs/train-12,dst=/output/train-12", text)
        self.assertNotIn("/tmp/rlab/payloads:/input/payloads", text)
        self.assertNotIn("/tmp/rlab/outputs:/output", text)
        self.assertNotIn("RLAB_ROM_CACHE_DIR", text)
        self.assertNotIn("/roms", text)

    def test_rom_container_mounts_only_the_required_digest_read_only(self) -> None:
        result = subprocess.CompletedProcess([], 0, "", "")
        with mock.patch.object(docker_host, "_run_machine_docker", return_value=result) as run:
            outcome = DockerRunnerHost(machine()).create_train_container(
                launch_id="train-12",
                container_name="rlab-job-beast-test-train-train-12",
                runtime_image_ref=RUNTIME_IMAGE_REF,
                labels={LAUNCH_ID_LABEL: "train-12"},
                rom_asset_manifest=ROM_MANIFEST,
            )

        self.assertTrue(outcome.ok)
        text = " ".join(run.call_args.args[1])
        digest = ROM_MANIFEST["sha256"]
        self.assertIn(
            f"src=/tmp/rlab/rom-cache/sha256/{digest},"
            f"dst=/rom-cache/sha256/{digest},bind-recursive=disabled,readonly",
            text,
        )
        self.assertIn("RLAB_ROM_CACHE_DIR=/rom-cache", text)
        self.assertNotIn("RLAB_ROM_DIR", text)
        self.assertNotIn("RLAB_IMPORT_ROMS", text)

    def test_rom_cache_helper_uses_runtime_image_and_writable_cache(self) -> None:
        result = subprocess.CompletedProcess([], 0, "", "")
        with mock.patch.object(docker_host, "_run_machine_docker", return_value=result) as run:
            outcome = DockerRunnerHost(machine()).ensure_rom_cache(
                launch_id="train-12",
                runtime_image_ref=RUNTIME_IMAGE_REF,
                rom_asset_manifest=ROM_MANIFEST,
            )

        self.assertTrue(outcome.ok)
        text = " ".join(run.call_args.args[1])
        self.assertIn("run --rm", text)
        digest = ROM_MANIFEST["sha256"]
        self.assertIn(
            f"src=/tmp/rlab/rom-cache/sha256/{digest},dst=/rom-cache/sha256/{digest}",
            text,
        )
        self.assertNotIn("/tmp/rlab/rom-cache:/rom-cache", text)
        self.assertIn("python -m rlab.rom_cache", text)
        self.assertIn("/input/payloads/train-12.json", text)

    def test_list_job_containers_filters_managed_job_label(self) -> None:
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
        result = subprocess.CompletedProcess([], 0, output, "")
        with mock.patch.object(docker_host, "_run_machine_docker", return_value=result):
            containers = DockerRunnerHost(machine()).list_job_containers()

        self.assertEqual([container.name for container in containers], ["current"])
        self.assertEqual(containers[0].launch_id, "train-12")
        self.assertEqual(containers[0].labels[JOB_CONTAINER_LABEL], "true")

    def test_result_observation_distinguishes_absent_and_unreachable(self) -> None:
        host = DockerRunnerHost(machine())
        with mock.patch.object(
            docker_host,
            "_run_machine_shell",
            return_value=subprocess.CompletedProcess([], 4, "", ""),
        ):
            self.assertEqual(host.observe_result("/output/train-1").state, "absent")
        with mock.patch.object(
            docker_host,
            "_run_machine_shell",
            return_value=subprocess.CompletedProcess([], 255, "", "unreachable"),
        ):
            observation = host.observe_result("/output/train-1")
        self.assertEqual(observation.state, "error")
        self.assertIn("unreachable", observation.error or "")

    def test_local_payload_and_result_round_trip_uses_real_subprocess(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            host = DockerRunnerHost(machine(host_root=str(root)))
            Path(host.machine.paths.outputs_dir).mkdir(parents=True)
            host.prepare_legacy_workspace("train-12", {"job_id": 12})
            self.assertEqual(
                json.loads(Path(host.payload_host_path("train-12")).read_text(encoding="utf-8")),
                {"job_id": 12},
            )
            output = Path(host.output_host_path("train-12"))
            self.assertTrue(output.is_dir())
            host.prepare_legacy_workspace("train-12", {"job_id": 12})
            (output / "result.json").write_text('{"status":"succeeded"}\n', encoding="utf-8")
            observation = host.observe_result(str(output))

        self.assertEqual(observation.state, "present")
        self.assertEqual(observation.payload, {"status": "succeeded"})

    def test_remote_legacy_workspace_creates_only_exact_leaf_with_sudo(self) -> None:
        host = DockerRunnerHost(
            machine(backend="docker_ssh", host_root="/home/tsilva/rlab")
        )
        result = subprocess.CompletedProcess([], 0, "", "")
        with mock.patch.object(
            docker_host,
            "_run_machine_shell",
            side_effect=[result, result],
        ) as run:
            host.prepare_legacy_workspace("train-176", {"job_id": 176})

        prepare_script = run.call_args_list[0].args[1]
        self.assertIn("sudo -n install -d", prepare_script)
        self.assertIn("-m 0700 /home/tsilva/rlab/outputs/train-176", prepare_script)
        self.assertNotIn("chown", prepare_script)
        self.assertIn("stat -c %u:%g", prepare_script)

    def test_checkpoint_coordinator_recovery_command_is_bounded(self) -> None:
        host = DockerRunnerHost(machine())
        result = subprocess.CompletedProcess([], 0, "", "")
        with (
            mock.patch.object(docker_host, "_run_machine_docker", return_value=result) as run,
            mock.patch.object(host, "container_is_absent", return_value=True),
        ):
            docker_host.run_checkpoint_coordinator_container(
                host,
                launch_id="train-12",
                run_name="run-12",
                runtime_image_ref=RUNTIME_IMAGE_REF,
            )

        args = run.call_args_list[0].args[1]
        text = " ".join(args)
        self.assertIn("--drain-and-exit", args)
        self.assertIn("/output/train-12/runs/run-12", text)
        self.assertEqual(run.call_args.kwargs["timeout"], 900)

    def test_wandb_recovery_container_is_cpu_only_and_bounded(self) -> None:
        host = DockerRunnerHost(machine())
        result = subprocess.CompletedProcess([], 0, "", "")
        attestation = {"container_id": "cid", "container_name": "recovery"}
        with (
            mock.patch.object(docker_host, "_run_machine_docker", return_value=result) as run,
            mock.patch.object(host, "attest_recovery_container", return_value=attestation),
            mock.patch.object(
                host,
                "remove_container",
                return_value=docker_host.HostOperationResult(ok=True),
            ),
            mock.patch.object(host, "container_is_absent", return_value=True),
        ):
            docker_host.run_wandb_publisher_recovery_container(
                host,
                launch_id="train-12",
                run_name="run-12",
                runtime_image_ref=RUNTIME_IMAGE_REF,
            )

        args = run.call_args_list[0].args[1]
        self.assertIn("rlab.wandb_publisher", args)
        self.assertIn("/output/train-12/publisher.stop", args)
        self.assertNotIn("--gpus", args)
        self.assertIn("rlab-wandb-recovery-train-12", args)
        self.assertIn("105s", args)
        start_call = next(call for call in run.call_args_list if call.args[1][:2] == ["start", "--attach"])
        self.assertEqual(start_call.kwargs["timeout"], 120)

    def test_import_order_has_no_cycles(self) -> None:
        env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import rlab.docker_host, rlab.job_queue, rlab.fleet, rlab.modal_eval_cli",
            ],
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_train_attestation_rejects_an_extra_broad_workspace_bind(self) -> None:
        host = DockerRunnerHost(machine())
        document = [
            {
                "Id": "container-id",
                "Config": {
                    "Image": docker_host.docker_image_ref(RUNTIME_IMAGE_REF),
                    "Labels": {
                        LAUNCH_ID_LABEL: "train-12",
                        docker_host.MANAGED_LABEL: "true",
                    },
                    "Entrypoint": ["rlab-container-entrypoint"],
                    "Cmd": ["rlab", "run-job"],
                },
                "Mounts": [
                    {
                        "Type": "bind",
                        "Source": host.payload_host_path("train-12"),
                        "Destination": host.payload_container_path("train-12"),
                        "RW": False,
                    },
                    {
                        "Type": "bind",
                        "Source": host.output_host_path("train-12"),
                        "Destination": host.output_container_path("train-12"),
                        "RW": True,
                    },
                    {
                        "Type": "bind",
                        "Source": host.machine.paths.host_root,
                        "Destination": "/unexpected",
                        "RW": True,
                    },
                ],
            }
        ]
        result = subprocess.CompletedProcess([], 0, json.dumps(document), "")
        with mock.patch.object(docker_host, "_run_machine_docker", return_value=result):
            with self.assertRaisesRegex(RuntimeError, "broad/exclusive"):
                host.attest_train_container(
                    container_name="train-container",
                    launch_id="train-12",
                    runtime_image_ref=RUNTIME_IMAGE_REF,
                )


if __name__ == "__main__":
    unittest.main()
