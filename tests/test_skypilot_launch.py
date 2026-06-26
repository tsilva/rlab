from __future__ import annotations

import json
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from rlab.skypilot_cli import cmd_launch_runner, cmd_queue_train
from rlab.skypilot_launch import (
    EndpointCheck,
    REQUIRED_ENV_KEYS,
    build_launch_command,
    build_runner_launch_command,
    configured_api_endpoints,
    collect_results,
    ensure_skypilot_api,
    format_results_table,
    launch_report,
    manifest_from_wandb_config,
    preflight_checks,
    preflight_runner_profile,
    render_runner_task_yaml,
    render_task_yaml,
    runner_cluster_name,
    sparse_log_events,
    write_launch_report,
)
from rlab.modal_launch import modal_launch_summary, preflight_modal_manifest


INSTANCE_CONFIG = {
    "instances": {
        "rtx4090": {
            "aliases": ["beast-3"],
            "accelerator": "RTX4090",
            "cpus": "12+",
            "memory": "48+",
            "image_id": "docker:test",
            "infra": "k8s/beast-3",
            "max_children": 5,
            "env_threads": 4,
            "api_endpoints": ["http://healthy.example", "http://lan.example"],
        },
        "runpod-l4": {
            "accelerator": "L4",
            "cpus": "5+",
            "memory": "29+",
            "image_id": "docker:runpod",
            "infra": "runpod",
            "max_children": 1,
            "children": 1,
            "env_threads": 2,
        },
        "local-macbook": {
            "kind": "local",
            "accelerator": "MPS",
            "max_children": 1,
            "children": 1,
            "env_threads": 0,
        },
        "modal-t4": {
            "aliases": ["modal"],
            "kind": "modal",
            "accelerator": "T4",
            "modal_gpu": "T4",
            "cpu": 16.0,
            "memory_mib": 32768,
            "max_children": 1,
            "children": 1,
            "env_threads": 0,
        }
    }
}

RUNTIME_IMAGE_REF = (
    "docker:ghcr.io/tsilva/rlab/rlab-train@sha256:"
    "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
)


def sample_manifest() -> dict:
    return {
        "name": "retro-test-4090",
        "cluster": "cluster-test",
        "run_name_prefix": "test_post12",
        "wandb_group_prefix": "group-test",
        "log_dir": "logs/test_4090",
        "venv_name": "venv-test",
        "game": "TestGame-Platform",
        "rom_source": "rom.bin",
        "rom_mount_path": "~/roms/TestGame-Platform/rom.bin",
        "wandb_tags": ["screen", "rtx4090"],
        "base_train": {
            "timesteps": 1024,
            "n_envs": 16,
            "n_steps": 512,
            "batch_size": 512,
            "n_epochs": 10,
            "eval_freq": 0,
            "eval_episodes": 0,
            "device": "cuda",
            "env_threads": 4,
            "torch_num_threads": 1,
            "wandb": True,
            "normalize_advantage": False,
        },
        "runs": [
            {
                "suffix": "lr15e4_seed23",
                "seed": 23,
                "learning_rate": 0.00015,
                "run_description": "Focused RTX4090 launch rendering test.",
            }
        ],
    }


def sample_runner_profile() -> dict:
    return {
        "name": "runner-test-4090",
        "cluster": "runner-cluster",
        "profile_id": "mario-ppo/post20/rtx4090-task-conditioned-v1",
        "game": "TestGame-Platform",
        "rom_source": "rom.bin",
        "rom_mount_path": "~/roms/TestGame-Platform/rom.bin",
        "extra_file_mounts": {
            "~/roms/TestGame-Platform/Level1-1.state": "states/Level1-1.state",
        },
        "runtime_image_ref": RUNTIME_IMAGE_REF,
        "stable_retro_turbo_version": "1.0.0.post21",
        "venv_name": "runner-test-venv",
        "log_dir": "logs/train_runner",
        "workers": 5,
        "max_jobs": 0,
        "poll_seconds": 15,
        "status_goal": "mario-level1-1-1-2-100of100",
        "smoke": {
            "env": {
                "states": ["Level1-1", "Level1-2"],
                "state_probs": [0.5, 0.5],
                "task_conditioning": True,
                "hud_crop_top": 32,
                "reward_mode": "score",
                "action_set": "simple",
                "done_on_info_json": (
                    '{"life_loss":["lives","decrease"],'
                    '"level_change":[["levelHi","levelLo"],"change"]}'
                ),
                "completion_x_threshold": 0,
                "max_pool_frames": False,
            }
        },
    }


class SkyPilotLaunchTests(unittest.TestCase):
    def test_render_task_injects_standard_rtx4090_training_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            rom_path = (repo_root / "rom.bin").resolve()
            manifest = sample_manifest()
            yaml = render_task_yaml(manifest, INSTANCE_CONFIG, repo_root)

        self.assertIn("accelerators: {RTX4090: 1}", yaml)
        self.assertIn('GROUP="group-test-${TS}"', yaml)
        self.assertIn("--run-description", yaml)
        self.assertIn("--env-threads 4", yaml)
        self.assertIn("--no-normalize-advantage", yaml)
        self.assertIn("--game TestGame-Platform", yaml)
        self.assertIn("retro.data.get_romfile_path('TestGame-Platform')", yaml)
        self.assertIn(str(rom_path), yaml)

    def test_render_task_uses_named_target_resources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            manifest = sample_manifest()
            manifest["target"] = "runpod-l4"
            yaml = render_task_yaml(manifest, INSTANCE_CONFIG, repo_root)

        self.assertIn("accelerators: {L4: 1}", yaml)
        self.assertIn("cpus: 5+", yaml)
        self.assertIn("memory: 29+", yaml)
        self.assertIn("target=runpod", yaml)

    def test_render_task_accepts_target_alias_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            yaml = render_task_yaml(
                sample_manifest(),
                INSTANCE_CONFIG,
                repo_root,
                target_override="beast-3",
            )

        self.assertIn("accelerators: {RTX4090: 1}", yaml)

    def test_render_task_includes_extra_file_mounts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            manifest = sample_manifest()
            manifest["extra_file_mounts"] = {
                "~/roms/TestGame-Platform/Level1-2.state": "roms/Level1-2.state",
            }
            yaml = render_task_yaml(manifest, INSTANCE_CONFIG, repo_root)

        self.assertIn(
                "  ~/roms/TestGame-Platform/Level1-2.state: "
                f"{(repo_root / 'roms/Level1-2.state').resolve()}",
            yaml,
        )

    def test_preflight_flags_missing_secrets_and_empty_descriptions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "pyproject.toml").write_text(
                'dependencies = ["stable-retro-turbo==1.0.0.post21"]\n',
                encoding="utf-8",
            )
            manifest = sample_manifest()
            manifest["runs"][0]["run_description"] = ""
            checks = preflight_checks(manifest, INSTANCE_CONFIG, repo_root, env={})

        messages = [check.message for check in checks]
        self.assertTrue(any("missing env/secrets" in message for message in messages))
        self.assertTrue(any("empty run_description" in message for message in messages))
        self.assertTrue(any(check.level == "error" for check in checks))

    def test_preflight_passes_with_required_env_and_existing_rom(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "pyproject.toml").write_text(
                'dependencies = ["stable-retro-turbo==1.0.0.post21"]\n',
                encoding="utf-8",
            )
            (repo_root / "rom.bin").write_bytes(b"rom")
            env = {key: "value" for key in REQUIRED_ENV_KEYS}
            checks = preflight_checks(sample_manifest(), INSTANCE_CONFIG, repo_root, env=env)

        self.assertFalse(any(check.level == "error" for check in checks))
        self.assertTrue(any(check.level == "ok" for check in checks))

    def test_preflight_flags_extensionless_remote_rom_mount(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "pyproject.toml").write_text(
                'dependencies = ["stable-retro-turbo==1.0.0.post21"]\n',
                encoding="utf-8",
            )
            (repo_root / "rom.bin").write_bytes(b"rom")
            env = {key: "value" for key in REQUIRED_ENV_KEYS}
            manifest = sample_manifest()
            manifest["rom_mount_path"] = "~/roms/TestGame-Platform/rom"
            checks = preflight_checks(manifest, INSTANCE_CONFIG, repo_root, env=env)

        messages = [check.message for check in checks]
        self.assertTrue(any("preserve the ROM file extension" in message for message in messages))
        self.assertTrue(any(check.level == "error" for check in checks))

    def test_preflight_checks_checkpoint_bucket_rom_prefix_from_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "pyproject.toml").write_text(
                'dependencies = ["stable-retro-turbo==1.0.0.post21"]\n',
                encoding="utf-8",
            )
            (repo_root / "rom.bin").write_bytes(b"rom")
            env = {key: "value" for key in REQUIRED_ENV_KEYS}
            env["CHECKPOINT_BUCKET_URI"] = "s3://bucket/wandb"
            manifest = sample_manifest()
            manifest["base_train"]["wandb_artifact_storage_uri"] = "${CHECKPOINT_BUCKET_URI}"
            checks = preflight_checks(manifest, INSTANCE_CONFIG, repo_root, env=env)

        messages = [check.message for check in checks]
        self.assertTrue(any("does not include the game id" in message for message in messages))
        self.assertFalse(any(check.level == "error" for check in checks))

    def test_launch_command_uses_standard_env_and_secret_flags(self) -> None:
        cmd = build_launch_command("cluster", Path("task.yaml"), infra="runpod")
        self.assertEqual(cmd[:5], ["sky", "launch", "-c", "cluster", "-y"])
        self.assertIn("--infra", cmd)
        self.assertIn("runpod", cmd)
        self.assertIn("--env", cmd)
        self.assertIn("CHECKPOINT_BUCKET_URI", cmd)
        self.assertIn("--secret", cmd)
        self.assertIn("WANDB_API_KEY", cmd)
        self.assertNotIn("DATABASE_URL", cmd)

    def test_runner_launch_command_includes_available_database_secret(self) -> None:
        env = {key: "value" for key in REQUIRED_ENV_KEYS}
        env["DATABASE_URL"] = "postgres://example"

        cmd = build_runner_launch_command("cluster", Path("task.yaml"), env=env)

        self.assertIn("--secret", cmd)
        self.assertIn("DATABASE_URL", cmd)
        self.assertNotIn("TRAIN_QUEUE_DATABASE_URL", cmd)
        self.assertNotIn("DIRECT_DATABASE_URL", cmd)

    def test_runner_launch_command_requires_database_secret(self) -> None:
        env = {key: "value" for key in REQUIRED_ENV_KEYS}

        with self.assertRaisesRegex(ValueError, "database"):
            build_runner_launch_command("cluster", Path("task.yaml"), env=env)

    def test_launch_runner_command_accepts_cluster_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "experiments").mkdir()
            (repo_root / "experiments" / "instances.json").write_text(
                json.dumps(INSTANCE_CONFIG),
                encoding="utf-8",
            )
            (repo_root / ".env").write_text(
                "\n".join(
                    [
                        *(f"{key}=value" for key in REQUIRED_ENV_KEYS),
                        "DATABASE_URL=postgres://example",
                    ]
                ),
                encoding="utf-8",
            )
            (repo_root / "rom.bin").write_bytes(b"rom")
            (repo_root / "states").mkdir()
            (repo_root / "states" / "Level1-1.state").write_bytes(b"state")
            profile_path = repo_root / "profile.json"
            profile_path.write_text(json.dumps(sample_runner_profile()), encoding="utf-8")
            output_path = repo_root / "runner.yaml"
            args = type(
                "Args",
                (),
                {
                    "profile": profile_path,
                    "repo_root": str(repo_root),
                    "instances": None,
                    "target": "runpod-l4",
                    "output": output_path,
                    "cluster": "runner-runpod-l4",
                    "detach_run": False,
                    "execute": False,
                    "sparse": True,
                    "log_output": None,
                    "down_on_complete": False,
                },
            )()

            output = io.StringIO()
            with redirect_stdout(output):
                returncode = cmd_launch_runner(args)

        self.assertEqual(returncode, 0)
        self.assertIn("cluster: runner-runpod-l4", output.getvalue())
        self.assertIn("sky launch -c runner-runpod-l4", output.getvalue())
        self.assertIn("--infra runpod", output.getvalue())

    def test_queue_train_dry_run_renders_digest_runner_without_enqueueing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "experiments").mkdir()
            (repo_root / "experiments" / "instances.json").write_text(
                json.dumps(INSTANCE_CONFIG),
                encoding="utf-8",
            )
            (repo_root / ".env").write_text(
                "\n".join(
                    [
                        *(f"{key}=value" for key in REQUIRED_ENV_KEYS),
                        "DATABASE_URL=postgres://example",
                    ]
                ),
                encoding="utf-8",
            )
            (repo_root / "rom.bin").write_bytes(b"rom")
            (repo_root / "states").mkdir()
            (repo_root / "states" / "Level1-1.state").write_bytes(b"state")
            profile_path = repo_root / "profile.json"
            profile_path.write_text(json.dumps(sample_runner_profile()), encoding="utf-8")
            output_path = repo_root / "runner.yaml"
            args = type(
                "Args",
                (),
                {
                    "profile": profile_path,
                    "repo_root": str(repo_root),
                    "instances": None,
                    "target": "beast-3",
                    "goal": "mario-goal",
                    "spec_id": 123,
                    "train_config_json": '{"timesteps":1024}',
                    "runtime_image_ref": RUNTIME_IMAGE_REF,
                    "priority": 0,
                    "max_attempts": 1,
                    "run_name": "digest_smoke",
                    "run_description": "Digest-pinned smoke.",
                    "wandb_group": None,
                    "wandb_tag": [],
                    "origin_decision_id": None,
                    "direct": False,
                    "ensure_runner": True,
                    "cluster": None,
                    "output": output_path,
                    "execute": False,
                    "detach_run": True,
                    "sparse": True,
                    "log_output": None,
                },
            )()

            expected_cluster = runner_cluster_name(
                profile_id=sample_runner_profile()["profile_id"],
                target="rtx4090",
                runtime_image_ref=RUNTIME_IMAGE_REF,
            )
            output = io.StringIO()
            with redirect_stdout(output):
                returncode = cmd_queue_train(args)
            rendered_exists = output_path.exists()

        self.assertEqual(returncode, 0)
        text = output.getvalue()
        self.assertIn(f"cluster: {expected_cluster}", text)
        self.assertIn(f"runtime_image_ref: {RUNTIME_IMAGE_REF}", text)
        self.assertIn("job_target: rtx4090", text)
        self.assertIn("--detach-run", text)
        self.assertIn("dry_run: pass --execute to enqueue the train job", text)
        self.assertTrue(rendered_exists)

    def test_render_runner_task_claims_profile_without_embedding_train_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            profile = sample_runner_profile()
            yaml = render_runner_task_yaml(profile, INSTANCE_CONFIG, repo_root)

        self.assertIn("name: runner-test-4090", yaml)
        self.assertIn("stable-retro-turbo==1.0.0.post21", yaml)
        self.assertIn("-m rlab.train_runner", yaml)
        self.assertIn("--profile mario-ppo/post20/rtx4090-task-conditioned-v1", yaml)
        self.assertIn(f"--runtime-image-ref {RUNTIME_IMAGE_REF}", yaml)
        self.assertIn("--run-target rtx4090", yaml)
        self.assertIn("--workers 5", yaml)
        self.assertIn("--max-jobs 0", yaml)
        self.assertIn("--status-goal mario-level1-1-1-2-100of100", yaml)
        self.assertIn("PPO('MultiInputPolicy' if isinstance(obs, dict) else 'CnnPolicy'", yaml)
        self.assertNotIn("-m rlab.train --", yaml)

    def test_render_runner_task_uses_prebuilt_image_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            profile = sample_runner_profile()
            profile["image_id"] = RUNTIME_IMAGE_REF
            profile["prebuilt_image"] = True
            yaml = render_runner_task_yaml(profile, INSTANCE_CONFIG, repo_root)

        self.assertIn(
            f'image_id: "{RUNTIME_IMAGE_REF}"',
            yaml,
        )
        self.assertIn("rlab-container-entrypoint rlab-container-smoke", yaml)
        self.assertIn("rlab-container-entrypoint python -m rlab.train_runner", yaml)
        self.assertIn('export RLAB_ROM_DIR="$HOME/roms"', yaml)
        self.assertNotIn("workdir: .", yaml)
        self.assertNotIn("uv sync", yaml)
        self.assertNotIn("stable-retro-turbo==1.0.0.post21", yaml)
        self.assertNotIn('PY="$HOME/runner-test-venv/bin/python"', yaml)

    def test_render_runner_task_rejects_prebuilt_mutable_image_tag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            profile = sample_runner_profile()
            profile["image_id"] = "docker:ghcr.io/tsilva/rlab/rlab-train:git-test"
            profile["runtime_image_ref"] = profile["image_id"]
            profile["prebuilt_image"] = True

            with self.assertRaisesRegex(ValueError, "immutable docker digest ref"):
                render_runner_task_yaml(profile, INSTANCE_CONFIG, repo_root)

    def test_render_runner_task_resolves_relative_mounts(self) -> None:
        repo_root = Path(".")
        profile = sample_runner_profile()

        yaml = render_runner_task_yaml(profile, INSTANCE_CONFIG, repo_root)

        cwd = Path.cwd().resolve()
        self.assertIn(
            f"  ~/roms/TestGame-Platform/rom.bin: {cwd / 'rom.bin'}",
            yaml,
        )
        self.assertIn(
            f"  ~/roms/TestGame-Platform/Level1-1.state: {cwd / 'states/Level1-1.state'}",
            yaml,
        )

    def test_preflight_runner_profile_passes_with_required_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "pyproject.toml").write_text(
                'dependencies = ["stable-retro-turbo==1.0.0.post21"]\n',
                encoding="utf-8",
            )
            (repo_root / "rom.bin").write_bytes(b"rom")
            (repo_root / "states").mkdir()
            (repo_root / "states" / "Level1-1.state").write_bytes(b"state")
            env = {key: "value" for key in REQUIRED_ENV_KEYS}
            env["TRAIN_QUEUE_DATABASE_URL"] = "postgres://example"
            checks = preflight_runner_profile(
                sample_runner_profile(),
                INSTANCE_CONFIG,
                repo_root,
                env=env,
            )

        self.assertFalse(any(check.level == "error" for check in checks))
        self.assertTrue(any(check.level == "ok" for check in checks))

    def test_preflight_runner_profile_requires_database_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "pyproject.toml").write_text(
                'dependencies = ["stable-retro-turbo==1.0.0.post21"]\n',
                encoding="utf-8",
            )
            (repo_root / "rom.bin").write_bytes(b"rom")
            (repo_root / "states").mkdir()
            (repo_root / "states" / "Level1-1.state").write_bytes(b"state")
            env = {key: "value" for key in REQUIRED_ENV_KEYS}
            checks = preflight_runner_profile(
                sample_runner_profile(),
                INSTANCE_CONFIG,
                repo_root,
                env=env,
            )

        messages = [check.message for check in checks]
        self.assertTrue(any("database URL" in message for message in messages))
        self.assertTrue(any(check.level == "error" for check in checks))

    def test_preflight_runner_profile_rejects_mutable_runtime_tag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "pyproject.toml").write_text(
                'dependencies = ["stable-retro-turbo==1.0.0.post21"]\n',
                encoding="utf-8",
            )
            (repo_root / "rom.bin").write_bytes(b"rom")
            (repo_root / "states").mkdir()
            (repo_root / "states" / "Level1-1.state").write_bytes(b"state")
            env = {key: "value" for key in REQUIRED_ENV_KEYS}
            env["TRAIN_QUEUE_DATABASE_URL"] = "postgres://example"
            profile = sample_runner_profile()
            profile["runtime_image_ref"] = "docker:ghcr.io/tsilva/rlab/rlab-train:latest"
            checks = preflight_runner_profile(
                profile,
                INSTANCE_CONFIG,
                repo_root,
                env=env,
            )

        messages = [check.message for check in checks]
        self.assertTrue(any("immutable docker digest ref" in message for message in messages))
        self.assertTrue(any(check.level == "error" for check in checks))

    def test_preflight_rejects_non_skypilot_target_for_skypilot_launch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "pyproject.toml").write_text(
                'dependencies = ["stable-retro-turbo==1.0.0.post21"]\n',
                encoding="utf-8",
            )
            (repo_root / "rom.bin").write_bytes(b"rom")
            env = {key: "value" for key in REQUIRED_ENV_KEYS}
            checks = preflight_checks(
                sample_manifest(),
                INSTANCE_CONFIG,
                repo_root,
                env=env,
                target_override="modal-t4",
            )

        messages = [check.message for check in checks]
        self.assertTrue(any("matching compute launcher" in message for message in messages))
        self.assertTrue(any(check.level == "error" for check in checks))

    def test_modal_launch_summary_renders_manifest_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "experiments").mkdir()
            (repo_root / "experiments" / "instances.json").write_text(
                json.dumps(INSTANCE_CONFIG),
                encoding="utf-8",
            )
            manifest_path = repo_root / "manifest.json"
            manifest = sample_manifest()
            manifest["target"] = "modal-t4"
            summary = modal_launch_summary(
                manifest,
                manifest_path,
                repo_root=repo_root,
                instances_path=None,
                target_override=None,
            )

        self.assertEqual(summary.target, "modal-t4")
        self.assertEqual(summary.gpu, "T4")
        self.assertEqual(summary.cpu, 16.0)
        self.assertEqual(summary.memory_mib, 32768)
        self.assertEqual(summary.command[:3], ["modal", "run", "src/rlab/modal_app.py::launch_manifest"])
        self.assertIn("--target", summary.command)
        self.assertIn("modal-t4", summary.command)

    def test_modal_preflight_allows_volume_uploaded_roms(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            manifest = sample_manifest()
            manifest["target"] = "modal-t4"
            checks = preflight_modal_manifest(manifest, INSTANCE_CONFIG, repo_root)

        messages = [check.message for check in checks]
        self.assertFalse(any(check.level == "error" for check in checks))
        self.assertTrue(any("Modal preflight passed" in message for message in messages))
        self.assertTrue(any("Modal can still run if ROMs were uploaded" in message for message in messages))

    def test_collect_results_parses_log_and_run_markers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log_dir = root / "logs"
            run_dir = root / "runs" / "candidate"
            log_dir.mkdir()
            run_dir.mkdir(parents=True)
            (log_dir / "candidate.log").write_text(
                "\n".join(
                    [
                        "wandb artifact logged: candidate-final (s3://bucket/prefix/final.zip)",
                        "|    total_timesteps                | 123456      |",
                        "| train/done/                       |             |",
                        "|    all                            | 80          |",
                        "|    level_change                   | 50          |",
                        "candidate exit status: 0",
                    ]
                ),
                encoding="utf-8",
            )
            (run_dir / "early_stop.txt").write_text(
                "completion_rate=0.800000\n"
                "total_completed_episodes=80\n"
                "timesteps=123456\n",
                encoding="utf-8",
            )
            rows = collect_results(log_dir, root / "runs")

        self.assertEqual(rows[0]["artifact_plane"], "r2")
        self.assertEqual(rows[0]["exit_status"], "0")
        self.assertEqual(rows[0]["timesteps"], "123456")
        self.assertIn("candidate", format_results_table(rows))

    def test_manifest_from_wandb_config_softcodes_rom_and_clones_train_shape(self) -> None:
        manifest = manifest_from_wandb_config(
            "tsilva/SuperMarioBros-NES/lexxixz3",
            {
                "game": "OtherGame-Genesis",
                "state": "Level1",
                "seed": 24,
                "timesteps": 5_000_000,
                "n_envs": 16,
                "env_threads": 4,
                "target_kl": 0.04,
                "states": ["Level1", "Level2"],
                "wandb_tags": "baseline,post12",
            },
            "roms/other.md",
            artifact_storage_uri="s3://bucket/wandb",
        )

        self.assertEqual(manifest["game"], "OtherGame-Genesis")
        self.assertEqual(manifest["rom_source"], "roms/other.md")
        self.assertEqual(manifest["rom_mount_path"], "~/roms/OtherGame-Genesis/other.md")
        self.assertEqual(manifest["base_train"]["states"], "Level1,Level2")
        self.assertEqual(manifest["base_train"]["target_kl"], 0.04)
        self.assertEqual(manifest["base_train"]["wandb_project"], "SuperMarioBros-NES")
        self.assertIn("source-run-lexxixz3", manifest["wandb_tags"])

    def test_sparse_log_events_emit_only_milestones(self) -> None:
        events = sparse_log_events(
            "\n".join(
                [
                    "wandb: View run at https://wandb.ai/e/p/runs/abc",
                    "|    total_timesteps                | 1000000      |",
                    "| train/done/                       |             |",
                    "|    all                            | 10           |",
                    "wandb artifact logged: candidate-final (s3://bucket/game/run/final.zip)",
                    "candidate exit status: 0",
                ]
            )
        )

        messages = [event.message for event in events]
        self.assertTrue(any("wandb.ai/e/p/runs/abc" in message for message in messages))
        self.assertTrue(any("done count crossed 1" in message for message in messages))
        self.assertTrue(any("done count crossed 10" in message for message in messages))
        self.assertTrue(any("candidate-final" in message for message in messages))

    def test_launch_report_writes_structured_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log_path = root / "launch.log"
            report_path = root / "report.json"
            log_path.write_text(
                "\n".join(
                    [
                        "wandb artifact logged: candidate-final (s3://bucket/game/run/final.zip)",
                        "| train/done/                       |             |",
                        "|    all                            | 90          |",
                        "candidate exit status: 0",
                    ]
                ),
                encoding="utf-8",
            )
            write_launch_report(log_path, report_path)
            report = launch_report(log_path)
            report_exists = report_path.exists()

        self.assertEqual(report["summary"]["done_all"], "90")
        self.assertEqual(report["artifacts"][0]["plane"], "r2")
        self.assertTrue(report_exists)

    def test_configured_api_endpoints_and_dry_run_selection(self) -> None:
        self.assertEqual(
            configured_api_endpoints(INSTANCE_CONFIG),
            ["http://healthy.example", "http://lan.example"],
        )

        original = __import__("rlab.skypilot_launch", fromlist=["check_api_endpoint"])
        check_api_endpoint = original.check_api_endpoint
        try:
            original.check_api_endpoint = lambda endpoint: EndpointCheck(
                endpoint,
                endpoint.endswith("healthy.example"),
                "fake",
            )
            checks, command = ensure_skypilot_api(
                INSTANCE_CONFIG,
                repo_root=Path("."),
                execute=False,
            )
        finally:
            original.check_api_endpoint = check_api_endpoint

        self.assertTrue(checks[0].ok)
        self.assertEqual(command, ["sky", "api", "login", "-e", "http://healthy.example"])


if __name__ == "__main__":
    unittest.main()
