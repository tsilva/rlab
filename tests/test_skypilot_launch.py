from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from stable_retro_ppo.skypilot_launch import (
    EndpointCheck,
    REQUIRED_ENV_KEYS,
    build_launch_command,
    configured_api_endpoints,
    collect_results,
    ensure_skypilot_api,
    format_results_table,
    launch_report,
    manifest_from_wandb_config,
    preflight_checks,
    render_task_yaml,
    sparse_log_events,
    write_launch_report,
)


INSTANCE_CONFIG = {
    "instances": {
        "rtx4090": {
            "accelerator": "RTX4090",
            "cpus": "12+",
            "memory": "48+",
            "image_id": "docker:test",
            "max_children": 5,
            "env_threads": 4,
            "api_endpoints": ["http://healthy.example", "http://lan.example"],
        }
    }
}


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


class SkyPilotLaunchTests(unittest.TestCase):
    def test_render_task_injects_standard_rtx4090_training_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            rom_path = repo_root / "rom.bin"
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

    def test_preflight_flags_missing_secrets_and_empty_descriptions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "pyproject.toml").write_text(
                'dependencies = ["stable-retro-turbo==1.0.0.post14"]\n',
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
                'dependencies = ["stable-retro-turbo==1.0.0.post14"]\n',
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
                'dependencies = ["stable-retro-turbo==1.0.0.post14"]\n',
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
                'dependencies = ["stable-retro-turbo==1.0.0.post14"]\n',
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
        cmd = build_launch_command("cluster", Path("task.yaml"))
        self.assertEqual(cmd[:5], ["sky", "launch", "-c", "cluster", "-y"])
        self.assertIn("--env", cmd)
        self.assertIn("CHECKPOINT_BUCKET_URI", cmd)
        self.assertIn("--secret", cmd)
        self.assertIn("WANDB_API_KEY", cmd)

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
                        "|    completion_episode_rate        | 0.8         |",
                        "|    completion_episodes_total      | 80          |",
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
                    "|    completion_episodes_total      | 1            |",
                    "|    completion_episode_rate        | 0.8          |",
                    "wandb artifact logged: candidate-final (s3://bucket/game/run/final.zip)",
                    "candidate exit status: 0",
                ]
            )
        )

        messages = [event.message for event in events]
        self.assertTrue(any("wandb.ai/e/p/runs/abc" in message for message in messages))
        self.assertTrue(any("first completion" in message for message in messages))
        self.assertTrue(any("crossed 0.80" in message for message in messages))
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
                        "|    completion_episode_rate        | 0.9         |",
                        "candidate exit status: 0",
                    ]
                ),
                encoding="utf-8",
            )
            write_launch_report(log_path, report_path)
            report = launch_report(log_path)
            report_exists = report_path.exists()

        self.assertEqual(report["summary"]["completion_rate"], "0.9")
        self.assertEqual(report["artifacts"][0]["plane"], "r2")
        self.assertTrue(report_exists)

    def test_configured_api_endpoints_and_dry_run_selection(self) -> None:
        self.assertEqual(
            configured_api_endpoints(INSTANCE_CONFIG),
            ["http://healthy.example", "http://lan.example"],
        )

        original = __import__("stable_retro_ppo.skypilot_launch", fromlist=["check_api_endpoint"])
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
