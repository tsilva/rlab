from __future__ import annotations

import argparse
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rlab.benchmark import main as benchmark_main
from rlab.benchmark_profiles import (
    build_benchmark_commands,
    find_benchmark_profile,
    load_benchmark_profile,
    load_benchmark_profiles,
)
from rlab.main import COMMANDS
from experiments.scripts.benchmarks.benchmark_env_sps import benchmark_config


class BenchmarkProfileTests(unittest.TestCase):
    def test_stable_retro_env_throughput_benchmark_always_requests_all_info(self) -> None:
        config = benchmark_config(
            argparse.Namespace(
                env_provider="stable-retro-turbo",
                game="SuperMarioBros-Nes-v0",
                state="Level1-1",
            )
        )

        self.assertEqual(config.env_args["info_filter"], "all")

    def test_checked_in_benchmark_profiles_validate(self) -> None:
        profiles = load_benchmark_profiles()

        self.assertGreaterEqual(len(profiles), 6)
        self.assertEqual(
            sorted(profile.name for profile in profiles),
            [
                "artifact-storage-smoke-mario-l11",
                "container-smoke-train-image",
                "eval-contract-mario-l11",
                "fleet-capacity-rtx4090",
                "local-smoke-mario-l11",
                "retro-env-throughput-mario-l11",
                "train-loop-throughput-mario-l11",
            ],
        )
        self.assertTrue(all(profile.path.suffix == ".yaml" for profile in profiles))

    def test_checked_in_benchmark_profiles_are_yaml_not_json(self) -> None:
        benchmark_files = sorted(Path("experiments/benchmarks").rglob("*"))
        json_profiles = [
            path
            for path in benchmark_files
            if path.suffix == ".json" and path.parent.name in {"benchmarks", "profiles"}
        ]
        self.assertEqual(json_profiles, [])

    def test_benchmark_profile_loader_rejects_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.json"
            path.write_text('{"name":"bad","kind":"local_smoke"}\n', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "must be YAML"):
                load_benchmark_profile(path)

    def test_env_throughput_profile_rejects_state_none_without_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.yaml"
            path.write_text(
                """
schema_version: 1
name: bad
kind: env_throughput
game: SuperMarioBros-Nes-v0
state: State.NONE
modes: [fast]
envs: [1]
steps: 10
warmup: 1
expectations: {}
""",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "actual saved state"):
                load_benchmark_profile(path)

    def test_benchmark_profile_rejects_non_executable_gate_vocabulary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.yaml"
            path.write_text(
                """
schema_version: 1
name: bad
kind: env_throughput
game: SuperMarioBros-Nes-v0
state: Level1-1
steps: 10
warmup: 1
gates: {}
""",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "gates is unsupported"):
                load_benchmark_profile(path)

    def test_benchmark_profile_rejects_unknown_kind_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.yaml"
            path.write_text(
                """
schema_version: 1
name: bad
kind: env_throughput
game: SuperMarioBros-Nes-v0
state: Level1-1
steps: 10
warmup: 1
timstepz: 10
expectations: {}
""",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "unknown field.*timstepz"):
                load_benchmark_profile(path)

    def test_local_smoke_uses_queue_backed_local_fleet_commands(self) -> None:
        profile = find_benchmark_profile("local-smoke-mario-l11")
        commands = build_benchmark_commands(profile)

        self.assertEqual(
            [command.label for command in commands],
            ["train-local-smoke", "local-jobs-status"],
        )
        self.assertEqual(commands[0].argv[:2], ("rlab", "train"))
        self.assertNotIn("local", commands[0].argv[:3])
        self.assertIn("--machine", commands[0].argv)
        self.assertIn("local-macbook", commands[0].argv)
        self.assertIn("--wait", commands[0].argv)
        self.assertIn("terminal", commands[0].argv)
        self.assertEqual(commands[1].argv[:3], ("rlab", "jobs", "status"))
        self.assertIn("local-macbook", commands[1].argv)

    def test_env_throughput_generates_mode_env_matrix(self) -> None:
        profile = find_benchmark_profile("retro-env-throughput-mario-l11")
        commands = build_benchmark_commands(profile)

        self.assertEqual(
            [command.label for command in commands],
            ["compare-1env", "compare-16env", "compare-32env"],
        )
        for command in commands:
            self.assertIn("experiments/scripts/benchmarks/benchmark_env_sps.py", command.argv)
            self.assertIn("stable-retro-turbo", command.argv)
            self.assertIn("Level1-1", command.argv)
            self.assertEqual(command.env, {"STABLE_RETRO_DISABLE_AUDIO": "1"})
            self.assertEqual(command.argv[command.argv.index("--max-overhead") + 1], "0.05")

    def test_fleet_capacity_uses_unified_rlab_commands(self) -> None:
        profile = find_benchmark_profile("fleet-capacity-rtx4090")
        commands = build_benchmark_commands(profile)

        self.assertEqual(commands[0].argv[:2], ("rlab", "train"))
        self.assertIn("--machine", commands[0].argv)
        self.assertIn("beast-3", commands[0].argv)
        self.assertIn("--wait", commands[0].argv)
        self.assertIn("running", commands[0].argv)
        self.assertEqual(commands[1].argv[:3], ("rlab", "jobs", "status"))
        self.assertIn("--machine", commands[1].argv)
        self.assertIn("beast-3", commands[1].argv)

    def test_fleet_capacity_rejects_unknown_spec_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.yaml"
            path.write_text(
                """
schema_version: 1
name: bad
kind: fleet_capacity
spec_file: experiments/goals/example/recipes/candidate.yaml
runtime_image_ref_file: rlab-train-image.json
expectations: {}
""",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "recipe_file"):
                load_benchmark_profile(path)

    def test_fleet_capacity_rejects_workers_above_machine_capacity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.yaml"
            path.write_text(
                """
schema_version: 1
name: bad
kind: fleet_capacity
machine: local-macbook
requested_workers: 2
goal_file: experiments/goals/SuperMarioBros-Nes-v0/Level1-1/_goal.yaml
recipe_file: experiments/recipes/mario/single/ppo.yaml
runtime_image_ref_file: rlab-train-image.json
expectations: {}
""",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "exceeds.*capacity"):
                load_benchmark_profile(path)

    def test_benchmark_train_config_rejects_unknown_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.yaml"
            path.write_text(
                """
schema_version: 1
name: bad
kind: train_loop_throughput
goal_file: experiments/goals/SuperMarioBros-Nes-v0/Level1-1/_goal.yaml
recipe_file: experiments/recipes/mario/single/ppo.yaml
recipe_overrides:
- train.backend.config.timstepz=512
expectations: {}
""",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "unexpected fields.*timstepz"):
                load_benchmark_profile(path)

    def test_train_benchmarks_derive_environment_contract_from_recipe(self) -> None:
        for name in (
            "train-loop-throughput-mario-l11",
            "artifact-storage-smoke-mario-l11",
        ):
            with self.subTest(name=name):
                profile = find_benchmark_profile(name)
                self.assertIn("goal_file", profile.payload)
                self.assertIn("recipe_file", profile.payload)
                self.assertNotIn("train_config", profile.payload)
                command = build_benchmark_commands(profile)[0]
                self.assertEqual(
                    command.argv[-2:],
                    ("--train-config-json", "/dev/stdin"),
                )
                self.assertIsNotNone(command.stdin)
                train_config = json.loads(command.stdin)
                self.assertIn("task", train_config)
                self.assertIn("timesteps", train_config)
                backend = train_config["training_backend"]
                self.assertEqual(backend["id"], "sb3.ppo")

    def test_queue_backed_benchmark_commands_include_goal_and_recipe(self) -> None:
        for name in ("local-smoke-mario-l11", "fleet-capacity-rtx4090"):
            with self.subTest(name=name):
                command = build_benchmark_commands(find_benchmark_profile(name))[0]
                self.assertIn("--goal-file", command.argv)
                self.assertIn("--recipe-file", command.argv)

    def test_benchmark_is_registered_on_unified_cli(self) -> None:
        self.assertIn("benchmark", COMMANDS)

    def test_benchmark_list_json_cli(self) -> None:
        stdout = io.StringIO()
        with patch("sys.stdout", stdout):
            exit_code = benchmark_main(["list", "--json"])

        self.assertEqual(exit_code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertIn("retro-env-throughput-mario-l11", {row["name"] for row in payload})

    def test_benchmark_show_emits_installed_cli_commands(self) -> None:
        stdout = io.StringIO()
        with patch("sys.stdout", stdout):
            exit_code = benchmark_main(["show", "fleet-capacity-rtx4090"])

        self.assertEqual(exit_code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertNotIn("execution_mode", payload)
        self.assertEqual(payload["commands"][0]["argv"][:2], ["rlab", "train"])


if __name__ == "__main__":
    unittest.main()
