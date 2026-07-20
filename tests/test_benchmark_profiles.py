from __future__ import annotations

import argparse
from dataclasses import replace
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rlab.benchmark import main as benchmark_main, validate_benchmark_results
from rlab.benchmark_profiles import (
    build_benchmark_commands,
    find_benchmark_profile,
    load_benchmark_profile,
    load_benchmark_profiles,
)
from rlab.main import COMMANDS
from rlab.metric_store import MetricStore, metric_store_path
from experiments.scripts.benchmarks.benchmark_env_sps import benchmark_config


class BenchmarkProfileTests(unittest.TestCase):
    def test_mario_env_throughput_benchmark_always_requests_all_info(self) -> None:
        config = benchmark_config(
            argparse.Namespace(
                env_provider="supermariobrosnes-turbo",
                game="SuperMarioBros-Nes-v0",
                state="Level1-1",
            )
        )

        self.assertEqual(config.env_args["info_filter"], "all")

    def test_checked_in_benchmark_profiles_validate(self) -> None:
        profiles = load_benchmark_profiles()

        self.assertEqual(len(profiles), 3)
        self.assertEqual(
            sorted(profile.name for profile in profiles),
            [
                "local-smoke-mario-l11",
                "mario-env-throughput-l11",
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
env_provider: supermariobrosnes-turbo
game: SuperMarioBros-Nes-v0
state: State.NONE
modes: [fast]
envs: [1]
steps: 10
warmup: 1
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
env_provider: supermariobrosnes-turbo
game: SuperMarioBros-Nes-v0
state: Level1-1
envs: [1]
modes: [compare]
steps: 10
warmup: 1
gates: {}
""",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "executable kind-specific gate fields"):
                load_benchmark_profile(path)

    def test_benchmark_profile_rejects_unknown_kind_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.yaml"
            path.write_text(
                """
schema_version: 1
name: bad
kind: env_throughput
env_provider: supermariobrosnes-turbo
game: SuperMarioBros-Nes-v0
state: Level1-1
modes: [compare]
envs: [1]
steps: 10
warmup: 1
max_runtime_overhead: 0.05
timstepz: 10
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
            ["train-local-smoke"],
        )
        self.assertEqual(
            commands[0].argv[:4],
            ("rlab", "experiment", "launch", "--from-head"),
        )
        self.assertNotIn("local", commands[0].argv[:4])
        self.assertIn("--machine", commands[0].argv)
        self.assertIn("local-macbook", commands[0].argv)
        self.assertIn("--wait", commands[0].argv)
        self.assertIn("terminal", commands[0].argv)
        self.assertEqual(
            commands[0].argv[commands[0].argv.index("--checkpoint-eval-backend") + 1],
            "none",
        )
        self.assertIn("--json", commands[0].argv)
        self.assertEqual(commands[0].argv[commands[0].argv.index("--seed") + 1], "123")

        from rlab.experiment_cli import build_parser as build_experiment_parser

        parsed = build_experiment_parser().parse_args(list(commands[0].argv[2:]))
        self.assertEqual(parsed.command, "launch")
        self.assertTrue(parsed.from_head)
        self.assertEqual(parsed.machine, "local-macbook")
        self.assertEqual(parsed.wait, "terminal")
        self.assertTrue(parsed.output_json)

    def test_env_throughput_generates_mode_env_matrix(self) -> None:
        profile = find_benchmark_profile("mario-env-throughput-l11")
        commands = build_benchmark_commands(profile)

        self.assertEqual(
            [command.label for command in commands],
            ["compare-1env", "compare-16env", "compare-32env"],
        )
        for command in commands:
            self.assertIn("experiments/scripts/benchmarks/benchmark_env_sps.py", command.argv)
            self.assertIn("supermariobrosnes-turbo", command.argv)
            self.assertIn("Level1-1", command.argv)
            self.assertEqual(command.env, {"STABLE_RETRO_DISABLE_AUDIO": "1"})
            self.assertEqual(command.argv[command.argv.index("--max-overhead") + 1], "0.05")

    def test_benchmark_train_config_rejects_unknown_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.yaml"
            path.write_text(
                """
schema_version: 1
name: bad
kind: train_loop_throughput
goal_file: experiments/goals/SuperMarioBros-Nes-v0/Level1-1/_goal.yaml
recipe_file: experiments/goals/SuperMarioBros-Nes-v0/Level1-1/recipes/ppo.yaml
recipe_overrides:
- train.backend.config.timstepz=512
required_metrics: [train/throughput/loop_fps]
""",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "unexpected fields.*timstepz"):
                load_benchmark_profile(path)

    def test_train_benchmark_rejects_unknown_required_metric(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.yaml"
            path.write_text(
                """
schema_version: 1
name: bad
kind: train_loop_throughput
goal_file: experiments/goals/SuperMarioBros-Nes-v0/Level1-1/_goal.yaml
recipe_file: experiments/goals/SuperMarioBros-Nes-v0/Level1-1/recipes/ppo.yaml
required_metrics: [train/throughput/not_real]
""",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "unknown metric name"):
                load_benchmark_profile(path)

    def test_train_benchmark_derives_current_core_config_from_recipe(self) -> None:
        profile = find_benchmark_profile("train-loop-throughput-mario-l11")
        self.assertIn("goal_file", profile.payload)
        self.assertIn("recipe_file", profile.payload)
        self.assertNotIn("train_config", profile.payload)
        command = build_benchmark_commands(profile)[0]
        self.assertEqual(command.argv[-2:], ("--train-config-json", "/dev/stdin"))
        self.assertIsNotNone(command.stdin)
        train_config = json.loads(command.stdin)
        self.assertEqual(train_config["training_backend"]["id"], "sb3.ppo")
        self.assertEqual(train_config["checkpoint_eval_backend"], "none")
        self.assertEqual(train_config["checkpoint_eval_stages"], [])
        self.assertIsNone(train_config["early_stop"])
        self.assertEqual(train_config["post_train_eval_episodes"], 0)
        self.assertFalse(train_config["wandb"])
        self.assertEqual(train_config["wandb_mode"], "disabled")
        self.assertEqual(train_config["seed"], 123)

    def test_queue_backed_benchmark_commands_include_goal_and_recipe(self) -> None:
        command = build_benchmark_commands(find_benchmark_profile("local-smoke-mario-l11"))[0]
        self.assertIn("--goal-file", command.argv)
        self.assertIn("--recipe-file", command.argv)

    def test_local_smoke_result_gate_requires_successful_terminal_jobs(self) -> None:
        profile = find_benchmark_profile("local-smoke-mario-l11")
        commands = build_benchmark_commands(profile)
        result = {
            "label": "train-local-smoke",
            "returncode": 0,
            "stdout": json.dumps({"wait": {"reached": True}, "jobs": [{"status": "failed"}]}),
        }

        validation = validate_benchmark_results(profile, commands, [result])

        self.assertFalse(validation["passed"])
        self.assertIn("not all successful", validation["issues"][0])

    def test_env_throughput_result_gate_rejects_excess_overhead(self) -> None:
        profile = find_benchmark_profile("mario-env-throughput-l11")
        commands = build_benchmark_commands(profile)
        results = [
            {
                "label": command.label,
                "returncode": 0,
                "stdout": json.dumps(
                    {
                        "mode": "compare",
                        "envs": int(command.argv[command.argv.index("--envs") + 1]),
                        "results": {},
                        "runtime_overhead_fraction": 0.06,
                        "overhead_gate_passed": False,
                    }
                ),
            }
            for command in commands
        ]

        validation = validate_benchmark_results(profile, commands, results)

        self.assertFalse(validation["passed"])
        self.assertEqual(len(validation["issues"]), 3)

    def test_train_loop_result_gate_requires_declared_metrics(self) -> None:
        profile = find_benchmark_profile("train-loop-throughput-mario-l11")
        command = build_benchmark_commands(profile)[0]
        with tempfile.TemporaryDirectory() as tmp:
            config = json.loads(command.stdin or "{}")
            config["runs_dir"] = tmp
            command = replace(command, stdin=json.dumps(config))
            store = MetricStore(metric_store_path(Path(tmp) / config["run_name"]))
            store.init()
            store.append_metrics(
                {name: 1.0 for name in profile.payload["required_metrics"]},
                step=1,
                source="test",
                publish=False,
            )
            result = {"label": "train", "returncode": 0, "stdout": ""}

            validation = validate_benchmark_results(profile, [command], [result])

        self.assertTrue(validation["passed"])
        self.assertEqual(
            sorted(validation["evidence"]["metrics"]),
            sorted(profile.payload["required_metrics"]),
        )

    def test_benchmark_is_registered_on_unified_cli(self) -> None:
        self.assertIn("benchmark", COMMANDS)

    def test_benchmark_list_json_cli(self) -> None:
        stdout = io.StringIO()
        with patch("sys.stdout", stdout):
            exit_code = benchmark_main(["list", "--json"])

        self.assertEqual(exit_code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertIn("mario-env-throughput-l11", {row["name"] for row in payload})

    def test_benchmark_show_emits_installed_cli_commands(self) -> None:
        stdout = io.StringIO()
        with patch("sys.stdout", stdout):
            exit_code = benchmark_main(["show", "local-smoke-mario-l11"])

        self.assertEqual(exit_code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertNotIn("execution_mode", payload)
        self.assertEqual(
            payload["commands"][0]["argv"][:4],
            ["rlab", "experiment", "launch", "--from-head"],
        )


if __name__ == "__main__":
    unittest.main()
