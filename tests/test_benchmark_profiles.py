from __future__ import annotations

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


class BenchmarkProfileTests(unittest.TestCase):
    def test_checked_in_benchmark_profiles_validate(self) -> None:
        profiles = load_benchmark_profiles()

        self.assertGreaterEqual(len(profiles), 7)
        self.assertEqual(
            sorted(profile.name for profile in profiles),
            [
                "artifact-storage-smoke-mario-l11",
                "container-smoke-train-image",
                "eval-contract-mario-l11",
                "fleet-capacity-rtx4090",
                "local-smoke-mario-l11",
                "ppo-loop-throughput-mario-l11",
                "retro-env-throughput-mario-l11",
            ],
        )
        self.assertTrue(all(profile.path.suffix == ".yaml" for profile in profiles))

    def test_checked_in_benchmark_specs_are_yaml_not_json(self) -> None:
        benchmark_files = sorted(Path("experiments/benchmarks").rglob("*"))
        json_specs = [
            path
            for path in benchmark_files
            if path.suffix == ".json"
            and path.parent.name in {"benchmarks", "profiles"}
        ]
        self.assertEqual(json_specs, [])

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
gates: {}
""",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "actual saved state"):
                load_benchmark_profile(path)

    def test_local_smoke_command_uses_active_python_and_eval_model_path(self) -> None:
        profile = find_benchmark_profile("local-smoke-mario-l11")
        commands = build_benchmark_commands(profile)

        self.assertEqual([command.label for command in commands], ["train-smoke", "eval-smoke"])
        self.assertIn("-m", commands[0].argv)
        self.assertIn("rlab.main", commands[0].argv)
        self.assertIn("local", commands[0].argv)
        self.assertIn("--preset", commands[0].argv)
        self.assertIn("runs/benchmark_local_smoke_mario_l11/final_model.zip", commands[1].argv)

    def test_env_throughput_generates_mode_env_matrix(self) -> None:
        profile = find_benchmark_profile("retro-env-throughput-mario-l11")
        commands = build_benchmark_commands(profile)

        self.assertEqual([command.label for command in commands], ["fast-1env", "fast-16env", "fast-32env"])
        for command in commands:
            self.assertIn("experiments/scripts/benchmarks/benchmark_env_sps.py", command.argv)
            self.assertIn("Level1-1", command.argv)
            self.assertEqual(command.env, {"STABLE_RETRO_DISABLE_AUDIO": "1"})

    def test_fleet_capacity_uses_unified_rlab_commands(self) -> None:
        profile = find_benchmark_profile("fleet-capacity-rtx4090")
        commands = build_benchmark_commands(profile)

        self.assertEqual(commands[0].argv[1:4], ("-m", "rlab.main", "train"))
        self.assertEqual(commands[1].argv[1:5], ("-m", "rlab.main", "fleet", "plan"))
        self.assertEqual(commands[2].argv[1:5], ("-m", "rlab.main", "fleet", "reconcile"))

    def test_fleet_capacity_rejects_legacy_spec_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.yaml"
            path.write_text(
                """
schema_version: 1
name: bad
kind: fleet_capacity
spec_file: experiments/goals/example/recipes/candidate.yaml
runtime_image_ref_file: rlab-train-image.json
gates: {}
""",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "recipe_file"):
                load_benchmark_profile(path)

    def test_benchmark_is_registered_on_unified_cli(self) -> None:
        self.assertIn("benchmark", COMMANDS)

    def test_benchmark_list_json_cli(self) -> None:
        stdout = io.StringIO()
        with patch("sys.stdout", stdout):
            exit_code = benchmark_main(["list", "--json"])

        self.assertEqual(exit_code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertIn("retro-env-throughput-mario-l11", {row["name"] for row in payload})


if __name__ == "__main__":
    unittest.main()
