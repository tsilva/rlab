from __future__ import annotations

import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


SCRIPT = Path(__file__).parents[1] / "scripts" / "study.py"
SPEC = importlib.util.spec_from_file_location("autoresearch_study", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def return_evidence(mean: float, p05: float, peak: float) -> dict[str, object]:
    return {
        "evidence_mode": "return",
        "return_evidence_valid": True,
        "return_tail_mean": mean,
        "return_tail_p05": p05,
        "return_peak": peak,
    }


class ReturnEvidenceTests(unittest.TestCase):
    def test_infers_return_mode_without_success_termination(self) -> None:
        train = {
            "task": {"termination": {"failure": ["serve_stall"]}},
            "selection_rank": ["max(eval/full/episode/return/mean)"],
        }
        self.assertEqual(study.infer_evidence_mode(train), "return")

    def test_success_termination_takes_precedence(self) -> None:
        train = {
            "task": {"termination": {"success": ["level_complete"]}},
            "selection_rank": ["max(eval/full/episode/return/mean)"],
        }
        self.assertEqual(study.infer_evidence_mode(train), "success")

    def test_return_pair_ranking_prioritizes_worst_seed_tail_mean(self) -> None:
        state = {"evidence_mode": "return", "rung_caps": {"pair": 50}}
        stable = {
            "id": "stable",
            "pair_runs": [
                {"training_evidence": return_evidence(100.0, 50.0, 180.0)},
                {"training_evidence": return_evidence(90.0, 45.0, 170.0)},
            ],
        }
        fragile = {
            "id": "fragile",
            "pair_runs": [
                {"training_evidence": return_evidence(140.0, 70.0, 220.0)},
                {"training_evidence": return_evidence(80.0, 40.0, 230.0)},
            ],
        }
        stable_score = study.candidate_score(state, stable)
        fragile_score = study.candidate_score(state, fragile)
        self.assertLess(study.evidence_key(stable_score), study.evidence_key(fragile_score))

    def test_return_screen_uses_tail_mean_then_floor_then_peak(self) -> None:
        state = {
            "evidence_mode": "return",
            "waves": [
                {
                    "phase": "search-screen",
                    "round": 1,
                    "candidate_id": "lower-floor",
                    "seeds": [123],
                    "terminal_runs": [
                        {"training_evidence": return_evidence(100.0, 30.0, 200.0)}
                    ],
                },
                {
                    "phase": "search-screen",
                    "round": 1,
                    "candidate_id": "higher-floor",
                    "seeds": [123],
                    "terminal_runs": [
                        {"training_evidence": return_evidence(100.0, 40.0, 180.0)}
                    ],
                },
            ],
        }
        selected = study.selected_return_screen(state, 1)
        self.assertIsNotNone(selected)
        self.assertEqual(selected["candidate_id"], "higher-floor")

    def test_percentile_interpolates(self) -> None:
        self.assertEqual(study.percentile([0.0, 10.0], 0.5), 5.0)


class StudyDiscoveryTests(unittest.TestCase):
    def test_init_ignores_incomplete_study_from_different_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
            goal = root / "goal.yaml"
            recipe = root / "recipe.yaml"
            goal.write_text("goal: current\n", encoding="utf-8")
            recipe.write_text("recipe: current\n", encoding="utf-8")
            subprocess.run(["git", "add", "goal.yaml", "recipe.yaml"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-qm", "current"], cwd=root, check=True)
            head = study.git_head(root)
            studies = root / "runs" / "autoresearch" / "old"
            studies.mkdir(parents=True)
            (studies / "study.json").write_text(
                json.dumps(
                    {
                        "schema_version": study.SCHEMA_VERSION,
                        "status": "paused",
                        "source_sha": "different-source",
                        "goal_path": "goal.yaml",
                        "recipe_path": "recipe.yaml",
                        "recipe_preimage_sha256": study.file_sha256(recipe),
                        "apply": None,
                        "policy": {"strong_threshold": study.DEFAULT_STRONG_THRESHOLD},
                        "input_hash": "old-input",
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                root=str(root),
                goal="goal.yaml",
                recipe="recipe.yaml",
                strong_threshold=study.DEFAULT_STRONG_THRESHOLD,
            )
            with mock.patch.object(study, "compose_train_document", side_effect=RuntimeError("new-study")):
                with self.assertRaisesRegex(RuntimeError, "new-study"):
                    study.command_init(args)
            self.assertEqual(study.git_head(root), head)

    def test_materializes_seed_in_reserved_description(self) -> None:
        state = {
            "study_id": "abcdef123456",
            "candidates": {"candidate": {"delta": {"learning_rate": 0.000125}}},
        }
        wave = {
            "candidate_id": "candidate",
            "phase": "search-pair",
            "timesteps": 50,
        }
        overrides = study.materialized_recipe_overrides(state, wave, 139)
        self.assertEqual(overrides[0], "train.backend.config.learning_rate=0.000125")
        self.assertIn("seed 139", overrides[-1])
        self.assertNotIn("{seed}", overrides[-1])


if __name__ == "__main__":
    unittest.main()
