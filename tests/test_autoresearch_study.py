from __future__ import annotations

import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / ".codex/skills/autoresearch/scripts/study.py"
SPEC = importlib.util.spec_from_file_location("autoresearch_study", SCRIPT)
assert SPEC and SPEC.loader
study = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(study)


def base_state(root: Path) -> dict:
    recipe = root / "recipe.yaml"
    goal = root / "goal.yaml"
    recipe.write_text("recipe_id: candidate\n", encoding="utf-8")
    goal.write_text("goal: candidate\n", encoding="utf-8")
    backend = {
        "n_steps": 64,
        "batch_size": 128,
        "n_epochs": 4,
        "learning_rate": 0.001,
        "gamma": 0.9,
        "gae_lambda": 0.95,
    }
    baseline_id = study.candidate_id({})
    return {
        "schema_version": 1,
        "study_id": "a" * 32,
        "input_hash": "b" * 64,
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
        "status": "active",
        "repo_root": str(root),
        "source_sha": "c" * 40,
        "goal_path": "goal.yaml",
        "recipe_path": "recipe.yaml",
        "recipe_preimage": recipe.read_text(encoding="utf-8"),
        "recipe_preimage_sha256": study.file_sha256(recipe),
        "source_files": [
            {"path": "goal.yaml", "sha256": study.file_sha256(goal)},
            {"path": "recipe.yaml", "sha256": study.file_sha256(recipe)},
        ],
        "backend": "sb3.ppo",
        "n_envs": 16,
        "search_seeds": [123, 139],
        "confirmation_seeds": [155, 171, 187, 203, 219],
        "runtime": None,
        "policy": {
            "machine": "beast-3",
            "max_reserved_jobs": 48,
            "stale_round_limit": 3,
            "confirmation_runs": 5,
            "confirmation_required": 4,
        },
        "baseline": {
            "backend_config": backend,
            "tunables": study.numeric_tunables(backend),
            "train_config": {
                "timesteps": 100_000,
                "training_backend": {"id": "sb3.ppo", "config": backend},
            },
            "update_work_per_env_step": study.update_work(backend, "sb3.ppo", 16),
            "censor_step": 100_352,
        },
        "baseline_candidate_id": baseline_id,
        "candidates": {
            baseline_id: {
                "id": baseline_id,
                "delta": {},
                "created_round": 0,
                "search_runs": [
                    {
                        "run_id": 1,
                        "seed": 123,
                        "accepted_verified": True,
                        "promoted_step": 80_000,
                    },
                    {
                        "run_id": 2,
                        "seed": 139,
                        "accepted_verified": False,
                        "promoted_step": None,
                    },
                ],
                "confirmation_runs": [],
            }
        },
        "waves": [],
        "reserved_jobs": 2,
        "search_round": 0,
        "stale_rounds": 0,
        "incumbent_candidate_id": baseline_id,
        "incumbent_evidence": None,
        "excluded_candidates": [],
        "confirmation": None,
        "winner": None,
        "apply": None,
    }


class AutoresearchStudyTests(unittest.TestCase):
    def test_init_pins_composition_and_resumes_the_single_incomplete_study(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            goal = root / "goal.yaml"
            recipe = root / "recipe.yaml"
            goal.write_text("goal: demo\n", encoding="utf-8")
            recipe.write_text("recipe_id: demo\n", encoding="utf-8")
            backend = {
                "n_steps": 64,
                "batch_size": 128,
                "n_epochs": 4,
                "learning_rate": 0.001,
            }
            document = {
                "train_config": {
                    "timesteps": 100_000,
                    "n_envs": 16,
                    "checkpoint_eval_backend": "modal",
                    "stop_on_acceptance": True,
                    "training_backend": {"id": "sb3.ppo", "config": backend},
                },
                "_composition": {
                    "source_files": [
                        {"path": str(goal), "sha256": study.file_sha256(goal)},
                        {"path": str(recipe), "sha256": study.file_sha256(recipe)},
                    ]
                },
            }
            args = SimpleNamespace(root=str(root), goal=str(goal), recipe=str(recipe))
            first_output = io.StringIO()
            second_output = io.StringIO()
            with (
                mock.patch.object(study, "git_head", return_value="a" * 40),
                mock.patch.object(
                    study,
                    "git_blob_sha256",
                    side_effect=lambda _root, _sha, path: study.file_sha256(root / path),
                ),
                mock.patch.object(study, "compose_train_document", return_value=document),
                mock.patch.object(study, "accepts_first_training_success", return_value=False),
            ):
                with redirect_stdout(first_output):
                    study.command_init(args)
                with redirect_stdout(second_output):
                    study.command_init(args)

            first = json.loads(first_output.getvalue())
            second = json.loads(second_output.getvalue())
            state = study.load_state(Path(first["study"]))

        self.assertFalse(first["resumed"])
        self.assertTrue(second["resumed"])
        self.assertEqual(first["study"], second["study"])
        self.assertEqual(state["search_seeds"], [123, 139])
        self.assertEqual(state["confirmation_seeds"], [155, 171, 187, 203, 219])

    def test_candidate_identity_excludes_trace_only_fields(self) -> None:
        left = study.candidate_id({"learning_rate": 0.002, "description": "left"})
        right = study.candidate_id({"description": "right", "learning_rate": 0.002})

        self.assertEqual(left, right)

    def test_candidate_validation_enforces_coherent_groups_and_ppo_rollout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = base_state(Path(temporary))
            self.assertEqual(
                study.validate_delta(state, {"learning_rate": 0.002}),
                {"learning_rate": 0.002},
            )
            with self.assertRaisesRegex(ValueError, "one coherent group"):
                study.validate_delta(state, {"learning_rate": 0.002, "gamma": 0.95})
            with self.assertRaisesRegex(ValueError, "must divide fixed rollout"):
                study.validate_delta(state, {"batch_size": 300})

    def test_score_censors_rejections_at_frozen_effective_cap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = base_state(Path(temporary))
            candidate = state["candidates"][state["baseline_candidate_id"]]

            score = study.candidate_score(state, candidate)

        self.assertEqual(score["accepted_verified"], 1)
        self.assertEqual(score["worst_censored_step"], study.frozen_censor_step(state))

    def test_search_reservation_uses_three_parallel_paired_cohorts_at_six_slots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state_path = root / "runs/autoresearch/test/study.json"
            state = base_state(root)
            study.atomic_json(state_path, state)
            args = SimpleNamespace(
                study=str(state_path),
                phase="search",
                candidates_json=json.dumps(
                    [
                        {"delta": {"learning_rate": 0.0005}},
                        {"delta": {"learning_rate": 0.0015}},
                        {"delta": {"learning_rate": 0.002}},
                    ]
                ),
                effective_capacity=6,
                active_reservations=0,
            )
            output = io.StringIO()
            with (
                mock.patch.object(study, "git_head", return_value=state["source_sha"]),
                redirect_stdout(output),
            ):
                study.command_reserve(args)

            payload = json.loads(output.getvalue())
            updated = study.load_state(state_path)

        self.assertTrue(payload["launch_concurrently"])
        self.assertEqual(len(payload["reserved"]), 3)
        self.assertEqual(updated["reserved_jobs"], 8)
        self.assertTrue(
            all("--existing-runtime-only" in row["command"] for row in payload["reserved"])
        )
        self.assertEqual(
            {len(wave["seeds"]) for wave in updated["waves"]},
            {2},
        )

    def test_next_action_reserves_exactly_five_confirmation_runs_after_staleness(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = base_state(Path(temporary))
            state["stale_rounds"] = 3

            action = study.next_action(state)

        self.assertEqual(action["action"], "reserve_confirmation")
        self.assertEqual(len(state["confirmation_seeds"]), 5)

    def test_resume_requires_a_pinned_source_and_records_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state_path = root / "runs/autoresearch/test/study.json"
            state = base_state(root)
            state["status"] = "paused"
            state["pause_reason"] = {"event": "attention_required"}
            study.atomic_json(state_path, state)
            output = io.StringIO()
            with (
                mock.patch.object(study, "git_head", return_value=state["source_sha"]),
                redirect_stdout(output),
            ):
                study.command_resume(
                    SimpleNamespace(
                        study=str(state_path), reason="budget block cleared on monitored run"
                    )
                )
            resumed = study.load_state(state_path)

        self.assertTrue(json.loads(output.getvalue())["resumed"])
        self.assertEqual(resumed["status"], "active")
        self.assertEqual(len(resumed["resume_history"]), 1)

    def test_forward_lifecycle_confirms_four_of_five_and_applies_exact_postimage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state_path = root / "runs/autoresearch/test/study.json"
            state = base_state(root)
            study.atomic_json(state_path, state)
            candidate_delta = {"learning_rate": 0.0005}

            def reserve(phase: str, candidates: list[dict] | None = None) -> dict:
                output = io.StringIO()
                with redirect_stdout(output):
                    study.command_reserve(
                        SimpleNamespace(
                            study=str(state_path),
                            phase=phase,
                            candidates_json=json.dumps(candidates or []),
                            effective_capacity=6,
                            active_reservations=4 if phase == "search" else 0,
                        )
                    )
                return json.loads(output.getvalue())

            def launch_payload(wave: dict, start_run: int) -> dict:
                current = study.load_state(state_path)
                overrides = study.expected_recipe_overrides(current, wave)
                runtime = {
                    "image_ref": "docker:ghcr.io/owner/image@sha256:" + "d" * 64,
                    "input_sha256": "e" * 64,
                    "build_source_sha": "f" * 40,
                }
                rows = [
                    {
                        "run_id": start_run + index,
                        "source_sha": current["source_sha"],
                        "submission": {
                            "key": wave["submission_key"],
                            "seed": seed,
                            "request_hash": "1" * 64,
                            "goal_path": current["goal_path"],
                            "goal_sha256": next(
                                item["sha256"]
                                for item in current["source_files"]
                                if item["path"] == current["goal_path"]
                            ),
                            "recipe_path": current["recipe_path"],
                            "recipe_sha256": current["recipe_preimage_sha256"],
                            "recipe_overrides": overrides,
                        },
                        "runtime": runtime,
                    }
                    for index, seed in enumerate(wave["seeds"])
                ]
                return {
                    "batch_id": wave["batch_id"],
                    "run_ids": [row["run_id"] for row in rows],
                    "runs": rows,
                    "runtime_image_ref": runtime["image_ref"],
                    "runtime_input_sha256": runtime["input_sha256"],
                    "runtime_build_source_sha": runtime["build_source_sha"],
                }

            def record_launch(wave: dict, start_run: int) -> None:
                output = io.StringIO()
                with redirect_stdout(output):
                    study.command_record_launch(
                        SimpleNamespace(
                            study=str(state_path),
                            submission_key=wave["submission_key"],
                            payload_json=json.dumps(launch_payload(wave, start_run)),
                            payload_file=None,
                        )
                    )
                self.assertFalse(json.loads(output.getvalue())["paused"])

            def record_terminal(
                wave: dict, run_id: int, seed: int, *, accepted: bool, step: int | None
            ) -> None:
                event = {
                    "event": "terminal",
                    "run_id": run_id,
                    "submission": {"key": wave["submission_key"], "seed": seed},
                    "terminal_classification": "accepted" if accepted else "goal_rejected",
                    "verified_success": accepted,
                    "wandb": {
                        "remote_verified": accepted,
                        "url": f"https://wandb.ai/e/p/runs/{run_id}",
                    },
                    "evaluation": {"promoted_step": step},
                    "artifacts": {
                        "wandb_artifact": f"e/p/run-{run_id}-checkpoint:step-{step}"
                        if step is not None
                        else None
                    },
                }
                with redirect_stdout(io.StringIO()):
                    study.command_record_terminal(
                        SimpleNamespace(
                            study=str(state_path),
                            submission_key=None,
                            run_id=None,
                            seed=None,
                            event_json=json.dumps(event),
                            event_file=None,
                        )
                    )

            with mock.patch.object(study, "git_head", return_value=state["source_sha"]):
                search = reserve("search", [{"delta": candidate_delta}])
                search_wave = study.load_state(state_path)["waves"][-1]
                record_launch(search_wave, 10)
                record_terminal(
                    search_wave, 10, search_wave["seeds"][0], accepted=True, step=50_000
                )
                record_terminal(
                    search_wave, 11, search_wave["seeds"][1], accepted=True, step=60_000
                )
                with redirect_stdout(io.StringIO()):
                    study.command_close_round(SimpleNamespace(study=str(state_path), round=1))

                after_search = study.load_state(state_path)
                after_search["policy"]["stale_round_limit"] = 0
                study.atomic_json(state_path, after_search)

                self.assertEqual(
                    search["reserved"][0]["candidate_id"], study.candidate_id(candidate_delta)
                )
                self.assertEqual(
                    study.next_action(study.load_state(state_path))["action"],
                    "reserve_confirmation",
                )

                reserve("confirmation")
                confirmation_wave = study.load_state(state_path)["waves"][-1]
                self.assertEqual(len(confirmation_wave["seeds"]), 5)
                record_launch(confirmation_wave, 20)
                for index, seed in enumerate(confirmation_wave["seeds"]):
                    record_terminal(
                        confirmation_wave,
                        20 + index,
                        seed,
                        accepted=index < 4,
                        step=40_000 + index * 1_000 if index < 4 else None,
                    )
                with redirect_stdout(io.StringIO()):
                    study.command_close_confirmation(SimpleNamespace(study=str(state_path)))

                confirmed = study.load_state(state_path)
                self.assertEqual(confirmed["winner"]["accepted_verified"], 4)
                self.assertEqual(confirmed["winner"]["delta"], candidate_delta)

                postimage = root / "planned-recipe.yaml"
                postimage.write_text(
                    "recipe_id: candidate\ntrain:\n  backend:\n    config:\n"
                    "      learning_rate: 0.0005\n",
                    encoding="utf-8",
                )
                with redirect_stdout(io.StringIO()):
                    study.command_prepare_apply(
                        SimpleNamespace(study=str(state_path), postimage_file=str(postimage))
                    )
                (root / "recipe.yaml").write_text(
                    postimage.read_text(encoding="utf-8"), encoding="utf-8"
                )
                expected_train = study.load_state(state_path)["baseline"]["train_config"]
                expected_train = json.loads(json.dumps(expected_train))
                expected_train["training_backend"]["config"].update(candidate_delta)
                with (
                    mock.patch.object(
                        study,
                        "compose_train_document",
                        return_value={"train_config": expected_train},
                    ),
                    redirect_stdout(io.StringIO()),
                ):
                    study.command_complete_apply(SimpleNamespace(study=str(state_path)))

            completed = study.load_state(state_path)

        self.assertEqual(completed["status"], "done")
        self.assertEqual(completed["apply"]["kind"], "recipe_patch")


if __name__ == "__main__":
    unittest.main()
