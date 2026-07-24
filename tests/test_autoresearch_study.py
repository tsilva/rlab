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


def immutable_run_id(value: int | str) -> str:
    if isinstance(value, str) and value.startswith("rlab-"):
        return value
    return f"rlab-{int(value):032x}"


def training_evidence(
    *,
    all_starts: bool = True,
    strong: bool = False,
    step: int | None = None,
    peak: float | None = None,
    run_id: int | str = 1,
) -> dict:
    resolved_run_id = immutable_run_id(run_id)
    return {
        "all_starts_succeeded": all_starts,
        "success_counts_by_start": {"Level1-3": 1 if all_starts else 0},
        "peak_window_100_rate_min": peak,
        "first_strong_step": step,
        "strong": strong,
        "observed_max_step": 50_176,
        "strong_threshold": 0.9,
        "wandb_run_id": resolved_run_id,
        "wandb_url": f"https://wandb.ai/e/p/runs/{resolved_run_id}",
        "wandb_state": "finished",
        "collected_at": "2026-01-01T00:00:00Z",
    }


def run_record(
    run_id: int | str,
    seed: int,
    *,
    all_starts: bool = True,
    strong: bool = False,
    step: int | None = None,
    peak: float | None = None,
) -> dict:
    resolved_run_id = immutable_run_id(run_id)
    return {
        "run_id": resolved_run_id,
        "seed": seed,
        "classification": "completed",
        "wandb_run_id": resolved_run_id,
        "wandb_url": f"https://wandb.ai/e/p/runs/{resolved_run_id}",
        "training_evidence": training_evidence(
            all_starts=all_starts,
            strong=strong,
            step=step,
            peak=peak,
            run_id=run_id,
        ),
    }


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
        "schema_version": study.SCHEMA_VERSION,
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
        "configured_starts": ["Level1-3"],
        "screen_seed": 123,
        "pair_seeds": [139, 155],
        "confirmation_seeds": [171, 187, 203, 219, 235],
        "rung_caps": {
            "screen": 20_480,
            "pair": 50_176,
            "confirmation": 100_000,
            "confirmation_effective": 100_352,
        },
        "runtime": None,
        "policy": {
            "compute_target": "b3",
            "max_reserved_jobs": 48,
            "stale_round_limit": 3,
            "confirmation_runs": 5,
            "confirmation_required": 4,
            "strong_threshold": 0.9,
            "checkpoint_evaluation": "disabled",
        },
        "baseline": {
            "backend_config": backend,
            "tunables": study.numeric_tunables(backend),
            "train_config": {
                "timesteps": 100_000,
                "checkpoint_eval_backend": "modal",
                "stop_on_acceptance": True,
                "environment": {"env_config": {"state": "Level1-3", "n_envs": 16}},
                "training_backend": {"id": "sb3.ppo", "config": backend},
            },
            "update_work_per_env_step": study.update_work(backend, "sb3.ppo", 16),
            "rollout_quantum": 1024,
        },
        "baseline_candidate_id": baseline_id,
        "candidates": {
            baseline_id: {
                "id": baseline_id,
                "delta": {},
                "created_round": 0,
                "screen_runs": [],
                "pair_runs": [],
                "confirmation_runs": [],
            }
        },
        "waves": [],
        "reserved_jobs": 0,
        "search_round": 0,
        "stale_rounds": 0,
        "incumbent_candidate_id": None,
        "incumbent_evidence": None,
        "excluded_candidates": [],
        "confirmation": None,
        "winner": None,
        "apply": None,
    }


def add_closed_baseline(state: dict, *, passed: bool = False) -> None:
    baseline = state["candidates"][state["baseline_candidate_id"]]
    record = run_record(1, state["screen_seed"], all_starts=passed)
    baseline["screen_runs"] = [record]
    state["waves"].append(
        {
            "phase": "baseline-screen",
            "round": 0,
            "candidate_id": baseline["id"],
            "seeds": [state["screen_seed"]],
            "timesteps": state["rung_caps"]["screen"],
            "submission_key": "baseline-screen",
            "status": "evidence_complete",
            "run_ids": [record["run_id"]],
            "terminal_runs": [record],
            "closed": True,
        }
    )


def add_ranked_candidate(
    state: dict,
    delta: dict,
    *,
    candidate_run_ids: tuple[int, int] = (10, 11),
    strong: tuple[bool, bool] = (True, True),
    steps: tuple[int | None, int | None] = (30_000, 40_000),
    peaks: tuple[float, float] = (0.95, 0.92),
) -> str:
    identifier = study.candidate_id(delta)
    runs = [
        run_record(
            run_id,
            seed,
            strong=is_strong,
            step=step,
            peak=peak,
        )
        for run_id, seed, is_strong, step, peak in zip(
            candidate_run_ids,
            state["pair_seeds"],
            strong,
            steps,
            peaks,
            strict=True,
        )
    ]
    state["candidates"][identifier] = {
        "id": identifier,
        "delta": delta,
        "created_round": 1,
        "screen_runs": [],
        "pair_runs": runs,
        "confirmation_runs": [],
    }
    state["incumbent_candidate_id"] = identifier
    state["incumbent_evidence"] = study.candidate_score(state, state["candidates"][identifier])
    return identifier


def reserve_args(state_path: Path, phase: str, **kwargs) -> SimpleNamespace:
    return SimpleNamespace(
        study=str(state_path),
        phase=phase,
        candidate_id=kwargs.get("candidate_id"),
        candidates_json=json.dumps(kwargs.get("candidates", [])),
        effective_capacity=kwargs.get("capacity", 1),
        active_reservations=kwargs.get("active", 0),
    )


class AutoresearchStudyTests(unittest.TestCase):
    def test_init_ignores_old_schema_and_resumes_v3_with_distinct_rungs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            goal = root / "goal.yaml"
            recipe = root / "recipe.yaml"
            goal.write_text("goal: demo\n", encoding="utf-8")
            recipe.write_text("recipe_id: demo\n", encoding="utf-8")
            old = root / "runs/autoresearch/old/study.json"
            old.parent.mkdir(parents=True)
            old.write_text(json.dumps({"schema_version": 1, "status": "active"}))
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
                    "environment": {"env_config": {"state": "Level1-3", "n_envs": 16}},
                    "training_backend": {"id": "sb3.ppo", "config": backend},
                },
                "_composition": {
                    "source_files": [
                        {"path": str(goal), "sha256": study.file_sha256(goal)},
                        {"path": str(recipe), "sha256": study.file_sha256(recipe)},
                    ]
                },
            }
            args = SimpleNamespace(
                root=str(root),
                goal=str(goal),
                recipe=str(recipe),
                strong_threshold=0.9,
            )
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
        self.assertEqual(state["schema_version"], study.SCHEMA_VERSION)
        self.assertEqual(state["screen_seed"], 123)
        self.assertEqual(state["pair_seeds"], [139, 155])
        self.assertEqual(state["confirmation_seeds"], [171, 187, 203, 219, 235])
        self.assertLess(state["rung_caps"]["screen"], state["rung_caps"]["pair"])
        self.assertLess(state["rung_caps"]["pair"], state["rung_caps"]["confirmation_effective"])
        self.assertEqual(state["baseline"]["train_config"]["checkpoint_eval_backend"], "modal")

    def test_schema_v1_is_explicitly_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "study.json"
            path.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "v1/v2 studies are historical"):
                study.load_state(path)

    def test_candidate_identity_and_validation_remain_bounded(self) -> None:
        left = study.candidate_id({"learning_rate": 0.002, "description": "left"})
        right = study.candidate_id({"description": "right", "learning_rate": 0.002})
        self.assertEqual(left, right)
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

    def test_configured_starts_supports_materialized_top_level_provider_fields(self) -> None:
        self.assertEqual(study.configured_starts({"state": "Level1-3"}), ["Level1-3"])
        self.assertEqual(study.configured_starts({"states": ["A", "B"]}), ["A", "B"])

    def test_baseline_screen_launch_is_training_only_and_uses_twenty_percent_cap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state_path = root / "runs/autoresearch/test/study.json"
            state = base_state(root)
            study.atomic_json(state_path, state)
            output = io.StringIO()
            with (
                mock.patch.object(study, "git_head", return_value=state["source_sha"]),
                redirect_stdout(output),
            ):
                study.command_reserve(reserve_args(state_path, "baseline-screen"))
            payload = json.loads(output.getvalue())
            wave = study.load_state(state_path)["waves"][0]

        command = payload["reserved"][0]["commands"][0]["command"]
        self.assertEqual(wave["seeds"], [123])
        self.assertEqual(wave["timesteps"], 20_480)
        self.assertIn("--checkpoint-eval-backend", command)
        self.assertEqual(command[command.index("--checkpoint-eval-backend") + 1], "none")
        self.assertIn("train.timesteps=20480", command)
        self.assertNotIn("checkpoint_eval_backend", wave["candidate_id"])

    def test_failed_screen_skips_pair_and_passing_screen_reserves_two_fresh_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            failed = base_state(root)
            add_closed_baseline(failed, passed=False)
            failed["waves"][0]["closed"] = False
            self.assertEqual(study.next_action(failed), {"action": "close_round", "round": 0})

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state_path = root / "runs/autoresearch/test/study.json"
            state = base_state(root)
            add_closed_baseline(state, passed=True)
            state["waves"][0]["closed"] = False
            study.atomic_json(state_path, state)
            action = study.next_action(state)
            self.assertEqual(action["action"], "reserve_baseline_pair")
            with (
                mock.patch.object(study, "git_head", return_value=state["source_sha"]),
                redirect_stdout(io.StringIO()),
            ):
                study.command_reserve(reserve_args(state_path, "baseline-pair"))
            wave = study.load_state(state_path)["waves"][-1]

        self.assertEqual(wave["seeds"], [139, 155])
        self.assertNotIn(123, wave["seeds"])
        self.assertEqual(wave["timesteps"], 50_176)

    def test_search_screen_reserves_one_candidate_for_the_single_b3_slot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state_path = root / "runs/autoresearch/test/study.json"
            state = base_state(root)
            add_closed_baseline(state)
            study.atomic_json(state_path, state)
            candidates = [{"delta": {"learning_rate": 0.0005}}]
            output = io.StringIO()
            with (
                mock.patch.object(study, "git_head", return_value=state["source_sha"]),
                redirect_stdout(output),
            ):
                study.command_reserve(
                    reserve_args(state_path, "search-screen", candidates=candidates)
                )
            payload = json.loads(output.getvalue())
            updated = study.load_state(state_path)

        self.assertFalse(payload["launch_concurrently"])
        self.assertEqual(len(payload["reserved"]), 1)
        self.assertEqual({len(wave["seeds"]) for wave in updated["waves"][1:]}, {1})
        self.assertEqual(updated["reserved_jobs"], 1)
        self.assertTrue(
            all(
                "--checkpoint-eval-backend" in row["commands"][0]["command"]
                for row in payload["reserved"]
            )
        )

    def test_score_uses_strong_count_crossing_censor_and_worst_peak(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = base_state(Path(temporary))
            identifier = add_ranked_candidate(
                state,
                {"learning_rate": 0.0005},
                strong=(True, False),
                steps=(30_000, None),
                peaks=(0.95, 0.7),
            )
            score = study.candidate_score(state, state["candidates"][identifier])

        self.assertEqual(score["strong_seeds"], 1)
        self.assertEqual(score["median_censored_first_strong_step"], 40_088)
        self.assertEqual(score["worst_censored_first_strong_step"], 50_176)
        self.assertEqual(score["worst_peak_window_100_rate_min"], 0.7)

    def test_terminal_then_remote_evidence_is_persisted_idempotently(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state_path = root / "runs/autoresearch/test/study.json"
            state = base_state(root)
            study.atomic_json(state_path, state)
            with (
                mock.patch.object(study, "git_head", return_value=state["source_sha"]),
                redirect_stdout(io.StringIO()),
            ):
                study.command_reserve(reserve_args(state_path, "baseline-screen"))
            current = study.load_state(state_path)
            wave = current["waves"][0]
            runtime = {
                "image_ref": "docker:ghcr.io/o/i@sha256:" + "d" * 64,
                "input_sha256": "e" * 64,
                "build_source_sha": "f" * 40,
            }
            run_id = immutable_run_id(7)
            launch = {
                "runs": [
                    {
                        "run_id": run_id,
                        "seed": 123,
                        "submission_key": wave["submission_key"],
                        "source_sha": state["source_sha"],
                        "goal_file": state["goal_path"],
                        "recipe_file": state["recipe_path"],
                        "recipe_overrides": study.expected_recipe_overrides(
                            state, wave
                        ),
                        "run_description": study.run_description(state, wave, 123),
                        "checkpoint_eval_backend": "none",
                        "image_digest": runtime["image_ref"],
                        "runtime_input_sha256": runtime["input_sha256"],
                        "runtime_build_source_sha": runtime["build_source_sha"],
                    }
                ]
            }
            with (
                mock.patch.object(study, "git_head", return_value=state["source_sha"]),
                redirect_stdout(io.StringIO()),
            ):
                study.command_record_launch(
                    SimpleNamespace(
                        study=str(state_path),
                        submission_key=wave["submission_key"],
                        payload_json=json.dumps(launch),
                        payload_file=None,
                    )
                )
            event = {
                "run_id": run_id,
                "scientific_success": False,
                "dstack": {"terminal": True},
                "attempt_terminal": {
                    "state": "succeeded",
                    "acceptance_required": False,
                    "wandb_high_water_mark": 12,
                    "drain": {
                        "complete": True,
                        "wandb_remote_high_water_mark": 12,
                    },
                },
                "semantic": {
                    "terminal": None,
                    "manifest": {
                        "seed": 123,
                        "compute": {"submission_key": wave["submission_key"]},
                        "wandb": {
                            "run_id": run_id,
                            "url": f"https://wandb.ai/e/p/runs/{run_id}",
                        },
                    },
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
            self.assertEqual(
                study.next_action(study.load_state(state_path)),
                {"action": "collect_training_evidence", "run_id": run_id},
            )
            evidence = training_evidence(
                all_starts=True,
                strong=True,
                step=18_000,
                peak=0.94,
                run_id=run_id,
            )
            with (
                mock.patch.object(study, "git_head", return_value=state["source_sha"]),
                mock.patch.object(study, "fetch_training_evidence", return_value=evidence),
                redirect_stdout(io.StringIO()),
            ):
                study.command_collect_training_evidence(
                    SimpleNamespace(study=str(state_path), run_id=run_id)
                )
            output = io.StringIO()
            with (
                mock.patch.object(study, "git_head", return_value=state["source_sha"]),
                mock.patch.object(study, "fetch_training_evidence") as fetch,
                redirect_stdout(output),
            ):
                study.command_collect_training_evidence(
                    SimpleNamespace(study=str(state_path), run_id=run_id)
                )
            persisted = study.load_state(state_path)

        self.assertTrue(json.loads(output.getvalue())["duplicate"])
        fetch.assert_not_called()
        self.assertEqual(
            persisted["candidates"][persisted["baseline_candidate_id"]]["screen_runs"][0][
                "training_evidence"
            ]["peak_window_100_rate_min"],
            0.94,
        )

    def test_passed_screen_is_promoted_while_other_screens_are_still_running(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = base_state(Path(temporary))
            add_closed_baseline(state)
            first_delta = {"learning_rate": 0.0005}
            second_delta = {"learning_rate": 0.0015}
            first_id = study.candidate_id(first_delta)
            second_id = study.candidate_id(second_delta)
            state["candidates"][first_id] = {
                "id": first_id,
                "delta": first_delta,
                "created_round": 1,
                "screen_runs": [],
                "pair_runs": [],
                "confirmation_runs": [],
            }
            state["candidates"][second_id] = {
                "id": second_id,
                "delta": second_delta,
                "created_round": 1,
                "screen_runs": [],
                "pair_runs": [],
                "confirmation_runs": [],
            }
            passed = run_record(20, 123, all_starts=True)
            state["waves"].extend(
                [
                    {
                        "phase": "search-screen",
                        "round": 1,
                        "candidate_id": first_id,
                        "seeds": [123],
                        "timesteps": 20_480,
                        "submission_key": "first-screen",
                        "status": "evidence_complete",
                        "run_ids": [passed["run_id"]],
                        "terminal_runs": [passed],
                        "closed": False,
                    },
                    {
                        "phase": "search-screen",
                        "round": 1,
                        "candidate_id": second_id,
                        "seeds": [123],
                        "timesteps": 20_480,
                        "submission_key": "second-screen",
                        "status": "launched",
                        "run_ids": [immutable_run_id(21)],
                        "terminal_runs": [],
                        "closed": False,
                    },
                ]
            )

            action = study.next_action(state)

        self.assertEqual(
            action,
            {"action": "reserve_search_pair", "candidate_id": first_id},
        )

    def test_authoritative_evidence_uses_all_start_counts_and_canonical_peak(
        self,
    ) -> None:
        run = SimpleNamespace(
            id="rlab-42",
            url="https://wandb.ai/e/p/runs/rlab-42",
            state="finished",
            scan_history=lambda **_kwargs: [
                {
                "global_step": 50_176,
                "train/outcome/success/from/A/count": 2,
                "train/outcome/success/from/B/count": 0,
                "train/outcome/success/window_100/rate/min": 0.91,
                }
            ],
        )
        api = SimpleNamespace(run=lambda _path: run)
        fake_wandb = SimpleNamespace(Api=lambda: api)
        with (
            mock.patch.dict("sys.modules", {"wandb": fake_wandb}),
            mock.patch("rlab.wandb_utils.load_wandb_env"),
        ):
            evidence = study.fetch_training_evidence(
                url="https://wandb.ai/e/p/runs/rlab-42",
                expected_run_id="rlab-42",
                starts=["A", "B"],
                strong_threshold=0.9,
            )

        self.assertFalse(evidence["all_starts_succeeded"])
        self.assertEqual(evidence["peak_window_100_rate_min"], 0.91)
        self.assertEqual(evidence["first_strong_step"], 50_176)
        self.assertTrue(evidence["strong"])
        self.assertEqual(evidence["authority"], "wandb_history")

    def test_eval_backed_or_unverified_terminal_pauses(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state_path = root / "runs/autoresearch/test/study.json"
            state = base_state(root)
            run_id = immutable_run_id(9)
            wave = {
                "phase": "baseline-screen",
                "round": 0,
                "candidate_id": state["baseline_candidate_id"],
                "seeds": [123],
                "timesteps": 20_480,
                "submission_key": "bad-terminal",
                "status": "launched",
                "run_ids": [run_id],
                "terminal_runs": [],
                "closed": False,
            }
            state["waves"].append(wave)
            study.atomic_json(state_path, state)
            event = {
                "run_id": run_id,
                "scientific_success": True,
                "dstack": {"terminal": True},
                "attempt_terminal": {
                    "state": "succeeded",
                    "acceptance_required": True,
                    "wandb_high_water_mark": 12,
                    "drain": {
                        "complete": True,
                        "wandb_remote_high_water_mark": 12,
                    },
                },
                "semantic": {
                    "terminal": {"state": "succeeded"},
                    "manifest": {
                        "seed": 123,
                        "compute": {"submission_key": "bad-terminal"},
                        "wandb": {
                            "run_id": run_id,
                            "url": f"https://wandb.ai/e/p/runs/{run_id}",
                        },
                    },
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
            updated = study.load_state(state_path)

        self.assertEqual(updated["status"], "paused")
        self.assertEqual(
            updated["pause_reason"]["event"],
            "terminal_without_valid_training_evidence_source",
        )

    def test_confirmation_requires_four_of_five_and_redacts_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state_path = root / "runs/autoresearch/pass/study.json"
            state = base_state(root)
            add_closed_baseline(state)
            identifier = add_ranked_candidate(state, {"learning_rate": 0.0005})
            state["stale_rounds"] = 3
            study.atomic_json(state_path, state)
            with (
                mock.patch.object(study, "git_head", return_value=state["source_sha"]),
                redirect_stdout(io.StringIO()),
            ):
                study.command_reserve(reserve_args(state_path, "confirmation"))
            confirmed = study.load_state(state_path)
            wave = confirmed["waves"][-1]
            wave["terminal_runs"] = [
                run_record(100 + index, seed, strong=index < 4, step=60_000 if index < 4 else None, peak=0.95 if index < 4 else 0.6)
                for index, seed in enumerate(confirmed["confirmation_seeds"])
            ]
            wave["status"] = "evidence_complete"
            confirmed["candidates"][identifier]["confirmation_runs"] = list(wave["terminal_runs"])
            study.atomic_json(state_path, confirmed)
            with redirect_stdout(io.StringIO()):
                study.command_close_confirmation(SimpleNamespace(study=str(state_path)))
            passed = study.load_state(state_path)

        self.assertTrue(passed["winner"]["training_signal_confirmed"])
        self.assertEqual(passed["winner"]["strong_seed_count"], 4)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state_path = root / "runs/autoresearch/fail/study.json"
            state = base_state(root)
            add_closed_baseline(state)
            identifier = add_ranked_candidate(state, {"learning_rate": 0.0005})
            state["stale_rounds"] = 3
            study.atomic_json(state_path, state)
            with (
                mock.patch.object(study, "git_head", return_value=state["source_sha"]),
                redirect_stdout(io.StringIO()),
            ):
                study.command_reserve(reserve_args(state_path, "confirmation"))
            failed = study.load_state(state_path)
            wave = failed["waves"][-1]
            wave["terminal_runs"] = [
                run_record(200 + index, seed, strong=index < 3, step=70_000 if index < 3 else None, peak=0.92 if index < 3 else 0.4)
                for index, seed in enumerate(failed["confirmation_seeds"])
            ]
            wave["status"] = "evidence_complete"
            failed["candidates"][identifier]["confirmation_runs"] = list(wave["terminal_runs"])
            study.atomic_json(state_path, failed)
            with redirect_stdout(io.StringIO()):
                study.command_close_confirmation(SimpleNamespace(study=str(state_path)))
            rejected = study.load_state(state_path)

        self.assertIsNone(rejected["winner"])
        self.assertIn(identifier, rejected["excluded_candidates"])
        self.assertEqual(
            rejected["candidates"][identifier]["confirmation_runs"],
            [{"redacted": True, "strong_seed_count": 3, "total": 5}],
        )

    def test_prepare_and_complete_apply_preserve_goal_eval_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state_path = root / "runs/autoresearch/test/study.json"
            state = base_state(root)
            delta = {"learning_rate": 0.0005}
            identifier = add_ranked_candidate(state, delta)
            state["winner"] = {
                "candidate_id": identifier,
                "delta": delta,
                "training_signal_confirmed": True,
                "strong_seed_count": 5,
                "total_seeds": 5,
                "strong_threshold": 0.9,
                "runs": [],
            }
            study.atomic_json(state_path, state)
            postimage = root / "planned.yaml"
            postimage.write_text(
                "recipe_id: candidate\ntrain:\n  backend:\n    config:\n      learning_rate: 0.0005\n",
                encoding="utf-8",
            )
            with (
                mock.patch.object(study, "git_head", return_value=state["source_sha"]),
                redirect_stdout(io.StringIO()),
            ):
                study.command_prepare_apply(
                    SimpleNamespace(study=str(state_path), postimage_file=str(postimage))
                )
            (root / "recipe.yaml").write_text(postimage.read_text(encoding="utf-8"), encoding="utf-8")
            expected = json.loads(json.dumps(state["baseline"]["train_config"]))
            expected["training_backend"]["config"].update(delta)
            with (
                mock.patch.object(study, "git_head", return_value=state["source_sha"]),
                mock.patch.object(
                    study,
                    "compose_train_document",
                    return_value={"train_config": expected},
                ),
                redirect_stdout(io.StringIO()),
            ):
                study.command_complete_apply(SimpleNamespace(study=str(state_path)))
            completed = study.load_state(state_path)
            report = json.loads((state_path.parent / "report.json").read_text(encoding="utf-8"))

        self.assertEqual(completed["status"], "done")
        self.assertEqual(
            completed["baseline"]["train_config"]["checkpoint_eval_backend"], "modal"
        )
        self.assertFalse(report["checkpoint_promoted"])
        self.assertFalse(report["goal_accepted"])

    def test_confirmed_baseline_writes_a_noop_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state_path = root / "runs/autoresearch/test/study.json"
            state = base_state(root)
            state["winner"] = {
                "candidate_id": state["baseline_candidate_id"],
                "delta": {},
                "training_signal_confirmed": True,
                "strong_seed_count": 5,
                "total_seeds": 5,
                "strong_threshold": 0.9,
                "runs": [],
            }
            study.atomic_json(state_path, state)
            postimage = root / "planned.yaml"
            postimage.write_text(state["recipe_preimage"], encoding="utf-8")
            with (
                mock.patch.object(study, "git_head", return_value=state["source_sha"]),
                redirect_stdout(io.StringIO()),
            ):
                study.command_prepare_apply(
                    SimpleNamespace(study=str(state_path), postimage_file=str(postimage))
                )
            completed = study.load_state(state_path)
            report = json.loads((state_path.parent / "report.json").read_text(encoding="utf-8"))

        self.assertEqual(completed["status"], "done")
        self.assertEqual(completed["apply"]["kind"], "baseline_noop")
        self.assertTrue(report["winner"]["training_signal_confirmed"])
        self.assertFalse(report["checkpoint_promoted"])


if __name__ == "__main__":
    unittest.main()
