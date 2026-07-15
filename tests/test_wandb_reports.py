from __future__ import annotations

import copy
import dataclasses
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rlab.config_validation import load_goal_contract
from rlab.metric_names import validate_metric_name
from rlab.recipe_documents import compose_train_document, goal_contract_sha256
from rlab.wandb_reports import (
    LEGACY_PORTFOLIO_TITLE,
    GoalReportSpec,
    PortfolioReportSpec,
    _preflight_existing,
    _replace_and_save,
    _structure_sha256,
    build_wandb_report,
    compile_report_specs,
    extract_report_identity,
    sync_reports,
    verify_reports,
)


ROOT = Path(__file__).resolve().parents[1]
FAMILY_ROOT = ROOT / "experiments" / "goals" / "SuperMarioBros-Nes-v0"
LEVEL1_1_GOAL = FAMILY_ROOT / "Level1-1" / "_goal.yaml"
MIXED_GOAL = FAMILY_ROOT / "Levels_1-1_1-2" / "_goal.yaml"
MARIO_RECIPE = ROOT / "experiments" / "recipes" / "mario" / "single" / "ppo.yaml"


def goal_spec(goal_id: str) -> GoalReportSpec:
    return next(
        spec
        for spec in compile_report_specs(ROOT)
        if isinstance(spec, GoalReportSpec) and spec.goal_id == goal_id
    )


class FakeApi:
    def __init__(self, reports):
        self._reports = reports

    def reports(self, _path, per_page=100):
        del per_page
        return list(self._reports)


class FakeSavedReport:
    def __init__(self, *, identity: str, url: str, display_name: str = "generated"):
        self.description = f"<!-- rlab-report-id:{identity} -->"
        self.url = url
        self.display_name = display_name


class WandbReportCompilationTests(unittest.TestCase):
    def test_compile_is_deterministic_and_portfolio_is_last(self) -> None:
        first = [spec.to_json() for spec in compile_report_specs(ROOT)]
        second = [spec.to_json() for spec in compile_report_specs(ROOT)]

        self.assertEqual(first, second)
        self.assertEqual(first[-1]["kind"], "portfolio")
        self.assertTrue(first[-1]["goals"])

    def test_mixed_goal_exposes_balanced_metrics_and_both_starts(self) -> None:
        spec = goal_spec("Levels_1-1_1-2")
        report = build_wandb_report(spec, entity="entity", source_sha="a" * 40)
        payload = dataclasses.asdict(report)

        self.assertEqual(spec.starts, ("Level1-1", "Level1-2"))
        serialized = str(payload)
        self.assertIn("max(eval/full/outcome/success/rate/min)", serialized)
        self.assertLess(
            serialized.index("eval/full/outcome/success/rate/min"),
            serialized.index("train/outcome/success/window_100/rate/min"),
        )
        self.assertIn("eval/full/outcome/success/from/Level1-1/rate", serialized)
        self.assertIn("eval/full/outcome/success/from/Level1-2/rate", serialized)

    def test_compiled_panel_metrics_are_registered(self) -> None:
        report = build_wandb_report(
            goal_spec("Level1-1"), entity="entity", source_sha="a" * 40
        )
        for block in report.blocks:
            for panel in getattr(block, "panels", ()):
                for metric in getattr(panel, "y", ()):
                    validate_metric_name(metric)
                x = getattr(panel, "x", None)
                if x:
                    validate_metric_name(x)
                table_name = getattr(panel, "table_name", None)
                if table_name:
                    validate_metric_name(table_name)

    def test_goal_fingerprint_is_semantic_and_reaches_train_config(self) -> None:
        document = load_goal_contract(LEVEL1_1_GOAL, ROOT)
        reordered = dict(reversed(list(document.items())))
        changed = copy.deepcopy(document)
        changed["eval"]["episodes"] += 1

        self.assertEqual(goal_contract_sha256(document), goal_contract_sha256(reordered))
        self.assertNotEqual(goal_contract_sha256(document), goal_contract_sha256(changed))
        train_document = compose_train_document(LEVEL1_1_GOAL, MARIO_RECIPE)
        self.assertEqual(
            train_document["train_config"]["goal_contract_sha256"],
            goal_contract_sha256(document),
        )

    def test_goal_override_replaces_sections_and_rejects_unknown_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = root / "experiments" / "goals" / "SuperMarioBros-Nes-v0"
            shutil.copytree(FAMILY_ROOT, target)
            override = target / "Level1-1" / "_report.yaml"
            override.write_text(
                "title: 'Custom {goal_id}'\nsections:\n- objective_summary\n",
                encoding="utf-8",
            )
            compiled = compile_report_specs(root, goal="Level1-1")
            spec = next(item for item in compiled if isinstance(item, GoalReportSpec))
            self.assertEqual(spec.title, "Custom Level1-1")
            self.assertEqual(spec.sections, ("objective_summary",))

            override.write_text("unknown: true\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown field"):
                compile_report_specs(root, goal="Level1-1")

    def test_orphan_override_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = root / "experiments" / "goals" / "SuperMarioBros-Nes-v0"
            shutil.copytree(FAMILY_ROOT, target)
            orphan = target / "orphan" / "_report.yaml"
            orphan.parent.mkdir()
            orphan.write_text("enabled: true\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "no sibling _goal.yaml"):
                compile_report_specs(root)

    def test_unknown_semantic_section_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = root / "experiments" / "goals" / "SuperMarioBros-Nes-v0"
            shutil.copytree(FAMILY_ROOT, target)
            override = target / "Level1-1" / "_report.yaml"
            override.write_text("sections:\n- raw_panel_dsl\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "unknown section id"):
                compile_report_specs(root, goal="Level1-1")


class WandbReportSyncTests(unittest.TestCase):
    def test_identity_marker_round_trips(self) -> None:
        identity = "SuperMarioBros-Nes-v0/goal/Level1-1"
        report = build_wandb_report(
            goal_spec("Level1-1"), entity="entity", source_sha="a" * 40
        )
        self.assertEqual(extract_report_identity(report.description), identity)

    def test_structure_fingerprint_ignores_generated_ids_but_detects_content(self) -> None:
        first = build_wandb_report(
            goal_spec("Level1-1"), entity="entity", source_sha="a" * 40
        )
        second = build_wandb_report(
            goal_spec("Level1-1"), entity="entity", source_sha="a" * 40
        )

        self.assertEqual(_structure_sha256(first), _structure_sha256(second))
        second.title = "manually edited"
        self.assertNotEqual(_structure_sha256(first), _structure_sha256(second))

    def test_preflight_rejects_duplicate_identities_and_adopts_legacy_portfolio(self) -> None:
        specs = compile_report_specs(ROOT, goal="Level1-1")
        identity = specs[0].identity
        duplicate = [
            FakeSavedReport(identity=identity, url="https://one"),
            FakeSavedReport(identity=identity, url="https://two"),
        ]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            _preflight_existing(specs, duplicate)

        legacy = SimpleNamespace(
            description="",
            display_name=LEGACY_PORTFOLIO_TITLE,
            url="https://legacy",
        )
        existing = _preflight_existing(specs, [legacy])
        portfolio = next(item for item in specs if isinstance(item, PortfolioReportSpec))
        self.assertIs(existing[portfolio.identity], legacy)

    def test_existing_report_is_replaced_from_source(self) -> None:
        desired = build_wandb_report(
            goal_spec("Level1-1"), entity="entity", source_sha="a" * 40
        )
        current = SimpleNamespace(
            entity="old",
            project="old",
            title="manual title",
            description="manual description",
            width="readable",
            blocks=["manual block"],
            url="https://updated",
            save=lambda: None,
        )
        existing = SimpleNamespace(url="https://existing")
        with patch(
            "wandb_workspaces.reports.v2.Report.from_url", return_value=current
        ) as from_url:
            saved = _replace_and_save(desired, existing)

        from_url.assert_called_once_with("https://existing")
        self.assertIs(saved, current)
        self.assertEqual(current.title, desired.title)
        self.assertEqual(current.description, desired.description)
        self.assertEqual(current.blocks, desired.blocks)

    def test_sync_creates_goals_before_portfolio_and_retry_converges(self) -> None:
        specs = compile_report_specs(ROOT, goal="Level1-1")
        api = FakeApi([])
        saved_order: list[str] = []
        failed_once = False

        def save(desired, existing):
            nonlocal failed_once
            identity = extract_report_identity(desired.description)
            assert identity is not None
            if identity.endswith("/portfolio") and not failed_once:
                failed_once = True
                raise ConnectionError("simulated network interruption")
            saved_order.append(identity)
            if existing is None:
                existing = FakeSavedReport(
                    identity=identity,
                    url=f"https://wandb.ai/entity/project/reports/{identity}",
                )
                api._reports.append(existing)
            return SimpleNamespace(url=existing.url)

        with (
            patch(
                "wandb_workspaces.reports.v2.interface._get_api",
                return_value=object(),
            ),
            patch(
                "wandb_workspaces.reports.v2.interface.execute_graphql",
                return_value={"project": {"internalId": "project-id"}},
            ),
            patch("rlab.wandb_reports._replace_and_save", side_effect=save),
        ):
            with self.assertRaisesRegex(ConnectionError, "simulated"):
                sync_reports(
                    specs, api=api, entity="entity", source_sha="a" * 40
                )
            result = sync_reports(
                specs, api=api, entity="entity", source_sha="a" * 40
            )

        goal = next(item for item in specs if isinstance(item, GoalReportSpec))
        portfolio = next(item for item in specs if isinstance(item, PortfolioReportSpec))
        self.assertEqual(saved_order, [goal.identity, goal.identity, portfolio.identity])
        self.assertEqual([row["identity"] for row in result], [goal.identity, portfolio.identity])
        self.assertEqual(len(api._reports), 2)

    def test_verify_detects_orphans_and_accepts_matching_structures(self) -> None:
        specs = compile_report_specs(ROOT, goal="Level1-1")
        goal = next(item for item in specs if isinstance(item, GoalReportSpec))
        portfolio = next(item for item in specs if isinstance(item, PortfolioReportSpec))
        goal_row = FakeSavedReport(identity=goal.identity, url="https://goal")
        portfolio_row = FakeSavedReport(identity=portfolio.identity, url="https://portfolio")
        orphan = FakeSavedReport(
            identity="SuperMarioBros-Nes-v0/goal/removed", url="https://orphan"
        )
        desired_goal = build_wandb_report(
            goal, entity="entity", source_sha="a" * 40
        )
        desired_portfolio = build_wandb_report(
            portfolio,
            entity="entity",
            source_sha="a" * 40,
            goal_urls={goal.identity: goal_row.url},
        )
        loaded = {goal_row.url: desired_goal, portfolio_row.url: desired_portfolio}

        result = verify_reports(
            specs,
            api=FakeApi([goal_row, portfolio_row, orphan]),
            entity="entity",
            source_sha="a" * 40,
            report_loader=loaded.__getitem__,
            include_orphans=True,
        )

        self.assertFalse(result["ok"])
        self.assertEqual(
            result["issues"],
            [{"identity": "SuperMarioBros-Nes-v0/goal/removed", "issue": "orphan"}],
        )

    def test_verify_detects_source_owned_content_drift(self) -> None:
        specs = compile_report_specs(ROOT, goal="Level1-1")
        goal = next(item for item in specs if isinstance(item, GoalReportSpec))
        portfolio = next(item for item in specs if isinstance(item, PortfolioReportSpec))
        goal_row = FakeSavedReport(identity=goal.identity, url="https://goal")
        portfolio_row = FakeSavedReport(identity=portfolio.identity, url="https://portfolio")
        drifted_goal = build_wandb_report(
            goal, entity="entity", source_sha="a" * 40
        )
        drifted_goal.title = "manual W&B edit"
        desired_portfolio = build_wandb_report(
            portfolio,
            entity="entity",
            source_sha="a" * 40,
            goal_urls={goal.identity: goal_row.url},
        )

        result = verify_reports(
            specs,
            api=FakeApi([goal_row, portfolio_row]),
            entity="entity",
            source_sha="a" * 40,
            report_loader={
                goal_row.url: drifted_goal,
                portfolio_row.url: desired_portfolio,
            }.__getitem__,
        )

        self.assertEqual(
            result["issues"],
            [{"identity": goal.identity, "issue": "content_drift"}],
        )


if __name__ == "__main__":
    unittest.main()
