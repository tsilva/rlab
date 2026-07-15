from __future__ import annotations

import asyncio
import contextlib
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rlab import fleet_service, fleet_watch


def completed(argv, returncode: int = 0, stdout: str = ""):
    return subprocess.CompletedProcess(argv, returncode, stdout, "")


def sample_snapshot() -> dict:
    return {
        "schema_version": 1,
        "captured_at": "2026-07-15T12:00:00Z",
        "data_freshness": {"source_at": "2026-07-15T12:00:00Z", "errors": [], "stale": False},
        "service": {
            "state": "HEALTHY",
            "installed": True,
            "loaded": True,
            "interval_seconds": 30,
            "last_pass_status": "ok",
            "consecutive_failures": 0,
        },
        "scheduler": {
            "state": "RECONCILING",
            "pass_id": "abc123",
            "started_at": "2026-07-15T11:59:55Z",
            "source": {"sha": "deadbeef", "dirty": True},
        },
        "work": {"in_progress": 2, "waiting": 1, "retrying": 1, "needs_action": 1},
        "needs_action": [
            {
                "id": "artifact:train/15",
                "entity": "train/15",
                "title": "Artifact recovery required",
                "detail": "Evaluation cannot continue",
                "resolution": "manual action",
                "blast_radius": "promotion only",
                "command": "rlab eval modal recover 15",
            }
        ],
        "now": [
            {
                "id": "beast-3",
                "entity": "beast-3",
                "title": "STARTING TRAIN",
                "detail": "Capacity became available",
                "started_at": "2026-07-15T11:59:58Z",
                "resolution": "automatic",
            }
        ],
        "retrying": [
            {
                "id": "publish:31",
                "entity": "train/31",
                "title": "Retrying publication",
                "detail": "Retry scheduled",
                "resolution": "automatic",
            }
        ],
        "waiting": [
            {
                "id": "capacity:32",
                "entity": "train/32",
                "title": "Waiting for beast-3 capacity",
                "detail": "6/6 train slots reserved",
                "resolution": "automatic",
            }
        ],
        "recent_changes": [
            {
                "entity": "train/31",
                "title": "Claimed by beast-3",
                "timestamp": "2026-07-15T11:59:50Z",
            }
        ],
    }


class FleetWatchTests(unittest.TestCase):
    def make_paths(self, root: Path) -> fleet_service.ServicePaths:
        repo = root / "repo"
        python = repo / ".venv" / "bin" / "python"
        python.parent.mkdir(parents=True)
        python.write_text("#!/bin/sh\n", encoding="utf-8")
        python.chmod(0o755)
        paths = fleet_service.ServicePaths(
            repo_root=repo.resolve(),
            python=python.resolve(),
            state_dir=(root / "state").resolve(),
            launch_agents_dir=(root / "LaunchAgents").resolve(),
        )
        paths.state_dir.mkdir(parents=True)
        paths.plist.parent.mkdir(parents=True)
        paths.plist.write_text("installed", encoding="utf-8")
        return paths

    def test_collect_snapshot_separates_service_and_workload_health(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.make_paths(Path(temporary))
            paths.last_pass.write_text(
                json.dumps(
                    {
                        "pass_id": "pass1",
                        "status": "ok",
                        "finished_at": fleet_service._iso_utc(),
                        "workload": {
                            "in_progress": 0,
                            "needs_action": sample_snapshot()["needs_action"],
                            "retrying": [],
                            "waiting": [],
                        },
                    }
                ),
                encoding="utf-8",
            )
            snapshot = fleet_watch.collect_watch_snapshot(
                paths,
                runner=lambda argv, **_kwargs: completed(argv),
            )

        self.assertEqual(snapshot["service"]["state"], "HEALTHY")
        self.assertEqual(snapshot["scheduler"]["state"], "IDLE")
        self.assertEqual(snapshot["work"]["needs_action"], 1)
        self.assertEqual(snapshot["needs_action"][0]["entity"], "train/15")

    def test_collect_snapshot_never_opens_queue_or_mutation_helpers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.make_paths(Path(temporary))
            paths.last_pass.write_text(
                json.dumps({"status": "ok", "finished_at": fleet_service._iso_utc()}),
                encoding="utf-8",
            )
            with (
                mock.patch.object(
                    fleet_service,
                    "_connect_queue",
                    side_effect=AssertionError("watcher opened queue"),
                ),
                mock.patch.object(
                    fleet_service,
                    "kick_service",
                    side_effect=AssertionError("watcher mutated service"),
                ),
            ):
                snapshot = fleet_watch.collect_watch_snapshot(
                    paths,
                    runner=lambda argv, **_kwargs: completed(argv),
                )
        self.assertEqual(snapshot["scheduler"]["state"], "IDLE")

    def test_current_pass_classifies_stuck_and_interrupted_from_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.make_paths(Path(temporary))
            paths.current_pass.write_text(
                json.dumps(
                    {
                        "pass_id": "pass2",
                        "started_at": "2026-07-15T11:00:00Z",
                        "updated_at": "2026-07-15T11:00:01Z",
                        "deadline_at": "2026-07-15T11:05:00Z",
                        "lanes": {},
                    }
                ),
                encoding="utf-8",
            )
            now = fleet_watch._parse_time("2026-07-15T11:06:00Z")
            assert now is not None
            stuck = fleet_watch.collect_watch_snapshot(
                paths,
                runner=lambda argv, **_kwargs: completed(argv, stdout="state = running"),
                now=now,
            )
            interrupted = fleet_watch.collect_watch_snapshot(
                paths,
                runner=lambda argv, **_kwargs: completed(argv, returncode=113),
                now=now,
            )

        self.assertEqual(stuck["scheduler"]["state"], "STUCK")
        self.assertEqual(interrupted["scheduler"]["state"], "INTERRUPTED")

    def test_plain_snapshot_is_readable_and_contains_no_terminal_control(self) -> None:
        rendered = fleet_watch.render_plain_snapshot(sample_snapshot())
        self.assertIn("SERVICE HEALTHY", rendered)
        self.assertIn("Artifact recovery required", rendered)
        self.assertNotIn("\x1b", rendered)

    def test_plain_stream_suppresses_noop_polls(self) -> None:
        output = io.StringIO()
        ticks = iter((0.0, 1.0, 2.0))
        with contextlib.redirect_stdout(output):
            result = fleet_watch.stream_plain_watch(
                sample_snapshot,
                max_iterations=3,
                clock=lambda: next(ticks),
                sleep=lambda _seconds: None,
            )
        self.assertEqual(result, 0)
        self.assertEqual(output.getvalue().count("FLEET SCHEDULER"), 1)

    def test_corrupt_state_is_reported_and_once_returns_one(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.make_paths(Path(temporary))
            paths.current_pass.write_text("{broken", encoding="utf-8")
            snapshot = fleet_watch.collect_watch_snapshot(
                paths,
                runner=lambda argv, **_kwargs: completed(argv),
            )
        self.assertTrue(snapshot["data_freshness"]["errors"])

    def test_textual_app_navigation_freeze_details_and_responsive_sizes(self) -> None:
        async def exercise(size: tuple[int, int]) -> None:
            app = fleet_watch.FleetWatchApp(sample_snapshot, initial=sample_snapshot())
            async with app.run_test(size=size) as pilot:
                await pilot.pause()
                self.assertIsNotNone(app.query_one("#summary").render())
                await pilot.press("space")
                self.assertTrue(app.frozen)
                self.assertTrue(app.query_one("#frozen").has_class("visible"))
                await pilot.press("down", "enter")
                self.assertEqual(len(app.screen_stack), 2)
                await pilot.press("escape")
                self.assertEqual(len(app.screen_stack), 1)
                await pilot.press("q")
                await pilot.pause()
                self.assertEqual(app.return_value, 0)

        for size in ((160, 45), (120, 35), (100, 30), (80, 24), (60, 20), (50, 16)):
            with self.subTest(size=size):
                asyncio.run(exercise(size))

    def test_parser_output_modes_are_mutually_exclusive(self) -> None:
        parser = fleet_service.build_parser()
        args = parser.parse_args(["service", "watch", "--json"])
        self.assertIs(args.func, fleet_service.cmd_watch)
        with self.assertRaises(SystemExit):
            parser.parse_args(["service", "watch", "--json", "--once"])


if __name__ == "__main__":
    unittest.main()
