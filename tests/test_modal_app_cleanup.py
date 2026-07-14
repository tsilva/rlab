from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
import unittest
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest import mock

from rlab.modal_app_cleanup import (
    DefaultModalAppClient,
    ModalAppInfo,
    protected_modal_app_names,
    run_modal_app_cleanup,
)
from rlab.modal_eval_config import load_modal_eval_config


def app(
    index: int,
    name: str,
    *,
    created_at: datetime,
    state: str = "deployed",
    tasks: int = 0,
) -> ModalAppInfo:
    return ModalAppInfo(
        app_id="ap-" + f"{index:022d}",
        name=name,
        state=state,
        running_tasks=tasks,
        created_at=created_at,
    )


class FakeModalAppClient:
    def __init__(self, apps: list[ModalAppInfo], *, failures: set[str] | None = None):
        self.apps = tuple(apps)
        self.failures = failures or set()
        self.list_calls = 0
        self.stop_calls: list[str] = []

    def list_apps(self, *, deadline_monotonic: float):
        self.list_calls += 1
        return self.apps

    def stop_app(self, app_id: str, *, deadline_monotonic: float) -> None:
        self.stop_calls.append(app_id)
        if app_id in self.failures:
            raise RuntimeError("stop failed")


class ModalAppCleanupTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = replace(load_modal_eval_config(), cleanup_enabled=True)
        self.now = datetime.now(UTC)
        self.deadline = time.monotonic() + 60

    def test_stops_only_old_unprotected_zero_task_owned_apps_with_cap(self) -> None:
        protected_name = "rlab-eval-aaaaaaaaaaaa"
        inventory = [
            app(1, "sigmaevolve-runner", created_at=self.now - timedelta(days=30)),
            app(2, "rlab-eval-not-a-digest", created_at=self.now - timedelta(days=30)),
            app(
                3,
                "rlab-eval-bbbbbbbbbbbb",
                created_at=self.now - timedelta(days=30),
                state="stopped",
            ),
            app(
                4,
                "rlab-eval-cccccccccccc",
                created_at=self.now - timedelta(days=30),
                tasks=1,
            ),
            app(5, "rlab-eval-dddddddddddd", created_at=self.now - timedelta(hours=23)),
            app(6, protected_name, created_at=self.now - timedelta(days=30)),
        ]
        stale = [
            app(
                100 + index,
                f"rlab-eval-{index:012x}",
                created_at=self.now - timedelta(hours=48 + index),
            )
            for index in range(12)
        ]
        client = FakeModalAppClient([*inventory, *stale])
        with (
            tempfile.TemporaryDirectory() as temporary,
            mock.patch(
                "rlab.modal_app_cleanup.protected_modal_app_names",
                return_value={protected_name},
            ),
        ):
            result = run_modal_app_cleanup(
                mock.MagicMock(),
                self.config,
                repo_root=Path(temporary),
                deadline_monotonic=self.deadline,
                client=client,
                now=self.now,
            )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["owned_deployed"], 15)
        self.assertEqual(result["protected"], 1)
        self.assertEqual(result["candidates"], 12)
        self.assertEqual(result["stopped"], 10)
        expected = [item.app_id for item in reversed(stale[2:])]
        self.assertEqual(client.stop_calls, expected)
        self.assertNotIn(protected_name, result["stopped_apps"])

    def test_global_protection_failure_stops_nothing(self) -> None:
        client = FakeModalAppClient(
            [app(1, "rlab-eval-aaaaaaaaaaaa", created_at=self.now - timedelta(days=2))]
        )
        with (
            tempfile.TemporaryDirectory() as temporary,
            mock.patch(
                "rlab.modal_app_cleanup.protected_modal_app_names",
                side_effect=RuntimeError("latest lookup failed"),
            ),
        ):
            result = run_modal_app_cleanup(
                mock.MagicMock(),
                self.config,
                repo_root=Path(temporary),
                deadline_monotonic=self.deadline,
                client=client,
                now=self.now,
            )

        self.assertEqual(result["status"], "error")
        self.assertEqual(client.list_calls, 0)
        self.assertEqual(client.stop_calls, [])

    def test_invalid_owned_app_identity_fails_closed(self) -> None:
        invalid = ModalAppInfo(
            app_id="not-an-app-id",
            name="rlab-eval-aaaaaaaaaaaa",
            state="deployed",
            running_tasks=0,
            created_at=self.now - timedelta(days=2),
        )
        client = FakeModalAppClient([invalid])
        with (
            tempfile.TemporaryDirectory() as temporary,
            mock.patch(
                "rlab.modal_app_cleanup.protected_modal_app_names",
                return_value=set(),
            ),
        ):
            result = run_modal_app_cleanup(
                mock.MagicMock(),
                self.config,
                repo_root=Path(temporary),
                deadline_monotonic=self.deadline,
                client=client,
                now=self.now,
            )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["errors"], ["invalid owned Modal app id"])
        self.assertEqual(client.stop_calls, [])

    def test_individual_stop_failure_is_isolated_and_reported(self) -> None:
        first = app(1, "rlab-eval-aaaaaaaaaaaa", created_at=self.now - timedelta(days=3))
        second = app(2, "rlab-eval-bbbbbbbbbbbb", created_at=self.now - timedelta(days=2))
        client = FakeModalAppClient([first, second], failures={first.app_id})
        with (
            tempfile.TemporaryDirectory() as temporary,
            mock.patch(
                "rlab.modal_app_cleanup.protected_modal_app_names",
                return_value=set(),
            ),
        ):
            result = run_modal_app_cleanup(
                mock.MagicMock(),
                self.config,
                repo_root=Path(temporary),
                deadline_monotonic=self.deadline,
                client=client,
                now=self.now,
            )

        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["stopped_apps"], [second.name])
        self.assertEqual(client.stop_calls, [first.app_id, second.app_id])

    def test_inventory_runs_at_most_hourly_and_skips_short_deadline(self) -> None:
        client = FakeModalAppClient([])
        with (
            tempfile.TemporaryDirectory() as temporary,
            mock.patch(
                "rlab.modal_app_cleanup.protected_modal_app_names",
                return_value=set(),
            ),
        ):
            root = Path(temporary)
            first = run_modal_app_cleanup(
                mock.MagicMock(),
                self.config,
                repo_root=root,
                deadline_monotonic=self.deadline,
                client=client,
                now=self.now,
            )
            second = run_modal_app_cleanup(
                mock.MagicMock(),
                self.config,
                repo_root=root,
                deadline_monotonic=self.deadline,
                client=client,
                now=self.now,
            )
            deadline = run_modal_app_cleanup(
                mock.MagicMock(),
                self.config,
                repo_root=root / "other",
                deadline_monotonic=time.monotonic() + 1,
                client=client,
                now=self.now,
            )

        self.assertEqual(first["status"], "ok")
        self.assertEqual(second["status"], "not_due")
        self.assertEqual(deadline["status"], "skipped_deadline")
        self.assertEqual(client.list_calls, 1)

    def test_protected_set_covers_all_nonterminal_work_classes(self) -> None:
        cursor = mock.MagicMock()
        cursor.fetchall.side_effect = [
            [{"runtime_image_ref": "docker:repo/image@sha256:" + "a" * 64}],
            [{"modal_app_name": "rlab-eval-bbbbbbbbbbbb"}],
        ]
        conn = mock.MagicMock()
        conn.cursor.return_value.__enter__.return_value = cursor

        names = protected_modal_app_names(
            conn,
            self.config,
            latest_runtime_ref=lambda: "docker:repo/image@sha256:" + "c" * 64,
        )

        self.assertEqual(
            names,
            {
                "rlab-eval-aaaaaaaaaaaa",
                "rlab-eval-bbbbbbbbbbbb",
                "rlab-eval-cccccccccccc",
            },
        )
        query = " ".join(str(call.args[0]) for call in cursor.execute.call_args_list)
        for state in (
            "pending",
            "launching",
            "starting",
            "running",
            "active",
            "awaiting_artifact_recovery",
            "finalizing",
            "dispatching",
            "submitted",
            "blocked_budget",
        ):
            self.assertIn(state, query)

    def test_default_client_uses_locked_modal_cli_without_shell(self) -> None:
        payload = [
            {
                "app_id": "ap-" + "a" * 22,
                "description": "rlab-eval-aaaaaaaaaaaa",
                "state": "deployed",
                "tasks": "0",
                "created_at": self.now.isoformat(),
                "stopped_at": None,
            }
        ]
        completed = subprocess.CompletedProcess([], 0, stdout=json.dumps(payload), stderr="")
        with mock.patch("rlab.modal_app_cleanup.subprocess.run", return_value=completed) as run:
            client = DefaultModalAppClient()
            apps = client.list_apps(deadline_monotonic=self.deadline)
            client.stop_app(apps[0].app_id, deadline_monotonic=self.deadline)

        self.assertEqual(apps[0].name, "rlab-eval-aaaaaaaaaaaa")
        self.assertEqual(
            run.call_args_list[0].args[0],
            [sys.executable, "-m", "modal", "app", "list", "--json"],
        )
        self.assertEqual(
            run.call_args_list[1].args[0],
            [sys.executable, "-m", "modal", "app", "stop", apps[0].app_id, "--yes"],
        )
        self.assertFalse(run.call_args_list[0].kwargs.get("shell", False))

    def test_default_client_rejects_malformed_inventory(self) -> None:
        completed = subprocess.CompletedProcess([], 0, stdout="{}", stderr="")
        with mock.patch("rlab.modal_app_cleanup.subprocess.run", return_value=completed):
            with self.assertRaisesRegex(ValueError, "inventory must be a list"):
                DefaultModalAppClient().list_apps(deadline_monotonic=self.deadline)


if __name__ == "__main__":
    unittest.main()
