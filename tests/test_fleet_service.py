from __future__ import annotations

import json
import os
import plistlib
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from rlab import fleet_service


def completed(argv, returncode: int = 0, stdout: str = "", stderr: str = ""):
    return subprocess.CompletedProcess(argv, returncode, stdout, stderr)


class FleetServiceTests(unittest.TestCase):
    def make_paths(self, root: Path) -> fleet_service.ServicePaths:
        repo = root / "repo"
        python = repo / ".venv" / "bin" / "python"
        python.parent.mkdir(parents=True)
        python.write_text("#!/bin/sh\n", encoding="utf-8")
        python.chmod(0o755)
        return fleet_service.ServicePaths(
            repo_root=repo.resolve(),
            python=python.resolve(),
            state_dir=(root / "state").resolve(),
            launch_agents_dir=(root / "LaunchAgents").resolve(),
        )

    def test_plist_is_short_lived_absolute_and_contains_no_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.make_paths(Path(temporary))
            payload = plistlib.loads(fleet_service.render_launch_agent_plist(paths))

        self.assertEqual(payload["StartInterval"], 30)
        self.assertIs(payload["KeepAlive"], False)
        self.assertNotIn("RunAtLoad", payload)
        self.assertNotIn("EnvironmentVariables", payload)
        self.assertEqual(payload["ProgramArguments"][0], str(paths.python))
        self.assertTrue(Path(payload["WorkingDirectory"]).is_absolute())
        self.assertEqual(payload["ProgramArguments"][1:4], ["-m", "rlab.fleet_service", "run-once"])
        self.assertNotIn("DATABASE_URL", json.dumps(payload))

    def test_kick_never_uses_kill_flag(self) -> None:
        commands: list[list[str]] = []

        def runner(argv, **_kwargs):
            commands.append(list(argv))
            return completed(argv)

        self.assertTrue(fleet_service.kick_service(runner=runner, uid=501))
        self.assertEqual(
            commands,
            [["launchctl", "kickstart", "gui/501/com.rlab.fleet-service"]],
        )
        self.assertNotIn("-k", commands[0])

    def test_redaction_removes_presigned_url_queries_and_url_fields(self) -> None:
        url = "https://r2.example/checkpoint?X-Amz-Credential=secret&X-Amz-Signature=value"
        self.assertEqual(
            fleet_service.redact(url),
            "https://r2.example/checkpoint?[REDACTED]",
        )
        self.assertEqual(
            fleet_service.redact({"model_get_url": url}), {"model_get_url": "[REDACTED]"}
        )

    def test_install_validates_bootstraps_and_kicks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.make_paths(Path(temporary))
            commands: list[list[str]] = []

            def runner(argv, **_kwargs):
                commands.append(list(argv))
                if argv[:2] == ["launchctl", "print"]:
                    return completed(argv, returncode=113)
                return completed(argv)

            with mock.patch.object(fleet_service.shutil, "which", return_value=None):
                result = fleet_service.install_service(paths, runner=runner)

            self.assertTrue(result.installed)
            self.assertTrue(result.kicked)
            self.assertTrue(paths.plist.is_file())
            plistlib.loads(paths.plist.read_bytes())
            self.assertIn(
                ["launchctl", "bootstrap", f"gui/{os.getuid()}", str(paths.plist)], commands
            )
            self.assertIn(
                ["launchctl", "kickstart", f"gui/{os.getuid()}/{paths.label}"],
                commands,
            )

    def test_failed_replacement_restores_previous_plist_and_registration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.make_paths(Path(temporary))
            paths.plist.parent.mkdir(parents=True)
            old_data = b"old plist\n"
            paths.plist.write_bytes(old_data)
            commands: list[list[str]] = []

            def runner(argv, **kwargs):
                commands.append(list(argv))
                if argv[:2] == ["launchctl", "print"]:
                    return completed(argv)
                if argv[:2] == ["launchctl", "bootstrap"] and kwargs.get("check"):
                    raise subprocess.CalledProcessError(5, argv)
                return completed(argv)

            with (
                mock.patch.object(fleet_service.shutil, "which", return_value=None),
                self.assertRaises(subprocess.CalledProcessError),
            ):
                fleet_service.install_service(paths, replace=True, runner=runner)

            self.assertEqual(paths.plist.read_bytes(), old_data)
            self.assertIn(["launchctl", "bootout", f"gui/{os.getuid()}/{paths.label}"], commands)
            self.assertEqual(
                sum(command[:2] == ["launchctl", "bootstrap"] for command in commands), 2
            )

    def test_uninstall_refuses_nonterminal_jobs_without_force(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.make_paths(Path(temporary))
            paths.plist.parent.mkdir(parents=True)
            paths.plist.write_text("installed", encoding="utf-8")
            runner = mock.Mock(return_value=completed([]))

            with self.assertRaisesRegex(RuntimeError, "2 nonterminal"):
                fleet_service.uninstall_service(
                    paths,
                    count_nonterminal_jobs=lambda _root: 2,
                    runner=runner,
                )

            self.assertTrue(paths.plist.exists())
            runner.assert_not_called()

    def test_force_uninstall_boots_out_loaded_service(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.make_paths(Path(temporary))
            paths.plist.parent.mkdir(parents=True)
            paths.plist.write_text("installed", encoding="utf-8")
            commands: list[list[str]] = []

            def runner(argv, **_kwargs):
                commands.append(list(argv))
                return completed(argv)

            changed = fleet_service.uninstall_service(paths, force=True, runner=runner)

            self.assertTrue(changed)
            self.assertFalse(paths.plist.exists())
            self.assertIn(["launchctl", "bootout", f"gui/{os.getuid()}/{paths.label}"], commands)

    def test_default_count_nonterminal_jobs_delegates_to_queue_api(self) -> None:
        connection = mock.Mock()
        with (
            mock.patch.object(fleet_service, "_connect_queue", return_value=connection),
            mock.patch(
                "rlab.job_queue.count_nonterminal_jobs",
                return_value=3,
            ) as count_nonterminal_jobs,
        ):
            count = fleet_service._default_count_nonterminal_jobs(Path("/repo"))

        self.assertEqual(count, 3)
        count_nonterminal_jobs.assert_called_once_with(connection)
        connection.close.assert_called_once_with()

    def test_default_discover_machines_delegates_to_queue_api(self) -> None:
        connection = mock.Mock()
        registry = mock.Mock(machines={})
        with (
            mock.patch.object(fleet_service, "_connect_queue", return_value=connection),
            mock.patch(
                "rlab.job_queue.machines_with_service_work",
                return_value=("beast-3",),
            ) as machines_with_service_work,
            mock.patch("rlab.machines.load_machine_registry", return_value=registry),
        ):
            machines = fleet_service._default_discover_machines(Path("/repo"))

        self.assertEqual(machines, ("beast-3",))
        machines_with_service_work.assert_called_once_with(connection)
        connection.close.assert_called_once_with()

    def test_default_reconcile_machine_delegates_to_fleet_api(self) -> None:
        expected = {"reconciled": 1}
        with mock.patch(
            "rlab.fleet.run_service_machine_pass",
            return_value=expected,
        ) as run_service_machine_pass:
            result = fleet_service._default_reconcile_machine(Path("/repo"), "beast-3", 42.0)

        self.assertIs(result, expected)
        run_service_machine_pass.assert_called_once_with(
            machine_name="beast-3",
            machines_path=Path("/repo/experiments/machines.yaml"),
            repo_root=Path("/repo"),
            deadline_monotonic=42.0,
        )

    def test_run_once_fans_out_and_writes_redacted_atomic_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            barrier = threading.Barrier(2)

            def discover(_repo):
                return ["beast-3", "beast-2", "beast-3"]

            def reconcile(_repo, machine, deadline):
                self.assertGreater(deadline, time.monotonic())
                barrier.wait(timeout=1)
                return {"machine": machine, "token": "sensitive", "launched": 1}

            summary = fleet_service.run_service_pass(
                repo_root=root,
                state_dir=root / "state",
                discover_machines=discover,
                reconcile_machine=reconcile,
                max_machine_lanes=2,
                pass_timeout_seconds=2,
            )

            self.assertEqual(summary["status"], "ok")
            self.assertEqual(
                [row["machine"] for row in summary["machines"]], ["beast-2", "beast-3"]
            )
            saved = json.loads((root / "state" / "last-pass.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["machines"][0]["detail"]["token"], "[REDACTED]")
            log_text = (root / "state" / "service.jsonl").read_text(encoding="utf-8")
            self.assertNotIn("sensitive", log_text)

    def test_run_once_records_machine_failure_without_blocking_other_lane(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def reconcile(_repo, machine, _deadline):
                if machine == "beast-2":
                    raise RuntimeError("unreachable password=hunter2")
                return {"reconciled": 1}

            summary = fleet_service.run_service_pass(
                repo_root=root,
                state_dir=root / "state",
                discover_machines=lambda _root: ["beast-2", "beast-3"],
                reconcile_machine=reconcile,
            )

            self.assertEqual(summary["status"], "degraded")
            by_machine = {row["machine"]: row for row in summary["machines"]}
            self.assertEqual(by_machine["beast-3"]["status"], "ok")
            self.assertEqual(by_machine["beast-2"]["error"], "unreachable password=[REDACTED]")

    def test_eval_health_rejects_fresh_failed_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.make_paths(Path(temporary))
            paths.state_dir.mkdir(parents=True)
            paths.last_pass.write_text(
                json.dumps(
                    {
                        "finished_at": fleet_service._iso_utc(),
                        "eval": {"status": "error", "error": "bad rank"},
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch.object(fleet_service, "service_is_loaded", return_value=True):
                health = fleet_service.eval_service_health(paths)

        self.assertFalse(health["ready"])
        self.assertEqual(health["eval_status"], "error")

    def test_eval_health_requires_successful_eval_detail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.make_paths(Path(temporary))
            paths.state_dir.mkdir(parents=True)
            paths.last_pass.write_text(
                json.dumps(
                    {
                        "finished_at": fleet_service._iso_utc(),
                        "eval": {"status": "ok", "detail": {"status": "ok"}},
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch.object(fleet_service, "service_is_loaded", return_value=True):
                health = fleet_service.eval_service_health(paths)

        self.assertTrue(health["ready"])

    def test_idle_pass_does_not_reconcile(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            reconcile = mock.Mock()
            summary = fleet_service.run_service_pass(
                repo_root=Path(temporary),
                state_dir=Path(temporary) / "state",
                discover_machines=lambda _root: [],
                reconcile_machine=reconcile,
            )
        self.assertEqual(summary["status"], "idle")
        reconcile.assert_not_called()

    def test_parser_exposes_public_lifecycle_and_internal_run_once(self) -> None:
        parser = fleet_service.build_parser()
        for argv, function in (
            (["service", "install"], fleet_service.cmd_install),
            (["service", "status"], fleet_service.cmd_status),
            (["service", "doctor"], fleet_service.cmd_doctor),
            (["service", "logs"], fleet_service.cmd_logs),
            (["service", "uninstall"], fleet_service.cmd_uninstall),
            (["run-once"], fleet_service.cmd_run_once),
        ):
            self.assertIs(parser.parse_args(argv).func, function)


if __name__ == "__main__":
    unittest.main()
