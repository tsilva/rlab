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
    def test_service_queue_connections_bypass_the_pool_for_session_locks(self) -> None:
        connection = object()
        with (
            mock.patch.object(fleet_service, "_load_repo_environment"),
            mock.patch("rlab.job_queue.database_url", return_value="direct") as database_url,
            mock.patch("rlab.job_queue.connect", return_value=connection) as connect,
        ):
            actual = fleet_service._connect_queue(Path("/repo"))

        self.assertIs(actual, connection)
        database_url.assert_called_once_with(use_direct=True)
        connect.assert_called_once_with("direct")

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
        self.assertEqual(
            payload["ProgramArguments"][1:4],
            ["-m", "rlab.fleet_service_entrypoint", "run-once"],
        )
        self.assertNotIn("DATABASE_URL", json.dumps(payload))

    def test_split_controller_plists_are_persistent_isolated_and_secret_free(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.make_paths(Path(temporary))
            payloads = {
                name: plistlib.loads(
                    fleet_service.render_controller_launch_agent_plist(paths, name)
                )
                for name in fleet_service.CONTROLLER_NAMES
            }

        self.assertEqual(set(payloads), {"machine", "evaluation", "wandb"})
        for name, payload in payloads.items():
            self.assertIs(payload["KeepAlive"], True)
            self.assertEqual(payload["ThrottleInterval"], 2)
            self.assertNotIn("RunAtLoad", payload)
            self.assertNotIn("StartInterval", payload)
            self.assertNotIn("EnvironmentVariables", payload)
            self.assertEqual(
                payload["ProgramArguments"][1:4],
                ["-m", "rlab.fleet_controllers", name],
            )
            self.assertEqual(
                payload["ProgramArguments"][-2:],
                ["--protocol-version", str(fleet_service.CONTROL_PLANE_PROTOCOL_VERSION)],
            )
            self.assertNotIn("DATABASE_URL", json.dumps(payload))

    def test_split_controller_install_bootstraps_three_labels_and_retires_combined(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.make_paths(Path(temporary))
            paths.plist.parent.mkdir(parents=True)
            paths.plist.write_text("legacy", encoding="utf-8")
            commands: list[list[str]] = []

            def runner(argv, **_kwargs):
                commands.append(list(argv))
                if argv[:2] == ["launchctl", "print"]:
                    target = str(argv[-1])
                    return completed(
                        argv,
                        returncode=0 if target.endswith(paths.label) else 113,
                    )
                return completed(argv)

            with mock.patch.object(fleet_service.shutil, "which", return_value=None):
                result = fleet_service.install_controller_services(paths, runner=runner)

            self.assertTrue(result.installed)
            self.assertTrue(result.kicked)
            self.assertFalse(paths.plist.exists())
            for name in fleet_service.CONTROLLER_NAMES:
                item = fleet_service.controller_service_paths(paths, name)
                self.assertTrue(item.plist.is_file())
                self.assertIn(
                    ["launchctl", "bootstrap", f"gui/{os.getuid()}", str(item.plist)],
                    commands,
                )
                self.assertIn(
                    ["launchctl", "kickstart", f"gui/{os.getuid()}/{item.label}"],
                    commands,
                )
            self.assertIn(["launchctl", "bootout", f"gui/{os.getuid()}/{paths.label}"], commands)

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

    def test_controller_reload_is_quiescent_and_restarts_only_requested_lane(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.make_paths(Path(temporary))
            item = fleet_service.controller_service_paths(paths, "evaluation")
            item.plist.parent.mkdir(parents=True)
            item.plist.write_text("installed", encoding="utf-8")
            commands: list[list[str]] = []

            def runner(argv, **_kwargs):
                commands.append(list(argv))
                if argv[:2] == ["launchctl", "print"]:
                    return completed(argv, stdout="state = running")
                return completed(argv)

            reloaded = fleet_service.reload_controller_service(
                paths,
                "evaluation",
                count_nonterminal_jobs=lambda _root: 0,
                runner=runner,
            )

        self.assertTrue(reloaded)
        target = f"gui/{os.getuid()}/{item.label}"
        self.assertIn(["launchctl", "bootout", target], commands)
        self.assertIn(
            ["launchctl", "bootstrap", f"gui/{os.getuid()}", str(item.plist)],
            commands,
        )
        self.assertFalse(any("machine" in " ".join(command) for command in commands))

    def test_controller_reload_refuses_while_jobs_are_nonterminal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.make_paths(Path(temporary))
            runner = mock.MagicMock()

            with self.assertRaisesRegex(RuntimeError, "2 nonterminal job"):
                fleet_service.reload_controller_service(
                    paths,
                    "evaluation",
                    count_nonterminal_jobs=lambda _root: 2,
                    runner=runner,
                )

        runner.assert_not_called()

    def test_launch_preflight_accepts_compatible_controllers_with_active_jobs(self) -> None:
        status = {
            "installed": True,
            "loaded": True,
            "running": True,
            "protocol_compatible": True,
            "controllers": {
                name: {
                    "installed": True,
                    "loaded": True,
                    "running": True,
                    "protocol_compatible": True,
                }
                for name in fleet_service.CONTROLLER_NAMES
            },
        }
        with (
            mock.patch.object(fleet_service, "controller_services_status", return_value=status),
            mock.patch.object(
                fleet_service,
                "_default_count_nonterminal_jobs",
                side_effect=AssertionError("preflight inspected active jobs"),
            ),
        ):
            self.assertIs(fleet_service.require_compatible_controller_services(), status)

    def test_launch_preflight_rejects_protocol_mismatch_with_exact_remediation(self) -> None:
        status = {
            "installed": True,
            "loaded": True,
            "running": True,
            "protocol_compatible": False,
            "controllers": {
                name: {
                    "installed": True,
                    "loaded": True,
                    "running": True,
                    "protocol_compatible": name != "evaluation",
                }
                for name in fleet_service.CONTROLLER_NAMES
            },
        }
        with mock.patch.object(fleet_service, "controller_services_status", return_value=status):
            with self.assertRaisesRegex(RuntimeError, "rlab fleet service install --replace"):
                fleet_service.require_compatible_controller_services()

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
                mock.patch.object(fleet_service.time, "sleep"),
                self.assertRaises(subprocess.CalledProcessError),
            ):
                fleet_service.install_service(paths, replace=True, runner=runner)

            self.assertEqual(paths.plist.read_bytes(), old_data)
            self.assertIn(["launchctl", "bootout", f"gui/{os.getuid()}/{paths.label}"], commands)
            self.assertEqual(
                sum(command[:2] == ["launchctl", "bootstrap"] for command in commands),
                fleet_service.SERVICE_BOOTSTRAP_ATTEMPTS + 1,
            )

    def test_replacement_retries_launchd_busy_after_bootout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.make_paths(Path(temporary))
            paths.plist.parent.mkdir(parents=True)
            paths.plist.write_bytes(b"old plist\n")
            bootstrap_attempts = 0

            def runner(argv, **kwargs):
                nonlocal bootstrap_attempts
                if argv[:2] == ["launchctl", "print"]:
                    return completed(argv)
                if argv[:2] == ["launchctl", "bootstrap"] and kwargs.get("check"):
                    bootstrap_attempts += 1
                    if bootstrap_attempts == 1:
                        raise subprocess.CalledProcessError(5, argv)
                return completed(argv)

            with (
                mock.patch.object(fleet_service.shutil, "which", return_value=None),
                mock.patch.object(fleet_service.time, "sleep") as sleep,
            ):
                result = fleet_service.install_service(paths, replace=True, runner=runner)

            self.assertTrue(result.replaced)
            self.assertEqual(bootstrap_attempts, 2)
            sleep.assert_called_once_with(fleet_service.SERVICE_BOOTSTRAP_RETRY_SECONDS)

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

    def test_idle_machine_maintenance_failure_is_not_retried_every_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            connection = mock.Mock()
            machine = mock.Mock(prewarm_latest_runtime=False)
            registry = mock.Mock(machines={"beast-2": machine})
            with (
                mock.patch.object(fleet_service, "_connect_queue", return_value=connection),
                mock.patch("rlab.job_queue.machines_with_service_work", return_value=()),
                mock.patch("rlab.machines.load_machine_registry", return_value=registry),
            ):
                first = fleet_service._default_discover_machines(repo)
                second = fleet_service._default_discover_machines(repo)

        self.assertEqual(first, ("beast-2",))
        self.assertEqual(second, ())

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

    def test_eval_health_uses_the_persistent_evaluation_controller(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.make_paths(Path(temporary))
            for name in fleet_service.CONTROLLER_NAMES:
                controller = fleet_service.controller_service_paths(paths, name)
                controller.plist.parent.mkdir(parents=True, exist_ok=True)
                controller.plist.write_text("installed", encoding="utf-8")

            def runner(argv, **_kwargs):
                label = str(argv[-1])
                if label.endswith(".evaluation"):
                    return completed(argv, stdout="state = running")
                return completed(argv, returncode=113)

            health = fleet_service.eval_service_health(paths, runner=runner)

        self.assertTrue(health["ready"])
        self.assertTrue(health["loaded"])
        self.assertEqual(health["eval_status"], "ok")

    def test_eval_health_requires_successful_eval_detail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.make_paths(Path(temporary))
            paths.state_dir.mkdir(parents=True)
            paths.last_pass.write_text(
                json.dumps(
                    {
                        "finished_at": fleet_service._iso_utc(),
                        "eval": {
                            "status": "ok",
                            "detail": {
                                "status": "ok",
                                "app_cleanup": {"status": "partial", "stopped": 1},
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch.object(fleet_service, "service_is_loaded", return_value=True):
                health = fleet_service.eval_service_health(paths)

        self.assertTrue(health["ready"])
        self.assertEqual(
            health["app_cleanup"],
            {"status": "partial", "stopped": 1},
        )

    def test_service_health_alerts_after_two_failures_and_on_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self.make_paths(Path(temporary))
            notifications = []

            def notifier(title, message):
                notifications.append((title, message))

            failed = {
                "status": "degraded",
                "eval": {"status": "ok"},
                "machines": [{"machine": "beast-2", "status": "error", "error": "unreachable"}],
            }

            first = fleet_service.record_service_health(paths, failed, notifier=notifier)
            second = fleet_service.record_service_health(paths, failed, notifier=notifier)
            recovered = fleet_service.record_service_health(
                paths,
                {"status": "ok", "eval": {"status": "ok"}, "machines": []},
                notifier=notifier,
            )

        self.assertEqual(first["consecutive_failures"], 1)
        self.assertFalse(first["alert_active"])
        self.assertEqual(second["consecutive_failures"], 2)
        self.assertTrue(second["alert_active"])
        self.assertTrue(recovered["healthy"])
        self.assertEqual(
            [title for title, _message in notifications],
            ["rlab fleet service failure", "rlab fleet service recovered"],
        )

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

    def test_wake_requests_coalesce_and_are_consumed_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state_dir = Path(temporary) / "state"
            fleet_service.record_wake_request(
                "train_enqueue",
                entity_kind="batch",
                entity_id="bx1",
                state_dir=state_dir,
            )
            fleet_service.record_wake_request(
                "machine_resume",
                entity_kind="machine",
                entity_id="beast-3",
                state_dir=state_dir,
            )
            consumed = fleet_service.consume_wake_requests(state_dir)
            second = fleet_service.consume_wake_requests(state_dir)

        self.assertEqual([row["reason"] for row in consumed], ["train_enqueue", "machine_resume"])
        self.assertEqual(second, [])

    def test_live_pass_state_is_atomic_visible_and_removed_after_last_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entered = threading.Event()
            release = threading.Event()
            result: dict[str, object] = {}

            def reconcile(_repo, _machine, _deadline):
                entered.set()
                self.assertTrue(release.wait(timeout=2))
                return {"launched": 1}

            def run() -> None:
                result.update(
                    fleet_service.run_service_pass(
                        repo_root=root,
                        state_dir=root / "state",
                        discover_machines=lambda _repo: ["beast-3"],
                        reconcile_machine=reconcile,
                        reconcile_eval=lambda _repo, _deadline: {"status": "ok"},
                        workload_snapshot=lambda _repo: {
                            "captured_at": fleet_service._iso_utc(),
                            "in_progress": 1,
                            "needs_action": [],
                            "retrying": [],
                            "waiting": [],
                            "counts": {},
                        },
                    )
                )

            thread = threading.Thread(target=run)
            thread.start()
            self.assertTrue(entered.wait(timeout=2))
            current_path = root / "state" / "current-pass.json"
            current = json.loads(current_path.read_text(encoding="utf-8"))
            self.assertTrue(current["pass_id"])
            self.assertIn("beast-3", current["lanes"])
            self.assertEqual(current["workload"]["in_progress"], 1)
            release.set()
            thread.join(timeout=3)

            saved = json.loads((root / "state" / "last-pass.json").read_text(encoding="utf-8"))
            self.assertFalse(current_path.exists())

        self.assertFalse(thread.is_alive())
        self.assertEqual(saved["pass_id"], result["pass_id"])
        self.assertEqual(saved["schema_version"], 2)
        self.assertIn("source", saved)

    def test_kick_with_reason_persists_trigger_before_launchctl(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state_dir = Path(temporary) / "state"
            observed: list[dict[str, object]] = []

            def runner(argv, **_kwargs):
                payload = json.loads((state_dir / "wake-requests.json").read_text())
                observed.extend(payload["requests"])
                return completed(argv)

            kicked = fleet_service.kick_service(
                reason="train_cancel",
                entity_kind="train",
                entity_id=31,
                state_dir=state_dir,
                runner=runner,
            )

        self.assertTrue(kicked)
        self.assertEqual(observed[0]["reason"], "train_cancel")
        self.assertEqual(observed[0]["entity_id"], "31")

    def test_kick_routes_work_to_the_owning_split_controller(self) -> None:
        commands: list[list[str]] = []

        def runner(argv, **_kwargs):
            commands.append(list(argv))
            return completed(argv)

        for entity_kind in ("train", "batch", "machine"):
            self.assertTrue(
                fleet_service.kick_service(entity_kind=entity_kind, runner=runner, uid=501)
            )
        self.assertTrue(fleet_service.kick_service(entity_kind="eval", runner=runner, uid=501))
        self.assertTrue(fleet_service.kick_service(entity_kind="wandb", runner=runner, uid=501))

        self.assertEqual(
            [command[-1] for command in commands],
            [
                "gui/501/com.rlab.fleet-service.machine",
                "gui/501/com.rlab.fleet-service.machine",
                "gui/501/com.rlab.fleet-service.machine",
                "gui/501/com.rlab.fleet-service.evaluation",
                "gui/501/com.rlab.fleet-service.wandb",
            ],
        )

    def test_parser_exposes_public_lifecycle_and_internal_run_once(self) -> None:
        parser = fleet_service.build_parser()
        for argv, function in (
            (["service", "install"], fleet_service.cmd_install),
            (["service", "status"], fleet_service.cmd_status),
            (
                ["service", "reload", "--controller", "evaluation"],
                fleet_service.cmd_reload,
            ),
            (["service", "doctor"], fleet_service.cmd_doctor),
            (["service", "logs"], fleet_service.cmd_logs),
            (["service", "watch"], fleet_service.cmd_watch),
            (["service", "uninstall"], fleet_service.cmd_uninstall),
            (["run-once"], fleet_service.cmd_run_once),
        ):
            self.assertIs(parser.parse_args(argv).func, function)


if __name__ == "__main__":
    unittest.main()
