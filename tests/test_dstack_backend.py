from __future__ import annotations

import json
import subprocess
import unittest
from unittest import mock

from rlab.dstack_backend import (
    DSTACK_VERSION,
    ComputeRequest,
    DstackBackend,
    TaskRequest,
    render_fleet_config,
    render_task_config,
)
from rlab.run_contracts import new_run_id


class DstackBackendTests(unittest.TestCase):
    def compute(self, **overrides) -> ComputeRequest:
        values = {
            "kind": "local",
            "target": "b3",
            "max_price": None,
            "max_cost_usd": None,
            "allow_on_demand": False,
            "max_duration_seconds": 24 * 3600,
        }
        values.update(overrides)
        return ComputeRequest(**values)

    def task(self, **overrides) -> TaskRequest:
        run_id = new_run_id()
        values = {
            "run_id": run_id,
            "task_name": run_id,
            "image": "docker:registry.example/rlab@sha256:" + "a" * 64,
            "manifest_uri": "s3://control/runs/manifest.json",
            "compute": self.compute(),
            "secret_env": [
                "RLAB_CONTROL_R2_ACCESS_KEY_ID",
                "RLAB_CONTROL_R2_SECRET_ACCESS_KEY",
            ],
            "rom_mount": "/srv/rlab/roms-ro:/roms",
        }
        values.update(overrides)
        return TaskRequest(**values)

    def test_local_config_reuses_one_b3_fleet_machine(self) -> None:
        config = render_task_config(self.task())
        self.assertEqual(config["fleets"], ["b3"])
        self.assertEqual(config["creation_policy"], "reuse")
        self.assertEqual(config["resources"]["cpu"], "12..")
        self.assertEqual(config["resources"]["memory"], "40GB..")
        self.assertEqual(config["resources"]["gpu"], "1")
        self.assertEqual(config["resources"]["disk"], "50GB..")
        self.assertEqual(config["retry"]["on_events"], ["no-capacity", "interruption"])
        self.assertNotIn("error", config["retry"]["on_events"])
        self.assertEqual(config["max_duration"], "1d")
        self.assertNotIn("max_price", config)
        self.assertEqual(config["volumes"], ["/srv/rlab/roms-ro:/roms"])
        self.assertIn("RLAB_ROM_CACHE_READ_ONLY=1", config["env"])

    def test_spot_requires_both_price_and_total_cost(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires --max-price"):
            self.compute(kind="spot", target="aws").validate()
        compute = self.compute(
            kind="spot",
            target="aws",
            max_price=1.0,
            max_cost_usd=2.0,
            max_duration_seconds=10 * 3600,
        )
        config = render_task_config(self.task(compute=compute))
        self.assertEqual(config["spot_policy"], "spot")
        self.assertEqual(config["max_price"], 1.0)
        self.assertEqual(config["max_duration"], "2h")

    def test_on_demand_requires_explicit_permission(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires --allow-on-demand"):
            self.compute(
                kind="on-demand",
                max_price=1.0,
                max_cost_usd=10.0,
            ).validate()

    def test_auto_without_budget_waits_for_local_capacity(self) -> None:
        config = render_task_config(
            self.task(compute=self.compute(kind="auto", target=None))
        )
        self.assertEqual(config["fleets"], ["b3"])
        self.assertEqual(config["creation_policy"], "reuse")

    def test_auto_with_budget_prefers_reuse_and_allows_bounded_spot(self) -> None:
        config = render_task_config(
            self.task(
                compute=self.compute(
                    kind="auto",
                    target=None,
                    max_price=1.0,
                    max_cost_usd=3.0,
                ),
            )
        )
        self.assertEqual(config["creation_policy"], "reuse-or-create")
        self.assertEqual(config["spot_policy"], "spot")
        self.assertNotIn("fleets", config)
        self.assertEqual(config["max_duration"], "3h")

    def test_auto_selection_uses_idle_b3_before_spot(self) -> None:
        backend = DstackBackend(environment={})
        request = self.compute(
            kind="auto",
            target=None,
            max_price=1.0,
            max_cost_usd=3.0,
        )
        with (
            mock.patch.object(backend, "preflight"),
            mock.patch.object(
                backend,
                "_command",
                return_value=subprocess.CompletedProcess(
                    ["dstack", "offer"],
                    0,
                    json.dumps(
                        {
                            "offers": [
                                {
                                    "backend": "ssh",
                                    "availability": "idle",
                                    "price": 0.0,
                                }
                            ]
                        }
                    ),
                    "",
                ),
            ),
        ):
            selected, offer = backend.select_compute(request)
        self.assertEqual(selected.kind, "local")
        self.assertEqual(selected.target, "b3")
        self.assertEqual(offer["backend"], "ssh")

    def test_auto_selection_falls_back_to_bounded_spot_when_b3_is_busy(self) -> None:
        backend = DstackBackend(environment={})
        request = self.compute(
            kind="auto",
            target=None,
            max_price=1.0,
            max_cost_usd=3.0,
        )
        with (
            mock.patch.object(backend, "preflight"),
            mock.patch.object(
                backend,
                "_command",
                return_value=subprocess.CompletedProcess(
                    ["dstack", "offer"],
                    0,
                    json.dumps({"offers": [{"availability": "busy"}]}),
                    "",
                ),
            ),
        ):
            selected, offer = backend.select_compute(request)
        self.assertEqual(selected.kind, "spot")
        self.assertIsNone(selected.target)
        self.assertIsNone(offer)

    def test_fleet_is_one_unsplit_ssh_host(self) -> None:
        config = render_fleet_config(
            name="b3",
            hostname="host.docker.internal",
            user="tsilva",
            identity_file="~/.ssh/id_ed25519",
        )
        self.assertEqual(config["blocks"], 1)
        self.assertEqual(config["ssh_config"]["hosts"], ["host.docker.internal"])

    @mock.patch("rlab.dstack_backend.shutil.which", return_value="/bin/dstack")
    @mock.patch("rlab.dstack_backend.subprocess.run")
    def test_submit_checks_version_and_sends_yaml_on_stdin(self, run, _which) -> None:
        run.side_effect = [
            subprocess.CompletedProcess(["dstack", "-v"], 0, DSTACK_VERSION + "\n", ""),
            subprocess.CompletedProcess(["dstack", "apply"], 0, "submitted\n", ""),
        ]
        backend = DstackBackend(
            environment={
                "PATH": "/bin",
                "DSTACK_SERVER_URL": "http://127.0.0.1:3000",
                "DSTACK_TOKEN": "secret",
            }
        )
        request = self.task()
        task = backend.submit(request)
        self.assertEqual(task.name, request.task_name)
        submitted = run.call_args_list[1]
        self.assertIn("on_events:", submitted.kwargs["input"])
        self.assertNotIn("DSTACK_TOKEN", submitted.kwargs["input"])

    @mock.patch("rlab.dstack_backend.subprocess.run")
    def test_status_reads_json(self, run) -> None:
        run.return_value = subprocess.CompletedProcess(
            ["dstack", "ps"],
            0,
            json.dumps(
                {
                    "runs": [
                        {
                            "run_name": None,
                            "submitted_at": "2026-07-24T12:00:00Z",
                            "status": "terminated",
                            "run_spec": {"configuration": {"name": "run-one"}},
                        },
                        {
                            "run_name": None,
                            "submitted_at": "2026-07-24T13:00:00Z",
                            "status": "running",
                            "run_spec": {"configuration": {"name": "run-one"}},
                        },
                    ]
                }
            ),
            "",
        )
        backend = DstackBackend(environment={})
        self.assertEqual(backend.status("run-one").status, "running")


if __name__ == "__main__":
    unittest.main()
