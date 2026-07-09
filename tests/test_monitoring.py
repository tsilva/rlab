from __future__ import annotations

import unittest
from pathlib import Path

from rlab.monitoring.server import format_monitor_state
from rlab.monitoring.state import (
    DeviceProbe,
    MonitorOptions,
    base_devices,
    collect_state,
    devices_from_jobs,
    infer_device_key,
    job_from_train_row,
    parse_probe_metrics,
    resource_metrics,
    sample_jobs,
    target_label,
)


class MonitoringStateTests(unittest.TestCase):
    def test_monitor_cli_table_is_not_html_dashboard(self) -> None:
        text = format_monitor_state(
            {
                "source": {"queue": "sample", "message": "offline sample"},
                "refreshed_at": "2026-06-26T12:00:00Z",
                "jobs": [
                    {
                        "id": "train-1",
                        "target": "Mario L1",
                        "device": "beast-3",
                        "container": "rlab-beast-3-any-profile",
                        "state": "running",
                        "progress": "42%",
                        "attention": "",
                    },
                ],
                "devices": [],
            },
            view="jobs",
        )

        self.assertIn("rlab monitor: sample - offline sample", text)
        self.assertIn("Job", text)
        self.assertIn("Device", text)
        self.assertIn("Container", text)
        self.assertIn("train-1", text)
        self.assertNotIn("<html", text)
        self.assertNotIn("target=\"_blank\"", text)

    def test_target_label_summarizes_mixed_mario_state_probabilities(self) -> None:
        label = target_label(
            {
                "game": "SuperMarioBros-Nes-v0",
                "states": ["Level1-1", "Level1-2"],
                "state_probs": [0.5, 0.5],
            }
        )

        self.assertEqual(label, "Mario mixed 0.5 / 0.5")

    def test_target_label_compresses_repeated_fixed_states(self) -> None:
        label = target_label(
            {
                "game": "SuperMarioBros-Nes-v0",
                "states": ["Level1-1", "Level1-1", "Level1-2", "Level1-2"],
            }
        )

        self.assertEqual(label, "Mario L1-1 x2 + L1-2 x2")

    def test_devices_reflect_active_jobs(self) -> None:
        devices = devices_from_jobs(Path("."), sample_jobs())
        by_id = {device["id"]: device for device in devices}

        self.assertEqual(by_id["rtx4090"]["state"], "warning")
        self.assertIn("train-184", by_id["rtx4090"]["current_job"])
        self.assertEqual(by_id["local-macbook"]["state"], "available")
        self.assertEqual(by_id["rtx2060"]["state"], "available")

    def test_base_devices_include_all_configured_compute_targets(self) -> None:
        devices = base_devices(Path("."))
        by_id = {device["id"]: device for device in devices}

        self.assertIn("rtx4090", by_id)
        self.assertIn("rtx2060", by_id)
        self.assertIn("local-macbook", by_id)
        self.assertEqual(by_id["rtx4090"]["target"], "docker/beast-3")
        self.assertEqual(by_id["rtx4090"]["capacity"], "6 workers")
        self.assertEqual(by_id["rtx4090"]["details"]["manager"], "rlab fleet")
        self.assertEqual(by_id["rtx4090"]["details"]["machine"], "beast-3")
        self.assertEqual(by_id["rtx4090"]["details"]["runner_capacity"], 6)
        self.assertEqual(by_id["local-macbook"]["target"], "local_docker/local-macbook")
        self.assertEqual(by_id["local-macbook"]["details"]["manager"], "rlab fleet")
        self.assertEqual(by_id["local-macbook"]["details"]["machine"], "local-macbook")

    def test_jobs_can_route_to_beast_target_rows(self) -> None:
        jobs = [
            {
                "id": "train-9",
                "kind": "train",
                "target": "Mario L1",
                "device": "beast-2",
                "container": "rlab-beast-2-any-profile",
                "device_key": "rtx2060",
                "state": "running",
                "progress": "",
                "attention": "",
                "details": {},
            }
        ]

        devices = devices_from_jobs(Path("."), jobs)
        by_id = {device["id"]: device for device in devices}

        self.assertEqual(by_id["rtx2060"]["state"], "busy")
        self.assertEqual(by_id["rtx2060"]["current_job"], "train-9")

    def test_target_with_4090_routes_to_beast_3(self) -> None:
        device_key = infer_device_key(
            "train",
            "train-launch",
            {"device": "cuda"},
            run_target="rtx4090",
        )

        self.assertEqual(device_key, "rtx4090")

    def test_explicit_run_target_drives_device_inference(self) -> None:
        device_key = infer_device_key(
            "train",
            "train-launch",
            {"device": "cuda"},
            run_target="rtx2060",
        )

        self.assertEqual(device_key, "rtx2060")

    def test_pending_jobs_do_not_make_device_busy(self) -> None:
        jobs = [
            {
                "id": "train-4",
                "kind": "train",
                "target": "checkpoint training",
                "device": "beast-3",
                "container": "",
                "device_key": "rtx4090",
                "state": "pending",
                "progress": "",
                "attention": "",
                "details": {},
            }
        ]

        devices = devices_from_jobs(Path("."), jobs)
        by_id = {device["id"]: device for device in devices}

        self.assertEqual(by_id["rtx4090"]["state"], "available")
        self.assertEqual(by_id["rtx4090"]["current_job"], "")
        self.assertEqual(by_id["rtx4090"]["queued_job"], "train-4")
        self.assertEqual(by_id["rtx4090"]["attention"], "1 queued")
        self.assertEqual(by_id["rtx4090"]["details"]["running jobs"], "")
        self.assertEqual(by_id["rtx4090"]["details"]["queued jobs"], "train-4")

    def test_train_job_includes_full_payload_for_queue_inspection(self) -> None:
        job = job_from_train_row(
            {
                "id": 12,
                "goal_slug": "goal",
                "recipe_slug": "rtx4090-screening",
                "train_config": {"game": "SuperMarioBros-Nes-v0", "n_envs": 32},
                "status": "pending",
                "lease_owner": None,
                "heartbeat_at": None,
                "lease_expires_at": None,
                "error": None,
                "artifact_refs": [],
                "metrics_json": {},
                "wandb_url": None,
                "run_name": "run-12",
                "job_payload": {
                    "id": 12,
                    "recipe_slug": "rtx4090-screening",
                    "train_config": {"game": "SuperMarioBros-Nes-v0", "n_envs": 32},
                    "status": "pending",
                },
                "result_payload": None,
            }
        )

        self.assertEqual(job["payload"]["table"], "train_jobs")
        self.assertEqual(job["payload"]["config_key"], "train_config")
        self.assertEqual(job["payload"]["job"]["train_config"]["n_envs"], 32)
        self.assertIn("train_config", job["payload"]["schema"])

    def test_running_train_job_uses_queue_state_without_result_projection(self) -> None:
        job = job_from_train_row(
            {
                "id": 12,
                "goal_slug": "goal",
                "recipe_slug": "rtx4090-screening",
                "run_target": "rtx4090",
                "train_config": {"game": "SuperMarioBros-Nes-v0", "n_envs": 32},
                "status": "running",
                "lease_owner": "rlab-beast-3-rtx4090-any-profile-aaaaaaaaaaaa-4-56ea2c67",
                "heartbeat_at": None,
                "lease_expires_at": None,
                "error": None,
                "run_name": "run-12",
                "job_payload": {
                    "id": 12,
                    "recipe_slug": "rtx4090-screening",
                    "train_config": {"game": "SuperMarioBros-Nes-v0", "n_envs": 32},
                    "status": "running",
                },
            }
        )

        self.assertEqual(job["wandb_url"], "")
        self.assertEqual(job["device"], "beast-3")
        self.assertEqual(job["container"], "rlab-beast-3-rtx4090-any-profile-aaaaaaaaaaaa")
        self.assertEqual(job["details"]["device"], "beast-3")
        self.assertEqual(job["details"]["container"], "rlab-beast-3-rtx4090-any-profile-aaaaaaaaaaaa")
        self.assertEqual(job["details"]["wandb"], "")
        self.assertEqual(job["state"], "running")

    def test_profileless_train_job_surfaces_target_and_runtime_digest(self) -> None:
        job = job_from_train_row(
            {
                "id": 13,
                "goal_slug": "goal",
                "recipe_slug": None,
                "runtime_image_ref": "docker:ghcr.io/tsilva/rlab/rlab-train@sha256:"
                + "a" * 64,
                "run_target": "rtx4090",
                "train_config": {"game": "SuperMarioBros-Nes-v0", "state": "Level1-1"},
                "status": "pending",
                "attempts": 0,
                "max_attempts": 1,
                "cancel_requested": True,
                "drain_requested": False,
                "lease_owner": None,
                "heartbeat_at": None,
                "lease_expires_at": None,
                "error": None,
                "artifact_refs": [],
                "metrics_json": {},
                "wandb_url": None,
                "run_name": "run-13",
                "job_payload": {
                    "id": 13,
                    "recipe_slug": None,
                    "runtime_image_ref": "docker:ghcr.io/tsilva/rlab/rlab-train@sha256:"
                    + "a" * 64,
                    "run_target": "rtx4090",
                    "train_config": {"game": "SuperMarioBros-Nes-v0", "state": "Level1-1"},
                    "status": "pending",
                },
                "result_payload": None,
            }
        )

        self.assertEqual(job["device_key"], "rtx4090")
        self.assertEqual(job["device"], "beast-3")
        self.assertEqual(job["container"], "")
        self.assertEqual(job["attention"], "cancel requested")
        self.assertEqual(job["details"]["device"], "beast-3")
        self.assertEqual(job["details"]["container"], "")
        self.assertEqual(job["details"]["run_target"], "rtx4090")
        self.assertEqual(job["details"]["runtime_image"], "sha256:aaaaaaaaaaaa")
        self.assertEqual(job["details"]["attempts"], "0/1")

    def test_offline_probe_overrides_idle_device_state(self) -> None:
        devices = devices_from_jobs(
            Path("."),
            [],
            probes={
                "rtx2060": DeviceProbe(
                    ok=False,
                    label="SSH timeout",
                    detail="connection timed out",
                )
            },
        )
        by_id = {device["id"]: device for device in devices}

        self.assertEqual(by_id["rtx2060"]["state"], "offline")
        self.assertEqual(by_id["rtx2060"]["attention"], "unreachable")
        self.assertEqual(by_id["rtx2060"]["last_check"], "unreachable")
        self.assertEqual(by_id["rtx2060"]["details"]["reachability"], "unreachable")
        self.assertEqual(by_id["rtx2060"]["details"]["health check"], "SSH timeout")

    def test_probe_metrics_parse_gpu_cpu_and_memory(self) -> None:
        host, metrics = parse_probe_metrics(
            "\n".join(
                [
                    "host=BEAST-3",
                    "gpu_util_pct=81",
                    "vram_used_mib=12000",
                    "vram_total_mib=24564",
                    "ram_used_mib=32768",
                    "ram_total_mib=65536",
                    "cpu1=cpu  100 0 100 800 0 0 0 0 0 0",
                    "cpu2=cpu  110 0 120 870 0 0 0 0 0 0",
                ]
            )
        )

        self.assertEqual(host, "BEAST-3")
        self.assertEqual(metrics["gpu_util_pct"], 81)
        self.assertEqual(metrics["vram_total_mib"], 24564)
        self.assertAlmostEqual(metrics["cpu_util_pct"], 30.0)

    def test_resource_metrics_prepare_bar_percentages(self) -> None:
        metrics = resource_metrics(
            {
                "gpu_util_pct": 50,
                "cpu_util_pct": 25,
                "ram_used_mib": 2048,
                "ram_total_mib": 4096,
                "vram_used_mib": 1024,
                "vram_total_mib": 4096,
            }
        )

        self.assertEqual(metrics["gpu"]["percent"], 50)
        self.assertEqual(metrics["cpu"]["percent"], 25)
        self.assertEqual(metrics["memory"]["percent"], 50)
        self.assertEqual(metrics["vram"]["percent"], 25)

    def test_sample_state_contains_jobs_and_devices(self) -> None:
        state = collect_state(MonitorOptions(repo_root=Path("."), sample=True))

        self.assertEqual(state["source"]["queue"], "sample")
        self.assertGreaterEqual(len(state["jobs"]), 1)
        self.assertGreaterEqual(len(state["devices"]), 1)

    def test_sample_running_jobs_have_wandb_urls(self) -> None:
        running_jobs = [job for job in sample_jobs() if job["state"] == "running"]

        self.assertGreaterEqual(len(running_jobs), 1)
        self.assertTrue(
            all(str(job.get("wandb_url") or "").startswith("https://wandb.ai/") for job in running_jobs)
        )


if __name__ == "__main__":
    unittest.main()
