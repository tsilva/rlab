from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from rlab.dotenv import load_env_file
from rlab.env_registry import resolve_env_provider
from rlab.job_queue import connect, database_url
from rlab.modal_eval_assets import asset_manifest_for_game, sync_rom_asset
from rlab.modal_eval_config import load_modal_eval_config, modal_app_name
from rlab.modal_eval_storage import ObjectStore, object_store_base_uri
from rlab.runtime_refs import normalize_runtime_image_ref


MODAL_SCHEMA_COLUMNS = {
    "eval_runs": {
        "train_job_id",
        "contract_json",
        "artifact_projection_attempts",
        "artifact_projection_next_retry_at",
        "promoted_artifact_projected_at",
    },
    "eval_jobs": {
        "train_job_id",
        "job_key",
        "stage_index",
        "contract_json",
        "projection_attempts",
        "projection_next_retry_at",
        "retry_round",
    },
    "eval_attempts": {"eval_job_id", "modal_call_id", "result_uri", "retry_round"},
    "eval_backend_state": {"backend", "drained"},
}


def _conn():
    return connect(database_url())


def _kick(reason: str, *, entity_kind: str = "eval", entity_id: str | int = "") -> None:
    from rlab.fleet_service import kick_service

    kick_service(reason=reason, entity_kind=entity_kind, entity_id=entity_id)


def _missing_schema_tables(conn) -> list[str]:
    missing: list[str] = []
    with conn.cursor() as cur:
        for table, required_columns in MODAL_SCHEMA_COLUMNS.items():
            cur.execute("SELECT to_regclass(%(table)s) AS table_name", {"table": table})
            row = cur.fetchone()
            if not row or not row.get("table_name"):
                missing.append(table)
                continue
            cur.execute(
                """
                SELECT column_name FROM information_schema.columns
                WHERE table_schema = current_schema() AND table_name = %(table)s
                """,
                {"table": table},
            )
            columns = {str(value["column_name"]) for value in cur.fetchall()}
            if not required_columns.issubset(columns):
                missing.append(table)
    return missing


def cmd_status(_args: argparse.Namespace) -> int:
    conn = _conn()
    try:
        missing = _missing_schema_tables(conn)
        if missing:
            print(
                json.dumps(
                    {
                        "ready": False,
                        "missing_tables": missing,
                        "remediation": "run: rlab fleet queue setup",
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 1
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM eval_backend_state WHERE backend = 'modal'")
            backend = dict(cur.fetchone() or {})
            cur.execute(
                "SELECT status, count(*) AS count FROM eval_runs GROUP BY status ORDER BY status"
            )
            runs = {str(row["status"]): int(row["count"]) for row in cur.fetchall()}
            cur.execute(
                "SELECT status, count(*) AS count FROM eval_jobs GROUP BY status ORDER BY status"
            )
            jobs = {str(row["status"]): int(row["count"]) for row in cur.fetchall()}
            cur.execute(
                "SELECT status, count(*) AS count FROM eval_attempts GROUP BY status ORDER BY status"
            )
            attempts = {str(row["status"]): int(row["count"]) for row in cur.fetchall()}
        from rlab.fleet_service import default_service_paths, eval_service_health

        fleet_health = eval_service_health(default_service_paths())
        print(
            json.dumps(
                {
                    "ready": bool(fleet_health["ready"]),
                    "backend": backend,
                    "fleet_eval_service": fleet_health,
                    "runs": runs,
                    "jobs": jobs,
                    "attempts": attempts,
                },
                indent=2,
                default=str,
                sort_keys=True,
            )
        )
        return 0 if fleet_health["ready"] else 1
    finally:
        conn.close()


def modal_preflight(
    *,
    runtime_image_ref: str,
    game: str,
    env_provider: str = "",
    runtime_input_sha256: str = "",
    runtime_build_source_sha: str = "",
) -> dict[str, Any]:
    from rlab.runtime_contract import train_config_contract_sha256

    runtime_image_ref = normalize_runtime_image_ref(runtime_image_ref)
    config = load_modal_eval_config()
    app_name = modal_app_name(config.app_name_prefix, runtime_image_ref)
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: object) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": str(detail)})

    add(
        "config_guards",
        config.hard_max_active == 10 and config.max_containers == 10,
        f"enabled={str(config.enabled).lower()} hard_cap={config.hard_max_active} max_containers={config.max_containers}",
    )
    try:
        from rlab.fleet_service import default_service_paths, eval_service_health

        fleet_health = eval_service_health(default_service_paths())
        add(
            "fleet_eval_service",
            bool(fleet_health["ready"]),
            json.dumps(fleet_health, sort_keys=True),
        )
    except Exception as exc:
        add("fleet_eval_service", False, type(exc).__name__)
    conn = None
    try:
        conn = _conn()
        missing = _missing_schema_tables(conn)
        add("postgres_schema", not missing, "ok" if not missing else f"missing={','.join(missing)}")
        if not missing:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM eval_backend_state WHERE backend = 'modal'")
                backend = dict(cur.fetchone() or {})
            add(
                "backend_state",
                bool(backend) and not bool(backend.get("drained")),
                (f"drained={str(bool(backend.get('drained'))).lower()}" if backend else "missing"),
            )
    except Exception as exc:
        add("postgres_schema", False, type(exc).__name__)
    finally:
        if conn is not None:
            conn.close()

    provider = str(env_provider or "").strip()
    requires_rom_asset = not provider or resolve_env_provider(provider).uses_stable_retro_roms
    if requires_rom_asset:
        try:
            manifest = asset_manifest_for_game(game)
            store = ObjectStore(object_store_base_uri())
            head = store.head(str(manifest["object_uri"]))
            remote_sha = str(head.get("metadata", {}).get("sha256") or "")
            expected_sha = str(manifest["sha256"])
            add(
                "rom_asset",
                int(head["size"]) > 0 and (not remote_sha or remote_sha == expected_sha),
                f"game={game} size={int(head['size'])} sha256={expected_sha[:12]}",
            )
        except Exception as exc:
            add("rom_asset", False, type(exc).__name__)
    else:
        add("rom_asset", True, f"not_required provider={provider}")

    try:
        import modal

        modal.Function.from_name(app_name, config.function_name).hydrate()
        add("modal_deployment", True, f"app={app_name} function={config.function_name}")
    except Exception as exc:
        add("modal_deployment", False, f"app={app_name} error={type(exc).__name__}")
    else:
        try:
            probe = modal.Function.from_name(app_name, "startup_probe").remote()
            expected_contract = train_config_contract_sha256()
            probe_ok = (
                isinstance(probe, dict)
                and probe.get("app_name") == app_name
                and probe.get("runtime_image_ref") == runtime_image_ref
                and probe.get("train_config_contract_sha256") == expected_contract
            )
            if runtime_input_sha256:
                probe_ok = probe_ok and (probe.get("runtime_input_sha256") == runtime_input_sha256)
            if runtime_build_source_sha:
                probe_ok = probe_ok and (
                    probe.get("runtime_build_source_sha") == runtime_build_source_sha
                )
            add(
                "modal_startup_probe",
                probe_ok,
                (
                    f"app={app_name} contract={expected_contract[:12]}"
                    if probe_ok
                    else f"app={app_name} incompatible startup receipt"
                ),
            )
        except Exception as exc:
            add("modal_startup_probe", False, f"app={app_name} error={type(exc).__name__}")

    return {
        "ready": all(bool(check["ok"]) for check in checks),
        "runtime_image_ref": runtime_image_ref,
        "app_name": app_name,
        "game": game,
        "env_provider": provider or None,
        "checks": checks,
    }


def cmd_preflight(args: argparse.Namespace) -> int:
    report = modal_preflight(
        runtime_image_ref=args.runtime_image_ref,
        game=args.game,
        env_provider=args.env_provider,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ready"] else 1


def _set_backend(*, drained: bool, reason: str | None) -> int:
    conn = _conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE eval_backend_state
                    SET drained = %(drained)s,
                        reason = %(reason)s, updated_at = now()
                    WHERE backend = 'modal'
                    RETURNING *
                    """,
                    {"drained": drained, "reason": reason},
                )
                row = dict(cur.fetchone())
        print(json.dumps(row, sort_keys=True, default=str))
        _kick(
            "modal_drain" if drained else "modal_resume", entity_kind="backend", entity_id="modal"
        )
        return 0
    finally:
        conn.close()


def cmd_drain(args: argparse.Namespace) -> int:
    return _set_backend(drained=True, reason=args.reason or "operator drain")


def cmd_resume(args: argparse.Namespace) -> int:
    return _set_backend(drained=False, reason=args.reason)


def cmd_retry(args: argparse.Namespace) -> int:
    conn = _conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE eval_jobs j SET status = 'pending', error = NULL,
                      finished_at = NULL, retry_round = retry_round + 1,
                      updated_at = now()
                    WHERE j.id = %(id)s AND j.status IN ('failed', 'blocked_budget')
                    RETURNING *
                    """,
                    {"id": int(args.eval_job_id)},
                )
                row = cur.fetchone()
                if row:
                    cur.execute(
                        """
                        UPDATE eval_runs SET
                          status = CASE WHEN complete_announcement_seen
                            THEN 'finalizing' ELSE 'active' END,
                          error = NULL, updated_at = now()
                        WHERE train_job_id = %(train_job_id)s AND status = 'failed'
                        """,
                        {"train_job_id": int(row["train_job_id"])},
                    )
                    cur.execute(
                        """
                        UPDATE train_jobs SET status = 'finalizing', error = NULL,
                          finished_at = NULL
                        WHERE id = %(train_job_id)s AND status = 'finalization_failed'
                        """,
                        {"train_job_id": int(row["train_job_id"])},
                    )
        if not row:
            raise ValueError("eval job is not retryable")
        print(json.dumps(dict(row), sort_keys=True, default=str))
        _kick("modal_eval_retry", entity_id=int(args.eval_job_id))
        return 0
    finally:
        conn.close()


def cmd_recover(args: argparse.Namespace) -> int:
    from rlab.docker_host import DockerRunnerHost, run_checkpoint_coordinator_container
    from rlab.machines import load_machine_registry, resolve_machine

    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT t.*, l.launch_id, l.output_uri, r.status AS eval_run_status
                FROM train_jobs t
                JOIN job_launches l ON l.job_id = t.id
                LEFT JOIN eval_runs r ON r.train_job_id = t.id
                WHERE t.id = %(id)s
                """,
                {"id": int(args.train_job_id)},
            )
            row = cur.fetchone()
        if not row:
            raise ValueError("train job or stable launch was not found")
        row = dict(row)
        train_status = str(row.get("status") or "")
        if train_status not in {
            "finalizing",
            "succeeded",
            "failed",
            "finalization_failed",
            "canceled",
        }:
            raise ValueError(
                f"train job {args.train_job_id} is {train_status or 'unknown'}, not terminal"
            )
        eval_run_status = str(row.get("eval_run_status") or "")
        if eval_run_status != "awaiting_artifact_recovery":
            raise ValueError(
                f"eval run for train job {args.train_job_id} is "
                f"{eval_run_status or 'missing'}, not awaiting_artifact_recovery"
            )
        machine = resolve_machine(load_machine_registry(), str(row["machine"]))
        host = DockerRunnerHost(machine)
        run_name = str(row.get("run_name") or f"train_job_{row['id']}")
        run_checkpoint_coordinator_container(
            host,
            launch_id=str(row["launch_id"]),
            run_name=run_name,
            runtime_image_ref=str(row["runtime_image_ref"]),
        )
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE eval_runs
                    SET status = 'active', error = NULL, updated_at = now()
                    WHERE train_job_id = %(id)s
                      AND status = 'awaiting_artifact_recovery'
                    RETURNING train_job_id
                    """,
                    {"id": int(args.train_job_id)},
                )
                if not cur.fetchone():
                    raise RuntimeError("eval run left awaiting_artifact_recovery during recovery")
        print(json.dumps({"train_job_id": int(args.train_job_id), "recovered": True}))
        _kick("modal_artifact_recovery", entity_kind="train", entity_id=int(args.train_job_id))
        return 0
    finally:
        conn.close()


def cmd_abandon(args: argparse.Namespace) -> int:
    """Close evaluation work that cannot complete after terminal training failure."""

    train_job_id = int(args.train_job_id)
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT t.status AS train_status, r.status AS eval_run_status,
                  (SELECT count(*) FROM eval_attempts a
                   JOIN eval_jobs j ON j.id = a.eval_job_id
                   WHERE j.train_job_id = t.id
                     AND a.status IN ('dispatching', 'submitted')) AS active_attempts
                FROM train_jobs t
                JOIN eval_runs r ON r.train_job_id = t.id
                WHERE t.id = %(id)s
                """,
                {"id": train_job_id},
            )
            row = cur.fetchone()
        if not row:
            raise ValueError("train job or evaluation run was not found")
        train_status = str(row.get("train_status") or "")
        eval_run_status = str(row.get("eval_run_status") or "")
        if train_status not in {"failed", "finalization_failed", "canceled"}:
            raise ValueError(
                f"train job {train_job_id} is {train_status or 'unknown'}, not terminal-failed"
            )
        if eval_run_status in {"complete", "failed", "canceled"}:
            print(
                json.dumps(
                    {
                        "train_job_id": train_job_id,
                        "abandoned": False,
                        "eval_run_status": eval_run_status,
                    },
                    sort_keys=True,
                )
            )
            return 0
        if int(row.get("active_attempts") or 0):
            raise ValueError(
                f"eval run for train job {train_job_id} still has active Modal attempts"
            )
        if eval_run_status not in {"active", "awaiting_artifact_recovery", "finalizing"}:
            raise ValueError(
                f"eval run for train job {train_job_id} is {eval_run_status or 'unknown'}, "
                "not abandonable"
            )
        terminal_eval_status = "canceled" if train_status == "canceled" else "failed"
        error = f"evaluation abandoned after terminal training state: {train_status}"
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE eval_attempts a
                    SET status = 'canceled', finished_at = now(), error = %(error)s
                    FROM eval_jobs j
                    WHERE j.id = a.eval_job_id
                      AND j.train_job_id = %(id)s
                      AND a.status IN ('dispatching', 'submitted')
                    """,
                    {"id": train_job_id, "error": error},
                )
                attempts = int(cur.rowcount)
                cur.execute(
                    """
                    UPDATE eval_jobs
                    SET status = 'canceled', finished_at = now(), updated_at = now(),
                      error = %(error)s
                    WHERE train_job_id = %(id)s
                      AND status IN ('pending', 'dispatching', 'submitted', 'blocked_budget')
                    """,
                    {"id": train_job_id, "error": error},
                )
                jobs = int(cur.rowcount)
                cur.execute(
                    """
                    UPDATE eval_runs
                    SET status = %(status)s, error = %(error)s, updated_at = now()
                    WHERE train_job_id = %(id)s
                      AND status IN ('active', 'awaiting_artifact_recovery', 'finalizing')
                    """,
                    {
                        "id": train_job_id,
                        "status": terminal_eval_status,
                        "error": error,
                    },
                )
                runs = int(cur.rowcount)
        print(
            json.dumps(
                {
                    "train_job_id": train_job_id,
                    "abandoned": bool(runs),
                    "eval_run_status": terminal_eval_status,
                    "canceled_attempts": attempts,
                    "canceled_jobs": jobs,
                },
                sort_keys=True,
            )
        )
        _kick("modal_eval_abandon", entity_kind="train", entity_id=train_job_id)
        return 0
    finally:
        conn.close()


def cmd_assets_sync(args: argparse.Namespace) -> int:
    manifest = sync_rom_asset(
        args.game,
        rom_path=args.rom_path,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    _kick("modal_assets_sync", entity_kind="game", entity_id=args.game)
    return 0


def cmd_smoke_local(_args: argparse.Namespace) -> int:
    from rlab.checkpoint_coordinator import import_decisions, process_upload, write_complete_marker
    from rlab.metric_names import checkpoint_eval_stage_metric, staged_metric_name
    from rlab.metric_store import MetricStore
    from rlab.modal_eval_projection import project_payload
    from rlab.modal_eval_protocol import (
        PROTOCOL_SCHEMA_VERSION,
        SEED_PROTOCOL,
        apply_decision_rules,
        stage_job_descriptor,
        validate_attempt_result,
    )
    from rlab.modal_eval_storage import ObjectStore

    with tempfile.TemporaryDirectory(prefix="rlab-modal-smoke-") as temporary:
        root = Path(temporary)
        object_store = ObjectStore((root / "objects").resolve().as_uri())
        model = root / "checkpoint.zip"
        metadata = root / "checkpoint.metadata.json"
        model.write_bytes(b"fake-checkpoint")
        metadata.write_text('{"metadata_version": 3}\n', encoding="utf-8")
        asset = root / "game.nes"
        asset.write_bytes(b"NES\x1a" + bytes(12) + b"fake-rom")
        from rlab.modal_eval_storage import file_sha256

        asset_manifest = {
            "schema_version": 1,
            "game": "Smoke-Nes-v0",
            "filename": asset.name,
            "sha256": file_sha256(asset),
            "object_uri": object_store.put_file(
                "modal-assets/smoke/game.nes",
                asset,
                sha256=file_sha256(asset),
            ),
            "provider_rom_identity": "0" * 40,
            "provider_rom_identity_algorithm": "sha1-provider-body-v1",
        }
        args = SimpleNamespace(
            queue_train_job_id=1,
            runtime_image_ref="docker:example.invalid/rlab@sha256:" + "1" * 64,
            wandb_run_id="rlab-smoke",
            wandb_artifact_storage_uri=object_store.base_uri,
            checkpoint_eval_environment={"game": "Smoke-Nes-v0", "task": {}},
            checkpoint_eval_stages=[
                {
                    "name": "screen",
                    "episodes": 2,
                    "n_envs": 2,
                    "pass": [
                        {
                            "metric": "eval/full/episode/return/mean",
                            "operator": ">=",
                            "threshold": 1.0,
                        }
                    ],
                    "candidate_stop": True,
                }
            ],
            checkpoint_eval_asset_manifest=asset_manifest,
            checkpoint_eval_n_envs=2,
            checkpoint_eval_seed_protocol=SEED_PROTOCOL,
            post_train_eval_max_steps=100,
            post_train_eval_episodes=2,
        )
        ledger = MetricStore(root / "metrics.sqlite3")
        ledger.init()
        checkpoint_id = ledger.record_checkpoint(
            run_name="smoke",
            kind="checkpoint",
            step=10,
            path=model,
            metadata_path=metadata,
        )
        row = ledger.pending_artifact_uploads(limit=1)[0]
        if not process_upload(ledger, object_store, args, row):
            raise RuntimeError("checkpoint coordinator did not upload the smoke checkpoint")
        announcement = object_store.get_json("artifact-announcements/1/00000001.json")
        descriptor = stage_job_descriptor(announcement, stage_index=0)

        class FakeModalInvoker:
            def __init__(self, cap: int):
                self.cap = cap
                self.active = 0
                self.queued = 0

            def spawn(self) -> None:
                if self.active >= self.cap:
                    self.queued += 1
                else:
                    self.active += 1

        invoker = FakeModalInvoker(cap=2)
        for _ in range(3):
            invoker.spawn()
        if (invoker.active, invoker.queued) != (2, 1):
            raise RuntimeError("bounded fake Modal dispatch failed")
        attempt_id = "smoke-attempt"
        raw_metrics = {"eval/full/episode/return/mean": 1.0}
        result = {
            "schema_version": PROTOCOL_SCHEMA_VERSION,
            "contract_schema_version": descriptor["contract"]["schema_version"],
            "attempt_id": attempt_id,
            "execution_key": descriptor["execution_key"],
            "checkpoint_sha256": announcement["sha256"],
            "runtime_image_ref": descriptor["contract"]["runtime_image_ref"],
            "rom_sha256": descriptor["contract"]["asset"]["sha256"],
            "seed_protocol": descriptor["contract"]["seed_protocol"],
            "n_envs": descriptor["contract"]["n_envs"],
            "episodes": descriptor["contract"]["episodes"],
            "status": "succeeded",
            "duration_seconds": 0.01,
            "metrics": raw_metrics,
            "episode_results": [
                {
                    "seed": 10_000,
                    "seed_protocol": SEED_PROTOCOL,
                    "seed_lane": lane,
                    "seed_episode_ordinal": 0,
                    "start_state": "Start",
                }
                for lane in range(2)
            ],
        }
        validate_attempt_result(result, contract=descriptor["contract"], attempt_id=attempt_id)
        passed, observed = apply_decision_rules(raw_metrics, descriptor["decision_rules"])
        decision_metrics = {
            "global_step": 10.0,
            staged_metric_name("screen", "eval/full/episode/return/mean"): 1.0,
            checkpoint_eval_stage_metric("screen", "pass"): 1.0,
        }
        decision = {
            "schema_version": PROTOCOL_SCHEMA_VERSION,
            "job_key": descriptor["job_key"],
            "execution_key": descriptor["execution_key"],
            "attempt_id": attempt_id,
            "train_job_id": 1,
            "ledger_id": checkpoint_id,
            "stage_name": "screen",
            "stage_index": 0,
            "purpose": "screen",
            "passed": passed,
            "candidate_stop": True,
            "observed_rules": observed,
            "metrics": decision_metrics,
            "raw_metrics": raw_metrics,
            "result_uri": object_store.uri("eval-attempts/smoke/smoke-attempt.json"),
        }
        object_store.put_json(
            f"eval-decisions/1/{descriptor['job_key']}.json", decision, create_only=True
        )
        if import_decisions(ledger, object_store, args) != 1:
            raise RuntimeError("Modal decision was not imported into the local ledger")
        if not write_complete_marker(ledger, object_store, args):
            raise RuntimeError("coordinator completion marker was not written")
        project_payload(
            {
                "train_config": {"wandb": False},
                "decision": decision,
                "purpose": "screen",
                "checkpoint_uri": announcement["model_uri"],
                "checkpoint_sha256": announcement["sha256"],
                "checkpoint_step": 10,
            }
        )
    print(
        json.dumps(
            {
                "modal_eval_local_smoke": "ok",
                "flow": [
                    "checkpoint_save",
                    "coordinator_upload",
                    "announcement_ingestion",
                    "bounded_dispatch",
                    "attempt_result",
                    "decision",
                    "early_stop_import",
                    "post_train_projection",
                ],
            }
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rlab eval modal")
    commands = parser.add_subparsers(dest="command", required=True)
    status = commands.add_parser("status")
    status.set_defaults(func=cmd_status)
    preflight = commands.add_parser("preflight")
    preflight.add_argument("--runtime-image-ref", required=True)
    preflight.add_argument("--game", required=True)
    preflight.add_argument("--env-provider", default="")
    preflight.set_defaults(func=cmd_preflight)
    drain = commands.add_parser("drain")
    drain.add_argument("--reason", default="")
    drain.set_defaults(func=cmd_drain)
    resume = commands.add_parser("resume")
    resume.add_argument("--reason", default="")
    resume.set_defaults(func=cmd_resume)
    retry = commands.add_parser("retry")
    retry.add_argument("--eval-job", dest="eval_job_id", type=int, required=True)
    retry.set_defaults(func=cmd_retry)
    recover = commands.add_parser("recover")
    recover.add_argument("--run", dest="train_job_id", type=int, required=True)
    recover.set_defaults(func=cmd_recover)
    abandon = commands.add_parser("abandon")
    abandon.add_argument("--run", dest="train_job_id", type=int, required=True)
    abandon.set_defaults(func=cmd_abandon)
    assets = commands.add_parser("assets")
    asset_commands = assets.add_subparsers(dest="asset_command", required=True)
    sync = asset_commands.add_parser("sync")
    sync.add_argument("--game", required=True)
    sync.add_argument("--rom-path", type=Path, default=None)
    sync.set_defaults(func=cmd_assets_sync)
    smoke = commands.add_parser("smoke-local")
    smoke.set_defaults(func=cmd_smoke_local)
    return parser


def main(argv: list[str] | None = None) -> int:
    load_env_file()
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
