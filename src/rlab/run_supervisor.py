from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Any

from rlab.checkpoint_acceptance import manifest_index
from rlab.dstack_backend import DSTACK_VERSION
from rlab.env import resolve_env_config
from rlab.env_config import env_config_from_args
from rlab.eval_metrics import eval_by_start_rows
from rlab.eval_backend import EvalBackend, EvalHandle
from rlab.file_utils import file_sha256
from rlab.metric_names import (
    EVAL_ACCEPTANCE_DURATION_SECONDS,
    EVAL_ACCEPTANCE_EPISODES_COMPLETED,
    EVAL_ACCEPTANCE_EPISODES_PLANNED,
    EVAL_ACCEPTANCE_FAILURE_COUNT,
    EVAL_ACCEPTANCE_PASS,
    METRICS_SCHEMA_VERSION,
    ORCHESTRATION_CHECKPOINT_BACKLOG,
    ORCHESTRATION_IDLE_GPU_TAIL_SECONDS,
    ORCHESTRATION_INGRESS_RATE,
    ORCHESTRATION_LOCAL_HIGH_WATER,
    ORCHESTRATION_OLDEST_UNPUBLISHED_SECONDS,
    ORCHESTRATION_PENDING_EVALS,
    ORCHESTRATION_PUBLICATION_CAPACITY_RATIO,
    ORCHESTRATION_PUBLISH_RATE,
    ORCHESTRATION_QUEUE_DEPTH,
    ORCHESTRATION_R2_HIGH_WATER,
    ORCHESTRATION_RESULT_TO_STOP_SECONDS,
    ORCHESTRATION_SCRATCH_USED_FRACTION,
    ORCHESTRATION_WANDB_HIGH_WATER,
    ORCHESTRATION_WANDB_REMOTE_HIGH_WATER,
    ORCHESTRATION_WANDB_REMOTE_VISIBLE_LAG_SECONDS,
)
from rlab.metric_store import MetricStore, metric_store_path
from rlab.model_sources import download_public_checkpoint_manifest_source
from rlab.modal_eval_backend import ModalEvalBackend
from rlab.modal_eval_config import load_modal_eval_config
from rlab.modal_eval_protocol import (
    PROTOCOL_SCHEMA_VERSION,
    execution_key,
    validate_attempt_result,
)
from rlab.policy_bundle import (
    build_recipe_document,
    canonical_json_sha256,
    evaluation_contract,
    evaluation_contract_sha256,
    write_canonical_json,
)
from rlab.r2_store import ConditionalWriteConflict, RunStorageConfig
from rlab.recipe_documents import (
    compose_train_document,
    prepare_checkpoint_eval_mode,
    recipe_tags,
)
from rlab.rom_assets import (
    CONTAINER_ROM_CACHE,
    cache_path,
    install_rom_file,
    validate_rom_asset_manifest,
    verify_rom_file,
)
from rlab.run_authority import (
    LEASE_MISSES_BEFORE_STOP,
    LEASE_RENEW_SECONDS,
    Lease,
    LeaseUnavailable,
    RunAuthority,
)
from rlab.run_contracts import (
    CheckpointManifest,
    EvalIntent,
    EvalResult,
    PromotionReceipt,
    RunManifest,
    TerminalReceipt,
    document_sha256,
    eval_idempotency_key,
    utc_now,
)
from rlab.runtime_contract import runtime_contract
from rlab.supervisor_ledger import SupervisorLedger
from rlab.train_config import (
    materialized_train_args,
    validate_and_normalize_train_config,
)
from rlab.trusted_inputs import stage_model_input
from rlab.wandb_publisher import (
    WandbProjector,
    publish_pending_frames,
    publish_promotion_summary,
)


METRIC_SEGMENT_SECONDS = 5.0
METRIC_SEGMENT_EVENTS = 1_000
EVAL_POLL_SECONDS = 2.0
WANDB_WARNING_SECONDS = 45.0
WANDB_UNHEALTHY_SECONDS = 60.0
WANDB_DRAIN_TIMEOUT_SECONDS = 300.0
SCRATCH_STOP_FRACTION = 0.80
METRIC_JOURNAL_RETENTION_DAYS = 7
HEALTH_SAMPLE_SECONDS = 15.0
WANDB_REMOTE_PROBE_SECONDS = 30.0
WANDB_DRAIN_REMOTE_PROBE_SECONDS = 2.0


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _timestamp(unix_seconds: float) -> str:
    return datetime.fromtimestamp(unix_seconds, UTC).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _manifest_from_document(value: Mapping[str, Any]) -> RunManifest:
    manifest = RunManifest(**dict(value))
    manifest.validate()
    return manifest


def _bind_evaluation_contract(
    config: dict[str, Any],
    *,
    recipe_document: Mapping[str, Any],
    evaluation_required: bool,
) -> dict[str, Any]:
    if evaluation_required:
        contract = evaluation_contract(recipe_document)
        config["checkpoint_eval_contract"] = contract
        return contract
    config.pop("checkpoint_eval_contract", None)
    return {}


class RunSupervisor:
    """Own all network-side effects for one learner container."""

    def __init__(
        self,
        *,
        manifest_uri: str,
        storage: RunStorageConfig | None = None,
        eval_backend: EvalBackend | None = None,
        repo_root: Path | None = None,
        work_root: Path | None = None,
    ):
        self.storage = storage or RunStorageConfig.from_env()
        self.authority = RunAuthority(self.storage)
        manifest_key = self.authority.control.key_from_uri(manifest_uri)
        self.manifest = _manifest_from_document(
            self.authority.control.get_json(manifest_key)
        )
        accepted_manifest_keys = {
            f"runs/{self.manifest.run_id}/manifest.json",
            f"runs/{self.manifest.run_id}/attempts/"
            f"{self.manifest.attempt_id}/manifest.json",
        }
        if manifest_key not in accepted_manifest_keys:
            raise ValueError(
                "run manifest is not at its canonical run or attempt key: "
                f"{manifest_key}"
            )
        self.repo_root = (
            repo_root
            or Path(os.environ.get("RLAB_PROJECT_ROOT") or Path(__file__).resolve().parents[2])
        ).resolve()
        self.work_root = (
            work_root
            or Path(os.environ.get("RLAB_RUN_WORK_ROOT") or "/workspace")
        ).resolve()
        self.output_root = self.work_root / "rlab" / self.manifest.run_id
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.run_dir = self.output_root / "runs" / self.manifest.run_id
        self.config_path = self.output_root / "train-config.json"
        self.recipe_path = self.output_root / "recipe.json"
        self.learner_log_path = self.output_root / "learner.log"
        self.cancel_requested = False
        self.lease_lost = False
        self.stop_reason = ""
        self.learner: subprocess.Popen[str] | None = None
        self.projector: WandbProjector | None = None
        self.lease: Lease | None = None
        self.last_lease_renewal = 0.0
        self.lease_misses = 0
        self.last_segment = 0.0
        self.last_eval_poll = 0.0
        self.last_health_sample = 0.0
        self.last_health_local_high_water = 0
        self.last_health_wandb_high_water = 0
        self.peak_ingress_rate = 0.0
        self.peak_publish_rate = 0.0
        self.peak_publish_capacity = 0.0
        self.last_remote_probe = 0.0
        self.wandb_remote_high_water = 0
        self.wandb_remote_visible_lag_seconds = 0.0
        self.accepted_observed_at: float | None = None
        self.recovery_mode = str(
            self.manifest.compute.get("recovery_mode") or "resume-training"
        )
        if self.recovery_mode not in {"resume-training", "drain-only"}:
            raise ValueError(f"unsupported recovery mode: {self.recovery_mode}")
        self.evaluation_required = bool(self.manifest.modal["enabled"])
        self.eval_backend = eval_backend
        if self.eval_backend is None and self.evaluation_required:
            self.eval_backend = ModalEvalBackend(
                app_name=str(self.manifest.modal["app_name"]),
                function_name=str(
                    self.manifest.modal.get("function_name") or "evaluate_checkpoint"
                ),
                environment_name=str(
                    self.manifest.modal.get("environment_name") or "rlab-eval"
                ),
            )
        self.metric_store = MetricStore(metric_store_path(self.run_dir))
        self.ledger = SupervisorLedger(metric_store_path(self.run_dir))
        self.train_config: dict[str, Any] = {}
        self.recipe_document: dict[str, Any] = {}
        self.eval_contract: dict[str, Any] = {}
        self.modal_config = load_modal_eval_config(
            self.repo_root / "experiments" / "modal_eval.yaml"
        )

    @staticmethod
    def _checkpoint_manifest_url(checkpoint: Mapping[str, Any]) -> str:
        model_url = str(checkpoint.get("public_url") or "")
        if not model_url.endswith("/model.zip"):
            raise ValueError("public checkpoint URL is malformed")
        return f"{model_url.removesuffix('/model.zip')}/manifest.json"

    def _configure_resume(self, config: dict[str, Any]) -> None:
        if self.recovery_mode != "resume-training":
            return
        index = self.authority.models.get_json_optional(
            f"runs/{self.manifest.run_id}/index.json"
        )
        checkpoints = [
            dict(row)
            for row in (index or {}).get("checkpoints") or []
            if isinstance(row, Mapping)
        ]
        if not checkpoints:
            return
        checkpoint = max(
            checkpoints,
            key=lambda row: (int(row["step"]), str(row["sha256"])),
        )
        manifest_url = self._checkpoint_manifest_url(checkpoint)
        resolved = download_public_checkpoint_manifest_source(
            manifest_url,
            root=self.output_root / ".resume-source",
        )
        staged = stage_model_input(
            resolved.model_path,
            source_identity=manifest_url,
        )
        try:
            backend = copy.deepcopy(dict(config["training_backend"]))
            backend_config = copy.deepcopy(dict(backend["config"]))
            backend_config["resume"] = manifest_url
            backend_config["resume_approval_hash"] = staged.manifest_hash
            backend_config["resume_manifest"] = [
                entry.as_dict() for entry in staged.manifest
            ]
            backend["config"] = backend_config
            config["training_backend"] = backend
        finally:
            staged.cleanup()

    def validate_runtime(self) -> None:
        runtime = runtime_contract(runtime_image_ref=self.manifest.image_digest)
        observed_source = str(runtime.get("runtime_build_source_sha") or "")
        expected_source = str(
            self.manifest.compute.get("runtime_build_source_sha") or ""
        )
        if observed_source != expected_source:
            raise RuntimeError(
                "runtime build source SHA does not match the immutable run manifest: "
                f"{observed_source or 'missing'} != {expected_source or 'missing'}"
            )
        observed_input = str(runtime.get("runtime_input_sha256") or "")
        expected_input = str(self.manifest.compute.get("runtime_input_sha256") or "")
        if observed_input != expected_input:
            raise RuntimeError(
                "runtime input SHA-256 does not match the immutable run manifest: "
                f"{observed_input or 'missing'} != {expected_input or 'missing'}"
            )
        if str(os.environ.get("RLAB_ORCHESTRATOR") or "") != "dstack":
            raise RuntimeError("run supervisor may execute only inside a dstack task")

    def materialize(self) -> None:
        goal_path = (
            self.repo_root / "experiments" / "goals" / self.manifest.goal_slug / "_goal.yaml"
        )
        recipe_path = goal_path.parent / "recipes" / f"{self.manifest.recipe_slug}.yaml"
        materialized = compose_train_document(
            goal_path,
            recipe_path,
            recipe_overrides=self.manifest.recipe_overrides,
            prepare_materialized=partial(
                prepare_checkpoint_eval_mode,
                checkpoint_eval_backend=(
                    "modal" if self.evaluation_required else "none"
                ),
            ),
        )
        materialized_goal_hash = str(
            materialized["train_config"]["effective_goal_contract_sha256"]
        )
        if materialized_goal_hash != self.manifest.goal_sha256:
            raise RuntimeError("effective goal hash does not match the run manifest")
        materialized_environment_hash = str(
            materialized.get("environment_hash") or ""
        ).removeprefix("sha256:")
        if materialized_environment_hash != self.manifest.environment_sha256:
            raise RuntimeError("materialized environment hash does not match the run manifest")

        config = dict(materialized["train_config"])
        asset = self.manifest.modal.get("rom_asset_manifest")
        if isinstance(asset, Mapping):
            normalized_asset = validate_rom_asset_manifest(
                asset,
                expected_game=str(config["game"]),
            )
            cached_rom = cache_path(CONTAINER_ROM_CACHE, normalized_asset)
            try:
                verify_rom_file(cached_rom, normalized_asset)
            except (FileNotFoundError, ValueError):
                if str(os.environ.get("RLAB_ROM_CACHE_READ_ONLY") or "") == "1":
                    raise
                object_key = self.authority.evaluation.key_from_uri(
                    str(normalized_asset["object_uri"])
                )
                with tempfile.TemporaryDirectory(
                    prefix="rlab-rom-",
                    dir=self.output_root,
                ) as temporary:
                    staged = Path(temporary) / str(normalized_asset["filename"])
                    staged.write_bytes(self.authority.evaluation.get_bytes(object_key))
                    install_rom_file(staged, normalized_asset, CONTAINER_ROM_CACHE)
            config["rom_asset_manifest"] = normalized_asset
        materialized["train_config"] = config
        self.recipe_document = build_recipe_document(
            materialized,
            repo_root=self.repo_root,
            source_commit=self.manifest.source_sha,
            run_description=self.manifest.run_description,
            seed=self.manifest.seed,
            runtime_image_ref=self.manifest.image_digest,
        )
        if canonical_json_sha256(self.recipe_document) != self.manifest.recipe_sha256:
            raise RuntimeError("portable recipe hash does not match the run manifest")
        config.update(
            {
                "seed": int(self.manifest.seed),
                "run_name": self.manifest.run_id,
                "run_description": self.manifest.run_description,
                "runs_dir": str(self.output_root / "runs"),
                "goal_slug": self.manifest.goal_slug,
                "goal_path": str(goal_path),
                "goal_sha256": self.manifest.goal_sha256,
                "recipe_slug": self.manifest.recipe_slug,
                "recipe_path": str(recipe_path),
                "recipe_sha256": self.manifest.recipe_sha256,
                "source_sha": self.manifest.source_sha,
                "runtime_build_source_sha": str(
                    self.manifest.compute["runtime_build_source_sha"]
                ),
                "runtime_input_sha256": str(
                    os.environ.get("RLAB_RUNTIME_INPUT_SHA256") or ""
                ),
                "runtime_image_ref": self.manifest.image_digest,
                "compute_target": str(
                    dict(
                        self.manifest.compute.get("selected")
                        or self.manifest.compute.get("request")
                        or {}
                    ).get("target")
                    or ""
                ),
                "attempt_id": self.manifest.attempt_id,
                "dstack_task": str(self.manifest.compute.get("dstack_task") or ""),
                "wandb": True,
                "wandb_mode": "online",
                "wandb_run_id": str(
                    self.manifest.wandb.get("run_id") or self.manifest.run_id
                ),
                "wandb_entity": str(self.manifest.wandb.get("entity") or ""),
                "wandb_project": str(self.manifest.wandb.get("project") or ""),
                "wandb_group": str(self.manifest.wandb.get("group") or ""),
                "wandb_tags": ",".join(
                    [
                        *recipe_tags(materialized),
                        f"rlab_run_id:{self.manifest.run_id}",
                        f"attempt_id:{self.manifest.attempt_id}",
                        "orchestrator:dstack",
                    ]
                ),
                "checkpoint_eval_backend": (
                    "modal" if self.evaluation_required else "none"
                ),
                "metrics_schema_version": METRICS_SCHEMA_VERSION,
            }
        )
        materialized["train_config"] = config
        write_canonical_json(self.recipe_path, self.recipe_document)
        self.eval_contract = _bind_evaluation_contract(
            config,
            recipe_document=self.recipe_document,
            evaluation_required=self.evaluation_required,
        )
        config["recipe_json_path"] = str(self.recipe_path)
        config["recipe_composition"] = dict(materialized.get("_composition") or {})
        self._configure_resume(config)
        self.train_config = validate_and_normalize_train_config(
            config,
            label="dstack run train_config",
            required_keys=("training_backend",),
        )
        write_canonical_json(self.config_path, self.train_config)

    def _start_wandb(self) -> None:
        args = materialized_train_args(self.config_path)
        config = resolve_env_config(env_config_from_args(args, include_states=True))
        receipt_key = f"runs/{self.manifest.run_id}/wandb.json"
        existing = self.authority.control.get_json_optional(receipt_key)
        if existing is None:
            self.projector = WandbProjector.start_live(
                args,
                run_dir=str(self.run_dir),
                config=config,
            )
            run = self.projector.run
            receipt = {
                "schema_version": 1,
                "run_id": self.manifest.run_id,
                "wandb_run_id": str(getattr(run, "id", "") or ""),
                "url": str(getattr(run, "url", "") or ""),
                "created_at": utc_now(),
            }
            self.authority.control.put_json(receipt_key, receipt, create_only=True)
        else:
            self.projector = WandbProjector.resume(
                self.train_config,
                allow_create=False,
            )

    def _start_learner(self) -> None:
        environment = os.environ.copy()
        for name in tuple(environment):
            if (
                name in {"WANDB_API_KEY", "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"}
                or name.startswith("RLAB_CONTROL_R2_")
                or name.startswith("RLAB_EVAL_R2_")
                or name.startswith("RLAB_MODELS_R2_")
                or name.startswith("AWS_")
            ):
                environment.pop(name, None)
        environment["RLAB_INTERNAL_LEARNER"] = "1"
        environment["RLAB_ROM_CACHE_DIR"] = str(CONTAINER_ROM_CACHE)
        command = [
            sys.executable,
            "-m",
            "rlab.train",
            "--train-config-json",
            str(self.config_path),
        ]
        log = self.learner_log_path.open("a", encoding="utf-8")
        self.learner = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=environment,
        )
        self.learner._rlab_log = log  # type: ignore[attr-defined]
        print(
            f"learner started pid={self.learner.pid} log={self.learner_log_path}",
            flush=True,
        )

    def _recover_durable_state(self) -> None:
        prefix = f"runs/{self.manifest.run_id}"
        control_keys = list(self.authority.control.iter_keys(f"{prefix}/attempts"))
        expiring_journal_keys = list(
            self.authority.control.iter_keys(
                f"expiring-metric-journals/{self.manifest.run_id}"
            )
        )
        attempts = []
        for key in control_keys:
            if not key.endswith("/manifest.json"):
                continue
            document = self.authority.control.get_json(key)
            attempts.append(
                (
                    str(document.get("created_at") or ""),
                    str(document.get("attempt_id") or ""),
                )
            )
        attempt_order = {
            attempt_id: index
            for index, (_created_at, attempt_id) in enumerate(sorted(attempts))
        }
        segment_keys = [
            key
            for key in control_keys
            if "/metric-segments/" in key and key.endswith(".jsonl")
        ]
        segment_keys.extend(
            key for key in expiring_journal_keys if key.endswith(".jsonl")
        )

        def segment_attempt_id(key: str) -> str:
            if "/attempts/" in key:
                return key.split("/attempts/", 1)[1].split("/", 1)[0]
            return key.split(
                f"expiring-metric-journals/{self.manifest.run_id}/",
                1,
            )[1].split("/", 1)[0]

        segment_keys.sort(
            key=lambda key: (
                attempt_order.get(segment_attempt_id(key), 1_000_000),
                key,
            )
        )
        recovered_events = 0
        for key in segment_keys:
            for encoded in self.authority.control.get_bytes(key).splitlines():
                if not encoded:
                    continue
                event = json.loads(encoded)
                if not isinstance(event, Mapping):
                    raise ValueError(f"metric journal event is not a mapping: {key}")
                self.metric_store.enqueue_event(
                    kind=str(event["kind"]),
                    payload=dict(event["payload"]),
                    step=(
                        None
                        if event.get("step") is None
                        else int(event["step"])
                    ),
                    source=str(event["source"]),
                    event_id=str(event["event_id"]),
                    created_at=float(event["created_at"]),
                )
                recovered_events += 1

        index = self.authority.models.get_json_optional(f"{prefix}/index.json")
        checkpoints = [
            dict(row)
            for row in (index or {}).get("checkpoints") or []
            if isinstance(row, Mapping)
        ]
        checkpoints.sort(key=lambda row: (int(row["step"]), str(row["sha256"])))
        for position, document in enumerate(checkpoints, start=1):
            checkpoint = CheckpointManifest(**document)
            checkpoint.validate()
            ledger_id = -position
            self.ledger.record_checkpoint_publication(
                checkpoint_ledger_id=ledger_id,
                manifest=checkpoint.to_dict(),
            )
            self._ensure_eval(ledger_id, checkpoint)

        for initial in self.ledger.evals(statuses=("pending",)):
            key = str(initial["idempotency_key"])
            selected_attempt = 0
            selected_prepared: Mapping[str, Any] | None = None
            selected_dispatch: Mapping[str, Any] | None = None
            for attempt in range(1, int(self.modal_config.max_attempts) + 1):
                prepared = self.authority.eval_attempt(
                    run_id=self.manifest.run_id,
                    idempotency_key=key,
                    attempt=attempt,
                )
                if prepared is None:
                    continue
                selected_attempt = attempt
                selected_prepared = prepared
                selected_dispatch = self.authority.eval_dispatch(
                    run_id=self.manifest.run_id,
                    idempotency_key=key,
                    attempt=attempt,
                )
            if selected_prepared is not None:
                self.ledger.mark_eval_submitted(
                    idempotency_key=key,
                    attempt=selected_attempt,
                    modal_call_id=str((selected_dispatch or {}).get("modal_call_id") or ""),
                    attempt_expires_at=float(selected_prepared["expires_at"]),
                )

            verified = self.authority.evaluation.get_json_optional(
                f"{prefix}/evals/{key}/verified-result.json"
            )
            raw = self.authority.eval_result(
                run_id=self.manifest.run_id,
                idempotency_key=key,
            )
            if verified is not None:
                result = EvalResult(**verified)
                result.validate()
                self.ledger.mark_eval_terminal(
                    idempotency_key=key,
                    status=result.status,
                    result=result.to_dict(),
                )
                if raw is not None:
                    self._record_eval_metrics(
                        self.ledger.eval(key) or initial,
                        result,
                        raw,
                    )
                if result.status == "accepted":
                    self.stop_reason = "eval_acceptance"
                continue
            if raw is not None:
                row = self.ledger.eval(key)
                if row is None:
                    raise RuntimeError(f"recovered eval disappeared from the ledger: {key}")
                if int(row["attempt"] or 0) == 0:
                    raw_attempt_id = str(raw.get("attempt_id") or "")
                    try:
                        attempt = int(raw_attempt_id.rsplit("-a", 1)[1])
                    except (IndexError, ValueError) as exc:
                        raise ValueError("raw eval result has no recoverable attempt") from exc
                    self.ledger.mark_eval_submitted(
                        idempotency_key=key,
                        attempt=attempt,
                        modal_call_id="",
                        attempt_expires_at=0.0,
                    )
                    row = self.ledger.eval(key)
                    assert row is not None
                self._observe_result(row)
        if recovered_events or checkpoints:
            print(
                f"recovered durable state events={recovered_events} "
                f"checkpoints={len(checkpoints)} evals={len(self.ledger.evals())}",
                flush=True,
            )

    def _request_learner_stop(self, reason: str) -> None:
        if not self.stop_reason:
            self.stop_reason = reason
        if self.learner is not None and self.learner.poll() is None:
            graceful = getattr(signal, "SIGUSR1", signal.SIGTERM)
            self.learner.send_signal(graceful)
            print(f"learner stop requested: reason={reason}", flush=True)

    def _reconcile_verified_eval_result(
        self,
        row: Mapping[str, Any],
    ) -> bool:
        document = self.authority.evaluation.get_json_optional(
            f"runs/{self.manifest.run_id}/evals/"
            f"{row['idempotency_key']}/verified-result.json"
        )
        if document is None:
            return False
        result = EvalResult(**document)
        result.validate()
        self.ledger.mark_eval_terminal(
            idempotency_key=result.idempotency_key,
            status=result.status,
            result=result.to_dict(),
        )
        return True

    def _cancel_outstanding_evals(self) -> None:
        for row in self.ledger.evals(statuses=("pending", "submitted")):
            if self._observe_result(row) or self._reconcile_verified_eval_result(row):
                continue
            call_id = str(row.get("modal_call_id") or "")
            if call_id:
                try:
                    assert self.eval_backend is not None
                    self.eval_backend.cancel(
                        EvalHandle(provider="modal", call_id=call_id)
                    )
                except Exception as exc:
                    print(f"Modal cancel failed call={call_id}: {exc}", flush=True)
            result = EvalResult(
                run_id=self.manifest.run_id,
                checkpoint_id=str(row["checkpoint_id"]),
                idempotency_key=str(row["idempotency_key"]),
                modal_call_id=call_id or "not-submitted",
                status="canceled",
                episode_results=[],
                aggregates={},
                timings={"canceled_at": utc_now()},
                evidence_sha256=[],
                completed_at=utc_now(),
                error="run canceled",
            )
            try:
                self.authority.put_verified_eval_result(result)
            except ConditionalWriteConflict:
                if not self._reconcile_verified_eval_result(row):
                    raise
                continue
            self.ledger.mark_eval_terminal(
                idempotency_key=result.idempotency_key,
                status=result.status,
                result=result.to_dict(),
            )

    def _renew_lease(self, now: float) -> None:
        if now - self.last_lease_renewal < LEASE_RENEW_SECONDS:
            return
        assert self.lease is not None
        try:
            self.lease = self.authority.renew_lease(self.lease)
        except Exception as exc:
            self.lease_misses += 1
            print(
                f"writer lease renewal failed ({self.lease_misses}/"
                f"{LEASE_MISSES_BEFORE_STOP}): {exc}",
                flush=True,
            )
            if self.lease_misses >= LEASE_MISSES_BEFORE_STOP:
                self.lease_lost = True
                self._request_learner_stop("writer_lease_lost")
        else:
            self.lease_misses = 0
            self.last_lease_renewal = now

    def _seal_metrics(self, now: float, *, force: bool = False) -> int:
        events = self.ledger.next_metric_events(limit=METRIC_SEGMENT_EVENTS)
        if not events:
            self.last_segment = now
            return 0
        if (
            not force
            and len(events) < METRIC_SEGMENT_EVENTS
            and now - self.last_segment < METRIC_SEGMENT_SECONDS
        ):
            return 0
        key, digest = self.authority.seal_metric_segment(
            run_id=self.manifest.run_id,
            attempt_id=self.manifest.attempt_id,
            events=events,
        )
        self.ledger.record_metric_segment(
            events=events,
            object_key=key,
            sha256=digest,
        )
        self.last_segment = now
        return len(events)

    def _publish_wandb(self) -> int:
        if self.projector is None:
            return 0
        args = materialized_train_args(self.config_path)
        config = resolve_env_config(env_config_from_args(args, include_states=True))
        started = time.monotonic()
        published = publish_pending_frames(
            self.metric_store,
            self.projector.run,
            args=args,
            config=config,
            limit=250,
        )
        elapsed = max(time.monotonic() - started, 1e-6)
        if published:
            self.peak_publish_capacity = max(
                self.peak_publish_capacity,
                published / elapsed,
            )
        return published

    def _publish_checkpoints(self) -> int:
        published = 0
        contract_hashes = {
            "goal_sha256": self.manifest.goal_sha256,
            "recipe_sha256": self.manifest.recipe_sha256,
            "environment_sha256": self.manifest.environment_sha256,
            "evaluation_contract_sha256": (
                evaluation_contract_sha256(self.recipe_document)
                if self.evaluation_required
                else canonical_json_sha256(
                    {
                        "training_only": True,
                        "playback": self.recipe_document["recipe"]["playback"],
                    }
                )
            ),
        }
        for checkpoint in self.metric_store.checkpoints():
            ledger_id = int(checkpoint["id"])
            if self.ledger.checkpoint_publication(ledger_id) is not None:
                continue
            path = Path(str(checkpoint["path"]))
            if not path.is_file():
                continue
            kind = str(checkpoint["kind"])
            purpose = "final" if kind in {"final", "interrupted"} else "periodic"
            try:
                manifest = self.authority.publish_checkpoint(
                    run_id=self.manifest.run_id,
                    model_path=path,
                    step=int(checkpoint["step"] or 0),
                    purpose=purpose,
                    contract_hashes=contract_hashes,
                    recovery_sidecar={
                        "schema_version": 1,
                        "run_id": self.manifest.run_id,
                        "attempt_id": self.manifest.attempt_id,
                        "checkpoint_ledger_id": ledger_id,
                        "kind": kind,
                        "local_path": str(path),
                    },
                    created_at=_timestamp(float(checkpoint["created_at"])),
                )
            except Exception as exc:
                self.metric_store.mark_checkpoint_upload_failed(ledger_id, repr(exc))
                print(f"checkpoint publication failed id={ledger_id}: {exc}", flush=True)
                continue
            existing = self.ledger.checkpoint_publication_by_id(
                manifest.checkpoint_id
            )
            if existing is not None:
                self.metric_store.mark_checkpoint_uploaded(
                    ledger_id,
                    manifest.public_url,
                )
                continue
            self.ledger.record_checkpoint_publication(
                checkpoint_ledger_id=ledger_id,
                manifest=manifest.to_dict(),
            )
            self.metric_store.mark_checkpoint_uploaded(
                ledger_id,
                manifest.public_url,
            )
            if bool(checkpoint["eval_required"]):
                self._ensure_eval(ledger_id, manifest)
            print(
                f"checkpoint published id={manifest.checkpoint_id} "
                f"step={manifest.step} url={manifest.public_url}",
                flush=True,
            )
            published += 1
        return published

    def _execution_contract(self, checkpoint: CheckpointManifest) -> dict[str, Any]:
        contract = dict(self.eval_contract)
        contract.update(
            {
                "schema_version": PROTOCOL_SCHEMA_VERSION,
                "checkpoint_sha256": checkpoint.sha256,
                "runtime_image_ref": self.manifest.image_digest,
                "recipe_sha256": checkpoint.recipe_document_sha256,
                "recipe_format_version": int(self.recipe_document["format_version"]),
                "evaluation_contract_sha256": checkpoint.evaluation_contract_sha256,
            }
        )
        asset = contract.get("asset")
        if isinstance(asset, Mapping):
            contract["asset"] = {
                str(key): value
                for key, value in asset.items()
                if str(key) != "object_uri"
            }
        manifest_index(contract)
        return contract

    def _ensure_eval(
        self,
        checkpoint_ledger_id: int,
        checkpoint: CheckpointManifest,
    ) -> None:
        contract = self._execution_contract(checkpoint)
        episode_manifest_sha = document_sha256(contract["manifest"])
        key = eval_idempotency_key(
            run_id=self.manifest.run_id,
            checkpoint_sha256=checkpoint.sha256,
            evaluation_contract_sha256=checkpoint.evaluation_contract_sha256,
            episode_manifest_sha256=episode_manifest_sha,
            protocol="modal-acceptance-v3",
        )
        timeout = int(self.modal_config.acceptance_timeout_seconds)
        created = _parse_timestamp(checkpoint.created_at)
        result_key = f"runs/{self.manifest.run_id}/evals/{key}/result.json"
        intent = EvalIntent(
            run_id=self.manifest.run_id,
            checkpoint_id=checkpoint.checkpoint_id,
            idempotency_key=key,
            checkpoint_sha256=checkpoint.sha256,
            goal_sha256=self.manifest.goal_sha256,
            recipe_sha256=self.manifest.recipe_sha256,
            environment_sha256=self.manifest.environment_sha256,
            evaluation_contract_sha256=checkpoint.evaluation_contract_sha256,
            episode_manifest_sha256=episode_manifest_sha,
            protocol="modal-acceptance-v3",
            execution_contract=contract,
            result_key=result_key,
            timeout_seconds=timeout,
            created_at=created.isoformat().replace("+00:00", "Z"),
            expires_at=(created + timedelta(seconds=timeout))
            .isoformat()
            .replace("+00:00", "Z"),
        )
        self.authority.put_eval_intent(intent)
        self.ledger.ensure_eval(
            checkpoint_ledger_id=checkpoint_ledger_id,
            intent={
                **intent.to_dict(),
                "checkpoint_step": checkpoint.step,
                "checkpoint": checkpoint.to_dict(),
            },
        )

    def _eval_payload(
        self,
        row: Mapping[str, Any],
        *,
        attempt: int,
        expires_at: float,
    ) -> dict[str, Any]:
        intent = dict(row["intent"])
        checkpoint = dict(intent["checkpoint"])
        contract = dict(intent["execution_contract"])
        timeout = int(intent["timeout_seconds"])
        result_key = str(intent["result_key"])
        attempt_id = f"{intent['idempotency_key'][:20]}-a{attempt}"
        payload: dict[str, Any] = {
            "attempt_id": attempt_id,
            "contract": contract,
            "expires_at": expires_at,
            "child_timeout_seconds": max(
                1,
                timeout - int(self.modal_config.child_margin_seconds),
            ),
            "model_get_url": str(checkpoint["public_url"]),
            "model_document_get_url": str(checkpoint["model_document_url"]),
            "model_document_sha256": str(checkpoint["model_document_sha256"]),
            "recipe_get_url": str(checkpoint["recipe_document_url"]),
            "result_uri": self.authority.evaluation.uri(result_key),
            "result_put_url": self.authority.evaluation.presign_put(
                result_key,
                expires_seconds=timeout + int(self.modal_config.expiry_margin_seconds),
            ),
        }
        asset = self.manifest.modal.get("rom_asset_manifest")
        if isinstance(asset, Mapping):
            rom_key = self.authority.evaluation.key_from_uri(str(asset["object_uri"]))
            payload["rom_get_url"] = self.authority.evaluation.presign_get(
                rom_key,
                expires_seconds=timeout + int(self.modal_config.expiry_margin_seconds),
            )
        return payload

    def _submit_pending_evals(self) -> int:
        submitted = 0
        for row in self.ledger.evals(statuses=("pending",)):
            attempt = int(row["attempt"] or 0) + 1
            if attempt > int(self.modal_config.max_attempts):
                self._mark_expired(row, error="eval exhausted two attempts")
                continue
            prepared = self.authority.eval_attempt(
                run_id=self.manifest.run_id,
                idempotency_key=str(row["idempotency_key"]),
                attempt=attempt,
            )
            if prepared is not None:
                dispatch = self.authority.eval_dispatch(
                    run_id=self.manifest.run_id,
                    idempotency_key=str(row["idempotency_key"]),
                    attempt=attempt,
                )
                self.ledger.mark_eval_submitted(
                    idempotency_key=str(row["idempotency_key"]),
                    attempt=attempt,
                    modal_call_id=str((dispatch or {}).get("modal_call_id") or ""),
                    attempt_expires_at=float(prepared["expires_at"]),
                )
                continue
            expires_at = time.time() + int(row["intent"]["timeout_seconds"])
            self.authority.prepare_eval_attempt(
                run_id=self.manifest.run_id,
                idempotency_key=str(row["idempotency_key"]),
                attempt=attempt,
                expires_at=expires_at,
            )
            payload = self._eval_payload(
                row,
                attempt=attempt,
                expires_at=expires_at,
            )
            try:
                assert self.eval_backend is not None
                handle = self.eval_backend.submit(payload)
            except Exception as exc:
                self.ledger.mark_eval_submitted(
                    idempotency_key=str(row["idempotency_key"]),
                    attempt=attempt,
                    modal_call_id="",
                    attempt_expires_at=expires_at,
                )
                self.ledger.record_eval_error(
                    idempotency_key=str(row["idempotency_key"]),
                    error=f"ambiguous submit: {exc!r}",
                )
                print(
                    f"Modal spawn ambiguous key={row['idempotency_key']}: {exc}",
                    flush=True,
                )
                continue
            self.authority.put_eval_dispatch(
                run_id=self.manifest.run_id,
                idempotency_key=str(row["idempotency_key"]),
                attempt=attempt,
                modal_call_id=handle.call_id,
            )
            self.ledger.mark_eval_submitted(
                idempotency_key=str(row["idempotency_key"]),
                attempt=attempt,
                modal_call_id=handle.call_id,
                attempt_expires_at=expires_at,
            )
            print(
                f"Modal eval submitted checkpoint={row['checkpoint_id']} "
                f"call={handle.call_id} attempt={attempt}",
                flush=True,
            )
            submitted += 1
        return submitted

    def _verified_result(
        self,
        row: Mapping[str, Any],
        raw: Mapping[str, Any],
    ) -> EvalResult:
        intent = dict(row["intent"])
        attempt = int(row["attempt"])
        attempt_id = f"{intent['idempotency_key'][:20]}-a{attempt}"
        contract = dict(intent["execution_contract"])
        raw_status = str(raw.get("status") or "")
        if raw_status == "succeeded":
            validated = validate_attempt_result(
                raw,
                contract=contract,
                attempt_id=attempt_id,
            )
            verdict = str(validated.get("verdict") or "")
            status = "accepted" if verdict == "accepted" else "rejected"
            episodes = list(validated.get("episode_results") or [])
            aggregates = dict(validated.get("claimed_aggregates") or {})
            error = None
        else:
            if str(raw.get("attempt_id") or "") != attempt_id:
                raise ValueError("failed eval result attempt id mismatch")
            if str(raw.get("execution_key") or "") != execution_key(contract):
                raise ValueError("failed eval result execution key mismatch")
            status = "expired" if raw_status == "expired" else "failed"
            episodes = list(raw.get("episode_results") or [])
            aggregates = dict(raw.get("claimed_aggregates") or {})
            error = str(raw.get("error") or f"Modal eval status={raw_status or 'unknown'}")
        evidence_values = [
            episodes,
            raw.get("evaluation_evidence") or {},
            raw.get("preview") or {},
        ]
        evidence_hashes = [
            document_sha256({"evidence": value})
            for value in evidence_values
            if value not in (None, {}, [])
        ]
        return EvalResult(
            run_id=self.manifest.run_id,
            checkpoint_id=str(row["checkpoint_id"]),
            idempotency_key=str(row["idempotency_key"]),
            modal_call_id=str(row["modal_call_id"] or ""),
            status=status,  # type: ignore[arg-type]
            episode_results=episodes,
            aggregates=aggregates,
            timings={
                "duration_seconds": float(raw.get("duration_seconds") or 0.0),
                "result_observed_at": utc_now(),
            },
            evidence_sha256=evidence_hashes,
            completed_at=utc_now(),
            error=error,
        )

    def _record_eval_metrics(
        self,
        row: Mapping[str, Any],
        result: EvalResult,
        raw: Mapping[str, Any],
    ) -> None:
        metrics = {
            str(name): value
            for name, value in dict(raw.get("metrics") or {}).items()
        }
        metrics.update(
            {
                EVAL_ACCEPTANCE_PASS: 1.0 if result.status == "accepted" else 0.0,
                EVAL_ACCEPTANCE_EPISODES_PLANNED: float(
                    row["intent"]["execution_contract"]["episodes"]
                ),
                EVAL_ACCEPTANCE_EPISODES_COMPLETED: float(
                    len(result.episode_results)
                ),
                EVAL_ACCEPTANCE_FAILURE_COUNT: float(
                    result.aggregates.get("failure_count") or 0
                ),
                EVAL_ACCEPTANCE_DURATION_SECONDS: float(
                    raw.get("duration_seconds") or 0.0
                ),
            }
        )
        self.metric_store.append_metrics(
            metrics,
            step=int(row["checkpoint_step"]),
            source=f"eval:{row['idempotency_key']}",
        )
        if result.status == "accepted":
            self.metric_store.enqueue_event(
                kind="eval_by_start",
                payload={
                    "rows": eval_by_start_rows(
                        [dict(episode) for episode in result.episode_results]
                    )
                },
                step=int(row["checkpoint_step"]),
                source=f"eval:{row['idempotency_key']}:by-start",
            )

    def _observe_result(self, row: Mapping[str, Any]) -> bool:
        raw = self.authority.eval_result(
            run_id=self.manifest.run_id,
            idempotency_key=str(row["idempotency_key"]),
        )
        if raw is None:
            return False
        try:
            result = self._verified_result(row, raw)
        except Exception as exc:
            self.ledger.record_eval_error(
                idempotency_key=str(row["idempotency_key"]),
                error=f"invalid result: {exc!r}",
            )
            print(f"invalid Modal result ignored key={row['idempotency_key']}: {exc}", flush=True)
            return False
        self.authority.put_verified_eval_result(result)
        self.ledger.mark_eval_terminal(
            idempotency_key=result.idempotency_key,
            status=result.status,
            result=result.to_dict(),
        )
        self._record_eval_metrics(row, result, raw)
        print(
            f"Modal eval terminal checkpoint={result.checkpoint_id} status={result.status}",
            flush=True,
        )
        if result.status == "accepted":
            observed = time.time()
            self.accepted_observed_at = self.accepted_observed_at or observed
            self._request_learner_stop("eval_acceptance")
            signal_sent = time.time()
            requested = self.ledger.mark_stop_requested(
                idempotency_key=result.idempotency_key
            )
            result_to_stop = signal_sent - observed
            self.metric_store.append_metrics(
                {ORCHESTRATION_RESULT_TO_STOP_SECONDS: result_to_stop},
                step=int(row["checkpoint_step"]),
                source=f"orchestration:stop:{result.idempotency_key}",
            )
            if result_to_stop > 10.0 or requested - observed > 10.0:
                raise RuntimeError("accepted eval did not issue stop within ten seconds")
        return True

    def _mark_expired(self, row: Mapping[str, Any], *, error: str) -> None:
        result = EvalResult(
            run_id=self.manifest.run_id,
            checkpoint_id=str(row["checkpoint_id"]),
            idempotency_key=str(row["idempotency_key"]),
            modal_call_id=str(row.get("modal_call_id") or "not-submitted"),
            status="expired",
            episode_results=[],
            aggregates={},
            timings={"result_observed_at": utc_now()},
            evidence_sha256=[],
            completed_at=utc_now(),
            error=error,
        )
        self.authority.put_verified_eval_result(result)
        self.ledger.mark_eval_terminal(
            idempotency_key=result.idempotency_key,
            status=result.status,
            result=result.to_dict(),
        )

    def _poll_evals(self, now: float) -> int:
        if now - self.last_eval_poll < EVAL_POLL_SECONDS:
            return 0
        self.last_eval_poll = now
        wall_now = time.time()
        completed = 0
        for row in self.ledger.evals(statuses=("submitted",)):
            if self._observe_result(row):
                completed += 1
                continue
            call_id = str(row["modal_call_id"] or "")
            if call_id:
                handle = EvalHandle(provider="modal", call_id=call_id)
                assert self.eval_backend is not None
                poll = self.eval_backend.poll(handle)
                if poll.status == "failed" and poll.error:
                    self.ledger.record_eval_error(
                        idempotency_key=str(row["idempotency_key"]),
                        error=poll.error,
                    )
            expires_at = float(row.get("attempt_expires_at") or 0)
            if expires_at > wall_now:
                continue
            if int(row["attempt"] or 0) < int(self.modal_config.max_attempts):
                self.ledger.reset_expired_eval(
                    idempotency_key=str(row["idempotency_key"]),
                    error="attempt expired without a valid result",
                )
            else:
                self._mark_expired(
                    row,
                    error="eval expired twice without a valid result",
                )
                completed += 1
        return completed

    def _scratch_guard(self) -> None:
        usage = shutil.disk_usage(self.output_root)
        fraction = usage.used / max(usage.total, 1)
        if fraction >= SCRATCH_STOP_FRACTION:
            self._request_learner_stop("scratch_storage_above_80_percent")
            raise RuntimeError(
                f"scratch storage is {fraction:.1%} full; stopped before evidence loss"
            )

    def _frame_high_waters(self) -> tuple[int, int]:
        with self.ledger.connection() as connection:
            row = connection.execute(
                """
                SELECT COALESCE(MAX(id), 0) AS local_high_water,
                       COALESCE(MAX(id) FILTER (WHERE status = 'published'), 0)
                         AS wandb_high_water
                FROM metric_frames
                """
            ).fetchone()
        return int(row["local_high_water"]), int(row["wandb_high_water"])

    def _probe_wandb_remote(
        self,
        now: float,
        *,
        local_high_water: int,
        force: bool = False,
    ) -> None:
        minimum_interval = (
            WANDB_DRAIN_REMOTE_PROBE_SECONDS if force else WANDB_REMOTE_PROBE_SECONDS
        )
        if self.projector is None or now - self.last_remote_probe < minimum_interval:
            return
        self.last_remote_probe = now
        try:
            import wandb

            path = getattr(self.projector.run, "path", "")
            run_path = (
                "/".join(str(part) for part in path)
                if isinstance(path, list | tuple)
                else str(path)
            )
            api = wandb.Api(timeout=10)
            flush = getattr(api, "flush", None)
            if callable(flush):
                flush()
            remote = api.run(run_path)
            summary_value = dict(getattr(remote, "summary", {}) or {}).get(
                "orchestration/event_seq"
            )
            if isinstance(summary_value, Mapping):
                summary_value = summary_value.get("max") or summary_value.get("last")
            remote_high_water = int(summary_value or 0)
            self.wandb_remote_high_water = max(
                self.wandb_remote_high_water,
                remote_high_water,
            )
            with self.ledger.connection() as connection:
                unseen = connection.execute(
                    "SELECT MIN(created_at) FROM metric_frames WHERE id > ?",
                    (self.wandb_remote_high_water,),
                ).fetchone()
            oldest_unseen = unseen[0] if unseen is not None else None
            self.wandb_remote_visible_lag_seconds = (
                0.0
                if oldest_unseen is None or self.wandb_remote_high_water >= local_high_water
                else max(0.0, time.time() - float(oldest_unseen))
            )
        except Exception as exc:
            self.ledger.set_state(
                "wandb_remote_probe_error",
                {"error": repr(exc)[:1000], "at": utc_now()},
            )

    def _emit_health(self, now: float) -> None:
        if now - self.last_health_sample < HEALTH_SAMPLE_SECONDS:
            return
        interval = (
            HEALTH_SAMPLE_SECONDS
            if self.last_health_sample == 0.0
            else max(now - self.last_health_sample, 0.001)
        )
        local_high_water, wandb_high_water = self._frame_high_waters()
        ingress_rate = max(
            0.0,
            (local_high_water - self.last_health_local_high_water) / interval,
        )
        publish_rate = max(
            0.0,
            (wandb_high_water - self.last_health_wandb_high_water) / interval,
        )
        self.peak_ingress_rate = max(self.peak_ingress_rate, ingress_rate)
        self.peak_publish_rate = max(self.peak_publish_rate, publish_rate)
        capacity_ratio = (
            self.peak_publish_capacity / self.peak_ingress_rate
            if self.peak_ingress_rate > 0.0
            else 0.0
        )
        self._probe_wandb_remote(now, local_high_water=local_high_water)
        usage = shutil.disk_usage(self.output_root)
        metrics = {
            ORCHESTRATION_QUEUE_DEPTH: float(
                self.metric_store.metric_outbox_stats()["frames"]
            ),
            ORCHESTRATION_OLDEST_UNPUBLISHED_SECONDS: self._oldest_unpublished_age(),
            ORCHESTRATION_INGRESS_RATE: ingress_rate,
            ORCHESTRATION_PUBLISH_RATE: publish_rate,
            ORCHESTRATION_PUBLICATION_CAPACITY_RATIO: capacity_ratio,
            ORCHESTRATION_LOCAL_HIGH_WATER: float(local_high_water),
            ORCHESTRATION_R2_HIGH_WATER: float(
                self.ledger.metric_segment_high_water()
            ),
            ORCHESTRATION_WANDB_HIGH_WATER: float(wandb_high_water),
            ORCHESTRATION_WANDB_REMOTE_HIGH_WATER: float(
                self.wandb_remote_high_water
            ),
            ORCHESTRATION_WANDB_REMOTE_VISIBLE_LAG_SECONDS: (
                self.wandb_remote_visible_lag_seconds
            ),
            ORCHESTRATION_CHECKPOINT_BACKLOG: float(
                len(self.metric_store.checkpoints())
                - len(self.ledger.checkpoint_publications())
            ),
            ORCHESTRATION_PENDING_EVALS: float(
                len(
                    self.ledger.evals(
                        statuses=("pending", "submitted"),
                    )
                )
            ),
            ORCHESTRATION_SCRATCH_USED_FRACTION: (
                usage.used / max(usage.total, 1)
            ),
        }
        step = int(self.metric_store.latest_metric("global_step") or 0)
        self.metric_store.append_metrics(
            metrics,
            step=step,
            source="orchestration:health",
        )
        self.ledger.set_state(
            "backpressure",
            {
                **metrics,
                "publication_capacity_sufficient": capacity_ratio >= 2.0,
                "sampled_at": utc_now(),
            },
        )
        self.last_health_sample = now
        self.last_health_local_high_water = local_high_water
        self.last_health_wandb_high_water = wandb_high_water

    def _oldest_unpublished_age(self) -> float:
        health = self.metric_store.outbox_health()
        oldest = health.get("oldest_created_at")
        return 0.0 if oldest is None else max(0.0, time.time() - float(oldest))

    def _all_ready_checkpoints_published(self) -> bool:
        for checkpoint in self.metric_store.checkpoints():
            ledger_id = int(checkpoint["id"])
            if self.ledger.checkpoint_publication(ledger_id) is not None:
                continue
            digest = str(checkpoint.get("sha256") or "")
            if not digest:
                path = Path(str(checkpoint["path"]))
                if not path.is_file():
                    return False
                digest = file_sha256(path)
            checkpoint_id = f"checkpoint-{int(checkpoint['step'] or 0)}-{digest[:16]}"
            if self.ledger.checkpoint_publication_by_id(checkpoint_id) is None:
                return False
        return True

    def _drain(self) -> None:
        deadline = time.monotonic() + WANDB_DRAIN_TIMEOUT_SECONDS
        while True:
            now = time.monotonic()
            activity = 0
            self._renew_lease(now)
            if self.lease_lost:
                raise LeaseUnavailable("writer lease was lost while draining")
            activity += self._seal_metrics(now, force=True)
            activity += self._publish_checkpoints()
            if self.cancel_requested:
                self._cancel_outstanding_evals()
            else:
                activity += self._submit_pending_evals()
            activity += self._poll_evals(now)
            activity += self._publish_wandb()
            pending_frames = self.metric_store.metric_outbox_stats()["frames"]
            local_high_water, _wandb_high_water = self._frame_high_waters()
            all_checkpoints_published = self._all_ready_checkpoints_published()
            if (
                all_checkpoints_published
                and self.ledger.all_evals_terminal()
                and pending_frames == 0
            ):
                self._probe_wandb_remote(
                    now,
                    local_high_water=local_high_water,
                    force=True,
                )
                remote_visible = self.wandb_remote_high_water >= local_high_water
                if remote_visible:
                    if (
                        self.peak_ingress_rate > 0.0
                        and self.peak_publish_capacity
                        < 2.0 * self.peak_ingress_rate
                    ):
                        raise RuntimeError(
                            "measured W&B publication capacity is below "
                            "twice peak metric ingress"
                        )
                    return
            if now >= deadline:
                raise TimeoutError(
                    "terminal drain exceeded 300 seconds before checkpoints, "
                    "evals, local W&B delivery, and remote W&B visibility converged"
                )
            if activity == 0:
                time.sleep(0.5)

    def _create_promotion(self) -> PromotionReceipt | None:
        existing = self.authority.control.get_json_optional(
            f"runs/{self.manifest.run_id}/promotion.json"
        )
        if existing is not None:
            receipt = PromotionReceipt(**existing)
            receipt.validate()
            self.authority.create_promotion(receipt)
            return receipt
        accepted = self.ledger.evals(statuses=("accepted",))
        if not accepted:
            return None
        selected = min(
            accepted,
            key=lambda row: (
                int(row["checkpoint_step"]),
                str(row["checkpoint_id"]),
            ),
        )
        result = dict(selected["result"])
        receipt = PromotionReceipt(
            run_id=self.manifest.run_id,
            checkpoint_id=str(selected["checkpoint_id"]),
            checkpoint_step=int(selected["checkpoint_step"]),
            eval_idempotency_key=str(selected["idempotency_key"]),
            eval_result_sha256=document_sha256(result),
            accepted_episode_count=len(result.get("episode_results") or []),
            promoted_at=utc_now(),
        )
        self.authority.create_promotion(receipt)
        return receipt

    def _publish_promotion(self, receipt: PromotionReceipt) -> None:
        selected = self.ledger.eval(receipt.eval_idempotency_key)
        if selected is None:
            raise RuntimeError("promoted eval is absent from the supervisor ledger")
        raw = self.authority.eval_result(
            run_id=self.manifest.run_id,
            idempotency_key=receipt.eval_idempotency_key,
        )
        if raw is None:
            raise RuntimeError("promoted eval raw result is absent from private R2")
        checkpoint = self.ledger.checkpoint_publication_by_id(
            receipt.checkpoint_id
        )
        if checkpoint is None:
            raise RuntimeError("promoted checkpoint is absent from the public inventory")
        metrics = dict(raw.get("metrics") or {})
        metrics.update(dict(selected["result"].get("aggregates") or {}))
        assert self.projector is not None
        publish_promotion_summary(
            self.projector.run,
            checkpoint_step=receipt.checkpoint_step,
            checkpoint_url=str(checkpoint["public_url"]),
            metrics=metrics,
            updated_at=receipt.promoted_at,
        )

    def _wait_for_remote_promotion(self, receipt: PromotionReceipt) -> None:
        assert self.projector is not None
        deadline = time.monotonic() + WANDB_DRAIN_TIMEOUT_SECONDS
        while True:
            try:
                import wandb

                path = getattr(self.projector.run, "path", "")
                run_path = (
                    "/".join(str(part) for part in path)
                    if isinstance(path, list | tuple)
                    else str(path)
                )
                api = wandb.Api(timeout=10)
                flush = getattr(api, "flush", None)
                if callable(flush):
                    flush()
                summary = dict(getattr(api.run(run_path), "summary", {}) or {})
                if (
                    str(summary.get("rlab/goal/outcome") or "") == "accepted"
                    and int(summary.get("leader/checkpoint/step") or -1)
                    == receipt.checkpoint_step
                ):
                    return
            except Exception as exc:
                self.ledger.set_state(
                    "wandb_promotion_probe_error",
                    {"error": repr(exc)[:1000], "at": utc_now()},
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "W&B promotion summary did not become remotely visible "
                    "within 300 seconds"
                )
            time.sleep(WANDB_DRAIN_REMOTE_PROBE_SECONDS)

    def _wandb_high_water(self) -> int:
        with self.ledger.connection() as connection:
            row = connection.execute(
                "SELECT COALESCE(MAX(id), 0) FROM metric_frames WHERE status = 'published'"
            ).fetchone()
        return int(row[0] if row else 0)

    def _finish_wandb(self) -> int:
        high_water = self._wandb_high_water()
        if self.projector is not None:
            self.projector.close()
            self.projector = None
        return high_water

    def _terminal_inventory(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        checkpoints = self.ledger.checkpoint_publications()
        evals = []
        for row in self.ledger.evals():
            result = dict(row.get("result") or {})
            evals.append(
                {
                    "checkpoint_id": str(row["checkpoint_id"]),
                    "checkpoint_step": int(row["checkpoint_step"]),
                    "idempotency_key": str(row["idempotency_key"]),
                    "status": str(row["status"]),
                    "modal_call_id": str(row.get("modal_call_id") or ""),
                    "episode_count": len(result.get("episode_results") or []),
                    "result_sha256": document_sha256(result) if result else None,
                }
            )
        return checkpoints, evals

    def run(self) -> int:
        self.validate_runtime()
        self.materialize()
        holder = f"{uuid.uuid4().hex}@{os.uname().nodename}"
        self.lease = self.authority.acquire_lease(
            run_id=self.manifest.run_id,
            attempt_id=self.manifest.attempt_id,
            holder_id=holder,
        )
        self.last_lease_renewal = time.monotonic()
        self.metric_store.init()
        self.ledger.init()
        self.metric_store.reset_interrupted_metric_frames()
        self._recover_durable_state()
        self._start_wandb()
        if self.recovery_mode == "drain-only":
            print("drain-only recovery: learner will not restart", flush=True)
        else:
            self._start_learner()
        previous_term = signal.getsignal(signal.SIGTERM)
        previous_int = signal.getsignal(signal.SIGINT)
        learner_exited_at: float | None = (
            time.time() if self.recovery_mode == "drain-only" else None
        )

        def cancel(_signum, _frame) -> None:
            self.cancel_requested = True
            self._request_learner_stop("canceled")

        signal.signal(signal.SIGTERM, cancel)
        signal.signal(signal.SIGINT, cancel)
        failure: BaseException | None = None
        try:
            while self.learner is not None and self.learner.poll() is None:
                now = time.monotonic()
                self._renew_lease(now)
                self._seal_metrics(now)
                self._publish_checkpoints()
                self._submit_pending_evals()
                self._poll_evals(now)
                self._publish_wandb()
                self._emit_health(now)
                self._scratch_guard()
                unpublished_age = self._oldest_unpublished_age()
                if unpublished_age >= WANDB_WARNING_SECONDS:
                    print(
                        f"warning: oldest unpublished W&B event is "
                        f"{unpublished_age:.1f}s old",
                        flush=True,
                    )
                if unpublished_age >= WANDB_UNHEALTHY_SECONDS:
                    self.ledger.set_state(
                        "wandb_unhealthy",
                        {"oldest_unpublished_seconds": unpublished_age, "at": utc_now()},
                    )
                if self.lease_lost:
                    break
                time.sleep(0.25)
            if self.learner is not None:
                learner_returncode = self.learner.wait()
                learner_exited_at = time.time()
                log = getattr(self.learner, "_rlab_log", None)
                if log is not None:
                    log.close()
                print(f"learner exited returncode={learner_returncode}", flush=True)
                if learner_returncode != 0:
                    raise RuntimeError(f"learner exited with code {learner_returncode}")
            if self.lease_lost:
                raise LeaseUnavailable("writer lease was lost")
            if self.cancel_requested:
                self._cancel_outstanding_evals()
                raise RuntimeError("run canceled")
            self._publish_checkpoints()
            self.ledger.set_state(
                "checkpoint_set_frozen",
                {
                    "checkpoint_ledger_ids": [
                        int(row["id"]) for row in self.metric_store.checkpoints()
                    ],
                    "frozen_at": utc_now(),
                },
            )
            self._drain()
            assert learner_exited_at is not None
            self.metric_store.append_metrics(
                {
                    ORCHESTRATION_IDLE_GPU_TAIL_SECONDS: max(
                        0.0,
                        time.time() - learner_exited_at,
                    )
                },
                step=max(
                    (
                        int(row.get("step") or 0)
                        for row in self.metric_store.checkpoints()
                    ),
                    default=0,
                ),
                source="orchestration:drain",
            )
            self._drain()
            if self.evaluation_required:
                promotion = self._create_promotion()
                if promotion is None:
                    raise RuntimeError("training ended without an accepted checkpoint")
                self._publish_promotion(promotion)
                self._wait_for_remote_promotion(promotion)
        except BaseException as exc:
            failure = exc
            self._request_learner_stop("supervisor_failure")
            if self.learner is not None and self.learner.poll() is None:
                try:
                    self.learner.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    self.learner.terminate()
                    self.learner.wait(timeout=30)
            if not self.lease_lost:
                try:
                    self._drain()
                except Exception as drain_exc:
                    print(f"failure drain incomplete: {drain_exc}", flush=True)
        finally:
            signal.signal(signal.SIGTERM, previous_term)
            signal.signal(signal.SIGINT, previous_int)

        if self.lease_lost:
            self.projector = None
            print("run stopped after writer lease loss; no further state was mutated", flush=True)
            return 1
        try:
            wandb_high_water = self._finish_wandb()
        except Exception as exc:
            failure = failure or exc
            wandb_high_water = self._wandb_high_water()
        journal_archive: dict[str, Any] | None = None
        journal_expires_at: str | None = None
        if failure is None:
            try:
                journal_archive = self.authority.archive_metric_journals(
                    run_id=self.manifest.run_id
                )
                journal_expires_at = (
                    datetime.now(UTC)
                    + timedelta(days=METRIC_JOURNAL_RETENTION_DAYS)
                ).isoformat().replace("+00:00", "Z")
            except Exception as exc:
                failure = exc
        checkpoints, evals = self._terminal_inventory()
        final_step = max((int(row["step"]) for row in checkpoints), default=0)
        state = (
            "canceled"
            if self.cancel_requested
            else "resumable_failure"
            if failure is not None
            else "succeeded"
        )
        stop_reason = (
            self.stop_reason
            or (
                "completed_after_eval_acceptance"
                if state == "succeeded" and self.evaluation_required
                else "training_cap_complete"
                if state == "succeeded"
                else "supervisor_failure"
            )
        )
        receipt = TerminalReceipt(
            run_id=self.manifest.run_id,
            attempt_id=self.manifest.attempt_id,
            state=state,  # type: ignore[arg-type]
            acceptance_required=self.evaluation_required,
            stop_reason=stop_reason,
            final_step=final_step,
            checkpoint_inventory=checkpoints,
            eval_inventory=evals,
            wandb_high_water_mark=wandb_high_water,
            drain={
                "complete": failure is None,
                "metric_segment_high_water": self.ledger.metric_segment_high_water(),
                "eval_terminal_count": self.ledger.terminal_eval_count(),
                "journal_archive": journal_archive,
                "journal_expires_at": journal_expires_at,
                "wandb_remote_high_water_mark": self.wandb_remote_high_water,
                "publication_capacity_ratio": (
                    self.peak_publish_capacity / self.peak_ingress_rate
                    if self.peak_ingress_rate > 0.0
                    else None
                ),
                "failure": repr(failure)[:4000] if failure is not None else None,
            },
            completed_at=utc_now(),
        )
        self.authority.create_attempt_terminal(receipt)
        if failure is not None:
            print(f"run failed: {failure!r}", flush=True)
            return 1
        if self.evaluation_required:
            self.authority.create_terminal(receipt)
        print(
            f"{'run accepted' if self.evaluation_required else 'training-only run completed'}: "
            f"run_id={self.manifest.run_id} "
            f"final_step={final_step} dstack={DSTACK_VERSION}",
            flush=True,
        )
        return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Supervise one immutable dstack rlab training run."
    )
    parser.add_argument("--manifest-uri", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return RunSupervisor(manifest_uri=args.manifest_uri).run()


if __name__ == "__main__":
    raise SystemExit(main())
