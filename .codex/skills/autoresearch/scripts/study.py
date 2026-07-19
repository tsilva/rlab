#!/usr/bin/env python3
"""Durable, deterministic state manager for the local autoresearch skill."""

from __future__ import annotations

import argparse
import copy
import difflib
import fcntl
import hashlib
import json
import math
import os
import shlex
import subprocess
import tempfile
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator, Mapping

from rlab.job_queue import submission_batch_id
from rlab.provider_config import provider_num_envs
from rlab.recipe_documents import compose_train_document
from rlab.training_backend import accepts_first_training_success, training_backend_id


SCHEMA_VERSION = 1
SUPPORTED_BACKENDS = frozenset({"sb3.ppo", "sb3.a2c"})
TRACE_ONLY_OVERRIDES = frozenset({"campaign_id", "description", "recipe_id"})
FROZEN_BACKEND_KEYS = frozenset({"n_steps"})
MAX_RESERVED_JOBS = 48
CONFIRMATION_RUNS = 5
SEARCH_SEED_COUNT = 2
MAX_CANDIDATES_PER_WAVE = 3
DEFAULT_STALE_ROUNDS = 3

GROUPS: dict[str, frozenset[str]] = {
    "learning_rate": frozenset(
        {"learning_rate", "learning_rate_final", "learning_rate_schedule_timesteps"}
    ),
    "entropy": frozenset({"ent_coef", "ent_coef_final", "ent_coef_schedule_timesteps"}),
    "discounting": frozenset({"gamma", "gae_lambda"}),
    "value": frozenset({"vf_coef"}),
    "ppo_update": frozenset({"batch_size", "n_epochs", "clip_range", "target_kl", "adam_eps"}),
    "a2c_optimizer": frozenset(
        {"learning_rate", "learning_rate_final", "max_grad_norm", "rms_prop_eps", "vf_coef"}
    ),
}

SEMANTIC_BOUNDS: dict[str, tuple[float, float]] = {
    "learning_rate": (1e-7, 1.0),
    "learning_rate_final": (0.0, 1.0),
    "learning_rate_schedule_timesteps": (1.0, float("inf")),
    "ent_coef": (0.0, 10.0),
    "ent_coef_final": (0.0, 10.0),
    "ent_coef_schedule_timesteps": (1.0, float("inf")),
    "gamma": (0.0, 1.0),
    "gae_lambda": (0.0, 1.0),
    "vf_coef": (0.0, 10.0),
    "batch_size": (1.0, float("inf")),
    "n_epochs": (1.0, 1000.0),
    "clip_range": (1e-6, 1.0),
    "target_kl": (1e-8, 10.0),
    "adam_eps": (1e-12, 1.0),
    "max_grad_norm": (0.0, 1000.0),
    "rms_prop_eps": (1e-12, 1.0),
}


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def git_head(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def git_blob_sha256(root: Path, source_sha: str, path: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{source_sha}:{Path(path).as_posix()}"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if result.returncode:
        detail = result.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"composition source is not tracked at {source_sha}: {path}; {detail}")
    return hashlib.sha256(result.stdout).hexdigest()


def relative(root: Path, value: str | Path) -> str:
    path = Path(value).resolve()
    try:
        return str(path.relative_to(root.resolve()))
    except ValueError as exc:
        raise ValueError(f"composition source is outside the repository: {path}") from exc


@contextmanager
def lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_state(path: Path) -> dict[str, Any]:
    state = json.loads(path.read_text(encoding="utf-8"))
    if int(state.get("schema_version") or 0) != SCHEMA_VERSION:
        raise ValueError(f"unsupported study schema in {path}")
    return state


@contextmanager
def edit_state(path: Path) -> Iterator[dict[str, Any]]:
    with lock(path.parent / ".lock"):
        state = load_state(path)
        yield state
        state["updated_at"] = utc_now()
        atomic_json(path, state)


def study_path(value: str | Path) -> Path:
    path = Path(value)
    return path / "study.json" if path.is_dir() else path


def emit(value: Any) -> None:
    print(json.dumps(value, sort_keys=True))


def numeric_tunables(config: Mapping[str, Any]) -> dict[str, int | float]:
    return {
        str(key): value
        for key, value in config.items()
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value is not None
        and key not in FROZEN_BACKEND_KEYS
    }


def normalize_delta(delta: Mapping[str, Any]) -> dict[str, int | float]:
    return {
        str(key): value
        for key, value in sorted(delta.items())
        if str(key) not in TRACE_ONLY_OVERRIDES
    }


def candidate_id(delta: Mapping[str, Any]) -> str:
    return "c" + digest({"backend_delta": normalize_delta(delta)})[:12]


def group_for_delta(backend: str, keys: set[str]) -> str:
    allowed = (
        {name: value for name, value in GROUPS.items() if name != "a2c_optimizer"}
        if backend == "sb3.ppo"
        else {
            name: value
            for name, value in GROUPS.items()
            if name in {"entropy", "discounting", "value", "a2c_optimizer"}
        }
    )
    matches = [name for name, group in allowed.items() if keys <= group]
    if not matches:
        raise ValueError(
            f"candidate changes must stay within one coherent group; keys={sorted(keys)}"
        )
    return sorted(matches)[0]


def update_work(config: Mapping[str, Any], backend: str, n_envs: int) -> float:
    if backend != "sb3.ppo":
        return 1.0
    rollout = int(config["n_steps"]) * int(n_envs)
    return float(config.get("n_epochs", 1)) * rollout / float(config.get("batch_size", rollout))


def validate_delta(
    state: Mapping[str, Any], raw_delta: Mapping[str, Any]
) -> dict[str, int | float]:
    delta = normalize_delta(raw_delta)
    if not delta:
        raise ValueError("search candidate delta must not be empty")
    baseline = state["baseline"]["backend_config"]
    tunables = state["baseline"]["tunables"]
    unknown = sorted(set(delta) - set(tunables))
    if unknown:
        raise ValueError(f"candidate changes frozen, categorical, null, or absent keys: {unknown}")
    for key, value in delta.items():
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
        ):
            raise ValueError(f"candidate {key} must be a finite numeric value")
        original = float(tunables[key])
        if original == 0.0:
            if float(value) != 0.0:
                raise ValueError(f"candidate {key} cannot move a zero baseline under ratio bounds")
        elif not 0.25 * original <= float(value) <= 4.0 * original:
            raise ValueError(f"candidate {key} must remain within 0.25x..4x of {original:g}")
        low, high = SEMANTIC_BOUNDS.get(key, (-float("inf"), float("inf")))
        if not low <= float(value) <= high:
            raise ValueError(f"candidate {key}={value} violates semantic bounds {low:g}..{high:g}")
        if isinstance(baseline[key], int) and not isinstance(value, int):
            raise ValueError(f"candidate {key} must remain an integer")
    group_for_delta(str(state["backend"]), set(delta))
    effective = {**baseline, **delta}
    if state["backend"] == "sb3.ppo":
        rollout = int(effective["n_steps"]) * int(state["n_envs"])
        batch_size = int(effective["batch_size"])
        if rollout % batch_size:
            raise ValueError(f"PPO batch_size={batch_size} must divide fixed rollout={rollout}")
    candidate_work = update_work(effective, str(state["backend"]), int(state["n_envs"]))
    if candidate_work > 2.0 * float(state["baseline"]["update_work_per_env_step"]):
        raise ValueError("candidate update work per environment step exceeds 2x baseline")
    return delta


def source_guard(state: Mapping[str, Any], *, allow_postimage: bool = False) -> None:
    root = Path(state["repo_root"])
    if git_head(root) != state["source_sha"]:
        raise RuntimeError("repository HEAD changed; pause this study")
    leaf = root / state["recipe_path"]
    leaf_hash = file_sha256(leaf)
    allowed_leaf_hashes = {state["recipe_preimage_sha256"]}
    apply = state.get("apply") or {}
    if allow_postimage and apply.get("postimage_sha256"):
        allowed_leaf_hashes.add(str(apply["postimage_sha256"]))
    if leaf_hash not in allowed_leaf_hashes:
        raise RuntimeError("target recipe changed outside the study's preimage/postimage contract")
    for item in state["source_files"]:
        if item["path"] == state["recipe_path"]:
            continue
        if file_sha256(root / item["path"]) != item["sha256"]:
            raise RuntimeError(f"composition source changed: {item['path']}")


def record_source_pause(state: dict[str, Any], error: Exception) -> dict[str, Any]:
    state["status"] = "paused"
    state["pause_reason"] = {"event": "source_drift", "detail": str(error)}
    return {"action": "pause", "reason": state["pause_reason"]}


def frozen_censor_step(state: Mapping[str, Any]) -> int:
    train = state["baseline"]["train_config"]
    quantum = int(train["training_backend"]["config"]["n_steps"]) * int(state["n_envs"])
    return int(math.ceil(int(train["timesteps"]) / quantum) * quantum)


def candidate_score(state: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    runs = list(candidate.get("search_runs") or [])
    censor = frozen_censor_step(state)
    values: list[int] = []
    accepted = 0
    for run in runs:
        if run.get("accepted_verified"):
            accepted += 1
            values.append(int(run["promoted_step"]))
        else:
            values.append(censor)
    while len(values) < SEARCH_SEED_COUNT:
        values.append(censor)
    values = sorted(values[:SEARCH_SEED_COUNT])
    median = sum(values) / len(values)
    return {
        "accepted_verified": accepted,
        "median_censored_step": median,
        "worst_censored_step": max(values),
        "candidate_id": candidate["id"],
    }


def evidence_key(score: Mapping[str, Any]) -> tuple[float, float, float]:
    return (
        -float(score["accepted_verified"]),
        float(score["median_censored_step"]),
        float(score["worst_censored_step"]),
    )


def ranked_candidates(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    excluded = set(state.get("excluded_candidates") or [])
    for candidate in state["candidates"].values():
        if candidate["id"] in excluded or len(candidate.get("search_runs") or []) != 2:
            continue
        score = candidate_score(state, candidate)
        rows.append({"candidate": candidate, "score": score})
    return sorted(rows, key=lambda row: (*evidence_key(row["score"]), row["score"]["candidate_id"]))


def next_action(state: Mapping[str, Any]) -> dict[str, Any]:
    if state["status"] == "paused":
        return {"action": "pause", "reason": state.get("pause_reason")}
    if state["status"] == "done":
        return {"action": "done", "winner": state.get("winner")}
    if state["status"] == "apply_pending":
        return {"action": "apply_postimage", "winner": state.get("winner")}
    for wave in state["waves"]:
        if wave["status"] == "reserved":
            return {
                "action": "reconcile_submission",
                "batch_id": wave["batch_id"],
                "submission_key": wave["submission_key"],
            }
        if wave["status"] == "launched" and len(wave["terminal_runs"]) < len(wave["seeds"]):
            return {"action": "await_runs", "run_ids": wave["run_ids"]}
    baseline = state["candidates"][state["baseline_candidate_id"]]
    if len(baseline["search_runs"]) < SEARCH_SEED_COUNT:
        return {"action": "reserve_baseline"}
    unclosed = [
        wave
        for wave in state["waves"]
        if wave["phase"] in {"baseline", "search"}
        and wave["status"] == "terminal"
        and not wave.get("closed")
    ]
    if unclosed:
        return {"action": "close_round", "round": min(int(wave["round"]) for wave in unclosed)}
    confirmation = state.get("confirmation")
    if confirmation and not confirmation.get("closed"):
        return {"action": "close_confirmation", "candidate_id": confirmation["candidate_id"]}
    if state.get("winner"):
        return {"action": "prepare_winner", "winner": state["winner"]}
    remaining = MAX_RESERVED_JOBS - int(state["reserved_jobs"])
    if int(state["stale_rounds"]) >= int(state["policy"]["stale_round_limit"]) or remaining < (
        SEARCH_SEED_COUNT + CONFIRMATION_RUNS
    ):
        if state.get("incumbent_candidate_id") and remaining >= CONFIRMATION_RUNS:
            return {
                "action": "reserve_confirmation",
                "candidate_id": state["incumbent_candidate_id"],
            }
        return {"action": "finish_no_winner", "reason": "search budget or evidence exhausted"}
    return {
        "action": "propose_search",
        "round": int(state["search_round"]) + 1,
        "max_candidates": MAX_CANDIDATES_PER_WAVE,
    }


def command_init(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    goal = Path(args.goal)
    recipe = Path(args.recipe)
    goal = (root / goal).resolve() if not goal.is_absolute() else goal.resolve()
    recipe = (root / recipe).resolve() if not recipe.is_absolute() else recipe.resolve()
    head = git_head(root)
    goal_path = relative(root, goal)
    recipe_path = relative(root, recipe)
    input_hash = digest({"goal": goal_path, "recipe": recipe_path, "source_sha": head})
    studies_root = root / "runs" / "autoresearch"
    with lock(studies_root / ".discovery.lock"):
        matches = []
        current_leaf = file_sha256(recipe)
        for path in sorted(studies_root.glob("*/study.json")):
            prior = load_state(path)
            if prior.get("status") == "done":
                continue
            apply = prior.get("apply") or {}
            recognized = current_leaf in {
                prior.get("recipe_preimage_sha256"),
                apply.get("postimage_sha256"),
            }
            if prior.get("input_hash") == input_hash or (
                prior.get("goal_path") == goal_path
                and prior.get("recipe_path") == recipe_path
                and recognized
            ):
                matches.append(path)
        if len(matches) > 1:
            raise RuntimeError(f"ambiguous incomplete studies: {[str(path) for path in matches]}")
        if matches:
            state = load_state(matches[0])
            source_guard(state, allow_postimage=True)
            emit({"study": str(matches[0]), "resumed": True, "next": next_action(state)})
            return
        document = compose_train_document(goal, recipe)
        train = document["train_config"]
        backend = training_backend_id(train)
        if backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"autoresearch v1 supports only {sorted(SUPPORTED_BACKENDS)}, got {backend}"
            )
        if accepts_first_training_success(train):
            raise ValueError("autoresearch requires checkpoint-evaluated acceptance")
        if str(train.get("checkpoint_eval_backend") or "none") == "none":
            raise ValueError("autoresearch requires an available checkpoint evaluation backend")
        if not bool(train.get("stop_on_acceptance")):
            raise ValueError("autoresearch requires goal-owned stop-on-acceptance promotion")
        composition = document.get("_composition") or {}
        sources = [
            {"path": relative(root, item["path"]), "sha256": str(item["sha256"])}
            for item in composition.get("source_files") or []
        ]
        if recipe_path not in {item["path"] for item in sources}:
            sources.append({"path": recipe_path, "sha256": file_sha256(recipe)})
        for item in sources:
            committed_hash = git_blob_sha256(root, head, item["path"])
            if (
                committed_hash != item["sha256"]
                or file_sha256(root / item["path"]) != committed_hash
            ):
                raise RuntimeError(
                    f"composition source differs from committed HEAD and cannot be launched "
                    f"with --from-head: {item['path']}"
                )
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        directory = studies_root / f"{stamp}-{input_hash[:12]}"
        directory.mkdir(parents=True, exist_ok=False)
        backend_config = copy.deepcopy(train["training_backend"]["config"])
        n_envs = provider_num_envs(train, explicit_n_envs=train.get("n_envs"))
        search_seeds = [123 + index * n_envs for index in range(SEARCH_SEED_COUNT)]
        confirmation_seeds = [
            123 + (SEARCH_SEED_COUNT + index) * n_envs for index in range(CONFIRMATION_RUNS)
        ]
        baseline_id = candidate_id({})
        state = {
            "schema_version": SCHEMA_VERSION,
            "study_id": uuid.uuid4().hex,
            "input_hash": input_hash,
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "status": "active",
            "repo_root": str(root),
            "source_sha": head,
            "goal_path": goal_path,
            "recipe_path": recipe_path,
            "recipe_preimage": recipe.read_text(encoding="utf-8"),
            "recipe_preimage_sha256": file_sha256(recipe),
            "source_files": sources,
            "backend": backend,
            "n_envs": n_envs,
            "search_seeds": search_seeds,
            "confirmation_seeds": confirmation_seeds,
            "runtime": None,
            "policy": {
                "machine": "beast-3",
                "max_reserved_jobs": MAX_RESERVED_JOBS,
                "stale_round_limit": DEFAULT_STALE_ROUNDS,
                "confirmation_runs": CONFIRMATION_RUNS,
                "confirmation_required": 4,
            },
            "baseline": {
                "backend_config": backend_config,
                "tunables": numeric_tunables(backend_config),
                "train_config": copy.deepcopy(train),
                "update_work_per_env_step": update_work(backend_config, backend, n_envs),
                "censor_step": int(
                    math.ceil(int(train["timesteps"]) / (int(backend_config["n_steps"]) * n_envs))
                    * int(backend_config["n_steps"])
                    * n_envs
                ),
            },
            "baseline_candidate_id": baseline_id,
            "candidates": {
                baseline_id: {
                    "id": baseline_id,
                    "delta": {},
                    "created_round": 0,
                    "search_runs": [],
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
        atomic_json(directory / "study.json", state)
    emit({"study": str(directory / "study.json"), "resumed": False, "next": next_action(state)})


def command_status(args: argparse.Namespace) -> None:
    path = study_path(args.study)
    state = load_state(path)
    emit({"study": str(path), "state": state, "next": next_action(state)})


def command_next(args: argparse.Namespace) -> None:
    path = study_path(args.study)
    with edit_state(path) as state:
        if state["status"] == "done":
            action = next_action(state)
        else:
            try:
                source_guard(state, allow_postimage=True)
            except RuntimeError as exc:
                action = record_source_pause(state, exc)
            else:
                action = next_action(state)
    emit({"study": str(path), "next": action})


def parse_candidates(value: str) -> list[dict[str, Any]]:
    payload = json.loads(value)
    if not isinstance(payload, list):
        raise ValueError("--candidates-json must be a JSON list")
    return [dict(item) for item in payload]


def expected_recipe_overrides(state: Mapping[str, Any], wave: Mapping[str, Any]) -> list[str]:
    candidate = state["candidates"][wave["candidate_id"]]
    overrides = [
        f"train.backend.config.{key}={json.dumps(value)}"
        for key, value in candidate["delta"].items()
    ]
    description = (
        f"Autoresearch {state['study_id'][:8]} {wave['phase']} candidate "
        f"{wave['candidate_id']} seed {{seed}}."
    )
    return [*overrides, f"description={description}"]


def launch_command(state: Mapping[str, Any], wave: Mapping[str, Any]) -> list[str]:
    command = [
        "rlab",
        "experiment",
        "launch",
        "--from-head",
        "--goal-file",
        state["goal_path"],
        "--recipe-file",
        state["recipe_path"],
        "--machine",
        state["policy"]["machine"],
        "--request-id",
        wave["submission_key"],
        "--existing-runtime-only",
    ]
    runtime = state.get("runtime")
    if runtime:
        command.extend(
            [
                "--expected-runtime-image-ref",
                runtime["image_ref"],
                "--expected-runtime-input-sha256",
                runtime["input_sha256"],
                "--expected-runtime-build-source-sha",
                runtime["build_source_sha"],
            ]
        )
    for seed in wave["seeds"]:
        command.extend(["--seed", str(seed)])
    for override in expected_recipe_overrides(state, wave):
        command.extend(["--set", override])
    command.append("--json")
    return command


def reserve_one(
    state: dict[str, Any],
    *,
    phase: str,
    round_number: int,
    candidate: Mapping[str, Any],
    seeds: list[int],
) -> dict[str, Any]:
    delta = dict(candidate.get("delta") or {})
    if phase == "baseline":
        identifier = state["baseline_candidate_id"]
    else:
        delta = validate_delta(state, delta) if phase == "search" else delta
        identifier = candidate_id(delta)
        if phase == "confirmation" and identifier != state["incumbent_candidate_id"]:
            raise ValueError("confirmation must use the current incumbent")
        state["candidates"].setdefault(
            identifier,
            {
                "id": identifier,
                "delta": delta,
                "created_round": round_number,
                "search_runs": [],
                "confirmation_runs": [],
            },
        )
        if state["candidates"][identifier]["delta"] != delta:
            raise RuntimeError(f"candidate id collision for {identifier}")
    nonce = state["study_id"][:10]
    submission_key = f"autoresearch-{nonce}-{phase}-r{round_number}-{identifier}"
    wave = {
        "phase": phase,
        "round": round_number,
        "candidate_id": identifier,
        "seeds": seeds,
        "submission_key": submission_key,
        "batch_id": submission_batch_id(submission_key),
        "status": "reserved",
        "run_ids": [],
        "terminal_runs": [],
        "closed": False,
        "reserved_at": utc_now(),
    }
    state["waves"].append(wave)
    state["reserved_jobs"] += len(seeds)
    return wave


def command_reserve(args: argparse.Namespace) -> None:
    path = study_path(args.study)
    with edit_state(path) as state:
        try:
            source_guard(state)
        except RuntimeError as exc:
            action = record_source_pause(state, exc)
            emit({"study": str(path), "next": action})
            return
        if state["status"] != "active":
            raise RuntimeError(f"cannot reserve while study is {state['status']}")
        action = next_action(state)["action"]
        phase = args.phase
        if phase == "baseline" and action != "reserve_baseline":
            raise RuntimeError(f"state expects {action}, not baseline reservation")
        if phase == "search" and action != "propose_search":
            raise RuntimeError(f"state expects {action}, not search reservation")
        if phase == "confirmation" and action != "reserve_confirmation":
            raise RuntimeError(f"state expects {action}, not confirmation reservation")
        free = max(int(args.effective_capacity) - int(args.active_reservations), 0)
        if free <= 0:
            raise RuntimeError("beast-3 has no available slot; wait without reserving a wave")
        candidates = parse_candidates(args.candidates_json)
        if phase == "baseline":
            candidates = [{"delta": {}}]
            seeds = list(state["search_seeds"])
            cohort_limit = 1
            round_number = 0
        elif phase == "confirmation":
            incumbent = state["candidates"][state["incumbent_candidate_id"]]
            candidates = [{"delta": incumbent["delta"]}]
            seeds = list(state["confirmation_seeds"])
            cohort_limit = 1
            round_number = int(state["search_round"])
        else:
            seeds = list(state["search_seeds"])
            round_number = int(state["search_round"]) + 1
            budget_cohorts = (
                MAX_RESERVED_JOBS - int(state["reserved_jobs"]) - CONFIRMATION_RUNS
            ) // SEARCH_SEED_COUNT
            cohort_limit = min(
                MAX_CANDIDATES_PER_WAVE,
                max(1, math.ceil(free / SEARCH_SEED_COUNT)),
                budget_cohorts,
            )
        if not candidates:
            raise ValueError("at least one candidate is required")
        if phase == "search" and len(candidates) < cohort_limit:
            raise ValueError(
                f"provide exactly {cohort_limit} new candidates to fill available beast-3 slots"
            )
        candidates = sorted(candidates, key=lambda item: candidate_id(item.get("delta") or {}))
        candidates = candidates[:cohort_limit]
        identifiers = [candidate_id(item.get("delta") or {}) for item in candidates]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("a search wave cannot reserve the same candidate twice")
        if phase == "search":
            existing = set(state["candidates"])
            repeated = sorted(set(identifiers) & existing)
            if repeated:
                raise ValueError(f"search candidates must be new to the study: {repeated}")
        needed = len(candidates) * len(seeds)
        reserve_for_confirmation = 0 if phase == "confirmation" else CONFIRMATION_RUNS
        if int(state["reserved_jobs"]) + needed + reserve_for_confirmation > MAX_RESERVED_JOBS:
            raise RuntimeError(
                "reservation would consume the confirmation reserve or exceed 48 jobs"
            )
        waves = [
            reserve_one(
                state,
                phase=phase,
                round_number=round_number,
                candidate=candidate,
                seeds=seeds,
            )
            for candidate in candidates
        ]
        if phase == "confirmation":
            state["confirmation"] = {
                "candidate_id": waves[0]["candidate_id"],
                "submission_key": waves[0]["submission_key"],
                "closed": False,
            }
        output = [
            {
                "candidate_id": wave["candidate_id"],
                "submission_key": wave["submission_key"],
                "batch_id": wave["batch_id"],
                "command": launch_command(state, wave),
                "shell_command": shlex.join(launch_command(state, wave)),
            }
            for wave in waves
        ]
    emit({"study": str(path), "reserved": output, "launch_concurrently": len(output) > 1})


def find_wave(state: Mapping[str, Any], submission_key: str) -> dict[str, Any]:
    matches = [wave for wave in state["waves"] if wave["submission_key"] == submission_key]
    if len(matches) != 1:
        raise ValueError(f"submission key maps to {len(matches)} waves")
    return matches[0]


def read_json_arg(value: str | None, path: str | None) -> dict[str, Any]:
    if bool(value) == bool(path):
        raise ValueError("provide exactly one JSON value or JSON file")
    return json.loads(value) if value else json.loads(Path(str(path)).read_text(encoding="utf-8"))


def command_record_launch(args: argparse.Namespace) -> None:
    payload = read_json_arg(args.payload_json, args.payload_file)
    path = study_path(args.study)
    with edit_state(path) as state:
        try:
            source_guard(state)
        except RuntimeError as exc:
            action = record_source_pause(state, exc)
            emit({"study": str(path), "next": action})
            return
        wave = find_wave(state, args.submission_key)
        errors: list[str] = []
        observed_batch = str(
            payload.get("batch_id") or (payload.get("selector") or {}).get("batch_id") or ""
        )
        if observed_batch != wave["batch_id"]:
            errors.append("batch does not match deterministic batch id")
        rows = payload.get("jobs") or payload.get("runs") or []
        run_ids = (
            payload.get("run_ids")
            or payload.get("job_ids")
            or [row.get("run_id", row.get("id")) for row in rows]
        )
        run_ids = [int(value) for value in run_ids if value is not None]
        if len(run_ids) != len(wave["seeds"]) or len(set(run_ids)) != len(run_ids):
            errors.append("submission is partial or has an unexpected run count")
        if rows:
            observed_seeds = sorted(
                int((row.get("submission") or {}).get("seed", row.get("seed"))) for row in rows
            )
            if observed_seeds != sorted(wave["seeds"]):
                errors.append("submission rows do not match the reserved training seeds")
            submissions = [row.get("submission") or {} for row in rows]
            observed_keys = {
                str(item.get("key", row.get("submission_key")) or "")
                for item, row in zip(submissions, rows, strict=True)
            }
            if observed_keys != {wave["submission_key"]}:
                errors.append("submission rows do not match the reserved submission key")
            request_hashes = {str(item.get("request_hash") or "") for item in submissions}
            if request_hashes == {""}:
                request_hashes = {str(row.get("request_hash") or "") for row in rows}
            if len(request_hashes) != 1 or "" in request_hashes:
                errors.append("submission rows do not share one non-empty request hash")
            observed_sources = {
                str(row.get("source_sha") or row.get("repo_git_commit") or "") for row in rows
            }
            if observed_sources != {state["source_sha"]}:
                errors.append("submission rows do not match the pinned source revision")
            goal_paths = {
                str(item.get("goal_path", row.get("goal_path")) or "")
                for item, row in zip(submissions, rows, strict=True)
            }
            recipe_paths = {
                str(item.get("recipe_path", row.get("recipe_path")) or "")
                for item, row in zip(submissions, rows, strict=True)
            }
            if goal_paths != {state["goal_path"]} or recipe_paths != {state["recipe_path"]}:
                errors.append("submission rows do not match the pinned goal and recipe paths")
            pinned_hashes = {item["path"]: item["sha256"] for item in state["source_files"]}
            goal_hashes = {
                str(item.get("goal_sha256", row.get("goal_sha256")) or "")
                for item, row in zip(submissions, rows, strict=True)
            }
            recipe_hashes = {
                str(item.get("recipe_sha256", row.get("recipe_sha256")) or "")
                for item, row in zip(submissions, rows, strict=True)
            }
            if goal_hashes != {pinned_hashes[state["goal_path"]]} or recipe_hashes != {
                state["recipe_preimage_sha256"]
            }:
                errors.append("submission rows do not match the pinned goal and recipe hashes")
            observed_overrides = []
            for item, row in zip(submissions, rows, strict=True):
                recipe_payload = row.get("recipe_payload_json") or {}
                recipe_document = recipe_payload.get("recipe") or {}
                observed_overrides.append(
                    list(
                        item.get("recipe_overrides")
                        or recipe_payload.get("recipe_overrides")
                        or recipe_document.get("recipe_overrides")
                        or []
                    )
                )
            if any(value != expected_recipe_overrides(state, wave) for value in observed_overrides):
                errors.append("submission rows do not match the reserved recipe overrides")
        runtime = {
            "image_ref": str(payload.get("runtime_image_ref") or ""),
            "input_sha256": str(payload.get("runtime_input_sha256") or ""),
            "build_source_sha": str(payload.get("runtime_build_source_sha") or ""),
        }
        if not all(runtime.values()) and rows:
            first = rows[0]
            runtime_projection = first.get("runtime") or {}
            runtime = {
                "image_ref": str(
                    runtime_projection.get("image_ref") or first.get("runtime_image_ref") or ""
                ),
                "input_sha256": str(runtime_projection.get("input_sha256") or ""),
                "build_source_sha": str(runtime_projection.get("build_source_sha") or ""),
            }
        if rows:
            row_runtimes = set()
            for row in rows:
                projected = row.get("runtime") or {}
                config = row.get("train_config") or {}
                row_runtimes.add(
                    (
                        str(projected.get("image_ref") or row.get("runtime_image_ref") or ""),
                        str(
                            projected.get("input_sha256")
                            or config.get("runtime_input_sha256")
                            or ""
                        ),
                        str(
                            projected.get("build_source_sha")
                            or config.get("runtime_build_source_sha")
                            or ""
                        ),
                    )
                )
            if len(row_runtimes) != 1 or next(iter(row_runtimes)) != tuple(runtime.values()):
                errors.append("submission rows do not share the resolved runtime triplet")
        if not all(runtime.values()):
            errors.append("launch/status payload lacks the complete runtime triplet")
        if state.get("runtime") and state["runtime"] != runtime:
            errors.append("recovered launch uses a different runtime triplet")
        if errors:
            state["status"] = "paused"
            state["pause_reason"] = {
                "event": "submission_reconciliation_mismatch",
                "submission_key": args.submission_key,
                "errors": errors,
            }
        else:
            state["runtime"] = runtime
            wave["run_ids"] = sorted(run_ids)
            wave["status"] = "launched"
            wave["launched_at"] = utc_now()
        result = {
            "study": str(path),
            "submission_key": args.submission_key,
            "run_ids": sorted(run_ids),
            "paused": bool(errors),
            "errors": errors,
        }
    emit(result)


def command_record_terminal(args: argparse.Namespace) -> None:
    payload = read_json_arg(args.event_json, args.event_file)
    submission = payload.get("submission") or {}
    submission_key = str(args.submission_key or submission.get("key") or "")
    run_id = int(args.run_id if args.run_id is not None else payload.get("run_id"))
    seed = int(args.seed if args.seed is not None else submission.get("seed"))
    if not submission_key:
        raise ValueError("terminal event lacks submission.key")
    path = study_path(args.study)
    with edit_state(path) as state:
        wave = find_wave(state, submission_key)
        if run_id not in wave["run_ids"]:
            raise RuntimeError("terminal run is not part of the recorded launch")
        if seed not in wave["seeds"]:
            raise RuntimeError("terminal seed is not part of the reservation")
        if any(int(item["run_id"]) == run_id for item in wave["terminal_runs"]):
            emit({"study": str(path), "duplicate": True, "run_id": run_id})
            return
        classification = str(payload.get("terminal_classification") or "")
        if classification not in {
            "accepted",
            "goal_rejected",
            "completed",
            "canceled",
            "operational_failure",
        }:
            raise ValueError("payload is not a terminal follow event")
        wandb = payload.get("wandb") or {}
        evaluation = payload.get("evaluation") or {}
        accepted_verified = (
            classification == "accepted"
            and bool(payload.get("verified_success"))
            and bool(wandb.get("remote_verified"))
        )
        promoted_step = evaluation.get("promoted_step")
        if accepted_verified and promoted_step is None:
            raise RuntimeError("accepted+verified terminal event lacks promoted_step")
        record = {
            "run_id": run_id,
            "seed": seed,
            "classification": classification,
            "accepted_verified": accepted_verified,
            "promoted_step": int(promoted_step) if promoted_step is not None else None,
            "wandb_url": wandb.get("url"),
            "artifact": (payload.get("artifacts") or {}).get("wandb_artifact"),
        }
        wave["terminal_runs"].append(record)
        wave["terminal_runs"].sort(key=lambda item: item["seed"])
        candidate = state["candidates"][wave["candidate_id"]]
        destination = "confirmation_runs" if wave["phase"] == "confirmation" else "search_runs"
        candidate[destination].append(record)
        candidate[destination].sort(key=lambda item: (item["seed"], item["run_id"]))
        if len(wave["terminal_runs"]) == len(wave["seeds"]):
            wave["status"] = "terminal"
        if classification in {"canceled", "operational_failure", "completed"} or (
            classification == "accepted" and not accepted_verified
        ):
            state["status"] = "paused"
            state["pause_reason"] = {
                "event": "terminal_without_valid_research_evidence",
                "run_id": run_id,
                "classification": classification,
                "remote_verified": bool(wandb.get("remote_verified")),
            }
    emit({"study": str(path), "run_id": run_id, "next": next_action(load_state(path))})


def command_close_round(args: argparse.Namespace) -> None:
    path = study_path(args.study)
    with edit_state(path) as state:
        round_number = int(args.round)
        waves = [
            wave
            for wave in state["waves"]
            if int(wave["round"]) == round_number and wave["phase"] in {"baseline", "search"}
        ]
        if not waves or any(wave["status"] != "terminal" for wave in waves):
            raise RuntimeError("round barrier requires every cohort to be terminal")
        if any(wave.get("closed") for wave in waves):
            raise RuntimeError("round was already closed")
        ranked = ranked_candidates(state)
        if not ranked:
            raise RuntimeError("no complete paired candidate evidence is available")
        best = ranked[0]
        prior = state.get("incumbent_evidence")
        improved = prior is None or evidence_key(best["score"]) < evidence_key(prior)
        if round_number > 0:
            state["search_round"] = round_number
            state["stale_rounds"] = 0 if improved else int(state["stale_rounds"]) + 1
        state["incumbent_candidate_id"] = best["candidate"]["id"]
        state["incumbent_evidence"] = best["score"]
        for wave in waves:
            wave["closed"] = True
    emit({"study": str(path), "ranking": [row["score"] for row in ranked], "improved": improved})


def command_close_confirmation(args: argparse.Namespace) -> None:
    path = study_path(args.study)
    with edit_state(path) as state:
        confirmation = state.get("confirmation")
        if not confirmation or confirmation.get("closed"):
            raise RuntimeError("no open confirmation exists")
        wave = find_wave(state, confirmation["submission_key"])
        if wave["status"] != "terminal" or len(wave["terminal_runs"]) != CONFIRMATION_RUNS:
            raise RuntimeError("confirmation barrier requires exactly five terminal runs")
        accepted = sum(bool(item["accepted_verified"]) for item in wave["terminal_runs"])
        confirmation["closed"] = True
        confirmation["accepted_verified"] = accepted
        candidate_id_value = confirmation["candidate_id"]
        if accepted >= int(state["policy"]["confirmation_required"]):
            state["winner"] = {
                "candidate_id": candidate_id_value,
                "delta": state["candidates"][candidate_id_value]["delta"],
                "accepted_verified": accepted,
                "runs": wave["terminal_runs"],
            }
        else:
            state["excluded_candidates"].append(candidate_id_value)
            state["stale_rounds"] = 0
            redacted = {
                "redacted": True,
                "accepted_verified_count": accepted,
                "total": CONFIRMATION_RUNS,
            }
            state["candidates"][candidate_id_value]["confirmation_runs"] = [redacted]
            wave["terminal_runs"] = [redacted]
            ranked = ranked_candidates(state)
            state["incumbent_candidate_id"] = ranked[0]["candidate"]["id"] if ranked else None
            state["incumbent_evidence"] = ranked[0]["score"] if ranked else None
            state["confirmation"] = None
    emit({"study": str(path), "accepted_verified": accepted, "next": next_action(load_state(path))})


def command_attention(args: argparse.Namespace) -> None:
    payload = read_json_arg(args.event_json, args.event_file)
    path = study_path(args.study)
    with edit_state(path) as state:
        state["status"] = "paused"
        state["pause_reason"] = {
            "event": payload.get("event"),
            "attention": payload.get("attention"),
            "incident": payload.get("incident"),
            "run_id": payload.get("run_id"),
        }
    emit({"study": str(path), "next": {"action": "pause", "reason": state["pause_reason"]}})


def command_resume(args: argparse.Namespace) -> None:
    path = study_path(args.study)
    with edit_state(path) as state:
        if state["status"] != "paused":
            raise RuntimeError("only a paused study can resume")
        source_guard(state, allow_postimage=True)
        prior = copy.deepcopy(state.get("pause_reason"))
        state.setdefault("resume_history", []).append(
            {"resumed_at": utc_now(), "reason": str(args.reason), "prior_pause": prior}
        )
        state["status"] = "active"
        state["pause_reason"] = None
        action = next_action(state)
    emit({"study": str(path), "resumed": True, "next": action})


def command_prepare_apply(args: argparse.Namespace) -> None:
    path = study_path(args.study)
    postimage = Path(args.postimage_file).read_text(encoding="utf-8")
    with edit_state(path) as state:
        try:
            source_guard(state)
        except RuntimeError as exc:
            action = record_source_pause(state, exc)
            emit({"study": str(path), "next": action})
            return
        winner = state.get("winner")
        if not winner:
            raise RuntimeError("no holdout-confirmed winner exists")
        if not winner["delta"]:
            state["status"] = "done"
            state["apply"] = {"kind": "baseline_noop", "completed_at": utc_now()}
        else:
            preimage = state["recipe_preimage"]
            if postimage == preimage:
                raise ValueError("winner has a delta but the planned recipe postimage is unchanged")
            diff = "".join(
                difflib.unified_diff(
                    preimage.splitlines(keepends=True),
                    postimage.splitlines(keepends=True),
                    fromfile=state["recipe_path"],
                    tofile=state["recipe_path"],
                )
            )
            state["apply"] = {
                "kind": "recipe_patch",
                "postimage": postimage,
                "postimage_sha256": hashlib.sha256(postimage.encode()).hexdigest(),
                "diff": diff,
                "prepared_at": utc_now(),
            }
            state["status"] = "apply_pending"
    emit({"study": str(path), "status": state["status"], "apply": state["apply"]})


def command_complete_apply(args: argparse.Namespace) -> None:
    path = study_path(args.study)
    with edit_state(path) as state:
        if state["status"] != "apply_pending":
            raise RuntimeError("study is not apply_pending")
        try:
            source_guard(state, allow_postimage=True)
        except RuntimeError as exc:
            action = record_source_pause(state, exc)
            emit({"study": str(path), "next": action})
            return
        recipe = Path(state["repo_root"]) / state["recipe_path"]
        if file_sha256(recipe) != state["apply"]["postimage_sha256"]:
            raise RuntimeError("recipe is not the exact preregistered postimage")
        document = compose_train_document(
            Path(state["repo_root"]) / state["goal_path"],
            recipe,
        )
        expected = copy.deepcopy(state["baseline"]["train_config"])
        expected["training_backend"]["config"].update(state["winner"]["delta"])
        if document["train_config"] != expected:
            raise RuntimeError(
                "recomposition changed frozen fields or did not materialize the winner"
            )
        state["status"] = "done"
        state["apply"]["completed_at"] = utc_now()
        report = {
            "schema_version": 1,
            "study_id": state["study_id"],
            "goal_path": state["goal_path"],
            "recipe_path": state["recipe_path"],
            "source_sha": state["source_sha"],
            "runtime": state["runtime"],
            "winner": state["winner"],
            "reserved_jobs": state["reserved_jobs"],
            "completed_at": utc_now(),
        }
        atomic_json(path.parent / "report.json", report)
    emit({"study": str(path), "status": "done", "report": str(path.parent / "report.json")})


def command_finish_no_winner(args: argparse.Namespace) -> None:
    path = study_path(args.study)
    with edit_state(path) as state:
        if next_action(state)["action"] != "finish_no_winner":
            raise RuntimeError("study still has a supported next action")
        state["status"] = "done"
        state["winner"] = None
        state["no_winner_reason"] = str(args.reason)
        atomic_json(
            path.parent / "report.json",
            {
                "schema_version": 1,
                "study_id": state["study_id"],
                "winner": None,
                "reason": state["no_winner_reason"],
                "reserved_jobs": state["reserved_jobs"],
                "completed_at": utc_now(),
            },
        )
    emit({"study": str(path), "status": "done", "winner": None})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)

    initialize = commands.add_parser("init")
    initialize.add_argument("--root", default=".")
    initialize.add_argument("--goal", required=True)
    initialize.add_argument("--recipe", required=True)
    initialize.set_defaults(handler=command_init)

    for name, handler in (("status", command_status), ("next", command_next)):
        child = commands.add_parser(name)
        child.add_argument("--study", required=True)
        child.set_defaults(handler=handler)

    reserve = commands.add_parser("reserve-wave")
    reserve.add_argument("--study", required=True)
    reserve.add_argument("--phase", choices=("baseline", "search", "confirmation"), required=True)
    reserve.add_argument("--candidates-json", default="[]")
    reserve.add_argument("--effective-capacity", type=int, required=True)
    reserve.add_argument("--active-reservations", type=int, required=True)
    reserve.set_defaults(handler=command_reserve)

    launched = commands.add_parser("record-launch")
    launched.add_argument("--study", required=True)
    launched.add_argument("--submission-key", required=True)
    launched.add_argument("--payload-json")
    launched.add_argument("--payload-file")
    launched.set_defaults(handler=command_record_launch)

    terminal = commands.add_parser("record-terminal")
    terminal.add_argument("--study", required=True)
    terminal.add_argument("--submission-key")
    terminal.add_argument("--run-id", type=int)
    terminal.add_argument("--seed", type=int)
    terminal.add_argument("--event-json")
    terminal.add_argument("--event-file")
    terminal.set_defaults(handler=command_record_terminal)

    close_round = commands.add_parser("close-round")
    close_round.add_argument("--study", required=True)
    close_round.add_argument("--round", type=int, required=True)
    close_round.set_defaults(handler=command_close_round)

    close_confirmation = commands.add_parser("close-confirmation")
    close_confirmation.add_argument("--study", required=True)
    close_confirmation.set_defaults(handler=command_close_confirmation)

    attention = commands.add_parser("record-attention")
    attention.add_argument("--study", required=True)
    attention.add_argument("--event-json")
    attention.add_argument("--event-file")
    attention.set_defaults(handler=command_attention)

    resume = commands.add_parser("resume")
    resume.add_argument("--study", required=True)
    resume.add_argument("--reason", required=True)
    resume.set_defaults(handler=command_resume)

    prepare = commands.add_parser("prepare-apply")
    prepare.add_argument("--study", required=True)
    prepare.add_argument("--postimage-file", required=True)
    prepare.set_defaults(handler=command_prepare_apply)

    complete = commands.add_parser("complete-apply")
    complete.add_argument("--study", required=True)
    complete.set_defaults(handler=command_complete_apply)

    finish = commands.add_parser("finish-no-winner")
    finish.add_argument("--study", required=True)
    finish.add_argument("--reason", required=True)
    finish.set_defaults(handler=command_finish_no_winner)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.handler(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
