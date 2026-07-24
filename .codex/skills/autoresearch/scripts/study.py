#!/usr/bin/env python3
"""Durable state manager for training-signal-only autoresearch studies."""

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
from statistics import median
from typing import Any, Iterator, Mapping
from urllib.parse import urlparse

from rlab.metric_names import (
    GLOBAL_STEP,
    TRAIN_EPISODE_RETURN_SHAPED_MEAN,
    TRAIN_OUTCOME_SUCCESS_WINDOW_100_RATE_MIN,
)
from rlab.provider_config import provider_num_envs
from rlab.recipe_documents import compose_train_document
from rlab.run_contracts import RUN_ID_PATTERN
from rlab.training_backend import training_backend_id


SCHEMA_VERSION = 3
SUPPORTED_BACKENDS = frozenset({"sb3.ppo", "sb3.a2c"})
TRACE_ONLY_OVERRIDES = frozenset({"campaign_id", "description", "recipe_id"})
FROZEN_BACKEND_KEYS = frozenset({"n_steps"})
MAX_RESERVED_JOBS = 48
CONFIRMATION_RUNS = 5
PAIR_SEED_COUNT = 2
MAX_CANDIDATES_PER_WAVE = 3
DEFAULT_STALE_ROUNDS = 3
DEFAULT_STRONG_THRESHOLD = 0.90
SCREEN_FRACTION = 0.20
PAIR_FRACTION = 0.50
RETURN_TAIL_FRACTION = 0.10

EVIDENCE_SUCCESS = "success"
EVIDENCE_RETURN = "return"
EVIDENCE_MODES = frozenset({EVIDENCE_SUCCESS, EVIDENCE_RETURN})

SCREEN_PHASES = frozenset({"baseline-screen", "search-screen"})
PAIR_PHASES = frozenset({"baseline-pair", "search-pair"})
SEARCH_PHASES = SCREEN_PHASES | PAIR_PHASES
PHASES = SEARCH_PHASES | {"confirmation"}

GROUPS: dict[str, frozenset[str]] = {
    "learning_rate": frozenset(
        {"learning_rate", "learning_rate_final", "learning_rate_schedule_timesteps"}
    ),
    "entropy": frozenset({"ent_coef", "ent_coef_final", "ent_coef_schedule_timesteps"}),
    "discounting": frozenset({"gamma", "gae_lambda"}),
    "value": frozenset({"vf_coef"}),
    "ppo_update": frozenset(
        {"batch_size", "n_epochs", "clip_range", "target_kl", "adam_eps"}
    ),
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
    version = int(state.get("schema_version") or 0)
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported autoresearch study schema {version} in {path}; "
            "queue-backed v1/v2 studies are historical and cannot be resumed"
        )
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
    return float(config.get("n_epochs", 1)) * rollout / float(
        config.get("batch_size", rollout)
    )


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


def effective_cap(timesteps: int, quantum: int) -> int:
    return int(math.ceil(int(timesteps) / int(quantum)) * int(quantum))


def rung_caps(timesteps: int, quantum: int) -> dict[str, int]:
    full_effective = effective_cap(timesteps, quantum)
    screen = effective_cap(math.ceil(timesteps * SCREEN_FRACTION), quantum)
    pair = effective_cap(math.ceil(timesteps * PAIR_FRACTION), quantum)
    if not screen < pair < full_effective:
        raise ValueError(
            "autoresearch requires distinct 20%, 50%, and full effective training caps"
        )
    return {
        "screen": screen,
        "pair": pair,
        "confirmation": int(timesteps),
        "confirmation_effective": full_effective,
    }


def configured_starts(train: Mapping[str, Any]) -> list[str]:
    environment = train.get("environment") or {}
    config = environment.get("env_config") or {}
    raw_states = train.get("states") or config.get("states")
    if raw_states:
        values = [str(value) for value in raw_states]
    elif train.get("state") is not None or config.get("state") is not None:
        values = [str(train.get("state", config.get("state")))]
    else:
        raise ValueError("autoresearch requires explicit configured training start states")
    values = list(dict.fromkeys(values))
    if not values or any(not value for value in values):
        raise ValueError("autoresearch training start states must be non-empty")
    return values


def infer_evidence_mode(train: Mapping[str, Any]) -> str:
    task = train.get("task") or {}
    termination = task.get("termination") or {}
    if termination.get("success"):
        return EVIDENCE_SUCCESS
    ranking = [str(value) for value in train.get("selection_rank") or []]
    if any("episode/return" in value for value in ranking):
        return EVIDENCE_RETURN
    return EVIDENCE_SUCCESS


def evidence_mode(state: Mapping[str, Any]) -> str:
    mode = str(state.get("evidence_mode") or EVIDENCE_SUCCESS)
    if mode not in EVIDENCE_MODES:
        raise ValueError(f"unsupported autoresearch evidence mode: {mode}")
    return mode


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * float(quantile)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def return_evidence_valid(evidence: Mapping[str, Any]) -> bool:
    return bool(evidence.get("return_evidence_valid")) and all(
        evidence.get(key) is not None
        for key in ("return_tail_mean", "return_tail_p05", "return_peak")
    )


def screen_evidence_passed(state: Mapping[str, Any], evidence: Mapping[str, Any]) -> bool:
    if evidence_mode(state) == EVIDENCE_RETURN:
        return return_evidence_valid(evidence)
    return bool(evidence.get("all_starts_succeeded"))


def return_screen_key(evidence: Mapping[str, Any]) -> tuple[float, float, float]:
    if not return_evidence_valid(evidence):
        return (-float("inf"), -float("inf"), -float("inf"))
    return (
        float(evidence["return_tail_mean"]),
        float(evidence["return_tail_p05"]),
        float(evidence["return_peak"]),
    )


def selected_return_screen(
    state: Mapping[str, Any], round_number: int
) -> dict[str, Any] | None:
    screens = [
        wave
        for wave in state["waves"]
        if int(wave["round"]) == int(round_number) and wave["phase"] in SCREEN_PHASES
    ]
    if not screens or any(not wave_evidence_complete(wave) for wave in screens):
        return None
    eligible = [
        wave
        for wave in screens
        if screen_evidence_passed(state, wave["terminal_runs"][0]["training_evidence"])
    ]
    if not eligible:
        return None
    return sorted(
        eligible,
        key=lambda wave: (
            tuple(-value for value in return_screen_key(
                wave["terminal_runs"][0]["training_evidence"]
            )),
            wave["candidate_id"],
        ),
    )[0]


def candidate_score(state: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    runs = list(candidate.get("pair_runs") or [])
    if len(runs) != PAIR_SEED_COUNT:
        raise ValueError("paired candidate evidence requires exactly two runs")
    if evidence_mode(state) == EVIDENCE_RETURN:
        evidence = [run["training_evidence"] for run in runs]
        if not all(return_evidence_valid(item) for item in evidence):
            raise ValueError("paired return evidence is incomplete")
        tail_means = [float(item["return_tail_mean"]) for item in evidence]
        tail_p05s = [float(item["return_tail_p05"]) for item in evidence]
        peaks = [float(item["return_peak"]) for item in evidence]
        return {
            "evidence_mode": EVIDENCE_RETURN,
            "worst_return_tail_mean": min(tail_means),
            "median_return_tail_mean": float(median(tail_means)),
            "worst_return_tail_p05": min(tail_p05s),
            "median_return_tail_p05": float(median(tail_p05s)),
            "median_return_peak": float(median(peaks)),
            "candidate_id": candidate["id"],
        }
    censor = int(state["rung_caps"]["pair"])
    strong = 0
    crossing_steps: list[int] = []
    peaks: list[float] = []
    for run in runs:
        evidence = run["training_evidence"]
        step = evidence.get("first_strong_step")
        if bool(evidence.get("strong")) and step is not None:
            strong += 1
            crossing_steps.append(int(step))
        else:
            crossing_steps.append(censor)
        peak = evidence.get("peak_window_100_rate_min")
        peaks.append(float(peak) if peak is not None else 0.0)
    return {
        "evidence_mode": EVIDENCE_SUCCESS,
        "strong_seeds": strong,
        "median_censored_first_strong_step": float(median(crossing_steps)),
        "worst_censored_first_strong_step": max(crossing_steps),
        "worst_peak_window_100_rate_min": min(peaks),
        "candidate_id": candidate["id"],
    }


def evidence_key(score: Mapping[str, Any]) -> tuple[float, ...]:
    if str(score.get("evidence_mode") or EVIDENCE_SUCCESS) == EVIDENCE_RETURN:
        return (
            -float(score["worst_return_tail_mean"]),
            -float(score["median_return_tail_mean"]),
            -float(score["worst_return_tail_p05"]),
            -float(score["median_return_tail_p05"]),
            -float(score["median_return_peak"]),
        )
    return (
        -float(score["strong_seeds"]),
        float(score["median_censored_first_strong_step"]),
        float(score["worst_censored_first_strong_step"]),
        -float(score["worst_peak_window_100_rate_min"]),
    )


def ranked_candidates(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    excluded = set(state.get("excluded_candidates") or [])
    for candidate in state["candidates"].values():
        runs = list(candidate.get("pair_runs") or [])
        if candidate["id"] in excluded or len(runs) != PAIR_SEED_COUNT:
            continue
        if evidence_mode(state) == EVIDENCE_RETURN:
            if not all(return_evidence_valid(run.get("training_evidence") or {}) for run in runs):
                continue
        elif not all(
            bool((run.get("training_evidence") or {}).get("all_starts_succeeded"))
            for run in runs
        ):
            continue
        score = candidate_score(state, candidate)
        rows.append({"candidate": candidate, "score": score})
    return sorted(rows, key=lambda row: (*evidence_key(row["score"]), row["score"]["candidate_id"]))


def find_wave(state: Mapping[str, Any], submission_key: str) -> dict[str, Any]:
    matches = [wave for wave in state["waves"] if wave["submission_key"] == submission_key]
    if len(matches) != 1:
        raise ValueError(f"submission key maps to {len(matches)} waves")
    return matches[0]


def candidate_wave(
    state: Mapping[str, Any], candidate_id_value: str, phases: frozenset[str]
) -> dict[str, Any] | None:
    matches = [
        wave
        for wave in state["waves"]
        if wave["candidate_id"] == candidate_id_value and wave["phase"] in phases
    ]
    if len(matches) > 1:
        raise RuntimeError(f"candidate {candidate_id_value} has duplicate rung waves")
    return matches[0] if matches else None


def wave_evidence_complete(wave: Mapping[str, Any]) -> bool:
    return len(wave.get("terminal_runs") or []) == len(wave["seeds"]) and all(
        item.get("training_evidence") is not None for item in wave.get("terminal_runs") or []
    )


def round_ready_to_close(state: Mapping[str, Any], round_number: int) -> bool:
    screens = [
        wave
        for wave in state["waves"]
        if int(wave["round"]) == round_number and wave["phase"] in SCREEN_PHASES
    ]
    if not screens or any(not wave_evidence_complete(wave) for wave in screens):
        return False
    if evidence_mode(state) == EVIDENCE_RETURN:
        selected = selected_return_screen(state, round_number)
        if selected is None:
            return True
        pair = candidate_wave(state, selected["candidate_id"], PAIR_PHASES)
        return pair is not None and wave_evidence_complete(pair)
    for screen in screens:
        record = screen["terminal_runs"][0]
        if bool(record["training_evidence"]["all_starts_succeeded"]):
            pair = candidate_wave(state, screen["candidate_id"], PAIR_PHASES)
            if pair is None or not wave_evidence_complete(pair):
                return False
    return True


def next_action(state: Mapping[str, Any]) -> dict[str, Any]:
    if state["status"] == "paused":
        return {"action": "pause", "reason": state.get("pause_reason")}
    if state["status"] == "done":
        return {"action": "done", "winner": state.get("winner")}
    if state["status"] == "apply_pending":
        return {"action": "apply_postimage", "winner": state.get("winner")}
    open_waves = [wave for wave in state["waves"] if not wave.get("closed")]
    for wave in open_waves:
        if wave["status"] == "reserved":
            return {
                "action": "reconcile_submission",
                "submission_key": wave["submission_key"],
                "seeds": list(wave["seeds"]),
            }
    for wave in open_waves:
        for run in wave.get("terminal_runs") or []:
            if run.get("training_evidence") is None:
                return {"action": "collect_training_evidence", "run_id": str(run["run_id"])}

    baseline = state["candidates"][state["baseline_candidate_id"]]
    baseline_screen = candidate_wave(state, baseline["id"], SCREEN_PHASES)
    if baseline_screen is None:
        return {"action": "reserve_baseline_screen"}
    if wave_evidence_complete(baseline_screen):
        screen_passed = screen_evidence_passed(
            state, baseline_screen["terminal_runs"][0]["training_evidence"]
        )
        if screen_passed and candidate_wave(state, baseline["id"], PAIR_PHASES) is None:
            return {"action": "reserve_baseline_pair", "candidate_id": baseline["id"]}
    if not baseline_screen.get("closed") and round_ready_to_close(state, 0):
        return {"action": "close_round", "round": 0}

    current_round = int(state["search_round"]) + 1
    screens = [
        wave
        for wave in state["waves"]
        if int(wave["round"]) == current_round and wave["phase"] == "search-screen"
    ]
    if evidence_mode(state) == EVIDENCE_RETURN:
        selected = selected_return_screen(state, current_round)
        if selected is not None and candidate_wave(
            state, selected["candidate_id"], PAIR_PHASES
        ) is None:
            return {"action": "reserve_search_pair", "candidate_id": selected["candidate_id"]}
    else:
        for screen in screens:
            if not wave_evidence_complete(screen):
                continue
            passed = bool(
                screen["terminal_runs"][0]["training_evidence"]["all_starts_succeeded"]
            )
            if passed and candidate_wave(state, screen["candidate_id"], PAIR_PHASES) is None:
                return {"action": "reserve_search_pair", "candidate_id": screen["candidate_id"]}

    awaiting = [
        str(run_id)
        for wave in open_waves
        if wave["status"] == "launched"
        for run_id in wave["run_ids"]
        if str(run_id)
        not in {str(item["run_id"]) for item in wave.get("terminal_runs") or []}
    ]
    if awaiting:
        return {"action": "await_runs", "run_ids": sorted(awaiting)}

    if screens and round_ready_to_close(state, current_round):
        return {"action": "close_round", "round": current_round}

    confirmation = state.get("confirmation")
    if confirmation and not confirmation.get("closed"):
        return {"action": "close_confirmation", "candidate_id": confirmation["candidate_id"]}
    if state.get("winner"):
        return {"action": "prepare_winner", "winner": state["winner"]}

    remaining = MAX_RESERVED_JOBS - int(state["reserved_jobs"])
    stop_search = int(state["stale_rounds"]) >= int(state["policy"]["stale_round_limit"])
    stop_search = stop_search or remaining < (1 + PAIR_SEED_COUNT + CONFIRMATION_RUNS)
    if stop_search:
        if state.get("incumbent_candidate_id") and remaining >= CONFIRMATION_RUNS:
            return {
                "action": "reserve_confirmation",
                "candidate_id": state["incumbent_candidate_id"],
            }
        return {"action": "finish_no_winner", "reason": "search budget or evidence exhausted"}
    if screens:
        raise RuntimeError("current search round is incomplete but has no deterministic action")
    return {
        "action": "propose_search",
        "round": current_round,
        "max_candidates": min(
            MAX_CANDIDATES_PER_WAVE,
            max(0, (remaining - CONFIRMATION_RUNS) // (1 + PAIR_SEED_COUNT)),
        ),
    }


def command_init(args: argparse.Namespace) -> None:
    threshold = float(args.strong_threshold)
    if not 0.0 < threshold <= 1.0:
        raise ValueError("--strong-threshold must be greater than zero and at most one")
    root = Path(args.root).resolve()
    goal = Path(args.goal)
    recipe = Path(args.recipe)
    goal = (root / goal).resolve() if not goal.is_absolute() else goal.resolve()
    recipe = (root / recipe).resolve() if not recipe.is_absolute() else recipe.resolve()
    head = git_head(root)
    goal_path = relative(root, goal)
    recipe_path = relative(root, recipe)
    input_hash = digest(
        {
            "schema_version": SCHEMA_VERSION,
            "goal": goal_path,
            "recipe": recipe_path,
            "source_sha": head,
            "strong_threshold": threshold,
        }
    )
    studies_root = root / "runs" / "autoresearch"
    with lock(studies_root / ".discovery.lock"):
        matches: list[Path] = []
        current_leaf = file_sha256(recipe)
        for path in sorted(studies_root.glob("*/study.json")):
            raw = json.loads(path.read_text(encoding="utf-8"))
            if int(raw.get("schema_version") or 0) != SCHEMA_VERSION:
                continue
            if raw.get("status") == "done":
                continue
            apply = raw.get("apply") or {}
            recognized = current_leaf in {
                raw.get("recipe_preimage_sha256"),
                apply.get("postimage_sha256"),
            }
            if raw.get("input_hash") == input_hash or (
                raw.get("source_sha") == head
                and raw.get("goal_path") == goal_path
                and raw.get("recipe_path") == recipe_path
                and float((raw.get("policy") or {}).get("strong_threshold", -1)) == threshold
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
                f"autoresearch v2 supports only {sorted(SUPPORTED_BACKENDS)}, got {backend}"
            )
        composition = document.get("_composition") or {}
        sources = [
            {"path": relative(root, item["path"]), "sha256": str(item["sha256"])}
            for item in composition.get("source_files") or []
        ]
        if recipe_path not in {item["path"] for item in sources}:
            sources.append({"path": recipe_path, "sha256": file_sha256(recipe)})
        for item in sources:
            committed_hash = git_blob_sha256(root, head, item["path"])
            if committed_hash != item["sha256"] or file_sha256(root / item["path"]) != committed_hash:
                raise RuntimeError(
                    "composition source differs from committed HEAD and cannot be launched "
                    f"as exact committed source: {item['path']}"
                )

        backend_config = copy.deepcopy(train["training_backend"]["config"])
        n_envs = provider_num_envs(train, explicit_n_envs=train.get("n_envs"))
        quantum = int(backend_config["n_steps"]) * int(n_envs)
        caps = rung_caps(int(train["timesteps"]), quantum)
        starts = configured_starts(train)
        mode = infer_evidence_mode(train)
        screen_seed = 123
        pair_seeds = [123 + index * n_envs for index in range(1, 1 + PAIR_SEED_COUNT)]
        confirmation_seeds = [
            123 + index * n_envs
            for index in range(1 + PAIR_SEED_COUNT, 1 + PAIR_SEED_COUNT + CONFIRMATION_RUNS)
        ]
        baseline_id = candidate_id({})
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        directory = studies_root / f"{stamp}-{input_hash[:12]}"
        directory.mkdir(parents=True, exist_ok=False)
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
            "configured_starts": starts,
            "evidence_mode": mode,
            "screen_seed": screen_seed,
            "pair_seeds": pair_seeds,
            "confirmation_seeds": confirmation_seeds,
            "rung_caps": caps,
            "runtime": None,
            "policy": {
                "compute_target": "b3",
                "max_reserved_jobs": MAX_RESERVED_JOBS,
                "stale_round_limit": DEFAULT_STALE_ROUNDS,
                "confirmation_runs": CONFIRMATION_RUNS,
                "confirmation_required": 4,
                "strong_threshold": threshold,
                "return_tail_fraction": RETURN_TAIL_FRACTION,
                "checkpoint_evaluation": "disabled",
            },
            "baseline": {
                "backend_config": backend_config,
                "tunables": numeric_tunables(backend_config),
                "train_config": copy.deepcopy(train),
                "update_work_per_env_step": update_work(backend_config, backend, n_envs),
                "rollout_quantum": quantum,
            },
            "baseline_candidate_id": baseline_id,
            "candidates": {
                baseline_id: {
                    "id": baseline_id,
                    "delta": {},
                    "created_round": 0,
                    "screen_runs": [],
                    "pair_runs": [],
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
    overrides.append(f"train.timesteps={int(wave['timesteps'])}")
    return overrides


def run_description(
    state: Mapping[str, Any],
    wave: Mapping[str, Any],
    seed: int,
) -> str:
    return (
        f"Autoresearch {state['study_id'][:8]} {wave['phase']} candidate "
        f"{wave['candidate_id']} seed {int(seed)}. "
        "Training-only evidence; no evaluation or promotion."
    )


def materialized_recipe_overrides(
    state: Mapping[str, Any], wave: Mapping[str, Any], seed: int
) -> list[str]:
    del seed
    return expected_recipe_overrides(state, wave)


def launch_command(
    state: Mapping[str, Any],
    wave: Mapping[str, Any],
    seed: int,
) -> list[str]:
    command = [
        "rlab",
        "experiment",
        "launch",
        "--goal-file",
        state["goal_path"],
        "--recipe-file",
        state["recipe_path"],
        "--seed",
        str(int(seed)),
        "--run-description",
        run_description(state, wave, seed),
        "--compute",
        "local",
        "--target",
        state["policy"]["compute_target"],
        "--submission-key",
        wave["submission_key"],
        "--existing-runtime-only",
        "--checkpoint-eval-backend",
        "none",
    ]
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
    timesteps: int,
) -> dict[str, Any]:
    delta = dict(candidate.get("delta") or {})
    if phase.startswith("baseline"):
        identifier = state["baseline_candidate_id"]
    else:
        delta = validate_delta(state, delta) if phase == "search-screen" else delta
        identifier = candidate_id(delta)
        state["candidates"].setdefault(
            identifier,
            {
                "id": identifier,
                "delta": delta,
                "created_round": round_number,
                "screen_runs": [],
                "pair_runs": [],
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
        "timesteps": int(timesteps),
        "submission_key": submission_key,
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
        action = next_action(state)
        phase = args.phase
        expected_actions = {
            "baseline-screen": "reserve_baseline_screen",
            "baseline-pair": "reserve_baseline_pair",
            "search-screen": "propose_search",
            "search-pair": "reserve_search_pair",
            "confirmation": "reserve_confirmation",
        }
        if action["action"] != expected_actions[phase]:
            raise RuntimeError(f"state expects {action['action']}, not {phase} reservation")
        free = max(int(args.effective_capacity) - int(args.active_reservations), 0)
        if free <= 0:
            raise RuntimeError("B3 has no available slot; wait without reserving a wave")

        candidates = parse_candidates(args.candidates_json)
        if phase == "baseline-screen":
            candidates = [{"delta": {}}]
            seeds = [int(state["screen_seed"])]
            timesteps = int(state["rung_caps"]["screen"])
            round_number = 0
        elif phase == "baseline-pair":
            candidates = [{"delta": {}}]
            seeds = list(state["pair_seeds"])
            timesteps = int(state["rung_caps"]["pair"])
            round_number = 0
        elif phase == "search-screen":
            seeds = [int(state["screen_seed"])]
            timesteps = int(state["rung_caps"]["screen"])
            round_number = int(state["search_round"]) + 1
            remaining = MAX_RESERVED_JOBS - int(state["reserved_jobs"])
            cohort_limit = min(
                MAX_CANDIDATES_PER_WAVE,
                free,
                max(0, (remaining - CONFIRMATION_RUNS) // (1 + PAIR_SEED_COUNT)),
            )
            if cohort_limit <= 0:
                raise RuntimeError("job budget cannot support another complete screened candidate")
            if len(candidates) != cohort_limit:
                raise ValueError(f"provide exactly {cohort_limit} new screen candidates")
        elif phase == "search-pair":
            candidate_value = str(args.candidate_id or action.get("candidate_id") or "")
            if candidate_value != str(action.get("candidate_id") or ""):
                raise ValueError("--candidate-id does not match the deterministic next action")
            incumbent = state["candidates"].get(candidate_value)
            if not incumbent:
                raise ValueError(f"unknown candidate: {candidate_value}")
            candidates = [{"delta": incumbent["delta"]}]
            seeds = list(state["pair_seeds"])
            timesteps = int(state["rung_caps"]["pair"])
            round_number = int(incumbent["created_round"])
        else:
            candidate_value = str(state["incumbent_candidate_id"] or "")
            incumbent = state["candidates"].get(candidate_value)
            if not incumbent:
                raise RuntimeError("confirmation requires a ranked incumbent")
            candidates = [{"delta": incumbent["delta"]}]
            seeds = list(state["confirmation_seeds"])
            timesteps = int(state["rung_caps"]["confirmation"])
            round_number = int(state["search_round"])

        if not candidates:
            raise ValueError("at least one candidate is required")
        candidates = sorted(candidates, key=lambda item: candidate_id(item.get("delta") or {}))
        identifiers = [candidate_id(item.get("delta") or {}) for item in candidates]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("a wave cannot reserve the same candidate twice")
        if phase == "search-screen":
            repeated = sorted(set(identifiers) & set(state["candidates"]))
            if repeated:
                raise ValueError(f"search candidates must be new to the study: {repeated}")
        needed = len(candidates) * len(seeds)
        reserve_for_confirmation = 0 if phase == "confirmation" else CONFIRMATION_RUNS
        if int(state["reserved_jobs"]) + needed + reserve_for_confirmation > MAX_RESERVED_JOBS:
            raise RuntimeError("reservation would consume the confirmation reserve or exceed 48 jobs")
        waves = [
            reserve_one(
                state,
                phase=phase,
                round_number=round_number,
                candidate=candidate,
                seeds=seeds,
                timesteps=timesteps,
            )
            for candidate in candidates
        ]
        if phase == "confirmation":
            confirmation_floor = None
            if evidence_mode(state) == EVIDENCE_RETURN:
                confirmation_floor = float(
                    candidate_score(state, state["candidates"][candidate_value])[
                        "worst_return_tail_mean"
                    ]
                )
            state["confirmation"] = {
                "candidate_id": waves[0]["candidate_id"],
                "submission_key": waves[0]["submission_key"],
                "closed": False,
                "return_floor": confirmation_floor,
            }
        output = [
            {
                "candidate_id": wave["candidate_id"],
                "submission_key": wave["submission_key"],
                "commands": [
                    {
                        "seed": int(seed),
                        "command": launch_command(state, wave, int(seed)),
                        "shell_command": shlex.join(
                            launch_command(state, wave, int(seed))
                        ),
                    }
                    for seed in wave["seeds"]
                ],
            }
            for wave in waves
        ]
    emit({"study": str(path), "reserved": output, "launch_concurrently": False})


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
        rows = payload.get("runs")
        if rows is None and payload.get("run_id"):
            rows = [payload]
        rows = [dict(row) for row in rows or []]
        run_ids = [str(row.get("run_id") or "") for row in rows]
        if len(run_ids) != len(wave["seeds"]) or len(set(run_ids)) != len(run_ids):
            errors.append("submission is partial or has an unexpected run count")
        if any(RUN_ID_PATTERN.fullmatch(run_id) is None for run_id in run_ids):
            errors.append("submission contains a malformed immutable run id")
        if rows:
            observed_seeds = sorted(int(row.get("seed")) for row in rows)
            if observed_seeds != sorted(wave["seeds"]):
                errors.append("submission rows do not match the reserved training seeds")
            observed_keys = {str(row.get("submission_key") or "") for row in rows}
            if observed_keys != {wave["submission_key"]}:
                errors.append("submission rows do not match the reserved submission key")
            observed_sources = {str(row.get("source_sha") or "") for row in rows}
            if observed_sources != {state["source_sha"]}:
                errors.append("submission rows do not match the pinned source revision")
            goal_paths = {str(row.get("goal_file") or "") for row in rows}
            recipe_paths = {str(row.get("recipe_file") or "") for row in rows}
            if goal_paths != {state["goal_path"]} or recipe_paths != {state["recipe_path"]}:
                errors.append("submission rows do not match the pinned goal and recipe paths")
            if any(
                list(row.get("recipe_overrides") or [])
                != expected_recipe_overrides(state, wave)
                for row in rows
            ):
                errors.append("submission rows do not match the reserved recipe overrides")
            if any(
                str(row.get("run_description") or "")
                != run_description(state, wave, int(row["seed"]))
                for row in rows
            ):
                errors.append("submission rows do not match the reserved descriptions")
            if any(
                str(row.get("checkpoint_eval_backend") or "") != "none"
                for row in rows
            ):
                errors.append("submission row did not materialize training-only execution")
        row_runtimes = {
            (
                str(row.get("image_digest") or ""),
                str(row.get("runtime_input_sha256") or ""),
                str(row.get("runtime_build_source_sha") or ""),
            )
            for row in rows
        }
        runtime = {
            "image_ref": next(iter(row_runtimes))[0] if len(row_runtimes) == 1 else "",
            "input_sha256": next(iter(row_runtimes))[1] if len(row_runtimes) == 1 else "",
            "build_source_sha": next(iter(row_runtimes))[2] if len(row_runtimes) == 1 else "",
        }
        if rows and len(row_runtimes) != 1:
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
    semantic = payload.get("semantic") or {}
    manifest = semantic.get("manifest") or {}
    submission_key = str(
        args.submission_key
        or (manifest.get("compute") or {}).get("submission_key")
        or ""
    )
    run_id = str(args.run_id or payload.get("run_id") or "")
    seed = int(args.seed if args.seed is not None else manifest.get("seed"))
    if not submission_key:
        raise ValueError("terminal event lacks the autoresearch submission key")
    if RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError("terminal event lacks a valid immutable run id")
    path = study_path(args.study)
    with edit_state(path) as state:
        wave = find_wave(state, submission_key)
        if run_id not in wave["run_ids"]:
            raise RuntimeError("terminal run is not part of the recorded launch")
        if seed not in wave["seeds"]:
            raise RuntimeError("terminal seed is not part of the reservation")
        if any(str(item["run_id"]) == run_id for item in wave["terminal_runs"]):
            emit({"study": str(path), "duplicate": True, "run_id": run_id})
            return
        terminal = payload.get("attempt_terminal") or {}
        drain = terminal.get("drain") or {}
        wandb = manifest.get("wandb") or {}
        classification = (
            "completed" if str(terminal.get("state") or "") == "succeeded" else "failed"
        )
        high_water = int(terminal.get("wandb_high_water_mark") or 0)
        remote_high_water = int(drain.get("wandb_remote_high_water_mark") or 0)
        valid = (
            classification == "completed"
            and terminal.get("acceptance_required") is False
            and drain.get("complete") is True
            and high_water > 0
            and remote_high_water >= high_water
            and bool(wandb.get("url"))
            and bool(wandb.get("run_id"))
            and semantic.get("terminal") is None
            and payload.get("scientific_success") is False
            and bool((payload.get("dstack") or {}).get("terminal"))
        )
        record = {
            "run_id": run_id,
            "seed": seed,
            "classification": classification,
            "wandb_run_id": wandb.get("run_id"),
            "wandb_url": wandb.get("url"),
            "training_evidence": None,
        }
        wave["terminal_runs"].append(record)
        wave["terminal_runs"].sort(key=lambda item: item["seed"])
        if not valid:
            state["status"] = "paused"
            state["pause_reason"] = {
                "event": "terminal_without_valid_training_evidence_source",
                "run_id": run_id,
                "classification": classification,
                "acceptance_required": terminal.get("acceptance_required"),
                "drain_complete": drain.get("complete"),
                "wandb_high_water_mark": high_water,
                "wandb_remote_high_water_mark": remote_high_water,
                "scientific_success": payload.get("scientific_success"),
                "dstack_terminal": bool((payload.get("dstack") or {}).get("terminal")),
            }
        elif len(wave["terminal_runs"]) == len(wave["seeds"]):
            wave["status"] = "awaiting_evidence"
    emit({"study": str(path), "run_id": run_id, "next": next_action(load_state(path))})


def _summary_scalar(value: Any) -> Any:
    if not isinstance(value, dict) and hasattr(value, "keys"):
        value = dict(value)
    if isinstance(value, dict) and len(value) == 1:
        return next(iter(value.values()))
    return value


def _wandb_run_path(url: str) -> str:
    parts = [part for part in urlparse(url).path.split("/") if part]
    if len(parts) < 4 or parts[-2] != "runs":
        raise ValueError(f"unrecognized W&B run URL: {url}")
    return f"{parts[0]}/{parts[1]}/{parts[-1]}"


def fetch_training_evidence(
    *,
    url: str,
    expected_run_id: str,
    starts: list[str],
    strong_threshold: float,
    mode: str = EVIDENCE_SUCCESS,
    return_tail_fraction: float = RETURN_TAIL_FRACTION,
) -> dict[str, Any]:
    from rlab.wandb_utils import load_wandb_env

    load_wandb_env()
    import wandb

    run = wandb.Api().run(_wandb_run_path(url))
    if str(getattr(run, "id", "") or "") != str(expected_run_id):
        raise RuntimeError("W&B run identity does not match the recorded dstack run")
    count_keys = [f"train/outcome/success/from/{start}/count" for start in starts]
    keys = [
        GLOBAL_STEP,
        *count_keys,
        TRAIN_OUTCOME_SUCCESS_WINDOW_100_RATE_MIN,
        TRAIN_EPISODE_RETURN_SHAPED_MEAN,
    ]
    history = [dict(row) for row in run.scan_history(keys=keys)]
    if not history:
        raise RuntimeError("W&B run has no remotely visible training history")
    counts = {
        start: max(
            (
                int(row.get(metric) or 0)
                for row in history
                if row.get(metric) is not None
            ),
            default=0,
        )
        for start, metric in zip(starts, count_keys, strict=True)
    }
    all_starts_succeeded = all(value > 0 for value in counts.values())
    rate_rows = [
        (int(row.get(GLOBAL_STEP) or 0), float(row[TRAIN_OUTCOME_SUCCESS_WINDOW_100_RATE_MIN]))
        for row in history
        if row.get(TRAIN_OUTCOME_SUCCESS_WINDOW_100_RATE_MIN) is not None
    ]
    peak = max((value for _step, value in rate_rows), default=None)
    first_strong_step = next(
        (
            step
            for step, value in sorted(rate_rows)
            if value >= float(strong_threshold)
        ),
        None,
    )
    observed_max_step = max((int(row.get(GLOBAL_STEP) or 0) for row in history), default=0)
    returns = [
        float(row[TRAIN_EPISODE_RETURN_SHAPED_MEAN])
        for row in history
        if row.get(TRAIN_EPISODE_RETURN_SHAPED_MEAN) is not None
    ]
    tail_count = max(1, math.ceil(len(returns) * float(return_tail_fraction))) if returns else 0
    tail_values = returns[-tail_count:] if tail_count else []
    tail_mean_raw = sum(tail_values) / len(tail_values) if tail_values else None
    return_valid = tail_mean_raw is not None and tail_count >= 10
    return_points = len(returns)
    sorted_tail = sorted(tail_values)
    tail_p05 = (
        sorted_tail[max(0, math.ceil(len(sorted_tail) * 0.05) - 1)]
        if sorted_tail
        else None
    )
    tail_std = (
        math.sqrt(
            sum((value - float(tail_mean_raw)) ** 2 for value in tail_values)
            / len(tail_values)
        )
        if tail_values
        else None
    )
    result = {
        "evidence_mode": mode,
        "all_starts_succeeded": all_starts_succeeded,
        "success_counts_by_start": counts,
        "peak_window_100_rate_min": peak,
        "first_strong_step": first_strong_step,
        "strong": first_strong_step is not None,
        "observed_max_step": observed_max_step,
        "strong_threshold": strong_threshold,
        "wandb_run_id": str(expected_run_id),
        "wandb_url": str(getattr(run, "url", "") or url),
        "wandb_state": str(getattr(run, "state", "") or "unknown"),
        "authority": "wandb_history",
        "rank_direction": "maximize",
        "collected_at": utc_now(),
        "return_metric": TRAIN_EPISODE_RETURN_SHAPED_MEAN,
        "return_points": return_points,
        "return_tail_fraction": float(return_tail_fraction),
        "return_tail_points": tail_count,
        "return_evidence_valid": return_valid,
        "return_peak": max(returns, default=None),
        "return_last": returns[-1] if returns else None,
        "return_tail_mean": tail_mean_raw,
        "return_tail_p05": tail_p05,
        "return_tail_std": tail_std,
    }
    if mode == EVIDENCE_RETURN and not return_valid:
        result["strong"] = False
    return result


def command_collect_training_evidence(args: argparse.Namespace) -> None:
    path = study_path(args.study)
    state = load_state(path)
    if state["status"] != "active":
        raise RuntimeError(f"cannot collect evidence while study is {state['status']}")
    try:
        source_guard(state)
    except RuntimeError as exc:
        with edit_state(path) as current:
            action = record_source_pause(current, exc)
        emit({"study": str(path), "next": action})
        return
    matches: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for wave in state["waves"]:
        for record in wave.get("terminal_runs") or []:
            if str(record["run_id"]) == str(args.run_id):
                matches.append((wave, record))
    if len(matches) != 1:
        raise ValueError(f"run {args.run_id} maps to {len(matches)} terminal records")
    _, record = matches[0]
    if record.get("training_evidence") is not None:
        emit({"study": str(path), "run_id": str(args.run_id), "duplicate": True})
        return
    evidence = fetch_training_evidence(
        url=str(record["wandb_url"]),
        expected_run_id=str(record["wandb_run_id"]),
        starts=list(state["configured_starts"]),
        strong_threshold=float(state["policy"]["strong_threshold"]),
        mode=evidence_mode(state),
        return_tail_fraction=float(
            state["policy"].get("return_tail_fraction", RETURN_TAIL_FRACTION)
        ),
    )
    with edit_state(path) as current:
        source_guard(current)
        current_matches: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for wave in current["waves"]:
            for item in wave.get("terminal_runs") or []:
                if str(item["run_id"]) == str(args.run_id):
                    current_matches.append((wave, item))
        if len(current_matches) != 1:
            raise RuntimeError("terminal record changed while collecting W&B evidence")
        wave, item = current_matches[0]
        if item.get("training_evidence") is not None:
            emit({"study": str(path), "run_id": str(args.run_id), "duplicate": True})
            return
        if str(item["wandb_run_id"]) != evidence["wandb_run_id"]:
            raise RuntimeError("W&B identity changed while collecting evidence")
        item["training_evidence"] = evidence
        candidate = current["candidates"][wave["candidate_id"]]
        destination = (
            "screen_runs"
            if wave["phase"] in SCREEN_PHASES
            else "pair_runs"
            if wave["phase"] in PAIR_PHASES
            else "confirmation_runs"
        )
        candidate[destination].append(copy.deepcopy(item))
        candidate[destination].sort(key=lambda value: (value["seed"], value["run_id"]))
        if wave_evidence_complete(wave):
            wave["status"] = "evidence_complete"
    emit(
        {
            "study": str(path),
            "run_id": str(args.run_id),
            "training_evidence": evidence,
            "next": next_action(load_state(path)),
        }
    )


def command_upgrade_return_mode(args: argparse.Namespace) -> None:
    path = study_path(args.study)
    state = load_state(path)
    source_guard(state)
    if evidence_mode(state) == EVIDENCE_RETURN:
        emit({"study": str(path), "duplicate": True, "next": next_action(state)})
        return
    if state["status"] != "active":
        raise RuntimeError(f"cannot upgrade evidence while study is {state['status']}")
    if infer_evidence_mode(state["baseline"]["train_config"]) != EVIDENCE_RETURN:
        raise RuntimeError("the frozen goal is not a return-only training objective")
    if state.get("confirmation") or state.get("winner") or state.get("apply"):
        raise RuntimeError("return-mode upgrade must precede confirmation or winner application")
    if any(candidate.get("pair_runs") for candidate in state["candidates"].values()):
        raise RuntimeError("return-mode upgrade must precede paired evidence")

    records: dict[str, dict[str, Any]] = {}
    for wave in state["waves"]:
        for item in wave.get("terminal_runs") or []:
            run_id = str(item["run_id"])
            records[run_id] = fetch_training_evidence(
                url=str(item["wandb_url"]),
                expected_run_id=str(item["wandb_run_id"]),
                starts=list(state["configured_starts"]),
                strong_threshold=float(state["policy"]["strong_threshold"]),
                mode=EVIDENCE_RETURN,
                return_tail_fraction=RETURN_TAIL_FRACTION,
            )

    with edit_state(path) as current:
        source_guard(current)
        if evidence_mode(current) == EVIDENCE_RETURN:
            emit({"study": str(path), "duplicate": True, "next": next_action(current)})
            return
        current["evidence_mode"] = EVIDENCE_RETURN
        current["policy"]["return_tail_fraction"] = RETURN_TAIL_FRACTION
        for wave in current["waves"]:
            if wave["phase"] in SEARCH_PHASES:
                wave["closed"] = False
            for item in wave.get("terminal_runs") or []:
                item["training_evidence"] = copy.deepcopy(records[str(item["run_id"])])
        for candidate in current["candidates"].values():
            for destination in ("screen_runs", "pair_runs", "confirmation_runs"):
                for item in candidate.get(destination) or []:
                    run_id = item.get("run_id")
                    if run_id is not None and str(run_id) in records:
                        item["training_evidence"] = copy.deepcopy(records[str(run_id)])
        current["search_round"] = 0
        current["stale_rounds"] = 0
        current["incumbent_candidate_id"] = None
        current["incumbent_evidence"] = None
        current.setdefault("migration_history", []).append(
            {
                "event": "success_to_return_evidence",
                "migrated_at": utc_now(),
                "run_ids": sorted(records),
            }
        )
        action = next_action(current)
    emit(
        {
            "study": str(path),
            "evidence_mode": EVIDENCE_RETURN,
            "migrated_run_ids": sorted(records),
            "next": action,
        }
    )


def command_close_round(args: argparse.Namespace) -> None:
    path = study_path(args.study)
    with edit_state(path) as state:
        round_number = int(args.round)
        waves = [
            wave
            for wave in state["waves"]
            if int(wave["round"]) == round_number and wave["phase"] in SEARCH_PHASES
        ]
        if not waves or not round_ready_to_close(state, round_number):
            raise RuntimeError("round barrier requires complete screen and promoted-pair evidence")
        if any(wave.get("closed") for wave in waves):
            raise RuntimeError("round was already closed")
        ranked = ranked_candidates(state)
        best = ranked[0] if ranked else None
        prior = state.get("incumbent_evidence")
        improved = bool(best) and (prior is None or evidence_key(best["score"]) < evidence_key(prior))
        if round_number > 0:
            state["search_round"] = round_number
            state["stale_rounds"] = 0 if improved else int(state["stale_rounds"]) + 1
        if best:
            state["incumbent_candidate_id"] = best["candidate"]["id"]
            state["incumbent_evidence"] = best["score"]
        for wave in waves:
            wave["closed"] = True
    emit(
        {
            "study": str(path),
            "ranking": [row["score"] for row in ranked],
            "improved": improved,
            "next": next_action(load_state(path)),
        }
    )


def command_close_confirmation(args: argparse.Namespace) -> None:
    path = study_path(args.study)
    with edit_state(path) as state:
        confirmation = state.get("confirmation")
        if not confirmation or confirmation.get("closed"):
            raise RuntimeError("no open confirmation exists")
        wave = find_wave(state, confirmation["submission_key"])
        if not wave_evidence_complete(wave) or len(wave["terminal_runs"]) != CONFIRMATION_RUNS:
            raise RuntimeError("confirmation barrier requires five remotely evidenced runs")
        if evidence_mode(state) == EVIDENCE_RETURN:
            floor = confirmation.get("return_floor")
            if floor is None:
                raise RuntimeError("return confirmation lacks its frozen paired-evidence floor")
            strong = sum(
                return_evidence_valid(item["training_evidence"])
                and float(item["training_evidence"]["return_tail_mean"]) >= float(floor)
                for item in wave["terminal_runs"]
            )
        else:
            strong = sum(
                bool(item["training_evidence"]["strong"]) for item in wave["terminal_runs"]
            )
        confirmation["closed"] = True
        confirmation["strong_seed_count"] = strong
        candidate_id_value = confirmation["candidate_id"]
        if strong >= int(state["policy"]["confirmation_required"]):
            wave["closed"] = True
            state["winner"] = {
                "candidate_id": candidate_id_value,
                "delta": state["candidates"][candidate_id_value]["delta"],
                "training_signal_confirmed": True,
                "strong_seed_count": strong,
                "total_seeds": CONFIRMATION_RUNS,
                "strong_threshold": state["policy"]["strong_threshold"],
                "evidence_mode": evidence_mode(state),
                "return_floor": confirmation.get("return_floor"),
                "runs": wave["terminal_runs"],
            }
        else:
            state["excluded_candidates"].append(candidate_id_value)
            state["stale_rounds"] = 0
            redacted = {
                "redacted": True,
                "strong_seed_count": strong,
                "total": CONFIRMATION_RUNS,
            }
            state["candidates"][candidate_id_value]["confirmation_runs"] = [redacted]
            wave["terminal_runs"] = [redacted]
            wave["closed"] = True
            ranked = ranked_candidates(state)
            state["incumbent_candidate_id"] = ranked[0]["candidate"]["id"] if ranked else None
            state["incumbent_evidence"] = ranked[0]["score"] if ranked else None
            state["confirmation"] = None
    emit(
        {
            "study": str(path),
            "strong_seed_count": strong,
            "training_signal_confirmed": strong
            >= int(state["policy"]["confirmation_required"]),
            "next": next_action(load_state(path)),
        }
    )


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
        if not winner or not winner.get("training_signal_confirmed"):
            raise RuntimeError("no training-signal-confirmed winner exists")
        if not winner["delta"]:
            state["status"] = "done"
            state["apply"] = {"kind": "baseline_noop", "completed_at": utc_now()}
            atomic_json(
                path.parent / "report.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "study_id": state["study_id"],
                    "goal_path": state["goal_path"],
                    "recipe_path": state["recipe_path"],
                    "source_sha": state["source_sha"],
                    "runtime": state["runtime"],
                    "rung_caps": state["rung_caps"],
                    "checkpoint_evaluation": "disabled",
                    "checkpoint_promoted": False,
                    "goal_accepted": False,
                    "winner": winner,
                    "reserved_jobs": state["reserved_jobs"],
                    "completed_at": utc_now(),
                },
            )
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
        document = compose_train_document(Path(state["repo_root"]) / state["goal_path"], recipe)
        expected = copy.deepcopy(state["baseline"]["train_config"])
        expected["training_backend"]["config"].update(state["winner"]["delta"])
        if document["train_config"] != expected:
            raise RuntimeError("recomposition changed frozen fields or did not materialize the winner")
        state["status"] = "done"
        state["apply"]["completed_at"] = utc_now()
        report = {
            "schema_version": SCHEMA_VERSION,
            "study_id": state["study_id"],
            "goal_path": state["goal_path"],
            "recipe_path": state["recipe_path"],
            "source_sha": state["source_sha"],
            "runtime": state["runtime"],
            "rung_caps": state["rung_caps"],
            "checkpoint_evaluation": "disabled",
            "checkpoint_promoted": False,
            "goal_accepted": False,
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
                "schema_version": SCHEMA_VERSION,
                "study_id": state["study_id"],
                "winner": None,
                "checkpoint_evaluation": "disabled",
                "checkpoint_promoted": False,
                "goal_accepted": False,
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
    initialize.add_argument("--strong-threshold", type=float, default=DEFAULT_STRONG_THRESHOLD)
    initialize.set_defaults(handler=command_init)

    for name, handler in (("status", command_status), ("next", command_next)):
        child = commands.add_parser(name)
        child.add_argument("--study", required=True)
        child.set_defaults(handler=handler)

    reserve = commands.add_parser("reserve-wave")
    reserve.add_argument("--study", required=True)
    reserve.add_argument("--phase", choices=sorted(PHASES), required=True)
    reserve.add_argument("--candidate-id")
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
    terminal.add_argument("--run-id")
    terminal.add_argument("--seed", type=int)
    terminal.add_argument("--event-json")
    terminal.add_argument("--event-file")
    terminal.set_defaults(handler=command_record_terminal)

    evidence = commands.add_parser("collect-training-evidence")
    evidence.add_argument("--study", required=True)
    evidence.add_argument("--run-id", required=True)
    evidence.set_defaults(handler=command_collect_training_evidence)

    return_mode = commands.add_parser("upgrade-return-mode")
    return_mode.add_argument("--study", required=True)
    return_mode.set_defaults(handler=command_upgrade_return_mode)

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
