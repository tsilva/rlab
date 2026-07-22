from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

from rlab.env import EnvConfig, make_eval_vec_env, task_termination, with_task_termination
from rlab.env import assert_provider_runtime_available, resolve_env_config
from rlab.env_metadata import env_config_from_config_dict
from rlab.eval_metrics import (
    drain_episode_records,
    episode_rank,
    episode_result_from_record,
    run_eval_episode,
    summarize_episode_results,
)
from rlab.metric_names import EVAL_FULL_DURATION_SECONDS
from rlab.modal_eval_protocol import SEED_PROTOCOL
from rlab.checkpoint_acceptance import acceptance_aggregates, manifest_index
from rlab.eval_metrics import episode_is_complete
from rlab.policy_bundle import (
    PolicyBundle,
    PolicyDocumentError,
    evaluation_contract,
    evaluation_contract_sha256,
    model_document_as_metadata,
)
from rlab.env_registry import resolve_env_provider
from rlab.rom_assets import rom_asset_manifest_for_game
from rlab.rom_runtime import (
    RomRuntimeBinding,
    ensure_local_rom_binding,
)
from rlab.policy_models import (
    load_external_policy_model,
    load_internal_policy_model,
    resolve_policy_algorithm,
)
from rlab.targets import EvalSemantics, target_for_game
from rlab.video import PolicyObservationPreview, write_video


def _eval_runtime_config(
    config: EnvConfig,
    *,
    max_steps: int,
    semantics: EvalSemantics,
    exact_task_contract: bool = False,
) -> EnvConfig:
    if exact_task_contract:
        return with_task_termination(config, max_episode_steps=max_steps)
    termination = task_termination(config)
    success = list(termination.get("success", ()))
    failure = [name for name in termination.get("failure", ()) if name != "life_loss"]
    if semantics.completion_reason == "level_change":
        success = list(dict.fromkeys((*success, "level_change")))
    return with_task_termination(
        config,
        max_episode_steps=max_steps,
        success=success,
        failure=failure,
    )


def _bind_policy_action_space(model: Any, action_space: Any) -> None:
    bind_action_space = getattr(model, "bind_action_space", None)
    if callable(bind_action_space):
        bind_action_space(action_space)


def _reset_policy_episode(model: Any) -> None:
    reset_episode = getattr(model, "reset_episode", None)
    if callable(reset_episode):
        reset_episode()


def _reset_policy_lanes(model: Any, dones: Any) -> None:
    reset_lanes = getattr(model, "reset_lanes", None)
    if callable(reset_lanes):
        reset_lanes(dones)


def _acceptance_runtime_config(
    config: EnvConfig,
    *,
    acceptance_contract: Mapping[str, Any] | None,
    n_envs: int,
) -> EnvConfig:
    """Pin each vector lane to the start declared by the evidence manifest."""

    if acceptance_contract is None:
        return config
    planned = manifest_index(acceptance_contract)
    lane_starts: list[str] = []
    for lane in range(n_envs):
        starts = {
            str(entry.get("start_state") or "").strip()
            for (entry_lane, _ordinal), entry in planned.items()
            if entry_lane == lane
        }
        if starts == {""}:
            return config
        if "" in starts or len(starts) != 1:
            raise ValueError(
                "acceptance manifest must declare one fixed start state per vector lane"
            )
        lane_starts.append(starts.pop())
    return replace(config, state="", states=tuple(lane_starts), state_probs=())


def _evaluate_model_episodes_vector(
    *,
    model,
    config: EnvConfig,
    episodes: int,
    seed: int,
    n_envs: int,
    max_steps: int,
    deterministic: bool,
    semantics: EvalSemantics,
    progress_bar: Any | None = None,
    preview_capture: PolicyObservationPreview | None = None,
    exact_task_contract: bool = False,
    acceptance_contract: dict[str, Any] | None = None,
    rom_binding: RomRuntimeBinding | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    vec_config = _eval_runtime_config(
        config,
        max_steps=max_steps,
        semantics=semantics,
        exact_task_contract=exact_task_contract,
    )
    vec_config = _acceptance_runtime_config(
        vec_config,
        acceptance_contract=acceptance_contract,
        n_envs=n_envs,
    )
    eval_env = make_eval_vec_env(
        config=vec_config,
        n_envs=n_envs,
        seed=seed,
        rom_binding=rom_binding,
    )
    episode_results: list[dict[str, Any]] = []
    best_episode_result: dict[str, Any] | None = None
    lane_episode_ordinals: dict[int, int] = {}
    planned = manifest_index(acceptance_contract) if acceptance_contract is not None else None
    rejected = False
    try:
        _bind_policy_action_space(model, getattr(eval_env, "action_space", None))
        _reset_policy_episode(model)
        torch.manual_seed(seed)
        obs = eval_env.reset()
        while len(episode_results) < episodes and not rejected:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, _step_rewards, dones, infos = eval_env.step(action)
            _reset_policy_lanes(model, dones)
            if preview_capture is not None:
                preview_capture.capture(obs)
            terminal_infos = {
                index: dict(infos[index]) for index in np.flatnonzero(np.asarray(dones, dtype=bool))
            }
            for record in drain_episode_records(eval_env):
                lane = int(record.lane)
                lane_ordinal = lane_episode_ordinals.get(lane, 0)
                lane_episode_ordinals[lane] = lane_ordinal + 1
                manifest_entry = (
                    planned.get((lane, lane_ordinal)) if planned is not None else None
                )
                if planned is not None and manifest_entry is None:
                    # A lane that completed its fixed quota remains alive while slower
                    # lanes finish, but its extra episodes are outside the manifest.
                    continue
                result = episode_result_from_record(
                    record,
                    semantics=semantics,
                    terminal_info=terminal_infos.get(int(record.lane), {}),
                )
                result = {
                    "episode": len(episode_results) + 1,
                    "seed": (
                        int(manifest_entry["seed"])
                        if manifest_entry is not None
                        else seed
                    ),
                    "seed_protocol": SEED_PROTOCOL,
                    "seed_lane": lane,
                    "seed_episode_ordinal": lane_ordinal,
                    **(
                        {"episode_id": str(manifest_entry["episode_id"])}
                        if manifest_entry is not None
                        else {}
                    ),
                    **result,
                }
                episode_results.append(result)
                if progress_bar is not None:
                    progress_bar.update(1)
                if best_episode_result is None or episode_rank(result, semantics) > episode_rank(
                    best_episode_result, semantics
                ):
                    best_episode_result = result
                if acceptance_contract is not None and not episode_is_complete(result):
                    rejected = True
                    break
                if len(episode_results) >= episodes:
                    break
    finally:
        eval_env.close()

    return episode_results, best_episode_result


def evaluate_model_episodes(
    *,
    model,
    config: EnvConfig,
    episodes: int,
    seed: int,
    max_steps: int,
    deterministic: bool,
    n_envs: int = 1,
    capture_best_video: bool = False,
    video_path: Path | None = None,
    video_fps: float = 30.0,
    video_scale: int = 4,
    extra: dict[str, Any] | None = None,
    progress: bool = False,
    progress_description: str = "eval episodes",
    preview_capture: PolicyObservationPreview | None = None,
    exact_task_contract: bool = False,
    acceptance_contract: dict[str, Any] | None = None,
    rom_binding: RomRuntimeBinding | None = None,
) -> tuple[dict[str, Any], Path | None]:
    if deterministic:
        raise ValueError("deterministic policy evaluation is unsupported; use stochastic sampling")
    started_at = time.perf_counter()
    episode_results: list[dict[str, Any]] = []
    best_episode_result: dict[str, Any] | None = None
    best_episode_actions: list[int] = []
    best_episode_seed: int | None = None

    if n_envs < 1:
        raise ValueError("n_envs must be >= 1")
    if n_envs > 1 and capture_best_video:
        raise ValueError("capture_best_video requires n_envs=1")
    semantics = target_for_game(config.game).eval_semantics
    planned = manifest_index(acceptance_contract) if acceptance_contract is not None else None
    rejected = False
    with tqdm(
        total=episodes,
        desc=progress_description,
        unit="episode",
        disable=not progress,
        leave=True,
    ) as progress_bar:
        if n_envs == 1:
            eval_config = _eval_runtime_config(
                config,
                max_steps=max_steps,
                semantics=semantics,
                exact_task_contract=exact_task_contract,
            )
            eval_env = make_eval_vec_env(
                config=eval_config,
                n_envs=1,
                seed=seed,
                rom_binding=rom_binding,
            )
            try:
                _bind_policy_action_space(model, getattr(eval_env, "action_space", None))
                for episode_idx in range(episodes):
                    manifest_entry = (
                        planned.get((0, episode_idx)) if planned is not None else None
                    )
                    if planned is not None and manifest_entry is None:
                        raise ValueError("acceptance manifest is missing a single-lane episode")
                    episode_seed = (
                        int(manifest_entry["seed"])
                        if manifest_entry is not None
                        else seed + episode_idx
                    )
                    torch.manual_seed(episode_seed)
                    result = run_eval_episode(
                        eval_env,
                        model,
                        max_steps=max_steps,
                        deterministic=deterministic,
                        seed=episode_seed,
                        capture_actions=capture_best_video,
                        default_start_state=eval_config.state,
                        semantics=semantics,
                        observation_callback=(
                            preview_capture.capture if preview_capture is not None else None
                        ),
                    )
                    actions = result.pop("actions")
                    result = {
                        "episode": episode_idx + 1,
                        "seed": episode_seed,
                        "seed_protocol": SEED_PROTOCOL,
                        "seed_lane": 0,
                        "seed_episode_ordinal": episode_idx,
                        **(
                            {"episode_id": str(manifest_entry["episode_id"])}
                            if manifest_entry is not None
                            else {}
                        ),
                        **result,
                    }
                    episode_results.append(result)
                    progress_bar.update(1)
                    if best_episode_result is None or episode_rank(
                        result,
                        semantics,
                    ) > episode_rank(
                        best_episode_result,
                        semantics,
                    ):
                        best_episode_result = result
                        best_episode_actions = actions
                        best_episode_seed = episode_seed
                    if acceptance_contract is not None and not episode_is_complete(result):
                        rejected = True
                        break
            finally:
                eval_env.close()
        else:
            episode_results, best_episode_result = _evaluate_model_episodes_vector(
                model=model,
                config=config,
                episodes=episodes,
                seed=seed,
                n_envs=n_envs,
                max_steps=max_steps,
                deterministic=deterministic,
                progress_bar=progress_bar,
                semantics=semantics,
                preview_capture=preview_capture,
                exact_task_contract=exact_task_contract,
                acceptance_contract=acceptance_contract,
                rom_binding=rom_binding,
            )

    if acceptance_contract is not None:
        aggregates = acceptance_aggregates(
            episode_results,
            contract=acceptance_contract,
        )
        rejected = rejected or int(aggregates["failure_count"]) > 0
        if rejected:
            return (
                {
                    "episode_results": episode_results,
                    "acceptance_verdict": "rejected",
                    "acceptance_aggregates": aggregates,
                },
                None,
            )
        if len(episode_results) != int(acceptance_contract["episodes"]):
            raise RuntimeError("acceptance evaluation ended without complete evidence")

    metrics = summarize_episode_results(
        episode_results,
        deterministic=deterministic,
        extra={"eval_n_envs": n_envs, **(extra or {})},
        semantics=semantics,
        event_names=tuple(task_termination(config).get("failure", ())),
        track_success=bool(task_termination(config).get("success")),
    )
    metrics["best_episode"] = best_episode_result
    written_video = None
    if (
        capture_best_video
        and video_path is not None
        and best_episode_actions
        and best_episode_seed is not None
    ):
        video_config = with_task_termination(
            config,
            max_episode_steps=max_steps,
            failure=[],
            success=[],
        )
        video_env = make_eval_vec_env(
            config=video_config,
            n_envs=1,
            seed=best_episode_seed,
            rom_binding=rom_binding,
        )
        try:
            video_env.seed(best_episode_seed)
            video_env.reset()
            frames = [np.asarray(video_env.get_images()[0]).copy()]
            for action in best_episode_actions:
                batched_action = np.expand_dims(np.asarray(action), axis=0)
                _obs, _rewards, dones, _infos = video_env.step(batched_action)
                frames.append(np.asarray(video_env.get_images()[0]).copy())
                if bool(np.asarray(dones)[0]):
                    break
        finally:
            video_env.close()
        write_video(frames, video_path, fps=video_fps, scale=video_scale)
        metrics["best_episode_video"] = str(video_path)
        written_video = video_path

    metrics[EVAL_FULL_DURATION_SECONDS] = time.perf_counter() - started_at
    if acceptance_contract is not None:
        metrics["acceptance_verdict"] = "accepted"
        metrics["acceptance_aggregates"] = acceptance_aggregates(
            episode_results,
            contract=acceptance_contract,
        )
    return metrics, written_video


def evaluate_policy_bundle(
    bundle: PolicyBundle,
    *,
    device: str = "auto",
    episodes: int | None = None,
    n_envs: int | None = None,
    progress: bool = False,
    capture_best_video: bool = False,
    video_path: Path | None = None,
    semantic_overrides: dict[str, Any] | None = None,
    preview_capture: PolicyObservationPreview | None = None,
    acceptance_contract: dict[str, Any] | None = None,
    rom_binding: RomRuntimeBinding | None = None,
    approval_hash: str | None = None,
    internal_execution_id: str | None = None,
) -> tuple[dict[str, Any], Path | None]:
    request = normalized_evaluation_request(
        bundle,
        episodes=episodes,
        n_envs=n_envs,
        semantic_overrides=semantic_overrides,
    )
    contract = evaluation_contract(bundle.recipe)
    if contract["seed_protocol"] != SEED_PROTOCOL:
        raise PolicyDocumentError(
            f"{bundle.recipe_path} uses unsupported eval seed protocol "
            f"{contract['seed_protocol']!r}; supported protocol is {SEED_PROTOCOL!r}"
        )
    overrides = dict(request["semantic_overrides"])
    environment = env_config_from_config_dict(request["environment"])
    if environment is None:
        raise PolicyDocumentError(f"{bundle.recipe_path} has no evaluation environment")
    config = resolve_env_config(environment)
    asset = contract.get("asset")
    if rom_binding is None and isinstance(asset, Mapping):
        rom_binding = ensure_local_rom_binding(asset, game=config.game)
    elif rom_binding is None and resolve_env_provider(config.env_provider).requires_external_rom_asset:
        rom_binding = ensure_local_rom_binding(
            rom_asset_manifest_for_game(config.game),
            game=config.game,
        )
    assert_provider_runtime_available(config, rom_binding=rom_binding)
    metadata = model_document_as_metadata(bundle.model)
    algorithm_id = resolve_policy_algorithm(metadata)
    if internal_execution_id:
        model = load_internal_policy_model(
            bundle.checkpoint_path,
            execution_id=internal_execution_id,
            device=device,
            metadata=metadata,
        )
    else:
        model = load_external_policy_model(
            bundle.checkpoint_path,
            device=device,
            metadata=metadata,
            source_identity=bundle.source,
            approval_hash=approval_hash,
        )
    requested_episodes = int(request["episodes"])
    requested_n_envs = int(request["n_envs"])
    requested_seed = int(request["seed"])
    requested_max_steps = int(request["max_steps"])
    effective_acceptance_contract = acceptance_contract
    if effective_acceptance_contract is None and "manifest" in contract:
        if (
            requested_episodes == int(contract["episodes"])
            and requested_n_envs == int(contract["n_envs"])
            and requested_seed == int(contract["seed"])
            and requested_max_steps == int(contract["max_steps"])
            and request["environment"] == contract["environment"]
        ):
            effective_acceptance_contract = contract
        else:
            from rlab.checkpoint_acceptance import build_checkpoint_eval_contract

            effective_acceptance_contract = build_checkpoint_eval_contract(
                environment=request["environment"],
                episodes=requested_episodes,
                n_envs=requested_n_envs,
                max_steps=requested_max_steps,
                seed=requested_seed,
                seed_protocol=str(contract["seed_protocol"]),
                acceptance=contract["acceptance"],
                asset=contract.get("asset"),
            )
    exact_contract = not overrides
    evidence = {
        "source": bundle.source,
        "source_revision": bundle.revision,
        "checkpoint_sha256": bundle.checkpoint_sha256,
        "recipe_sha256": bundle.recipe_sha256,
        "recipe_format_version": bundle.recipe["format_version"],
        "model_format_version": bundle.model["format_version"],
        "evaluation_contract_sha256": evaluation_contract_sha256(bundle.recipe),
        "exact_contract": exact_contract,
        "semantic_overrides": overrides,
        "operational_overrides": {
            "episodes": requested_episodes,
            "n_envs": requested_n_envs,
            "progress": bool(progress),
            "capture_best_video": bool(capture_best_video),
        },
        "seed_protocol": contract["seed_protocol"],
        "policy": algorithm_id,
    }
    summary, written_video = evaluate_model_episodes(
        model=model,
        config=config,
        episodes=requested_episodes,
        seed=requested_seed,
        max_steps=requested_max_steps,
        deterministic=False,
        n_envs=requested_n_envs,
        capture_best_video=capture_best_video,
        video_path=video_path,
        progress=progress,
        progress_description="eval policy bundle",
        exact_task_contract=True,
        acceptance_contract=effective_acceptance_contract,
        extra=evidence,
        preview_capture=preview_capture,
        rom_binding=rom_binding,
    )
    summary["evaluation_evidence"] = evidence
    summary["episode_seeds"] = [
        {
            "episode": result.get("episode"),
            "seed": result.get("seed"),
            "lane": result.get("seed_lane"),
            "lane_episode_ordinal": result.get("seed_episode_ordinal"),
        }
        for result in summary.get("episode_results", [])
    ]
    return summary, written_video


def normalized_evaluation_request(
    bundle: PolicyBundle,
    *,
    episodes: int | None = None,
    n_envs: int | None = None,
    semantic_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    contract = evaluation_contract(bundle.recipe)
    if contract["seed_protocol"] != SEED_PROTOCOL:
        raise PolicyDocumentError(
            f"{bundle.recipe_path} uses unsupported eval seed protocol "
            f"{contract['seed_protocol']!r}; supported protocol is {SEED_PROTOCOL!r}"
        )
    overrides = dict(semantic_overrides or {})
    allowed_semantic_overrides = {"seed", "max_steps", "environment"}
    unknown_overrides = sorted(set(overrides) - allowed_semantic_overrides)
    if unknown_overrides:
        raise PolicyDocumentError(
            "unknown semantic evaluation override(s): " + ", ".join(unknown_overrides)
        )
    requested_episodes = int(episodes if episodes is not None else contract["episodes"])
    requested_n_envs = int(n_envs if n_envs is not None else contract["n_envs"])
    if requested_episodes < 1 or requested_n_envs < 1:
        raise PolicyDocumentError("evaluation episodes and n_envs must be positive")
    seed = int(overrides.get("seed", contract["seed"]))
    environment = dict(contract["environment"])
    environment_overrides = overrides.get("environment")
    if environment_overrides is not None:
        if not isinstance(environment_overrides, Mapping):
            raise PolicyDocumentError("environment semantic override must be an object")
        from rlab.train_config import env_config_allowed_keys

        unknown_environment = sorted(
            set(environment_overrides) - env_config_allowed_keys()
        )
        if unknown_environment:
            raise PolicyDocumentError(
                "unknown environment semantic override(s): "
                + ", ".join(unknown_environment)
            )
        environment.update(environment_overrides)
    manifest_contract = contract
    if "manifest" in contract and (
        requested_episodes != int(contract["episodes"])
        or requested_n_envs != int(contract["n_envs"])
        or seed != int(contract["seed"])
        or int(overrides.get("max_steps", contract["max_steps"]))
        != int(contract["max_steps"])
        or environment != contract["environment"]
    ):
        from rlab.checkpoint_acceptance import build_checkpoint_eval_contract

        manifest_contract = build_checkpoint_eval_contract(
            environment=environment,
            episodes=requested_episodes,
            n_envs=requested_n_envs,
            max_steps=int(overrides.get("max_steps", contract["max_steps"])),
            seed=seed,
            seed_protocol=str(contract["seed_protocol"]),
            acceptance=contract["acceptance"],
            asset=contract.get("asset"),
        )
    return {
        "checkpoint_sha256": bundle.checkpoint_sha256,
        "recipe_sha256": bundle.recipe_sha256,
        "evaluation_contract_sha256": evaluation_contract_sha256(bundle.recipe),
        "environment": environment,
        "action_sampling": contract["action_sampling"],
        "episodes": requested_episodes,
        "n_envs": requested_n_envs,
        "max_steps": int(overrides.get("max_steps", contract["max_steps"])),
        "seed": seed,
        "seed_protocol": contract["seed_protocol"],
        "semantic_overrides": overrides,
        "seed_assignments": (
            [
                {
                    "lane": int(entry["lane"]),
                    "lane_episode_ordinal": int(entry["lane_episode_ordinal"]),
                    "seed": int(entry["seed"]),
                }
                for entry in manifest_contract["manifest"]["episodes"]
            ]
            if "manifest" in manifest_contract
            else (
                [seed + episode for episode in range(requested_episodes)]
                if requested_n_envs == 1
                else [
                    {"lane": lane, "base_seed": seed}
                    for lane in range(requested_n_envs)
                ]
            )
        ),
    }
