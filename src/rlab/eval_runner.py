from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

from rlab.env import EnvConfig, make_eval_vec_env, task_termination, with_task_termination
from rlab.eval_metrics import (
    drain_episode_records,
    episode_rank,
    episode_result_from_record,
    run_eval_episode,
    summarize_episode_results,
)
from rlab.metric_names import EVAL_FULL_DURATION_SECONDS
from rlab.modal_eval_protocol import SEED_PROTOCOL
from rlab.targets import EvalSemantics, target_for_game
from rlab.video import PolicyObservationPreview, write_video


def _eval_runtime_config(
    config: EnvConfig,
    *,
    max_steps: int,
    semantics: EvalSemantics,
) -> EnvConfig:
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
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    vec_config = _eval_runtime_config(
        config,
        max_steps=max_steps,
        semantics=semantics,
    )
    eval_env = make_eval_vec_env(config=vec_config, n_envs=n_envs, seed=seed)
    episode_results: list[dict[str, Any]] = []
    best_episode_result: dict[str, Any] | None = None
    lane_episode_ordinals: dict[int, int] = {}
    try:
        _bind_policy_action_space(model, getattr(eval_env, "action_space", None))
        torch.manual_seed(seed)
        obs = eval_env.reset()
        while len(episode_results) < episodes:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, _step_rewards, dones, infos = eval_env.step(action)
            if preview_capture is not None:
                preview_capture.capture(obs)
            terminal_infos = {
                index: dict(infos[index]) for index in np.flatnonzero(np.asarray(dones, dtype=bool))
            }
            for record in drain_episode_records(eval_env):
                lane = int(record.lane)
                lane_ordinal = lane_episode_ordinals.get(lane, 0)
                lane_episode_ordinals[lane] = lane_ordinal + 1
                result = episode_result_from_record(
                    record,
                    semantics=semantics,
                    terminal_info=terminal_infos.get(int(record.lane), {}),
                )
                result = {
                    "episode": len(episode_results) + 1,
                    "seed": seed,
                    "seed_protocol": SEED_PROTOCOL,
                    "seed_lane": lane,
                    "seed_episode_ordinal": lane_ordinal,
                    **result,
                }
                episode_results.append(result)
                if progress_bar is not None:
                    progress_bar.update(1)
                if best_episode_result is None or episode_rank(result, semantics) > episode_rank(
                    best_episode_result, semantics
                ):
                    best_episode_result = result
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
            )
            eval_env = make_eval_vec_env(config=eval_config, n_envs=1, seed=seed)
            try:
                _bind_policy_action_space(model, getattr(eval_env, "action_space", None))
                for episode_idx in range(episodes):
                    episode_seed = seed + episode_idx
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
            )

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
        video_env = make_eval_vec_env(config=video_config, n_envs=1, seed=best_episode_seed)
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
    return metrics, written_video
