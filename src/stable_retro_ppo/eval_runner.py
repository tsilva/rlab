from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from stable_retro_ppo.env import EnvConfig, make_eval_vec_env, make_rendered_replay_env
from stable_retro_ppo.eval_metrics import (
    episode_rank,
    is_level_complete,
    run_eval_episode,
    summarize_episode_results,
)
from stable_retro_ppo.video import replay_actions_for_video, write_video


def _evaluate_model_episodes_vector(
    *,
    model,
    config: EnvConfig,
    episodes: int,
    seed: int,
    n_envs: int,
    max_steps: int,
    deterministic: bool,
    completion_x_threshold: int,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    vec_config = replace(
        config,
        terminate_on_completion=False,
        terminate_on_level_change=False,
        max_episode_steps=0,
        no_progress_timeout_steps=0,
    )
    eval_env = make_eval_vec_env(config=vec_config, n_envs=n_envs, seed=seed)
    episode_results: list[dict[str, Any]] = []
    best_episode_result: dict[str, Any] | None = None
    rewards = np.zeros(n_envs, dtype=np.float64)
    steps = np.zeros(n_envs, dtype=np.int64)
    max_x_positions = np.zeros(n_envs, dtype=np.int64)
    max_level_x_positions = np.zeros(n_envs, dtype=np.int64)
    active = np.ones(n_envs, dtype=bool)

    try:
        torch.manual_seed(seed)
        obs = eval_env.reset()
        while len(episode_results) < episodes:
            if not active.any():
                obs = eval_env.reset()
                active[:] = True
                rewards[:] = 0.0
                steps[:] = 0
                max_x_positions[:] = 0
                max_level_x_positions[:] = 0

            action, _ = model.predict(obs, deterministic=deterministic)
            obs, step_rewards, dones, infos = eval_env.step(action)

            for env_index, info_obj in enumerate(infos):
                if not active[env_index]:
                    continue

                info = dict(info_obj)
                rewards[env_index] += float(step_rewards[env_index])
                steps[env_index] += 1
                max_x_positions[env_index] = max(
                    max_x_positions[env_index],
                    int(info.get("max_x_pos", 0)),
                )
                max_level_x_positions[env_index] = max(
                    max_level_x_positions[env_index],
                    int(info.get("level_max_x_pos", 0)),
                )

                completed = is_level_complete(
                    info,
                    int(max_x_positions[env_index]),
                    completion_x_threshold,
                )
                timed_out = steps[env_index] >= max_steps
                if bool(dones[env_index]) or completed or timed_out:
                    died = bool(info.get("died", False))
                    death_x_pos = info.get("death_x_pos")
                    if died and death_x_pos is None:
                        death_x_pos = int(max_x_positions[env_index])

                    result = {
                        "episode": len(episode_results) + 1,
                        "seed": None,
                        "env_index": int(env_index),
                        "start_state": info.get("start_state") or info.get("state") or config.state,
                        "reward": float(rewards[env_index]),
                        "max_x_pos": int(max_x_positions[env_index]),
                        "max_level_x_pos": int(max_level_x_positions[env_index]),
                        "score": int(info.get("score", 0)),
                        "lives": int(info.get("lives", 0)),
                        "time": int(info.get("time", 0)),
                        "steps": int(steps[env_index]),
                        "terminated": bool(dones[env_index]) or completed,
                        "truncated": timed_out or bool(info.get("TimeLimit.truncated", False)),
                        "level_complete": completed,
                        "died": died,
                        "death_x_pos": int(death_x_pos) if death_x_pos is not None else None,
                        "final_info": info,
                    }
                    episode_results.append(result)
                    if best_episode_result is None or episode_rank(result) > episode_rank(
                        best_episode_result
                    ):
                        best_episode_result = result

                    rewards[env_index] = 0.0
                    steps[env_index] = 0
                    max_x_positions[env_index] = 0
                    max_level_x_positions[env_index] = 0
                    active[env_index] = False

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
    completion_x_threshold: int,
    n_envs: int = 1,
    capture_best_video: bool = False,
    video_path: Path | None = None,
    video_fps: float = 30.0,
    video_scale: int = 4,
    extra: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], Path | None]:
    episode_results: list[dict[str, Any]] = []
    best_episode_result: dict[str, Any] | None = None
    best_episode_actions: list[int] = []
    best_episode_seed: int | None = None

    if n_envs < 1:
        raise ValueError("n_envs must be >= 1")
    if n_envs > 1 and capture_best_video:
        raise ValueError("capture_best_video requires n_envs=1")

    if n_envs == 1:
        eval_env = make_eval_vec_env(config=config, n_envs=1, seed=seed)
        try:
            for episode_idx in range(episodes):
                episode_seed = seed + episode_idx
                torch.manual_seed(episode_seed)
                result = run_eval_episode(
                    eval_env,
                    model,
                    max_steps=max_steps,
                    deterministic=deterministic,
                    seed=episode_seed,
                    completion_x_threshold=completion_x_threshold,
                    capture_actions=capture_best_video,
                    default_start_state=config.state,
                )
                actions = result.pop("actions")
                result = {"episode": episode_idx + 1, "seed": episode_seed, **result}
                episode_results.append(result)
                if best_episode_result is None or episode_rank(result) > episode_rank(
                    best_episode_result
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
            completion_x_threshold=completion_x_threshold,
        )

    metrics = summarize_episode_results(
        episode_results,
        deterministic=deterministic,
        state_metric_root="eval",
        extra={"eval_n_envs": n_envs, **(extra or {})},
    )
    metrics["best_episode"] = best_episode_result

    written_video = None
    if (
        capture_best_video
        and video_path is not None
        and best_episode_actions
        and best_episode_seed is not None
    ):
        video_env = make_rendered_replay_env(config=config, seed=best_episode_seed)
        try:
            frames = replay_actions_for_video(
                video_env,
                actions=best_episode_actions,
                seed=best_episode_seed,
            )
        finally:
            video_env.close()
        write_video(frames, video_path, fps=video_fps, scale=video_scale)
        metrics["best_episode_video"] = str(video_path)
        written_video = video_path

    return metrics, written_video
