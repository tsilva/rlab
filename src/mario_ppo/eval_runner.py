from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from mario_ppo.env import EnvConfig, make_eval_vec_env, make_rendered_replay_env
from mario_ppo.eval_metrics import episode_rank, run_eval_episode, summarize_episode_results
from mario_ppo.video import replay_actions_for_video, write_video


def evaluate_model_episodes(
    *,
    model,
    config: EnvConfig,
    episodes: int,
    seed: int,
    max_steps: int,
    deterministic: bool,
    completion_x_threshold: int,
    capture_best_video: bool = False,
    video_path: Path | None = None,
    video_fps: float = 30.0,
    video_scale: int = 4,
    extra: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], Path | None]:
    eval_env = make_eval_vec_env(config=config, n_envs=1, seed=seed)
    episode_results: list[dict[str, Any]] = []
    best_episode_result: dict[str, Any] | None = None
    best_episode_actions: list[int] = []
    best_episode_seed: int | None = None
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

    metrics = summarize_episode_results(
        episode_results,
        deterministic=deterministic,
        extra=extra,
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
