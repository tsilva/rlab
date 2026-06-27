from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from rlab.env import EnvConfig, make_eval_vec_env, make_rendered_replay_env
from rlab.metric_names import (
    EVAL_BEST_REWARD,
    EVAL_BEST_VIDEO,
    EVAL_BEST_X,
    EVAL_DEATH_COUNT,
    EVAL_DEATH_RATE,
    EVAL_DEATH_X_HIST,
    EVAL_DONE_ALL,
    EVAL_DONE_LEVEL_CHANGE,
    EVAL_DONE_LEVEL_CHANGE_FROM_RATE_MEAN,
    EVAL_DONE_LEVEL_CHANGE_FROM_RATE_MIN,
    EVAL_DONE_LEVEL_CHANGE_RATE,
    EVAL_DONE_MAX_STEPS,
    EVAL_DONE_MAX_STEPS_RATE,
    EVAL_DONE_UNCLASSIFIED,
    EVAL_DONE_UNCLASSIFIED_RATE,
    EVAL_INFO_LEVEL_COMPLETE_RATE_MEAN_LAST,
    EVAL_INFO_LEVEL_COMPLETE_RATE_MIN_LAST,
    EVAL_PROGRESS_LEVEL_X_MAX,
    EVAL_PROGRESS_LEVEL_X_MEAN,
    EVAL_PROGRESS_X_MAX,
    EVAL_PROGRESS_X_MEAN,
    EVAL_REWARD_MAX,
    EVAL_REWARD_MEAN,
    EVAL_REWARD_STD,
    GLOBAL_STEP,
    eval_done_value_metric,
)
from rlab.video import replay_actions_for_video, write_video


def is_level_complete(info: dict[str, Any], max_x_pos: int, completion_x_threshold: int) -> bool:
    if "completion_event" in info or "level_complete" in info:
        return bool(info.get("completion_event", info.get("level_complete", False)))
    return bool(info.get("level_changed", False)) and not bool(
        info.get("died", False) or info.get("life_loss", False),
    )


def death_location_histogram(death_x_positions: list[int], bin_size: int = 100) -> dict[str, int]:
    bins: dict[str, int] = {}
    for x_pos in death_x_positions:
        start = (int(x_pos) // bin_size) * bin_size
        key = f"{start}-{start + bin_size - 1}"
        bins[key] = bins.get(key, 0) + 1
    return dict(sorted(bins.items(), key=lambda item: int(item[0].split("-", 1)[0])))


def episode_start_state(episode: dict[str, Any]) -> str | None:
    state = episode.get("start_state") or episode.get("state")
    final_info = episode.get("final_info")
    if not state and isinstance(final_info, dict):
        state = final_info.get("start_state") or final_info.get("state")
    return str(state) if state else None


def serializable_info(info: dict[str, Any]) -> dict[str, Any]:
    result = dict(info)
    result.pop("terminal_observation", None)
    return result


def eval_done_from_metrics(episode_results: list[dict[str, Any]]) -> dict[str, int | float]:
    metrics: dict[str, int | float] = {}
    level_change_rates: list[float] = []
    states = sorted(
        {state for episode in episode_results if (state := episode_start_state(episode))}
    )
    for state in states:
        state_episodes = [
            episode for episode in episode_results if episode_start_state(episode) == state
        ]
        denominator = len(state_episodes)
        level_change_count = sum(1 for episode in state_episodes if episode["level_complete"])
        max_steps_count = sum(1 for episode in state_episodes if episode.get("truncated"))
        unclassified_count = sum(
            1
            for episode in state_episodes
            if not episode["level_complete"] and not episode.get("truncated")
        )
        level_change_rate = level_change_count / denominator
        level_change_rates.append(level_change_rate)

        all_metric = eval_done_value_metric("all", "from", state)
        level_change_metric = eval_done_value_metric("level_change", "from", state)
        max_steps_metric = eval_done_value_metric("max_steps", "from", state)
        unclassified_metric = eval_done_value_metric("unclassified", "from", state)
        metrics.update(
            {
                all_metric: denominator,
                level_change_metric: level_change_count,
                f"{level_change_metric}/rate": level_change_rate,
                max_steps_metric: max_steps_count,
                f"{max_steps_metric}/rate": max_steps_count / denominator,
                unclassified_metric: unclassified_count,
                f"{unclassified_metric}/rate": unclassified_count / denominator,
            },
        )
    if level_change_rates:
        level_change_rate_min = min(level_change_rates)
        level_change_rate_mean = float(np.mean(level_change_rates))
        metrics[EVAL_DONE_LEVEL_CHANGE_FROM_RATE_MIN] = level_change_rate_min
        metrics[EVAL_DONE_LEVEL_CHANGE_FROM_RATE_MEAN] = level_change_rate_mean
        metrics[EVAL_INFO_LEVEL_COMPLETE_RATE_MIN_LAST] = level_change_rate_min
        metrics[EVAL_INFO_LEVEL_COMPLETE_RATE_MEAN_LAST] = level_change_rate_mean
    return metrics


def flat_numeric_metrics(metrics: dict[str, Any], prefix: str) -> dict[str, int | float]:
    return {
        key: value
        for key, value in metrics.items()
        if key.startswith(prefix) and isinstance(value, int | float) and not isinstance(value, bool)
    }


def episode_rank(result: dict[str, Any]) -> tuple[int, float, float]:
    return (
        int(bool(result["level_complete"])),
        float(result["max_x_pos"]),
        float(result["reward"]),
    )


def single_env_action(action) -> int | np.ndarray:
    action_array = np.asarray(action)
    if action_array.shape == ():
        return int(action_array)
    first = np.asarray(action_array[0])
    if first.shape == ():
        return int(first)
    return first.astype(np.int8, copy=True)


def summarize_episode_results(
    episode_results: list[dict[str, Any]],
    *,
    deterministic: bool,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not episode_results:
        raise ValueError("episode_results must not be empty")

    rewards = np.array([episode["reward"] for episode in episode_results], dtype=np.float64)
    max_x_positions = np.array(
        [episode["max_x_pos"] for episode in episode_results],
        dtype=np.float64,
    )
    max_level_x_positions = np.array(
        [episode["max_level_x_pos"] for episode in episode_results],
        dtype=np.float64,
    )
    death_x_positions = [
        int(episode["death_x_pos"])
        for episode in episode_results
        if episode.get("death_x_pos") is not None
    ]
    completion_count = sum(1 for episode in episode_results if episode["level_complete"])
    death_count = sum(1 for episode in episode_results if episode["died"])
    terminated_count = sum(
        1
        for episode in episode_results
        if episode.get("terminated") and not episode.get("truncated")
    )
    truncated_count = sum(1 for episode in episode_results if episode.get("truncated"))
    unclassified_count = sum(
        1
        for episode in episode_results
        if not episode["level_complete"] and not episode.get("truncated")
    )
    episode_count = len(episode_results)
    metrics: dict[str, Any] = {
        "episodes": episode_count,
        "deterministic": deterministic,
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std()),
        "reward_max": float(rewards.max()),
        "max_x_mean": float(max_x_positions.mean()),
        "max_x_max": int(max_x_positions.max()),
        "max_level_x_mean": float(max_level_x_positions.mean()),
        "max_level_x_max": int(max_level_x_positions.max()),
        "completion_count": completion_count,
        "completion_rate": completion_count / episode_count,
        "death_count": death_count,
        "death_rate": death_count / episode_count,
        "terminated_count": terminated_count,
        "terminated_rate": terminated_count / episode_count,
        "truncated_count": truncated_count,
        "truncated_rate": truncated_count / episode_count,
        "unclassified_count": unclassified_count,
        "unclassified_rate": unclassified_count / episode_count,
        EVAL_DONE_ALL: episode_count,
        EVAL_DONE_LEVEL_CHANGE: completion_count,
        EVAL_DONE_LEVEL_CHANGE_RATE: completion_count / episode_count,
        EVAL_DONE_MAX_STEPS: truncated_count,
        EVAL_DONE_MAX_STEPS_RATE: truncated_count / episode_count,
        EVAL_DONE_UNCLASSIFIED: unclassified_count,
        EVAL_DONE_UNCLASSIFIED_RATE: unclassified_count / episode_count,
        "death_x_histogram": death_location_histogram(death_x_positions),
        "episode_results": episode_results,
    }
    metrics.update(eval_done_from_metrics(episode_results))
    if extra:
        metrics = {**extra, **metrics}
    return metrics


def run_eval_episode(
    env,
    model,
    max_steps: int,
    deterministic: bool,
    seed: int,
    completion_x_threshold: int,
    capture_actions: bool = False,
    default_start_state: str | None = None,
) -> dict[str, Any]:
    env.seed(seed)
    obs = env.reset()
    actions: list[Any] = []
    total_reward = 0.0
    max_x_pos = 0
    max_level_x_pos = 0
    final_info: dict[str, Any] = {}
    terminated = False
    truncated = False
    completed = False
    died = False
    death_x_pos: Any | None = None
    start_state = default_start_state
    steps_taken = 0

    for step_idx in range(max_steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        action_value = single_env_action(action)
        if capture_actions:
            actions.append(action_value)
        obs, rewards, dones, infos = env.step(action)
        info = dict(infos[0])
        steps_taken = step_idx + 1
        terminated = bool(dones[0])
        truncated = bool(info.get("TimeLimit.truncated", False))
        total_reward += float(rewards[0])
        if start_state is None:
            start_state = info.get("start_state") or info.get("state")
        max_x_pos = max(max_x_pos, int(info.get("max_x_pos", 0)))
        max_level_x_pos = max(max_level_x_pos, int(info.get("level_max_x_pos", 0)))
        final_info = info
        completed = completed or is_level_complete(
            final_info,
            max_x_pos,
            completion_x_threshold,
        )
        if bool(info.get("died", False)):
            died = True
            if death_x_pos is None:
                death_x_pos = info.get("death_x_pos")
                if death_x_pos is None:
                    death_x_pos = max_x_pos
        if terminated:
            break
    else:
        truncated = True

    return {
        "start_state": start_state
        or final_info.get("start_state")
        or final_info.get("state")
        or default_start_state,
        "reward": total_reward,
        "max_x_pos": max_x_pos,
        "max_level_x_pos": max_level_x_pos,
        "score": int(final_info.get("score", 0)),
        "lives": int(final_info.get("lives", 0)),
        "time": int(final_info.get("time", 0)),
        "steps": steps_taken,
        "terminated": terminated,
        "truncated": truncated,
        "level_complete": completed,
        "died": died,
        "death_x_pos": int(death_x_pos) if death_x_pos is not None else None,
        "final_info": serializable_info(final_info),
        "actions": actions,
    }


class RetroEvalCallback(BaseCallback):
    def __init__(
        self,
        config: EnvConfig,
        run_dir: str,
        best_model_save_path: str,
        eval_freq: int,
        n_eval_episodes: int,
        deterministic: bool,
        seed: int,
        completion_x_threshold: int,
        wandb_run=None,
        record_video: bool = True,
        video_fps: float = 30.0,
        video_scale: int = 4,
    ):
        super().__init__()
        self.config = config
        self.run_dir = Path(run_dir)
        self.best_model_save_path = Path(best_model_save_path)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.seed = seed
        self.completion_x_threshold = completion_x_threshold
        self.wandb_run = wandb_run
        self.record_video = record_video
        self.video_fps = video_fps
        self.video_scale = video_scale
        self.best_eval_score = (-float("inf"), -float("inf"), -float("inf"))

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or self.n_calls % self.eval_freq != 0:
            return True
        self.evaluate()
        return True

    def evaluate(self) -> None:
        eval_env = make_eval_vec_env(
            config=self.config, n_envs=1, seed=self.seed + self.num_timesteps
        )
        episode_results: list[dict[str, Any]] = []
        best_episode_result: dict[str, Any] | None = None
        best_episode_actions: list[int] = []
        best_episode_seed: int | None = None
        try:
            for episode_idx in range(self.n_eval_episodes):
                episode_seed = self.seed + self.num_timesteps + episode_idx
                result = run_eval_episode(
                    eval_env,
                    self.model,
                    max_steps=self.config.max_episode_steps,
                    deterministic=self.deterministic,
                    seed=episode_seed,
                    completion_x_threshold=self.completion_x_threshold,
                    capture_actions=self.record_video,
                    default_start_state=self.config.state,
                )
                actions = result.pop("actions")
                result = {"episode": episode_idx + 1, **result}
                episode_results.append(result)
                if best_episode_result is None or episode_rank(result) > episode_rank(
                    best_episode_result
                ):
                    best_episode_result = result
                    best_episode_actions = actions
                    best_episode_seed = episode_seed
        finally:
            eval_env.close()

        death_x_positions = [
            int(episode["death_x_pos"])
            for episode in episode_results
            if episode.get("death_x_pos") is not None
        ]
        metrics = summarize_episode_results(
            episode_results,
            deterministic=self.deterministic,
            extra={"timesteps": self.num_timesteps},
        )
        metrics["best_model_score"] = [
            metrics.get(EVAL_DONE_LEVEL_CHANGE_FROM_RATE_MIN, metrics["completion_rate"]),
            metrics["max_x_max"],
            metrics["reward_mean"],
        ]
        metrics["best_episode"] = best_episode_result

        video_path = None
        if self.record_video and best_episode_actions and best_episode_seed is not None:
            video_path = (
                self.run_dir / "eval_videos" / f"best_episode_{self.num_timesteps}_steps.mp4"
            )
            video_env = make_rendered_replay_env(config=self.config, seed=best_episode_seed)
            try:
                best_episode_frames = replay_actions_for_video(
                    video_env,
                    actions=best_episode_actions,
                    seed=best_episode_seed,
                )
            finally:
                video_env.close()
            write_video(best_episode_frames, video_path, fps=self.video_fps, scale=self.video_scale)
            metrics["best_episode_video"] = str(video_path)

        self.run_dir.mkdir(parents=True, exist_ok=True)
        with Path(self.run_dir, "eval_metrics.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(metrics) + "\n")

        self.logger.record(EVAL_REWARD_MEAN, metrics["reward_mean"])
        self.logger.record(EVAL_REWARD_STD, metrics["reward_std"])
        self.logger.record(EVAL_REWARD_MAX, metrics["reward_max"])
        self.logger.record(EVAL_PROGRESS_X_MEAN, metrics["max_x_mean"])
        self.logger.record(EVAL_PROGRESS_X_MAX, metrics["max_x_max"])
        self.logger.record(EVAL_PROGRESS_LEVEL_X_MEAN, metrics["max_level_x_mean"])
        self.logger.record(EVAL_PROGRESS_LEVEL_X_MAX, metrics["max_level_x_max"])
        self.logger.record(EVAL_DEATH_RATE, metrics["death_rate"])
        self.logger.record(EVAL_DEATH_COUNT, metrics["death_count"])
        for key, value in flat_numeric_metrics(metrics, "eval/done/").items():
            self.logger.record(key, value)
        for key, value in flat_numeric_metrics(metrics, "eval/info/").items():
            self.logger.record(key, value)
        self.logger.record("time/total_timesteps", self.num_timesteps)
        self.logger.dump(self.num_timesteps)

        eval_score = (
            metrics.get(EVAL_DONE_LEVEL_CHANGE_FROM_RATE_MIN, metrics["completion_rate"]),
            metrics["max_x_max"],
            metrics["reward_mean"],
        )
        if eval_score > self.best_eval_score:
            self.best_eval_score = eval_score
            self.best_model_save_path.mkdir(parents=True, exist_ok=True)
            self.model.save(self.best_model_save_path / "best_model")

        if self.wandb_run is not None:
            self.log_wandb(metrics, death_x_positions, video_path)

        print(
            "Retro eval "
            f"steps={self.num_timesteps} "
            f"reward_mean={metrics['reward_mean']:.2f} "
            f"max_x_mean={metrics['max_x_mean']:.2f} "
            f"max_x_max={metrics['max_x_max']} "
            f"completion_rate={metrics['completion_rate']:.3f} "
            f"death_rate={metrics['death_rate']:.3f}",
            flush=True,
        )

    def log_wandb(
        self,
        metrics: dict[str, Any],
        death_x_positions: list[int],
        video_path: Path | None,
    ) -> None:
        import wandb

        payload: dict[str, Any] = {
            GLOBAL_STEP: self.num_timesteps,
            EVAL_REWARD_MEAN: metrics["reward_mean"],
            EVAL_REWARD_STD: metrics["reward_std"],
            EVAL_REWARD_MAX: metrics["reward_max"],
            EVAL_PROGRESS_X_MEAN: metrics["max_x_mean"],
            EVAL_PROGRESS_X_MAX: metrics["max_x_max"],
            EVAL_PROGRESS_LEVEL_X_MEAN: metrics["max_level_x_mean"],
            EVAL_PROGRESS_LEVEL_X_MAX: metrics["max_level_x_max"],
            EVAL_DEATH_COUNT: metrics["death_count"],
            EVAL_DEATH_RATE: metrics["death_rate"],
            EVAL_BEST_REWARD: metrics["best_episode"]["reward"],
            EVAL_BEST_X: metrics["best_episode"]["max_x_pos"],
        }
        payload.update(flat_numeric_metrics(metrics, "eval/done/"))
        payload.update(flat_numeric_metrics(metrics, "eval/info/"))
        if death_x_positions:
            payload[EVAL_DEATH_X_HIST] = wandb.Histogram(death_x_positions)
        if video_path is not None and video_path.is_file():
            payload[EVAL_BEST_VIDEO] = wandb.Video(
                str(video_path), fps=self.video_fps, format="mp4"
            )
        self.wandb_run.log(payload, step=self.num_timesteps)
