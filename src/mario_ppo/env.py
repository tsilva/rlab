from __future__ import annotations

# ruff: noqa: E402

import os
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import replace
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", os.path.abspath(".matplotlib"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import gymnasium as gym
import cv2
import numpy as np
import stable_retro as retro
from stable_retro import StableRetroNativeVecEnv
from stable_baselines3.common.atari_wrappers import ClipRewardEnv
from stable_baselines3.common.vec_env import VecEnvWrapper, VecMonitor, VecTransposeImage

GAME = "SuperMarioBros-Nes-v0"
DEFAULT_STATE = "Level1-1"
DEFAULT_OBS_RESIZE_ALGORITHM = "area"
DEFAULT_HUD_CROP_TOP = 32

# stable-retro button order for NES:
# ['B', None, 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A']
BUTTON_B = 0
BUTTON_LEFT = 6
BUTTON_RIGHT = 7
BUTTON_A = 8


def _button_mask(*buttons: int) -> np.ndarray:
    mask = np.zeros(9, dtype=np.int8)
    for button in buttons:
        mask[button] = 1
    return mask


ACTION_LIBRARY = {
    "noop": _button_mask(),
    "right": _button_mask(BUTTON_RIGHT),
    "right_b": _button_mask(BUTTON_RIGHT, BUTTON_B),
    "right_a": _button_mask(BUTTON_RIGHT, BUTTON_A),
    "right_a_b": _button_mask(BUTTON_RIGHT, BUTTON_A, BUTTON_B),
    "a": _button_mask(BUTTON_A),
    "left": _button_mask(BUTTON_LEFT),
}

ACTION_SETS = {
    "simple": ("noop", "right", "right_b", "right_a", "right_a_b", "a", "left"),
    "right": ("right", "right_b", "right_a", "right_a_b"),
}

ACTION_NAMES = ACTION_SETS["simple"]


def action_names_for_set(action_set: str) -> tuple[str, ...]:
    if action_set not in ACTION_SETS:
        valid = ", ".join(sorted(ACTION_SETS))
        raise ValueError(f"unknown action_set {action_set!r}; valid values: {valid}")
    return ACTION_SETS[action_set]


@dataclass(frozen=True)
class EnvConfig:
    game: str = GAME
    state: str = DEFAULT_STATE
    states: tuple[str, ...] = ()
    frame_skip: int = 4
    max_pool_frames: bool = True
    max_episode_steps: int = 4500
    observation_size: int = 84
    hud_crop_top: int = DEFAULT_HUD_CROP_TOP
    obs_resize_algorithm: str = DEFAULT_OBS_RESIZE_ALGORITHM
    use_retro_reward: bool = False
    clip_rewards: bool = False
    reward_mode: str = "bounded"
    progress_reward_cap: float = 30.0
    progress_reward_scale: float = 1.0
    terminal_reward: float = 30.0
    reward_scale: float = 30.0
    time_penalty: float = 0.0
    death_penalty: float = 25.0
    completion_reward: float = 0.0
    completion_x_threshold: int = 0
    terminate_on_life_loss: bool = True
    terminate_on_level_change: bool = False
    terminate_on_completion: bool = False
    action_set: str = "simple"


class DiscreteMarioActions(gym.ActionWrapper):
    """Map a small discrete action set to stable-retro's NES MultiBinary controls."""

    def __init__(self, env: gym.Env, action_set: str):
        super().__init__(env)
        self.action_names = action_names_for_set(action_set)
        self.actions = tuple(ACTION_LIBRARY[name] for name in self.action_names)
        self.action_space = gym.spaces.Discrete(len(self.actions))

    def action(self, action: int) -> np.ndarray:
        return self.actions[int(action)].copy()


class VecDiscreteMarioActions(VecEnvWrapper):
    """Map discrete SB3 actions to stable-retro's NES MultiBinary controls."""

    def __init__(self, venv, action_set: str):
        self.action_names = action_names_for_set(action_set)
        self.actions = np.stack([ACTION_LIBRARY[name] for name in self.action_names]).astype(
            np.int8,
        )
        super().__init__(
            venv,
            observation_space=venv.observation_space,
            action_space=gym.spaces.Discrete(len(self.actions)),
        )

    def reset(self):
        return self.venv.reset()

    def step_async(self, actions):
        action_indices = np.asarray(actions, dtype=np.int64).reshape(-1)
        self.venv.step_async(self.actions[action_indices])

    def step_wait(self):
        return self.venv.step_wait()


class VecMarioProgressInfo(VecEnvWrapper):
    """Vectorized Mario reward shaping and progress metrics.

    Image preprocessing, frame skip, frame stacking, and max-pooling stay inside
    StableRetroNativeVecEnv. This wrapper only rewrites rewards and annotates info.
    """

    def __init__(self, venv, config: EnvConfig):
        super().__init__(venv)
        self.config = config
        n_envs = self.num_envs
        self.level_x_pos = np.zeros(n_envs, dtype=np.int64)
        self.level_max_x_pos = np.zeros(n_envs, dtype=np.int64)
        self.completed_level_base = np.zeros(n_envs, dtype=np.int64)
        self.max_global_x_pos = np.zeros(n_envs, dtype=np.int64)
        self.curr_score = np.zeros(n_envs, dtype=np.int64)
        self.prev_lives: list[int | None] = [None] * n_envs
        self.initial_level: list[tuple[int, int] | None] = [None] * n_envs
        self.current_level: list[tuple[int, int] | None] = [None] * n_envs
        self.completed_level_count = np.zeros(n_envs, dtype=np.int64)
        self.current_level_completion_awarded = np.zeros(n_envs, dtype=bool)
        self.completed = np.zeros(n_envs, dtype=bool)
        self.episode_steps = np.zeros(n_envs, dtype=np.int64)

    def reset(self):
        obs = self.venv.reset()
        self._reset_tracking(range(self.num_envs), getattr(self.venv, "reset_infos", None))
        return obs

    def _reset_tracking(self, indices, infos=None) -> None:
        infos = infos or [{} for _ in range(self.num_envs)]
        for index in indices:
            info = infos[index] if index < len(infos) else {}
            self.level_x_pos[index] = 0
            self.level_max_x_pos[index] = 0
            self.completed_level_base[index] = 0
            self.max_global_x_pos[index] = 0
            self.curr_score[index] = int(info.get("score", 0))
            lives = info.get("lives")
            self.prev_lives[index] = int(lives) if lives is not None else None
            if "levelHi" in info or "levelLo" in info:
                level = (int(info.get("levelHi", 0)), int(info.get("levelLo", 0)))
                self.initial_level[index] = level
                self.current_level[index] = level
            else:
                self.initial_level[index] = None
                self.current_level[index] = None
            self.completed_level_count[index] = 0
            self.current_level_completion_awarded[index] = False
            self.completed[index] = False
            self.episode_steps[index] = 0

    def step_async(self, actions):
        self.venv.step_async(actions)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        rewards = np.asarray(rewards, dtype=np.float32)
        dones = np.asarray(dones, dtype=bool)
        infos = [dict(info) for info in infos]
        shaped_rewards = np.zeros(self.num_envs, dtype=np.float32)
        custom_dones = np.zeros(self.num_envs, dtype=bool)

        for index, info in enumerate(infos):
            shaped_rewards[index] = self._shape_one(index, rewards[index], info, dones[index])
            self.episode_steps[index] += 1
            if (
                self.config.max_episode_steps > 0
                and self.episode_steps[index] >= self.config.max_episode_steps
            ):
                custom_dones[index] = True
                info["_custom_truncated"] = True
                info["TimeLimit.truncated"] = True
            if info.pop("_custom_done", False):
                custom_dones[index] = True

        if self.config.clip_rewards:
            shaped_rewards = np.sign(shaped_rewards).astype(np.float32)

        dones = np.logical_or(dones, custom_dones)
        native_done_indices = [
            idx for idx, done in enumerate(dones) if done and not custom_dones[idx]
        ]
        if native_done_indices:
            self._reset_tracking(native_done_indices)

        if custom_dones.any():
            terminal_obs = np.asarray(obs).copy()
            for index, info in enumerate(infos):
                info.setdefault("terminal_observation", terminal_obs[index])
                if custom_dones[index]:
                    if info.pop("_custom_terminal", False):
                        info["TimeLimit.truncated"] = False
                    elif info.pop("_custom_truncated", False):
                        info["TimeLimit.truncated"] = True
                    else:
                        info.setdefault("TimeLimit.truncated", False)
                else:
                    # StableRetroNativeVecEnv does not expose per-env reset yet.
                    # If one Python-defined terminal condition fires, all slots
                    # must be reset together. Non-terminal slots are treated as
                    # time-limit-style truncations so SB3 bootstraps from the
                    # saved terminal_observation instead of cutting value flow.
                    info["global_reset"] = True
                    info["TimeLimit.truncated"] = True
            obs = self.venv.reset()
            dones[:] = True
            self._reset_tracking(range(self.num_envs), getattr(self.venv, "reset_infos", None))

        return obs, shaped_rewards, dones, infos

    def _shape_one(
        self, index: int, native_reward: float, info: dict[str, Any], done: bool
    ) -> float:
        config = self.config
        x_pos = int(info.get("xscrollHi", 0)) * 256 + int(info.get("xscrollLo", 0))
        lives = info.get("lives")
        level = (int(info.get("levelHi", 0)), int(info.get("levelLo", 0)))
        if self.initial_level[index] is None:
            self.initial_level[index] = level
        if self.current_level[index] is None:
            self.current_level[index] = level

        level_changed = level != self.current_level[index]
        level_completion_event = False
        if level_changed:
            self.completed_level_base[index] += self.level_max_x_pos[index]
            self.completed_level_count[index] += 1
            level_completion_event = not self.current_level_completion_awarded[index]
            self.current_level[index] = level
            self.level_max_x_pos[index] = 0
            self.current_level_completion_awarded[index] = False

        self.level_x_pos[index] = x_pos
        self.level_max_x_pos[index] = max(int(self.level_max_x_pos[index]), x_pos)
        global_x_pos = int(self.completed_level_base[index] + self.level_x_pos[index])
        global_max_x_pos = int(self.completed_level_base[index] + self.level_max_x_pos[index])
        progress_delta = max(0, global_max_x_pos - int(self.max_global_x_pos[index]))
        self.max_global_x_pos[index] = max(int(self.max_global_x_pos[index]), global_max_x_pos)

        threshold_complete = (
            config.completion_x_threshold > 0
            and int(self.level_max_x_pos[index]) >= config.completion_x_threshold
        )
        threshold_completion_event = (
            threshold_complete and not self.current_level_completion_awarded[index]
        )
        completion_event = level_completion_event or threshold_completion_event
        if threshold_completion_event:
            self.current_level_completion_awarded[index] = True
        if completion_event:
            self.completed[index] = True
        if level_completion_event and config.terminate_on_level_change:
            info["_custom_done"] = True
            info["_custom_terminal"] = True
        if completion_event and config.terminate_on_completion:
            info["_custom_done"] = True
            info["_custom_terminal"] = True

        died = False
        if (
            self.prev_lives[index] is not None
            and lives is not None
            and int(lives) < self.prev_lives[index]
        ):
            died = True
            if config.terminate_on_life_loss:
                info["_custom_done"] = True
                info["_custom_terminal"] = True
        if lives is not None:
            self.prev_lives[index] = int(lives)

        progress_reward = min(float(progress_delta), config.progress_reward_cap)
        score = int(info.get("score", 0))
        score_delta = max(0, score - int(self.curr_score[index]))
        self.curr_score[index] = score

        if config.reward_mode == "bounded":
            raw_reward = progress_reward
            if completion_event:
                raw_reward = config.terminal_reward
            if died:
                raw_reward = -config.terminal_reward
            shaped_reward = raw_reward / config.reward_scale if config.reward_scale else raw_reward
        elif config.reward_mode == "score":
            shaped_reward = (
                (float(native_reward) if config.use_retro_reward else 0.0)
                + config.progress_reward_scale * float(progress_delta)
                + 0.01 * float(score_delta)
            )
            if completion_event:
                shaped_reward += config.completion_reward
            if died:
                shaped_reward -= config.death_penalty
        else:
            shaped_reward = (
                float(native_reward) if config.use_retro_reward else 0.0
            ) + config.progress_reward_scale * float(progress_delta)
            if completion_event:
                shaped_reward += config.completion_reward
            if died:
                shaped_reward -= config.death_penalty

        shaped_reward -= config.time_penalty
        if done:
            info["_native_done"] = True
        info["x_pos"] = global_x_pos
        info["level_x_pos"] = int(self.level_x_pos[index])
        info["max_x_pos"] = int(self.max_global_x_pos[index])
        info["level_max_x_pos"] = int(self.level_max_x_pos[index])
        info["progress_delta"] = int(progress_delta)
        info["progress_reward"] = float(progress_reward)
        info["score_delta"] = int(score_delta)
        info["level_changed"] = level_changed
        info["level_complete"] = bool(completion_event)
        info["completed_level_count"] = int(self.completed_level_count[index])
        info["died"] = died
        if died:
            info["death_x_pos"] = int(self.max_global_x_pos[index])
            info["death_level_x_pos"] = int(self.level_max_x_pos[index])
        return float(shaped_reward)


class FrameSkip(gym.Wrapper):
    """Repeat one action for several emulator frames and sum reward."""

    def __init__(self, env: gym.Env, skip: int, max_pool: bool = False):
        super().__init__(env)
        if skip < 1:
            raise ValueError("frame_skip must be >= 1")
        self.skip = skip
        self.max_pool = max_pool

    def step(self, action: Any):
        total_reward = 0.0
        final_obs = None
        pooled_obs: list[np.ndarray] = []
        final_info: dict[str, Any] = {}
        terminated = False
        truncated = False
        for step_idx in range(self.skip):
            final_obs, reward, terminated, truncated, final_info = self.env.step(action)
            total_reward += float(reward)
            if self.max_pool and step_idx >= self.skip - 2 and final_obs is not None:
                pooled_obs.append(final_obs)
            if terminated or truncated:
                break
        if self.max_pool and pooled_obs:
            final_obs = np.maximum.reduce(pooled_obs)
        return final_obs, total_reward, terminated, truncated, final_info


class MarioProgressInfo(gym.Wrapper):
    """Reward true forward progress and add stable x-position metrics."""

    def __init__(
        self,
        env: gym.Env,
        use_retro_reward: bool = False,
        reward_mode: str = "bounded",
        progress_reward_cap: float = 30.0,
        progress_reward_scale: float = 1.0,
        terminal_reward: float = 30.0,
        reward_scale: float = 30.0,
        time_penalty: float = 0.0,
        death_penalty: float = 25.0,
        completion_reward: float = 0.0,
        completion_x_threshold: int = 0,
        terminate_on_life_loss: bool = True,
        terminate_on_level_change: bool = False,
        terminate_on_completion: bool = False,
    ):
        super().__init__(env)
        self.use_retro_reward = use_retro_reward
        if reward_mode not in {"bounded", "additive", "score"}:
            raise ValueError("reward_mode must be 'bounded', 'additive', or 'score'")
        if progress_reward_cap < 0:
            raise ValueError("progress_reward_cap must be >= 0")
        if terminal_reward < 0:
            raise ValueError("terminal_reward must be >= 0")
        if reward_scale < 0:
            raise ValueError("reward_scale must be >= 0")
        self.reward_mode = reward_mode
        self.progress_reward_cap = progress_reward_cap
        self.progress_reward_scale = progress_reward_scale
        self.terminal_reward = terminal_reward
        self.reward_scale = reward_scale
        self.time_penalty = time_penalty
        self.death_penalty = death_penalty
        self.completion_reward = completion_reward
        self.completion_x_threshold = completion_x_threshold
        self.terminate_on_life_loss = terminate_on_life_loss
        self.terminate_on_level_change = terminate_on_level_change
        self.terminate_on_completion = terminate_on_completion
        self.level_x_pos = 0
        self.level_max_x_pos = 0
        self.completed_level_base = 0
        self.max_global_x_pos = 0
        self.curr_score = 0
        self.prev_lives: int | None = None
        self.initial_level: tuple[int, int] | None = None
        self.current_level: tuple[int, int] | None = None
        self.completed_level_count = 0
        self.current_level_completion_awarded = False
        self.completed = False

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.level_x_pos = 0
        self.level_max_x_pos = 0
        self.completed_level_base = 0
        self.max_global_x_pos = 0
        self.curr_score = int(info.get("score", 0))
        lives = info.get("lives")
        self.prev_lives = int(lives) if lives is not None else None
        if "levelHi" in info or "levelLo" in info:
            level = (int(info.get("levelHi", 0)), int(info.get("levelLo", 0)))
            self.initial_level = level
            self.current_level = level
        else:
            self.initial_level = None
            self.current_level = None
        self.completed_level_count = 0
        self.current_level_completion_awarded = False
        self.completed = False
        return obs, info

    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(action)
        x_pos = int(info.get("xscrollHi", 0)) * 256 + int(info.get("xscrollLo", 0))
        lives = info.get("lives")
        level = (int(info.get("levelHi", 0)), int(info.get("levelLo", 0)))
        if self.initial_level is None:
            self.initial_level = level
        if self.current_level is None:
            self.current_level = level

        level_changed = level != self.current_level
        level_completion_event = False
        if level_changed:
            self.completed_level_base += self.level_max_x_pos
            self.completed_level_count += 1
            level_completion_event = not self.current_level_completion_awarded
            self.current_level = level
            self.level_max_x_pos = 0
            self.current_level_completion_awarded = False

        self.level_x_pos = x_pos
        self.level_max_x_pos = max(self.level_max_x_pos, x_pos)
        global_x_pos = self.completed_level_base + self.level_x_pos
        global_max_x_pos = self.completed_level_base + self.level_max_x_pos
        progress_delta = max(0, global_max_x_pos - self.max_global_x_pos)
        self.max_global_x_pos = max(self.max_global_x_pos, global_max_x_pos)

        threshold_complete = (
            self.completion_x_threshold > 0 and self.level_max_x_pos >= self.completion_x_threshold
        )
        threshold_completion_event = (
            threshold_complete and not self.current_level_completion_awarded
        )
        completion_event = level_completion_event or threshold_completion_event
        if threshold_completion_event:
            self.current_level_completion_awarded = True
        if completion_event:
            self.completed = True
        if level_completion_event and self.terminate_on_level_change:
            terminated = True
        if completion_event and self.terminate_on_completion:
            terminated = True

        died = False
        if self.prev_lives is not None and lives is not None and int(lives) < self.prev_lives:
            died = True
            if self.terminate_on_life_loss:
                terminated = True
        if lives is not None:
            self.prev_lives = int(lives)

        progress_reward = min(float(progress_delta), self.progress_reward_cap)
        score = int(info.get("score", self.curr_score))
        score_delta = max(0, score - self.curr_score)
        self.curr_score = score
        completion_bonus = 0.0
        terminal_reward = 0.0
        if self.reward_mode == "bounded":
            raw_reward = self.progress_reward_scale * progress_reward
            if self.use_retro_reward:
                raw_reward += float(reward)
            if died:
                terminal_reward = -self.terminal_reward
                raw_reward = terminal_reward
            elif completion_event:
                terminal_reward = self.terminal_reward
                raw_reward = terminal_reward
            else:
                raw_reward -= self.time_penalty
            clipped_reward = float(np.clip(raw_reward, -self.terminal_reward, self.terminal_reward))
            shaped_reward = (
                clipped_reward / self.reward_scale if self.reward_scale > 0 else clipped_reward
            )
        else:
            if self.reward_mode == "score":
                raw_reward = float(reward) + score_delta / 40.0
                if died:
                    terminal_reward = -self.terminal_reward
                    raw_reward += terminal_reward
                elif completion_event:
                    terminal_reward = self.terminal_reward
                    raw_reward += terminal_reward
                clipped_reward = raw_reward
                shaped_reward = (
                    raw_reward / self.reward_scale if self.reward_scale > 0 else raw_reward
                )
            else:
                raw_reward = self.progress_reward_scale * progress_delta - self.time_penalty
                if self.use_retro_reward:
                    raw_reward += float(reward)
                if completion_event:
                    completion_bonus = self.completion_reward
                    raw_reward += completion_bonus
                if died:
                    raw_reward -= self.death_penalty
                clipped_reward = raw_reward
                shaped_reward = raw_reward

        info = dict(info)
        info["x_pos"] = global_x_pos
        info["max_x_pos"] = self.max_global_x_pos
        info["level_x_pos"] = self.level_x_pos
        info["level_max_x_pos"] = self.level_max_x_pos
        info["completed_level_base"] = self.completed_level_base
        info["global_x_pos"] = global_x_pos
        info["global_max_x_pos"] = self.max_global_x_pos
        info["progress_delta"] = progress_delta
        info["level_id"] = f"{level[0]}-{level[1]}"
        info["level_changed"] = level_changed
        info["completed_level_count"] = self.completed_level_count
        info["threshold_complete"] = threshold_complete
        info["level_complete"] = self.completed
        info["completion_event"] = completion_event
        info["terminate_on_level_change"] = self.terminate_on_level_change
        info["terminate_on_completion"] = self.terminate_on_completion
        info["completion_bonus"] = completion_bonus
        info["reward_mode"] = self.reward_mode
        info["progress_reward"] = progress_reward
        info["score_delta"] = score_delta
        info["terminal_reward"] = terminal_reward
        info["raw_reward"] = raw_reward
        info["clipped_reward"] = clipped_reward
        info["reward_scale"] = self.reward_scale
        info["time_penalty"] = self.time_penalty
        info["shaped_reward"] = shaped_reward
        info["died"] = died
        if died:
            info["death_x_pos"] = self.max_global_x_pos
            info["death_level_x_pos"] = self.level_max_x_pos
        return obs, shaped_reward, terminated, truncated, info


class MarioPreprocess(gym.ObservationWrapper):
    """Crop optional HUD rows, then convert RGB frames to grayscale observations."""

    def __init__(self, env: gym.Env, size: int = 84, hud_crop_top: int = 0):
        super().__init__(env)
        if hud_crop_top < 0:
            raise ValueError("hud_crop_top must be >= 0")
        self.size = size
        self.hud_crop_top = hud_crop_top
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(size, size, 1),
            dtype=np.uint8,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        if self.hud_crop_top >= observation.shape[0]:
            raise ValueError(
                f"hud_crop_top={self.hud_crop_top} must be less than frame height {observation.shape[0]}",
            )
        frame = observation[self.hud_crop_top :, :, :]
        gray = np.dot(frame[..., :3], np.array([0.299, 0.587, 0.114])).astype(np.uint8)
        resized = cv2.resize(gray, (self.size, self.size), interpolation=cv2.INTER_AREA)
        return resized[..., None]


def make_mario_env(config: EnvConfig | None = None, seed: int | None = None) -> gym.Env:
    config = config or EnvConfig()
    env = retro.make(config.game, state=config.state, render_mode="rgb_array")
    return wrap_mario_env(env, config=config, seed=seed)


def make_fast_mario_env(config: EnvConfig | None = None, seed: int | None = None) -> gym.Env:
    config = config or EnvConfig()
    env = retro.make(
        config.game,
        state=config.state,
        render_mode="rgb_array",
        obs_resize=(config.observation_size, config.observation_size),
        obs_crop=(config.hud_crop_top, 0, 0, 0) if config.hud_crop_top else None,
        obs_grayscale=True,
        obs_resize_algorithm=config.obs_resize_algorithm,
        frame_skip=config.frame_skip,
        frame_stack=4,
        maxpool_last_two=config.max_pool_frames,
    )
    env = DiscreteMarioActions(env, action_set=config.action_set)
    env = MarioProgressInfo(
        env,
        use_retro_reward=config.use_retro_reward,
        reward_mode=config.reward_mode,
        progress_reward_cap=config.progress_reward_cap,
        progress_reward_scale=config.progress_reward_scale,
        terminal_reward=config.terminal_reward,
        reward_scale=config.reward_scale,
        time_penalty=config.time_penalty,
        death_penalty=config.death_penalty,
        completion_reward=config.completion_reward,
        completion_x_threshold=config.completion_x_threshold,
        terminate_on_life_loss=config.terminate_on_life_loss,
        terminate_on_level_change=config.terminate_on_level_change,
        terminate_on_completion=config.terminate_on_completion,
    )
    env = gym.wrappers.TimeLimit(env, max_episode_steps=config.max_episode_steps)
    if config.clip_rewards:
        env = ClipRewardEnv(env)
    if seed is not None:
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    return env


def make_rendered_mario_env(config: EnvConfig | None = None, seed: int | None = None) -> gym.Env:
    config = config or EnvConfig()
    env = retro.make(config.game, state=config.state, render_mode="human")
    return wrap_mario_env(env, config=config, seed=seed)


def wrap_mario_env(env: gym.Env, config: EnvConfig, seed: int | None = None) -> gym.Env:
    env = DiscreteMarioActions(env, action_set=config.action_set)
    env = FrameSkip(env, config.frame_skip, max_pool=config.max_pool_frames)
    env = MarioProgressInfo(
        env,
        use_retro_reward=config.use_retro_reward,
        reward_mode=config.reward_mode,
        progress_reward_cap=config.progress_reward_cap,
        progress_reward_scale=config.progress_reward_scale,
        terminal_reward=config.terminal_reward,
        reward_scale=config.reward_scale,
        time_penalty=config.time_penalty,
        death_penalty=config.death_penalty,
        completion_reward=config.completion_reward,
        completion_x_threshold=config.completion_x_threshold,
        terminate_on_life_loss=config.terminate_on_life_loss,
        terminate_on_level_change=config.terminate_on_level_change,
        terminate_on_completion=config.terminate_on_completion,
    )
    env = MarioPreprocess(env, config.observation_size, hud_crop_top=config.hud_crop_top)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=config.max_episode_steps)
    if config.clip_rewards:
        env = ClipRewardEnv(env)
    if seed is not None:
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    return env


def make_env_fn(rank: int, seed: int, config: EnvConfig) -> Callable[[], gym.Env]:
    def _init() -> gym.Env:
        env_config = config
        if config.states:
            env_config = replace(config, state=config.states[rank % len(config.states)])
        env = make_fast_mario_env(config=env_config, seed=seed + rank)
        env.reset(seed=seed + rank)
        return env

    return _init


def make_vec_envs(config: EnvConfig, n_envs: int, seed: int, start_method: str = "fork"):
    os.environ.setdefault("STABLE_RETRO_DISABLE_AUDIO", "1")
    if config.states:
        raise ValueError(
            "StableRetroNativeVecEnv supports one homogeneous state per vector env; "
            "use --state instead of --states for native rollouts.",
        )
    vec_env = StableRetroNativeVecEnv(
        config.game,
        num_envs=n_envs,
        state=config.state,
        num_threads=min(max(n_envs, 1), 16),
        render_mode="rgb_array",
        obs_resize=(config.observation_size, config.observation_size),
        obs_crop=(config.hud_crop_top, 0, 0, 0) if config.hud_crop_top else None,
        obs_grayscale=True,
        obs_resize_algorithm=config.obs_resize_algorithm,
        frame_skip=config.frame_skip,
        frame_stack=4,
        maxpool_last_two=config.max_pool_frames,
        copy_observations=False,
    )
    vec_env.seed(seed)
    vec_env = VecDiscreteMarioActions(vec_env, action_set=config.action_set)
    vec_env = VecMarioProgressInfo(vec_env, config=config)
    vec_env = VecMonitor(vec_env)
    return VecTransposeImage(vec_env)


def assert_rom_imported(game: str = GAME) -> str:
    try:
        return retro.data.get_romfile_path(game)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"{game} is not imported. Run: uv run python scripts/import_roms.py ~/Desktop/roms",
        ) from exc


def default_run_dir(run_name: str, runs_dir: str = "runs") -> str:
    return os.path.join(runs_dir, run_name)
