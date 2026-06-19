from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

from stable_retro_ppo.env import EnvConfig, resolve_env_config


DEFAULT_EVAL_PROFILE = "mario_level1_v1"


@dataclass(frozen=True)
class EvalProfile:
    name: str
    game: str
    state: str
    frame_skip: int
    max_pool_frames: bool
    frame_stack: int
    grayscale: bool
    observation_size: int
    hud_crop_top: int
    reward_mode: str
    progress_reward_cap: float
    progress_reward_scale: float
    terminal_reward: float
    reward_scale: float
    time_penalty: float
    death_penalty: float
    completion_reward: float
    score_progress_clipped: bool
    no_progress_timeout_steps: int
    no_progress_min_delta: int
    action_set: str
    max_episode_steps: int
    completion_x_threshold: int
    terminate_on_life_loss: bool
    terminate_on_level_change: bool
    terminate_on_completion: bool
    n_envs: int
    max_steps: int
    deterministic: bool

    def env_config(self) -> EnvConfig:
        return resolve_env_config(
            EnvConfig(
                game=self.game,
                state=self.state,
                frame_skip=self.frame_skip,
                max_pool_frames=self.max_pool_frames,
                max_episode_steps=self.max_episode_steps,
                observation_size=self.observation_size,
                hud_crop_top=self.hud_crop_top,
                reward_mode=self.reward_mode,
                progress_reward_cap=self.progress_reward_cap,
                progress_reward_scale=self.progress_reward_scale,
                terminal_reward=self.terminal_reward,
                reward_scale=self.reward_scale,
                time_penalty=self.time_penalty,
                death_penalty=self.death_penalty,
                completion_reward=self.completion_reward,
                score_progress_clipped=self.score_progress_clipped,
                no_progress_timeout_steps=self.no_progress_timeout_steps,
                no_progress_min_delta=self.no_progress_min_delta,
                completion_x_threshold=self.completion_x_threshold,
                terminate_on_life_loss=self.terminate_on_life_loss,
                terminate_on_level_change=self.terminate_on_level_change,
                terminate_on_completion=self.terminate_on_completion,
                action_set=self.action_set,
            )
        )

    def metadata(self) -> dict[str, Any]:
        return asdict(self)


MARIO_LEVEL1_V1 = EvalProfile(
    name="mario_level1_v1",
    game="SuperMarioBros-Nes-v0",
    state="Level1-1",
    frame_skip=4,
    max_pool_frames=True,
    frame_stack=4,
    grayscale=True,
    observation_size=84,
    hud_crop_top=32,
    reward_mode="score",
    progress_reward_cap=30.0,
    progress_reward_scale=1.0,
    terminal_reward=50.0,
    reward_scale=10.0,
    time_penalty=0.0,
    death_penalty=25.0,
    completion_reward=0.0,
    score_progress_clipped=False,
    no_progress_timeout_steps=0,
    no_progress_min_delta=0,
    action_set="simple",
    max_episode_steps=4500,
    completion_x_threshold=3160,
    terminate_on_life_loss=True,
    terminate_on_level_change=False,
    terminate_on_completion=True,
    n_envs=1,
    max_steps=2500,
    deterministic=False,
)

MARIO_LEVEL1_NO_LIFE_LOSS_V1 = replace(
    MARIO_LEVEL1_V1,
    name="mario_level1_no_life_loss_v1",
    terminate_on_life_loss=False,
)

MARIO_LEVEL1_VEC8_V1 = replace(
    MARIO_LEVEL1_V1,
    name="mario_level1_vec8_v1",
    n_envs=8,
)

MARIO_LEVEL1_VEC20_V1 = replace(
    MARIO_LEVEL1_V1,
    name="mario_level1_vec20_v1",
    n_envs=20,
)

MARIO_LEVEL1_VEC24_V1 = replace(
    MARIO_LEVEL1_V1,
    name="mario_level1_vec24_v1",
    n_envs=24,
)


EVAL_PROFILES: dict[str, EvalProfile] = {
    MARIO_LEVEL1_V1.name: MARIO_LEVEL1_V1,
    MARIO_LEVEL1_NO_LIFE_LOSS_V1.name: MARIO_LEVEL1_NO_LIFE_LOSS_V1,
    MARIO_LEVEL1_VEC8_V1.name: MARIO_LEVEL1_VEC8_V1,
    MARIO_LEVEL1_VEC20_V1.name: MARIO_LEVEL1_VEC20_V1,
    MARIO_LEVEL1_VEC24_V1.name: MARIO_LEVEL1_VEC24_V1,
}


def get_eval_profile(name: str = DEFAULT_EVAL_PROFILE) -> EvalProfile:
    try:
        return EVAL_PROFILES[name]
    except KeyError as exc:
        known = ", ".join(sorted(EVAL_PROFILES))
        raise ValueError(f"unknown eval profile {name!r}; known profiles: {known}") from exc
