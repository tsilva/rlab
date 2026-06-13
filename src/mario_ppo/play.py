from __future__ import annotations

# ruff: noqa: E402

import argparse
import os
import time
from collections import deque
from itertools import count

os.environ.setdefault("MPLCONFIGDIR", os.path.abspath(".matplotlib"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import numpy as np
import pygame
import torch
from stable_baselines3 import PPO

from mario_ppo.device import resolve_sb3_device
from mario_ppo.env import DEFAULT_HUD_CROP_TOP, EnvConfig, assert_rom_imported, make_mario_env


def stacked_obs(frames: deque[np.ndarray]) -> np.ndarray:
    # Model was trained with VecFrameStack + VecTransposeImage: (n_env, 4, 84, 84).
    return np.stack([frame[..., 0] for frame in frames], axis=0)[None, ...]


class PygameViewer:
    def __init__(self, frame_shape: tuple[int, int, int], scale: int, fps: float):
        if scale < 1:
            raise ValueError("--scale must be >= 1")
        height, width, _channels = frame_shape
        self.size = (width * scale, height * scale)
        self.fps = fps
        pygame.init()
        pygame.display.set_caption("Mario PPO")
        self.screen = pygame.display.set_mode(self.size)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, max(16, 5 * scale))

    def show(self, frame: np.ndarray, overlay: list[str] | None = None) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        surface = pygame.transform.scale(surface, self.size)
        self.screen.blit(surface, (0, 0))
        if overlay:
            self.draw_overlay(overlay)
        pygame.display.flip()
        if self.fps > 0:
            self.clock.tick(self.fps)
        return True

    def draw_overlay(self, lines: list[str]) -> None:
        padding = 6
        line_height = self.font.get_height() + 2
        width = max(self.font.size(line)[0] for line in lines) + padding * 2
        height = line_height * len(lines) + padding * 2
        background = pygame.Surface((width, height), pygame.SRCALPHA)
        background.fill((0, 0, 0, 160))
        self.screen.blit(background, (0, 0))
        for idx, line in enumerate(lines):
            text = self.font.render(line, True, (255, 255, 255))
            self.screen.blit(text, (padding, padding + idx * line_height))

    def close(self) -> None:
        pygame.quit()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Show a PPO checkpoint playing Mario in a GUI window")
    parser.add_argument("--model", default="runs/smoke_doc/final_model.zip")
    parser.add_argument("--state", default="Level1-1")
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--max-pool-frames", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument(
        "--hud-crop-top",
        type=int,
        default=DEFAULT_HUD_CROP_TOP,
        help="Crop this many pixels from the top of raw frames before grayscale resize.",
    )
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes; use 0 to run forever")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--random-seeds", action="store_true", help="Use a fresh random seed each episode")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--stochastic", action="store_true", help="Sample from the policy")
    parser.add_argument("--reward-mode", choices=["bounded", "additive", "score"], default="bounded")
    parser.add_argument("--progress-reward-cap", type=float, default=30.0)
    parser.add_argument("--progress-reward-scale", type=float, default=1.0)
    parser.add_argument("--terminal-reward", type=float, default=30.0)
    parser.add_argument("--reward-scale", type=float, default=30.0)
    parser.add_argument("--time-penalty", type=float, default=0.0)
    parser.add_argument("--death-penalty", type=float, default=25.0)
    parser.add_argument("--completion-reward", type=float, default=0.0)
    parser.add_argument("--completion-x-threshold", type=int, default=0)
    parser.add_argument("--no-terminate-on-life-loss", action="store_true")
    parser.add_argument("--terminate-on-level-change", action="store_true")
    parser.add_argument("--terminate-on-completion", action="store_true")
    parser.add_argument("--action-set", choices=["simple", "right"], default="simple")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    assert_rom_imported()
    model = PPO.load(args.model, device=resolve_sb3_device(args.device))
    config = EnvConfig(
        state=args.state,
        frame_skip=args.frame_skip,
        max_pool_frames=args.max_pool_frames,
        max_episode_steps=args.max_steps,
        hud_crop_top=args.hud_crop_top,
        reward_mode=args.reward_mode,
        progress_reward_cap=args.progress_reward_cap,
        progress_reward_scale=args.progress_reward_scale,
        terminal_reward=args.terminal_reward,
        reward_scale=args.reward_scale,
        time_penalty=args.time_penalty,
        death_penalty=args.death_penalty,
        completion_reward=args.completion_reward,
        completion_x_threshold=args.completion_x_threshold,
        terminate_on_life_loss=not args.no_terminate_on_life_loss,
        terminate_on_level_change=args.terminate_on_level_change,
        terminate_on_completion=args.terminate_on_completion,
        action_set=args.action_set,
    )
    env = make_mario_env(config=config, seed=args.seed)
    seed_rng = np.random.default_rng() if args.random_seeds else None

    obs, _ = env.reset(seed=args.seed)
    first_frame = env.render()
    viewer = PygameViewer(first_frame.shape, scale=args.scale, fps=args.fps)

    try:
        if not viewer.show(first_frame, ["r_step: 0.00", "r_total: 0.00", "max_x: 0", "step: 0"]):
            return
        episode_iter = count() if args.episodes <= 0 else range(args.episodes)
        for episode in episode_iter:
            episode_seed = (
                int(seed_rng.integers(0, np.iinfo(np.int32).max))
                if seed_rng is not None
                else args.seed + episode
            )
            torch.manual_seed(episode_seed)
            obs, _ = env.reset(seed=episode_seed)
            frame = env.render()
            if not viewer.show(
                frame,
                [
                    "r_step: 0.00",
                    "r_total: 0.00",
                    "dx: 0 penalty: 0.00",
                    "max_x: 0",
                    f"step: 0 seed: {episode_seed}",
                ],
            ):
                break
            frames: deque[np.ndarray] = deque([obs] * 4, maxlen=4)
            total_reward = 0.0
            max_x_pos = 0
            final_info = {}
            for step_idx in range(args.max_steps):
                action, _ = model.predict(stacked_obs(frames), deterministic=not args.stochastic)
                obs, reward, terminated, truncated, info = env.step(int(action[0]))
                frames.append(obs)
                total_reward += float(reward)
                max_x_pos = max(max_x_pos, int(info.get("max_x_pos", 0)))
                final_info = dict(info)
                frame = env.render()
                overlay = [
                    f"r_step: {float(reward):.2f}",
                    f"r_total: {total_reward:.2f}",
                    (
                        f"dx: {int(info.get('progress_delta', 0))} "
                        f"penalty: {float(info.get('time_penalty', 0.0)):.2f}"
                    ),
                    (
                        f"bonus: {float(info.get('completion_bonus', 0.0)):.0f} "
                        f"shaped: {float(info.get('shaped_reward', reward)):.2f}"
                    ),
                    f"max_x: {max_x_pos}",
                    f"step: {step_idx + 1} seed: {episode_seed}",
                ]
                if not viewer.show(frame, overlay):
                    return
                if terminated or truncated:
                    status = "terminated" if terminated else "truncated"
                    print(
                        "episode="
                        f"{episode + 1} seed={episode_seed} reward={total_reward:.2f} "
                        f"max_x={max_x_pos} steps={step_idx + 1} status={status} "
                        f"died={bool(final_info.get('died', False))} "
                        f"complete={bool(final_info.get('level_complete', False))}",
                        flush=True,
                    )
                    time.sleep(0.5)
                    break
            else:
                print(
                    "episode="
                    f"{episode + 1} seed={episode_seed} reward={total_reward:.2f} "
                    f"max_x={max_x_pos} steps={args.max_steps} status=max_steps "
                    f"died={bool(final_info.get('died', False))} "
                    f"complete={bool(final_info.get('level_complete', False))}",
                    flush=True,
                )
    finally:
        viewer.close()
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
