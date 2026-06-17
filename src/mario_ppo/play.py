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
from mario_ppo.env import DEFAULT_HUD_CROP_TOP, assert_rom_imported, make_rendered_replay_env
from mario_ppo.env_config import env_config_from_args
from mario_ppo.eval_metrics import single_env_action


def stacked_obs(frames: deque[np.ndarray]) -> np.ndarray:
    # Model was trained with VecFrameStack + VecTransposeImage: (n_env, 4, 84, 84).
    return np.stack([frame[..., 0] for frame in frames], axis=0)[None, ...]


def render_obs_stack(frames: deque[np.ndarray], scale: int) -> np.ndarray:
    if scale < 1:
        raise ValueError("--obs-stack-scale must be >= 1")
    panels = []
    for idx, frame in enumerate(frames):
        gray = frame[..., 0]
        panel = np.repeat(gray[..., None], 3, axis=2)
        if scale != 1:
            panel = np.repeat(np.repeat(panel, scale, axis=0), scale, axis=1)
        panels.append(panel)
    image = np.concatenate(panels, axis=1)

    try:
        import cv2
    except ImportError:
        return image

    label_height = max(18, 14 * scale)
    canvas = np.zeros((image.shape[0] + label_height, image.shape[1], 3), dtype=np.uint8)
    canvas[label_height:, :, :] = image
    panel_width = panels[0].shape[1]
    labels = ["t-3", "t-2", "t-1", "t"]
    for idx, label in enumerate(labels):
        cv2.putText(
            canvas,
            label,
            (idx * panel_width + 4, label_height - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35 * scale,
            (220, 220, 220),
            max(1, scale),
            cv2.LINE_AA,
        )
    return canvas


class PygameViewer:
    def __init__(
        self, frame_shape: tuple[int, int, int], scale: int, position: tuple[int, int] | None = None
    ):
        if scale < 1:
            raise ValueError("--scale must be >= 1")
        height, width, _channels = frame_shape
        self.size = (width * scale, height * scale)
        if position is not None:
            os.environ["SDL_VIDEO_WINDOW_POS"] = f"{position[0]},{position[1]}"
        pygame.init()
        pygame.display.set_caption("Mario PPO")
        self.screen = pygame.display.set_mode(self.size)
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


class OptionsPanel:
    def __init__(self, fps: float, show_obs_stack: bool, position: tuple[int, int] | None = None):
        self.window_name = "Mario PPO controls"
        self.cv2 = None
        self.show_obs_stack = show_obs_stack
        self.obs_button_rect = (10, 8, 170, 28)

        try:
            import cv2
        except ImportError:
            print("cv2 is not installed; --control-panel is disabled.", flush=True)
            return

        self.cv2 = cv2
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        if position is not None:
            cv2.moveWindow(self.window_name, position[0], position[1])
        cv2.createTrackbar(
            "FPS", self.window_name, int(max(0, min(round(fps), 240))), 240, lambda _v: None
        )
        cv2.setMouseCallback(self.window_name, self._on_mouse)

    def _on_mouse(self, event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if self.cv2 is None or event != self.cv2.EVENT_LBUTTONDOWN:
            return
        rect_x, rect_y, rect_w, rect_h = self.obs_button_rect
        if rect_x <= x <= rect_x + rect_w and rect_y <= y <= rect_y + rect_h:
            self.show_obs_stack = not self.show_obs_stack

    def poll(self, actual_fps: float | None = None) -> tuple[bool, float, bool]:
        if self.cv2 is None:
            return True, 0.0, False

        fps_pos = self.cv2.getTrackbarPos("FPS", self.window_name)
        fps = 0.0 if fps_pos <= 0 else float(fps_pos)

        canvas = np.zeros((94, 260, 3), dtype=np.uint8)

        rect_x, rect_y, rect_w, rect_h = self.obs_button_rect
        button_color = (40, 130, 70) if self.show_obs_stack else (70, 70, 70)
        border_color = (90, 220, 120) if self.show_obs_stack else (170, 170, 170)
        self.cv2.rectangle(
            canvas, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), button_color, -1
        )
        self.cv2.rectangle(
            canvas, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), border_color, 1
        )
        label = f"Obs stack: {'ON' if self.show_obs_stack else 'OFF'}"
        self.cv2.putText(
            canvas,
            label,
            (rect_x + 10, rect_y + 20),
            self.cv2.FONT_HERSHEY_PLAIN,
            1.2,
            (255, 255, 255),
            1,
            self.cv2.LINE_AA,
        )

        lines = [
            f"target_fps   : {'max' if fps <= 0 else int(fps)}",
            f"measured_fps : {'...' if actual_fps is None else f'{actual_fps:.1f}'}",
        ]
        for idx, line in enumerate(lines):
            self.cv2.putText(
                canvas,
                line,
                (10, 56 + idx * 18),
                self.cv2.FONT_HERSHEY_PLAIN,
                1.0,
                (210, 210, 210),
                1,
                self.cv2.LINE_AA,
            )
        self.cv2.imshow(self.window_name, canvas)
        key = self.cv2.waitKey(1) & 0xFF
        return key not in {27, ord("q")}, fps, self.show_obs_stack

    def close(self) -> None:
        if self.cv2 is None:
            return
        self.cv2.destroyWindow(self.window_name)


class ObsStackViewer:
    def __init__(self, scale: int, position: tuple[int, int] | None = None):
        self.scale = scale
        self.window_name = "Mario PPO obs framestack"
        self.cv2 = None

        try:
            import cv2
        except ImportError:
            print("cv2 is not installed; --show-obs-stack is disabled.", flush=True)
            return

        self.cv2 = cv2
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        if position is not None:
            cv2.moveWindow(self.window_name, position[0], position[1])

    def show(self, frames: deque[np.ndarray]) -> bool:
        if self.cv2 is None:
            return True
        image = render_obs_stack(frames, self.scale)
        self.cv2.imshow(self.window_name, self.cv2.cvtColor(image, self.cv2.COLOR_RGB2BGR))
        key = self.cv2.waitKey(1) & 0xFF
        return key not in {27, ord("q")}

    def close(self) -> None:
        if self.cv2 is None:
            return
        self.cv2.destroyWindow(self.window_name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Show a PPO checkpoint playing Mario in a GUI window"
    )
    parser.add_argument("--model", default="runs/smoke_doc/final_model.zip")
    parser.add_argument("--game", default="SuperMarioBros-Nes-v0")
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
    parser.add_argument(
        "--episodes", type=int, default=3, help="Number of episodes; use 0 to run forever"
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument(
        "--random-seeds", action="store_true", help="Use a fresh random seed each episode"
    )
    parser.add_argument("--fps", type=float, default=0.0)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument(
        "--show-obs-stack",
        action="store_true",
        help="Open a second window showing the four preprocessed frames fed to the policy.",
    )
    parser.add_argument("--obs-stack-scale", type=int, default=4)
    parser.add_argument(
        "--control-panel",
        action="store_true",
        help="Open controls for FPS and the observation framestack diagnostic window.",
    )
    parser.add_argument("--stochastic", action="store_true", help="Sample from the policy")
    parser.add_argument(
        "--reward-mode",
        choices=["baseline", "bounded", "additive", "score", "native"],
        default="baseline",
    )
    parser.add_argument("--progress-reward-cap", type=float, default=30.0)
    parser.add_argument("--progress-reward-scale", type=float, default=1.0)
    parser.add_argument("--terminal-reward", type=float, default=50.0)
    parser.add_argument("--reward-scale", type=float, default=10.0)
    parser.add_argument("--time-penalty", type=float, default=0.0)
    parser.add_argument("--death-penalty", type=float, default=25.0)
    parser.add_argument("--completion-reward", type=float, default=0.0)
    parser.add_argument("--score-progress-clipped", action="store_true")
    parser.add_argument("--no-progress-timeout-steps", type=int, default=0)
    parser.add_argument("--no-progress-min-delta", type=int, default=0)
    parser.add_argument("--completion-x-threshold", type=int, default=0)
    parser.add_argument("--no-terminate-on-life-loss", action="store_true")
    parser.add_argument("--terminate-on-level-change", action="store_true")
    parser.add_argument("--terminate-on-completion", action="store_true")
    parser.add_argument("--action-set", choices=["simple", "right", "native"], default="simple")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    assert_rom_imported()
    model = PPO.load(args.model, device=resolve_sb3_device(args.device))
    config = env_config_from_args(args, max_episode_steps_attr="max_steps")
    env = make_rendered_replay_env(config=config, seed=args.seed)
    seed_rng = np.random.default_rng() if args.random_seeds else None

    obs, _ = env.reset(seed=args.seed)
    first_frame = env.render()
    game_position = (460, 60) if args.control_panel else None
    controls_position = (40, 60)
    obs_stack_position = (40, 240)
    viewer = PygameViewer(first_frame.shape, scale=args.scale, position=game_position)
    obs_viewer = (
        ObsStackViewer(scale=args.obs_stack_scale, position=obs_stack_position)
        if args.show_obs_stack
        else None
    )
    options_panel = (
        OptionsPanel(fps=args.fps, show_obs_stack=args.show_obs_stack, position=controls_position)
        if args.control_panel
        else None
    )
    current_fps = args.fps
    actual_fps: float | None = None
    fps_ema_alpha = 0.12
    last_frame_at = time.perf_counter()

    def throttle() -> None:
        nonlocal actual_fps, last_frame_at
        if current_fps <= 0:
            now = time.perf_counter()
            elapsed = now - last_frame_at
            if elapsed > 0:
                instantaneous_fps = 1.0 / elapsed
                actual_fps = (
                    instantaneous_fps
                    if actual_fps is None
                    else (1.0 - fps_ema_alpha) * actual_fps + fps_ema_alpha * instantaneous_fps
                )
            last_frame_at = time.perf_counter()
            return

        target_interval = 1.0 / current_fps
        now = time.perf_counter()
        target_frame_at = last_frame_at + target_interval
        while now < target_frame_at:
            delay = target_frame_at - now
            time.sleep(min(delay, 0.02))
            now = time.perf_counter()
        elapsed = now - last_frame_at
        if elapsed > 0:
            instantaneous_fps = 1.0 / elapsed
            actual_fps = (
                instantaneous_fps
                if actual_fps is None
                else (1.0 - fps_ema_alpha) * actual_fps + fps_ema_alpha * instantaneous_fps
            )
        last_frame_at = now

    def update_controls(frames: deque[np.ndarray] | None = None) -> bool:
        nonlocal current_fps, obs_viewer
        if options_panel is None:
            if obs_viewer is not None and frames is not None:
                return obs_viewer.show(frames)
            return True

        should_continue, fps, show_obs_stack = options_panel.poll(actual_fps=actual_fps)
        if not should_continue:
            return False
        current_fps = fps
        if show_obs_stack and obs_viewer is None:
            obs_viewer = ObsStackViewer(scale=args.obs_stack_scale, position=obs_stack_position)
        elif not show_obs_stack and obs_viewer is not None:
            obs_viewer.close()
            obs_viewer = None
        if obs_viewer is not None and frames is not None:
            return obs_viewer.show(frames)
        return True

    try:
        if not update_controls():
            return
        if not viewer.show(first_frame, ["r_step: 0.00", "r_total: 0.00", "max_x: 0", "step: 0"]):
            return
        throttle()
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
            throttle()
            frames: deque[np.ndarray] = deque([obs] * 4, maxlen=4)
            if not update_controls(frames):
                return
            total_reward = 0.0
            max_x_pos = 0
            final_info = {}
            for step_idx in range(args.max_steps):
                action, _ = model.predict(stacked_obs(frames), deterministic=not args.stochastic)
                obs, reward, terminated, truncated, info = env.step(single_env_action(action))
                frames.append(obs)
                if not update_controls(frames):
                    return
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
                throttle()
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
        if options_panel is not None:
            options_panel.close()
        if obs_viewer is not None:
            obs_viewer.close()
        viewer.close()
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
