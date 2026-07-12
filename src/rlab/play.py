from __future__ import annotations

# ruff: noqa: E402

import argparse
import contextlib
import os
import sys
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import replace
from itertools import count
from types import ModuleType

os.environ.setdefault("MPLCONFIGDIR", os.path.abspath(".matplotlib"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import torch

from rlab.artifacts import load_playback_env_config
from rlab.device import resolve_sb3_device
from rlab.env import (
    assert_provider_runtime_available,
    info_value_from_state_name,
    make_eval_vec_env,
    state_name_candidates_from_level_id,
    task_action_set,
    task_conditioning,
    task_conditioning_info_values,
    task_max_episode_steps,
    task_reward,
    task_termination,
    with_task_termination,
)
from rlab.eval_metrics import (
    batch_metrics_for_lane,
    drain_runtime_records,
    episode_records,
    episode_result_from_record,
    is_level_complete,
    single_env_action,
)
from rlab.env_registry import ALE_PY_PROVIDER, is_stable_retro_atari_env
from rlab.model_sources import (
    model_source_ref,
    positional_model_source_arg,
    resolve_single_model_source,
)
from rlab.play_attribution import PolicyActionAttributor
from rlab.policy_observation import (
    model_observation,
    task_info_value_from_info,
    task_info_vars,
    task_state_names,
)
from rlab.seeds import DEFAULT_EVAL_SEED, EVAL_SEED_START, validate_eval_seed
from rlab.targets import target_for_game
from rlab.wandb_utils import default_wandb_project_path


ANSI_RESET = "\033[0m"
ANSI_STYLES = {
    "bold": "\033[1m",
    "dim": "\033[2m",
    "cyan": "\033[36m",
    "green": "\033[32m",
    "blue": "\033[34m",
    "yellow": "\033[33m",
    "magenta": "\033[35m",
}
DEFAULT_VIEWER_SCALE = 4
DEFAULT_OBS_VIEWER_SCALE = 4
ATTRIBUTION_MODES = ("none", "gradcam", "occlusion")


def _color(text: str, style: str) -> str:
    if os.environ.get("NO_COLOR") or os.environ.get("RLAB_NO_COLOR"):
        return text
    return f"{ANSI_STYLES[style]}{text}{ANSI_RESET}"


def _summary_line(icon: str, label: str, value: str, style: str) -> str:
    return f"  {_color(icon, style)} {_color(label + ':', 'dim')} {value}"


def _format_sequence(value) -> str:
    if not value:
        return "-"
    if isinstance(value, str):
        return value
    return ",".join(str(item) for item in value)


@contextlib.contextmanager
def _suppress_native_stderr():
    """Hide noisy native-library import chatter while keeping normal stderr intact."""

    try:
        stderr_fd = sys.stderr.fileno()
    except AttributeError, OSError, ValueError:
        yield
        return
    saved_stderr_fd = os.dup(stderr_fd)
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), stderr_fd)
        yield
    finally:
        os.dup2(saved_stderr_fd, stderr_fd)
        os.close(saved_stderr_fd)


def import_pygame() -> ModuleType:
    if "pygame" in sys.modules:
        return sys.modules["pygame"]
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    with _suppress_native_stderr():
        import pygame
    return pygame


def fast_env_image_obs(obs) -> np.ndarray:
    if isinstance(obs, Mapping):
        if "image" not in obs:
            raise ValueError(f"dict fast env obs is missing 'image'; keys={tuple(obs)}")
        obs = obs["image"]
    return np.asarray(obs)


def fast_env_obs(obs: np.ndarray) -> np.ndarray:
    # Older native envs may expose HWC stacks; post12 exposes CHW stacks.
    # The policy always receives SB3's channel-first batch layout.
    arr = fast_env_image_obs(obs)
    if arr.ndim == 4 and arr.shape[0] == 1 and arr.shape[-1] == 4:
        return np.transpose(arr, (0, 3, 1, 2))
    if arr.ndim == 4 and arr.shape[0] == 1 and arr.shape[1] == 4:
        return arr
    if arr.ndim == 3 and arr.shape[-1] == 4:
        return np.transpose(arr, (2, 0, 1))[None, ...]
    if arr.ndim == 3 and arr.shape[0] == 4:
        return arr[None, ...]
    raise ValueError(f"expected fast env obs with 4 stacked frames, got shape {arr.shape}")


def fast_env_frames(obs: np.ndarray) -> deque[np.ndarray]:
    arr = fast_env_image_obs(obs)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[-1] == 4:
        return deque([arr[..., idx : idx + 1] for idx in range(arr.shape[-1])], maxlen=4)
    if arr.ndim == 3 and arr.shape[0] == 4:
        return deque([arr[idx, ..., None] for idx in range(arr.shape[0])], maxlen=4)
    raise ValueError(f"expected fast env obs with 4 stacked frames, got shape {arr.shape}")


def task_conditioning_change_message(
    *,
    episode: int,
    step: int,
    old_task: object,
    new_task: object,
    task_index: int,
    task_count: int,
) -> str:
    task_vector = [1 if index == task_index else 0 for index in range(task_count)]
    return (
        "task_conditioning_change "
        f"episode={episode} step={step} old={old_task!r} new={new_task!r} "
        f"index={task_index} one_hot={task_vector}"
    )


def task_conditioning_start_message(
    *,
    episode: int,
    step: int,
    task: object,
    task_index: int,
    task_count: int,
) -> str:
    task_vector = [1 if index == task_index else 0 for index in range(task_count)]
    return (
        "task_conditioning_start "
        f"episode={episode} step={step} task={task!r} "
        f"index={task_index} one_hot={task_vector}"
    )


def task_state_from_info(info: dict, task_states: tuple[str, ...]) -> str | None:
    level_id = info.get("level_id")
    if not isinstance(level_id, str) or not level_id:
        return None
    for level_state in state_name_candidates_from_level_id(level_id):
        if level_state in task_states:
            return level_state
    return None


def playback_should_end_episode(terminated: bool, truncated: bool, completed: bool) -> bool:
    # Completion is shown in playback output, but GUI playback keeps going unless
    # the environment actually terminates or truncates the episode.
    del completed
    return bool(terminated or truncated)


def playback_step_indices(max_episode_steps: int):
    """Iterate forever when zero denotes an unbounded episode."""
    if max_episode_steps <= 0:
        return count()
    return range(max_episode_steps)


def vector_env_frame(env) -> np.ndarray:
    images = env.get_images()
    if not images or images[0] is None:
        raise RuntimeError("native vector provider did not return an RGB frame for lane 0")
    return np.asarray(images[0]).copy()


def playback_runtime_config(config):
    """Make record-producing task completion explicit for interactive playback."""
    semantics = target_for_game(config.game).eval_semantics
    if semantics.completion_reason != "level_change":
        return config
    termination = task_termination(config)
    success = list(dict.fromkeys((*termination.get("success", ()), "level_change")))
    return with_task_termination(config, success=success)


def _heatmap_color(heatmap: np.ndarray) -> np.ndarray:
    heatmap = np.clip(np.asarray(heatmap, dtype=np.float32), 0.0, 1.0)
    red = np.clip(1.6 * heatmap, 0.0, 1.0)
    green = np.clip(1.8 * heatmap - 0.25, 0.0, 1.0)
    blue = np.clip(1.0 - heatmap, 0.0, 1.0) * 0.35
    return np.stack([red, green, blue], axis=2) * 255.0


def render_obs_stack(
    frames: deque[np.ndarray],
    scale: int,
    heatmap: np.ndarray | None = None,
    heatmap_opacity: float = 0.45,
) -> np.ndarray:
    if scale < 1:
        raise ValueError("obs viewer scale must be >= 1")
    scaled_heatmap = None
    if heatmap is not None:
        scaled_heatmap = np.asarray(heatmap, dtype=np.float32)
        if scaled_heatmap.ndim != 2:
            raise ValueError(f"attribution heatmap must be 2D, got shape {scaled_heatmap.shape}")
        if scale != 1:
            scaled_heatmap = np.repeat(np.repeat(scaled_heatmap, scale, axis=0), scale, axis=1)
        scaled_heatmap = np.clip(scaled_heatmap, 0.0, 1.0)
        heat_color = _heatmap_color(scaled_heatmap)
        alpha = (float(heatmap_opacity) * scaled_heatmap)[..., None]
    panels = []
    for frame in frames:
        gray = frame[..., 0]
        panel = np.repeat(gray[..., None], 3, axis=2)
        if scale != 1:
            panel = np.repeat(np.repeat(panel, scale, axis=0), scale, axis=1)
        if scaled_heatmap is not None:
            if scaled_heatmap.shape != panel.shape[:2]:
                raise ValueError(
                    "attribution heatmap shape does not match observation frame: "
                    f"{scaled_heatmap.shape} vs {panel.shape[:2]}"
                )
            panel = ((1.0 - alpha) * panel.astype(np.float32) + alpha * heat_color).astype(np.uint8)
        panels.append(panel)
    return np.concatenate(panels, axis=1)


class PygameViewer:
    def __init__(
        self, frame_shape: tuple[int, int, int], scale: int, position: tuple[int, int] | None = None
    ):
        if scale < 1:
            raise ValueError("viewer scale must be >= 1")
        self.pygame = import_pygame()
        height, width, _channels = frame_shape
        self.size = (width * scale, height * scale)
        if position is not None:
            os.environ["SDL_VIDEO_WINDOW_POS"] = f"{position[0]},{position[1]}"
        with _suppress_native_stderr():
            self.pygame.init()
            self.pygame.display.set_caption("rlab")
            self.screen = self.pygame.display.set_mode(self.size)
            self.font = self.pygame.font.Font(None, max(16, 5 * scale))
        self.step_controls = StepOverControls()

    def show(self, frame: np.ndarray, overlay: list[str] | None = None) -> bool:
        for event in self.pygame.event.get():
            if event.type == self.pygame.QUIT:
                return False
            self.step_controls.handle_event(
                event,
                keydown_type=self.pygame.KEYDOWN,
                keyup_type=self.pygame.KEYUP,
                step_key=self.pygame.K_SPACE,
            )
        surface = self.pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        surface = self.pygame.transform.scale(surface, self.size)
        self.screen.blit(surface, (0, 0))
        if overlay:
            self.draw_overlay(overlay)
        self.pygame.display.flip()
        return True

    def draw_overlay(self, lines: list[str]) -> None:
        padding = 6
        line_height = self.font.get_height() + 2
        width = max(self.font.size(line)[0] for line in lines) + padding * 2
        height = line_height * len(lines) + padding * 2
        background = self.pygame.Surface((width, height), self.pygame.SRCALPHA)
        background.fill((0, 0, 0, 160))
        self.screen.blit(background, (0, 0))
        for idx, line in enumerate(lines):
            text = self.font.render(line, True, (255, 255, 255))
            self.screen.blit(text, (padding, padding + idx * line_height))

    def close(self) -> None:
        self.pygame.quit()


class StepOverControls:
    def __init__(self) -> None:
        self.step_key_pressed = False
        self.step_requested = False

    def handle_event(self, event, *, keydown_type: int, keyup_type: int, step_key: int) -> None:
        if event.type == keydown_type and getattr(event, "key", None) == step_key:
            self.step_key_pressed = True
            self.step_requested = True
        elif event.type == keyup_type and getattr(event, "key", None) == step_key:
            self.step_key_pressed = False

    def consume_step(self) -> bool:
        if self.step_key_pressed:
            self.step_requested = False
            return True
        if self.step_requested:
            self.step_requested = False
            return True
        return False


class ObsStackViewer:
    def __init__(self, scale: int, position: tuple[int, int] | None = None):
        self.scale = scale
        self.window_name = "rlab obs framestack"
        self.cv2 = None

        try:
            import cv2
        except ImportError:
            print("cv2 is not installed; --show-obs is disabled.", flush=True)
            return

        self.cv2 = cv2
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        if position is not None:
            cv2.moveWindow(self.window_name, position[0], position[1])

    def show(
        self,
        frames: deque[np.ndarray],
        heatmap: np.ndarray | None = None,
        heatmap_opacity: float = 0.45,
    ) -> bool:
        if self.cv2 is None:
            return True
        image = render_obs_stack(frames, self.scale, heatmap, heatmap_opacity)
        self.cv2.imshow(self.window_name, self.cv2.cvtColor(image, self.cv2.COLOR_RGB2BGR))
        key = self.cv2.waitKey(1) & 0xFF
        return key not in {27, ord("q")}

    def close(self) -> None:
        if self.cv2 is None:
            return
        self.cv2.destroyWindow(self.window_name)


def add_play_source_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "artifact_ref",
        nargs="?",
        type=positional_model_source_arg,
        help=(
            "Model source: W&B run name or URL, full W&B artifact ref, Hugging Face "
            "model ref, or use --model for a local checkpoint."
        ),
    )
    parser.add_argument(
        "--model",
        default="runs/smoke/final_model.zip",
        help="Local SB3 checkpoint path. The checkpoint must have a .metadata.json sidecar.",
    )
    parser.set_defaults(
        artifact=None,
        artifact_run=None,
        artifact_project=default_wandb_project_path(),
        artifact_kind="checkpoint",
        artifact_version="latest",
        artifact_root="runs/wandb_artifacts",
        hf_file=None,
        hf_revision=None,
        hf_model_root="runs/hf_models",
    )


def positive_int_arg(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def attribution_opacity_arg(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be in [0, 1]")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Show a PPO checkpoint playing a provider environment in a GUI window"
    )
    add_play_source_args(parser)
    parser.add_argument(
        "--episodes", type=int, default=0, help="Number of episodes; use 0 to run forever"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_EVAL_SEED,
        help=(
            "Base playback seed. The default lives in the eval/play-reserved seed "
            f"range >= {EVAL_SEED_START}; overrides must stay in that range."
        ),
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--fps", type=float, default=0.0)
    parser.add_argument(
        "--step-over",
        action="store_true",
        help=(
            "Pause before each policy step. Press Space to advance one step; "
            "hold Space to keep playing and release it to pause again."
        ),
    )
    parser.add_argument(
        "--show-obs",
        action="store_true",
        help="Open a second window showing the four preprocessed frames fed to the policy.",
    )
    parser.add_argument(
        "--attribution",
        choices=ATTRIBUTION_MODES,
        default="none",
        help=(
            "Overlay policy-input attribution in the observation window. "
            "Grad-CAM is fast; occlusion is slower but perturbation-based."
        ),
    )
    parser.add_argument(
        "--attribution-interval",
        type=positive_int_arg,
        default=None,
        help=(
            "Compute attribution every N policy steps. Defaults to 1 for Grad-CAM "
            "and 8 for occlusion."
        ),
    )
    parser.add_argument(
        "--attribution-opacity",
        type=attribution_opacity_arg,
        default=0.45,
        help="Heatmap opacity for --attribution overlays, in [0, 1].",
    )
    return parser


def display_replay_config(config):
    env_args = dict(config.env_args or {})
    if is_stable_retro_atari_env(config.env_provider, config.game):
        env_args.update(
            {
                "obs_resize": (210, 160),
                "obs_grayscale": False,
                "frame_stack": 1,
            }
        )
        return replace(
            config,
            env_args=env_args,
            obs_crop=(0, 0, 0, 0),
            hud_crop_top=0,
        )
    if config.env_provider == ALE_PY_PROVIDER.provider_id:
        env_args.update(
            {
                "img_height": 210,
                "img_width": 160,
                "grayscale": False,
                "stack_num": 1,
            }
        )
        return replace(
            config,
            env_args=env_args,
            obs_crop=(0, 0, 0, 0),
            hud_crop_top=0,
        )
    return config


def resolved_play_launch_lines(
    args: argparse.Namespace,
    *,
    argv: list[str],
    artifact_ref: str | None,
    policy_config,
    display_config,
) -> list[str]:
    states = policy_config.states or ((policy_config.state,) if policy_config.state else ())
    return [
        _color("▶ resolved play launch", "bold"),
        _summary_line("›", "argv", " ".join(argv) if argv else "-", "cyan"),
        _summary_line("◇", "artifact", artifact_ref or "-", "magenta"),
        _summary_line("▣", "model", args.model, "magenta"),
        _summary_line(
            "●",
            "policy/eval env",
            f"{policy_config.env_provider} game={policy_config.game} "
            f"state={policy_config.state or '-'} states={_format_sequence(states)}",
            "green",
        ),
        _summary_line(
            "○",
            "viewer env",
            f"{display_config.env_provider} game={display_config.game} "
            f"state={display_config.state or '-'} visual_only=True",
            "blue",
        ),
        _summary_line(
            "▶",
            "policy",
            f"device={args.device} stochastic=True "
            f"seed={args.seed} episodes={args.episodes} "
            f"max_steps={task_max_episode_steps(policy_config)} "
            f"step_over={getattr(args, 'step_over', False)}",
            "green",
        ),
        _summary_line(
            "◎",
            "attribution",
            f"mode={getattr(args, 'attribution', 'none')} "
            f"interval={getattr(args, 'attribution_interval', None) or '-'} "
            f"opacity={getattr(args, 'attribution_opacity', 0.45):.2f}",
            "magenta",
        ),
        _summary_line(
            "▤",
            "preprocessing",
            f"frame_skip={policy_config.frame_skip} max_pool={policy_config.max_pool_frames} "
            f"sticky={policy_config.sticky_action_prob} "
            f"obs={policy_config.observation_size} crop={_format_sequence(policy_config.obs_crop)} "
            f"crop_mode={policy_config.obs_crop_mode} crop_fill={policy_config.obs_crop_fill} "
            f"crop_top={policy_config.hud_crop_top} "
            f"resize={policy_config.obs_resize_algorithm}",
            "yellow",
        ),
        _summary_line(
            "⚙",
            "action/reward",
            f"action_set={task_action_set(policy_config)} "
            f"reward_mode={task_reward(policy_config).get('reward_mode')} "
            f"reward_scale={task_reward(policy_config).get('reward_scale')} "
            f"clip_rewards={task_reward(policy_config).get('clip_rewards')}",
            "yellow",
        ),
        _summary_line(
            "◆",
            "task/events",
            f"task_conditioning={bool(task_conditioning(policy_config).get('enabled'))} "
            f"task_info_vars={_format_sequence(task_info_vars(policy_config))} "
            f"termination_events={_format_sequence((*task_termination(policy_config).get('failure', ()), *task_termination(policy_config).get('success', ())))}",
            "cyan",
        ),
        _summary_line(
            "✓",
            "source of truth",
            "policy/eval env supplies model observations, rewards, dones, and info",
            "green",
        ),
    ]


def print_resolved_play_launch(
    args: argparse.Namespace,
    *,
    argv: list[str],
    artifact_ref: str | None,
    policy_config,
    display_config,
) -> None:
    lines = resolved_play_launch_lines(
        args,
        argv=argv,
        artifact_ref=artifact_ref,
        policy_config=policy_config,
        display_config=display_config,
    )
    print("\n".join(lines), flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    argv_list = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(argv_list)
    args.seed = validate_eval_seed(args.seed)
    if args.attribution_interval is None:
        args.attribution_interval = 8 if args.attribution == "occlusion" else 1
    ref = model_source_ref(args)
    if ref is not None:
        print(f"Downloading {ref}", flush=True)
    source = resolve_single_model_source(args)
    args.model = str(source.model_path)
    if ref is not None:
        print(f"Downloaded model: {args.model}", flush=True)
    artifact_config = load_playback_env_config(source.model_path)
    config = playback_runtime_config(artifact_config)
    display_config = display_replay_config(artifact_config)
    print_resolved_play_launch(
        args,
        argv=argv_list,
        artifact_ref=ref,
        policy_config=config,
        display_config=display_config,
    )
    assert_provider_runtime_available(config)
    from stable_baselines3 import PPO

    model = PPO.load(args.model, device=resolve_sb3_device(args.device))
    attributor = PolicyActionAttributor(model) if args.attribution != "none" else None
    policy_env = make_eval_vec_env(config=config, n_envs=1, seed=args.seed)
    display_env = make_eval_vec_env(config=display_config, n_envs=1, seed=args.seed)

    policy_env.seed(args.seed)
    policy_env.reset()
    display_env.seed(args.seed)
    display_env.reset()
    first_frame = vector_env_frame(display_env)
    obs_stack_position = (40, 240)
    viewer = PygameViewer(first_frame.shape, scale=DEFAULT_VIEWER_SCALE, position=None)
    obs_viewer = (
        ObsStackViewer(scale=DEFAULT_OBS_VIEWER_SCALE, position=obs_stack_position)
        if args.show_obs or attributor is not None
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

    def update_controls(
        frames: deque[np.ndarray] | None = None,
        heatmap: np.ndarray | None = None,
    ) -> bool:
        if obs_viewer is not None and frames is not None:
            return obs_viewer.show(
                frames,
                heatmap=heatmap,
                heatmap_opacity=args.attribution_opacity,
            )
        return True

    def step_over_overlay(overlay: list[str]) -> list[str]:
        if not args.step_over:
            return overlay
        return [*overlay, "step_over: SPACE"]

    def wait_for_step(frame: np.ndarray, overlay: list[str]) -> bool:
        if not args.step_over:
            return True
        overlay = step_over_overlay(overlay)
        while True:
            if not viewer.show(frame, overlay):
                return False
            if viewer.step_controls.consume_step():
                return True
            time.sleep(0.02)

    try:
        if not update_controls():
            return 0
        initial_overlay = step_over_overlay(
            ["r_step: 0.00", "r_total: 0.00", "max_x: 0", "step: 0"]
        )
        if not viewer.show(first_frame, initial_overlay):
            return 0
        throttle()
        episode_iter = count() if args.episodes <= 0 else range(args.episodes)
        for episode in episode_iter:
            episode_seed = args.seed + episode
            torch.manual_seed(episode_seed)
            policy_env.seed(episode_seed)
            policy_obs = policy_env.reset()
            policy_reset_info = dict(policy_env.reset_infos[0])
            display_env.seed(episode_seed)
            display_env.reset()
            frame = vector_env_frame(display_env)
            overlay = [
                "r_step: 0.00",
                "r_total: 0.00",
                "dx: 0 penalty: 0.00",
                "max_x: 0",
                f"step: 0 seed: {episode_seed}",
            ]
            if not viewer.show(frame, step_over_overlay(overlay)):
                break
            throttle()
            frames: deque[np.ndarray] = fast_env_frames(policy_obs)
            conditioning_enabled = bool(task_conditioning(config).get("enabled"))
            info_vars = task_info_vars(config)
            configured_task_states = task_state_names(config) if conditioning_enabled else ()
            active_task_state = (
                config.state or configured_task_states[0] if configured_task_states else None
            )
            active_info_value = (
                task_info_value_from_info(policy_reset_info, config)
                or info_value_from_state_name(
                    active_task_state or "",
                    info_vars,
                )
                if info_vars
                else None
            )
            if info_vars and active_info_value is not None:
                values = task_conditioning_info_values(config)
                print(
                    task_conditioning_start_message(
                        episode=episode + 1,
                        step=0,
                        task=active_info_value,
                        task_index=values.index(active_info_value),
                        task_count=len(values),
                    ),
                    flush=True,
                )
            elif (
                not info_vars
                and configured_task_states
                and active_task_state is not None
            ):
                reset_task_state = task_state_from_info(policy_reset_info, configured_task_states)
                if reset_task_state is not None:
                    active_task_state = reset_task_state
                print(
                    task_conditioning_start_message(
                        episode=episode + 1,
                        step=0,
                        task=active_task_state,
                        task_index=configured_task_states.index(active_task_state),
                        task_count=len(configured_task_states),
                    ),
                    flush=True,
                )
            if not update_controls(frames):
                return 0
            total_reward = 0.0
            max_x_pos = 0
            final_info = {}
            max_episode_steps = task_max_episode_steps(config)
            for step_idx in playback_step_indices(max_episode_steps):
                if not wait_for_step(frame, overlay):
                    return 0
                image_obs = fast_env_obs(policy_obs)
                model_obs = model_observation(
                    model,
                    image_obs,
                    config,
                    active_task_state=active_task_state,
                    active_info_value=active_info_value,
                )
                action, _ = model.predict(model_obs, deterministic=False)
                heatmap = None
                if attributor is not None and step_idx % args.attribution_interval == 0:
                    heatmap = attributor.attribute(args.attribution, model_obs, action)
                if attributor is not None and not update_controls(frames, heatmap):
                    return 0
                env_action = single_env_action(action)
                policy_obs, rewards, dones, infos = policy_env.step(np.asarray([env_action]))
                reward = float(np.asarray(rewards)[0])
                done = bool(np.asarray(dones)[0])
                truncated = bool(infos[0].get("TimeLimit.truncated", False))
                terminated = done and not truncated
                records = drain_runtime_records(policy_env)
                step_metrics = batch_metrics_for_lane(records, 0)
                info = {**dict(infos[0]), **step_metrics}
                completed_records = episode_records(records)
                episode_result = None
                if completed_records:
                    episode_result = episode_result_from_record(
                        completed_records[0],
                        semantics=target_for_game(config.game).eval_semantics,
                        terminal_info=info,
                    )
                    terminated = bool(episode_result["terminated"])
                    truncated = bool(episode_result["truncated"])
                if info_vars:
                    next_info_value = (
                        (int(info["level_hi"]), int(info["level_lo"]))
                        if "level_hi" in info and "level_lo" in info
                        else task_info_value_from_info(info, config)
                    )
                    if next_info_value is not None and next_info_value != active_info_value:
                        values = task_conditioning_info_values(config)
                        print(
                            task_conditioning_change_message(
                                episode=episode + 1,
                                step=step_idx + 1,
                                old_task=active_info_value,
                                new_task=next_info_value,
                                task_index=values.index(next_info_value),
                                task_count=len(values),
                            ),
                            flush=True,
                        )
                        active_info_value = next_info_value
                else:
                    if "level_hi" in info and "level_lo" in info:
                        info["level_id"] = f"{int(info['level_hi'])}-{int(info['level_lo'])}"
                    next_task_state = task_state_from_info(info, configured_task_states)
                    if next_task_state is not None and next_task_state != active_task_state:
                        print(
                            task_conditioning_change_message(
                                episode=episode + 1,
                                step=step_idx + 1,
                                old_task=active_task_state,
                                new_task=next_task_state,
                                task_index=configured_task_states.index(next_task_state),
                                task_count=len(configured_task_states),
                            ),
                            flush=True,
                        )
                        active_task_state = next_task_state
                display_env.step(np.asarray([env_action]))
                drain_runtime_records(display_env)
                frames = fast_env_frames(policy_obs)
                if attributor is None and not update_controls(frames):
                    return 0
                total_reward += float(reward)
                max_x_pos = max(max_x_pos, int(info.get("max_x_pos", 0)))
                final_info = dict(info)
                if episode_result is not None:
                    total_reward = float(episode_result["reward"])
                    max_x_pos = max(max_x_pos, int(episode_result.get("max_x_pos", 0)))
                    final_info = dict(episode_result.get("final_info", {}))
                    completed = bool(episode_result.get("level_complete", False))
                else:
                    completed = is_level_complete(final_info)
                frame = vector_env_frame(display_env)
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
                if not viewer.show(frame, step_over_overlay(overlay)):
                    return 0
                throttle()
                if playback_should_end_episode(terminated, truncated, completed):
                    status = (
                        "complete" if completed else "terminated" if terminated else "truncated"
                    )
                    print(
                        "episode="
                        f"{episode + 1} seed={episode_seed} reward={total_reward:.2f} "
                        f"max_x={max_x_pos} steps={step_idx + 1} status={status} "
                        f"died={bool(final_info.get('died', False))} "
                        f"complete={completed}",
                        flush=True,
                    )
                    time.sleep(0.5)
                    break
            else:
                print(
                    "episode="
                    f"{episode + 1} seed={episode_seed} reward={total_reward:.2f} "
                    f"max_x={max_x_pos} steps={max_episode_steps} status=max_steps "
                    f"died={bool(final_info.get('died', False))} "
                    f"complete={bool(final_info.get('level_complete', False))}",
                    flush=True,
                )
    finally:
        if obs_viewer is not None:
            obs_viewer.close()
        viewer.close()
        try:
            display_env.close()
            if policy_env is not display_env:
                policy_env.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
