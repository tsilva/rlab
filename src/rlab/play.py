from __future__ import annotations

# ruff: noqa: E402

import argparse
import contextlib
import os
import sys
import time
from collections import deque
from dataclasses import replace
from itertools import count
from types import ModuleType
from typing import TYPE_CHECKING

os.environ.setdefault("MPLCONFIGDIR", os.path.abspath(".matplotlib"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import torch

if TYPE_CHECKING:
    from stable_baselines3 import PPO

from rlab.artifacts import explicit_arg_dests
from rlab.cli_args import add_env_config_args
from rlab.device import resolve_sb3_device
from rlab.env import (
    assert_rom_imported,
    info_value_from_state_name,
    make_eval_vec_env,
    make_rendered_replay_env,
    make_visual_replay_env,
    native_vec_env_supports_rgb_render,
    resolve_env_config,
    state_name_candidates_from_level_id,
    task_conditioning_info_values,
)
from rlab.env_config import env_config_from_args
from rlab.env_registry import STABLE_RETRO_TURBO_PROVIDER
from rlab.eval_metrics import single_env_action
from rlab.eval_metrics import is_level_complete
from rlab.model_sources import (
    add_model_source_args,
    apply_model_source_defaults,
    model_source_ref,
    resolve_single_model_source,
)
from rlab.seeds import DEFAULT_EVAL_SEED, EVAL_SEED_START, validate_eval_seed


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
    except (AttributeError, OSError, ValueError):
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


def stacked_obs(frames: deque[np.ndarray]) -> np.ndarray:
    # Rendered replay stacks grayscale frames into the channel-first model layout.
    return np.stack([frame[..., 0] for frame in frames], axis=0)[None, ...]


def fast_env_obs(obs: np.ndarray) -> np.ndarray:
    # Older native envs may expose HWC stacks; post12 exposes CHW stacks.
    # The policy always receives SB3's channel-first batch layout.
    arr = np.asarray(obs)
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
    arr = np.asarray(obs)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[-1] == 4:
        return deque([arr[..., idx : idx + 1] for idx in range(arr.shape[-1])], maxlen=4)
    if arr.ndim == 3 and arr.shape[0] == 4:
        return deque([arr[idx, ..., None] for idx in range(arr.shape[0])], maxlen=4)
    raise ValueError(f"expected fast env obs with 4 stacked frames, got shape {arr.shape}")


def task_state_names(config) -> tuple[str, ...]:
    if config.task_conditioning_info_vars:
        return tuple(
            ",".join(str(part) for part in value)
            for value in task_conditioning_info_values(config)
        )
    states = tuple(dict.fromkeys(config.states or ((config.state,) if config.state else ())))
    if not states:
        raise ValueError("task-conditioned playback requires at least one configured state")
    return states


def task_vector_for_state(
    config,
    task_dim: int,
    active_state: str | None = None,
    active_info_value: tuple[int | str, ...] | None = None,
) -> np.ndarray:
    if config.task_conditioning_info_vars:
        values = task_conditioning_info_values(config)
        if not values:
            raise ValueError("info-var task-conditioned playback has no configured task values")
        if task_dim != len(values):
            raise ValueError(
                "task-conditioned model expects "
                f"{task_dim} task values, but playback metadata has {len(values)} "
                f"info-value rows: {values}",
            )
        active_info_value = active_info_value or info_value_from_state_name(
            active_state or config.state,
            config.task_conditioning_info_vars,
        )
        if active_info_value not in values:
            raise ValueError(
                f"playback info value {active_info_value!r} is not in "
                f"task-conditioned info values {values!r}",
            )
        task = np.zeros((1, task_dim), dtype=np.float32)
        task[0, values.index(active_info_value)] = 1.0
        return task

    states = task_state_names(config)
    if task_dim != len(states):
        raise ValueError(
            "task-conditioned model expects "
            f"{task_dim} task values, but playback metadata has {len(states)} state(s): {states}",
        )
    active_state = active_state or config.state or states[0]
    if active_state not in states:
        raise ValueError(
            f"playback state {active_state!r} is not in task-conditioned state list {states!r}",
        )
    task = np.zeros((1, task_dim), dtype=np.float32)
    task[0, states.index(active_state)] = 1.0
    return task


def model_observation(
    model: PPO,
    image_obs: np.ndarray,
    config,
    *,
    active_task_state: str | None = None,
    active_info_value: tuple[int | str, ...] | None = None,
) -> np.ndarray | dict[str, np.ndarray]:
    observation_space = model.observation_space
    spaces = getattr(observation_space, "spaces", None)
    if not isinstance(spaces, dict):
        return image_obs
    if "image" not in spaces or "task" not in spaces:
        raise ValueError(
            "dict-observation model must have 'image' and 'task' observation keys, "
            f"got {tuple(spaces)}",
        )
    task_shape = getattr(spaces["task"], "shape", None)
    if task_shape is None or len(task_shape) != 1:
        raise ValueError(f"expected one-dimensional task observation, got {spaces['task']!r}")
    return {
        "image": image_obs,
        "task": task_vector_for_state(
            config,
            task_dim=int(task_shape[0]),
            active_state=active_task_state,
            active_info_value=active_info_value,
        ),
    }


def task_info_value_from_info(info: dict, config) -> tuple[int | str, ...] | None:
    if not config.task_conditioning_info_vars:
        return None
    try:
        value = tuple(info[var] for var in config.task_conditioning_info_vars)
    except KeyError:
        return None
    return value if value in task_conditioning_info_values(config) else None


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


def playback_env_config(config):
    return replace(config, done_on_events=())


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

    def show(self, frame: np.ndarray, overlay: list[str] | None = None) -> bool:
        for event in self.pygame.event.get():
            if event.type == self.pygame.QUIT:
                return False
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


class OptionsPanel:
    def __init__(self, fps: float, show_obs_stack: bool, position: tuple[int, int] | None = None):
        self.window_name = "rlab controls"
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
        self.window_name = "rlab obs framestack"
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
        description="Show a PPO checkpoint playing a Stable Retro game in a GUI window"
    )
    add_model_source_args(
        parser,
        positional_artifact=True,
        model_default="runs/smoke/final_model.zip",
        default_kind="checkpoint",
    )
    parser.add_argument("--download-only", action="store_true")
    add_env_config_args(parser, max_steps_default=1200)
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
    parser.add_argument(
        "--policy-env",
        choices=["fast", "rendered"],
        default="fast",
        help=(
            "Observation path used for the model. 'fast' matches native-vector training "
            "preprocessing; 'rendered' uses the older manual GUI frame stack."
        ),
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic argmax actions instead of stochastic policy sampling.",
    )
    return parser


def display_replay_config(config):
    if config.env_provider == STABLE_RETRO_TURBO_PROVIDER.provider_id:
        return config
    if native_vec_env_supports_rgb_render(config):
        return config
    return replace(config, env_provider=STABLE_RETRO_TURBO_PROVIDER.provider_id, env_threads=0)


def resolved_play_launch_lines(
    args: argparse.Namespace,
    *,
    argv: list[str],
    artifact_ref: str | None,
    policy_config,
    display_config,
) -> list[str]:
    wrapper_ids = [
        str(spec.get("id", ""))
        for spec in policy_config.env_wrappers
        if isinstance(spec, dict) and spec.get("id")
    ]
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
            f"state={policy_config.state or '-'} states={_format_sequence(states)} "
            f"threads={policy_config.env_threads}",
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
            f"policy_env={args.policy_env} device={args.device} "
            f"stochastic={not args.deterministic} seed={args.seed} "
            f"episodes={args.episodes} max_steps={args.max_steps}",
            "green",
        ),
        _summary_line(
            "▤",
            "preprocessing",
            f"frame_skip={policy_config.frame_skip} max_pool={policy_config.max_pool_frames} "
            f"sticky={policy_config.sticky_action_prob} "
            f"obs={policy_config.observation_size} crop={_format_sequence(policy_config.obs_crop)} "
            f"crop_top={policy_config.hud_crop_top} "
            f"resize={policy_config.obs_resize_algorithm}",
            "yellow",
        ),
        _summary_line(
            "⚙",
            "action/reward",
            f"action_set={policy_config.action_set} reward_mode={policy_config.reward_mode} "
            f"reward_scale={policy_config.reward_scale} clip_rewards={policy_config.clip_rewards}",
            "yellow",
        ),
        _summary_line(
            "◆",
            "task/events",
            f"task_conditioning={policy_config.task_conditioning} "
            f"task_info_vars={_format_sequence(policy_config.task_conditioning_info_vars)} "
            f"done_on={_format_sequence(policy_config.done_on_events)}",
            "cyan",
        ),
        _summary_line("✚", "wrappers", _format_sequence(wrapper_ids), "blue"),
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


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    argv_list = list(sys.argv[1:] if argv is None else argv)
    parser_defaults = vars(parser.parse_args([]))
    explicit_dests = explicit_arg_dests(parser, argv_list)
    args = parser.parse_args(argv_list)
    args.seed = validate_eval_seed(args.seed)
    ref = model_source_ref(args)
    if ref is not None:
        print(f"Downloading {ref}", flush=True)
    source = resolve_single_model_source(args)
    args.model = str(source.model_path)
    if ref is not None:
        print(f"Downloaded model: {args.model}", flush=True)
    apply_model_source_defaults(
        args,
        source,
        parser,
        parser_defaults,
        explicit_dests,
        infer_artifact_config=True,
        print_loaded_metadata=True,
    )
    config = playback_env_config(
        resolve_env_config(
            env_config_from_args(
                args,
                max_episode_steps_attr="max_steps",
                include_states=True,
                include_env_threads=True,
            )
        )
    )
    effective_policy_env = args.policy_env
    display_config = display_replay_config(config)
    print_resolved_play_launch(
        args,
        argv=argv_list,
        artifact_ref=ref,
        policy_config=config,
        display_config=display_config,
    )
    if args.download_only:
        if ref is None:
            raise SystemExit(
                "--download-only requires an hf:// model ref, positional artifact ref, "
                "--artifact, or --artifact-run"
            )
        return
    assert_rom_imported(args.game)
    from stable_baselines3 import PPO

    model = PPO.load(args.model, device=resolve_sb3_device(args.device))
    if (
        effective_policy_env == "rendered"
        and config.env_provider != STABLE_RETRO_TURBO_PROVIDER.provider_id
    ):
        raise SystemExit(
            "--policy-env rendered only supports stable-retro-turbo; use --policy-env fast "
            f"to replay policies trained with {config.env_provider}",
        )
    if args.policy_env == "fast":
        display_env = make_visual_replay_env(config=display_config, seed=args.seed)
        policy_env = make_eval_vec_env(config=config, n_envs=1, seed=args.seed)
    else:
        display_env = make_rendered_replay_env(config=display_config, seed=args.seed)
        policy_env = display_env
    seed_rng = np.random.default_rng() if args.random_seeds else None

    if effective_policy_env == "fast":
        policy_env.seed(args.seed)
        policy_env.reset()
    display_env.reset(seed=args.seed)
    first_frame = display_env.render()
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
                int(seed_rng.integers(EVAL_SEED_START, np.iinfo(np.int32).max))
                if seed_rng is not None
                else args.seed + episode
            )
            torch.manual_seed(episode_seed)
            if effective_policy_env == "fast":
                policy_env.seed(episode_seed)
                policy_obs = policy_env.reset()
                policy_reset_info = {}
                display_obs, _ = display_env.reset(seed=episode_seed)
            else:
                policy_obs, policy_reset_info = policy_env.reset(seed=episode_seed)
                display_obs = policy_obs
            frame = display_env.render()
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
            frames: deque[np.ndarray] = (
                fast_env_frames(policy_obs)
                if effective_policy_env == "fast"
                else deque([display_obs] * 4, maxlen=4)
            )
            configured_task_states = task_state_names(config) if config.task_conditioning else ()
            active_task_state = (
                config.state or configured_task_states[0]
                if configured_task_states
                else None
            )
            active_info_value = (
                task_info_value_from_info(policy_reset_info, config)
                or info_value_from_state_name(
                    active_task_state or "",
                    config.task_conditioning_info_vars,
                )
                if config.task_conditioning_info_vars
                else None
            )
            if config.task_conditioning_info_vars and active_info_value is not None:
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
                not config.task_conditioning_info_vars
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
                return
            total_reward = 0.0
            max_x_pos = 0
            final_info = {}
            for step_idx in range(args.max_steps):
                image_obs = (
                    fast_env_obs(policy_obs)
                    if effective_policy_env == "fast"
                    else stacked_obs(frames)
                )
                model_obs = model_observation(
                    model,
                    image_obs,
                    config,
                    active_task_state=active_task_state,
                    active_info_value=active_info_value,
                )
                action, _ = model.predict(model_obs, deterministic=args.deterministic)
                env_action = single_env_action(action)
                if effective_policy_env == "fast":
                    policy_obs, rewards, dones, infos = policy_env.step(np.asarray([env_action]))
                    reward = float(np.asarray(rewards)[0])
                    done = bool(np.asarray(dones)[0])
                    terminated = done
                    truncated = bool(infos[0].get("TimeLimit.truncated", False))
                    info = dict(infos[0])
                else:
                    policy_obs, reward, terminated, truncated, info = policy_env.step(env_action)
                if config.task_conditioning_info_vars:
                    next_info_value = task_info_value_from_info(info, config)
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
                if effective_policy_env == "rendered":
                    display_obs = policy_obs
                    frames.append(display_obs)
                else:
                    display_env.step(env_action)
                    frames = fast_env_frames(policy_obs)
                if not update_controls(frames):
                    return
                total_reward += float(reward)
                max_x_pos = max(max_x_pos, int(info.get("max_x_pos", 0)))
                final_info = dict(info)
                completed = is_level_complete(final_info)
                frame = display_env.render()
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
                if playback_should_end_episode(terminated, truncated, completed):
                    status = "complete" if completed else "terminated" if terminated else "truncated"
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
            display_env.close()
            if policy_env is not display_env:
                policy_env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
