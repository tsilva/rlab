from __future__ import annotations

# ruff: noqa: E402

import argparse
import contextlib
import os
import selectors
import signal
import sys
import time
from collections import deque
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from itertools import count
from types import ModuleType

os.environ.setdefault("MPLCONFIGDIR", os.path.abspath(".matplotlib"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import torch
from tqdm import tqdm

from rlab.artifacts import load_playback_env_config, playback_env_config
from rlab.batch_runtime import StepDiagnostics
from rlab.cli_args import explicit_arg_dests
from rlab.device import resolve_sb3_device
from rlab.env import (
    assert_provider_runtime_available,
    info_value_from_state_name,
    make_eval_vec_env,
    state_name_candidates_from_level_id,
    task_action_set,
    task_conditioning,
    task_max_episode_steps,
    task_reward,
    task_termination,
    resolve_env_config,
)
from rlab.env_metadata import env_config_from_config_dict
from rlab.eval_metrics import (
    batch_metrics_for_lane,
    drain_runtime_records,
    episode_records,
    episode_result_from_record,
    is_level_complete,
)
from rlab.model_sources import (
    model_source_ref,
    positional_model_source_arg,
    resolve_single_model_source,
)
from rlab.play_attribution import PolicyActionAttributor
from rlab.policy_bundle import evaluation_contract
from rlab.play_debug import (
    DebugCommandError,
    PolicyDecision,
    action_display_name,
    ansi,
    debug_prompt,
    debug_help,
    field,
    format_action,
    format_model_input,
    format_policy_detail,
    format_raw,
    model_input_lines,
    policy_summary_lines,
    reward_text,
    section,
    status_message,
    terminal_panel,
    inspect_policy,
    parse_debug_command,
    sample_policy_decision,
)
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

    def show(self, frame: np.ndarray, overlay: list[str] | None = None) -> bool:
        for event in self.pygame.event.get():
            if event.type in {self.pygame.QUIT, self.pygame.WINDOWCLOSE}:
                return False
            if event.type == self.pygame.KEYDOWN and getattr(event, "key", None) in {
                self.pygame.K_ESCAPE,
                self.pygame.K_q,
            }:
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


class ObsStackViewer:
    def __init__(self, scale: int, position: tuple[int, int] | None = None):
        self.scale = scale
        self.position = position
        self.pygame = import_pygame()
        self.window = None
        self.surface = None

    def _ensure_window(self, image: np.ndarray) -> None:
        height, width = image.shape[:2]
        if self.window is not None and self.window.size == (width, height):
            return
        if self.window is not None:
            self.window.destroy()
        kwargs = {"size": (width, height)}
        if self.position is not None:
            kwargs["position"] = self.position
        self.window = self.pygame.Window("rlab obs framestack", **kwargs)
        self.surface = self.window.get_surface()

    def show(
        self,
        frames: deque[np.ndarray],
        heatmap: np.ndarray | None = None,
        heatmap_opacity: float = 0.45,
    ) -> bool:
        image = render_obs_stack(frames, self.scale, heatmap, heatmap_opacity)
        self._ensure_window(image)
        frame_surface = self.pygame.surfarray.make_surface(np.transpose(image, (1, 0, 2)))
        self.surface.blit(frame_surface, (0, 0))
        self.window.flip()
        return True

    def close(self) -> None:
        if self.window is not None:
            self.window.destroy()
            self.window = None
            self.surface = None


def add_play_source_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "artifact_ref",
        nargs="?",
        type=positional_model_source_arg,
        help=(
            "Model source: W&B run name or URL, full W&B artifact ref, Hugging Face "
            "model ref, or use --model for a local checkpoint. A W&B run resolves "
            "the confirmed promotion, then its newest confirmed checkpoint/final model."
        ),
    )
    parser.add_argument(
        "--model",
        default="runs/smoke/final_model.zip",
        help="Local rlab policy path. The artifact must have a .metadata.json sidecar.",
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
        prog="rlab play",
        description="Show an rlab policy artifact playing a provider environment in a GUI window",
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
    parser.add_argument(
        "--env-provider",
        help=(
            "Run the artifact's unchanged evaluation contract through an equivalent provider. "
            "The provider must support the recorded game and constructor arguments."
        ),
    )
    parser.add_argument("--fps", type=float, default=0.0)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Open an interactive policy debugger; press Enter to advance one step.",
    )
    parser.add_argument(
        "--continuous-play",
        action="store_true",
        help=(
            "Ignore the recipe's task success, failure, stall, and step-limit boundaries. "
            "This is a semantic deviation intended only for continuous interactive play."
        ),
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable model-download and player-startup progress bars.",
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
    """Return the policy config because playback has exactly one environment.

    Native providers expose their best available frame through ``get_images``.
    Creating a second RGB-configured environment can silently diverge from the
    policy state when the provider has stochastic transitions.
    """
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
            "viewer source",
            f"{display_config.env_provider} game={display_config.game} "
            f"state={display_config.state or '-'} shared_with_policy=True",
            "blue",
        ),
        _summary_line(
            "▶",
            "policy",
            f"device={args.device} stochastic=True "
            f"seed={args.seed} episodes={args.episodes} "
            f"max_steps={task_max_episode_steps(policy_config)} "
            f"debug={getattr(args, 'debug', False)} "
            f"respect_task_termination={getattr(args, 'respect_task_termination', False)}",
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
            "one policy/eval env supplies both viewers, observations, rewards, dones, and info",
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


@contextlib.contextmanager
def startup_progress(name: str, *, disabled: bool = False):
    """Show one independently timed progress bar for a startup operation."""

    with tqdm(
        total=1,
        desc=name,
        unit="operation",
        dynamic_ncols=True,
        disable=disabled,
    ) as progress:
        yield
        progress.update(1)


def optional_vector_env_frame(env) -> np.ndarray | None:
    try:
        return vector_env_frame(env)
    except AttributeError, NotImplementedError, RuntimeError, TypeError, ValueError:
        return None


def optional_fast_env_frames(obs) -> deque[np.ndarray] | None:
    try:
        return fast_env_frames(obs)
    except KeyError, TypeError, ValueError:
        return None


def playback_model_observation(
    model,
    policy_obs,
    config,
    *,
    active_task_state: str | None,
    active_info_value: tuple[int | str, ...] | None,
):
    spaces = getattr(getattr(model, "observation_space", None), "spaces", None)
    if isinstance(spaces, dict) and {"image", "task"}.issubset(spaces):
        return model_observation(
            model,
            fast_env_obs(policy_obs),
            config,
            active_task_state=active_task_state,
            active_info_value=active_info_value,
        )
    if isinstance(policy_obs, Mapping):
        return policy_obs
    try:
        return fast_env_obs(policy_obs)
    except ValueError:
        return np.asarray(policy_obs)


def _observation_shape(value) -> str:
    if isinstance(value, Mapping):
        return (
            "{"
            + ", ".join(f"{name}:{np.asarray(item).shape}" for name, item in value.items())
            + "}"
        )
    return str(np.asarray(value).shape)


@dataclass(frozen=True)
class _PlaybackTransition:
    episode: int
    step: int
    seed: int | None
    start_id: str | None
    model_obs: object
    decision: PolicyDecision
    diagnostics: StepDiagnostics | None
    info: dict[str, object]
    pre_task: object
    next_task: object
    reward: float
    total_reward: float
    max_x_pos: int
    terminated: bool
    truncated: bool
    completed: bool
    boundary: bool

    @property
    def events(self) -> tuple[str, ...]:
        return self.diagnostics.events if self.diagnostics is not None else ()


class _PlaybackSession:
    """The sole mutable owner of one playback trajectory."""

    def __init__(
        self,
        *,
        model,
        env,
        config,
        initial_seed: int,
        attributor: PolicyActionAttributor | None,
        attribution_mode: str,
        attribution_interval: int,
        attribution_opacity: float,
    ):
        self.model = model
        self.env = env
        self.config = config
        self.initial_seed = initial_seed
        self.attributor = attributor
        self.attribution_mode = attribution_mode
        self.attribution_interval = attribution_interval
        self.attribution_opacity = attribution_opacity
        self.viewer: PygameViewer | None = None
        self.obs_viewer: ObsStackViewer | None = None
        self.info_vars = task_info_vars(config)
        self.conditioning_enabled = bool(task_conditioning(config).get("enabled"))
        self.configured_task_states = task_state_names(config) if self.conditioning_enabled else ()
        try:
            self.action_names = target_for_game(config.game).action_names_for_set(
                task_action_set(config)
            )
        except ValueError:
            self.action_names = ()
        self.policy_obs = None
        self.current_frame: np.ndarray | None = None
        self.frames: deque[np.ndarray] | None = None
        self.active_task_state: str | None = None
        self.active_info_value: tuple[int | str, ...] | None = None
        self.active_seed: int | None = initial_seed
        self.episode = 1
        self.step_index = 0
        self.total_reward = 0.0
        self.max_x_pos = 0
        self.last_transition: _PlaybackTransition | None = None

    @property
    def active_task(self):
        return (
            self.active_info_value if self.active_info_value is not None else self.active_task_state
        )

    @property
    def model_obs(self):
        return playback_model_observation(
            self.model,
            self.policy_obs,
            self.config,
            active_task_state=self.active_task_state,
            active_info_value=self.active_info_value,
        )

    def _set_initial_conditioning(self, reset_info: Mapping[str, object]) -> None:
        self.active_task_state = (
            (self.config.state or self.configured_task_states[0])
            if self.configured_task_states
            else None
        )
        self.active_info_value = None
        if self.info_vars:
            self.active_info_value = task_info_value_from_info(reset_info, self.config)
            if self.active_info_value is None:
                self.active_info_value = info_value_from_state_name(
                    self.active_task_state or "",
                    self.info_vars,
                )
        elif self.configured_task_states:
            self._update_conditioning(reset_info)

    def _update_conditioning(self, info: Mapping[str, object]) -> None:
        if self.info_vars:
            next_value = None
            if "level_hi" in info and "level_lo" in info:
                next_value = (int(info["level_hi"]), int(info["level_lo"]))
            if next_value is None:
                next_value = task_info_value_from_info(info, self.config)
            if next_value is not None:
                self.active_info_value = next_value
            return
        if not self.configured_task_states:
            return
        candidate = info.get("start_id") or info.get("start_state") or info.get("state")
        if isinstance(candidate, str) and candidate in self.configured_task_states:
            self.active_task_state = candidate
            return
        mutable_info = dict(info)
        if "level_hi" in mutable_info and "level_lo" in mutable_info:
            mutable_info["level_id"] = (
                f"{int(mutable_info['level_hi'])}-{int(mutable_info['level_lo'])}"
            )
        next_state = task_state_from_info(mutable_info, self.configured_task_states)
        if next_state is not None:
            self.active_task_state = next_state

    def restart(self, seed: int | None = None) -> None:
        seed = self.initial_seed if seed is None else seed
        torch.manual_seed(seed)
        if bool(getattr(self.model, "use_sde", False)):
            self.model.policy.reset_noise()
        self.env.seed(seed)
        self.policy_obs = self.env.reset()
        reset_episode = getattr(self.model, "reset_episode", None)
        if callable(reset_episode):
            reset_episode()
        reset_info = dict(self.env.reset_infos[0])
        self._set_initial_conditioning(reset_info)
        self.active_seed = seed
        self.episode = 1
        self.step_index = 0
        self.total_reward = 0.0
        self.max_x_pos = 0
        self.last_transition = None
        self.current_frame = optional_vector_env_frame(self.env)
        self.frames = optional_fast_env_frames(self.policy_obs)

    def inspect_policy(self) -> PolicyDecision:
        return inspect_policy(self.model, self.model_obs)

    def step(self) -> _PlaybackTransition:
        model_obs = self.model_obs
        model_obs_snapshot = deepcopy(model_obs)
        pre_task = deepcopy(self.active_task)
        decision = sample_policy_decision(self.model, model_obs)
        if (
            self.attributor is not None
            and self.frames is not None
            and self.step_index % self.attribution_interval == 0
        ):
            heatmap = self.attributor.attribute(
                self.attribution_mode,
                model_obs,
                decision.raw_action,
            )
            if self.obs_viewer is not None:
                self.obs_viewer.show(
                    self.frames,
                    heatmap=heatmap,
                    heatmap_opacity=self.attribution_opacity,
                )

        batched_action = np.expand_dims(np.asarray(decision.executed_action), axis=0)
        policy_obs, rewards, dones, infos = self.env.step(batched_action)
        diagnostics = self.env.take_step_diagnostics()
        records = drain_runtime_records(self.env)
        step_metrics = batch_metrics_for_lane(records, 0)
        info: dict[str, object] = {}
        if diagnostics is not None:
            info.update(diagnostics.provider_info)
            info.update(diagnostics.task_metrics)
        info.update(dict(infos[0]))
        info.update(step_metrics)

        reward = float(np.asarray(rewards)[0])
        done = bool(np.asarray(dones)[0])
        truncated = bool(info.get("TimeLimit.truncated", False))
        terminated = done and not truncated
        completed_records = episode_records(records)
        episode_result = None
        if completed_records:
            episode_result = episode_result_from_record(
                completed_records[0],
                semantics=target_for_game(self.config.game).eval_semantics,
                terminal_info=info,
            )
            terminated = bool(episode_result["terminated"])
            truncated = bool(episode_result["truncated"])

        self.total_reward += reward
        self.max_x_pos = max(self.max_x_pos, int(info.get("max_x_pos", 0)))
        final_info = info
        if episode_result is not None:
            self.total_reward = float(episode_result["return"])
            self.max_x_pos = max(
                self.max_x_pos,
                int(episode_result.get("max_x_pos", 0)),
            )
            final_info = dict(episode_result.get("final_info", {}))
            completed = bool(episode_result.get("level_complete", False))
        else:
            completed = is_level_complete(final_info)
        boundary = playback_should_end_episode(terminated, truncated, completed)

        next_conditioning_info = dict(info.get("reset_info", {})) if boundary else info
        self._update_conditioning(next_conditioning_info)
        next_task = deepcopy(self.active_task)
        transition = _PlaybackTransition(
            episode=self.episode,
            step=self.step_index + 1,
            seed=self.active_seed,
            start_id=(diagnostics.start_id if diagnostics is not None else None),
            model_obs=model_obs_snapshot,
            decision=decision,
            diagnostics=diagnostics,
            info=dict(info),
            pre_task=pre_task,
            next_task=next_task,
            reward=reward,
            total_reward=self.total_reward,
            max_x_pos=self.max_x_pos,
            terminated=terminated,
            truncated=truncated,
            completed=completed,
            boundary=boundary,
        )
        self.last_transition = transition
        self.policy_obs = policy_obs
        self.current_frame = optional_vector_env_frame(self.env)
        self.frames = optional_fast_env_frames(policy_obs)
        if boundary:
            reset_lanes = getattr(self.model, "reset_lanes", None)
            if callable(reset_lanes):
                reset_lanes([True])
            self.episode += 1
            self.step_index = 0
            self.total_reward = 0.0
            self.max_x_pos = 0
            if diagnostics is not None:
                self.active_seed = diagnostics.next_episode_seed
        else:
            self.step_index += 1
        return transition

    def render(self) -> bool:
        if self.viewer is not None and self.current_frame is not None:
            transition = self.last_transition
            overlay = (
                ["r_step: 0.00", "r_total: 0.00", f"step: 0 seed: {self.active_seed}"]
                if transition is None
                else [
                    f"r_step: {transition.reward:.2f}",
                    f"r_total: {transition.total_reward:.2f}",
                    f"max_x: {transition.max_x_pos}",
                    f"step: {transition.step} seed: {transition.seed}",
                ]
            )
            if not self.viewer.show(self.current_frame, overlay):
                return False
        if self.obs_viewer is not None and self.frames is not None:
            return self.obs_viewer.show(self.frames)
        return True


def _transition_debug_text(
    transition: _PlaybackTransition,
    action_names: tuple[str, ...],
) -> str:
    diagnostics = transition.diagnostics
    if diagnostics is None:
        raise RuntimeError("debug playback step did not produce runtime diagnostics")
    components = {
        name: value
        for name, value in diagnostics.task_metrics.items()
        if name.endswith("_component") and np.any(np.asarray(value) != 0)
    }
    deltas = {
        name: value
        for name, value in diagnostics.task_metrics.items()
        if name.endswith("_delta") and np.any(np.asarray(value) != 0)
    }
    selected = transition.decision.selected_discrete_action
    controller = (
        action_display_name(selected, action_names)
        if selected is not None
        else format_action(diagnostics.native_action)
    )
    event_labels = []
    for event in diagnostics.events:
        lowered = event.lower()
        icon = (
            "💀"
            if "life_loss" in lowered or "death" in lowered or "fail" in lowered
            else "🏁"
            if "complete" in lowered or "success" in lowered
            else "◆"
        )
        style = "red" if icon == "💀" else "green" if icon == "🏁" else "yellow"
        event_labels.append(f"{icon} {ansi(event.replace('_', ' ').upper(), style)}")
    boundary_parts = []
    if diagnostics.provider_terminated:
        boundary_parts.append("provider terminated")
    if diagnostics.provider_truncated:
        boundary_parts.append("provider truncated")
    if diagnostics.task_terminated:
        boundary_parts.append("task terminated")
    if diagnostics.task_truncated:
        boundary_parts.append("task truncated")
    outcome = diagnostics.outcome.name.lower()
    outcome_style = "green" if outcome == "success" else "red" if outcome == "failure" else "dim"

    lines = [
        section("👁", "INPUT", style="blue"),
        field("trajectory", f"episode {transition.episode}  ·  policy step {transition.step}"),
        field("scenario", f"seed {transition.seed}  ·  start {transition.start_id or 'default'}"),
        field(
            "conditioning",
            repr(transition.pre_task) if transition.pre_task is not None else ansi("none", "dim"),
        ),
        field("observation", _observation_shape(transition.model_obs)),
        "",
        section("🎲", "POLICY", style="magenta"),
        *policy_summary_lines(transition.decision, action_names),
        "",
        section("⚙", "TRANSITION", style="yellow"),
        field("controller", ansi(controller, "bold")),
        field(
            "reward",
            f"provider {reward_text(diagnostics.provider_reward)}  →  "
            f"training {reward_text(diagnostics.reward)}",
        ),
    ]
    for name, value in components.items():
        component_value = float(np.asarray(value).reshape(-1)[0])
        lines.append(
            field(
                f"↳ {name.removesuffix('_component').replace('_', ' ')}",
                reward_text(component_value),
            )
        )
    if deltas:
        lines.append(
            field(
                "signal deltas",
                "  ·  ".join(
                    f"{name.removesuffix('_delta').replace('_', ' ')} {format_action(value)}"
                    for name, value in deltas.items()
                ),
            )
        )
    if diagnostics.event_transitions:
        lines.append(
            field(
                "signal changes",
                "  ·  ".join(
                    f"{name.replace('_', ' ')} {format_action(source)}→{format_action(target)}"
                    for name, (source, target) in diagnostics.event_transitions.items()
                ),
            )
        )
    lines.extend(
        [
            field("events", "  ·  ".join(event_labels) if event_labels else ansi("none", "dim")),
            field(
                "boundary",
                ansi("  ·  ".join(boundary_parts), "red" if outcome == "failure" else "yellow")
                if boundary_parts
                else ansi("continuing", "green"),
            ),
            field("outcome", ansi(outcome.upper(), outcome_style)),
            "",
            section("↻" if transition.boundary else "→", "NEXT", style="green"),
            field(
                "conditioning",
                repr(transition.next_task)
                if transition.next_task is not None
                else ansi("none", "dim"),
            ),
        ]
    )
    if transition.boundary:
        lines.extend(
            [
                field("observation", ansi("same-step reset", "yellow")),
                field("terminal frame", ansi("preserved for inspection", "green")),
                field("next seed", diagnostics.next_episode_seed),
            ]
        )
    else:
        lines.append(field("observation", ansi("ordinary successor", "green")))
    title = f"TRANSITION  ·  EPISODE {transition.episode}  ·  STEP {transition.step}"
    accent = "red" if outcome == "failure" else "green" if outcome == "success" else "cyan"
    return terminal_panel(title, lines, accent=accent)


def _raw_transition_payload(transition: _PlaybackTransition) -> dict[str, object]:
    diagnostics = transition.diagnostics
    terminal_observation = transition.info.get("terminal_observation")
    return {
        "runtime": None if diagnostics is None else asdict(diagnostics),
        "sb3_info_keys": sorted(transition.info),
        "terminal_observation_present": "terminal_observation" in transition.info,
        "terminal_observation": (
            None if terminal_observation is None else model_input_lines(terminal_observation)
        ),
        "reset_info": transition.info.get("reset_info"),
    }


def _read_debug_line(session: _PlaybackSession) -> str | None:
    prompt = debug_prompt()
    if session.viewer is None or not sys.stdin.isatty():
        try:
            return input(prompt)
        except EOFError:
            return None
    try:
        selector = selectors.DefaultSelector()
        selector.register(sys.stdin, selectors.EVENT_READ)
    except AttributeError, OSError, ValueError:
        try:
            return input(prompt)
        except EOFError:
            return None
    print(prompt, end="", flush=True)
    try:
        while True:
            if not session.render():
                print()
                return None
            if selector.select(timeout=0.02):
                line = sys.stdin.readline()
                return None if line == "" else line.rstrip("\n")
    finally:
        selector.close()


@contextlib.contextmanager
def _deferred_sigint():
    interrupted = False
    previous = signal.getsignal(signal.SIGINT)

    def handle_interrupt(_signum, _frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handle_interrupt)
    try:
        yield lambda: interrupted
    finally:
        signal.signal(signal.SIGINT, previous)


def _run_debugger(
    session: _PlaybackSession,
    args: argparse.Namespace,
    config_text: str,
) -> int:
    event_names = tuple(session.env.runtime.kernel.event_names)
    boundaries = 0
    print(
        terminal_panel(
            "INTERACTIVE POLICY DEBUGGER",
            [
                section("🧪", "READY", style="cyan"),
                field("Enter", "take one policy step"),
                field("continue", "run to an event or boundary"),
                field("inspect", "show policy  ·  show input  ·  show raw"),
                field("commands", "enter help for the full command list"),
            ],
            accent="cyan",
        ),
        flush=True,
    )
    if not session.render():
        return 0
    while True:
        try:
            line = _read_debug_line(session)
        except KeyboardInterrupt:
            print(
                "\n" + status_message("■", "interrupted; debugger is still active", style="yellow"),
                flush=True,
            )
            continue
        if line is None:
            return 0
        try:
            command = parse_debug_command(line, event_names)
        except DebugCommandError as exc:
            print(status_message("✗", str(exc), style="red"), flush=True)
            continue
        if command.name == "quit":
            return 0
        if command.name == "help":
            print(debug_help(event_names), flush=True)
            continue
        if command.name == "show":
            if command.target == "policy":
                print(
                    format_policy_detail(session.inspect_policy(), session.action_names),
                    flush=True,
                )
            elif command.target == "input":
                print(format_model_input(session.model_obs), flush=True)
            elif command.target == "raw":
                if session.last_transition is None:
                    print(
                        status_message("○", "no transition has been stepped", style="yellow"),
                        flush=True,
                    )
                else:
                    print(format_raw(_raw_transition_payload(session.last_transition)), flush=True)
            elif command.target == "config":
                print(
                    terminal_panel(
                        "PLAYBACK CONFIG",
                        [
                            *config_text.splitlines(),
                            "",
                            section("●", "ACTIVE SESSION", style="green"),
                            field("seed", session.active_seed),
                            field("conditioning", repr(session.active_task)),
                        ],
                        accent="blue",
                    ),
                    flush=True,
                )
            elif session.last_transition is None:
                print(
                    status_message("○", "no transition has been stepped", style="yellow"),
                    flush=True,
                )
            else:
                print(
                    _transition_debug_text(session.last_transition, session.action_names),
                    flush=True,
                )
            continue
        if command.name == "reset":
            try:
                seed = (
                    session.initial_seed
                    if command.seed is None
                    else validate_eval_seed(command.seed)
                )
                session.restart(seed)
            except ValueError as exc:
                print(status_message("✗", str(exc), style="red"), flush=True)
                continue
            print(status_message("↻", f"reset complete  ·  seed {seed}", style="green"), flush=True)
            if not session.render():
                return 0
            continue

        try:
            if command.name == "step":
                with _deferred_sigint() as interrupted:
                    for _ in range(command.count):
                        transition = session.step()
                        print(
                            _transition_debug_text(transition, session.action_names),
                            flush=True,
                        )
                        if not session.render():
                            return 0
                        if transition.boundary:
                            boundaries += 1
                            if args.episodes > 0 and boundaries >= args.episodes:
                                return 0
                        if interrupted():
                            print(
                                status_message(
                                    "■",
                                    "interrupted after the completed step",
                                    style="yellow",
                                ),
                                flush=True,
                            )
                            break
            else:
                transition = None
                advanced = 0
                matched = False
                with _deferred_sigint() as interrupted:
                    for advanced in range(1, 10_001):
                        transition = session.step()
                        if not session.render():
                            return 0
                        if transition.boundary:
                            boundaries += 1
                        matched = (
                            transition.boundary
                            if command.target == "done"
                            else bool(transition.events)
                            if command.target is None
                            else command.target in transition.events
                        )
                        if matched or transition.boundary or interrupted():
                            break
                        if args.episodes > 0 and boundaries >= args.episodes:
                            break
                was_interrupted = interrupted()
                if transition is not None:
                    print(
                        status_message("⏩", f"advanced {advanced:,} steps", style="cyan"),
                        flush=True,
                    )
                    print(
                        _transition_debug_text(transition, session.action_names),
                        flush=True,
                    )
                if was_interrupted:
                    print(
                        status_message("■", "interrupted after the completed step", style="yellow"),
                        flush=True,
                    )
                if (
                    advanced == 10_000
                    and transition is not None
                    and not (matched or transition.boundary)
                ):
                    print(
                        status_message(
                            "⚠",
                            "continue stopped at the 10,000-step safety limit",
                            style="yellow",
                        ),
                        flush=True,
                    )
                if args.episodes > 0 and boundaries >= args.episodes:
                    return 0
        except KeyboardInterrupt:
            print(
                "\n" + status_message("■", "interrupted; debugger is still active", style="yellow"),
                flush=True,
            )


def _run_normal_playback(
    session: _PlaybackSession,
    args: argparse.Namespace,
    throttle,
) -> int:
    episode_iter = count() if args.episodes <= 0 else range(args.episodes)
    for episode in episode_iter:
        if episode:
            session.restart(args.seed + episode)
        if not session.render():
            return 0
        throttle()
        max_episode_steps = task_max_episode_steps(session.config)
        final_transition = None
        for _ in playback_step_indices(max_episode_steps):
            final_transition = session.step()
            if not session.render():
                return 0
            throttle()
            if final_transition.boundary:
                status = (
                    "complete"
                    if final_transition.completed
                    else "terminated"
                    if final_transition.terminated
                    else "truncated"
                )
                print(
                    f"episode={episode + 1} seed={final_transition.seed} "
                    f"reward={final_transition.total_reward:.2f} "
                    f"max_x={final_transition.max_x_pos} steps={final_transition.step} "
                    f"status={status} complete={final_transition.completed}",
                    flush=True,
                )
                time.sleep(0.5)
                break
        else:
            print(
                f"episode={episode + 1} seed={args.seed + episode} "
                f"reward={0.0 if final_transition is None else final_transition.total_reward:.2f} "
                f"max_x={0 if final_transition is None else final_transition.max_x_pos} "
                f"steps={max_episode_steps} status=max_steps",
                flush=True,
            )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    argv_list = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(argv_list)
    args.respect_task_termination = not args.continuous_play
    explicit_dests = explicit_arg_dests(parser, argv_list)
    if args.attribution_interval is None:
        args.attribution_interval = 8 if args.attribution == "occlusion" else 1
    with startup_progress("Resolving model reference", disabled=args.no_progress):
        ref = model_source_ref(args)
    with startup_progress(
        "Downloading model" if ref is not None else "Opening local model",
        disabled=args.no_progress,
    ):
        source = resolve_single_model_source(args, resolved_ref=ref)
    args.model = str(source.model_path)
    if ref is not None:
        print(f"Downloaded model: {args.model}", flush=True)
    with startup_progress("Loading playback metadata", disabled=args.no_progress):
        if source.bundle is not None:
            contract = evaluation_contract(source.bundle.recipe)
            artifact_config = env_config_from_config_dict(contract["environment"])
            if artifact_config is None:
                raise ValueError("policy bundle recipe has no evaluation environment")
            artifact_config = resolve_env_config(artifact_config)
            if "seed" not in explicit_dests:
                args.seed = int(contract["seed"])
            if args.continuous_play:
                artifact_config = playback_env_config(
                    artifact_config,
                    respect_task_termination=False,
                )
        else:
            artifact_config = load_playback_env_config(
                source.model_path,
                respect_task_termination=args.respect_task_termination,
            )
        if args.env_provider:
            artifact_config = resolve_env_config(
                replace(artifact_config, env_provider=str(args.env_provider))
            )
    args.seed = validate_eval_seed(args.seed)
    config = artifact_config
    display_config = display_replay_config(config)
    print_resolved_play_launch(
        args,
        argv=argv_list,
        artifact_ref=ref,
        policy_config=config,
        display_config=display_config,
    )
    with startup_progress("Checking provider runtime", disabled=args.no_progress):
        assert_provider_runtime_available(config)

    with startup_progress("Loading policy runtime", disabled=args.no_progress):
        from rlab.policy_models import load_policy_model

    with startup_progress("Loading model checkpoint", disabled=args.no_progress):
        model = load_policy_model(args.model, device=resolve_sb3_device(args.device))
    if args.attribution != "none":
        if not hasattr(model, "policy"):
            raise ValueError("policy attribution is unavailable for non-neural policies")
        with startup_progress("Preparing policy attribution", disabled=args.no_progress):
            attributor = PolicyActionAttributor(model)
    else:
        attributor = None

    with startup_progress("Creating policy environment", disabled=args.no_progress):
        policy_env = make_eval_vec_env(
            config=config,
            n_envs=1,
            seed=args.seed,
            capture_step_diagnostics=args.debug,
        )
    bind_action_space = getattr(model, "bind_action_space", None)
    if callable(bind_action_space):
        bind_action_space(policy_env.action_space)

    session = _PlaybackSession(
        model=model,
        env=policy_env,
        config=config,
        initial_seed=args.seed,
        attributor=attributor,
        attribution_mode=args.attribution,
        attribution_interval=args.attribution_interval,
        attribution_opacity=args.attribution_opacity,
    )
    with startup_progress("Resetting policy environment", disabled=args.no_progress):
        session.restart(args.seed)
    if session.current_frame is not None:
        with startup_progress("Creating game viewer", disabled=args.no_progress):
            session.viewer = PygameViewer(
                session.current_frame.shape,
                scale=DEFAULT_VIEWER_SCALE,
                position=None,
            )
    if (args.show_obs or attributor is not None) and session.frames is not None:
        with startup_progress("Creating observation viewer", disabled=args.no_progress):
            session.obs_viewer = ObsStackViewer(
                scale=DEFAULT_OBS_VIEWER_SCALE,
                position=(40, 240),
            )
    elif args.show_obs:
        print("warning: policy observation is not a four-frame image stack", flush=True)
    if session.viewer is None and not args.debug and args.episodes <= 0:
        raise ValueError("non-rendering playback requires --debug or a positive --episodes limit")
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

    try:
        config_text = "\n".join(
            resolved_play_launch_lines(
                args,
                argv=argv_list,
                artifact_ref=ref,
                policy_config=config,
                display_config=display_config,
            )
        )
        config_text += (
            "\n"
            f"checkpoint_step={source.checkpoint_step or '-'} "
            f"environment_hash={source.run_config.get('environment_hash', '-')}"
        )
        if args.debug:
            return _run_debugger(session, args, config_text)
        return _run_normal_playback(session, args, throttle)
    finally:
        if session.obs_viewer is not None:
            session.obs_viewer.close()
        if session.viewer is not None:
            session.viewer.close()
        try:
            policy_env.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
