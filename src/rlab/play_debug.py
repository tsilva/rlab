from __future__ import annotations

import hashlib
import os
import pprint
import re
import shlex
import shutil
import textwrap
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
)


ANSI_RESET = "\033[0m"
ANSI_STYLES = {
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
}
ANSI_PATTERN = re.compile(r"\x1b\[[0-9;]*m")


def ansi(text: object, style: str) -> str:
    rendered = str(text)
    if os.environ.get("NO_COLOR") or os.environ.get("RLAB_NO_COLOR"):
        return rendered
    return f"{ANSI_STYLES[style]}{rendered}{ANSI_RESET}"


def _visible_length(text: str) -> int:
    width = 0
    for character in ANSI_PATTERN.sub("", text):
        if unicodedata.combining(character):
            continue
        codepoint = ord(character)
        if 0x1F000 <= codepoint <= 0x1FAFF:
            width += 2
        else:
            width += 2 if unicodedata.east_asian_width(character) in {"F", "W"} else 1
    return width


def terminal_panel(title: str, lines: Sequence[str], *, accent: str = "cyan") -> str:
    """Render a compact scrollback-friendly terminal card."""

    width = max(40, min(112, shutil.get_terminal_size((96, 24)).columns))
    inner_width = width - 4
    plain_title = ANSI_PATTERN.sub("", title)
    if _visible_length(plain_title) > width - 7:
        plain_title = plain_title[: width - 8] + "…"
    top_fill = "─" * max(1, width - _visible_length(plain_title) - 5)

    def border(value: str) -> str:
        return ansi(value, accent)

    rendered = [f"{border('╭─')} {ansi(plain_title, 'bold')} {border(top_fill + '╮')}"]
    for source_line in lines:
        source_line = str(source_line)
        if not source_line:
            wrapped = [""]
        elif _visible_length(source_line) <= inner_width:
            wrapped = [source_line]
        else:
            # Long diagnostic values are more useful wrapped than clipped. Strip inline
            # styling only for the wrapped line so ANSI bytes do not distort its width.
            wrapped = textwrap.wrap(
                ANSI_PATTERN.sub("", source_line),
                width=inner_width,
                subsequent_indent="  ",
                replace_whitespace=False,
                drop_whitespace=True,
            ) or [""]
        for line in wrapped:
            padding = " " * max(0, inner_width - _visible_length(line))
            rendered.append(f"{border('│')} {line}{padding} {border('│')}")
    rendered.append(border("╰" + "─" * (width - 2) + "╯"))
    return "\n".join(rendered)


def section(icon: str, title: str, *, style: str = "cyan") -> str:
    return f"{ansi(icon, style)}  {ansi(title, 'bold')}"


def field(label: str, value: object, *, indent: int = 3) -> str:
    return f"{' ' * indent}{ansi(f'{label:<17}', 'dim')} {value}"


def humanize_name(value: str) -> str:
    if value.lower() in {"noop", "no_op", "none"}:
        return "NOOP"
    return " + ".join(part.upper() for part in value.split("_") if part)


def probability_bar(probability: float, *, width: int = 18) -> str:
    probability = min(1.0, max(0.0, probability))
    filled = int(round(probability * width))
    style = "green" if probability >= 0.5 else "yellow" if probability >= 0.2 else "magenta"
    return ansi("█" * filled, style) + ansi("░" * (width - filled), "dim")


def reward_text(value: float) -> str:
    style = "green" if value > 0 else "red" if value < 0 else "dim"
    return ansi(f"{value:+.5g}", style)


def status_message(icon: str, message: str, *, style: str = "cyan") -> str:
    return f"{ansi(icon, style)}  {message}"


def debug_prompt() -> str:
    return f"{ansi('🧪 rlab', 'cyan')} {ansi('debug', 'dim')} {ansi('›', 'bold')} "


@dataclass(frozen=True)
class PolicyDecision:
    distribution_kind: str
    raw_action: np.ndarray
    executed_action: np.ndarray
    value: float
    log_probability: float
    entropy: float | None
    mode: np.ndarray
    probabilities: np.ndarray | None = None
    component_probabilities: tuple[np.ndarray, ...] = ()
    mean: np.ndarray | None = None
    stddev: np.ndarray | None = None
    sampled: bool = True

    @property
    def selected_discrete_action(self) -> int | None:
        if self.probabilities is None or self.raw_action.size != 1:
            return None
        return int(self.raw_action.reshape(-1)[0])

    @property
    def selected_probability(self) -> float | None:
        action = self.selected_discrete_action
        if action is None:
            return None
        return float(self.probabilities[action])

    @property
    def selected_rank(self) -> int | None:
        probability = self.selected_probability
        if probability is None:
            return None
        return 1 + int(np.count_nonzero(self.probabilities > probability))


def _as_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def _postprocess_action(policy: Any, raw_action: np.ndarray) -> np.ndarray:
    action = np.asarray(raw_action).reshape((-1, *policy.action_space.shape))
    if isinstance(policy.action_space, gym.spaces.Box):
        if policy.squash_output:
            action = policy.unscale_action(action)
        else:
            action = np.clip(action, policy.action_space.low, policy.action_space.high)
    return np.asarray(action)


def _decision_from_distribution(
    policy: Any,
    distribution: Any,
    *,
    raw_tensor: torch.Tensor,
    value_tensor: torch.Tensor,
    log_probability_tensor: torch.Tensor,
    sampled: bool,
) -> PolicyDecision:
    entropy_tensor = distribution.entropy()
    mode_tensor = distribution.mode()
    raw_action = _as_numpy(raw_tensor).reshape((-1, *policy.action_space.shape))
    executed_action = _postprocess_action(policy, raw_action)
    entropy = None if entropy_tensor is None else float(_as_numpy(entropy_tensor).reshape(-1)[0])
    common = {
        "raw_action": raw_action[0].copy(),
        "executed_action": executed_action[0].copy(),
        "value": float(_as_numpy(value_tensor).reshape(-1)[0]),
        "log_probability": float(_as_numpy(log_probability_tensor).reshape(-1)[0]),
        "entropy": entropy,
        "mode": _as_numpy(mode_tensor)[0].copy(),
        "sampled": sampled,
    }
    if isinstance(distribution, CategoricalDistribution):
        return PolicyDecision(
            **common,
            distribution_kind="categorical",
            probabilities=_as_numpy(distribution.distribution.probs)[0].copy(),
        )
    if isinstance(distribution, MultiCategoricalDistribution):
        return PolicyDecision(
            **common,
            distribution_kind="multi_categorical",
            component_probabilities=tuple(
                _as_numpy(component.probs)[0].copy() for component in distribution.distribution
            ),
        )
    if isinstance(distribution, BernoulliDistribution):
        return PolicyDecision(
            **common,
            distribution_kind="bernoulli",
            component_probabilities=(_as_numpy(distribution.distribution.probs)[0].copy(),),
        )
    if isinstance(distribution, (DiagGaussianDistribution, StateDependentNoiseDistribution)):
        return PolicyDecision(
            **common,
            distribution_kind="gaussian",
            mean=_as_numpy(distribution.distribution.mean)[0].copy(),
            stddev=_as_numpy(distribution.distribution.stddev)[0].copy(),
        )
    raise TypeError(f"unsupported PPO distribution {type(distribution).__name__}")


def sample_policy_decision(model: Any, model_obs: Any) -> PolicyDecision:
    """Sample once from PPO and describe that same state without another sample."""

    policy = model.policy
    policy.set_training_mode(False)
    obs_tensor, _vectorized = policy.obs_to_tensor(model_obs)
    with torch.no_grad():
        raw_tensor, value_tensor, log_probability_tensor = policy.forward(
            obs_tensor,
            deterministic=False,
        )
        distribution = policy.get_distribution(obs_tensor)
    return _decision_from_distribution(
        policy,
        distribution,
        raw_tensor=raw_tensor,
        value_tensor=value_tensor,
        log_probability_tensor=log_probability_tensor,
        sampled=True,
    )


def inspect_policy(model: Any, model_obs: Any) -> PolicyDecision:
    """Inspect a state using its mode without sampling or changing policy RNG."""

    policy = model.policy
    policy.set_training_mode(False)
    obs_tensor, _vectorized = policy.obs_to_tensor(model_obs)
    with torch.no_grad():
        distribution = policy.get_distribution(obs_tensor)
        mode_tensor = distribution.mode()
        value_tensor = policy.predict_values(obs_tensor)
        log_probability_tensor = distribution.log_prob(mode_tensor)
    return _decision_from_distribution(
        policy,
        distribution,
        raw_tensor=mode_tensor,
        value_tensor=value_tensor,
        log_probability_tensor=log_probability_tensor,
        sampled=False,
    )


def _format_number(value: Any) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.5g}"
    return str(value)


def format_action(value: Any) -> str:
    arr = np.asarray(value)
    if arr.ndim == 0:
        return _format_number(arr.item())
    return np.array2string(arr, precision=4, separator=",", threshold=16)


def action_name(action: int, names: Sequence[str]) -> str:
    return names[action] if 0 <= action < len(names) else str(action)


def action_display_name(action: int, names: Sequence[str]) -> str:
    return humanize_name(action_name(action, names))


def format_policy_compact(decision: PolicyDecision, action_names: Sequence[str] = ()) -> str:
    selected = decision.selected_discrete_action
    entropy = "unavailable" if decision.entropy is None else f"{decision.entropy:.4g} nats"
    if selected is not None and decision.probabilities is not None:
        ranked = np.argsort(-decision.probabilities)[:3]
        alternatives = ", ".join(
            f"{action_name(int(index), action_names)}={decision.probabilities[index]:.3f}"
            for index in ranked
        )
        return (
            f"action={action_name(selected, action_names)} "
            f"p={decision.selected_probability:.3f} rank={decision.selected_rank} "
            f"top=[{alternatives}] entropy={entropy} V(s_t)={decision.value:.4g}"
        )
    log_label = "log_density" if decision.distribution_kind == "gaussian" else "log_probability"
    raw = ""
    if not np.allclose(decision.raw_action, decision.executed_action):
        raw = f"raw={format_action(decision.raw_action)} "
    return (
        f"{raw}action={format_action(decision.executed_action)} "
        f"{log_label}={decision.log_probability:.4g} "
        f"entropy={entropy} V(s_t)={decision.value:.4g}"
    )


def policy_summary_lines(
    decision: PolicyDecision,
    action_names: Sequence[str] = (),
) -> list[str]:
    selected = decision.selected_discrete_action
    entropy = "unavailable" if decision.entropy is None else f"{decision.entropy:.4g} nats"
    if selected is not None and decision.probabilities is not None:
        probability = float(decision.probabilities[selected])
        selected_name = action_display_name(selected, action_names)
        ranked = [
            int(index)
            for index in np.argsort(-decision.probabilities)
            if int(index) != selected
        ][:3]
        alternatives = "  ·  ".join(
            f"{action_display_name(index, action_names)} "
            f"{ansi(f'{decision.probabilities[index]:.1%}', 'dim')}"
            for index in ranked
        )
        return [
            field(
                "sampled action" if decision.sampled else "policy mode",
                f"{ansi(selected_name, 'bold')}  {probability_bar(probability)}  "
                f"{ansi(f'{probability:.1%}', 'green' if probability >= 0.5 else 'yellow')}  "
                f"{ansi(f'#{decision.selected_rank}', 'dim')}",
            ),
            field("alternatives", alternatives or ansi("none", "dim")),
            field(
                "policy stats",
                f"entropy {entropy}  ·  critic V(sₜ) {decision.value:+.4g}",
            ),
        ]
    log_label = "log density" if decision.distribution_kind == "gaussian" else "log probability"
    raw_changed = not np.allclose(decision.raw_action, decision.executed_action)
    lines = []
    if raw_changed:
        lines.append(field("raw sample", format_action(decision.raw_action)))
    lines.extend(
        [
            field("executed action", format_action(decision.executed_action)),
            field(log_label, f"{decision.log_probability:+.4g}"),
            field(
                "policy stats",
                f"entropy {entropy}  ·  critic V(sₜ) {decision.value:+.4g}",
            ),
        ]
    )
    return lines


def format_policy_detail(
    decision: PolicyDecision,
    action_names: Sequence[str] = (),
) -> str:
    lines = [section("🎲", "POLICY DISTRIBUTION", style="magenta")]
    lines.extend(policy_summary_lines(decision, action_names))
    lines.append("")
    if decision.probabilities is not None:
        lines.append(section("▦", "ALL ACTIONS", style="blue"))
        for rank, index in enumerate(np.argsort(-decision.probabilities), start=1):
            marker = (
                ansi("● sampled", "green")
                if decision.sampled and int(index) == decision.selected_discrete_action
                else ansi("◆ mode", "cyan")
                if not decision.sampled and int(index) == decision.selected_discrete_action
                else ansi("·", "dim")
            )
            lines.append(
                f"   {rank:>2}  {action_display_name(int(index), action_names):<18} "
                f"{probability_bar(float(decision.probabilities[index]), width=14)}  "
                f"{decision.probabilities[index]:>7.2%}  {marker}"
            )
    elif decision.component_probabilities:
        heading = "BIT MARGINALS" if decision.distribution_kind == "bernoulli" else "COMPONENTS"
        lines.append(section("▦", heading, style="blue"))
        if decision.distribution_kind == "bernoulli":
            probabilities = np.asarray(decision.component_probabilities[0]).reshape(-1)
            for component, probability in enumerate(probabilities):
                bit = int(np.asarray(decision.raw_action).reshape(-1)[component])
                lines.append(
                    f"   bit {component:<2} sampled={bit}  "
                    f"P(1) {probability_bar(float(probability), width=14)} "
                    f"{probability:>7.2%}"
                )
        else:
            for component, probabilities in enumerate(decision.component_probabilities):
                sampled = int(np.asarray(decision.raw_action).reshape(-1)[component])
                lines.append(
                    field(
                        f"component {component}",
                        f"sampled {sampled}  probabilities {format_action(probabilities)}",
                    )
                )
    else:
        lines.append(section("∿", "CONTINUOUS POLICY", style="blue"))
        lines.extend(
            [
                field(
                    "raw sample" if decision.sampled else "policy mode",
                    format_action(decision.raw_action),
                ),
                field("executed action", format_action(decision.executed_action)),
                field("mode", format_action(decision.mode)),
                field("mean", format_action(decision.mean)),
                field("standard deviation", format_action(decision.stddev)),
            ]
        )
    lines.extend(
        [
            "",
            field(
                "critic meaning",
                "expected discounted return on the checkpoint training-reward scale",
            ),
        ]
    )
    return terminal_panel("POLICY INSPECTOR", lines, accent="magenta")


def _input_leaf_lines(name: str, value: Any) -> list[str]:
    arr = np.asarray(value)
    if arr.dtype == object:
        return [f"{name}: {value!r}"]
    if arr.size <= 32:
        return [f"{name}: shape={arr.shape} dtype={arr.dtype} values={format_action(arr)}"]
    digest = hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]
    finite = arr[np.isfinite(arr)] if np.issubdtype(arr.dtype, np.number) else np.array([])
    value_range = (
        f" min={_format_number(finite.min())} max={_format_number(finite.max())}"
        if finite.size
        else ""
    )
    return [f"{name}: shape={arr.shape} dtype={arr.dtype}{value_range} sha256={digest}"]


def model_input_lines(model_obs: Any) -> list[str]:
    if isinstance(model_obs, Mapping):
        lines: list[str] = []
        for name, value in model_obs.items():
            lines.extend(_input_leaf_lines(str(name), value))
        return lines
    return _input_leaf_lines("observation", model_obs)


def format_model_input(model_obs: Any) -> str:
    body = model_input_lines(model_obs)
    return terminal_panel(
        "MODEL INPUT",
        [section("👁", "OBSERVATION", style="blue"), *(f"   {line}" for line in body)],
        accent="blue",
    )


def _raw_leaf(value: Any) -> str:
    if isinstance(value, np.ndarray):
        return np.array2string(value, precision=7, separator=",", threshold=value.size)
    return pprint.pformat(value, sort_dicts=True, width=72, compact=True)


def _flatten_raw(prefix: str, value: Any) -> list[tuple[str, str]]:
    if isinstance(value, Mapping):
        if not value:
            return [(prefix, "{}")]
        flattened: list[tuple[str, str]] = []
        for name, item in value.items():
            nested = f"{prefix}.{name}" if prefix else str(name)
            flattened.extend(_flatten_raw(nested, item))
        return flattened
    return [(prefix, _raw_leaf(value))]


def format_raw(value: Any) -> str:
    if not isinstance(value, Mapping):
        lines = [section("{ }", "CAPTURED FIELDS", style="yellow"), _raw_leaf(value)]
    else:
        lines = []
        icons = {
            "runtime": "⚙",
            "sb3_info_keys": "ⓘ",
            "terminal_observation": "▧",
            "terminal_observation_present": "●",
            "reset_info": "↻",
        }
        for group, group_value in value.items():
            if lines:
                lines.append("")
            label = str(group).replace("_", " ").upper()
            lines.append(section(icons.get(str(group), "◆"), label, style="yellow"))
            flattened = _flatten_raw("", group_value)
            for name, item in flattened:
                readable_name = name.replace("_", " ") if name else "value"
                item_lines = item.splitlines() or [""]
                lines.append(field(readable_name, item_lines[0]))
                lines.extend(f"{' ' * 21}{line}" for line in item_lines[1:])
    return terminal_panel(
        "RAW TRANSITION",
        lines,
        accent="yellow",
    )


DebugCommandName = Literal["step", "continue", "show", "reset", "help", "quit"]


@dataclass(frozen=True)
class DebugCommand:
    name: DebugCommandName
    count: int = 1
    target: str | None = None
    seed: int | None = None


class DebugCommandError(ValueError):
    pass


def parse_debug_command(line: str, event_names: Sequence[str] = ()) -> DebugCommand:
    try:
        parts = shlex.split(line)
    except ValueError as exc:
        raise DebugCommandError(str(exc)) from exc
    if not parts:
        return DebugCommand("step")
    name = parts[0].lower()
    if name == "step":
        if len(parts) > 2:
            raise DebugCommandError("usage: step [N]")
        try:
            count = int(parts[1]) if len(parts) == 2 else 1
        except ValueError as exc:
            raise DebugCommandError("step count must be an integer") from exc
        if not 1 <= count <= 100:
            raise DebugCommandError("step count must be in [1, 100]")
        return DebugCommand("step", count=count)
    if name == "continue":
        if len(parts) > 2:
            raise DebugCommandError("usage: continue [done|EVENT]")
        target = parts[1] if len(parts) == 2 else None
        if target is not None and target != "done" and target not in event_names:
            available = ", ".join(event_names) or "none configured"
            raise DebugCommandError(f"unknown event {target!r}; available events: {available}")
        return DebugCommand("continue", target=target)
    if name == "show":
        if len(parts) > 2:
            raise DebugCommandError("usage: show [last|policy|input|raw|config]")
        target = parts[1].lower() if len(parts) == 2 else "last"
        if target not in {"last", "policy", "input", "raw", "config"}:
            raise DebugCommandError("usage: show [last|policy|input|raw|config]")
        return DebugCommand("show", target=target)
    if name == "reset":
        if len(parts) > 2:
            raise DebugCommandError("usage: reset [SEED]")
        try:
            seed = int(parts[1]) if len(parts) == 2 else None
        except ValueError as exc:
            raise DebugCommandError("reset seed must be an integer") from exc
        return DebugCommand("reset", seed=seed)
    if name in {"help", "quit"}:
        if len(parts) != 1:
            raise DebugCommandError(f"usage: {name}")
        return DebugCommand(name)  # type: ignore[arg-type]
    raise DebugCommandError(f"unknown command {name!r}; enter help for commands")


def debug_help(event_names: Sequence[str]) -> str:
    events = ", ".join(event_names) or "none configured"
    return terminal_panel(
        "DEBUGGER COMMANDS",
        [
            section("↵", "MOVE", style="green"),
            field("Enter / step [N]", "advance 1–100 policy transitions"),
            field("continue", "run until any event or a boundary"),
            field("continue done", "run until a boundary"),
            field("continue EVENT", "run until that event (10,000-step cap)"),
            "",
            section("⌕", "INSPECT", style="blue"),
            field("show / show last", "redisplay the last transition"),
            field("show policy", "complete current distribution; does not advance"),
            field("show input", "current model input and conditioning"),
            field("show raw", "all captured provider and task fields"),
            field("show config", "resolved playback configuration"),
            "",
            section("↻", "SESSION", style="yellow"),
            field("reset [SEED]", "restart original seed, or start SEED"),
            field("help / quit", "show this card / leave cleanly"),
            field("configured events", events),
            field("interrupt", "Ctrl-C stops a run; Ctrl-D exits"),
        ],
        accent="cyan",
    )
