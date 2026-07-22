from __future__ import annotations

import argparse
import asyncio
import io
import json
import queue
import secrets
import struct
import threading
import time
import uuid
import webbrowser
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
from aiohttp import WSMsgType, web
from PIL import Image

from rlab.play import _PlaybackSession, _PlaybackTransition, render_obs_stack
from rlab.play_debug import ANSI_PATTERN, PolicyDecision, model_input_lines


PROTOCOL_VERSION = 1
HISTORY_LIMIT = 4096
COMMAND_QUEUE_LIMIT = 64
CLIENT_QUEUE_LIMIT = 64
FRAME_HEADER = struct.Struct(">4sBBHQ")
FRAME_MAGIC = b"RLP1"
FRAME_CODEC_PNG = 1
FRAME_GAME = 1
FRAME_OBSERVATION = 2
MAX_JSON_DEPTH = 5
MAX_JSON_ITEMS = 128
MAX_JSON_TEXT = 4096
INPUT_HEARTBEAT_SECONDS = 0.25
LAST_CLIENT_GRACE_SECONDS = 30.0


def _json_value(value: Any, *, depth: int = 0) -> Any:
    if depth >= MAX_JSON_DEPTH:
        return f"<{type(value).__name__}>"
    if value is None or isinstance(value, bool | int | str):
        if isinstance(value, str) and len(value) > MAX_JSON_TEXT:
            return value[:MAX_JSON_TEXT] + "…"
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else str(value)
    if isinstance(value, np.generic):
        return _json_value(value.item(), depth=depth + 1)
    if isinstance(value, np.ndarray):
        if value.size > MAX_JSON_ITEMS:
            finite = value[np.isfinite(value)] if np.issubdtype(value.dtype, np.number) else ()
            return {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "min": float(np.min(finite)) if len(finite) else None,
                "max": float(np.max(finite)) if len(finite) else None,
            }
        return _json_value(value.tolist(), depth=depth + 1)
    if is_dataclass(value):
        return _json_value(asdict(value), depth=depth + 1)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= MAX_JSON_ITEMS:
                result["…"] = f"{len(value) - MAX_JSON_ITEMS} more entries"
                break
            name = str(key)
            lowered = name.casefold()
            if any(token in lowered for token in ("password", "secret", "credential", "token")):
                result[name] = "<redacted>"
            elif name in {"terminal_observation", "final_observation"}:
                result[name] = _json_value(np.asarray(item), depth=depth + 1)
            else:
                result[name] = _json_value(item, depth=depth + 1)
        return result
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray | memoryview):
        rendered = [_json_value(item, depth=depth + 1) for item in value[:MAX_JSON_ITEMS]]
        if len(value) > MAX_JSON_ITEMS:
            rendered.append(f"<{len(value) - MAX_JSON_ITEMS} more entries>")
        return rendered
    if isinstance(value, bytes | bytearray | memoryview):
        return f"<{len(value)} bytes>"
    return str(value)[:MAX_JSON_TEXT]


def _decision_payload(decision: PolicyDecision | None) -> dict[str, Any] | None:
    if decision is None:
        return None
    return {
        "distribution": decision.distribution_kind,
        "raw_action": _json_value(decision.raw_action),
        "executed_action": _json_value(decision.executed_action),
        "value": decision.value,
        "log_probability": decision.log_probability,
        "entropy": decision.entropy,
        "mode": _json_value(decision.mode),
        "probabilities": _json_value(decision.probabilities),
        "component_probabilities": _json_value(decision.component_probabilities),
        "mean": _json_value(decision.mean),
        "stddev": _json_value(decision.stddev),
        "sampled": decision.sampled,
        "selected_action": decision.selected_discrete_action,
        "selected_probability": decision.selected_probability,
        "selected_rank": decision.selected_rank,
    }


def _numeric_signals(value: Mapping[str, Any]) -> dict[str, float]:
    signals: dict[str, float] = {}
    for key, item in value.items():
        if isinstance(item, bool):
            signals[str(key)] = float(item)
        elif isinstance(item, int | float | np.number):
            numeric = float(item)
            if np.isfinite(numeric):
                signals[str(key)] = numeric
        elif isinstance(item, np.ndarray) and item.size == 1:
            numeric = float(item.reshape(-1)[0])
            if np.isfinite(numeric):
                signals[str(key)] = numeric
    return dict(sorted(signals.items())[:MAX_JSON_ITEMS])


def transition_payload(transition: _PlaybackTransition) -> dict[str, Any]:
    diagnostics = transition.diagnostics
    provider_reward = transition.reward
    task_reward = transition.reward
    outcome = "continuing"
    task_metrics: Mapping[str, Any] = {}
    event_transitions: Mapping[str, Any] = {}
    boundary_reasons: list[str] = []
    if diagnostics is not None:
        provider_reward = diagnostics.provider_reward
        task_reward = diagnostics.reward
        outcome = diagnostics.outcome.name.lower()
        task_metrics = diagnostics.task_metrics
        event_transitions = diagnostics.event_transitions
        if diagnostics.provider_terminated:
            boundary_reasons.append("provider_terminated")
        if diagnostics.provider_truncated:
            boundary_reasons.append("provider_truncated")
        if diagnostics.task_terminated:
            boundary_reasons.append("task_terminated")
        if diagnostics.task_truncated:
            boundary_reasons.append("task_truncated")
    components = {
        name.removesuffix("_component"): float(np.asarray(value).reshape(-1)[0])
        for name, value in task_metrics.items()
        if name.endswith("_component") and np.asarray(value).size == 1
    }
    return {
        "sequence": transition.sequence,
        "episode": transition.episode,
        "step": transition.step,
        "seed": transition.seed,
        "start_id": transition.start_id,
        "action_source": transition.action_source,
        "executed_action": _json_value(transition.executed_action),
        "decision": _decision_payload(transition.decision),
        "before": {
            "task": _json_value(transition.pre_task),
            "model_input": model_input_lines(transition.model_obs),
            "game_frame": transition.before_frame is not None,
            "observation_frames": len(transition.before_frames),
        },
        "after": {
            "task": _json_value(transition.next_task),
            "game_frame": transition.after_frame is not None,
            "observation_frames": len(transition.after_frames),
        },
        "reward": {
            "provider": provider_reward,
            "shaped": task_reward,
            "step": transition.reward,
            "return": transition.total_reward,
            "components": components,
        },
        "events": list(transition.events),
        "event_transitions": _json_value(event_transitions),
        "signals": _numeric_signals(transition.info),
        "info": _json_value(transition.info),
        "max_x_pos": transition.max_x_pos,
        "terminated": transition.terminated,
        "truncated": transition.truncated,
        "completed": transition.completed,
        "boundary": transition.boundary,
        "boundary_reasons": boundary_reasons,
        "outcome": outcome,
        "attribution": transition.attribution is not None,
    }


def history_point(transition: _PlaybackTransition) -> dict[str, Any]:
    payload = transition_payload(transition)
    decision = payload["decision"] or {}
    reward = payload["reward"]
    return {
        "sequence": transition.sequence,
        "episode": transition.episode,
        "step": transition.step,
        "action": decision.get("selected_action"),
        "action_source": transition.action_source,
        "reward_provider": reward["provider"],
        "reward_shaped": reward["shaped"],
        "return": reward["return"],
        "value": decision.get("value"),
        "entropy": decision.get("entropy"),
        "events": payload["events"],
        "boundary": transition.boundary,
        "signals": payload["signals"],
        "components": reward["components"],
    }


class FrameEncoder:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._pending: dict[int, tuple[int, np.ndarray]] = {}
        self._latest: dict[int, tuple[int, bytes]] = {}
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="rlab-frame-encoder")

    def start(self) -> None:
        self._thread.start()

    def submit(self, kind: int, sequence: int, frame: np.ndarray | None) -> None:
        if frame is None:
            return
        owned = np.asarray(frame, dtype=np.uint8).copy()
        with self._condition:
            if self._closed:
                return
            self._pending[kind] = (sequence, owned)
            self._condition.notify()

    def latest(self) -> dict[int, tuple[int, bytes]]:
        with self._condition:
            return dict(self._latest)

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._pending and not self._closed:
                    self._condition.wait()
                if self._closed and not self._pending:
                    return
                pending = self._pending
                self._pending = {}
            for kind, (sequence, frame) in pending.items():
                output = io.BytesIO()
                Image.fromarray(frame, mode="RGB").save(
                    output,
                    format="PNG",
                    compress_level=1,
                )
                packet = FRAME_HEADER.pack(
                    FRAME_MAGIC,
                    kind,
                    FRAME_CODEC_PNG,
                    0,
                    sequence,
                ) + output.getvalue()
                with self._condition:
                    self._latest[kind] = (sequence, packet)


@dataclass(frozen=True)
class PlaybackCommand:
    command_id: str
    client_id: str
    name: str
    payload: Mapping[str, Any]
    expected_revision: int | None


@dataclass(frozen=True)
class PlaybackResponse:
    client_id: str
    payload: Mapping[str, Any]


class WebPlaybackRunner:
    """The only thread allowed to call the policy or environment."""

    def __init__(
        self,
        session: _PlaybackSession,
        args: argparse.Namespace,
        *,
        config_text: str,
    ) -> None:
        self.session = session
        self.args = args
        self.config_text = ANSI_PATTERN.sub("", config_text)
        self.commands: queue.Queue[PlaybackCommand] = queue.Queue(COMMAND_QUEUE_LIMIT)
        self.responses: queue.SimpleQueue[PlaybackResponse] = queue.SimpleQueue()
        self.encoder = FrameEncoder()
        self.history: deque[dict[str, Any]] = deque(maxlen=HISTORY_LIMIT)
        self._snapshot_lock = threading.Lock()
        self._latest_snapshot: dict[str, Any] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="rlab-playback-runtime")
        self.revision = 0
        self.run_state = "paused"
        self.driver = "policy"
        self.target_fps = max(0.0, float(args.fps))
        self.remaining_steps = 0
        self.continue_target: str | None = None
        self.continue_count = 0
        self.boundaries = 0
        self._input_lock = threading.Lock()
        self._pressed: tuple[str, ...] = ()
        self._input_updated_at = 0.0
        self._input_focused = False
        self._status_message: str | None = None

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def start(self) -> None:
        self.encoder.start()
        self._publish(None)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=10.0)
        self.encoder.close()

    def submit(self, command: PlaybackCommand) -> None:
        self.commands.put_nowait(command)

    def update_input(self, labels: Sequence[str], *, focused: bool) -> None:
        with self._input_lock:
            self._pressed = tuple(sorted({str(label).casefold() for label in labels}))
            self._input_updated_at = time.monotonic()
            self._input_focused = bool(focused)

    def clear_input(self) -> None:
        with self._input_lock:
            self._pressed = ()
            self._input_focused = False
            self._input_updated_at = 0.0

    def snapshot(self) -> dict[str, Any]:
        with self._snapshot_lock:
            return dict(self._latest_snapshot)

    def history_payload(self) -> dict[str, Any]:
        return {"type": "history", "points": list(self.history)}

    def _response(self, command: PlaybackCommand, *, ok: bool, **extra: Any) -> None:
        self.responses.put(
            PlaybackResponse(
                command.client_id,
                {
                    "type": "command_result",
                    "id": command.command_id,
                    "ok": ok,
                    "revision": self.revision,
                    **extra,
                },
            )
        )

    def _snapshot_payload(self, transition: _PlaybackTransition | None) -> dict[str, Any]:
        current = transition_payload(transition) if transition is not None else None
        try:
            event_names = list(self.session.env.runtime.kernel.event_names)
        except AttributeError:
            event_names = []
        return {
            "type": "snapshot",
            "protocol": PROTOCOL_VERSION,
            "revision": self.revision,
            "sequence": self.session.sequence,
            "run_state": self.run_state,
            "driver": self.driver,
            "interactive": self.session.interactive,
            "status_message": self._status_message,
            "session": {
                "episode": self.session.episode,
                "step": self.session.step_index,
                "seed": self.session.active_seed,
                "task": _json_value(self.session.active_task),
                "total_reward": self.session.total_reward,
                "max_x_pos": self.session.max_x_pos,
                "action_names": list(self.session.action_names),
                "event_names": event_names,
                "target_fps": self.target_fps,
                "episodes_limit": int(self.args.episodes),
                "history_size": len(self.history),
                "config": self.config_text,
            },
            "transition": current,
        }

    def _publish(self, transition: _PlaybackTransition | None) -> None:
        if transition is not None and (
            not self.history or int(self.history[-1]["sequence"]) != transition.sequence
        ):
            self.history.append(history_point(transition))
        if transition is not None:
            game_frame = transition.after_frame
            obs_frames = transition.before_frames
            attribution = transition.attribution
            sequence = transition.sequence
        else:
            game_frame = self.session.current_frame
            obs_frames = tuple(self.session.frames or ())
            attribution = None
            sequence = self.session.sequence
        self.encoder.submit(FRAME_GAME, sequence, game_frame)
        if obs_frames:
            obs_image = render_obs_stack(
                deque(obs_frames, maxlen=len(obs_frames)),
                scale=1,
                heatmap=attribution,
                heatmap_opacity=self.session.attribution_opacity,
            )
            self.encoder.submit(FRAME_OBSERVATION, sequence, obs_image)
        payload = self._snapshot_payload(transition)
        with self._snapshot_lock:
            self._latest_snapshot = payload

    def _set_state(self, state: str, *, message: str | None = None) -> None:
        self.run_state = state
        self._status_message = message
        self.revision += 1
        self._publish(self.session.last_transition)

    def _apply(self, command: PlaybackCommand) -> None:
        if (
            bool(command.payload.get("strict_revision", False))
            and command.expected_revision is not None
            and command.name not in {"pause", "stop"}
            and command.expected_revision != self.revision
        ):
            self._response(
                command,
                ok=False,
                error=f"stale revision {command.expected_revision}; current is {self.revision}",
            )
            return
        try:
            if command.name == "pause":
                self.remaining_steps = 0
                self.continue_target = None
                self.clear_input()
                self._set_state("paused", message="paused at a completed transition")
            elif command.name == "play":
                self.driver = str(command.payload.get("driver") or self.driver)
                if self.driver not in {"policy", "human"}:
                    raise ValueError(f"unsupported driver {self.driver!r}")
                self._set_state("playing")
            elif command.name == "step":
                count = int(command.payload.get("count", 1))
                if not 1 <= count <= 100:
                    raise ValueError("step count must be in [1, 100]")
                self.remaining_steps = count
                self.continue_target = None
                self._set_state("stepping")
            elif command.name == "continue":
                self.driver = "policy"
                self.continue_target = str(command.payload.get("target") or "any")
                self.continue_count = 0
                self.remaining_steps = 0
                self._set_state("continuing")
            elif command.name == "reset":
                seed_value = command.payload.get("seed")
                seed = self.session.initial_seed if seed_value in {None, ""} else int(seed_value)
                self.session.restart(seed)
                self.boundaries = 0
                self.driver = "policy"
                self._set_state("paused", message=f"reset to seed {seed}")
            elif command.name == "set_fps":
                fps = float(command.payload.get("fps", 0.0))
                if fps < 0 or not np.isfinite(fps):
                    raise ValueError("fps must be a finite value >= 0")
                self.target_fps = fps
                self.revision += 1
                self._publish(self.session.last_transition)
            elif command.name == "set_driver":
                driver = str(command.payload.get("driver") or "policy")
                if driver not in {"policy", "human"}:
                    raise ValueError(f"unsupported driver {driver!r}")
                self.driver = driver
                self.clear_input()
                self._set_state("paused", message=f"{driver} control selected")
            elif command.name == "inspect_policy":
                decision = self.session.inspect_policy()
                self._response(
                    command,
                    ok=True,
                    inspection={"kind": "policy", "decision": _decision_payload(decision)},
                )
                return
            elif command.name == "stop":
                self._response(command, ok=True)
                self._stop.set()
                return
            else:
                raise ValueError(f"unknown playback command {command.name!r}")
        except Exception as exc:
            self._set_state("paused", message=str(exc))
            self._response(command, ok=False, error=str(exc))
            return
        self._response(command, ok=True)

    def _drain_commands(self) -> None:
        for _ in range(COMMAND_QUEUE_LIMIT):
            try:
                command = self.commands.get_nowait()
            except queue.Empty:
                return
            self._apply(command)

    def _human_labels(self) -> tuple[str, ...]:
        with self._input_lock:
            fresh = time.monotonic() - self._input_updated_at <= INPUT_HEARTBEAT_SECONDS
            if not fresh or not self._input_focused:
                raise RuntimeError("human input lease expired; playback paused")
            return self._pressed

    def _step_once(self) -> _PlaybackTransition | None:
        try:
            transition = (
                self.session.step_human(self._human_labels())
                if self.driver == "human"
                else self.session.step()
            )
        except Exception as exc:
            self._set_state("paused", message=str(exc))
            return None
        self.revision += 1
        if transition.boundary:
            self.boundaries += 1
        if self.run_state == "stepping":
            self.remaining_steps -= 1
            if self.remaining_steps <= 0:
                self.run_state = "paused"
        elif self.run_state == "continuing":
            self.continue_count += 1
            target = self.continue_target or "any"
            matched = (
                transition.boundary
                if target == "done"
                else bool(transition.events)
                if target == "any"
                else target in transition.events
            )
            if matched or transition.boundary or self.continue_count >= 10_000:
                self.run_state = "paused"
                if self.continue_count >= 10_000 and not matched:
                    self._status_message = "continue reached the 10,000-step safety limit"
        if int(self.args.episodes) > 0 and self.boundaries >= int(self.args.episodes):
            self.run_state = "paused"
            self._status_message = f"episode limit reached ({self.boundaries})"
        self._publish(transition)
        return transition

    def _run(self) -> None:
        next_step_at = time.perf_counter()
        while not self._stop.is_set():
            self._drain_commands()
            if self.run_state not in {"playing", "stepping", "continuing"}:
                time.sleep(0.005)
                continue
            fps = 60.0 if self.driver == "human" and self.target_fps <= 0 else self.target_fps
            if fps > 0:
                now = time.perf_counter()
                if now < next_step_at:
                    time.sleep(min(next_step_at - now, 0.005))
                    continue
                next_step_at = max(next_step_at + 1.0 / fps, now)
            self._step_once()


class HumanRecordingRunner:
    """Bridge the synchronous dataset recorder to the shared web dashboard."""

    def __init__(self, session: Any, args: argparse.Namespace) -> None:
        self.session = session
        self.args = args
        self.encoder = FrameEncoder()
        self.responses: queue.SimpleQueue[PlaybackResponse] = queue.SimpleQueue()
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._snapshot_lock = threading.Lock()
        self._latest_snapshot: dict[str, Any] = {}
        self._pressed: tuple[str, ...] = ()
        self._input_updated_at = 0.0
        self._input_focused = False
        self._next_action_at = time.perf_counter()
        self.history: deque[dict[str, Any]] = deque(maxlen=HISTORY_LIMIT)
        self._transition: dict[str, Any] | None = None
        self._last_action: Any = None
        self.total_reward = 0.0
        self.revision = 0
        self.sequence = 0
        self.run_state = "paused"
        self.target_fps = max(float(getattr(args, "fps", None) or session.fps), 1.0)
        self._status_message = "Focus the game view, then press Play to begin recording"

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def start(self) -> None:
        self.encoder.start()
        self._publish()

    def stop(self) -> None:
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        self.encoder.close()

    def snapshot(self) -> dict[str, Any]:
        with self._snapshot_lock:
            return dict(self._latest_snapshot)

    def history_payload(self) -> dict[str, Any]:
        return {"type": "history", "points": list(self.history)}

    def clear_input(self) -> None:
        with self._condition:
            self._pressed = ()
            self._input_focused = False
            self._input_updated_at = 0.0
            self._condition.notify_all()

    def update_input(self, labels: Sequence[str], *, focused: bool) -> None:
        with self._condition:
            self._pressed = tuple(sorted({str(label).upper() for label in labels}))
            self._input_updated_at = time.monotonic()
            self._input_focused = bool(focused)
            self._condition.notify_all()

    def _response(self, command: PlaybackCommand, *, ok: bool, **extra: Any) -> None:
        self.responses.put(
            PlaybackResponse(
                command.client_id,
                {
                    "type": "command_result",
                    "id": command.command_id,
                    "ok": ok,
                    "revision": self.revision,
                    **extra,
                },
            )
        )

    def submit(self, command: PlaybackCommand) -> None:
        try:
            if command.name == "play":
                self.run_state = "playing"
                self._status_message = "Recording human controls"
            elif command.name == "pause":
                self.run_state = "paused"
                self.clear_input()
                self._status_message = "Recording paused"
            elif command.name == "set_fps":
                fps = float(command.payload.get("fps", self.target_fps))
                if not np.isfinite(fps) or fps <= 0:
                    raise ValueError("recording FPS must be a finite value > 0")
                self.target_fps = fps
                self._status_message = f"Recording at {fps:g} FPS"
            elif command.name == "set_driver":
                if command.payload.get("driver") != "human":
                    raise ValueError("dataset recording only supports human control")
                self.run_state = "paused"
                self._status_message = "Human control selected"
            elif command.name == "stop":
                self._response(command, ok=True)
                self.stop()
                return
            else:
                raise ValueError(f"{command.name or 'that command'} is unavailable while recording")
        except Exception as exc:
            self._status_message = str(exc)
            self._response(command, ok=False, error=str(exc))
        else:
            self._response(command, ok=True)
        finally:
            self.revision += 1
            self._publish()
            with self._condition:
                self._condition.notify_all()

    def _publish(self) -> None:
        with self._snapshot_lock:
            self._latest_snapshot = {
                "type": "snapshot",
                "protocol": PROTOCOL_VERSION,
                "mode": "recording",
                "revision": self.revision,
                "sequence": self.sequence,
                "run_state": self.run_state,
                "driver": "human",
                "interactive": True,
                "status_message": self._status_message,
                "session": {
                    "episode": 1,
                    "step": self.sequence,
                    "seed": None,
                    "task": None,
                    "total_reward": self.total_reward,
                    "max_x_pos": 0,
                    "action_names": [],
                    "event_names": [],
                    "target_fps": self.target_fps,
                    "episodes_limit": int(getattr(self.args, "episodes", None) or 0),
                    "history_size": len(self.history),
                    "config": (
                        "Human dataset recording. Browser input is translated through the "
                        "provider's declared control labels. This session is never promotion evidence."
                    ),
                },
                "transition": self._transition,
            }

    def action(self, frame: np.ndarray) -> tuple[Any | None, bool]:
        self.encoder.submit(FRAME_GAME, self.sequence, frame)
        self._publish()
        while not self.stopped:
            with self._condition:
                fresh = time.monotonic() - self._input_updated_at <= INPUT_HEARTBEAT_SECONDS
                if not (self.run_state == "playing" and self._input_focused and fresh):
                    self._condition.wait(timeout=0.05)
                    continue
                labels = self._pressed
            now = time.perf_counter()
            if now < self._next_action_at:
                time.sleep(self._next_action_at - now)
            self._next_action_at = max(self._next_action_at + 1.0 / self.target_fps, now)
            try:
                action = self.session.action_from_labels(labels)
            except ValueError as exc:
                self.run_state = "paused"
                self.clear_input()
                self._status_message = str(exc)
                self.revision += 1
                self._publish()
                continue
            self.sequence += 1
            self.revision += 1
            self._last_action = _json_value(action)
            return action, True
        return None, False

    def observe_transition(
        self,
        *,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Mapping[str, Any],
        next_frame: np.ndarray,
    ) -> None:
        self.total_reward += float(reward)
        boundary = bool(terminated or truncated)
        self._transition = {
            "sequence": self.sequence,
            "episode": 1,
            "step": self.sequence,
            "seed": None,
            "start_id": None,
            "action_source": "human",
            "executed_action": self._last_action,
            "decision": None,
            "before": {
                "task": None,
                "model_input": [],
                "game_frame": True,
                "observation_frames": 0,
            },
            "after": {"task": None, "game_frame": True, "observation_frames": 0},
            "reward": {
                "provider": float(reward),
                "shaped": float(reward),
                "step": float(reward),
                "return": self.total_reward,
                "components": {},
            },
            "events": [],
            "event_transitions": {},
            "signals": _numeric_signals(info),
            "info": _json_value(info),
            "max_x_pos": int(info.get("max_x_pos", 0)),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "completed": False,
            "boundary": boundary,
            "boundary_reasons": [
                name
                for name, active in (
                    ("provider_terminated", terminated),
                    ("provider_truncated", truncated),
                )
                if active
            ],
            "outcome": "boundary" if boundary else "continuing",
            "attribution": False,
        }
        self.history.append(
            {
                "sequence": self.sequence,
                "episode": 1,
                "step": self.sequence,
                "action": None,
                "action_source": "human",
                "reward_provider": float(reward),
                "reward_shaped": float(reward),
                "return": self.total_reward,
                "value": None,
                "entropy": None,
                "events": [],
                "boundary": boundary,
                "signals": _numeric_signals(info),
                "components": {},
            }
        )
        self.encoder.submit(FRAME_GAME, self.sequence, next_frame)
        self.revision += 1
        self._publish()


class WebClient:
    def __init__(
        self,
        client_id: str,
        socket: web.WebSocketResponse,
        subscriptions: set[str],
    ) -> None:
        self.client_id = client_id
        self.socket = socket
        self.subscriptions = subscriptions
        self.reliable: asyncio.Queue[str | bytes] = asyncio.Queue(CLIENT_QUEUE_LIMIT)
        self.event = asyncio.Event()
        self.latest_snapshot: str | None = None
        self.latest_snapshot_key: tuple[int, int, int] = (-1, -1, -1)
        self.sent_snapshot_key: tuple[int, int, int] = (-1, -1, -1)
        self.latest_frames: dict[int, tuple[int, bytes]] = {}
        self.sent_frames: dict[int, int] = {}
        self.closed = False

    def offer_reliable(self, payload: Mapping[str, Any]) -> None:
        rendered = json.dumps(payload, separators=(",", ":"), allow_nan=False)
        try:
            self.reliable.put_nowait(rendered)
        except asyncio.QueueFull:
            self.closed = True
        self.event.set()

    def offer_snapshot(self, payload: Mapping[str, Any]) -> None:
        key = (
            int(payload.get("revision", 0)),
            int(payload.get("sequence", 0)),
            int(payload.get("control_epoch", 0)),
        )
        if key >= self.latest_snapshot_key:
            self.latest_snapshot_key = key
            self.latest_snapshot = json.dumps(payload, separators=(",", ":"), allow_nan=False)
            self.event.set()

    def offer_frame(self, kind: int, sequence: int, packet: bytes) -> None:
        if sequence >= self.latest_frames.get(kind, (-1, b""))[0]:
            self.latest_frames[kind] = (sequence, packet)
            self.event.set()

    async def write(self) -> None:
        while not self.closed and not self.socket.closed:
            await self.event.wait()
            self.event.clear()
            while not self.reliable.empty():
                value = self.reliable.get_nowait()
                if isinstance(value, bytes):
                    await self.socket.send_bytes(value)
                else:
                    await self.socket.send_str(value)
            if (
                self.latest_snapshot is not None
                and self.latest_snapshot_key > self.sent_snapshot_key
            ):
                await self.socket.send_str(self.latest_snapshot)
                self.sent_snapshot_key = self.latest_snapshot_key
            for kind, (sequence, packet) in tuple(self.latest_frames.items()):
                if sequence > self.sent_frames.get(kind, -1):
                    await self.socket.send_bytes(packet)
                    self.sent_frames[kind] = sequence


class PlaybackWebServer:
    def __init__(self, runner: Any, args: argparse.Namespace) -> None:
        self.runner = runner
        self.args = args
        self.token = secrets.token_urlsafe(32)
        self.origin = ""
        self.clients: dict[str, WebClient] = {}
        self.control_holder: str | None = None
        self.control_epoch = 0
        self.stop_event = asyncio.Event()
        self.ever_connected = False
        self.last_client_at = time.monotonic()
        self._auto_started = False

    @property
    def asset_root(self) -> Path:
        return Path(__file__).with_name("web_player")

    @web.middleware
    async def security_headers(self, request: web.Request, handler):
        response = await handler(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' blob:; "
            "connect-src 'self' ws: wss:; object-src 'none'; base-uri 'none'; "
            "frame-ancestors 'none'"
        )
        return response

    async def page(self, _request: web.Request) -> web.FileResponse:
        return web.FileResponse(self.asset_root / "index.html")

    async def asset(self, request: web.Request) -> web.FileResponse:
        name = request.match_info["name"]
        if name not in {"app.js", "styles.css"}:
            raise web.HTTPNotFound()
        return web.FileResponse(self.asset_root / name)

    def _snapshot_for(self, client: WebClient, snapshot: Mapping[str, Any]) -> dict[str, Any]:
        return {
            **snapshot,
            "control_epoch": self.control_epoch,
            "control": {
                "client_id": client.client_id,
                "holder": self.control_holder,
                "has_control": self.control_holder == client.client_id,
            },
        }

    def _broadcast_control(self) -> None:
        snapshot = self.runner.snapshot()
        for client in self.clients.values():
            client.offer_snapshot(self._snapshot_for(client, snapshot))

    async def websocket(self, request: web.Request) -> web.WebSocketResponse:
        if request.headers.get("Origin") != self.origin:
            raise web.HTTPForbidden(text="invalid websocket origin")
        socket = web.WebSocketResponse(
            heartbeat=10.0,
            compress=False,
            max_msg_size=256 * 1024,
            writer_limit=256 * 1024,
        )
        await socket.prepare(request)
        client: WebClient | None = None
        writer: asyncio.Task[None] | None = None
        try:
            try:
                first = await asyncio.wait_for(socket.receive(), timeout=5.0)
            except TimeoutError:
                await socket.close(code=1008, message=b"authentication timeout")
                return socket
            if first.type != WSMsgType.TEXT:
                await socket.close(code=1008, message=b"hello required")
                return socket
            try:
                hello = json.loads(first.data)
            except json.JSONDecodeError:
                await socket.close(code=1008, message=b"invalid hello")
                return socket
            if hello.get("type") != "hello" or not secrets.compare_digest(
                str(hello.get("token") or ""), self.token
            ):
                await socket.close(code=1008, message=b"authentication failed")
                return socket
            subscriptions = {
                str(value)
                for value in hello.get("subscriptions", ("telemetry",))
                if str(value) in {"telemetry", "game", "observation"}
            }
            client_id = uuid.uuid4().hex
            client = WebClient(client_id, socket, subscriptions)
            self.clients[client_id] = client
            self.ever_connected = True
            if self.control_holder is None:
                self.control_holder = client_id
                self.control_epoch += 1
            client.offer_reliable(
                {
                    "type": "welcome",
                    "protocol": PROTOCOL_VERSION,
                    "client_id": client_id,
                    "history_limit": HISTORY_LIMIT,
                }
            )
            client.offer_reliable(self.runner.history_payload())
            client.offer_snapshot(self._snapshot_for(client, self.runner.snapshot()))
            for frame_kind, (sequence, packet) in self.runner.encoder.latest().items():
                subscription = "game" if frame_kind == FRAME_GAME else "observation"
                if subscription in client.subscriptions:
                    client.offer_frame(frame_kind, sequence, packet)
            writer = asyncio.create_task(client.write())
            self._broadcast_control()
            if not self._auto_started and not bool(getattr(self.args, "debug", False)):
                self._auto_started = True
                self.runner.submit(
                    PlaybackCommand(uuid.uuid4().hex, client_id, "play", {}, None)
                )
            async for message in socket:
                if message.type == WSMsgType.ERROR:
                    break
                if message.type != WSMsgType.TEXT:
                    continue
                try:
                    payload = json.loads(message.data)
                except json.JSONDecodeError:
                    client.offer_reliable({"type": "error", "error": "invalid JSON message"})
                    continue
                kind = str(payload.get("type") or "")
                if kind == "acquire_control":
                    if self.control_holder != client_id:
                        self.control_holder = client_id
                        self.control_epoch += 1
                        self.runner.clear_input()
                        self._broadcast_control()
                elif kind == "subscribe":
                    client.subscriptions = {
                        str(value)
                        for value in payload.get("subscriptions", ())
                        if str(value) in {"telemetry", "game", "observation"}
                    }
                elif kind == "history":
                    client.offer_reliable(self.runner.history_payload())
                elif kind == "input":
                    if self.control_holder != client_id:
                        client.offer_reliable({"type": "error", "error": "control lease required"})
                    else:
                        labels = payload.get("pressed", ())
                        self.runner.update_input(
                            labels if isinstance(labels, list) else (),
                            focused=bool(payload.get("focused", False)),
                        )
                elif kind == "command":
                    if self.control_holder != client_id:
                        client.offer_reliable(
                            {
                                "type": "command_result",
                                "id": str(payload.get("id") or ""),
                                "ok": False,
                                "error": "control lease required",
                            }
                        )
                        continue
                    try:
                        self.runner.submit(
                            PlaybackCommand(
                                str(payload.get("id") or uuid.uuid4().hex),
                                client_id,
                                str(payload.get("name") or ""),
                                payload.get("payload")
                                if isinstance(payload.get("payload"), Mapping)
                                else {},
                                int(payload["expected_revision"])
                                if payload.get("expected_revision") is not None
                                else None,
                            )
                        )
                    except queue.Full:
                        client.offer_reliable(
                            {
                                "type": "command_result",
                                "id": str(payload.get("id") or ""),
                                "ok": False,
                                "error": "command queue is full",
                            }
                        )
        finally:
            if client is not None:
                client.closed = True
                client.event.set()
                self.clients.pop(client.client_id, None)
                if self.control_holder == client.client_id:
                    self.control_holder = None
                    self.control_epoch += 1
                    self.runner.clear_input()
                    try:
                        self.runner.submit(
                            PlaybackCommand(
                                uuid.uuid4().hex,
                                client.client_id,
                                "pause",
                                {},
                                None,
                            )
                        )
                    except queue.Full:
                        pass
                self.last_client_at = time.monotonic()
                self._broadcast_control()
            if writer is not None:
                writer.cancel()
                await asyncio.gather(writer, return_exceptions=True)
        return socket

    async def pump(self) -> None:
        latest_snapshot_key = (-1, -1)
        latest_frames: dict[int, int] = {}
        while not self.stop_event.is_set():
            snapshot = self.runner.snapshot()
            key = (int(snapshot.get("revision", 0)), int(snapshot.get("sequence", 0)))
            if key != latest_snapshot_key:
                latest_snapshot_key = key
                for client in tuple(self.clients.values()):
                    if "telemetry" in client.subscriptions:
                        client.offer_snapshot(self._snapshot_for(client, snapshot))
            for kind, (sequence, packet) in self.runner.encoder.latest().items():
                if sequence == latest_frames.get(kind):
                    continue
                latest_frames[kind] = sequence
                subscription = "game" if kind == FRAME_GAME else "observation"
                for client in tuple(self.clients.values()):
                    if subscription in client.subscriptions:
                        client.offer_frame(kind, sequence, packet)
            while True:
                try:
                    response = self.runner.responses.get_nowait()
                except queue.Empty:
                    break
                client = self.clients.get(response.client_id)
                if client is not None:
                    client.offer_reliable(response.payload)
            for client_id, client in tuple(self.clients.items()):
                if client.closed:
                    await client.socket.close(code=1013, message=b"client is too slow")
                    self.clients.pop(client_id, None)
            if self.runner.stopped:
                self.stop_event.set()
                break
            if (
                self.ever_connected
                and not self.clients
                and time.monotonic() - self.last_client_at >= LAST_CLIENT_GRACE_SECONDS
            ):
                self.runner.stop()
                self.stop_event.set()
                break
            await asyncio.sleep(1.0 / 120.0)

    async def run(self) -> int:
        app = web.Application(middlewares=[self.security_headers])
        app.add_routes(
            [
                web.get("/", self.page),
                web.get("/panel/{panel}", self.page),
                web.get("/assets/{name}", self.asset),
                web.get("/ws", self.websocket),
            ]
        )
        app_runner = web.AppRunner(app, access_log=None)
        await app_runner.setup()
        site = web.TCPSite(app_runner, "127.0.0.1", int(self.args.port))
        await site.start()
        sockets = tuple(site._server.sockets) if site._server is not None else ()
        if not sockets:
            raise RuntimeError("player web server did not bind a socket")
        port = int(sockets[0].getsockname()[1])
        self.origin = f"http://127.0.0.1:{port}"
        url = f"{self.origin}/#token={self.token}"
        dashboard_label = str(getattr(self.args, "dashboard_label", "Player dashboard"))
        print(f"{dashboard_label}: {url}", flush=True)
        self.runner.start()
        pump = asyncio.create_task(self.pump())
        if not bool(getattr(self.args, "no_open", False)):
            webbrowser.open(url, new=1, autoraise=True)
        try:
            await self.stop_event.wait()
        finally:
            pump.cancel()
            await asyncio.gather(pump, return_exceptions=True)
            for client in tuple(self.clients.values()):
                await client.socket.close(code=1001, message=b"player shutting down")
            self.runner.stop()
            await app_runner.cleanup()
        return 0


def run_web_playback(
    session: _PlaybackSession,
    args: argparse.Namespace,
    *,
    config_text: str,
) -> int:
    runner = WebPlaybackRunner(session, args, config_text=config_text)
    server = PlaybackWebServer(runner, args)
    try:
        return asyncio.run(server.run())
    except KeyboardInterrupt:
        runner.stop()
        return 130


class WebHumanController:
    """Synchronous human controller backed by a loopback web dashboard."""

    def __init__(self, session: Any, args: argparse.Namespace) -> None:
        self.runner = HumanRecordingRunner(session, args)
        args.dashboard_label = "Recording dashboard"
        self.server = PlaybackWebServer(self.runner, args)
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._serve,
            name="rlab-recording-dashboard",
            daemon=True,
        )
        self._thread.start()
        deadline = time.monotonic() + 10.0
        while not self.server.origin and self._thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.01)
        if self._error is not None:
            raise RuntimeError("recording dashboard failed to start") from self._error
        if not self.server.origin:
            self.close()
            raise RuntimeError("recording dashboard did not start within 10 seconds")

    def _serve(self) -> None:
        try:
            asyncio.run(self.server.run())
        except BaseException as exc:
            self._error = exc
            self.runner.stop()

    def action(self, frame: np.ndarray) -> tuple[Any | None, bool]:
        if self._error is not None:
            raise RuntimeError("recording dashboard stopped unexpectedly") from self._error
        return self.runner.action(frame)

    def observe_transition(
        self,
        *,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Mapping[str, Any],
        next_frame: np.ndarray,
    ) -> None:
        self.runner.observe_transition(
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
            next_frame=next_frame,
        )

    def close(self) -> None:
        self.runner.stop()
        if self._thread.is_alive():
            self._thread.join(timeout=10.0)


__all__ = [
    "FRAME_CODEC_PNG",
    "FRAME_GAME",
    "FRAME_HEADER",
    "FRAME_MAGIC",
    "FRAME_OBSERVATION",
    "HumanRecordingRunner",
    "PlaybackCommand",
    "PlaybackWebServer",
    "PROTOCOL_VERSION",
    "WebPlaybackRunner",
    "WebHumanController",
    "history_point",
    "run_web_playback",
    "transition_payload",
]
