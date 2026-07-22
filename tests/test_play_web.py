from __future__ import annotations

import argparse
import asyncio
import io
import time
from pathlib import Path

import numpy as np
from aiohttp import ClientSession, WSServerHandshakeError, WSMsgType
from PIL import Image

from rlab.dataset_cli import build_parser as build_dataset_parser
from rlab.play import _PlaybackSession, _PlaybackTransition
from rlab.play_web import (
    FRAME_CODEC_PNG,
    FRAME_GAME,
    FRAME_HEADER,
    FRAME_MAGIC,
    FrameEncoder,
    HumanRecordingRunner,
    PlaybackCommand,
    PlaybackWebServer,
    transition_payload,
)


class FakeHumanSession:
    fps = 60.0

    def action_from_labels(self, labels):
        return tuple(sorted(labels))


def human_args(**overrides):
    values = {
        "fps": 240.0,
        "episodes": 1,
        "port": 0,
        "no_open": True,
        "debug": True,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_human_dataset_recording_defaults_to_web_dashboard() -> None:
    args = build_dataset_parser().parse_args(
        ["record", "local-session", "--env-id", "Game-v0"]
    )

    assert args.agent == "human"
    assert args.ui == "web"
    assert args.port == 0
    assert args.no_open is False


def test_browser_button_chords_map_to_declared_discrete_actions() -> None:
    session = argparse.Namespace(action_names=("noop", "right", "right_a", "left"))

    assert _PlaybackSession.manual_action(session, []) == 0
    assert _PlaybackSession.manual_action(session, ["RIGHT", "a"]) == 2


def test_frame_encoder_emits_versioned_latest_only_png_packet() -> None:
    encoder = FrameEncoder()
    encoder.start()
    try:
        encoder.submit(FRAME_GAME, 7, np.full((3, 4, 3), 91, dtype=np.uint8))
        deadline = time.monotonic() + 2.0
        while FRAME_GAME not in encoder.latest() and time.monotonic() < deadline:
            time.sleep(0.005)
        sequence, packet = encoder.latest()[FRAME_GAME]
    finally:
        encoder.close()

    magic, kind, codec, flags, header_sequence = FRAME_HEADER.unpack(
        packet[: FRAME_HEADER.size]
    )
    image = Image.open(io.BytesIO(packet[FRAME_HEADER.size :]))
    assert (magic, kind, codec, flags, header_sequence, sequence) == (
        FRAME_MAGIC,
        FRAME_GAME,
        FRAME_CODEC_PNG,
        0,
        7,
        7,
    )
    assert image.size == (4, 3)


def test_transition_payload_keeps_before_decision_after_alignment() -> None:
    transition = _PlaybackTransition(
        sequence=3,
        episode=1,
        step=3,
        seed=40_000,
        start_id="Level1-1",
        model_obs=np.zeros((1, 4, 84, 84), dtype=np.uint8),
        decision=None,
        action_source="human",
        executed_action=2,
        diagnostics=None,
        info={"x_pos": 12, "credential_token": "do-not-stream"},
        before_frame=np.zeros((2, 2, 3), dtype=np.uint8),
        after_frame=np.ones((2, 2, 3), dtype=np.uint8),
        before_frames=(np.zeros((2, 2, 1), dtype=np.uint8),),
        after_frames=(np.ones((2, 2, 1), dtype=np.uint8),),
        attribution=None,
        pre_task="Level1-1",
        next_task="Level1-1",
        reward=1.5,
        total_reward=4.0,
        max_x_pos=12,
        terminated=False,
        truncated=False,
        completed=False,
        boundary=False,
    )

    payload = transition_payload(transition)

    assert payload["sequence"] == 3
    assert payload["before"]["task"] == "Level1-1"
    assert payload["decision"] is None
    assert payload["after"]["task"] == "Level1-1"
    assert payload["signals"]["x_pos"] == 12.0
    assert payload["info"]["credential_token"] == "<redacted>"


def test_human_recording_runner_requires_fresh_focus_and_streams_transition_stats() -> None:
    runner = HumanRecordingRunner(FakeHumanSession(), human_args())
    frame = np.zeros((4, 5, 3), dtype=np.uint8)
    runner.start()
    try:
        runner.submit(PlaybackCommand("play", "client", "play", {"driver": "human"}, None))
        runner.update_input(["right", "a"], focused=True)
        action, keep_recording = runner.action(frame)
        runner.observe_transition(
            reward=2.5,
            terminated=False,
            truncated=False,
            info={"x_pos": 11},
            next_frame=np.ones_like(frame),
        )
        snapshot = runner.snapshot()
    finally:
        runner.stop()

    assert keep_recording
    assert action == ("A", "RIGHT")
    assert snapshot["mode"] == "recording"
    assert snapshot["interactive"] is True
    assert snapshot["transition"]["reward"]["return"] == 2.5
    assert snapshot["transition"]["signals"] == {"x_pos": 11.0}
    assert runner.history_payload()["points"][0]["action_source"] == "human"


def test_loopback_server_requires_exact_origin_and_fragment_token() -> None:
    async def scenario() -> None:
        runner = HumanRecordingRunner(FakeHumanSession(), human_args())
        server = PlaybackWebServer(runner, human_args())
        task = asyncio.create_task(server.run())
        try:
            deadline = asyncio.get_running_loop().time() + 3.0
            while not server.origin and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(0.01)
            assert server.origin.startswith("http://127.0.0.1:")
            runner.encoder.submit(FRAME_GAME, 0, np.zeros((2, 3, 3), dtype=np.uint8))
            while FRAME_GAME not in runner.encoder.latest():
                await asyncio.sleep(0.005)
            async with ClientSession() as client:
                response = await client.get(server.origin)
                assert response.status == 200
                assert response.headers["Cache-Control"] == "no-store"
                assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
                try:
                    await client.ws_connect(f"{server.origin}/ws", origin="http://example.test")
                except WSServerHandshakeError as exc:
                    assert exc.status == 403
                else:
                    raise AssertionError("cross-origin websocket unexpectedly connected")

                socket = await client.ws_connect(f"{server.origin}/ws", origin=server.origin)
                await socket.send_json(
                    {
                        "type": "hello",
                        "token": server.token,
                        "subscriptions": ["telemetry", "game"],
                    }
                )
                received_types = set()
                received_frame = False
                for _ in range(6):
                    message = await asyncio.wait_for(socket.receive(), timeout=2.0)
                    if message.type == WSMsgType.TEXT:
                        received_types.add(message.json()["type"])
                    elif message.type == WSMsgType.BINARY:
                        received_frame = True
                    if {"welcome", "snapshot"}.issubset(received_types) and received_frame:
                        break
                assert {"welcome", "snapshot"}.issubset(received_types)
                assert received_frame

                observer = await client.ws_connect(f"{server.origin}/ws", origin=server.origin)
                await observer.send_json(
                    {
                        "type": "hello",
                        "token": server.token,
                        "subscriptions": ["telemetry"],
                    }
                )
                observer_snapshot = None
                for _ in range(4):
                    message = await asyncio.wait_for(observer.receive(), timeout=2.0)
                    if message.type == WSMsgType.TEXT and message.json()["type"] == "snapshot":
                        observer_snapshot = message.json()
                        break
                assert observer_snapshot is not None
                assert observer_snapshot["control"]["has_control"] is False

                await observer.send_json({"type": "acquire_control"})
                acquired_snapshot = None
                for _ in range(3):
                    message = await asyncio.wait_for(observer.receive(), timeout=2.0)
                    if message.type == WSMsgType.TEXT and message.json()["type"] == "snapshot":
                        acquired_snapshot = message.json()
                        break
                assert acquired_snapshot is not None
                assert acquired_snapshot["control"]["has_control"] is True
                assert acquired_snapshot["control_epoch"] > observer_snapshot["control_epoch"]

                await observer.send_json(
                    {"type": "command", "id": "stop", "name": "stop", "payload": {}}
                )
                await observer.close()
                await socket.close()
            await asyncio.wait_for(task, timeout=3.0)
        finally:
            runner.stop()
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


def test_web_dashboard_assets_are_packaged_beside_server() -> None:
    root = Path(__file__).parents[1] / "src" / "rlab" / "web_player"
    assert (root / "index.html").is_file()
    assert (root / "styles.css").is_file()
    markup = (root / "index.html").read_text(encoding="utf-8")
    styles = (root / "styles.css").read_text(encoding="utf-8")
    script = (root / "app.js").read_text(encoding="utf-8")
    assert markup.index('data-panel="game"') < markup.index('data-panel="controls"')
    assert '<aside class="panel control-panel transport"' in markup
    assert 'height: calc(100dvh - 4.3rem)' in styles
    assert "object-fit: contain" in styles
    assert "requestFullscreen" in script
    assert "/panel/" in script
    assert "visibilitychange" in script
