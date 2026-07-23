from __future__ import annotations

import argparse
import asyncio
import io
import time
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

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
    WebPlaybackRunner,
    _session_environment_id,
    run_web_playback,
    transition_payload,
)


class FakeHumanSession:
    fps = 60.0
    environment_id = "Game-v0"

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


def test_playback_environment_title_uses_configured_env_id() -> None:
    session = argparse.Namespace(
        config={"game": "BreakoutTurbo-v0"},
        environment_id="recording-fallback",
    )

    assert _session_environment_id(session, human_args()) == "BreakoutTurbo-v0"


def test_web_playback_sampling_mode_selects_policy_action_path() -> None:
    transition = argparse.Namespace(boundary=False, events=())
    session = argparse.Namespace(
        config={"game": "Game-v0"},
        last_transition=None,
        step=Mock(return_value=transition),
    )
    runner = WebPlaybackRunner(session, human_args(), config_text="")
    runner._publish = Mock()

    runner._apply(
        PlaybackCommand(
            "sampling",
            "client",
            "set_sampling_mode",
            {"mode": "deterministic"},
            None,
        )
    )
    runner.run_state = "playing"
    runner._step_once()

    assert runner.sampling_mode == "deterministic"
    session.step.assert_called_once_with(deterministic=True)


def test_web_playback_requires_explicit_command_after_episode_boundary() -> None:
    transition = argparse.Namespace(boundary=True, events=(), episode=1)
    session = argparse.Namespace(
        config={"game": "Game-v0"},
        episode=2,
        last_transition=transition,
        step=Mock(return_value=transition),
    )
    runner = WebPlaybackRunner(session, human_args(episodes=0), config_text="")
    runner._publish = Mock()
    runner.run_state = "playing"

    runner._step_once()

    assert runner.run_state == "paused"
    assert runner.awaiting_next_episode is True
    assert runner._can_start_next_episode() is True
    assert runner.remaining_steps == 0

    runner._apply(PlaybackCommand("play", "client", "play", {}, None))
    blocked = runner.responses.get_nowait().payload
    assert blocked["ok"] is False
    assert blocked["error"] == "episode complete; choose Play next episode"
    assert runner.awaiting_next_episode is True

    runner._apply(
        PlaybackCommand("next", "client", "next_episode", {}, runner.revision)
    )
    accepted = runner.responses.get_nowait().payload
    assert accepted["ok"] is True
    assert runner.awaiting_next_episode is False
    assert runner.run_state == "playing"


def test_web_playback_episode_limit_disables_next_episode() -> None:
    transition = argparse.Namespace(boundary=True, events=(), episode=1)
    session = argparse.Namespace(
        config={"game": "Game-v0"},
        episode=2,
        last_transition=transition,
        step=Mock(return_value=transition),
    )
    runner = WebPlaybackRunner(session, human_args(episodes=1), config_text="")
    runner._publish = Mock()
    runner.run_state = "playing"

    runner._step_once()

    assert runner.awaiting_next_episode is True
    assert runner._can_start_next_episode() is False
    assert runner._status_message == "episode limit reached (1)"


def test_human_dataset_recording_defaults_to_web_dashboard() -> None:
    args = build_dataset_parser().parse_args(
        ["record", "local-session", "--env-id", "Game-v0"]
    )

    assert args.agent == "human"
    assert args.ui == "web"
    assert args.port == 0
    assert args.no_open is False


def test_run_web_playback_requests_paired_browser_windows() -> None:
    args = human_args()
    runner = object()
    server = AsyncMock()
    server.run.return_value = 0
    with (
        patch("rlab.play_web.WebPlaybackRunner", return_value=runner),
        patch("rlab.play_web.PlaybackWebServer", return_value=server) as server_type,
    ):
        assert run_web_playback(object(), args, config_text="config") == 0

    server_type.assert_called_once_with(runner, args, paired_windows=True)


def test_paired_playback_server_opens_play_and_stats_windows() -> None:
    async def scenario() -> None:
        runner = HumanRecordingRunner(FakeHumanSession(), human_args())
        server = PlaybackWebServer(
            runner,
            human_args(no_open=False),
            paired_windows=True,
        )
        with patch("rlab.play_web.webbrowser.open") as open_browser:
            task = asyncio.create_task(server.run())
            try:
                deadline = asyncio.get_running_loop().time() + 3.0
                while (
                    (not server.origin or open_browser.call_count < 2)
                    and asyncio.get_running_loop().time() < deadline
                ):
                    await asyncio.sleep(0.01)
                urls = server.dashboard_urls()
                assert urls == (
                    f"{server.origin}/?workspace=paired#token={server.token}",
                    f"{server.origin}/workspace/stats?workspace=paired#token={server.token}",
                )
                assert [call.args[0] for call in open_browser.call_args_list] == list(urls)
                assert all(call.kwargs == {"new": 1, "autoraise": True} for call in open_browser.call_args_list)
            finally:
                runner.stop()
                await asyncio.wait_for(task, timeout=3.0)

    asyncio.run(scenario())


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
    assert snapshot["session"]["env_id"] == "Game-v0"
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
                icon_response = await client.get(f"{server.origin}/assets/tabler-icons.svg")
                assert icon_response.status == 200
                assert "image/svg+xml" in icon_response.headers["Content-Type"]
                assert 'id="ti-player-play"' in await icon_response.text()
                panel_response = await client.get(
                    f"{server.origin}/assets/panels/catalog.js"
                )
                assert panel_response.status == 200
                assert "javascript" in panel_response.headers["Content-Type"]
                assert "PANEL_CATALOG" in await panel_response.text()
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

                await observer.send_json(
                    {"type": "subscribe", "subscriptions": ["telemetry", "game"]}
                )
                observer_received_latest_frame = False
                for _ in range(4):
                    message = await asyncio.wait_for(observer.receive(), timeout=2.0)
                    if message.type == WSMsgType.BINARY:
                        observer_received_latest_frame = True
                        break
                assert observer_received_latest_frame

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

                sibling = await client.ws_connect(f"{server.origin}/ws", origin=server.origin)
                await sibling.send_json(
                    {
                        "type": "hello",
                        "token": server.token,
                        "subscriptions": ["telemetry"],
                        "workspace_id": acquired_snapshot["control"]["workspace_id"],
                        "window_id": "analysis-window",
                    }
                )
                sibling_snapshot = None
                for _ in range(4):
                    message = await asyncio.wait_for(sibling.receive(), timeout=2.0)
                    if message.type == WSMsgType.TEXT and message.json()["type"] == "snapshot":
                        sibling_snapshot = message.json()
                        break
                assert sibling_snapshot is not None
                assert sibling_snapshot["control"]["has_control"] is True
                assert sibling_snapshot["control"]["window_id"] == "analysis-window"
                await sibling.close()

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
    assert (root / "tabler-icons.svg").is_file()
    panel_root = root / "panels"
    panel_names = {
        "game",
        "controls",
        "policy",
        "reward",
        "actions",
        "observation",
        "signals",
        "events",
        "raw",
    }
    assert all((panel_root / f"{name}.js").is_file() for name in panel_names)
    markup = (root / "index.html").read_text(encoding="utf-8")
    styles = (root / "styles.css").read_text(encoding="utf-8")
    script = (root / "app.js").read_text(encoding="utf-8")
    icons = (root / "tabler-icons.svg").read_text(encoding="utf-8")
    catalog = (panel_root / "catalog.js").read_text(encoding="utf-8")
    runtime = (panel_root / "runtime.js").read_text(encoding="utf-8")
    shared = (panel_root / "shared.js").read_text(encoding="utf-8")
    game_markup = (panel_root / "game.js").read_text(encoding="utf-8")
    controls_markup = (panel_root / "controls.js").read_text(encoding="utf-8")
    policy_markup = (panel_root / "policy.js").read_text(encoding="utf-8")
    reward_markup = (panel_root / "reward.js").read_text(encoding="utf-8")
    signals_markup = (panel_root / "signals.js").read_text(encoding="utf-8")
    raw_markup = (panel_root / "raw.js").read_text(encoding="utf-8")
    assert '<main id="dashboard" class="dashboard"></main>' in markup
    assert '<h1 id="page-title">Environment</h1>' in markup
    assert 'value="Default layout"' in markup
    assert "Mario debug" not in markup
    assert 'data-panel="' not in markup
    assert 'className: "control-panel transport"' in controls_markup
    assert "ENVIRONMENT" not in game_markup
    assert "Focus the game for human input" not in game_markup
    assert 'class="game-actions panel-actions"' in game_markup
    assert "aspect-ratio: 256 / 240" in styles
    assert "grid-template-columns: repeat(12" in styles
    assert ".icon-only" in styles
    assert 'id="timeline-scrubber"' in markup
    assert "data-return-chart" in reward_markup
    assert controls_markup.count("data-playback-toggle data-requires-active-episode class=") == 1
    assert 'data-command="play" data-playback-toggle data-requires-active-episode class="primary icon-only"' in controls_markup
    assert 'data-command="pause" class="icon-only"' not in controls_markup
    assert "services.getState().liveSnapshot?.driver" in controls_markup
    assert "if (view.inspection) snapshot = services.getState().liveSnapshot || snapshot;" in controls_markup
    assert 'playbackToggle.dataset.command = command' in controls_markup
    assert "if (playbackToggle.dataset.command === command) return;" in controls_markup
    assert 'playbackIcon.setAttribute("href", `/assets/tabler-icons.svg#ti-player-${command}`)' in controls_markup
    assert "repeat(5, minmax(0, 1fr))" in styles
    assert 'data-command="step-ten" data-requires-active-episode class="icon-only" aria-label="Step 10 times"' in controls_markup
    assert 'data-command="next-episode" data-next-episode' in controls_markup
    assert 'services.command("next_episode")' in controls_markup
    assert "Boolean(session.awaiting_next_episode)" in controls_markup
    assert "nextEpisode.disabled = !state.hasControl || !session.can_start_next_episode" in controls_markup
    assert '<label for="playback-fps">Play FPS</label>' in controls_markup
    assert 'id="playback-fps" data-fps type="number" min="0"' in controls_markup
    assert 'data-command="set-fps"' in controls_markup
    assert '<select id="playback-sampling" data-sampling' in controls_markup
    assert '<option value="stochastic">Stochastic</option>' in controls_markup
    assert '<option value="deterministic">Deterministic</option>' in controls_markup
    assert 'services.command("set_sampling_mode", { mode: sampling.value })' in controls_markup
    assert 'fps.addEventListener("keydown"' in controls_markup
    assert 'commands["set-fps"]();' in controls_markup
    assert 'element.querySelector(".session-settings").hidden = recording' in controls_markup
    assert ".playback-fps" in styles
    assert ".playback-sampling" in styles
    assert 'id="layouts-toggle" class="quiet icon-only"' in markup
    assert 'id="save-layout" class="primary button-with-icon" type="button" title="Save layout"' in markup
    assert 'id="reset-layout" class="quiet button-with-icon" type="button" title="Reset default layout"' in markup
    assert 'id="panel-hide" class="button-with-icon" type="button" title="Hide panel"' in markup
    assert "ti-device-desktop-share" in icons
    assert "separate scale" not in markup
    assert "shared scale" not in markup
    assert "Research workspace" not in markup
    assert "panel-kicker" not in markup
    assert 'id="workspace-sequence"' not in markup
    assert '<details class="control-section session-settings">' in controls_markup
    assert '#workspace-sequence' not in script
    assert "panel-shelf-title" not in script
    assert "scrollIntoView" not in script
    assert '${snapshot.run_state.toUpperCase()} · ${snapshot.driver.toUpperCase()}' in controls_markup
    assert "drawLines(returnChart" in reward_markup
    assert "data-drag-handle" in game_markup
    assert 'aria-label="Move game panel"' in game_markup
    assert "/assets/tabler-icons.svg#ti-player-play" in controls_markup
    assert 'id="ti-grip-vertical"' in icons
    assert 'id="ti-player-play"' in icons
    assert 'id="panel-shelf" class="floating-menu panel-shelf" hidden' in markup
    assert "requestFullscreen" in game_markup
    assert 'data-transition class="json-view"' in raw_markup
    assert "function renderJson(" in shared
    assert "function niceTickStep(" in shared
    assert "function lineChartScale(" in shared
    assert "function formatAxisValue(" in shared
    assert "context.fillText(labels[index], plot.left - 6, y)" in shared
    assert "const scale = lineChartScale([0, max])" in shared
    assert 'class="signal-toolbar-label">Chart signal</span>' in signals_markup
    assert "grid-template-columns: max-content minmax(0, 1fr)" in styles
    assert ".signal-toolbar select" in styles
    assert "text-overflow: ellipsis" in styles
    for token_class in (
        "json-key",
        "json-string",
        "json-number",
        "json-boolean",
        "json-null",
    ):
        assert f".{token_class}" in styles
    assert "/panel/" in script
    assert "/workspace/" in script
    assert "workspace_id" in script
    assert "BroadcastChannel" in script
    assert 'type: "panel-drag-start"' in script
    assert 'type: "panel-drag-move"' in script
    assert 'type: "panel-drag-target"' in script
    assert 'type: "panel-drag-end"' in script
    assert "setPointerCapture" in script
    assert "clientPointFromScreen" in script
    assert "preview.style.width" in script
    assert "preview.style.height" in script
    assert ".panel-drag-overlay" in styles
    assert ".dashboard.drag-receiving" in styles
    assert "visibilitychange" in game_markup
    assert "PANEL_CATALOG" in catalog
    assert "const PAIRED_PANEL_LAYOUT" in catalog
    assert 'window: "stats"' in catalog
    assert "defaultPanelLayout({ paired = false } = {})" in catalog
    assert 'module: "./game.js"' in catalog
    assert "defaultPanelLayout" in catalog
    assert "import(definition.module)" in runtime
    assert "async ensureMounted" in runtime
    assert "this.unmount(name)" in runtime
    assert "new PanelRuntime" in script
    assert "load.title = `Load layout ${name}`" in script
    assert "remove.title = `Delete layout ${name}`" in script
    assert 'button.title = button.getAttribute("aria-label")' in script
    assert 'handle.title = handle.getAttribute("aria-label")' in script
    assert 'id="sampling-status" class="badge muted" hidden' in markup
    assert 'samplingMode === "deterministic" ? "Deterministic" : "Stochastic"' in script
    assert 'decision.sampled ? "Stochastic" : "Deterministic"' in policy_markup
    assert 'panelRuntime.invoke("controls", "render", snapshot)' in script
    assert 'name: "Default layout"' in script
    assert "state.liveSnapshot?.session?.env_id" in script
    assert "Mario debug" not in script
    assert 'new URLSearchParams(location.search).get("workspace") === "paired"' in script
    assert '"rlab.player.workspace.layout.v2"' in script
    assert "pairedWorkspace && closedWindow === STATS_WINDOW_ID" in script
    assert "body.stats-window #timeline" in styles
