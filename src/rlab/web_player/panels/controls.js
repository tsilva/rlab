import { createPanel, text } from "./shared.js";

export function mount({ definition, services }) {
  const element = createPanel({
    id: definition.id,
    label: definition.label,
    tag: "aside",
    className: "control-panel transport",
    headerClass: "control-panel-header",
    body: `
      <p data-session-summary class="session-summary" aria-live="polite">Waiting for session…</p>
      <section class="control-section" aria-labelledby="playback-controls-heading">
        <h3 id="playback-controls-heading" class="control-label">Playback</h3>
        <div class="control-grid transport-grid">
          <button data-command="play" data-playback-toggle data-requires-active-episode class="primary icon-only" aria-label="Play" title="Play with the selected driver"><svg class="icon" aria-hidden="true"><use data-playback-icon href="/assets/tabler-icons.svg#ti-player-play"></use></svg></button>
          <button data-command="step" data-requires-active-episode class="icon-only" aria-label="Step once" title="Step once"><svg class="icon" aria-hidden="true"><use href="/assets/tabler-icons.svg#ti-player-skip-forward"></use></svg></button>
          <button data-command="step-ten" data-requires-active-episode class="icon-only" aria-label="Step 10 times" title="Step 10 times"><svg class="icon" aria-hidden="true"><use href="/assets/tabler-icons.svg#ti-player-track-next"></use></svg></button>
          <button data-command="continue-event" data-requires-active-episode class="icon-only" aria-label="Continue to next event" title="Continue to next event"><svg class="icon" aria-hidden="true"><use href="/assets/tabler-icons.svg#ti-activity-heartbeat"></use></svg></button>
          <button data-command="next-episode" data-next-episode class="primary button-with-icon control-wide" aria-label="Play next episode" title="Start the prepared next episode" hidden><svg class="icon" aria-hidden="true"><use href="/assets/tabler-icons.svg#ti-player-play"></use></svg><span>Play next episode</span></button>
        </div>
        <details class="playback-settings">
          <summary>Playback settings</summary>
          <div class="playback-settings-body">
            <div class="playback-fps">
              <label for="playback-fps">Play FPS</label>
              <input id="playback-fps" data-fps type="number" min="0" step="1" value="0" inputmode="decimal" aria-describedby="playback-fps-hint">
              <button data-command="set-fps" class="quiet icon-only" aria-label="Apply play FPS" title="Apply play FPS"><svg class="icon" aria-hidden="true"><use href="/assets/tabler-icons.svg#ti-check"></use></svg></button>
            </div>
            <p id="playback-fps-hint" class="control-hint">0 runs playback uncapped</p>
            <div class="playback-sampling">
              <label for="playback-sampling">Sampling</label>
              <select id="playback-sampling" data-sampling aria-describedby="playback-sampling-hint">
                <option value="stochastic">Stochastic</option>
                <option value="deterministic">Deterministic</option>
              </select>
            </div>
            <p id="playback-sampling-hint" class="control-hint">Playback only · evaluation remains stochastic</p>
          </div>
        </details>
      </section>
      <details class="control-section session-settings">
        <summary>Session settings</summary>
        <div class="session-settings-body">
          <div class="control-field-row">
            <label>Seed <input data-seed inputmode="numeric"></label>
            <button data-command="reset" class="icon-only" aria-label="Reset with this seed" title="Reset with this seed"><svg class="icon" aria-hidden="true"><use href="/assets/tabler-icons.svg#ti-refresh"></use></svg></button>
          </div>
        </div>
      </details>
      <section class="control-section" aria-labelledby="driver-controls-heading">
        <h3 id="driver-controls-heading" class="control-label">Driver</h3>
        <div class="driver-controls">
          <div class="driver-switch" role="group" aria-label="Driver selection">
            <button data-command="policy" data-driver-option="policy" class="quiet button-with-icon driver-option" aria-pressed="true" aria-label="Use policy driver" title="Use policy driver"><svg class="icon" aria-hidden="true"><use href="/assets/tabler-icons.svg#ti-brain"></use></svg><span>Policy</span></button>
            <button data-command="human" data-driver-option="human" class="quiet button-with-icon driver-option" aria-pressed="false" aria-label="Take human control" title="Take human control"><svg class="icon" aria-hidden="true"><use href="/assets/tabler-icons.svg#ti-hand-grab"></use></svg><span>Human</span></button>
          </div>
          <button data-command="inspect" class="quiet icon-only" aria-label="Inspect policy" title="Inspect policy"><svg class="icon" aria-hidden="true"><use href="/assets/tabler-icons.svg#ti-search"></use></svg></button>
        </div>
      </section>
    `,
  });

  const seed = element.querySelector("[data-seed]");
  const fps = element.querySelector("[data-fps]");
  const sampling = element.querySelector("[data-sampling]");
  const playbackToggle = element.querySelector("[data-playback-toggle]");
  const playbackIcon = playbackToggle.querySelector("[data-playback-icon]");
  const nextEpisode = element.querySelector("[data-next-episode]");
  const driverControls = element.querySelector(".driver-controls");
  const driverSwitch = element.querySelector(".driver-switch");
  const driverOptions = [...element.querySelectorAll("[data-driver-option]")];
  const commands = {
    pause: () => services.pauseCurrentPlayback(),
    play: () => services.playFromCurrentPosition(
      services.getState().liveSnapshot?.driver
        || services.getState().snapshot?.driver
        || "policy",
    ),
    step: () => services.command("step", { count: 1 }),
    "step-ten": () => services.command("step", { count: 10 }),
    "continue-event": () => services.command("continue", { target: "any" }),
    "next-episode": () => services.command("next_episode"),
    reset: () => services.command("reset", { seed: seed.value }),
    "set-fps": () => services.command("set_fps", { fps: Number(fps.value) }),
    "set-sampling-mode": () => services.command("set_sampling_mode", { mode: sampling.value }),
    policy: () => services.command("set_driver", { driver: "policy" }),
    human: () => services.command("set_driver", { driver: "human" }),
    inspect: () => services.command("inspect_policy"),
  };
  element.querySelectorAll("[data-command]").forEach((button) => {
    button.addEventListener("click", () => commands[button.dataset.command]());
  });
  fps.addEventListener("keydown", (event) => {
    if (event.key !== "Enter") return;
    event.preventDefault();
    commands["set-fps"]();
  });
  sampling.addEventListener("change", commands["set-sampling-mode"]);

  const updateControl = () => {
    const state = services.getState();
    element.querySelectorAll("button:not([data-panel-menu]):not([data-drag-handle]):not(.panel-resize)")
      .forEach((button) => { button.disabled = !state.hasControl; });
    const session = state.liveSnapshot?.session || state.snapshot?.session || {};
    element.querySelectorAll("[data-requires-active-episode]")
      .forEach((button) => {
        button.disabled = !state.hasControl || Boolean(session.awaiting_next_episode);
      });
    playbackToggle.disabled = !state.hasControl || (
      Boolean(session.awaiting_next_episode)
      && !state.replayingInspection
      && !services.canReplayInspection()
    );
    nextEpisode.disabled = !state.hasControl || !session.can_start_next_episode;
    sampling.disabled = !state.hasControl;
  };

  const renderPlaybackToggle = (runState) => {
    const running = services.getState().replayingInspection
      || ["playing", "stepping", "continuing"].includes(runState);
    const command = running ? "pause" : "play";
    playbackToggle.title = running
      ? "Pause after the current transition"
      : (services.canReplayInspection()
        ? "Replay from the selected step"
        : "Play with the selected driver");
    if (playbackToggle.dataset.command === command) return;
    const label = running ? "Pause" : "Play";
    playbackToggle.dataset.command = command;
    playbackToggle.classList.toggle("primary", !running);
    playbackToggle.setAttribute("aria-label", label);
    playbackIcon.setAttribute("href", `/assets/tabler-icons.svg#ti-player-${command}`);
  };

  return {
    element,
    updateControl,
    render(snapshot, view = {}) {
      if (view.inspection) snapshot = services.getState().liveSnapshot || snapshot;
      if (!snapshot) { updateControl(); return; }
      const session = snapshot.session || {};
      seed.value = text(session.seed, "");
      if (document.activeElement !== fps) fps.value = Number(session.target_fps || 0);
      if (document.activeElement !== sampling) {
        sampling.value = session.sampling_mode || "stochastic";
      }
      element.querySelector("[data-session-summary]").textContent = `${snapshot.run_state.toUpperCase()} · ${snapshot.driver.toUpperCase()}`;
      renderPlaybackToggle(snapshot.run_state);
      driverOptions.forEach((option) => {
        option.setAttribute(
          "aria-pressed",
          String(option.dataset.driverOption === snapshot.driver),
        );
      });
      const recording = snapshot.mode === "recording";
      driverControls.classList.toggle("recording", recording);
      driverSwitch.classList.toggle("single-option", recording);
      nextEpisode.hidden = recording || !session.awaiting_next_episode;
      nextEpisode.title = session.can_start_next_episode
        ? "Start the prepared next episode"
        : "The configured episode limit has been reached";
      fps.min = recording ? "1" : "0";
      element.querySelector("#playback-fps-hint").textContent = recording
        ? "Recording FPS must be at least 1"
        : "0 runs playback uncapped";
      ["step", "step-ten", "continue-event", "reset", "policy", "inspect"].forEach((name) => {
        element.querySelector(`[data-command="${name}"]`).hidden = recording;
      });
      element.querySelector(".session-settings").hidden = recording;
      element.querySelector(".playback-sampling").hidden = recording;
      element.querySelector("#playback-sampling-hint").hidden = recording;
      seed.closest("label").hidden = recording;
      const human = element.querySelector('[data-command="human"]');
      const humanLabel = recording ? "Human controls" : "Take human control";
      human.setAttribute("aria-label", humanLabel);
      human.title = humanLabel;
      updateControl();
    },
  };
}
