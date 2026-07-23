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
          <button data-command="play" data-playback-toggle class="primary icon-only" aria-label="Play" title="Play with the selected driver"><svg class="icon" aria-hidden="true"><use data-playback-icon href="/assets/tabler-icons.svg#ti-player-play"></use></svg></button>
          <button data-command="step" class="icon-only" aria-label="Step once" title="Step once"><svg class="icon" aria-hidden="true"><use href="/assets/tabler-icons.svg#ti-player-skip-forward"></use></svg></button>
          <button data-command="step-ten" class="icon-only" aria-label="Step 10 times" title="Step 10 times"><svg class="icon" aria-hidden="true"><use href="/assets/tabler-icons.svg#ti-player-track-next"></use></svg></button>
          <button data-command="continue-event" class="icon-only" aria-label="Continue to next event" title="Continue to next event"><svg class="icon" aria-hidden="true"><use href="/assets/tabler-icons.svg#ti-activity-heartbeat"></use></svg></button>
          <button data-command="continue-done" class="icon-only" aria-label="Continue to episode boundary" title="Continue to episode boundary"><svg class="icon" aria-hidden="true"><use href="/assets/tabler-icons.svg#ti-flag-3"></use></svg></button>
        </div>
      </section>
      <details class="control-section session-settings">
        <summary>Session settings</summary>
        <div class="session-settings-body">
          <div class="control-field-row">
            <label>Seed <input data-seed inputmode="numeric"></label>
            <button data-command="reset" class="icon-only" aria-label="Reset with this seed" title="Reset with this seed"><svg class="icon" aria-hidden="true"><use href="/assets/tabler-icons.svg#ti-refresh"></use></svg></button>
          </div>
          <div class="control-field-row">
            <label>FPS <input data-fps type="number" min="0" step="1" value="0"></label>
            <button data-command="set-fps" class="quiet icon-only" aria-label="Apply FPS limit" title="Apply FPS limit"><svg class="icon" aria-hidden="true"><use href="/assets/tabler-icons.svg#ti-check"></use></svg></button>
          </div>
        </div>
      </details>
      <section class="control-section" aria-labelledby="driver-controls-heading">
        <h3 id="driver-controls-heading" class="control-label">Driver</h3>
        <div class="control-grid driver-grid">
          <button data-command="policy" class="quiet icon-only" aria-label="Use policy driver" title="Use policy driver"><svg class="icon" aria-hidden="true"><use href="/assets/tabler-icons.svg#ti-brain"></use></svg></button>
          <button data-command="human" class="quiet icon-only" aria-label="Take human control" title="Take human control"><svg class="icon" aria-hidden="true"><use href="/assets/tabler-icons.svg#ti-hand-grab"></use></svg></button>
          <button data-command="inspect" class="quiet icon-only" aria-label="Inspect policy" title="Inspect policy"><svg class="icon" aria-hidden="true"><use href="/assets/tabler-icons.svg#ti-search"></use></svg></button>
          <button data-acquire class="quiet icon-only" aria-label="Control from this window" title="Control from this window"><svg class="icon" aria-hidden="true"><use href="/assets/tabler-icons.svg#ti-device-desktop-share"></use></svg></button>
        </div>
      </section>
      <button data-command="stop" class="danger quiet icon-only control-end" aria-label="End session" title="End session"><svg class="icon" aria-hidden="true"><use href="/assets/tabler-icons.svg#ti-power"></use></svg></button>
    `,
  });

  const seed = element.querySelector("[data-seed]");
  const fps = element.querySelector("[data-fps]");
  const acquire = element.querySelector("[data-acquire]");
  const playbackToggle = element.querySelector("[data-playback-toggle]");
  const playbackIcon = playbackToggle.querySelector("[data-playback-icon]");
  const commands = {
    pause: () => services.command("pause"),
    play: () => services.command("play", { driver: services.getState().snapshot?.driver || "policy" }),
    step: () => services.command("step", { count: 1 }),
    "step-ten": () => services.command("step", { count: 10 }),
    "continue-event": () => services.command("continue", { target: "any" }),
    "continue-done": () => services.command("continue", { target: "done" }),
    reset: () => services.command("reset", { seed: seed.value }),
    "set-fps": () => services.command("set_fps", { fps: Number(fps.value) }),
    policy: () => services.command("set_driver", { driver: "policy" }),
    human: () => services.command("set_driver", { driver: "human" }),
    inspect: () => services.command("inspect_policy"),
    stop: () => services.command("stop"),
  };
  element.querySelectorAll("[data-command]").forEach((button) => {
    button.addEventListener("click", () => commands[button.dataset.command]());
  });
  acquire.addEventListener("click", () => services.send({ type: "acquire_control" }));

  const updateControl = () => {
    const state = services.getState();
    element.querySelectorAll("button:not([data-acquire]):not([data-panel-menu]):not([data-drag-handle]):not(.panel-resize)")
      .forEach((button) => { button.disabled = !state.hasControl; });
    acquire.disabled = !state.connected || state.hasControl;
  };

  const renderPlaybackToggle = (runState) => {
    const running = ["playing", "stepping", "continuing"].includes(runState);
    const command = running ? "pause" : "play";
    const label = running ? "Pause" : "Play";
    playbackToggle.dataset.command = command;
    playbackToggle.classList.toggle("primary", !running);
    playbackToggle.setAttribute("aria-label", label);
    playbackToggle.title = running
      ? "Pause after the current transition"
      : "Play with the selected driver";
    playbackIcon.setAttribute("href", `/assets/tabler-icons.svg#ti-player-${command}`);
  };

  return {
    element,
    updateControl,
    render(snapshot) {
      if (!snapshot) { updateControl(); return; }
      const session = snapshot.session || {};
      seed.value = text(session.seed, "");
      if (document.activeElement !== fps) fps.value = Number(session.target_fps || 0);
      element.querySelector("[data-session-summary]").textContent = `${snapshot.run_state.toUpperCase()} · ${snapshot.driver.toUpperCase()}`;
      renderPlaybackToggle(snapshot.run_state);
      const recording = snapshot.mode === "recording";
      ["step", "step-ten", "continue-event", "continue-done", "reset", "policy", "inspect"].forEach((name) => {
        element.querySelector(`[data-command="${name}"]`).hidden = recording;
      });
      seed.closest("label").hidden = recording;
      const human = element.querySelector('[data-command="human"]');
      const humanLabel = recording ? "Human controls" : "Take human control";
      human.setAttribute("aria-label", humanLabel);
      human.title = humanLabel;
      updateControl();
    },
  };
}
