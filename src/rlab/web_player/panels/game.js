const FRAME_GAME = 1;

export function mount({ definition, services }) {
  const element = document.createElement("section");
  element.className = "panel game-panel";
  element.dataset.panel = definition.id;
  element.innerHTML = `
    <div id="game-stage" class="game-stage">
      <div id="game-frame" class="game-frame">
        <canvas id="game-canvas" tabindex="0" aria-label="Live game frame. Focus it for human controls: arrows move, Z is B, X is A, Enter is Start, and Shift is Select."></canvas>
      </div>
      <div id="game-empty" class="game-empty empty-state">This environment has no RGB renderer.</div>
      <div class="game-actions panel-actions">
        <button data-drag-handle class="icon-button icon-only panel-drag" type="button" aria-label="Move game panel" title="Move game panel"><svg class="icon" aria-hidden="true"><use href="/assets/tabler-icons.svg#ti-grip-vertical"></use></svg></button>
        <button data-fullscreen class="icon-button icon-only" type="button" aria-label="Fullscreen game" title="Fullscreen game"><svg class="icon" aria-hidden="true"><use href="/assets/tabler-icons.svg#ti-maximize"></use></svg></button>
        <button data-panel-menu="game" class="icon-button icon-only" type="button" aria-label="Game panel options" title="Game panel options"><svg class="icon" aria-hidden="true"><use href="/assets/tabler-icons.svg#ti-dots-vertical"></use></svg></button>
      </div>
    </div>
  `;

  const stage = element.querySelector(".game-stage");
  const frame = element.querySelector(".game-frame");
  const canvas = element.querySelector("canvas");
  const empty = element.querySelector(".game-empty");
  const pressed = new Set();
  let focused = false;
  let aspect = 256 / 240;
  const mapping = new Map([
    ["ArrowUp", "up"], ["ArrowDown", "down"], ["ArrowLeft", "left"], ["ArrowRight", "right"],
    ["z", "b"], ["Z", "b"], ["x", "a"], ["X", "a"], ["Enter", "start"], ["Shift", "select"],
  ]);

  const publish = (hasFocus = focused) => services.send({
    type: "input",
    pressed: [...pressed],
    focused: hasFocus,
  });
  const fit = () => {
    const width = stage.clientWidth;
    const height = stage.clientHeight;
    if (!width || !height) return;
    const fittedWidth = Math.min(width, height * aspect);
    const fittedHeight = fittedWidth / aspect;
    frame.style.width = `${Math.max(1, Math.floor(fittedWidth))}px`;
    frame.style.height = `${Math.max(1, Math.floor(fittedHeight))}px`;
    frame.style.aspectRatio = String(aspect);
  };
  const loseFocus = () => {
    if (!focused && !pressed.size) return;
    focused = false;
    pressed.clear();
    publish(false);
  };
  const visibility = () => { if (document.hidden) loseFocus(); };

  canvas.addEventListener("focus", () => { focused = true; publish(true); });
  canvas.addEventListener("blur", loseFocus);
  canvas.addEventListener("keydown", (event) => {
    const label = mapping.get(event.key);
    if (!label) return;
    event.preventDefault();
    pressed.add(label);
    publish(true);
  });
  canvas.addEventListener("keyup", (event) => {
    const label = mapping.get(event.key);
    if (!label) return;
    event.preventDefault();
    pressed.delete(label);
    publish(true);
  });
  document.addEventListener("visibilitychange", visibility);
  element.querySelector("[data-fullscreen]").addEventListener("click", () => {
    stage.requestFullscreen({ navigationUI: "hide" }).catch((error) => services.showToast(error.message, true));
  });
  const keepalive = setInterval(() => {
    const state = services.getState();
    if (focused && state.hasControl && state.snapshot?.driver === "human") publish(true);
  }, 50);

  return {
    element,
    async renderFrame(kind, blob) {
      if (kind !== FRAME_GAME || !blob) return false;
      const bitmap = await createImageBitmap(blob);
      aspect = bitmap.width / Math.max(1, bitmap.height);
      fit();
      canvas.width = bitmap.width;
      canvas.height = bitmap.height;
      const context = canvas.getContext("2d", { alpha: false });
      context.imageSmoothingEnabled = false;
      context.drawImage(bitmap, 0, 0);
      bitmap.close();
      empty.hidden = true;
      return true;
    },
    resize: fit,
    destroy() {
      loseFocus();
      clearInterval(keepalive);
      document.removeEventListener("visibilitychange", visibility);
    },
  };
}
