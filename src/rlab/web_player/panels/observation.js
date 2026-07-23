import { createPanel } from "./shared.js";

const FRAME_OBSERVATION = 2;

export function mount({ definition }) {
  const element = createPanel({
    id: definition.id,
    label: definition.label,
    className: "observation-panel",
    body: `
      <div class="observation-stage">
        <canvas id="observation-canvas" data-canvas></canvas>
        <div id="observation-empty" data-empty class="empty-state">No image stack is available.</div>
      </div>
      <pre id="model-input" data-input class="compact-pre">No policy input yet.</pre>
    `,
  });
  const canvas = element.querySelector("[data-canvas]");
  const empty = element.querySelector("[data-empty]");

  return {
    element,
    render(snapshot) {
      element.querySelector("[data-input]").textContent = snapshot?.transition?.before?.model_input?.join("\n")
        || "No policy input yet.";
    },
    async renderFrame(kind, blob) {
      if (kind !== FRAME_OBSERVATION || !blob) return false;
      const bitmap = await createImageBitmap(blob);
      canvas.width = bitmap.width;
      canvas.height = bitmap.height;
      const context = canvas.getContext("2d", { alpha: false });
      context.imageSmoothingEnabled = false;
      context.drawImage(bitmap, 0, 0);
      bitmap.close();
      empty.hidden = true;
      return true;
    },
  };
}
