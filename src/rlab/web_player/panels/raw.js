import { createPanel, renderJson } from "./shared.js";

export function mount({ definition }) {
  const element = createPanel({
    id: definition.id,
    label: definition.label,
    className: "raw-panel",
    body: `
      <details open><summary>Current transition</summary><pre data-transition class="json-view">No transition yet.</pre></details>
      <details><summary>Resolved playback configuration</summary><pre data-config>Waiting…</pre></details>
    `,
  });

  return {
    element,
    render(snapshot) {
      renderJson(
        element.querySelector("[data-transition]"),
        snapshot?.transition,
        "No transition yet.",
      );
      element.querySelector("[data-config]").textContent = snapshot?.session?.config
        || "No configuration supplied.";
    },
  };
}
