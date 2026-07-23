import { createPanel, drawHistogram } from "./shared.js";

export function mount({ definition }) {
  const element = createPanel({
    id: definition.id,
    label: "Action histogram",
    body: `
      <canvas data-chart class="chart" aria-label="Action histogram"></canvas>
      <p data-caption class="panel-foot">No policy actions observed.</p>
    `,
  });
  const canvas = element.querySelector("[data-chart]");
  const caption = element.querySelector("[data-caption]");
  let history = [];
  let snapshot = null;

  const render = () => {
    const names = snapshot?.session?.action_names || [];
    const counts = Array.from({ length: names.length || 1 }, () => 0);
    history.slice(-1024).forEach((point) => {
      if (Number.isInteger(point.action) && point.action >= 0) {
        counts[point.action] = (counts[point.action] || 0) + 1;
      }
    });
    drawHistogram(canvas, counts, names);
    const total = counts.reduce((sum, value) => sum + value, 0);
    caption.textContent = total
      ? `${total} sampled policy actions in the visible history.`
      : "No policy actions observed.";
  };

  return {
    element,
    render(next) { snapshot = next; render(); },
    renderHistory(next, current) { history = next; snapshot = current; render(); },
    resize: render,
  };
}
