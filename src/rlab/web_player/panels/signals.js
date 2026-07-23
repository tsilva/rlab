import { createPanel, drawLines, number } from "./shared.js";

const SIGNAL_KEY = "rlab.player.signal";

export function mount({ definition }) {
  const element = createPanel({
    id: definition.id,
    label: definition.label,
    body: `
      <div class="signal-toolbar">
        <label>Chart signal <select data-select><option value="">Choose a signal</option></select></label>
      </div>
      <canvas data-chart class="chart" aria-label="Selected signal history"></canvas>
      <div class="table-scroll"><table><tbody data-body></tbody></table></div>
    `,
  });
  const select = element.querySelector("[data-select]");
  const chart = element.querySelector("[data-chart]");
  const body = element.querySelector("[data-body]");
  let history = [];
  let selected = localStorage.getItem(SIGNAL_KEY) || "";

  const renderChart = () => {
    const name = select.value;
    drawLines(chart, [{
      values: history.slice(-1024).map((point) => Number(point.signals?.[name])),
      color: "#f0c36a",
    }]);
  };
  const renderSignals = (signals = {}) => {
    const known = new Set([
      ...history.flatMap((point) => Object.keys(point.signals || {})),
      ...Object.keys(signals),
    ]);
    const current = new Set([...select.options].map((option) => option.value));
    [...known].sort().forEach((name) => {
      if (current.has(name)) return;
      const option = document.createElement("option");
      option.value = name;
      option.textContent = name;
      select.append(option);
    });
    if (selected && known.has(selected)) select.value = selected;
    body.replaceChildren(...Object.entries(signals).map(([name, value]) => {
      const row = document.createElement("tr");
      const key = document.createElement("td");
      key.textContent = name;
      const rendered = document.createElement("td");
      rendered.textContent = number(value, 4);
      row.append(key, rendered);
      return row;
    }));
    renderChart();
  };
  select.addEventListener("change", () => {
    selected = select.value;
    localStorage.setItem(SIGNAL_KEY, selected);
    renderChart();
  });

  return {
    element,
    render(snapshot) { renderSignals(snapshot?.transition?.signals || {}); },
    renderHistory(next) { history = next; renderSignals(); },
    resize: renderChart,
  };
}
