import { createPanel, drawLines, number, setStats } from "./shared.js";

function legend(target, items) {
  target.replaceChildren(...items.map(([label, color]) => {
    const item = document.createElement("span");
    item.textContent = label;
    item.style.setProperty("--legend-color", color);
    return item;
  }));
}

export function mount({ definition }) {
  const element = createPanel({
    id: definition.id,
    label: definition.label,
    body: `
      <div data-stats class="stat-grid"></div>
      <div class="reward-plots">
        <section class="reward-plot" aria-labelledby="reward-plot-title">
          <div class="chart-heading"><span id="reward-plot-title">Step reward</span></div>
          <canvas data-reward-chart class="chart reward-chart" aria-label="Provider and shaped step reward history"></canvas>
          <div data-reward-legend class="legend"></div>
        </section>
        <section class="reward-plot" aria-labelledby="return-plot-title">
          <div class="chart-heading"><span id="return-plot-title">Episode return</span></div>
          <canvas data-return-chart class="chart reward-chart" aria-label="Episode return history"></canvas>
          <div data-return-legend class="legend"></div>
        </section>
      </div>
    `,
  });
  const rewardChart = element.querySelector("[data-reward-chart]");
  const returnChart = element.querySelector("[data-return-chart]");
  let history = [];

  const renderHistory = (next) => {
    history = next;
    const points = history.slice(-1024);
    drawLines(rewardChart, [
      { values: points.map((point) => Number(point.reward_provider)), color: "#76a9ff" },
      { values: points.map((point) => Number(point.reward_shaped)), color: "#d794ff" },
    ]);
    drawLines(returnChart, [
      { values: points.map((point) => Number(point.return)), color: "#60d394" },
    ]);
    legend(element.querySelector("[data-reward-legend]"), [
      ["Provider reward", "#76a9ff"], ["Shaped reward", "#d794ff"],
    ]);
    legend(element.querySelector("[data-return-legend]"), [["Return", "#60d394"]]);
  };

  return {
    element,
    render(snapshot) {
      const transition = snapshot?.transition;
      const reward = transition?.reward || {};
      setStats(element.querySelector("[data-stats]"), [
        ["Provider r", number(reward.provider, 3)],
        ["Shaped r", number(reward.shaped, 3)],
        ["Return", number(reward.return, 2)],
        ["Outcome", transition?.outcome || "continuing"],
      ]);
    },
    renderHistory,
    resize() { renderHistory(history); },
  };
}
