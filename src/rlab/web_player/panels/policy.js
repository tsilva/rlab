import { createPanel, number, setStats } from "./shared.js";

export function mount({ definition, services }) {
  const element = createPanel({
    id: definition.id,
    label: definition.label,
    body: `
      <div data-summary class="stat-grid"></div>
      <div data-actions class="action-probabilities empty-state">No sampled decision yet.</div>
    `,
  });
  const summary = element.querySelector("[data-summary]");
  const actions = element.querySelector("[data-actions]");

  const renderDecision = (decision) => {
    if (!decision) {
      setStats(summary, [["Mode", services.getState().snapshot?.driver || "—"], ["Decision", "Unavailable"]]);
      actions.className = "action-probabilities empty-state";
      actions.textContent = "No sampled policy decision for this transition.";
      return;
    }
    setStats(summary, [
      ["Mode", decision.sampled ? "Stochastic" : "Deterministic"],
      ["V(s)", number(decision.value, 4)],
      ["Entropy", number(decision.entropy, 4)],
      ["Log p", number(decision.log_probability, 4)],
    ]);
    const probabilities = decision.probabilities;
    if (!Array.isArray(probabilities)) {
      actions.className = "action-probabilities";
      actions.textContent = `Executed ${JSON.stringify(decision.executed_action)} · mean ${JSON.stringify(decision.mean)} · std ${JSON.stringify(decision.stddev)}`;
      return;
    }
    const names = services.getState().snapshot?.session?.action_names || [];
    actions.className = "action-probabilities";
    actions.replaceChildren(...probabilities.map((probability, index) => {
      const row = document.createElement("div");
      row.className = `action-row ${index === decision.selected_action ? "selected" : ""}`;
      const label = document.createElement("span");
      label.textContent = names[index] || `action ${index}`;
      const track = document.createElement("div");
      track.className = "probability-track";
      const fill = document.createElement("div");
      fill.className = "probability-fill";
      fill.style.width = `${Math.max(0, Math.min(100, Number(probability) * 100))}%`;
      track.append(fill);
      const amount = document.createElement("span");
      amount.textContent = `${(Number(probability) * 100).toFixed(1)}%`;
      row.append(label, track, amount);
      return row;
    }));
  };

  return {
    element,
    render(snapshot) { renderDecision(snapshot?.transition?.decision || null); },
  };
}
