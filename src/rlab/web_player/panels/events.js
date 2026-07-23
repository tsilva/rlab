import { createPanel } from "./shared.js";

export function mount({ definition }) {
  const element = createPanel({
    id: definition.id,
    label: definition.label,
    body: '<ol data-list class="event-list"><li class="empty-state">No events observed.</li></ol>',
  });
  const list = element.querySelector("[data-list]");

  return {
    element,
    renderHistory(history) {
      const events = history
        .filter((point) => point.boundary || point.events?.length)
        .slice(-100)
        .reverse();
      if (!events.length) {
        const empty = document.createElement("li");
        empty.className = "empty-state";
        empty.textContent = "No events observed.";
        list.replaceChildren(empty);
        return;
      }
      list.replaceChildren(...events.map((point) => {
        const item = document.createElement("li");
        item.className = `event-item ${point.boundary ? "boundary" : ""}`;
        const label = document.createElement("div");
        label.textContent = point.events?.length ? point.events.join(" · ") : "episode boundary";
        const meta = document.createElement("div");
        meta.className = "event-meta";
        meta.textContent = `seq ${point.sequence} · ep ${point.episode} · step ${point.step}`;
        item.append(label, meta);
        return item;
      }));
    },
  };
}
