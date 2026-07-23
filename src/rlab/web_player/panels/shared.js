const ICONS = "/assets/tabler-icons.svg";

export function text(value, fallback = "—") {
  return value === null || value === undefined || value === "" ? fallback : String(value);
}

export function number(value, digits = 3) {
  return Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "—";
}

export function createPanel({
  id,
  label,
  body = "",
  className = "",
  tag = "section",
  headerClass = "",
}) {
  const element = document.createElement(tag);
  element.className = `panel ${className}`.trim();
  element.dataset.panel = id;
  const heading = `${id}-panel-heading`;
  element.setAttribute("aria-labelledby", heading);
  element.innerHTML = `
    <header class="panel-header ${headerClass}">
      <button data-drag-handle class="icon-button icon-only panel-drag" type="button" aria-label="Move ${label.toLowerCase()} panel" title="Move ${label.toLowerCase()} panel"><svg class="icon" aria-hidden="true"><use href="${ICONS}#ti-grip-vertical"></use></svg></button>
      <div class="panel-title"><h2 id="${heading}">${label}</h2></div>
      <button data-panel-menu="${id}" class="icon-button icon-only" type="button" aria-label="${label} panel options" title="${label} panel options"><svg class="icon" aria-hidden="true"><use href="${ICONS}#ti-dots-vertical"></use></svg></button>
    </header>
    ${body}
  `;
  return element;
}

export function setStats(target, values) {
  target.replaceChildren(...values.map(([label, value]) => {
    const box = document.createElement("div");
    box.className = "stat";
    const key = document.createElement("span");
    key.className = "stat-label";
    key.textContent = label;
    const rendered = document.createElement("span");
    rendered.className = "stat-value";
    rendered.textContent = text(value);
    box.append(key, rendered);
    return box;
  }));
}

export function renderJson(target, value, fallback) {
  if (value === null || value === undefined) {
    target.textContent = fallback;
    return;
  }
  const source = JSON.stringify(value, null, 2);
  const tokens = /"(?:\\.|[^"\\])*"(?=\s*:)|"(?:\\.|[^"\\])*"|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|\b(?:true|false|null)\b/g;
  const fragment = document.createDocumentFragment();
  let cursor = 0;
  for (const match of source.matchAll(tokens)) {
    fragment.append(document.createTextNode(source.slice(cursor, match.index)));
    const token = document.createElement("span");
    const raw = match[0];
    if (raw.startsWith('"')) {
      token.className = source.slice(match.index + raw.length).match(/^\s*:/)
        ? "json-key"
        : "json-string";
    } else if (raw === "true" || raw === "false") token.className = "json-boolean";
    else if (raw === "null") token.className = "json-null";
    else token.className = "json-number";
    token.textContent = raw;
    fragment.append(token);
    cursor = match.index + raw.length;
  }
  fragment.append(document.createTextNode(source.slice(cursor)));
  target.replaceChildren(fragment);
}

function resizeCanvas(canvas) {
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(240, canvas.clientWidth);
  const height = Math.max(120, canvas.clientHeight);
  if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
  }
  return { context: canvas.getContext("2d"), ratio, width, height };
}

export function drawLines(canvas, series) {
  const { context, ratio, width, height } = resizeCanvas(canvas);
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#071117";
  context.fillRect(0, 0, width, height);
  const values = series.flatMap((item) => item.values.filter(Number.isFinite));
  if (!values.length) {
    context.fillStyle = "#8da6b2";
    context.font = "12px system-ui";
    context.fillText("No history yet", 12, 22);
    return;
  }
  let min = Math.min(...values);
  let max = Math.max(...values);
  if (min === max) { min -= 1; max += 1; }
  const padding = 12;
  context.strokeStyle = "#1d3541";
  context.lineWidth = 1;
  for (let index = 1; index < 4; index += 1) {
    const y = padding + ((height - padding * 2) * index) / 4;
    context.beginPath();
    context.moveTo(padding, y);
    context.lineTo(width - padding, y);
    context.stroke();
  }
  series.forEach(({ values: points, color }) => {
    context.strokeStyle = color;
    context.lineWidth = 1.5;
    context.beginPath();
    points.forEach((value, index) => {
      if (!Number.isFinite(value)) return;
      const x = padding + (index / Math.max(1, points.length - 1)) * (width - padding * 2);
      const y = height - padding - ((value - min) / (max - min)) * (height - padding * 2);
      if (index === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    });
    context.stroke();
  });
}

function fitCanvasLabel(context, value, maxWidth) {
  const label = String(value);
  if (context.measureText(label).width <= maxWidth) return label;
  let end = label.length;
  while (end > 0 && context.measureText(`${label.slice(0, end)}…`).width > maxWidth) end -= 1;
  return end > 0 ? `${label.slice(0, end)}…` : "…";
}

export function drawHistogram(canvas, counts, names) {
  const { context, ratio, width, height } = resizeCanvas(canvas);
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#071117";
  context.fillRect(0, 0, width, height);
  const max = Math.max(1, ...counts);
  const gap = 4;
  const barWidth = Math.max(4, (width - 24) / Math.max(1, counts.length) - gap);
  const plotBottom = height - 30;
  counts.forEach((count, index) => {
    const barHeight = (count / max) * Math.max(0, plotBottom - 12);
    const x = 12 + index * (barWidth + gap);
    context.fillStyle = "#53d4e8";
    context.fillRect(x, plotBottom - barHeight, barWidth, barHeight);
    context.fillStyle = "#d7e5ea";
    context.font = "600 12px system-ui, sans-serif";
    context.textAlign = "center";
    context.textBaseline = "middle";
    context.fillText(
      fitCanvasLabel(context, names[index] || String(index), barWidth + gap - 4),
      x + barWidth / 2,
      height - 14,
    );
  });
}
