const FRAME_HEADER_BYTES = 16;
const FRAME_GAME = 1;
const FRAME_OBSERVATION = 2;
const panelName = location.pathname.startsWith("/panel/")
  ? location.pathname.slice("/panel/".length)
  : null;
const workspaceWindowName = location.pathname.startsWith("/workspace/")
  ? location.pathname.slice("/workspace/".length)
  : null;
const token = new URLSearchParams(location.hash.slice(1)).get("token") || "";
const WORKSPACE_ID_KEY = "rlab.player.workspace.id";
const LAYOUT_KEY = "rlab.player.workspace.layout.v1";
const SAVED_LAYOUTS_KEY = "rlab.player.workspace.saved.v1";
const workspaceId = localStorage.getItem(WORKSPACE_ID_KEY) || crypto.randomUUID();
localStorage.setItem(WORKSPACE_ID_KEY, workspaceId);
const windowId = panelName ? `panel-${panelName}` : (workspaceWindowName || "main");
const PANEL_LABELS = {
  game: "Game",
  controls: "Controls",
  policy: "Policy distribution",
  reward: "Reward and return",
  actions: "Action history",
  observation: "Observation and attribution",
  signals: "Live signals",
  events: "Events",
  raw: "Transition inspector",
};

function defaultLayout() {
  return {
    version: 1,
    revision: 0,
    name: "Mario debug",
    panels: {
      game: { col: 1, row: 1, w: 7, h: 15, visible: true, window: "main" },
      controls: { col: 8, row: 1, w: 2, h: 15, visible: true, window: "main" },
      policy: { col: 10, row: 1, w: 3, h: 7, visible: true, window: "main" },
      reward: { col: 10, row: 8, w: 3, h: 8, visible: true, window: "main" },
      actions: { col: 1, row: 16, w: 4, h: 8, visible: false, window: "main" },
      observation: { col: 5, row: 16, w: 5, h: 8, visible: false, window: "main" },
      signals: { col: 10, row: 16, w: 3, h: 8, visible: false, window: "main" },
      events: { col: 1, row: 24, w: 4, h: 7, visible: false, window: "main" },
      raw: { col: 5, row: 24, w: 8, h: 7, visible: false, window: "main" },
    },
  };
}

const state = {
  socket: null,
  connected: false,
  clientId: null,
  snapshot: null,
  liveSnapshot: null,
  snapshots: new Map(),
  frameBlobs: new Map([[FRAME_GAME, new Map()], [FRAME_OBSERVATION, new Map()]]),
  inspectionSequence: null,
  timelineSequences: [],
  timelineWindow: Number(localStorage.getItem("rlab.player.timeline.window")) || 512,
  history: [],
  hasControl: false,
  frameSequence: new Map(),
  receivedFrameSequence: new Map(),
  pendingSnapshot: null,
  pressed: new Set(),
  gameFocused: false,
  mode: null,
  lastStatus: null,
  actionNamesKey: "",
  selectedSignal: localStorage.getItem("rlab.player.signal") || "",
  workspaceId,
  windowId,
  layout: null,
  selectedPanel: null,
  draggingPanel: null,
  dragSession: null,
  dragTarget: null,
  remoteDrag: null,
  activeWindows: new Map(),
  gameAspect: 256 / 240,
};

const workspaceChannel = "BroadcastChannel" in window
  ? new BroadcastChannel(`rlab-player-${workspaceId}`)
  : null;

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

function clamp(value, minimum, maximum) {
  return Math.max(minimum, Math.min(maximum, Number(value) || minimum));
}

function normalizeLayout(value) {
  const fallback = defaultLayout();
  const source = value && typeof value === "object" ? value : {};
  const panels = {};
  Object.entries(fallback.panels).forEach(([name, defaults]) => {
    const candidate = source.panels?.[name] || {};
    const w = clamp(candidate.w ?? defaults.w, name === "game" ? 4 : 2, 12);
    const h = clamp(candidate.h ?? defaults.h, name === "game" ? 8 : 4, 40);
    panels[name] = {
      col: clamp(candidate.col ?? defaults.col, 1, 13 - w),
      row: clamp(candidate.row ?? defaults.row, 1, 200),
      w,
      h,
      visible: candidate.visible === undefined ? defaults.visible : Boolean(candidate.visible),
      window: typeof candidate.window === "string" && candidate.window ? candidate.window : defaults.window,
    };
  });
  return {
    version: 1,
    revision: Number(source.revision) || 0,
    name: typeof source.name === "string" && source.name.trim() ? source.name.trim().slice(0, 48) : fallback.name,
    panels,
  };
}

function readStoredLayout() {
  try {
    return normalizeLayout(JSON.parse(localStorage.getItem(LAYOUT_KEY) || "null"));
  } catch {
    return defaultLayout();
  }
}

function panelsInThisWindow() {
  if (!state.layout) return [];
  return Object.entries(state.layout.panels)
    .filter(([, panel]) => panel.visible && panel.window === state.windowId)
    .map(([name]) => name);
}

function subscriptions() {
  const visible = new Set(panelsInThisWindow());
  const values = ["telemetry"];
  if (visible.has("game")) values.push("game");
  if (visible.has("observation")) values.push("observation");
  return values;
}

function setDetachedLayout() {
  const secondary = state.windowId !== "main";
  document.body.classList.toggle("secondary-window", secondary);
}

function showToast(message, error = false) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.style.borderColor = error ? "var(--red)" : "var(--cyan)";
  toast.classList.add("visible");
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => toast.classList.remove("visible"), 3200);
}

function updateConnection(label, kind = "") {
  const badge = $("#connection-status");
  badge.textContent = label;
  badge.className = `sync-status ${kind}`.trim();
}

function connect() {
  if (!token) {
    updateConnection("Missing session token", "error");
    showToast("Open the complete dashboard URL printed by rlab.", true);
    return;
  }
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${scheme}://${location.host}/ws`);
  state.socket = socket;
  socket.binaryType = "arraybuffer";
  updateConnection("Connecting", "warning");
  socket.addEventListener("open", () => {
    socket.send(JSON.stringify({
      type: "hello",
      token,
      subscriptions: subscriptions(),
      panel: panelName || "workspace",
      workspace_id: state.workspaceId,
      window_id: state.windowId,
    }));
  });
  socket.addEventListener("message", (event) => {
    if (typeof event.data === "string") handleMessage(JSON.parse(event.data));
    else handleFrame(event.data);
  });
  socket.addEventListener("close", () => {
    state.connected = false;
    state.hasControl = false;
    updateConnection("Disconnected", "error");
    updateControlState();
  });
  socket.addEventListener("error", () => updateConnection("Connection error", "error"));
}

function handleMessage(message) {
  if (message.type === "welcome") {
    state.connected = true;
    state.clientId = message.client_id;
    updateConnection("Synced", "");
    return;
  }
  if (message.type === "history") {
    state.history = Array.isArray(message.points) ? message.points : [];
    renderHistory();
    return;
  }
  if (message.type === "snapshot") {
    const frameKind = requiredFrameKind(message);
    if (frameKind && (state.frameSequence.get(frameKind) ?? -1) < message.sequence) {
      state.pendingSnapshot = message;
    } else {
      applySnapshot(message);
    }
    return;
  }
  if (message.type === "command_result") {
    if (!message.ok) showToast(message.error || "Command failed", true);
    else if (message.inspection?.kind === "policy") renderPolicy(message.inspection.decision, true);
    return;
  }
  if (message.type === "error") showToast(message.error || "Player error", true);
}

function rememberFrame(kind, sequence, blob) {
  const frames = state.frameBlobs.get(kind);
  frames.set(sequence, blob);
  while (frames.size > 1024) frames.delete(frames.keys().next().value);
}

function fitGameFrame() {
  const stage = $("#game-stage");
  const frame = $("#game-frame");
  if (!stage || !frame) return;
  const width = stage.clientWidth;
  const height = stage.clientHeight;
  if (!width || !height) return;
  const fittedWidth = Math.min(width, height * state.gameAspect);
  const fittedHeight = fittedWidth / state.gameAspect;
  frame.style.width = `${Math.max(1, Math.floor(fittedWidth))}px`;
  frame.style.height = `${Math.max(1, Math.floor(fittedHeight))}px`;
  frame.style.aspectRatio = String(state.gameAspect);
}

async function drawFrameBlob(kind, blob) {
  if (!blob) return false;
  const bitmap = await createImageBitmap(blob);
  const canvas = kind === FRAME_GAME ? $("#game-canvas") : $("#observation-canvas");
  if (!canvas) { bitmap.close(); return false; }
  if (kind === FRAME_GAME) {
    state.gameAspect = bitmap.width / Math.max(1, bitmap.height);
    fitGameFrame();
  }
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  const context = canvas.getContext("2d", { alpha: false });
  context.imageSmoothingEnabled = false;
  context.drawImage(bitmap, 0, 0);
  bitmap.close();
  if (kind === FRAME_GAME) $("#game-empty")?.setAttribute("hidden", "");
  if (kind === FRAME_OBSERVATION) $("#observation-empty")?.setAttribute("hidden", "");
  return true;
}

async function handleFrame(buffer) {
  const view = new DataView(buffer);
  if (buffer.byteLength <= FRAME_HEADER_BYTES) return;
  const magic = String.fromCharCode(...new Uint8Array(buffer, 0, 4));
  if (magic !== "RLP1") return;
  const kind = view.getUint8(4);
  const sequence = Number(view.getBigUint64(8));
  if (sequence <= (state.receivedFrameSequence.get(kind) ?? -1)) return;
  state.receivedFrameSequence.set(kind, sequence);
  const blob = new Blob([buffer.slice(FRAME_HEADER_BYTES)], { type: "image/png" });
  rememberFrame(kind, sequence, blob);
  if (state.inspectionSequence === null || state.inspectionSequence === sequence) {
    await drawFrameBlob(kind, blob);
  }
  state.frameSequence.set(kind, sequence);
  flushPendingSnapshot();
}

function requiredFrameKind(snapshot) {
  const visible = new Set(panelsInThisWindow());
  if (visible.has("game") && snapshot.transition?.after?.game_frame) return FRAME_GAME;
  if (visible.has("observation") && Number(snapshot.transition?.before?.observation_frames || 0) > 0) return FRAME_OBSERVATION;
  return null;
}

function applySnapshot(snapshot) {
  state.pendingSnapshot = null;
  state.liveSnapshot = snapshot;
  state.snapshots.set(Number(snapshot.sequence), snapshot);
  while (state.snapshots.size > 1024) state.snapshots.delete(state.snapshots.keys().next().value);
  state.hasControl = Boolean(snapshot.control?.has_control);
  if (state.inspectionSequence === null) {
    state.snapshot = snapshot;
    renderSnapshot();
  } else {
    updateControlState();
    renderWorkspaceStatus();
    renderTimeline();
  }
}

function flushPendingSnapshot() {
  const snapshot = state.pendingSnapshot;
  if (!snapshot) return;
  const frameKind = requiredFrameKind(snapshot);
  if (!frameKind || (state.frameSequence.get(frameKind) ?? -1) >= snapshot.sequence) applySnapshot(snapshot);
}

function send(value) {
  if (state.socket?.readyState === WebSocket.OPEN) state.socket.send(JSON.stringify(value));
}

function command(name, payload = {}) {
  if (!state.hasControl) {
    showToast("This window is an observer. Choose Control here first.", true);
    return;
  }
  send({
    type: "command",
    id: crypto.randomUUID(),
    name,
    payload,
    expected_revision: state.liveSnapshot?.revision ?? null,
  });
}

function text(value, fallback = "—") {
  return value === null || value === undefined || value === "" ? fallback : String(value);
}

function number(value, digits = 3) {
  return Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "—";
}

function stat(label, value) {
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
}

function setStats(target, values) {
  target.replaceChildren(...values.map(([label, value]) => stat(label, value)));
}

function renderJson(target, value, fallback) {
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
    } else if (raw === "true" || raw === "false") {
      token.className = "json-boolean";
    } else if (raw === "null") {
      token.className = "json-null";
    } else {
      token.className = "json-number";
    }
    token.textContent = raw;
    fragment.append(token);
    cursor = match.index + raw.length;
  }
  fragment.append(document.createTextNode(source.slice(cursor)));
  target.replaceChildren(fragment);
}

function updateControlState() {
  const control = $("#control-status");
  control.textContent = state.hasControl ? "Controller" : "Observer";
  control.className = `badge ${state.hasControl ? "" : "muted"}`.trim();
  $$(".transport button:not(#acquire-control):not([data-panel-menu]):not([data-drag-handle]):not(.panel-resize)")
    .forEach((button) => { button.disabled = !state.hasControl; });
  $("#acquire-control").disabled = !state.connected || state.hasControl;
}

function renderWorkspaceStatus() {
  const live = state.liveSnapshot;
  const shown = state.snapshot || live;
  $("#timeline-step").textContent = `EP ${text(shown?.session?.episode)} · STEP ${text(shown?.session?.step)}`;
  if (state.inspectionSequence === null) {
    $("#timeline-sequence").textContent = `SEQ ${text(shown?.sequence)}`;
  } else {
    $("#timeline-sequence").textContent = `SEQ ${text(shown?.sequence)} · LIVE ${text(live?.sequence)}`;
  }
}

function renderSnapshot() {
  const snapshot = state.snapshot;
  const session = snapshot.session || {};
  const transition = snapshot.transition;
  configureMode(snapshot.mode || "playback");
  updateControlState();
  renderWorkspaceStatus();
  $("#seed").value = text(session.seed, "");
  if (document.activeElement !== $("#fps")) $("#fps").value = Number(session.target_fps || 0);
  $("#session-summary").textContent = `${snapshot.run_state.toUpperCase()} · ${snapshot.driver.toUpperCase()}`;
  $("#resolved-config").textContent = session.config || "No configuration supplied.";
  renderJson($("#raw-transition"), transition, "No transition yet.");
  $("#model-input").textContent = transition?.before?.model_input?.join("\n") || "No policy input yet.";
  $("#game-overlay").textContent = transition
    ? `${state.inspectionSequence === null ? "" : "INSPECTING · "}EP ${transition.episode} · STEP ${transition.step} · r ${number(transition.reward?.step, 2)} · R ${number(transition.reward?.return, 2)} · ${String(transition.action_source || "—").toUpperCase()}${snapshot.interactive ? " · NON-EVIDENCE" : ""}`
    : `seed ${text(session.seed)} · ready`;
  const actionNamesKey = JSON.stringify(session.action_names || []);
  if (actionNamesKey !== state.actionNamesKey) {
    state.actionNamesKey = actionNamesKey;
    renderHistory();
  }
  if (state.inspectionSequence === null && snapshot.status_message && snapshot.status_message !== state.lastStatus) {
    state.lastStatus = snapshot.status_message;
    showToast(snapshot.status_message, snapshot.run_state === "paused" && /error|expired|unsupported|no configured/i.test(snapshot.status_message));
  }
  renderPolicy(transition?.decision || null, false);
  renderReward(transition);
  renderSignals(transition);
  renderEvents();
  if (state.inspectionSequence === null && transition && (!state.history.length || state.history.at(-1)?.sequence !== transition.sequence)) {
    state.history.push(historyFromTransition(transition));
    if (state.history.length > 4096) state.history.shift();
    renderHistory();
  }
  renderTimeline();
}

function configureMode(mode) {
  if (state.mode === mode) return;
  state.mode = mode;
  const recording = mode === "recording";
  document.body.classList.toggle("recording", recording);
  document.querySelector(".eyebrow").textContent = recording ? "HUMAN RECORDING" : "RLAB PLAYER";
  ["step", "step-ten", "continue-event", "continue-done", "reset", "policy-driver", "inspect-policy"].forEach((id) => {
    $(`#${id}`).hidden = recording;
  });
  $("#seed").closest("label").hidden = recording;
  const humanDriver = $("#human-driver");
  const humanDriverLabel = recording ? "Human controls" : "Take human control";
  humanDriver.setAttribute("aria-label", humanDriverLabel);
  humanDriver.title = humanDriverLabel;
}

function historyFromTransition(transition) {
  return {
    sequence: transition.sequence,
    episode: transition.episode,
    step: transition.step,
    action: transition.decision?.selected_action ?? null,
    action_source: transition.action_source,
    reward_provider: transition.reward?.provider,
    reward_shaped: transition.reward?.shaped,
    return: transition.reward?.return,
    value: transition.decision?.value,
    entropy: transition.decision?.entropy,
    events: transition.events || [],
    boundary: transition.boundary,
    signals: transition.signals || {},
    components: transition.reward?.components || {},
  };
}

function renderPolicy(decision, inspection) {
  const summary = $("#policy-summary");
  const actions = $("#policy-actions");
  if (!decision) {
    setStats(summary, [["Mode", state.snapshot?.driver || "—"], ["Decision", "Unavailable"]]);
    actions.className = "action-probabilities empty-state";
    actions.textContent = "No sampled policy decision for this transition.";
    return;
  }
  setStats(summary, [
    [inspection ? "Inspection" : "Mode", inspection ? "Policy" : (decision.sampled ? "Stochastic" : "Policy")],
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
  const names = state.snapshot?.session?.action_names || [];
  const rows = probabilities.map((probability, index) => {
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
  });
  actions.className = "action-probabilities";
  actions.replaceChildren(...rows);
}

function renderReward(transition) {
  const reward = transition?.reward || {};
  setStats($("#reward-stats"), [
    ["Provider r", number(reward.provider, 3)],
    ["Shaped r", number(reward.shaped, 3)],
    ["Return", number(reward.return, 2)],
    ["Outcome", transition?.outcome || "continuing"],
  ]);
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

function drawLines(canvas, series) {
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
    context.beginPath(); context.moveTo(padding, y); context.lineTo(width - padding, y); context.stroke();
  }
  series.forEach(({ values: points, color }) => {
    context.strokeStyle = color;
    context.lineWidth = 1.5;
    context.beginPath();
    points.forEach((value, index) => {
      if (!Number.isFinite(value)) return;
      const x = padding + (index / Math.max(1, points.length - 1)) * (width - padding * 2);
      const y = height - padding - ((value - min) / (max - min)) * (height - padding * 2);
      if (index === 0) context.moveTo(x, y); else context.lineTo(x, y);
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

function drawHistogram(canvas, counts, names) {
  const { context, ratio, width, height } = resizeCanvas(canvas);
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#071117"; context.fillRect(0, 0, width, height);
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
    const label = fitCanvasLabel(context, names[index] || String(index), barWidth + gap - 4);
    context.fillText(label, x + barWidth / 2, height - 14);
  });
}

function renderHistory() {
  const points = state.history.slice(-1024);
  drawLines($("#reward-chart"), [
    { values: points.map((point) => Number(point.reward_provider)), color: "#76a9ff" },
    { values: points.map((point) => Number(point.reward_shaped)), color: "#d794ff" },
  ]);
  drawLines($("#return-chart"), [
    { values: points.map((point) => Number(point.return)), color: "#60d394" },
  ]);
  $("#reward-legend").replaceChildren(...[
    ["Provider reward", "#76a9ff"], ["Shaped reward", "#d794ff"],
  ].map(([label, color]) => {
    const item = document.createElement("span"); item.textContent = label; item.style.setProperty("--legend-color", color); return item;
  }));
  $("#return-legend").replaceChildren(...[
    ["Return", "#60d394"],
  ].map(([label, color]) => {
    const item = document.createElement("span"); item.textContent = label; item.style.setProperty("--legend-color", color); return item;
  }));
  const names = state.snapshot?.session?.action_names || [];
  const counts = Array.from({ length: names.length || 1 }, () => 0);
  points.forEach((point) => { if (Number.isInteger(point.action) && point.action >= 0) counts[point.action] = (counts[point.action] || 0) + 1; });
  drawHistogram($("#action-chart"), counts, names);
  const total = counts.reduce((sum, value) => sum + value, 0);
  $("#action-caption").textContent = total ? `${total} sampled policy actions in the visible history.` : "No policy actions observed.";
  renderSignalChart();
  renderEvents();
  renderTimeline();
}

function nearestFrameBlob(kind, sequence) {
  const frames = state.frameBlobs.get(kind);
  if (frames.has(sequence)) return frames.get(sequence);
  const candidate = [...frames.keys()].filter((value) => value <= sequence).at(-1);
  return candidate === undefined ? null : frames.get(candidate);
}

async function showFramesForSequence(sequence) {
  const visible = new Set(panelsInThisWindow());
  const tasks = [];
  if (visible.has("game")) tasks.push(drawFrameBlob(FRAME_GAME, nearestFrameBlob(FRAME_GAME, sequence)));
  if (visible.has("observation")) tasks.push(drawFrameBlob(FRAME_OBSERVATION, nearestFrameBlob(FRAME_OBSERVATION, sequence)));
  const results = await Promise.all(tasks);
  if (tasks.length && !results.some(Boolean)) showToast("This retained transition has telemetry but no retained image frame.", true);
}

function inspectSequence(sequence) {
  const snapshot = state.snapshots.get(Number(sequence));
  if (!snapshot) return;
  state.inspectionSequence = Number(sequence);
  state.snapshot = snapshot;
  $("#return-live").hidden = false;
  renderSnapshot();
  showFramesForSequence(Number(sequence));
}

function returnToLive() {
  state.inspectionSequence = null;
  state.snapshot = state.liveSnapshot;
  $("#return-live").hidden = true;
  if (state.snapshot) {
    renderSnapshot();
    showFramesForSequence(Number(state.snapshot.sequence));
  }
}

function renderTimeline() {
  const scrubber = $("#timeline-scrubber");
  if (!scrubber) return;
  const all = [...state.snapshots.keys()].sort((a, b) => a - b);
  state.timelineSequences = all.slice(-state.timelineWindow);
  const sequences = state.timelineSequences;
  scrubber.min = "0";
  scrubber.max = String(Math.max(0, sequences.length - 1));
  scrubber.disabled = sequences.length < 2;
  const selected = state.inspectionSequence ?? sequences.at(-1);
  const selectedIndex = sequences.indexOf(selected);
  scrubber.value = String(selectedIndex < 0 ? Math.max(0, sequences.length - 1) : selectedIndex);
  $("#timeline-zoom-label").textContent = `${state.timelineWindow} steps`;
  renderWorkspaceStatus();

  const markers = $("#timeline-markers");
  if (!sequences.length) { markers.replaceChildren(); return; }
  const minimum = sequences[0];
  const maximum = sequences.at(-1);
  const range = Math.max(1, maximum - minimum);
  const interesting = state.history.filter((point) =>
    Number(point.sequence) >= minimum
    && Number(point.sequence) <= maximum
    && (point.boundary || point.events?.length)
  );
  markers.replaceChildren(...interesting.slice(-120).map((point) => {
    const marker = document.createElement("span");
    marker.className = "timeline-marker";
    marker.style.left = `${((Number(point.sequence) - minimum) / range) * 100}%`;
    marker.style.setProperty("--marker-color", point.boundary ? "var(--red)" : "var(--magenta)");
    return marker;
  }));
}

function renderSignals(transition) {
  const signals = transition?.signals || {};
  const select = $("#signal-select");
  const known = new Set([...state.history.flatMap((point) => Object.keys(point.signals || {})), ...Object.keys(signals)]);
  const currentOptions = new Set([...select.options].map((option) => option.value));
  [...known].sort().forEach((name) => {
    if (!currentOptions.has(name)) {
      const option = document.createElement("option"); option.value = name; option.textContent = name; select.append(option);
    }
  });
  if (state.selectedSignal && known.has(state.selectedSignal)) select.value = state.selectedSignal;
  const body = $("#signals-table tbody");
  body.replaceChildren(...Object.entries(signals).map(([name, value]) => {
    const row = document.createElement("tr");
    const key = document.createElement("td"); key.textContent = name;
    const rendered = document.createElement("td"); rendered.textContent = number(value, 4);
    row.append(key, rendered); return row;
  }));
  renderSignalChart();
}

function renderSignalChart() {
  const name = $("#signal-select").value;
  drawLines($("#signal-chart"), [{ values: state.history.slice(-1024).map((point) => Number(point.signals?.[name])), color: "#f0c36a" }]);
}

function renderEvents() {
  const events = state.history.filter((point) => point.boundary || point.events?.length).slice(-100).reverse();
  const list = $("#event-list");
  if (!events.length) {
    const empty = document.createElement("li"); empty.className = "empty-state"; empty.textContent = "No events observed."; list.replaceChildren(empty); return;
  }
  list.replaceChildren(...events.map((point) => {
    const item = document.createElement("li"); item.className = `event-item ${point.boundary ? "boundary" : ""}`;
    const label = document.createElement("div"); label.textContent = point.events?.length ? point.events.join(" · ") : "episode boundary";
    const meta = document.createElement("div"); meta.className = "event-meta"; meta.textContent = `seq ${point.sequence} · ep ${point.episode} · step ${point.step}`;
    item.append(label, meta); return item;
  }));
}

function panelsOverlap(a, b) {
  return a.col < b.col + b.w
    && a.col + a.w > b.col
    && a.row < b.row + b.h
    && a.row + a.h > b.row;
}

function resolveCollisions(movedName) {
  const moved = state.layout.panels[movedName];
  if (!moved?.visible) return;
  moved.col = clamp(moved.col, 1, 13 - moved.w);
  moved.row = clamp(moved.row, 1, 200);
  const placed = [moved];
  const others = Object.entries(state.layout.panels)
    .filter(([name, panel]) => name !== movedName && panel.visible && panel.window === moved.window)
    .sort(([, a], [, b]) => a.row - b.row || a.col - b.col);
  others.forEach(([, panel]) => {
    let collisions = placed.filter((candidate) => panelsOverlap(panel, candidate));
    while (collisions.length) {
      panel.row = Math.max(...collisions.map((candidate) => candidate.row + candidate.h));
      collisions = placed.filter((candidate) => panelsOverlap(panel, candidate));
    }
    placed.push(panel);
  });
}

function maxPanelRow(targetWindow = state.windowId) {
  return Math.max(0, ...Object.values(state.layout.panels)
    .filter((panel) => panel.visible && panel.window === targetWindow)
    .map((panel) => panel.row + panel.h - 1));
}

function persistLayout({ announce = true } = {}) {
  state.layout.revision = Number(state.layout.revision || 0) + 1;
  localStorage.setItem(LAYOUT_KEY, JSON.stringify(state.layout));
  if (announce) workspaceChannel?.postMessage({ type: "layout", layout: state.layout, source: state.windowId });
}

function updateLayoutTitle() {
  $("#page-title").textContent = panelName
    ? `${PANEL_LABELS[panelName] || panelName} window`
    : state.layout.name;
  $("#layout-name-input").value = state.layout.name;
  document.title = `${state.layout.name} · rlab player`;
}

function applyLayout() {
  const dashboard = $("#dashboard");
  const visibleHere = [];
  $$("[data-panel]").forEach((panel) => {
    const name = panel.dataset.panel;
    const config = state.layout.panels[name];
    if (!config) return;
    const shown = config.visible && config.window === state.windowId;
    panel.hidden = !shown;
    if (!shown) return;
    visibleHere.push(name);
    panel.style.gridColumn = `${config.col} / span ${config.w}`;
    panel.style.gridRow = `${config.row} / span ${config.h}`;
  });
  document.body.classList.toggle("empty-workspace", visibleHere.length === 0);
  const rows = Math.max(8, maxPanelRow());
  dashboard.style.minHeight = `${rows * 32 + Math.max(0, rows - 1) * 10 + 12}px`;
  updateLayoutTitle();
  renderPanelShelf();
  renderSavedLayouts();
  requestAnimationFrame(() => { fitGameFrame(); renderHistory(); });
  send({ type: "subscribe", subscriptions: subscriptions() });
}

function readSavedLayouts() {
  try {
    const value = JSON.parse(localStorage.getItem(SAVED_LAYOUTS_KEY) || "{}");
    return value && typeof value === "object" ? value : {};
  } catch {
    return {};
  }
}

function renderSavedLayouts() {
  const target = $("#saved-layouts");
  const saved = readSavedLayouts();
  const rows = Object.keys(saved).sort().map((name) => {
    const row = document.createElement("div");
    row.className = "saved-layout-row";
    const load = document.createElement("button");
    load.type = "button";
    load.className = "quiet";
    load.textContent = name;
    load.addEventListener("click", () => {
      state.layout = normalizeLayout(saved[name]);
      state.layout.name = name;
      persistLayout();
      applyLayout();
      $("#layout-menu").hidden = true;
      showToast(`Loaded layout “${name}”.`);
    });
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "quiet danger";
    remove.textContent = "Delete";
    remove.addEventListener("click", () => {
      const next = readSavedLayouts();
      delete next[name];
      localStorage.setItem(SAVED_LAYOUTS_KEY, JSON.stringify(next));
      renderSavedLayouts();
    });
    row.append(load, remove);
    return row;
  });
  if (!rows.length) {
    const empty = document.createElement("span");
    empty.className = "empty-state";
    empty.textContent = "No named layouts saved yet.";
    target.replaceChildren(empty);
  } else target.replaceChildren(...rows);
}

function renderPanelShelf() {
  const target = $("#panel-shelf-items");
  const entries = Object.entries(PANEL_LABELS).filter(([name]) => {
    const panel = state.layout.panels[name];
    return !panel.visible || panel.window !== state.windowId;
  });
  const buttons = entries.map(([name, label]) => {
    const config = state.layout.panels[name];
    const button = document.createElement("button");
    button.type = "button";
    button.className = "shelf-item";
    button.setAttribute("aria-label", config.visible ? `Move ${label} to this window` : `Show ${label}`);
    const title = document.createElement("span");
    title.textContent = label;
    button.append(title);
    if (config.visible) {
      const status = document.createElement("small");
      status.textContent = "Other window";
      button.append(status);
    }
    button.addEventListener("click", () => {
      const windowHasPanels = Object.entries(state.layout.panels).some(([otherName, panel]) =>
        otherName !== name && panel.visible && panel.window === state.windowId
      );
      config.visible = true;
      config.window = state.windowId;
      if (!windowHasPanels) {
        config.col = 1;
        config.row = 1;
      } else if (Object.entries(state.layout.panels).some(([otherName, panel]) =>
        otherName !== name && panel.visible && panel.window === state.windowId && panelsOverlap(config, panel)
      )) config.row = maxPanelRow() + 1;
      resolveCollisions(name);
      persistLayout();
      applyLayout();
      $("#panel-shelf").hidden = true;
      $("#panels-toggle").setAttribute("aria-expanded", "false");
      showToast(`${label} moved into this window.`);
    });
    return button;
  });
  if (!buttons.length) {
    const empty = document.createElement("span");
    empty.className = "empty-state";
    empty.textContent = "Every panel is visible in this window.";
    target.replaceChildren(empty);
  } else target.replaceChildren(...buttons);
}

function gridMetrics() {
  const dashboard = $("#dashboard");
  const rect = dashboard.getBoundingClientRect();
  const style = getComputedStyle(dashboard);
  const gap = Number.parseFloat(style.columnGap) || 10;
  const row = Number.parseFloat(style.gridAutoRows) || 32;
  const paddingLeft = Number.parseFloat(style.paddingLeft) || 0;
  const paddingRight = Number.parseFloat(style.paddingRight) || 0;
  const paddingTop = Number.parseFloat(style.paddingTop) || 0;
  const paddingBottom = Number.parseFloat(style.paddingBottom) || 0;
  const contentWidth = Math.max(0, rect.width - paddingLeft - paddingRight);
  const contentHeight = Math.max(0, rect.height - paddingTop - paddingBottom);
  const column = Math.max(0, (contentWidth - gap * 11) / 12);
  return {
    dashboard,
    rect,
    gap,
    column,
    row,
    paddingLeft,
    paddingTop,
    contentWidth,
    contentHeight,
    rowPitch: row + gap,
    columnPitch: column + gap,
  };
}

function dropTargetAt(clientX, clientY, config) {
  const metrics = gridMetrics();
  const { rect, paddingLeft, paddingTop, contentWidth, contentHeight, columnPitch, rowPitch } = metrics;
  const x = clientX - rect.left - paddingLeft;
  const y = clientY - rect.top - paddingTop;
  if (x < 0 || x > contentWidth || y < 0 || y > contentHeight) return null;
  return {
    cell: {
      col: clamp(Math.floor(x / columnPitch) + 1, 1, 13 - config.w),
      row: clamp(Math.floor(y / rowPitch) + 1, 1, 200),
    },
    metrics,
  };
}

function showDropPreview(config, cell, metrics = gridMetrics()) {
  const preview = $("#drop-preview");
  const { paddingLeft, paddingTop, column, row, gap, columnPitch, rowPitch } = metrics;
  preview.hidden = false;
  preview.style.left = `${paddingLeft + (cell.col - 1) * columnPitch}px`;
  preview.style.top = `${paddingTop + (cell.row - 1) * rowPitch}px`;
  preview.style.width = `${column * config.w + gap * (config.w - 1)}px`;
  preview.style.height = `${row * config.h + gap * (config.h - 1)}px`;
  preview.dataset.panel = state.draggingPanel || state.remoteDrag?.name || "";
  preview.dataset.targetWindow = state.windowId;
}

function hideDropPreview() {
  const preview = $("#drop-preview");
  preview.hidden = true;
  delete preview.dataset.panel;
  delete preview.dataset.targetWindow;
}

function ensurePanelDragOverlay() {
  let overlay = $("#panel-drag-overlay");
  if (overlay) return overlay;
  overlay = document.createElement("div");
  overlay.id = "panel-drag-overlay";
  overlay.className = "panel-drag-overlay";
  overlay.hidden = true;
  overlay.setAttribute("aria-hidden", "true");
  document.body.append(overlay);
  return overlay;
}

function showPanelDragOverlay(name, clientX, clientY) {
  const overlay = ensurePanelDragOverlay();
  overlay.textContent = PANEL_LABELS[name] || name;
  overlay.style.left = `${clientX}px`;
  overlay.style.top = `${clientY}px`;
  overlay.hidden = clientX < 0 || clientX > innerWidth || clientY < 0 || clientY > innerHeight;
}

function setPanelDragUi(active, { source = false } = {}) {
  document.body.classList.toggle("panel-drag-active", active);
  $("#dashboard").classList.toggle("drag-origin", active && source);
  $("#dashboard").classList.toggle("drag-receiving", active && !source);
  if (active) return;
  hideDropPreview();
  const overlay = $("#panel-drag-overlay");
  if (overlay) overlay.hidden = true;
}

function clientPointFromScreen(screenX, screenY) {
  const sideChrome = Math.max(0, (outerWidth - innerWidth) / 2);
  const topChrome = Math.max(0, outerHeight - innerHeight - sideChrome);
  return {
    x: Number(screenX) - window.screenX - sideChrome,
    y: Number(screenY) - window.screenY - topChrome,
  };
}

function publishPanelDragMove(session, event) {
  const screenX = Number(event.screenX);
  const screenY = Number(event.screenY);
  if (session.lastScreen?.x === screenX && session.lastScreen?.y === screenY) return;
  session.lastScreen = { x: screenX, y: screenY };
  session.move += 1;
  const config = state.layout.panels[session.name];
  const localTarget = dropTargetAt(event.clientX, event.clientY, config);
  state.dragTarget = localTarget
    ? { window: state.windowId, cell: localTarget.cell, move: session.move }
    : null;
  showPanelDragOverlay(session.name, event.clientX, event.clientY);
  if (localTarget) showDropPreview(config, localTarget.cell, localTarget.metrics);
  else hideDropPreview();
  workspaceChannel?.postMessage({
    type: "panel-drag-move",
    drag: session.id,
    source: state.windowId,
    name: session.name,
    move: session.move,
    screenX,
    screenY,
  });
}

function clearPanelDragSession(session, panel) {
  if (state.dragSession?.id !== session.id) return;
  panel.classList.remove("dragging");
  state.draggingPanel = null;
  state.dragSession = null;
  state.dragTarget = null;
  setPanelDragUi(false);
  workspaceChannel?.postMessage({ type: "panel-drag-end", drag: session.id, source: state.windowId });
}

function ensureResizeHandle(panel) {
  if (panel.querySelector(".panel-resize")) return;
  const handle = document.createElement("button");
  handle.type = "button";
  handle.className = "panel-resize";
  handle.setAttribute("aria-label", `Resize ${PANEL_LABELS[panel.dataset.panel] || panel.dataset.panel}`);
  const label = document.createElement("span");
  label.textContent = "Resize";
  handle.append(label);
  panel.append(handle);
  handle.addEventListener("pointerdown", (event) => beginResize(event, panel));
}

function beginResize(event, panel) {
  event.preventDefault();
  event.stopPropagation();
  const name = panel.dataset.panel;
  const config = state.layout.panels[name];
  const start = { x: event.clientX, y: event.clientY, w: config.w, h: config.h };
  const { columnPitch, rowPitch } = gridMetrics();
  panel.classList.add("resizing");
  const move = (next) => {
    const minW = name === "game" ? 4 : 2;
    const minH = name === "game" ? 8 : 4;
    config.w = clamp(start.w + Math.round((next.clientX - start.x) / columnPitch), minW, 13 - config.col);
    config.h = clamp(start.h + Math.round((next.clientY - start.y) / rowPitch), minH, 40);
    panel.style.gridColumn = `${config.col} / span ${config.w}`;
    panel.style.gridRow = `${config.row} / span ${config.h}`;
    requestAnimationFrame(fitGameFrame);
  };
  const finish = () => {
    panel.classList.remove("resizing");
    document.removeEventListener("pointermove", move);
    document.removeEventListener("pointerup", finish);
    resolveCollisions(name);
    persistLayout();
    applyLayout();
    showToast(`${PANEL_LABELS[name]} resized.`);
  };
  document.addEventListener("pointermove", move);
  document.addEventListener("pointerup", finish, { once: true });
}

function beginPanelDrag(event, panel) {
  if (event.button !== 0) return;
  const name = panel.dataset.panel;
  const config = state.layout.panels[name];
  const handle = event.currentTarget;
  const start = { x: event.clientX, y: event.clientY };
  let moved = false;
  let finishing = false;
  try { handle.setPointerCapture(event.pointerId); } catch { /* Pointer capture is optional. */ }
  const move = (next) => {
    if (!moved && Math.hypot(next.clientX - start.x, next.clientY - start.y) < 5) return;
    next.preventDefault();
    if (!moved) {
      moved = true;
      const session = { id: crypto.randomUUID(), name, move: 0, lastScreen: null };
      state.draggingPanel = name;
      state.dragSession = session;
      state.dragTarget = null;
      panel.classList.add("dragging");
      setPanelDragUi(true, { source: true });
      workspaceChannel?.postMessage({
        type: "panel-drag-start",
        drag: session.id,
        source: state.windowId,
        name,
        width: config.w,
        height: config.h,
      });
    }
    publishPanelDragMove(state.dragSession, next);
  };
  const removeListeners = () => {
    document.removeEventListener("pointermove", move);
    document.removeEventListener("pointerup", finish);
    document.removeEventListener("pointercancel", cancel);
    document.removeEventListener("keydown", keydown);
  };
  const complete = (cancelled = false) => {
    const session = state.dragSession;
    try { handle.releasePointerCapture(event.pointerId); } catch { /* Capture may already be released. */ }
    if (!moved || !session) return;
    if (!cancelled && state.dragTarget?.move === session.move) {
      const target = state.dragTarget;
      Object.assign(config, target.cell, { visible: true, window: target.window });
      resolveCollisions(name);
      persistLayout();
      applyLayout();
      const destination = target.window === state.windowId ? "" : " to the other window";
      showToast(`${PANEL_LABELS[name]} moved${destination}.`);
    }
    clearPanelDragSession(session, panel);
  };
  const finish = (next) => {
    if (finishing) return;
    finishing = true;
    removeListeners();
    if (!moved) { complete(); return; }
    publishPanelDragMove(state.dragSession, next);
    // Give the destination window one animation frame to claim the final pointer position.
    setTimeout(() => complete(false), 50);
  };
  const cancel = () => {
    if (finishing) return;
    finishing = true;
    removeListeners();
    complete(true);
  };
  const keydown = (next) => {
    if (next.key !== "Escape") return;
    next.preventDefault();
    cancel();
  };
  document.addEventListener("pointermove", move);
  document.addEventListener("pointerup", finish, { once: true });
  document.addEventListener("pointercancel", cancel, { once: true });
  document.addEventListener("keydown", keydown);
}

function bindPanelLayout() {
  const dashboard = $("#dashboard");
  dashboard.append($("#drop-preview"));
  $$("[data-panel]").forEach((panel) => {
    ensureResizeHandle(panel);
    const handle = panel.querySelector("[data-drag-handle]");
    if (!handle) return;
    handle.draggable = false;
    handle.addEventListener("pointerdown", (event) => beginPanelDrag(event, panel));
    handle.addEventListener("keydown", (event) => {
      if (!event.altKey || !["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(event.key)) return;
      event.preventDefault();
      const config = state.layout.panels[panel.dataset.panel];
      const amount = event.key === "ArrowLeft" || event.key === "ArrowUp" ? -1 : 1;
      if (event.shiftKey) {
        if (event.key === "ArrowLeft" || event.key === "ArrowRight") config.w = clamp(config.w + amount, panel.dataset.panel === "game" ? 4 : 2, 13 - config.col);
        else config.h = clamp(config.h + amount, panel.dataset.panel === "game" ? 8 : 4, 40);
      } else if (event.key === "ArrowLeft" || event.key === "ArrowRight") config.col = clamp(config.col + amount, 1, 13 - config.w);
      else config.row = clamp(config.row + amount, 1, 200);
      resolveCollisions(panel.dataset.panel);
      persistLayout();
      applyLayout();
    });
  });
}

function positionMenu(menu, anchor) {
  const rect = anchor.getBoundingClientRect();
  menu.hidden = false;
  const width = menu.offsetWidth || 304;
  menu.style.left = `${Math.max(8, Math.min(window.innerWidth - width - 8, rect.right - width))}px`;
  menu.style.top = `${Math.min(window.innerHeight - menu.offsetHeight - 8, rect.bottom + 6)}px`;
}

function openPanelMenu(name, anchor) {
  state.selectedPanel = name;
  $("#panel-menu-title").textContent = PANEL_LABELS[name] || name;
  $("#panel-dock-main").hidden = state.windowId === "main";
  positionMenu($("#panel-menu"), anchor);
}

function windowUrl(targetWindow) {
  return `${location.origin}/workspace/${encodeURIComponent(targetWindow)}#token=${encodeURIComponent(token)}`;
}

function movePanelToNewWindow(name) {
  const targetWindow = `window-${crypto.randomUUID().slice(0, 8)}`;
  const popup = window.open(windowUrl(targetWindow), `rlab-${targetWindow}`, "popup");
  if (!popup) { showToast("The browser blocked the new workspace window.", true); return; }
  const config = state.layout.panels[name];
  config.window = targetWindow;
  config.visible = true;
  config.col = 1;
  config.row = 1;
  persistLayout();
  applyLayout();
  showToast(`${PANEL_LABELS[name]} moved to a synchronized window.`);
}

function bindWorkspaceMenus() {
  $("#layouts-toggle").addEventListener("click", (event) => {
    $("#panel-shelf").hidden = true;
    $("#panels-toggle").setAttribute("aria-expanded", "false");
    positionMenu($("#layout-menu"), event.currentTarget);
  });
  $$("[data-panel-menu]").forEach((button) => button.addEventListener("click", (event) => {
    event.stopPropagation();
    openPanelMenu(button.dataset.panelMenu, button);
  }));
  $("#save-layout").addEventListener("click", () => {
    const name = $("#layout-name-input").value.trim().slice(0, 48) || "Workspace";
    state.layout.name = name;
    const saved = readSavedLayouts();
    saved[name] = state.layout;
    localStorage.setItem(SAVED_LAYOUTS_KEY, JSON.stringify(saved));
    persistLayout();
    applyLayout();
    $("#layout-menu").hidden = true;
    showToast(`Layout “${name}” saved.`);
  });
  $("#reset-layout").addEventListener("click", () => {
    state.layout = defaultLayout();
    persistLayout();
    applyLayout();
    $("#layout-menu").hidden = true;
    showToast("Default research layout restored.");
  });
  $("#panel-new-window").addEventListener("click", () => {
    if (state.selectedPanel) movePanelToNewWindow(state.selectedPanel);
    $("#panel-menu").hidden = true;
  });
  $("#panel-dock-main").addEventListener("click", () => {
    const name = state.selectedPanel;
    if (!name) return;
    const config = state.layout.panels[name];
    const appendRow = maxPanelRow("main") + 1;
    config.window = "main";
    config.visible = true;
    if (Object.entries(state.layout.panels).some(([otherName, panel]) =>
      otherName !== name && panel.visible && panel.window === "main" && panelsOverlap(config, panel)
    )) config.row = appendRow;
    resolveCollisions(name);
    persistLayout();
    applyLayout();
    $("#panel-menu").hidden = true;
    showToast(`${PANEL_LABELS[name]} docked to the main window.`);
    if (state.windowId !== "main" && !panelsInThisWindow().length) setTimeout(() => window.close(), 250);
  });
  $("#panel-hide").addEventListener("click", () => {
    const name = state.selectedPanel;
    if (!name) return;
    state.layout.panels[name].visible = false;
    persistLayout();
    applyLayout();
    $("#panel-menu").hidden = true;
    showToast(`${PANEL_LABELS[name]} moved to the panel shelf.`);
  });
  $("#panel-reset-size").addEventListener("click", () => {
    const name = state.selectedPanel;
    if (!name) return;
    const defaults = defaultLayout().panels[name];
    Object.assign(state.layout.panels[name], { w: defaults.w, h: defaults.h });
    state.layout.panels[name].col = clamp(state.layout.panels[name].col, 1, 13 - defaults.w);
    resolveCollisions(name);
    persistLayout();
    applyLayout();
    $("#panel-menu").hidden = true;
  });
  $("#panels-toggle").addEventListener("click", (event) => {
    const shelf = $("#panel-shelf");
    const opening = shelf.hidden;
    $("#layout-menu").hidden = true;
    shelf.hidden = true;
    event.currentTarget.setAttribute("aria-expanded", String(opening));
    if (opening) {
      renderPanelShelf();
      positionMenu(shelf, event.currentTarget);
    }
  });
  $("#new-window").addEventListener("click", () => {
    const targetWindow = `window-${crypto.randomUUID().slice(0, 8)}`;
    const popup = window.open(windowUrl(targetWindow), `rlab-${targetWindow}`, "popup");
    if (!popup) showToast("The browser blocked the new workspace window.", true);
  });
  document.addEventListener("click", (event) => {
    if (!$("#panel-menu").contains(event.target) && !event.target.closest("[data-panel-menu]")) $("#panel-menu").hidden = true;
    if (!$("#layout-menu").contains(event.target) && !event.target.closest("#layouts-toggle")) $("#layout-menu").hidden = true;
    if (!$("#panel-shelf").contains(event.target) && !event.target.closest("#panels-toggle")) {
      $("#panel-shelf").hidden = true;
      $("#panels-toggle").setAttribute("aria-expanded", "false");
    }
  });
}

function reclaimWindow(closedWindow) {
  if (state.windowId !== "main") return;
  let changed = false;
  Object.entries(state.layout.panels).forEach(([name, panel]) => {
    if (panel.visible && panel.window === closedWindow) {
      const appendRow = maxPanelRow("main") + 1;
      panel.window = "main";
      if (Object.entries(state.layout.panels).some(([otherName, candidate]) =>
        otherName !== name && candidate.visible && candidate.window === "main" && panelsOverlap(panel, candidate)
      )) panel.row = appendRow;
      resolveCollisions(name);
      changed = true;
    }
  });
  if (changed) {
    persistLayout();
    applyLayout();
    showToast("Panels from a closed window returned to the main workspace.");
  }
}

function bindWorkspaceSync() {
  if (workspaceChannel) {
    workspaceChannel.addEventListener("message", (event) => {
      const message = event.data || {};
      if (message.type === "layout" && message.source !== state.windowId) {
        const next = normalizeLayout(message.layout);
        if (next.revision >= Number(state.layout.revision || 0)) {
          state.layout = next;
          applyLayout();
        }
      } else if (message.type === "heartbeat") {
        state.activeWindows.set(message.window, Date.now());
      } else if (message.type === "panel-drag-start" && message.source !== state.windowId && PANEL_LABELS[message.name]) {
        state.remoteDrag = { id: message.drag, source: message.source, name: message.name };
        setPanelDragUi(true, { source: false });
      } else if (message.type === "panel-drag-move" && state.remoteDrag?.id === message.drag) {
        const config = state.layout.panels[state.remoteDrag.name];
        const point = clientPointFromScreen(message.screenX, message.screenY);
        const target = dropTargetAt(point.x, point.y, config);
        showPanelDragOverlay(state.remoteDrag.name, point.x, point.y);
        if (target) showDropPreview(config, target.cell, target.metrics);
        else hideDropPreview();
        workspaceChannel.postMessage({
          type: "panel-drag-target",
          drag: message.drag,
          source: message.source,
          target: state.windowId,
          move: message.move,
          cell: target?.cell || null,
        });
      } else if (message.type === "panel-drag-target" && state.dragSession?.id === message.drag && message.source === state.windowId) {
        if (message.move < state.dragSession.move) return;
        if (message.cell) {
          state.dragTarget = {
            window: message.target,
            cell: message.cell,
            move: message.move,
          };
        } else if (state.dragTarget?.window === message.target && state.dragTarget.move <= message.move) {
          state.dragTarget = null;
        }
      } else if (message.type === "panel-drag-end" && state.remoteDrag?.id === message.drag) {
        state.remoteDrag = null;
        setPanelDragUi(false);
      } else if (message.type === "window-closing" && state.windowId === "main") {
        setTimeout(() => {
          const lastSeen = state.activeWindows.get(message.window) || 0;
          if (Date.now() - lastSeen > 1800) reclaimWindow(message.window);
        }, 2000);
      }
    });
  }
  window.addEventListener("storage", (event) => {
    if (event.key !== LAYOUT_KEY || !event.newValue) return;
    try {
      const next = normalizeLayout(JSON.parse(event.newValue));
      if (next.revision >= Number(state.layout.revision || 0)) {
        state.layout = next;
        applyLayout();
      }
    } catch { /* Ignore malformed local data. */ }
  });
  const heartbeat = () => workspaceChannel?.postMessage({ type: "heartbeat", window: state.windowId });
  heartbeat();
  setInterval(heartbeat, 1000);
  window.addEventListener("beforeunload", () => {
    if (state.dragSession) workspaceChannel?.postMessage({ type: "panel-drag-end", drag: state.dragSession.id, source: state.windowId });
    workspaceChannel?.postMessage({ type: "window-closing", window: state.windowId });
  });
}

function bindTimeline() {
  $("#timeline-scrubber").addEventListener("input", (event) => {
    const sequence = state.timelineSequences[Number(event.target.value)];
    if (sequence !== undefined) inspectSequence(sequence);
  });
  $("#return-live").addEventListener("click", returnToLive);
  $("#timeline-zoom-out").addEventListener("click", () => {
    state.timelineWindow = clamp(state.timelineWindow * 2, 128, 1024);
    localStorage.setItem("rlab.player.timeline.window", String(state.timelineWindow));
    renderTimeline();
  });
  $("#timeline-zoom-in").addEventListener("click", () => {
    state.timelineWindow = clamp(state.timelineWindow / 2, 128, 1024);
    localStorage.setItem("rlab.player.timeline.window", String(state.timelineWindow));
    renderTimeline();
  });
}

function initWorkspace() {
  state.layout = readStoredLayout();
  if (panelName && state.layout.panels[panelName]) {
    state.layout.panels[panelName].visible = true;
    state.layout.panels[panelName].window = state.windowId;
    persistLayout();
  }
  setDetachedLayout();
  bindPanelLayout();
  bindWorkspaceMenus();
  bindWorkspaceSync();
  bindTimeline();
  applyLayout();
}

function bindControls() {
  $("#acquire-control").addEventListener("click", () => send({ type: "acquire_control" }));
  $("#pause").addEventListener("click", () => command("pause"));
  $("#play").addEventListener("click", () => command("play", { driver: state.snapshot?.driver || "policy" }));
  $("#step").addEventListener("click", () => command("step", { count: 1 }));
  $("#step-ten").addEventListener("click", () => command("step", { count: 10 }));
  $("#continue-event").addEventListener("click", () => command("continue", { target: "any" }));
  $("#continue-done").addEventListener("click", () => command("continue", { target: "done" }));
  $("#reset").addEventListener("click", () => command("reset", { seed: $("#seed").value }));
  $("#set-fps").addEventListener("click", () => command("set_fps", { fps: Number($("#fps").value) }));
  $("#policy-driver").addEventListener("click", () => command("set_driver", { driver: "policy" }));
  $("#human-driver").addEventListener("click", () => command("set_driver", { driver: "human" }));
  $("#inspect-policy").addEventListener("click", () => command("inspect_policy"));
  $("#end-session").addEventListener("click", () => command("stop"));
  $("#signal-select").addEventListener("change", (event) => {
    state.selectedSignal = event.target.value;
    localStorage.setItem("rlab.player.signal", state.selectedSignal);
    renderSignalChart();
  });
  $$('[data-fullscreen]').forEach((button) => button.addEventListener("click", () => {
    $("#game-stage").requestFullscreen({ navigationUI: "hide" }).catch((error) => showToast(error.message, true));
  }));
}

function bindHumanInput() {
  const canvas = $("#game-canvas");
  const mapping = new Map([
    ["ArrowUp", "up"], ["ArrowDown", "down"], ["ArrowLeft", "left"], ["ArrowRight", "right"],
    ["z", "b"], ["Z", "b"], ["x", "a"], ["X", "a"], ["Enter", "start"], ["Shift", "select"],
  ]);
  const publish = (focused = state.gameFocused) => send({ type: "input", pressed: [...state.pressed], focused });
  canvas.addEventListener("focus", () => { state.gameFocused = true; publish(true); });
  canvas.addEventListener("blur", () => { state.gameFocused = false; state.pressed.clear(); publish(false); });
  canvas.addEventListener("keydown", (event) => {
    const label = mapping.get(event.key);
    if (!label) return;
    event.preventDefault();
    state.pressed.add(label); publish(true);
  });
  canvas.addEventListener("keyup", (event) => {
    const label = mapping.get(event.key);
    if (!label) return;
    event.preventDefault();
    state.pressed.delete(label); publish(true);
  });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) { state.gameFocused = false; state.pressed.clear(); publish(false); }
  });
  setInterval(() => {
    if (state.gameFocused && state.hasControl && state.snapshot?.driver === "human") publish(true);
  }, 50);
}

window.addEventListener("resize", () => { fitGameFrame(); renderHistory(); });
initWorkspace();
bindControls();
bindHumanInput();
updateControlState();
const chartObserver = new ResizeObserver(() => renderHistory());
$$('.chart').forEach((canvas) => chartObserver.observe(canvas));
const gameObserver = new ResizeObserver(fitGameFrame);
gameObserver.observe($("#game-stage"));
connect();
