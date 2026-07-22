const FRAME_HEADER_BYTES = 16;
const FRAME_GAME = 1;
const FRAME_OBSERVATION = 2;
const panelName = location.pathname.startsWith("/panel/")
  ? location.pathname.slice("/panel/".length)
  : null;
const token = new URLSearchParams(location.hash.slice(1)).get("token") || "";

const state = {
  socket: null,
  connected: false,
  clientId: null,
  snapshot: null,
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
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

function subscriptions() {
  if (!panelName) return ["telemetry", "game", "observation"];
  if (panelName === "game") return ["telemetry", "game"];
  if (panelName === "observation") return ["telemetry", "observation"];
  return ["telemetry"];
}

function setDetachedLayout() {
  if (!panelName) return;
  document.body.classList.add("detached");
  const target = document.querySelector(`[data-panel="${CSS.escape(panelName)}"]`);
  if (target) target.classList.add("detached-target");
  $("#page-title").textContent = `${panelName[0].toUpperCase()}${panelName.slice(1)} panel`;
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
  badge.className = `badge ${kind}`.trim();
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
    socket.send(JSON.stringify({ type: "hello", token, subscriptions: subscriptions(), panel: panelName || "dashboard" }));
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
    updateConnection("Live", "");
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

async function handleFrame(buffer) {
  const view = new DataView(buffer);
  if (buffer.byteLength <= FRAME_HEADER_BYTES) return;
  const magic = String.fromCharCode(...new Uint8Array(buffer, 0, 4));
  if (magic !== "RLP1") return;
  const kind = view.getUint8(4);
  const sequence = Number(view.getBigUint64(8));
  if (sequence <= (state.receivedFrameSequence.get(kind) ?? -1)) return;
  state.receivedFrameSequence.set(kind, sequence);
  const bitmap = await createImageBitmap(new Blob([buffer.slice(FRAME_HEADER_BYTES)], { type: "image/png" }));
  if (sequence < (state.receivedFrameSequence.get(kind) ?? -1)) { bitmap.close(); return; }
  const canvas = kind === FRAME_GAME ? $("#game-canvas") : $("#observation-canvas");
  if (!canvas) { bitmap.close(); return; }
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  const context = canvas.getContext("2d", { alpha: false });
  context.imageSmoothingEnabled = false;
  context.drawImage(bitmap, 0, 0);
  bitmap.close();
  state.frameSequence.set(kind, sequence);
  if (kind === FRAME_GAME) $("#game-empty")?.setAttribute("hidden", "");
  if (kind === FRAME_OBSERVATION) $("#observation-empty")?.setAttribute("hidden", "");
  flushPendingSnapshot();
}

function requiredFrameKind(snapshot) {
  if ((panelName === null || panelName === "game") && snapshot.transition?.after?.game_frame) return FRAME_GAME;
  if (panelName === "observation" && Number(snapshot.transition?.before?.observation_frames || 0) > 0) return FRAME_OBSERVATION;
  return null;
}

function applySnapshot(snapshot) {
  state.pendingSnapshot = null;
  state.snapshot = snapshot;
  state.hasControl = Boolean(snapshot.control?.has_control);
  renderSnapshot();
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
    expected_revision: state.snapshot?.revision ?? null,
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

function updateControlState() {
  const control = $("#control-status");
  control.textContent = state.hasControl ? "Controller" : "Observer";
  control.className = `badge ${state.hasControl ? "" : "muted"}`.trim();
  $$(`.transport button:not(#acquire-control):not([data-detach])`).forEach((button) => { button.disabled = !state.hasControl; });
  $("#acquire-control").disabled = !state.connected || state.hasControl;
}

function renderSnapshot() {
  const snapshot = state.snapshot;
  const session = snapshot.session || {};
  const transition = snapshot.transition;
  configureMode(snapshot.mode || "playback");
  updateControlState();
  $("#seed").value = text(session.seed, "");
  if (document.activeElement !== $("#fps")) $("#fps").value = Number(session.target_fps || 0);
  $("#session-summary").textContent = `EP ${session.episode} · STEP ${session.step} · SEQ ${snapshot.sequence} · ${snapshot.run_state.toUpperCase()} · ${snapshot.driver.toUpperCase()}`;
  $("#resolved-config").textContent = session.config || "No configuration supplied.";
  $("#raw-transition").textContent = transition ? JSON.stringify(transition, null, 2) : "No transition yet.";
  $("#model-input").textContent = transition?.before?.model_input?.join("\n") || "No policy input yet.";
  $("#game-overlay").textContent = transition
    ? `seq ${transition.sequence} · ep ${transition.episode} · step ${transition.step}\nr ${number(transition.reward?.step, 2)} · return ${number(transition.reward?.return, 2)}\n${transition.action_source}${snapshot.interactive ? " · NON-EVIDENCE" : ""}`
    : `seed ${text(session.seed)} · ready`;
  const actionNamesKey = JSON.stringify(session.action_names || []);
  if (actionNamesKey !== state.actionNamesKey) {
    state.actionNamesKey = actionNamesKey;
    renderHistory();
  }
  if (snapshot.status_message && snapshot.status_message !== state.lastStatus) {
    state.lastStatus = snapshot.status_message;
    showToast(snapshot.status_message, snapshot.run_state === "paused" && /error|expired|unsupported|no configured/i.test(snapshot.status_message));
  }
  renderPolicy(transition?.decision || null, false);
  renderReward(transition);
  renderSignals(transition);
  renderEvents();
  if (transition && (!state.history.length || state.history.at(-1)?.sequence !== transition.sequence)) {
    state.history.push(historyFromTransition(transition));
    if (state.history.length > 4096) state.history.shift();
    renderHistory();
  }
}

function configureMode(mode) {
  if (state.mode === mode) return;
  state.mode = mode;
  const recording = mode === "recording";
  document.body.classList.toggle("recording", recording);
  if (recording) {
    $("#page-title").textContent = panelName ? `${panelName[0].toUpperCase()}${panelName.slice(1)} panel` : "Human recording dashboard";
    document.querySelector(".eyebrow").textContent = "HUMAN RECORDING";
  }
  ["step", "step-ten", "continue-event", "continue-done", "reset", "policy-driver", "inspect-policy"].forEach((id) => {
    $(`#${id}`).hidden = recording;
  });
  $("#seed").closest("label").hidden = recording;
  $("#human-driver").textContent = recording ? "Human controls" : "Take control";
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
    setStats(summary, [["Source", state.snapshot?.driver || "—"], ["Decision", "Human / unavailable"]]);
    actions.className = "action-probabilities empty-state";
    actions.textContent = "No sampled policy decision for this transition.";
    return;
  }
  setStats(summary, [
    [inspection ? "Inspection" : "Source", inspection ? "Policy mode" : (decision.sampled ? "Stochastic sample" : "Policy mode")],
    ["Value V(s)", number(decision.value, 4)],
    ["Entropy", number(decision.entropy, 4)],
    ["Log probability", number(decision.log_probability, 4)],
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
    ["Provider", number(reward.provider, 3)],
    ["Shaped", number(reward.shaped, 3)],
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

function drawHistogram(canvas, counts, names) {
  const { context, ratio, width, height } = resizeCanvas(canvas);
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#071117"; context.fillRect(0, 0, width, height);
  const max = Math.max(1, ...counts);
  const gap = 4;
  const barWidth = Math.max(4, (width - 24) / Math.max(1, counts.length) - gap);
  counts.forEach((count, index) => {
    const barHeight = (count / max) * (height - 42);
    const x = 12 + index * (barWidth + gap);
    context.fillStyle = "#53d4e8";
    context.fillRect(x, height - 22 - barHeight, barWidth, barHeight);
    context.fillStyle = "#8da6b2";
    context.font = "9px ui-monospace";
    context.save(); context.translate(x + 2, height - 7); context.rotate(-0.45);
    context.fillText((names[index] || String(index)).slice(0, 9), 0, 0); context.restore();
  });
}

function renderHistory() {
  const points = state.history.slice(-1024);
  drawLines($("#reward-chart"), [
    { values: points.map((point) => Number(point.reward_provider)), color: "#76a9ff" },
    { values: points.map((point) => Number(point.reward_shaped)), color: "#d794ff" },
    { values: points.map((point) => Number(point.return)), color: "#60d394" },
  ]);
  $("#reward-legend").replaceChildren(...[
    ["Provider reward", "#76a9ff"], ["Shaped reward", "#d794ff"], ["Return", "#60d394"],
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
  $$('[data-detach]').forEach((button) => button.addEventListener("click", () => {
    const panel = button.dataset.detach;
    window.open(`${location.origin}/panel/${encodeURIComponent(panel)}#token=${encodeURIComponent(token)}`, `rlab-${panel}`, "popup,noopener");
  }));
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

window.addEventListener("resize", () => renderHistory());
setDetachedLayout();
bindControls();
bindHumanInput();
updateControlState();
connect();
