export const FRAME_GAME = 1;
export const FRAME_OBSERVATION = 2;

export const PANEL_CATALOG = Object.freeze({
  game: {
    label: "Game",
    module: "./game.js",
    layout: { col: 1, row: 1, w: 7, h: 15, visible: true, window: "main" },
    minimum: { w: 4, h: 8 },
    subscriptions: ["game"],
    frameKinds: [FRAME_GAME],
  },
  controls: {
    label: "Controls",
    module: "./controls.js",
    layout: { col: 8, row: 1, w: 2, h: 15, visible: true, window: "main" },
    minimum: { w: 2, h: 4 },
    subscriptions: [],
    frameKinds: [],
  },
  policy: {
    label: "Policy distribution",
    module: "./policy.js",
    layout: { col: 10, row: 1, w: 3, h: 7, visible: true, window: "main" },
    minimum: { w: 2, h: 4 },
    subscriptions: [],
    frameKinds: [],
  },
  reward: {
    label: "Reward and return",
    module: "./reward.js",
    layout: { col: 10, row: 8, w: 3, h: 8, visible: true, window: "main" },
    minimum: { w: 2, h: 4 },
    subscriptions: [],
    frameKinds: [],
  },
  actions: {
    label: "Action history",
    module: "./actions.js",
    layout: { col: 1, row: 16, w: 4, h: 8, visible: false, window: "main" },
    minimum: { w: 2, h: 4 },
    subscriptions: [],
    frameKinds: [],
  },
  observation: {
    label: "Observation and attribution",
    module: "./observation.js",
    layout: { col: 5, row: 16, w: 5, h: 8, visible: false, window: "main" },
    minimum: { w: 2, h: 4 },
    subscriptions: ["observation"],
    frameKinds: [FRAME_OBSERVATION],
  },
  signals: {
    label: "Live signals",
    module: "./signals.js",
    layout: { col: 10, row: 16, w: 3, h: 8, visible: false, window: "main" },
    minimum: { w: 2, h: 4 },
    subscriptions: [],
    frameKinds: [],
  },
  events: {
    label: "Events",
    module: "./events.js",
    layout: { col: 1, row: 24, w: 4, h: 7, visible: false, window: "main" },
    minimum: { w: 2, h: 4 },
    subscriptions: [],
    frameKinds: [],
  },
  raw: {
    label: "Transition inspector",
    module: "./raw.js",
    layout: { col: 5, row: 24, w: 8, h: 7, visible: false, window: "main" },
    minimum: { w: 2, h: 4 },
    subscriptions: [],
    frameKinds: [],
  },
});

export function panelLabels() {
  return Object.fromEntries(
    Object.entries(PANEL_CATALOG).map(([name, definition]) => [name, definition.label]),
  );
}

export function defaultPanelLayout() {
  return Object.fromEntries(
    Object.entries(PANEL_CATALOG).map(([name, definition]) => [name, { ...definition.layout }]),
  );
}

export function panelSubscriptions(names) {
  const values = new Set(["telemetry"]);
  names.forEach((name) => {
    PANEL_CATALOG[name]?.subscriptions.forEach((subscription) => values.add(subscription));
  });
  return [...values];
}
