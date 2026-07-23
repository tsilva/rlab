export class PanelRuntime {
  constructor({ catalog, container, services, onMount, onError }) {
    this.catalog = catalog;
    this.container = container;
    this.services = services;
    this.onMount = onMount;
    this.onError = onError;
    this.instances = new Map();
    this.loading = new Map();
    this.desired = new Set();
    this.generation = 0;
    this.view = { snapshot: null, history: [], inspection: false };
  }

  async sync(layout, windowId) {
    const generation = ++this.generation;
    this.desired = new Set(
      Object.entries(layout.panels)
        .filter(([, config]) => config.visible && config.window === windowId)
        .map(([name]) => name),
    );

    [...this.instances.keys()].forEach((name) => {
      if (!this.desired.has(name)) this.unmount(name);
    });

    await Promise.all([...this.desired].map((name) => this.ensureMounted(name)));
    if (generation !== this.generation) return;
    this.instances.forEach((instance, name) => {
      const config = layout.panels[name];
      if (!config || !this.desired.has(name)) return;
      instance.element.style.gridColumn = `${config.col} / span ${config.w}`;
      instance.element.style.gridRow = `${config.row} / span ${config.h}`;
    });
  }

  async ensureMounted(name) {
    if (this.instances.has(name)) return this.instances.get(name);
    if (this.loading.has(name)) return this.loading.get(name);
    const loading = this.loadPanel(name);
    this.loading.set(name, loading);
    try {
      return await loading;
    } finally {
      this.loading.delete(name);
    }
  }

  async loadPanel(name) {
    const definition = this.catalog[name];
    if (!definition) return null;
    try {
      const module = await import(definition.module);
      if (!this.desired.has(name)) return null;
      const instance = await module.mount({
        definition: { ...definition, id: name },
        services: this.services,
      });
      if (!instance?.element) throw new Error(`Panel ${name} did not return an element.`);
      if (!this.desired.has(name)) {
        instance.destroy?.();
        return null;
      }
      instance.element.dataset.panel = name;
      this.container.append(instance.element);
      const observer = new ResizeObserver(() => this.safeCall(name, "resize"));
      observer.observe(instance.element);
      instance.observer = observer;
      this.instances.set(name, instance);
      this.onMount(instance.element, name, definition);
      this.safeCall(name, "render", this.view.snapshot, this.view);
      this.safeCall(name, "renderHistory", this.view.history, this.view.snapshot);
      return instance;
    } catch (error) {
      this.onError(name, error);
      return null;
    }
  }

  unmount(name) {
    const instance = this.instances.get(name);
    if (!instance) return;
    instance.observer?.disconnect();
    this.safeCall(name, "destroy");
    instance.element.remove();
    this.instances.delete(name);
  }

  safeCall(name, method, ...args) {
    const callback = this.instances.get(name)?.[method];
    if (typeof callback !== "function") return undefined;
    try {
      return callback(...args);
    } catch (error) {
      this.onError(name, error);
      return undefined;
    }
  }

  renderSnapshot(snapshot, view = {}) {
    this.view = { ...this.view, ...view, snapshot };
    this.instances.forEach((_, name) => this.safeCall(name, "render", snapshot, this.view));
  }

  renderHistory(history, snapshot = this.view.snapshot) {
    this.view = { ...this.view, history, snapshot };
    this.instances.forEach((_, name) => this.safeCall(name, "renderHistory", history, snapshot));
  }

  async renderFrame(kind, blob) {
    const tasks = [...this.instances.entries()]
      .filter(([name]) => this.catalog[name]?.frameKinds.includes(kind))
      .map(([name]) => Promise.resolve(this.safeCall(name, "renderFrame", kind, blob))
        .catch((error) => {
          this.onError(name, error);
          return false;
        }));
    const results = await Promise.all(tasks);
    return results.some(Boolean);
  }

  invoke(name, method, ...args) {
    return this.safeCall(name, method, ...args);
  }

  resize() {
    this.instances.forEach((_, name) => this.safeCall(name, "resize"));
  }
}
