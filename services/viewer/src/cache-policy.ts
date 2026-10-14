import type * as Cesium from "cesium";

import type { DisplayMode, ModelKey, ModelState, ModelStates, ViewName } from "./types";
import type { RuntimeMonitor } from "./runtime-monitor";

export interface CachePolicySettings {
  modelKey: ModelKey;
  resourcePrefix: string;
  retainMs: number;
  softBytes: number;
  hardBytes: number;
}

export interface CacheSample {
  enabled: boolean;
  ready: boolean;
  elapsedMs: number;
  loaded: number;
  visible: number;
  memoryBytes: number;
}

export type TrimDecision =
  | "disabled"
  | "not-ready"
  | "no-spare"
  | "hard-pressure"
  | "retain-recent-view"
  | "under-soft-budget"
  | "idle-over-soft-budget";

export class SettledTrimGate {
  epoch = 0;
  passes = 0;
  stopReason: string | null = null;
  private signature: string | null = null;
  private readySince: number | null = null;
  private lastAttemptAt = Number.NEGATIVE_INFINITY;
  private lastAttemptFingerprint = "";
  private currentFingerprint = "";
  private now = 0;
  private currentReady = false;

  constructor(
    readonly settleMs = 2000,
    readonly cooldownMs = 3000,
    readonly maxPasses = 3,
  ) {
    if (
      ![settleMs, cooldownMs].every((value) => Number.isFinite(value) && value >= 0)
      || !Number.isInteger(maxPasses)
      || maxPasses < 1
    ) {
      throw new Error("invalid convergence limits");
    }
  }

  sample(now: number, signature: string, ready: boolean, fingerprint = ""): boolean {
    if (signature !== this.signature) {
      this.signature = signature;
      this.epoch += 1;
      this.readySince = null;
      this.passes = 0;
      this.lastAttemptAt = Number.NEGATIVE_INFINITY;
      this.lastAttemptFingerprint = "";
      this.stopReason = null;
    }
    this.now = now;
    this.currentReady = ready;
    this.currentFingerprint = fingerprint;
    if (!ready) {
      this.readySince = null;
      return false;
    }
    if (this.readySince === null) this.readySince = now;
    if (
      this.passes > 0
      && now - this.lastAttemptAt >= this.cooldownMs
      && fingerprint === this.lastAttemptFingerprint
    ) {
      this.stopReason = "engine-protected-or-no-progress";
    }
    if (this.passes >= this.maxPasses) this.stopReason = "bounded-pass-limit";
    return this.stopReason === null
      && now - this.readySince >= this.settleMs
      && now - this.lastAttemptAt >= this.cooldownMs;
  }

  claim(epoch: number): boolean {
    if (
      epoch !== this.epoch
      || !this.currentReady
      || this.stopReason !== null
      || this.passes >= this.maxPasses
      || this.readySince === null
      || this.now - this.readySince < this.settleMs
      || this.now - this.lastAttemptAt < this.cooldownMs
    ) {
      return false;
    }
    this.passes += 1;
    this.lastAttemptAt = this.now;
    this.lastAttemptFingerprint = this.currentFingerprint;
    return true;
  }
}

export function retentionDecision(
  sample: CacheSample,
  settings: CachePolicySettings,
): TrimDecision {
  if (!sample.enabled) return "disabled";
  if (!sample.ready) return "not-ready";
  if (sample.loaded <= sample.visible) return "no-spare";
  if (sample.memoryBytes > settings.hardBytes) return "hard-pressure";
  if (sample.elapsedMs < settings.retainMs) return "retain-recent-view";
  if (sample.memoryBytes <= settings.softBytes) return "under-soft-budget";
  return "idle-over-soft-budget";
}

export function cachePolicySettings(
  options: Partial<CachePolicySettings> = {},
): CachePolicySettings {
  const result: CachePolicySettings = {
    modelKey: "a1",
    resourcePrefix: "/a1/",
    retainMs: 30000,
    softBytes: 256 * 1024 * 1024,
    hardBytes: 1024 * 1024 * 1024,
    ...options,
  };
  if (!result.resourcePrefix.startsWith("/")) throw new Error("invalid resource prefix");
  for (const field of ["retainMs", "softBytes", "hardBytes"] as const) {
    if (!Number.isFinite(result[field]) || result[field] < 0) {
      throw new Error(`invalid cache setting: ${field}`);
    }
  }
  if (result.hardBytes < result.softBytes) throw new Error("hard budget below soft budget");
  return result;
}

interface AppStateSnapshot {
  mode: DisplayMode;
  activeView: ViewName;
}

interface CacheTelemetry {
  policy: "baseline" | "convergent-demand-v55";
  loads: number;
  unloads: number;
  reloads: number;
  trims: number;
  trimDecision: string;
  epoch: number;
  convergencePasses: number;
  convergenceStopReason: string | null;
}

export interface CachePolicyController {
  attach: (state: ModelState) => void;
  optionsFor: (key: ModelKey) => Partial<Cesium.Cesium3DTileset.ConstructorOptions>;
  snapshot: () => object;
}

export function createCachePolicy(
  viewer: Cesium.Viewer,
  states: ModelStates,
  getAppState: () => AppStateSnapshot,
  query: URLSearchParams,
  monitor: RuntimeMonitor,
  options: Partial<CachePolicySettings> = {},
): CachePolicyController {
  const settings = cachePolicySettings(options);
  const enabled = query.get("policy") !== "baseline";
  const gate = new SettledTrimGate();
  const loadedOnce = new WeakSet<Cesium.Cesium3DTile>();
  const tileIds = new WeakMap<Cesium.Cesium3DTile, number>();
  const telemetry: CacheTelemetry = {
    policy: enabled ? "convergent-demand-v55" : "baseline",
    loads: 0,
    unloads: 0,
    reloads: 0,
    trims: 0,
    trimDecision: enabled ? "not-ready" : "disabled",
    epoch: 0,
    convergencePasses: 0,
    convergenceStopReason: null,
  };
  let nextTileId = 0;
  let lastActivity = performance.now();
  let cameraValues: number[] | undefined;
  let cameraRevision = 0;
  let viewStarted = performance.now();
  let lastSignature = "";

  function state(): ModelState {
    return states[settings.modelKey];
  }

  function signature(): string {
    const camera = viewer.camera;
    const values = [
      camera.position.x,
      camera.position.y,
      camera.position.z,
      camera.direction.x,
      camera.direction.y,
      camera.direction.z,
      camera.up.x,
      camera.up.y,
      camera.up.z,
    ];
    if (
      !cameraValues
      || values.some((value, index) => Math.abs(value - cameraValues![index]) > (index < 3 ? 0.001 : 1e-7))
    ) {
      cameraValues = values;
      cameraRevision += 1;
    }
    const current = state();
    const visibleIds = [...current.lastVisibleTiles].map((tile) => {
      let id = tileIds.get(tile);
      if (id === undefined) {
        id = ++nextTileId;
        tileIds.set(tile, id);
      }
      return id;
    }).sort((left, right) => left - right);
    const app = getAppState();
    return [
      cameraRevision,
      app.mode,
      app.activeView,
      current.tileset?.maximumScreenSpaceError,
      viewer.scene.canvas.clientWidth,
      viewer.scene.canvas.clientHeight,
      visibleIds.join(","),
    ].join(":");
  }

  function ready(now = performance.now()): boolean {
    const current = state();
    const tileset = current.tileset;
    return !document.hidden
      && getAppState().mode !== "flicker"
      && Boolean(tileset && !tileset.isDestroyed() && tileset.show && tileset.tilesLoaded)
      && current.pending === 0
      && current.processing === 0
      && current.failures === 0
      && current.lastVisibleTiles.size > 0
      && now - lastActivity >= 500;
  }

  function snapshot(): object {
    const current = state();
    const tileset = current.tileset;
    const resource = monitor.resources.totals(settings.resourcePrefix);
    return {
      ...telemetry,
      settings,
      enabled,
      pending: current.pending,
      processing: current.processing,
      failures: current.failures,
      visibleTiles: current.lastVisibleTiles.size,
      loadedTiles: current.loadedTiles.size,
      nonVisibleCachedTiles: Math.max(0, current.loadedTiles.size - current.lastVisibleTiles.size),
      memoryBytes: tileset?.totalMemoryUsageInBytes ?? 0,
      resource,
      activeView: getAppState().activeView,
      mode: getAppState().mode,
      capturedAt: new Date().toISOString(),
    };
  }

  function tick(): void {
    const current = state();
    const tileset = current.tileset;
    if (!tileset || tileset.isDestroyed()) return;
    const now = performance.now();
    const currentSignature = signature();
    if (currentSignature !== lastSignature) {
      viewStarted = now;
      lastSignature = currentSignature;
    }
    const fingerprint = `${current.loadedTiles.size}:${current.lastVisibleTiles.size}:${tileset.totalMemoryUsageInBytes}`;
    const eligible = gate.sample(now, currentSignature, ready(now), fingerprint);
    telemetry.epoch = gate.epoch;
    telemetry.convergencePasses = gate.passes;
    telemetry.convergenceStopReason = gate.stopReason;
    telemetry.trimDecision = retentionDecision({
      enabled,
      ready: eligible && ready(now),
      elapsedMs: now - viewStarted,
      loaded: current.loadedTiles.size,
      visible: current.lastVisibleTiles.size,
      memoryBytes: tileset.totalMemoryUsageInBytes,
    }, settings);
    if (
      (telemetry.trimDecision === "hard-pressure" || telemetry.trimDecision === "idle-over-soft-budget")
      && ready(now)
      && signature() === currentSignature
      && gate.claim(gate.epoch)
      && current.loadedTiles.size > current.lastVisibleTiles.size
    ) {
      tileset.trimLoadedTiles();
      telemetry.trims += 1;
      viewer.scene.requestRender();
    }
  }

  monitor.startLoop(tick);

  return {
    attach(modelState) {
      if (modelState.key !== settings.modelKey || !modelState.tileset) return;
      modelState.tileset.tileLoad.addEventListener((tile) => {
        telemetry.loads += 1;
        if (loadedOnce.has(tile)) telemetry.reloads += 1;
        loadedOnce.add(tile);
        lastActivity = performance.now();
        monitor.touch();
      });
      modelState.tileset.tileUnload.addEventListener(() => {
        telemetry.unloads += 1;
        lastActivity = performance.now();
        monitor.touch();
      });
    },
    optionsFor(key) {
      if (!enabled || key !== settings.modelKey) return {};
      return {
        cacheBytes: settings.hardBytes,
        maximumCacheOverflowBytes: Math.min(settings.softBytes, 256 * 1024 * 1024),
        skipLevelOfDetail: true,
        immediatelyLoadDesiredLevelOfDetail: true,
        loadSiblings: false,
      };
    },
    snapshot,
  };
}
