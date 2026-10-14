import type * as Cesium from "cesium";

import type { RuntimeMonitor } from "./runtime-monitor";
import type { ModelKey, ModelState, ModelStates } from "./types";

export function createModelState(key: ModelKey): ModelState {
  return {
    key,
    loadedTiles: new Map(),
    visibleTiles: new Set(),
    lastVisibleTiles: new Set(),
    pending: 0,
    processing: 0,
    initialReady: false,
    failures: 0,
  };
}

export function createModelStates(): ModelStates {
  return {
    terra: createModelState("terra"),
    a1: createModelState("a1"),
  };
}

function tileTriangles(tile: Cesium.Cesium3DTile): number {
  const value = tile.content?.trianglesLength;
  return Number.isFinite(value) ? value : 0;
}

export function markModelStarted(state: ModelState, now = performance.now()): void {
  if (!Number.isFinite(now) || now < 0) throw new Error("invalid model start time");
  if (state.startedAt === undefined) state.startedAt = now;
}

export function markVisibleModelsStarted(states: ModelStates, now = performance.now()): void {
  for (const state of Object.values(states)) {
    if (state.tileset?.show) markModelStarted(state, now);
  }
}

export interface TilesetCallbacks {
  onProgress: (state: ModelState) => void;
  onInitialReady: (state: ModelState) => void;
}

export function registerTileset(
  state: ModelState,
  tileset: Cesium.Cesium3DTileset,
  callbacks: TilesetCallbacks,
): void {
  state.tileset = tileset;
  tileset.loadProgress.addEventListener((pending, processing) => {
    state.pending = pending;
    state.processing = processing;
    callbacks.onProgress(state);
  });
  tileset.initialTilesLoaded.addEventListener(() => {
    if (state.initialLoadMs === undefined) {
      state.initialLoadMs = performance.now() - (state.startedAt ?? performance.now());
    }
    state.initialReady = true;
    callbacks.onInitialReady(state);
  });
  tileset.tileLoad.addEventListener((tile) => {
    state.loadedTiles.set(tile, tileTriangles(tile));
  });
  tileset.tileUnload.addEventListener((tile) => {
    state.loadedTiles.delete(tile);
    state.visibleTiles.delete(tile);
    state.lastVisibleTiles.delete(tile);
  });
  tileset.tileVisible.addEventListener((tile) => {
    state.visibleTiles.add(tile);
  });
  tileset.tileFailed.addEventListener((error) => {
    state.failures += 1;
    console.error(`[${state.key}] tile failed`, error);
  });
}

function formatInteger(value: number): string {
  return new Intl.NumberFormat("zh-CN", { maximumFractionDigits: 0 }).format(value || 0);
}

function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 MB";
  return `${(bytes / 1048576).toFixed(bytes >= 104857600 ? 0 : 1)} MB`;
}

function networkStats(
  key: ModelKey,
  monitor: RuntimeMonitor,
): { requests: number; bytes: number } {
  const totals = monitor.resources.totals(key === "terra" ? "/terra/" : "/a1/");
  return {
    requests: totals.requests,
    bytes: totals.encodedBodyBytes || totals.decodedBodyBytes,
  };
}

function sumTriangles(tiles: Set<Cesium.Cesium3DTile>): number {
  let sum = 0;
  for (const tile of tiles) sum += tileTriangles(tile);
  return sum;
}

function setMetric(
  key: ModelKey,
  metric: string,
  value: string,
  monitor: RuntimeMonitor,
): void {
  const metricElement = document.querySelector(
    `[data-metrics="${key}"] [data-key="${metric}"]`,
  );
  monitor.setText(metricElement, value);
}

function updateMetrics(modelStates: ModelStates, monitor: RuntimeMonitor): void {
  for (const key of ["terra", "a1"] as const) {
    const state = modelStates[key];
    const network = networkStats(key, monitor);
    const visibleTriangles = sumTriangles(state.lastVisibleTiles);
    setMetric(key, "tiles", `${state.lastVisibleTiles.size} / ${state.loadedTiles.size}`, monitor);
    setMetric(key, "queue", `${state.pending} / ${state.processing}`, monitor);
    setMetric(key, "triangles", formatInteger(visibleTriangles), monitor);
    setMetric(key, "memory", formatBytes(state.tileset?.totalMemoryUsageInBytes || 0), monitor);
    setMetric(key, "network", `${network.requests} / ${formatBytes(network.bytes)}`, monitor);
    setMetric(
      key,
      "loadTime",
      state.initialLoadMs === undefined ? "等待" : `${(state.initialLoadMs / 1000).toFixed(2)} s`,
      monitor,
    );

    let status = "初始化";
    if (state.tileset) status = state.pending || state.processing ? "加载中" : "当前视图就绪";
    if (state.failures) status = `${state.failures} 个内容失败`;
    setMetric(key, "status", status, monitor);
  }
}

export function startModelMetrics(
  viewer: Cesium.Viewer,
  modelStates: ModelStates,
  monitor: RuntimeMonitor,
): void {
  viewer.scene.postRender.addEventListener(() => {
    monitor.touch();
    for (const state of Object.values(modelStates)) {
      state.lastVisibleTiles = new Set(state.visibleTiles);
      state.visibleTiles.clear();
    }
  });
  monitor.startLoop(() => updateMetrics(modelStates, monitor));
}
