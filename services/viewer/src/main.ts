import * as Cesium from "cesium";

import { createCachePolicy } from "./cache-policy";
import { startDiagnostics } from "./diagnostics";
import { createUi } from "./dom";
import { setupMeasurement } from "./measurement";
import {
  createModelStates,
  markModelStarted,
  markVisibleModelsStarted,
  registerTileset,
  startModelMetrics,
} from "./model-state";
import { RuntimeMonitor } from "./runtime-monitor";
import "cesium/Build/Cesium/Widgets/widgets.css";
import "./styles.css";
import type {
  ComparisonApi,
  DisplayMode,
  ModelKey,
  ModelLabels,
  ViewName,
  ViewerConfig,
} from "./types";

performance.setResourceTimingBufferSize(10000);

const SHARED_TILESET_CACHE_BYTES = 512 * 1024 * 1024;

function isDisplayMode(value: string | null | undefined): value is DisplayMode {
  return value === "split" || value === "terra" || value === "a1" || value === "flicker";
}

function isViewName(value: string | undefined): value is ViewName {
  return value === "overview"
    || value === "building"
    || value === "edge"
    || value === "texture"
    || value === "top";
}

const query = new URLSearchParams(window.location.search);
const captureMode = query.get("capture") === "1";
const requestedSse = Number(query.get("sse"));
const modeParameter = query.get("mode");
const requestedMode: DisplayMode = isDisplayMode(modeParameter) ? modeParameter : "a1";
const previewProfile = query.get("preview");
const previewMode = previewProfile === "1" || previewProfile === "medium";
const ui = createUi();

const viewer = new Cesium.Viewer("cesiumContainer", {
  animation: false,
  baseLayer: false,
  baseLayerPicker: false,
  fullscreenButton: false,
  geocoder: false,
  homeButton: false,
  infoBox: false,
  navigationHelpButton: false,
  sceneModePicker: false,
  selectionIndicator: false,
  shadows: false,
  timeline: false,
  terrainProvider: new Cesium.EllipsoidTerrainProvider(),
  contextOptions: { webgl: { preserveDrawingBuffer: captureMode } },
});

viewer.clock.currentTime = Cesium.JulianDate.fromIso8601("2026-04-21T04:00:00Z");
viewer.clock.shouldAnimate = false;
viewer.scene.globe.show = false;
viewer.scene.backgroundColor = Cesium.Color.fromCssColorString("#07100d");
if (viewer.scene.skyAtmosphere) viewer.scene.skyAtmosphere.show = false;
if (viewer.scene.skyBox) viewer.scene.skyBox.show = false;
if (viewer.scene.sun) viewer.scene.sun.show = false;
if (viewer.scene.moon) viewer.scene.moon.show = false;
viewer.scene.highDynamicRange = false;
viewer.scene.postProcessStages.fxaa.enabled = true;
viewer.scene.screenSpaceCameraController.enableCollisionDetection = false;
viewer.screenSpaceEventHandler.removeInputAction(Cesium.ScreenSpaceEventType.LEFT_DOUBLE_CLICK);

const modelStates = createModelStates();
let ensureModels: ((mode: DisplayMode) => Promise<void>) | undefined;
const labels: ModelLabels = { terra: "左侧模型", a1: "右侧模型" };
const appState: {
  config?: ViewerConfig;
  mode: DisplayMode;
  flickerTimer?: number;
  flickerPhase: ModelKey;
  activeView: ViewName;
  toastTimer?: number;
} = {
  mode: requestedMode,
  flickerPhase: "terra",
  activeView: "overview",
};
const runtimeMonitor = new RuntimeMonitor(window.location.href, ["/terra/", "/a1/"]);
const cachePolicy = createCachePolicy(
  viewer,
  modelStates,
  () => ({ mode: appState.mode, activeView: appState.activeView }),
  query,
  runtimeMonitor,
);

function showToast(message: string, duration = 2600): void {
  window.clearTimeout(appState.toastTimer);
  ui.toast.textContent = message;
  ui.toast.classList.add("show");
  appState.toastTimer = window.setTimeout(() => ui.toast.classList.remove("show"), duration);
}

function stopFlicker(): void {
  window.clearInterval(appState.flickerTimer);
  appState.flickerTimer = undefined;
}

function applyVisibility(): void {
  const terra = modelStates.terra.tileset;
  const a1 = modelStates.a1.tileset;

  stopFlicker();
  if (terra) {
    terra.splitDirection = Cesium.SplitDirection.NONE;
    terra.show = appState.mode !== "a1";
  }
  if (a1) {
    a1.splitDirection = Cesium.SplitDirection.NONE;
    a1.show = appState.mode !== "terra";
  }

  if (appState.mode === "split" && terra && a1) {
    terra.show = true;
    a1.show = true;
    terra.splitDirection = Cesium.SplitDirection.LEFT;
    a1.splitDirection = Cesium.SplitDirection.RIGHT;
  } else if (appState.mode === "flicker" && terra && a1) {
    appState.flickerPhase = "terra";
    terra.show = true;
    a1.show = false;
    appState.flickerTimer = window.setInterval(() => {
      appState.flickerPhase = appState.flickerPhase === "terra" ? "a1" : "terra";
      terra.show = appState.flickerPhase === "terra";
      a1.show = appState.flickerPhase === "a1";
    }, 650);
  }
  markVisibleModelsStarted(modelStates);
}

function setDisplayMode(mode: DisplayMode, announce = true): void {
  appState.mode = mode;
  document.body.className = `mode-${mode}`;
  document.querySelectorAll<HTMLButtonElement>("[data-mode]").forEach((button) => {
    button.classList.toggle("active", button.dataset.mode === mode);
  });
  applyVisibility();
  if (requiredModels(mode).some((key) => !modelStates[key].tileset) && ensureModels) {
    void ensureModels(mode).catch((error) => {
      showToast(`模型加载失败：${String(error)}`, 5200);
      console.error(error);
    });
  }
  if (announce) {
    const modeLabels: Record<DisplayMode, string> = {
      split: "左右裁切对比",
      terra: `仅显示 ${labels.terra}`,
      a1: `仅显示 ${labels.a1}`,
      flicker: "650 ms 闪烁切换",
    };
    showToast(modeLabels[mode]);
  }
}

function offsetCenter(
  center: Cesium.Cartesian3,
  east: number,
  north: number,
  up: number,
): Cesium.Cartesian3 {
  const frame = Cesium.Transforms.eastNorthUpToFixedFrame(center);
  return Cesium.Matrix4.multiplyByPoint(
    frame,
    new Cesium.Cartesian3(east, north, up),
    new Cesium.Cartesian3(),
  );
}

function goToView(name: ViewName, duration = 1.1): void {
  const source = modelStates.a1.tileset || modelStates.terra.tileset;
  if (!source) return;
  const { center, radius } = source.boundingSphere;
  const presets: Record<ViewName, {
    center: Cesium.Cartesian3;
    heading: number;
    pitch: number;
    range: number;
  }> = {
    overview: { center, heading: 25, pitch: -34, range: radius * 2.25 },
    building: {
      center: offsetCenter(center, radius * 0.08, radius * -0.1, 0),
      heading: 42,
      pitch: -19,
      range: radius * 0.72,
    },
    edge: {
      center: offsetCenter(center, radius * -0.34, radius * 0.24, 0),
      heading: 126,
      pitch: -26,
      range: radius * 0.78,
    },
    texture: {
      center: offsetCenter(center, radius * -0.34, radius * 0.24, 0),
      heading: 126,
      pitch: -18,
      range: radius * 0.42,
    },
    top: { center, heading: 0, pitch: -89.5, range: radius * 1.65 },
  };
  const preset = presets[name];
  appState.activeView = name;
  document.querySelectorAll<HTMLButtonElement>("[data-view]").forEach((button) => {
    button.classList.toggle("active", button.dataset.view === name);
  });
  viewer.camera.flyToBoundingSphere(
    new Cesium.BoundingSphere(preset.center, Math.max(radius * 0.05, 10)),
    {
      duration,
      offset: new Cesium.HeadingPitchRange(
        Cesium.Math.toRadians(preset.heading),
        Cesium.Math.toRadians(preset.pitch),
        preset.range,
      ),
    },
  );
}

function setLabel(name: string, value: string): void {
  document.querySelectorAll<HTMLElement>(`[data-label="${name}"]`).forEach((node) => {
    node.textContent = value;
  });
}

function applyConfigLabels(config: ViewerConfig): void {
  labels.terra = config.models.terra.label;
  labels.a1 = config.models.a1.label;
  document.title = `${config.dataset} · ${labels.terra} / ${labels.a1} 对比台`;
  setLabel("title", `${labels.terra} × ${labels.a1} 对比台`);
  setLabel("terra", labels.terra);
  setLabel("a1", labels.a1);
  setLabel("mode-terra", `仅 ${labels.terra}`);
  setLabel("mode-a1", `仅 ${labels.a1}`);
  setLabel("target-terra", `在 ${labels.terra} 取点`);
  setLabel("target-a1", `在 ${labels.a1} 取点`);
  setLabel("result-terra", `${labels.terra} 结果`);
  setLabel("result-a1", `${labels.a1} 结果`);
  setLabel("split-terra", `LEFT · ${labels.terra}`);
  setLabel("split-a1", `${labels.a1} · RIGHT`);
}

document.querySelectorAll<HTMLButtonElement>("[data-mode]").forEach((button) => {
  button.addEventListener("click", () => {
    if (isDisplayMode(button.dataset.mode)) setDisplayMode(button.dataset.mode);
  });
});
document.querySelectorAll<HTMLButtonElement>("[data-view]").forEach((button) => {
  button.addEventListener("click", () => {
    if (isViewName(button.dataset.view)) goToView(button.dataset.view);
  });
});

function setSplitPosition(value: number): void {
  viewer.scene.splitPosition = value / 100;
  ui.splitLine.style.left = `${value}%`;
}
ui.splitRange.addEventListener("input", () => setSplitPosition(Number(ui.splitRange.value)));
setSplitPosition(50);

ui.sseRange.addEventListener("input", () => {
  const value = Number(ui.sseRange.value);
  ui.sseValue.value = String(value);
  for (const state of Object.values(modelStates)) {
    if (state.tileset) state.tileset.maximumScreenSpaceError = value;
  }
});

let fpsFrames = 0;
let fpsWindowStart = performance.now();
viewer.scene.postRender.addEventListener(() => {
  fpsFrames += 1;
});
runtimeMonitor.startLoop(() => {
  const now = performance.now();
  const elapsed = now - fpsWindowStart;
  if (elapsed < 750) return;
  runtimeMonitor.setText(
    ui.fps,
    fpsFrames > 0 ? `${Math.round((fpsFrames * 1000) / elapsed)} FPS` : "静止",
  );
  fpsFrames = 0;
  fpsWindowStart = now;
});
runtimeMonitor.onLifecycle(
  () => {
    stopFlicker();
    fpsFrames = 0;
    fpsWindowStart = performance.now();
  },
  () => {
    fpsFrames = 0;
    fpsWindowStart = performance.now();
    if (appState.mode === "flicker") applyVisibility();
  },
);
startModelMetrics(viewer, modelStates, runtimeMonitor);

const measurement = setupMeasurement({
  viewer,
  ui,
  labels,
  showToast,
  setDisplayMode,
});

function requiredModels(mode: DisplayMode): ModelKey[] {
  if (mode === "terra" || mode === "a1") return [mode];
  return ["terra", "a1"];
}

async function fetchConfig(): Promise<ViewerConfig> {
  const response = await fetch("/api/config", { cache: "no-store" });
  if (!response.ok) throw new Error(`配置请求失败：HTTP ${response.status}`);
  return response.json() as Promise<ViewerConfig>;
}

async function loadComparison(): Promise<void> {
  try {
    const config = await fetchConfig();
    appState.config = config;
    applyConfigLabels(config);
    const maximumScreenSpaceError = Number.isFinite(requestedSse)
      && requestedSse >= 1
      && requestedSse <= 128
      ? requestedSse
      : config.maximumScreenSpaceError;
    if (previewProfile === "1") config.models.a1.tilesetUrl = "/api/a1-preview";
    if (previewProfile === "medium") config.models.a1.tilesetUrl = "/api/a1-medium";
    ui.sseRange.value = String(maximumScreenSpaceError);
    ui.sseValue.value = String(maximumScreenSpaceError);

    const commonOptions = {
      maximumScreenSpaceError,
      cacheBytes: SHARED_TILESET_CACHE_BYTES,
      maximumCacheOverflowBytes: SHARED_TILESET_CACHE_BYTES,
      dynamicScreenSpaceError: true,
      foveatedScreenSpaceError: true,
      foveatedTimeDelay: 0,
      progressiveResolutionHeightFraction: 0.3,
      skipLevelOfDetail: false,
      preloadWhenHidden: false,
      preloadFlightDestinations: true,
      preferLeaves: false,
      cullWithChildrenBounds: false,
      cullRequestsWhileMoving: false,
      shadows: Cesium.ShadowMode.DISABLED,
    };
    const modelLoads: Partial<Record<ModelKey, Promise<void>>> = {};
    const loadModel = (key: ModelKey): Promise<void> => {
      if (modelStates[key].tileset) return Promise.resolve();
      if (modelLoads[key]) return modelLoads[key];

      const load = (async () => {
        markModelStarted(modelStates[key]);
        const tileset = await Cesium.Cesium3DTileset.fromUrl(
          config.models[key].tilesetUrl,
          { ...commonOptions, ...cachePolicy.optionsFor(key) },
        );
        viewer.scene.primitives.add(tileset);
        registerTileset(modelStates[key], tileset, {
          onProgress(state) {
            if (!ui.loading.classList.contains("hidden") && (state.pending || state.processing)) {
              ui.loadingMessage.textContent = `${labels[state.key]} 正在加载：${state.pending} 个请求，${state.processing} 个内容处理中。`;
            }
          },
          onInitialReady() {
            if (requiredModels(appState.mode).every((required) => modelStates[required].initialReady)) {
              ui.loading.classList.add("hidden");
            }
          },
        });
        cachePolicy.attach(modelStates[key]);
      })().catch((error) => {
        delete modelLoads[key];
        throw error;
      });
      modelLoads[key] = load;
      return load;
    };
    ensureModels = async (mode) => {
      await Promise.all(requiredModels(mode).map(loadModel));
      if (appState.mode === mode) applyVisibility();
    };
    await ensureModels(appState.mode);
    setDisplayMode(appState.mode, false);
    goToView("overview", 0);

    ui.loadingMessage.textContent = previewMode && appState.mode === "a1"
      ? previewProfile === "medium"
        ? `正在加载 ${labels.a1} 中清预览（最多 17 个内容瓦片）。`
        : `正在加载 ${labels.a1} 单瓦片预览。`
      : appState.mode === "terra"
        ? `先加载 ${labels.terra}；显示后可通过对比模式加载两套模型。`
        : "两套瓦片集已连接，正在准备完整首屏；加载完成前暂不展示不完整的局部瓦片。";

    const comparisonApi: ComparisonApi = {
      viewer,
      modelStates,
      measurement,
      goToView,
      setDisplayMode,
      runtime: {
        version: 56,
        snapshot: () => ({ monitor: runtimeMonitor.snapshot(), cache: cachePolicy.snapshot() }),
      },
    };
    (window as Window & { __comparison?: ComparisonApi }).__comparison = comparisonApi;

    window.setTimeout(() => {
      if (ui.loading.classList.contains("hidden")) return;
      const outstanding = Object.values(modelStates)
        .reduce((total, state) => total + state.pending + state.processing, 0);
      ui.loading.classList.add("hidden");
      showToast(
        outstanding > 0
          ? `已进入页面，剩余 ${outstanding} 个模型请求在后台继续加载。`
          : "模型页面已就绪。",
        5200,
      );
    }, 20000);

    startDiagnostics(
      viewer,
      modelStates.a1,
      () => ({ activeView: appState.activeView, mode: appState.mode }),
      runtimeMonitor,
      cachePolicy,
    );
  } catch (error) {
    ui.loadingMessage.textContent = `加载失败：${String(error)}`;
    console.error(error);
  }
}

void loadComparison();
