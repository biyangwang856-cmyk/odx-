import * as Cesium from "cesium";

import { element, type UiElements } from "./dom";
import type {
  MeasurementMode,
  MeasurementState,
  ModelKey,
  ModelLabels,
} from "./types";

interface MeasurementOptions {
  viewer: Cesium.Viewer;
  ui: UiElements;
  labels: ModelLabels;
  showToast: (message: string, duration?: number) => void;
  setDisplayMode: (mode: ModelKey, announce?: boolean) => void;
}

function isMeasurementMode(value: string | undefined): value is MeasurementMode {
  return value === "distance" || value === "height" || value === "area";
}

function isModelKey(value: string | undefined): value is ModelKey {
  return value === "terra" || value === "a1";
}

function formatMeters(value: number): string {
  if (Math.abs(value) >= 1000) return `${(value / 1000).toFixed(3)} km`;
  return `${value.toFixed(3)} m`;
}

export function setupMeasurement(options: MeasurementOptions): MeasurementState {
  const { viewer, ui, labels, showToast, setDisplayMode } = options;
  const measurement: MeasurementState = {
    target: "terra",
    points: [],
    entities: [],
    results: {},
  };

  function measurementColor(alpha = 1): Cesium.Color {
    const color = measurement.target === "terra"
      ? Cesium.Color.fromCssColorString("#efb34d")
      : Cesium.Color.fromCssColorString("#50d7a0");
    return color.withAlpha(alpha);
  }

  function updateMeasurementResults(): void {
    for (const target of ["terra", "a1"] as const) {
      const value = measurement.results[target];
      const output = target === "terra" ? ui.terraResult : ui.a1Result;
      output.textContent = value ? value.result : "尚未量测";
    }
  }

  function clearCurrentMeasurement(): void {
    for (const entity of measurement.entities) viewer.entities.remove(entity);
    measurement.entities = [];
    measurement.points = [];
    if (measurement.dynamicEntity) viewer.entities.remove(measurement.dynamicEntity);
    measurement.dynamicEntity = undefined;
  }

  function setMeasurementTarget(target: ModelKey, clear = true): void {
    if (clear) clearCurrentMeasurement();
    measurement.target = target;
    document.querySelectorAll<HTMLButtonElement>("[data-target]").forEach((button) => {
      button.classList.toggle("active", button.dataset.target === target);
    });
    if (measurement.mode) {
      setDisplayMode(target, false);
      showToast(`量测目标：${labels[target]}`);
    }
  }

  function setMeasurementMode(mode: MeasurementMode): void {
    clearCurrentMeasurement();
    measurement.mode = measurement.mode === mode ? undefined : mode;
    document.querySelectorAll<HTMLButtonElement>("[data-measure]").forEach((button) => {
      button.classList.toggle("active", button.dataset.measure === measurement.mode);
    });
    const messages: Record<MeasurementMode, string> = {
      distance: "距离：在模型表面依次点击两个点。",
      height: "高差：先点基准点，再点目标点。",
      area: "面积：依次点击至少三个点，右键或点击“完成当前”。",
    };
    ui.measureHint.textContent = measurement.mode
      ? messages[measurement.mode]
      : "选择距离、高差或面积后，先指定模型再在表面取点。";
    if (measurement.mode) {
      setMeasurementTarget(measurement.target, false);
      showToast(messages[measurement.mode]);
    }
  }

  function pickModelPosition(windowPosition: Cesium.Cartesian2): Cesium.Cartesian3 | undefined {
    if (!viewer.scene.pickPositionSupported) return undefined;
    const picked = viewer.scene.pick(windowPosition);
    if (!Cesium.defined(picked)) return undefined;
    const position = viewer.scene.pickPosition(windowPosition);
    return Cesium.defined(position) ? Cesium.Cartesian3.clone(position) : undefined;
  }

  function addMeasurementPoint(position: Cesium.Cartesian3): void {
    const index = measurement.points.length + 1;
    const entity = viewer.entities.add({
      position,
      point: {
        color: measurementColor(),
        outlineColor: Cesium.Color.BLACK,
        outlineWidth: 2,
        pixelSize: 10,
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
      },
      label: {
        text: String(index),
        font: "700 12px sans-serif",
        fillColor: Cesium.Color.WHITE,
        outlineColor: Cesium.Color.BLACK,
        outlineWidth: 3,
        style: Cesium.LabelStyle.FILL_AND_OUTLINE,
        pixelOffset: new Cesium.Cartesian2(0, -20),
        disableDepthTestDistance: Number.POSITIVE_INFINITY,
      },
    });
    measurement.entities.push(entity);
  }

  function rebuildMeasurementShape(): void {
    if (measurement.dynamicEntity) viewer.entities.remove(measurement.dynamicEntity);
    measurement.dynamicEntity = undefined;
    if (measurement.points.length < 2) return;

    const positions = [...measurement.points];
    if (measurement.mode === "area" && positions.length >= 3) positions.push(positions[0]);
    measurement.dynamicEntity = viewer.entities.add({
      polyline: {
        positions,
        width: 3,
        material: measurementColor(),
        clampToGround: false,
      },
      polygon: measurement.mode === "area" && measurement.points.length >= 3
        ? {
            hierarchy: new Cesium.PolygonHierarchy(measurement.points),
            material: measurementColor(0.14),
            perPositionHeight: true,
          }
        : undefined,
    });
  }

  function horizontalArea(points: Cesium.Cartesian3[]): number {
    const frame = Cesium.Transforms.eastNorthUpToFixedFrame(points[0]);
    const inverse = Cesium.Matrix4.inverseTransformation(frame, new Cesium.Matrix4());
    const local = points.map((point) => (
      Cesium.Matrix4.multiplyByPoint(inverse, point, new Cesium.Cartesian3())
    ));
    let twiceArea = 0;
    for (let index = 0; index < local.length; index += 1) {
      const current = local[index];
      const next = local[(index + 1) % local.length];
      twiceArea += current.x * next.y - next.x * current.y;
    }
    return Math.abs(twiceArea) * 0.5;
  }

  function horizontalDistance(first: Cesium.Cartesian3, second: Cesium.Cartesian3): number {
    const firstCartographic = Cesium.Cartographic.fromCartesian(first);
    const secondCartographic = Cesium.Cartographic.fromCartesian(second);
    return new Cesium.EllipsoidGeodesic(firstCartographic, secondCartographic).surfaceDistance;
  }

  function finishMeasurement(): void {
    const { mode, points, target } = measurement;
    if (!mode) {
      showToast("请先选择量测类型");
      return;
    }
    const required = mode === "area" ? 3 : 2;
    if (points.length < required) {
      showToast(`当前至少需要 ${required} 个有效表面点`);
      return;
    }

    let result: string;
    if (mode === "distance") {
      let spatial = 0;
      let horizontal = 0;
      for (let index = 1; index < points.length; index += 1) {
        spatial += Cesium.Cartesian3.distance(points[index - 1], points[index]);
        horizontal += horizontalDistance(points[index - 1], points[index]);
      }
      result = `空间 ${formatMeters(spatial)}\n水平 ${formatMeters(horizontal)}`;
    } else if (mode === "height") {
      const first = Cesium.Cartographic.fromCartesian(points[0]);
      const last = Cesium.Cartographic.fromCartesian(points[points.length - 1]);
      result = `高差 ${formatMeters(last.height - first.height)}\n水平 ${formatMeters(horizontalDistance(points[0], points[points.length - 1]))}`;
    } else {
      let perimeter = 0;
      for (let index = 0; index < points.length; index += 1) {
        perimeter += horizontalDistance(points[index], points[(index + 1) % points.length]);
      }
      result = `水平面积 ${horizontalArea(points).toFixed(3)} m²\n周长 ${formatMeters(perimeter)}`;
    }

    measurement.results[target] = { mode, result };
    updateMeasurementResults();
    const measureLabel = mode === "distance" ? "距离" : mode === "height" ? "高差" : "面积";
    showToast(`${labels[target]} ${measureLabel}已记录`);
    measurement.points = [];
  }

  function clearAllMeasurements(): void {
    clearCurrentMeasurement();
    measurement.results = {};
    updateMeasurementResults();
    showToast("量测记录已清除");
  }

  document.querySelectorAll<HTMLButtonElement>("[data-measure]").forEach((button) => {
    button.addEventListener("click", () => {
      if (isMeasurementMode(button.dataset.measure)) setMeasurementMode(button.dataset.measure);
    });
  });
  document.querySelectorAll<HTMLButtonElement>("[data-target]").forEach((button) => {
    button.addEventListener("click", () => {
      if (isModelKey(button.dataset.target)) setMeasurementTarget(button.dataset.target);
    });
  });

  const measureHandler = new Cesium.ScreenSpaceEventHandler(viewer.scene.canvas);
  measureHandler.setInputAction((event: { position: Cesium.Cartesian2 }) => {
    if (!measurement.mode) return;
    const position = pickModelPosition(event.position);
    if (!position) {
      showToast("未拾取到模型表面，请在可见纹理区域取点");
      return;
    }
    measurement.points.push(position);
    addMeasurementPoint(position);
    rebuildMeasurementShape();
    if (measurement.mode !== "area" && measurement.points.length === 2) finishMeasurement();
  }, Cesium.ScreenSpaceEventType.LEFT_CLICK);
  measureHandler.setInputAction(() => {
    if (measurement.mode === "area") finishMeasurement();
  }, Cesium.ScreenSpaceEventType.RIGHT_CLICK);

  element<HTMLButtonElement>("#finishMeasure").addEventListener("click", finishMeasurement);
  element<HTMLButtonElement>("#clearMeasure").addEventListener("click", clearAllMeasurements);
  return measurement;
}
