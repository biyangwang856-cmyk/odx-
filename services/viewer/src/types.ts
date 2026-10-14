import type * as Cesium from "cesium";

export type ModelKey = "terra" | "a1";
export type DisplayMode = ModelKey | "split" | "flicker";
export type ViewName = "overview" | "building" | "edge" | "texture" | "top";
export type MeasurementMode = "distance" | "height" | "area";

export interface ViewerConfig {
  dataset: string;
  maximumScreenSpaceError: number;
  models: Record<ModelKey, {
    label: string;
    tilesetUrl: string;
  }>;
}

export interface ModelState {
  key: ModelKey;
  tileset?: Cesium.Cesium3DTileset;
  loadedTiles: Map<Cesium.Cesium3DTile, number>;
  visibleTiles: Set<Cesium.Cesium3DTile>;
  lastVisibleTiles: Set<Cesium.Cesium3DTile>;
  pending: number;
  processing: number;
  initialLoadMs?: number;
  initialReady: boolean;
  failures: number;
  startedAt?: number;
}

export type ModelStates = Record<ModelKey, ModelState>;

export interface MeasurementResult {
  mode: MeasurementMode;
  result: string;
}

export interface MeasurementState {
  mode?: MeasurementMode;
  target: ModelKey;
  points: Cesium.Cartesian3[];
  entities: Cesium.Entity[];
  dynamicEntity?: Cesium.Entity;
  results: Partial<Record<ModelKey, MeasurementResult>>;
}

export interface ModelLabels {
  terra: string;
  a1: string;
}

export interface ComparisonApi {
  viewer: Cesium.Viewer;
  modelStates: ModelStates;
  measurement: MeasurementState;
  goToView: (name: ViewName, duration?: number) => void;
  setDisplayMode: (mode: DisplayMode, announce?: boolean) => void;
  runtime: {
    version: 56;
    snapshot: () => object;
  };
}
