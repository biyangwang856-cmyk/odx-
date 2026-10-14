import type * as Cesium from "cesium";

import type { CachePolicyController } from "./cache-policy";
import type { RuntimeMonitor } from "./runtime-monitor";
import type { DisplayMode, ModelState, ViewName } from "./types";

interface DiagnosticsState {
  activeView: ViewName;
  mode: DisplayMode;
}

interface RuntimeTile extends Cesium.Cesium3DTile {
  _contentResource?: { url?: string };
  _depth?: number;
  _geometricError?: number;
  _header?: { content?: { uri?: string; url?: string } };
  _boundingVolume?: { boundingSphere?: Cesium.BoundingSphere };
}

interface RuntimeContent {
  _resource?: { url?: string };
  _url?: string;
  trianglesLength?: number;
}

function contentUrl(tile: RuntimeTile): string | null {
  const content = tile.content as RuntimeContent | undefined;
  return tile._contentResource?.url
    || content?._resource?.url
    || content?._url
    || tile._header?.content?.uri
    || tile._header?.content?.url
    || null;
}

function tileSphere(tile: RuntimeTile): Cesium.BoundingSphere | undefined {
  const sphere = tile.boundingSphere;
  return sphere || tile._boundingVolume?.boundingSphere;
}

export function startDiagnostics(
  viewer: Cesium.Viewer,
  modelState: ModelState,
  getState: () => DiagnosticsState,
  monitor: RuntimeMonitor,
  cachePolicy: CachePolicyController,
): void {
  const publish = (): void => {
    const state = getState();
    const camera = viewer.camera;
    const visible = [...modelState.lastVisibleTiles].map((rawTile) => {
      const tile = rawTile as RuntimeTile;
      const sphere = tileSphere(tile);
      return {
        contentUrl: contentUrl(tile),
        triangles: tile.content?.trianglesLength || 0,
        geometricError: Number.isFinite(tile.geometricError)
          ? tile.geometricError
          : tile._geometricError,
        depth: Number.isInteger(tile._depth) ? tile._depth : null,
        center: sphere?.center
          ? [sphere.center.x, sphere.center.y, sphere.center.z]
          : null,
        radius: sphere && Number.isFinite(sphere.radius) ? sphere.radius : null,
      };
    });
    const fov = "fov" in camera.frustum ? camera.frustum.fov : null;
    fetch("/api/diagnostics", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        activeView: state.activeView,
        mode: state.mode,
        pending: modelState.pending,
        processing: modelState.processing,
        camera: {
          position: [camera.position.x, camera.position.y, camera.position.z],
          fov,
          viewportHeight: viewer.canvas.clientHeight,
        },
        visible,
        runtime: monitor.snapshot(),
        cache: cachePolicy.snapshot(),
        capturedAt: new Date().toISOString(),
      }),
    }).catch(() => undefined);
  };

  monitor.startLoop(publish, { activeMs: 5000, idleMs: 30000, immediate: false });
  publish();
}
