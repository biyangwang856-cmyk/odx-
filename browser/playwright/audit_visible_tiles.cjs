#!/usr/bin/env node
"use strict";

// Audit the complete visible-tile set after the selected view settles.

const fs = require("fs");
const os = require("os");
const path = require("path");

const playwrightRoot = process.env.PLAYWRIGHT_ROOT
  || "playwright";
const { chromium } = require(playwrightRoot);

const viewerUrl = process.env.VIEWER_URL || "http://127.0.0.1:8150";
const output = process.env.AUDIT_OUTPUT;
const screenshot = process.env.AUDIT_SCREENSHOT;
const view = process.env.AUDIT_VIEW || "texture";
const sse = Number(process.env.AUDIT_SSE || 32);
const timeoutMs = Number(process.env.AUDIT_TIMEOUT_MS || 30000);

if (!output) throw new Error("AUDIT_OUTPUT is required");

let browser;

function unique(values) {
  return [...new Set(values.filter(Boolean))];
}

(async () => {
  const localBrowserLibraries = path.join(os.homedir(), ".cache/ms-playwright/local-libs/usr/lib/x86_64-linux-gnu");
  browser = await chromium.launch({
    headless: true,
    args: ["--enable-webgl", "--ignore-gpu-blocklist", "--use-angle=swiftshader"],
    env: {
      ...process.env,
      LD_LIBRARY_PATH: [localBrowserLibraries, process.env.LD_LIBRARY_PATH].filter(Boolean).join(":"),
    },
  });
  const context = await browser.newContext({ viewport: { width: 1400, height: 900 }, deviceScaleFactor: 1 });
  const page = await context.newPage();
  const consoleErrors = [];
  const failedRequests = [];
  page.on("console", (message) => {
    if (message.type() === "error") consoleErrors.push(message.text());
  });
  page.on("pageerror", (error) => consoleErrors.push(String(error)));
  page.on("requestfailed", (request) => {
    failedRequests.push(`${request.url()} :: ${request.failure()?.errorText}`);
  });

  const target = new URL(viewerUrl);
  target.searchParams.set("mode", "a1");
  target.searchParams.set("sse", String(sse));
  await page.goto(target.href, { waitUntil: "domcontentloaded", timeout: 30000 });
  await page.waitForFunction(() => Boolean(window.__comparison?.modelStates?.a1?.tileset), undefined, { timeout: 30000 });
  await page.evaluate((viewName) => {
    window.__comparison.setDisplayMode("a1", false);
    window.__comparison.goToView(viewName, 0);
  }, view);

  const startedAt = Date.now();
  let quietStreak = 0;
  while (Date.now() - startedAt < timeoutMs) {
    const state = await page.evaluate(() => {
      const model = window.__comparison.modelStates.a1;
      return { pending: model.pending, processing: model.processing, visible: model.lastVisibleTiles.size };
    });
    if (state.visible > 0 && state.pending === 0 && state.processing === 0) quietStreak += 1;
    else quietStreak = 0;
    if (quietStreak >= 4) break;
    await page.waitForTimeout(250);
  }

  const audit = await page.evaluate(() => {
    const state = window.__comparison.modelStates.a1;
    const tileset = state.tileset;
    const identity = new Map();
    const visit = (tile, tilePath) => {
      identity.set(tile, tilePath);
      const children = tile.children || tile._children || [];
      children.forEach((child, index) => visit(child, `${tilePath}.${index}`));
    };
    visit(tileset.root, "root");

    const resourceUrl = (tile) => tile?._contentResource?.url
      || tile?.content?._resource?.url
      || tile?.content?._url
      || tile?._header?.content?.uri
      || tile?._header?.content?.url
      || null;
    const center = (tile) => {
      const sphere = tile?.boundingSphere?.boundingSphere || tile?.boundingSphere || tile?._boundingVolume?.boundingSphere;
      const point = sphere?.center;
      return point ? [point.x, point.y, point.z] : null;
    };
    const radius = (tile) => {
      const sphere = tile?.boundingSphere?.boundingSphere || tile?.boundingSphere || tile?._boundingVolume?.boundingSphere;
      return Number.isFinite(sphere?.radius) ? sphere.radius : null;
    };
    const describe = (tile) => ({
      runtimePath: identity.get(tile) || null,
      depth: Number.isInteger(tile?._depth) ? tile._depth : null,
      geometricError: Number.isFinite(tile?.geometricError) ? tile.geometricError : tile?._geometricError,
      refine: tile?.refine || tile?._refine || null,
      contentUrl: resourceUrl(tile),
      headerContent: tile?._header?.content || null,
      contentConstructor: tile?.content?.constructor?.name || null,
      triangles: tile?.content?.trianglesLength || 0,
      features: tile?.content?.featuresLength || 0,
      center: center(tile),
      radius: radius(tile),
      children: (tile?.children || tile?._children || []).length,
      parentPath: tile?.parent ? identity.get(tile.parent) || null : null,
    });

    const visible = [...state.lastVisibleTiles].map(describe);
    const loaded = [...state.loadedTiles.keys()].map(describe);
    const resources = performance.getEntriesByType("resource")
      .filter((entry) => entry.name.includes("/a1/"))
      .map((entry) => ({
        url: entry.name,
        transferBytes: entry.transferSize || entry.encodedBodySize || entry.decodedBodySize || 0,
        durationMs: entry.duration,
      }));
    const camera = tileset._scene?.camera || window.__comparison.viewer.camera;
    return {
      pending: state.pending,
      processing: state.processing,
      memoryBytes: tileset.totalMemoryUsageInBytes || 0,
      camera: {
        position: [camera.position.x, camera.position.y, camera.position.z],
        direction: [camera.direction.x, camera.direction.y, camera.direction.z],
        up: [camera.up.x, camera.up.y, camera.up.z],
        right: [camera.right.x, camera.right.y, camera.right.z],
        frustum: {
          fov: camera.frustum.fov,
          aspectRatio: camera.frustum.aspectRatio,
          near: camera.frustum.near,
          far: camera.frustum.far,
        },
      },
      visible,
      loaded,
      resources,
    };
  });

  if (screenshot) {
    fs.mkdirSync(path.dirname(screenshot), { recursive: true });
    await page.screenshot({ path: screenshot, fullPage: false, timeout: 90000 });
  }

  const report = {
    viewerUrl: target.href,
    view,
    sse,
    timeoutMs,
    elapsedMs: Date.now() - startedAt,
    ...audit,
    visibleContentUrls: unique(audit.visible.map((item) => item.contentUrl)),
    loadedContentUrls: unique(audit.loaded.map((item) => item.contentUrl)),
    consoleErrors,
    failedRequests,
    screenshot,
  };
  fs.mkdirSync(path.dirname(output), { recursive: true });
  fs.writeFileSync(output, `${JSON.stringify(report, null, 2)}\n`);
  process.stdout.write(`${JSON.stringify({
    output,
    elapsedMs: report.elapsedMs,
    pending: report.pending,
    processing: report.processing,
    memoryBytes: report.memoryBytes,
    visible: report.visible.length,
    loaded: report.loaded.length,
    visibleContentUrls: report.visibleContentUrls,
    errors: consoleErrors.length,
    failedRequests: failedRequests.length,
  }, null, 2)}\n`);
  await browser.close();
  if (consoleErrors.length || failedRequests.length) process.exitCode = 1;
})().catch(async (error) => {
  console.error(error);
  if (browser) await browser.close().catch(() => {});
  process.exitCode = 1;
});
