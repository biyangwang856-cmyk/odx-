#!/usr/bin/env node
"use strict";

// Capture loading behavior across a repeatable interactive camera path.

const fs = require("fs");
const os = require("os");
const path = require("path");

const playwrightRoot = process.env.PLAYWRIGHT_ROOT
  || "playwright";
const { chromium } = require(playwrightRoot);

const viewerUrl = process.env.VIEWER_URL || "http://127.0.0.1:8121";
const output = process.env.PROFILE_OUTPUT;
const screenshotRoot = process.env.PROFILE_SCREENSHOT_ROOT;
const sse = Number(process.env.PROFILE_SSE || 32);
const phaseTimeoutMs = Number(process.env.PROFILE_PHASE_TIMEOUT_MS || 15000);
const flightDurationSeconds = Number(process.env.PROFILE_FLIGHT_SECONDS || 0.8);
const views = (process.env.PROFILE_VIEWS || "building,texture,edge,top,overview")
  .split(",")
  .map((value) => value.trim())
  .filter(Boolean);

if (!output) throw new Error("PROFILE_OUTPUT is required");

function summarizeResources(resources) {
  const byExtension = {};
  let bytes = 0;
  for (const resource of resources) {
    const pathname = new URL(resource.name).pathname;
    const extension = path.extname(pathname).toLowerCase() || "(none)";
    const summary = byExtension[extension] || { count: 0, bytes: 0 };
    summary.count += 1;
    summary.bytes += resource.transferBytes;
    byExtension[extension] = summary;
    bytes += resource.transferBytes;
  }
  return { count: resources.length, bytes, byExtension };
}

let browser;

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
  const context = await browser.newContext({ viewport: { width: 1200, height: 760 }, deviceScaleFactor: 1 });
  const page = await context.newPage();
  const errors = [];
  const failedRequests = [];
  const canceledRequests = [];
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });
  page.on("pageerror", (error) => errors.push(String(error)));
  page.on("requestfailed", (request) => {
    const failure = `${request.url()} :: ${request.failure()?.errorText}`;
    if (request.failure()?.errorText === "net::ERR_ABORTED") canceledRequests.push(failure);
    else failedRequests.push(failure);
  });

  const target = new URL(viewerUrl);
  target.searchParams.set("mode", "a1");
  target.searchParams.set("sse", String(sse));
  const navigationStartedAt = Date.now();
  await page.goto(target.href, { waitUntil: "domcontentloaded", timeout: 30000 });
  await page.waitForFunction(() => Boolean(window.__comparison?.modelStates?.a1?.tileset), undefined, { timeout: 30000 });
  await page.evaluate(() => window.__comparison.setDisplayMode("a1", false));

  const phases = [];
  for (let phaseIndex = 0; phaseIndex < views.length; phaseIndex += 1) {
    const view = views[phaseIndex];
    const resourceOffset = await page.evaluate(() => performance.getEntriesByType("resource").length);
    const phaseStartedAt = Date.now();
    await page.evaluate(({ name, duration }) => {
      window.__comparison.goToView(name, duration);
    }, { name: view, duration: flightDurationSeconds });

    const samples = [];
    let quietStreak = 0;
    let settledMs;
    while (Date.now() - phaseStartedAt < phaseTimeoutMs) {
      const sample = await page.evaluate(() => {
        const state = window.__comparison.modelStates.a1;
        const depths = {};
        for (const tile of state.lastVisibleTiles) {
          const depth = Number.isInteger(tile._depth) ? tile._depth : -1;
          depths[depth] = (depths[depth] || 0) + 1;
        }
        return {
          pending: state.pending,
          processing: state.processing,
          loaded: state.loadedTiles.size,
          visible: state.lastVisibleTiles.size,
          depths,
          triangles: [...state.lastVisibleTiles]
            .reduce((sum, tile) => sum + (tile.content?.trianglesLength || 0), 0),
          memoryBytes: state.tileset.totalMemoryUsageInBytes || 0,
        };
      });
      sample.elapsedMs = Date.now() - phaseStartedAt;
      samples.push(sample);
      const minimumFlightWait = flightDurationSeconds * 1000 + 500;
      if (sample.elapsedMs >= minimumFlightWait && sample.pending === 0 && sample.processing === 0) quietStreak += 1;
      else quietStreak = 0;
      if (quietStreak >= 3) {
        settledMs = sample.elapsedMs;
        break;
      }
      await page.waitForTimeout(250);
    }

    const newResources = await page.evaluate((offset) => performance.getEntriesByType("resource")
      .slice(offset)
      .filter((entry) => entry.name.includes("/a1/"))
      .map((entry) => ({
        name: entry.name,
        durationMs: entry.duration,
        transferBytes: entry.transferSize || entry.encodedBodySize || entry.decodedBodySize || 0,
      })), resourceOffset);
    let screenshot;
    if (screenshotRoot) {
      screenshot = path.join(screenshotRoot, `${String(phaseIndex + 1).padStart(2, "0")}-${view}.png`);
      fs.mkdirSync(path.dirname(screenshot), { recursive: true });
      const bounds = await page.locator("#cesiumContainer canvas").boundingBox();
      if (bounds) await page.screenshot({ path: screenshot, clip: bounds });
      else errors.push(`screenshot failed for ${view}: Cesium canvas has no bounds`);
    }
    phases.push({
      view,
      timedOut: settledMs === undefined,
      settledMs,
      final: samples.at(-1) || {},
      peaks: {
        pending: Math.max(0, ...samples.map((sample) => sample.pending)),
        processing: Math.max(0, ...samples.map((sample) => sample.processing)),
        visible: Math.max(0, ...samples.map((sample) => sample.visible)),
        memoryBytes: Math.max(0, ...samples.map((sample) => sample.memoryBytes)),
      },
      resources: summarizeResources(newResources),
      screenshot,
      samples,
    });
  }

  const allResources = await page.evaluate(() => performance.getEntriesByType("resource")
    .filter((entry) => entry.name.includes("/a1/"))
    .map((entry) => ({
      name: entry.name,
      durationMs: entry.duration,
      transferBytes: entry.transferSize || entry.encodedBodySize || entry.decodedBodySize || 0,
    })));
  const report = {
    viewerUrl: target.href,
    renderer: "SwiftShader; interaction selection/request regression, not FPS benchmarking",
    sse,
    flightDurationSeconds,
    phaseTimeoutMs,
    views,
    elapsedMs: Date.now() - navigationStartedAt,
    phases,
    resources: summarizeResources(allResources),
    errors,
    failedRequests,
    canceledRequests,
  };
  fs.mkdirSync(path.dirname(output), { recursive: true });
  fs.writeFileSync(output, `${JSON.stringify(report, null, 2)}\n`);
  process.stdout.write(`${JSON.stringify({
    output,
    elapsedMs: report.elapsedMs,
    phases: phases.map(({ view, timedOut, settledMs, final, peaks, resources }) => ({
      view, timedOut, settledMs, final, peaks, resources,
    })),
    resources: report.resources,
    errors,
    failedRequests,
    canceledRequests,
  }, null, 2)}\n`);
  await browser.close();
  if (errors.length || failedRequests.length) process.exitCode = 1;
})().catch(async (error) => {
  console.error(error);
  if (browser) await browser.close().catch(() => {});
  process.exitCode = 1;
});
