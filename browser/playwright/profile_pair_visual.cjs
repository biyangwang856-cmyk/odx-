#!/usr/bin/env node
"use strict";

// Capture paired publications from identical Cesium camera views.

const fs = require("fs");
const { createHash } = require("crypto");
const os = require("os");
const path = require("path");
const {
  createProfileIdentity,
  frozenPositiveInteger,
  frozenViews,
  normalizePairState,
  profileFailed,
  remainingStartupMs,
  settledAt: acceptedSettlement,
  splitLabels,
  summarizeFinite,
  validateViewerBinding,
  viewerTilesetUrls,
} = require("./profile_contract.cjs");
const playwrightRoot = process.env.PLAYWRIGHT_ROOT || "playwright";
const { chromium } = require(playwrightRoot);

const viewerUrl = process.env.VIEWER_URL;
const output = process.env.PROFILE_OUTPUT;
const screenshotRoot = process.env.PROFILE_SCREENSHOT_ROOT;
const profileIdentity = createProfileIdentity(process.env);
const leftLabel = profileIdentity.models.left.label;
const rightLabel = profileIdentity.models.right.label;
const sse = frozenPositiveInteger(process.env.PROFILE_SSE || "8", "PROFILE_SSE", 8);
const navigationTimeoutMs = frozenPositiveInteger(
  process.env.PROFILE_NAVIGATION_TIMEOUT_MS || "90000",
  "PROFILE_NAVIGATION_TIMEOUT_MS",
  90000,
);
const phaseTimeoutMs = frozenPositiveInteger(
  process.env.PROFILE_PHASE_TIMEOUT_MS || "30000",
  "PROFILE_PHASE_TIMEOUT_MS",
  30000,
);
const views = frozenViews(process.env.PROFILE_VIEWS || "overview,building,texture,edge");

if (!viewerUrl || !output || !screenshotRoot) {
  throw new Error("VIEWER_URL, PROFILE_OUTPUT and PROFILE_SCREENSHOT_ROOT are required");
}

function summarizeResources(resources, marker) {
  const selected = resources.filter((entry) => entry.name.includes(marker));
  const byExtension = {};
  let bytes = 0;
  for (const entry of selected) {
    const extension = path.extname(new URL(entry.name).pathname).toLowerCase() || "(none)";
    const summary = byExtension[extension] || { count: 0, bytes: 0 };
    summary.count += 1;
    summary.bytes += entry.transferBytes;
    byExtension[extension] = summary;
    bytes += entry.transferBytes;
  }
  return { count: selected.length, bytes, byExtension };
}

function stateExpression() {
  const describe = (key) => {
    const state = window.__comparison.modelStates[key];
    const depths = {};
    const geometricErrors = [];
    for (const tile of state.lastVisibleTiles) {
      const depth = Number.isInteger(tile._depth) ? tile._depth : -1;
      depths[depth] = (depths[depth] || 0) + 1;
      const geometricError = Number.isFinite(tile.geometricError)
        ? tile.geometricError
        : tile._geometricError;
      if (Number.isFinite(geometricError)) geometricErrors.push(geometricError);
    }
    return {
      pending: state.pending,
      processing: state.processing,
      loaded: state.loadedTiles.size,
      visible: state.lastVisibleTiles.size,
      triangles: [...state.lastVisibleTiles]
        .reduce((sum, tile) => sum + (tile.content?.trianglesLength || 0), 0),
      memoryBytes: state.tileset.totalMemoryUsageInBytes || 0,
      depths,
      geometricError: geometricErrors,
    };
  };
  return { terra: describe("terra"), a1: describe("a1") };
}

function percentile(values, fraction) {
  if (!values.length) return null;
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.min(sorted.length - 1, Math.ceil(sorted.length * fraction) - 1)];
}

async function fetchJson(url, timeoutMs) {
  const response = await fetch(url, {
    headers: { Accept: "application/json" },
    signal: AbortSignal.timeout(timeoutMs),
  });
  if (!response.ok) throw new Error(`viewer binding request failed: ${url} HTTP ${response.status}`);
  return response.json();
}

async function fetchDigest(url, timeoutMs) {
  const response = await fetch(url, { signal: AbortSignal.timeout(timeoutMs) });
  if (!response.ok) throw new Error(`viewer binding request failed: ${url} HTTP ${response.status}`);
  const body = Buffer.from(await response.arrayBuffer());
  return `sha256:${createHash("sha256").update(body).digest("hex")}`;
}

let browser;

(async () => {
  const startupStartedAt = Date.now();
  const configUrl = new URL("/api/config", viewerUrl);
  const viewerConfig = await fetchJson(
    configUrl,
    remainingStartupMs(startupStartedAt, navigationTimeoutMs),
  );
  const tilesetUrls = viewerTilesetUrls(viewerConfig);
  const manifestBudgetMs = remainingStartupMs(startupStartedAt, navigationTimeoutMs);
  const [leftDigest, rightDigest] = await Promise.all([
    fetchDigest(new URL(tilesetUrls.left, configUrl), manifestBudgetMs),
    fetchDigest(new URL(tilesetUrls.right, configUrl), manifestBudgetMs),
  ]);
  const viewerBinding = validateViewerBinding(profileIdentity, viewerConfig, {
    left: leftDigest,
    right: rightDigest,
  });
  const localBrowserLibraries = path.join(os.homedir(), ".cache/ms-playwright/local-libs/usr/lib/x86_64-linux-gnu");
  browser = await chromium.launch({
    timeout: remainingStartupMs(startupStartedAt, navigationTimeoutMs),
    headless: true,
    args: ["--enable-webgl", "--ignore-gpu-blocklist", "--use-angle=swiftshader"],
    env: {
      ...process.env,
      LD_LIBRARY_PATH: [localBrowserLibraries, process.env.LD_LIBRARY_PATH].filter(Boolean).join(":"),
    },
  });
  const context = await browser.newContext({ viewport: { width: 1400, height: 900 }, deviceScaleFactor: 1 });
  const page = await context.newPage();
  const errors = [];
  const failedRequests = [];
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });
  page.on("pageerror", (error) => errors.push(String(error)));
  page.on("requestfailed", (request) => {
    if (request.failure()?.errorText !== "net::ERR_ABORTED") {
      failedRequests.push(`${request.url()} :: ${request.failure()?.errorText}`);
    }
  });

  const target = new URL(viewerUrl);
  target.searchParams.set("mode", "split");
  target.searchParams.set("sse", String(sse));
  target.searchParams.set("policy", "baseline");
  await page.goto(target.href, {
    waitUntil: "domcontentloaded",
    timeout: remainingStartupMs(startupStartedAt, navigationTimeoutMs),
  });
  await page.waitForFunction(
    () => Boolean(window.__comparison?.modelStates?.terra?.tileset)
      && Boolean(window.__comparison?.modelStates?.a1?.tileset),
    undefined,
    { timeout: remainingStartupMs(startupStartedAt, navigationTimeoutMs) },
  );
  const startupMs = Date.now() - startupStartedAt;
  await page.evaluate(({ left, right, split }) => {
    document.title = `${left} × ${right}`;
    document.querySelector("h1").textContent = `${left} × ${right} 对比`;
    document.querySelector(".terra-text").textContent = left;
    document.querySelector(".a1-text").textContent = right;
    document.querySelector(".split-label.terra").textContent = split.left;
    document.querySelector(".split-label.a1").textContent = split.right;
    window.__comparison.setDisplayMode("split", false);
  }, { left: leftLabel, right: rightLabel, split: splitLabels(profileIdentity.models) });

  fs.mkdirSync(screenshotRoot, { recursive: true });
  const phases = [];
  for (let index = 0; index < views.length; index += 1) {
    const view = views[index];
    await page.evaluate((name) => {
      window.__comparison.setDisplayMode("split", false);
      window.__comparison.goToView(name, 0);
    }, view);
    const startedAt = Date.now();
    const samples = [];
    let quietStreak = 0;
    let settledMs;
    while (Date.now() - startedAt < phaseTimeoutMs) {
      const sample = normalizePairState(await page.evaluate(stateExpression));
      sample.left.geometricError = summarizeFinite(sample.left.geometricError);
      sample.right.geometricError = summarizeFinite(sample.right.geometricError);
      const remainingMs = phaseTimeoutMs - (Date.now() - startedAt);
      if (remainingMs <= 0) break;
      sample.frameTimeMs = await page.evaluate((limitMs) => new Promise((resolve) => {
        const timeout = window.setTimeout(() => resolve(null), limitMs);
        requestAnimationFrame((first) => requestAnimationFrame((second) => {
          window.clearTimeout(timeout);
          resolve(second - first);
        }));
      }), Math.min(1000, remainingMs));
      sample.elapsedMs = Date.now() - startedAt;
      samples.push(sample);
      const quiet = [sample.left, sample.right]
        .every((state) => state.visible > 0 && state.pending === 0 && state.processing === 0);
      quietStreak = quiet ? quietStreak + 1 : 0;
      settledMs = acceptedSettlement(quietStreak, sample.elapsedMs, phaseTimeoutMs);
      if (settledMs !== undefined) {
        break;
      }
      await page.waitForTimeout(250);
    }

    const prefix = `${String(index + 1).padStart(2, "0")}-${view}`;
    const screenshots = {};
    for (const [side, mode] of [["split", "split"], ["left", "terra"], ["right", "a1"]]) {
      await page.evaluate((value) => window.__comparison.setDisplayMode(value, false), mode);
      await page.waitForTimeout(500);
      const file = path.join(screenshotRoot, `${prefix}-${side}.png`);
      try {
        await page.screenshot({ path: file, timeout: 90000 });
        screenshots[side] = file;
      } catch (error) {
        errors.push(`screenshot ${prefix}-${side} failed: ${String(error)}`);
      }
    }
    phases.push({
      view,
      timedOut: settledMs === undefined,
      settledMs,
      final: samples.at(-1),
      peaks: {
        leftMemoryBytes: Math.max(0, ...samples.map((sample) => sample.left.memoryBytes)),
        rightMemoryBytes: Math.max(0, ...samples.map((sample) => sample.right.memoryBytes)),
        leftProcessing: Math.max(0, ...samples.map((sample) => sample.left.processing)),
        rightProcessing: Math.max(0, ...samples.map((sample) => sample.right.processing)),
        frameTimeMsP95: percentile(
          samples.map((sample) => sample.frameTimeMs).filter(Number.isFinite),
          0.95,
        ),
      },
      screenshots,
      samples,
    });
  }

  const resources = await page.evaluate(() => performance.getEntriesByType("resource").map((entry) => ({
    name: entry.name,
    transferBytes: entry.transferSize || entry.encodedBodySize || entry.decodedBodySize || 0,
  })));
  const report = {
    schemaVersion: "mesh-pair-profile.v1",
    ...profileIdentity,
    viewerBinding,
    viewerUrl: target.href,
    renderer: "Cesium + Playwright + SwiftShader; shared camera and SSE",
    sse,
    navigationTimeoutMs,
    startupMs,
    phaseTimeoutMs,
    views,
    phases,
    resources: {
      left: summarizeResources(resources, "/terra/"),
      right: summarizeResources(resources, "/a1/"),
    },
    errors,
    failedRequests,
  };
  fs.mkdirSync(path.dirname(output), { recursive: true });
  fs.writeFileSync(output, `${JSON.stringify(report, null, 2)}\n`);
  process.stdout.write(`${JSON.stringify({
    output,
    phases: phases.map(({ view, timedOut, settledMs, final, peaks, screenshots }) => ({
      view, timedOut, settledMs, final, peaks, screenshots,
    })),
    resources: report.resources,
    errors,
    failedRequests,
  }, null, 2)}\n`);
  await browser.close();
  if (profileFailed(phases, errors, failedRequests)) process.exitCode = 1;
})().catch(async (error) => {
  console.error(error);
  if (browser) await browser.close().catch(() => {});
  process.exitCode = 1;
});
