import assert from "node:assert/strict";
import test from "node:test";

import {
  SettledTrimGate,
  cachePolicySettings,
  retentionDecision,
} from "../src/cache-policy.ts";
import { RefreshCadence, ResourceLedger } from "../src/runtime-monitor.ts";

function resource(
  name: string,
  encodedBodySize = 0,
  transferSize = 0,
): PerformanceResourceTiming {
  return { name, encodedBodySize, decodedBodySize: 0, transferSize } as PerformanceResourceTiming;
}

test("resource totals are incremental and distinguish cached transfer bytes", () => {
  const ledger = new ResourceLedger("http://localhost:9999/", ["/a1/"]);
  ledger.add([
    resource("/a1/a.glb", 100, 120),
    resource("/a1/a.glb", 100, 0),
  ]);
  assert.deepEqual(ledger.totals("/a1/"), {
    requests: 2,
    encodedBodyBytes: 200,
    decodedBodyBytes: 0,
    transferBytes: 120,
  });
});

test("resource matching is path-based and same-origin", () => {
  const ledger = new ResourceLedger("http://localhost/", ["/a1/"]);
  ledger.add([
    resource("/other?target=/a1/a.glb"),
    resource("http://elsewhere/a1/a.glb"),
    resource("/a1/a.glb", 3),
  ]);
  assert.equal(ledger.totals("/a1/").requests, 1);
  assert.equal(ledger.ignored, 1);
});

test("invalid resource sizes cannot corrupt later measurements", () => {
  const ledger = new ResourceLedger("http://localhost/", ["/a1/"]);
  ledger.add([
    resource("/a1/bad.glb", -1),
    resource("/a1/good.glb", 3),
  ]);
  assert.equal(ledger.ignored, 1);
  assert.equal(ledger.totals("/a1/").encodedBodyBytes, 3);
  const copy = ledger.totals("/a1/");
  copy.requests = 99;
  assert.equal(ledger.totals("/a1/").requests, 1);
});

test("active work returns to idle cadence and touch re-arms activity", () => {
  const cadence = new RefreshCadence();
  cadence.touch(0);
  assert.equal(cadence.delay(1999), 250);
  assert.equal(cadence.delay(2000), 1000);
  cadence.touch(5000);
  assert.equal(cadence.delay(5001), 250);
  assert.throws(() => new RefreshCadence({ activeMs: 0 }));
});

test("settled cache demand must converge before trim is claimed", () => {
  const gate = new SettledTrimGate(2, 3);
  assert.equal(gate.sample(0, "near", true, "200:20:512"), false);
  assert.equal(gate.claim(gate.epoch), false);
  assert.equal(gate.sample(2, "near", true, "200:20:512"), true);
  assert.equal(gate.claim(gate.epoch), true);
  assert.equal(gate.claim(gate.epoch), false);
});

test("no-progress protected tiles stop repeated cache cleanup", () => {
  const gate = new SettledTrimGate(2, 3);
  gate.sample(0, "near", true, "25:20:300");
  gate.sample(2, "near", true, "25:20:300");
  gate.claim(gate.epoch);
  assert.equal(gate.sample(5, "near", true, "25:20:300"), false);
  assert.equal(gate.stopReason, "engine-protected-or-no-progress");
});

test("retention protects visible-only, recent, and under-budget caches", () => {
  const settings = cachePolicySettings();
  const sample = {
    enabled: true,
    ready: true,
    elapsedMs: 0,
    memoryBytes: 500 * 1048576,
    loaded: 50,
    visible: 10,
  };
  assert.equal(retentionDecision(sample, settings), "retain-recent-view");
  assert.equal(retentionDecision({ ...sample, elapsedMs: 30000, memoryBytes: 200 * 1048576 }, settings), "under-soft-budget");
  assert.equal(retentionDecision({ ...sample, loaded: 10 }, settings), "no-spare");
});

test("invalid cache budgets fail closed", () => {
  assert.throws(() => cachePolicySettings({ softBytes: 10, hardBytes: 9 }));
  assert.throws(() => new SettledTrimGate(-1));
});
