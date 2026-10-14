"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");

const {
  createProfileIdentity,
  frozenPositiveInteger,
  frozenViews,
  normalizePairState,
  positiveInteger,
  profileFailed,
  remainingStartupMs,
  settledAt,
  splitLabels,
  summarizeFinite,
  validateViewerBinding,
  viewerTilesetUrls,
} = require("./profile_contract.cjs");

test("profile identity requires generic candidate ids and the frozen protocol", () => {
  assert.throws(
    () => createProfileIdentity({}),
    /LEFT_CANDIDATE_ID, RIGHT_CANDIDATE_ID and PROFILE_PROTOCOL_DIGEST are required/,
  );
  assert.deepEqual(createProfileIdentity({
    LEFT_CANDIDATE_ID: "odx-3.8.3-eb9fd56",
    RIGHT_CANDIDATE_ID: "remesh-v55",
    LEFT_LABEL: "spoofed",
    RIGHT_LABEL: "spoofed",
    PROFILE_PROTOCOL_DIGEST: "sha256:b36886942e7a0a6c9db8c14e31bd631707d73a13d7bc4b544ecafd4881e19aee",
  }), {
    protocolDigest: "sha256:b36886942e7a0a6c9db8c14e31bd631707d73a13d7bc4b544ecafd4881e19aee",
    models: {
      left: { candidateId: "odx-3.8.3-eb9fd56", label: "ODX 3.8.3" },
      right: { candidateId: "remesh-v55", label: "REMESH_v55_runtime" },
    },
  });
  assert.throws(
    () => createProfileIdentity({
      LEFT_CANDIDATE_ID: "terra-5.1.1",
      RIGHT_CANDIDATE_ID: "remesh-v55",
      PROFILE_PROTOCOL_DIGEST: `sha256:${"0".repeat(64)}`,
    }),
    /PROFILE_PROTOCOL_DIGEST does not match the frozen protocol/,
  );
  assert.throws(
    () => createProfileIdentity({
      LEFT_CANDIDATE_ID: "terra-5.1.1",
      RIGHT_CANDIDATE_ID: "odx-3.8.3-eb9fd56",
      PROFILE_PROTOCOL_DIGEST: "sha256:b36886942e7a0a6c9db8c14e31bd631707d73a13d7bc4b544ecafd4881e19aee",
    }),
    /candidate pair is not declared by the frozen protocol/,
  );
  assert.throws(
    () => createProfileIdentity({
      LEFT_CANDIDATE_ID: Symbol("terra"),
      RIGHT_CANDIDATE_ID: "remesh-v55",
      PROFILE_PROTOCOL_DIGEST: "sha256:b36886942e7a0a6c9db8c14e31bd631707d73a13d7bc4b544ecafd4881e19aee",
    }),
    /candidate ids are invalid/,
  );
});

test("internal viewer keys never leak into candidate metrics", () => {
  assert.deepEqual(
    normalizePairState({ terra: { pending: 1 }, a1: { pending: 2 } }),
    { left: { pending: 1 }, right: { pending: 2 } },
  );
});

test("a quiet sample after the deadline remains a timeout", () => {
  assert.equal(settledAt(3, 30000, 30000), 30000);
  assert.equal(settledAt(3, 30001, 30000), undefined);
  assert.equal(settledAt(2, 1000, 30000), undefined);
});

test("split labels use candidate identity rather than viewer storage keys", () => {
  assert.deepEqual(
    splitLabels({ left: { label: "ODX 3.8.3" }, right: { label: "Remesh v55" } }),
    { left: "LEFT · ODX 3.8.3", right: "Remesh v55 · RIGHT" },
  );
});

test("navigation and sampling timeouts require positive integers", () => {
  assert.equal(positiveInteger("90000", "PROFILE_NAVIGATION_TIMEOUT_MS"), 90000);
  assert.throws(
    () => positiveInteger("0", "PROFILE_NAVIGATION_TIMEOUT_MS"),
    /PROFILE_NAVIGATION_TIMEOUT_MS must be a positive integer/,
  );
  assert.equal(frozenPositiveInteger("30000", "PROFILE_PHASE_TIMEOUT_MS", 30000), 30000);
  assert.throws(
    () => frozenPositiveInteger("30001", "PROFILE_PHASE_TIMEOUT_MS", 30000),
    /PROFILE_PHASE_TIMEOUT_MS must be exactly 30000/,
  );
});

test("the frozen view sequence cannot omit or reorder evidence", () => {
  assert.deepEqual(
    frozenViews("overview,building,texture,edge"),
    ["overview", "building", "texture", "edge"],
  );
  assert.throws(() => frozenViews("overview,building,texture"), /PROFILE_VIEWS must equal/);
  assert.throws(() => frozenViews("building,overview,texture,edge"), /PROFILE_VIEWS must equal/);
});

test("timeouts remain report evidence and make the process fail", () => {
  assert.equal(profileFailed([{ timedOut: true }], [], []), true);
  assert.equal(profileFailed([{ timedOut: false }], [], []), false);
});

test("geometric error aggregation does not spread the tile array", () => {
  assert.deepEqual(summarizeFinite([3, Number.NaN, 1, 2]), {
    count: 3,
    min: 1,
    max: 3,
    mean: 2,
  });
});

test("the reported pair is bound to viewer labels and root manifests", () => {
  const identity = createProfileIdentity({
    LEFT_CANDIDATE_ID: "odx-3.8.3-eb9fd56",
    RIGHT_CANDIDATE_ID: "remesh-v55",
    PROFILE_PROTOCOL_DIGEST: "sha256:b36886942e7a0a6c9db8c14e31bd631707d73a13d7bc4b544ecafd4881e19aee",
  });
  const config = {
    models: {
      terra: { label: "ODX", tilesetUrl: "/terra/tileset.json" },
      a1: { label: "REMESH_v55_runtime", tilesetUrl: "/a1/tileset.json" },
    },
  };
  const digests = {
    left: "sha256:e7d0d57a75e400b6edee57c4e03da8a2cd2d7b5ca1b5a00645e1496c729951a9",
    right: "sha256:7ee09e40fea72d70ae70b709c70b8e957c9f7d92c1175e4e6ccb8e6ec18278f2",
  };

  assert.equal(validateViewerBinding(identity, config, digests).models.left.serviceLabel, "ODX");
  config.models.terra.label = "DJI_Terra";
  assert.throws(
    () => validateViewerBinding(identity, config, digests),
    /viewer binding does not match the declared candidate pair/,
  );
});

test("viewer root URLs are validated before URL construction", () => {
  assert.deepEqual(viewerTilesetUrls({
    models: {
      terra: { tilesetUrl: "/terra/tileset.json" },
      a1: { tilesetUrl: "/a1/tileset.json" },
    },
  }), { left: "/terra/tileset.json", right: "/a1/tileset.json" });
  assert.throws(
    () => viewerTilesetUrls({ models: {} }),
    /viewer binding does not match the declared candidate pair/,
  );
});

test("all startup work shares one deadline", () => {
  assert.equal(remainingStartupMs(1000, 90000, 5000), 86000);
  assert.throws(() => remainingStartupMs(1000, 90000, 91001), /startup timeout exceeded/);
});
