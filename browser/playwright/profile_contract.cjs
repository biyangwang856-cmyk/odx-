"use strict";

const CANDIDATE_ID = /^[a-z0-9][a-z0-9.-]*$/;
const FROZEN_PROTOCOL_DIGEST =
  "sha256:b36886942e7a0a6c9db8c14e31bd631707d73a13d7bc4b544ecafd4881e19aee";
const PROFILE_PAIRS = new Map([
  ["terra-5.1.1/remesh-v55", {
    leftDisplayLabel: "DJI_Terra",
    rightDisplayLabel: "REMESH_v55_runtime",
    leftLabel: "DJI_Terra",
    rightLabel: "REMESH_v55_runtime",
    leftDigest: "sha256:ad064f4976baa3f5d486c0e9452ee7784b3b9b011d610fdf4c4d236bff158b26",
    rightDigest: "sha256:7ee09e40fea72d70ae70b709c70b8e957c9f7d92c1175e4e6ccb8e6ec18278f2",
  }],
  ["odx-3.8.3-eb9fd56/remesh-v55", {
    leftDisplayLabel: "ODX 3.8.3",
    rightDisplayLabel: "REMESH_v55_runtime",
    leftLabel: "ODX",
    rightLabel: "REMESH_v55_runtime",
    leftDigest: "sha256:e7d0d57a75e400b6edee57c4e03da8a2cd2d7b5ca1b5a00645e1496c729951a9",
    rightDigest: "sha256:7ee09e40fea72d70ae70b709c70b8e957c9f7d92c1175e4e6ccb8e6ec18278f2",
  }],
]);
const REQUIRED_VIEWS = ["overview", "building", "texture", "edge"];
const VIEWER_TILESET_URLS = {
  left: "/terra/tileset.json",
  right: "/a1/tileset.json",
};

function createProfileIdentity(environment) {
  const leftId = environment.LEFT_CANDIDATE_ID;
  const rightId = environment.RIGHT_CANDIDATE_ID;
  const protocolDigest = environment.PROFILE_PROTOCOL_DIGEST;
  if (!leftId || !rightId || !protocolDigest) {
    throw new Error(
      "LEFT_CANDIDATE_ID, RIGHT_CANDIDATE_ID and PROFILE_PROTOCOL_DIGEST are required",
    );
  }
  if (
    typeof leftId !== "string"
    || typeof rightId !== "string"
    || !CANDIDATE_ID.test(leftId)
    || !CANDIDATE_ID.test(rightId)
  ) {
    throw new Error("candidate ids are invalid");
  }
  if (protocolDigest !== FROZEN_PROTOCOL_DIGEST) {
    throw new Error("PROFILE_PROTOCOL_DIGEST does not match the frozen protocol");
  }
  const pair = PROFILE_PAIRS.get(`${leftId}/${rightId}`);
  if (!pair) {
    throw new Error("candidate pair is not declared by the frozen protocol");
  }
  return {
    protocolDigest,
    models: {
      left: { candidateId: leftId, label: pair.leftDisplayLabel },
      right: { candidateId: rightId, label: pair.rightDisplayLabel },
    },
  };
}

function normalizePairState(internalState) {
  return { left: internalState.terra, right: internalState.a1 };
}

function positiveInteger(value, name) {
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed <= 0) {
    throw new Error(`${name} must be a positive integer`);
  }
  return parsed;
}

function frozenPositiveInteger(value, name, expected) {
  const parsed = positiveInteger(value, name);
  if (parsed !== expected) throw new Error(`${name} must be exactly ${expected}`);
  return parsed;
}

function frozenViews(value) {
  const views = value.split(",").map((item) => item.trim()).filter(Boolean);
  if (views.length !== REQUIRED_VIEWS.length || views.some((view, index) => view !== REQUIRED_VIEWS[index])) {
    throw new Error(`PROFILE_VIEWS must equal ${REQUIRED_VIEWS.join(",")}`);
  }
  return views;
}

function settledAt(quietStreak, elapsedMs, timeoutMs) {
  return quietStreak >= 3 && elapsedMs <= timeoutMs ? elapsedMs : undefined;
}

function profileFailed(phases, errors, failedRequests) {
  return errors.length > 0 || failedRequests.length > 0 || phases.some((phase) => phase.timedOut);
}

function remainingStartupMs(startedAt, timeoutMs, now = Date.now()) {
  const remaining = timeoutMs - (now - startedAt);
  if (remaining <= 0) throw new Error("startup timeout exceeded");
  return remaining;
}

function summarizeFinite(values) {
  let count = 0;
  let min = Infinity;
  let max = -Infinity;
  let sum = 0;
  for (const value of values) {
    if (!Number.isFinite(value)) continue;
    count += 1;
    min = Math.min(min, value);
    max = Math.max(max, value);
    sum += value;
  }
  return {
    count,
    min: count ? min : null,
    max: count ? max : null,
    mean: count ? sum / count : null,
  };
}

function validateViewerBinding(identity, config, digests) {
  const pair = PROFILE_PAIRS.get(
    `${identity.models.left.candidateId}/${identity.models.right.candidateId}`,
  );
  const left = config?.models?.terra;
  const right = config?.models?.a1;
  if (
    !pair
    || left?.label !== pair.leftLabel
    || right?.label !== pair.rightLabel
    || left?.tilesetUrl !== VIEWER_TILESET_URLS.left
    || right?.tilesetUrl !== VIEWER_TILESET_URLS.right
    || digests.left !== pair.leftDigest
    || digests.right !== pair.rightDigest
  ) {
    throw new Error("viewer binding does not match the declared candidate pair");
  }
  return {
    models: {
      left: {
        candidateId: identity.models.left.candidateId,
        serviceLabel: left.label,
        tilesetUrl: left.tilesetUrl,
        tilesetDigest: digests.left,
      },
      right: {
        candidateId: identity.models.right.candidateId,
        serviceLabel: right.label,
        tilesetUrl: right.tilesetUrl,
        tilesetDigest: digests.right,
      },
    },
  };
}

function viewerTilesetUrls(config) {
  const left = config?.models?.terra?.tilesetUrl;
  const right = config?.models?.a1?.tilesetUrl;
  if (left !== VIEWER_TILESET_URLS.left || right !== VIEWER_TILESET_URLS.right) {
    throw new Error("viewer binding does not match the declared candidate pair");
  }
  return { left, right };
}

function splitLabels(models) {
  return {
    left: `LEFT · ${models.left.label}`,
    right: `${models.right.label} · RIGHT`,
  };
}

module.exports = {
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
};
