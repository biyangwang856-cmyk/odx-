"use strict";

const assert = require("node:assert/strict");
const crypto = require("node:crypto");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const test = require("node:test");
const gpu = require("../app/core/pipelines/gpu_textures.cjs");

function fixture() {
  const binary = Buffer.alloc(80);
  const document = {
    asset: { version: "2.0" },
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0 }],
    meshes: [{ primitives: [{ attributes: { POSITION: 0, TEXCOORD_0: 1 }, indices: 2, material: 0 }] }],
    accessors: [
      { bufferView: 1, componentType: 5126, count: 3, type: "VEC3" },
      { bufferView: 2, componentType: 5126, count: 3, type: "VEC2" },
      { bufferView: 3, componentType: 5123, count: 3, type: "SCALAR" },
    ],
    buffers: [{ byteLength: 80 }],
    bufferViews: [
      { buffer: 0, byteLength: 8 },
      { buffer: 0, byteOffset: 8, byteLength: 36 },
      { buffer: 0, byteOffset: 44, byteLength: 24 },
      { buffer: 0, byteOffset: 68, byteLength: 6 },
    ],
    images: [{ bufferView: 0, mimeType: "image/webp" }],
    textures: [{ sampler: 0, extensions: { EXT_texture_webp: { source: 0 } } }],
    materials: [{ pbrMetallicRoughness: { baseColorTexture: { index: 0 } } }],
    samplers: [{ minFilter: 9728, magFilter: 9728 }],
    extensionsUsed: ["EXT_texture_webp"],
    extensionsRequired: ["EXT_texture_webp"],
  };
  return { document, binary };
}

function withMeshoptFallback(value) {
  value.document.buffers.push({
    byteLength: 66,
    extensions: { EXT_meshopt_compression: { fallback: true } },
  });
  const compressed = [
    { byteOffset: 8, byteLength: 8, mode: "ATTRIBUTES", byteStride: 4, count: 3 },
    { byteOffset: 16, byteLength: 8, mode: "ATTRIBUTES", byteStride: 8, count: 3 },
    { byteOffset: 24, byteLength: 8, mode: "TRIANGLES", byteStride: 2, count: 3 },
  ];
  for (let index = 1; index < value.document.bufferViews.length; index += 1) {
    const view = value.document.bufferViews[index];
    view.buffer = 1;
    view.byteOffset = index === 1 ? 0 : index === 2 ? 36 : 60;
    view.extensions = { EXT_meshopt_compression: { buffer: 0, ...compressed[index - 1] } };
  }
  value.document.extensionsUsed.push("EXT_meshopt_compression");
  value.document.extensionsRequired.push("EXT_meshopt_compression");
  return value;
}

test("image repacking preserves exact geometry and UV bytes", () => {
  const value = fixture();
  const before = gpu.glbBytes(value.document, value.binary);
  const after = gpu.repack(value.document, value.binary, new Map([[0, Buffer.alloc(104, 9)]]));
  assert.equal(gpu.geometryPreserved(before, after), true);
  const parsed = gpu.readGlb(after);
  assert.equal(parsed.document.images[0].mimeType, "image/ktx2");
  assert.equal(parsed.document.textures[0].extensions.KHR_texture_basisu.source, 0);
  assert.equal(parsed.document.extensionsUsed.includes("EXT_texture_webp"), false);
  assert.equal(parsed.document.extensionsRequired.includes("KHR_texture_basisu"), true);
});

test("Meshopt fallback buffers preserve compressed geometry while repacking images", () => {
  const value = withMeshoptFallback(fixture());
  const before = gpu.glbBytes(value.document, value.binary);
  const after = gpu.repack(value.document, value.binary, new Map([[0, Buffer.alloc(104, 9)]]));
  assert.equal(gpu.geometryPreserved(before, after), true);
  const parsed = gpu.readGlb(after);
  assert.equal(parsed.document.buffers.length, 2);
  assert.equal(parsed.document.buffers[1].extensions.EXT_meshopt_compression.fallback, true);
  assert.equal(parsed.document.images[0].mimeType, "image/ktx2");
});

test("undeclared secondary buffers are rejected", () => {
  const value = fixture();
  value.document.buffers.push({ byteLength: 16 });
  assert.throws(
    () => gpu.readGlb(gpu.glbBytes(value.document, value.binary)),
    /Only declared Meshopt fallback buffers are supported/,
  );
});

test("altering geometry or UV bytes is rejected", () => {
  const value = fixture();
  const before = gpu.glbBytes(value.document, value.binary);
  const repacked = gpu.readGlb(gpu.repack(value.document, value.binary, new Map([[0, Buffer.alloc(104)]])));
  repacked.binary[repacked.document.bufferViews[2].byteOffset] = 1;
  assert.throws(
    () => gpu.geometryPreserved(before, gpu.glbBytes(repacked.document, repacked.binary)),
    /buffer changed/,
  );
});

test("sampler and node changes invalidate preservation", () => {
  const value = fixture();
  const before = gpu.glbBytes(value.document, value.binary);
  value.document.samplers[0].magFilter = 9729;
  assert.throws(() => gpu.geometryPreserved(before, gpu.glbBytes(value.document, value.binary)));
});

test("data textures and unsupported sampler demand remain original", () => {
  const value = fixture();
  assert.deepEqual(gpu.eligibleImages(value.document), [0]);
  value.document.materials[0].normalTexture = { index: 0 };
  assert.deepEqual(gpu.eligibleImages(value.document), []);
  delete value.document.materials[0].normalTexture;
  value.document.samplers[0].minFilter = 9987;
  assert.deepEqual(gpu.eligibleImages(value.document), []);
});

test("image accessor overlap is rejected", () => {
  const value = fixture();
  value.document.accessors[0].bufferView = 0;
  assert.throws(
    () => gpu.repack(value.document, value.binary, new Map([[0, Buffer.alloc(104)]])),
    /overlaps a geometry accessor/,
  );
});

test("locked toolchain binds official source identity, binary, and policy", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "basisu-provenance-test-"));
  try {
    const binary = path.join(root, "basisu");
    const provenance = path.join(root, "provenance.json");
    const binaryBytes = Buffer.from("locked-linux-elf");
    fs.writeFileSync(binary, binaryBytes);
    fs.writeFileSync(provenance, JSON.stringify({
      version: "1.16.3",
      gitCommit: "1531cfaf9ed5232248a0a45736686a849ca3befc",
      sourceSha256: "b89563aa5879eed20f56b9cfa03b52848e759531fd5a1d51a8f63c846f96c2ac",
      binaryPlatform: "linux-x86_64",
      binarySha256: crypto.createHash("sha256").update(binaryBytes).digest("hex"),
      binaryBytes: binaryBytes.length,
    }));
    const toolchain = gpu.lockedToolchain(binary, provenance, gpu.DEFAULT_POLICY);
    assert.equal(toolchain.basisUniversalVersion, "1.16.3");
    assert.match(toolchain.basisUniversalBinarySha256, /^[0-9a-f]{64}$/u);
    assert.match(toolchain.policySha256, /^[0-9a-f]{64}$/u);
    fs.writeFileSync(provenance, JSON.stringify({
      version: "1.16.3",
      gitCommit: "0".repeat(40),
      sourceSha256: "0".repeat(64),
    }));
    assert.throws(() => gpu.lockedToolchain(binary, provenance, gpu.DEFAULT_POLICY), /locked/);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("KTX2 adapter validates single-level 2D dimensions", () => {
  const bytes = Buffer.alloc(104);
  Buffer.from([171, 75, 84, 88, 32, 50, 48, 187, 13, 10, 26, 10]).copy(bytes);
  bytes.writeUInt32LE(64, 20);
  bytes.writeUInt32LE(64, 24);
  bytes.writeUInt32LE(1, 36);
  bytes.writeUInt32LE(1, 40);
  assert.deepEqual(gpu.ktx2Info(bytes), { width: 64, height: 64, levels: 1 });
  bytes.writeUInt32LE(2, 40);
  assert.throws(() => gpu.ktx2Info(bytes));
  assert.throws(() => gpu.ktx2Info(Buffer.alloc(104)));
});
