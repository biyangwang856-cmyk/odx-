#!/usr/bin/env node
"use strict";

const assert = require("node:assert/strict");
const crypto = require("node:crypto");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { execFile } = require("node:child_process");
const { promisify } = require("node:util");

const run = promisify(execFile);
const KTX2_MAGIC = Buffer.from([0xab, 0x4b, 0x54, 0x58, 0x20, 0x32, 0x30, 0xbb, 0x0d, 0x0a, 0x1a, 0x0a]);
const DEFAULT_POLICY = Object.freeze({
  codec: "etc1s",
  quality: 255,
  compressionLevel: 1,
  maxThreads: 2,
  minimumBc7RgbPsnr: 32,
  maximumByteRatio: 1.15,
});

function sha256(bytes) {
  return crypto.createHash("sha256").update(bytes).digest("hex");
}

function aligned(value) {
  return (value + 3) & ~3;
}

function readGlb(bytes) {
  if (bytes.length < 20 || bytes.subarray(0, 4).toString("ascii") !== "glTF") {
    throw new Error("Invalid GLB header");
  }
  if (bytes.readUInt32LE(4) !== 2 || bytes.readUInt32LE(8) !== bytes.length) {
    throw new Error("Invalid GLB version or declared length");
  }
  const jsonLength = bytes.readUInt32LE(12);
  if (bytes.readUInt32LE(16) !== 0x4e4f534a || jsonLength % 4) {
    throw new Error("Invalid GLB JSON chunk");
  }
  const binaryHeader = 20 + jsonLength;
  if (binaryHeader + 8 > bytes.length) throw new Error("Missing GLB BIN chunk");
  const binaryLength = bytes.readUInt32LE(binaryHeader);
  if (bytes.readUInt32LE(binaryHeader + 4) !== 0x004e4942 || binaryHeader + 8 + binaryLength !== bytes.length) {
    throw new Error("Invalid GLB BIN chunk");
  }
  const document = JSON.parse(bytes.subarray(20, binaryHeader).toString("utf8").replace(/[\u0000 ]+$/u, ""));
  const binary = bytes.subarray(binaryHeader + 8);
  if (!Array.isArray(document.buffers) || !document.buffers.length) {
    throw new Error("GLB physical buffer is missing");
  }
  const physical = document.buffers[0];
  if (
    !Number.isInteger(physical.byteLength)
    || physical.byteLength < 0
    || physical.byteLength > binary.length
    || binary.length - physical.byteLength > 3
  ) throw new Error("GLB physical buffer length is invalid");
  for (let index = 1; index < document.buffers.length; index += 1) {
    const fallback = document.buffers[index];
    if (
      fallback.uri !== undefined
      || !Number.isInteger(fallback.byteLength)
      || fallback.byteLength < 0
      || fallback.extensions?.EXT_meshopt_compression?.fallback !== true
    ) throw new Error("Only declared Meshopt fallback buffers are supported");
  }
  for (const view of document.bufferViews || []) {
    if (!Number.isInteger(view.buffer) || !document.buffers[view.buffer]) {
      throw new Error("BufferView references a missing buffer");
    }
    const offset = view.byteOffset || 0;
    if (!Number.isInteger(offset) || !Number.isInteger(view.byteLength) || offset < 0 || view.byteLength < 0) {
      throw new Error("BufferView range is invalid");
    }
    if (offset + view.byteLength > document.buffers[view.buffer].byteLength) {
      throw new Error("BufferView exceeds its declared buffer");
    }
    const meshopt = view.extensions?.EXT_meshopt_compression;
    if (view.buffer !== 0 && !meshopt) throw new Error("Meshopt fallback BufferView is missing compression data");
    if (meshopt) {
      const compressedOffset = meshopt.byteOffset || 0;
      if (
        meshopt.buffer !== 0
        || !Number.isInteger(compressedOffset)
        || !Number.isInteger(meshopt.byteLength)
        || compressedOffset < 0
        || meshopt.byteLength < 0
        || compressedOffset + meshopt.byteLength > physical.byteLength
      ) throw new Error("Meshopt compressed range is invalid");
    }
  }
  return { document, binary };
}

function glbBytes(document, binary) {
  const json = Buffer.from(JSON.stringify(document));
  const jsonPadded = Buffer.alloc(aligned(json.length), 0x20);
  json.copy(jsonPadded);
  const binaryPadded = Buffer.alloc(aligned(binary.length));
  binary.copy(binaryPadded);
  const output = Buffer.alloc(12 + 8 + jsonPadded.length + 8 + binaryPadded.length);
  output.write("glTF", 0, "ascii");
  output.writeUInt32LE(2, 4);
  output.writeUInt32LE(output.length, 8);
  output.writeUInt32LE(jsonPadded.length, 12);
  output.writeUInt32LE(0x4e4f534a, 16);
  jsonPadded.copy(output, 20);
  const binaryHeader = 20 + jsonPadded.length;
  output.writeUInt32LE(binaryPadded.length, binaryHeader);
  output.writeUInt32LE(0x004e4942, binaryHeader + 4);
  binaryPadded.copy(output, binaryHeader + 8);
  return output;
}

function textureSource(texture) {
  const extensions = texture.extensions || {};
  const webp = extensions.EXT_texture_webp;
  if (webp !== undefined && (typeof webp !== "object" || webp === null || !Number.isInteger(webp.source))) {
    throw new Error("Invalid EXT_texture_webp source");
  }
  const source = webp?.source ?? texture.source;
  if (!Number.isInteger(source)) throw new Error("Texture source is missing");
  return source;
}

function eligibleImages(document) {
  const permitted = new Set();
  const blocked = new Set();
  const materials = document.materials || [];
  const textures = document.textures || [];
  const samplers = document.samplers || [];
  const usages = [
    ["color", (material) => material.pbrMetallicRoughness?.baseColorTexture],
    ["color", (material) => material.emissiveTexture],
    ["data", (material) => material.normalTexture],
    ["data", (material) => material.occlusionTexture],
    ["data", (material) => material.pbrMetallicRoughness?.metallicRoughnessTexture],
  ];
  const referencedMaterials = new Set();
  for (const mesh of document.meshes || []) {
    for (const primitive of mesh.primitives || []) {
      if (primitive.material !== undefined) referencedMaterials.add(primitive.material);
    }
  }
  for (const materialIndex of referencedMaterials) {
    const material = materials[materialIndex];
    if (!material) throw new Error("Primitive references a missing material");
    if (material.extensions && Object.keys(material.extensions).length) {
      throw new Error("Material extension adapter is required");
    }
    for (const [kind, read] of usages) {
      const info = read(material);
      if (!info) continue;
      const texture = textures[info.index];
      if (!texture) throw new Error("Material references a missing texture");
      const extensionNames = Object.keys(texture.extensions || {});
      if (extensionNames.some((name) => name !== "EXT_texture_webp")) {
        throw new Error("Texture variant adapter is required");
      }
      const source = textureSource(texture);
      if (texture.source !== undefined && texture.extensions?.EXT_texture_webp && texture.source !== source) {
        throw new Error("Texture source variants differ");
      }
      const sampler = texture.sampler === undefined ? {} : samplers[texture.sampler];
      if (!sampler) throw new Error("Texture references a missing sampler");
      if (kind === "data" || sampler.minFilter !== 9728 || sampler.magFilter !== 9728) blocked.add(source);
      else permitted.add(source);
    }
  }
  return [...permitted].filter((source) => !blocked.has(source)).sort((a, b) => a - b);
}

function imageBytes(document, binary, imageIndex) {
  const image = document.images?.[imageIndex];
  if (!image || image.uri !== undefined || !Number.isInteger(image.bufferView)) {
    throw new Error("Only embedded image sources are supported");
  }
  const view = document.bufferViews?.[image.bufferView];
  if (!view || view.buffer !== 0) throw new Error("Image is not stored in the physical GLB buffer");
  const offset = view.byteOffset || 0;
  return binary.subarray(offset, offset + view.byteLength);
}

function ktx2Info(bytes) {
  if (bytes.length < 104 || !bytes.subarray(0, 12).equals(KTX2_MAGIC)) throw new Error("Invalid KTX2 output");
  const width = bytes.readUInt32LE(20);
  const height = bytes.readUInt32LE(24);
  const depth = bytes.readUInt32LE(28);
  const layers = bytes.readUInt32LE(32);
  const faces = bytes.readUInt32LE(36);
  const levels = bytes.readUInt32LE(40);
  if (!width || !height || depth !== 0 || layers !== 0 || faces !== 1 || levels !== 1) {
    throw new Error("Only single-level 2D KTX2 textures are supported");
  }
  return { width, height, levels };
}

function repack(document, binary, replacements) {
  const next = structuredClone(document);
  const chunks = [];
  const segments = new Map();
  let length = 0;
  const append = (bytes, key) => {
    if (segments.has(key)) return segments.get(key);
    const padding = (4 - (length % 4)) % 4;
    if (padding) chunks.push(Buffer.alloc(padding));
    length += padding;
    const offset = length;
    chunks.push(bytes);
    length += bytes.length;
    segments.set(key, offset);
    return offset;
  };

  const replacementViews = new Map();
  for (const [imageIndex, bytes] of replacements) {
    const image = next.images?.[imageIndex];
    if (!image || image.uri !== undefined || !Number.isInteger(image.bufferView)) {
      throw new Error("Only embedded image replacement is supported");
    }
    if (replacementViews.has(image.bufferView)) throw new Error("Shared image bufferView adapter is required");
    if ((document.accessors || []).some((accessor) => accessor.bufferView === image.bufferView)) {
      throw new Error("Image bufferView overlaps a geometry accessor");
    }
    replacementViews.set(image.bufferView, bytes);
    image.mimeType = "image/ktx2";
  }

  for (let index = 0; index < next.bufferViews.length; index += 1) {
    const view = next.bufferViews[index];
    const original = document.bufferViews[index];
    if (replacementViews.has(index)) {
      const bytes = replacementViews.get(index);
      view.byteOffset = append(bytes, `image:${index}`);
      view.byteLength = bytes.length;
    } else if (view.buffer === 0) {
      const offset = original.byteOffset || 0;
      view.byteOffset = append(binary.subarray(offset, offset + original.byteLength), `view:${offset}:${original.byteLength}`);
    }
    const meshopt = view.extensions?.EXT_meshopt_compression;
    const originalMeshopt = original.extensions?.EXT_meshopt_compression;
    if (meshopt) {
      if (!originalMeshopt || originalMeshopt.buffer !== 0) throw new Error("Compressed geometry adapter is required");
      const offset = originalMeshopt.byteOffset || 0;
      meshopt.byteOffset = append(
        binary.subarray(offset, offset + originalMeshopt.byteLength),
        `meshopt:${offset}:${originalMeshopt.byteLength}`,
      );
    }
  }

  for (const texture of next.textures || []) {
    const source = textureSource(texture);
    if (!replacements.has(source)) continue;
    delete texture.source;
    texture.extensions = { ...(texture.extensions || {}), KHR_texture_basisu: { source } };
    delete texture.extensions.EXT_texture_webp;
  }
  next.extensionsUsed = [...new Set([...(next.extensionsUsed || []), "KHR_texture_basisu"])];
  next.extensionsRequired = [...new Set([...(next.extensionsRequired || []), "KHR_texture_basisu"])];
  if (!(next.textures || []).some((texture) => texture.extensions?.EXT_texture_webp)) {
    next.extensionsUsed = next.extensionsUsed.filter((name) => name !== "EXT_texture_webp");
    next.extensionsRequired = next.extensionsRequired.filter((name) => name !== "EXT_texture_webp");
  }
  next.buffers[0].byteLength = length;
  return glbBytes(next, Buffer.concat(chunks));
}

function accessorPayload(document, binary, accessor) {
  if (!Number.isInteger(accessor.bufferView)) throw new Error("Sparse or implicit accessor adapter is required");
  const view = document.bufferViews[accessor.bufferView];
  const meshopt = view.extensions?.EXT_meshopt_compression;
  const offset = meshopt?.byteOffset ?? view.byteOffset ?? 0;
  const length = meshopt?.byteLength ?? view.byteLength;
  if ((meshopt?.buffer ?? view.buffer) !== 0) return null;
  return binary.subarray(offset, offset + length);
}

function normalizedView(view) {
  const result = structuredClone(view);
  if (result.buffer === 0) delete result.byteOffset;
  if (result.extensions?.EXT_meshopt_compression) delete result.extensions.EXT_meshopt_compression.byteOffset;
  return result;
}

function geometryPreserved(originalBytes, candidateBytes) {
  const original = readGlb(originalBytes);
  const candidate = readGlb(candidateBytes);
  for (const key of ["nodes", "scenes", "meshes", "accessors", "materials", "samplers"]) {
    assert.deepEqual(candidate.document[key], original.document[key], `Scene/attribute/material semantics changed: ${key}`);
  }
  for (let index = 0; index < original.document.accessors.length; index += 1) {
    const beforeAccessor = original.document.accessors[index];
    const afterAccessor = candidate.document.accessors[index];
    const beforeView = original.document.bufferViews[beforeAccessor.bufferView];
    const afterView = candidate.document.bufferViews[afterAccessor.bufferView];
    const before = accessorPayload(original.document, original.binary, beforeAccessor);
    const after = accessorPayload(candidate.document, candidate.binary, afterAccessor);
    if (!before || !after || !before.equals(after)) throw new Error("Geometry or UV buffer changed");
    assert.deepEqual(normalizedView(afterView), normalizedView(beforeView), "Geometry decoding semantics changed");
  }
  return true;
}

async function encodeTexture(source, sharp, basisu, work, policy) {
  const sourceDigest = sha256(source);
  const metadata = await sharp(source).metadata();
  if (
    metadata.icc
    || (metadata.orientation && metadata.orientation !== 1)
    || metadata.pages > 1
    || !metadata.width
    || !metadata.height
    || metadata.width * metadata.height > 16_777_216
  ) return { status: "original-preserved", reason: "unsupported-image-metadata-or-size", sourceSha256: sourceDigest };

  const pixels = await sharp(source).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
  if (pixels.info.width % 4 || pixels.info.height % 4 || pixels.info.channels !== 4) {
    return { status: "original-preserved", reason: "unsupported-block-dimensions", sourceSha256: sourceDigest };
  }
  for (let index = 3; index < pixels.data.length; index += 4) {
    if (pixels.data[index] !== 255) {
      return { status: "original-preserved", reason: "alpha-adapter-required", sourceSha256: sourceDigest };
    }
  }

  const input = path.join(work, `${sourceDigest}.png`);
  const output = path.join(work, `${sourceDigest}.ktx2`);
  fs.writeFileSync(input, await sharp(pixels.data, { raw: pixels.info }).png().toBuffer());
  const args = [
    "-ktx2", "-q", String(policy.quality), "-comp_level", String(policy.compressionLevel),
    "-max_threads", String(policy.maxThreads), "-no_alpha", "-stats",
    "-file", input, "-output_file", output,
  ];
  const completed = await run(basisu, args, { timeout: 120_000, maxBuffer: 4 * 1024 * 1024 });
  const log = `${completed.stdout}\n${completed.stderr}`;
  const match = log.match(/BC7 RGB Avg:[^\n]*PSNR:\s*([\d.]+) dB/u);
  if (!match) throw new Error("BasisU did not emit the required BC7 RGB quality metric");
  const encoded = fs.readFileSync(output);
  const info = ktx2Info(encoded);
  if (info.width !== metadata.width || info.height !== metadata.height) throw new Error("BasisU resized the texture");
  const psnr = Number(match[1]);
  const qualityPassed = psnr >= policy.minimumBc7RgbPsnr;
  const byteBudgetPassed = encoded.length <= source.length * policy.maximumByteRatio;
  return {
    status: qualityPassed && byteBudgetPassed ? "converted" : "original-preserved",
    reason: qualityPassed ? (byteBudgetPassed ? null : "encoded-byte-budget") : "quality-threshold",
    sourceSha256: sourceDigest,
    sourceBytes: source.length,
    encodedSha256: sha256(encoded),
    encodedBytes: encoded.length,
    bc7RgbPsnr: psnr,
    qualityPassed,
    byteBudgetPassed,
    width: info.width,
    height: info.height,
    bytes: qualityPassed && byteBudgetPassed ? encoded : undefined,
  };
}

function lockedToolchain(basisu, provenancePath, policy) {
  const provenance = JSON.parse(fs.readFileSync(provenancePath, "utf8"));
  const binary = fs.readFileSync(basisu);
  const binarySha256 = sha256(binary);
  if (
    provenance.version !== "1.16.3"
    || provenance.gitCommit !== "1531cfaf9ed5232248a0a45736686a849ca3befc"
    || provenance.sourceSha256 !== "b89563aa5879eed20f56b9cfa03b52848e759531fd5a1d51a8f63c846f96c2ac"
    || provenance.binaryPlatform !== "linux-x86_64"
    || provenance.binarySha256 !== binarySha256
    || provenance.binaryBytes !== binary.length
  ) throw new Error("BasisU provenance does not identify the locked 1.16.3 Linux toolchain");
  return {
    basisUniversalVersion: provenance.version,
    basisUniversalGitCommit: provenance.gitCommit,
    basisUniversalSourceSha256: provenance.sourceSha256,
    basisUniversalBinarySha256: binarySha256,
    policySha256: sha256(Buffer.from(JSON.stringify(policy))),
  };
}

async function optimizeFile(sourcePath, destinationPath, options) {
  const policy = { ...DEFAULT_POLICY, ...(options.policy || {}) };
  if (policy.codec !== "etc1s") throw new Error("Only ETC1S is supported by the v55 clean-break policy");
  const basisu = path.resolve(options.basisu);
  const provenancePath = path.resolve(options.provenance);
  const toolchain = lockedToolchain(basisu, provenancePath, policy);
  const sharp = require(path.resolve(options.sharpModule));
  const originalBytes = fs.readFileSync(sourcePath);
  const originalDigest = sha256(originalBytes);
  const { document, binary } = readGlb(originalBytes);
  const candidates = eligibleImages(document);
  const work = fs.mkdtempSync(path.join(os.tmpdir(), "mesh-v55-textures-"));
  const cache = new Map();
  const rows = [];
  const replacements = new Map();
  try {
    for (const imageIndex of candidates) {
      const source = imageBytes(document, binary, imageIndex);
      const digest = sha256(source);
      let result = cache.get(digest);
      if (!result) {
        result = await encodeTexture(source, sharp, basisu, work, policy);
        cache.set(digest, result);
      }
      rows.push({ imageIndex, ...result, bytes: undefined });
      if (result.status === "converted") replacements.set(imageIndex, result.bytes);
    }
    const candidateBytes = replacements.size ? repack(document, binary, replacements) : originalBytes;
    geometryPreserved(originalBytes, candidateBytes);
    if (sha256(fs.readFileSync(sourcePath)) !== originalDigest) throw new Error("Source GLB changed during optimization");
    fs.mkdirSync(path.dirname(destinationPath), { recursive: true });
    const temporary = `${destinationPath}.tmp-${process.pid}`;
    fs.writeFileSync(temporary, candidateBytes);
    fs.renameSync(temporary, destinationPath);
    return {
      schemaVersion: "mesh-gpu-texture-report.v1",
      status: replacements.size ? "converted" : "unchanged-no-eligible-texture",
      sourceSha256: `sha256:${originalDigest}`,
      outputSha256: `sha256:${sha256(candidateBytes)}`,
      sourceBytes: originalBytes.length,
      outputBytes: candidateBytes.length,
      eligibleImages: candidates.length,
      convertedImages: replacements.size,
      minimumAcceptedBc7RgbPsnr: rows.some((row) => row.status === "converted")
        ? rows.filter((row) => row.status === "converted")
          .reduce((value, row) => Math.min(value, row.bc7RgbPsnr), Infinity)
        : null,
      policy,
      toolchain,
      images: rows,
      geometryAndUvBytesPreserved: true,
    };
  } finally {
    fs.rmSync(work, { recursive: true, force: true });
  }
}

async function main(argv) {
  const [source, destination, basisu, sharpModule, provenance, policyJson] = argv;
  if (!source || !destination || !basisu || !sharpModule || !provenance) {
    throw new Error("Usage: gpu_textures.cjs SOURCE.glb DESTINATION.glb BASISU SHARP_MODULE BASISU_PROVENANCE [POLICY_JSON]");
  }
  const policy = policyJson ? JSON.parse(policyJson) : undefined;
  const report = await optimizeFile(path.resolve(source), path.resolve(destination), {
    basisu,
    sharpModule,
    provenance,
    policy,
  });
  process.stdout.write(`${JSON.stringify(report)}\n`);
}

module.exports = {
  DEFAULT_POLICY,
  eligibleImages,
  geometryPreserved,
  glbBytes,
  imageBytes,
  ktx2Info,
  lockedToolchain,
  optimizeFile,
  readGlb,
  repack,
  textureSource,
};

if (require.main === module) {
  main(process.argv.slice(2)).catch((error) => {
    console.error(error.stack);
    process.exitCode = 1;
  });
}
