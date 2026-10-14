"""Fetch verified public ODX OCI layers and source; unpack with upstream umoci.

Only writes beneath the supplied persistent runtime directory. Uses stdlib.
An image VERSION alone does not establish source identity. Overlay is permitted
only after all pinned build definitions and Python requirements match the image.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tarfile
import time
import urllib.error
import urllib.request

REVISION = "eb9fd56ac430fc3735401d88aaef9945ebd00e14"
IMAGES = {
    "cpu": "sha256:4c9d191ab09b7344212491733391f9da8280f9cf6c99bdb35d39cb6f1d389408",
    "gpu": "sha256:f67dac749fa33009ae4f42bcf84e36f14b39bf5f7d280d838f87b8b6128c14a5",
}
REGISTRY = "https://registry-1.docker.io/v2/webodm/odx"
USER_AGENT = "mesh-research-runtime-recovery/1"


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return "sha256:" + hashlib.file_digest(handle, "sha256").hexdigest()


def request(url: str, headers=None):
    return urllib.request.urlopen(urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, **(headers or {})}), timeout=90)


def get_json(url: str, headers=None):
    with request(url, headers) as handle:
        return json.loads(handle.read(4 * 1024 * 1024))


def write_json(path: Path, value):
    part = path.with_suffix(path.suffix + ".tmp")
    part.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    part.replace(path)


def fetch(url: str, destination: Path, expected_digest=None, expected_size=None, headers=None):
    if destination.exists():
        if expected_digest and digest(destination) != expected_digest:
            raise RuntimeError("Existing verified-cache path has a different hash: " + str(destination))
        if expected_size is not None and destination.stat().st_size != expected_size:
            raise RuntimeError("Existing verified-cache path has a different size: " + str(destination))
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    if partial.exists() and expected_size is not None and partial.stat().st_size == expected_size:
        if not expected_digest or digest(partial) == expected_digest:
            partial.replace(destination)
            return
        raise RuntimeError("Complete partial-cache file has a SHA-256 mismatch")
    for attempt in range(6):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            extra = {**(headers or {})}
            if offset:
                extra["Range"] = f"bytes={offset}-"
            with request(url, extra) as response:
                if offset and response.status != 206:
                    offset = 0
                with partial.open("ab" if offset else "wb") as stream:
                    while data := response.read(4 * 1024 * 1024):
                        stream.write(data)
            if expected_size is not None and partial.stat().st_size != expected_size:
                raise RuntimeError("Downloaded size mismatch")
            if expected_digest and digest(partial) != expected_digest:
                raise RuntimeError("Downloaded SHA-256 mismatch")
            partial.replace(destination)
            print(json.dumps({"downloaded": destination.name, "bytes": destination.stat().st_size}), flush=True)
            return
        except Exception as error:
            if "SHA-256 mismatch" in str(error):
                raise
            if attempt == 5:
                raise
            print(json.dumps({"retry": attempt + 1, "file": destination.name,
                              "error": type(error).__name__}), flush=True)
            time.sleep(min(5 * (attempt + 1), 25))


def obtain_umoci(runtime: Path) -> Path:
    existing = shutil.which("umoci")
    if existing:
        return Path(existing)
    tools = runtime / "tools"
    tools.mkdir(exist_ok=True)
    binary = tools / "umoci-0.5.1-amd64"
    if binary.exists() and (tools / "umoci-provenance.json").exists():
        previous = json.loads((tools / "umoci-provenance.json").read_text())
        if digest(binary) != previous["sha256"]:
            raise RuntimeError("umoci binary changed")
        return binary
    release = get_json("https://api.github.com/repos/opencontainers/umoci/releases/tags/v0.5.1")
    assets = {item["name"]: item for item in release["assets"]}
    name = next((name for name in assets if name in {"umoci.linux.amd64", "umoci.amd64", "umoci.x86_64"}), None)
    if name is None:
        raise RuntimeError("Cannot identify official static amd64 umoci release asset")
    asset = assets[name]
    expected = asset.get("digest")
    # Fixed official v0.5.1 linux-amd64 SHA, also published in expanded_assets.
    if name == "umoci.linux.amd64":
        expected = "sha256:4406d5ea5dff7ad92982f592e80ad665eb90318e083b1adb57083528bd19fa84"
    checksum_asset = assets.get(name + ".sha256sum") or assets.get("umoci.sha256sum")
    if checksum_asset:
        checksum_path = tools / checksum_asset["name"]
        fetch(checksum_asset["browser_download_url"], checksum_path)
        lines = [line for line in checksum_path.read_text().splitlines()
                 if len(line.split()) >= 2 and line.split()[-1].lstrip("*") == name]
        if len(lines) != 1 or not re.fullmatch(r"[a-fA-F0-9]{64}", lines[0].split()[0]):
            raise RuntimeError("Invalid upstream umoci checksum asset")
        checksum_digest = "sha256:" + lines[0].split()[0].lower()
        if expected and expected != checksum_digest:
            raise RuntimeError("Official release asset digest and checksum disagree")
        expected = checksum_digest
    if not expected or not re.fullmatch(r"sha256:[a-f0-9]{64}", expected):
        raise RuntimeError("No upstream SHA-256 available for umoci release binary")
    fetch(asset["browser_download_url"], binary, expected, asset["size"])
    binary.chmod(0o755)
    write_json(tools / "umoci-provenance.json", {
        "url": asset["browser_download_url"], "version": "0.5.1", "sha256": digest(binary),
        "expectedDigest": expected,
    })
    return binary


def obtain_source(runtime: Path) -> Path:
    source = runtime / "ODX-eb9fd56"
    if source.exists():
        if (source / "VERSION").read_text().strip() != "3.8.3":
            raise RuntimeError("Existing pinned-source VERSION differs")
        return source
    archive_path = runtime / ("ODX-" + REVISION + ".tar.gz")
    url = "https://codeload.github.com/WebODM/ODX/tar.gz/" + REVISION
    fetch(url, archive_path)
    partial = runtime / "ODX-eb9fd56.partial"
    if partial.exists():
        raise RuntimeError("An incomplete source unpack exists; inspect it before resuming")
    partial.mkdir()
    with tarfile.open(archive_path) as archive:
        members = archive.getmembers()
        prefix = "ODX-" + REVISION
        for member in members:
            path = PurePosixPath(member.name)
            if not path.parts or path.parts[0] != prefix or path.is_absolute() or ".." in path.parts:
                raise RuntimeError("Unsafe source archive path")
            if member.issym() or member.islnk() or member.isdev():
                raise RuntimeError("Source archive link/device requires explicit review")
        archive.extractall(partial, filter="data")
    unpacked = partial / ("ODX-" + REVISION)
    unpacked.rename(source)
    partial.rmdir()
    if (source / "VERSION").read_text().strip() != "3.8.3":
        raise RuntimeError("Pinned source VERSION differs")
    write_json(runtime / "source-provenance.json", {
        "revision": REVISION, "url": url, "archiveSha256": digest(archive_path),
        "version": "3.8.3",
    })
    return source


def obtain_image(runtime: Path, kind: str, umoci: Path) -> Path:
    image_dir = runtime / ("odx-oci-" + kind)
    image_dir.mkdir(exist_ok=True)
    blobs = image_dir / "blobs/sha256"
    blobs.mkdir(parents=True, exist_ok=True)
    token = get_json("https://auth.docker.io/token?service=registry.docker.io&scope=repository:webodm/odx:pull")["token"]
    headers = {"Authorization": "Bearer " + token,
               "Accept": "application/vnd.docker.distribution.manifest.v2+json,application/vnd.oci.image.manifest.v1+json"}
    original_digest = IMAGES[kind]
    remote_path = runtime / (kind + "-registry-manifest.json")
    fetch(REGISTRY + "/manifests/" + original_digest, remote_path, original_digest, headers=headers)
    manifest = json.loads(remote_path.read_text())
    if "layers" not in manifest:
        raise RuntimeError("Requested manifest is an index; a platform image is required")
    for descriptor in [manifest["config"], *manifest["layers"]]:
        sha = descriptor["digest"]
        if not re.fullmatch(r"sha256:[a-f0-9]{64}", sha):
            raise RuntimeError("Unexpected blob digest")
        # Docker Hub bearer tokens are short lived; each new blob gets a fresh
        # token, so a large prior layer cannot invalidate the next request.
        token = get_json("https://auth.docker.io/token?service=registry.docker.io&scope=repository:webodm/odx:pull")["token"]
        headers["Authorization"] = "Bearer " + token
        fetch(REGISTRY + "/blobs/" + sha, blobs / sha.split(":")[1],
              sha, descriptor["size"], headers=headers)
    # OCI media type conversion preserves all verified layer/config blob bytes.
    manifest["mediaType"] = "application/vnd.oci.image.manifest.v1+json"
    manifest["config"]["mediaType"] = "application/vnd.oci.image.config.v1+json"
    mapping = {
        "application/vnd.docker.image.rootfs.diff.tar.gzip": "application/vnd.oci.image.layer.v1.tar+gzip",
        "application/vnd.docker.image.rootfs.diff.tar": "application/vnd.oci.image.layer.v1.tar",
    }
    for layer in manifest["layers"]:
        layer["mediaType"] = mapping.get(layer["mediaType"], layer["mediaType"])
    manifest_bytes = json.dumps(manifest, separators=(",", ":")).encode()
    oci_digest = "sha256:" + hashlib.sha256(manifest_bytes).hexdigest()
    (blobs / oci_digest.split(":")[1]).write_bytes(manifest_bytes)
    write_json(image_dir / "oci-layout", {"imageLayoutVersion": "1.0.0"})
    write_json(image_dir / "index.json", {"schemaVersion": 2, "manifests": [{
        "mediaType": manifest["mediaType"], "digest": oci_digest,
        "size": len(manifest_bytes), "annotations": {"org.opencontainers.image.ref.name": "odx"},
    }]})
    bundle = runtime / ("odx-runtime-" + kind)
    if not bundle.exists():
        subprocess.run([str(umoci), "unpack", "--rootless", "--image", str(image_dir) + ":odx", str(bundle)], check=True)
    if not (bundle / "rootfs/code/VERSION").is_file():
        raise RuntimeError("Incomplete OCI unpack")
    write_json(bundle / "image-provenance.json", {
        "repository": "docker.io/webodm/odx", "registryManifestDigest": original_digest,
        "derivedOciManifestDigest": oci_digest,
        "compressedBytes": sum(item["size"] for item in manifest["layers"]),
        "versionBeforeOverlay": (bundle / "rootfs/code/VERSION").read_text().strip(),
        "unpacker": str(umoci), "scriptSha256": digest(Path(__file__)),
    })
    return bundle


def pin_source(bundle: Path, source: Path):
    root = bundle / "rootfs"
    code = root / "code"
    proof = bundle / "source-overlay-validation.json"
    if proof.exists():
        record = json.loads(proof.read_text())
        if record.get("state") != "pinned":
            raise RuntimeError("Prior source validation rejected this runtime")
        for relative, expected in record["pinnedSourceFiles"].items():
            if digest(code / relative) != expected:
                raise RuntimeError("Pinned source changed after overlay: " + relative)
        return
    build_files = [path for path in (source / "SuperBuild").rglob("*") if path.is_file()]
    build_files.append(source / "requirements.txt")
    mismatches = []
    matched = []
    for path in build_files:
        relative = path.relative_to(source)
        target = code / relative
        if not target.is_file() or digest(path) != digest(target):
            mismatches.append(str(relative))
        else:
            matched.append(str(relative))
    record = {"state": "rejected" if mismatches else "validated", "sourceRevision": REVISION,
              "matchedBuildDefinitions": matched, "mismatchedBuildDefinitions": mismatches}
    write_json(proof, record)
    if mismatches:
        raise RuntimeError("Image dependency build definitions differ from pinned source: " + ", ".join(mismatches))
    original = root / "code-image-original"
    if original.exists():
        raise RuntimeError("Original-source backup path already exists")
    code.rename(original)
    shutil.copytree(source, code)
    # Move only compiled dependencies. The original Python source remains backed up.
    (original / "SuperBuild/install").rename(code / "SuperBuild/install")
    record.update(state="pinned", version="3.8.3", originalCodeBackup=str(original),
                  pinnedSourceFiles={str(path.relative_to(source)): digest(path)
                                     for path in source.rglob("*") if path.is_file()})
    write_json(proof, record)
    print(json.dumps({"pinned": str(code), "revision": REVISION}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--kind", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--source-only", action="store_true")
    args = parser.parse_args()
    runtime = args.runtime.resolve()
    allowed = Path("/volume/wzhang/mesh-research-workspace/runtime")
    if runtime != allowed:
        raise RuntimeError("This recovery launcher may only write to its assigned runtime directory")
    runtime.mkdir(parents=True, exist_ok=True)
    source = obtain_source(runtime)
    if args.source_only:
        return
    umoci = obtain_umoci(runtime)
    bundle = obtain_image(runtime, args.kind, umoci)
    pin_source(bundle, source)
    print(json.dumps({"state": "ready-for-smoke-test", "bundle": str(bundle),
                      "version": "3.8.3", "revision": REVISION}), flush=True)


if __name__ == "__main__":
    main()
