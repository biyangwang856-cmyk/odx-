"""Package real image bytes and an experiment profile for the ODX adapter."""

from __future__ import annotations

import argparse
import json
import tarfile
from io import BytesIO
from pathlib import Path

from core.artifacts import image_tree_identity
from core.contracts.photogrammetry_odx import validate_photogrammetry_odx_input


def prepare(images: Path, profile: Path, archive: Path) -> None:
    manifest = json.loads(profile.read_text(encoding="utf-8"))
    identity = image_tree_identity(images)
    manifest["images"] = {
        "artifactId": "artifact:images:" + identity["digest"].split(":", 1)[1][:24],
        "contentDigest": identity["digest"],
        "bytes": identity["bytes"],
        "fileCount": identity["count"],
        "treeAlgorithm": identity["algorithm"],
        "mediaType": "application/vnd.surveying.image-tree",
    }
    validate_photogrammetry_odx_input(manifest)
    encoded = (json.dumps(manifest, indent=2) + "\n").encode()
    with tarfile.open(archive, "x") as output:
        entry = tarfile.TarInfo("input-manifest.json")
        entry.size = len(encoded)
        output.addfile(entry, BytesIO(encoded))
        output.add(images, arcname="images")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.images, args.profile, args.archive)


if __name__ == "__main__":
    main()
