#!/usr/bin/env python3
"""Validate the frozen candidate-comparison protocol and image identity."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from core.artifacts import image_tree_identity

FROZEN_PROTOCOL_DIGEST = "sha256:b36886942e7a0a6c9db8c14e31bd631707d73a13d7bc4b544ecafd4881e19aee"


def protocol_digest(protocol: object) -> str:
    canonical = json.dumps(protocol, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


def validate_protocol(protocol: dict[str, Any]) -> None:
    """Reject any change from the reviewed v55 clean-break comparison contract."""
    observed = protocol_digest(protocol)
    if observed != FROZEN_PROTOCOL_DIGEST:
        raise ValueError(f"protocol digest mismatch: expected {FROZEN_PROTOCOL_DIGEST}, observed {observed}")


def verify_protocol(protocol: dict[str, Any], image_root: Path) -> dict[str, object]:
    validate_protocol(protocol)
    observed = image_tree_identity(image_root)
    expected = protocol["dataset"]["images"]
    if observed != expected:
        raise ValueError(
            "image identity mismatch: "
            f"expected {json.dumps(expected, sort_keys=True)}, "
            f"observed {json.dumps(observed, sort_keys=True)}"
        )
    return {
        "schemaVersion": "mesh-candidate-protocol-validation.v1",
        "protocolSchemaVersion": protocol["schemaVersion"],
        "protocolDigest": protocol_digest(protocol),
        "datasetId": protocol["dataset"]["id"],
        "acquisitionId": protocol["dataset"]["acquisitionId"],
        "observedImages": observed,
        "candidateIds": [candidate["id"] for candidate in protocol["candidates"]],
        "valid": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    report = verify_protocol(protocol, args.image_root)
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
