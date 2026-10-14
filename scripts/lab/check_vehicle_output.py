"""Read-only pixel integrity checks for a completed full OGS2 output.

This is a normal output check, not vehicle detection or a defect diagnosis.
It cannot prove vehicle identity, deghosting, complete shadows or survey accuracy.
Only the new JSON check record is written; all supplied inputs are read-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import rasterio as rio


def signature(path: Path) -> tuple[int, int]:
    info = path.stat()
    return info.st_size, info.st_mtime_ns


def grid(dataset) -> dict:
    return {
        "crs": str(dataset.crs),
        "transform": list(dataset.transform),
        "width": dataset.width,
        "height": dataset.height,
    }


def check(baseline: Path, candidate: Path, source_index: Path, report_path: Path) -> dict:
    paths = (baseline, candidate, source_index, report_path)
    before = {str(path): signature(path) for path in paths}
    report_bytes = report_path.read_bytes()
    report = json.loads(report_bytes)
    if report.get("schemaVersion") != "ogs2-run/v1":
        raise ValueError("Require the full OGS2 report schema ogs2-run/v1")
    raw_mapping = report.get("sourceIndex")
    if not isinstance(raw_mapping, dict):
        raise ValueError("Run report must contain the sourceIndex mapping")
    mapping = {}
    for key, name in raw_mapping.items():
        try:
            identifier = int(key)
        except (TypeError, ValueError) as error:
            raise ValueError("Source mapping keys must be positive integer strings") from error
        if str(identifier) != key or not 1 <= identifier <= 65535:
            raise ValueError("Source mapping keys must be canonical IDs in 1..65535")
        if not isinstance(name, str) or not name or Path(name).name != name:
            raise ValueError("Source mapping values must be local photo basenames")
        mapping[identifier] = name
    if len(set(mapping.values())) != len(mapping) or report.get("imageCount") != len(mapping):
        raise ValueError("Source mapping must uniquely cover report.imageCount photos")
    if not isinstance(report.get("output"), str) or Path(report["output"]).resolve() != candidate:
        raise ValueError("Run report output path does not identify the supplied candidate")

    with rio.open(baseline) as old, rio.open(candidate) as new, rio.open(source_index) as ids:
        if not old.crs or grid(old) != grid(new) or grid(old) != grid(ids):
            raise ValueError("Baseline, candidate and source-index must share the exact map grid")
        if old.count != 4 or new.count != 4 or old.dtypes != new.dtypes:
            raise ValueError("Baseline and candidate must share four-band RGBA dtype")
        if ids.count != 1 or not np.issubdtype(np.dtype(ids.dtypes[0]), np.integer):
            raise ValueError("Source-index must contain one integer band")
        if report.get("grid") != grid(new):
            raise ValueError("Run report grid differs from the actual candidate")

        totals = Counter()
        source_counts = Counter()
        for _, window in new.block_windows(1):
            original = old.read(window=window)
            actual = new.read(window=window)
            selected = ids.read(1, window=window)
            different = np.any(actual != original, axis=0)
            rgb_different = np.any(actual[:3] != original[:3], axis=0)
            covered = original[3] > 0
            fallback = selected == 0
            values, counts = np.unique(selected, return_counts=True)
            source_counts.update({int(value): int(count) for value, count in zip(values, counts)})
            totals.update(
                blocks=1,
                pixels=selected.size,
                validPixels=int(covered.sum()),
                changedPixels=int(different.sum()),
                changedValidRGBPixels=int((rgb_different & covered).sum()),
                alphaDifferentPixels=int((actual[3] != original[3]).sum()),
                sourceZeroDifferentPixels=int((different & fallback).sum()),
                validSourceZeroPixels=int((covered & fallback).sum()),
                validPhotoSourcePixels=int((covered & ~fallback).sum()),
                nonzeroSourceOutsideBaselineCoveragePixels=int((~covered & ~fallback).sum()),
            )

        illegal = {
            str(identifier): count
            for identifier, count in sorted(source_counts.items())
            if identifier != 0 and identifier not in mapping
        }
        failed = {
            key: totals[key]
            for key in (
                "alphaDifferentPixels",
                "sourceZeroDifferentPixels",
                "nonzeroSourceOutsideBaselineCoveragePixels",
            )
            if totals[key] != 0
        }
        if illegal:
            failed["unknownSourceIdPixels"] = sum(illegal.values())
        result = {
            "grid": grid(new),
            "counts": dict(totals),
            "sourcePixelCounts": {str(key): value for key, value in sorted(source_counts.items())},
            "unknownSourceIds": illegal,
            "failedPixelChecks": failed,
            "checksPassed": not failed,
        }

    after = {str(path): signature(path) for path in paths}
    if before != after:
        raise ValueError("An input changed while the read-only check was running")
    result.update(
        reportByteSha256=hashlib.sha256(report_bytes).hexdigest(),
        reportSchema=report["schemaVersion"],
        mappedPhotoCount=len(mapping),
        inputFilesStableDuringCheck=True,
        inputFileSignatures={
            path: {"bytes": value[0], "mtimeNs": value[1]} for path, value in before.items()
        },
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "candidate", "source-index", "report", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    baseline, candidate, source_index, report_path, output = [
        getattr(args, name).resolve()
        for name in ("baseline", "candidate", "source_index", "report", "output")
    ]
    if output in (baseline, candidate, source_index, report_path):
        parser.error("The new JSON record may not overwrite any input")
    if output.exists():
        parser.error("Use a fresh JSON output path; prior check records are preserved")
    temporary = output.with_name(output.name + ".partial")
    if temporary.exists():
        parser.error("A prior partial JSON record exists; choose a fresh output path")
    started = time.monotonic()
    result = {
        "schemaVersion": "ogs2-pixel-integrity-check/v1",
        "checkedAtUtc": datetime.now(UTC).isoformat(),
        "baseline": str(baseline),
        "candidate": str(candidate),
        "sourceIndex": str(source_index),
        "runReport": str(report_path),
        "scope": "Full supplied rasters, blockwise, with exact grid and RGBA comparison",
        "limitations": [
            "Pixel integrity does not prove vehicle identity, duplicate removal or complete shadows.",
            "Source IDs identify reported photo choices; original-photo pixel provenance is not rechecked.",
            "No geometric accuracy, segmentation quality or visual improvement claim is made.",
            "File size and mtime stability are checked; the full input file hashes are not recomputed.",
        ],
    }
    try:
        result.update(check(baseline, candidate, source_index, report_path))
    except Exception as error:
        result.update(checksPassed=False, error=f"{type(error).__name__}: {error}")
    result["elapsedSeconds"] = time.monotonic() - started
    output.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    if output.exists():
        temporary.unlink()
        raise FileExistsError("A check record appeared at the output path; it was preserved")
    temporary.replace(output)
    print(json.dumps({"output": str(output), "checksPassed": result["checksPassed"]}, ensure_ascii=False))
    return 0 if result["checksPassed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
