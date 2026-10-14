"""Audit EXIF / DJI XMP without treating camera metadata as calibrated poses.

Run this on the lab CPU before reconstruction. Only Pillow is required.
EXIF wall times stay timezone-unspecified: they are not UTC timestamps.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

from PIL import Image


def scalar(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace").rstrip("\x00")
    if isinstance(value, (str, int, float)) or value is None:
        return value
    try:
        return float(value)
    except (ValueError, TypeError):
        return str(value)


def decimal_coordinate(values, reference):
    if values is None or reference not in ("N", "S", "E", "W"):
        return None
    value = float(values[0]) + float(values[1]) / 60 + float(values[2]) / 3600
    return value * (-1 if reference in ("S", "W") else 1)


def xmp_dji(path):
    # Metadata is in the JPEG header. Do not decode full-resolution image pixels.
    with path.open("rb") as stream:
        header = stream.read(1024 * 1024)
    text = header.decode("utf-8", "ignore")
    pairs = re.findall(r'drone-dji:([A-Za-z0-9_]+)\s*=\s*[\"\']([^\"\']+)[\"\']', text)
    return dict(pairs)


def read_image(path, root, include_hash=False):
    with Image.open(path) as image:
        exif = image.getexif()
        gps = exif.get_ifd(34853)
        details = exif.get_ifd(34665)
        gps_height = float(gps[6]) if 6 in gps else None
        if gps_height is not None and gps.get(5) in (1, b"\x01"):
            gps_height = -gps_height
        record = {
            "file": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "width": image.width,
            "height": image.height,
            "orientation": exif.get(274, 1),
            "make": scalar(exif.get(271)),
            "model": scalar(exif.get(272)),
            "exifBodySerial": scalar(details.get(42033)),
            "focalLengthMm": scalar(details.get(37386)),
            "focalLength35mm": scalar(details.get(41989)),
            "dateTimeOriginal": scalar(details.get(36867, exif.get(306))),
            "offsetTimeOriginal": scalar(details.get(36881)),
            "gpsLatitude": decimal_coordinate(gps.get(2), gps.get(1)),
            "gpsLongitude": decimal_coordinate(gps.get(4), gps.get(3)),
            "gpsAltitudeM": gps_height,
            "djiXmp": xmp_dji(path),
        }
    if include_hash:
        with path.open("rb") as stream:
            record["sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    return record


def summarize(records):
    geographic = [r for r in records if r["gpsLatitude"] is not None and r["gpsLongitude"] is not None]
    times = [r["dateTimeOriginal"] for r in records if r["dateTimeOriginal"]]
    zones = sorted({max(1, min(60, math.floor((r["gpsLongitude"] + 180) / 6) + 1)) for r in geographic})
    crs = None
    extent = None
    if geographic:
        longitude = statistics.median(r["gpsLongitude"] for r in geographic)
        latitude = statistics.median(r["gpsLatitude"] for r in geographic)
        zone = max(1, min(60, math.floor((longitude + 180) / 6) + 1))
        crs = f"EPSG:{(32600 if latitude >= 0 else 32700) + zone}"
        extent = {
            "latitude": [min(r["gpsLatitude"] for r in geographic), max(r["gpsLatitude"] for r in geographic)],
            "longitude": [min(r["gpsLongitude"] for r in geographic), max(r["gpsLongitude"] for r in geographic)],
        }
    warnings = []
    if len(zones) > 1:
        warnings.append("The footprint crosses UTM zone boundaries. Use one coherent projected CRS, then verify ODX output CRS; never assign a different CRS per photograph.")
    if len(geographic) != len(records):
        warnings.append("Some photographs lack EXIF GPS. Check DJI XMP and SfM registration before export.")
    if any(not r["offsetTimeOriginal"] for r in records):
        warnings.append("EXIF wall times have unspecified timezones. Use within-flight elapsed times; do not equate them with MRK GPS seconds or UTC.")
    if any(r["orientation"] != 1 for r in records):
        warnings.append("Non-default EXIF orientation exists; check pixel orientation before applying intrinsics.")
    warnings.append("GPS altitude, DJI AbsoluteAltitude and reconstructed local Z may use different vertical references. Use ODX georeferencing offset consistently; do not replace calibrated SfM poses with EXIF GPS.")
    if any(r["djiXmp"].get("DewarpFlag") == "0" for r in records):
        warnings.append("DJI DewarpFlag=0 exists. Check ODX camera metadata support; do not assume raw photographs are already undistorted or reuse calibration from an earlier camera.")
    if any(r["djiXmp"].get("UTCAtExposure") for r in records):
        warnings.append("DJI UTCAtExposure is retained verbatim. Validate its clock against MRK GPS times before interpreting it as civil UTC; do not mix it with EXIF wall time for vehicle association.")
    return {
        "photoCount": len(records),
        "gpsPhotoCount": len(geographic),
        "pixelDimensions": dict(Counter(f'{r["width"]}x{r["height"]}' for r in records)),
        "cameraModels": dict(Counter(f'{r["make"]} {r["model"]}' for r in records)),
        "focalLengthsMm": dict(Counter(str(r["focalLengthMm"]) for r in records)),
        "djiDewarpFlags": dict(Counter(str(r["djiXmp"].get("DewarpFlag")) for r in records)),
        "djiCalibratedFocalLengthsPixels": dict(Counter(str(r["djiXmp"].get("CalibratedFocalLength")) for r in records)),
        "djiDewarpData": sorted({r["djiXmp"]["DewarpData"] for r in records if r["djiXmp"].get("DewarpData")}),
        "captureWallTimeRange": [min(times), max(times)] if times else None,
        "timeConvention": "EXIF wall clock; timezone retained only if explicitly supplied by EXIF",
        "gpsExtentWgs84": extent,
        "observedUtmZones": zones,
        "suggestedProjectedCrsFromMedianGps": crs,
        "suggestionIsNotVerifiedOutputCrs": True,
        "warnings": warnings,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sha256", action="store_true", help="Read each image fully to record its content digest")
    args = parser.parse_args()
    root = args.images.resolve()
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".tif", ".tiff"})
    records, errors = [], []
    for path in files:
        try:
            records.append(read_image(path, root, args.sha256))
        except Exception as error:
            errors.append({"file": str(path.relative_to(root)), "error": f"{type(error).__name__}: {error}"})
    result = {
        "schemaVersion": "aerial-metadata-audit/v1",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "sourceDirectory": str(root),
        "summary": summarize(records),
        "sidecars": [{"file": p.relative_to(root).as_posix(), "bytes": p.stat().st_size} for p in sorted(root.rglob("*")) if p.is_file() and p.suffix.lower() in {".mrk", ".nav", ".obs", ".rtk", ".pbk"}],
        "errors": errors,
        "photos": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError("Refuse to overwrite an existing metadata audit")
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    if errors or not records:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
