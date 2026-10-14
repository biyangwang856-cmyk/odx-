"""Small fail-closed primitives for versioned JSON manifests."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any

IDENTITY_RE = re.compile(r"^[a-z0-9][a-z0-9._:-]{2,127}$")
SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
MEDIA_TYPE_RE = re.compile(r"^[^\s/]+/[^\s/]+$")


def canonical_json_digest(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def require_object(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    return value


def require_keys(
    value: Mapping[str, Any], path: str, *, required: set[str], optional: set[str] | None = None
) -> None:
    optional = optional or set()
    missing = sorted(required - value.keys())
    unknown = sorted(value.keys() - required - optional)
    if missing:
        raise ValueError(f"{path} missing fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{path} unknown fields: {', '.join(unknown)}")


def require_string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path} must be a non-empty string")
    return value


def require_identity(value: object, path: str) -> str:
    identity = require_string(value, path)
    if IDENTITY_RE.fullmatch(identity) is None:
        raise ValueError(f"{path} is not a stable identity")
    return identity


def require_sha256(value: object, path: str) -> str:
    digest = require_string(value, path)
    if SHA256_RE.fullmatch(digest) is None:
        raise ValueError(f"{path} must be sha256:<64 lowercase hex characters>")
    return digest


def require_positive_int(value: object, path: str, *, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{path} must be a positive integer")
    if maximum is not None and value > maximum:
        raise ValueError(f"{path} must not exceed {maximum}")
    return value


def require_enum(value: object, path: str, allowed: set[str]) -> str:
    text = require_string(value, path)
    if text not in allowed:
        raise ValueError(f"{path} must be one of {', '.join(sorted(allowed))}")
    return text


def require_media_type(value: object, path: str) -> str:
    media_type = require_string(value, path)
    if not 3 <= len(media_type) <= 127 or MEDIA_TYPE_RE.fullmatch(media_type) is None:
        raise ValueError(f"{path} must be a media type")
    return media_type


def require_relative_path(value: object, path: str) -> str:
    text = require_string(value, path)
    pure = PurePosixPath(text)
    if (
        len(text) > 1024
        or "\\" in text
        or "\x00" in text
        or pure.is_absolute()
        or any(component in {"", ".", ".."} for component in text.split("/"))
    ):
        raise ValueError(f"{path} must be a normalized relative POSIX path")
    return text


def require_finite_number(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{path} must be a finite number")
    try:
        converted = float(value)
    except OverflowError as error:
        raise ValueError(f"{path} must be a finite number") from error
    if not math.isfinite(converted):
        raise ValueError(f"{path} must be a finite number")
    return converted
