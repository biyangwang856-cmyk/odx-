"""Derive lightweight preview tilesets from an arbitrary publication tree."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

JsonObject = dict[str, Any]


def read_tileset(root: Path) -> JsonObject:
    return json.loads((root / "tileset.json").read_text(encoding="utf-8"))


def _rewrite_content(content: JsonObject) -> JsonObject:
    result = deepcopy(content)
    for key in ("uri", "url"):
        value = result.get(key)
        if isinstance(value, str) and not value.startswith("/") and "://" not in value:
            result[key] = f"/a1/{value}"
    return result


def create_preview_tileset(source: JsonObject) -> JsonObject:
    pending = [source["root"]]
    selected: JsonObject | None = None
    while pending and selected is None:
        tile = pending.pop(0)
        if tile.get("content"):
            selected = tile
        else:
            pending[0:0] = tile.get("children", [])
    if selected is None:
        raise ValueError("right tileset has no content tiles")

    root = deepcopy(selected)
    root["content"] = _rewrite_content(selected["content"])
    root["geometricError"] = source["geometricError"]
    root["refine"] = "REPLACE"
    root.pop("children", None)
    return {**deepcopy(source), "root": root}


def create_medium_tileset(source: JsonObject, *, content_limit: int = 17) -> JsonObject:
    remaining = content_limit

    def select_tile(tile: JsonObject) -> JsonObject | None:
        nonlocal remaining
        result = {key: deepcopy(value) for key, value in tile.items() if key not in {"children", "content"}}
        selected_content = False
        if tile.get("content") and remaining > 0:
            result["content"] = _rewrite_content(tile["content"])
            remaining -= 1
            selected_content = True

        children = []
        for child in tile.get("children", []):
            if remaining <= 0:
                break
            selected_child = select_tile(child)
            if selected_child is not None:
                children.append(selected_child)
        if children:
            result["children"] = children
        return result if selected_content or children else None

    root = select_tile(source["root"])
    if root is None:
        raise ValueError("right tileset has no content tiles")
    return {**deepcopy(source), "root": root}
