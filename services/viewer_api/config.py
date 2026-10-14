"""Runtime configuration for the result publication service."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DATA_ROOT = Path("/mnt/juicefs/shared/mesh-research-runs/20260814-lglglg-adaptive-hlod-v1")
DEFAULT_TERRA_ROOT = Path("/mnt/juicefs/surveying/staging/scene/LGLGLG/models/pc/0/terra_b3dms")

TARGETS = {
    "v55": (
        "REMESH_v55_runtime",
        "results/v55-optimized-runtime-local",
        8155,
    ),
}


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ViewerSettings:
    """Resolved paths and labels needed by one Viewer API process."""

    viewer_dist: Path
    left_root: Path
    right_root: Path
    left_label: str
    right_label: str
    port: int
    dataset: str = "LGLGLG"
    maximum_screen_space_error: int = 8
    publications_root: Path | None = None
    publication_registry: Path | None = None
    publication_bearer_token: str | None = None
    publication_cors_origins: tuple[str, ...] = ("*",)

    @classmethod
    def for_target(cls, target: str = "v55") -> ViewerSettings:
        try:
            default_label, relative_right_root, default_port = TARGETS[target]
        except KeyError as error:
            choices = ", ".join(sorted(TARGETS))
            raise ValueError(f"unknown viewer target {target!r}; choose one of: {choices}") from error

        data_root = Path(os.environ.get("MESH_RESEARCH_DATA_ROOT", DEFAULT_DATA_ROOT)).expanduser()
        terra_root = Path(os.environ.get("MESH_RESEARCH_TERRA_ROOT", DEFAULT_TERRA_ROOT)).expanduser()
        publications_root = os.environ.get("MESH_PUBLICATIONS_ROOT")
        publication_registry = os.environ.get("MESH_PUBLICATION_REGISTRY")
        cors_origins = tuple(
            origin.strip()
            for origin in os.environ.get("MESH_PUBLICATION_CORS_ORIGINS", "*").split(",")
            if origin.strip()
        )
        return cls(
            viewer_dist=repository_root() / "services/viewer/dist",
            left_root=Path(os.environ.get("LEFT_TILES_ROOT", terra_root)).expanduser().resolve(),
            right_root=Path(os.environ.get("RIGHT_TILES_ROOT", data_root / relative_right_root))
            .expanduser()
            .resolve(),
            left_label=os.environ.get("LEFT_LABEL", "DJI_Terra"),
            right_label=os.environ.get("RIGHT_LABEL", default_label),
            port=int(os.environ.get("PORT", default_port)),
            publications_root=Path(publications_root).expanduser().resolve() if publications_root else None,
            publication_registry=Path(publication_registry).expanduser().resolve()
            if publication_registry
            else None,
            publication_bearer_token=os.environ.get("MESH_PUBLICATION_BEARER_TOKEN"),
            publication_cors_origins=cors_origins,
        )

    def validate(self) -> None:
        """Fail before binding a port when a publication or frontend build is missing."""
        for label, root in (("left", self.left_root), ("right", self.right_root)):
            tileset = root / "tileset.json"
            if not tileset.is_file():
                raise FileNotFoundError(f"{label} tileset not found: {tileset}")
        index = self.viewer_dist / "index.html"
        if not index.is_file():
            raise FileNotFoundError(f"viewer build not found: {index}; run viewer-build first")
        publication_settings = (
            self.publications_root,
            self.publication_registry,
            self.publication_bearer_token,
        )
        if any(value is not None for value in publication_settings) and not all(publication_settings):
            raise ValueError("publication root, registry, and bearer token must be configured together")
        if self.publications_root is not None and not self.publications_root.is_dir():
            raise FileNotFoundError("publication root not found")
        if self.publication_registry is not None and not self.publication_registry.is_file():
            raise FileNotFoundError("publication registry not found")
        if not self.publication_cors_origins:
            raise ValueError("at least one publication CORS origin is required")
