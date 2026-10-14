"""Validated browser diagnostics accepted by the Viewer API."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ApiModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class CameraDiagnostic(ApiModel):
    position: tuple[float, float, float]
    fov: float | None
    viewport_height: int = Field(alias="viewportHeight")


class VisibleTileDiagnostic(ApiModel):
    content_url: str | None = Field(alias="contentUrl")
    triangles: int
    geometric_error: float | None = Field(alias="geometricError")
    depth: int | None
    center: tuple[float, float, float] | None
    radius: float | None


class RuntimeDiagnostic(ApiModel):
    version: Literal[56]
    metric_dom_writes: int = Field(alias="metricDomWrites", ge=0)
    resource_batches: int = Field(alias="resourceBatches", ge=0)
    full_resource_scans: int = Field(alias="fullResourceScans", ge=0)
    hidden_suspensions: int = Field(alias="hiddenSuspensions", ge=0)
    resource_mode: Literal["observer", "buffer-fallback"] = Field(alias="resourceMode")
    cumulative_resource_entries: int = Field(alias="cumulativeResourceEntries", ge=0)
    ignored_resource_entries: int = Field(alias="ignoredResourceEntries", ge=0)
    cadence_ms: int = Field(alias="cadenceMs", ge=1)
    idle: bool
    hidden: bool
    resource_counters_retain_timing_entries: Literal[False] = Field(
        alias="resourceCountersRetainTimingEntries"
    )
    limitations: list[str]


class CacheSettingsDiagnostic(ApiModel):
    model_key: Literal["a1"] = Field(alias="modelKey")
    resource_prefix: Literal["/a1/"] = Field(alias="resourcePrefix")
    retain_ms: int = Field(alias="retainMs", ge=0)
    soft_bytes: int = Field(alias="softBytes", ge=0)
    hard_bytes: int = Field(alias="hardBytes", ge=0)


class ResourceTotalsDiagnostic(ApiModel):
    requests: int = Field(ge=0)
    encoded_body_bytes: int = Field(alias="encodedBodyBytes", ge=0)
    decoded_body_bytes: int = Field(alias="decodedBodyBytes", ge=0)
    transfer_bytes: int = Field(alias="transferBytes", ge=0)


class CacheDiagnostic(ApiModel):
    policy: Literal["baseline", "convergent-demand-v55"]
    loads: int = Field(ge=0)
    unloads: int = Field(ge=0)
    reloads: int = Field(ge=0)
    trims: int = Field(ge=0)
    trim_decision: str = Field(alias="trimDecision")
    epoch: int = Field(ge=0)
    convergence_passes: int = Field(alias="convergencePasses", ge=0)
    convergence_stop_reason: str | None = Field(alias="convergenceStopReason")
    settings: CacheSettingsDiagnostic
    enabled: bool
    pending: int = Field(ge=0)
    processing: int = Field(ge=0)
    failures: int = Field(ge=0)
    visible_tiles: int = Field(alias="visibleTiles", ge=0)
    loaded_tiles: int = Field(alias="loadedTiles", ge=0)
    non_visible_cached_tiles: int = Field(alias="nonVisibleCachedTiles", ge=0)
    memory_bytes: int = Field(alias="memoryBytes", ge=0)
    resource: ResourceTotalsDiagnostic
    active_view: Literal["overview", "building", "edge", "texture", "top"] = Field(alias="activeView")
    mode: Literal["split", "terra", "a1", "flicker"]
    captured_at: datetime = Field(alias="capturedAt")


class DiagnosticsPayload(ApiModel):
    active_view: Literal["overview", "building", "edge", "texture", "top"] = Field(alias="activeView")
    mode: Literal["split", "terra", "a1", "flicker"]
    pending: int
    processing: int
    camera: CameraDiagnostic
    visible: list[VisibleTileDiagnostic]
    runtime: RuntimeDiagnostic
    cache: CacheDiagnostic
    captured_at: datetime = Field(alias="capturedAt")
