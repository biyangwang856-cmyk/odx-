# LGLGLG REMESH v38 quality candidate

Date: 2026-08-30

## Status

`REMESH_v38_quality` is a visual test candidate. It is not registered as a
backend Worker artifact and must not replace the current integration target.

The existing `REMESH_v37_fast` entry remains on remote port 8147 / local port
9147. The quality candidate has an isolated entry on remote port 8148 / local
port 9148.

## Motivation

The v37 fast candidate reduced geometry to about 3.91 million triangles. This
was useful for browser loading, but very close camera positions exposed large
facets. The old texture-detail preset also placed the shared camera inside
foreground vegetation and too close to a building surface, so it was not a
valid comparison view.

## Candidate construction

Source:

`results/lglglg-remesh-viewer-v1/tiles-v34-browser-ratio30`

Output:

`results/lglglg-remesh-viewer-v1/tiles-v38-quality-ratio30-webp82-meshopt`

Current registered build entry:

`just remesh-quality`

The original 2026-08-30 run used `code/build_remesh_quality_webp.sh`; that legacy wrapper was removed
when the build was migrated into the task registry.

The candidate keeps all 6,211,035 triangles from the ratio-30 source, converts
633 embedded PNG textures to WebP quality 82, and applies Meshopt compression
to all 205 GLB tiles. No additional geometry simplification is performed.

| Metric | Source | v38 quality |
| --- | ---: | ---: |
| GLB tiles | 205 | 205 |
| Triangles | 6,211,035 | 6,211,035 |
| Embedded images | 633 PNG | 633 WebP |
| GLB bytes | 477,317,596 | 89,024,984 |
| Meshopt tiles | 0 | 205 |

The detailed audit is stored in `quality-webp-report.json` inside the output
directory.

## Viewer changes

The texture-detail preset now reuses the validated building-side target with a
closer but safe range (`0.42 * bounding-sphere radius`) and a stable pitch. It
no longer places the camera between foreground leaves and the wall, while the
building facade occupies enough of the viewport for texture inspection.

Test startup:

```bash
just viewer v38
```

Configuration check:

```text
http://127.0.0.1:8148/api/config
```

Windows launcher:

`打开质量候选REMESH-v38对比.cmd`

## Verification

The overview, building, edge, texture-detail, and top-down presets were checked
at SSE 32. The v38 side remained visually continuous, the texture-detail view
showed the complete building facade and roof instead of clipped foliage, and no
browser console errors were recorded.

Observed cost in the full loaded comparison was approximately 732 MB of GPU
cache for the v38 side. Frame rate was lower than the fast candidate. This is
why v38 remains a quality reference rather than an integration recommendation.

The coarse appearance on the left side is the existing DJI Terra baseline and
does not indicate fragmentation of the v38 output.
