# Mesh Research Viewer

TypeScript/Cesium frontend for comparing the reference model with the clean-break v55 production publication. The v56 runtime layer adds bounded cache policy, request/lifecycle monitoring, and diagnostics without changing the v55 model identity.

## Commands

```bash
just viewer-build
just viewer-check
just viewer v55
```

`just viewer` builds `dist/` before starting FastAPI. The default right-side publication is `results/v55-optimized-runtime-local`; override it with `RIGHT_TILES_ROOT` when testing a Worker-generated v55 publication.
