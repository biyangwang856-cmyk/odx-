# Viewer API

FastAPI serves the built Viewer, the comparison tileset, and the clean-break v55 production publication with CORS and byte-range support.

```bash
just viewer v55
```

`MESH_RESEARCH_DATA_ROOT`, `MESH_RESEARCH_TERRA_ROOT`, `LEFT_TILES_ROOT`, and `RIGHT_TILES_ROOT` can override the default roots. Published immutable artifacts may also be exposed through `MESH_PUBLICATIONS_ROOT`, `MESH_PUBLICATION_REGISTRY`, and `MESH_PUBLICATION_BEARER_TOKEN`.
