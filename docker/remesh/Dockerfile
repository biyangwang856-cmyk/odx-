ARG PIXI_IMAGE
ARG REMESH_TOOLCHAIN_IMAGE

FROM ${REMESH_TOOLCHAIN_IMAGE} AS remesh-toolchain

FROM ${PIXI_IMAGE} AS production-environment
WORKDIR /environment
COPY environment/pixi.toml environment/pixi.lock ./
COPY environment/pixi-config.toml /etc/pixi/config.toml
RUN --mount=type=cache,id=mesh-pixi-rattler,target=/root/.cache/rattler,sharing=locked \
    PIXI_CACHE_DIR=/root/.cache/rattler timeout 900s sh -c \
    'pixi install --locked --environment production \
    && pixi install --locked --environment runtime-abi'

FROM ${PIXI_IMAGE}
COPY --from=production-environment /environment/.pixi/envs/production /opt/mesh-production
COPY --from=production-environment /environment/.pixi/envs/runtime-abi /opt/runtime-abi
COPY --from=remesh-toolchain /opt/gltf-transform /opt/gltf-transform
COPY --from=remesh-toolchain /usr/local/bin/basisu /opt/basisu/bin/basisu
COPY --from=remesh-toolchain /usr/share/licenses/basisu/LICENSE /opt/basisu/LICENSE
COPY --from=remesh-toolchain /usr/share/basisu/provenance.json /opt/basisu/provenance.json
COPY toolchain/basisu.json /tmp/expected-basisu.json
COPY toolchain/gltf-transform.json /tmp/expected-gltf-transform.json

ENV LD_LIBRARY_PATH=/opt/runtime-abi/lib \
    PATH=/opt/mesh-production/bin:/opt/gltf-transform/node_modules/.bin:/opt/basisu/bin:/usr/local/bin:/usr/bin:/bin \
    BASISU=/opt/basisu/bin/basisu \
    BASISU_PROVENANCE=/opt/basisu/provenance.json \
    SHARP_MODULE=/opt/gltf-transform/node_modules/sharp \
    PYTHONPATH=/app/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN /opt/mesh-production/bin/python - <<'PY'
import hashlib
import json
from pathlib import Path

checks = (
    (Path("/tmp/expected-basisu.json"), Path("/opt/basisu/provenance.json")),
    (Path("/tmp/expected-gltf-transform.json"), Path("/opt/gltf-transform/provenance.json")),
)
for expected_path, provenance_path in checks:
    if json.loads(provenance_path.read_text()) != json.loads(expected_path.read_text()):
        raise SystemExit(f"toolchain provenance mismatch: {provenance_path}")
    expected_path.unlink()
basisu = Path("/opt/basisu/bin/basisu")
provenance = json.loads(Path("/opt/basisu/provenance.json").read_text())
actual = hashlib.sha256(basisu.read_bytes()).hexdigest()
if actual != provenance["binarySha256"]:
    raise SystemExit(f"BasisU binary checksum mismatch: {actual}")
PY
RUN basisu -version 2>&1 | grep -F 'v1.16.3' \
    && gltf-transform --version | grep -Fx '4.5.0' \
    && node -e 'const sharp = require(process.env.SHARP_MODULE); if (sharp.versions.sharp !== "0.35.4") process.exit(1)'

WORKDIR /app
COPY --chmod=0755 worker-runtime/ai-worker-runtime /usr/local/bin/ai-worker-runtime
COPY runtime-root/ ./

ENTRYPOINT ["/usr/local/bin/ai-worker-runtime"]
