ARG PIXI_IMAGE
ARG ODX_BASE_IMAGE

FROM ${PIXI_IMAGE} AS runtime-abi
WORKDIR /environment
COPY environment/pixi.toml environment/pixi.lock ./
COPY environment/pixi-config.toml /etc/pixi/config.toml
RUN --mount=type=cache,id=mesh-pixi-rattler,target=/root/.cache/rattler,sharing=locked \
    PIXI_CACHE_DIR=/root/.cache/rattler timeout 900s sh -c \
    'pixi install --locked --environment runtime-abi'

FROM ${ODX_BASE_IMAGE}
COPY --from=runtime-abi /environment/.pixi/envs/runtime-abi /opt/runtime-abi

ENV LD_LIBRARY_PATH=/opt/runtime-abi/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/code/SuperBuild/install/lib \
    PATH=/opt/runtime-abi/bin:/usr/local/bin:/usr/bin:/bin \
    PYTHONPATH=/app/app:/code/SuperBuild/install/lib/python3.12:/code/SuperBuild/install/bin/opensfm \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY --chmod=0755 worker-runtime/ai-worker-runtime /usr/local/bin/ai-worker-runtime
COPY runtime-root/ ./

ENTRYPOINT ["/usr/local/bin/ai-worker-runtime"]
