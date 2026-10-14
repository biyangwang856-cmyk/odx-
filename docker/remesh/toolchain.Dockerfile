FROM scratch
COPY --chmod=0755 basisu /usr/local/bin/basisu
COPY --chmod=0644 basisu-LICENSE /usr/share/licenses/basisu/LICENSE
COPY --chmod=0644 basisu-provenance.json /usr/share/basisu/provenance.json
COPY gltf-transform/ /opt/gltf-transform/
COPY --chmod=0644 gltf-transform-provenance.json /opt/gltf-transform/provenance.json
