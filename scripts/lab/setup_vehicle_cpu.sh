set -eu
root=/volume/wzhang/mesh-research-workspace/runtime
lab_script_dir=$(cd -- "$(dirname -- "$0")" && pwd)
python="$root/vehicle-env/bin/python"
if [ ! -x "$python" ]; then
    python3 -m venv "$root/vehicle-env"
fi
mkdir -p "$root/vehicle-system-libs/packages" "$root/vehicle-system-libs/rootfs"
(
    cd "$root/vehicle-system-libs/packages"
    apt-get download libgl1 libglx0 libglvnd0
    for pkg in ./*.deb; do dpkg-deb -x "$pkg" "$root/vehicle-system-libs/rootfs"; done
    sha256sum ./*.deb > "$root/vehicle-system-libs/package-sha256.txt"
)
export LD_LIBRARY_PATH="$root/vehicle-system-libs/rootfs/usr/lib/x86_64-linux-gnu"
export PIP_CONFIG_FILE=/dev/null
export PIP_CACHE_DIR="$root/pip-cache"
export HF_HOME="$root/hf-cache"
export SAM2_BUILD_CUDA=0
"$python" -m pip install --index-url https://pypi.org/simple numpy==1.26.4 scipy==1.14.1 opencv-python==4.10.0.84 rasterio==1.4.4 orthority==0.7.0 numba==0.61.2 plyfile==1.1.2 pillow pytest jsonschema hydra-core==1.3.2 iopath==0.1.10 transformers==4.49.0 huggingface-hub==0.28.1 setuptools wheel
"$python" -m pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 torchvision==0.21.0
if [ ! -e "$root/sam2-source/.git" ]; then
    git clone --filter=blob:none --no-checkout https://github.com/facebookresearch/sam2.git "$root/sam2-source"
fi
git -C "$root/sam2-source" checkout --detach 2b90b9f5ceec907a1c18123530e92e794ad901a4
"$python" -m pip install --no-deps --no-build-isolation -e "$root/sam2-source"
"$python" "$lab_script_dir/download_vehicle_models.py"
"$python" -m pip check
"$python" -m pip freeze > "$root/vehicle-env.freeze.txt"
date -u +%FT%TZ > "$root/vehicle-env.ready"
