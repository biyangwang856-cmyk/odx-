"""Smoke recovered pinned ODX, without touching project code or input images."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess

RUNTIME = Path("/volume/wzhang/mesh-research-workspace/runtime")


def digest(path):
    with path.open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def patch_native_opencv(bundle, root):
    """Prefix OpenCV's build-time absolute paths only during native execution.

    The image was built at /code and /usr/local. Without this conditional
    prefix, cv2 can accidentally load the host's older binary extension.
    The environment variable remains unset inside the CPU chroot.
    """
    marker = "# mesh-research conditional native-root prefix v1"
    directory = root / "usr/local/lib/python3.12/dist-packages/cv2"
    targets = {"config.py": "BINARIES_PATHS", "config-3.12.py": "PYTHON_EXTENSIONS_PATHS"}
    provenance = bundle / "native-opencv-prefix-patch.json"
    records = json.loads(provenance.read_text()) if provenance.exists() else {"files": {}}
    for name, variable in targets.items():
        path = directory / name
        if not path.is_file():
            raise RuntimeError("OpenCV native config path unavailable: " + str(path))
        text = path.read_text()
        if marker in text:
            recorded = records["files"].get(name)
            if not recorded or digest(path) != recorded["patchedSha256"]:
                raise RuntimeError("Existing cv2 prefix patch lacks matching provenance")
            continue
        if variable not in text:
            raise RuntimeError("Unexpected OpenCV config contents: " + name)
        original = bundle / "native-opencv-originals" / name
        original.parent.mkdir(exist_ok=True)
        if original.exists():
            raise RuntimeError("cv2 original backup path already exists")
        original.write_bytes(path.read_bytes())
        suffix = (
            "\n" + marker + "\n"
            "import os as _odx_native_os\n"
            "_odx_native_root = _odx_native_os.environ.get('ODX_NATIVE_ROOT')\n"
            "if _odx_native_root:\n"
            f"    {variable} = [_odx_native_os.path.join(_odx_native_root, _odx_p.lstrip('/')) "
            "if _odx_native_os.path.isabs(_odx_p) and "
            "not _odx_p.startswith(_odx_native_root.rstrip('/') + '/') else _odx_p "
            f"for _odx_p in {variable}]\n"
        )
        path.write_text(text + suffix)
        records["files"][name] = {"path": str(path), "backup": str(original),
                                  "originalSha256": digest(original), "patchedSha256": digest(path)}
        records.update(state="conditional-prefix-applied", environmentVariable="ODX_NATIVE_ROOT",
                       root=str(root), chrootBehavior="Unchanged when ODX_NATIVE_ROOT is unset")
        provenance.write_text(json.dumps(records, indent=2) + "\n")


def record_command(command, environment, log, timeout=180):
    with log.open("wb") as stream:
        process = subprocess.run(command, env=environment, stdout=stream,
                                 stderr=subprocess.STDOUT, timeout=timeout)
    return {"command": command, "exitCode": process.returncode, "log": str(log)}


def prepare_cpu_root(root):
    for name in ["dev", "dev/shm", "proc", "sys", "tmp", "run"]:
        (root / name).mkdir(exist_ok=True)
    (root / "tmp").chmod(0o1777)
    (root / "dev/shm").chmod(0o1777)
    for name, minor in [("null", 3), ("zero", 5), ("random", 8), ("urandom", 9)]:
        target = root / "dev" / name
        if not target.exists():
            os.mknod(target, stat.S_IFCHR | 0o666, os.makedev(1, minor))
    # A mount can be forbidden inside a VM/container. These snapshots implement
    # the prior verified CPU fallback, including the GDAL RAM/cgroup files.
    for name in ["cpuinfo", "meminfo", "stat", "loadavg", "uptime", "version",
                 "sys/kernel/osrelease", "self/cgroup", "self/mountinfo", "mounts"]:
        source = Path("/proc") / name
        destination = root / "proc" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())


def native_environment(root):
    candidates = [root / "usr/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2",
                  root / "lib/x86_64-linux-gnu/ld-linux-x86-64.so.2"]
    loader = next((path for path in candidates if path.is_file()), None)
    if loader is None:
        raise RuntimeError("Pinned glibc loader not found")
    library_dirs = [root / "code/SuperBuild/install/lib", root / "usr/local/lib",
                    root / "usr/lib/x86_64-linux-gnu", root / "usr/lib", root / "lib",
                    root / "usr/local/cuda/targets/x86_64-linux/lib",
                    root / "usr/local/cuda/lib64", Path("/usr/local/nvidia/lib64"),
                    Path("/usr/local/nvidia/lib"), Path("/usr/lib/x86_64-linux-gnu")]
    libraries = ":".join(str(path) for path in library_dirs if path.is_dir())
    python_paths = [root / "code", root / "usr/local/lib/python3.12/dist-packages",
                    root / "usr/lib/python3/dist-packages",
                    root / "code/SuperBuild/install/lib/python3.12",
                    root / "code/SuperBuild/install/lib/python3.12/dist-packages",
                    root / "code/SuperBuild/install/bin/opensfm"]
    environment = {**os.environ,
        "ODX_NATIVE_ROOT": str(root),
        "PYTHONHOME": str(root / "usr"),
        "PYTHONPATH": ":".join(str(path) for path in python_paths if path.is_dir()),
        "LD_LIBRARY_PATH": libraries,
        "GDAL_DATA": str(root / "usr/share/gdal"),
        "PROJ_DATA": str(root / "usr/share/proj"),
        "PROJ_LIB": str(root / "usr/share/proj"),
        "PDAL_DRIVER_PATH": str(root / "code/SuperBuild/install/bin"),
        "PATH": ":".join(str(root / name) for name in ["code/SuperBuild/install/bin", "usr/local/bin", "usr/bin", "bin"]) + ":" + os.environ.get("PATH", ""),
        "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "4", "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1"}
    plugins = root / "usr/lib/x86_64-linux-gnu/gdalplugins"
    if plugins.is_dir():
        environment["GDAL_DRIVER_PATH"] = str(plugins)
    python = root / "usr/bin/python3.12"
    if not python.is_file():
        raise RuntimeError("Pinned Python 3.12 executable not found")
    return loader, python, libraries, environment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--native", action="store_true")
    args = parser.parse_args()
    bundle = RUNTIME / ("odx-runtime-" + args.kind)
    root = bundle / "rootfs"
    identity = json.loads((bundle / "source-overlay-validation.json").read_text())
    if identity.get("state") != "pinned":
        raise RuntimeError("Runtime has no pinned-source validation")
    mode = "native" if args.native else "chroot"
    logs = bundle / ("smoke-" + mode)
    logs.mkdir(exist_ok=True)
    checks = []
    if args.native:
        patch_native_opencv(bundle, root)
        loader, python, libraries, environment = native_environment(root)
        prefix = [str(loader), "--library-path", libraries, str(python)]
        run_path = str(root / "code/run.py")
        write = {"loader": str(loader), "python": str(python),
                 "libraryPath": libraries,
                 "environment": {key: environment[key] for key in ["ODX_NATIVE_ROOT", "PYTHONHOME", "PYTHONPATH", "LD_LIBRARY_PATH", "GDAL_DATA", "PROJ_DATA", "PROJ_LIB", "PDAL_DRIVER_PATH", "PATH", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "PYTHONUNBUFFERED", "PYTHONDONTWRITEBYTECODE"]}}
        (bundle / "native-loader-python-preflight.json").write_text(json.dumps(write, indent=2) + "\n")
    else:
        prepare_cpu_root(root)
        environment = dict(os.environ)
        environment.pop("ODX_NATIVE_ROOT", None)
        inner_env = ["PYTHONPATH=/code:/usr/local/lib/python3.12/dist-packages:/usr/lib/python3/dist-packages:/code/SuperBuild/install/lib/python3.12:/code/SuperBuild/install/lib/python3.12/dist-packages:/code/SuperBuild/install/bin/opensfm",
                     "LD_LIBRARY_PATH=/code/SuperBuild/install/lib", "PDAL_DRIVER_PATH=/code/SuperBuild/install/bin",
                     "OPENBLAS_NUM_THREADS=1", "OMP_NUM_THREADS=4", "PYTHONDONTWRITEBYTECODE=1"]
        prefix = ["chroot", str(root), "/usr/bin/env", *inner_env, "/usr/bin/python3"]
        run_path = "/code/run.py"
    checks.append(record_command([*prefix, run_path, "--help"], environment, logs / "help.log"))
    probe = "import sys,json,importlib.util; print('python',sys.executable,sys.prefix); print('sys.path',json.dumps(sys.path)); print('opencv_spec',importlib.util.find_spec('cv2')); from opensfm import io,pymap; from osgeo import gdal; import numpy,cv2; print('OpenSfM/GDAL/NumPy/OpenCV OK',gdal.VersionInfo(),numpy.__version__,cv2.__version__); print('numpyFile',numpy.__file__); print('cv2File',cv2.__file__)"
    checks.append(record_command([*prefix, "-c", probe], environment, logs / "imports.log"))
    if args.native and args.kind == "gpu":
        cuda_probe = (
            "import ctypes; c=ctypes.CDLL('libcuda.so.1'); "
            "code=c.cuInit(0); print('cuInit',code); "
            "assert code==0, 'CUDA driver initialization failed'; "
            "count=ctypes.c_int(); assert c.cuDeviceGetCount(ctypes.byref(count))==0; "
            "print('visibleCUDADevices',count.value); assert count.value>0"
        )
        checks.append(record_command([*prefix, "-c", cuda_probe], environment, logs / "cuda-driver.log"))
    result = {"state": "passed" if all(check["exitCode"] == 0 for check in checks) else "failed",
              "mode": mode, "kind": args.kind, "checks": checks}
    (logs / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    if result["state"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
