"""Invoke pinned ODX natively while bounding its GDAL memory estimates.

Launch with the loader, Python and environment from native preflight JSON.
Arguments after -- are passed unchanged to the actual ODX CLI. This wrapper
does not skip SfM/MVS or apply any old dataset-specific checkpoint assumptions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, required=True)
    options, engine_arguments = parser.parse_known_args()
    if engine_arguments[:1] == ["--"]:
        engine_arguments = engine_arguments[1:]
    code = options.code_root.resolve()
    allowed = Path("/volume/wzhang/mesh-research-workspace/runtime").resolve()
    if not code.is_relative_to(allowed) or not (code / "run.py").is_file():
        raise RuntimeError("Code must be a pinned runtime underneath the assigned persistent directory")
    if (code / "VERSION").read_text().strip() != "3.8.3":
        raise RuntimeError("Expected pinned ODX 3.8.3")
    # ODX may concatenate independent reconstructions. A single output JSON
    # does not prove visual connectivity, so audit full (unclipped) residuals
    # before spending resources on dense reconstruction of such a model.
    if "--end-with" in engine_arguments:
        end = engine_arguments[engine_arguments.index("--end-with") + 1]
        if end == "odm_orthophoto":
            datasets = Path(engine_arguments[engine_arguments.index("--project-path") + 1])
            project = datasets / engine_arguments[-1]
            job = datasets.parent
            log = job / "sfm.log"
            if (project / "opensfm/reconstruction.topocentric.json").exists():
                gate = job / "sfm-merge-quality-gate.json"
                deadline = time.monotonic() + 900
                while not gate.exists() and time.monotonic() < deadline:
                    print("Waiting for untruncated SfM merge audit", flush=True)
                    time.sleep(15)
                if not gate.exists():
                    raise RuntimeError("Independent partials require an explicit full-residual audit before dense reconstruction")
                proof = json.loads(gate.read_text())
                reconstruction = project / "opensfm/reconstruction.json"
                with reconstruction.open("rb") as stream:
                    observed = hashlib.file_digest(stream, "sha256").hexdigest()
                if proof.get("state") != "passed" or proof.get("reconstructionSha256") != observed:
                    raise RuntimeError("Merged SfM audit failed or belongs to different geometry; preserve this attempt")
                for relative, key in [("opensfm/undistorted/reconstruction.json", "undistortedReconstructionSha256"),
                                      ("opensfm/undistorted/reconstruction.nvm", "nvmSha256"),
                                      ("odm_georeferencing/coords.txt", "coordsSha256")]:
                    with (project / relative).open("rb") as stream:
                        digest = hashlib.file_digest(stream, "sha256").hexdigest()
                    if digest != proof.get(key):
                        raise RuntimeError("SfM gate no longer matches " + relative)
    budget = int(os.environ["LAB_MAX_MEMORY_BYTES"])
    if budget <= 0:
        raise RuntimeError("Positive memory budget is required")
    sys.path.insert(0, str(code))
    from opendm import concurrency
    cache = min(4 * 1024**3, budget // 8)
    physical = concurrency.virtual_memory().total
    original_mb = concurrency.get_max_memory_mb
    original_percent = concurrency.get_max_memory
    concurrency.get_max_memory_mb = lambda *args, **kwargs: min(original_mb(*args, **kwargs), cache / 1024**2)
    concurrency.get_max_memory = lambda *args, **kwargs: min(original_percent(*args, **kwargs), cache / physical * 100)
    os.environ["GDAL_CACHEMAX"] = str(max(1, cache // 1024**2))
    print("Runtime GDAL cache cap:", cache, "bytes", flush=True)
    sys.argv = [str(code / "run.py"), *engine_arguments]
    runpy.run_path(str(code / "run.py"), run_name="__main__")


if __name__ == "__main__":
    main()
