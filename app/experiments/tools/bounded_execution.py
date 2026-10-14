"""Run a local Runtime acceptance probe and retain its verified file submissions."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import NamedTemporaryFile

from core.artifacts import sha256_file

IDENTITY_FIELDS = (
    "task_run_id",
    "workflow_run_id",
    "temporal_workflow_id",
    "temporal_run_id",
    "step_run_id",
    "shard_id",
    "attempt_id",
    "job_uid",
    "pod_uid",
    "pod_execution_id",
    "fencing_epoch",
)
KEY = re.compile(r"ai/v1/sha256/([0-9a-f]{64}(?:\.tif)?)")
SHA256 = re.compile(r"[0-9a-f]{64}")
MAX_BYTES = 64 * 1024**3


class CommitReceiver:
    def __init__(self, output: Path, manifest: dict[str, object]):
        self.output = output
        self.identity = {name: manifest[name] for name in IDENTITY_FIELDS}
        self.artifacts: dict[str, dict[str, object]] = {}
        self.result: dict[str, object] | None = None

    def accept(self, handler: BaseHTTPRequestHandler) -> None:
        for name, expected in self.identity.items():
            if handler.headers.get("x-surveying-" + name.replace("_", "-")) != str(expected):
                raise ValueError(f"commit identity mismatch: {name}")
        size = int(handler.headers.get("Content-Length", "0"))
        kind = handler.headers.get("x-surveying-result-kind", "artifact")
        if size <= 0 or size > (16 * 1024**2 if kind == "manifest" else MAX_BYTES):
            raise ValueError("commit size exceeds acceptance budget")
        if kind == "manifest":
            result = json.loads(handler.rfile.read(size))
            if (
                result.get("protocol_version") != "bounded-result-v1"
                or result.get("identity") != self.identity
            ):
                raise ValueError("terminal result protocol or identity mismatch")
            declared = result.get("artifacts", [])
            if not declared or len(declared) != len(self.artifacts):
                raise ValueError("terminal result does not cover uploaded artifacts")
            for item in declared:
                received = self.artifacts[item["artifact_key"]]
                reference = item["artifact"]
                if (
                    reference["bytes"] != received["bytes"]
                    or reference["content_digest"] != received["contentDigest"]
                ):
                    raise ValueError("terminal result artifact does not match received bytes")
            self.result = result
            with (self.output / "result.json").open("x", encoding="utf-8") as target:
                json.dump(result, target, indent=2)
            return
        key = handler.headers.get("x-surveying-artifact-key", "")
        match = KEY.fullmatch(key)
        expected_sha = handler.headers.get("x-surveying-artifact-sha256", "")
        if not match or not SHA256.fullmatch(expected_sha) or key in self.artifacts:
            raise ValueError("invalid or duplicate artifact key/checksum")
        digest = hashlib.sha256()
        partial: Path | None = None
        try:
            with NamedTemporaryFile(dir=self.output, prefix=".receiving-", delete=False) as target:
                partial = Path(target.name)
                remaining = size
                while remaining:
                    chunk = handler.rfile.read(min(remaining, 1024 * 1024))
                    if not chunk:
                        raise ValueError("artifact transfer ended before declared size")
                    target.write(chunk)
                    digest.update(chunk)
                    remaining -= len(chunk)
            if digest.hexdigest() != expected_sha:
                raise ValueError("received artifact checksum mismatch")
            path = self.output / match[1]
            partial.rename(path)
            partial = None
            self.artifacts[key] = {
                "path": str(path),
                "bytes": size,
                "contentDigest": "sha256:" + expected_sha,
                "mediaType": handler.headers.get("Content-Type"),
            }
        finally:
            if partial is not None:
                partial.unlink(missing_ok=True)


def run(runtime: Path, config: Path, manifest_path: Path, output: Path) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=False)
    receiver = CommitReceiver(output, manifest)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            try:
                if self.path != "/commit":
                    raise ValueError("unknown commit path")
                receiver.accept(self)
            except (KeyError, OSError, TypeError, ValueError) as error:
                self.send_error(400, str(error))
            else:
                self.send_response(201)
                self.end_headers()

        def log_message(self, *_args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    command = [
        str(runtime),
        "--config",
        str(config),
        "--batch-manifest",
        str(manifest_path),
        "--batch-commit-url",
        f"http://127.0.0.1:{server.server_port}/commit",
    ]
    try:
        with (output / "runtime.log").open("wb") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    if completed.returncode != 0:
        raise RuntimeError(f"Runtime exited with {completed.returncode}; inspect runtime.log")
    if "tiles" in manifest:
        if len(receiver.artifacts) != len(manifest["tiles"]):
            raise ValueError("classification did not submit every tile")
    elif receiver.result is None:
        raise ValueError("oneshot did not submit a terminal result")
    report = {
        "runtimeSha256": sha256_file(runtime),
        "configSha256": sha256_file(config),
        "manifestSha256": sha256_file(manifest_path),
        "artifacts": receiver.artifacts,
        "runtimeExitCode": completed.returncode,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime", "config", "manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    report = run(
        args.runtime.resolve(), args.config.resolve(), args.manifest.resolve(), args.output.resolve()
    )
    print(json.dumps({"verified": True, "artifacts": len(report["artifacts"])}))


if __name__ == "__main__":
    main()
