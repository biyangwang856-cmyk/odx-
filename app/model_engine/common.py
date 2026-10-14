"""Shared adapter framing and job-root utilities."""

from __future__ import annotations

import json
import os
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from core.artifacts import artifact_ref


def event_artifact_ref(path: Path, root: Path, media_type: str, role: str) -> dict[str, object]:
    reference = artifact_ref(path, root, media_type, role)
    return {
        "artifact_id": reference["artifactId"],
        "relative_path": reference["relativePath"],
        "content_digest": reference["contentDigest"],
        "bytes": reference["bytes"],
        "media_type": reference["mediaType"],
    }


@dataclass(frozen=True)
class AdapterEnvironment:
    attempt_id: str
    input_path: Path
    job_root: Path
    task_name: str
    task_revision: str
    task_config: dict[str, object]

    @classmethod
    def load(cls) -> AdapterEnvironment:
        protocol = os.environ.get("WORKER_RUNTIME_PROTOCOL")
        if protocol != "oneshot-events-v1":
            raise ValueError("WORKER_RUNTIME_PROTOCOL must be oneshot-events-v1")
        return cls(
            attempt_id=os.environ["WORKER_ATTEMPT_ID"],
            input_path=Path(os.environ["WORKER_INPUT_PATH"]).resolve(),
            job_root=Path(os.environ["WORKER_JOB_ROOT"]).resolve(),
            task_name=os.environ["WORKER_TASK_NAME"],
            task_revision=os.environ["WORKER_TASK_REVISION"],
            task_config=json.loads(os.environ.get("WORKER_TASK_CONFIG_JSON", "{}")),
        )


class EventEmitter:
    def __init__(self, environment: AdapterEnvironment, adapter_id: str, stream: BinaryIO | None = None):
        self.environment = environment
        self.adapter_id = adapter_id
        self.stream = stream or sys.stdout.buffer
        self.sequence = 0

    def emit(self, event: str, **fields: object) -> None:
        header = {
            "protocol_version": "oneshot-events-v1",
            "event": event,
            "adapter_id": self.adapter_id,
            "attempt_id": self.environment.attempt_id,
            "sequence": self.sequence,
            **fields,
        }
        encoded = json.dumps(header, separators=(",", ":")).encode()
        self.stream.write(struct.pack(">I", len(encoded)))
        self.stream.write(encoded)
        self.stream.write(struct.pack(">Q", 0))
        self.stream.flush()
        self.sequence += 1

    def ready(self) -> None:
        self.emit(
            "ready",
            task_name=self.environment.task_name,
            task_revision=self.environment.task_revision,
        )

    def result_error(self, code: str, message: str, stage_id: str | None = None) -> None:
        self.emit(
            "result",
            status="error",
            manifest=None,
            error={"code": code, "message": message[:4096], "stage_id": stage_id},
        )
