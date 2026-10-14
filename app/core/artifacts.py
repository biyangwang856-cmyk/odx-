"""Shared file identity and safe artifact materialization."""

from __future__ import annotations

import hashlib
import shutil
import stat
import tarfile
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def tree_identity(root: Path) -> dict[str, object]:
    root_stat = root.stat(follow_symlinks=False)
    if stat.S_ISLNK(root_stat.st_mode):
        raise ValueError(f"artifact tree root is a symbolic link: {root}")
    if not stat.S_ISDIR(root_stat.st_mode):
        raise ValueError(f"artifact tree root is not a directory: {root}")
    paths = list(root.rglob("*"))
    if symlink := next((path for path in paths if path.is_symlink()), None):
        raise ValueError(f"artifact tree contains a symbolic link: {symlink.relative_to(root)}")
    files = sorted(path for path in paths if path.is_file())
    if not files:
        raise ValueError(f"artifact tree is empty: {root}")
    digest = hashlib.sha256()
    total = 0
    for path in files:
        relative = path.relative_to(root).as_posix()
        file_digest = sha256_file(path).removeprefix("sha256:")
        digest.update(f"{file_digest}  ./{relative}\n".encode())
        total += path.stat().st_size
    current_root_stat = root.stat(follow_symlinks=False)
    if (current_root_stat.st_dev, current_root_stat.st_ino) != (
        root_stat.st_dev,
        root_stat.st_ino,
    ):
        raise ValueError(f"artifact tree root changed while hashing: {root}")
    return {
        "algorithm": "sha256-tree-v1",
        "contentDigest": f"sha256:{digest.hexdigest()}",
        "files": len(files),
        "bytes": total,
    }


def artifact_ref(path: Path, root: Path, media_type: str, role: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"artifact must be a regular non-symlink file: {path}")
    digest = sha256_file(path)
    relative = path.relative_to(root).as_posix()
    identity = hashlib.sha256(f"{role}\0{relative}\0{digest}".encode()).hexdigest()[:24]
    return {
        "artifactId": f"artifact:{role}:{identity}",
        "relativePath": relative,
        "contentDigest": digest,
        "bytes": path.stat().st_size,
        "mediaType": media_type,
    }


def safe_extract_tar(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    with tarfile.open(archive, "r:*") as source:
        for member in source:
            relative = Path(member.name)
            parts = relative.parts
            if (
                relative.is_absolute()
                or not parts
                or any(part in {"", ".", ".."} for part in parts)
                or "\\" in member.name
                or member.issym()
                or member.islnk()
                or not (member.isdir() or member.isfile())
            ):
                raise ValueError(f"unsafe archive member: {member.name}")
            normalized = relative.as_posix()
            if normalized in seen:
                raise ValueError(f"duplicate archive member: {member.name}")
            seen.add(normalized)
            target = destination / relative
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            extracted = source.extractfile(member)
            if extracted is None:
                raise ValueError(f"archive member has no content: {member.name}")
            with target.open("xb") as output:
                shutil.copyfileobj(extracted, output)


def image_tree_identity(image_root: Path) -> dict[str, object]:
    """Match `find | sort | sha256sum | sha256sum` using relative paths."""
    if image_root.is_symlink():
        raise ValueError(f"image root is a symbolic link: {image_root}")
    if not image_root.is_dir():
        raise FileNotFoundError(f"image root does not exist: {image_root}")
    paths = list(image_root.rglob("*"))
    for path in paths:
        if path.is_symlink():
            raise ValueError(f"image tree contains a symbolic link: {path.relative_to(image_root)}")
    files = sorted(
        (path for path in paths if path.is_file()),
        key=lambda path: path.relative_to(image_root).as_posix().encode("utf-8"),
    )
    if not files:
        raise ValueError(f"image root contains no files: {image_root}")

    tree_hash = hashlib.sha256()
    total_bytes = 0
    for path in files:
        file_hash = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                file_hash.update(chunk)
        total_bytes += path.stat().st_size
        relative = path.relative_to(image_root).as_posix()
        tree_hash.update(f"{file_hash.hexdigest()}  ./{relative}\n".encode())
    return {
        "algorithm": "sha256-tree-v1",
        "count": len(files),
        "bytes": total_bytes,
        "digest": f"sha256:{tree_hash.hexdigest()}",
    }
