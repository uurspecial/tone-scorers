"""Portable, checksum-verified scorer artifacts; no ML dependencies needed."""
from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_manifest(path: Path | None = None) -> dict:
    data = json.loads((path or ROOT / "models.json").read_text(encoding="utf-8"))
    ids = [m["id"] for m in data["models"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate scorer IDs in manifest")
    for item in data["models"]:
        if Path(item["id"]).name != item["id"] or item["id"] in {".", ".."}:
            raise ValueError("Invalid scorer ID")
        for name in item["files"]:
            if Path(name).name != name or name in {".", ".."}:
                raise ValueError("Invalid artifact filename")
    return data


def verify_adapter(directory: Path, descriptor: dict) -> None:
    for name, expected in descriptor["files"].items():
        path = directory / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Missing regular adapter file: {name}")
        if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            raise ValueError(f"Adapter checksum mismatch: {name}")


def install_archive(archive: Path, adapter_root: Path, descriptor: dict) -> Path:
    """Extract only the exact allowlisted regular files, then install atomically."""
    if sha256(archive) != descriptor["archive_sha256"]:
        raise ValueError("Release archive checksum mismatch")
    adapter_root.mkdir(parents=True, exist_ok=True)
    destination = adapter_root / descriptor["id"]
    if destination.exists():
        verify_adapter(destination, descriptor)
        return destination
    staging = Path(tempfile.mkdtemp(prefix=".extract-", dir=adapter_root))
    try:
        with tarfile.open(archive, "r:gz") as handle:
            seen = set()
            for member in handle:
                if member.name not in descriptor["files"] or not member.isfile() or member.name in seen:
                    raise ValueError("Unexpected or unsafe archive member")
                expected = descriptor["files"][member.name]
                if member.size != expected["bytes"]:
                    raise ValueError("Unexpected archive member size")
                with handle.extractfile(member) as source, (staging / member.name).open("wb") as target:
                    shutil.copyfileobj(source, target)
                seen.add(member.name)
            if seen != set(descriptor["files"]):
                raise ValueError("Incomplete release archive")
        verify_adapter(staging, descriptor)
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return destination
