from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "bridge"
DEFAULT_BUNDLE_NAME = "bridge_inference.bundle"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _cache_path_for_url(bundle_url: str, cache_dir: Path) -> Path:
    parsed = urlparse(bundle_url)
    name = Path(parsed.path).name or DEFAULT_BUNDLE_NAME
    suffix = "".join(Path(name).suffixes)
    stem = name[: -len(suffix)] if suffix else name
    key = _sha256_text(bundle_url)[:12]
    final_name = f"{stem}-{key}{suffix}" if suffix else f"{stem}-{key}"
    return cache_dir / final_name


def _validate_checksum(path: Path, expected_sha256: str | None) -> None:
    if expected_sha256 is None:
        return
    expected = expected_sha256.strip().lower()
    if not expected:
        return
    observed = _sha256_file(path)
    if observed != expected:
        raise ValueError(
            f"Checksum mismatch for {path}: expected sha256={expected}, got {observed}"
        )


def _download_to_path(bundle_url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with urlopen(bundle_url) as response, tmp_path.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        tmp_path.replace(destination)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def resolve_bundle_path(
    *,
    bundle: Path | None,
    bundle_url: str | None,
    bundle_cache_dir: Path | None = None,
    bundle_sha256: str | None = None,
) -> Path | None:
    """Resolve a bundle path from local path or URL (+ cache)."""
    if bundle is not None and bundle_url is not None:
        raise ValueError("Use either bundle path or bundle URL, not both.")

    if bundle is not None:
        path = Path(bundle).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Bundle file not found: {path}")
        _validate_checksum(path, bundle_sha256)
        return path

    if bundle_url is None:
        return None

    cache_dir = Path(bundle_cache_dir).expanduser() if bundle_cache_dir else DEFAULT_CACHE_DIR
    cache_path = _cache_path_for_url(bundle_url, cache_dir)
    if cache_path.exists():
        _validate_checksum(cache_path, bundle_sha256)
        return cache_path

    _download_to_path(bundle_url, cache_path)
    try:
        _validate_checksum(cache_path, bundle_sha256)
    except Exception:
        if cache_path.exists():
            cache_path.unlink()
        raise
    return cache_path


def env_path(var_name: str) -> Path | None:
    value = os.getenv(var_name)
    if value is None or not value.strip():
        return None
    return Path(value).expanduser()
