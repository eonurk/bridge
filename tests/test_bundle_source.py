from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pytest

from bridge import bundle_source
from bridge.bundle_source import resolve_bundle_path


class _BytesResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def test_resolve_bundle_local_path(tmp_path: Path) -> None:
    bundle = tmp_path / "bridge.bundle"
    bundle.write_bytes(b"payload")

    resolved = resolve_bundle_path(bundle=bundle, bundle_url=None)
    assert resolved == bundle


def test_resolve_bundle_local_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_bundle_path(bundle=tmp_path / "missing.bundle", bundle_url=None)


def test_resolve_bundle_downloads_once_and_reuses_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"bundle-bytes"
    calls: list[str] = []

    def fake_urlopen(url: str):
        calls.append(url)
        return _BytesResponse(payload)

    monkeypatch.setattr(bundle_source, "urlopen", fake_urlopen)

    cache_dir = tmp_path / "cache"
    url = "https://example.org/models/bridge_inference.bundle"
    first = resolve_bundle_path(bundle=None, bundle_url=url, bundle_cache_dir=cache_dir)
    second = resolve_bundle_path(bundle=None, bundle_url=url, bundle_cache_dir=cache_dir)

    assert first is not None
    assert first == second
    assert first.exists()
    assert first.read_bytes() == payload
    assert len(calls) == 1


def test_resolve_bundle_checksum_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = b"bundle-bytes"
    sha = hashlib.sha256(payload).hexdigest()

    def fake_urlopen(_url: str):
        return _BytesResponse(payload)

    monkeypatch.setattr(bundle_source, "urlopen", fake_urlopen)

    cache_dir = tmp_path / "cache"
    url = "https://example.org/models/bridge_inference.bundle"
    resolve_bundle_path(
        bundle=None,
        bundle_url=url,
        bundle_cache_dir=cache_dir,
        bundle_sha256=sha,
    )

    with pytest.raises(ValueError, match="Checksum mismatch"):
        resolve_bundle_path(
            bundle=None,
            bundle_url=url,
            bundle_cache_dir=cache_dir,
            bundle_sha256="f" * 64,
        )


def test_resolve_bundle_bad_download_is_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"bundle-bytes"

    def fake_urlopen(_url: str):
        return _BytesResponse(payload)

    monkeypatch.setattr(bundle_source, "urlopen", fake_urlopen)

    cache_dir = tmp_path / "cache"
    url = "https://example.org/models/another.bundle"
    with pytest.raises(ValueError, match="Checksum mismatch"):
        resolve_bundle_path(
            bundle=None,
            bundle_url=url,
            bundle_cache_dir=cache_dir,
            bundle_sha256="0" * 64,
        )

    assert not any(cache_dir.glob("*"))


def test_resolve_bundle_rejects_path_and_url(tmp_path: Path) -> None:
    bundle = tmp_path / "bridge.bundle"
    bundle.write_bytes(b"payload")
    with pytest.raises(ValueError, match="either bundle path or bundle URL"):
        resolve_bundle_path(bundle=bundle, bundle_url="https://example.org/model.bundle")
