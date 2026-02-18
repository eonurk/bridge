from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from bridge.bundle import (
    BUNDLE_VERSION,
    CHECKPOINT_NAME,
    CLASSIFIER_NAME,
    MANIFEST_NAME,
    METADATA_NAME,
    load_bundle,
)


def _write_bundle(
    path: Path,
    *,
    version: int = BUNDLE_VERSION,
    include_classifier: bool = True,
) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(MANIFEST_NAME, json.dumps({"version": version}))
        zf.writestr(CHECKPOINT_NAME, b"checkpoint")
        zf.writestr(METADATA_NAME, b"metadata")
        if include_classifier:
            zf.writestr(CLASSIFIER_NAME, b"classifier")


def test_load_bundle_extracts_and_cleans_tmpdir(tmp_path: Path) -> None:
    bundle_path = tmp_path / "ok.bundle"
    _write_bundle(bundle_path)

    assets = load_bundle(bundle_path)
    tmpdir = assets.checkpoint_path.parent
    assert assets.checkpoint_path.exists()
    assert assets.metadata_path.exists()
    assert assets.classifier_path.exists()

    assets.cleanup()
    assert not tmpdir.exists()


def test_load_bundle_rejects_missing_entries(tmp_path: Path) -> None:
    bundle_path = tmp_path / "missing.bundle"
    _write_bundle(bundle_path, include_classifier=False)

    with pytest.raises(ValueError, match="missing entries"):
        load_bundle(bundle_path)


def test_load_bundle_rejects_version_mismatch(tmp_path: Path) -> None:
    bundle_path = tmp_path / "wrong-version.bundle"
    _write_bundle(bundle_path, version=BUNDLE_VERSION + 1)

    with pytest.raises(ValueError, match="Unsupported bundle version"):
        load_bundle(bundle_path)

