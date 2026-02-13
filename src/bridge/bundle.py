from __future__ import annotations

import json
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

BUNDLE_VERSION = 1
MANIFEST_NAME = "manifest.json"
CHECKPOINT_NAME = "checkpoint.ckpt"
METADATA_NAME = "metadata.joblib"
CLASSIFIER_NAME = "classifier.joblib"


@dataclass
class BundleAssets:
    checkpoint_path: Path
    metadata_path: Path
    classifier_path: Path
    _tempdir: tempfile.TemporaryDirectory[str]

    def cleanup(self) -> None:
        self._tempdir.cleanup()


def load_bundle(bundle_path: str | Path) -> BundleAssets:
    path = Path(bundle_path)
    if not path.exists():
        raise FileNotFoundError(f"Bundle file not found: {path}")

    tempdir = tempfile.TemporaryDirectory(prefix="bridge_bundle_")
    out_dir = Path(tempdir.name)

    with zipfile.ZipFile(path, mode="r") as zf:
        names = set(zf.namelist())
        required = {MANIFEST_NAME, CHECKPOINT_NAME, METADATA_NAME, CLASSIFIER_NAME}
        missing = sorted(required.difference(names))
        if missing:
            tempdir.cleanup()
            raise ValueError(f"Invalid bundle, missing entries: {missing}")

        manifest = json.loads(zf.read(MANIFEST_NAME).decode("utf-8"))
        version = manifest.get("version")
        if version != BUNDLE_VERSION:
            tempdir.cleanup()
            raise ValueError(
                f"Unsupported bundle version {version}; expected {BUNDLE_VERSION}"
            )

        zf.extract(CHECKPOINT_NAME, path=out_dir)
        zf.extract(METADATA_NAME, path=out_dir)
        zf.extract(CLASSIFIER_NAME, path=out_dir)

    return BundleAssets(
        checkpoint_path=out_dir / CHECKPOINT_NAME,
        metadata_path=out_dir / METADATA_NAME,
        classifier_path=out_dir / CLASSIFIER_NAME,
        _tempdir=tempdir,
    )
