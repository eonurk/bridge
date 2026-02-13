from __future__ import annotations

from pathlib import Path
from typing import Iterable, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

METADATA_VERSION = 1
MatrixInput = Union[str, Path, pd.DataFrame]


def default_metadata_path(checkpoint_path: Path) -> Path:
    suffix = ".meta.joblib"
    try:
        return checkpoint_path.with_suffix(suffix)
    except ValueError:
        return checkpoint_path.parent / f"{checkpoint_path.name}{suffix}"


def load_preprocessing_metadata(metadata_path: Path) -> dict:
    payload = joblib.load(metadata_path)
    version = payload.get("version")
    if version is not None and version != METADATA_VERSION:
        raise ValueError(
            f"Incompatible metadata version: expected {METADATA_VERSION}, got {version}"
        )
    required = {"rna_features", "meth_features", "rna_scaler", "meth_scaler"}
    missing = sorted(required.difference(payload.keys()))
    if missing:
        raise KeyError(f"Metadata file is missing keys: {missing}")
    return payload


def _read_matrix(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, sep=None, engine="python", index_col=0)
    if df.empty:
        raise ValueError(f"{path} produced an empty matrix.")
    return df


def read_matrix(matrix: MatrixInput) -> pd.DataFrame:
    if isinstance(matrix, pd.DataFrame):
        df = matrix.copy()
    else:
        path = Path(matrix)
        df = _read_matrix(path)
    if "Gene" in df.columns:
        df = df.set_index("Gene")
    if df.index.name is None:
        df.index.name = "feature"
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.dropna(axis=1, how="all")
    if numeric.empty:
        raise ValueError("Matrix has no numeric columns after parsing.")
    return numeric


def orient_samples_by_features(df: pd.DataFrame, feature_names: Iterable[str]) -> pd.DataFrame:
    feature_set = set(feature_names)
    overlap_cols = len(feature_set.intersection(df.columns))
    overlap_rows = len(feature_set.intersection(df.index))

    if overlap_rows > overlap_cols:
        df = df.transpose()
        df.index = df.index.astype(str)
        df.columns = df.columns.astype(str)
        overlap_cols = len(feature_set.intersection(df.columns))

    if overlap_cols == 0:
        raise ValueError(
            "Unable to align matrix with training features. "
            "No overlap with feature names in rows or columns."
        )
    return df


def _normalize_rna(df: pd.DataFrame, method: str) -> pd.DataFrame:
    method = method.lower()
    if method == "none":
        return df
    if method == "cpm":
        library_sizes = df.sum(axis=1)
        library_sizes = library_sizes.mask(library_sizes == 0, 1.0)
        return df.div(library_sizes, axis=0) * 1e6
    if method == "zscore":
        stds = df.std(axis=1, ddof=0).replace(0, 1.0)
        return df.subtract(df.mean(axis=1), axis=0).div(stds, axis=0)
    raise ValueError(f"Unsupported RNA normalization method: {method}")


def _scale_with_reference(df: pd.DataFrame, feature_names: list[str], scaler: StandardScaler) -> np.ndarray:
    aligned = df.reindex(columns=feature_names)
    means = pd.Series(np.asarray(scaler.mean_, dtype=np.float64), index=feature_names)
    aligned = aligned.fillna(means)

    values = aligned.to_numpy(dtype=np.float64, copy=True)
    if hasattr(scaler, "scale_"):
        scales = np.asarray(scaler.scale_, dtype=np.float64)
    else:
        scales = np.sqrt(np.asarray(scaler.var_, dtype=np.float64))
    scales = np.where(scales == 0.0, 1.0, scales)

    scaled = (values - np.asarray(scaler.mean_, dtype=np.float64)) / scales
    return scaled.astype(np.float32, copy=False)


def prepare_rna_matrix(
    matrix: MatrixInput,
    feature_names: list[str],
    scaler: StandardScaler,
    normalization: str = "cpm",
    log1p: bool = True,
) -> tuple[np.ndarray, list[str]]:
    df = read_matrix(matrix)
    df = orient_samples_by_features(df, feature_names)
    df = _coerce_numeric(df)
    df = _normalize_rna(df, normalization)
    if log1p:
        df = np.log1p(df)

    sample_ids = df.index.astype(str).tolist()
    scaled = _scale_with_reference(df, feature_names, scaler)
    return scaled, sample_ids


def prepare_methylation_matrix(
    matrix: MatrixInput,
    feature_names: list[str],
    scaler: StandardScaler,
) -> tuple[np.ndarray, list[str]]:
    df = read_matrix(matrix)
    df = orient_samples_by_features(df, feature_names)
    df = _coerce_numeric(df).clip(0.0, 1.0)

    sample_ids = df.index.astype(str).tolist()
    scaled = _scale_with_reference(df, feature_names, scaler)
    return scaled, sample_ids
