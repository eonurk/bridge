from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .model import BridgeModule
from .bundle import BundleAssets, load_bundle
from .preprocessing import (
    MatrixInput,
    default_metadata_path,
    load_preprocessing_metadata,
    prepare_methylation_matrix,
    prepare_rna_matrix,
)


@dataclass
class PredictionResult:
    combined: pd.DataFrame
    predictions: pd.DataFrame
    probabilities: pd.DataFrame
    latents: pd.DataFrame


class BridgePredictor:
    """Inference helper that maps count matrices to bridge latents and labels."""

    def __init__(
        self,
        checkpoint: Optional[str | Path] = None,
        metadata: Optional[str | Path] = None,
        classifier: Optional[str | Path] = None,
        bundle: Optional[str | Path] = None,
        device: str = "auto",
    ) -> None:
        self._bundle_assets: Optional[BundleAssets] = None
        if bundle is not None:
            if checkpoint is not None or metadata is not None or classifier is not None:
                raise ValueError(
                    "When 'bundle' is provided, do not also pass checkpoint/metadata/classifier."
                )
            self._bundle_assets = load_bundle(bundle)
            self.checkpoint_path = self._bundle_assets.checkpoint_path
            self.metadata_path = self._bundle_assets.metadata_path
            classifier = self._bundle_assets.classifier_path
        else:
            if checkpoint is None:
                raise ValueError("Provide either 'checkpoint' or 'bundle'.")
            self.checkpoint_path = Path(checkpoint)
            self.metadata_path = (
                Path(metadata) if metadata else default_metadata_path(self.checkpoint_path)
            )

        payload = load_preprocessing_metadata(self.metadata_path)
        self.metadata = payload
        self.options = payload.get("options", {})
        self.rna_features: list[str] = list(payload["rna_features"])
        self.meth_features: list[str] = list(payload["meth_features"])
        self.rna_scaler = payload["rna_scaler"]
        self.meth_scaler = payload["meth_scaler"]

        self.device = self._choose_device(device)
        self.model = BridgeModule.load_from_checkpoint(
            str(self.checkpoint_path),
            map_location=self.device,
        ).to(self.device)
        self.model.eval()

        self.classifier_path: Optional[Path] = None
        self.classifier_model = None
        self.classifier_feature_columns: Optional[list[str]] = None
        self.classifier_classes: Optional[np.ndarray] = None
        self.latents_id_column = "sample_id"

        if classifier is not None:
            self.load_classifier(classifier)

    def close(self) -> None:
        if self._bundle_assets is not None:
            self._bundle_assets.cleanup()
            self._bundle_assets = None

    @staticmethod
    def _choose_device(device: str) -> torch.device:
        req = device.lower()
        if req == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(req)

    def load_classifier(self, classifier: str | Path) -> None:
        self.classifier_path = Path(classifier)
        payload = joblib.load(self.classifier_path)

        if isinstance(payload, dict) and "model" in payload:
            self.classifier_model = payload["model"]
            self.classifier_feature_columns = payload.get("feature_columns")
            classes = payload.get("classes")
            self.classifier_classes = np.asarray(classes) if classes is not None else None
            self.latents_id_column = payload.get("latents_id_column", "sample_id")
        else:
            self.classifier_model = payload
            self.classifier_feature_columns = None
            classes = getattr(payload, "classes_", None)
            self.classifier_classes = np.asarray(classes) if classes is not None else None
            self.latents_id_column = "sample_id"

    def _encode_batches(self, matrix: np.ndarray, encode_fn, batch_size: int) -> np.ndarray:
        tensor = torch.from_numpy(matrix)
        loader = DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=False)
        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(self.device)
                z = encode_fn(batch)
                chunks.append(z.detach().cpu().numpy())
        if not chunks:
            latent_dim = int(self.model.latent_dim)
            return np.empty((0, latent_dim), dtype=np.float32)
        return np.vstack(chunks)

    def encode_rna(
        self,
        rna_counts: MatrixInput,
        batch_size: int = 256,
        use_attention: bool = False,
    ) -> pd.DataFrame:
        normalization = str(self.options.get("rna_normalization", "cpm"))
        log1p = bool(self.options.get("log1p_rna", True))
        matrix, sample_ids = prepare_rna_matrix(
            rna_counts,
            feature_names=self.rna_features,
            scaler=self.rna_scaler,
            normalization=normalization,
            log1p=log1p,
        )
        latents = self._encode_batches(
            matrix,
            encode_fn=lambda batch: self.model.encode_rna(batch, use_attention=use_attention),
            batch_size=batch_size,
        )
        columns = [f"rna_latent_{i}" for i in range(latents.shape[1])]
        out = pd.DataFrame(latents, columns=columns)
        out.insert(0, "sample_id", sample_ids)
        return out

    def encode_methylation(
        self,
        methylation_values: MatrixInput,
        batch_size: int = 256,
        use_attention: bool = False,
    ) -> pd.DataFrame:
        matrix, sample_ids = prepare_methylation_matrix(
            methylation_values,
            feature_names=self.meth_features,
            scaler=self.meth_scaler,
        )
        latents = self._encode_batches(
            matrix,
            encode_fn=lambda batch: self.model.encode_meth(batch, use_attention=use_attention),
            batch_size=batch_size,
        )
        columns = [f"meth_latent_{i}" for i in range(latents.shape[1])]
        out = pd.DataFrame(latents, columns=columns)
        out.insert(0, "sample_id", sample_ids)
        return out

    def predict_rna(
        self,
        rna_counts: MatrixInput,
        batch_size: int = 256,
        use_attention: bool = False,
    ) -> PredictionResult:
        if self.classifier_model is None:
            raise RuntimeError("No classifier loaded. Pass classifier=... or call load_classifier(...).")

        latents = self.encode_rna(rna_counts, batch_size=batch_size, use_attention=use_attention)

        if self.classifier_feature_columns is None:
            feature_columns = [c for c in latents.columns if c != "sample_id"]
        else:
            feature_columns = list(self.classifier_feature_columns)

        missing = [c for c in feature_columns if c not in latents.columns]
        if missing:
            raise KeyError(
                "Latent feature mismatch with classifier model. "
                f"Missing columns: {missing[:10]}"
            )

        X = latents[feature_columns].to_numpy(dtype=np.float32)
        proba = self.classifier_model.predict_proba(X)
        classes = self.classifier_classes
        if classes is None:
            classes = np.asarray(self.classifier_model.classes_)

        pred_index = np.argmax(proba, axis=1)
        pred_labels = classes[pred_index]
        pred_scores = np.max(proba, axis=1)

        predictions = pd.DataFrame(
            {
                "sample_id": latents["sample_id"].to_numpy(),
                "predicted_label": pred_labels,
                "predicted_proba": pred_scores,
            }
        )
        probabilities = pd.DataFrame(proba, columns=[f"proba_{c}" for c in classes])
        probabilities.insert(0, "sample_id", latents["sample_id"].to_numpy())
        combined = predictions.merge(
            probabilities.drop(columns=["sample_id"]),
            left_index=True,
            right_index=True,
            how="left",
        ).merge(
            latents.drop(columns=["sample_id"]),
            left_index=True,
            right_index=True,
            how="left",
        )

        return PredictionResult(
            combined=combined,
            predictions=predictions,
            probabilities=probabilities,
            latents=latents,
        )
