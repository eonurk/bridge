from __future__ import annotations

__all__ = ["BridgePredictor", "PredictionResult"]


def __getattr__(name: str):
    if name in {"BridgePredictor", "PredictionResult"}:
        from .predictor import BridgePredictor, PredictionResult

        return {"BridgePredictor": BridgePredictor, "PredictionResult": PredictionResult}[name]
    raise AttributeError(f"module 'bridge' has no attribute {name!r}")
