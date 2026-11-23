"""Load the frozen autoencoder and turn reconstructions into anomaly scores."""
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd

from . import config
from .features import Preprocessor


def load_autoencoder(path=config.MODEL_PATH):
    import keras
    return keras.models.load_model(path, compile=False)  # inference only; no optimizer state needed


def feature_errors(model, X: np.ndarray) -> np.ndarray:
    """Squared reconstruction error per row and per feature, shape (n, 52)."""
    return np.square(X - model.predict(X, batch_size=4096, verbose=0))


def anomaly_score(model, X: np.ndarray) -> np.ndarray:
    """Per-row MSE, the anomaly score used throughout (same definition as the original notebook)."""
    return feature_errors(model, X).mean(axis=1)


class Detector:
    """Preprocessor + autoencoder + frozen threshold. Used by the API and the evaluation."""

    def __init__(self, model=None, pre=None, threshold=None):
        self.model = model or load_autoencoder()
        self.pre = pre or Preprocessor.load()
        self.threshold = threshold if threshold is not None else json.loads(config.THRESHOLD_PATH.read_text())["value"]

    def score(self, df: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
        X = self.pre.transform(df)
        fe = feature_errors(self.model, X)
        s = fe.mean(axis=1)
        top = np.argsort(-fe, axis=1)[:, :top_k]
        feats = np.array(self.pre.features)
        return pd.DataFrame({
            "score": s,
            "is_anomaly": s > self.threshold,
            "top_features": [feats[i].tolist() for i in top],  # simple per-row explanation
        }, index=df.index)
