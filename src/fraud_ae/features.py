"""Inference-time preprocessing that reproduces the training transform exactly.

Fixes the original train/serve skew: the hackathon API re-fitted LabelEncoder and MinMaxScaler
on every uploaded batch, so the same user got a different vector depending on who else was in
the file. Here everything is fitted once on training data and only *applied* at inference.
"""
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from . import config

UNSEEN = -1  # same fallback as the original `encode_with_fallback`; see ADR-004


def fit_label_classes(train: pd.DataFrame) -> dict:
    """Sorted unique values per categorical column == what sklearn LabelEncoder learns."""
    return {c: sorted(train[c].astype(str).unique().tolist()) for c in config.CATEGORICAL_COLS}


class Preprocessor:
    def __init__(self, classes: dict, scaler):
        self.classes = classes
        self.lookup = {c: {v: i for i, v in enumerate(vals)} for c, vals in classes.items()}
        self.scaler = scaler
        self.features = list(scaler.feature_names_in_)

    @classmethod
    def load(cls, encoders_path=config.ENCODERS_PATH, scaler_path=config.SCALER_PATH):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # sklearn version-mismatch warning; scaler is plain min/max arrays
            scaler = joblib.load(scaler_path)  # trusted artifact shipped in this repo
        return cls(json.loads(encoders_path.read_text()), scaler)

    def encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate schema, label-encode categoricals, return numeric frame in training column order."""
        missing = [c for c in self.features if c not in df.columns]
        if missing:
            raise ValueError(f"missing required columns: {missing}")
        out = df[self.features].copy()
        for c in config.CATEGORICAL_COLS:
            out[c] = out[c].astype(str).map(self.lookup[c]).fillna(UNSEEN).astype(int)
        num = out.drop(columns=config.CATEGORICAL_COLS).apply(pd.to_numeric, errors="coerce")
        if num.isna().any().any():
            bad = num.columns[num.isna().any()].tolist()
            raise ValueError(f"non-numeric or missing values in: {bad}")
        out[num.columns] = num
        return out.astype(float)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        # No clipping: out-of-range values (>1 or <0) are exactly the evidence we want the AE to see.
        return self.scaler.transform(self.encode(df)).astype("float32")

    def unseen_rate(self, df: pd.DataFrame) -> pd.Series:
        """Share of rows per categorical column whose value was never seen in training (drift signal)."""
        return pd.Series({c: (~df[c].astype(str).isin(self.lookup[c])).mean() for c in config.CATEGORICAL_COLS})
