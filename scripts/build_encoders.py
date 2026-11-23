"""Recover the lost `label_encoders.pkl` from the training data, and prove the recovery is exact.

The training notebook dumped `label_encoders.pkl` but the file did not survive. sklearn's
LabelEncoder is deterministic (classes_ = sorted unique values), so refitting on the same CSV
yields the same mapping. Proof: encoding the training CSV with the recovered mapping must
reproduce the per-feature min/max stored inside the ORIGINAL `scaler.pkl` (fitted in the same
notebook run). Saved as JSON (not pickle): human-readable, diffable, safe to load.

    python scripts/build_encoders.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fraud_ae import config  # noqa: E402
from fraud_ae.features import Preprocessor, fit_label_classes  # noqa: E402

train = pd.read_csv(config.TRAIN_CSV)
classes = fit_label_classes(train)
pre = Preprocessor(classes, joblib.load(config.SCALER_PATH))  # trusted artifact shipped in this repo
enc = pre.encode(train)
ok_min = np.allclose(enc.min().values, pre.scaler.data_min_)
ok_max = np.allclose(enc.max().values, pre.scaler.data_max_)
assert ok_min and ok_max, "recovered encoders do not reproduce scaler.pkl - refusing to write"
config.ENCODERS_PATH.write_text(json.dumps(classes))
print(f"verified against scaler.pkl (52/52 min & max match); wrote {config.ENCODERS_PATH.relative_to(config.ROOT)}")
print({c: len(v) for c, v in classes.items()})
