import json

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fraud_ae import config
from fraud_ae.features import Preprocessor


@pytest.fixture(scope="module")
def normals():
    return pd.read_csv(config.TRAIN_CSV, nrows=2000)


@pytest.fixture(scope="module")
def fraud():
    return pd.read_csv(config.FRAUD_CSV, nrows=200)


def test_encoders_reproduce_original_scaler():
    """Recovered encoders must give the exact min/max stored in the March-2025 scaler.pkl."""
    pre = Preprocessor.load()
    enc = pre.encode(pd.read_csv(config.TRAIN_CSV))
    assert np.allclose(enc.min().values, pre.scaler.data_min_)
    assert np.allclose(enc.max().values, pre.scaler.data_max_)


def test_transform_is_row_independent(normals):
    """Fixes the original train/serve skew: a row's vector must not depend on the batch."""
    pre = Preprocessor.load()
    assert np.allclose(pre.transform(normals)[:5], pre.transform(normals.head(5)))


def test_unseen_category_and_schema(fraud):
    pre = Preprocessor.load()
    assert (pre.encode(fraud)["email_domain"] == -1).any()  # tempmail.com etc. never seen in training
    with pytest.raises(ValueError, match="missing required columns"):
        pre.transform(fraud.drop(columns=["age"]))


def test_api_separates_populations(normals, fraud):
    from api.app import app
    c = TestClient(app)
    body = pd.concat([normals.head(300), fraud.drop(columns="Unnamed: 0")]).to_csv(index=False)
    r = c.post("/detect_anomalies/", files={"file": ("x.csv", body)})
    assert r.status_code == 200
    flagged = set(r.json()["flagged_users"])
    assert set(fraud.user_id) <= flagged
    assert len(flagged & set(normals.user_id.head(300))) <= 15  # ~1% FPR, generous bound
    assert c.post("/detect_anomalies/", files={"file": ("x.csv", "a,b\n1,2")}).status_code == 422


def test_threshold_artifact():
    t = json.loads(config.THRESHOLD_PATH.read_text())
    assert 0 < t["value"] < 1 and t["model"] == "night_4.h5"
