"""FastAPI service for the frozen detector.

    PYTHONPATH=src uvicorn api.app:app --port 8000
    curl -F file=@data/fraud_night_1.csv localhost:8000/detect_anomalies/

Differences from the hackathon api.py (legacy/api_original.py) are listed in docs/DECISIONS.md, ADR-006.
"""
import io

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile

from fraud_ae import config
from fraud_ae.model import Detector

MAX_BYTES = 50 * 1024 * 1024
app = FastAPI(title="Fraud anomaly detector", version="2.0.0")
detector = Detector()


@app.get("/health")
def health():
    return {"status": "ok", "model": config.MODEL_PATH.name, "threshold": detector.threshold}


@app.post("/detect_anomalies/")
async def detect_anomalies(file: UploadFile = File(...)):
    raw = await file.read()
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, "file too large")
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception:
        raise HTTPException(400, "could not parse CSV")
    if config.ID_COL not in df.columns:
        raise HTTPException(422, f"missing column: {config.ID_COL}")
    try:
        res = detector.score(df)
    except ValueError as e:  # schema problems, message is ours and safe to return
        raise HTTPException(422, str(e))
    flagged = res[res.is_anomaly]
    return {
        "rows": len(df),
        "threshold": detector.threshold,
        "flagged_users": df.loc[flagged.index, config.ID_COL].tolist(),
        "details": [{"user_id": u, "score": round(float(s), 5), "top_features": f}
                    for u, s, f in zip(df.loc[flagged.index, config.ID_COL], flagged.score, flagged.top_features)],
        "unseen_category_rate": detector.pre.unseen_rate(df).round(4).to_dict(),
    }
