
import io
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Load model bundle produced by the Streamlit app (model.pkl)
with open("model.pkl", "rb") as f:
    bundle = joblib.load(f)

sup_model = bundle["sup_model"]
iso = bundle["iso_model"]
feature_names = bundle["feature_names"]
iso_min = bundle.get("iso_min", 0.0)
iso_max = bundle.get("iso_max", 1.0)

app = FastAPI(title="Fraud Hybrid Scoring API")

class ScoreRequest(BaseModel):
    records: list  # list of dicts with feature_name: value
    blend: float = 0.7
    threshold: float = 0.9

@app.post("/score")
def score(req: ScoreRequest):
    df = pd.DataFrame(req.records)

    # Ensure numeric types (simple category encoding)
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.Categorical(df[c]).codes

    # Add any missing features as zeros, enforce column order
    for f in feature_names:
        if f not in df.columns:
            df[f] = 0
    df = df[feature_names]

    proba = sup_model.predict_proba(df)[:, 1]
    anom_raw = -iso.score_samples(df)
    anom_scaled = (anom_raw - iso_min) / (iso_max - iso_min + 1e-9)
    hybrid = req.blend * proba + (1 - req.blend) * anom_scaled
    pred = (hybrid >= req.threshold).astype(int)

    return {
        "supervised_proba": proba.tolist(),
        "anomaly_scaled": anom_scaled.tolist(),
        "hybrid_score": hybrid.tolist(),
        "fraud_pred_hybrid": pred.tolist(),
    }

@app.get("/")
def root():
    return {"status": "ok", "message": "Use POST /score"}

