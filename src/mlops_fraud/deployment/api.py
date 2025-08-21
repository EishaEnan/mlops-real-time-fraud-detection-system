#!/usr/bin/env python3
"""FastAPI service to serve the latest fraud model from MLflow."""

from __future__ import annotations

import json
import os
from typing import List

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient

from mlops_fraud.features import build_features
from mlops_fraud.schemas import TransactionRequest

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5500")
MODEL_NAME = os.getenv("MODEL_NAME", "fraud_xgb")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "staging")  # alias, stage or version

app = FastAPI(title="Fraud Detection API")


def _get_model_version(client: MlflowClient, name: str, alias: str):
    """Resolve model version using alias, stage, or direct version."""
    try:
        return client.get_model_version_by_alias(name, alias)
    except Exception:
        if alias.isdigit():
            return client.get_model_version(name, alias)
        versions = client.get_latest_versions(name, stages=[alias])
        if versions:
            return versions[0]
        raise RuntimeError(f"No model version found for {name} with alias/stage '{alias}'")


def _load_feature_order(client: MlflowClient, mv) -> List[str]:
    """Download feature_order.json from the run associated with the model."""
    path = client.download_artifacts(mv.run_id, "feature_order.json")
    with open(path) as f:
        return json.load(f)


def load_model():
    """Load model and its feature order from MLflow."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    mv = _get_model_version(client, MODEL_NAME, MODEL_ALIAS)

    # Try loading via alias first (@alias), fallback to /stage or /version
    try:
        model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
        model = mlflow.xgboost.load_model(model_uri)
    except Exception:
        model_uri = f"models:/{MODEL_NAME}/{MODEL_ALIAS}"
        model = mlflow.xgboost.load_model(model_uri)

    feature_order = _load_feature_order(client, mv)
    return model, feature_order


model, feature_order = load_model()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
def predict(req: TransactionRequest) -> dict:
    if not req.rows:
        raise HTTPException(status_code=400, detail="No rows provided")
    df = pd.DataFrame(req.rows)
    X = build_features(df, for_inference=True).reindex(columns=feature_order, fill_value=0)
    preds = model.predict_proba(X)[:, 1]
    return {"predictions": preds.tolist()}


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
