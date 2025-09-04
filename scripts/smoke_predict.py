import json
import os
import tempfile

import boto3
import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd

# ----- Config from env -----
TRACKING = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
ART_ROOT = os.getenv("ARTIFACTS_URI", "").rstrip("/")  # e.g., s3://mlops-fraud-dvc
MODEL_NAME = os.getenv("MODEL_NAME", "fraud_xgb")
ALIAS = os.getenv("MODEL_ALIAS", "staging")
REGION = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")

assert ART_ROOT.startswith("s3://"), "ARTIFACTS_URI must be s3://<bucket>[/prefix]"
assert REGION, "Set AWS_REGION or AWS_DEFAULT_REGION"


# ----- Helpers -----
def map_mlflow_artifacts_to_s3(artifacts_uri: str, artifacts_root: str) -> str:
    # artifacts_uri looks like: mlflow-artifacts:/artifacts/<exp_name>/<run_id>/artifacts
    if not artifacts_uri.startswith("mlflow-artifacts:/"):
        raise ValueError(f"Unexpected artifacts_uri: {artifacts_uri}")
    tail = artifacts_uri.removeprefix("mlflow-artifacts:/").lstrip("/")
    base = artifacts_root.rstrip("/")
    # if both contain '/artifacts', avoid double-joining
    if tail.startswith("artifacts/") and base.endswith("/artifacts"):
        base = base[: -len("/artifacts")]
    return f"{base}/{tail}"


def discover_registry_model_dir(exp_name: str) -> str:
    """
    Return s3://.../artifacts/<exp_name>/models/m-*/artifacts (latest by LastModified).
    """
    _, rest = ART_ROOT.split("s3://", 1)
    bucket, *prefix = rest.split("/", 1)
    base_prefix = (prefix[0] + "/") if prefix else ""
    models_prefix = f"{base_prefix}artifacts/{exp_name}/models/"

    s3 = boto3.client("s3", region_name=REGION)
    paginator = s3.get_paginator("list_objects_v2")

    newest_key, newest_time = None, None
    for page in paginator.paginate(Bucket=bucket, Prefix=models_prefix):
        for obj in page.get("Contents", []) or []:
            key = obj["Key"]
            if key.endswith("artifacts/MLmodel"):
                lm = obj["LastModified"]
                if newest_time is None or lm > newest_time:
                    newest_time, newest_key = lm, key
    if not newest_key:
        raise RuntimeError(f"No registry MLmodel under s3://{bucket}/{models_prefix}")
    return "s3://" + bucket + "/" + newest_key.rsplit("/", 1)[0]  # .../artifacts


# ----- Resolve model, run, and paths -----
mlflow.set_tracking_uri(TRACKING)
c = MlflowClient()
mv = c.get_model_version_by_alias(MODEL_NAME, ALIAS)
run = c.get_run(mv.run_id)
exp = c.get_experiment(run.info.experiment_id)

print(f"Resolved {MODEL_NAME}@{ALIAS} -> v{mv.version}, run_id={mv.run_id}")
print("run.info.artifact_uri:", run.info.artifact_uri)
print("experiment:", exp.name)

# Prefer registry folder (works even if mv.source points to a run path)
s3_model_dir = discover_registry_model_dir(exp.name)
print("Using model registry dir:", s3_model_dir)

# ----- Download model + feature order -----
with tempfile.TemporaryDirectory() as td:
    local_model_dir = mlflow.artifacts.download_artifacts(s3_model_dir, dst_path=td)
    print("Local model dir:", local_model_dir, "contains:", os.listdir(local_model_dir))

    # Try to fetch feature_order.json from RUN
    feature_order = None
    try:
        p = mlflow.artifacts.download_artifacts(f"runs:/{mv.run_id}/feature_order.json", dst_path=td)
        with open(p) as f:
            feature_order = json.load(f)
        print(f"Loaded feature_order.json with {len(feature_order)} columns")
    except Exception as e:
        print("No feature_order.json — proceeding with DF column order. Reason:", e)

    # ----- Load model (XGBoost → PyFunc) -----
    try:
        from mlflow import xgboost as mlf_xgb

        model = mlf_xgb.load_model(local_model_dir)
        flavor = "xgboost"
        print("✅ Loaded XGBoost flavor")
    except Exception as e1:
        print("XGBoost failed:", e1, "→ trying PyFunc")
        model = mlflow.pyfunc.load_model(local_model_dir)
        flavor = "pyfunc"
        print("✅ Loaded PyFunc flavor")

    # ----- Build features on a sample row and predict -----
    from mlops_fraud.features import build_features

    sample = pd.DataFrame(
        [
            {
                "type": "PAYMENT",
                "amount": 123.45,
                "step": 1,
                "oldbalanceOrg": 1000.0,
                "newbalanceOrig": 876.55,
                "oldbalanceDest": 500.0,
                "newbalanceDest": 623.45,
            }
        ]
    )
    X = build_features(sample, for_inference=True)
    if feature_order:
        X = X.reindex(columns=feature_order, fill_value=0)

    if flavor == "xgboost" and hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        score = float(np.asarray(proba)[:, 1][0])
    else:
        y = np.asarray(model.predict(X))
        if y.ndim == 2 and y.shape[1] == 2:
            y = y[:, 1]
        score = float(y.ravel()[0])

    print("\n=== SMOKE PREDICTION ===")
    print("flavor      :", flavor)
    print("model_ref   :", f"models:/{MODEL_NAME}/{mv.version}")
    print("score       :", score)
