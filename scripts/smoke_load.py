# scripts/smoke_load.py
import os
import tempfile

import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = os.getenv("MODEL_NAME", "fraud_xgb")
ALIAS = os.getenv("MODEL_ALIAS", "staging")
TRACKING = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
ART_ROOT = os.getenv("ARTIFACTS_URI")  # e.g. s3://mlops-fraud-dvc  (no trailing slash)

assert ART_ROOT and ART_ROOT.startswith("s3://"), (
    "Set ARTIFACTS_URI=s3://<bucket>[/optional-prefix] in env (ex: s3://mlops-fraud-dvc)"
)


def map_mlflow_artifacts_to_s3(artifacts_uri: str, artifacts_root: str) -> str:
    """
    artifacts_uri: 'mlflow-artifacts:/artifacts/<exp>/<run_id>/artifacts'
    artifacts_root: 's3://bucket' OR 's3://bucket/artifacts'
    returns: 's3://bucket/artifacts/<exp>/<run_id>/artifacts'
    (prevents double 'artifacts' when root already ends with /artifacts)
    """
    if not artifacts_uri.startswith("mlflow-artifacts:/"):
        raise ValueError(f"Unexpected artifacts_uri: {artifacts_uri}")
    tail = artifacts_uri.removeprefix("mlflow-artifacts:/").lstrip("/")  # 'artifacts/...'
    base = artifacts_root.rstrip("/")
    if tail.startswith("artifacts/") and base.endswith("/artifacts"):
        base = base.rsplit("/artifacts", 1)[0]
    return f"{base}/{tail}"


print("Tracking URI:", TRACKING)
print("ARTIFACTS_URI:", ART_ROOT)
mlflow.set_tracking_uri(TRACKING)
client = MlflowClient()

# Resolve alias -> version
mv = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
run = client.get_run(mv.run_id)
print(f"Model: {MODEL_NAME}  alias: {ALIAS}  -> version: {mv.version}")
print("run_id:", mv.run_id)
print("run.info.artifact_uri:", run.info.artifact_uri)

# Map to direct S3 and download the *model* directory
s3_run_root = map_mlflow_artifacts_to_s3(run.info.artifact_uri, ART_ROOT)
s3_model_dir = f"{s3_run_root}/model"
print("Direct S3 model dir:", s3_model_dir)

with tempfile.TemporaryDirectory() as td:
    local_dir = mlflow.artifacts.download_artifacts(s3_model_dir, dst_path=td)
    import os as _os

    print("Downloaded to:", local_dir, "files:", _os.listdir(local_dir))

    # Load flavor (XGB -> PyFunc)
    try:
        from mlflow import xgboost as mlf_xgb

        model = mlf_xgb.load_model(local_dir)
        print("✅ Loaded XGBoost flavor")
    except Exception as e1:
        print("XGBoost failed:", e1, "→ trying PyFunc")
        model = mlflow.pyfunc.load_model(local_dir)
        print("✅ Loaded PyFunc flavor")

print("SMOKE LOAD: OK")
