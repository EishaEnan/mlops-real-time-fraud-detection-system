# training/pipelines/evaluate.py
from __future__ import annotations

import json
import os

import mlflow
import pandas as pd

from mlops_fraud.features import build_features

LABEL = os.getenv("LABEL_COL", "isFraud")
VALID_PATH = os.getenv("VALID_PATH", "data/processed/valid.csv")
FEATURE_ORDER_PATH = os.getenv("FEATURE_ORDER_PATH", "")  # optional path to feature_order.json

EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT", "fraud_train")
ARTIFACTS_URI = os.getenv("ARTIFACTS_URI", "").rstrip("/")
EXP_ARTIFACT_DIR = os.getenv("EXP_ARTIFACT_DIR", "fraud_train")


def ensure_experiment(name: str, artifacts_uri: str, subdir: str) -> str:
    """Create (if needed) the experiment with a fixed artifact root."""
    exp = mlflow.get_experiment_by_name(name)
    desired = f"{artifacts_uri}/artifacts/{subdir}" if artifacts_uri else None
    if exp is None:
        exp_id = mlflow.create_experiment(name, artifact_location=desired)
        return exp_id
    # If exists but different location, just warn; changing it in-place isn’t supported.
    if desired and exp.artifact_location != desired:
        print(f"[warn] Experiment '{name}' artifact_location={exp.artifact_location} != desired={desired}")
    return exp.experiment_id


# Call once at start of main()
exp_id = ensure_experiment(EXPERIMENT_NAME, ARTIFACTS_URI, EXP_ARTIFACT_DIR)
mlflow.set_experiment(experiment_name=EXPERIMENT_NAME)


def main():
    df = pd.read_csv(VALID_PATH)
    # y = df[LABEL].astype(int)
    X = build_features(df.drop(columns=[LABEL]), for_inference=False)

    # Placeholder: save feature schema snapshot for debugging.
    snap = {"columns": X.columns.tolist(), "n_rows": int(X.shape[0])}
    with open("eval_snapshot.json", "w") as f:
        json.dump(snap, f)
    print("[eval] wrote eval_snapshot.json")


if __name__ == "__main__":
    main()
