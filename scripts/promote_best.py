#!/usr/bin/env python3
from __future__ import annotations
import os
import sys
import mlflow
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5500")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT", "fraud_train")
MODEL_NAME = os.getenv("MODEL_NAME", "fraud_xgb")
PRIMARY_METRIC = os.getenv("PRIMARY_METRIC", "PR_auc") # fallback to roc_auc if not set
ALIAS = os.getenv("PROMOTE_ALIAS", "Staging")  # or "Production"
ARCHIVE_OLD = os.getenv("ARCHIVE_OLD", "false").lower() == "true"

def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    c = MlflowClient()

    exp = c.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        print(f"[error] experiment '{EXPERIMENT_NAME}' not found", file=sys.stderr)
        sys.exit(2)

    # Prefer primary metric; if missing on a run, use roc_auc
    order = [f"metrics.{PRIMARY_METRIC} DESC", "attributes.start_time DESC"]
    runs = c.search_runs(
        [exp.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=order,
        max_results=50,
    )

    if not runs:
        print(f"[error] no finished runs found in experiment '{EXPERIMENT_NAME}'", file=sys.stderr)
        sys.exit(3)
    
    # Pick first run that has either primary or roc/auc
    best = None
    best_metric_name = PRIMARY_METRIC
    for run in runs:
        if PRIMARY_METRIC in run.data.metrics:
            best = run
            best_metric_name = PRIMARY_METRIC
            break
        elif "roc_auc" in run.data.metrics:
            best = run
            best_metric_name = "roc_auc"
            break
    
    if best is None:
        print(f"[error] no suitable run found with {PRIMARY_METRIC} or roc_auc", file=sys.stderr)
        sys.exit(4)
    
    metric_val = best.data.metrics.get(best_metric_name)
    run_id = best.info.run_id
    src = f"runs:/{run_id}/artifacts/model"

    # Ensure the registered model exists
    try:
        c.get_registered_model(MODEL_NAME)
    except Exception:
        c.create_registered_model(MODEL_NAME)
    
    mv = c.create_model_version(name=MODEL_NAME, source=src, run_id=run_id)

    # Optional alias (supported on recent MLflow versions)
    try:
        c.set_registered_model_alias(
            name=MODEL_NAME,
            alias=ALIAS.lower(),
            version=mv.version,
        )
    except Exception:
        pass
    
    # Tag the version with some context
    try:
        c.set_model_version_tag(
            name=MODEL_NAME,
            version=mv.version,
            key="source_experiment",
            value=EXPERIMENT_NAME)
        c.set_model_version_tag(
            name=MODEL_NAME,
            version=mv.version,
            key="primary_metric",
            value=best_metric_name)
        c.set_model_version_tag(
            name=MODEL_NAME,
            version=mv.version,
            key=best_metric_name,
            value=str(metric_val)
        )

    except Exception:
        pass
    
    ui = f"{MLFLOW_TRACKING_URI}/#/models/{MODEL_NAME}/versions/{mv.version}"
    print(f"[ok] registered {MODEL_NAME} v{mv.version} from run {run_id} with {best_metric_name}={metric_val:.4f} stage={ALIAS}")
    print(f"[ui]: {ui}")

if __name__ == "__main__":
    main()