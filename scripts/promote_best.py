# scripts/promote_best.py
from __future__ import annotations
import os, sys
import mlflow
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5500")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT", "fraud_train")  # standardized
MODEL_NAME = os.getenv("MODEL_NAME", "fraud_xgb")
PRIMARY_METRIC = os.getenv("PRIMARY_METRIC", "pr_auc")               # matches training logs
ALIAS = os.getenv("PROMOTE_ALIAS", "staging").lower()                # 'staging' | 'production'
ARCHIVE_OLD = os.getenv("ARCHIVE_OLD", "false").lower() == "true"

def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    c = MlflowClient()

    exp = c.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        print(f"[error] experiment '{EXPERIMENT_NAME}' not found", file=sys.stderr)
        sys.exit(2)

    # Sort by primary metric, then recency
    order = [f"metrics.{PRIMARY_METRIC} DESC", "attributes.start_time DESC"]
    runs = c.search_runs(
        [exp.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=order,
        max_results=50,
    )
    if not runs:
        print(f"[error] no finished runs in experiment '{EXPERIMENT_NAME}'", file=sys.stderr)
        sys.exit(3)

    # Pick first run with PRIMARY_METRIC/pr_auc/roc_auc
    best = None
    metric_name = PRIMARY_METRIC
    for r in runs:
        if PRIMARY_METRIC in r.data.metrics:
            best, metric_name = r, PRIMARY_METRIC
            break
        for alt in ("pr_auc", "roc_auc"):
            if alt in r.data.metrics:
                best, metric_name = r, alt
                break
        if best:
            break

    if not best:
        print(f"[error] no suitable run found with {PRIMARY_METRIC}/pr_auc/roc_auc", file=sys.stderr)
        sys.exit(4)

    run_id = best.info.run_id
    metric_val = best.data.metrics[metric_name]

    # Ensure model exists
    try:
        c.get_registered_model(MODEL_NAME)
    except Exception:
        c.create_registered_model(MODEL_NAME)

    # Register from run artifact
    src = f"runs:/{run_id}/model"
    mv = c.create_model_version(name=MODEL_NAME, source=src, run_id=run_id)

    # Set alias (best effort)
    try:
        c.set_registered_model_alias(MODEL_NAME, ALIAS, mv.version)
    except Exception:
        pass

    # Link model card into registry version
    try:
        run = c.get_run(run_id)
        base = run.info.artifact_uri.rstrip("/")
        card_uri = f"{base}/model_card.md"
        desc = (
            f"Auto-promoted from exp='{EXPERIMENT_NAME}' run_id={run_id} "
            f"| {metric_name}={metric_val:.4f}\nModel card: {card_uri}"
        )
        c.update_model_version(name=MODEL_NAME, version=mv.version, description=desc)
        c.set_model_version_tag(MODEL_NAME, mv.version, "model_card", card_uri)
        # propagate if run already tagged with a canonical URI
        mc_uri = run.data.tags.get("model_card_uri")
        if mc_uri:
            c.set_model_version_tag(MODEL_NAME, mv.version, "model_card_uri", mc_uri)
    except Exception as e:
        print(f"[warn] could not link model card: {e}", file=sys.stderr)

    print(f"[ok] {MODEL_NAME} v{mv.version} <- run {run_id} ({metric_name}={metric_val:.4f}) alias={ALIAS}")

if __name__ == "__main__":
    main()