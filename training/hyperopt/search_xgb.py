# training/pipelines/train_xgb.py
from __future__ import annotations
import os, json, sys, logging
import pandas as pd
import mlflow, xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from mlops_fraud.features import prepare_training, build_features
try:
    from mlops_fraud.schemas import TrainingSchema
except Exception:
    TrainingSchema = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

# ---- Config (match search_xgb.py) ----
TRACKING = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5500")
EXP_NAME = os.getenv("MLFLOW_EXPERIMENT", "fraud_hyperopt_search")   # to keep the search runs separate
ARTIFACTS_URI = os.getenv("ARTIFACTS_URI", "").rstrip("/")
EXP_ARTIFACT_DIR = os.getenv("EXP_ARTIFACT_DIR", EXP_NAME)

MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "fraud_xgb")
TRAIN_PATH = os.getenv("TRAIN_PATH", "data/processed/train.csv")
VALID_PATH = os.getenv("VALID_PATH", "data/processed/valid.csv")
RUN_NAME   = os.getenv("RUN_NAME", "xgb_final")
LABEL      = os.getenv("LABEL_COL", "isFraud")  # ≤ keep consistent with search
BEST_PARAMS_JSON = os.getenv("BEST_PARAMS_JSON", "best_xgb_params.json")

def _ensure_experiment(name: str) -> str:
    """Create experiment with S3 artifact root like s3://<bucket>/artifacts/<EXP_ARTIFACT_DIR>."""
    client = MlflowClient()
    exp = client.get_experiment_by_name(name)
    if exp:
        return exp.experiment_id
    artifact_location = f"{ARTIFACTS_URI}/{EXP_ARTIFACT_DIR}" if ARTIFACTS_URI else None
    return client.create_experiment(name, artifact_location=artifact_location)

def _load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if TrainingSchema is not None and LABEL in df.columns:
        try:
            TrainingSchema.validate(df)
        except Exception as e:
            log.warning(f"schema validation failed for {path}: {e}")
    return df

def main():
    # ---- MLflow wiring ----
    mlflow.set_tracking_uri(TRACKING)
    exp_id = _ensure_experiment(EXP_NAME)
    mlflow.set_experiment(EXP_NAME)

    # ---- Data ----
    log.info(f"Loading train: {TRAIN_PATH}")
    df_tr = _load_csv(TRAIN_PATH)
    log.info(f"Loading valid: {VALID_PATH}")
    df_va = _load_csv(VALID_PATH)

    Xtr, ytr, feature_order = prepare_training(df_tr, label_col=LABEL)
    yva = df_va[LABEL].astype(int)
    Xva = build_features(df_va.drop(columns=[LABEL]), for_inference=False).reindex(columns=feature_order, fill_value=0)

    # ---- Params (use best if present) ----
    params = dict(
        max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
        objective="binary:logistic", eval_metric="aucpr",
        n_estimators=400, tree_method="hist",
    )
    if os.path.exists(BEST_PARAMS_JSON):
        log.info(f"Loading best params from {BEST_PARAMS_JSON}")
        with open(BEST_PARAMS_JSON) as f:
            best = json.load(f)
        # coerce integer params
        for k in ("max_depth", "n_estimators"):
            if k in best: best[k] = int(best[k])
        params.update(best)

    # ---- Train + log ----
    with mlflow.start_run(run_name=RUN_NAME):
        run_id = mlflow.active_run().info.run_id
        log.info(f"Experiment ID: {exp_id} | Run ID: {run_id}")

        mlflow.log_params(params)
        clf = xgb.XGBClassifier(**params)
        clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)

        p = clf.predict_proba(Xva)[:, 1]
        pr_auc = float(average_precision_score(yva, p))
        roc    = float(roc_auc_score(yva, p))
        mlflow.log_metric("pr_auc", pr_auc)
        mlflow.log_metric("roc_auc", roc)
        log.info(f"PR AUC: {pr_auc:.4f} | ROC AUC: {roc:.4f}")

        # Signature ensures correct input schema at load/serve time
        sig = infer_signature(Xva, p)
        mlflow.xgboost.log_model(
            xgb_model=clf,
            artifact_path="model",
            signature=sig,
            registered_model_name=MODEL_NAME,  # optional registry
        )

        mlflow.log_text(json.dumps(feature_order), "feature_order.json")
        mlflow.log_dict({"pr_auc": pr_auc, "roc_auc": roc}, "metrics.json")

    print(f"[done] Run: {run_id} | PR-AUC={pr_auc:.4f} ROC-AUC={roc:.4f}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback; traceback.print_exc()
        sys.exit(1)
