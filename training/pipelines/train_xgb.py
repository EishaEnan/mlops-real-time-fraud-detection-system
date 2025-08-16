# training/pipelines/train_xgb.py
from __future__ import annotations
import os, json, sys, logging
import pandas as pd
import mlflow
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from mlops_fraud.features import prepare_training, build_features
try:
    from mlops_fraud.schemas import TrainingSchema
except Exception:
    TrainingSchema = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---- Config (env) ----
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5500")
EXP_NAME = os.getenv("MLFLOW_EXPERIMENT", "fraud_train")          # experiment name
EXP_ARTIFACT_DIR = os.getenv("EXP_ARTIFACT_DIR", EXP_NAME)        # subfolder under artifacts/
MODEL_NAME = os.getenv("MODEL_NAME", "fraud_xgb")
TRAIN_PATH = os.getenv("TRAIN_PATH", "data/processed/train.csv")
VALID_PATH = os.getenv("VALID_PATH", "data/processed/valid.csv")
RUN_NAME = os.getenv("RUN_NAME", "xgb_final")
LABEL = os.getenv("LABEL_COL", "isFraud")                        # ensure it matches your data
BEST_PARAMS_JSON = os.getenv("BEST_PARAMS_JSON", "best_xgb_params.json")

def _ensure_experiment(name: str) -> str:
    """
    Create the experiment with an artifact root routed via the MLflow artifact proxy:
      mlflow-artifacts:/artifacts/<EXP_ARTIFACT_DIR>
    The tracking server maps this to your S3 bucket/prefix (ARTIFACTS_URI).
    """
    client = MlflowClient()
    exp = client.get_experiment_by_name(name)
    if exp:
        return exp.experiment_id
    artifact_location = f"mlflow-artifacts:/artifacts/{EXP_ARTIFACT_DIR}"
    return client.create_experiment(name, artifact_location=artifact_location)

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if TrainingSchema is not None and LABEL in df.columns:
        try:
            TrainingSchema.validate(df)
        except Exception as e:
            logger.warning(f"schema validation failed for {path}: {e}")
    return df

def main():
    # ---- Wire MLflow ----
    logger.info(f"Setting MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    exp_id = _ensure_experiment(EXP_NAME)
    mlflow.set_experiment(EXP_NAME)

    # ---- Data ----
    logger.info(f"Loading train: {TRAIN_PATH}")
    df_tr = load_csv(TRAIN_PATH)
    logger.info(f"Loading valid: {VALID_PATH}")
    df_va = load_csv(VALID_PATH)

    logger.info("Building features")
    Xtr, ytr, feature_order = prepare_training(df_tr, label_col=LABEL)
    yva = df_va[LABEL].astype(int)
    Xva = build_features(df_va.drop(columns=[LABEL]), for_inference=False).reindex(columns=feature_order, fill_value=0)

    # ---- Params (use best if present) ----
    base_params = dict(
        max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
        objective="binary:logistic", eval_metric="aucpr", n_estimators=400, tree_method="hist",
    )
    if os.path.exists(BEST_PARAMS_JSON):
        logger.info(f"Loading best params from {BEST_PARAMS_JSON}")
        with open(BEST_PARAMS_JSON) as f:
            best = json.load(f)
        # coerce possible floats to ints for integer params
        best["max_depth"] = int(best.get("max_depth", base_params["max_depth"]))
        best["n_estimators"] = int(best.get("n_estimators", base_params["n_estimators"]))
        params = {**base_params, **best}
    else:
        logger.info("best_xgb_params.json not found; using base params")
        params = base_params

    # ---- Train + log ----
    with mlflow.start_run(run_name=RUN_NAME):
        logger.info(f"Experiment ID: {exp_id}  |  Run: {mlflow.active_run().info.run_id}")

        mlflow.log_params(params)

        clf = xgb.XGBClassifier(**params)
        clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)

        p = clf.predict_proba(Xva)[:, 1]
        pr_auc = float(average_precision_score(yva, p))
        roc = float(roc_auc_score(yva, p))
        logger.info(f"PR AUC: {pr_auc:.4f}  |  ROC AUC: {roc:.4f}")

        mlflow.log_metric("pr_auc", pr_auc)
        mlflow.log_metric("roc_auc", roc)

        # log model via server proxy (works without client AWS creds)
        sig = infer_signature(Xva, p)
        mlflow.xgboost.log_model(
            xgb_model=clf,
            artifact_path="model",
            signature=sig,
            registered_model_name=MODEL_NAME,   # optional: registers in Model Registry
        )

        # artifacts: feature order + metrics blob
        mlflow.log_text(json.dumps(feature_order), "feature_order.json")
        mlflow.log_dict({"pr_auc": pr_auc, "roc_auc": roc}, "metrics.json")

    print(f"[done] PR-AUC={pr_auc:.4f} ROC-AUC={roc:.4f}")

if __name__ == "__main__":
    try:
        main()
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)
