# training/pipelines/train_xgb.py
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys

import mlflow
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
import xgboost as xgb

from mlops_fraud.features import build_features, prepare_training

try:
    from mlops_fraud.schemas import TrainingSchema
except Exception:
    TrainingSchema = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---- Config (env) ----
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5500")
EXP_NAME = os.getenv("MLFLOW_EXPERIMENT", "fraud_train")          # experiment name
EXP_ARTIFACT_DIR = os.getenv("EXP_ARTIFACT_DIR", "fraud_train")        # subfolder under artifacts/
ARTIFACTS_URI = os.getenv("ARTIFACTS_URI", "").rstrip("/")
MODEL_NAME = os.getenv("MODEL_NAME", "fraud_xgb")
TRAIN_PATH = os.getenv("TRAIN_PATH", "data/processed/train.csv")
VALID_PATH = os.getenv("VALID_PATH", "data/processed/valid.csv")
RUN_NAME = os.getenv("RUN_NAME", "xgb_final")
LABEL = os.getenv("LABEL_COL", "isFraud")                        # ensure it matches your data
BEST_PARAMS_JSON = os.getenv("BEST_PARAMS_JSON", "best_xgb_params.json")

def _ensure_experiment(name: str, artifacts_uri: str, subdir: str) -> str:
    """
    Ensure experiment exists with a stable artifact root:
      - If ARTIFACTS_URI is set:   s3://.../artifacts/<subdir>
      - Else (fallback/proxy):     mlflow-artifacts:/artifacts/<subdir>
    """
    client = MlflowClient()
    exp = client.get_experiment_by_name(name)
    desired = (f"{artifacts_uri}/artifacts/{subdir}" if artifacts_uri
               else f"mlflow-artifacts:/artifacts/{subdir}")
    if exp is None:
        return client.create_experiment(name, artifact_location=desired)
    if exp.artifact_location != desired:
        logger.warning("Experiment '%s' artifact_location=%s != desired=%s",
                       name, exp.artifact_location, desired)
    return exp.experiment_id

# ---- Data loading and feature preparation ----
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if TrainingSchema is not None and LABEL in df.columns:
        try:
            TrainingSchema.validate(df)
        except Exception as e:
            logger.warning(f"schema validation failed for {path}: {e}")
    return df

def _sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    # ---- Wire MLflow ----
    logger.info(f"Setting MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    exp_id = _ensure_experiment(EXP_NAME, ARTIFACTS_URI, EXP_ARTIFACT_DIR)
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

        # Build model card
        ds_path = "data/processed/paysim_features.csv"  
        dataset_hash = _sha256_file(ds_path) if os.path.exists(ds_path) else "n/a"

        # --- Model card (Markdown) ---
        run_id = mlflow.active_run().info.run_id
        feature_list = list(feature_order)

        card_lines = [
            f"# Model Card: {MODEL_NAME}",
            "",
            "## Summary",
            f"- **Run ID:** `{run_id}`",
            f"- **Experiment:** `{EXP_NAME}`",
            f"- **Artifact Subdir:** `{EXP_ARTIFACT_DIR}`",
            f"- **Train file:** `{TRAIN_PATH}`",
            f"- **Valid file:** `{VALID_PATH}`",
            f"- **Dataset hash (SHA256):** `{dataset_hash}`",
            f"- **Feature count:** {len(feature_list)}",
            f"- **Key metrics:** PR_AUC={pr_auc:.6f}, ROC_AUC={roc:.6f}",
            "",
            "## Params",
            "```json",
            json.dumps(params, indent=2),
            "```",
            "",
            "## Features",
            "```json",
            json.dumps(feature_list, indent=2),
            "```",
        ]
        card = "\n".join(card_lines)

        # Store alongside other run artifacts
        mlflow.log_text(card, "model_card.md")

        # Tag the run with discoverable pointers
        mlflow.set_tags({
            "model_card": "model_card.md",
            "model_card_uri": mlflow.get_artifact_uri("model_card.md"),
            "dataset_hash": dataset_hash,
            "feature_count": str(len(feature_list)),
            "exp_artifact_dir": EXP_ARTIFACT_DIR,
        })

    print(f"[done] PR-AUC={pr_auc:.4f} ROC-AUC={roc:.4f}")

if __name__ == "__main__":
    try:
        main()
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)
