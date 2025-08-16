# training/pipelines/evaluate.py
from __future__ import annotations
import os, json
import pandas as pd
from sklearn.metrics import precision_recall_curve
from mlops_fraud.features import build_features

LABEL = os.getenv("LABEL_COL", "isFraud")
VALID_PATH = os.getenv("VALID_PATH", "data/processed/valid.csv")
FEATURE_ORDER_PATH = os.getenv("FEATURE_ORDER_PATH", "")  # optional path to feature_order.json

def main():
    df = pd.read_csv(VALID_PATH)
    y = df[LABEL].astype(int)
    X = build_features(df.drop(columns=[LABEL]), for_inference=False)

    # Placeholder: save feature schema snapshot for debugging.
    snap = {"columns": X.columns.tolist(), "n_rows": int(X.shape[0])}
    with open("eval_snapshot.json", "w") as f:
        json.dump(snap, f)
    print("[eval] wrote eval_snapshot.json")

if __name__ == "__main__":
    main()
