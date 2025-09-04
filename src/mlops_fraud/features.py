# common/features.py
from __future__ import annotations

import numpy as np
import pandas as pd

# Canonical categories for 'type' (keeps OHE columns stable accross training runs)
TYPE_CATS = ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']
ID_COLS = ['nameOrig', 'nameDest']
RAW_BAL_COLS = ['oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
LABEL_COL = "isFraud"
_OHE_PREFIX = 'type_'

def _ohe_type(df: pd.DataFrame) -> pd.DataFrame:
    if "type" in df.columns:
        df = df.copy()
        df["type"] = pd.Categorical(df["type"], categories=TYPE_CATS)
        dummies = pd.get_dummies(df["type"], prefix="type").astype("int8")
        df = df.drop(columns=["type"]).join(dummies)
    # ensure all dummy columns exist (stable schema)
    for cat in TYPE_CATS:
        col = f"{_OHE_PREFIX}{cat}"
        if col not in df.columns:
            df[col] = 0
    return df


def build_features(df: pd.DataFrame, *, for_inference: bool = False) -> pd.DataFrame:
    """
    Deterministic pipeline from ../notebooks/02_feature_engineering.ipynb:
      - drop IDs
      - OHE 'type'
      - log1p(amount)
      - deltaOrig/deltaDest
      - errorOrig/errorDest (+abs)
      - hour/day
      - is_high_value
      - drop raw balance cols + amount + errorOrig/errorDest
      - cast bool->int
      - if for_inference: drop label
    """
    X = df.copy()

    # 1) Drop identifiers if present
    X = X.drop(columns=[c for c in ID_COLS if c in X.columns], errors="ignore")

    # 2) One-hot 'type'
    X = _ohe_type(X)

    # 3) log1p(amount)
    if "amount" in X.columns:
        X["log_amount"] = np.log1p(X["amount"].astype(float))

    # 4) Balance deltas
    if set(RAW_BAL_COLS).issubset(X.columns):
        X["deltaOrig"] = X["oldbalanceOrg"] - X["newbalanceOrig"]
        X["deltaDest"] = X["oldbalanceDest"] - X["newbalanceDest"]

    # 5) Balance errors (+ absolutes)
    if {"amount", *RAW_BAL_COLS}.issubset(X.columns):
        X["errorOrig"] = X["oldbalanceOrg"] - X["amount"] - X["newbalanceOrig"]
        X["errorDest"] = X["oldbalanceDest"] + X["amount"] - X["newbalanceDest"]
        X["abs_errorOrig"] = X["errorOrig"].abs()
        X["abs_errorDest"] = X["errorDest"].abs()

    # 6) Time features from step
    if "step" in X.columns:
        X["hour"] = X["step"] % 24
        X["day"]  = X["step"] // 24

    # 7) High-value flag
    if "amount" in X.columns:
        X["is_high_value"] = (X["amount"] > 200_000).astype("int8")

    # 8) Drop raw balances and redundant columns
    drop_cols = [
        "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
        "amount", "errorOrig", "errorDest"
    ]
    X = X.drop(columns=[c for c in drop_cols if c in X.columns], errors="ignore")

    # 9) Normalize bools -> small ints
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype("int8")

    # 10) Never include label at inference
    if for_inference and LABEL_COL in X.columns:
        X = X.drop(columns=[LABEL_COL])

    return X

def prepare_training(df: pd.DataFrame, label_col: str = LABEL_COL) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Split label, build features, return (X, y, feature_order).
    Keep feature_order to enforce alignment at validation/inference time.
    """
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found")
    y = df[label_col].astype(int)
    X = df.drop(columns=[label_col])
    X = build_features(X, for_inference=False)
    feature_order = X.columns.tolist()
    return X, y, feature_order