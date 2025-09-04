# tests/test_features.py
import numpy as np
import pandas as pd

from mlops_fraud.features import TYPE_CATS, build_features, prepare_training


def _mk_rows(types):
    rows = []
    for i, t in enumerate(types, start=1):
        rows.append(
            {
                "step": i,
                "type": t,
                "amount": 1000.0 * i,
                "nameOrig": f"C{i}",
                "oldbalanceOrg": 5000.0 * i,
                "newbalanceOrig": 4000.0 * i,
                "nameDest": f"M{i}",
                "oldbalanceDest": 1000.0 * i,
                "newbalanceDest": 1500.0 * i,
                "isFlaggedFraud": 0,
                "isFraud": int(i % 2 == 0),
            }
        )
    return pd.DataFrame(rows)


def test_build_features():
    df = _mk_rows(["PAYMENT"])
    X = build_features(df, for_inference=True)

    # one-hot columns for ALL categories must exist (even if zeros)
    for cat in TYPE_CATS:
        assert f"type_{cat}" in X.columns

    # engineered columns exist
    for col in [
        "log_amount",
        "deltaOrig",
        "deltaDest",
        "abs_errorOrig",
        "abs_errorDest",
        "hour",
        "day",
        "is_high_value",
    ]:
        assert col in X.columns

    # raw balance columns and amount should be dropped
    for col in [
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "amount",
        "errorOrig",
        "errorDest",
    ]:
        assert col not in X.columns

    # label should not be present at inference
    assert "isFraud" not in X.columns

    # quick value sanity checks
    assert np.isclose(X.loc[0, "log_amount"], np.log1p(1000.0))
    assert X.loc[0, "hour"] == df.loc[0, "step"] % 24
    assert X.loc[0, "day"] == df.loc[0, "step"] // 24


def test_training_features():
    # Train sees some types; validation sees a different subset
    df_tr = _mk_rows(["PAYMENT", "TRANSFER", "CASH_OUT", "PAYMENT"])
    Xtr, ytr, feature_order = prepare_training(df_tr)

    # Inference batch with missing categories (e.g., DEBIT, CASH_IN only)
    df_va = _mk_rows(["DEBIT", "CASH_IN"]).drop(columns=["isFraud"])
    Xva = build_features(df_va, for_inference=True).reindex(columns=feature_order, fill_value=0)

    # Columns align 1:1 with the saved training feature order
    assert Xtr.columns.tolist() == feature_order
    assert Xva.columns.tolist() == feature_order

    # Reindexing fills missing categories with zeros
    type_cols = [c for c in feature_order if c.startswith("type_")]
    assert Xva[type_cols].isin([0, 1]).all().all()
