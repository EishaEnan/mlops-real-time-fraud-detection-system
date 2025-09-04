# tests/test_feature_alignment.py
import numpy as np
import pandas as pd

from mlops_fraud.features import build_features


def _row(t):
    # Minimal raw row with required fields; label intentionally absent (inference mode)
    return {
        "type": t,
        "amount": 123.0,
        "step": 1,
        "nameOrig": "C1",
        "oldbalanceOrg": 1000.0,
        "newbalanceOrig": 900.0,
        "nameDest": "M1",
        "oldbalanceDest": 500.0,
        "newbalanceDest": 600.0,
        "isFlaggedFraud": 0,
    }


def _score_linear(X: pd.DataFrame, w: np.ndarray, b: float = 0.0) -> np.ndarray:
    """Tiny, deterministic scorer: sigmoid(W·x + b). Used to verify alignment only."""
    z = X.values @ w + b
    return 1.0 / (1.0 + np.exp(-z))


def test_feature_reindex_preserves_scores_on_permutation():
    """
    Feature alignment contract:
    Reindexing to the saved feature_order must yield identical scores
    even if incoming columns are permuted.
    """
    # Build deterministic features for two rows
    df = pd.DataFrame([_row("PAYMENT"), _row("TRANSFER")])
    X = build_features(df, for_inference=True)

    # The canonical order captured at train time (simulated here)
    feature_order = X.columns.tolist()

    # Make a fixed random weight vector so the test is stable
    rng = np.random.default_rng(42)
    w = rng.normal(size=len(feature_order))

    # Reference scores on the canonical column order
    y_ref = _score_linear(X, w)

    # --- Simulate a scrambled inference payload ---
    perm = rng.permutation(feature_order)  # a shuffled column order
    X_perm = X[perm]  # mis-ordered features as they might arrive

    # Critical step: reindex back to the canonical training order
    X_fixed = X_perm.reindex(columns=feature_order, fill_value=0)

    # Scores must be identical after reindexing (alignment works)
    y_fixed = _score_linear(X_fixed, w)
    assert np.allclose(y_ref, y_fixed)
