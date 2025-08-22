# tests/test_schemas.py
import importlib
import pandas as pd
import pytest

pa = importlib.util.find_spec("pandera")
pytestmark = pytest.mark.skipif(pa is None, reason="Pandera not installed")

from mlops_fraud.schemas import TrainingSchema

def test_training_schema():
    df = pd.DataFrame([{
        "step": 1, "type": "PAYMENT", "amount": 100.0,
        "nameOrig": "C1", "oldbalanceOrg": 500.0, "newbalanceOrig": 400.0,
        "nameDest": "M1", "oldbalanceDest": 100.0, "newbalanceDest": 200.0,
        "isFlaggedFraud": 0, "isFraud": 0,
    }])
    out = TrainingSchema.validate(df)  # should not raise
    assert len(out) == 1
