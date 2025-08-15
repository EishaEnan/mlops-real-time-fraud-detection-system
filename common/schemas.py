# common/schemas.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any

# -------- Pydantic (API request body) --------
class TransactionRequest(BaseModel):
    rows: List[Dict[str, Any]] = Field(..., description="List of raw feature rows (col->value)")

# -------- Pandera (optional: data checks in training/inference) --------
try:
    import pandera as pa
    import pandera.typing as pat

    # Raw TRAINING rows (label present)
    class TrainingSchema(pa.DataFrameModel):
        step: pat.Series[int] = pa.Field(description="Transaction step")
        type: pat.Series[str] = pa.Field(description="Transaction type")
        amount: pat.Series[float] = pa.Field(description="Amount")
        nameOrig: pat.Series[str]
        oldbalanceOrg: pat.Series[float]
        newbalanceOrig: pat.Series[float]
        nameDest: pat.Series[str]
        oldbalanceDest: pat.Series[float]
        newbalanceDest: pat.Series[float]
        isFlaggedFraud: pat.Series[int] = pa.Field(coerce=True)
        isFraud: pat.Series[int] = pa.Field(coerce=True, description="Label (0/1)")

        class Config:
            strict = False  # allow extra columns

    # Raw INFERENCE rows (no label required)
    class InferenceSchema(pa.DataFrameModel):
        step: pat.Series[int]
        type: pat.Series[str]
        amount: pat.Series[float]
        nameOrig: pat.Series[str]
        oldbalanceOrg: pat.Series[float]
        newbalanceOrig: pat.Series[float]
        nameDest: pat.Series[str]
        oldbalanceDest: pat.Series[float]
        newbalanceDest: pat.Series[float]
        isFlaggedFraud: pat.Series[int] = pa.Field(coerce=True)

        class Config:
            strict = False
except Exception:
    # Pandera not installed in some runtimes (e.g., Lambda). That's fine.
    TrainingSchema = None
    InferenceSchema = None
