#src/mlops_fraud/deployment/api.py
# --- imports ---
from __future__ import annotations
import os, json, time, logging, threading, shutil, tempfile
from typing import List, Dict, Any, Optional, Tuple, Literal
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

import numpy as np
import pandas as pd
import boto3
import mlflow
from mlflow.tracking import MlflowClient
from mlflow import xgboost as mlf_xgb

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ConfigDict, Field 

from mlops_fraud.features import build_features

from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# ---------- Logging ----------
LOG = logging.getLogger("uvicorn")
LOG.setLevel(logging.INFO)

# ---------- Config ----------
TRACKING       = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5500")
MODEL_NAME     = os.getenv("MODEL_NAME", "fraud_xgb")
MODEL_ALIAS    = os.getenv("MODEL_ALIAS", "staging")
MODEL_STAGE    = os.getenv("MODEL_STAGE", "")
MODEL_VERSION  = os.getenv("MODEL_VERSION", "")
REFRESH_SECS   = int(os.getenv("MODEL_REFRESH_SECS", "120"))
READY_TIMEOUT  = int(os.getenv("MODEL_READY_TIMEOUT_SECS", "10"))
HTTP_TIMEOUT   = int(os.getenv("MLFLOW_HTTP_REQUEST_TIMEOUT", "30"))
ARTIFACTS_URI  = os.getenv("ARTIFACTS_URI", "").rstrip("/")  # e.g. s3://mlops-fraud-dvc
AWS_REGION     = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")

mlflow.set_tracking_uri(TRACKING)
client = MlflowClient()
app = FastAPI(title="Fraud XGB Inference", version="1.0")

# ---------- Validation Errors (422) ----------
@app.exception_handler(RequestValidationError)
async def _validation_handler(request: Request, exc: RequestValidationError):
    detail = [{"loc": e["loc"], "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"error": "Invalid request", "detail": detail})

# ---------- Metrics ----------
REQ_COUNT = Counter("api_requests_total", "Total API requests", ["method", "path", "status"])
REQ_LATENCY = Histogram(
    "api_request_latency_seconds", "Request latency (seconds)",
    ["method", "path", "status"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5],
)

@app.middleware("http")
async def _metrics_middleware(request: Request, call_next):
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
        return response
    finally:
        dt = time.perf_counter() - t0
        method = request.method
        path = getattr(request.scope.get("route"), "path", request.url.path)
        status = str(getattr(locals().get("response", None), "status_code", 500))
        REQ_COUNT.labels(method, path, status).inc()
        REQ_LATENCY.labels(method, path, status).observe(dt)
        # grep-friendly timing line
        LOG.info(f"{method} {path} -> {status} {dt:.4f}s")

@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ---------- Schemas ----------
class Transaction(BaseModel):
    model_config = ConfigDict(extra="forbid")  # reject unknown keys

    type: Literal["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
    amount: float = Field(gt=0)
    step: int = Field(ge=0)
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float
    isFlaggedFraud: int = Field(ge=0, le=1, default=0)

class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rows: List[Transaction] = Field(..., min_length=1, description="List of input rows")

class PredictResponse(BaseModel):
    scores: List[float]
    n: int
    model_ref: str

# ---------- Globals ----------
_model_xgb = None
_model_pyfunc = None
_feature_order: Optional[list] = None
_last_loaded_ts = 0.0
_last_source = ""
_artifacts_dir: Optional[str] = None  # persistent temp folder for loaded model
_lock = threading.RLock()

REQUIRED_MIN_COLS = [
    "type", "amount", "step",
    "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
]

# ---------- Helpers ----------
def _resolve_model_uri_and_runid() -> Tuple[str, Optional[str], str]:
    """
    Returns (models:/name/version, run_id, version_str).
    Always resolves to a numeric version (never alias/stage in the URI).
    """
    LOG.info(f"Resolving model → name={MODEL_NAME}, alias={MODEL_ALIAS}, stage={MODEL_STAGE}, version={MODEL_VERSION}")

    # 1) Alias
    if MODEL_ALIAS:
        try:
            mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
            return f"models:/{MODEL_NAME}/{mv.version}", mv.run_id, str(mv.version)
        except Exception as e:
            LOG.warning(f"Alias resolution failed: {e}")

    # 2) Stage
    if MODEL_STAGE:
        try:
            mvs = client.search_model_versions(f"name='{MODEL_NAME}'")
            staged = [mv for mv in mvs if mv.current_stage and mv.current_stage.lower() == MODEL_STAGE.lower()]
            if staged:
                mv = max(staged, key=lambda m: int(m.version))
                return f"models:/{MODEL_NAME}/{mv.version}", mv.run_id, str(mv.version)
            LOG.warning(f"No versions found at stage={MODEL_STAGE}")
        except Exception as e:
            LOG.warning(f"Stage resolution failed: {e}")

    # 3) Explicit version
    if MODEL_VERSION:
        try:
            mv = client.get_model_version(MODEL_NAME, MODEL_VERSION)
            return f"models:/{MODEL_NAME}/{mv.version}", mv.run_id, str(mv.version)
        except Exception as e:
            LOG.warning(f"Version resolution failed: {e}")

    # 4) Latest version
    mvs = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not mvs:
        raise RuntimeError(f"No registered versions found for model '{MODEL_NAME}'")
    mv = max(mvs, key=lambda m: int(m.version))
    return f"models:/{MODEL_NAME}/{mv.version}", mv.run_id, str(mv.version)

def _experiment_name_for_run(run_id: str) -> str:
    run = client.get_run(run_id)
    exp = client.get_experiment(run.info.experiment_id)
    return exp.name

def _discover_registry_s3_dir(run_id: str) -> str:
    """
    Locate: s3://<bucket>/<optional-prefix>/artifacts/<experiment_name>/models/m-*/artifacts
    Pick latest folder that contains 'MLmodel'.
    """
    if not (ARTIFACTS_URI.startswith("s3://") and AWS_REGION):
        raise RuntimeError("ARTIFACTS_URI must be s3://… and AWS_REGION/DEFAULT_REGION must be set")

    _, rest = ARTIFACTS_URI.split("s3://", 1)
    bucket, *prefix = rest.split("/", 1)
    base_prefix = (prefix[0] + "/") if prefix else ""

    exp_name = _experiment_name_for_run(run_id)
    models_prefix = f"{base_prefix}artifacts/{exp_name}/models/"

    s3 = boto3.client("s3", region_name=AWS_REGION)
    paginator = s3.get_paginator("list_objects_v2")

    newest_key = None
    newest_time = None
    for page in paginator.paginate(Bucket=bucket, Prefix=models_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("artifacts/MLmodel"):
                lm = obj["LastModified"]
                if newest_time is None or lm > newest_time:
                    newest_time = lm
                    newest_key = key

    if not newest_key:
        raise RuntimeError(f"No registry MLmodel found under s3://{bucket}/{models_prefix}")

    return "s3://" + bucket + "/" + newest_key.rsplit("/", 1)[0]  # .../artifacts

def _download_models_uri_with_timeout(uri: str, dst_dir: str) -> str:
    """Download artifacts for models:/... into dst_dir with a timeout; returns dst_dir."""
    def _work():
        return mlflow.artifacts.download_artifacts(uri, dst_path=dst_dir)
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_work)
        fut.result(timeout=HTTP_TIMEOUT)
    return dst_dir

def _prepare_clean_dir() -> str:
    global _artifacts_dir
    # remove previous staged dir
    if _artifacts_dir and os.path.isdir(_artifacts_dir):
        shutil.rmtree(_artifacts_dir, ignore_errors=True)
    _artifacts_dir = tempfile.mkdtemp(prefix="model_artifacts_")
    return _artifacts_dir

def _download_feature_order(run_id: str) -> Optional[list]:
    if not run_id:
        return None
    try:
        with tempfile.TemporaryDirectory() as td:
            p = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/feature_order.json", dst_path=td)
            with open(p, "r") as f:
                return json.load(f)
    except Exception as e:
        LOG.warning(f"No feature_order.json (run={run_id}): {e}")
        return None

# ---------- Loader ----------
def _load_model(force: bool = False):
    global _model_xgb, _model_pyfunc, _feature_order, _last_loaded_ts, _last_source
    with _lock:
        if not force and (time.time() - _last_loaded_ts) < REFRESH_SECS and (_model_xgb or _model_pyfunc):
            return

        uri, run_id, version_str = _resolve_model_uri_and_runid()
        _last_source = uri

        # Honor SKIP_REGISTRY_RESOLVER (or fallback only if resolver fails)
        use_direct = os.getenv("SKIP_REGISTRY_RESOLVER", "0") == "1"
        local_dir = _prepare_clean_dir()

        if use_direct:
            # ---- DIRECT S3 (fast path) ----
            LOG.info("SKIP_REGISTRY_RESOLVER=1 → loading directly from registry S3")
            s3_dir = _discover_registry_s3_dir(run_id)
            LOG.info("Discovered registry S3 dir: %s", s3_dir)
            local_dir = mlflow.artifacts.download_artifacts(s3_dir, dst_path=local_dir)
            LOG.info("Downloaded registry artifacts to %s", local_dir)
        else:
            # ---- TRY REGISTRY RESOLVER, THEN FALL BACK ----
            LOG.info("Loading model via registry resolver: %s (timeout=%ss)", uri, HTTP_TIMEOUT)
            try:
                _download_models_uri_with_timeout(uri, local_dir)
                LOG.info("Downloaded registry artifacts to %s", local_dir)
            except FuturesTimeout:
                LOG.warning("Timed out downloading %s; falling back to direct S3 registry discovery", uri)
                shutil.rmtree(local_dir, ignore_errors=True)
                local_dir = _prepare_clean_dir()
            except Exception as e:
                LOG.warning("Registry download failed (%s); falling back to direct S3 registry discovery", e)
                shutil.rmtree(local_dir, ignore_errors=True)
                local_dir = _prepare_clean_dir()

            if not os.listdir(local_dir):
                s3_dir = _discover_registry_s3_dir(run_id)
                LOG.info("Discovered registry S3 dir: %s", s3_dir)
                local_dir = mlflow.artifacts.download_artifacts(s3_dir, dst_path=local_dir)
                LOG.info("Downloaded registry artifacts to %s", local_dir)

        # ---- Load model from local_dir ----
        try:
            _model_xgb = mlf_xgb.load_model(local_dir)
            _model_pyfunc = None
            LOG.info("Loaded XGBoost flavor from local dir")
        except Exception as e1:
            LOG.warning("XGBoost load failed: %s; trying PyFunc", e1)
            _model_xgb = None
            _model_pyfunc = mlflow.pyfunc.load_model(local_dir)
            LOG.info("Loaded PyFunc flavor from local dir")

        # ---- Feature order from RUN ----
        _feature_order = _download_feature_order(run_id) if run_id else None
        if _feature_order:
            LOG.info("Loaded feature_order.json (%d features)", len(_feature_order))
        else:
            LOG.info("feature_order.json not found; using dataframe column order")

        _last_loaded_ts = time.time()


# ---------- Scoring ----------
def _score(df: pd.DataFrame) -> pd.Series:
    X = build_features(df, for_inference=True)
    if _feature_order:
        X = X.reindex(columns=_feature_order, fill_value=0)
    if _model_xgb is not None:
        proba = _model_xgb.predict_proba(X)
        return pd.Series(np.asarray(proba)[:, 1].ravel(), index=X.index)
    y = np.asarray(_model_pyfunc.predict(X))
    if y.ndim == 2 and y.shape[1] == 2:
        y = y[:, 1]
    return pd.Series(y.ravel(), index=X.index)

# ---------- Lifecycle ----------
def _warmup():
    try:
        _load_model(force=True)
    except Exception as e:
        LOG.warning("Warmup load failed: %s", e)

@app.on_event("startup")
def startup():
    threading.Thread(target=_warmup, daemon=True).start()

# ---------- Endpoints ----------
@app.get("/healthz")
def healthz():
    status = "ready" if (_model_xgb or _model_pyfunc) else "starting"
    return {"status": status, "model_ref": _last_source, "last_loaded_ts": _last_loaded_ts}

def _try_load_with_timeout(seconds: int):
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_load_model, True)
        return fut.result(timeout=seconds)

@app.get("/readyz")
def readyz():
    # if already loaded, don't poke the network
    if _model_xgb or _model_pyfunc:
        return {"status": "ready", "model_ref": _last_source}
    try:
        # prefer direct S3 to keep this quick
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_load_model, True)  # SKIP_REGISTRY_RESOLVER=1 handles direct path
            fut.result(timeout=READY_TIMEOUT)
        return {"status": "ready", "model_ref": _last_source}
    except FuturesTimeout:
        return {"status": "loading-timeout", "model_ref": _last_source}
    except Exception as e:
        return {"status": "error", "detail": str(e), "model_ref": _last_source}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        if not (_model_xgb or _model_pyfunc):
            _load_model(force=True)
        if not (_model_xgb or _model_pyfunc):
            raise HTTPException(status_code=503, detail="Model not ready")

        # Pydantic already validated + forbade unknown keys
        df = pd.DataFrame([r.model_dump() for r in req.rows])

        # (Optional) keep this guard if you want belt-and-suspenders
        missing = [c for c in REQUIRED_MIN_COLS if c not in df.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing required columns: {missing}")

        scores = _score(df)
        return PredictResponse(scores=scores.tolist(), n=len(scores), model_ref=_last_source)
    except HTTPException:
        raise
    except Exception as e:
        LOG.error("Prediction failed", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

@app.post("/reload")
def reload_model():
    _load_model(force=True)
    return {"status": "reloaded", "model_ref": _last_source}
