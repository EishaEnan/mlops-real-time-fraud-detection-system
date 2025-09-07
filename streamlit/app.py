# streamlit/app.py
import io
import os
import queue
import threading
import time

import altair as alt
import mlflow
import numpy as np
import pandas as pd
import requests

import streamlit as st

# ---------------- Config ----------------
API_ENV = os.getenv("API_BASE_URL")  # may be None
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
DEFAULT_SAMPLE = "streamlit/data/replay_sample.csv"
GITHUB_REPO_URL = "https://github.com/EishaEnan/mlops-real-time-fraud-detection-system"
PROJECTS_URL = "https://projects.eishaenan.com/"

st.set_page_config(page_title="Fraud Monitor", layout="wide")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# ---------------- Hero / Branding ----------------
st.markdown(
    f"""
    <style>
      .hero {{
        padding: 18px 20px; border-radius: 14px;
        background: linear-gradient(90deg,#eef6ff 0%, #f4fff6 100%);
        border: 1px solid #dfeaff; margin-bottom: 10px;
      }}
      .hero-head {{
        display:flex; align-items:center; justify-content:space-between; gap:12px;
      }}
      .hero h1 {{ margin: 0; font-size: 32px; }}
      .pill {{ display:inline-block; padding:4px 10px; margin-right:8px;
              border-radius:999px; border:1px solid #d0d7ff; background:#f7f9ff; font-size:12px; }}
      .pill-row {{ margin-top:8px; }}
      .linkbar {{ display:flex; gap:8px; flex-wrap:wrap; }}
      .linkbtn {{
        display:inline-block; padding:6px 12px;
        border-radius:10px; border:1px solid #cfd8ff;
        background:#ffffff; color:#1f3b6d; text-decoration:none;
        font-size:12px; font-weight:600;
      }}
      .linkbtn:hover {{ background:#f3f6ff; }}
      .summary {{ padding:14px;border-radius:12px;background:#F6FAFF;border:1px solid #DCEBFF;margin-top:10px; }}
    </style>

    <div class="hero">
      <div class="hero-head">
        <h1>🛡️ Real-Time Fraud Detection System</h1>
        <div class="linkbar">
          <a class="linkbtn" href="{GITHUB_REPO_URL}" target="_blank">🐙 GitHub Repo</a>
          <a class="linkbtn" href="{PROJECTS_URL}" target="_blank">🧰 More Projects</a>
        </div>
      </div>

      <div class="pill-row">
        <span class="pill">FastAPI: Real-time prediction</span>
        <span class="pill">Streamlit: Monitor & Simulator</span>
        <span class="pill">MLflow: Experiments & Registry</span>
        <span class="pill">Airflow: Orchestration</span>
        <span class="pill">DVC: Data/Model Versioning</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------- Helpers ----------------
@st.cache_data(ttl=60)
def fetch_latest_metrics(experiment_name="fraud_train"):
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if not exp:
        return None, None
    runs = client.search_runs(exp.experiment_id, order_by=["attributes.start_time DESC"], max_results=1)
    if not runs:
        return None, None
    r = runs[0]
    return r.data.metrics, r


REQUIRED = ["type", "amount", "step", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]


def _pick_metric(m: dict, *names):
    for n in names:
        if n in m:
            return m[n]
    return None


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        lc = c.lower().strip()
        if lc in ("isfraud", "is_fraud", "fraud", "label", "y"):
            rename[c] = "isFraud"
        elif lc == "isflaggedfraud":
            rename[c] = "isFlaggedFraud"
        elif lc in ("oldbalance_org", "oldbalanceorg"):
            rename[c] = "oldbalanceOrg"
        elif lc in ("newbalance_orig", "newbalanceorig"):
            rename[c] = "newbalanceOrig"
        elif lc in ("oldbalance_dest", "oldbalancedest"):
            rename[c] = "oldbalanceDest"
        elif lc in ("newbalance_dest", "newbalancedest"):
            rename[c] = "newbalanceDest"
    return df.rename(columns=rename) if rename else df


def _as_source_blob(src):
    """Return bytes if src is an UploadedFile; otherwise return the path string."""
    if hasattr(src, "getvalue"):  # Streamlit UploadedFile
        return src.getvalue()
    return src


def _read_csv_any(src):
    if isinstance(src, bytes | bytearray):
        return pd.read_csv(io.BytesIO(src), low_memory=False)
    if hasattr(src, "read"):
        try:
            src.seek(0)
        except Exception:
            pass
        return pd.read_csv(src, low_memory=False)
    return pd.read_csv(src, low_memory=False)


def _rows_and_labels_from_source(src):
    df = _normalize_cols(_read_csv_any(src))
    has_label = "isFraud" in df.columns
    for _, r in df.iterrows():
        row = {
            "type": str(r["type"]),
            "amount": float(r["amount"]),
            "step": int(r["step"]),
            "oldbalanceOrg": float(r["oldbalanceOrg"]),
            "newbalanceOrig": float(r["newbalanceOrig"]),
            "oldbalanceDest": float(r["oldbalanceDest"]),
            "newbalanceDest": float(r["newbalanceDest"]),
            "isFlaggedFraud": int(r.get("isFlaggedFraud", 0)),
        }
        y = int(float(r["isFraud"])) if has_label else None
        yield row, y


def _validate_source(src):
    try:
        head = _normalize_cols(_read_csv_any(src).head(5))
    except Exception as e:
        return False, f"Failed to read CSV: {e}"
    miss = [c for c in REQUIRED if c not in head.columns]
    if "isFraud" not in head.columns:
        return (
            False,
            f"Replay CSV missing label `isFraud`. Found: {list(head.columns)[:20]}{'...' if head.shape[1] > 20 else ''}",
        )
    if miss:
        return (
            False,
            f"Replay CSV missing required columns: {miss}. Found: {list(head.columns)[:20]}{'...' if head.shape[1] > 20 else ''}",
        )
    return True, None


def _api_candidates():
    # order matters; first responsive wins
    return [
        API_ENV,
        "http://api:8080",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://host.docker.internal:8080",
    ]


def _resolve_api(cands):
    tried = []
    for base in [c for c in cands if c]:
        tried.append(base)
        try:
            r = requests.get(f"{base}/readyz", timeout=1.5)
            if r.ok:
                return base, r.json(), tried
        except Exception:
            continue
    return (cands[0] or "http://localhost:8080"), None, tried


def _worker(api_base, eps, secs, src, stop_event, out_q):
    end = time.time() + secs
    it = _rows_and_labels_from_source(src)
    lats, preds, labels = [], [], []
    while time.time() < end and not stop_event.is_set():
        t0 = time.time()
        try:
            row, y_true = next(it)
        except StopIteration:
            break
        try:
            t1 = time.time()
            resp = requests.post(f"{api_base}/predict", json={"rows": [row]}, timeout=3)
            lat_ms = (time.time() - t1) * 1000.0
            if resp.ok:
                sc = resp.json().get("scores", [None])[0]
                lats.append(lat_ms)
                preds.append(sc)
                labels.append(y_true)
                out_q.put(("ok", lat_ms, sc))
            else:
                out_q.put(("err", resp.status_code, resp.text))
        except Exception as ex:
            out_q.put(("err", 0, str(ex)))
        time.sleep(max(0.0, (1.0 / eps) - (time.time() - t0)))
    out_q.put(
        (
            "done",
            len(lats),
            len([p for p in preds if p is not None]),
            float(np.median(lats)) if lats else 0.0,
            preds,
            labels,
        )
    )


# ---------------- State ----------------
if "sim_q" not in st.session_state:
    st.session_state.sim_q = queue.Queue()
if "running" not in st.session_state:
    st.session_state.running = False
if "stop_event" not in st.session_state:
    st.session_state.stop_event = None
if "worker" not in st.session_state:
    st.session_state.worker = None
if "api_base" not in st.session_state:
    st.session_state.api_base = None

# ---------------- Top: Latest MLflow metrics + Ready check ----------------
metrics, run = fetch_latest_metrics("fraud_train")
c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
with c1:
    st.metric("ROC AUC", f"{(metrics or {}).get('roc_auc', 0):.3f}" if metrics else "—")
with c2:
    st.metric("PR AUC", f"{(metrics or {}).get('pr_auc', 0):.3f}" if metrics else "—")
with c3:
    f1_mlflow = _pick_metric(metrics or {}, "f1", "f1_score", "val_f1", "f1_binary", "f1_macro", "f1_micro")
    st.metric("F1", "—" if f1_mlflow is None else f"{f1_mlflow:.3f}")
with c4:
    if st.button("🔄 Check /readyz"):
        base, rz, tried = _resolve_api(_api_candidates())
        st.session_state.api_base = base
        if rz:
            st.success(f"{rz.get('status')} • {rz.get('model_ref', '')}  @ {base}")
        else:
            st.error(f"API not reachable. Tried: {', '.join(tried)}")
    if run:
        st.caption(f"Run: {run.info.run_id} • Start: {pd.to_datetime(run.info.start_time, unit='ms')}")
st.divider()

# ---------------- Controls (upload or bundled sample) ----------------
cL, cR = st.columns([2, 2])
with cL:
    st.markdown("**Traffic**")
    rate = st.slider("Events/sec", 1, 100, 20)
    duration = st.slider("Duration (s)", 5, 80, 20)
with cR:
    st.markdown("**Replay & Decision**")
    uploaded = st.file_uploader("Upload replay CSV", type=["csv"], accept_multiple_files=False)
    use_sample = st.checkbox("Use bundled sample", value=(uploaded is None))
    thresh = st.slider("Threshold (fraud if score ≥)", 0.0, 1.0, 0.5, 0.01)
    with st.expander("📄 Required CSV schema", expanded=False):
        st.markdown("""
        **Columns expected:**
        - `type` (e.g., PAYMENT, TRANSFER, CASH_OUT, …)
        - `amount` *(float)*, `step` *(int)*
        - `oldbalanceOrg`, `newbalanceOrig` *(float)*
        - `oldbalanceDest`, `newbalanceDest` *(float)*
        - `isFraud` *(0/1 label)*
        - *(optional)* `isFlaggedFraud`
        """)

source = uploaded if uploaded is not None else DEFAULT_SAMPLE

# ---------------- Start/Stop ----------------
cStart, cStop, _sp = st.columns([1, 1, 6])
start_btn = cStart.button("▶️ Start", width="content")
stop_btn = cStop.button("⏸️ Stop", width="content")

# Placeholders
status_ph = st.empty()
latency_chart_ph = st.empty()
summary_ph = st.empty()
cm_ph = st.empty()

# ---------------- Start/Stop logic ----------------
if start_btn and not st.session_state.running:
    # 1) Resolve API first; fail fast if unreachable
    base, rz, tried = _resolve_api(_api_candidates())
    st.session_state.api_base = base
    if not rz:
        st.error(f"API not reachable. Tried: {', '.join(tried)}. Start your FastAPI container then click /readyz.")
    else:
        # 2) Validate & start
        chosen_blob = _as_source_blob(source)
        ok, msg = _validate_source(chosen_blob)
        if not ok:
            st.error(msg)
        else:
            summary_ph.empty()
            cm_ph.empty()
            st.session_state.sim_q = queue.Queue()
            st.session_state.stop_event = threading.Event()
            st.session_state.worker = threading.Thread(
                target=_worker,
                args=(base, rate, duration, chosen_blob, st.session_state.stop_event, st.session_state.sim_q),
                daemon=True,
            )
            st.session_state.running = True
            st.session_state.worker.start()

if stop_btn and st.session_state.running:
    st.session_state.stop_event.set()
    st.session_state.running = False

# ---------------- Streaming loop (status + Altair latency chart) ----------------
rows, lats = [], []
preds, labels = [], []
start_ts = time.time()
ROLL_MAX = 500

while st.session_state.running and (time.time() - start_ts) < (duration + 2):
    try:
        kind, a, b, *rest = st.session_state.sim_q.get(timeout=0.2)
    except queue.Empty:
        kind = None

    if kind == "ok":
        lat_ms, score = a, b
        lats.append(lat_ms)
        lats = lats[-ROLL_MAX:]
        rows.append({"ms": round(lat_ms, 2), "score": score})
        rows = rows[-ROLL_MAX:]
    elif kind == "err":
        status_or_0, msg = a, b
        rows.append({"ms": 0, "score": f"ERR {status_or_0}: {str(msg)[:80]}"})
        rows = rows[-ROLL_MAX:]
    elif kind == "done":
        preds = rest[1] if len(rest) >= 2 else []
        labels = rest[2] if len(rest) >= 3 else []
        st.session_state.running = False

    sent_now = len(rows)
    ok_now = sum(1 for r in rows if isinstance(r["score"], int | float | np.floating))
    name = getattr(source, "name", os.path.basename(str(source)))
    base = st.session_state.api_base or "?"
    status_ph.caption(
        f"Streaming from **{name}** → API **{base}** at ~**{rate} eps** for **{duration}s** · sent **{sent_now}** \
            · ok **{ok_now}**"
    )

    if lats:
        base_idx = max(0, len(lats) - 200)
        df_lat = pd.DataFrame({"Event #": np.arange(base_idx + 1, len(lats) + 1), "Latency (ms)": lats[-200:]})
        chart = (
            alt.Chart(df_lat)
            .mark_line()
            .encode(x=alt.X("Event #:Q", title="Event #"), y=alt.Y("Latency (ms):Q", title="Latency (ms)"))
            .properties(height=200, title="Latency (ms) over events")
        )
        latency_chart_ph.altair_chart(chart, use_container_width=True)

# ---------------- End-of-run evaluation ----------------
if preds and any(lbl is not None for lbl in labels):
    y_pred = np.asarray(preds, dtype=float)
    y_true = np.asarray([lbl if lbl is not None else -1 for lbl in labels])
    mask = y_true >= 0
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true):
        y_hat = (y_pred >= thresh).astype(int)
        correct = int((y_hat == y_true).sum())
        total = int(len(y_true))
        acc = (correct / total) if total else 0.0

        summary_ph.markdown(
            f"""
            <div class="summary">
              <div style="font-size:18px;font-weight:600;margin-bottom:4px;">
                Replay evaluation @ threshold {thresh:.2f}
              </div>
              <div style="font-size:28px;font-weight:700;">
                {correct}/{total} correct &nbsp;•&nbsp; {acc * 100:.2f}% accuracy
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        tp = int(((y_hat == 1) & (y_true == 1)).sum())
        tn = int(((y_hat == 0) & (y_true == 0)).sum())
        fp = int(((y_hat == 1) & (y_true == 0)).sum())
        fn = int(((y_hat == 0) & (y_true == 1)).sum())
        cm_ph.dataframe(
            pd.DataFrame([[tn, fp], [fn, tp]], index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]),
            width="content",
            height=140,
        )
