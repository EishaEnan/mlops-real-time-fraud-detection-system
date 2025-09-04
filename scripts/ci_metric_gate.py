#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile


def _load_json(p: Path) -> dict:
    with p.open("r") as f:
        return json.load(f)


def _ensure_float(x):
    try:
        return float(x)
    except Exception:
        # We don't need the original traceback here; suppress it explicitly.
        raise SystemExit(f"[metric-gate] ERROR: metric value {x!r} is not numeric") from None


def _fetch_from_mlflow(model_name: str, alias: str, artifact_path: str) -> dict:
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if not uri:
        raise SystemExit("[metric-gate] ERROR: MLFLOW_TRACKING_URI not set for MLflow mode.") from None

    try:
        from mlflow.tracking import MlflowClient
    except Exception as e:
        # Keep the cause so it's easier to diagnose missing package/env issues.
        raise SystemExit(f"[metric-gate] ERROR: mlflow not installed: {e}") from e

    client = MlflowClient()
    try:
        mv = client.get_model_version_by_alias(model_name, alias)
    except Exception as e:
        raise SystemExit(f"[metric-gate] ERROR: cannot resolve model {model_name!r} with alias {alias!r}: {e}") from e

    run_id = mv.run_id
    tmp = tempfile.mkdtemp(prefix="metric_gate_")
    try:
        local = client.download_artifacts(run_id, artifact_path, tmp)
    except Exception as e:
        raise SystemExit(
            f"[metric-gate] ERROR: failed to download artifact {artifact_path!r} from run {run_id}: {e}"
        ) from e

    p = Path(local)
    if p.is_dir():
        p = p / Path(artifact_path).name
    if not p.exists():
        raise SystemExit(f"[metric-gate] ERROR: downloaded artifact missing expected file at {p}") from None
    return _load_json(p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="Path to eval snapshot JSON (if present, used directly)")
    ap.add_argument("--metric", default="pr_auc")
    ap.add_argument("--threshold", type=float, default=0.90)
    ap.add_argument("--model-name", help="Registered model name (fallback if --file missing)")
    ap.add_argument("--alias", default=os.getenv("MODEL_ALIAS", "staging"), help="Model alias")
    ap.add_argument(
        "--artifact-path",
        default=os.getenv("EVAL_ARTIFACT", "eval_snapshot.json"),
        help="Path within run artifacts",
    )
    args = ap.parse_args()

    # 1) Local file path wins if it exists
    if args.file and Path(args.file).exists():
        data = _load_json(Path(args.file))
    else:
        # 2) Fallback to MLflow fetch via model registry
        model_name = args.model_name or os.getenv("MODEL_NAME")
        if not model_name:
            print(
                "[metric-gate] File not found and no model provided. Supply --model-name or set MODEL_NAME.",
                file=sys.stderr,
            )
            return 2
        data = _fetch_from_mlflow(model_name, args.alias, args.artifact_path)

    # Accept top-level or nested under 'metrics'
    metrics = data.get("metrics", data)
    if args.metric not in metrics:
        print(f"[metric-gate] Metric '{args.metric}' missing in eval JSON", file=sys.stderr)
        return 3

    val = _ensure_float(metrics[args.metric])
    print(f"[metric-gate] {args.metric}={val:.6f} (threshold={args.threshold:.6f})")
    if val < args.threshold:
        print("[metric-gate] ❌ Threshold not met — failing CI", file=sys.stderr)
        return 1
    print("[metric-gate] ✅ Threshold met")
    return 0


if __name__ == "__main__":
    sys.exit(main())
