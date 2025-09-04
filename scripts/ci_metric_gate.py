#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="Path to eval snapshot JSON")
    ap.add_argument("--metric", default="pr_auc")
    ap.add_argument("--threshold", type=float, default=0.90)
    args = ap.parse_args()

    if not os.path.exists(args.file):
        print(f"[metric-gate] File not found: {args.file}", file=sys.stderr)
        sys.exit(2)

    with open(args.file) as f:
        data = json.load(f)

    # accept top-level or nested under 'metrics'
    metrics = data.get("metrics", data)
    if args.metric not in metrics:
        print(f"[metric-gate] Metric '{args.metric}' missing in {args.file}", file=sys.stderr)
        sys.exit(3)

    val = float(metrics[args.metric])
    print(f"[metric-gate] {args.metric}={val:.5f} (threshold={args.threshold:.2f})")
    if val < args.threshold:
        print("[metric-gate] ❌ Threshold not met — failing CI", file=sys.stderr)
        sys.exit(1)
    print("[metric-gate] ✅ Threshold met")
    return 0


if __name__ == "__main__":
    sys.exit(main())
