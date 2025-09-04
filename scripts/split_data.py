#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd


def split_time(df, label_col, step_col, train, valid, test, seed=42, shuffle_within=False):
    df = df.sort_values(step_col).reset_index(drop=True)
    n = len(df)
    n_train = int(n * train)
    n_valid = int(n * valid)
    tr = df.iloc[:n_train]
    va = df.iloc[n_train:n_train+n_valid]
    te = df.iloc[n_train+n_valid:]
    if shuffle_within:
        tr = tr.sample(frac=1, random_state=seed)
        va = va.sample(frac=1, random_state=seed)
        te = te.sample(frac=1, random_state=seed)
    return tr, va, te

def split_stratified(df, label_col, train, valid, test, seed=42):
    from sklearn.model_selection import train_test_split
    tr, tmp = train_test_split(df, test_size=1-train, stratify=df[label_col], random_state=seed)
    rel = valid / (valid + test)
    va, te = train_test_split(tmp, test_size=1-rel, stratify=tmp[label_col], random_state=seed)
    return tr, va, te

def stats(df, label_col, step_col, name):
    n = len(df)
    pos = int(df[label_col].sum()) if label_col in df.columns else 0
    rate = float(pos) / n if n else 0.0
    return dict(split=name, rows=int(n), positives=pos, rate=rate,
                min_step=int(df[step_col].min()), max_step=int(df[step_col].max()))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/processed/paysim_clean.csv")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--label-col", default="isFraud")
    ap.add_argument("--step-col", default="step")
    ap.add_argument("--method", choices=["time","stratified"], default="time")
    ap.add_argument("--train", type=float, default=0.70)
    ap.add_argument("--valid", type=float, default=0.15)
    ap.add_argument("--test",  type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    assert abs(args.train + args.valid + args.test - 1.0) < 1e-6, "fractions must sum to 1"

    df = pd.read_csv(args.input)
    if args.method == "time":
        tr, va, te = split_time(df, args.label_col, args.step_col, args.train, args.valid, args.test, seed=args.seed)
    else:
        tr, va, te = split_stratified(df, args.label_col, args.train, args.valid, args.test, seed=args.seed)

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    tr.to_csv(out/"train.csv", index=False)
    va.to_csv(out/"valid.csv", index=False)
    te.to_csv(out/"test.csv",  index=False)

    rep = pd.DataFrame([
        stats(tr, args.label_col, args.step_col, "train"),
        stats(va, args.label_col, args.step_col, "valid"),
        stats(te, args.label_col, args.step_col, "test"),
    ])
    rep.to_csv(out/"split_stats.csv", index=False)
    print(rep.to_string(index=False))

if __name__ == "__main__":
    main()
