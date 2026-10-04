"""Classic ML baselines on hand-crafted flow statistics.

These are what CipherFlow must beat — especially in the few-shot regime. We compute the
kind of aggregate features prior work feeds to RandomForest/XGBoost, so the comparison is
apples-to-apples on the SAME flows, differing only in representation + pretraining.

Run:
    python -m cipherflow.finetune.baselines --few-shot 10
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from cipherflow.data.flow_io import load_flows
from cipherflow.finetune.clf_dataset import encode_labels
from cipherflow.finetune.train_clf import few_shot_subsample
from cipherflow.utils import load_config, resolve_path, set_seed


def flow_features(row) -> np.ndarray:
    """Aggregate statistics from one flow's SPLT arrays (payload-free, like CipherFlow)."""
    sizes = np.asarray(row["splt_ps"], dtype=float)
    iats = np.asarray(row["splt_iat"], dtype=float)
    dirs = np.asarray(row["splt_dir"], dtype=float)
    if sizes.size == 0:
        return np.zeros(14)
    up = sizes[dirs == 0]; down = sizes[dirs == 1]
    return np.array([
        sizes.mean(), sizes.std(), sizes.min(), sizes.max(), np.median(sizes),
        iats.mean(), iats.std(), iats.max(),
        dirs.mean(),                         # downstream fraction
        sizes.sum(), len(sizes),
        up.mean() if up.size else 0.0,
        down.mean() if down.size else 0.0,
        (down.sum() / (up.sum() + 1e-6)),    # down/up byte ratio
    ])


def build_matrix(df) -> np.ndarray:
    return np.stack([flow_features(r) for _, r in df.iterrows()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=None)
    ap.add_argument("--few-shot", type=int, default=None)
    ap.add_argument("--metrics-out", type=str, default=None, help="write best baseline metrics json")
    ap.add_argument("--set", dest="overrides", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(overrides=args.overrides)
    set_seed(cfg["seed"])
    data_path = Path(args.data) if args.data else (resolve_path(cfg, "paths", "data_dir") / "synthetic.parquet")
    df = load_flows(data_path)
    df, y, label_names = encode_labels(df)
    X = build_matrix(df)

    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.15, stratify=y, random_state=cfg["seed"])
    if args.few_shot:
        tr = tr[few_shot_subsample(y[tr], args.few_shot, cfg["seed"])]
    print(f"baselines | train={len(tr)} test={len(te)} classes={len(label_names)}"
          + (f" (few-shot={args.few_shot}/class)" if args.few_shot else ""))

    models = {"RandomForest": RandomForestClassifier(n_estimators=300, random_state=cfg["seed"], n_jobs=-1)}
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9, tree_method="hist",
            random_state=cfg["seed"], num_class=len(label_names),
            objective="multi:softmax", eval_metric="mlogloss",
        )
    except ImportError:
        print("(xgboost not installed — skipping)")

    best = {"macro_f1": -1.0, "method": None}
    for name, clf in models.items():
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        f1 = f1_score(y[te], pred, average="macro")
        print(f"  {name:13s} acc {accuracy_score(y[te], pred):.4f} macro-F1 {f1:.4f}")
        if f1 > best["macro_f1"]:
            best = {"macro_f1": float(f1), "method": name}

    if args.metrics_out:
        import json
        best["few_shot"] = args.few_shot
        Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.metrics_out).write_text(json.dumps(best, indent=2))
        print(f"Wrote metrics -> {args.metrics_out}")


if __name__ == "__main__":
    main()
