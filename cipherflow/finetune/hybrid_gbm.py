"""Hybrid classifier: CipherFlow self-supervised embeddings + statistical features -> XGBoost.

On small/noisy real traffic, gradient-boosted trees are the strongest classifier, and a
neural head underfits. This hybrid keeps the tree classifier but *augments* the hand-crafted
statistical features with CipherFlow's self-supervised flow-shape embedding. If
[stats + embedding] beats [stats] alone, the learned representation adds real, measurable
value on top of classic features -- an honest, patentable win on real data.

Reports macro-F1 for three feature sets on the SAME split:
    stats only   |   CipherFlow embedding only   |   stats + embedding (HYBRID)

Run:
    python -m cipherflow.finetune.hybrid_gbm --data data_out/selfcap_f12.parquet
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from cipherflow.data.flow_io import load_flows, to_flow_records
from cipherflow.finetune.clf_dataset import encode_labels
from cipherflow.finetune.features import compute_features
from cipherflow.inference import embed_records
from cipherflow.model.flowformer import FlowFormer
from cipherflow.tokenizer.flow_tokenizer import FlowTokenizer
from cipherflow.utils import get_device, load_config, resolve_path, set_seed


def load_backbone(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["cfg"]
    tok = FlowTokenizer(cfg)
    model = FlowFormer(cfg, ckpt["size_bins"], ckpt["iat_bins"]).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    model._hybrid = False  # ensure embed_records returns pure pooled embeddings
    return model, tok


def xgb_or_rf(seed, n_classes):
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.08,
                             subsample=0.9, colsample_bytree=0.9, tree_method="hist",
                             random_state=seed, num_class=n_classes,
                             objective="multi:softmax", eval_metric="mlogloss"), "XGBoost"
    except ImportError:
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=400, random_state=seed, n_jobs=-1), "RandomForest"


def score(clf, Xtr, ytr, Xte, yte):
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    return accuracy_score(yte, pred), f1_score(yte, pred, average="macro")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--backbone", default="artifacts/pretrained.pt",
                    help="self-supervised backbone for embeddings")
    ap.add_argument("--metrics-out", default=None)
    ap.add_argument("--set", dest="overrides", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(overrides=args.overrides)
    set_seed(cfg["seed"])
    device = get_device()

    df = load_flows(args.data)
    df, y, label_names = encode_labels(df)
    records = to_flow_records(df)

    bpath = args.backbone if Path(args.backbone).is_absolute() else str(resolve_path(cfg, "paths", "artifacts_dir").parent / args.backbone)
    model, tok = load_backbone(bpath, device)

    E = embed_records(model, tok, records, device)     # [N, 2*d] self-supervised embedding
    S = compute_features(records)                      # [N, 14] statistical features
    H = np.concatenate([S, E], axis=1)                 # hybrid
    print(f"{len(records)} flows | {len(label_names)} classes | embed dim={E.shape[1]} stats dim={S.shape[1]}")

    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=cfg["seed"])
    results = {}
    for name, X in [("stats_only", S), ("embedding_only", E), ("hybrid_stats+embed", H)]:
        clf, algo = xgb_or_rf(cfg["seed"], len(label_names))
        acc, f1 = score(clf, X[tr], y[tr], X[te], y[te])
        results[name] = {"accuracy": float(acc), "macro_f1": float(f1), "algo": algo}
        print(f"  {name:22s} [{algo}]  acc {acc:.4f}  macro-F1 {f1:.4f}")

    gain = results["hybrid_stats+embed"]["macro_f1"] - results["stats_only"]["macro_f1"]
    print(f"\nHybrid gain over stats-only: {gain:+.4f} macro-F1 "
          f"({'CipherFlow embedding ADDS value' if gain > 0 else 'no gain'})")

    if args.metrics_out:
        Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.metrics_out).write_text(json.dumps(results, indent=2))
        print(f"Wrote metrics -> {args.metrics_out}")


if __name__ == "__main__":
    main()
