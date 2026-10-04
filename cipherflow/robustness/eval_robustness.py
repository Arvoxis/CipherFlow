"""Measure a classifier's robustness to evasion (patent Claim C evidence).

Evaluates a trained classifier on (1) clean test flows and (2) the same flows after an
attacker applies padding + timing jitter. The accuracy DROP quantifies vulnerability;
compare a normally-trained checkpoint vs. one trained with `--augment` to show the defense.

Run:
    python -m cipherflow.robustness.eval_robustness --ckpt artifacts/classifier.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from cipherflow.data.flow_io import load_flows, to_flow_records
from cipherflow.finetune.clf_dataset import encode_labels
from cipherflow.inference import load_classifier, predict_records
from cipherflow.robustness.transforms import evade_records
from cipherflow.utils import get_device, load_config, resolve_path, set_seed


def score(model, head, tok, records, y, device, label_names):
    probs = predict_records(model, head, tok, records, device)
    pred = probs.argmax(-1)
    return accuracy_score(y, pred), f1_score(y, pred, average="macro")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="artifacts/classifier.pt")
    ap.add_argument("--data", type=str, default=None)
    ap.add_argument("--set", dest="overrides", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(overrides=args.overrides)
    set_seed(cfg["seed"])
    device = get_device()

    model, head, tok, label_names, _ = load_classifier(args.ckpt, device)

    data_path = (
        Path(args.data)
        if args.data
        else (resolve_path(cfg, "paths", "data_dir") / "synthetic.parquet")
    )
    df = load_flows(data_path)
    df, y, _ = encode_labels(df, label_names=label_names)
    records = to_flow_records(df)

    # Reproduce the same test split used in training.
    idx = np.arange(len(y))
    _, tmp = train_test_split(idx, test_size=0.3, stratify=y, random_state=cfg["seed"])
    _, te = train_test_split(
        tmp, test_size=0.5, stratify=y[tmp], random_state=cfg["seed"]
    )
    test_records = [records[i] for i in te]
    y_te = y[te]

    clean_acc, clean_f1 = score(
        model, head, tok, test_records, y_te, device, label_names
    )
    evaded = evade_records(
        test_records, cfg["robustness"], cfg["seed"], cfg["flow"]["max_size"]
    )
    ev_acc, ev_f1 = score(model, head, tok, evaded, y_te, device, label_names)

    print(f"\n=== Robustness report: {args.ckpt} ===")
    print(f"  clean   : acc {clean_acc:.4f}  macro-F1 {clean_f1:.4f}")
    print(f"  evaded  : acc {ev_acc:.4f}  macro-F1 {ev_f1:.4f}")
    print(
        f"  DROP    : acc {clean_acc - ev_acc:+.4f}  macro-F1 {clean_f1 - ev_f1:+.4f}  "
        f"(positive = accuracy lost under evasion)"
    )
    print("  (train with --augment to shrink this drop — Claim C.)")


if __name__ == "__main__":
    main()
