"""Plotting + metric helpers: confusion matrix and the few-shot label-efficiency curve.

Can be run directly to render a confusion matrix for a saved classifier:
    python -m cipherflow.eval.metrics confusion --ckpt artifacts/classifier.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

from cipherflow.utils import ensure_dir


def plot_confusion(y_true, y_pred, labels: list[str], out_path: str | Path, title="Confusion matrix"):
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(1.2 * len(labels) + 2, 1.0 * len(labels) + 2))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels,
                yticklabels=labels, ax=ax, vmin=0, vmax=1, cbar=True)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    fig.tight_layout()
    ensure_dir(Path(out_path).parent)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def plot_fewshot_curve(results: dict[str, list[tuple[int, float]]], out_path: str | Path,
                       title="Label efficiency (macro-F1 vs labels/class)"):
    """results: {method_name: [(k, macro_f1), ...]}. k = labels/class (use a large int for 'all')."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, pts in results.items():
        pts = sorted(pts)
        xs = [k for k, _ in pts]; ys = [f for _, f in pts]
        ax.plot(xs, ys, marker="o", label=name)
    ax.set_xscale("log")
    ax.set_xlabel("Labeled flows per class")
    ax.set_ylabel("macro-F1")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    ensure_dir(Path(out_path).parent)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def _confusion_cli(args):
    import torch
    from sklearn.model_selection import train_test_split

    from cipherflow.data.flow_io import load_flows, to_flow_records
    from cipherflow.finetune.clf_dataset import encode_labels
    from cipherflow.inference import load_classifier, predict_records
    from cipherflow.utils import get_device, load_config, resolve_path

    cfg = load_config()
    device = get_device()
    model, head, tok, label_names, _ = load_classifier(args.ckpt, device)
    data_path = Path(args.data) if args.data else (resolve_path(cfg, "paths", "data_dir") / "synthetic.parquet")
    df = load_flows(data_path)
    df, y, _ = encode_labels(df, label_names=label_names)
    records = to_flow_records(df)
    idx = np.arange(len(y))
    _, tmp = train_test_split(idx, test_size=0.3, stratify=y, random_state=cfg["seed"])
    _, te = train_test_split(tmp, test_size=0.5, stratify=y[tmp], random_state=cfg["seed"])
    probs = predict_records(model, head, tok, [records[i] for i in te], device)
    out = resolve_path(cfg, "paths", "artifacts_dir") / "confusion.png"
    plot_confusion(y[te], probs.argmax(-1), label_names, out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("confusion")
    c.add_argument("--ckpt", default="artifacts/classifier.pt")
    c.add_argument("--data", default=None)
    args = ap.parse_args()
    if args.cmd == "confusion":
        _confusion_cli(args)
