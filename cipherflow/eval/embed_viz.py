"""Visualize the learned flow representation (patent Claim B evidence).

Projects CipherFlow's [CLS] flow embeddings to 2-D (PCA, optionally t-SNE) and colors by
class. Well-separated clusters demonstrate that the self-supervised + fine-tuned encoder
learns a transferable, discriminative representation of flow shape — a strong demo visual
and supporting evidence for the "transferable flow representation" claim.

Run:
    python -m cipherflow.eval.embed_viz --ckpt artifacts/classifier.pt
    python -m cipherflow.eval.embed_viz --ckpt artifacts/classifier.pt --method tsne --max 1500
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from cipherflow.data.flow_io import load_flows, to_flow_records
from cipherflow.finetune.clf_dataset import encode_labels
from cipherflow.inference import embed_records, load_classifier
from cipherflow.utils import ensure_dir, get_device, load_config, resolve_path


def project(embs: np.ndarray, method: str) -> np.ndarray:
    if method == "tsne":
        from sklearn.manifold import TSNE
        perp = min(30, max(5, len(embs) // 20))
        return TSNE(n_components=2, perplexity=perp, init="pca", random_state=0).fit_transform(embs)
    return PCA(n_components=2, random_state=0).fit_transform(embs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="artifacts/classifier.pt")
    ap.add_argument("--data", default=None)
    ap.add_argument("--method", choices=["pca", "tsne"], default="pca")
    ap.add_argument("--max", type=int, default=2000, help="max flows to plot (subsampled)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config()
    device = get_device()
    model, head, tok, label_names, _ = load_classifier(args.ckpt, device)

    data_path = Path(args.data) if args.data else (resolve_path(cfg, "paths", "data_dir") / "synthetic.parquet")
    df = load_flows(data_path)
    df, y, _ = encode_labels(df, label_names=label_names)

    if len(df) > args.max:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(df), args.max, replace=False)
        df, y = df.iloc[idx].reset_index(drop=True), y[idx]

    embs = embed_records(model, tok, to_flow_records(df), device)
    xy = project(embs, args.method)

    fig, ax = plt.subplots(figsize=(7.5, 6))
    for c, name in enumerate(label_names):
        m = y == c
        ax.scatter(xy[m, 0], xy[m, 1], s=10, alpha=0.6, label=name)
    ax.set_title(f"CipherFlow flow embeddings ({args.method.upper()}) — colored by class")
    ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2")
    ax.legend(markerscale=2, fontsize=8, loc="best")
    fig.tight_layout()

    out = args.out or (resolve_path(cfg, "paths", "artifacts_dir") / f"embeddings_{args.method}.png")
    ensure_dir(Path(out).parent)
    fig.savefig(out, dpi=130)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
