"""Measure and plot clean vs. evaded accuracy on real captured traffic (patent Claim C).

Scores three models on the same held-out split, first clean and then under the padding and
jitter attack: a RandomForest over aggregate flow statistics, a conventionally trained
CipherFlow classifier, and a CipherFlow classifier trained with `--augment`. Writes both the
numbers and the figure, so the document and the plot can never drift apart.

Run:
    python -m cipherflow.deliverables.real_robustness
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from cipherflow.data.flow_io import load_flows, to_flow_records
from cipherflow.finetune.clf_dataset import encode_labels
from cipherflow.finetune.features import compute_features
from cipherflow.inference import load_classifier, predict_records
from cipherflow.robustness.transforms import evade_records
from cipherflow.utils import ensure_dir, get_device, load_config, resolve_path, set_seed


def test_split(y, seed):
    """The same stratified split train_clf uses, so these numbers match the CLI evaluation."""
    idx = np.arange(len(y))
    tr, tmp = train_test_split(idx, test_size=0.3, stratify=y, random_state=seed)
    _, te = train_test_split(tmp, test_size=0.5, stratify=y[tmp], random_state=seed)
    return tr, te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_out/selfcap_f12.parquet")
    ap.add_argument("--ckpt", default="artifacts/classifier.pt")
    ap.add_argument("--ckpt-aug", default="artifacts/classifier_aug.pt")
    ap.add_argument("--set", dest="overrides", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(overrides=args.overrides)
    set_seed(cfg["seed"])
    seed, rc = cfg["seed"], cfg["robustness"]
    device = get_device()

    df = load_flows(args.data)
    df, y, _ = encode_labels(df)
    records = to_flow_records(df)
    tr, te = test_split(y, seed)
    evaded = evade_records([records[i] for i in te], rc, seed, cfg["flow"]["max_size"])
    print(f"{len(df)} flows | test split {len(te)} | device {device}")

    results = []

    # RandomForest on aggregate statistics: the bar the deep model has to clear.
    X = compute_features(records)
    rf = RandomForestClassifier(n_estimators=300, random_state=seed).fit(X[tr], y[tr])
    clean = accuracy_score(y[te], rf.predict(X[te]))
    ev = accuracy_score(y[te], rf.predict(compute_features(evaded)))
    results.append(("RandomForest\n(flow statistics)", clean, ev))

    for label, path in [
        ("CipherFlow\n(conventional)", args.ckpt),
        ("CipherFlow\n(evasion-augmented)", args.ckpt_aug),
    ]:
        model, head, tok, names, _ = load_classifier(path, device)
        _, yy, _ = encode_labels(load_flows(args.data), label_names=names)
        c = accuracy_score(
            yy[te], predict_records(model, head, tok, [records[i] for i in te], device).argmax(-1)
        )
        e = accuracy_score(yy[te], predict_records(model, head, tok, evaded, device).argmax(-1))
        results.append((label, c, e))

    art = resolve_path(cfg, "paths", "artifacts_dir")
    ensure_dir(art)
    payload = [
        {
            "model": m.replace("\n", " "),
            "clean": round(c, 4),
            "evaded": round(e, 4),
            "change": round(e - c, 4),
            "relative_loss": round((c - e) / c, 4) if c else 0.0,
        }
        for m, c, e in results
    ]
    (art / "real_robustness.json").write_text(json.dumps(payload, indent=2))
    for r in payload:
        print(
            f"  {r['model']:34s} clean {r['clean']:.4f} -> evaded {r['evaded']:.4f} "
            f"({r['relative_loss']:+.1%})"
        )

    # ---- figure
    labels = [m for m, _, _ in results]
    cleans = [c for _, c, _ in results]
    evs = [e for _, _, e in results]
    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    b1 = ax.bar(x - w / 2, cleans, w, label="Clean traffic", color="#7FB3D5", edgecolor="#2E4A62")
    b2 = ax.bar(
        x + w / 2, evs, w, label="Under padding + jitter attack", color="#2E7D32", edgecolor="#1B4D1F"
    )
    for bars in (b1, b2):
        for bar in bars:
            ax.annotate(
                f"{bar.get_height():.3f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                textcoords="offset points",
                xytext=(0, 3),
                ha="center",
                fontsize=9,
            )
    for i, (_, c, e) in enumerate(results):
        drop = (c - e) / c if c else 0
        ax.annotate(
            "no loss" if drop <= 0 else f"−{drop:.1%}",
            (i, max(c, e) + 0.055),
            ha="center",
            fontsize=9.5,
            color=("#1B5E20" if drop <= 0 else "#B23B3B"),
            weight="bold",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("Accuracy on held-out test split")
    ax.set_ylim(0, max(max(cleans), max(evs)) + 0.17)
    ax.set_title(
        "Evasion robustness on real captured traffic\n(5 activity classes, 1797 flows, 270 held out)",
        fontsize=11,
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out = art / "real_robustness.png"
    fig.savefig(out, dpi=190)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
