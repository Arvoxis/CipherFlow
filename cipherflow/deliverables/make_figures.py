"""Generate patent/report figures programmatically.

Produces clean schematic drawings that match the claims and the deck outline:
  FIG. 1 — system pipeline (capture -> tokenizer -> encoder -> task heads)
  FIG. 2 — flow-shape tokenization worked example (Claim A)
  FIG. 3 — masked-flow-token pretraining schematic (Claim B)
FIG. 5 (label-efficiency curve) and the confusion matrix are produced by the eval scripts.

Run:
    python -m cipherflow.deliverables.make_figures
Outputs -> artifacts/figures/fig1_pipeline.png, fig2_tokenization.png, fig3_pretraining.png
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from cipherflow.tokenizer.flow_tokenizer import FlowTokenizer
from cipherflow.tokenizer.inspect import tokenize_table
from cipherflow.utils import ensure_dir, load_config, resolve_path

BLUE = "#2b6cb0"; GREY = "#4a5568"; GREEN = "#2f855a"; AMBER = "#b7791f"; LIGHT = "#ebf4ff"


def _box(ax, x, y, w, h, text, fc=LIGHT, ec=BLUE, fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fc, ec=ec, lw=1.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color="#1a202c", wrap=True)


def _arrow(ax, x1, y1, x2, y2, color=GREY):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                                 lw=1.6, color=color))


def fig1_pipeline(out):
    fig, ax = plt.subplots(figsize=(11.6, 3.1)); ax.axis("off"); ax.set_xlim(0, 11.3); ax.set_ylim(0, 3)
    boxes = [
        (0.15, "Encrypted\ntraffic\n(TLS/QUIC)", LIGHT),
        (2.0, "Flow sensor\n(size, IAT,\ndirection)", LIGHT),
        (3.95, "Flow-shape\ntokenizer\n(Claim A)", "#fefcbf"),
        (5.9, "Transformer\nencoder\n(FlowFormer)", "#c6f6d5"),
        (7.85, "[CLS] flow\nembedding", LIGHT),
        (9.55, "App-ID /\nthreat\nlabel", "#fed7d7"),
    ]
    w, h, y = 1.5, 1.5, 0.8
    for i, (x, t, fc) in enumerate(boxes):
        _box(ax, x, y, w, h, t, fc=fc)
        if i:
            _arrow(ax, boxes[i - 1][0] + w, y + h / 2, x, y + h / 2)
    ax.text(6.65, 2.65, "Pretraining (Claim B): masked-flow-token reconstruction on UNLABELED flows",
            ha="center", fontsize=8.5, color=GREEN, style="italic")
    ax.text(5.5, 0.35, "No payload bytes are inspected at any stage.",
            ha="center", fontsize=8.5, color=AMBER, style="italic")
    ax.set_title("FIG. 1 — CipherFlow system pipeline", fontsize=11, loc="left")
    fig.tight_layout(); ensure_dir(out.parent); fig.savefig(out, dpi=140); plt.close(fig)


def fig2_tokenization(out, tok: FlowTokenizer):
    sizes = [64, 1460, 88, 40, 1460, 120]; iats = [0.0, 3.2, 15.1, 800.0, 2.1, 40.0]; dirs = [0, 1, 0, 1, 1, 0]
    tb = tokenize_table(tok, sizes, iats, dirs)
    fig, ax = plt.subplots(figsize=(10.5, 3.3)); ax.axis("off")
    cols = ["pkt", "size_B", "iat_ms", "dir", "size_bin", "iat_bin", "token_id", "token_range"]
    cell = [[str(r[c]) for c in cols] for _, r in tb.iterrows()]
    t = ax.table(cellText=cell, colLabels=cols, loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(8.5); t.scale(1, 1.5)
    for j in range(len(cols)):
        t[0, j].set_facecolor(BLUE); t[0, j].set_text_props(color="white", weight="bold")
    ax.set_title("FIG. 2 — Flow-shape tokenization (Claim A): (size, IAT, direction) → discrete token id",
                 fontsize=10.5, loc="left")
    fig.tight_layout(); ensure_dir(out.parent); fig.savefig(out, dpi=140); plt.close(fig)


def fig3_pretraining(out):
    fig, ax = plt.subplots(figsize=(10.5, 3.6)); ax.axis("off"); ax.set_xlim(0, 10.5); ax.set_ylim(0, 3.6)
    seq = ["t1", "[MASK]", "t3", "t4", "[MASK]", "t6"]
    x0, w, gap, y = 1.4, 1.15, 0.25, 2.2
    centers = []
    for i, tokv in enumerate(seq):
        x = x0 + i * (w + gap)
        masked = tokv == "[MASK]"
        _box(ax, x, y, w, 0.7, tokv, fc="#fed7d7" if masked else LIGHT, ec="#c53030" if masked else BLUE, fs=8.5)
        centers.append(x + w / 2)
    _box(ax, x0, 0.9, centers[-1] + w / 2 - x0, 0.7, "Transformer encoder (FlowFormer)", fc="#c6f6d5", ec=GREEN, fs=9.5)
    for cx in centers:
        _arrow(ax, cx, y, cx, 1.6)
    # reconstruction heads over masked positions
    for i, tokv in enumerate(seq):
        if tokv == "[MASK]":
            cx = centers[i]
            _arrow(ax, cx, 0.9, cx, 0.35, color="#c53030")
            ax.text(cx, 0.2, "predict\nsize/iat/dir", ha="center", va="top", fontsize=7.5, color="#c53030")
    ax.text(centers[0], 3.15, "15% of packet tokens masked; three heads reconstruct their "
            "(size, IAT, direction) bins from context — unlabeled.", fontsize=8.5, color=GREY, va="center")
    ax.set_title("FIG. 3 — Masked-flow-token pretraining (Claim B)", fontsize=11, loc="left")
    fig.tight_layout(); ensure_dir(out.parent); fig.savefig(out, dpi=140); plt.close(fig)


def main():
    cfg = load_config()
    tok = FlowTokenizer(cfg)
    figdir = resolve_path(cfg, "paths", "artifacts_dir") / "figures"
    fig1_pipeline(figdir / "fig1_pipeline.png")
    fig2_tokenization(figdir / "fig2_tokenization.png", tok)
    fig3_pretraining(figdir / "fig3_pretraining.png")
    print(f"Wrote figures -> {figdir}")
    print("  fig1_pipeline.png, fig2_tokenization.png, fig3_pretraining.png")
    print("  (FIG.5 label-efficiency = artifacts/fewshot_curve.png; confusion = artifacts/confusion.png)")


if __name__ == "__main__":
    main()
