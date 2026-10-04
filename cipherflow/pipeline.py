"""One-command end-to-end pipeline — reproduce the whole CipherFlow demo.

Runs, in order: synthetic data -> pretrain -> fine-tune -> baselines -> robustness ->
confusion matrix -> embedding map -> patent figures. Prints a summary of artifacts.

Run:
    python -m cipherflow.pipeline --quick              # small model, fast (CPU-friendly)
    python -m cipherflow.pipeline                      # default config (use a GPU)
    python -m cipherflow.pipeline --quick --fewshot    # also run the label-efficiency sweep
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

from cipherflow.utils import REPO_ROOT, load_config, resolve_path

# Compact architecture used by --quick (matches across pretrain + fine-tune).
QUICK = ["model.d_model=96", "model.n_layers=3", "model.n_heads=3", "model.d_ff=192"]


def run(step: str, cmd: list[str]):
    print(f"\n{'='*70}\n>> {step}\n{'='*70}")
    t0 = time.time()
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}  # keep child prints crash-proof on Windows
    subprocess.run([sys.executable, "-m"] + cmd, check=True, cwd=REPO_ROOT, env=env)
    print(f"  ({step}: {time.time()-t0:.1f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small model + fewer epochs (CPU)")
    ap.add_argument("--fewshot", action="store_true", help="also run the label-efficiency sweep (slow)")
    ap.add_argument("--n-per-class", type=int, default=1200)
    ap.add_argument("--mode", choices=["sequential", "easy"], default="sequential")
    args = ap.parse_args()

    arch = QUICK if args.quick else []
    arch_block = (["--set"] + arch) if arch else []
    pre_epochs = ["--set", "pretrain.epochs=12"] if args.quick else []
    ft_epochs = ["--set", "finetune.epochs=15"] if args.quick else []

    def merge_set(*groups):
        items = []
        for g in groups:
            if g and g[0] == "--set":
                items += g[1:]
        return (["--set"] + items) if items else []

    run("1/8 synthetic data", ["cipherflow.data.make_synthetic", "--mode", args.mode,
                               "--n-per-class", str(args.n_per_class)])
    run("2/8 pretrain (Claim B)", ["cipherflow.pretrain.train_mlm"] + merge_set(arch_block, pre_epochs))
    run("3/8 fine-tune", ["cipherflow.finetune.train_clf", "--pretrained", "artifacts/pretrained.pt"]
        + merge_set(arch_block, ft_epochs))
    run("4/8 baselines", ["cipherflow.finetune.baselines"])
    run("5/8 robustness (Claim C)", ["cipherflow.robustness.eval_robustness", "--ckpt", "artifacts/classifier.pt"])
    run("6/8 confusion matrix", ["cipherflow.eval.metrics", "confusion", "--ckpt", "artifacts/classifier.pt"])
    run("7/8 embedding map", ["cipherflow.eval.embed_viz", "--ckpt", "artifacts/classifier.pt"])
    run("8/8 patent figures", ["cipherflow.deliverables.make_figures"])

    if args.fewshot:
        ks = ["5", "25", "100", "500"]
        run("bonus: few-shot curve", ["cipherflow.eval.run_fewshot", "--ks"] + ks
            + (["--epochs", "40"] if args.quick else []) + merge_set(arch_block))

    art = resolve_path(load_config(), "paths", "artifacts_dir")
    print(f"\n{'='*70}\nDONE. Artifacts in {art}:")
    for name in ["pretrained.pt", "classifier.pt", "confusion.png", "embeddings_pca.png",
                 "fewshot_curve.png", "figures/fig1_pipeline.png", "figures/fig2_tokenization.png",
                 "figures/fig3_pretraining.png"]:
        p = art / name
        print(f"  {'[x]' if p.exists() else '[ ]'} {name}")
    print("Run the live demo (from repo root):  python -m streamlit run cipherflow/serve/app.py")


if __name__ == "__main__":
    main()
