"""Orchestrate the headline experiment: label-efficiency of CipherFlow vs. baselines.

For each label budget K it trains/evaluates three methods on the SAME data:
  * CipherFlow (pretrained)  — fine-tune the pretrained backbone
  * From-scratch Transformer — same architecture, no pretraining
  * Classic baseline         — best of RandomForest / XGBoost on hand-crafted features

Then it plots macro-F1 vs. K. Pretraining should dominate at small K — the core claim.

Prereqs: run make_synthetic and train_mlm first (see README). Run:
    python -m cipherflow.eval.run_fewshot --ks 5 10 25 50 200
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from cipherflow.eval.metrics import plot_fewshot_curve
from cipherflow.utils import REPO_ROOT, ensure_dir, load_config, resolve_path


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def read_f1(path: Path) -> float:
    return json.loads(path.read_text())["macro_f1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ks", type=int, nargs="+", default=[5, 10, 25, 50, 200], help="labels-per-class budgets to sweep"
    )
    ap.add_argument("--pretrained", default="artifacts/pretrained.pt")
    ap.add_argument("--epochs", type=int, default=None, help="override finetune epochs for speed")
    ap.add_argument(
        "--set",
        dest="overrides",
        nargs="*",
        default=[],
        help="config overrides forwarded to sub-runs (e.g. model.d_model=64) — "
        "must match the pretrained checkpoint's architecture",
    )
    args = ap.parse_args()

    cfg = load_config()
    art = resolve_path(cfg, "paths", "artifacts_dir")
    tmp = ensure_dir(art / "fewshot")
    py = [sys.executable, "-m"]
    # Build a single --set block combining epoch override + forwarded config overrides.
    set_items = list(args.overrides)
    if args.epochs:
        set_items.append(f"finetune.epochs={args.epochs}")
    # Park every sub-run's checkpoint in the scratch dir. Without this the sweep trains 2*len(ks)
    # throwaway models straight over artifacts/classifier.pt, quietly replacing the real
    # classifier with whichever few-shot run finished last.
    set_items.append(f"finetune.ckpt_path={(tmp / 'sweep_ckpt.pt').relative_to(REPO_ROOT).as_posix()}")
    epoch_override = ["--set"] + set_items  # forwarded to the transformer sub-runs

    results = {"CipherFlow (pretrained)": [], "From-scratch": [], "Classic (RF/XGB)": []}
    for k in args.ks:
        m_pre = tmp / f"pre_{k}.json"
        m_scr = tmp / f"scr_{k}.json"
        m_base = tmp / f"base_{k}.json"
        run(
            py
            + [
                "cipherflow.finetune.train_clf",
                "--pretrained",
                args.pretrained,
                "--few-shot",
                str(k),
                "--metrics-out",
                str(m_pre),
            ]
            + epoch_override
        )
        run(
            py
            + [
                "cipherflow.finetune.train_clf",
                "--pretrained",
                "none",
                "--few-shot",
                str(k),
                "--metrics-out",
                str(m_scr),
            ]
            + epoch_override
        )
        run(py + ["cipherflow.finetune.baselines", "--few-shot", str(k), "--metrics-out", str(m_base)])
        results["CipherFlow (pretrained)"].append((k, read_f1(m_pre)))
        results["From-scratch"].append((k, read_f1(m_scr)))
        results["Classic (RF/XGB)"].append((k, read_f1(m_base)))

    out = art / "fewshot_curve.png"
    plot_fewshot_curve(results, out)
    (art / "fewshot_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nDone. Curve -> {out}\nResults -> {art / 'fewshot_results.json'}")


if __name__ == "__main__":
    main()
