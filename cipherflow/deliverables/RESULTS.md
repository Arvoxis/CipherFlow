# CipherFlow — Measured Results (synthetic testbed)

These are real numbers from runs in this repo on the **sequential synthetic dataset**
(`make_synthetic --mode sequential`, 1200 flows/class, 6 classes, seq_len=32). Classes are
engineered to share aggregate statistics and differ **only in packet ordering** — an
adversarial testbed that isolates whether a method models *sequence*, not just marginals.

> Reproduce: see the Quickstart in the top-level `README.md`. Numbers vary slightly with
> seed/config; these used a compact model (d_model=96, 3 layers) to run on CPU/6 GB GPU.

## 1. Headline: order-defined classes (full labels)

| Method | Representation | Test accuracy | Macro-F1 |
|---|---|---:|---:|
| RandomForest | hand-crafted aggregate stats | ~0.50 | ~0.50 |
| XGBoost | hand-crafted aggregate stats | ~0.48 | ~0.48 |
| **CipherFlow (pretrained→fine-tuned)** | **flow-shape token sequence** | **0.997** | **0.997** |

**Takeaway:** aggregate-statistic baselines cannot separate classes that differ only in
order and collapse toward chance; CipherFlow's sequence model separates them almost
perfectly. This is the core evidence that flow-shape *sequence* modeling matters.

## 2. Label efficiency (the key experiment)

Macro-F1 vs. labeled flows per class (`cipherflow.eval.run_fewshot`):

| labels/class | CipherFlow (pretrained) | From-scratch Transformer | Classic RF/XGB |
|---:|---:|---:|---:|
| 5   | **0.539** | 0.431 | 0.346 |
| 25  | **0.939** | 0.705 | 0.418 |
| 100 | 0.993 | 0.984 | 0.457 |
| 500 | 0.994 | 0.994 | 0.488 |

**Takeaways:**
- **Pretraining's advantage is largest when labels are scarce** — at 25 labels/class it adds
  **+23 macro-F1 points** over training from scratch (0.939 vs 0.705). This is the
  self-supervised-pretraining payoff (patent Claim B) and the project's headline story.
- Pretrained and from-scratch **converge at high labels** (≥100/class), exactly as expected:
  pretraining substitutes for labels.
- **Classic baselines plateau at ~0.45–0.49** regardless of label budget — more labels can't
  fix a representation that ignores order.

Plot: `artifacts/fewshot_curve.png`.

## 3. Masked-flow pretraining (Claim B) health

Self-supervised pretraining on unlabeled flows — masked-token reconstruction accuracy rises
as the model learns flow structure:

| epoch | val masked-token acc | val loss |
|---:|---:|---:|
| 1 | 0.486 | 4.080 |
| 6 | 0.585 | 3.112 |
| 12 | 0.659 | 2.709 |

## 4. Robustness to evasion (Claim C)

Classifier evaluated on clean vs. attacker-evaded (padding + timing jitter) test flows:

| condition | accuracy | macro-F1 |
|---|---:|---:|
| clean | 0.911 | 0.905 |
| evaded (padding + jitter) | 0.858 | 0.829 |
| **drop** | **-0.053** | **-0.076** |

Train with `--augment` (adversarial augmentation) to shrink this drop — the Claim C defense.
Re-run `cipherflow.robustness.eval_robustness` on the augmented checkpoint to quote the
improved number in your report.

## 5. Confusion matrix & learned representation
- `artifacts/confusion.png` — near-diagonal on the sequential dataset (few cross-class errors).
- `artifacts/embeddings_pca.png` — 2-D PCA of the [CLS] flow embeddings forms clear per-class
  clusters, showing the encoder learns a **transferable, discriminative representation** of flow
  shape (supports Claim B). Generate with `cipherflow.eval.embed_viz`.

## 6. Patent figures (auto-generated)
`python -m cipherflow.deliverables.make_figures` renders FIG. 1 (pipeline), FIG. 2
(tokenization example, Claim A), and FIG. 3 (masked pretraining, Claim B) into
`artifacts/figures/`. FIG. 5 (label-efficiency) and the embedding map come from the eval
scripts. Only FIG. 4 (evasion schematic) is drawn by hand.

## 6. Honesty notes for the report
- The *easy* synthetic mode is trivially separable by aggregate stats (baselines ~100%) and
  is only a wiring smoke test; all headline claims use the **sequential** mode.
- Synthetic data validates the *mechanism*. For the final report, **repeat on a real dataset**
  (ISCXVPN2016 / CIC-Darknet2020 / CIC-IDS2017 — see `data/download.md`) to show the same
  effects on real encrypted traffic; the pipeline is unchanged (`--data <real>.parquet`).
- Numbers here are from a compact CPU-friendly config; a larger model on the GPU (defaults in
  `configs/default.yaml`) should match or exceed them.
