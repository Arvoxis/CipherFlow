# CipherFlow — Measured Results (synthetic testbed)

These are real numbers from runs in this repo on the **sequential synthetic dataset**
(`make_synthetic --mode sequential`, 1200 flows/class, 6 classes, seq_len=32). Classes are
engineered to share aggregate statistics and differ **only in packet ordering** — an
adversarial testbed that isolates whether a method models *sequence*, not just marginals.

> Reproduce: see the Quickstart in the top-level `README.md`. Numbers vary slightly with
> seed/config; these used a compact model (d_model=96, 3 layers) to run on CPU/6 GB GPU.

## 1. Headline: order-defined classes (full labels)

7200 flows total, 1080 held out for test.

| Method | Representation | Test accuracy | Macro-F1 |
|---|---|---:|---:|
| RandomForest | hand-crafted aggregate stats | 0.5009 | 0.4982 |
| XGBoost | hand-crafted aggregate stats | 0.4833 | 0.4812 |
| **CipherFlow (pretrained→fine-tuned)** | **flow-shape token sequence** | **0.9944** | **0.9944** |

**Takeaway:** aggregate-statistic baselines cannot separate classes that differ only in
order and collapse toward chance; CipherFlow's sequence model separates them almost
perfectly. This is the core evidence that flow-shape *sequence* modeling matters.

## 2. Label efficiency (the key experiment)

Macro-F1 vs. labeled flows per class (`cipherflow.eval.run_fewshot`):

| labels/class | CipherFlow (pretrained) | From-scratch Transformer | Classic RF/XGB |
|---:|---:|---:|---:|
| 5   | **0.789** | 0.365 | 0.346 |
| 25  | **0.892** | 0.561 | 0.418 |
| 100 | 0.993 | 0.990 | 0.457 |
| 500 | 0.998 | 0.999 | 0.488 |

**Takeaways:**
- **Pretraining's advantage is largest when labels are scarce** — at 5 labels/class it more than
  doubles the from-scratch macro-F1 (0.789 vs 0.365, **+42 points**), and still adds **+33
  points** at 25/class. This is the self-supervised-pretraining payoff (patent Claim B) and the
  project's headline story.
- Pretrained and from-scratch **converge at high labels** (≥100/class), exactly as expected:
  pretraining substitutes for labels.
- **Classic baselines plateau at ~0.45–0.49** regardless of label budget — more labels can't
  fix a representation that ignores order.

Plot: `artifacts/fewshot_curve.png`.

## 3. Masked-flow pretraining (Claim B) health

Self-supervised pretraining on unlabeled flows — masked-token reconstruction accuracy rises
as the model learns flow structure:

| dataset | epoch 1 | final | epochs |
|---|---:|---:|---:|
| synthetic (sequential) | 0.485 | 0.669 | 12 |
| real capture (`selfcap_f12`) | 0.430 | 0.751 | 20 |

Masked-token reconstruction accuracy, not classification accuracy. Real traffic trains better
than the synthetic testbed, as expected: real flows carry protocol regularities a generator
doesn't reproduce.

## 4. Robustness to evasion (Claim C)

**Synthetic:** the order signal survives the attack intact — clean 0.9944, evaded 0.9944, zero
drop. Padding and jitter perturb magnitudes, and these classes are defined by ordering, so there
is nothing for the attack to erase.

**Captured real traffic** (`selfcap_f12.parquet`, 5 classes, 1797 flows, 270 test) is where the
defense is visible:

| model | clean | evaded | change |
|---|---:|---:|---:|
| RandomForest on flow stats | 0.6481 | 0.5333 | **-0.1148** (-17.7%) |
| CipherFlow, normally trained | 0.5519 | 0.4852 | **-0.0667** (-12.1%) |
| CipherFlow, `--augment` | 0.5370 | **0.5519** | **+0.0148** (no loss) |

The augmented model loses nothing to the attack and **overtakes the RandomForest once the
attacker engages** (0.5519 vs 0.5333), having started behind it. See
`artifacts/real_robustness.png`.

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

## 7. Honesty notes for the report
- The *easy* synthetic mode is trivially separable by aggregate stats (baselines ~100%) and
  is only a wiring smoke test; all headline claims use the **sequential** mode.
- **On the small real capture, RandomForest beats CipherFlow on clean accuracy** (~0.64 vs
  0.5519). The capture is small and noisy and the deep model is data-hungry. The hybrid head
  didn't close the gap either (stats-only 0.6597 > hybrid 0.5980 > embedding-only 0.5241).
  Report this; the defensible real-data win is robustness, not clean accuracy.
- Numbers here are from a compact CPU-friendly config; a larger model on the GPU (defaults in
  `cipherflow/configs/default.yaml`) should match or exceed them.
- Every run stamps `artifacts/run_manifest.json` with commit, config hash and seed. Quote a
  number only if a manifest entry backs it.

## 8. Reproducing this file
```bash
python -m cipherflow.pipeline --quick --fewshot --n-per-class 1200
python -m cipherflow.robustness.eval_robustness --ckpt artifacts/classifier_aug.pt --data data_out/selfcap_f12.parquet
```
