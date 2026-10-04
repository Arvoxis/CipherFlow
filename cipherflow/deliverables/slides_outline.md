# Defense Slide Deck — Outline

A ~12-slide deck for the viva/defense. Diagram list at the bottom maps to the patent figures.

1. **Title** — CipherFlow: Self-Supervised Encrypted-Traffic Classification from Flow Shape.
   Your name, course, date. One-line hook: *"Classifying traffic you can't read."*

2. **Problem** — TLS 1.3 / QUIC encrypt payloads → DPI is blind. Malware hides in encryption.
   Labeling traffic is expensive; new apps/malware appear constantly.

3. **Gap in prior work** — payload-byte methods (ET-BERT) need readable bytes; supervised
   flow-statistic models need many labels and generalize poorly. *(One comparison table.)*

4. **Key idea** — learn from **flow shape only** (size, timing, direction), self-supervised,
   then fine-tune on a handful of labels. No payloads, ever.

5. **Flow-shape tokenization (Claim A)** — FIG. 2. Show a real flow → quantized tokens.

6. **Model + pretraining (Claim B)** — FIG. 1 + FIG. 3. Masked-flow-token reconstruction;
   [CLS] = flow embedding. Emphasize it's small (fits a 6 GB laptop GPU).

7. **Label efficiency result** — FIG. 5. macro-F1 vs. labels/class: pretrained ≫ from-scratch
   ≫ classic at low labels. **This is the money slide.**

8. **Downstream tasks** — app-ID (ISCXVPN2016), darknet, malware/C2 detection. Confusion
   matrix. Accuracy/macro-F1 table.

9. **Robustness to evasion (Claim C)** — FIG. 4. Clean vs. padded+jittered accuracy; show the
   augmented model's smaller drop.

10. **Live demo** — screenshot/GIF of the Streamlit dashboard streaming flows → labels.

11. **Novelty & patent** — the three claims in plain English; how CipherFlow differs from
    ET-BERT and supervised baselines. Mention provisional-filing readiness.

12. **Limitations & future work** — synthetic-vs-real gap, more protocols, on-device sensor,
    adversarial-RL attacker co-training, larger pretraining corpus.

---
## Diagrams — most are auto-generated now
Run `python -m cipherflow.deliverables.make_figures` (+ the eval scripts) to produce:
- **FIG. 1** system pipeline — `artifacts/figures/fig1_pipeline.png` ✅ auto
- **FIG. 2** tokenization worked example — `artifacts/figures/fig2_tokenization.png` ✅ auto
- **FIG. 3** masked pretraining schematic — `artifacts/figures/fig3_pretraining.png` ✅ auto
- **FIG. 4** evasion augmentation (before/after) — draw by hand (draw.io/Excalidraw)
- **FIG. 5** label-efficiency curve — `artifacts/fewshot_curve.png` ✅ auto (`eval.run_fewshot`)
- **Embedding map** (Claim B evidence) — `artifacts/embeddings_pca.png` ✅ auto (`eval.embed_viz`)
- **Confusion matrix** — `artifacts/confusion.png` ✅ auto (`eval.metrics confusion`)

*Only FIG. 4 needs manual drawing; everything else is generated from the trained model.*
Live-demo GIF for slide 10: screen-record the Streamlit app's three tabs.
