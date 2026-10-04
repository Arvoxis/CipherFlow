# CipherFlow 🔐

**Self-supervised foundation model for encrypted-traffic classification — from flow *shape*
alone (packet size + inter-arrival time + direction), with zero payload inspection.**

CipherFlow pretrains a small Transformer on **unlabeled** traffic with a masked-flow-token
objective, then fine-tunes on a tiny labeled set to identify applications and detect
malicious (C2) flows. It works on fully-encrypted TLS 1.3 / QUIC where deep packet inspection
is blind, and stays accurate when an attacker pads packets or jitters timing.

> **Why it's novel (patent angle):** prior work like ET-BERT tokenizes *payload bytes*.
> CipherFlow tokenizes *flow shape only* — no payload — so it survives full encryption and
> byte-level obfuscation. See [`cipherflow/deliverables/patent_draft.md`](cipherflow/deliverables/patent_draft.md).

## Architecture

```
pcap ──► extract_flows (nfstream) ──► flows.parquet ──► FlowTokenizer (Claim A)
                                                             │  (size,iat,dir) → tokens
                                                             ▼
                          ┌──────────── FlowFormer encoder (small Transformer) ───────────┐
                          │                                                               │
          Claim B ►  MLM head (masked-token pretraining, unlabeled)      ClassifierHead ◄ fine-tune
                          │                                                               │
                          └────────────► pretrained backbone ────────────────────────────┘
                                                             │
                       Claim C ► padding+jitter augmentation (robust training / eval)
                                                             ▼
                                     Streamlit demo: live flow → label
```

## Setup

**Option A — conda (recommended; easiest GPU):**
```bash
conda create -n ml python=3.11 -y
conda activate ml
pip install -r requirements.txt
# torch CUDA build for the RTX 3050:
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**Option B — venv:**
```bash
python -m venv .venv
.venv\Scripts\activate                      # Windows PowerShell
pip install -r requirements.txt
```

> **Whichever env you pick, install ALL of `requirements.txt` into it** — running with a
> different interpreter that is missing `pyarrow`/`xgboost`/`streamlit` will fail at the first
> parquet write. Verify with: `python -c "import torch,pyarrow,xgboost,streamlit; print('ok')"`.

**GPU (RTX 3050 6 GB):** the CUDA torch build above enables GPU training (auto-detected).
Everything runs on CPU too (just slower). The model is sized to fit 6 GB VRAM.

## One-command demo (everything, reproducible)

```bash
python -m cipherflow.pipeline --quick            # data → pretrain → finetune → baselines →
                                                 # robustness → confusion → embeddings → figures
python -m cipherflow.pipeline --quick --fewshot  # also run the label-efficiency sweep
```
Then launch the live demo: `python -m streamlit run cipherflow/serve/app.py` (from the repo root).

## Quickstart (step by step — runs today, no downloads)

```bash
# 1) generate a synthetic dataset (schema-identical to real extraction)
#    sequential mode = order-defined classes (the headline result); --mode easy = quick smoke test
python -m cipherflow.data.make_synthetic --mode sequential --n-per-class 1500

# 2) self-supervised pretraining (Claim B)  — quick config shown
python -m cipherflow.pretrain.train_mlm --set pretrain.epochs=10

# 3) fine-tune a classifier from the pretrained backbone
python -m cipherflow.finetune.train_clf --pretrained artifacts/pretrained.pt

# 4) headline experiment: label-efficiency curve (pretrained vs scratch vs classic)
python -m cipherflow.eval.run_fewshot --ks 5 10 25 50 200 --epochs 8

# 5) robustness to evasion (Claim C)
python -m cipherflow.finetune.train_clf --augment                 # adversarially-trained model
python -m cipherflow.robustness.eval_robustness --ckpt artifacts/classifier.pt

# 6) visuals: confusion matrix, embedding map (Claim B), patent figures (FIG 1-3)
python -m cipherflow.eval.metrics confusion --ckpt artifacts/classifier.pt
python -m cipherflow.eval.embed_viz --ckpt artifacts/classifier.pt        # embeddings_pca.png
python -m cipherflow.deliverables.make_figures                            # artifacts/figures/

# 7) inspect the tokenization (Claim A, made concrete)
python -m cipherflow.tokenizer.inspect --from-data

# 8) live demo (tabs: live classification, tokenization inspector, embedding map)
python -m streamlit run cipherflow/serve/app.py
```

## Using real data
See [`cipherflow/data/download.md`](cipherflow/data/download.md) for ISCXVPN2016,
CIC-Darknet2020, CIC-IDS2017/CTU-13, then:
```bash
python -m cipherflow.data.extract_flows --pcap "raw/*.pcap" --dataset iscx --out data_out/iscx.parquet
```
All later commands accept `--data data_out/iscx.parquet`.

## Layout
| Path | What |
|---|---|
| `cipherflow/tokenizer/` | Flow-shape tokenizer + quantization (**Claim A**) |
| `cipherflow/model/` | `FlowFormer` encoder + MLM/classifier heads |
| `cipherflow/pretrain/` | Masked-flow-token pretraining (**Claim B**) |
| `cipherflow/finetune/` | Fine-tuning + RandomForest/XGBoost baselines |
| `cipherflow/robustness/` | Evasion augmentation + robustness eval (**Claim C**) |
| `cipherflow/eval/` | Few-shot experiment, plots, **embedding map** |
| `cipherflow/tokenizer/inspect.py` | **Tokenization inspector** (Claim A, CLI + demo) |
| `cipherflow/serve/` | Streamlit live demo (3 tabs) |
| `cipherflow/deliverables/` | Patent draft, results, slide outline, **figure generator** |
| `cipherflow/pipeline.py` | One-command end-to-end orchestrator |
| `configs/default.yaml` | All hyperparameters (override with `--set key=value`) |

## Config overrides
Any config value can be overridden inline, e.g. `--set model.d_model=64 pretrain.epochs=3`.

## Sanity check
```bash
python -m tests.test_smoke        # tokenizer round-trip + tiny forward/backward pass
```
