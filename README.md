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
pcap ──► extract_pcap (dpkt) ──► flows.parquet ──► FlowTokenizer (Claim A)
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

```bash
conda create -n ml python=3.11 -y
conda activate ml
# torch CUDA build first, so nothing else drags in a CPU wheel over it:
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[dev]"
```

The editable install is what makes `python -m cipherflow.*` work from any directory instead
of only from the repo root. Verify the env in one line:

```bash
python -c "import torch,pyarrow,xgboost,streamlit,dpkt; print('ok', torch.cuda.is_available())"
```

**GPU (RTX 3050 6 GB):** the CUDA wheel above enables GPU training, auto-detected. Everything
runs on CPU too, just slower, and the model is sized to fit 6 GB.

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
CIC-Darknet2020, CIC-IDS2017/CTU-13. Extraction goes through dpkt:
```bash
python -m cipherflow.data.extract_pcap --pcap "raw/*.pcap" --dataset iscx --out data_out/iscx.parquet
python -m cipherflow.data.extract_pcap --dir raw/mycapture --out data_out/selfcap.parquet   # label = file stem
```
All later commands accept `--data data_out/iscx.parquet`.

> `nfstream` hangs on Windows when reading a capture, so `extract_flows.py` is kept only as a
> non-Windows fallback. Use `extract_pcap.py`.

Capturing your own traffic (Windows, needs Wireshark's `dumpcap` on PATH):
```powershell
.\scripts\capture_traffic.ps1
```

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
| `cipherflow/serve/` | Streamlit live demo (5 tabs) |
| `cipherflow/deliverables/` | Patent draft, results, slide outline, **figure generator** |
| `cipherflow/pipeline.py` | One-command end-to-end orchestrator |
| `cipherflow/provenance.py` | Run stamping: commit + config hash + seed per artifact |
| `cipherflow/configs/default.yaml` | All hyperparameters (override with `--set key=value`) |

## Config overrides
Any config value can be overridden inline on any entry point, the pipeline included:
```bash
python -m cipherflow.pipeline --quick --set model.d_model=64 pretrain.epochs=3
```

## Provenance
Every pipeline run appends a record to `artifacts/run_manifest.json`: UTC timestamp, git
commit, dirty flag, config hash, seed, device and package versions. That is the dated
reduction-to-practice trail behind the claims, and it makes "reproduce table 2" mechanical.

```bash
python -m cipherflow.provenance        # print the current stamp
```

## Sanity check
```bash
pytest -q                         # tokenizer round-trip + tiny forward/backward pass
ruff check cipherflow tests
```
CI (`.github/workflows/ci.yml`) runs the same two plus a tiny end-to-end pipeline on every push.
