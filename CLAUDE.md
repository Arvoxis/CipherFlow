# CipherFlow — encrypted-traffic classification from flow shape

Self-supervised Transformer that classifies encrypted network flows using **only** packet
size, inter-arrival time and direction — no payload inspection at all. Works on TLS 1.3 /
QUIC where DPI is blind, and survives padding and timing jitter.

There is a patent angle here (`cipherflow/deliverables/patent_draft.md`): prior work like
ET-BERT tokenizes payload *bytes*; CipherFlow tokenizes flow *shape*. Keep that distinction
intact — any change that starts reading payload bytes breaks the core claim.

## Pipeline

```
pcap ─► extract_pcap (dpkt) ─► flows.parquet ─► FlowTokenizer
     ─► masked-flow-token pretrain (unlabeled) ─► finetune (small labeled set)
     ─► eval / robustness ─► serve
```

## Layout

| Path | What lives there |
|---|---|
| `cipherflow/data/` | Flow extraction (`extract_pcap.py`, dpkt) + synthetic generator |
| `cipherflow/tokenizer/` | Flow-shape tokenization — the novel bit |
| `cipherflow/pretrain/` | Masked-flow-token self-supervised training |
| `cipherflow/finetune/` | Supervised heads: app ID, C2 detection |
| `cipherflow/model/` | Transformer definition |
| `cipherflow/eval/`, `robustness/` | Metrics, plus padding/jitter attack simulations |
| `cipherflow/serve/` | Streamlit demo |
| `cipherflow/pipeline.py`, `inference.py` | Entry points |
| `cipherflow/configs/` | YAML training configs |
| `scripts/capture_traffic.ps1` | PowerShell pcap capture |
| `raw/`, `data_out/`, `artifacts/` | Data and outputs — gitignored |

## Rules

- **6 GB VRAM.** Model is small by design; keep it that way. AMP on, batch size tuned to fit.
  Code auto-detects CUDA and must keep working on CPU.
- Baselines (`scikit-learn`, `xgboost`) are the comparison bar — don't delete them when
  the Transformer wins, the paper needs both numbers.
- Parquet via pyarrow is the data interchange format. Don't reintroduce CSV.
- Real pcaps go through `cipherflow/data/extract_pcap.py` (dpkt). `nfstream` **hangs on Windows**
  when reading a capture, so `extract_flows.py` is kept only as a non-Windows fallback. The
  synthetic generator needs neither — use synthetic data for development.
- `tests/test_smoke.py` should stay runnable without any real data present.
- **Env: conda `ml`** (torch+cu121, pyarrow, nfstream, xgboost, streamlit — all present).
  The leftover `.venv/` directory is stale; ignore it.
- **Not on GitHub yet** — intentional, it's pre-patent.

## Running

```bash
conda activate ml          # in my own shell; tool calls use the env python.exe directly
python -m cipherflow.pipeline --quick            # full synthetic run; add --fewshot for the sweep
python -m cipherflow.pipeline --quick --set model.d_model=64   # config overrides are --set key=value
streamlit run cipherflow/serve/app.py
pytest tests/
```
