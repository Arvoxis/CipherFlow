# Datasets — download & prepare

CipherFlow uses **public** encrypted-traffic datasets. Download the raw pcaps/CSVs from the
official sources below, then convert them to CipherFlow's parquet schema with
`cipherflow.data.extract_flows` (for pcaps).

> You do **not** need any of these to develop: `python -m cipherflow.data.make_synthetic`
> produces a schema-identical dataset so the full pipeline runs immediately.

## 1. ISCXVPN2016 — application traffic (primary app-ID task)
- Source: University of New Brunswick, CIC — "VPN-nonVPN dataset (ISCXVPN2016)"
  https://www.unb.ca/cic/datasets/vpn.html
- Format: `.pcap` per application/activity (browsing, chat, streaming, VoIP, file transfer, …).
- Extract (label defaults to the filename stem):
  ```bash
  python -m cipherflow.data.extract_flows --pcap "raw/iscx/*.pcap" --dataset iscx --out data_out/iscx.parquet
  ```

## 2. CIC-Darknet2020 — Tor/VPN + app categories (generalization task)
- Source: CIC — "Darknet 2020"  https://www.unb.ca/cic/datasets/darknet2020.html
- Has pcaps (extract as above) and a labeled CSV of flow features.

## 3. Malware / C2 — threat-detection task (pick ONE)
- **CIC-IDS2017**: https://www.unb.ca/cic/datasets/ids-2017.html (benign + attack pcaps)
- **CTU-13**: https://www.stratosphere.ips.cz/datasets-ctu13 (botnet C2 captures)
- Extract with `extract_flows`, giving each capture a `--label` (e.g. `benign`, `botnet_c2`).

## Combining datasets
Concatenate parquet files (all share the schema) for the **pretraining pool** (labels are
ignored during pretraining) and keep per-task files for fine-tuning:
```python
import pandas as pd
from cipherflow.data.flow_io import load_flows, save_flows
dfs = [load_flows(p) for p in ["data_out/iscx.parquet", "data_out/darknet.parquet"]]
save_flows(pd.concat(dfs, ignore_index=True), "data_out/pretrain_pool.parquet")
```

## Notes
- Terms of use: these datasets are free for research/education — cite the CIC/Stratosphere
  papers in your report.
- `seq_len` (packets/flow) comes from `configs/default.yaml`; extraction respects it.
- No payloads are stored — only size/timing/direction — which is central to the privacy and
  patent story.
