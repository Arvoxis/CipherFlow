"""Extract flow-shape features from real .pcap files using nfstream (Week 1 pipeline).

Produces the SAME parquet schema as the synthetic generator, so every downstream stage is
unchanged. nfstream's SPLT analysis gives us the first-N packet sizes / inter-arrival times
/ directions per flow — exactly CipherFlow's inputs, with zero payload inspection.

Labeling: pass --label explicitly, or let it default to each pcap's filename stem (handy for
per-application datasets like ISCXVPN2016 where the app name is in the filename).

Run:
    python -m cipherflow.data.extract_flows --pcap "raw/*.pcap" --dataset iscx --out data_out/iscx.parquet
"""
from __future__ import annotations

import argparse
import glob
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from cipherflow.data.flow_io import save_flows
from cipherflow.utils import load_config


def _trim(splt_list, n):
    """nfstream pads SPLT arrays with -1; keep only the first n real entries."""
    arr = np.asarray(splt_list, dtype=float)[:n]
    arr = arr[arr >= 0]
    return arr


def extract_file(path: str, label: str, dataset: str, n: int, min_packets: int) -> list[dict]:
    try:
        from nfstream import NFStreamer
    except ImportError as e:
        raise SystemExit(
            "nfstream is required for real pcap extraction. Install with `pip install nfstream`.\n"
            "(You can prototype the whole pipeline without it via cipherflow.data.make_synthetic.)"
        ) from e

    streamer = NFStreamer(source=path, statistical_analysis=True, splt_analysis=n)
    rows = []
    for flow in streamer:
        sizes = _trim(flow.splt_ps, n)
        iats = _trim(flow.splt_piat_ms, n)
        dirs = _trim(flow.splt_direction, n)
        m = min(len(sizes), len(iats), len(dirs))
        if m < min_packets:
            continue
        rows.append({
            "flow_id": uuid.uuid4().hex,
            "splt_ps": sizes[:m].astype(int).tolist(),
            "splt_iat": iats[:m].round(3).tolist(),
            "splt_dir": np.clip(dirs[:m], 0, 1).astype(int).tolist(),
            "n_packets": int(m),
            "label": label,
            "dataset": dataset,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcap", required=True, help="pcap file, glob, or directory")
    ap.add_argument("--label", default=None, help="label for all flows (default: filename stem)")
    ap.add_argument("--dataset", default="custom")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-packets", type=int, default=4)
    ap.add_argument("--set", dest="overrides", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(overrides=args.overrides)
    n = cfg["flow"]["seq_len"]

    p = Path(args.pcap)
    if p.is_dir():
        files = [str(f) for f in p.glob("*.pcap")] + [str(f) for f in p.glob("*.pcapng")]
    else:
        files = glob.glob(args.pcap)
    if not files:
        raise SystemExit(f"no pcap files matched: {args.pcap}")

    all_rows = []
    for f in files:
        label = args.label or Path(f).stem
        rows = extract_file(f, label, args.dataset, n, args.min_packets)
        print(f"  {Path(f).name}: {len(rows)} flows (label='{label}')")
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    save_flows(df, args.out)
    print(f"\nWrote {len(df)} flows -> {args.out}")
    if "label" in df.columns and len(df):
        print(df["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
