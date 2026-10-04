"""Inspect flow-shape tokenization (patent Claim A, made concrete).

Given a flow (from a dataset, or a hand-typed example), print the packet-by-packet mapping
    (size, iat, direction)  ->  (size_bin, iat_bin, dir)  ->  composite token id  + readable range
This is the clearest single artifact for explaining Claim A in the report/viva, and it backs
the demo's "Tokenization inspector" tab.

Run:
    python -m cipherflow.tokenizer.inspect                      # a built-in example
    python -m cipherflow.tokenizer.inspect --from-data          # first flow of the dataset
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cipherflow.tokenizer.flow_tokenizer import FlowTokenizer
from cipherflow.utils import load_config, resolve_path


def tokenize_table(tok: FlowTokenizer, sizes, iats, dirs) -> pd.DataFrame:
    """Return a per-packet DataFrame of the tokenization (for CLI print or Streamlit)."""
    enc = tok.encode(sizes, iats, dirs)
    n = int(enc["attn_mask"].sum())
    rows = []
    for i in range(n):
        s, ii, d = int(enc["size_ids"][i]), int(enc["iat_ids"][i]), int(enc["dir_ids"][i])
        rows.append({
            "pkt": i,
            "size_B": int(round(sizes[i])),
            "iat_ms": round(float(iats[i]), 2),
            "dir": "up" if d == 0 else "down",
            "size_bin": s,
            "iat_bin": ii,
            "token_id": int(enc["composite_ids"][i]),
            "token_range": tok.q.describe_bin(s, ii, d),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-data", action="store_true", help="use the first flow of the dataset")
    ap.add_argument("--data", default=None)
    args = ap.parse_args()

    cfg = load_config()
    tok = FlowTokenizer(cfg)

    if args.from_data:
        from cipherflow.data.flow_io import load_flows
        data_path = Path(args.data) if args.data else (resolve_path(cfg, "paths", "data_dir") / "synthetic.parquet")
        df = load_flows(data_path)
        row = df.iloc[0]
        sizes, iats, dirs = row["splt_ps"], row["splt_iat"], row["splt_dir"]
        print(f"Flow label: {row.get('label', '?')}  ({len(sizes)} packets)\n")
    else:
        sizes = [64, 1460, 88, 40, 1460, 120, 1460, 52]
        iats = [0.0, 3.2, 15.1, 800.0, 2.1, 40.0, 1.8, 5.0]
        dirs = [0, 1, 0, 1, 1, 0, 1, 0]
        print("Built-in example flow (a small request followed by large responses):\n")

    table = tokenize_table(tok, sizes, iats, dirs)
    print(table.to_string(index=False))
    print(f"\nComposite vocabulary size: {tok.composite_vocab_size} "
          f"({tok.size_bins} size-bins x {tok.iat_bins} iat-bins x 2 directions + specials)")
    print("This (size,iat,direction) -> discrete-token mapping IS patent Claim A "
          "— note: no payload bytes are read.")


if __name__ == "__main__":
    main()
