"""Canonical on-disk format for flows + load/save helpers.

One parquet row == one flow. Columns:
    flow_id   : str
    splt_ps   : list[int]    packet sizes (bytes), first-N packets
    splt_iat  : list[float]  inter-arrival times (ms)
    splt_dir  : list[int]    direction per packet (0 up / 1 down)
    n_packets : int          number of real packets (<= len of the SPLT lists)
    label     : str          class label ("unlabeled" for pretraining-only flows)
    dataset   : str          source dataset name

Both the synthetic generator and the nfstream extractor emit this exact schema, so every
downstream stage (tokenizer, pretrain, finetune, demo) reads one format.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

COLUMNS = ["flow_id", "splt_ps", "splt_iat", "splt_dir", "n_packets", "label", "dataset"]
UNLABELED = "unlabeled"


def save_flows(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"flow dataframe missing columns: {missing}")
    df[COLUMNS].to_parquet(path, index=False)
    return path


def load_flows(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # Ensure list columns come back as python lists (pyarrow may give numpy arrays).
    for col in ("splt_ps", "splt_iat", "splt_dir"):
        df[col] = df[col].apply(list)
    return df


def to_flow_records(df: pd.DataFrame) -> list[dict]:
    """Rows -> list of dicts consumed by FlowTokenizer.encode_batch."""
    return df[["splt_ps", "splt_iat", "splt_dir"]].to_dict("records")
