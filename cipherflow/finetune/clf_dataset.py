"""Labeled dataset helpers for downstream classification / fine-tuning."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from cipherflow.data.flow_io import UNLABELED


def encode_labels(df: pd.DataFrame, label_names: list[str] | None = None):
    """Map string labels -> contiguous ints. Drops UNLABELED rows.

    Returns (filtered_df, y int array, label_names list). Pass label_names to keep a
    stable class ordering across train/val/test splits.
    """
    df = df[df["label"] != UNLABELED].reset_index(drop=True)
    if label_names is None:
        label_names = sorted(df["label"].unique().tolist())
    idx = {name: i for i, name in enumerate(label_names)}
    y = df["label"].map(idx).to_numpy()
    if np.isnan(y.astype(float)).any():
        unknown = set(df["label"]) - set(label_names)
        raise ValueError(f"labels not in label_names: {unknown}")
    return df, y.astype(np.int64), label_names


def clf_collate(batch: list[dict]) -> dict:
    """Plain stacking collator (no masking) for classification."""
    out = {
        "size_ids": torch.stack([b["size_ids"] for b in batch]),
        "iat_ids": torch.stack([b["iat_ids"] for b in batch]),
        "dir_ids": torch.stack([b["dir_ids"] for b in batch]),
        "attn_mask": torch.stack([b["attn_mask"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
    }
    if "feats" in batch[0]:
        out["feats"] = torch.stack([b["feats"] for b in batch])
    return out


def test_split(y: np.ndarray, seed: int, n: int | None = None) -> np.ndarray:
    """Indices of the held-out test split train_clf created: 70/15/15, stratified.

    Every evaluator has to reproduce this exact split or its accuracy means nothing, so the
    arithmetic lives here once. Pass ``n`` to subsample the split while keeping class balance.
    """
    from sklearn.model_selection import train_test_split

    _, tmp = train_test_split(np.arange(len(y)), test_size=0.3, stratify=y, random_state=seed)
    _, te = train_test_split(tmp, test_size=0.5, stratify=y[tmp], random_state=seed)
    if n is not None and n < len(te):
        te, _ = train_test_split(te, train_size=n, stratify=y[te], random_state=seed)
    return te
