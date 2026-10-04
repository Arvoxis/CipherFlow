"""Statistical flow features + standardization for the hybrid classifier.

The hybrid model concatenates CipherFlow's learned flow-shape embedding with these
hand-crafted aggregate features, so the classifier sees both the deep sequential
representation and the aggregate volume/rate/direction signal that tree models exploit.
"""
from __future__ import annotations

import numpy as np

from cipherflow.finetune.baselines import flow_features


def compute_features(records: list[dict]) -> np.ndarray:
    """[N, F] float32 matrix of per-flow statistical features."""
    if not records:
        return np.zeros((0, 14), dtype=np.float32)
    return np.stack([flow_features(r) for r in records]).astype(np.float32)


def fit_scaler(feats: np.ndarray):
    """Return (mean, std) for standardization; std floored to avoid divide-by-zero."""
    mean = feats.mean(0)
    std = feats.std(0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def apply_scaler(feats: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((feats - mean) / std).astype(np.float32)
