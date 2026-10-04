"""Task heads that sit on top of the shared FlowFormer backbone.

MLMHead     — predicts the (size_bin, iat_bin, direction) of masked packets (Claim B).
ClassifierHead — predicts the flow's application/threat class from the CLS embedding.

Both are deliberately tiny; the representation power lives in the backbone.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MLMHead(nn.Module):
    """Three parallel classifiers reconstructing each factor of a masked packet token."""

    def __init__(self, d_model: int, size_bins: int, iat_bins: int):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.size_out = nn.Linear(d_model, size_bins)
        self.iat_out = nn.Linear(d_model, iat_bins)
        self.dir_out = nn.Linear(d_model, 2)

    def forward(self, seq_out: torch.Tensor):
        """seq_out: [B, L+1, d]. We drop CLS (index 0) and predict over packet positions."""
        h = self.proj(seq_out[:, 1:])  # [B, L, d]
        return self.size_out(h), self.iat_out(h), self.dir_out(h)


class ClassifierHead(nn.Module):
    """Small MLP over the pooled CLS embedding -> class logits."""

    def __init__(self, d_model: int, n_classes: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.net(pooled)
