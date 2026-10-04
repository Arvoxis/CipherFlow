"""Quantization vocabulary for flow-shape tokens (patent Claim A, primitives).

A packet is described by three observable, payload-free quantities:
  * size      — bytes on the wire (we clip to one MTU)
  * iat       — inter-arrival time since the previous packet, in milliseconds
  * direction — 0 = client->server (upstream), 1 = server->client (downstream)

Sizes and IATs are heavy-tailed, so we quantize them on a **log scale**: most of the
resolution goes to the small/fast packets where application signatures actually live.

We expose both:
  * factorized ids  (size_bin, iat_bin, dir) — used by the model's factorized embedding
  * a single composite token id               — used for demos / classic single-vocab BERT
"""
from __future__ import annotations

import numpy as np

# Reserved special tokens for the *composite* single-id vocabulary. The model's
# factorized path handles PAD/CLS/MASK with dedicated learned vectors instead, so these
# constants matter mainly for the composite/demo view and for anyone building single-vocab BERT.
PAD_ID = 0
MASK_ID = 1
CLS_ID = 2
UNK_ID = 3
N_SPECIAL = 4

DIR_UP = 0
DIR_DOWN = 1
N_DIR = 2


class Quantizer:
    """Log-scale quantizer with fixed, reproducible bin edges."""

    def __init__(self, size_bins: int, iat_bins: int, max_size: int, max_iat_ms: float):
        self.size_bins = size_bins
        self.iat_bins = iat_bins
        self.max_size = max_size
        self.max_iat_ms = max_iat_ms
        # Edges are the RIGHT boundary of each bin (np.searchsorted -> bin index).
        # Sizes: 1..max_size on a log scale.
        self.size_edges = np.logspace(0, np.log10(max_size), size_bins, base=10.0)
        # IATs: 0..max_iat on a log1p scale (handles iat == 0 cleanly).
        self.iat_edges = np.expm1(
            np.linspace(0, np.log1p(max_iat_ms), iat_bins)
        )

    def size_to_bin(self, size: np.ndarray) -> np.ndarray:
        size = np.clip(size, 1, self.max_size)
        return np.clip(np.searchsorted(self.size_edges, size, side="left"),
                       0, self.size_bins - 1).astype(np.int64)

    def iat_to_bin(self, iat_ms: np.ndarray) -> np.ndarray:
        iat_ms = np.clip(iat_ms, 0.0, self.max_iat_ms)
        return np.clip(np.searchsorted(self.iat_edges, iat_ms, side="left"),
                       0, self.iat_bins - 1).astype(np.int64)

    # ---- composite single-id view (for demos / analysis) --------------------
    @property
    def composite_vocab_size(self) -> int:
        return N_SPECIAL + N_DIR * self.size_bins * self.iat_bins

    def composite_id(self, size_bin: np.ndarray, iat_bin: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """Pack (size_bin, iat_bin, dir) into one integer id above the special-token block."""
        return (N_SPECIAL
                + direction * (self.size_bins * self.iat_bins)
                + size_bin * self.iat_bins
                + iat_bin).astype(np.int64)

    def describe_bin(self, size_bin: int, iat_bin: int, direction: int) -> str:
        """Human-readable range for a token — handy for the demo/slides."""
        s_lo = 1 if size_bin == 0 else self.size_edges[size_bin - 1]
        s_hi = self.size_edges[min(size_bin, self.size_bins - 1)]
        t_lo = 0 if iat_bin == 0 else self.iat_edges[iat_bin - 1]
        t_hi = self.iat_edges[min(iat_bin, self.iat_bins - 1)]
        d = "up" if direction == DIR_UP else "down"
        return f"[{d}] size~{s_lo:.0f}-{s_hi:.0f}B  iat~{t_lo:.1f}-{t_hi:.1f}ms"
