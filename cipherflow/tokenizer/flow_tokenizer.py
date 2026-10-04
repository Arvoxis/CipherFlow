"""FlowTokenizer — turn a raw flow (SPLT arrays) into fixed-length token tensors.

*** This is the concrete implementation of patent Claim A. ***

Input to `encode` is the first-N-packet "SPLT" view of one flow:
    sizes : sequence of packet sizes (bytes)          e.g. [120, 1460, 88, ...]
    iats  : inter-arrival times (ms)                  e.g. [0.0, 3.2, 15.1, ...]
    dirs  : direction per packet (0 up / 1 down)      e.g. [0, 1, 0, ...]

Output is a dict of length-`seq_len` int arrays plus an attention mask:
    size_ids, iat_ids, dir_ids  — factorized category ids for the model embedding
    attn_mask                   — 1 for real packets, 0 for padding
    composite_ids               — single-id view (demo / classic BERT / analysis)

Flows shorter than seq_len are padded; longer flows are truncated to the first seq_len
packets (application signatures live in the handshake / early packets).
"""
from __future__ import annotations

import numpy as np

from cipherflow.tokenizer.vocab import PAD_ID, Quantizer


class FlowTokenizer:
    def __init__(self, cfg: dict):
        f = cfg["flow"]
        self.seq_len = f["seq_len"]
        self.q = Quantizer(
            size_bins=f["size_bins"],
            iat_bins=f["iat_bins"],
            max_size=f["max_size"],
            max_iat_ms=f["max_iat_ms"],
        )

    @property
    def size_bins(self) -> int:
        return self.q.size_bins

    @property
    def iat_bins(self) -> int:
        return self.q.iat_bins

    @property
    def composite_vocab_size(self) -> int:
        return self.q.composite_vocab_size

    def _fit_length(self, arr: np.ndarray, pad_value) -> np.ndarray:
        """Truncate or pad a 1-D array to exactly seq_len."""
        arr = np.asarray(arr).ravel()
        if len(arr) >= self.seq_len:
            return arr[: self.seq_len]
        out = np.full(self.seq_len, pad_value, dtype=arr.dtype if arr.size else np.float64)
        out[: len(arr)] = arr
        return out

    def encode(self, sizes, iats, dirs) -> dict[str, np.ndarray]:
        """Encode a single flow. Returns dict of np.int64 arrays, all length seq_len."""
        n = min(len(sizes), self.seq_len)

        sizes = self._fit_length(np.asarray(sizes, dtype=np.float64), 0)
        iats = self._fit_length(np.asarray(iats, dtype=np.float64), 0.0)
        dirs = self._fit_length(np.asarray(dirs, dtype=np.int64), 0)

        size_ids = self.q.size_to_bin(sizes)
        iat_ids = self.q.iat_to_bin(iats)
        dir_ids = np.clip(dirs, 0, 1).astype(np.int64)

        attn_mask = np.zeros(self.seq_len, dtype=np.int64)
        attn_mask[:n] = 1

        # Zero-out (as PAD) everything past the real packets so padded content is inert.
        size_ids[n:] = 0
        iat_ids[n:] = 0
        dir_ids[n:] = 0

        composite = self.q.composite_id(size_ids, iat_ids, dir_ids)
        composite[attn_mask == 0] = PAD_ID

        return {
            "size_ids": size_ids,
            "iat_ids": iat_ids,
            "dir_ids": dir_ids,
            "attn_mask": attn_mask,
            "composite_ids": composite,
        }

    def encode_batch(self, flows: list[dict]) -> dict[str, np.ndarray]:
        """Encode many flows (each a dict with 'splt_ps','splt_iat','splt_dir')."""
        keys = ["size_ids", "iat_ids", "dir_ids", "attn_mask", "composite_ids"]
        out = {k: [] for k in keys}
        for fl in flows:
            enc = self.encode(fl["splt_ps"], fl["splt_iat"], fl["splt_dir"])
            for k in keys:
                out[k].append(enc[k])
        return {k: np.stack(v) for k, v in out.items()}
