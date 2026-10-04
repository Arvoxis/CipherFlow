"""Dataset + masking collator for masked-flow-token pretraining (patent Claim B).

The collator implements BERT-style masking adapted to factorized flow tokens:
  * select `mask_prob` of the *real* (non-pad) packet positions
  * of the selected, 80% are replaced by the learned [MASK] vector, 20% are left intact
    (the model must still predict them — this reduces train/inference mismatch)
  * the model reconstructs (size_bin, iat_bin, direction) at every selected position
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from cipherflow.tokenizer.flow_tokenizer import FlowTokenizer

IGNORE = -100  # cross-entropy ignore_index


class TokenizedFlowDataset(Dataset):
    """Pre-tokenizes a list of flow records into tensors held in memory."""

    def __init__(self, flow_records: list[dict], tokenizer: FlowTokenizer,
                 labels: np.ndarray | None = None, feats: np.ndarray | None = None):
        enc = tokenizer.encode_batch(flow_records)
        self.size_ids = torch.from_numpy(enc["size_ids"])
        self.iat_ids = torch.from_numpy(enc["iat_ids"])
        self.dir_ids = torch.from_numpy(enc["dir_ids"])
        self.attn_mask = torch.from_numpy(enc["attn_mask"])
        self.labels = torch.from_numpy(labels).long() if labels is not None else None
        self.feats = torch.from_numpy(feats).float() if feats is not None else None

    def __len__(self):
        return self.size_ids.shape[0]

    def __getitem__(self, i):
        item = {
            "size_ids": self.size_ids[i],
            "iat_ids": self.iat_ids[i],
            "dir_ids": self.dir_ids[i],
            "attn_mask": self.attn_mask[i],
        }
        if self.labels is not None:
            item["label"] = self.labels[i]
        if self.feats is not None:
            item["feats"] = self.feats[i]
        return item


class MLMCollator:
    """Stack samples and build the masking targets for pretraining."""

    def __init__(self, mask_prob: float = 0.15, seed: int = 0):
        self.mask_prob = mask_prob
        self.g = torch.Generator().manual_seed(seed)

    def __call__(self, batch: list[dict]) -> dict:
        size_ids = torch.stack([b["size_ids"] for b in batch])
        iat_ids = torch.stack([b["iat_ids"] for b in batch])
        dir_ids = torch.stack([b["dir_ids"] for b in batch])
        attn = torch.stack([b["attn_mask"] for b in batch])  # [B,L] 1=real

        # Randomly select positions among real packets.
        prob = torch.rand(size_ids.shape, generator=self.g)
        selected = (prob < self.mask_prob) & (attn == 1)

        # 80% of selected get the [MASK] vector; the rest stay but are still predicted.
        replace = selected & (torch.rand(size_ids.shape, generator=self.g) < 0.8)

        def targets(ids):
            t = ids.clone()
            t[~selected] = IGNORE
            return t

        out = {
            "size_ids": size_ids,
            "iat_ids": iat_ids,
            "dir_ids": dir_ids,
            "attn_mask": attn,
            "mlm_mask": replace,            # positions to feed [MASK]
            "tgt_size": targets(size_ids),
            "tgt_iat": targets(iat_ids),
            "tgt_dir": targets(dir_ids),
        }
        if "label" in batch[0]:
            out["label"] = torch.stack([b["label"] for b in batch])
        return out
