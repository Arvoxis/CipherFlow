"""Evasion transforms + augmented dataset (patent Claim C).

Threat model: an attacker who cannot break encryption but reshapes their flow to dodge a
classifier by (a) PADDING packets to larger sizes and (b) adding TIMING JITTER. These are
exactly the cheap, real-world evasions available over TLS/QUIC.

Used two ways:
  * as an ATTACK at eval time (measure accuracy drop)  -> eval_robustness.py
  * as AUGMENTATION at train time (defense)            -> train_clf.py --augment
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from cipherflow.tokenizer.flow_tokenizer import FlowTokenizer


def pad_sizes(sizes: np.ndarray, pad_prob: float, pad_max_frac: float, rng, max_size: int = 1500) -> np.ndarray:
    """Inflate a random subset of packet sizes by up to pad_max_frac (never past MTU)."""
    sizes = sizes.astype(float).copy()
    hit = rng.random(len(sizes)) < pad_prob
    frac = rng.uniform(0.0, pad_max_frac, len(sizes))
    sizes[hit] = np.minimum(sizes[hit] * (1 + frac[hit]), max_size)
    return sizes


def jitter_iats(iats: np.ndarray, jitter_std_ms: float, rng) -> np.ndarray:
    """Add non-negative Gaussian jitter to inter-arrival times."""
    iats = iats.astype(float).copy()
    return np.clip(iats + rng.normal(0.0, jitter_std_ms, len(iats)), 0.0, None)


def augment_record(record: dict, rc: dict, rng, max_size: int = 1500) -> dict:
    sizes = np.asarray(record["splt_ps"], dtype=float)
    iats = np.asarray(record["splt_iat"], dtype=float)
    sizes = pad_sizes(sizes, rc["pad_prob"], rc["pad_max_frac"], rng, max_size)
    iats = jitter_iats(iats, rc["jitter_std_ms"], rng)
    return {"splt_ps": sizes, "splt_iat": iats, "splt_dir": record["splt_dir"]}


class AugmentedFlowDataset(Dataset):
    """Tokenizes raw flow records on the fly, applying evasion augmentation each access.

    Fresh randomness per __getitem__ means every epoch sees different padding/jitter — the
    model learns shape features that survive the attack.
    """

    def __init__(self, records: list[dict], tokenizer: FlowTokenizer, cfg: dict,
                 labels: np.ndarray | None = None, seed: int = 0):
        self.records = records
        self.tok = tokenizer
        self.rc = cfg["robustness"]
        self.max_size = cfg["flow"]["max_size"]
        self.labels = labels
        self.base_seed = seed

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        rng = np.random.default_rng(self.base_seed + i + torch.randint(0, 1 << 30, (1,)).item())
        aug = augment_record(self.records[i], self.rc, rng, self.max_size)
        enc = self.tok.encode(aug["splt_ps"], aug["splt_iat"], aug["splt_dir"])
        item = {
            "size_ids": torch.from_numpy(enc["size_ids"]),
            "iat_ids": torch.from_numpy(enc["iat_ids"]),
            "dir_ids": torch.from_numpy(enc["dir_ids"]),
            "attn_mask": torch.from_numpy(enc["attn_mask"]),
        }
        if self.labels is not None:
            item["label"] = torch.tensor(int(self.labels[i]))
        return item


def evade_records(records: list[dict], rc: dict, seed: int, max_size: int = 1500) -> list[dict]:
    """Deterministically evade a whole list of records (for a fixed evaluation set)."""
    rng = np.random.default_rng(seed)
    return [augment_record(r, rc, rng, max_size) for r in records]
