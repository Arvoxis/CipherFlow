"""Load a trained CipherFlow classifier and run predictions on flow records.

Shared by robustness evaluation and the Streamlit demo so there is exactly one code path
from raw flow -> class probabilities.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from cipherflow.model.flowformer import FlowFormer
from cipherflow.model.heads import ClassifierHead
from cipherflow.tokenizer.flow_tokenizer import FlowTokenizer


def load_classifier(ckpt_path: str | Path, device=None):
    """Reconstruct (model, head, tokenizer, label_names, cfg) from a saved classifier."""
    device = device or torch.device("cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["cfg"]
    tok = FlowTokenizer(cfg)
    model = FlowFormer(cfg, ckpt["size_bins"], ckpt["iat_bins"]).to(device)
    n_feats = len(ckpt["feat_mean"]) if ckpt.get("hybrid") else 0
    head = ClassifierHead(model.pool_dim + n_feats, len(ckpt["label_names"]),
                          dropout=cfg["model"]["dropout"]).to(device)
    model.load_state_dict(ckpt["model"])
    head.load_state_dict(ckpt["head"])
    model.eval(); head.eval()
    # Attach hybrid feature scaler so predict/embed can rebuild the exact input.
    model._hybrid = bool(ckpt.get("hybrid", False))
    if model._hybrid:
        model._feat_mean = ckpt["feat_mean"]
        model._feat_std = ckpt["feat_std"]
    return model, head, tok, ckpt["label_names"], cfg


@torch.no_grad()
def _forward_pooled(model, tok, records, device, batch_size):
    """Yield (slice, pooled_feature) for records in batches — matches training pooling.

    For a hybrid checkpoint, appends the standardized statistical features so the classifier
    input matches training exactly.
    """
    enc = tok.encode_batch(records)
    hybrid = getattr(model, "_hybrid", False)
    if hybrid:
        from cipherflow.finetune.features import apply_scaler, compute_features
    for i in range(0, len(records), batch_size):
        sl = slice(i, i + batch_size)
        attn = torch.from_numpy(enc["attn_mask"][sl]).to(device)
        seq_out, _ = model(
            torch.from_numpy(enc["size_ids"][sl]).to(device),
            torch.from_numpy(enc["iat_ids"][sl]).to(device),
            torch.from_numpy(enc["dir_ids"][sl]).to(device),
            attn,
        )
        feat = model.pool(seq_out, attn)
        if hybrid:
            f = apply_scaler(compute_features(records[sl]), model._feat_mean, model._feat_std)
            feat = torch.cat([feat, torch.from_numpy(f).to(device)], dim=-1)
        yield sl, feat


@torch.no_grad()
def predict_records(model, head, tok: FlowTokenizer, records: list[dict], device=None, batch_size: int = 256):
    """Return softmax probabilities [N, n_classes] for a list of flow records."""
    device = device or next(model.parameters()).device
    probs = []
    for _, pooled in _forward_pooled(model, tok, records, device, batch_size):
        probs.append(torch.softmax(head(pooled), dim=-1).cpu().numpy())
    return np.concatenate(probs) if probs else np.zeros((0, head.net[-1].out_features))


@torch.no_grad()
def embed_records(model, tok: FlowTokenizer, records: list[dict], device=None, batch_size: int = 256):
    """Return the [N, d_model] CLS flow embeddings — the transferable representation (Claim B)."""
    device = device or next(model.parameters()).device
    embs = [pooled.cpu().numpy() for _, pooled in _forward_pooled(model, tok, records, device, batch_size)]
    return np.concatenate(embs) if embs else np.zeros((0, model.d_model))
