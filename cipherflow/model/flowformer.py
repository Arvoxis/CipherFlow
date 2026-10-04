"""FlowFormer — a small Transformer encoder over flow-shape tokens.

Design choices driven by the 6 GB VRAM budget:
  * factorized embedding (size + iat + direction) instead of one huge vocab -> tiny params
  * 4 layers x 128 dim by default (~1-2M params) -> pretrains in hours on an RTX 3050
  * a learned [CLS] vector is prepended; its output is the flow embedding used downstream
  * a learned [MASK] vector replaces masked positions during pretraining (Claim B)

Forward returns both the full sequence output and the pooled CLS embedding so the same
backbone serves masked-token pretraining and downstream classification.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FlowFormer(nn.Module):
    def __init__(self, cfg: dict, size_bins: int, iat_bins: int):
        super().__init__()
        m = cfg["model"]
        d = m["d_model"]
        self.d_model = d
        self.seq_len = cfg["flow"]["seq_len"]
        self.size_bins = size_bins
        self.iat_bins = iat_bins

        # --- factorized token embedding ---
        self.size_emb = nn.Embedding(size_bins, d)
        self.iat_emb = nn.Embedding(iat_bins, d)
        self.dir_emb = nn.Embedding(2, d)
        # positional embedding covers CLS (index 0) + seq_len packet positions
        self.pos_emb = nn.Embedding(self.seq_len + 1, d)

        # learned special vectors
        self.cls_vec = nn.Parameter(torch.zeros(1, 1, d))
        self.mask_vec = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.normal_(self.cls_vec, std=0.02)
        nn.init.normal_(self.mask_vec, std=0.02)

        self.emb_dropout = nn.Dropout(m["dropout"])

        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=m["n_heads"],
            dim_feedforward=m["d_ff"],
            dropout=m["dropout"],
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-LN: more stable for small models
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=m["n_layers"], enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)

    def embed_tokens(self, size_ids, iat_ids, dir_ids):
        """Sum the three factor embeddings -> per-packet token embedding [B, L, d]."""
        return self.size_emb(size_ids) + self.iat_emb(iat_ids) + self.dir_emb(dir_ids)

    def forward(self, size_ids, iat_ids, dir_ids, attn_mask, mlm_mask=None):
        """
        size_ids/iat_ids/dir_ids : [B, L] int64
        attn_mask                : [B, L] int64, 1=real packet, 0=pad
        mlm_mask                 : [B, L] bool, True where the position is masked (optional)
        Returns:
            seq_out : [B, L+1, d]  (index 0 is CLS, 1..L are packets)
            pooled  : [B, d]       (CLS output — the flow embedding)
        """
        B, L = size_ids.shape
        tok = self.embed_tokens(size_ids, iat_ids, dir_ids)  # [B, L, d]

        if mlm_mask is not None:
            # Replace masked packet embeddings with the learned mask vector.
            mask_vec = self.mask_vec.expand(B, L, self.d_model)
            tok = torch.where(mlm_mask.unsqueeze(-1), mask_vec, tok)

        # Prepend CLS.
        cls = self.cls_vec.expand(B, 1, self.d_model)
        x = torch.cat([cls, tok], dim=1)  # [B, L+1, d]

        # Positional embeddings.
        pos_ids = torch.arange(L + 1, device=x.device).unsqueeze(0)
        x = x + self.pos_emb(pos_ids)
        x = self.emb_dropout(x)

        # key_padding_mask: True where the model should IGNORE (pad). CLS is always valid.
        cls_valid = torch.ones(B, 1, dtype=attn_mask.dtype, device=attn_mask.device)
        valid = torch.cat([cls_valid, attn_mask], dim=1)  # [B, L+1], 1=valid
        key_padding_mask = valid == 0

        seq_out = self.encoder(x, src_key_padding_mask=key_padding_mask)
        seq_out = self.norm(seq_out)
        pooled = seq_out[:, 0]  # CLS
        return seq_out, pooled

    def pool(self, seq_out, attn_mask):
        """Classification feature = concat([CLS, masked-mean over real packets]) -> [B, 2*d].

        CLS captures a learned summary; the masked mean captures aggregate flow statistics
        (volume/rate/direction balance) where real activity classes differ. Combining both
        beats CLS-only pooling on noisy real traffic.
        """
        cls = seq_out[:, 0]                                  # [B, d]
        tok = seq_out[:, 1:]                                 # [B, L, d]
        m = attn_mask.unsqueeze(-1).to(tok.dtype)            # [B, L, 1]
        mean = (tok * m).sum(1) / m.sum(1).clamp(min=1.0)    # [B, d]
        return torch.cat([cls, mean], dim=-1)                # [B, 2d]

    @property
    def pool_dim(self) -> int:
        return self.d_model * 2


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
