"""End-to-end smoke test: tokenizer -> model -> MLM loss/backward -> classifier.

Runs in seconds on CPU, needs no data files. Use as a wiring check before long runs:
    python -m tests.test_smoke      (prints PASS/FAIL)
    pytest tests/test_smoke.py      (as a unit test)
"""
from __future__ import annotations

import numpy as np

from cipherflow.utils import load_config, set_seed


def _tiny_cfg():
    cfg = load_config()
    # shrink for speed
    cfg["model"].update(d_model=32, n_layers=2, n_heads=2, d_ff=64)
    cfg["flow"]["seq_len"] = 16
    return cfg


def test_tokenizer_shapes():
    from cipherflow.tokenizer.flow_tokenizer import FlowTokenizer
    cfg = _tiny_cfg()
    tok = FlowTokenizer(cfg)
    enc = tok.encode([120, 1460, 88, 40], [0.0, 3.2, 15.1, 200.0], [0, 1, 0, 1])
    L = cfg["flow"]["seq_len"]
    for k in ("size_ids", "iat_ids", "dir_ids", "attn_mask", "composite_ids"):
        assert enc[k].shape == (L,), (k, enc[k].shape)
    assert enc["attn_mask"].sum() == 4  # four real packets
    assert enc["size_ids"].max() < tok.size_bins
    assert enc["iat_ids"].max() < tok.iat_bins


def test_forward_and_mlm_backward():
    import torch

    from cipherflow.model.flowformer import FlowFormer, count_params
    from cipherflow.model.heads import ClassifierHead, MLMHead
    from cipherflow.pretrain.mlm_dataset import MLMCollator, TokenizedFlowDataset
    from cipherflow.pretrain.train_mlm import mlm_loss
    from cipherflow.tokenizer.flow_tokenizer import FlowTokenizer

    cfg = _tiny_cfg()
    set_seed(0)
    tok = FlowTokenizer(cfg)
    rng = np.random.default_rng(0)
    records = [{
        "splt_ps": rng.integers(40, 1500, 10).tolist(),
        "splt_iat": rng.uniform(0, 500, 10).tolist(),
        "splt_dir": rng.integers(0, 2, 10).tolist(),
    } for _ in range(8)]

    ds = TokenizedFlowDataset(records, tok)
    collate = MLMCollator(mask_prob=0.3, seed=0)
    batch = collate([ds[i] for i in range(len(ds))])

    model = FlowFormer(cfg, tok.size_bins, tok.iat_bins)
    mlm = MLMHead(model.d_model, tok.size_bins, tok.iat_bins)
    assert count_params(model) > 0

    seq_out, pooled = model(batch["size_ids"], batch["iat_ids"], batch["dir_ids"],
                            batch["attn_mask"], mlm_mask=batch["mlm_mask"])
    assert seq_out.shape[1] == cfg["flow"]["seq_len"] + 1   # +CLS
    assert pooled.shape == (len(records), model.d_model)

    preds = mlm(seq_out)
    loss, parts = mlm_loss(preds, batch)
    loss.backward()
    assert torch.isfinite(loss).item()
    assert model.size_emb.weight.grad is not None  # gradients flow

    clf = ClassifierHead(model.d_model, n_classes=6)
    logits = clf(pooled.detach())
    assert logits.shape == (len(records), 6)


if __name__ == "__main__":
    ok = True
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except Exception as e:  # noqa: BLE001
                ok = False
                print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print("\nALL PASS" if ok else "\nSOME FAILED")
    raise SystemExit(0 if ok else 1)
