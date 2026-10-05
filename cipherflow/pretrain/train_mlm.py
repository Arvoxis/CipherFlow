"""Self-supervised masked-flow pretraining (patent Claim B).

Trains FlowFormer + MLMHead to reconstruct masked (size, iat, direction) bins from the
surrounding flow shape. Labels are NOT used here — this learns from unlabeled traffic.
Saves the backbone weights for fine-tuning.

Run:
    python -m cipherflow.pretrain.train_mlm --data data_out/synthetic.parquet
    python -m cipherflow.pretrain.train_mlm --set pretrain.epochs=3 model.d_model=64   # quick test
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cipherflow.data.flow_io import load_flows, to_flow_records
from cipherflow.model.flowformer import FlowFormer, count_params
from cipherflow.model.heads import MLMHead
from cipherflow.pretrain.mlm_dataset import IGNORE, MLMCollator, TokenizedFlowDataset
from cipherflow.tokenizer.flow_tokenizer import FlowTokenizer
from cipherflow.utils import ensure_dir, get_device, load_config, resolve_path, set_seed


def mlm_loss(preds, batch):
    size_logits, iat_logits, dir_logits = preds
    B, L, _ = size_logits.shape
    ls = F.cross_entropy(size_logits.reshape(B * L, -1), batch["tgt_size"].reshape(-1), ignore_index=IGNORE)
    li = F.cross_entropy(iat_logits.reshape(B * L, -1), batch["tgt_iat"].reshape(-1), ignore_index=IGNORE)
    ld = F.cross_entropy(dir_logits.reshape(B * L, -1), batch["tgt_dir"].reshape(-1), ignore_index=IGNORE)
    return ls + li + ld, (ls.item(), li.item(), ld.item())


@torch.no_grad()
def masked_accuracy(preds, batch):
    """Fraction of masked positions whose size-bin is predicted exactly (a simple health metric)."""
    size_logits = preds[0]
    tgt = batch["tgt_size"]
    sel = tgt != IGNORE
    if sel.sum() == 0:
        return 0.0
    pred = size_logits.argmax(-1)
    return (pred[sel] == tgt[sel]).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=None, help="flows parquet (default: data_out/synthetic.parquet)")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--set", dest="overrides", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(overrides=args.overrides)
    set_seed(cfg["seed"])
    device = get_device()

    data_path = Path(args.data) if args.data else (resolve_path(cfg, "paths", "data_dir") / "synthetic.parquet")
    df = load_flows(data_path)
    records = to_flow_records(df)
    print(f"Loaded {len(records)} flows from {data_path} | device={device}")

    tok = FlowTokenizer(cfg)
    ds = TokenizedFlowDataset(records, tok)
    n_val = max(1, int(len(ds) * args.val_frac))
    n_train = len(ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [n_train, n_val], generator=torch.Generator().manual_seed(cfg["seed"])
    )

    pc = cfg["pretrain"]
    collate = MLMCollator(mask_prob=pc["mask_prob"], seed=cfg["seed"])
    train_dl = DataLoader(train_ds, batch_size=pc["batch_size"], shuffle=True, collate_fn=collate, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=pc["batch_size"], shuffle=False, collate_fn=collate)

    model = FlowFormer(cfg, tok.size_bins, tok.iat_bins).to(device)
    head = MLMHead(model.d_model, tok.size_bins, tok.iat_bins).to(device)
    print(f"Params: backbone={count_params(model):,}  head={count_params(head):,}")

    params = list(model.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=pc["lr"], weight_decay=pc["weight_decay"])
    use_amp = bool(pc["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def move(batch):
        return {k: v.to(device) for k, v in batch.items()}

    def run_epoch(dl, train: bool):
        model.train(train); head.train(train)
        tot, acc, steps = 0.0, 0.0, 0
        for batch in dl:
            batch = move(batch)
            with torch.set_grad_enabled(train), torch.autocast(device.type, enabled=use_amp):
                seq_out, _ = model(batch["size_ids"], batch["iat_ids"], batch["dir_ids"],
                                   batch["attn_mask"], mlm_mask=batch["mlm_mask"])
                preds = head(seq_out)
                loss, _ = mlm_loss(preds, batch)
            if train:
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            tot += loss.item(); acc += masked_accuracy(preds, batch); steps += 1
        return tot / steps, acc / steps

    out_ckpt = resolve_path(cfg, "pretrain", "ckpt_path")
    ensure_dir(out_ckpt.parent)
    best_val = float("inf")
    for ep in range(1, pc["epochs"] + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(train_dl, True)
        va_loss, va_acc = run_epoch(val_dl, False)
        print(f"epoch {ep:02d} | train loss {tr_loss:.3f} acc {tr_acc:.3f} "
              f"| val loss {va_loss:.3f} acc {va_acc:.3f} | {time.time()-t0:.1f}s")
        if va_loss < best_val:
            best_val = va_loss
            torch.save({"model": model.state_dict(), "cfg": cfg,
                        "size_bins": tok.size_bins, "iat_bins": tok.iat_bins}, out_ckpt)
    print(f"Saved pretrained backbone -> {out_ckpt}  (best val loss {best_val:.3f})")


if __name__ == "__main__":
    main()
