"""Fine-tune (or train from scratch) a flow classifier on top of FlowFormer.

Supports the core experiment of the project — comparing pretrained vs. from-scratch under
a shrinking label budget (`--few-shot K`). Loading the pretrained backbone should win big
when labels are scarce.

Run:
    # fine-tune from the pretrained backbone (full labels)
    python -m cipherflow.finetune.train_clf --pretrained artifacts/pretrained.pt
    # from-scratch baseline, only 10 labels/class
    python -m cipherflow.finetune.train_clf --pretrained none --few-shot 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from cipherflow.data.flow_io import load_flows, to_flow_records
from cipherflow.finetune.clf_dataset import clf_collate, encode_labels
from cipherflow.model.flowformer import FlowFormer
from cipherflow.model.heads import ClassifierHead
from cipherflow.pretrain.mlm_dataset import TokenizedFlowDataset
from cipherflow.tokenizer.flow_tokenizer import FlowTokenizer
from cipherflow.utils import ensure_dir, get_device, load_config, resolve_path, set_seed


def few_shot_subsample(y: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Return indices keeping at most k examples per class."""
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        keep.extend(idx[:k].tolist())
    return np.array(sorted(keep))


def load_pretrained_backbone(model: FlowFormer, path: str, device) -> None:
    ckpt = torch.load(path, map_location=device)
    state = ckpt["model"] if "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded pretrained backbone from {path} (missing={len(missing)}, unexpected={len(unexpected)})")


def build_loaders(records, y, tok, cfg, seed, few_shot=None, augment=False, hybrid=False):
    """Returns (train_dl, val_dl, test_dl, scaler). scaler=(mean,std) when hybrid, else None."""
    idx = np.arange(len(y))
    tr, tmp = train_test_split(idx, test_size=0.3, stratify=y, random_state=seed)
    va, te = train_test_split(tmp, test_size=0.5, stratify=y[tmp], random_state=seed)
    if few_shot:
        sub = few_shot_subsample(y[tr], few_shot, seed)
        tr = tr[sub]
    print(
        f"split: train={len(tr)} val={len(va)} test={len(te)}"
        + (f" (few-shot={few_shot}/class)" if few_shot else "")
        + (" [adversarial aug]" if augment else "")
        + (" [hybrid feats]" if hybrid else "")
    )

    # Hybrid: statistical features standardized with train-set statistics only.
    feats_all, scaler = None, None
    if hybrid:
        from cipherflow.finetune.features import apply_scaler, compute_features, fit_scaler

        feats_all = compute_features(records)
        mean, std = fit_scaler(feats_all[tr])
        feats_all = apply_scaler(feats_all, mean, std)
        scaler = (mean, std)

    def make(indices):
        recs = [records[i] for i in indices]
        f = feats_all[indices] if feats_all is not None else None
        return TokenizedFlowDataset(recs, tok, labels=y[indices], feats=f)

    # Optional evasion augmentation on the TRAIN split only (Claim C defense).
    if augment and not hybrid:
        from cipherflow.robustness.transforms import AugmentedFlowDataset

        train_ds = AugmentedFlowDataset([records[i] for i in tr], tok, cfg, labels=y[tr], seed=seed)
    else:
        train_ds = make(tr)

    bs = cfg["finetune"]["batch_size"]
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=clf_collate),
        DataLoader(make(va), batch_size=bs, shuffle=False, collate_fn=clf_collate),
        DataLoader(make(te), batch_size=bs, shuffle=False, collate_fn=clf_collate),
        scaler,
    )


def pooled_feature(model, batch):
    """Model pooling, concatenated with statistical features when present (hybrid)."""
    seq_out, _ = model(batch["size_ids"], batch["iat_ids"], batch["dir_ids"], batch["attn_mask"])
    feat = model.pool(seq_out, batch["attn_mask"])
    if "feats" in batch:
        feat = torch.cat([feat, batch["feats"]], dim=-1)
    return feat


@torch.no_grad()
def evaluate(model, head, dl, device):
    model.eval()
    head.eval()
    ys, ps = [], []
    for batch in dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = head(pooled_feature(model, batch))
        ps.append(logits.argmax(-1).cpu().numpy())
        ys.append(batch["label"].cpu().numpy())
    y_true, y_pred = np.concatenate(ys), np.concatenate(ps)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=None)
    ap.add_argument(
        "--pretrained",
        type=str,
        default="artifacts/pretrained.pt",
        help="path to pretrained backbone, or 'none' for from-scratch",
    )
    ap.add_argument("--few-shot", type=int, default=None, help="labels per class (default: all)")
    ap.add_argument("--augment", action="store_true", help="adversarial (padding+jitter) training aug")
    ap.add_argument(
        "--hybrid", action="store_true", help="hybrid head: CipherFlow embedding + statistical flow features"
    )
    ap.add_argument("--metrics-out", type=str, default=None, help="write metrics json here")
    ap.add_argument("--set", dest="overrides", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(overrides=args.overrides)
    set_seed(cfg["seed"])
    device = get_device()

    data_path = (
        Path(args.data) if args.data else (resolve_path(cfg, "paths", "data_dir") / "synthetic.parquet")
    )
    df = load_flows(data_path)
    df, y, label_names = encode_labels(df)
    records = to_flow_records(df)
    print(f"{len(records)} labeled flows | {len(label_names)} classes: {label_names} | device={device}")

    tok = FlowTokenizer(cfg)
    train_dl, val_dl, test_dl, feat_scaler = build_loaders(
        records, y, tok, cfg, cfg["seed"], args.few_shot, args.augment, args.hybrid
    )

    model = FlowFormer(cfg, tok.size_bins, tok.iat_bins).to(device)
    if args.pretrained and args.pretrained.lower() != "none":
        p = (
            args.pretrained
            if Path(args.pretrained).is_absolute()
            else str(resolve_path(cfg, "paths", "artifacts_dir").parent / args.pretrained)
        )
        load_pretrained_backbone(model, p, device)
    else:
        print("Training backbone from scratch (no pretraining).")

    n_feats = feat_scaler[0].shape[0] if feat_scaler else 0
    head = ClassifierHead(model.pool_dim + n_feats, len(label_names), dropout=cfg["model"]["dropout"]).to(
        device
    )

    fc = cfg["finetune"]
    if fc["freeze_encoder"]:
        for p in model.parameters():
            p.requires_grad = False
        params = list(head.parameters())
    else:
        params = list(model.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=fc["lr"], weight_decay=fc["weight_decay"])
    use_amp = bool(fc["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_f1, best_state = -1.0, None
    for ep in range(1, fc["epochs"] + 1):
        model.train(not fc["freeze_encoder"])
        head.train()
        tot = 0.0
        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device.type, enabled=use_amp):
                loss = F.cross_entropy(head(pooled_feature(model, batch)), batch["label"])
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item()
        val = evaluate(model, head, val_dl, device)
        print(
            f"epoch {ep:02d} | train loss {tot / len(train_dl):.3f} | val acc {val['accuracy']:.3f} f1 {val['macro_f1']:.3f}"
        )
        if val["macro_f1"] > best_f1:
            best_f1 = val["macro_f1"]
            best_state = (
                {k: v.cpu().clone() for k, v in model.state_dict().items()},
                {k: v.cpu().clone() for k, v in head.state_dict().items()},
            )

    # Restore best and report test metrics.
    if best_state:
        model.load_state_dict(best_state[0])
        head.load_state_dict(best_state[1])
    test = evaluate(model, head, test_dl, device)
    print(f"\nTEST | accuracy {test['accuracy']:.4f} | macro-F1 {test['macro_f1']:.4f}")

    ckpt_path = resolve_path(cfg, "finetune", "ckpt_path")
    ensure_dir(ckpt_path.parent)
    ckpt = {
        "model": model.state_dict(),
        "head": head.state_dict(),
        "label_names": label_names,
        "cfg": cfg,
        "size_bins": tok.size_bins,
        "iat_bins": tok.iat_bins,
        "hybrid": bool(args.hybrid),
        # Which dataset this was trained on, so the demo defaults to the matching one and
        # a reviewer can tell two same-class checkpoints apart.
        "data_path": str(data_path),
        "augment": bool(args.augment),
    }
    if feat_scaler is not None:
        ckpt["feat_mean"], ckpt["feat_std"] = feat_scaler
    torch.save(ckpt, ckpt_path)
    print(f"Saved classifier -> {ckpt_path}")

    if args.metrics_out:
        out = {
            "accuracy": test["accuracy"],
            "macro_f1": test["macro_f1"],
            "few_shot": args.few_shot,
            "pretrained": args.pretrained,
            "n_classes": len(label_names),
            "label_names": label_names,
        }
        ensure_dir(Path(args.metrics_out).parent)
        Path(args.metrics_out).write_text(json.dumps(out, indent=2))
        print(f"Wrote metrics -> {args.metrics_out}")


if __name__ == "__main__":
    main()
