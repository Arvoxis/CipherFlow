"""Generate a synthetic encrypted-traffic dataset for end-to-end development.

Two modes:

  * ``--mode sequential`` (default) — classes are defined by the *ordering* of packets
    (like real protocol handshakes / request-response patterns). Every class is engineered
    to have the SAME aggregate statistics (≈50% upstream, ≈50% large packets, same size and
    timing marginals); only the *sequence/position* pattern differs. This makes aggregate-
    statistic baselines (RandomForest/XGBoost) collapse toward chance, while a sequence model
    with positional awareness (CipherFlow) can separate them — demonstrating the core thesis.

  * ``--mode easy`` — classes with distinct size/timing marginals. Trivially separable by
    aggregate stats; useful only as a quick wiring smoke test, NOT for the headline result.

Schema is identical to real extraction (see data/flow_io.py), so nothing downstream changes.

Run:
    python -m cipherflow.data.make_synthetic --n-per-class 1500                 # sequential
    python -m cipherflow.data.make_synthetic --mode easy --n-per-class 1500     # easy
"""

from __future__ import annotations

import argparse
import uuid

import numpy as np
import pandas as pd

from cipherflow.data.flow_io import save_flows
from cipherflow.utils import load_config, resolve_path, set_seed

# ---- EASY mode (distinct marginals) ------------------------------------------
EASY_CLASSES = ["web", "video_stream", "voip", "file_transfer", "chat", "malware_c2"]


def _clip_sizes(x):
    return np.clip(np.round(x), 40, 1500).astype(int)


def gen_flow_easy(cls: str, rng: np.random.Generator, max_len: int = 40):
    n = rng.integers(12, max_len)
    if cls == "web":
        sizes = np.where(rng.random(n) < 0.4, rng.normal(200, 80, n), rng.normal(1100, 300, n))
        iats = np.abs(rng.exponential(8, n))
        dirs = (rng.random(n) < 0.7).astype(int)
    elif cls == "video_stream":
        sizes = rng.normal(1350, 120, n)
        iats = np.abs(rng.normal(6, 2, n))
        dirs = (rng.random(n) < 0.9).astype(int)
    elif cls == "voip":
        sizes = rng.normal(160, 25, n)
        iats = np.abs(rng.normal(20, 3, n))
        dirs = (rng.random(n) < 0.5).astype(int)
    elif cls == "file_transfer":
        sizes = rng.normal(1460, 30, n)
        iats = np.abs(rng.exponential(2, n))
        dirs = (rng.random(n) < 0.85).astype(int)
    elif cls == "chat":
        sizes = rng.normal(120, 40, n)
        iats = np.abs(rng.exponential(120, n))
        dirs = (rng.random(n) < 0.5).astype(int)
    elif cls == "malware_c2":
        sizes = rng.normal(90, 15, n)
        iats = np.abs(rng.normal(1000, 40, n))
        dirs = (rng.random(n) < 0.5).astype(int)
    else:
        raise ValueError(cls)
    return _clip_sizes(sizes), np.round(iats, 3), dirs.astype(int)


# ---- SEQUENTIAL mode (order-defined classes, matched marginals) --------------
# Each class = a fixed, balanced template of (direction, big/small) over positions. All
# templates are 50/50 in both axes, so per-flow aggregate stats are ~identical across classes;
# the discriminative signal lives purely in the ORDER.
SEQ_CLASSES = ["proto_A", "proto_B", "proto_C", "proto_D", "proto_E", "proto_F"]
SIZE_SMALL = (120, 35)  # mean, std
SIZE_BIG = (1350, 80)
IAT_SHARED = 10.0  # same timing distribution for ALL classes (no timing leakage)


def _balanced(L: int, rng) -> np.ndarray:
    v = np.array([0] * (L // 2) + [1] * (L - L // 2))
    rng.shuffle(v)
    return v


def build_templates(n_classes: int, L: int, seed: int):
    """Deterministic per-class (dir, big) templates — stable across pretrain/finetune runs."""
    rng = np.random.default_rng(seed + 777)
    return [(_balanced(L, rng), _balanced(L, rng)) for _ in range(n_classes)]


def gen_flow_seq(template, rng, L: int, flip_prob: float = 0.08):
    dir_tpl, big_tpl = template
    n = rng.integers(int(0.6 * L), L + 1)
    dirs = dir_tpl[:n].copy()
    bigs = big_tpl[:n].copy()
    # light perturbation so the model can't memorize an exact string
    flip = rng.random(n) < flip_prob
    dirs[flip] ^= 1
    flip2 = rng.random(n) < flip_prob
    bigs[flip2] ^= 1
    sizes = np.where(bigs == 1, rng.normal(*SIZE_BIG, n), rng.normal(*SIZE_SMALL, n))
    iats = np.abs(rng.exponential(IAT_SHARED, n))  # identical timing law for every class
    return _clip_sizes(sizes), np.round(iats, 3), dirs.astype(int)


def build(mode: str, n_per_class: int, seed: int, seq_len: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    if mode == "easy":
        classes = EASY_CLASSES
        gen = lambda c: gen_flow_easy(c, rng)
    else:
        classes = SEQ_CLASSES
        templates = build_templates(len(classes), seq_len, seed)
        tpl_by_cls = dict(zip(classes, templates, strict=True))
        gen = lambda c: gen_flow_seq(tpl_by_cls[c], rng, seq_len)

    for cls in classes:
        for _ in range(n_per_class):
            sizes, iats, dirs = gen(cls)
            rows.append(
                {
                    "flow_id": uuid.uuid4().hex,
                    "splt_ps": sizes.tolist(),
                    "splt_iat": iats.tolist(),
                    "splt_dir": dirs.tolist(),
                    "n_packets": int(len(sizes)),
                    "label": cls,
                    "dataset": f"synthetic_{mode}",
                }
            )
    return pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sequential", "easy"], default="sequential")
    ap.add_argument("--n-per-class", type=int, default=1500)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--set", dest="overrides", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(overrides=args.overrides)
    set_seed(cfg["seed"])
    out = args.out or (resolve_path(cfg, "paths", "data_dir") / "synthetic.parquet")

    df = build(args.mode, args.n_per_class, cfg["seed"], cfg["flow"]["seq_len"])
    save_flows(df, out)
    print(f"[{args.mode}] Wrote {len(df)} flows across {df['label'].nunique()} classes -> {out}")
    print(df["label"].value_counts().to_string())
    if args.mode == "sequential":
        print("\nNote: classes share aggregate statistics by design - order is the only signal.")


if __name__ == "__main__":
    main()
