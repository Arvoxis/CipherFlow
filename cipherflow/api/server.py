"""JSON API behind the CipherFlow React demo.

Everything here is a thin wrapper over the same code paths the CLI and the Streamlit app use
(`cipherflow.inference`, `cipherflow.tokenizer.inspect`, `cipherflow.robustness.transforms`),
so a number on screen is the same number `python -m cipherflow.robustness.eval_robustness`
prints. No model logic is reimplemented in this file.

Run:
    python -m uvicorn cipherflow.api.server:app --port 8000
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score

from cipherflow.data.flow_io import load_flows, to_flow_records
from cipherflow.finetune.clf_dataset import encode_labels, test_split
from cipherflow.inference import embed_records, load_classifier, predict_records
from cipherflow.robustness.transforms import evade_records
from cipherflow.tokenizer.inspect import tokenize_table
from cipherflow.utils import REPO_ROOT, load_config, resolve_path

CFG = load_config()
ART = resolve_path(CFG, "paths", "artifacts_dir")
DATA = resolve_path(CFG, "paths", "data_dir")
SEED = CFG["seed"]

app = FastAPI(title="CipherFlow API", version="1.0")
# The Vite dev server is a different origin (5173 vs 8000). Local demo only, so allow any.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------------------- loading (cached)
@lru_cache(maxsize=8)
def _model(ckpt: str):
    path = ART / ckpt
    if not path.exists():
        raise HTTPException(404, f"No checkpoint {ckpt} in artifacts/")
    return load_classifier(path)


@lru_cache(maxsize=8)
def _data(dataset: str):
    path = DATA / dataset
    if not path.exists():
        raise HTTPException(404, f"No dataset {dataset} in data_out/")
    return load_flows(path)


@lru_cache(maxsize=16)
def _held_out_ids(ckpt: str, dataset: str) -> frozenset:
    """flow_ids of the held-out test split for this checkpoint.

    The replay covers the whole dataset, which is the right demo but the wrong number: most
    of those flows were in training. Marking the test split lets the UI show both an overall
    running accuracy and the honest held-out one, instead of only the flattering figure.
    """
    _, _, _, label_names, _ = _model(ckpt)
    try:
        dfl, y, _ = encode_labels(_data(dataset), label_names=label_names)
    except Exception:
        return frozenset()  # labels this checkpoint does not know; no split to speak of
    return frozenset(dfl["flow_id"].iloc[test_split(y, SEED)].tolist())


@lru_cache(maxsize=16)
def _stream(ckpt: str, dataset: str):
    """Predictions over the whole dataset in a fixed shuffled order.

    Datasets are written grouped by class, so replaying them in file order shows one class at
    a time and makes a running accuracy meaningless. Shuffling once with a fixed seed gives
    arrival order AND makes any (offset, limit) window reproducible across requests.
    """
    model, head, tok, label_names, _ = _model(ckpt)
    df = _data(dataset).sample(frac=1.0, random_state=0).reset_index(drop=True)
    probs = predict_records(model, head, tok, to_flow_records(df))
    truth = df["label"].tolist() if "label" in df.columns else [None] * len(df)
    held = _held_out_ids(ckpt, dataset)
    ids = df["flow_id"].tolist()
    packets = df["n_packets"].tolist()
    return [
        {
            "flow": i,
            "prediction": label_names[int(p.argmax())],
            "confidence": float(p.max()),
            "actual": truth[i],
            "packets": int(packets[i]),
            "held_out": ids[i] in held,
        }
        for i, p in enumerate(probs)
    ]


@lru_cache(maxsize=64)
def _score(ckpt: str, dataset: str, n: int, evasion: tuple | None):
    """Accuracy / macro-F1 on the held-out test split, optionally under evasion."""
    model, head, tok, label_names, cfg = _model(ckpt)
    df, y, _ = encode_labels(_data(dataset), label_names=label_names)
    te = test_split(y, SEED, n)
    all_records = to_flow_records(df)
    records = [all_records[i] for i in te]
    y = y[te]
    if evasion is not None:
        pad_prob, pad_max_frac, jitter_std_ms = evasion
        rc = {"pad_prob": pad_prob, "pad_max_frac": pad_max_frac, "jitter_std_ms": jitter_std_ms}
        records = evade_records(records, rc, SEED, cfg["flow"]["max_size"])
    pred = predict_records(model, head, tok, records).argmax(-1)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "n": int(len(y)),
    }


# ----------------------------------------------------------------- endpoints
@app.get("/api/health")
def health():
    return {"ok": True, "device": "cuda" if torch.cuda.is_available() else "cpu"}


@app.get("/api/meta")
def meta():
    """Everything the UI needs to populate its pickers, in one round trip."""
    ckpts = []
    for p in sorted(ART.glob("*.pt")):
        if p.name.startswith("pretrained"):
            continue
        try:
            model, _, tok, label_names, _ = _model(p.name)
        except Exception:
            continue  # a checkpoint from an older schema should not blank the whole page
        ckpts.append(
            {
                "name": p.name,
                "classes": label_names,
                "trained_on": Path(model._data_path).name if model._data_path else None,
                "hybrid": bool(getattr(model, "_hybrid", False)),
                "augment": bool(getattr(model, "_augment", False)),
                "params": sum(q.numel() for q in model.parameters()),
                "vocab": tok.composite_vocab_size,
                "size_bins": tok.size_bins,
                "iat_bins": tok.iat_bins,
            }
        )
    datasets = []
    for p in sorted(DATA.glob("*.parquet")):
        try:
            df = _data(p.name)
        except Exception:
            continue
        datasets.append(
            {
                "name": p.name,
                "flows": int(len(df)),
                "labels": sorted(df["label"].unique().tolist()) if "label" in df else [],
            }
        )
    return {
        "checkpoints": ckpts,
        "datasets": datasets,
        "default_checkpoint": Path(CFG["finetune"]["ckpt_path"]).name,
        "robustness_defaults": {
            "pad_prob": CFG["robustness"]["pad_prob"],
            "pad_max_frac": CFG["robustness"]["pad_max_frac"],
            "jitter_std_ms": CFG["robustness"]["jitter_std_ms"],
        },
        "seed": SEED,
    }


@app.get("/api/stream")
def stream(
    ckpt: str,
    dataset: str,
    offset: int = 0,
    limit: int = Query(10, ge=1, le=500),
):
    """One window of the replay. The UI polls this with a rising offset to animate the feed."""
    rows = _stream(ckpt, dataset)
    window = rows[offset : offset + limit]
    seen = rows[: offset + limit]
    scored = [r for r in seen if r["actual"] is not None]
    held = [r for r in scored if r["held_out"]]
    counts: dict[str, int] = {}
    for r in seen:
        counts[r["prediction"]] = counts.get(r["prediction"], 0) + 1

    def _acc(rs):
        return sum(r["actual"] == r["prediction"] for r in rs) / len(rs) if rs else None

    return {
        "rows": window,
        "total": len(rows),
        "seen": len(seen),
        "counts": counts,
        "running_accuracy": _acc(scored),
        "held_out_accuracy": _acc(held),
        "held_out_seen": len(held),
        "held_out_total": sum(r["held_out"] for r in rows),
        "done": offset + limit >= len(rows),
    }


@app.get("/api/tokenize")
def tokenize(ckpt: str, dataset: str, idx: int = 0):
    """Claim A: one flow, packet by packet, as flow-shape tokens. No payload bytes read."""
    _, _, tok, _, _ = _model(ckpt)
    df = _data(dataset)
    if not 0 <= idx < len(df):
        raise HTTPException(400, f"idx out of range 0..{len(df) - 1}")
    row = df.iloc[idx]
    table = tokenize_table(tok, row["splt_ps"], row["splt_iat"], row["splt_dir"])
    return {
        "label": row.get("label"),
        "packets": int(row["n_packets"]),
        "tokens": table.to_dict("records"),
        "vocab": tok.composite_vocab_size,
        "size_bins": tok.size_bins,
        "iat_bins": tok.iat_bins,
        "payload_bytes_read": 0,
    }


@app.get("/api/embed")
def embed(ckpt: str, dataset: str, n: int = Query(600, ge=50, le=3000)):
    """Claim B: 2-D PCA of the pooled flow embeddings, colored by true class."""
    model, _, tok, label_names, _ = _model(ckpt)
    df = _data(dataset)
    if "label" not in df.columns:
        raise HTTPException(400, "Dataset has no labels to color by")
    # Datasets are written grouped by class, so head(n) would only ever reach the first
    # class or two. Sample instead, with a fixed seed so the map is stable across reloads.
    dfl, y, names = encode_labels(df.sample(n=min(n, len(df)), random_state=0), label_names=label_names)
    embs = embed_records(model, tok, to_flow_records(dfl))
    pca = PCA(n_components=2, random_state=0)
    xy = pca.fit_transform(embs)
    return {
        "classes": names,
        "points": [
            {"x": float(a), "y": float(b), "label": names[int(c)]} for (a, b), c in zip(xy, y, strict=True)
        ],
        "explained_variance": [float(v) for v in pca.explained_variance_ratio_],
        "dim": int(embs.shape[1]),
    }


@app.get("/api/evasion")
def evasion(
    dataset: str,
    ckpts: str = Query(..., description="comma-separated checkpoint filenames"),
    pad_prob: float = Query(0.5, ge=0.0, le=1.0),
    pad_max_frac: float = Query(0.5, ge=0.0, le=2.0),
    jitter_std_ms: float = Query(10.0, ge=0.0, le=200.0),
    n: int = Query(800, ge=50, le=4000),
):
    """Claim C: every selected checkpoint, clean vs under the attacker's padding and jitter."""
    rows = []
    for name in [c.strip() for c in ckpts.split(",") if c.strip()]:
        try:
            clean = _score(name, dataset, n, None)
            evaded = _score(name, dataset, n, (pad_prob, pad_max_frac, jitter_std_ms))
        except Exception as e:
            # One mismatched checkpoint must not blank the comparison for the others.
            rows.append({"checkpoint": name, "error": f"{type(e).__name__}: {e}"})
            continue
        drop = clean["accuracy"] - evaded["accuracy"]
        rows.append(
            {
                "checkpoint": name,
                "n": clean["n"],
                "clean_acc": clean["accuracy"],
                "evaded_acc": evaded["accuracy"],
                "acc_lost": drop,
                "relative_loss": drop / clean["accuracy"] if clean["accuracy"] else 0.0,
                "clean_f1": clean["macro_f1"],
                "evaded_f1": evaded["macro_f1"],
                "augment": bool(getattr(_model(name)[0], "_augment", False)),
            }
        )
    return {
        "rows": rows,
        "attack": {
            "pad_prob": pad_prob,
            "pad_max_frac": pad_max_frac,
            "jitter_std_ms": jitter_std_ms,
        },
    }


@app.get("/api/provenance")
def provenance(ckpt: str | None = None):
    """Reduction-to-practice record: revision, config hash, seed, plus the run log."""
    from cipherflow.provenance import stamp

    cur = stamp(CFG, step="demo")
    manifest = ART / "run_manifest.json"
    history = []
    if manifest.exists():
        try:
            history = json.loads(manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            history = []
    return {"current": cur, "checkpoint": ckpt, "history": history[-40:]}


# ----------------------------------------------------------------- built UI
# Mounted last so every /api route above wins. With a build present this one process serves
# the whole demo on one port, which is one less thing to go wrong in front of an audience.
# During development `npm run dev` proxies /api here instead and this mount is unused.
_DIST = REPO_ROOT / "frontend" / "dist"
if _DIST.exists():
    app.mount("/", StaticFiles(directory=_DIST, html=True), name="ui")


def demo():
    """Self-check: hit every endpoint in-process against whatever is on disk."""
    from fastapi.testclient import TestClient

    c = TestClient(app)
    assert c.get("/api/health").json()["ok"]
    m = c.get("/api/meta").json()
    assert m["checkpoints"] and m["datasets"], "no checkpoints or datasets on disk"
    ck = m["default_checkpoint"]
    ds = next(k["trained_on"] for k in m["checkpoints"] if k["name"] == ck) or m["datasets"][0]["name"]
    print(f"checkpoint={ck}  dataset={ds}")

    s = c.get("/api/stream", params={"ckpt": ck, "dataset": ds, "offset": 0, "limit": 5}).json()
    assert len(s["rows"]) == 5 and s["total"] > 5, s
    assert 0 < s["held_out_total"] < s["total"], "held-out split looks wrong"
    assert all("held_out" in r for r in s["rows"]), s["rows"][0]
    # The same window twice must be identical, or the replay order is not reproducible.
    again = c.get("/api/stream", params={"ckpt": ck, "dataset": ds, "offset": 0, "limit": 5}).json()
    assert s["rows"] == again["rows"], "stream order is not stable"

    t = c.get("/api/tokenize", params={"ckpt": ck, "dataset": ds, "idx": 0}).json()
    assert t["tokens"] and t["payload_bytes_read"] == 0, t

    e = c.get("/api/embed", params={"ckpt": ck, "dataset": ds, "n": 120}).json()
    assert e["points"] and e["classes"], e

    v = c.get("/api/evasion", params={"dataset": ds, "ckpts": ck, "n": 100}).json()
    assert "clean_acc" in v["rows"][0], v

    p = c.get("/api/provenance").json()
    assert len(p["current"]["config_hash"]) == 12

    assert c.get("/api/stream", params={"ckpt": "nope.pt", "dataset": ds}).status_code == 404
    print("api selfcheck PASS")


if __name__ == "__main__":
    demo()
