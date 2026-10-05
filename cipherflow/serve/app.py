"""CipherFlow live demo — a multi-tab Streamlit app for the viva / patent defense.

Tabs:
  1. Live classification    — replay flows, watch predictions + running accuracy.
  2. Tokenization (Claim A) — turn any flow into flow-shape tokens, with readable bins.
  3. Embedding map (Claim B)— 2-D projection of learned flow embeddings, colored by class.
  4. Evasion (Claim C)      — pad and jitter the test set live, watch accuracy hold or collapse.
  5. Provenance             — code revision, config hash and seed behind the loaded model.

Data sources: any dataset in data_out/, an uploaded flows parquet, or a .pcap (extracted
with the dpkt extractor).

Run:
    python -m streamlit run cipherflow/serve/app.py   (from the repo root)
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

# `streamlit run cipherflow/serve/app.py` puts THIS file's folder on sys.path, not the repo
# root, so `import cipherflow` would fail. Add the repo root (three levels up) so the app runs
# no matter the working directory or launch method.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from cipherflow.data.flow_io import load_flows, to_flow_records
from cipherflow.finetune.clf_dataset import encode_labels
from cipherflow.inference import embed_records, load_classifier, predict_records
from cipherflow.robustness.transforms import evade_records
from cipherflow.tokenizer.inspect import tokenize_table
from cipherflow.utils import load_config, resolve_path

st.set_page_config(page_title="CipherFlow", page_icon="🔐", layout="wide")

# Streamlit renamed use_container_width -> width="stretch" in 1.49; this keeps one spelling.
STRETCH = {"width": "stretch"}


@st.cache_resource
def _load(ckpt_path: str):
    return load_classifier(ckpt_path)


@st.cache_data
def _load_data(path: str) -> pd.DataFrame:
    return load_flows(path)


@st.cache_data(show_spinner=False)
def _trained_on(ckpt_path: str) -> str | None:
    """Basename of the dataset a checkpoint was fine-tuned on, or None on older checkpoints."""
    import torch

    dp = torch.load(ckpt_path, map_location="cpu", weights_only=False).get("data_path")
    return Path(dp).name if dp else None


@st.cache_data(show_spinner=False)
def _scored(ckpt_path: str, data_path: str, n: int, evasion: tuple | None, seed: int):
    """Accuracy / macro-F1 for one checkpoint on one dataset, optionally under evasion.

    Scores the HELD-OUT TEST SPLIT, reproducing the same stratified split train_clf used, so
    these numbers match `cipherflow.robustness.eval_robustness` instead of quietly scoring
    training data. Cached on the plain arguments so dragging a slider re-scores only what
    actually changed.
    """
    model, head, tok, label_names, cfg = _load(ckpt_path)
    df = _load_data(data_path)
    df, y, _ = encode_labels(df, label_names=label_names)
    records = to_flow_records(df)

    idx = np.arange(len(y))
    _, tmp = train_test_split(idx, test_size=0.3, stratify=y, random_state=seed)
    _, te = train_test_split(tmp, test_size=0.5, stratify=y[tmp], random_state=seed)
    if n < len(te):  # subsample the test split, keeping class balance
        te, _ = train_test_split(te, train_size=n, stratify=y[te], random_state=seed)
    records = [records[i] for i in te]
    y = y[te]

    if evasion is not None:
        pad_prob, pad_max_frac, jitter_std_ms = evasion
        rc = {"pad_prob": pad_prob, "pad_max_frac": pad_max_frac, "jitter_std_ms": jitter_std_ms}
        records = evade_records(records, rc, seed, cfg["flow"]["max_size"])
    pred = predict_records(model, head, tok, records).argmax(-1)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "n": int(len(y)),
    }


def _pcap_to_flows(uploaded, cfg) -> pd.DataFrame | None:
    """Extract flows from an uploaded pcap with the dpkt extractor."""
    try:
        from cipherflow.data.extract_pcap import extract_pcap_file
    except Exception as e:
        st.error(f"pcap extractor unavailable ({e}). `pip install dpkt` and retry.")
        return None
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pcap") as tf:
        tf.write(uploaded.getbuffer())
        tmp_path = tf.name
    try:
        rows = extract_pcap_file(
            tmp_path,
            label="unknown",
            dataset="uploaded",
            n=cfg["flow"]["seq_len"],
            min_packets=4,
        )
    except SystemExit as e:
        st.error(str(e))
        return None
    return pd.DataFrame(rows) if rows else None


def pick_checkpoint(cfg) -> str | None:
    """Sidebar checkpoint picker over whatever is in artifacts/, newest first."""
    art = resolve_path(cfg, "paths", "artifacts_dir")
    # The configured checkpoint leads the list so the demo opens on the model the write-up
    # quotes, whatever got trained last. The rest are alphabetical, not by mtime.
    default = Path(cfg["finetune"]["ckpt_path"]).name
    ckpts = [p for p in art.glob("*.pt") if not p.name.startswith("pretrained")]
    ckpts.sort(key=lambda p: (p.name != default, p.name))
    if not ckpts:
        st.warning(
            f"No classifier in `{art}`. Train one:\n"
            "`python -m cipherflow.finetune.train_clf --pretrained artifacts/pretrained.pt`"
        )
        return None
    names = [p.name for p in ckpts]
    pick = st.selectbox("Classifier checkpoint", names, help="Anything matching artifacts/*.pt")
    with st.expander("Other path"):
        manual = st.text_input("Load a checkpoint by path", "")
    return manual.strip() or str(art / pick)


def get_source_df(cfg, label_names, trained_on=None) -> tuple[pd.DataFrame | None, str | None]:
    """Return (dataframe, path). path is None for uploads, which have no cacheable identity."""
    src = st.radio(
        "Flow source",
        ["Sample dataset", "Upload flows parquet", "Upload .pcap"],
        horizontal=True,
    )
    if src == "Upload flows parquet":
        up = st.file_uploader("flows .parquet", type=["parquet"])
        return (load_flows(up) if up else None), None
    if src == "Upload .pcap":
        up = st.file_uploader("capture .pcap / .pcapng", type=["pcap", "pcapng"])
        return (_pcap_to_flows(up, cfg) if up else None), None

    data_dir = resolve_path(cfg, "paths", "data_dir")
    found = sorted(data_dir.glob("*.parquet"))
    if not found:
        st.warning(f"No dataset in `{data_dir}`. Run `python -m cipherflow.data.make_synthetic`.")
        return None, None
    # Default to the dataset the checkpoint was actually trained on; failing that, anything
    # whose classes overlap. Two datasets can share class names, so the exact path wins.
    trained_name = Path(trained_on).name if trained_on else None
    found.sort(
        key=lambda p: (
            p.name != trained_name,
            not (set(_load_data(str(p))["label"]) & set(label_names)),
            p.name,
        )
    )
    pick = st.selectbox("Dataset", [p.name for p in found])
    if trained_name and pick != trained_name:
        st.caption(f"Model was trained on `{trained_name}`.")
    path = str(data_dir / pick)
    return _load_data(path), path


# ------------------------------------------------------------------ tabs
def tab_live(model, head, tok, label_names, df):
    st.subheader("Live classification")
    st.caption(
        "Flows stream through the model one batch at a time, exactly as they would off a tap. "
        "Nothing but packet size, timing and direction reaches the classifier."
    )
    # Datasets land on disk grouped by class. Replaying them in file order shows one class at a
    # time and makes the running accuracy meaningless, so shuffle into arrival order first.
    df = df.sample(frac=1.0, random_state=0).reset_index(drop=True)
    truth = df["label"].tolist() if "label" in df.columns else None
    records = to_flow_records(df)
    c1, c2 = st.columns(2)
    speed = c1.slider("Replay speed (flows/sec)", 1, 200, 40)
    batch = c2.slider("Batch per tick", 1, 50, 5)

    if not st.button("Start", type="primary"):
        st.info("Press Start to stream flows through the model.")
        return

    counts = {c: 0 for c in label_names}
    correct = total = 0
    prog = st.progress(0.0)
    metric_area = st.empty()
    left, right = st.columns([1, 1])
    chart_area, feed_area = left.empty(), right.empty()

    for i in range(0, len(records), batch):
        probs = predict_records(model, head, tok, records[i : i + batch])
        preds, conf = probs.argmax(-1), probs.max(-1)
        feed = []
        for j, (p, c) in enumerate(zip(preds, conf, strict=True)):
            name = label_names[p]
            counts[name] += 1
            row = {"flow": i + j, "prediction": name, "confidence": f"{c:.2f}"}
            if truth is not None:
                row["actual"] = truth[i + j]
                total += 1
                correct += int(truth[i + j] == name)
            feed.append(row)
        chart_area.bar_chart(pd.Series(counts))
        feed_area.dataframe(pd.DataFrame(feed), hide_index=True, **STRETCH)
        if truth is not None and total:
            m1, m2 = metric_area.columns(2)
            m1.metric("Live accuracy", f"{correct / total:.1%}")
            m2.metric("Flows seen", f"{total}")
        prog.progress(min(1.0, (i + batch) / len(records)))
        time.sleep(batch / speed)
    st.success("Done streaming.")


def tab_tokenize(tok, df):
    st.subheader("Flow-shape tokenization — patent Claim A")
    st.caption(
        "Every packet becomes one token: (quantized size, quantized inter-arrival time, direction). "
        "**No payload bytes are read.** This is the step that keeps working under TLS 1.3 and QUIC."
    )
    idx = st.number_input("Pick a flow index", 0, max(0, len(df) - 1), 0)
    row = df.iloc[int(idx)]
    if "label" in df.columns:
        st.write(f"True label: **{row['label']}** · {len(row['splt_ps'])} packets")
    table = tokenize_table(tok, row["splt_ps"], row["splt_iat"], row["splt_dir"])
    st.dataframe(table, hide_index=True, **STRETCH)
    c1, c2, c3 = st.columns(3)
    c1.metric("Composite vocabulary", f"{tok.composite_vocab_size:,}")
    c2.metric("Size × IAT bins", f"{tok.size_bins} × {tok.iat_bins}")
    c3.metric("Payload bytes read", "0")


def tab_embeddings(model, tok, label_names, df):
    st.subheader("Learned flow embeddings — patent Claim B")
    st.caption(
        "A 2-D projection of the pooled flow embeddings. Clusters that separate without ever "
        "having seen a label mean the masked-flow-token objective learned something transferable."
    )
    if "label" not in df.columns:
        st.warning("Need labeled data to color the embedding map.")
        return
    n = st.slider("Flows to plot", 100, min(2000, len(df)), min(800, len(df)))
    dfl, y, names = encode_labels(df.head(n))
    embs = embed_records(model, tok, to_flow_records(dfl))
    from sklearn.decomposition import PCA

    xy = PCA(n_components=2, random_state=0).fit_transform(embs)
    fig, ax = plt.subplots(figsize=(7, 5))
    for c, name in enumerate(names):
        m = y == c
        ax.scatter(xy[m, 0], xy[m, 1], s=10, alpha=0.6, label=name)
    ax.legend(markerscale=2, fontsize=8)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    st.pyplot(fig)


def tab_evasion(cfg, ckpt_path, data_path, label_names):
    """Claim C made interactive: the attacker's knobs are the sliders."""
    st.subheader("Evasion robustness — patent Claim C")
    st.caption(
        "Take the attacker's position. You cannot break the encryption, so you reshape the flow: "
        "pad packets to hide their sizes, jitter the timing to smear the gaps. Turn the knobs and "
        "watch what it costs the classifier."
    )
    if data_path is None:
        st.info("Pick a dataset from `data_out/` for this tab. Uploads are not re-scored here.")
        return

    rc = cfg["robustness"]
    c1, c2, c3 = st.columns(3)
    pad_prob = c1.slider("Padding probability", 0.0, 1.0, float(rc["pad_prob"]), 0.05)
    pad_frac = c2.slider("Max size inflation", 0.0, 2.0, float(rc["pad_max_frac"]), 0.05)
    jitter = c3.slider("Timing jitter (ms, std)", 0.0, 100.0, float(rc["jitter_std_ms"]), 1.0)

    n = st.slider(
        "Test flows to score", 100, 4000, 800, 100, help="Capped at the size of the held-out test split."
    )
    art = resolve_path(cfg, "paths", "artifacts_dir")
    others = [p.name for p in sorted(art.glob("*.pt")) if not p.name.startswith("pretrained")]
    # Default to every checkpoint trained on the dataset in view. Comparing a normally-trained
    # model against an augmented one is the whole point of this tab, so defaulting to a single
    # checkpoint would hide the result it exists to show.
    same_data = [n for n in others if _trained_on(str(art / n)) == Path(data_path).name]
    compare = st.multiselect(
        "Compare checkpoints",
        others,
        default=same_data or ([Path(ckpt_path).name] if Path(ckpt_path).name in others else others[:1]),
        help="Put a normally-trained checkpoint next to one trained with --augment to see the defense.",
    )
    if not compare:
        st.info("Select at least one checkpoint.")
        return

    seed = cfg["seed"]
    rows = []
    with st.spinner("Scoring clean and evaded flows..."):
        for name in compare:
            p = str(art / name)
            try:
                clean = _scored(p, data_path, n, None, seed)
                evaded = _scored(p, data_path, n, (pad_prob, pad_frac, jitter), seed)
            except Exception as e:
                st.warning(f"{name}: skipped ({type(e).__name__}: {e})")
                continue
            drop = clean["accuracy"] - evaded["accuracy"]
            rows.append(
                {
                    "checkpoint": name,
                    "n": clean["n"],
                    "clean acc": clean["accuracy"],
                    "evaded acc": evaded["accuracy"],
                    "acc lost": drop,
                    "relative loss": drop / clean["accuracy"] if clean["accuracy"] else 0.0,
                    "clean F1": clean["macro_f1"],
                    "evaded F1": evaded["macro_f1"],
                }
            )
    if not rows:
        return

    res = pd.DataFrame(rows).sort_values("relative loss")
    best = res.iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Most robust", best["checkpoint"])
    c2.metric(
        "Accuracy under evasion",
        f"{best['evaded acc']:.1%}",
        f"{-best['acc lost']:.1%} vs clean",
    )
    c3.metric("Relative loss", f"{best['relative loss']:.1%}", delta_color="inverse")

    st.dataframe(
        res.style.format(
            {
                "clean acc": "{:.3f}",
                "evaded acc": "{:.3f}",
                "acc lost": "{:+.3f}",
                "relative loss": "{:.1%}",
                "clean F1": "{:.3f}",
                "evaded F1": "{:.3f}",
            }
        ),
        hide_index=True,
        **STRETCH,
    )

    chart = res.set_index("checkpoint")[["clean acc", "evaded acc"]]
    st.bar_chart(chart)
    st.caption(
        f"Scored on the held-out test split ({rows[0]['n']} flows), the same split "
        "`cipherflow.robustness.eval_robustness` uses, so these match the CLI. A short bar next "
        "to a tall one is a classifier the attacker just defeated. Two bars of similar height is "
        "Claim C holding up. Train with `--augment` to get the second picture."
    )


def tab_provenance(cfg, ckpt_path):
    st.subheader("Provenance")
    st.caption(
        "What exactly produced the loaded model. For a patent file this is the reduction-to-practice "
        "record: a revision, a config hash and a seed, not a claim that it worked once."
    )
    from cipherflow.provenance import stamp

    cur = stamp(cfg, step="demo")
    c1, c2, c3 = st.columns(3)
    c1.metric("Commit", (cur["git_commit"] or "no git")[:10])
    c2.metric("Config hash", cur["config_hash"])
    c3.metric("Seed", str(cur["seed"]))
    if cur["git_dirty"]:
        st.warning("Working tree is dirty, so this run is not reproducible from the commit alone.")
    st.json(cur, expanded=False)

    st.write("**Loaded checkpoint**")
    st.code(ckpt_path, language=None)

    manifest = resolve_path(cfg, "paths", "artifacts_dir") / "run_manifest.json"
    if manifest.exists():
        st.write("**Run log**")
        hist = json.loads(manifest.read_text(encoding="utf-8"))
        st.dataframe(
            pd.DataFrame(hist)[["timestamp_utc", "step", "git_commit", "config_hash", "device"]],
            hide_index=True,
            **STRETCH,
        )
    else:
        st.info("No run_manifest.json yet. It is written by `python -m cipherflow.pipeline`.")


def main():
    st.title("CipherFlow — Encrypted-Traffic Classifier")
    st.caption(
        "Classification from **flow shape alone**: packet size, inter-arrival time, direction. "
        "No payload inspection, no TLS termination, no decryption."
    )
    cfg = load_config()

    with st.sidebar:
        st.header("Model")
        ckpt_path = pick_checkpoint(cfg)
        if ckpt_path is None or not Path(ckpt_path).exists():
            if ckpt_path:
                st.error(f"Not found: {ckpt_path}")
            st.stop()
        model, head, tok, label_names, ckpt_cfg = _load(ckpt_path)
        st.success(f"{len(label_names)} classes")
        st.write(", ".join(label_names))
        tags = []
        if getattr(model, "_hybrid", False):
            tags.append("hybrid head (embedding + flow statistics)")
        if getattr(model, "_augment", False):
            tags.append("evasion-augmented (Claim C)")
        if tags:
            st.info(" · ".join(tags))
        st.divider()
        st.header("Data")
        df, data_path = get_source_df(cfg, label_names, getattr(model, "_data_path", None))

    if df is None:
        st.info("Pick a flow source in the sidebar to begin.")
        st.stop()

    # The checkpoint and the dataset must agree, otherwise live accuracy is meaningless.
    if "label" in df.columns and not set(df["label"]) & set(label_names):
        st.error(
            f"This dataset's classes {sorted(set(df['label']))} do not match the model's "
            f"classes {label_names}. Pick a matching dataset or retrain."
        )
        st.stop()

    t1, t2, t3, t4, t5 = st.tabs(
        [
            "Live classification",
            "Tokenization (Claim A)",
            "Embedding map (Claim B)",
            "Evasion (Claim C)",
            "Provenance",
        ]
    )
    with t1:
        tab_live(model, head, tok, label_names, df)
    with t2:
        tab_tokenize(tok, df)
    with t3:
        tab_embeddings(model, tok, label_names, df)
    with t4:
        tab_evasion(ckpt_cfg, ckpt_path, data_path, label_names)
    with t5:
        tab_provenance(cfg, ckpt_path)


if __name__ == "__main__":
    main()
