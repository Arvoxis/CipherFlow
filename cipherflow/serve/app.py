"""CipherFlow live demo — a multi-tab Streamlit app for the viva / defense.

Tabs:
  1. Live classification  — replay flows, watch predictions + running accuracy.
  2. Tokenization (Claim A) — turn any flow into flow-shape tokens, with readable bins.
  3. Embedding map (Claim B) — 2-D projection of learned flow embeddings, colored by class.

Data sources: the synthetic dataset, an uploaded flows parquet, or a .pcap (extracted with
nfstream if installed).

Run:
    python -m streamlit run cipherflow/serve/app.py   (from the repo root)
"""

from __future__ import annotations

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

from cipherflow.data.flow_io import load_flows, to_flow_records
from cipherflow.finetune.clf_dataset import encode_labels
from cipherflow.inference import embed_records, load_classifier, predict_records
from cipherflow.tokenizer.inspect import tokenize_table
from cipherflow.utils import load_config, resolve_path

st.set_page_config(page_title="CipherFlow", page_icon="🔐", layout="wide")


@st.cache_resource
def _load(ckpt_path: str):
    return load_classifier(ckpt_path)


@st.cache_data
def _load_data(path: str) -> pd.DataFrame:
    return load_flows(path)


def _pcap_to_flows(uploaded, cfg) -> pd.DataFrame | None:
    """Extract flows from an uploaded pcap using nfstream (if available)."""
    try:
        from cipherflow.data.extract_flows import extract_file
    except Exception:
        st.error(
            "nfstream not installed — `pip install nfstream` to classify raw pcaps."
        )
        return None
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pcap") as tf:
        tf.write(uploaded.getbuffer())
        tmp_path = tf.name
    try:
        rows = extract_file(
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


def get_source_df(cfg, label_names) -> pd.DataFrame | None:
    src = st.radio(
        "Flow source",
        ["Sample dataset", "Upload flows parquet", "Upload .pcap"],
        horizontal=True,
    )
    if src == "Upload flows parquet":
        up = st.file_uploader("flows .parquet", type=["parquet"])
        return load_flows(up) if up else None
    if src == "Upload .pcap":
        up = st.file_uploader("capture .pcap / .pcapng", type=["pcap", "pcapng"])
        return _pcap_to_flows(up, cfg) if up else None
    # List datasets matching the loaded checkpoint's classes first, so the default just works.
    data_dir = resolve_path(cfg, "paths", "data_dir")
    found = [
        data_dir / n
        for n in ("selfcap_f12.parquet", "selfcap.parquet", "synthetic.parquet")
        if (data_dir / n).exists()
    ]
    if not found:
        st.warning(
            f"No dataset in `{data_dir}`. Run `python -m cipherflow.data.make_synthetic`."
        )
        return None
    found.sort(key=lambda p: not (set(_load_data(str(p))["label"]) & set(label_names)))
    pick = st.selectbox("Dataset", [p.name for p in found])
    return _load_data(str(data_dir / pick))


# ------------------------------------------------------------------ tabs
def tab_live(model, head, tok, label_names, df):
    st.subheader("Live classification")
    truth = df["label"].tolist() if "label" in df.columns else None
    records = to_flow_records(df)
    c1, c2 = st.columns(2)
    speed = c1.slider("Replay speed (flows/sec)", 1, 200, 40)
    batch = c2.slider("Batch per tick", 1, 50, 5)

    if not st.button("▶ Start", type="primary"):
        st.info("Press Start to stream flows through the model.")
        return

    counts = {c: 0 for c in label_names}
    correct = total = 0
    prog = st.progress(0.0)
    left, right = st.columns([1, 1])
    chart_area, feed_area, metric_area = left.empty(), right.empty(), st.empty()

    for i in range(0, len(records), batch):
        probs = predict_records(model, head, tok, records[i : i + batch])
        preds, conf = probs.argmax(-1), probs.max(-1)
        feed = []
        for j, (p, c) in enumerate(zip(preds, conf)):
            name = label_names[p]
            counts[name] += 1
            row = {"flow": i + j, "prediction": name, "confidence": f"{c:.2f}"}
            if truth is not None:
                row["actual"] = truth[i + j]
                total += 1
                correct += int(truth[i + j] == name)
            feed.append(row)
        chart_area.bar_chart(pd.Series(counts))
        feed_area.dataframe(
            pd.DataFrame(feed), hide_index=True, use_container_width=True
        )
        if truth is not None and total:
            metric_area.metric(
                "Live accuracy", f"{correct / total:.1%}", f"{total} flows seen"
            )
        prog.progress(min(1.0, (i + batch) / len(records)))
        time.sleep(batch / speed)
    st.success("Done streaming.")


def tab_tokenize(tok, df):
    st.subheader("Flow-shape tokenization — patent Claim A")
    st.caption(
        "Every packet → (quantized size, quantized inter-arrival time, direction) → one token. "
        "**No payload bytes are read.**"
    )
    idx = st.number_input("Pick a flow index", 0, max(0, len(df) - 1), 0)
    row = df.iloc[int(idx)]
    if "label" in df.columns:
        st.write(f"True label: **{row['label']}**  ·  {len(row['splt_ps'])} packets")
    table = tokenize_table(tok, row["splt_ps"], row["splt_iat"], row["splt_dir"])
    st.dataframe(table, hide_index=True, use_container_width=True)
    st.info(
        f"Composite vocabulary size: **{tok.composite_vocab_size}** "
        f"({tok.size_bins} size-bins × {tok.iat_bins} iat-bins × 2 directions + specials)"
    )


def tab_embeddings(model, tok, label_names, df):
    st.subheader("Learned flow embeddings — patent Claim B")
    st.caption(
        "2-D projection of the [CLS] flow embeddings. Well-separated clusters = a "
        "discriminative, transferable representation learned from flow shape alone."
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


def main():
    st.title("🔐 CipherFlow — Encrypted-Traffic Classifier")
    st.caption(
        "Classifies flows from **shape alone** (packet size + timing + direction). No payload inspection."
    )
    cfg = load_config()

    with st.sidebar:
        st.header("Model")
        ckpt_path = st.text_input(
            "Classifier checkpoint", str(resolve_path(cfg, "finetune", "ckpt_path"))
        )
        if not Path(ckpt_path).exists():
            st.warning(
                "No checkpoint yet. Train one:\n`python -m cipherflow.finetune.train_clf`"
            )
            st.stop()
        model, head, tok, label_names, _ = _load(ckpt_path)
        st.success(f"{len(label_names)} classes")
        st.write(", ".join(label_names))

    df = get_source_df(cfg, label_names)
    if df is None:
        st.stop()

    # The checkpoint and the dataset must agree, otherwise live accuracy is meaningless.
    if "label" in df.columns and not set(df["label"]) & set(label_names):
        st.error(
            f"This dataset's classes {sorted(set(df['label']))} do not match the model's "
            f"classes {label_names}. Pick a matching dataset or retrain."
        )
        st.stop()

    t1, t2, t3 = st.tabs(
        [
            "▶ Live classification",
            "🔡 Tokenization (Claim A)",
            "🗺 Embedding map (Claim B)",
        ]
    )
    with t1:
        tab_live(model, head, tok, label_names, df)
    with t2:
        tab_tokenize(tok, df)
    with t3:
        tab_embeddings(model, tok, label_names, df)


if __name__ == "__main__":
    main()
