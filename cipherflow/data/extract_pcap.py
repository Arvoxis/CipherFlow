"""Self-contained pcap -> flow-shape extractor using dpkt (no nfstream needed).

Parses each packet's 5-tuple and records, per bidirectional flow, the first-N packets'
(size, inter-arrival-time, direction) — exactly CipherFlow's inputs, reading NO payload
bytes beyond L3/L4 headers. Direction 0 = from the flow's initiator (client), 1 = toward it.

This is the default extractor on Windows (nfstream's capture engine can hang there). Output
schema is identical to the synthetic generator, so everything downstream is unchanged.

Run:
    # one file, label from --label or filename stem:
    python -m cipherflow.data.extract_pcap --pcap raw/ustc/Benign/Skype.pcap --out data_out/skype.parquet
    # a directory tree (label = each file's stem), e.g. the whole USTC-TFC2016 set:
    python -m cipherflow.data.extract_pcap --dir raw/ustc --dataset ustc --out data_out/ustc.parquet
"""
from __future__ import annotations

import argparse
import glob
import uuid
from pathlib import Path

import dpkt
import numpy as np
import pandas as pd

from cipherflow.data.flow_io import save_flows
from cipherflow.utils import load_config


def _l3(buf, datalink):
    """Return the IP(4/6) layer from a raw frame, or None for non-IP frames."""
    try:
        if datalink == dpkt.pcap.DLT_EN10MB:            # Ethernet (USTC uses this)
            data = dpkt.ethernet.Ethernet(buf).data
        elif datalink == dpkt.pcap.DLT_LINUX_SLL:       # Linux cooked capture
            data = dpkt.sll.SLL(buf).data
        elif datalink in (dpkt.pcap.DLT_RAW, 12, 14):   # raw IP
            try:
                return dpkt.ip.IP(buf)
            except Exception:
                return dpkt.ip6.IP6(buf)
        else:
            data = dpkt.ethernet.Ethernet(buf).data
        return data if isinstance(data, (dpkt.ip.IP, dpkt.ip6.IP6)) else None
    except Exception:
        return None


def extract_pcap_file(path: str, label: str, dataset: str, n: int, min_packets: int) -> list[dict]:
    flows: dict = {}
    with open(path, "rb") as fp:
        try:
            reader = dpkt.pcap.Reader(fp)
        except ValueError:
            fp.seek(0)
            reader = dpkt.pcapng.Reader(fp)
        datalink = reader.datalink()
        for ts, buf in reader:
            ip = _l3(buf, datalink)
            if ip is None:
                continue
            l4 = ip.data
            if isinstance(l4, dpkt.tcp.TCP):
                proto = 6
            elif isinstance(l4, dpkt.udp.UDP):
                proto = 17
            else:
                continue
            src = (bytes(ip.src), int(l4.sport))
            dst = (bytes(ip.dst), int(l4.dport))
            key = (proto, frozenset((src, dst)))
            f = flows.get(key)
            if f is None:
                f = {"t": [], "s": [], "d": [], "client": src}
                flows[key] = f
            if len(f["s"]) >= n:          # only need the first N packets
                continue
            f["t"].append(float(ts))
            f["s"].append(len(buf))       # wire length (bytes)
            f["d"].append(0 if src == f["client"] else 1)

    rows = []
    for f in flows.values():
        m = len(f["s"])
        if m < min_packets:
            continue
        t = f["t"]
        iat = [0.0] + [max(0.0, (t[i] - t[i - 1]) * 1000.0) for i in range(1, m)]
        rows.append({
            "flow_id": uuid.uuid4().hex,
            "splt_ps": [int(x) for x in f["s"]],
            "splt_iat": [round(x, 3) for x in iat],
            "splt_dir": [int(x) for x in f["d"]],
            "n_packets": int(m),
            "label": label,
            "dataset": dataset,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcap", help="single pcap file or glob")
    ap.add_argument("--dir", help="directory tree of .pcap/.pcapng (label = each file's stem)")
    ap.add_argument("--label", default=None, help="label override (default: filename stem)")
    ap.add_argument("--dataset", default="pcap")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-packets", type=int, default=4)
    ap.add_argument("--set", dest="overrides", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(overrides=args.overrides)
    n = cfg["flow"]["seq_len"]

    files: list[str] = []
    if args.dir:
        for ext in ("*.pcap", "*.pcapng"):
            files += [str(p) for p in Path(args.dir).rglob(ext)]
    elif args.pcap:
        files = sorted(glob.glob(args.pcap))
    if not files:
        raise SystemExit("no pcap files matched")

    all_rows = []
    for fpath in sorted(files):
        label = args.label or Path(fpath).stem
        rows = extract_pcap_file(fpath, label, args.dataset, n, args.min_packets)
        print(f"  {Path(fpath).name:28s} -> {len(rows):6d} flows (label='{label}')")
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    save_flows(df, args.out)
    print(f"\nWrote {len(df)} flows across {df['label'].nunique()} classes -> {args.out}")
    print(df["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
