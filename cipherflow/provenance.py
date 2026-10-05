"""Stamp every run with a provenance record.

For the patent this is the point: a dated, machine-generated record tying each artifact to
an exact code revision, config and seed. That is what turns "we built it" into evidence of
reduction to practice. It also makes a reviewer's "can you reproduce table 2" a one-liner.

Writes artifacts/run_manifest.json. Appends, so the file is a run log, not a single run.

Run:
    python -m cipherflow.provenance          # print the current stamp without recording it
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cipherflow.utils import REPO_ROOT, ensure_dir, load_config, resolve_path


def _git(*args: str) -> str | None:
    """Run a git command in the repo, returning None if git or the repo is absent."""
    try:
        out = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def config_hash(cfg: dict) -> str:
    """Stable short hash of the fully-resolved config, overrides included."""
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def _versions() -> dict[str, str]:
    names = ("torch", "numpy", "pandas", "sklearn", "xgboost", "streamlit", "dpkt")
    found = {}
    for n in names:
        try:
            found[n] = __import__(n).__version__
        except Exception:
            pass
    return found


def _device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.get_device_name(0)}"
        return "cpu"
    except Exception:
        return "unknown"


def stamp(cfg: dict | None = None, step: str = "unknown", extra: dict[str, Any] | None = None) -> dict:
    """Build the provenance record for the current process."""
    cfg = cfg if cfg is not None else load_config()
    dirty = _git("status", "--porcelain")
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "step": step,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(dirty),
        "config_hash": config_hash(cfg),
        "seed": cfg.get("seed"),
        "device": _device(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "versions": _versions(),
        **(extra or {}),
    }


def record(cfg: dict | None = None, step: str = "unknown", extra: dict[str, Any] | None = None) -> Path:
    """Append a provenance record to artifacts/run_manifest.json and return its path."""
    cfg = cfg if cfg is not None else load_config()
    rec = stamp(cfg, step, extra)
    out = resolve_path(cfg, "paths", "artifacts_dir") / "run_manifest.json"
    ensure_dir(out.parent)
    history = []
    if out.exists():
        try:
            history = json.loads(out.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # A truncated manifest should not kill a training run; start a fresh log.
            history = []
    history.append(rec)
    out.write_text(json.dumps(history[-200:], indent=2), encoding="utf-8")
    return out


def demo():
    rec = stamp(step="selfcheck")
    assert rec["config_hash"] == config_hash(load_config()), "config hash not stable"
    assert len(rec["config_hash"]) == 12
    assert rec["timestamp_utc"].endswith("+00:00"), rec["timestamp_utc"]
    assert rec["device"] in ("cpu", "unknown") or rec["device"].startswith("cuda:")
    print(json.dumps(rec, indent=2))
    print("\nprovenance selfcheck PASS")


if __name__ == "__main__":
    demo()
