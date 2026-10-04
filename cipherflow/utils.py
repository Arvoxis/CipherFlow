"""Shared utilities: config loading, seeding, device selection, tiny logging.

Kept dependency-light on purpose so every module can import it cheaply.
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# Windows consoles default to cp1252, which crashes on non-ASCII prints (arrows, em-dashes,
# emoji). Every CipherFlow script imports this module, so reconfigure stdout/stderr to UTF-8
# here once — errors="replace" guarantees a print never raises, even on a legacy console.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Repository root = two levels up from this file (cipherflow/utils.py -> repo root).
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "default.yaml"


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> dict[str, Any]:
    """Load YAML config and apply ``key.subkey=value`` CLI overrides.

    Example: ``--set pretrain.epochs=5 model.d_model=64`` -> overrides=["pretrain.epochs=5", ...]
    Values are parsed with YAML so ``true``/``3``/``0.1`` get proper types.
    """
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Bad override '{item}', expected key.path=value")
        key, raw = item.split("=", 1)
        value = yaml.safe_load(raw)  # parse into int/float/bool/str
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return cfg


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (if available) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device():
    """Return the best available torch device (cuda if present, else cpu)."""
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_path(cfg: dict, *keys: str) -> Path:
    """Resolve a path from config relative to the repo root, creating parents."""
    p = cfg
    for k in keys:
        p = p[k]
    out = (REPO_ROOT / p).resolve()
    return out


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
