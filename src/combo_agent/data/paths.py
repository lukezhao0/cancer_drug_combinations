"""Filesystem locations for data artifacts.

Everything lives under one git-ignored root, `data_cache/` at the repo root by default:

    data_cache/raw/        downloaded files, byte-for-byte as served, plus manifest.json
    data_cache/interim/    converted or cleaned intermediates (e.g. parquet from TDC pickles)
    data_cache/processed/  the frozen pool tables and data card

Set COMBO_DATA_CACHE to put the root elsewhere (e.g. an external drive).
"""

from __future__ import annotations

import os
from pathlib import Path

#: src/combo_agent/data/paths.py -> repo root (valid for the editable install used here)
REPO_ROOT = Path(__file__).resolve().parents[3]


def data_cache() -> Path:
    """Root of all data artifacts."""
    return Path(os.environ.get("COMBO_DATA_CACHE", REPO_ROOT / "data_cache"))


def raw_dir() -> Path:
    return data_cache() / "raw"


def interim_dir() -> Path:
    return data_cache() / "interim"


def processed_dir() -> Path:
    return data_cache() / "processed"
