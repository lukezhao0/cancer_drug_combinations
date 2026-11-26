"""Feature helpers shared by the surrogates, built from the frozen feature tables only (no outcomes).

    _numeric      numeric feature columns with given prefixes (e.g. `fp_`, `target_`, `expr_`)
    _pca          top-k principal-component scores, deterministic sign
    _standardize  column mean / SD (SD 1 for constant columns) and the non-constant mask

Used by the GP's features (`models/gp.py:GPFeatures`) and by the legacy ensemble's
`models/legacy/ensemble_features.py:FeatureSet`. `tests/test_label_access.py` enforces that no
outcome is read here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _numeric(df: pd.DataFrame, prefixes: tuple[str, ...], exclude: tuple[str, ...] = ()) -> pd.DataFrame:
    cols = [c for c in df.columns if c.startswith(prefixes) and not c.startswith(exclude)
            and pd.api.types.is_numeric_dtype(df[c])]
    return df[cols].astype(np.float32)


def _pca(x: np.ndarray, k: int) -> np.ndarray:
    """Top-k principal-component scores (deterministic sign: largest loading positive)."""
    xc = x - x.mean(axis=0)
    u, s, vt = np.linalg.svd(xc, full_matrices=False)
    k = min(k, len(s))
    signs = np.sign(vt[np.arange(k), np.abs(vt[:k]).argmax(axis=1)])
    signs[signs == 0] = 1
    return (u[:, :k] * s[:k] * signs).astype(np.float32)


def _standardize(x: np.ndarray, ref: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ref = x if ref is None else ref
    mu, sd = ref.mean(axis=0), ref.std(axis=0)
    keep = sd > 1e-6
    return mu, np.where(keep, sd, 1.0), keep
