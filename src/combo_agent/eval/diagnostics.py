"""Per-round surrogate diagnostics, logged during a campaign without reading any hidden outcome.

Each round, after the surrogate predicts every candidate and before the batch is run, record its
predictions for two sets of experiments:

    batch   the experiments the selector picked (their outcomes are revealed right after)
    probe   a fixed random sample of 2,000 selectable experiments, drawn once per seed and the same
            for every selector and surrogate; rows already tested drop out

For each row: predicted mean, predictive std, latent std (the model's own uncertainty about the
mean), noise std, and the distance to the nearest experiment already in the history.

Distance is model-agnostic, so both surrogates are measured on the same ruler:

    pair similarity  = best-order mean of fingerprint Tanimoto: max((T[a,a'] + T[b,b']) / 2,
                                                                   (T[a,b'] + T[b,a']) / 2)
    cell similarity  = (1 + cosine of expression principal components) / 2
    distance         = 1 - max over history of (pair similarity x cell similarity)

0 means an exact repeat of a tested experiment; 1 means nothing similar has been tested. The
outcomes of `batch` rows are joined later from the campaign's reveals; the outcomes of `probe` rows
only by allowlisted evaluation code (`eval/compare.py`).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from combo_agent.lab.oracle import KEY, Result
from combo_agent.models.gp import GPFeatures
from combo_agent.models.response_model import Prediction

N_PROBE = 2000
PROBE_SALT = 7_777


class DistanceSpace:
    """Nearest-history distance on fixed drug and cell similarities (see the module docstring)."""

    def __init__(self, features: GPFeatures):
        self.features = features
        e = features.cell_pcs / np.maximum(np.linalg.norm(features.cell_pcs, axis=1, keepdims=True), 1e-12)
        self.cell_sim = (1.0 + e @ e.T) / 2.0
        self.t_fp = features.t_fp

    def nearest(self, query: np.ndarray, history: np.ndarray) -> np.ndarray:
        """Distance (rows,) from each encoded query row to its most similar encoded history row."""
        if len(history) == 0:
            return np.ones(len(query))
        t = self.t_fp
        qa, qb, qc = query[:, 0][:, None], query[:, 1][:, None], query[:, 2][:, None]
        ha, hb, hc = history[:, 0][None, :], history[:, 1][None, :], history[:, 2][None, :]
        pair = np.maximum((t[qa, ha] + t[qb, hb]) / 2, (t[qa, hb] + t[qb, ha]) / 2)
        sim = pair * self.cell_sim[qc, hc]
        return 1.0 - sim.max(axis=1)


def probe_keys(candidates: pd.DataFrame, seed: int, n: int = N_PROBE) -> pd.DataFrame:
    """The fixed probe set: n random rows of the round-1 candidates, from a seed-only rng."""
    rng = np.random.default_rng([seed, PROBE_SALT])
    rows = np.sort(rng.choice(len(candidates), size=min(n, len(candidates)), replace=False))
    return candidates.iloc[rows][KEY].reset_index(drop=True)


def round_records(rnd: int, candidates: pd.DataFrame, pred: Prediction, batch_rows: np.ndarray,
                  probe: pd.DataFrame, history: Sequence[Result], space: DistanceSpace) -> pd.DataFrame:
    """Diagnostic rows for one round (see the module docstring)."""
    pos = pd.Series(np.arange(len(candidates)), index=pd.MultiIndex.from_frame(candidates[KEY]))
    probe_rows = pos.reindex(pd.MultiIndex.from_frame(probe)).dropna().to_numpy(dtype=int)
    rows = np.concatenate([np.asarray(batch_rows, dtype=int), probe_rows])
    hist = space.features.encode([r.triplet for r in history])
    q = space.features.encode(candidates.iloc[rows])
    out = candidates.iloc[rows][KEY].reset_index(drop=True)
    out.insert(0, "set", ["batch"] * len(batch_rows) + ["probe"] * len(probe_rows))
    out.insert(0, "round", rnd)
    for t in ("syn", "inh"):
        out[f"mean_{t}"] = pred.mean(t)[rows]
        out[f"std_{t}"] = pred.std(t)[rows]
        lat, noi = pred.latent_std(t), pred.noise_std(t)
        out[f"latent_std_{t}"] = lat[rows] if lat is not None else np.nan
        out[f"noise_std_{t}"] = noi[rows] if noi is not None else np.nan
    out["dist"] = space.nearest(q, hist)
    return out
