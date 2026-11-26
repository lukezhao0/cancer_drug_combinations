"""What every surrogate returns, shared by the surrogates, the selectors and the diagnostics.

    Prediction         per-experiment predicted mean and predictive std of both outcomes
                       (optionally split into latent std and noise std)
    cosine_similarity  j -> cosine similarity of every row of an embedding to row j
                       (the diversity penalty of `bo_ucb` for embedding-based surrogates)

Moved verbatim from the former `models/response_model.py` when the Gaussian process became the
default surrogate and the ensemble moved to `models/legacy/`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Prediction:
    """Per-experiment predicted mean and std of both outcomes, in original units."""

    mean_syn: np.ndarray
    std_syn: np.ndarray
    mean_inh: np.ndarray
    std_inh: np.ndarray
    # Optional split of `std` (the predictive std, which selectors use) into the model's own
    # uncertainty about the mean ("latent") and the measurement-noise level. For diagnostics.
    latent_std_syn: np.ndarray | None = None
    noise_std_syn: np.ndarray | None = None
    latent_std_inh: np.ndarray | None = None
    noise_std_inh: np.ndarray | None = None

    def mean(self, target: str) -> np.ndarray:
        return self.mean_syn if target == "syn" else self.mean_inh

    def std(self, target: str) -> np.ndarray:
        return self.std_syn if target == "syn" else self.std_inh

    def latent_std(self, target: str) -> np.ndarray | None:
        return self.latent_std_syn if target == "syn" else self.latent_std_inh

    def noise_std(self, target: str) -> np.ndarray | None:
        return self.noise_std_syn if target == "syn" else self.noise_std_inh

    def ucb(self, target: str, beta: float = 2.0) -> np.ndarray:
        return self.mean(target) + beta * self.std(target)


def cosine_similarity(emb: np.ndarray):
    """j -> cosine similarity of every row of `emb` to row j (rows centered, then L2-normalized)."""
    e = emb - emb.mean(0)
    e = e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
    return lambda j: e @ e[j]
