"""The surrogate interface: what every response model offers the campaign runner and the selectors.

A surrogate is a cheap stand-in for the lab. Refit on the revealed history each round, it gives a
guess and an uncertainty for every untested experiment. Two implementations:

    ensemble   models/response_model.py:EnsembleModel    5 small neural networks (the M3 model)
    gp         models/gp.py:GPModel                       Gaussian process with a Tanimoto pair kernel

Choose one with `--surrogate` on `python -m combo_agent.eval.runner`. The selectors only use the
methods below, so they treat both the same way.

    fit(history)                              refit on oracle Results (revealed experiments only)
    predict(experiments) -> Prediction        mean and predictive std of both outcomes per experiment
                                              (+ the split into latent std and noise std)
    sample(experiments, n, target, rng,       joint draws (n x rows) for batch selection
           noise=True)                        (noise=False draws the underlying function only)
    similarity(experiments, target)           j -> similarity of every row to row j (diversity penalty)

`fit` takes `Result` objects rather than raw (X, y) arrays, so no selector can hand a model labels
from anywhere but the oracle.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol

import numpy as np

from combo_agent.lab.oracle import Result
from combo_agent.models.response_model import Prediction

SURROGATES = ("ensemble", "gp")


class Surrogate(Protocol):
    name: str

    def fit(self, history: Sequence[Result]) -> Surrogate: ...

    def predict(self, experiments) -> Prediction: ...

    def sample(self, experiments, n_samples: int, target: str, rng: np.random.Generator,
               noise: bool = True) -> np.ndarray: ...

    def similarity(self, experiments, target: str | None = None) -> Callable[[int], np.ndarray]: ...


def make_surrogate(name: str, features=None, seed: int = 0, **kw) -> Surrogate:
    """A fresh, unfitted surrogate. `features` is a FeatureSet (ensemble) or GPFeatures (gp)."""
    if name == "ensemble":
        from combo_agent.models.response_model import EnsembleModel
        return EnsembleModel(features, seed=seed, **kw)
    if name == "gp":
        from combo_agent.models.gp import GPModel
        return GPModel(features, seed=seed, **kw)
    raise ValueError(f"unknown surrogate {name!r}; known: {', '.join(SURROGATES)}")
