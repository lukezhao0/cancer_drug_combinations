"""The selector interface (target_design.md §7)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from combo_agent.lab.oracle import KEY, Result, Triplet
from combo_agent.models.response_model import Prediction
from combo_agent.models.surrogate import Surrogate

TARGETS = ("syn", "inh")


@dataclass(frozen=True)
class SelectionContext:
    """What a selector may see in one round.

    candidates  untested, selectable experiments (pool rows: keys and metadata, no outcomes)
    history     every revealed result so far (seed set first)
    model       the shared surrogate (ensemble or GP), refit on `history` this round (None for
                model-free selectors)
    pred        the model's prediction for every row of `candidates`, in the same order (or None)
    """

    candidates: pd.DataFrame
    history: Sequence[Result]
    model: Surrogate | None = None
    pred: Prediction | None = None

    def triplets(self, rows: np.ndarray) -> list[Triplet]:
        """Candidate rows (positional) as Triplets, in the given order."""
        keys = self.candidates[KEY].to_numpy()[np.asarray(rows, dtype=int)]
        return [Triplet(*map(str, k)) for k in keys]


class Selector(Protocol):
    name: str
    needs_model: bool

    def select(self, ctx: SelectionContext, batch_size: int, target: str,
               rng: np.random.Generator) -> list[Triplet]:
        """Exactly `batch_size` distinct candidates (fewer only if fewer candidates remain)."""
        ...


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k largest scores, best first; ties go to the earlier row (deterministic)."""
    k = min(k, len(scores))
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    return order[:k]
