"""The numeric baselines (target_design.md §7).

| name          | rule                                                                 |
|---------------|----------------------------------------------------------------------|
| random        | uniform over untested candidates (no model)                          |
| greedy        | highest predicted mean for the target                                |
| uncertainty   | highest predicted std for the target                                 |
| bo_ucb        | mean + beta*std, then diverse greedy selection (beta=2, alpha=0.5)   |
| bo_qei        | Monte-Carlo batch expected improvement over the surrogate's samples  |
| bo_ts         | batch Thompson sampling on the UCB shortlist (one draw per slot)     |

Ablation variants (not in the main grid): bo_ucb_b1, bo_ucb_b3 (beta 1 and 3), bo_ucb_nodiv
(alpha 0: plain top-k UCB, to show what the diversity penalty buys).

**Diversity penalty (bo_ucb).** M3 found that right after the seed set, UCB's top picks are the
most extreme extrapolations and pile onto a few drug pairs (16 picks on 3 pairs in a demo). So
bo_ucb takes the top `shortlist` candidates by UCB and picks greedily by maximal marginal
relevance: `(1 - alpha) * s - alpha * (max similarity to the picks so far)`. The similarity comes
from the surrogate: for the ensemble, the cosine of its last hidden layer (centered over the
shortlist, L2-normalized); for the GP, its normalized prior kernel. s is UCB
rescaled to [0, 1] over the shortlist. Rescaling bounds the score term, so a few extreme
extrapolations cannot outrun the penalty (with z-scores they did: a synthetic test kept all six
picks in one cluster at lambda = 1).

**Batch expected improvement (bo_qei).** Draw `n_samples` joint outcomes for the shortlist from the
surrogate (`sample`). For the ensemble, each sample picks one member at random and adds Gaussian
noise at that member's held-out noise level; for the GP, it is a draw from the joint posterior plus
noise. Starting from the best value revealed so far, add the
candidate that most raises the average over samples of max(best so far, batch values), and repeat.
The joint draw correlates candidates within a sample, so the batch hedges across possible worlds.

**Thompson sampling (bo_ts).** On the same UCB shortlist, draw one joint sample of the underlying
function per batch slot (no measurement noise) and give that slot the sample's best experiment not
already picked. Exact joint draws over all ~219k candidates are infeasible for the GP, so both
surrogates sample the shortlist.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from combo_agent.lab.oracle import Triplet
from combo_agent.models.prediction import cosine_similarity
from combo_agent.selectors.base import TARGETS, SelectionContext, top_k

SHORTLIST = 2000


def _check(ctx: SelectionContext, target: str, needs_pred: bool) -> None:
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}")
    if needs_pred and (ctx.pred is None or len(ctx.pred.mean_syn) != len(ctx.candidates)):
        raise ValueError("this selector needs model predictions for every candidate")


@dataclass
class RandomSelector:
    name: str = "random"
    needs_model: bool = False

    def select(self, ctx, batch_size, target, rng) -> list[Triplet]:
        _check(ctx, target, needs_pred=False)
        k = min(batch_size, len(ctx.candidates))
        return ctx.triplets(np.sort(rng.choice(len(ctx.candidates), size=k, replace=False)))


@dataclass
class GreedySelector:
    name: str = "greedy"
    needs_model: bool = True

    def select(self, ctx, batch_size, target, rng) -> list[Triplet]:
        _check(ctx, target, needs_pred=True)
        return ctx.triplets(top_k(ctx.pred.mean(target), batch_size))


@dataclass
class UncertaintySelector:
    name: str = "uncertainty"
    needs_model: bool = True

    def select(self, ctx, batch_size, target, rng) -> list[Triplet]:
        _check(ctx, target, needs_pred=True)
        return ctx.triplets(top_k(ctx.pred.std(target), batch_size))


def diverse_greedy(scores: np.ndarray, sim, k: int, alpha: float) -> np.ndarray:
    """Maximal marginal relevance: greedy picks maximizing
    (1 - alpha) * score rescaled to [0, 1] - alpha * max similarity to earlier picks.

    `sim` is either an embedding (rows x dims; cosine similarity, rows centered and L2-normalized
    here) or a function j -> similarity of every row to row j, from the surrogate's `similarity`.
    Returns positions into `scores`, in pick order.
    """
    k = min(k, len(scores))
    span = scores.max() - scores.min()
    z = (scores - scores.min()) / (span if span > 0 else 1.0)
    if alpha == 0 or k <= 1:
        return top_k(z, k)
    sim_to = sim if callable(sim) else cosine_similarity(sim)
    maxsim = np.full(len(z), -np.inf)
    taken = np.zeros(len(z), dtype=bool)
    picks = []
    for _ in range(k):
        penalty = np.where(np.isfinite(maxsim), maxsim, 0.0)
        s = np.where(taken, -np.inf, (1 - alpha) * z - alpha * penalty)
        j = int(np.argmax(s))
        picks.append(j)
        taken[j] = True
        maxsim = np.maximum(maxsim, sim_to(j))
    return np.array(picks)


@dataclass
class UCBSelector:
    name: str = "bo_ucb"
    beta: float = 2.0
    alpha: float = 0.5
    shortlist: int = SHORTLIST
    needs_model: bool = True

    def select(self, ctx, batch_size, target, rng) -> list[Triplet]:
        _check(ctx, target, needs_pred=True)
        ucb = ctx.pred.ucb(target, self.beta)
        short = top_k(ucb, max(self.shortlist, batch_size))
        if self.alpha == 0:
            return ctx.triplets(short[:batch_size])
        if ctx.model is None:
            raise ValueError("bo_ucb with a diversity penalty needs the model's similarity")
        sim = ctx.model.similarity(ctx.candidates.iloc[short], target)
        picks = diverse_greedy(ucb[short], sim, batch_size, self.alpha)
        return ctx.triplets(short[picks])


def greedy_qei(samples: np.ndarray, incumbent: float, k: int) -> np.ndarray:
    """Sequential greedy batch expected improvement over joint samples (n_samples x n_candidates)."""
    k = min(k, samples.shape[1])
    best = np.full(samples.shape[0], incumbent, dtype=float)
    taken = np.zeros(samples.shape[1], dtype=bool)
    picks = []
    for _ in range(k):
        value = np.maximum(best[:, None], samples).mean(0)
        value[taken] = -np.inf
        j = int(np.argmax(value))
        picks.append(j)
        taken[j] = True
        best = np.maximum(best, samples[:, j])
    return np.array(picks)


@dataclass
class QEISelector:
    name: str = "bo_qei"
    n_samples: int = 128
    shortlist: int = SHORTLIST
    beta: float = 2.0
    needs_model: bool = True

    def select(self, ctx, batch_size, target, rng) -> list[Triplet]:
        _check(ctx, target, needs_pred=True)
        if ctx.model is None:
            raise ValueError("bo_qei needs the model's joint samples")
        short = top_k(ctx.pred.ucb(target, self.beta), max(self.shortlist, batch_size))
        samples = ctx.model.sample(ctx.candidates.iloc[short], self.n_samples, target, rng, noise=True)
        values = [r.y_syn if target == "syn" else r.y_inh for r in ctx.history]
        incumbent = max(values) if values else -np.inf
        return ctx.triplets(short[greedy_qei(samples, incumbent, batch_size)])


@dataclass
class ThompsonSelector:
    """Batch Thompson sampling on the UCB shortlist: draw one possible world per batch slot (the
    underlying function, no measurement noise) and take its best experiment not chosen yet."""

    name: str = "bo_ts"
    shortlist: int = SHORTLIST
    beta: float = 2.0
    needs_model: bool = True

    def select(self, ctx, batch_size, target, rng) -> list[Triplet]:
        _check(ctx, target, needs_pred=True)
        if ctx.model is None:
            raise ValueError("bo_ts needs the model's joint samples")
        short = top_k(ctx.pred.ucb(target, self.beta), max(self.shortlist, batch_size))
        k = min(batch_size, len(short))
        draws = ctx.model.sample(ctx.candidates.iloc[short], k, target, rng, noise=False)
        taken = np.zeros(len(short), dtype=bool)
        picks = []
        for draw in draws:
            j = int(np.argmax(np.where(taken, -np.inf, draw)))
            picks.append(j)
            taken[j] = True
        return ctx.triplets(short[np.array(picks, dtype=int)])


SELECTORS = {
    "random": RandomSelector,
    "greedy": GreedySelector,
    "uncertainty": UncertaintySelector,
    "bo_ucb": UCBSelector,
    "bo_qei": QEISelector,
    "bo_ts": ThompsonSelector,
    "bo_ucb_b1": lambda: UCBSelector(name="bo_ucb_b1", beta=1.0),
    "bo_ucb_b3": lambda: UCBSelector(name="bo_ucb_b3", beta=3.0),
    "bo_ucb_nodiv": lambda: UCBSelector(name="bo_ucb_nodiv", alpha=0.0),
}


def make(name: str):
    if name not in SELECTORS:
        raise ValueError(f"unknown selector {name!r}; known: {', '.join(SELECTORS)}")
    return SELECTORS[name]()
