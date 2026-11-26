"""LEGACY BUILD (superseded by the Gaussian process, `models/gp.py`; old build tagged `ensemble-v1`).

The first response model (target_design.md §6): a deep ensemble that predicts both outcomes with
uncertainty, refit from scratch on a campaign's revealed history every round.

Every selector uses the same model with the same settings, so differences between selectors come
from how they choose experiments, not from a better predictor.

Per member (5 members; different random initialization and a different 20% early-stopping split):

    input [pair block, cell block] -> 256 ReLU -> 256 ReLU -> 2 outputs (y_syn, y_inh means)

* Loss: mean squared error on standardized targets; AdamW, gradient-norm clipping; early stopping
  on the member's held-out 20% (validation MSE).
* **Noise scale** per member and target, from its residuals on its own held-out 20%: an
  out-of-sample error estimate, so it includes the error from too little data. Default
  (`noise="quantile"`): the Gaussian sd whose 80% interval matches the 80th percentile of held-out
  |residual|. Bliss and CSS are heavy-tailed, so the mean squared residual (`noise="mse"`) made
  intervals too wide for typical points (coverage 0.91 vs 0.87 at n = 320; M3 report).
* Prediction: mean = average of member means; variance = variance of the member means (the members
  disagree where data are thin) + average held-out residual variance (noise and fit error).
* Targets are standardized with the mean and SD of the history being fit (design: seed-set
  statistics in round 0). With fewer than 10 rows there is no validation split; the noise variance
  is then the prior (the history's own variance).
* `std_scale` multiplies the std per target (1 by default; see the M3 calibration report).

**Deviation from the design (§6), on evidence.** The design gives each member a learned per-row
variance head (Gaussian NLL). At campaign sizes (32-352 rows) the network memorizes its training
rows and the learned variance collapses toward zero: on a planted test function, validation NLL
rose from -0.1 to 84 while validation MSE kept falling, i.e. confident and wrong. Held-out residual
variance cannot be memorized. Heteroscedastic noise can be revisited with more data.

The first layer is applied to the pair block once per drug pair and to the cell block once per
line, then summed per experiment. This is exact (the layer is linear) and keeps scoring the
~219k-candidate pool to seconds without building a ~5 GB input matrix.

Implemented in numpy (no deep-learning dependency); gradients are verified against finite
differences in `tests/test_legacy_ensemble.py`.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Protocol

import numpy as np
import pandas as pd
import scipy.sparse as sp

from combo_agent.lab.oracle import KEY, Result, Triplet
from combo_agent.models.legacy.ensemble_features import FeatureSet
from combo_agent.models.prediction import Prediction, cosine_similarity

TARGETS = ("syn", "inh")
CHUNK = 65_536
Z80 = 1.2815515655446004  # standard-normal 90th percentile: the 80% interval is mean +- Z80 * std
MIN_VAL_ROWS = 10


class ResponseModel(Protocol):
    def fit(self, history: Sequence[Result]) -> ResponseModel: ...
    def predict(self, experiments) -> Prediction: ...
    def embed(self, experiments) -> np.ndarray: ...


def _scatter(index: np.ndarray, n_groups: int) -> sp.csr_matrix:
    """Sparse (groups x rows) indicator, so `S @ grad_rows` sums row gradients per group."""
    return sp.csr_matrix((np.ones(len(index)), (index, np.arange(len(index)))), shape=(n_groups, len(index)))


class _MLP:
    """One ensemble member: parameters, factorized forward pass and its exact gradient."""

    NAMES = ("Wp", "Wc", "b1", "W2", "b2", "W3", "b3")

    def __init__(self, pair_dim: int, cell_dim: int, hidden: int, rng: np.random.Generator):
        fan_in = pair_dim + cell_dim
        he = lambda n_in, shape: (rng.standard_normal(shape) * np.sqrt(2.0 / n_in)).astype(np.float32)  # noqa: E731
        self.p = {
            "Wp": he(fan_in, (pair_dim, hidden)), "Wc": he(fan_in, (cell_dim, hidden)),
            "b1": np.zeros(hidden, np.float32), "W2": he(hidden, (hidden, hidden)),
            "b2": np.zeros(hidden, np.float32),
            "W3": (rng.standard_normal((hidden, 2)) * np.sqrt(1.0 / hidden)).astype(np.float32),
            "b3": np.zeros(2, np.float32),
        }
        self.noise_var = np.ones(2)

    def forward(self, P, C, pi, ci, keep: bool = False):
        p = self.p
        z1 = (P @ p["Wp"])[pi] + (C @ p["Wc"])[ci] + p["b1"]
        h1 = np.maximum(z1, 0)
        z2 = h1 @ p["W2"] + p["b2"]
        h2 = np.maximum(z2, 0)
        mu = h2 @ p["W3"] + p["b3"]
        return (mu, (z1, h1, z2, h2)) if keep else (mu, h2)

    def loss_and_grad(self, P, C, pi, ci, y):
        """Mean squared error over rows and both targets, and its gradient."""
        mu, (z1, h1, z2, h2) = self.forward(P, C, pi, ci, keep=True)
        r = mu - y
        loss = float(np.mean(r * r))
        do = 2.0 * r / r.size
        p, g = self.p, {}
        g["W3"], g["b3"] = h2.T @ do, do.sum(0)
        dz2 = (do @ p["W3"].T) * (z2 > 0)
        g["W2"], g["b2"] = h1.T @ dz2, dz2.sum(0)
        dz1 = (dz2 @ p["W2"].T) * (z1 > 0)
        g["b1"] = dz1.sum(0)
        g["Wp"] = P.T @ (_scatter(pi, len(P)) @ dz1)
        g["Wc"] = C.T @ (_scatter(ci, len(C)) @ dz1)
        return loss, {k: np.asarray(v, dtype=p[k].dtype) for k, v in g.items()}


class EnsembleModel:
    """Deep ensemble response model. See the module docstring."""

    def __init__(self, features: FeatureSet | None = None, mode: str = "full", n_members: int = 5,
                 hidden: int = 256, seed: int = 0, lr: float = 1e-3, weight_decay: float = 1e-4,
                 batch_size: int = 4096, max_epochs: int | None = None, patience: int | None = None,
                 val_frac: float = 0.2, clip: float = 10.0, std_scale: dict[str, float] | None = None,
                 noise: str = "quantile"):
        self.features = features if features is not None else FeatureSet.from_cache(mode)
        self.n_members, self.hidden, self.seed = n_members, hidden, seed
        self.lr, self.weight_decay, self.batch_size, self.clip = lr, weight_decay, batch_size, clip
        self.max_epochs, self.patience, self.val_frac = max_epochs, patience, val_frac
        if noise not in ("quantile", "mse"):
            raise ValueError("noise must be 'quantile' or 'mse'")
        self.noise = noise
        self.std_scale = dict(std_scale or {"syn": 1.0, "inh": 1.0})
        self.members: list[_MLP] = []
        self.fit_seconds = float("nan")
        self.epochs_run: list[int] = []

    # -- inputs -----------------------------------------------------------------------------------

    def _encode(self, experiments) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(experiments, pd.DataFrame):
            return self.features.encode(experiments["drug_a"], experiments["drug_b"], experiments["cellosaurus_ac"])
        trips = [t if isinstance(t, Triplet) else Triplet(*t) for t in experiments]
        return self.features.encode([t.drug_a for t in trips], [t.drug_b for t in trips],
                                    [t.cellosaurus_ac for t in trips])

    # -- fitting ----------------------------------------------------------------------------------

    def fit(self, history: Sequence[Result]) -> EnsembleModel:
        """Refit from scratch on revealed results."""
        if not history:
            raise ValueError("cannot fit on an empty history")
        if not all(isinstance(r, Result) for r in history):
            raise TypeError("fit takes oracle Results (revealed experiments only)")
        frame = pd.DataFrame([(*r.triplet, r.y_syn, r.y_inh) for r in history], columns=[*KEY, "y_syn", "y_inh"])
        return self.fit_frame(frame)

    def fit_frame(self, frame: pd.DataFrame) -> EnsembleModel:
        """Refit on a frame with the triplet keys plus y_syn and y_inh (evaluation code uses this)."""
        t0 = time.perf_counter()
        codes, lines = self._encode(frame)
        y = frame[["y_syn", "y_inh"]].to_numpy(np.float64)
        self.y_mu = y.mean(0)
        self.y_sd = np.where(y.std(0) > 1e-8, y.std(0), 1.0)
        ys = ((y - self.y_mu) / self.y_sd).astype(np.float32)
        uniq, pi = np.unique(codes, return_inverse=True)
        P, C = self.features.pair_block(uniq), self.features.cell_x
        n = len(ys)
        full_batch = n <= self.batch_size
        max_epochs = self.max_epochs or (2000 if full_batch else 60)
        patience = self.patience or (200 if full_batch else 6)
        self.members, self.epochs_run = [], []
        for m in range(self.n_members):
            rng = np.random.default_rng([self.seed, m])
            net = _MLP(P.shape[1], C.shape[1], self.hidden, rng)
            order = rng.permutation(n)
            n_val = int(round(self.val_frac * n)) if n >= MIN_VAL_ROWS else 0
            val, tr = order[:n_val], order[n_val:]
            self.epochs_run.append(self._train(net, P, C, pi, lines, ys, tr, val, rng, max_epochs, patience))
            self.members.append(net)
        self.fit_seconds = time.perf_counter() - t0
        return self

    def _train(self, net, P, C, pi, ci, y, tr, val, rng, max_epochs, patience) -> int:
        state = {k: (np.zeros_like(v), np.zeros_like(v)) for k, v in net.p.items()}
        b1, b2, eps, t = 0.9, 0.999, 1e-8, 0
        best, best_p, best_epoch, best_res, stale = np.inf, None, 0, None, 0
        bs = min(self.batch_size, len(tr))
        for epoch in range(1, max_epochs + 1):
            perm = rng.permutation(tr) if bs < len(tr) else tr
            for start in range(0, len(perm), bs):
                rows = perm[start:start + bs]
                _, g = net.loss_and_grad(P, C, pi[rows], ci[rows], y[rows])
                norm = np.sqrt(sum(float((v * v).sum()) for v in g.values()))
                scale = min(1.0, self.clip / (norm + 1e-12))
                t += 1
                for k, w in net.p.items():
                    m_, v_ = state[k]
                    gk = g[k] * scale
                    m_ *= b1; m_ += (1 - b1) * gk
                    v_ *= b2; v_ += (1 - b2) * gk * gk
                    if k.startswith("W"):
                        w -= self.lr * self.weight_decay * w
                    w -= (self.lr * (m_ / (1 - b1 ** t)) / (np.sqrt(v_ / (1 - b2 ** t)) + eps)).astype(np.float32)
            if len(val) == 0:
                continue
            mu, _ = net.forward(P, C, pi[val], ci[val])
            res = y[val] - mu
            score = float((res ** 2).mean())
            if score < best - 1e-6:
                best, best_p, best_epoch, best_res, stale = score, {k: v.copy() for k, v in net.p.items()}, epoch, res, 0
            else:
                stale += 1
                if stale >= patience:
                    break
        if best_p is None:  # no validation rows: keep the last weights, prior noise (standardized units)
            net.noise_var = np.ones(2)
            return max_epochs
        net.p = best_p
        r = best_res.astype(np.float64)
        if self.noise == "mse":
            noise_var = (r ** 2).mean(0)
        else:  # Gaussian sd whose 80% interval matches the held-out 80% absolute error (robust to tails)
            noise_var = (np.quantile(np.abs(r), 0.8, axis=0) / Z80) ** 2
        net.noise_var = np.maximum(noise_var, 1e-4)
        return best_epoch

    # -- prediction -------------------------------------------------------------------------------

    def _member_outputs(self, experiments):
        if not self.members:
            raise RuntimeError("fit the model first")
        codes, lines = self._encode(experiments)
        uniq, pi = np.unique(codes, return_inverse=True)
        P, C = self.features.pair_block(uniq), self.features.cell_x
        for net in self.members:
            ep, ec = P @ net.p["Wp"], C @ net.p["Wc"]
            mus, embs = [], []
            for s in range(0, len(pi), CHUNK):
                sl = slice(s, s + CHUNK)
                h1 = np.maximum(ep[pi[sl]] + ec[lines[sl]] + net.p["b1"], 0)
                h2 = np.maximum(h1 @ net.p["W2"] + net.p["b2"], 0)
                mus.append(h2 @ net.p["W3"] + net.p["b3"]); embs.append(h2)
            yield net, np.concatenate(mus), embs

    def predict(self, experiments) -> Prediction:
        """Mean and std of y_syn and y_inh for experiments (Triplets, tuples or a pool frame)."""
        mus, noise = [], []
        for net, mu, _ in self._member_outputs(experiments):
            mus.append(mu.astype(np.float64)); noise.append(net.noise_var)
        mu = np.stack(mus)
        mean = mu.mean(0) * self.y_sd + self.y_mu
        std = np.sqrt(mu.var(0) + np.mean(noise, axis=0)) * self.y_sd
        latent = np.sqrt(mu.var(0)) * self.y_sd
        noise_sd = np.broadcast_to(np.sqrt(np.mean(noise, axis=0)) * self.y_sd, latent.shape)
        s_syn, s_inh = self.std_scale["syn"], self.std_scale["inh"]
        return Prediction(mean[:, 0], std[:, 0] * s_syn, mean[:, 1], std[:, 1] * s_inh,
                          latent[:, 0] * s_syn, noise_sd[:, 0] * s_syn, latent[:, 1] * s_inh, noise_sd[:, 1] * s_inh)

    def member_means(self, experiments) -> tuple[np.ndarray, np.ndarray]:
        """Each member's predicted means (members x n x 2, original units, columns syn and inh) and
        its noise sd (members x 2, original units, before `std_scale`). For sampling-based
        selectors such as Monte-Carlo batch expected improvement."""
        mus, noise = [], []
        for net, mu, _ in self._member_outputs(experiments):
            mus.append(mu.astype(np.float64) * self.y_sd + self.y_mu)
            noise.append(np.sqrt(net.noise_var) * self.y_sd)
        return np.stack(mus), np.stack(noise)

    def embed(self, experiments) -> np.ndarray:
        """Member-averaged last hidden layer (n x hidden), for batch diversity."""
        total = None
        for _, _, embs in self._member_outputs(experiments):
            e = np.concatenate(embs).astype(np.float32)
            total = e if total is None else total + e
        return total / len(self.members)

    # -- the Surrogate interface (models/surrogate.py) -------------------------------------------

    name = "ensemble"

    def sample(self, experiments, n_samples: int, target: str, rng: np.random.Generator,
               noise: bool = True) -> np.ndarray:
        """Joint draws (n_samples x rows) in original units: each draw takes one member at random
        and, with `noise`, adds Gaussian noise at that member's noise sd. The random draws are made
        in the same order as the original bo_qei code, so its picks are unchanged."""
        means, noise_sd = self.member_means(experiments)
        col = TARGETS.index(target)
        members = rng.integers(0, means.shape[0], size=n_samples)
        draws = means[members, :, col]
        if noise:
            draws = draws + noise_sd[members, col][:, None] * rng.standard_normal((n_samples, means.shape[1]))
        return draws

    def similarity(self, experiments, target: str | None = None):
        """Similarity between experiments for the diversity penalty, as a function j -> column j:
        cosine of the member-averaged embeddings, centered over `experiments`."""
        return cosine_similarity(self.embed(experiments))
