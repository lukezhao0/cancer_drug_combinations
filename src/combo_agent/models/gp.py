"""Gaussian-process surrogate: an alternative to the 5-network ensemble (models/response_model.py).

A Gaussian process (GP) is a prior over functions. Two experiments that are similar under a
*kernel* (a similarity rule) get correlated outcomes. After seeing data, the GP's guess at an
untested experiment is a weighted blend of the outcomes of similar tested ones, and its uncertainty
shrinks near the data and grows back to the prior level far from it.

**Inputs** (`GPFeatures`, built once from the feature tables; no outcome is read):

* drug structure: Morgan fingerprint bits (radius 2, 2,048 bits), the `fp_*` columns;
* drug targets: ChEMBL target-gene multi-hots, the `target_*` columns (no pathway annotations exist
  in the data, so pathways are not used);
* cell line: the top 16 principal components of the 1,000 most variable expression genes,
  standardized. PCA is fit on all 57 lines' expression (no outcomes);
* no dose input: the outcomes are per-experiment summary scores, not dose responses.

**Kernel.** An experiment is (drug a, drug b, cell line c), passed to the GP as indices.

    k_drug(i, j)  = pi * Tanimoto_fp(i, j) + (1 - pi) * Tanimoto_targets(i, j)     pi learned in (0, 1)
    k_pair((a,b), (a',b')) = k_drug(a,a') k_drug(b,b') + k_drug(a,b') k_drug(b,a')   (order-free)
    k_cell(c, c') = Matern-5/2 with one lengthscale per principal component (ARD)

    syn (Bliss, already a synergy score):   K = s_int * k_pair * k_cell
    inh (CSS, a killing score):             K = s_int * k_pair * k_cell
                                              + s_single * [k_drug(a,a') + k_drug(a,b') + k_drug(b,a')
                                                            + k_drug(b,b')] * k_cell

The second inh term is the kernel of g(a, c) + g(b, c): each drug's own effect on the line, the
part of killing that does not depend on the partner. Bliss already removes the single-drug effects,
so syn has no such term. The Tanimoto matrices are computed once with GAUCHE's `TanimotoKernel`. A
drug with no target gene gets target similarity 1 to itself and 0 to every other drug, so two
untargeted drugs are not called identical.

**Likelihood.** Gaussian noise. Its variance sigma^2 is learned, and divided by `n_reps` for
experiments measured twice (their value is a mean of two replicates). Constant mean. y is
standardized on the revealed history every round.

**Fitting.** Exact marginal likelihood, maximized with BoTorch's scipy L-BFGS-B fitter, from 5
starts (one at the defaults, four drawn from the priors), keeping the best. Mild priors: LogNormal on
the cell lengthscales (BoTorch's dimension-scaled default), Gamma(2, 4) on the noise and on the
output scales (mean 0.5 in standardized units). Everything is float64. Cholesky jitter starts at
1e-6 and grows to 1e-4 if a factorization fails.

**Prediction** is closed form from the fitted Cholesky factor, in chunks, with the cross-kernel
built from small lookup tables (pairs x history, lines x history), so the ~219k-candidate pool
takes seconds. Exact GPs are used up to 5,000 history rows; campaigns stop at 352. Above that the
model refuses (a sparse variational GP would be needed and is not built).
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.linalg import cho_solve, solve_triangular

from combo_agent.data import paths
from combo_agent.lab.oracle import KEY, Result, Triplet
from combo_agent.models.features import _numeric, _pca
from combo_agent.models.response_model import Prediction

TARGETS = ("syn", "inh")
CELL_PCS = 16
MAX_EXACT = 5000
CHUNK = 32_768
JITTERS = (1e-6, 1e-5, 1e-4)
SINGLE_DRUG_TERMS = {"syn": False, "inh": True}


# -- inputs -----------------------------------------------------------------------------------------

def tanimoto(bits: np.ndarray, empty_is_identity: bool = True) -> np.ndarray:
    """Tanimoto similarity matrix of binary rows, via GAUCHE's kernel. Rows with no bits set get
    similarity 1 to themselves and 0 to all others (GAUCHE's eps would call any two empty rows
    identical)."""
    from gauche.kernels.fingerprint_kernels.tanimoto_kernel import TanimotoKernel

    x = torch.as_tensor(bits, dtype=torch.float64)
    with torch.no_grad():
        t = TanimotoKernel()(x, x).to_dense().numpy().copy()
    empty = bits.sum(1) == 0
    if empty_is_identity and empty.any():
        t[empty, :] = 0.0
        t[:, empty] = 0.0
        t[empty, empty] = 1.0
    t = 0.5 * (t + t.T)
    np.fill_diagonal(t, 1.0)
    return t


@dataclass
class GPFeatures:
    """Fixed inputs of the GP: drug similarity matrices and cell-line principal components."""

    drug_ids: list[str]
    t_fp: np.ndarray          # (n_drugs, n_drugs) fingerprint Tanimoto
    t_tg: np.ndarray          # (n_drugs, n_drugs) target-gene Tanimoto
    line_ids: list[str]
    cell_pcs: np.ndarray      # (n_lines, CELL_PCS) standardized expression principal components
    n_empty_targets: int = 0

    def __post_init__(self) -> None:
        self._drug_index = {d: i for i, d in enumerate(self.drug_ids)}
        self._line_index = {c: i for i, c in enumerate(self.line_ids)}

    def encode(self, experiments) -> np.ndarray:
        """(rows, 3) int array of (drug a, drug b, line) indices with a < b (order-free anyway)."""
        if isinstance(experiments, pd.DataFrame):
            a, b, c = (experiments[k].to_numpy() for k in KEY)
        else:
            trips = [t if isinstance(t, Triplet) else Triplet(*t) for t in experiments]
            a, b, c = ([getattr(t, k) for t in trips] for k in KEY)
        try:
            ia = np.fromiter((self._drug_index[d] for d in a), dtype=np.int64)
            ib = np.fromiter((self._drug_index[d] for d in b), dtype=np.int64)
            ic = np.fromiter((self._line_index[x] for x in c), dtype=np.int64)
        except KeyError as e:
            raise KeyError(f"{e.args[0]!r} has no GP features") from None
        return np.stack([np.minimum(ia, ib), np.maximum(ia, ib), ic], axis=1)

    @classmethod
    def build(cls, drugs: pd.DataFrame, cells: pd.DataFrame, pairs: pd.DataFrame, n_pcs: int = CELL_PCS) -> GPFeatures:
        drugs = drugs.set_index("drug_id")
        used = sorted(set(pairs["drug_a"]) | set(pairs["drug_b"]))
        drugs = drugs.loc[used]
        fp = (_numeric(drugs, ("fp_",)).to_numpy() > 0).astype(np.float64)
        tg = (_numeric(drugs, ("target_",)).to_numpy() > 0).astype(np.float64)
        cells = cells.set_index("cellosaurus_ac").sort_index()
        expr = _numeric(cells, ("expr_",)).to_numpy().astype(np.float64)
        pcs = _pca(expr, n_pcs).astype(np.float64)
        sd = pcs.std(0)
        pcs = (pcs - pcs.mean(0)) / np.where(sd > 1e-12, sd, 1.0)
        return cls(used, tanimoto(fp), tanimoto(tg), list(cells.index), pcs, int((tg.sum(1) == 0).sum()))

    @classmethod
    def from_cache(cls, root: Path | None = None) -> GPFeatures:
        root = root or paths.processed_dir()
        pool = pd.read_parquet(root / "pool.parquet", columns=["drug_a", "drug_b"]).drop_duplicates()
        return cls.build(pd.read_parquet(root / "drugs.parquet"), pd.read_parquet(root / "cells.parquet"), pool)


# -- the GPyTorch model -------------------------------------------------------------------------------

def _gp_classes():
    """Built lazily so importing this module (e.g. for GPFeatures) stays cheap."""
    import gpytorch
    from gpytorch.constraints import GreaterThan, Interval
    from gpytorch.kernels import Kernel, MaternKernel
    from gpytorch.likelihoods import _GaussianLikelihoodBase
    from gpytorch.likelihoods.noise_models import Noise
    from gpytorch.priors import GammaPrior, LogNormalPrior
    from linear_operator.operators import DiagLinearOperator

    class ComboKernel(Kernel):
        """Order-free drug-pair x cell-line kernel on index inputs (a, b, c, n_reps)."""

        has_lengthscale = False

        def __init__(self, t_fp, t_tg, cell_pcs, single: bool):
            super().__init__()
            self.register_buffer("t_fp", torch.as_tensor(t_fp, dtype=torch.float64))
            self.register_buffer("t_tg", torch.as_tensor(t_tg, dtype=torch.float64))
            self.register_buffer("cell_pcs", torch.as_tensor(cell_pcs, dtype=torch.float64))
            d = self.cell_pcs.shape[1]
            self.cell = MaternKernel(nu=2.5, ard_num_dims=d,
                                     lengthscale_prior=LogNormalPrior(math.sqrt(2) + math.log(d) / 2, math.sqrt(3)),
                                     lengthscale_constraint=GreaterThan(2.5e-2))
            self.single = single
            self.register_parameter("raw_mix", torch.nn.Parameter(torch.zeros(1, dtype=torch.float64)))
            self.register_constraint("raw_mix", Interval(0.0, 1.0))
            self.register_parameter("raw_s_int", torch.nn.Parameter(torch.zeros(1, dtype=torch.float64)))
            self.register_constraint("raw_s_int", GreaterThan(1e-4))
            self.register_prior("s_int_prior", GammaPrior(2.0, 4.0), lambda m: m.s_int, lambda m, v: m._set("raw_s_int", v))
            if single:
                self.register_parameter("raw_s_single", torch.nn.Parameter(torch.zeros(1, dtype=torch.float64)))
                self.register_constraint("raw_s_single", GreaterThan(1e-4))
                self.register_prior("s_single_prior", GammaPrior(2.0, 4.0), lambda m: m.s_single,
                                    lambda m, v: m._set("raw_s_single", v))

        def _set(self, name, value):
            c = getattr(self, f"{name}_constraint")
            getattr(self, name).data.copy_(c.inverse_transform(torch.as_tensor(value, dtype=torch.float64).reshape(1)))

        mix = property(lambda self: self.raw_mix_constraint.transform(self.raw_mix))
        s_int = property(lambda self: self.raw_s_int_constraint.transform(self.raw_s_int))
        s_single = property(lambda self: self.raw_s_single_constraint.transform(self.raw_s_single))

        def tables(self):
            """The learned drug kernel (drugs x drugs) and cell kernel (lines x lines)."""
            kd = self.mix * self.t_fp + (1 - self.mix) * self.t_tg
            kc = self.cell(self.cell_pcs, self.cell_pcs).to_dense()
            return kd, kc

        def forward(self, x1, x2, diag=False, **params):
            kd, kc = self.tables()
            a1, b1, c1 = (x1[..., i].long() for i in range(3))
            a2, b2, c2 = (x2[..., i].long() for i in range(3))
            if diag:
                pair = kd[a1, a2] * kd[b1, b2] + kd[a1, b2] * kd[b1, a2]
                out = self.s_int * pair * kc[c1, c2]
                if self.single:
                    out = out + self.s_single * (kd[a1, a2] + kd[a1, b2] + kd[b1, a2] + kd[b1, b2]) * kc[c1, c2]
                return out
            A1, B1, C1 = a1.unsqueeze(-1), b1.unsqueeze(-1), c1.unsqueeze(-1)
            A2, B2, C2 = a2.unsqueeze(-2), b2.unsqueeze(-2), c2.unsqueeze(-2)
            cell = kc[C1, C2]
            out = self.s_int * (kd[A1, A2] * kd[B1, B2] + kd[A1, B2] * kd[B1, A2]) * cell
            if self.single:
                out = out + self.s_single * (kd[A1, A2] + kd[A1, B2] + kd[B1, A2] + kd[B1, B2]) * cell
            return out

    class RepNoise(Noise):
        """Homoscedastic noise sigma^2, divided by the replicate count in input column 3."""

        def __init__(self):
            super().__init__()
            self.register_parameter("raw_noise", torch.nn.Parameter(torch.zeros(1, dtype=torch.float64)))
            self.register_constraint("raw_noise", GreaterThan(1e-4))
            self.register_prior("noise_prior", GammaPrior(2.0, 4.0), lambda m: m.noise, lambda m, v: m._set(v))

        def _set(self, value):
            self.raw_noise.data.copy_(self.raw_noise_constraint.inverse_transform(
                torch.as_tensor(value, dtype=torch.float64).reshape(1)))

        @property
        def noise(self):
            return self.raw_noise_constraint.transform(self.raw_noise)

        def forward(self, *params, shape=None, **kwargs):
            x = params[0]
            if isinstance(x, (list, tuple)):  # GPyTorch's eval path passes the train inputs as a list
                x = x[0]
            reps = x[..., 3].clamp_min(1.0)
            return DiagLinearOperator(self.noise / reps)

    class RepLikelihood(_GaussianLikelihoodBase):
        def __init__(self):
            super().__init__(noise_covar=RepNoise())

        @property
        def noise(self):
            return self.noise_covar.noise

    class ExactComboGP(gpytorch.models.ExactGP):
        def __init__(self, x, y, likelihood, kernel):
            super().__init__(x, y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = kernel

        def forward(self, x):
            return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))

        def transform_inputs(self, X, input_transform=None):  # BoTorch's fitter calls this; no transform
            return X

    return ComboKernel, RepLikelihood, ExactComboGP


@dataclass
class _Fit:
    """One fitted target: hyperparameters as numpy, plus the Cholesky factor of the training kernel."""

    x: np.ndarray            # (n, 3) training indices
    kd: np.ndarray           # (drugs, drugs)
    kc: np.ndarray           # (lines, lines)
    s_int: float
    s_single: float          # 0 when the target has no single-drug term
    noise: float             # sigma^2, standardized units
    const: float
    chol: np.ndarray         # lower Cholesky of K + noise/reps + jitter
    alpha: np.ndarray        # (K + noise)^-1 (y - const)
    y_mu: float
    y_sd: float
    mll: float
    hypers: dict

    def cross(self, q: np.ndarray, h: np.ndarray | None = None) -> np.ndarray:
        """Kernel between query rows q (m, 3) and rows h (default: training rows), (m, n)."""
        h = self.x if h is None else h
        kd, kc = self.kd, self.kc
        qa, qb, qc = q[:, 0], q[:, 1], q[:, 2]
        ha, hb, hc = h[:, 0], h[:, 1], h[:, 2]
        # Pair terms per unique query pair, then gathered (pairs repeat across lines).
        pairs, inv = np.unique(qa * len(kd) + qb, return_inverse=True)
        pa, pb = pairs // len(kd), pairs % len(kd)
        daa, dbb, dab, dba = kd[pa][:, ha], kd[pb][:, hb], kd[pa][:, hb], kd[pb][:, ha]
        pair = self.s_int * (daa * dbb + dab * dba)
        if self.s_single:
            pair_single = self.s_single * (daa + dbb + dab + dba)
        lines, linv = np.unique(qc, return_inverse=True)
        cell = kc[lines][:, hc]
        out = pair[inv] * cell[linv]
        if self.s_single:
            out += pair_single[inv] * cell[linv]
        return out

    def prior_diag(self, q: np.ndarray) -> np.ndarray:
        kab = self.kd[q[:, 0], q[:, 1]]
        out = self.s_int * (1.0 + kab ** 2)
        if self.s_single:
            out = out + self.s_single * (2.0 + 2.0 * kab)
        return out


class GPModel:
    """Exact GP surrogate for both outcomes (one independent GP per target). See the module docstring."""

    name = "gp"

    def __init__(self, features: GPFeatures | None = None, seed: int = 0, n_restarts: int = 5,
                 maxiter: int = 200, single_drug_terms: dict[str, bool] | None = None):
        self.features = features if features is not None else GPFeatures.from_cache()
        self.seed, self.n_restarts, self.maxiter = seed, n_restarts, maxiter
        self.single = dict(SINGLE_DRUG_TERMS if single_drug_terms is None else single_drug_terms)
        self.fits: dict[str, _Fit] = {}
        self.failed_starts: list[str] = []
        self.fit_seconds = float("nan")

    # -- fitting ------------------------------------------------------------------------------------

    def fit(self, history: Sequence[Result]) -> GPModel:
        """Refit from scratch on revealed results."""
        if not history:
            raise ValueError("cannot fit on an empty history")
        if not all(isinstance(r, Result) for r in history):
            raise TypeError("fit takes oracle Results (revealed experiments only)")
        frame = pd.DataFrame([(*r.triplet, r.y_syn, r.y_inh, r.n_reps) for r in history],
                             columns=[*KEY, "y_syn", "y_inh", "n_reps"])
        return self.fit_frame(frame)

    def fit_frame(self, frame: pd.DataFrame) -> GPModel:
        """Refit on a frame with the triplet keys, y_syn, y_inh and optionally n_reps."""
        t0 = time.perf_counter()
        if len(frame) > MAX_EXACT:
            raise NotImplementedError(f"{len(frame)} rows: above {MAX_EXACT} an exact GP is too slow; "
                                      "a sparse variational GP would be needed (not built)")
        self.failed_starts = []
        x = self.features.encode(frame)
        reps = frame["n_reps"].to_numpy(np.float64) if "n_reps" in frame else np.ones(len(frame))
        for t in TARGETS:
            self.fits[t] = self._fit_target(x, reps, frame[f"y_{t}"].to_numpy(np.float64), t)
        self.fit_seconds = time.perf_counter() - t0
        return self

    def _fit_target(self, x: np.ndarray, reps: np.ndarray, y: np.ndarray, target: str) -> _Fit:
        import gpytorch
        from botorch.optim.fit import fit_gpytorch_mll_scipy

        ComboKernel, RepLikelihood, ExactComboGP = _gp_classes()
        torch.use_deterministic_algorithms(True)
        y_mu = float(y.mean())
        y_sd = float(y.std()) if y.std() > 1e-8 else 1.0
        ys = torch.as_tensor((y - y_mu) / y_sd, dtype=torch.float64)
        X = torch.as_tensor(np.column_stack([x, reps]), dtype=torch.float64)
        rng = np.random.default_rng([int(self.seed), TARGETS.index(target)])
        f = self.features
        best = None
        for restart in range(self.n_restarts):
            lik = RepLikelihood().double()
            kern = ComboKernel(f.t_fp, f.t_tg, f.cell_pcs, self.single[target]).double()
            model = ExactComboGP(X, ys, lik, kern).double()
            self._init(kern, lik, restart, rng)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)
            model.train(); lik.train()
            try:
                with gpytorch.settings.cholesky_jitter(double_value=JITTERS[0]), gpytorch.settings.max_cholesky_size(10_000):
                    fit_gpytorch_mll_scipy(mll, options={"maxiter": self.maxiter})
                    with torch.no_grad():
                        value = float(mll(model(X), ys, X))
            except (RuntimeError, ValueError, np.linalg.LinAlgError) as e:  # skip a failed start
                self.failed_starts.append(f"{target} start {restart}: {type(e).__name__}: {str(e)[:120]}")
                continue
            if not np.isfinite(value):
                continue
            if best is None or value > best[0]:
                best = (value, kern, lik, model)
        if best is None:
            raise RuntimeError(f"GP fit failed from every start ({target})")
        value, kern, lik, model = best
        with torch.no_grad():
            kd, kc = (t.numpy().copy() for t in kern.tables())
            s_int = float(kern.s_int)
            s_single = float(kern.s_single) if kern.single else 0.0
            noise = float(lik.noise)
            const = float(model.mean_module.constant)
            ls = kern.cell.lengthscale.numpy().ravel().tolist()
            mix = float(kern.mix)
        fit = _Fit(x, kd, kc, s_int, s_single, noise, const, np.empty(0), np.empty(0), y_mu, y_sd, value,
                   {"mix_fp": mix, "s_int": s_int, "s_single": s_single, "noise": noise, "const": const,
                    "cell_lengthscales": ls})
        K = fit.cross(x) + np.diag(noise / reps)
        K = 0.5 * (K + K.T)
        for jitter in JITTERS:
            try:
                fit.chol = np.linalg.cholesky(K + jitter * np.eye(len(K)))
                break
            except np.linalg.LinAlgError:
                continue
        else:
            raise RuntimeError("training kernel is not positive definite even with jitter 1e-4")
        fit.alpha = cho_solve((fit.chol, True), ys.numpy() - const)
        return fit

    @staticmethod
    def _init(kern, lik, restart: int, rng: np.random.Generator) -> None:
        """Start 0 at fixed defaults; later starts drawn from the priors (mix uniform on 0.05-0.95)."""
        d = kern.cell_pcs.shape[1]
        if restart == 0:
            mix, s_int, s_single, noise = 0.5, 0.5, 0.5, 0.5
            ls = np.full(d, math.sqrt(d))
        else:
            mix = rng.uniform(0.05, 0.95)
            s_int, s_single, noise = np.maximum(rng.gamma(2.0, 1 / 4.0, size=3), 1e-3)
            ls = np.clip(np.exp(math.sqrt(2) + math.log(d) / 2 + math.sqrt(3) * rng.standard_normal(d)), 0.05, 1e3)
        kern._set("raw_mix", float(mix))
        kern._set("raw_s_int", float(s_int))
        if kern.single:
            kern._set("raw_s_single", float(s_single))
        lik.noise_covar._set(float(noise))
        kern.cell.lengthscale = torch.as_tensor(ls, dtype=torch.float64).reshape(kern.cell.lengthscale.shape)

    # -- prediction ---------------------------------------------------------------------------------

    def _check(self):
        if not self.fits:
            raise RuntimeError("fit the model first")

    def _posterior(self, target: str, q: np.ndarray):
        """Latent mean and variance (standardized units) for query rows."""
        fit = self.fits[target]
        means, variances = [], []
        for s in range(0, len(q), CHUNK):
            qs = q[s:s + CHUNK]
            ks = fit.cross(qs)
            means.append(fit.const + ks @ fit.alpha)
            v = solve_triangular(fit.chol, ks.T, lower=True, check_finite=False)
            variances.append(np.maximum(fit.prior_diag(qs) - (v * v).sum(0), 1e-12))
        return np.concatenate(means), np.concatenate(variances)

    def predict(self, experiments) -> Prediction:
        """Mean and predictive std (latent + noise for one replicate) of both outcomes."""
        self._check()
        q = self.features.encode(experiments)
        out = {}
        for t in TARGETS:
            fit = self.fits[t]
            m, v = self._posterior(t, q)
            latent = np.sqrt(v) * fit.y_sd
            noise = np.full(len(q), math.sqrt(fit.noise) * fit.y_sd)
            out[t] = (m * fit.y_sd + fit.y_mu, np.sqrt(latent ** 2 + noise ** 2), latent, noise)
        (ms, ss, ls, ns), (mi, si, li, ni) = out["syn"], out["inh"]
        return Prediction(ms, ss, mi, si, ls, ns, li, ni)

    def posterior_cov(self, experiments, target: str) -> tuple[np.ndarray, np.ndarray]:
        """Joint latent posterior mean (m,) and covariance (m, m), original units."""
        self._check()
        fit = self.fits[target]
        q = self.features.encode(experiments)
        ks = fit.cross(q)
        mean = fit.const + ks @ fit.alpha
        v = solve_triangular(fit.chol, ks.T, lower=True, check_finite=False)
        cov = fit.cross(q, q) - v.T @ v
        cov = 0.5 * (cov + cov.T)
        return mean * fit.y_sd + fit.y_mu, cov * fit.y_sd ** 2

    def sample(self, experiments, n_samples: int, target: str, rng: np.random.Generator,
               noise: bool = True) -> np.ndarray:
        """Joint posterior draws (n_samples x rows), original units; with `noise`, plus measurement
        noise for one replicate."""
        mean, cov = self.posterior_cov(experiments, target)
        scale = float(np.mean(np.diag(cov))) if len(cov) else 1.0
        for rel in (1e-8, 1e-6, 1e-4, 1e-2):
            try:
                L = np.linalg.cholesky(cov + rel * max(scale, 1e-12) * np.eye(len(cov)))
                break
            except np.linalg.LinAlgError:
                continue
        else:
            raise RuntimeError("posterior covariance is not positive definite")
        draws = mean + rng.standard_normal((n_samples, len(mean))) @ L.T
        if noise:
            fit = self.fits[target]
            draws = draws + math.sqrt(fit.noise) * fit.y_sd * rng.standard_normal(draws.shape)
        return draws

    def similarity(self, experiments, target: str | None = None):
        """j -> normalized prior kernel k(x, x_j) / sqrt(k(x, x) k(x_j, x_j)) for the target's GP."""
        self._check()
        fit = self.fits[target or "syn"]
        q = self.features.encode(experiments)
        k = fit.cross(q, q)
        d = np.sqrt(np.maximum(np.diag(k), 1e-12))
        s = k / d[:, None] / d[None, :]
        return lambda j: s[:, j]

    def kernel_matrix(self, experiments, target: str) -> np.ndarray:
        """Prior kernel between experiments (standardized units), for tests and diagnostics."""
        self._check()
        q = self.features.encode(experiments)
        return self.fits[target].cross(q, q)

    def hypers(self) -> dict:
        return {t: {**f.hypers, "mll": f.mll} for t, f in self.fits.items()}
