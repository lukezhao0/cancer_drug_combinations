"""Gaussian-process surrogate tests on a synthetic world (no real data)."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
import pytest
import torch

from combo_agent.lab.oracle import KEY
from combo_agent.models.gp import GPFeatures, GPModel, _gp_classes, tanimoto


def _tables(n_drugs=12, n_lines=8, n_bits=64, seed=0, far_drug=False):
    rng = np.random.default_rng(seed)
    ids = [f"CHEMBL{i:02d}" for i in range(n_drugs)]
    # Three chemical families: each drug is its family prototype with ~10% of bits flipped, so
    # structure similarity carries information (random bits would make every pair unrelated).
    protos = rng.random((3, n_bits)) < 0.3
    fp = (protos[np.arange(n_drugs) % 3] ^ (rng.random((n_drugs, n_bits)) < 0.1)).astype(int)
    tg = (rng.random((n_drugs, 6)) < 0.3).astype(int)
    tg[0] = 0  # one drug without any target gene
    if far_drug:  # last drug shares no fingerprint bit and no target with any other drug
        fp[:, -8:] = 0
        fp[-1] = 0
        fp[-1, -8:] = 1
        tg[:, -1] = 0
        tg[-1] = 0
        tg[-1, -1] = 1
    drugs = pd.DataFrame(fp, columns=[f"fp_{j}" for j in range(n_bits)])
    for j in range(tg.shape[1]):
        drugs[f"target_G{j}"] = tg[:, j]
    drugs.insert(0, "drug_id", ids)
    cells = pd.DataFrame(rng.normal(size=(n_lines, 20)), columns=[f"expr_G{j}" for j in range(20)])
    cells.insert(0, "cellosaurus_ac", [f"CVCL_{i:02d}" for i in range(n_lines)])
    pool = pd.DataFrame([(a, b, c) for a, b in itertools.combinations(ids, 2) for c in cells.cellosaurus_ac],
                        columns=KEY)
    return drugs, cells, pool


@pytest.fixture(scope="module")
def planted():
    """A world whose outcomes are drawn from the GP prior itself, with known hyperparameters."""
    drugs, cells, pool = _tables()
    fs = GPFeatures.build(drugs, cells, pool[["drug_a", "drug_b"]].drop_duplicates(), n_pcs=4)
    ComboKernel, _, _ = _gp_classes()
    out = {"fs": fs, "pool": pool}
    rng = np.random.default_rng(1)
    x = torch.as_tensor(fs.encode(pool), dtype=torch.float64)
    order = np.random.default_rng(2).permutation(len(pool))
    tr, te = order[:250], order[250:]
    for t, single in (("syn", False), ("inh", True)):
        k = ComboKernel(fs.t_fp, fs.t_tg, fs.cell_pcs, single).double()
        k._set("raw_mix", 0.6); k._set("raw_s_int", 1.0)
        if single:
            k._set("raw_s_single", 1.0)
        k.cell.lengthscale = torch.full_like(k.cell.lengthscale, 3.0)
        with torch.no_grad():
            K = k(x, x).to_dense().numpy()
        f = np.linalg.cholesky(K + 1e-8 * np.eye(len(K))) @ rng.standard_normal(len(K))
        noise = 0.1 * np.diag(K).mean()
        out[f"f_{t}"] = f
        out[f"y_{t}"] = 3.0 * f + 10.0 + 3.0 * math.sqrt(noise) * rng.standard_normal(len(f))
        out[f"noise_sd_{t}"] = 3.0 * math.sqrt(noise)
        # The best possible prediction: the posterior mean under the true kernel and noise.
        Ktt = K[np.ix_(tr, tr)] + noise * np.eye(len(tr))
        out[f"oracle_{t}"] = 3.0 * K[np.ix_(te, tr)] @ np.linalg.solve(Ktt, (out[f"y_{t}"][tr] - 10.0) / 3.0) + 10.0
    frame = pool.assign(y_syn=out["y_syn"], y_inh=out["y_inh"])
    out["train"], out["test"] = frame.iloc[tr], frame.iloc[te]
    out["model"] = GPModel(fs, seed=0, n_restarts=3).fit_frame(out["train"])
    return out


def test_tanimoto_matches_hand_computation_and_empty_rows_are_identity():
    bits = np.array([[1, 1, 0, 0], [1, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=float)
    t = tanimoto(bits)
    assert t[0, 1] == pytest.approx(1 / 3, abs=1e-5)       # |A∩B| / |A∪B| = 1 / 3
    assert t[0, 0] == t[2, 2] == 1.0
    assert t[2, 3] == 0.0 and t[0, 2] == 0.0                # two empty rows are not "identical"
    assert np.allclose(t, t.T) and np.linalg.eigvalsh(t).min() > -1e-10


def test_pair_symmetry_and_psd_kernel(planted):
    m, pool = planted["model"], planted["pool"]
    swapped = pool.rename(columns={"drug_a": "drug_b", "drug_b": "drug_a"})[KEY]
    p1, p2 = m.predict(pool), m.predict(swapped)
    for t in ("syn", "inh"):
        assert np.array_equal(p1.mean(t), p2.mean(t)) and np.array_equal(p1.std(t), p2.std(t))
        fit = m.fits[t]
        q = m.features.encode(pool[:120])
        k1 = fit.cross(q, q)
        k2 = fit.cross(q[:, [1, 0, 2]], q)  # raw kernel with a and b swapped on one side
        assert np.allclose(k1, k2, atol=1e-12)
        assert np.allclose(k1, k1.T, atol=1e-12)
        assert np.linalg.eigvalsh(k1).min() >= -1e-8
        assert np.allclose(np.diag(k1), fit.prior_diag(q))


def test_closed_form_matches_gpytorch_posterior(planted):
    """Our chunked closed-form prediction equals GPyTorch's own exact posterior."""
    import gpytorch
    m, train, test = planted["model"], planted["train"], planted["test"]
    fit = m.fits["inh"]
    ComboKernel, RepLikelihood, ExactComboGP = _gp_classes()
    fs = m.features
    X = torch.as_tensor(np.column_stack([fs.encode(train), np.ones(len(train))]), dtype=torch.float64)
    y = torch.as_tensor((train.y_inh.to_numpy() - fit.y_mu) / fit.y_sd)
    k = ComboKernel(fs.t_fp, fs.t_tg, fs.cell_pcs, True).double()
    k._set("raw_mix", fit.hypers["mix_fp"]); k._set("raw_s_int", fit.s_int); k._set("raw_s_single", fit.s_single)
    k.cell.lengthscale = torch.as_tensor(fit.hypers["cell_lengthscales"], dtype=torch.float64).reshape(1, -1)
    lik = RepLikelihood().double(); lik.noise_covar._set(fit.noise)
    gp = ExactComboGP(X, y, lik, k).double()
    gp.mean_module.constant.data.fill_(fit.const)
    gp.eval(); lik.eval()
    Xs = torch.as_tensor(np.column_stack([fs.encode(test), np.ones(len(test))]), dtype=torch.float64)
    with torch.no_grad(), gpytorch.settings.fast_pred_var(False), gpytorch.settings.cholesky_jitter(double_value=1e-6):
        post = gp(Xs)
    pred = m.predict(test)
    assert np.allclose(pred.mean_inh, post.mean.numpy() * fit.y_sd + fit.y_mu, atol=1e-4 * fit.y_sd)
    assert np.allclose(pred.latent_std_inh, np.sqrt(post.variance.numpy()) * fit.y_sd, rtol=1e-3, atol=1e-4)


def test_recovers_planted_function_with_calibrated_intervals(planted):
    m, test = planted["model"], planted["test"]
    p = m.predict(test)
    for t in ("syn", "inh"):
        y = test[f"y_{t}"].to_numpy()
        f = planted[f"f_{t}"][test.index.to_numpy()] * 3.0 + 10.0
        r_fit = np.corrcoef(p.mean(t), f)[0, 1]
        r_best = np.corrcoef(planted[f"oracle_{t}"], f)[0, 1]  # true hyperparameters: the ceiling
        assert r_fit > 0.7 and r_fit >= r_best - 0.05, (t, r_fit, r_best)
        cover1 = np.mean(np.abs(y - p.mean(t)) <= p.std(t))
        cover2 = np.mean(np.abs(y - p.mean(t)) <= 2 * p.std(t))
        assert 0.58 <= cover1 <= 0.78 and 0.90 <= cover2 <= 0.99, (t, cover1, cover2)
        assert p.noise_std(t)[0] == pytest.approx(planted[f"noise_sd_{t}"], rel=0.5)


def test_std_small_at_data_and_prior_level_far_away():
    drugs, cells, pool = _tables(far_drug=True)
    fs = GPFeatures.build(drugs, cells, pool[["drug_a", "drug_b"]].drop_duplicates(), n_pcs=4)
    far = drugs.drug_id.iloc[-1]
    rng = np.random.default_rng(0)
    seen = pool[(pool.drug_a != far) & (pool.drug_b != far)]
    train = seen.sample(150, random_state=0)
    y = rng.normal(size=len(train))
    train = train.assign(y_syn=y, y_inh=y)
    m = GPModel(fs, seed=0, n_restarts=2, single_drug_terms={"syn": False, "inh": False}).fit_frame(train)
    fit = m.fits["syn"]
    p_train = m.predict(train)
    away = pool[(pool.drug_a == far) | (pool.drug_b == far)]
    p_far = m.predict(away)
    prior_sd = np.sqrt(fit.prior_diag(fs.encode(away))) * fit.y_sd
    # Far from data (a drug unlike any tested one): the latent std is the prior std.
    assert np.allclose(p_far.latent_std_syn, prior_sd, rtol=1e-6)
    # At the training points: the predictive std is close to the noise level.
    noise_sd = math.sqrt(fit.noise) * fit.y_sd
    assert np.median(p_train.latent_std_syn) < np.median(p_far.latent_std_syn)
    assert np.median(p_train.std_syn) <= 1.5 * noise_sd + 1e-9


def test_determinism_sampling_and_similarity(planted):
    m, train, test = planted["model"], planted["train"], planted["test"]
    m2 = GPModel(m.features, seed=0, n_restarts=3).fit_frame(train)
    assert np.array_equal(m.predict(test).mean_syn, m2.predict(test).mean_syn)
    rows = test.iloc[:50]
    a = m.sample(rows, 4000, "inh", np.random.default_rng(5), noise=False)
    b = m.sample(rows, 4000, "inh", np.random.default_rng(5), noise=False)
    assert a.shape == (4000, 50) and np.array_equal(a, b)
    p = m.predict(rows)
    assert np.allclose(a.mean(0), p.mean_inh, atol=4 * p.latent_std_inh.max() / math.sqrt(4000))
    assert np.allclose(a.std(0), p.latent_std_inh, rtol=0.1)
    sim = m.similarity(rows, "inh")
    col = sim(3)
    assert col[3] == pytest.approx(1.0) and np.all(col <= 1 + 1e-9)


def test_refuses_above_exact_limit(planted, monkeypatch):
    import combo_agent.models.gp as gp
    monkeypatch.setattr(gp, "MAX_EXACT", 10)
    with pytest.raises(NotImplementedError):
        GPModel(planted["model"].features).fit_frame(planted["train"])
