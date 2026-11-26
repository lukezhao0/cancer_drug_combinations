"""Tests for combo_agent.models: features, the ensemble's gradients, symmetry, learnability."""

from __future__ import annotations

import itertools
import os
import time

import numpy as np
import pandas as pd
import pytest

from combo_agent.lab.oracle import KEY, Result, Triplet
from combo_agent.models.legacy.ensemble_features import FeatureSet
from combo_agent.models.legacy.ensemble import EnsembleModel, _MLP


def _tables(n_drugs=12, n_lines=10, seed=0):
    rng = np.random.default_rng(seed)
    drugs = pd.DataFrame({"drug_id": [f"CHEMBL{i:02d}" for i in range(n_drugs)], "target_status": "x"})
    for j in range(40):
        drugs[f"fp_{j}"] = rng.integers(0, 2, n_drugs)
    for g in ("EGFR", "MEK", "TOP2A"):
        drugs[f"target_{g}"] = rng.integers(0, 2, n_drugs)
    drugs["has_target"] = 1
    cells = pd.DataFrame({"cellosaurus_ac": [f"CVCL_{i:02d}" for i in range(n_lines)], "cn_source": "x"})
    for j in range(30):
        cells[f"expr_G{j}"] = rng.normal(size=n_lines)
    cells["mut_KRAS"] = rng.integers(0, 2, n_lines)
    cells["cn_MYC"] = rng.normal(size=n_lines)
    pairs = pd.DataFrame(list(itertools.combinations(sorted(drugs.drug_id), 2)), columns=["drug_a", "drug_b"])
    return drugs, cells, pairs


@pytest.fixture(scope="module")
def fs():
    d, c, p = _tables()
    return FeatureSet.build(d, c, p, mode="compact")


def _frame(fs, pairs, lines):
    return pd.DataFrame([(a, b, line) for a, b in pairs for line in lines], columns=KEY)


def _planted(fs, frame, noise=0.3, seed=0):
    """A smooth nonlinear function of the drugs' first fingerprint PCs and the cell features."""
    rng = np.random.default_rng(seed)
    ia, ib = fs.drug_index(frame.drug_a), fs.drug_index(frame.drug_b)
    ci = fs.line_index(frame.cellosaurus_ac)
    da, db, cx = fs.drug_x[ia, 0], fs.drug_x[ib, 0], fs.cell_x[ci, 0]
    syn = np.tanh(da * db) * 3 + 2 * cx * (da + db) + noise * rng.standard_normal(len(frame))
    inh = 10 * (da + db) - 5 * cx + noise * 10 * rng.standard_normal(len(frame))
    return frame.assign(y_syn=syn, y_inh=inh)


def test_feature_modes_and_symmetry():
    d, c, p = _tables()
    for mode in ("full", "compact"):
        f = FeatureSet.build(d, c, p, mode=mode)
        ab = f.pair_block(f.pair_code(f.drug_index(["CHEMBL01"]), f.drug_index(["CHEMBL05"])))
        ba = f.pair_block(f.pair_code(f.drug_index(["CHEMBL05"]), f.drug_index(["CHEMBL01"])))
        assert np.array_equal(ab, ba) and ab.shape[1] == f.pair_dim
        assert np.isfinite(f.cell_x).all()
    assert FeatureSet.build(d, c, p, "compact").pair_dim < FeatureSet.build(d, c, p, "full").pair_dim + 10
    with pytest.raises(KeyError, match="no features"):
        f.drug_index(["NOPE"])


def test_gradient_matches_finite_differences(fs):
    rng = np.random.default_rng(1)
    net = _MLP(fs.pair_dim, fs.cell_dim, 8, rng)
    net.p = {k: v.astype(np.float64) for k, v in net.p.items()}
    net.p["W3"] = rng.standard_normal(net.p["W3"].shape) * 0.3
    P, C = fs.pair_block(np.arange(5) * 12 + 7).astype(np.float64), fs.cell_x.astype(np.float64)
    pi, ci = np.array([0, 1, 2, 3, 4, 0, 2]), np.array([0, 1, 2, 3, 4, 5, 1])
    y = rng.standard_normal((7, 2))
    _, g = net.loss_and_grad(P, C, pi, ci, y)
    for name in _MLP.NAMES:
        w = net.p[name]
        for idx in [tuple(rng.integers(0, s) for s in w.shape) for _ in range(4)]:
            old = w[idx]
            w[idx] = old + 1e-6
            lp, _ = net.loss_and_grad(P, C, pi, ci, y)
            w[idx] = old - 1e-6
            lm, _ = net.loss_and_grad(P, C, pi, ci, y)
            w[idx] = old
            assert g[name][idx] == pytest.approx((lp - lm) / 2e-6, rel=1e-4, abs=1e-8), name


def test_factorized_forward_equals_dense_forward(fs):
    net = _MLP(fs.pair_dim, fs.cell_dim, 16, np.random.default_rng(2))
    codes = np.array([7, 7, 30, 55, 30])
    lines = np.array([0, 3, 3, 9, 1])
    uniq, pi = np.unique(codes, return_inverse=True)
    mu, _ = net.forward(fs.pair_block(uniq), fs.cell_x, pi, lines)
    X = fs.dense(codes, lines)
    W1 = np.concatenate([net.p["Wp"], net.p["Wc"]])
    h = np.maximum(np.maximum(X @ W1 + net.p["b1"], 0) @ net.p["W2"] + net.p["b2"], 0)
    o = h @ net.p["W3"] + net.p["b3"]
    np.testing.assert_allclose(mu, o, rtol=1e-4, atol=1e-5)


def test_learns_a_planted_function_and_is_calibrated(fs):
    d, c, p = _tables()
    lines = list(fs.line_ids)
    frame = _planted(fs, _frame(fs, p.values.tolist(), lines))
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(frame))
    train, test = frame.iloc[idx[:400]], frame.iloc[idx[400:]]
    m = EnsembleModel(fs, n_members=3, hidden=64, seed=0).fit_frame(train)
    pred = m.predict(test)
    r = np.corrcoef(pred.mean_syn, test.y_syn)[0, 1]
    assert r > 0.8, r
    z = np.abs(test.y_inh - pred.mean_inh) / pred.std_inh
    coverage = float(np.mean(z <= 1.2816))
    assert 0.6 <= coverage <= 0.97, coverage
    assert (pred.std_syn > 0).all() and (pred.std_inh > 0).all()


def test_swap_invariance_determinism_and_results_api(fs):
    d, c, p = _tables()
    frame = _planted(fs, _frame(fs, p.values.tolist()[:20], fs.line_ids[:5]))
    history = [Result(Triplet(a, b, line), s, i, float("nan"), float("nan"), 1, 0)
               for a, b, line, s, i in frame[[*KEY, "y_syn", "y_inh"]].itertuples(index=False)]
    m1 = EnsembleModel(fs, n_members=2, hidden=32, seed=4).fit(history)
    m2 = EnsembleModel(fs, n_members=2, hidden=32, seed=4).fit(history)
    q = [Triplet("CHEMBL01", "CHEMBL07", "CVCL_08"), Triplet("CHEMBL03", "CHEMBL04", "CVCL_02")]
    swapped = [Triplet(t.drug_b, t.drug_a, t.cellosaurus_ac) for t in q]
    a, b, c2 = m1.predict(q), m1.predict(swapped), m2.predict(q)
    np.testing.assert_array_equal(a.mean_syn, b.mean_syn)
    np.testing.assert_array_equal(a.std_inh, b.std_inh)
    np.testing.assert_array_equal(a.mean_syn, c2.mean_syn)
    assert m1.embed(q).shape == (2, 32)
    assert np.all(a.ucb("syn", 2.0) > a.mean_syn)
    with pytest.raises(TypeError):
        m1.fit([("CHEMBL01", "CHEMBL02", "CVCL_00", 1.0, 2.0)])
    with pytest.raises(ValueError):
        m1.fit([])


def test_tiny_history_still_fits(fs):
    d, c, p = _tables()
    frame = _planted(fs, _frame(fs, p.values.tolist()[:3], fs.line_ids[:2]))  # 6 rows: no validation split
    pred = EnsembleModel(fs, n_members=2, hidden=16).fit_frame(frame).predict(frame)
    assert np.isfinite(pred.mean_syn).all() and (pred.std_syn > 0).all()


# -- real data ------------------------------------------------------------------------------------

integration = pytest.mark.skipif(os.environ.get("COMBO_INTEGRATION") != "1",
                                 reason="needs the M1/M2 data; set COMBO_INTEGRATION=1")


@integration
@pytest.mark.parametrize("mode", ["compact", "full"])
def test_real_refit_and_pool_prediction_under_60s(mode):
    from combo_agent.lab.oracle import Oracle

    o = Oracle.from_cache(seed=0)
    frame = o.candidate_frame()
    o.run(frame.iloc[:16][KEY].values.tolist())
    for _ in range(19):
        frame = o.candidate_frame()
        o.run(frame.sample(16, random_state=len(o.history()))[KEY].values.tolist())
    assert len(o.history()) == 352
    t0 = time.perf_counter()
    m = EnsembleModel(mode=mode, seed=0).fit(o.history())
    pred = m.predict(o.candidate_frame())
    elapsed = time.perf_counter() - t0
    assert elapsed < 60, f"{mode}: refit + predict took {elapsed:.1f}s"
    print(f"{mode}: fit {m.fit_seconds:.1f}s, fit+predict {elapsed:.1f}s, epochs {m.epochs_run}")
    assert len(pred.mean_syn) == 219_239 - 352 and np.isfinite(pred.std_syn).all()


def test_noise_modes_change_width_not_means(fs):
    d, c, p = _tables()
    frame = _planted(fs, _frame(fs, p.values.tolist(), fs.line_ids))
    # heavy-tailed noise: the quantile scale should be narrower than the RMS scale
    rng = np.random.default_rng(3)
    frame = frame.assign(y_syn=frame.y_syn + rng.standard_t(2, len(frame)))
    q = EnsembleModel(fs, n_members=2, hidden=32, seed=1, noise="quantile").fit_frame(frame)
    m = EnsembleModel(fs, n_members=2, hidden=32, seed=1, noise="mse").fit_frame(frame)
    pq, pm = q.predict(frame), m.predict(frame)
    np.testing.assert_array_equal(pq.mean_syn, pm.mean_syn)
    assert pq.std_syn.mean() < pm.std_syn.mean()
    with pytest.raises(ValueError):
        EnsembleModel(fs, noise="bogus")


def test_member_means_agree_with_predict(fs):
    d, c, p = _tables()
    frame = _planted(fs, _frame(fs, p.values.tolist()[:30], fs.line_ids))
    m = EnsembleModel(fs, n_members=3, hidden=16, seed=2).fit_frame(frame)
    means, noise = m.member_means(frame)
    pred = m.predict(frame)
    assert means.shape == (3, len(frame), 2) and noise.shape == (3, 2)
    np.testing.assert_allclose(means.mean(0)[:, 0], pred.mean_syn, rtol=1e-6, atol=1e-6)
    total = np.sqrt(means.var(0)[:, 1] + (noise[:, 1] ** 2).mean())
    np.testing.assert_allclose(total, pred.std_inh, rtol=1e-5)
