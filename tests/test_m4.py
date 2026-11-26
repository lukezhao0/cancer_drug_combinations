"""M4 tests on a synthetic world: selectors, the campaign runner and the metrics. No real data."""

from __future__ import annotations

import itertools
import os

import numpy as np
import pandas as pd
import pytest

from combo_agent.eval import metrics as mx
from combo_agent.eval import runner as rn
from combo_agent.lab.oracle import KEY, Oracle, Triplet
from combo_agent.eval.diagnostics import DistanceSpace
from combo_agent.models.legacy.ensemble_features import FeatureSet
from combo_agent.models.gp import GPFeatures, GPModel
from combo_agent.models.legacy.ensemble import EnsembleModel
from combo_agent.selectors import SELECTORS, SelectionContext, make
from combo_agent.selectors.baselines import diverse_greedy, greedy_qei

TINY = {"n_members": 2, "hidden": 16, "max_epochs": 60, "patience": 10}


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("m4")
    rng = np.random.default_rng(0)
    drugs = pd.DataFrame({"drug_id": [f"CHEMBL{i:02d}" for i in range(10)]})
    for j in range(24):
        drugs[f"fp_{j}"] = rng.integers(0, 2, len(drugs))
    drugs["target_X"] = rng.integers(0, 2, len(drugs))
    cells = pd.DataFrame({"cellosaurus_ac": [f"CVCL_{i:02d}" for i in range(8)]})
    for j in range(12):
        cells[f"expr_G{j}"] = rng.normal(size=len(cells))
    cells["mut_KRAS"] = rng.integers(0, 2, len(cells))
    rows = [(a, b, c) for a, b in itertools.combinations(sorted(drugs.drug_id), 2) for c in cells.cellosaurus_ac]
    pool = pd.DataFrame(rows, columns=KEY)
    pool["split"] = np.where(pool.cellosaurus_ac.isin(["CVCL_06", "CVCL_07"]), "heldout", "selectable")
    pool["n_reps"] = 1
    fs = FeatureSet.build(drugs, cells, pool[["drug_a", "drug_b"]].drop_duplicates(), mode="compact")
    ia, ib = fs.drug_index(pool.drug_a), fs.drug_index(pool.drug_b)
    ci = fs.line_index(pool.cellosaurus_ac)
    signal = fs.drug_x[ia, 0] * fs.drug_x[ib, 0] + fs.cell_x[ci, 0]
    labels = pool[KEY].assign(y_syn=3 * signal + rng.normal(size=len(pool)),
                              y_inh=20 * signal + 30 + 5 * rng.normal(size=len(pool)),
                              bliss_sd=np.nan, css_sd=np.nan)
    path = tmp / "labels.parquet"
    labels.to_parquet(path, index=False)
    sel = pool[pool.split == "selectable"][KEY].sort_values(KEY).values.tolist()
    seed_sets = {str(s): [sel[i] for i in np.random.default_rng(s).choice(len(sel), 8, replace=False)] for s in range(3)}
    thresholds = {"y_syn": float(np.quantile(labels.y_syn, 0.8)), "y_inh": float(np.quantile(labels.y_inh, 0.8))}

    def factory(**kw):
        return Oracle(pool, labels_path=path, seed_sets=seed_sets, thresholds=thresholds, **kw)

    gfs = GPFeatures.build(drugs, cells, pool[["drug_a", "drug_b"]].drop_duplicates(), n_pcs=4)
    return {"fs": fs, "gfs": gfs, "factory": factory, "pool": pool, "tmp": tmp}


GP_TINY = {"n_restarts": 2, "maxiter": 50}


def _ctx(world, seed=0, surrogate="ensemble"):
    o = world["factory"](seed=seed)
    cand = o.candidate_frame()
    if surrogate == "gp":
        m = GPModel(world["gfs"], seed=1, **GP_TINY).fit(o.history())
    else:
        m = EnsembleModel(world["fs"], seed=1, **TINY).fit(o.history())
    return o, SelectionContext(cand, o.history(), m, m.predict(cand))


@pytest.mark.parametrize("surrogate", ["ensemble", "gp"])
@pytest.mark.parametrize("name", list(SELECTORS))
def test_every_selector_returns_unique_untested_selectable_and_is_deterministic(world, name, surrogate):
    o, ctx = _ctx(world, surrogate=surrogate)
    sel = make(name)
    a = sel.select(ctx, 6, "syn", np.random.default_rng(3))
    b = sel.select(ctx, 6, "syn", np.random.default_rng(3))
    assert a == b and len(a) == 6 and len(set(a)) == 6
    cands = set(map(tuple, ctx.candidates[KEY].values.tolist()))
    assert all(tuple(t) in cands for t in a)
    o.run(a)  # the oracle accepts it: untested, selectable, within batch size


def test_greedy_and_uncertainty_take_the_argmax(world):
    _, ctx = _ctx(world)
    g = make("greedy").select(ctx, 4, "inh", np.random.default_rng(0))
    u = make("uncertainty").select(ctx, 4, "syn", np.random.default_rng(0))
    order_mean = np.argsort(-ctx.pred.mean_inh, kind="stable")[:4]
    order_std = np.argsort(-ctx.pred.std_syn, kind="stable")[:4]
    assert g == ctx.triplets(order_mean) and u == ctx.triplets(order_std)


def test_diversity_penalty_spreads_picks():
    # 3 clusters of near-identical embeddings; cluster 0 has the best scores.
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(3, 8)) * 5
    emb = np.repeat(centers, 10, axis=0) + 0.01 * rng.normal(size=(30, 8))
    scores = np.concatenate([np.linspace(10, 9, 10), np.linspace(5, 4, 10), np.linspace(4.5, 3.5, 10)])
    plain = diverse_greedy(scores, emb, 6, alpha=0.0)
    div = diverse_greedy(scores, emb, 6, alpha=0.5)
    assert len({p // 10 for p in plain}) == 1
    assert len({p // 10 for p in div}) == 3
    assert div[0] == 0


def test_qei_prefers_the_dominant_candidate_and_hedges():
    samples = np.zeros((50, 4))
    samples[:, 2] = 5.0  # always best
    samples[:25, 0] = 8.0  # sometimes better
    picks = greedy_qei(samples, incumbent=1.0, k=2)
    assert picks[0] in (0, 2) and set(picks) == {0, 2}


def test_campaign_spends_budget_and_curves_are_monotone(world):
    for name in ("random", "bo_ucb", "bo_qei"):
        spec = rn.CampaignSpec(name, "syn", 0, budget=24, batch_size=8)
        rounds, reveals = rn.run_campaign(spec, world["fs"], TINY, world["factory"], surrogate="ensemble")
        assert rounds["n_charged"].tolist() == [0, 8, 16, 24]
        assert rounds["n_revealed"].iloc[-1] == 32 and len(reveals) == 32
        for col in ("hits_syn", "hits_inh", "hits_syn_strict", "hit_pairs_syn"):
            assert rounds[col].is_monotonic_increasing
        assert (rounds["hits_syn_strict"] <= rounds["hits_syn"]).all()
        if name == "random":
            assert rounds["fit_s"].sum() == 0.0
        assert rounds["hits_syn"].iloc[-1] == reveals[(reveals["round"] > 0)]["hit_syn"].sum()


def test_run_one_writes_atomically_and_resumes(world, tmp_path):
    spec = rn.CampaignSpec("greedy", "inh", 1, budget=16, batch_size=8)
    msg = rn.run_one(spec, tmp_path, "compact", TINY, world["fs"], world["factory"], surrogate="ensemble")
    assert msg.startswith("done")
    files = sorted(p.name for p in (tmp_path / "runs").iterdir())
    assert files == ["greedy__inh__s1.parquet", "greedy__inh__s1.reveals.parquet"]
    assert rn.run_one(spec, tmp_path, "compact", TINY, world["fs"], world["factory"], surrogate="ensemble").startswith("skip")


def test_ensemble_sample_and_similarity_match_the_original_selector_math(world):
    """The Surrogate methods reproduce the pre-interface bo_qei / bo_ucb arithmetic exactly."""
    _, ctx = _ctx(world)
    rows = ctx.candidates.iloc[:40]
    means, noise = ctx.model.member_means(rows)
    rng = np.random.default_rng(9)
    members = rng.integers(0, means.shape[0], size=64)
    old = means[members, :, 1] + noise[members, 1][:, None] * rng.standard_normal((64, 40))
    assert np.array_equal(ctx.model.sample(rows, 64, "inh", np.random.default_rng(9)), old)
    emb = ctx.model.embed(rows)
    e = emb - emb.mean(0)
    e = e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
    assert np.array_equal(ctx.model.similarity(rows)(7), e @ e[7])


def test_gp_campaign_with_diagnostics(world, tmp_path):
    spec = rn.CampaignSpec("bo_ts", "inh", 0, budget=16, batch_size=8)
    msg = rn.run_one(spec, tmp_path, "compact", GP_TINY, world["gfs"], world["factory"], surrogate="gp",
                     space=DistanceSpace(world["gfs"]))
    assert msg.startswith("done"), msg
    preds = pd.read_parquet(tmp_path / "runs" / "bo_ts__inh__s0.preds.parquet")
    reveals = pd.read_parquet(tmp_path / "runs" / "bo_ts__inh__s0.reveals.parquet")
    assert set(preds["set"]) == {"batch", "probe"} and (preds["surrogate"] == "gp").all()
    batch = preds[preds["set"] == "batch"]
    assert len(batch) == 16 and batch["round"].tolist() == [1] * 8 + [2] * 8
    paid = reveals[reveals["round"] > 0]
    assert batch[KEY].values.tolist() == paid[KEY].values.tolist()  # predictions logged before each run
    assert ((preds["dist"] >= 0) & (preds["dist"] <= 1)).all()
    assert (preds["std_inh"] >= preds["latent_std_inh"]).all()
    probe = preds[preds["set"] == "probe"]
    assert len(probe[probe["round"] == 1]) > 0 and not probe.duplicated(["round", *KEY]).any()


def test_diagnostics_do_not_change_the_picks(world, tmp_path):
    spec = rn.CampaignSpec("bo_qei", "syn", 1, budget=16, batch_size=8)
    _, plain = rn.run_campaign(spec, world["fs"], TINY, world["factory"], surrogate="ensemble")
    diag = []
    _, logged = rn.run_campaign(spec, world["fs"], TINY, world["factory"], surrogate="ensemble", space=DistanceSpace(world["gfs"]),
                                diagnostics=diag)
    assert plain.equals(logged) and len(diag) == 2


def test_gp_is_the_default_surrogate():
    """The GP is the main surrogate; the ensemble runs only when asked for (legacy build)."""
    import inspect
    assert rn.DEFAULT_SURROGATE == "gp"
    for f in (rn.run_campaign, rn.run_one, rn.run_grid, rn.write_manifest):
        assert inspect.signature(f).parameters["surrogate"].default == "gp", f.__name__
    called = {}
    orig = rn.run_grid
    try:
        rn.run_grid = lambda *a, **kw: called.update(kw)
        rn.main(["--grid", "pilot", "--out", "unused"])
    finally:
        rn.run_grid = orig
    assert called["surrogate"] == "gp"


def test_seed_parser_and_presets():
    assert rn._seeds("0-2,5") == (0, 1, 2, 5)
    assert set(rn.GRIDS["main"]["selectors"]) == {"random", "greedy", "uncertainty", "bo_ucb", "bo_qei"}
    assert len(rn.GRIDS["main"]["seeds"]) == 10
    assert set(rn.GRIDS["gpcompare"]["selectors"]) == {"greedy", "uncertainty", "bo_ucb", "bo_qei", "bo_ts"}
    assert rn.GRIDS["gpcompare"]["seeds"] == (0, 1, 2, 3, 4)


# -- metrics ----------------------------------------------------------------------------------------

def _curve(hits, step=16):
    return pd.DataFrame({"n_charged": np.arange(len(hits)) * step, "hits_syn": hits})


def test_curve_metrics_by_hand():
    c = _curve([0, 1, 1, 3, 10, 12])  # charged 0, 16, ..., 80
    assert mx.hits_at(c, 80, "hits_syn") == 12 and mx.hits_at(c, 40, "hits_syn") == 1
    assert mx.exp_to(c, "hits_syn") == (64.0, False)
    assert mx.exp_to(_curve([0, 1, 2]), "hits_syn") == (32.0, True)
    perfect = pd.DataFrame({"n_charged": [0, 16, 32], "hits_syn": [0, 16, 32]})
    assert mx.auc(perfect, "hits_syn") == pytest.approx(1.0)
    assert mx.auc(_curve([0, 0, 0]), "hits_syn") == 0.0


def test_holm_and_paired_p():
    adj = mx.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == pytest.approx(0.03) and adj["c"] == pytest.approx(0.06) and adj["b"] == pytest.approx(0.06)
    assert mx.paired_p(np.ones(5), np.ones(5)) == 1.0
    assert mx.paired_p(np.arange(10) + 5.0, np.arange(10)) < 0.01


def test_metrics_end_to_end_on_a_tiny_grid(world, tmp_path):
    for sel in ("random", "bo_ucb"):
        for seed in (0, 1, 2):
            rn.run_one(rn.CampaignSpec(sel, "inh", seed, budget=16, batch_size=8), tmp_path, "compact", TINY,
                       world["fs"], world["factory"], surrogate="ensemble")
    fame = world["pool"][["drug_a", "drug_b"]].drop_duplicates().assign(fame="middle")
    fame_path = tmp_path / "fame.parquet"
    fame.to_parquet(fame_path, index=False)
    files = mx.build(tmp_path, fame_path)
    summary = pd.read_parquet(files["summary"])
    assert set(summary.selector) == {"random", "bo_ucb"} and (summary.n_seeds == 3).all()
    runs = pd.read_parquet(files["runs_summary"])
    assert (runs["hits_middle"] == runs["hits_at_320"]).all()
    text = files["report"].read_text()
    assert "Gate: bo_ucb beats random" in text and "| inh |" in text


# -- real data (not run unless COMBO_INTEGRATION=1) -------------------------------------------------

integration = pytest.mark.skipif(os.environ.get("COMBO_INTEGRATION") != "1",
                                 reason="needs the M1-M3 data; set COMBO_INTEGRATION=1")


@integration
def test_real_short_campaign_every_selector(tmp_path):
    for name in ("random", "bo_ucb", "bo_qei"):
        spec = rn.CampaignSpec(name, "syn", 0, budget=32, batch_size=16)
        msg = rn.run_one(spec, tmp_path, "compact")
        assert msg.startswith("done"), msg
