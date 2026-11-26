"""Tests for combo_agent.lab.oracle: refusals, budget, seed sets, history and the log."""

from __future__ import annotations

import itertools
import json
import os

import numpy as np
import pandas as pd
import pytest

from combo_agent.lab.oracle import KEY, Oracle, OracleError, Triplet

DRUGS = ["D1", "D2", "D3", "D4"]
LINES = ["L1", "L2", "L3", "L4", "L5"]  # L5 is held out


def _world(tmp_path):
    rows = [(a, b, line) for a, b in itertools.combinations(DRUGS, 2) for line in LINES]
    pool = pd.DataFrame(rows, columns=KEY)
    pool["split"] = np.where(pool["cellosaurus_ac"] == "L5", "heldout", "selectable")
    pool["n_reps"] = 1
    pool.loc[0, "n_reps"] = 2
    labels = pool[KEY].copy()
    labels["y_syn"] = np.arange(len(pool), dtype=float)
    labels["y_inh"] = 100.0 - labels["y_syn"]
    labels["bliss_sd"] = np.where(pool["n_reps"] == 2, 1.5, np.nan)
    labels["css_sd"] = np.nan
    path = tmp_path / "labels.parquet"
    labels.to_parquet(path, index=False)
    sel = pool[pool["split"] == "selectable"][KEY].values.tolist()
    seed_sets = {"0": sel[:3], "1": sel[3:6]}
    return pool, path, seed_sets


@pytest.fixture
def world(tmp_path):
    return _world(tmp_path)


def make(world, **kw):
    pool, path, seed_sets = world
    kw.setdefault("seed_sets", seed_sets)
    kw.setdefault("thresholds", {"y_syn": 20.0, "y_inh": 90.0})
    return Oracle(pool, labels_path=path, budget=kw.pop("budget", 6), batch_size=kw.pop("batch_size", 3), **kw)


def test_seed_set_is_free_round_zero_and_depends_on_seed(world):
    o0, o1, again = make(world, seed=0), make(world, seed=1), make(world, seed=0)
    assert o0.budget_left() == 6 and o0.round == 0
    assert [r.round for r in o0.history()] == [0, 0, 0]
    assert [r.triplet for r in o0.history()] == [r.triplet for r in again.history()]
    assert {r.triplet for r in o0.history()}.isdisjoint({r.triplet for r in o1.history()})
    assert len(o0.candidates()) == 6 * 4 - 3  # 6 pairs x 4 selectable lines, minus the seed set


def test_no_seed_set_when_seed_is_none_and_error_when_missing(world):
    assert make(world, seed=None).history() == []
    with pytest.raises(ValueError, match="no seed set"):
        make(world, seed=7)


def test_run_reveals_label_values_and_charges_budget(world):
    o = make(world, seed=None)
    res = o.run([("D1", "D2", "L1"), ("D1", "D2", "L2")])
    assert [r.y_syn for r in res] == [0.0, 1.0] and [r.y_inh for r in res] == [100.0, 99.0]
    assert res[0].n_reps == 2 and res[0].y_syn_sd == 1.5 and np.isnan(res[1].y_syn_sd)
    assert o.budget_left() == 4 and o.round == 1 and res[0].round == 1


def test_swapped_drug_order_is_the_same_experiment(world):
    o = make(world, seed=None)
    (r,) = o.run([("D2", "D1", "L1")])
    assert r.triplet == Triplet("D1", "D2", "L1")
    with pytest.raises(OracleError, match="already revealed"):
        o.run([Triplet("D1", "D2", "L1")])


@pytest.mark.parametrize("batch, message", [
    ([], "empty"),
    ([("D1", "D2", "L1"), ("D1", "D2", "L2"), ("D1", "D2", "L3"), ("D1", "D2", "L4")], "exceeds batch_size"),
    ([("D1", "D2", "L1"), ("D2", "D1", "L1")], "repeats"),
    ([("D1", "D2", "L5")], "held-out"),
    ([("D1", "D9", "L1")], "not in the pool"),
    ([("D1", "D2")], "a triplet is"),
])
def test_refusals_change_nothing(world, batch, message):
    o = make(world, seed=0)
    before = (o.budget_left(), o.round, len(o.history()), len(o.candidates()))
    with pytest.raises(OracleError, match=message):
        o.run(batch)
    assert (o.budget_left(), o.round, len(o.history()), len(o.candidates())) == before


def test_rerun_of_seed_triplet_is_refused(world):
    o = make(world, seed=0)
    with pytest.raises(OracleError, match="already revealed"):
        o.run([o.history()[0].triplet])


def test_budget_is_enforced_atomically(world):
    o = make(world, seed=None, budget=4)
    cands = o.candidates()
    o.run(cands[:3])
    with pytest.raises(OracleError, match="remaining budget 1"):
        o.run(cands[3:5])
    assert o.budget_left() == 1 and len(o.history()) == 3
    o.run(cands[3:4])
    assert o.budget_left() == 0
    with pytest.raises(OracleError):
        o.run(cands[4:5])


def test_candidates_shrink_and_exclude_heldout(world):
    o = make(world, seed=None)
    c = o.candidates()
    assert all(t.cellosaurus_ac != "L5" for t in c) and len(c) == 24
    o.run(c[:3])
    assert len(o.candidates()) == 21
    frame = o.candidate_frame()
    assert len(frame) == 21 and not {"y_syn", "y_inh", "bliss_sd", "css_sd"} & set(frame.columns)


def test_public_surface_carries_no_outcomes(world):
    o = make(world, seed=None)
    public = {k: v for k, v in vars(o).items() if not k.startswith("_Oracle__")}
    for v in public.values():
        if isinstance(v, pd.DataFrame):
            assert not {"y_syn", "y_inh"} & set(v.columns)
    assert "y_syn" not in repr(o)


def test_is_hit_uses_public_thresholds(world):
    o = make(world, seed=None)
    lo, hi = o.run([("D1", "D2", "L1"), ("D3", "D4", "L4")])  # y_syn 0 and 28
    assert not o.is_hit(lo, "syn") and o.is_hit(hi, "syn")
    assert o.is_hit(lo, "inh") and not o.is_hit(hi, "inh")  # y_inh 100 and 72


def test_log_records_every_reveal(world, tmp_path):
    log = tmp_path / "runs" / "oracle.jsonl"
    o = make(world, seed=0, log_path=log, selector="random")
    o.run(o.candidates()[:2])
    lines = [json.loads(x) for x in log.read_text().splitlines()]
    assert [x["round"] for x in lines] == [0, 1] and [x["charged"] for x in lines] == [False, True]
    assert lines[1]["selector"] == "random" and len(lines[1]["results"]) == 2
    assert lines[0]["results"][0]["y_syn_sd"] is None or isinstance(lines[0]["results"][0]["y_syn_sd"], float)


def test_misaligned_labels_are_realigned_by_key(world, tmp_path):
    pool, path, seed_sets = world
    shuffled = pd.read_parquet(path).sample(frac=1, random_state=0)
    p2 = tmp_path / "shuffled.parquet"
    shuffled.to_parquet(p2, index=False)
    o = Oracle(pool, labels_path=p2, seed=None)
    (r,) = o.run([("D1", "D2", "L2")])
    assert r.y_syn == 1.0


# -- real data ------------------------------------------------------------------------------------

integration = pytest.mark.skipif(os.environ.get("COMBO_INTEGRATION") != "1",
                                 reason="needs the M1 data bundle; set COMBO_INTEGRATION=1")


@integration
def test_real_oracle_random_campaign():
    from combo_agent.data import paths

    o = Oracle.from_cache(seed=None)
    assert len(o.candidates()) == 219_239
    rng = np.random.default_rng(0)
    frame = o.candidate_frame()
    picks = frame.iloc[rng.choice(len(frame), 320, replace=False)][KEY].values.tolist()
    results = [r for i in range(20) for r in o.run(picks[16 * i:16 * (i + 1)])]
    assert o.budget_left() == 0 and o.round == 20 and len(results) == 320
    pool = pd.read_parquet(paths.processed_dir() / "pool.parquet")
    held = pool[pool["split"] == "heldout"].iloc[0]
    o2 = Oracle.from_cache(seed=None)
    with pytest.raises(OracleError, match="held-out"):
        o2.run([tuple(held[KEY])])
    labels = pd.read_parquet(paths.processed_dir() / "labels.parquet").set_index(KEY)
    for r in results:
        row = labels.loc[tuple(r.triplet)]
        assert r.y_syn == row["y_syn"] and o.is_hit(r, "syn") == bool(row["is_hit_syn"])
        assert o.is_hit(r, "inh") == bool(row["is_hit_inh"])
