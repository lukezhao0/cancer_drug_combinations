"""Tests for combo_agent.lab.splits: seed sets, PubMed queries, the API-key guard, fame classes."""

from __future__ import annotations

import itertools
import json
import os
import urllib.parse

import numpy as np
import pandas as pd
import pytest

from combo_agent.lab import splits as sp
from combo_agent.lab.oracle import KEY, Oracle


def _pool(n_drugs=6, n_lines=12):
    drugs = [f"CHEMBL{i}" for i in range(1, n_drugs + 1)]
    rows = [(a, b, f"L{j:02d}") for a, b in itertools.combinations(sorted(drugs), 2) for j in range(n_lines)]
    pool = pd.DataFrame(rows, columns=KEY)
    pool["split"] = np.where(pool["cellosaurus_ac"].isin(["L00", "L01"]), "heldout", "selectable")
    pool["n_reps"] = 1
    return pool


def test_seed_sets_are_deterministic_selectable_distinct_and_differ():
    pool = _pool()
    a, b = sp.seed_sets(pool, size=8), sp.seed_sets(pool.sample(frac=1, random_state=1), size=8)
    assert a == b  # independent of row order
    assert set(a) == {str(s) for s in sp.SEEDS}
    for trips in a.values():
        assert len({tuple(t) for t in trips}) == 8
        assert all(t[2] not in ("L00", "L01") for t in trips)
    assert a["0"] != a["1"]


def test_seed_sets_work_with_the_oracle(tmp_path):
    pool = _pool()
    labels = pool[KEY].assign(y_syn=1.0, y_inh=2.0, bliss_sd=np.nan, css_sd=np.nan)
    labels.to_parquet(tmp_path / "labels.parquet", index=False)
    o = Oracle(pool, labels_path=tmp_path / "labels.parquet", seed=3, seed_sets=sp.seed_sets(pool, size=8))
    assert len(o.history()) == 8 and o.budget_left() == o.budget


def test_queries_quote_terms_and_require_a_cancer_term():
    q = sp.pair_query(["fluorouracil", "5-FU"], ["cisplatin"])
    assert q.startswith('("fluorouracil"[tiab] OR "5-FU"[tiab]) AND ("cisplatin"[tiab]) AND (neoplasms[mh]')
    url = sp.count_url(q)
    params = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
    assert params["term"] == [q] and params["maxdate"] == [sp.MAXDATE] and params["rettype"] == ["count"]
    assert "api_key" not in params


def test_parse_count_rejects_errors():
    assert sp.parse_count({"esearchresult": {"count": "17"}}, "u") == 17
    with pytest.raises(sp.SplitsError):
        sp.parse_count({"esearchresult": {"count": "0", "ERROR": "bad"}}, "u")
    with pytest.raises(sp.SplitsError):
        sp.parse_count({"error": "API rate limit exceeded"}, "u")


@pytest.mark.parametrize("counts, famous, obscure, fallback", [
    ([0, 0, 1, 2, 3, 4, 5, 100], {5, 100}, {0}, False),          # q3 = 4.25
    ([0, 0, 0, 0, 0, 0, 0, 0, 3, 9], {9}, {0}, True),             # q3 = 0 -> top quartile of nonzero (7.5)
    ([0, 0, 0, 0], set(), {0}, True),
])
def test_classify(counts, famous, obscure, fallback):
    s = pd.Series(counts)
    fame, info = sp.classify(s)
    assert info["fallback"] == fallback
    assert set(s[fame == "famous"]) == famous
    assert set(s[fame == "obscure"]) <= obscure and (s[fame == "obscure"] == 0).all()
    assert (s[fame == "famous"] >= info["famous_min_count"]).all()
    assert ((s > 0) & (fame == "obscure")).sum() == 0
    if not fallback:
        assert info["famous_min_count"] == pytest.approx(s.quantile(0.75))


def test_api_key_never_reaches_the_cache(tmp_path):
    seen = []

    def fake(url):
        seen.append(url)
        return 200, {"esearchresult": {"count": "5"}}

    c = sp.Counter(tmp_path / "cache", offline=False, api_key="SECRET123", fetch=fake, workers=2)
    assert c.counts(["a", "b", "c"]) == [5, 5, 5]
    assert all("api_key=SECRET123" in u for u in seen)
    for f in (tmp_path / "cache").iterdir():
        assert "SECRET123" not in f.read_text()


def test_api_key_from_env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("NCBI_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("OPENAI_API_KEY=x\nNCBI_API_KEY='abc'\n")
    assert sp.api_key_from_env(env) == "abc"
    monkeypatch.setenv("NCBI_API_KEY", "def")
    assert sp.api_key_from_env(env) == "def"


def _fake_pubmed(single: dict[str, int], pair: dict[frozenset, int], terms: dict[str, list[str]]):
    """A fake E-utilities that answers by which drugs' clauses are in the query."""
    clause = {d: sp.drug_clause(t) for d, t in terms.items()}

    def fetch(url):
        q = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)["term"][0]
        ds = [d for d, c in clause.items() if c in q]
        n = single[ds[0]] if len(ds) == 1 else pair.get(frozenset(ds), 0)
        return 200, {"esearchresult": {"count": str(n)}}
    return fetch


def _build_world(tmp_path):
    pool = _pool(n_drugs=4)
    drugs = sorted(set(pool.drug_a) | set(pool.drug_b))
    labels = pool[KEY].copy()
    rng = np.random.default_rng(0)
    labels["y_syn"], labels["y_inh"] = rng.normal(size=len(pool)), rng.normal(size=len(pool))
    labels["bliss_sd"] = labels["css_sd"] = np.nan
    labels["is_hit_syn"] = (labels.y_syn > 1).astype(int)
    labels["is_hit_inh"] = (labels.y_inh > 1).astype(int)
    labels["syn_hit_low_kill"] = 0
    proc = tmp_path / "processed"
    proc.mkdir()
    pool.to_parquet(proc / "pool.parquet", index=False)
    labels.to_parquet(proc / "labels.parquet", index=False)
    pd.DataFrame({"drug_id": drugs, "preferred_name": [f"DRUG{i}" for i in range(len(drugs))]}).to_parquet(
        proc / "drugs.parquet", index=False)
    terms = {d: [f"drug{i}", f"d{i}x"] for i, d in enumerate(drugs)}
    tp = tmp_path / "terms.csv"
    tp.write_text("drug_id,preferred_name,terms\n" + "".join(f"{d},X,{'|'.join(t)}\n" for d, t in terms.items()))
    single = {drugs[0]: 500, drugs[1]: 300, drugs[2]: 40, drugs[3]: 0}  # drug 3 has no papers
    pair = {frozenset(drugs[:2]): 120, frozenset([drugs[0], drugs[2]]): 3}
    return proc, tp, _fake_pubmed(single, pair, terms), drugs


def test_build_then_offline_rebuild_is_byte_identical(tmp_path):
    proc, tp, fetch, drugs = _build_world(tmp_path)
    calls = []
    counting = lambda u: (calls.append(u), fetch(u))[1]  # noqa: E731
    a = sp.build(tmp_path / "a", offline=False, processed=proc, cache_dir=tmp_path / "cache", terms_path=tp,
                 fetch=counting, log=lambda m: None)
    n_online = len(calls)
    assert n_online == 4 + 3  # 4 single counts; 6 pairs minus the 3 with the paperless drug
    b = sp.build(tmp_path / "b", offline=True, processed=proc, cache_dir=tmp_path / "cache", terms_path=tp,
                 fetch=counting, log=lambda m: None)
    assert len(calls) == n_online
    for k in ("seed_sets", "pair_fame", "drug_pubmed", "m2_card"):
        assert a[k].read_bytes() == b[k].read_bytes(), k
    fame = pd.read_parquet(a["pair_fame"]).set_index(["drug_a", "drug_b"])
    assert fame.loc[tuple(drugs[:2]), "fame"] == "famous"
    assert (fame.loc[[k for k in fame.index if drugs[3] in k], "fame"] == "obscure").all()
    card = a["m2_card"].read_text()
    assert "DRUG0 + DRUG1" in card and "SECRET" not in card
    assert set(json.loads(a["seed_sets"].read_text())["seed_sets"]) == {str(s) for s in sp.SEEDS}


def test_build_fails_when_a_pair_exceeds_a_drug_count(tmp_path):
    proc, tp, fetch, drugs = _build_world(tmp_path)

    def bad(url):
        status, body = fetch(url)
        if "drug1" in url and "drug2" in url:
            body = {"esearchresult": {"count": "999"}}
        return status, body
    with pytest.raises(sp.SplitsError, match="exceed"):
        sp.build(tmp_path / "x", offline=False, processed=proc, cache_dir=tmp_path / "c", terms_path=tp,
                 fetch=bad, log=lambda m: None)


def test_curated_terms_cover_every_pool_drug_once():
    terms = sp.load_terms()
    assert len(terms) == 101
    assert terms["CHEMBL185"][:2] == ["fluorouracil", "5-fluorouracil"]
    assert "rapamycin" in terms["CHEMBL413"] and "SN-38" in terms["CHEMBL837"]


# -- real data ------------------------------------------------------------------------------------

integration = pytest.mark.skipif(os.environ.get("COMBO_INTEGRATION") != "1",
                                 reason="needs the M1 data bundle; set COMBO_INTEGRATION=1")


@integration
def test_real_terms_match_pool_and_seed_sets_reproduce():
    from combo_agent.data import paths

    pool = pd.read_parquet(paths.processed_dir() / "pool.parquet")
    assert set(sp.load_terms()) == set(pool.drug_a) | set(pool.drug_b)
    s = sp.seed_sets(pool)
    assert s == sp.seed_sets(pool) and all(len(v) == 32 for v in s.values())
    saved = paths.processed_dir() / "m2" / "seed_sets.json"
    if saved.exists():
        assert json.loads(saved.read_text())["seed_sets"] == s
