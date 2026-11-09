"""Tests for combo_agent.data.build_experiments (step 1e)."""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path

import pandas as pd
import pytest

from combo_agent.data import build_experiments as be

# A small world. L1 is an O'Neil line, L2 an ALMANAC-only line, LX a line 1d drops.
# A, B: ALMANAC-only drugs. S1, S2: drugs both studies use under one name. ON1: O'Neil-only.
# SYN: O'Neil's name for drug A. T1, T2: third study.
DRUGS = {"A": "dA", "B": "dB", "S1": "dS1", "S2": "dS2", "ON1": "dON1", "SYN": "dA",
         "T1": "dT1", "T2": "dT2"}


@pytest.fixture(autouse=True)
def small_rules(monkeypatch):
    monkeypatch.setattr(be, "ONEIL_LINES", frozenset({"L1"}))
    monkeypatch.setattr(be, "ONEIL_ONLY_DRUGS", frozenset({"ON1"}))
    monkeypatch.setattr(be, "ONEIL_SYNONYMS", frozenset({"SYN"}))
    monkeypatch.setattr(be, "ONEIL_LABELS", frozenset({"ON1", "SYN"}))
    monkeypatch.setattr(be, "THIRD_STUDY_DRUGS", frozenset({"T1", "T2"}))


def row(d1, d2, line, css, bliss=0.0):
    return {"Drug1_ID": d1, "Drug2_ID": d2, "Cell_Line_ID": line, "CSS": css, "Synergy_ZIP": 0.0,
            "Synergy_Bliss": bliss, "Synergy_Loewe": 0.0, "Synergy_HSA": 0.0}


ROWS = [
    row("A", "B", "L2", 10, 1),     # 0  kept, replicate 1
    row("B", "A", "L2", 20, 3),     # 1  kept, replicate 2 (other order)
    row("A", "S1", "L1", 30),       # 2  kept (A is ALMANAC-only)
    row("S1", "S2", "L1", 40),      # 3  oneil_ambiguous
    row("S1", "S2", "L2", 50),      # 4  kept
    row("ON1", "S1", "L1", 60),     # 5  oneil
    row("ON1", "S2", "L1", 61),     # 6  oneil
    row("SYN", "S1", "L1", 70),     # 7  oneil, same triplet as row 2 -> cross-study overlap
    row("T1", "T2", "L2", 80),      # 8  third_study
    row("A", "B", "LX", 90),        # 9  line_dropped
    row("A", "A", "L2", 5),         # 10 self_pair
]


def tables(rows=ROWS, drugs=DRUGS):
    df = pd.DataFrame(rows)
    drug_map = pd.DataFrame({"source_id": list(drugs), "drug_id": list(drugs.values())})
    line_map = pd.DataFrame({
        "source_id": ["L1", "L2", "LX"], "cellosaurus_ac": ["CVCL_1", "CVCL_2", "CVCL_X"],
        "depmap_id": ["ACH-1", "ACH-2", None], "depmap_is_proxy": [False, True, False],
        "family_id": ["FAM_1", "FAM_2", "FAM_X"], "keep": [True, True, False]})
    return df, drug_map, line_map


def test_each_row_gets_one_fate_in_precedence_order():
    r = be.build(*tables())
    assert list(r.row_fate.fate) == ["kept", "kept", "kept", "oneil_ambiguous", "kept", "oneil",
                                     "oneil", "oneil", "third_study", "line_dropped", "self_pair"]
    assert r.report["fates"] == {"third_study": 1, "oneil": 3, "oneil_ambiguous": 1,
                                 "line_dropped": 1, "self_pair": 1, "kept": 4}
    assert r.report["shared_drug_labels"] == ["S1", "S2"]  # derived from O'Neil rows' partners


def test_replicates_merge_across_drug_order_with_mean_sd_and_count():
    e = be.build(*tables()).experiments.set_index(be.KEY)
    ab = e.loc[("dA", "dB", "CVCL_2")]
    assert (ab.n_reps, ab.css_mean, ab.y_inh, ab.y_syn) == (2, 15.0, 15.0, 2.0)
    assert ab.css_sd == pytest.approx(math.sqrt(50))
    single = e.loc[("dA", "dS1", "CVCL_1")]
    assert single.n_reps == 1 and math.isnan(single.css_sd)
    assert set(e.study_id) == {"NCI-ALMANAC"} and len(e) == 3


def test_line_columns_and_proxy_flag_are_attached():
    e = be.build(*tables()).experiments.set_index(be.KEY)
    assert e.loc[("dA", "dB", "CVCL_2"), "depmap_id"] == "ACH-2"
    assert bool(e.loc[("dA", "dB", "CVCL_2"), "depmap_is_proxy"])
    assert e.loc[("dA", "dB", "CVCL_2"), "family_id"] == "FAM_2"


def test_self_pairs_go_to_their_own_table():
    r = be.build(*tables())
    assert list(zip(r.self_pairs.drug_a, r.self_pairs.drug_b)) == [("dA", "dA")]
    assert not (r.experiments.drug_a == r.experiments.drug_b).any()


def test_oneil_rows_go_to_side_table_and_overlap_is_reported():
    r = be.build(*tables())
    assert sorted(r.oneil_rows.source_row) == [3, 5, 6, 7]
    assert set(r.oneil_rows.fate) == {"oneil", "oneil_ambiguous"}
    assert len(r.overlap) == 1
    o = r.overlap.iloc[0]
    assert (o.drug_a, o.drug_b, o.css_mean_almanac, o.css_mean_oneil) == ("dA", "dS1", 30.0, 70.0)
    assert "dON1" not in set(r.experiments.drug_a) | set(r.experiments.drug_b)


def test_guard_oneil_row_off_the_oneil_lines():
    rows = ROWS + [row("ON1", "S1", "L2", 1)]
    with pytest.raises(be.ExperimentError, match="9 O'Neil lines"):
        be.build(*tables(rows))


def test_guard_unknown_study_confined_to_oneil_lines():
    rows = ROWS + [row("NEW", "A", "L1", 1)]
    with pytest.raises(be.ExperimentError, match="unknown study"):
        be.build(*tables(rows, {**DRUGS, "NEW": "dNEW"}))


def test_guard_unmapped_label():
    rows = ROWS + [row("MYSTERY", "A", "L2", 1)]
    with pytest.raises(be.ExperimentError, match="drug_map"):
        be.build(*tables(rows))


def test_guard_third_study_pairing_with_outsiders():
    rows = ROWS + [row("T1", "A", "L2", 1)]
    with pytest.raises(be.ExperimentError, match="third-study"):
        be.build(*tables(rows))


# -- real data ------------------------------------------------------------------------------------


@pytest.mark.skipif(os.environ.get("COMBO_INTEGRATION") != "1",
                    reason="needs the converted parquet and the 1c/1d outputs; set COMBO_INTEGRATION=1")
def test_real_build_counts_and_reproducibility(tmp_path, monkeypatch):
    monkeypatch.undo()  # the real rules, not the small ones
    from combo_agent.data import ingest_tdc, paths

    df = ingest_tdc.load("tdc_drugcomb")
    interim = paths.interim_dir()
    dm, lm = pd.read_parquet(interim / "drug_map.parquet"), pd.read_parquet(interim / "line_map.parquet")
    digests = []
    for run in ("a", "b"):
        out = be.write_outputs(be.build(df, dm, lm), tmp_path / run)
        digests.append({k: hashlib.sha256(Path(p).read_bytes()).hexdigest() for k, p in out.items()})
    assert digests[0] == digests[1]
    r = be.build(df, dm, lm).report
    assert r["fates"] == {"third_study": 10, "oneil": 4284, "oneil_ambiguous": 1076,
                          "line_dropped": 10024, "self_pair": 111, "kept": 281593}
    assert (r["experiments"], r["pairs"], r["lines"], r["drugs"]) == (272357, 4939, 57, 101)
    assert r["cross_study_overlap"]["triplets"] == 546
    assert r["cross_study_overlap"]["agreement"]["css"]["pearson"] == pytest.approx(0.217, abs=1e-3)
