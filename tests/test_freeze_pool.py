"""Tests for combo_agent.data.freeze_pool (step 1g)."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from combo_agent.data import freeze_pool as fz


def test_quotas_use_largest_remainder_and_sum_to_20_percent():
    sizes = pd.Series({"Lung": 8, "Skin": 8, "Kidney": 8, "Bowel": 7, "CNS": 6, "Ovary": 6, "Breast": 5,
                       "Lymphoid": 4, "Myeloid": 2, "Prostate": 2, "Pleura": 1})
    q = fz.quotas(sizes)
    assert q.sum() == round(0.2 * sizes.sum()) == 11
    assert q.to_dict() == {"Bowel": 1, "Breast": 1, "CNS": 1, "Kidney": 2, "Lung": 2, "Lymphoid": 1,
                           "Myeloid": 0, "Ovary": 1, "Pleura": 0, "Prostate": 0, "Skin": 2}


def _cells(n_per=(5, 5, 5), fams=None, bad=()):
    rows, k = [], 0
    for li, n in enumerate(n_per):
        for _ in range(n):
            ac = f"CVCL_{k:03d}"
            rows.append({"cellosaurus_ac": ac, "cellosaurus_name": f"L{k}", "lineage": f"Lin{li}",
                         "family_id": (fams or {}).get(ac, f"FAM_{ac}"), "depmap_id": f"ACH-{k}",
                         "depmap_is_proxy": False, "expression_source": "cellminer",
                         "mutation_source": "depmap"})
            k += 1
    c = pd.DataFrame(rows)
    for ac in bad:
        c.loc[c.cellosaurus_ac == ac, "expression_source"] = "imputed"
    return c


def test_eligibility_excludes_imputed_borrowed_proxy_and_their_families():
    c = _cells(fams={"CVCL_000": "FAM_X", "CVCL_001": "FAM_X"}, bad=["CVCL_001"])
    c.loc[c.cellosaurus_ac == "CVCL_005", "mutation_source"] = "depmap_parent"
    c.loc[c.cellosaurus_ac == "CVCL_006", "depmap_is_proxy"] = True
    ok = fz.eligible(c).set_axis(c.cellosaurus_ac)
    assert not ok["CVCL_000"]  # own data, but its family partner is imputed
    assert not ok[["CVCL_001", "CVCL_005", "CVCL_006"]].any()
    assert ok.sum() == 15 - 4


def test_heldout_keeps_families_together_and_is_deterministic():
    fams = {"CVCL_000": "FAM_X", "CVCL_001": "FAM_X"}  # a 2-line family in lineage 0 (quota 1)
    c = _cells(n_per=(5, 5, 5), fams=fams)
    a, rec = fz.choose_heldout(c, seed=0)
    b, _ = fz.choose_heldout(c, seed=0)
    assert a == b and len(a) == 3
    assert not ({"CVCL_000", "CVCL_001"} & set(a))  # a family of 2 never overshoots a quota of 1
    assert [r["quota"] for r in rec.values()] == [1, 1, 1]


def test_heldout_raises_when_a_quota_cannot_be_met():
    c = _cells(n_per=(5,), bad=["CVCL_000", "CVCL_001", "CVCL_002", "CVCL_003", "CVCL_004"])
    with pytest.raises(fz.FreezeError, match="quota"):
        fz.choose_heldout(c)


def _world():
    cells = _cells(n_per=(5, 5, 5))
    rng = np.random.default_rng(0)
    rows = []
    for ac in cells.cellosaurus_ac:
        for i in range(20):
            rows.append({"drug_a": f"D{i:02d}", "drug_b": f"D{i + 1:02d}", "cellosaurus_ac": ac, "n_reps": 1,
                         "study_id": "NCI-ALMANAC", "depmap_id": "x", "depmap_is_proxy": False, "family_id": "f",
                         **{c: float(rng.normal()) for c in fz.OUTCOME_COLS}})
    e = pd.DataFrame(rows)
    e["y_inh"] = e.css_mean * 20 + 20
    e["y_syn"] = e.bliss_mean * 5
    cell_features = cells[["cellosaurus_ac"]].assign(mut_KRAS=[i % 2 for i in range(15)], expr_G1=1.0)
    drug_ids = sorted(set(e.drug_a) | set(e.drug_b))
    drug_features = pd.DataFrame({"drug_id": drug_ids, "preferred_name": drug_ids, "in_pool": 1, "fp_0": 0})
    drugs = pd.DataFrame({"drug_id": drug_ids, "chembl_parent_id": drug_ids, "n_targets": [0] + [1] * 20,
                          "mechanism_source": [None] + drug_ids[1:2] + ["OTHER"] + drug_ids[3:]})
    self_pairs = e.iloc[:0]
    return e, self_pairs, cells, cell_features, drug_features, drugs


def test_build_splits_labels_from_pool_and_thresholds_on_selectable_only():
    e, sp, cells, cf, df, d = _world()
    r = fz.build(e, sp, cells, cf, df, d, upstream={"replicate_r_bliss": 0.2})
    pool, labels = r.tables["pool"], r.tables["labels"]
    assert not set(fz.OUTCOME_COLS) & set(pool.columns)
    assert labels[fz.KEY].equals(pool[fz.KEY])
    sel = pool.split == "selectable"
    t = r.splits["thresholds"]
    assert t["y_syn"] == pytest.approx(labels.loc[sel.values, "y_syn"].quantile(0.95), abs=1e-4)
    assert t["n_selectable"] == int(sel.sum())
    assert labels.loc[sel.values, "is_hit_syn"].mean() == pytest.approx(0.05, abs=0.01)
    held = set(pool.loc[~sel, "cellosaurus_ac"])
    assert held == set(r.splits["heldout_lines"]) and len(held) == 3
    status = r.tables["drugs"].set_index("drug_id").target_status
    assert status["D00"] == "none_in_chembl" and status["D02"] == "chembl_proxy" and status["D01"] == "chembl"


def test_data_card_is_deterministic_and_has_no_timestamps():
    args = _world()
    a = fz.build(*args).card
    b = fz.build(*args).card
    assert a == b
    assert "2026" not in a and "/Users" not in a


def test_syn_hit_low_kill_flags_synergy_hits_that_do_not_kill():
    e, sp, cells, cf, df, d = _world()
    e.loc[0, ["y_syn", "y_inh"]] = [100.0, -5.0]
    labels = fz.build(e, sp, cells, cf, df, d).tables["labels"].set_index(fz.KEY)
    key = tuple(e.loc[0, fz.KEY])
    assert labels.loc[key, "is_hit_syn"] == 1 and labels.loc[key, "syn_hit_low_kill"] == 1


@pytest.mark.skipif(os.environ.get("COMBO_INTEGRATION") != "1",
                    reason="needs the 1e/1f outputs; set COMBO_INTEGRATION=1")
def test_real_freeze_reproducible(tmp_path):
    for run in ("a", "b"):
        assert fz.main(["--out", str(tmp_path / run)]) == 0
    for p in (tmp_path / "a").iterdir():
        assert hashlib.sha256(p.read_bytes()).digest() == hashlib.sha256((tmp_path / "b" / p.name).read_bytes()).digest(), p.name
    s = json.loads((tmp_path / "a" / "splits.json").read_text())
    assert len(s["heldout_lines"]) == 11
    assert not set(s["heldout_names"]) & {"SR", "SNB-19", "NCI-H322M", "HL-60(TB)", "U-251MG"}
    assert s["thresholds"]["y_syn"] == pytest.approx(9.2989, abs=1e-3)
