"""Tests for step 1f: combo_agent.data.ingest_cellminer and combo_agent.data.build_features."""

from __future__ import annotations

import hashlib
import io
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from combo_agent.data import build_features as bf
from combo_agent.data import ingest_cellminer as ic

# -- CellMiner ------------------------------------------------------------------------------------


def _lines():
    line_map = pd.DataFrame({"source_id": ["HCT116", "A549", "NCIH23"],
                             "cellosaurus_ac": ["CVCL_0291", "CVCL_0023", "CVCL_1547"]})
    cell_lines = pd.DataFrame({"cellosaurus_ac": ["CVCL_0291", "CVCL_0023", "CVCL_1547"],
                               "cellosaurus_name": ["HCT 116", "A-549", "NCI-H23"]})
    return line_map, cell_lines


def test_line_label_strips_panel_and_atcc_suffix():
    assert ic.line_label("CO:HCT-116") == "HCT-116"
    assert ic.line_label("LC:A549/ATCC") == "A549"


def test_match_lines_by_normalized_name_and_skips_unknown_lines():
    m = ic.match_lines(["CO:HCT-116", "LC:A549/ATCC", "LC:NCI-H23", "ME:MDA-N"], *_lines())
    assert m == {"CO:HCT-116": "CVCL_0291", "LC:A549/ATCC": "CVCL_0023", "LC:NCI-H23": "CVCL_1547"}


def test_match_lines_refuses_a_column_matching_two_lines():
    line_map, cell_lines = _lines()
    line_map = pd.concat([line_map, pd.DataFrame({"source_id": ["HCT-116"], "cellosaurus_ac": ["CVCL_X"]})])
    with pytest.raises(ValueError, match="several lines"):
        ic.match_lines(["CO:HCT-116"], line_map, cell_lines)


def test_convert_command_is_isolated():
    cmd = ic.convert_command(Path("in.xls"), Path("out.parquet"))
    assert cmd[:4] == ["uv", "run", "--isolated", "--no-project"]
    assert "xlrd==2.0.1" in cmd and "-I" in cmd


def test_download_rejects_a_changed_file(tmp_path):
    ds = ic.DATASETS["expression"]
    (tmp_path / ds.zip_name).write_bytes(b"not the pinned file")
    with pytest.raises(ValueError, match="sha256"):
        ic.download(ds, tmp_path)


def test_extract_takes_only_the_workbook(tmp_path):
    ds = ic.DATASETS["expression"]
    zpath = tmp_path / ds.zip_name
    with zipfile.ZipFile(zpath, "w") as z:
        z.writestr(ds.workbook, b"xls bytes")
        z.writestr("output/html/big.html", b"ignored")
    out = ic.extract(zpath, ds)
    assert out.read_bytes() == b"xls bytes"
    assert not (zpath.with_suffix("") / "output/html/big.html").exists()


# -- identity check -------------------------------------------------------------------------------


def _profiles(seed=0, n_lines=6, n_genes=300):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n_lines, n_genes)) * 3
    acs = [f"CVCL_{i}" for i in range(n_lines)]
    deps = [f"ACH-{i}" for i in range(n_lines)]
    genes = [f"G{j}" for j in range(n_genes)]
    cm = pd.DataFrame(base + rng.normal(scale=0.3, size=base.shape), index=acs, columns=genes)
    dm = pd.DataFrame(base + rng.normal(scale=0.3, size=base.shape), index=deps, columns=genes)
    return cm, dm, dict(zip(acs, deps))


def test_identity_check_passes_matched_lines_and_flags_a_swapped_sample():
    cm, dm, own = _profiles()
    t = bf.identity_check(cm, dm, own, n_genes=200).set_index("cellosaurus_ac")
    assert t.own_is_best.all() and not t.suspect.any()
    cm.loc["CVCL_0"] = cm.loc["CVCL_3"].values  # CellMiner's sample for line 0 is really line 3
    t = bf.identity_check(cm, dm, own, n_genes=200).set_index("cellosaurus_ac")
    assert bool(t.loc["CVCL_0", "suspect"]) and t.loc["CVCL_0", "best_depmap_id"] == "ACH-3"
    assert t.suspect.sum() == 1


# -- drugs ----------------------------------------------------------------------------------------


def test_morgan_bits_shape_and_similarity():
    a, b = bf.morgan_bits("CCO"), bf.morgan_bits("CCCO")
    assert a.shape == (bf.FP_BITS,) and a.dtype == np.int8 and a.sum() > 0
    assert (a & b).sum() > 0  # shared fragments
    with pytest.raises(bf.FeatureError):
        bf.morgan_bits("not a smiles")


def test_drug_features_separate_any_target_from_gene_targets():
    drugs = pd.DataFrame({"drug_id": ["D1", "D2", "D3"], "preferred_name": ["KINASEI", "ALKYLATOR", "NONE"],
                          "smiles": ["CCO", "ClCCNCCCl", "CCN"], "smiles_origin": ["chembl_parent"] * 3,
                          "n_targets": [1, 1, 0]})
    targets = pd.DataFrame({"drug_id": ["D1", "D2"], "gene_symbols": ["EGFR|ERBB2", None],
                            "action_type": ["INHIBITOR", "CROSSLINKING AGENT"],
                            "target_type": ["SINGLE PROTEIN", "NUCLEIC-ACID"]})
    f = bf.drug_features(drugs, targets, {"D1", "D2"}).set_index("drug_id")
    assert list(f.has_target) == [1, 1, 0] and list(f.has_target_gene) == [1, 0, 0]
    assert f.loc["D1", ["target_EGFR", "target_ERBB2"]].tolist() == [1, 1]
    assert f.loc["D2", "ttype_NUCLEIC_ACID"] == 1 and f.loc["D3", "in_pool"] == 0
    assert not f.drop(columns=["preferred_name", "smiles_origin"]).isna().any().any()


# -- build ----------------------------------------------------------------------------------------


def test_impute_expression_prefers_arrays_then_depmap_then_mean():
    rng = np.random.default_rng(1)
    lines = [f"CVCL_{i}" for i in range(10)]
    truth = pd.DataFrame(rng.normal(5, 2, size=(10, 4)), index=lines, columns=["A", "B", "C", "D"])
    array = (truth[["A", "B"]] - 5) / 2 * 1.5  # a linear rescaling of the truth, genes A and B only
    dm = pd.DataFrame((truth[["C"]].values + 1.0), index=[f"ACH-{i}" for i in range(10)], columns=["C"])
    own = {a: f"ACH-{i}" for i, a in enumerate(lines)}
    vals, src = bf.impute_expression("CVCL_0", truth.drop(index="CVCL_0"), array, dm, own, "ACH-0")
    assert list(src) == ["cellminer_array", "cellminer_array", "depmap", "gene_mean"]
    assert vals[["A", "B", "C"]].values == pytest.approx(truth.loc["CVCL_0", ["A", "B", "C"]].values, abs=1e-6)
    assert vals["D"] == pytest.approx(truth.drop(index="CVCL_0")["D"].mean())


def test_build_prefers_own_then_parent_mutations_and_imputes_suspect_expression(monkeypatch):
    monkeypatch.setattr(bf, "N_EXPRESSION_GENES", 50)
    cm, dm, own = _profiles(n_lines=6, n_genes=300)
    acs = list(own)
    # line 0: CellMiner RNA is really line 3's (suspect). line 5: no own DepMap profile; parent = line 4.
    cm.loc["CVCL_0"] = cm.loc["CVCL_3"].values
    cell_lines = pd.DataFrame({
        "cellosaurus_ac": acs, "cellosaurus_name": [f"L{i}" for i in range(6)],
        "depmap_id": [own[a] for a in acs], "depmap_is_proxy": False,
        "family_id": [f"F{i}" for i in range(6)], "problematic": "",
        "derived_from": ["", "", "", "", "", "CVCL_4"]})
    experiments = pd.DataFrame({"drug_a": ["D1"] * 6, "drug_b": ["D2"] * 6, "cellosaurus_ac": acs})
    model = pd.DataFrame({"ModelID": [own[a] for a in acs], "RRID": acs,
                          "OncotreeLineage": ["Lung"] * 3 + ["Skin"] * 3,
                          "OncotreePrimaryDisease": "x", "OncotreeSubtype": "y"})
    calls = pd.DataFrame(0, index=[own[a] for a in acs[:5]], columns=list(bf.DRIVER_GENES), dtype="int8")
    calls.loc["ACH-4", "TP53"] = 1
    calls.loc["ACH-1", "KRAS"] = 1
    cm_mut = pd.DataFrame(0.0, index=acs, columns=list(bf.DRIVER_GENES))
    cm_mut.loc["CVCL_5", "BRAF"] = 50.0  # CellMiner's own call for line 5, kept for comparison only
    cm_cn = pd.DataFrame(0.1, index=acs, columns=["KRAS", "MYC"])
    cm_cn.loc["CVCL_2", "MYC"] = np.nan
    drugs = pd.DataFrame({"drug_id": ["D1", "D2"], "preferred_name": ["A", "B"], "smiles": ["CCO", "CCN"],
                          "smiles_origin": ["chembl_parent"] * 2, "n_targets": [1, 0]})
    targets = pd.DataFrame({"drug_id": ["D1"], "gene_symbols": ["EGFR"], "action_type": ["INHIBITOR"],
                            "target_type": ["SINGLE PROTEIN"]})
    truth0 = dm.loc["ACH-0"].values  # line 0's real profile, as the arrays see it
    cm_array = cm.copy()
    cm_array.loc["CVCL_0"] = truth0
    r = bf.build(experiments, cell_lines, cell_lines.rename(columns={"cellosaurus_name": "source_id"}),
                 drugs, targets, model, cm, cm_mut, cm_cn, dm, calls, set(calls.index), cm_array=cm_array)
    cells = r.tables["cells"].set_index("cellosaurus_ac")
    mut = r.tables["cell_mutations"].set_index("cellosaurus_ac")
    assert cells.loc["CVCL_0", "expression_imputed"] == 1 and cells.expression_imputed.sum() == 1
    assert cells.expression_missing.sum() == 0 and cells.loc["CVCL_0", "expression_source"] == "imputed"
    imputed = r.tables["cell_expression"].set_index("cellosaurus_ac").loc["CVCL_0"]
    swapped = cm.loc["CVCL_3"]
    real = pd.Series(truth0, index=cm.columns).clip(lower=0)
    assert np.corrcoef(imputed, real)[0, 1] > np.corrcoef(imputed, swapped)[0, 1]  # not the mixed-up sample
    assert mut.loc["CVCL_1", "KRAS"] == 1 and mut.loc["CVCL_1", "mutation_source"] == "depmap"
    assert mut.loc["CVCL_5", "mutation_source"] == "depmap_parent" and mut.loc["CVCL_5", "TP53"] == 1
    assert mut.loc["CVCL_5", "BRAF"] == 0  # parent wins over CellMiner
    assert r.tables["cell_mutations_cellminer"].set_index("cellosaurus_ac").loc["CVCL_5", "BRAF"] == 1
    f = r.tables["cell_features"].set_index("cellosaurus_ac")
    assert not f.isna().any().any()
    assert f.loc["CVCL_2", "cn_missing_MYC"] == 1 and f.loc["CVCL_2", "cn_MYC"] == 0.0
    assert f.loc["CVCL_0", "expression_imputed"] == 1
    assert sum(c.startswith("expr_") for c in f.columns) == 50
    assert {"lineage_Lung", "lineage_Skin"} <= set(f.columns)


# -- real data ------------------------------------------------------------------------------------


@pytest.mark.skipif(os.environ.get("COMBO_INTEGRATION") != "1",
                    reason="needs DepMap, CellMiner and the 1c-1e outputs; set COMBO_INTEGRATION=1")
def test_real_features(tmp_path):
    from combo_agent.data import paths

    rc = bf.main(["--out", str(tmp_path / "a")])
    assert rc == 0
    bf.main(["--out", str(tmp_path / "b")])
    for p in (tmp_path / "a").iterdir():
        assert hashlib.sha256(p.read_bytes()).digest() == hashlib.sha256((tmp_path / "b" / p.name).read_bytes()).digest()
    import json

    r = json.loads((tmp_path / "a" / "features_report.json").read_text())
    assert r["pool_lines"] == 57 and r["expression"]["missing_lines"] == []
    assert r["expression"]["imputed_lines"] == ["SR"]
    assert r["expression"]["imputation_loo_quality"]["deviation_r_median"] > 0.9
    assert r["mutations"]["source_counts"] == {"depmap": 54, "depmap_parent": 3}
    assert r["relationships"]["SNB-19_closest_cellminer"][0][0] == "U-251MG"
    assert r["drugs"]["in_pool"] == 101
    assert paths.interim_dir().exists()
