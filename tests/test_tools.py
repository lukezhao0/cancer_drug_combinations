"""Tests for combo_agent.tools: the store allowlist, rules, tool functions, registry and probes."""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from combo_agent.tools import ToolError, omics, registry, rules
from combo_agent.tools import probes as tool_probes
from combo_agent.tools.store import ALLOWED_TABLES, DENIED_FILES, FeatureStore

LINES = [  # ac, name, lineage, split, KRAS, PTEN mut, cn_PTEN, cn_MYC, EGFR expr
    ("CVCL_1", "A-549", "Lung", "heldout", 1, 0, 0.0, 1.5, 6.0),
    ("CVCL_2", "NCI-H23", "Lung", "selectable", 1, 0, -1.2, 0.0, 5.0),
    ("CVCL_3", "HCT 116", "Bowel", "selectable", 1, 0, 0.1, 0.2, 4.0),
    ("CVCL_4", "HT-29", "Bowel", "selectable", 0, 0, 0.0, 1.1, 3.0),
    ("CVCL_5", "MCF-7", "Breast", "selectable", 0, 1, -1.5, 0.0, 2.0),
    ("CVCL_6", "T-47D", "Breast", "selectable", 0, 0, 0.0, 0.0, 1.0),
    ("CVCL_7", "SR", "Lymphoid", "selectable", 0, 0, np.nan, 0.0, 0.5),
]


@pytest.fixture
def store(tmp_path: Path) -> FeatureStore:
    cells = pd.DataFrame({
        "cellosaurus_ac": [r[0] for r in LINES], "cellosaurus_name": [r[1] for r in LINES],
        "depmap_id": [f"ACH-00000{i}" for i in range(len(LINES))],
        "lineage": [r[2] for r in LINES], "primary_disease": "disease", "subtype": "subtype",
        "split": [r[3] for r in LINES], "problematic": None,
        "expression_imputed": [int(r[1] == "SR") for r in LINES], "mutation_from_parent": 0,
        "mut_KRAS": [r[4] for r in LINES], "mut_PTEN": [r[5] for r in LINES],
        "cn_PTEN": [0.0 if np.isnan(r[6]) else r[6] for r in LINES], "cn_MYC": [r[7] for r in LINES],
        "cn_missing_PTEN": [int(np.isnan(r[6])) for r in LINES], "cn_source": "cellminer",
        "expr_EGFR": [r[8] for r in LINES],
    })
    drugs = pd.DataFrame({
        "drug_id": ["CHEMBL1", "CHEMBL2", "CHEMBL3"],
        "preferred_name": ["TRAMETINIB", "GEFITINIB", "CISPLATIN"],
        "source_ids": ["Trametinib|GSK1120212", "Gefitinib", "Cisplatin"],
        "max_phase": [4.0, 4.0, np.nan],
    })
    targets = pd.DataFrame({
        "drug_id": ["CHEMBL1", "CHEMBL2", "CHEMBL3"],
        "target_pref_name": ["MEK1/2", "EGFR", "DNA"],
        "target_type": ["PROTEIN FAMILY", "SINGLE PROTEIN", "NUCLEIC-ACID"],
        "gene_symbols": ["MAP2K1|MAP2K2", "EGFR", ""],
        "action_type": ["INHIBITOR", "INHIBITOR", "CROSS-LINKING AGENT"],
        "mechanism_of_action": ["MEK inhibitor", "EGFR inhibitor", "DNA cross-linking agent"],
    })
    expression = pd.DataFrame({"cellosaurus_ac": [r[0] for r in LINES],
                               "EGFR": [r[8] for r in LINES], "MAP2K1": np.arange(7.0)})
    copy_number = pd.DataFrame({"cellosaurus_ac": [r[0] for r in LINES], "EGFR": 0.0})
    for key, df in {"cells": cells, "drugs": drugs, "drug_targets": targets,
                    "expression": expression, "copy_number": copy_number}.items():
        p = tmp_path / ALLOWED_TABLES[key]
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(p)
    # An outcome file sits beside the features, as in the real cache. Nothing may read it.
    pd.DataFrame({"y_syn": [12.345]}).to_parquet(tmp_path / "processed" / "labels.parquet")
    return FeatureStore(tmp_path)


# -- store ----------------------------------------------------------------------------------------


def test_store_refuses_tables_off_the_allowlist(store):
    for name in ("labels", "self_pairs", "pool", "../processed/labels"):
        with pytest.raises(PermissionError):
            store.path(name)
    assert not {Path(p).name for p in ALLOWED_TABLES.values()} & DENIED_FILES


def test_tools_package_names_no_outcome_file_outside_the_denylist():
    src = Path(omics.__file__).parent
    for path in src.glob("*.py"):
        text = path.read_text()
        if path.name != "store.py":
            assert "read_parquet" not in text and "pq." not in text, f"{path.name} reads from disk"
            for denied in DENIED_FILES:
                assert denied not in text, f"{path.name} names {denied}"
        assert not re.search(r"y_syn|y_inh|OUTCOME_COLS|freeze_pool", text), f"{path.name} names outcomes"


def test_resolve_drug_and_line_by_any_name(store):
    assert store.resolve_drug("trametinib") == store.resolve_drug("GSK-1120212") == "CHEMBL1"
    assert store.resolve_line("a549") == store.resolve_line("CVCL_1") == "CVCL_1"
    with pytest.raises(ToolError, match="did you mean TRAMETINIB"):
        store.resolve_drug("trametinb")


def test_copy_number_is_missing_not_zero_where_unmeasured(store):
    assert np.isnan(store.copy_number("PTEN")["CVCL_7"])


# -- rules ----------------------------------------------------------------------------------------


def test_rule_parses_and_evaluates(store):
    rule = rules.parse("KRAS_mut == 1 and PTEN_loss == 0")
    assert str(rule) == "KRAS_mut == 1 and PTEN_loss == 0"
    result = rule.evaluate(store)
    assert list(result.index[result.fillna(False).astype(bool)]) == ["CVCL_1", "CVCL_3"]
    assert pd.isna(rules.parse("PTEN_loss == 0").evaluate(store)["CVCL_7"])


def test_rule_lineage_gain_and_expression_tertile(store):
    assert rules.parse('lineage == "lung"').evaluate(store).sum() == 2
    assert rules.parse("MYC_gain == 1").evaluate(store).sum() == 2
    high = rules.parse("EGFR_expr == high").evaluate(store)
    assert set(high.index[high.astype(bool)]) == {"CVCL_1", "CVCL_2", "CVCL_3"}  # top third of 7, by rank


@pytest.mark.parametrize("text", [
    "KRAS_mut == 1 and PTEN_loss == 0 and lineage == Lung",  # three terms
    "KRAS_mut == 1 or PTEN_mut == 1",
    "KRAS_mut > 0",
    "FOO_mut == 1",  # not a driver gene
    "EGFR_expr == 1",
    "KRAS_mut == yes",
    "__import__('os')",
    "",
])
def test_rule_rejects_anything_outside_the_vocabulary(text):
    with pytest.raises(ToolError):
        rules.parse(text)


# -- tools ----------------------------------------------------------------------------------------


def test_get_drug_and_get_cell(store):
    drug = omics.get_drug(store, "trametinib")
    assert "MEK inhibitor" in drug["summary"] and drug["evidence"][0]["mechanisms"][0]["genes"] == ["MAP2K1", "MAP2K2"]
    assert "no gene" in omics.get_drug(store, "cisplatin")["summary"]
    cell = omics.get_cell(store, "A-549")["evidence"][0]
    assert cell["mutated_drivers"] == ["KRAS"] and cell["cn_gain"] == ["MYC"] and cell["split"] == "heldout"
    assert "imputed" in omics.get_cell(store, "SR")["summary"]


def test_find_lines_groups_and_warns_when_untestable(store):
    out = omics.find_lines(store, "KRAS_mut == 1")
    assert out["evidence"][0]["testable"] and "3 true, 4 false" in out["summary"]
    out = omics.find_lines(store, "KRAS_mut == 1", lineage="Lung")
    assert not out["evidence"][0]["testable"] and "cannot be tested" in out["summary"]
    assert "1 true" in omics.find_lines(store, "KRAS_mut == 1", lineage="Lung", split="selectable")["summary"]
    with pytest.raises(ToolError, match="lineages: Bowel"):
        omics.find_lines(store, "KRAS_mut == 1", lineage="Kidney")


def test_gene_expression_three_modes(store):
    by_rule = omics.gene_expression(store, "EGFR", rule="KRAS_mut == 1")["evidence"][0]
    assert by_rule["median_true"] == 5.0 and by_rule["median_false"] == 1.5 and by_rule["mann_whitney_p"] < 0.1
    by_line = omics.gene_expression(store, "EGFR", lines=["A549", "SR"])["evidence"]
    assert [r["tertile"] for r in by_line] == ["high", "low"]
    assert "Highest: A-549" in omics.gene_expression(store, "EGFR")["summary"]
    assert "no comparison" in omics.gene_expression(store, "EGFR", rule="PTEN_mut == 1")["summary"]
    with pytest.raises(ToolError, match="no expression data"):
        omics.gene_expression(store, "NOTAGENE")


def test_drug_targets_in_line_and_shared_targets(store):
    out = omics.drug_targets_in_line(store, "gefitinib", "A-549")["evidence"][0]["targets"]
    assert out == [{"gene": "EGFR", "log2_fpkm1": 6.0, "expr_tertile": "high", "cn_log2": 0.0}]
    assert "no gene-level target" in omics.drug_targets_in_line(store, "cisplatin", "A-549")["summary"]
    shared = omics.shared_targets(store, "trametinib", "gefitinib")["evidence"][0]
    assert shared["shared_genes"] == [] and shared["only_b"] == ["EGFR"]
    with pytest.raises(ToolError, match="same drug"):
        omics.shared_targets(store, "trametinib", "GSK1120212")


# -- registry and probes --------------------------------------------------------------------------


def test_only_tools_with_a_backend_and_an_implementation_are_offered(store):
    names = [t.name for t in registry.offered({"features": store})]
    assert names == ["get_drug", "get_cell", "find_lines", "gene_expression", "drug_targets_in_line", "shared_targets"]
    assert registry.offered({}) == []
    assert registry.offered({"model": object(), "oracle": object(), "literature": object()}) == []
    assert {t.name for t in registry.pending()} == {"query_model", "resolve", "get_history", "search_literature"}


def test_probes_run_and_report_bad_input_without_raising(store):
    probes = {p.name: p for p in tool_probes.as_probes({"features": store})}
    ok = asyncio.run(probes["find_lines"].run(rule="KRAS_mut == 1", lineage=None))
    assert ok.ok and "3 true" in ok.render()
    bad = asyncio.run(probes["get_drug"].run(drug="nonsense"))
    assert not bad.ok and "unknown drug" in bad.summary
    for probe in probes.values():
        definition = probe.tool_definition()
        assert definition["input_schema"]["additionalProperties"] is False
        json.dumps(definition)


def test_python_context_is_json_and_carries_no_outcomes(store):
    ctx = tool_probes.python_context(store)
    text = json.dumps(ctx)
    assert "12.345" not in text and "y_syn" not in text
    assert ctx["line_ids"][0] == "CVCL_1" and ctx["expr"]["EGFR"][0] == 6.0
    assert ctx["cells"][6]["cn"]["PTEN"] is None and ctx["drugs"][0]["targets"] == ["MAP2K1", "MAP2K2"]


def test_sandbox_runs_preloaded_data_and_refuses_data_paths(store):
    probe = tool_probes.GuardedPythonProbe(context=tool_probes.python_context(store))
    out = asyncio.run(probe.run(code="print(sum(c['mut']['KRAS'] for c in cells))"))
    assert out.ok and out.summary == "3"
    for code in ("open('data_cache/processed/labels.parquet')", "import pandas; pandas.read_parquet(p)"):
        refused = asyncio.run(probe.run(code=code))
        assert not refused.ok and "refused" in refused.summary


def test_no_tool_output_contains_a_hidden_label(store):
    """Design section 10: scan tool outputs for values matching hidden labels to 3 decimals."""
    outputs = _all_outputs(store)
    assert outputs and not _label_matches(outputs, {"12.345"})


def _all_outputs(store: FeatureStore) -> str:
    out = []
    drugs, lines = list(store.drugs.index), list(store.cells.index)
    for d in drugs:
        out.append(omics.get_drug(store, d))
        out.append(omics.drug_targets_in_line(store, d, lines[0]))
    for line in lines:
        out.append(omics.get_cell(store, line))
    for gene in store.mutation_genes:
        out.append(omics.find_lines(store, f"{gene}_mut == 1"))
        try:
            out.append(omics.gene_expression(store, gene, rule=f"{gene}_mut == 1"))
        except ToolError:
            pass
    out.append(omics.shared_targets(store, drugs[0], drugs[1]))
    return json.dumps(out, default=str)


def _label_matches(text: str, labels: set[str]) -> set[str]:
    return {m for m in re.findall(r"-?\d+\.\d{3}", text) if m in labels}


# -- real data ------------------------------------------------------------------------------------

integration = pytest.mark.skipif(os.environ.get("COMBO_INTEGRATION") != "1",
                                 reason="needs the M1 data bundle; set COMBO_INTEGRATION=1")


@integration
def test_real_tools_run_on_the_frozen_tables():
    store = FeatureStore()
    assert store.available() == []
    assert len(store.cells) == 57
    assert "KRAS" in omics.get_cell(store, "A-549")["summary"]
    assert omics.find_lines(store, "KRAS_mut == 1")["evidence"][0]["testable"]
    assert omics.gene_expression(store, "EGFR", rule="lineage == Lung")["evidence"][0]["n_true"] >= 3
    size = len(json.dumps(tool_probes.python_context(store)))
    assert size < 2_000_000, f"sandbox preload is {size} bytes"


@integration
def test_real_tool_outputs_match_no_hidden_label():
    """Tool outputs are rounded to 2 decimals, so any 3-decimal match with a label is a leak."""
    from combo_agent.data import paths  # labels are read here, in the test, never in tools

    labels = pd.read_parquet(paths.processed_dir() / "labels.parquet", columns=["y_syn", "y_inh"])
    hidden = {f"{v:.3f}" for v in pd.concat([labels["y_syn"], labels["y_inh"]]).dropna()}
    assert not _label_matches(_all_outputs(FeatureStore()), hidden)
