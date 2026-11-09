"""Tests for combo_agent.data.harmonize_lines and combo_agent.data.cellosaurus (step 1d)."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pandas as pd
import pytest

from combo_agent.data import cellosaurus as cs
from combo_agent.data import harmonize_lines as hl
from combo_agent.data.api_cache import CacheMiss, CachedJsonClient

# -- a fake Cellosaurus ---------------------------------------------------------------------------


def entry(ac, ident, synonyms=(), *, panel=True, depmap=("ACH-0",), category="Cancer cell line",
          human=True, parents=(), problem="", misspellings=()):
    comments = [{"category": "Part of", "value": cs.NCI60_PANEL}] if panel else []
    if problem:
        comments.append({"category": "Problematic cell line", "value": problem})
    e = {
        "accession-list": [{"type": "primary", "value": ac}],
        "name-list": [{"type": "identifier", "value": ident}]
        + [{"type": "synonym", "value": s} for s in synonyms],
        "category": category,
        "species-list": [{"accession": "9606" if human else "10090"}],
        "comment-list": comments,
        "xref-list": [{"database": "DepMap", "accession": d} for d in depmap]
        + [{"database": "Cosmic", "accession": "1"}],
        "derived-from": [{"accession": p, "label": p} for p in parents],
        "disease-list": [{"database": "NCIt", "accession": "C1", "label": "Carcinoma"}],
        "sex": "Female",
    }
    if misspellings:
        e["misspelling-list"] = [{"misspelling-name": m} for m in misspellings]
    return e


class FakeApi:
    def __init__(self, entries, search_hits=None):
        self.entries = {cs.accession(e): e for e in entries}
        self.search_hits = search_hits or {}
        self.searches = []

    def entry(self, ac):
        return self.entries.get(ac)

    def nci60_panel(self):
        return [e for e in self.entries.values() if cs.in_nci60_panel(e)]

    def search(self, query):
        self.searches.append(query)
        return self.search_hits.get(query, [])


def table(*labels):
    rows = [{"Cell_Line_ID": lab} for lab in labels for _ in range(3)]
    return pd.DataFrame(rows)


def curated(*rows):
    out = []
    for source_id, ac, *rest in rows:
        depmap = rest[0] if rest else ""
        out.append({"source_id": source_id, "cellosaurus_ac": ac, "cellosaurus_name": "",
                    "resolution": "nci60_panel", "matched_as": "identifier", "depmap_id": depmap,
                    "depmap_proxy_of": "", "note": ""})
    return out


# -- names ----------------------------------------------------------------------------------------


def test_normalize_drops_case_and_punctuation():
    assert cs.normalize_name("NCI/ADR-RES") == cs.normalize_name("nci adr res") == "NCIADRRES"
    assert cs.normalize_name("HL-60(TB)") == "HL60TB"


def test_match_kind_prefers_identifier_and_sees_misspellings():
    e = entry("CVCL_1", "HCT 15", ["HCT-15"], misspellings=["HTC15"])
    assert hl.match_kind("HCT-15", e) == "identifier"  # "HCT 15" normalizes to the same key
    assert hl.match_kind("HTC-15", e) == "misspelling"
    assert hl.match_kind("HCT-116", e) is None


# -- propose --------------------------------------------------------------------------------------


def test_propose_matches_exactly_one_panel_entry():
    api = FakeApi([entry("CVCL_0023", "A-549", ["A549"]), entry("CVCL_9", "A549-luc", ["A549 Luc"])])
    p = hl.propose_one("A549", api.nci60_panel(), api)
    # "A549" and the identifier "A-549" normalize to the same key, so this is an identifier match
    assert (p.resolution, p.cellosaurus_ac, p.matched_as) == ("nci60_panel", "CVCL_0023", "identifier")
    assert hl.propose_one("A549LUC", api.nci60_panel(), api).cellosaurus_ac == "CVCL_9"
    assert api.searches == []  # the panel answered; no free-text search needed


def test_propose_flags_two_panel_matches_as_ambiguous():
    api = FakeApi([entry("CVCL_1", "XY-1"), entry("CVCL_2", "XY1")])
    p = hl.propose_one("XY-1", api.nci60_panel(), api)
    assert p.resolution == "ambiguous" and "CVCL_1" in p.note and "CVCL_2" in p.note


def test_propose_falls_back_to_name_search_and_skips_non_human():
    q = 'id:"ZZ-9" OR sy:"ZZ-9"'
    human = entry("CVCL_H", "ZZ-9", panel=False)
    mouse = entry("CVCL_M", "ZZ9", panel=False, human=False)
    api = FakeApi([], search_hits={q: [human, mouse]})
    p = hl.propose_one("ZZ-9", api.nci60_panel(), api)
    assert (p.resolution, p.cellosaurus_ac) == ("name_search", "CVCL_H")
    assert "review" in p.note


def test_propose_reports_unresolved():
    api = FakeApi([])
    assert hl.propose_one("NOPE", [], api).resolution == "unresolved"


def test_csv_roundtrip_and_manual_rows_survive(tmp_path):
    auto = curated(("A549", "CVCL_0023"), ("SR", "CVCL_1711"))
    old = curated(("A549", "CVCL_0023"), ("SR", "CVCL_1711", "ACH-000338"))
    old[0] = {**old[0], "resolution": "manual", "note": "hand-checked"}
    merged = hl.merge_with_manual(auto, old)
    assert merged[0]["resolution"] == "manual"
    assert merged[1]["depmap_id"] == "ACH-000338"  # hand-chosen DepMap ID kept on an auto row
    path = tmp_path / "lines.csv"
    path.write_text(hl.render_csv(merged))
    assert hl.read_curated(path) == sorted(merged, key=lambda r: r["source_id"])


def test_read_curated_rejects_wrong_columns(tmp_path):
    path = tmp_path / "lines.csv"
    path.write_text("source_id,cellosaurus_ac\nA549,CVCL_0023\n")
    with pytest.raises(hl.HarmonizationError, match="columns"):
        hl.read_curated(path)


# -- families -------------------------------------------------------------------------------------


def test_families_join_derivatives_and_shared_parents_with_stable_ids():
    fam = hl.families({
        "CVCL_0021": set(),                      # U-251MG
        "CVCL_0535": {"CVCL_0021"},              # SNB-19, derived from U-251MG
        "CVCL_0292": {"CVCL_3449"},              # HCT 15 -> placeholder parent
        "CVCL_9999": {"CVCL_3449"},              # another line from the same individual
        "CVCL_0023": set(),                      # unrelated
    })
    assert fam["CVCL_0535"] == fam["CVCL_0021"] == "FAM_CVCL_0021"
    assert fam["CVCL_0292"] == fam["CVCL_9999"] == "FAM_CVCL_0292"
    assert fam["CVCL_0023"] == "FAM_CVCL_0023"


def test_ancestors_follow_chains_and_stop_on_cycles():
    a = entry("CVCL_A", "A", parents=["CVCL_B"])
    b = entry("CVCL_B", "B", parents=["CVCL_C"])
    c = entry("CVCL_C", "C", parents=["CVCL_A"])  # a cycle should not loop forever
    assert hl.ancestors(FakeApi([a, b, c]), "CVCL_A", {}) == {"CVCL_B", "CVCL_C"}


# -- build ----------------------------------------------------------------------------------------


def _world():
    return FakeApi([
        entry("CVCL_1629", "OVCAR-8", depmap=["ACH-000696"]),
        entry("CVCL_1452", "NCI-ADR-RES", ["NCI/ADR-RES"], depmap=[], parents=["CVCL_1629"],
              problem="Contaminated. Shown to be a OVCAR-8 derivative."),
        entry("CVCL_1711", "SR", depmap=["ACH-000338", "ACH-001665"]),
        entry("CVCL_0023", "A-549", ["A549"], depmap=["ACH-000681"]),
        entry("CVCL_N", "NOTCANCER", depmap=["ACH-9"], category="Finite cell line"),
    ])


def test_build_maps_drops_and_groups_families():
    df = table("OVCAR-8", "NCI/ADR-RES", "SR", "A549", "NOTCANCER")
    rows = curated(("OVCAR-8", "CVCL_1629"), ("NCI/ADR-RES", "CVCL_1452"), ("SR", "CVCL_1711"),
                   ("A549", "CVCL_0023"), ("NOTCANCER", "CVCL_N"))
    r = hl.build(df, rows, _world())
    cells = r.cell_lines.set_index("cellosaurus_name")
    assert cells.loc["A-549", "keep"] and cells.loc["A-549", "depmap_id"] == "ACH-000681"
    assert cells.loc["NCI-ADR-RES", "drop_reason"] == "no_depmap_id"
    assert cells.loc["SR", "drop_reason"] == "multiple_depmap_ids"
    assert cells.loc["NOTCANCER", "drop_reason"] == "not_cancer_cell_line"
    assert cells.loc["NCI-ADR-RES", "family_id"] == cells.loc["OVCAR-8", "family_id"]
    assert r.report["needs_depmap_choice"] == {"SR": "ACH-000338|ACH-001665"}
    assert r.report["families_with_more_than_one_line"] == {"FAM_CVCL_1452": ["NCI-ADR-RES", "OVCAR-8"]}
    assert r.report["rows_kept"] == 6 and r.report["rows_dropped"] == 9
    assert list(r.line_map.source_id) == sorted(df.Cell_Line_ID.unique())


def test_build_uses_a_valid_hand_chosen_depmap_id():
    rows = curated(("SR", "CVCL_1711", "ACH-001665"))
    r = hl.build(table("SR"), rows, _world())
    assert r.cell_lines.loc[0, "depmap_id"] == "ACH-001665" and bool(r.cell_lines.loc[0, "keep"])


def test_build_rejects_a_depmap_id_cellosaurus_does_not_list():
    with pytest.raises(hl.HarmonizationError, match="not among"):
        hl.build(table("SR"), curated(("SR", "CVCL_1711", "ACH-123")), _world())


def _proxy_world():
    return FakeApi([
        entry("CVCL_0002", "HL-60", panel=False, depmap=["ACH-000002"]),
        entry("CVCL_A794", "HL-60(TB)", depmap=[], parents=["CVCL_0002"]),
        entry("CVCL_0023", "A-549", ["A549"], depmap=["ACH-000681"]),
    ])


def _proxy_rows(proxy):
    rows = curated(("HL-60(TB)", "CVCL_A794"))
    rows[0]["depmap_proxy_of"] = proxy
    return rows


def test_build_uses_an_ancestor_as_depmap_proxy():
    r = hl.build(table("HL-60(TB)"), _proxy_rows("CVCL_0002"), _proxy_world())
    row = r.cell_lines.iloc[0]
    assert (row.depmap_id, bool(row.depmap_is_proxy), bool(row.keep)) == ("ACH-000002", True, True)
    assert r.report["depmap_proxies"] == {"HL-60(TB)": "ACH-000002 (from CVCL_0002)"}
    assert bool(r.line_map.loc[0, "depmap_is_proxy"])


def test_build_rejects_a_proxy_that_is_not_an_ancestor():
    with pytest.raises(hl.HarmonizationError, match="not an ancestor"):
        hl.build(table("HL-60(TB)"), _proxy_rows("CVCL_0023"), _proxy_world())


def test_build_rejects_depmap_id_and_proxy_on_one_row():
    rows = _proxy_rows("CVCL_0002")
    rows[0]["depmap_id"] = "ACH-000002"
    with pytest.raises(hl.HarmonizationError, match="both set"):
        hl.build(table("HL-60(TB)"), rows, _proxy_world())


def test_build_rejects_missing_rows_and_name_mismatches():
    with pytest.raises(hl.HarmonizationError, match="every label has a curated row"):
        hl.build(table("A549", "SR"), curated(("A549", "CVCL_0023")), _world())
    with pytest.raises(hl.HarmonizationError, match="no name"):
        hl.build(table("A549"), curated(("A549", "CVCL_1711")), _world())  # A549 is not "SR"


def test_build_accepts_a_manual_row_whose_names_differ():
    rows = curated(("A549-X", "CVCL_0023"))
    rows[0]["resolution"] = "manual"
    assert hl.build(table("A549-X"), rows, _world()).line_map.loc[0, "match_kind"] == "manual"


def test_build_rejects_secondary_accession():
    api = _world()
    api.entries["CVCL_OLD"] = api.entries["CVCL_0023"]  # the API answers with the new primary
    with pytest.raises(hl.HarmonizationError, match="secondary"):
        hl.build(table("A549"), curated(("A549", "CVCL_OLD")), api)


# -- cached client --------------------------------------------------------------------------------


def test_api_urls_are_cached_and_replay_offline(tmp_path):
    body = {"Cellosaurus": {"cell-line-list": [entry("CVCL_0023", "A-549")]}}
    seen = []

    def fetch(url):
        seen.append(url)
        return 200, body

    online = cs.CellosaurusApi(CachedJsonClient(tmp_path, offline=False, fetch=fetch))
    assert cs.identifier(online.entry("CVCL_0023")) == "A-549"
    assert "fields=" in seen[0] and "format=json" in seen[0]
    offline = cs.CellosaurusApi(CachedJsonClient(tmp_path, offline=True, fetch=fetch))
    assert cs.identifier(offline.entry("CVCL_0023")) == "A-549" and len(seen) == 1
    with pytest.raises(CacheMiss):
        offline.entry("CVCL_9999")


# -- real data ------------------------------------------------------------------------------------


@pytest.mark.skipif(os.environ.get("COMBO_INTEGRATION") != "1",
                    reason="needs the converted parquet and a filled Cellosaurus cache; "
                           "set COMBO_INTEGRATION=1")
def test_real_build_offline_is_reproducible(tmp_path):
    from combo_agent.data import ingest_tdc

    df = ingest_tdc.load("tdc_drugcomb")
    rows = hl.read_curated()
    digests = []
    for run in ("a", "b"):
        result = hl.build(df, rows, hl.make_api(offline=True))
        out = hl.write_outputs(result, tmp_path / run)
        digests.append({k: hashlib.sha256(Path(p).read_bytes()).hexdigest() for k, p in out.items()})
    assert digests[0] == digests[1]
    rep = result.report
    assert rep["source_labels"] == rep["cellosaurus_lines"] == 59
    assert rep["resolutions"] == {"nci60_panel": 59}
    fams = rep["families_with_more_than_one_line"]
    assert ["NCI-ADR-RES", "OVCAR-8"] in fams.values()
    assert ["M14", "MDA-MB-435"] in fams.values()
    assert ["SNB-19", "U-251MG"] in fams.values()
    # decisions 9-11 (2026-11-08)
    assert rep["depmap_chosen"] == {"SR": "ACH-000338", "UO-31": "ACH-000428"}
    assert rep["depmap_proxies"] == {"HL-60(TB)": "ACH-000002 (from CVCL_0002)"}
    assert rep["dropped"] == {"MDA-MB-435": "no_depmap_id", "NCI-ADR-RES": "no_depmap_id"}
    assert rep["kept_lines"] == 57
