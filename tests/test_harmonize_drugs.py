"""Tests for step 1c (drug harmonization): structure keys, the resolver, the checks, and the build.

Unit tests use a small in-memory stand-in for ChEMBL/PubChem. The integration tests at the bottom
(`COMBO_INTEGRATION=1`) replay the real build from the cached API responses, offline, and check the
outcome against the drug identity audit (7 synonym groups, 129 labels -> 122 molecules).
"""

from __future__ import annotations

import http.client
import json
import os
from pathlib import Path

import pandas as pd
import pytest

from combo_agent.data import harmonize_drugs as hd
from combo_agent.data import structure
from combo_agent.data.api_cache import ApiError, CachedJsonClient, CacheMiss

ERLOTINIB = "COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC"
ERLOTINIB_HCL = ERLOTINIB + ".Cl"
ERLOTINIB_KEY = "AAKJLRGGTJKAMG"
CISPLATIN_SRC = "N.N.Cl[Pt+2]Cl"
CARBOPLATIN_SRC = "C1CC(C1)(C(=O)O)C(=O)O.[NH2-].[NH2-].[Pt+2]"
GEFITINIB = "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"
ALANINE_FLAT, L_ALANINE, D_ALANINE = "CC(N)C(=O)O", "C[C@H](N)C(=O)O", "C[C@@H](N)C(=O)O"


# -- structure -------------------------------------------------------------------------------------


def test_connectivity_key_ignores_salt_and_matches_chembl_block():
    assert structure.connectivity_key(ERLOTINIB) == structure.connectivity_key(ERLOTINIB_HCL) == ERLOTINIB_KEY


def test_connectivity_key_ignores_stereo():
    assert structure.connectivity_key(L_ALANINE) == structure.connectivity_key(ALANINE_FLAT)
    assert not structure.has_stereo(ALANINE_FLAT) and structure.has_stereo(L_ALANINE)


def test_largest_fragment_would_destroy_cisplatin_so_metals_are_refused():
    # Why P3 exists: the plain recipe keeps ammonia.
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize as S
    naive = Chem.MolToSmiles(S.LargestFragmentChooser().choose(Chem.MolFromSmiles(CISPLATIN_SRC)))
    assert naive == "N"
    assert structure.has_metal(CISPLATIN_SRC)
    with pytest.raises(structure.StructureError, match="metal complex"):
        structure.parent_mol(CISPLATIN_SRC)
    assert hd.structure_signature(CISPLATIN_SRC) == ("formula", "Cl2N2Pt")


def test_heavy_atom_formula_matches_chembl_molformula():
    assert structure.heavy_atom_formula(CARBOPLATIN_SRC) == "C6N2O4Pt"
    assert structure.heavy_atom_formula_from_molformula("C6H12N2O4Pt") == "C6N2O4Pt"
    assert structure.heavy_atom_formula_from_molformula("Cl2H6N2Pt") == "Cl2N2Pt"


def test_sodium_salt_is_not_a_metal_complex():
    assert not structure.has_metal("CC(=O)[O-].[Na+]")


def test_unparseable_smiles_raises():
    with pytest.raises(structure.StructureError):
        structure.mol_from_smiles("not a smiles((")


# -- name variants ---------------------------------------------------------------------------------


@pytest.mark.parametrize("label, expected", [
    ("ERLOTINIB HYDROCHLORIDE", ["ERLOTINIB HYDROCHLORIDE", "ERLOTINIB"]),
    ("ABT-263 (NAVITOCLAX)", ["ABT-263 (NAVITOCLAX)", "ABT-263", "NAVITOCLAX"]),
    ("ELOXATIN (TN) (SANOFI SYNTHELAB)", ["ELOXATIN (TN) (SANOFI SYNTHELAB)", "ELOXATIN"]),
    ("NSC 85998", ["NSC 85998", "NSC-85998", "NSC85998"]),
    ("NSC141540", ["NSC141540", "NSC-141540", "NSC 141540"]),
    ("FLUDARABINE BASE", ["FLUDARABINE BASE", "FLUDARABINE"]),
    ("NAVELBINE DITARTRATE (TN)", ["NAVELBINE DITARTRATE (TN)", "NAVELBINE DITARTRATE", "NAVELBINE"]),
    ("5-FU", ["5-FU"]),
])
def test_name_queries(label, expected):
    assert hd.name_queries(label) == expected


# -- fake ChEMBL -----------------------------------------------------------------------------------


def mol(cid, smiles=None, parent=None, name=None, phase=None, formula=None):
    from rdkit import Chem
    structs = None
    if smiles:
        structs = {"canonical_smiles": smiles,
                   "standard_inchi_key": Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))}
    return {"molecule_chembl_id": cid, "pref_name": name, "max_phase": phase,
            "molecule_hierarchy": {"parent_chembl_id": parent or cid, "molecule_chembl_id": cid},
            "molecule_structures": structs, "molecule_properties": {"full_molformula": formula}}


class FakeApi:
    """The subset of ChemblApi the resolver and build use, backed by dicts."""

    def __init__(self, molecules, names=None, pubchem=None, mechanisms=None, targets=None):
        self.molecules = {m["molecule_chembl_id"]: m for m in molecules}
        self.names = {k.upper(): v for k, v in (names or {}).items()}
        self.pub = pubchem or {}
        self.mechs = mechanisms or {}
        self.targets = targets or {}

    def status(self):
        return {"chembl_db_version": "ChEMBL_TEST", "chembl_release_date": "2026-01-01"}

    def molecule(self, cid):
        return self.molecules.get(cid)

    def parent_id(self, cid):
        m = self.molecules.get(cid)
        return None if m is None else m["molecule_hierarchy"]["parent_chembl_id"]

    def molecules_by_name(self, name):
        return sorted(self.names.get(name.upper(), []))

    def molecules_by_inchikey(self, key):
        return sorted(c for c, m in self.molecules.items()
                      if (m["molecule_structures"] or {}).get("standard_inchi_key") == key)

    def molecules_by_connectivity(self, conn):
        return sorted(c for c, m in self.molecules.items()
                      if ((m["molecule_structures"] or {}).get("standard_inchi_key") or "").startswith(conn))

    def pubchem_by_name(self, name):
        return self.pub.get(name, [])

    def mechanisms(self, parent):
        return self.mechs.get(parent, [])

    def target(self, tid):
        return self.targets.get(tid)


def erlotinib_world():
    return FakeApi(
        [mol("C553", ERLOTINIB, name="ERLOTINIB", phase=4),
         mol("C1079742", ERLOTINIB_HCL, parent="C553", name="ERLOTINIB HYDROCHLORIDE", phase=4),
         mol("C939", GEFITINIB, name="GEFITINIB", phase=4),
         mol("C11359", None, name="CISPLATIN", phase=4, formula="Cl2H6N2Pt"),
         mol("CL", L_ALANINE, name="L-ALANINE"), mol("CD", D_ALANINE, name="D-ALANINE")],
        names={"ERLOTINIB": ["C553"], "ERLOTINIB HYDROCHLORIDE": ["C1079742"], "CISPLATINO": ["C11359"],
               "WRONGNAME": ["C939"]},
        mechanisms={"C553": [{"mec_id": 1, "target_chembl_id": "T203", "action_type": "INHIBITOR",
                              "mechanism_of_action": "EGFR inhibitor", "direct_interaction": True}]},
        targets={"T203": {"pref_name": "Epidermal growth factor receptor erbB1", "target_type": "SINGLE PROTEIN",
                          "organism": "Homo sapiens", "target_components": [{"target_component_synonyms": [
                              {"syn_type": "GENE_SYMBOL", "component_synonym": "EGFR"},
                              {"syn_type": "GENE_SYMBOL_OTHER", "component_synonym": "ERBB1"}]}]}},
    )


# -- resolver --------------------------------------------------------------------------------------


def test_salt_label_resolves_to_parent_by_name_and_structure():
    p = hd.propose_one("ERLOTINIB HYDROCHLORIDE", ERLOTINIB_HCL, erlotinib_world())
    assert (p.chembl_parent_id, p.resolution, p.structure_check) == ("C553", "name+structure", "connectivity")


def test_platinum_label_resolves_by_formula():
    p = hd.propose_one("CISPLATINO", CISPLATIN_SRC, erlotinib_world())
    assert (p.chembl_parent_id, p.resolution, p.structure_check) == ("C11359", "name+structure", "formula")


def test_name_match_with_wrong_structure_is_rejected_and_reported():
    # The name points at gefitinib, but the structure is erlotinib: the name must not win.
    p = hd.propose_one("WRONGNAME", ERLOTINIB, erlotinib_world())
    assert p.chembl_parent_id == "C553" and p.resolution == "structure_only"
    assert "failing the structure check: C939" in p.note


def test_structure_alone_with_several_stereoisomers_is_ambiguous():
    p = hd.propose_one("SOME ALANINE", ALANINE_FLAT, erlotinib_world())
    assert p.resolution == "ambiguous" and p.chembl_parent_id is None
    assert "CD" in p.note and "CL" in p.note


def test_unknown_drug_is_unresolved():
    p = hd.propose_one("MYSTERY", "c1ccc2ccccc2c1", erlotinib_world())
    assert (p.resolution, p.chembl_parent_id) == ("unresolved", None)


def test_pubchem_route_breaks_a_stereo_tie():
    api = erlotinib_world()
    from rdkit import Chem
    api.pub["L-ALA LABEL"] = [(5950, Chem.MolToInchiKey(Chem.MolFromSmiles(L_ALANINE)))]
    p = hd.propose_one("L-ALA LABEL", ALANINE_FLAT, api)
    assert (p.chembl_parent_id, p.resolution, p.pubchem_cid) == ("CL", "name+structure", 5950)


# -- checks ----------------------------------------------------------------------------------------


def _drugs(*pairs):
    return pd.DataFrame({"source_id": [a for a, _ in pairs], "smiles": [b for _, b in pairs],
                         "n_rows": 1, "n_lines": 1})


def _ok_rows(api, drugs):
    return hd.proposal_rows(drugs, api)


def _failed(checks):
    return {c.name for c in checks if not (c.passed or c.warning)}


def test_checks_pass_on_a_consistent_table():
    api = erlotinib_world()
    drugs = _drugs(("ERLOTINIB", ERLOTINIB), ("ERLOTINIB HYDROCHLORIDE", ERLOTINIB_HCL), ("CISPLATINO", CISPLATIN_SRC))
    rows = _ok_rows(api, drugs)
    assert _failed(hd.verify(drugs, rows, rows, api)) == set()
    assert {r["preferred_name"] for r in rows if r["chembl_parent_id"] == "C553"} == {"ERLOTINIB"}


def test_check_catches_missing_and_duplicated_labels():
    api = erlotinib_world()
    drugs = _drugs(("ERLOTINIB", ERLOTINIB), ("CISPLATINO", CISPLATIN_SRC))
    rows = _ok_rows(api, drugs)
    assert "coverage" in _failed(hd.verify(drugs, rows[:1], None, api))
    assert "coverage" in _failed(hd.verify(drugs, rows + rows[:1], None, api))


def test_check_catches_a_salt_id_used_as_parent():
    api = erlotinib_world()
    drugs = _drugs(("ERLOTINIB", ERLOTINIB))
    rows = _ok_rows(api, drugs)
    rows[0] |= {"chembl_parent_id": "C1079742", "resolution": "manual", "note": "x"}
    assert "chembl_id_is_parent" in _failed(hd.verify(drugs, rows, None, api))


def test_check_catches_a_manual_mapping_to_the_wrong_molecule():
    api = erlotinib_world()
    drugs = _drugs(("ERLOTINIB", ERLOTINIB))
    rows = _ok_rows(api, drugs)
    rows[0] |= {"chembl_parent_id": "C939", "resolution": "manual", "note": "typo", "preferred_name": "GEFITINIB"}
    assert "structure_agrees" in _failed(hd.verify(drugs, rows, None, api))


def test_check_catches_a_stereoisomer_split():
    # Two labels with the same flat skeleton mapped to two different parents: one skeleton, two IDs.
    api = erlotinib_world()
    drugs = _drugs(("ALA A", ALANINE_FLAT), ("ALA B", ALANINE_FLAT))
    rows = [hd._row("ALA A", "CL", "L-ALANINE", "manual", "connectivity", None, "hand"),
            hd._row("ALA B", "CD", "D-ALANINE", "manual", "connectivity", None, "hand")]
    assert hd.verify(drugs, rows, None, api) and "grouping_matches_structure" in _failed(hd.verify(drugs, rows, None, api))


def test_check_catches_manual_row_without_note_and_bad_vocabulary():
    api = erlotinib_world()
    drugs = _drugs(("ERLOTINIB", ERLOTINIB))
    rows = _ok_rows(api, drugs)
    assert "row_vocabulary" in _failed(hd.verify(drugs, [rows[0] | {"resolution": "manual", "note": ""}], None, api))
    assert "row_vocabulary" in _failed(hd.verify(drugs, [rows[0] | {"resolution": "guess"}], None, api))
    assert "row_vocabulary" in _failed(hd.verify(drugs, [rows[0] | {"resolution": "unresolved"}], None, api))


def test_check_catches_drift_from_the_resolver():
    api = erlotinib_world()
    drugs = _drugs(("ERLOTINIB", ERLOTINIB))
    rows = _ok_rows(api, drugs)
    stale = [rows[0] | {"note": "edited by hand without marking manual"}]
    assert _failed(hd.verify(drugs, stale, rows, api)) == {"curated_matches_resolver"}


def test_manual_rows_survive_a_new_proposal():
    auto = [hd._row("A", "C1", "X", "unresolved", None, None, "")]
    manual = [hd._row("A", "C2", "Y", "manual", "connectivity", None, "checked by hand")]
    assert hd.merge_with_manual(auto, manual) == manual


def test_csv_round_trip(tmp_path):
    rows = [hd._row("B, with comma", "C1", "X", "name+structure", "connectivity", 7, 'say "hi"'),
            hd._row("A", "", "A", "unresolved", None, None, "")]
    path = tmp_path / "s.csv"
    path.write_text(hd.render_csv(rows))
    assert hd.read_curated(path) == sorted(rows, key=lambda r: r["source_id"])
    assert hd.render_csv(hd.read_curated(path)) == path.read_text()


# -- build -----------------------------------------------------------------------------------------


def _tdc(rows):
    return pd.DataFrame(rows, columns=["Drug1_ID", "Drug2_ID", "Cell_Line_ID", "Drug1", "Drug2"])


def test_build_merges_salt_and_attaches_parent_targets(tmp_path):
    api = erlotinib_world()
    df = _tdc([("ERLOTINIB", "CISPLATINO", "L1", ERLOTINIB, CISPLATIN_SRC),
               ("ERLOTINIB HYDROCHLORIDE", "CISPLATINO", "L2", ERLOTINIB_HCL, CISPLATIN_SRC)])
    rows = hd.proposal_rows(hd.source_drugs(df), api)
    res = hd.build(df, rows, api, rows)
    assert list(res.drugs["drug_id"]) == ["C11359", "C553"]
    erl = res.drugs.set_index("drug_id").loc["C553"]
    assert erl["source_ids"] == "ERLOTINIB|ERLOTINIB HYDROCHLORIDE" and erl["smiles_origin"] == "chembl_parent"
    assert erl["target_genes"] == "EGFR" and erl["n_targets"] == 1
    pt = res.drugs.set_index("drug_id").loc["C11359"]
    assert pt["smiles_origin"] == "source" and pt["n_targets"] == 0  # warning, not failure
    assert res.report["merge_groups"] == {"C553": {"preferred_name": "ERLOTINIB",
                                                   "source_ids": ["ERLOTINIB", "ERLOTINIB HYDROCHLORIDE"]}}
    assert res.report["passed"]
    # Deterministic: two builds write identical bytes.
    a, b = hd.write_outputs(res, tmp_path / "a"), hd.write_outputs(hd.build(df, rows, api, rows), tmp_path / "b")
    assert all(a[k].read_bytes() == b[k].read_bytes() for k in a)


def test_build_refuses_a_failing_table():
    api = erlotinib_world()
    df = _tdc([("ERLOTINIB", "CISPLATINO", "L1", ERLOTINIB, CISPLATIN_SRC)])
    rows = hd.proposal_rows(hd.source_drugs(df), api)
    rows[0] |= {"chembl_parent_id": "C939", "resolution": "manual", "note": "typo", "preferred_name": "GEFITINIB"}
    with pytest.raises(hd.HarmonizationError, match="structure_agrees"):
        hd.build(df, rows, api, None)


def test_source_drugs_counts_rows_and_refuses_two_smiles_for_one_label():
    df = _tdc([("A", "B", "L1", ERLOTINIB, GEFITINIB), ("A", "A", "L2", ERLOTINIB, ERLOTINIB)])
    out = hd.source_drugs(df).set_index("source_id")
    assert out.loc["A", "n_rows"] == 2 and out.loc["A", "n_lines"] == 2 and out.loc["B", "n_rows"] == 1
    with pytest.raises(hd.HarmonizationError, match="several SMILES"):
        hd.source_drugs(_tdc([("A", "B", "L1", ERLOTINIB, GEFITINIB), ("A", "B", "L2", GEFITINIB, GEFITINIB)]))


# -- API cache -------------------------------------------------------------------------------------


def test_cache_replays_offline_and_caches_404(tmp_path):
    calls = []

    def fetch(url):
        calls.append(url)
        return (404, None) if "missing" in url else (200, {"u": url})

    online = CachedJsonClient(tmp_path, offline=False, fetch=fetch)
    assert online.get("https://x/a") == {"u": "https://x/a"} and online.get("https://x/missing") is None
    offline = CachedJsonClient(tmp_path, offline=True, fetch=fetch)
    assert offline.get("https://x/a") == {"u": "https://x/a"} and offline.get("https://x/missing") is None
    assert len(calls) == 2
    with pytest.raises(CacheMiss):
        offline.get("https://x/never-fetched")


def test_cache_retries_truncated_bodies_then_gives_up(tmp_path):
    n = []

    def flaky(url):
        n.append(1)
        if len(n) < 3:
            raise http.client.IncompleteRead(b"", 10)
        return 200, {"ok": True}

    c = CachedJsonClient(tmp_path, offline=False, fetch=flaky, sleep=lambda s: None)
    assert c.get("https://x/a") == {"ok": True} and len(n) == 3

    def dead(url):
        raise http.client.IncompleteRead(b"", 10)

    with pytest.raises(ApiError):
        CachedJsonClient(tmp_path, offline=False, fetch=dead, sleep=lambda s: None).get("https://x/b")
    assert not list(tmp_path.glob("*.part"))


# -- integration: the real table, replayed offline from the cache ----------------------------------

integration = pytest.mark.skipif(os.environ.get("COMBO_INTEGRATION") != "1",
                                 reason="set COMBO_INTEGRATION=1 (needs the parquet and the API cache)")

#: The 7 synonym groups found by the drug identity audit (2026-11-08), as sets of labels.
AUDIT_GROUPS = {
    frozenset({"ERLOTINIB", "ERLOTINIB HYDROCHLORIDE"}),
    frozenset({"TOPOTECAN", "TOPOTECAN HYDROCHLORIDE"}),
    frozenset({"VINBLASTINE", "VINBLASTINE SULFATE"}),
    frozenset({"VINORELBINE", "NAVELBINE DITARTRATE (TN)"}),
    frozenset({"ETOPOSIDE", "NSC141540"}),
    frozenset({"DOXORUBICIN", "ADM HYDROCHLORIDE"}),
    frozenset({"OXALIPLATIN", "ELOXATIN (TN) (SANOFI SYNTHELAB)"}),
}


@pytest.fixture(scope="module")
def real_build():
    from combo_agent.data import ingest_tdc
    api = hd.make_api(offline=True)
    df = ingest_tdc.load("tdc_drugcomb")
    rows = hd.read_curated()
    auto = hd.proposal_rows(hd.source_drugs(df), api)
    return df, rows, auto, hd.build(df, rows, api, auto, hd.read_proxies())


@integration
def test_real_build_passes_every_check(real_build):
    *_, res = real_build
    failed = [c for c in res.report["checks"] if not (c["passed"] or c["warning"])]
    assert res.report["passed"] and not failed, failed


@integration
def test_real_build_reproduces_the_audit_groups(real_build):
    *_, res = real_build
    groups = {frozenset(g["source_ids"]) for g in res.report["merge_groups"].values()}
    assert groups == AUDIT_GROUPS
    assert res.report["counts"]["source_labels"] == 129
    assert res.report["counts"]["drugs"] == 122


@integration
def test_real_build_platinum_drugs_are_checked_by_formula(real_build):
    *_, res = real_build
    dm = res.drug_map.set_index("source_id")
    for label in ("CISPLATINO", "CARBOPLATIN", "OXALIPLATIN", "ELOXATIN (TN) (SANOFI SYNTHELAB)"):
        assert dm.at[label, "structure_check"] == "formula", label
    names = set(dm.loc[["CISPLATINO", "CARBOPLATIN", "OXALIPLATIN"], "preferred_name"])
    assert names == {"CISPLATIN", "CARBOPLATIN", "OXALIPLATIN"}


@integration
@pytest.mark.parametrize("label, gene", [
    ("ERLOTINIB HYDROCHLORIDE", "EGFR"), ("LAPATINIB", "ERBB2"), ("METHOTREXATE", "DHFR"),
    ("BORTEZOMIB", "PSMB5"), ("IMATINIB", "ABL1"), ("VEMURAFENIB", "BRAF"), ("ZOLINZA", "HDAC1"),
    ("ABT-263 (NAVITOCLAX)", "BCL2"),
    # ChEMBL files rapalogs under their direct binding partner FKBP12, not mTOR (checked 2026-11-08).
    ("NSC733504", "FKBP1A"), ("MK-8669", "MTOR"),
])
def test_real_build_known_targets(real_build, label, gene):
    *_, res = real_build
    drug_id = res.drug_map.set_index("source_id").at[label, "drug_id"]
    genes = res.drugs.set_index("drug_id").at[drug_id, "target_genes"].split("|")
    assert gene in genes, (label, drug_id, genes)


@integration
def test_real_build_is_byte_reproducible(real_build, tmp_path):
    df, rows, auto, res = real_build
    again = hd.build(df, rows, hd.make_api(offline=True), auto, hd.read_proxies())
    a, b = hd.write_outputs(res, tmp_path / "a"), hd.write_outputs(again, tmp_path / "b")
    assert all(a[k].read_bytes() == b[k].read_bytes() for k in a)


@integration
def test_curated_csv_is_canonical():
    assert hd.render_csv(hd.read_curated()) == hd.CURATED_CSV.read_text()


def test_stereo_guard_flags_an_unnamed_stereoisomer_when_an_approved_sibling_exists():
    api = erlotinib_world()
    api.molecules["CL"]["max_phase"] = "4.0"  # ChEMBL_37 serves max_phase as a string
    api.molecules["CD"]["pref_name"] = None
    api.names["DRUGCOMB LABEL"] = ["CD"]
    drugs = _drugs(("DRUGCOMB LABEL", ALANINE_FLAT))
    rows = _ok_rows(api, drugs)
    assert rows[0]["chembl_parent_id"] == "CD"  # the name route lands on the unnamed stereoisomer
    assert "no_approved_stereo_sibling" in _failed(hd.verify(drugs, rows, rows, api))
    fixed = [rows[0] | {"chembl_parent_id": "CL", "preferred_name": "L-ALANINE", "resolution": "manual",
                        "note": "approved sibling"}]
    assert _failed(hd.verify(drugs, fixed, rows, api)) == set()


def test_build_keeps_an_unresolved_label_as_its_own_drug():
    # Regression: under pandas 3 the missing ChEMBL ID was NaN, so drug_id became NaN and groupby
    # dropped the drug silently.
    api = erlotinib_world()
    df = _tdc([("ERLOTINIB", "MYSTERY", "L1", ERLOTINIB, "c1ccc2ccccc2c1")])
    rows = hd.proposal_rows(hd.source_drugs(df), api)
    res = hd.build(df, rows, api, rows)
    assert list(res.drugs["drug_id"]) == ["C553", "UNRESOLVED:MYSTERY"]
    assert res.drugs.set_index("drug_id").at["UNRESOLVED:MYSTERY", "smiles_origin"] == "source_parent"
    assert res.report["counts"]["drugs"] == 2 and res.report["passed"]


def _proxy_world():
    api = erlotinib_world()
    api.molecules["C1"] = mol("C1", GEFITINIB.replace("F", "Cl"), name="PRODRUG X", phase="4.0")
    api.mechs["C1"] = [{"mec_id": 9, "target_chembl_id": "T203", "action_type": "INHIBITOR",
                        "mechanism_of_action": "EGFR inhibitor", "direct_interaction": True}]
    df = _tdc([("ERLOTINIB", "GEFITINIB", "L1", ERLOTINIB, GEFITINIB)])
    return api, df, hd.proposal_rows(hd.source_drugs(df), api)


def test_mechanism_proxy_fills_a_gap_and_is_tagged():
    api, df, rows = _proxy_world()
    proxies = [{"drug_id": "C939", "proxy_chembl_id": "C1", "relation": "prodrug", "note": "test"}]
    res = hd.build(df, rows, api, rows, proxies)
    t = res.targets[res.targets["drug_id"] == "C939"]
    assert list(t["mechanism_source"]) == ["C1"] and list(t["gene_symbols"]) == ["EGFR"]
    assert res.drugs.set_index("drug_id").at["C939", "mechanism_source"] == "C1"
    assert res.drugs.set_index("drug_id").at["C553", "mechanism_source"] == "C553"


@pytest.mark.parametrize("proxy, reason", [
    ({"drug_id": "C553", "proxy_chembl_id": "C1", "relation": "prodrug", "note": "x"}, "own mechanisms"),
    ({"drug_id": "C939", "proxy_chembl_id": "C1", "relation": "cousin", "note": "x"}, "relation"),
    ({"drug_id": "C939", "proxy_chembl_id": "C1", "relation": "prodrug", "note": " "}, "no note"),
    ({"drug_id": "C939", "proxy_chembl_id": "CL", "relation": "prodrug", "note": "x"}, "no targeted mechanism"),
    ({"drug_id": "NOPE", "proxy_chembl_id": "C1", "relation": "prodrug", "note": "x"}, "not in table"),
])
def test_invalid_mechanism_proxy_fails_the_build(proxy, reason):
    api, df, rows = _proxy_world()
    with pytest.raises(hd.HarmonizationError, match="mechanism_proxies_valid") as e:
        hd.build(df, rows, api, rows, [proxy])
    detail = next(c["detail"] for c in e.value.report["checks"] if c["name"] == "mechanism_proxies_valid")
    assert reason in detail


@integration
def test_real_build_drugs_without_targets_are_exactly_the_known_ones(real_build):
    *_, res = real_build
    no_target = set(res.drugs.loc[res.drugs["n_targets"] == 0, "preferred_name"])
    assert no_target == {"AMIFOSTINE", "MRK-003", "LLL-12", "GELDANAMYCIN", "TRIETHYLENEMELAMINE",
                         "QUINACRINE", "MK-4541"}


@integration
def test_real_build_proxies_give_fludarabine_and_plicamycin_targets(real_build):
    *_, res = real_build
    d = res.drugs.set_index("drug_id")
    assert d.at["CHEMBL1568", "mechanism_source"] == "CHEMBL1096882" and d.at["CHEMBL1568", "n_targets"] > 0
    assert d.at["CHEMBL257619", "mechanism_source"] == "CHEMBL1237054" and d.at["CHEMBL257619", "n_targets"] > 0


#: Identities the drug identity audit (§2.1) guessed for the obscure ALMANAC labels by reading the
#: structure. The resolver reaches them independently (name routes + ChEMBL), so agreement is a check.
AUDIT_OBSCURE = {
    "CHEMBL17639": "CLADRIBINE", "CHEMBL277800": "CLOFARABINE", "34793-34-5": "NELARABINE",
    "ANTIBIOTIC AY 22989": "SIROLIMUS", "NSC733504": "EVEROLIMUS", "NSC-127716": "DECITABINE",
    "NSC 85998": "STREPTOZOCIN", "NSC256439": "IDARUBICIN", "NSC707389": "ERIBULIN",
    "ANTIBIOTIC AD 32": "VALRUBICIN", "CO-V": "PENTOSTATIN", "EMCYT (PHARMACIA)": "ESTRAMUSTINE PHOSPHATE",
    "1-(5-DEOXYPENTOFURANOSYL)-5-FLUORO-4-{[(PENTYLOXY)CARBONYL]AMINO}PYRIMIDIN-2(1H)-ONE": "CAPECITABINE",
    "CISPLATINO": "CISPLATIN", "MITOMYCINE": "MITOMYCIN", "ZOLINZA": "VORINOSTAT",
}


@integration
def test_real_build_confirms_the_audit_guesses_for_obscure_labels(real_build):
    *_, res = real_build
    got = res.drug_map.set_index("source_id")["preferred_name"]
    assert {k: got[k] for k in AUDIT_OBSCURE} == AUDIT_OBSCURE
