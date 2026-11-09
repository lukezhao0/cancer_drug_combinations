"""Step 1c: one drug, one identity (target_design.md section 4, step 3; revised plan P1-P5).

Every drug label in the TDC table is mapped to a **ChEMBL parent molecule ID**, which is the merge key
(P1). A label is accepted only when a name lookup and the file's structure agree (P2):

* name routes: the label is itself a ChEMBL ID; an exact ChEMBL preferred-name/synonym match on the
  label or a cleaned variant of it (salt words, brand/supplier parentheticals, NSC spellings);
  PubChem name -> InChIKey (with stereo) -> ChEMBL.
* structure check: the candidate's connectivity key equals the one computed from the file's SMILES
  (`structure.connectivity_key`), or, for metal complexes, the heavy-atom formula equals ChEMBL's
  (P3: no salt-stripping of Pt drugs).

The resolution is written to a **committed, reviewable table** `curated/drug_synonyms.csv` (P5).
`propose` regenerates the automatic part of it; rows marked `manual` are kept as curated overrides.
`build` reads that table (never the live resolver output), re-verifies every row against ChEMBL,
attaches ChEMBL parent SMILES with stereo (P4) and mechanism targets (queried by
`parent_molecule_chembl_id`), and writes:

    data_cache/interim/drug_map.parquet      one row per source label (129)
    data_cache/interim/drugs.parquet         one row per drug (ChEMBL parent)
    data_cache/interim/drug_targets.parquet  one row per (drug, mechanism target)
    data_cache/interim/drug_harmonization_report.json   counts, merge groups, every check

Every ChEMBL/PubChem response is cached in `data_cache/raw/api_cache/`; `--offline` replays the
build from that cache only, which is how byte-for-byte reproducibility is checked.

Not done here (step 1e): per-row `study_id`, removing O'Neil rows, and merging replicates. This
module only says which labels are the same molecule.

Run:
    python -m combo_agent.data.harmonize_drugs propose [--write]
    python -m combo_agent.data.harmonize_drugs build [--offline]
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from rdkit import Chem

from combo_agent.data import ingest_tdc, paths, structure
from combo_agent.data.api_cache import CachedJsonClient
from combo_agent.data.chembl import ChemblApi, gene_symbols

CURATED_CSV = Path(__file__).with_name("curated") / "drug_synonyms.csv"
#: Drugs whose ChEMBL parent has no mechanism of its own, but a closely related named ChEMBL record
#: does (fludarabine -> fludarabine phosphate, a prodrug that is a separate parent). Identity stays
#: the structure-verified parent; only the mechanism targets are borrowed, and are tagged as such.
PROXY_CSV = CURATED_CSV.with_name("drug_mechanism_proxies.csv")
PROXY_COLUMNS = ("drug_id", "proxy_chembl_id", "relation", "note")
PROXY_RELATIONS = ("prodrug", "same_drug_other_record")
CSV_COLUMNS = ("source_id", "chembl_parent_id", "preferred_name", "resolution", "structure_check",
               "pubchem_cid", "note")
#: Automatic outcomes, plus `manual` for a hand-set row.
RESOLUTIONS = ("name+structure", "structure_only", "ambiguous", "unresolved", "manual")
#: Outcomes that assign a ChEMBL parent.
RESOLVED = ("name+structure", "structure_only", "manual")
#: Words that name a salt or form rather than the molecule; dropped to make a name variant.
SALT_WORDS = ("HYDROCHLORIDE", "DIHYDROCHLORIDE", "HCL", "SULFATE", "SULPHATE", "CITRATE",
              "DITARTRATE", "TARTRATE", "MESYLATE", "MALEATE", "ACETATE", "PHOSPHATE", "SODIUM",
              "BASE", "ANHYDROUS")
NAME_ROUTES = ("chembl_id_label", "chembl_name", "pubchem")


class HarmonizationError(RuntimeError):
    """At least one check failed; the report lists which."""


# -- source labels ---------------------------------------------------------------------------------


def source_drugs(df: pd.DataFrame) -> pd.DataFrame:
    """One row per drug label in the TDC table: its SMILES and how much data uses it.

    Raises if a label carries more than one SMILES or a SMILES RDKit cannot parse (design step 3,
    "drop drugs with no structure", becomes a hard error: there are none in this file).
    """
    long = pd.concat([
        df[["Drug1_ID", "Drug1", "Cell_Line_ID"]].set_axis(["source_id", "smiles", "line"], axis=1),
        df[["Drug2_ID", "Drug2", "Cell_Line_ID"]].set_axis(["source_id", "smiles", "line"], axis=1),
    ])
    n_smiles = long.groupby("source_id")["smiles"].nunique()
    if (n_smiles != 1).any():
        raise HarmonizationError(f"labels with several SMILES: {sorted(n_smiles[n_smiles != 1].index)}")
    out = long.groupby("source_id").agg(smiles=("smiles", "first"), n_lines=("line", "nunique"))
    rows = (pd.concat([df["Drug1_ID"], df["Drug2_ID"].where(df["Drug2_ID"] != df["Drug1_ID"])])
            .dropna().value_counts())
    out["n_rows"] = rows.reindex(out.index).astype("int64")
    out = out.reset_index().sort_values("source_id", ignore_index=True)
    for s in out["smiles"]:
        structure.mol_from_smiles(s)  # raises StructureError on an unparseable SMILES
    return out[["source_id", "smiles", "n_rows", "n_lines"]]


def structure_signature(smiles: str) -> tuple[str, str]:
    """('connectivity', key) for organic structures, ('formula', heavy formula) for metal complexes."""
    if structure.has_metal(smiles):
        return "formula", structure.heavy_atom_formula(smiles)
    return "connectivity", structure.connectivity_key(smiles)


def name_queries(label: str) -> list[str]:
    """The label plus cleaned variants to try as exact names, most specific first."""
    out = [label]
    if " (" in label:  # "ABT-263 (NAVITOCLAX)", "ELOXATIN (TN) (SANOFI SYNTHELAB)"
        outer = re.sub(r"\s*\([^()]*\)", "", label).strip()
        out.append(outer)
        if re.search(r"\d", outer):  # outer is a code, so the parenthetical may be the name
            out += [p.strip() for p in re.findall(r"\(([^()]*)\)", label)]
    # Drop trailing salt words from the label and from the cleaned variant: "ERLOTINIB
    # HYDROCHLORIDE" -> "ERLOTINIB"; "NAVELBINE DITARTRATE (TN)" -> "NAVELBINE DITARTRATE" -> "NAVELBINE".
    for q in list(out):
        words = q.split()
        while len(words) > 1 and words[-1] in SALT_WORDS:
            words = words[:-1]
        out.append(" ".join(words))
    if m := re.fullmatch(r"NSC[\s-]?(\d+)", label):
        out += [f"NSC-{m[1]}", f"NSC {m[1]}", f"NSC{m[1]}"]
    seen: list[str] = []
    for q in out:
        if q and q not in seen:
            seen.append(q)
    return seen


# -- matching a candidate to the source structure ------------------------------------------------


def structure_check(source_smiles: str, molecule: dict[str, Any]) -> str | None:
    """How a ChEMBL record agrees with the source SMILES: 'connectivity', 'tautomer', 'formula', or
    None if it does not agree."""
    kind, signature = structure_signature(source_smiles)
    if kind == "formula":
        formula = ((molecule.get("molecule_properties") or {}).get("full_molformula")) or ""
        return "formula" if structure.heavy_atom_formula_from_molformula(formula) == signature else None
    structs = molecule.get("molecule_structures") or {}
    key, smiles = structs.get("standard_inchi_key"), structs.get("canonical_smiles")
    if key and key[:14] == signature:
        return "connectivity"
    if smiles and not structure.has_metal(smiles):
        try:
            if structure.tautomer_key(smiles) == structure.tautomer_key(source_smiles):
                return "tautomer"
        except structure.StructureError:
            return None
    return None


UNVERIFIABLE = "unverifiable"


def _phase(molecule: dict[str, Any] | None) -> float:
    """ChEMBL `max_phase` as a number (ChEMBL_37 serves it as a string such as "4.0"); -1 if none."""
    value = (molecule or {}).get("max_phase")
    return float(value) if value is not None else -1.0


def _has_structure(molecule: dict[str, Any]) -> bool:
    structs = molecule.get("molecule_structures") or {}
    props = molecule.get("molecule_properties") or {}
    return bool(structs.get("standard_inchi_key") or props.get("full_molformula"))


@dataclass
class Candidate:
    parent_id: str
    routes: set[str] = field(default_factory=set)
    check: str | None = None
    max_phase: float = -1.0

    @property
    def route_kinds(self) -> set[str]:
        return {r.split(":", 1)[0] for r in self.routes}

    @property
    def named(self) -> bool:
        return bool(self.route_kinds & set(NAME_ROUTES))

    def rank(self) -> tuple[int, int, float]:
        """Higher is better: independent name routes, an exact ChEMBL name match, clinical phase."""
        kinds = self.route_kinds & set(NAME_ROUTES)
        return len(kinds), int("chembl_name" in kinds or "chembl_id_label" in kinds), self.max_phase


@dataclass
class Proposal:
    source_id: str
    chembl_parent_id: str | None
    resolution: str
    structure_check: str | None
    pubchem_cid: int | None
    note: str
    candidates: dict[str, Candidate]


def propose_one(source_id: str, smiles: str, api: ChemblApi) -> Proposal:
    """Resolve one label automatically (no curated overrides)."""
    kind, signature = structure_signature(smiles)
    cands: dict[str, Candidate] = {}

    def add(molecule_id: str, route: str) -> None:
        parent = api.parent_id(molecule_id)
        if parent is not None:
            cands.setdefault(parent, Candidate(parent)).routes.add(route)

    if re.fullmatch(r"CHEMBL\d+", source_id):
        add(source_id, "chembl_id_label")
    for q in name_queries(source_id):
        for mid in api.molecules_by_name(q):
            add(mid, f"chembl_name:{q}")
    pubchem_cid = None
    for q in name_queries(source_id):
        hits = api.pubchem_by_name(q)
        for cid, inchikey in hits:
            if kind == "connectivity" and inchikey[:14] == signature and pubchem_cid is None:
                pubchem_cid = cid
            for mid in api.molecules_by_inchikey(inchikey):
                add(mid, f"pubchem:{cid}")
        if hits:
            break  # the most specific name PubChem knows is enough
    if kind == "connectivity":
        for mid in api.molecules_by_connectivity(signature):
            add(mid, "structure")

    for c in cands.values():
        rec = api.molecule(c.parent_id) or {}
        c.check = structure_check(smiles, rec)
        c.max_phase = float(rec.get("max_phase") or -1)

    passing = [c for c in cands.values() if c.check]
    conflicts = sorted(c.parent_id for c in cands.values() if c.named and not c.check)
    notes = [f"name matches failing the structure check: {', '.join(conflicts)}"] if conflicts else []
    named = sorted((c for c in passing if c.named), key=lambda c: c.rank(), reverse=True)
    if named:
        top = [c for c in named if c.rank() == named[0].rank()]
        if len(top) == 1:
            if len(named) > 1:
                notes.append("also matched (lower rank): " + ", ".join(c.parent_id for c in named[1:]))
            best = top[0]
            return Proposal(source_id, best.parent_id, "name+structure", best.check, pubchem_cid,
                            "; ".join(notes), cands)
        notes.append("tied name candidates: " + ", ".join(sorted(c.parent_id for c in top)))
        return Proposal(source_id, None, "ambiguous", None, pubchem_cid, "; ".join(notes), cands)
    structural = [c for c in passing if "structure" in c.route_kinds]
    if len(structural) == 1:
        c = structural[0]
        return Proposal(source_id, c.parent_id, "structure_only", c.check, pubchem_cid,
                        "; ".join(notes + ["no name route; unique ChEMBL parent with this skeleton"]),
                        cands)
    if structural:
        notes.append("several ChEMBL parents share this skeleton and no name decides: "
                     + ", ".join(sorted(c.parent_id for c in structural)))
        return Proposal(source_id, None, "ambiguous", None, pubchem_cid, "; ".join(notes), cands)
    notes.append("no ChEMBL record matches by name or structure")
    return Proposal(source_id, None, "unresolved", None, pubchem_cid, "; ".join(notes), cands)


def preferred_name(api: ChemblApi, parent_id: str | None, source_ids: Iterable[str]) -> str:
    """ChEMBL's preferred name; if it has none, the alphabetically first source label."""
    rec = api.molecule(parent_id) if parent_id else None
    return (rec or {}).get("pref_name") or min(source_ids)


def proposal_rows(drugs: pd.DataFrame, api: ChemblApi) -> list[dict[str, str]]:
    """Automatic CSV rows for every label. `preferred_name` is per ChEMBL parent."""
    props = [propose_one(r.source_id, r.smiles, api) for r in drugs.itertuples()]
    members: dict[str, list[str]] = defaultdict(list)
    for p in props:
        if p.chembl_parent_id:
            members[p.chembl_parent_id].append(p.source_id)
    return [_row(p.source_id, p.chembl_parent_id,
                 preferred_name(api, p.chembl_parent_id, members.get(p.chembl_parent_id or "", [p.source_id])),
                 p.resolution, p.structure_check, p.pubchem_cid, p.note) for p in props]


def _row(source_id, parent, name, resolution, check, cid, note) -> dict[str, str]:
    return {"source_id": source_id, "chembl_parent_id": parent or "", "preferred_name": name,
            "resolution": resolution, "structure_check": check or "",
            "pubchem_cid": "" if cid is None else str(cid), "note": note}


# -- curated table ---------------------------------------------------------------------------------


def read_curated(path: Path = CURATED_CSV) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != CSV_COLUMNS:
            raise HarmonizationError(f"{path}: columns {reader.fieldnames}, expected {list(CSV_COLUMNS)}")
        return list(reader)


def render_csv(rows: list[dict[str, str]]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CSV_COLUMNS, lineterminator="\n")
    w.writeheader()
    w.writerows(sorted(rows, key=lambda r: r["source_id"]))
    return buf.getvalue()


def merge_with_manual(auto: list[dict[str, str]], curated: list[dict[str, str]]) -> list[dict[str, str]]:
    """Automatic rows, except that curated `manual` rows replace their automatic counterpart."""
    manual = {r["source_id"]: r for r in curated if r["resolution"] == "manual"}
    return [manual.get(r["source_id"], r) for r in auto]


# -- verification --------------------------------------------------------------------------------


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    #: A warning is reported but does not fail the build.
    warning: bool = False

    def __post_init__(self) -> None:
        self.passed = bool(self.passed)  # numpy bools are not JSON-serializable

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "warning": self.warning, "detail": self.detail}


def verify(drugs: pd.DataFrame, rows: list[dict[str, str]], auto: list[dict[str, str]] | None,
           api: ChemblApi) -> list[Check]:
    """Every check that step 1c achieves "one drug, one identity". `auto` (the automatic proposal)
    is optional only so that the structural checks can be unit-tested without a resolver run."""
    checks: list[Check] = []
    by_id = {r["source_id"]: r for r in rows}
    smiles = dict(zip(drugs["source_id"], drugs["smiles"]))

    # C1 coverage: every label exactly once, nothing extra.
    ids = [r["source_id"] for r in rows]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    missing, extra = sorted(set(smiles) - set(ids)), sorted(set(ids) - set(smiles))
    checks.append(Check("coverage", not (dup or missing or extra),
                        f"{len(smiles)} labels; duplicated {dup}; missing {missing}; extra {extra}"))

    # C2 vocabulary: a known resolution; resolved rows have a ChEMBL ID, others do not; manual has a note.
    bad = [r["source_id"] for r in rows
           if r["resolution"] not in RESOLUTIONS
           or bool(r["chembl_parent_id"]) != (r["resolution"] in RESOLVED)
           or (r["resolution"] == "manual" and not r["note"].strip())]
    checks.append(Check("row_vocabulary", not bad, f"bad rows: {bad}"))

    resolved = [r for r in rows if r["chembl_parent_id"] and r["source_id"] in smiles]
    # C3 each ChEMBL ID exists and is a parent (its own hierarchy parent), not a salt form.
    not_parent = []
    for r in resolved:
        rec = api.molecule(r["chembl_parent_id"])
        parent = ((rec or {}).get("molecule_hierarchy") or {}).get("parent_chembl_id")
        if rec is None or parent not in (None, r["chembl_parent_id"]):
            not_parent.append(f"{r['source_id']} -> {r['chembl_parent_id']} (parent {parent})")
    checks.append(Check("chembl_id_is_parent", not not_parent, f"not a parent: {not_parent}"))

    # C4 name and structure agree for every resolved label; recorded check type is the real one.
    # A ChEMBL record with neither a structure nor a formula (e.g. BLEOMYCIN, a mixture) cannot be
    # checked; that is accepted only on a manual row, which must explain itself in its note.
    disagree = []
    for r in resolved:
        rec = api.molecule(r["chembl_parent_id"]) or {}
        got = structure_check(smiles[r["source_id"]], rec)
        if got is None and r["resolution"] == "manual" and not _has_structure(rec):
            got = UNVERIFIABLE
        if got is None or got != r["structure_check"]:
            disagree.append(f"{r['source_id']} -> {r['chembl_parent_id']}: recorded "
                            f"{r['structure_check']!r}, actual {got!r}")
    checks.append(Check("structure_agrees", not disagree, f"disagreements: {disagree}"))

    # C5 one identity per molecule: the ChEMBL grouping equals the independent RDKit grouping
    # (connectivity key / heavy-atom formula). Catches a stereoisomer collision (two parents, one
    # skeleton) and a split (one parent, two skeletons).
    sig = {r["source_id"]: structure_signature(smiles[r["source_id"]]) for r in resolved}
    by_parent, by_sig = defaultdict(set), defaultdict(set)
    for r in resolved:
        by_parent[r["chembl_parent_id"]].add(r["source_id"])
        by_sig[sig[r["source_id"]]].add(r["source_id"])
    a = sorted(sorted(g) for g in by_parent.values())
    b = sorted(sorted(g) for g in by_sig.values())
    diff = [g for g in a if g not in b] + [g for g in b if g not in a]
    checks.append(Check("grouping_matches_structure", not diff,
                        f"{len(a)} ChEMBL parents vs {len(b)} structure groups; differing groups {diff}"))

    # C5b stereo guard. The source SMILES have no stereo, so the structure check cannot tell
    # stereoisomers apart, and a name route can land on an unnamed stereoisomer (e.g. the label
    # `CHEMBL17639` is not CLADRIBINE, CHEMBL1619). An automatic row whose skeleton has a named
    # ChEMBL parent with a higher clinical phase must be reviewed and made a manual row.
    suspect = []
    for r in resolved:
        if r["resolution"] == "manual":
            continue
        kind, key = sig[r["source_id"]]
        if kind != "connectivity":
            continue
        phase = _phase(api.molecule(r["chembl_parent_id"]))
        siblings = {api.parent_id(m) for m in api.molecules_by_connectivity(key)} - {r["chembl_parent_id"], None}
        better = sorted(s for s in siblings
                        if (rec := api.molecule(s)) and rec.get("pref_name") and _phase(rec) > phase)
        if better:
            suspect.append(f"{r['source_id']} -> {r['chembl_parent_id']} (phase {phase}); named "
                           f"higher-phase stereo-siblings: {better}")
    checks.append(Check("no_approved_stereo_sibling", not suspect, f"review: {suspect}"))

    # C6 preferred name is ChEMBL's, consistently per parent.
    wrong_name = []
    for parent, members in by_parent.items():
        expect = preferred_name(api, parent, members)
        wrong_name += [f"{m}: {by_id[m]['preferred_name']!r} != {expect!r}"
                       for m in sorted(members) if by_id[m]["preferred_name"] != expect]
    checks.append(Check("preferred_name_from_chembl", not wrong_name, f"mismatches: {wrong_name}"))

    # C7 the curated table still equals what the resolver finds today (except manual overrides).
    if auto is not None:
        auto_by = {r["source_id"]: r for r in auto}
        drift = [f"{r['source_id']}: curated {r['chembl_parent_id'] or '-'} / {r['resolution']}, "
                 f"automatic {auto_by[r['source_id']]['chembl_parent_id'] or '-'} / "
                 f"{auto_by[r['source_id']]['resolution']}"
                 for r in rows if r["resolution"] != "manual" and r["source_id"] in auto_by
                 and any(r[k] != auto_by[r["source_id"]][k] for k in CSV_COLUMNS)]
        checks.append(Check("curated_matches_resolver", not drift, f"drift: {drift}"))

    # Warnings: things to review, not failures.
    review = sorted(r["source_id"] for r in rows if r["resolution"] in ("structure_only", "ambiguous",
                                                                         "unresolved"))
    checks.append(Check("rows_needing_review", not review, f"{review}", warning=True))
    return checks


# -- build -----------------------------------------------------------------------------------------


@dataclass
class BuildResult:
    drug_map: pd.DataFrame
    drugs: pd.DataFrame
    targets: pd.DataFrame
    report: dict[str, Any]


def _targets_for(api: ChemblApi, drug_id: str, parent_id: str) -> tuple[list[dict[str, Any]], int]:
    """Target rows from the mechanisms of `parent_id` (the drug itself, or its proxy)."""
    mechs = api.mechanisms(parent_id)
    out = []
    for m in mechs:
        tid = m.get("target_chembl_id")
        if not tid:
            continue
        t = api.target(tid) or {}
        out.append({
            "drug_id": drug_id, "target_chembl_id": tid, "target_pref_name": t.get("pref_name"),
            "target_type": t.get("target_type"), "organism": t.get("organism"),
            "gene_symbols": "|".join(gene_symbols(t)), "action_type": m.get("action_type"),
            "mechanism_of_action": m.get("mechanism_of_action"),
            "direct_interaction": m.get("direct_interaction"), "mec_id": int(m["mec_id"]),
            "mechanism_source": parent_id,
        })
    return out, len(mechs)


def read_proxies(path: Path = PROXY_CSV) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != PROXY_COLUMNS:
            raise HarmonizationError(f"{path}: columns {reader.fieldnames}, expected {list(PROXY_COLUMNS)}")
        return list(reader)


def verify_proxies(proxies: list[dict[str, str]], drug_ids: set[str], api: ChemblApi) -> Check:
    """A proxy fills a gap only: the drug exists and has no mechanism of its own; the proxy is a named
    ChEMBL parent with at least one targeted mechanism; the relation is known; the note says why."""
    bad = []
    seen: set[str] = set()
    for p in proxies:
        d, x = p["drug_id"], p["proxy_chembl_id"]
        rec = api.molecule(x) or {}
        why = []
        if d in seen:
            why.append("listed twice")
        seen.add(d)
        if d not in drug_ids:
            why.append("drug not in table")
        elif api.mechanisms(d):
            why.append("drug has its own mechanisms")
        if not rec.get("pref_name") or api.parent_id(x) != x:
            why.append("proxy is not a named ChEMBL parent")
        if not any(m.get("target_chembl_id") for m in api.mechanisms(x)):
            why.append("proxy has no targeted mechanism")
        if p["relation"] not in PROXY_RELATIONS:
            why.append(f"relation {p['relation']!r} not in {PROXY_RELATIONS}")
        if not p["note"].strip():
            why.append("no note")
        if why:
            bad.append(f"{d} -> {x}: {', '.join(why)}")
    return Check("mechanism_proxies_valid", not bad, f"{len(proxies)} proxies; invalid: {bad}")


def build(df: pd.DataFrame, rows: list[dict[str, str]], api: ChemblApi,
          auto: list[dict[str, str]] | None, proxies: list[dict[str, str]] | None = None) -> BuildResult:
    """Verify the curated rows and assemble the drug tables. Raises HarmonizationError on a failed
    check (the report is attached to the exception as `.report`)."""
    drugs = source_drugs(df)
    checks = verify(drugs, rows, auto, api)
    by_id = {r["source_id"]: r for r in rows}
    proxies = proxies or []
    proxy_by = {p["drug_id"]: p["proxy_chembl_id"] for p in proxies}

    dm = drugs.copy()
    sigs = [structure_signature(s) for s in dm["smiles"]]
    dm = dm.rename(columns={"smiles": "source_smiles"})
    dm["n_fragments"] = [structure.n_fragments(s) for s in dm["source_smiles"]]
    dm["source_has_metal"] = [k == "formula" for k, _ in sigs]
    dm["source_has_stereo"] = [structure.has_stereo(s) for s in dm["source_smiles"]]
    dm["source_conn_key"] = [v if k == "connectivity" else None for k, v in sigs]
    dm["source_heavy_formula"] = [structure.heavy_atom_formula(s) for s in dm["source_smiles"]]
    for col in ("chembl_parent_id", "preferred_name", "resolution", "structure_check", "note"):
        dm[col] = [by_id.get(i, {}).get(col) or None for i in dm["source_id"]]
    dm["pubchem_cid"] = pd.array([int(by_id[i]["pubchem_cid"]) if by_id.get(i, {}).get("pubchem_cid")
                                  else None for i in dm["source_id"]], dtype="Int64")
    # pd.isna, not truthiness: under pandas 3 a missing string is NaN, which is truthy, and a NaN
    # drug_id is then silently dropped by groupby (this lost MK-4541 once).
    dm["drug_id"] = [f"UNRESOLVED:{i}" if pd.isna(p) else p
                     for i, p in zip(dm["source_id"], dm["chembl_parent_id"])]
    dm["preferred_name"] = dm["preferred_name"].fillna(dm["source_id"])

    drug_rows, target_rows = [], []
    for drug_id, g in dm.groupby("drug_id", sort=True, dropna=False):
        parent = None if pd.isna(g["chembl_parent_id"].iloc[0]) else g["chembl_parent_id"].iloc[0]
        rec = (api.molecule(parent) if parent else None) or {}
        structs = rec.get("molecule_structures") or {}
        if structs.get("canonical_smiles"):
            smi, origin = structs["canonical_smiles"], "chembl_parent"
        elif g["source_has_metal"].iloc[0]:
            smi, origin = g["source_smiles"].iloc[0], "source"  # ChEMBL stores no Pt structures
        else:
            smi = Chem.MolToSmiles(structure.parent_mol(g["source_smiles"].iloc[0]))
            origin = "source_parent"
        tgts, n_mech = _targets_for(api, drug_id, parent) if parent else ([], 0)
        if not tgts and drug_id in proxy_by:
            tgts, _ = _targets_for(api, drug_id, proxy_by[drug_id])
        target_rows += tgts
        drug_rows.append({
            "drug_id": drug_id, "chembl_parent_id": parent, "preferred_name": g["preferred_name"].iloc[0],
            "source_ids": "|".join(sorted(g["source_id"])), "n_source_ids": int(len(g)),
            "n_rows": int(g["n_rows"].sum()), "smiles": smi, "smiles_origin": origin,
            "inchikey": structs.get("standard_inchi_key"),
            "structure_key": g["source_conn_key"].iloc[0] or g["source_heavy_formula"].iloc[0],
            "has_stereo": structure.has_stereo(smi),
            "n_unassigned_stereocentres": structure.n_unassigned_stereocentres(smi),
            "max_phase": float(rec["max_phase"]) if rec.get("max_phase") is not None else None,
            "n_mechanisms": n_mech, "n_targets": len({t["target_chembl_id"] for t in tgts}),
            "mechanism_source": tgts[0]["mechanism_source"] if tgts else None,
            "target_genes": "|".join(sorted({s for t in tgts for s in t["gene_symbols"].split("|") if s})),
        })
    drugs_df = pd.DataFrame(drug_rows).sort_values("drug_id", ignore_index=True)
    targets_df = (pd.DataFrame(target_rows, columns=[
        "drug_id", "target_chembl_id", "target_pref_name", "target_type", "organism", "gene_symbols",
        "action_type", "mechanism_of_action", "direct_interaction", "mec_id", "mechanism_source"])
        .sort_values(["drug_id", "mec_id", "target_chembl_id"], ignore_index=True))
    dm = dm[["source_id", "drug_id", "chembl_parent_id", "preferred_name", "resolution",
             "structure_check", "pubchem_cid", "source_smiles", "n_rows", "n_lines", "n_fragments",
             "source_has_metal", "source_has_stereo", "source_conn_key", "source_heavy_formula", "note"]]

    # Post-build checks on the assembled tables.
    checks.append(verify_proxies(proxies, set(drugs_df["drug_id"]), api))
    listed = [i for ids in drugs_df["source_ids"] for i in ids.split("|")]
    lost = sorted(set(dm["source_id"]) - set(listed))
    twice = sorted({i for i in listed if listed.count(i) > 1})
    checks.append(Check("every_label_in_one_drug", not lost and not twice and dm["drug_id"].notna().all(),
                        f"labels missing from drugs: {lost}; in several drugs: {twice}"))
    resolved = drugs_df[drugs_df["chembl_parent_id"].notna()]
    organic = resolved[resolved["smiles_origin"] == "chembl_parent"]
    multi = sorted(organic.loc[[structure.n_fragments(s) > 1 for s in organic["smiles"]], "drug_id"])
    checks.append(Check("parent_smiles_single_fragment", not multi, f"multi-fragment parents: {multi}"))
    # Stereo comes from ChEMBL (P4), never from the flat source SMILES. Parents that still leave a
    # centre unspecified are racemates or partly defined natural products; listed for the data card.
    partial = sorted(f"{d} ({n})" for d, n in zip(organic["drug_id"], organic["n_unassigned_stereocentres"]) if n)
    checks.append(Check("parent_stereo_from_chembl", True,
                        f"{int(organic['has_stereo'].sum())} of {len(organic)} ChEMBL parent SMILES carry "
                        f"stereo; the rest are achiral or racemic. Unassigned centres: {partial}"))
    no_tgt = sorted(drugs_df.loc[drugs_df["n_targets"] == 0, "drug_id"])
    checks.append(Check("drugs_without_targets", not no_tgt,
                        f"{len(no_tgt)} drugs have no ChEMBL mechanism target (dropped at freeze, step "
                        f"7): {no_tgt}", warning=True))

    groups = {d: s.split("|") for d, s in zip(drugs_df["drug_id"], drugs_df["source_ids"]) if "|" in s}
    status = api.status() or {}
    report = {
        "chembl_release": status.get("chembl_db_version"),
        "chembl_release_date": status.get("chembl_release_date"),
        "counts": {
            "source_labels": int(len(dm)),
            "drugs": int(len(drugs_df)),
            "resolved_labels": int(dm["chembl_parent_id"].notna().sum()),
            "chembl_parents": int(resolved["drug_id"].nunique()),
            "merge_groups": len(groups),
            "drugs_with_targets": int((drugs_df["n_targets"] > 0).sum()),
            "target_rows": int(len(targets_df)),
            "source_smiles_with_stereo": int(dm["source_has_stereo"].sum()),
        },
        "resolution_counts": {k: int(v) for k, v in sorted(dm["resolution"].value_counts().items())},
        "structure_check_counts": {k: int(v) for k, v in
                                   sorted(dm["structure_check"].fillna("none").value_counts().items())},
        "merge_groups": {d: {"preferred_name": drugs_df.set_index("drug_id").at[d, "preferred_name"],
                             "source_ids": ids} for d, ids in sorted(groups.items())},
        "checks": [c.as_dict() for c in checks],
        "passed": all(c.passed or c.warning for c in checks),
    }
    result = BuildResult(dm, drugs_df, targets_df, report)
    if not report["passed"]:
        err = HarmonizationError("drug harmonization checks failed: "
                                 + ", ".join(c.name for c in checks if not (c.passed or c.warning)))
        err.report = report  # type: ignore[attr-defined]
        raise err
    return result


def write_outputs(result: BuildResult, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "drug_map": out_dir / "drug_map.parquet",
        "drugs": out_dir / "drugs.parquet",
        "drug_targets": out_dir / "drug_targets.parquet",
        "report": out_dir / "drug_harmonization_report.json",
    }
    result.drug_map.to_parquet(files["drug_map"], index=False)
    result.drugs.to_parquet(files["drugs"], index=False)
    result.targets.to_parquet(files["drug_targets"], index=False)
    files["report"].write_text(json.dumps(result.report, indent=2, sort_keys=True) + "\n")
    return files


# -- CLI -------------------------------------------------------------------------------------------


def make_api(offline: bool | None = None, cache_root: Path | None = None) -> ChemblApi:
    root = cache_root or paths.raw_dir() / "api_cache"
    return ChemblApi(CachedJsonClient(root / "chembl", offline=offline),
                     # PubChem asks for at most 5 requests per second
                     CachedJsonClient(root / "pubchem", offline=offline, min_interval=0.25))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Step 1c: harmonize drug identities")
    sub = p.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("propose", help="resolve all labels automatically and diff with the curated CSV")
    pp.add_argument("--write", action="store_true",
                    help=f"write the proposal to {CURATED_CSV.name}, keeping its manual rows")
    pb = sub.add_parser("build", help="verify the curated CSV and write the drug tables")
    for q in (pp, pb):
        q.add_argument("--offline", action="store_true", help="use cached API responses only")
    pb.add_argument("--out", type=Path, default=None, help="output dir (default data_cache/interim)")
    args = p.parse_args(argv)

    api = make_api(offline=args.offline or None)
    df = ingest_tdc.load("tdc_drugcomb")
    drugs = source_drugs(df)
    curated = read_curated() if CURATED_CSV.exists() else []
    auto = proposal_rows(drugs, api)

    if args.cmd == "propose":
        merged = merge_with_manual(auto, curated)
        text = render_csv(merged)
        old = render_csv(curated) if curated else ""
        changed = [r["source_id"] for r in merged
                   if {c["source_id"]: c for c in curated}.get(r["source_id"]) != r]
        print(f"{len(merged)} labels; {len(changed)} differ from {CURATED_CSV.name}: {changed}")
        for r in merged:
            if r["resolution"] not in ("name+structure",):
                print(f"  review: {r['source_id']}: {r['resolution']} {r['chembl_parent_id']} {r['note']}")
        if args.write and text != old:
            CURATED_CSV.parent.mkdir(parents=True, exist_ok=True)
            CURATED_CSV.write_text(text)
            print(f"wrote {CURATED_CSV}")
        return 0

    try:
        result = build(df, curated, api, auto, read_proxies())
    except HarmonizationError as e:
        report = getattr(e, "report", None)
        if report:
            for c in report["checks"]:
                if not (c["passed"] or c["warning"]):
                    print(f"FAIL {c['name']}: {c['detail']}", file=sys.stderr)
        raise
    files = write_outputs(result, args.out or paths.interim_dir())
    for c in result.report["checks"]:
        flag = "ok  " if c["passed"] else ("WARN" if c["warning"] else "FAIL")
        print(f"{flag} {c['name']}: {c['detail'][:200]}")
    print(json.dumps(result.report["counts"], indent=2))
    for name, path in files.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
