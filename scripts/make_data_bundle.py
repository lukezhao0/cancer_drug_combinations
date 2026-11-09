"""Package the frozen M1 data into a shareable, documented bundle (for Drive or Box).

Output, in `data_cache/bundles/` (git-ignored):

    m1-data-v1/                 the bundle folder
      README.md, SPLITS_AND_HITS.md, CAVEATS.md, PROVENANCE.md   copied from docs/data-bundle/
      DATA_DICTIONARY.md        generated here from the actual files (every column); a copy is also
                                written to docs/data-bundle/ so it can be reviewed in git
      SHA256SUMS                checksums of every file under data_cache/
      BUNDLE_INFO.json          version, build date, code commit, file counts
      data_cache/...            the data, laid out like the repo's data_cache/
    m1-data-v1.tgz              the folder, archived (upload this)
    m1-data-v1.tgz.sha256       checksum of the archive (record it next to the download link)
    m1_api_cache.tgz            optional (--with-api-cache): cached ChEMBL, PubChem and Cellosaurus
                                responses, for byte-identical rebuilds of steps 1c/1d

Run from the repo root, after the pipeline has been built (`freeze_pool` last):

    uv run python scripts/make_data_bundle.py [--version m1-data-v1] [--with-api-cache]
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / "data_cache"
DOCS = REPO / "docs" / "data-bundle"
STATIC_DOCS = ("README.md", "SPLITS_AND_HITS.md", "CAVEATS.md", "PROVENANCE.md")

# Files included, relative to data_cache/. Raw downloads and raw-derived intermediates are left
# out: they can be rebuilt, and some contain every outcome.
FILES = {
    "processed": ["pool.parquet", "labels.parquet", "cells.parquet", "drugs.parquet", "self_pairs.parquet",
                  "splits.json", "data_card.md", "manifest.json", "freeze_report.json"],
    "interim/features": ["cell_expression.parquet", "cell_copy_number.parquet", "cell_mutations.parquet",
                         "cell_mutations_cellminer.parquet", "expression_identity.parquet",
                         "features_report.json"],
    "interim": ["drugs.parquet", "drug_targets.parquet", "drug_map.parquet", "line_map.parquet",
                "cell_lines.parquet", "oneil_rows.parquet", "cross_study_overlap.parquet", "row_fate.parquet",
                "experiments_report.json", "drug_harmonization_report.json",
                "line_harmonization_report.json"],
}

# -- data dictionary ------------------------------------------------------------------------------

FILE_NOTES = {
    "processed/pool.parquet": "One row per experiment (drug A, drug B, cell line). **No outcomes.** Key: (drug_a, drug_b, cellosaurus_ac).",
    "processed/labels.parquet": "**Restricted: the hidden answers.** Same keys and row order as `pool.parquet`, plus outcomes and hit flags. Read only from oracle and evaluation code (see README).",
    "processed/cells.parquet": "One row per pool cell line: identity, split, data-source flags, and the model-ready cell features.",
    "processed/drugs.parquet": "One row per drug (122; `in_pool` = 101): annotation flags and the model-ready drug features.",
    "processed/self_pairs.parquet": "Drug-with-itself experiments. Not combinations; kept for score sanity checks (Loewe should be about 0).",
    "processed/splits.json": "Held-out lines (Cellosaurus IDs and names), per-lineage record, seed, hit thresholds.",
    "processed/data_card.md": "Human-readable summary of the pool.",
    "processed/manifest.json": "sha256 of every input and output of the freeze, plus raw-download records.",
    "processed/freeze_report.json": "Split statistics, mutation testability per gene, acceptance checks.",
    "interim/features/cell_expression.parquet": "Full expression profiles: 57 lines × all CellMiner genes, log2(FPKM+1). SR's row is imputed. For agent tools and expression rules (M6/M7).",
    "interim/features/cell_copy_number.parquet": "Full copy-number profiles: 57 lines × all genes, log2 ratio to normal DNA (0 = 2 copies; copy number N = 2·2^value). Empty where CellMiner has no value.",
    "interim/features/cell_mutations.parquet": "Driver-mutation calls used in the features: 57 lines × 24 genes (1 = mutated), with the source of each line's calls.",
    "interim/features/cell_mutations_cellminer.parquet": "CellMiner's own exome calls for the same 24 genes, for comparison only (less sensitive for tumour-suppressor genes).",
    "interim/features/expression_identity.parquet": "Identity check: each line's CellMiner expression vs its own DepMap profile (Spearman). `suspect` = likely sample mix-up (SR).",
    "interim/features/features_report.json": "Feature-build report: sources, imputation quality, checks.",
    "interim/drugs.parquet": "Drug identities: ChEMBL parent, name, structure, mechanism counts.",
    "interim/drug_targets.parquet": "One row per (drug, mechanism target) from ChEMBL: target, action, mechanism description. For the agent's drug lookups.",
    "interim/drug_map.parquet": "Source drug label (as in the TDC file) → drug identity, with how it was resolved.",
    "interim/line_map.parquet": "Source cell-line label → Cellosaurus accession, DepMap ID, family, keep/drop.",
    "interim/cell_lines.parquet": "All 59 source cell lines (including the 2 dropped): Cellosaurus details, problems, families.",
    "interim/oneil_rows.parquet": "**Contains outcomes** (O'Neil study). Rows removed from the pool (decision 13), for the second benchmark and analysis.",
    "interim/cross_study_overlap.parquet": "**Contains pool outcomes** for 546 experiments measured by both studies (O'Neil vs ALMANAC means). Treat like labels.",
    "interim/row_fate.parquet": "What happened to each of the 297,098 source rows (kept, O'Neil, dropped line, …).",
    "interim/experiments_report.json": "Experiment-build report: fate counts, replicate agreement, cross-study agreement.",
    "interim/drug_harmonization_report.json": "Drug-identity build report.",
    "interim/line_harmonization_report.json": "Cell-line build report.",
}

COLS = {
    "drug_a": "first drug's ID (ChEMBL parent ID, e.g. CHEMBL553); always `drug_a < drug_b`",
    "drug_b": "second drug's ID (ChEMBL parent ID)",
    "drug_id": "drug ID (ChEMBL parent ID); joins to `drug_a` / `drug_b`",
    "cellosaurus_ac": "cell-line ID (Cellosaurus accession, e.g. CVCL_0023 = A-549); the cell-line join key",
    "cellosaurus_name": "Cellosaurus name of the line",
    "split": "`selectable` (usable by selectors and models) or `heldout` (only for the final mechanism check)",
    "lineage": "tissue of origin (DepMap OncotreeLineage)",
    "primary_disease": "DepMap OncotreePrimaryDisease",
    "subtype": "DepMap OncotreeSubtype",
    "depmap_id": "DepMap model ID (ACH-…) used for this line's DepMap data",
    "depmap_is_proxy": "True if the DepMap profile is a parent line's (HL-60(TB) → HL-60)",
    "family_id": "lines sharing an origin share this ID; a family is never split across selectable and held out",
    "study_id": "source screen; always `NCI-ALMANAC` in this pool",
    "n_reps": "number of measurements averaged into this row (1 or 2)",
    "y_syn": "**synergy outcome** = mean Bliss score (above 0 = better than independent action)",
    "y_inh": "**killing outcome** = mean CSS, drug combination sensitivity score (higher = kills more)",
    "css_mean": "mean CSS (equals `y_inh`)", "css_sd": "SD of CSS across replicates (empty if n_reps = 1)",
    "bliss_mean": "mean Bliss synergy (equals `y_syn`)", "bliss_sd": "SD of Bliss across replicates",
    "zip_mean": "mean ZIP synergy", "zip_sd": "SD of ZIP",
    "loewe_mean": "mean Loewe synergy", "loewe_sd": "SD of Loewe",
    "hsa_mean": "mean HSA synergy", "hsa_sd": "SD of HSA",
    "is_hit_syn": "1 if `y_syn` ≥ the synergy threshold (p95 of selectable experiments, 9.30)",
    "is_hit_inh": "1 if `y_inh` ≥ the killing threshold (p95 of selectable experiments, 62.95)",
    "syn_hit_low_kill": "1 if a synergy hit has `y_inh` < 0 (report synergy results with and without these; decision 20)",
    "problematic": "Cellosaurus 'problematic cell line' note (contamination or misidentification), if any",
    "expression_source": "`cellminer` (measured) or `imputed` (SR; see CAVEATS)",
    "expression_missing": "1 if no expression at all (none in this version)",
    "expression_imputed": "1 if expression was imputed (SR)",
    "expression_identity_spearman": "Spearman between this line's CellMiner and own-DepMap expression (identity check)",
    "mutation_source": "`depmap` (own profile) or `depmap_parent` (borrowed from a parent line)",
    "mutation_model": "DepMap model the mutation calls came from",
    "mutation_from_parent": "1 if mutation calls are borrowed from a parent line",
    "mutation_from_cellminer": "1 if mutation calls came from CellMiner (none in this version)",
    "cn_source": "copy-number source (`cellminer`)",
    "eligible_for_heldout": "True if every feature of the line is its own measurement (decision 19)",
    "preferred_name": "drug name (ChEMBL preferred name)",
    "target_status": "`chembl` (own mechanism records), `chembl_proxy` (borrowed from a related record) or `none_in_chembl` (no target; kept, decision 8)",
    "smiles_origin": "where the structure came from: ChEMBL parent, or the source file (Pt drugs)",
    "in_pool": "1 if the drug appears in the pool (101 of 122)",
    "has_target": "1 if ChEMBL lists any mechanism target (DNA included)",
    "has_target_gene": "1 if a target has a gene symbol (DNA-targeting drugs have none)",
    "smiles_from_source": "1 if the fingerprint uses the source structure (3 Pt drugs)",
    "n_target_genes": "number of target genes",
    "source_id": "the label exactly as in the TDC source file",
    "source_row": "row index in the converted TDC table",
    "fate": "what happened to the source row (kept, oneil, oneil_ambiguous, line_dropped, self_pair, third_study)",
    # expression identity check
    "own_spearman": "Spearman between the line's CellMiner and own-DepMap expression (2,000 most variable genes)",
    "best_depmap_id": "the DepMap model whose expression matches this CellMiner profile best",
    "best_spearman": "Spearman with that best-matching model",
    "own_is_best": "True if the best match is the line's own DepMap profile",
    "suspect": "True if not its own best match and own_spearman < 0.80 (likely sample mix-up; SR)",
    # drug identities and mechanisms
    "chembl_parent_id": "ChEMBL parent molecule ID (salt forms resolved to the parent)",
    "source_ids": "all source labels merged into this identity (`|`-separated)",
    "n_source_ids": "number of source labels merged into this drug",
    "n_rows": "number of source rows that use this label or drug",
    "smiles": "structure used for fingerprints (ChEMBL parent with stereochemistry; source for Pt drugs)",
    "inchikey": "InChIKey of the structure",
    "structure_key": "first 14 characters of the InChIKey (atom connectivity, ignoring 3-D shape)",
    "has_stereo": "True if the structure carries stereochemistry",
    "n_unassigned_stereocentres": "stereocentres left unassigned in the structure",
    "max_phase": "highest clinical phase in ChEMBL (4 = approved)",
    "n_mechanisms": "number of ChEMBL mechanism records",
    "n_targets": "number of distinct mechanism targets (DNA counts as a target)",
    "mechanism_source": "ChEMBL ID the mechanism records were taken from (differs from the drug's own ID for proxies)",
    "target_genes": "target gene symbols (`|`-separated)",
    "target_chembl_id": "ChEMBL target ID",
    "target_pref_name": "target name",
    "target_type": "ChEMBL target type (SINGLE PROTEIN, NUCLEIC-ACID, PROTEIN FAMILY, …)",
    "organism": "target organism",
    "gene_symbols": "gene symbols of the target (`|`-separated; empty for DNA)",
    "action_type": "action on the target (INHIBITOR, AGONIST, …)",
    "mechanism_of_action": "ChEMBL's mechanism description, e.g. 'Epidermal growth factor receptor erbB1 inhibitor'",
    "direct_interaction": "1 if the drug binds the target directly",
    "mec_id": "ChEMBL mechanism record ID",
    # name mappings
    "resolution": "how the label was resolved (e.g. name+structure, manual, nci60_panel)",
    "structure_check": "how name and structure were confirmed to agree (connectivity, formula, …)",
    "pubchem_cid": "PubChem compound ID used in resolution, if any",
    "source_smiles": "the structure as given in the source file (no stereochemistry)",
    "n_lines": "number of cell lines the label appears on",
    "n_fragments": "number of disconnected fragments in the source structure (salts, Pt complexes)",
    "source_has_metal": "True if the source structure contains a metal (Pt drugs)",
    "source_has_stereo": "True if the source structure has stereochemistry (never, in this file)",
    "source_conn_key": "connectivity key computed from the source structure",
    "source_heavy_formula": "heavy-atom formula of the source structure (used to check Pt drugs)",
    "note": "reviewer's note explaining a manual or unusual mapping",
    "match_kind": "which Cellosaurus name matched (identifier, synonym, manual)",
    "keep": "True if the line is in the pool",
    "drop_reason": "why a line is not in the pool (e.g. no_depmap_id)",
    "depmap_ids_listed": "all DepMap IDs Cellosaurus lists for the line (`|`-separated)",
    "depmap_proxy_of": "Cellosaurus ID of the parent line whose DepMap profile is used, if any",
    "category": "Cellosaurus category (all: Cancer cell line)",
    "sex": "sex of the donor",
    "disease_ncit": "NCI Thesaurus disease code(s)",
    "disease": "disease name(s)",
    "in_nci60_panel": "True if Cellosaurus lists the line as part of the NCI-60 panel",
    "derived_from": "Cellosaurus IDs of parent lines (`|`-separated)",
    # audit tables (source column names kept from the TDC file)
    "Drug1_ID": "first drug label as in the source file", "Drug2_ID": "second drug label as in the source file",
    "Cell_Line_ID": "cell-line label as in the source file",
    "CSS": "CSS (killing) as in the source", "Synergy_Bliss": "Bliss synergy as in the source",
    "Synergy_ZIP": "ZIP synergy as in the source", "Synergy_Loewe": "Loewe synergy as in the source",
    "Synergy_HSA": "HSA synergy as in the source",
}
for score in ("css", "bliss", "zip", "loewe", "hsa"):
    for study, label in (("almanac", "NCI-ALMANAC"), ("oneil", "O'Neil")):
        COLS[f"{score}_mean_{study}"] = f"mean {score.upper()} measured by {label}"
COLS["n_reps_almanac"] = "number of NCI-ALMANAC measurements"
COLS["n_reps_oneil"] = "number of O'Neil measurements"

GROUPS = [  # (column regex, description) for wide tables
    (r"^fp_\d+$", "Morgan fingerprint bits (radius 2, 2,048 bits): 1 = the substructure is present"),
    (r"^target_(?!status$).+$", "target gene, multi-hot: 1 = the drug acts on this gene (from ChEMBL mechanisms)"),
    (r"^action_.+$", "mechanism action type, multi-hot (e.g. INHIBITOR, AGONIST)"),
    (r"^ttype_.+$", "target type, multi-hot (e.g. SINGLE_PROTEIN, NUCLEIC_ACID)"),
    (r"^lineage_.+$", "lineage one-hot (tissue)"),
    (r"^mut_.+$", "driver-gene mutation, 0/1: oncogenes = hotspot; tumour suppressors = damaging (TP53 also missense)"),
    (r"^cn_missing_.+$", "1 if this gene's copy number is missing for the line (value filled with 0 = 2 copies)"),
    (r"^cn_(?!source$|missing_).+$", "copy number of a driver / amplified gene, log2 ratio to normal (0 = 2 copies)"),
    (r"^expr_.+$", "expression of one of the 1,000 most variable genes, log2(FPKM+1) (CellMiner RNA-seq)"),
]


def describe_table(rel: str, path: Path) -> list[str]:
    schema = pq.read_schema(path)
    n = pq.read_metadata(path).num_rows
    cols = schema.names
    lines = [f"### `{rel}`", "", FILE_NOTES.get(rel, ""), "",
             f"{n:,} rows × {len(cols):,} columns.", ""]
    grouped: dict[str, list[str]] = {}
    singles = []
    wide = len(cols) > 200  # only the model-ready feature tables group columns by prefix
    for c in cols:
        for pat, desc in (GROUPS if wide else ()):
            if re.match(pat, c):
                grouped.setdefault(desc, []).append(c)
                break
        else:
            singles.append(c)
    gene_table = len(singles) > 60  # full expression / copy-number tables: one column per gene
    if gene_table:
        lines += [f"- `cellosaurus_ac`: {COLS['cellosaurus_ac']}",
                  f"- **{len(cols) - 1:,} gene columns**, named by HGNC gene symbol "
                  f"(e.g. `{'`, `'.join(c for c in singles[1:4])}`). Values as described above.", ""]
        return lines
    lines += ["| Column | Type | Meaning |", "|:--|:--|:--|"]
    for c in singles:
        t = str(schema.field(c).type).replace("large_string", "string")
        desc = COLS.get(c)
        if desc is None and re.fullmatch(r"[A-Z0-9]+", c):
            desc = "driver gene: 1 = mutated in this line"
        if desc is None:
            raise SystemExit(f"{rel}: column {c!r} has no description; add it to COLS")
        lines.append(f"| `{c}` | {t} | {desc} |")
    for desc, cs in grouped.items():
        lines.append(f"| `{cs[0]}` … `{cs[-1]}` ({len(cs):,} columns) | numeric | {desc} |")
    return lines + [""]


def dictionary(bundle: Path) -> str:
    out = ["# Data dictionary", "",
           "Generated from the files in this bundle by `scripts/make_data_bundle.py`. Every parquet file "
           "and column is listed. Wide tables group their feature columns by prefix.", "",
           "**Join keys:** experiments join to drugs on `drug_a` / `drug_b` = `drug_id`, and to cell lines on "
           "`cellosaurus_ac`. `pool` and `labels` share the key (drug_a, drug_b, cellosaurus_ac) and the same row "
           "order.", ""]
    for folder in FILES:
        out += [f"## `data_cache/{folder}/`", ""]
        for name in FILES[folder]:
            rel = f"{folder}/{name}"
            p = bundle / "data_cache" / folder / name
            if name.endswith(".parquet"):
                out += describe_table(rel, p)
            else:
                out += [f"### `{rel}`", "", FILE_NOTES.get(rel, ""), ""]
    return "\n".join(out) + "\n"


# -- packaging ------------------------------------------------------------------------------------


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def git_commit() -> str:
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True,
                             check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "tests"], cwd=REPO,
                               capture_output=True, text=True, check=True).stdout.strip()
        return rev + (" (with uncommitted pipeline changes)" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build(version: str, with_api_cache: bool) -> Path:
    out_root = CACHE / "bundles"
    bundle = out_root / version
    if bundle.exists():
        shutil.rmtree(bundle)
    (bundle / "data_cache").mkdir(parents=True)
    missing = [f"{d}/{n}" for d, ns in FILES.items() for n in ns if not (CACHE / d / n).exists()]
    if missing:
        sys.exit(f"missing inputs (build the pipeline first): {missing}")
    for d, names in FILES.items():
        (bundle / "data_cache" / d).mkdir(parents=True, exist_ok=True)
        for n in names:
            shutil.copy2(CACHE / d / n, bundle / "data_cache" / d / n)
    for doc in STATIC_DOCS:
        shutil.copy2(DOCS / doc, bundle / doc)
    text = dictionary(bundle)
    (bundle / "DATA_DICTIONARY.md").write_text(text)
    (DOCS / "DATA_DICTIONARY.md").write_text(text)  # keep a reviewable copy in git
    data_files = sorted(p for p in (bundle / "data_cache").rglob("*") if p.is_file())
    (bundle / "SHA256SUMS").write_text(
        "".join(f"{sha256(p)}  {p.relative_to(bundle)}\n" for p in data_files))
    info = {"version": version, "built": dt.date.today().isoformat(), "code_commit": git_commit(),
            "files": len(data_files), "bytes": sum(p.stat().st_size for p in data_files),
            "pool_manifest_sha256": sha256(bundle / "data_cache/processed/manifest.json")}
    (bundle / "BUNDLE_INFO.json").write_text(json.dumps(info, indent=2) + "\n")

    archive = out_root / f"{version}.tgz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(bundle, arcname=version)
    (out_root / f"{version}.tgz.sha256").write_text(f"{sha256(archive)}  {archive.name}\n")
    print(f"bundle:  {bundle}  ({info['files']} data files, {info['bytes'] / 1e6:.1f} MB)")
    print(f"archive: {archive}  ({archive.stat().st_size / 1e6:.1f} MB)")
    if with_api_cache:
        api = out_root / "m1_api_cache.tgz"
        with tarfile.open(api, "w:gz") as tar:
            tar.add(CACHE / "raw" / "api_cache", arcname="api_cache")
        (out_root / "m1_api_cache.tgz.sha256").write_text(f"{sha256(api)}  {api.name}\n")
        print(f"api cache: {api}  ({api.stat().st_size / 1e6:.1f} MB); unpack into data_cache/raw/")
    return bundle


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Package the frozen M1 data for sharing")
    p.add_argument("--version", default="m1-data-v1")
    p.add_argument("--with-api-cache", action="store_true", help="also archive the API-response cache")
    args = p.parse_args(argv)
    build(args.version, args.with_api_cache)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
