# Changelog

Newest first. Decision numbers refer to the table in
[`project-notes/m1-freeze-and-splits.md`](project-notes/m1-freeze-and-splits.md) §2.

## 2026-11-08: M1 (data and pool) complete

**Pipeline** (`src/combo_agent/data/`, tests in `tests/`):

| Step | Change |
|:--|:--|
| 1a | TDC NCI-60 table downloaded directly (no `tdc` package: its pins conflict with pandas 3). sha256-pinned, with a truncation check and retries; the pickle is converted with a restricted unpickler in an isolated environment |
| 1c | 129 drug labels → 122 ChEMBL parent molecules (7 cross-study synonym pairs). Name and structure must agree; Pt drugs checked by formula; reviewed `curated/drug_synonyms.csv`; targets by `parent_molecule_chembl_id`; 2 mechanism proxies |
| 1d | 59 line labels → 59 Cellosaurus lines, matched within the NCI-60 panel. 3 families (SNB-19 + U-251 MG, NCI/ADR-RES + OVCAR-8, MDA-MB-435 + M14). DepMap IDs: SR → ACH-000338, UO-31 → ACH-000428; HL-60(TB) proxied by HL-60. MDA-MB-435 and NCI/ADR-RES dropped (decisions 9–11) |
| 1e | Every source row gets one fate. **O'Neil and third-study rows removed** (decision 13); `study_id = NCI-ALMANAC` (12); replicates averaged within ALMANAC; self-pairs in their own table (5). Result: 272,357 experiments |
| 1f | Cell features: DepMap 26Q3 (lineage, 24 driver mutations) and NCI CellMiner v2.16 (expression, copy number) for all lines (decision 15). Parent mutation calls for SNB-19, NCI-H322M and HL-60(TB) (16). **SR's expression imputed** from CellMiner microarrays (17). Drug features: Morgan fingerprints, targets, action and target types |
| 1g | Held-out split: 11 lines, stratified by lineage, families kept together, eligibility rule (19). Thresholds y_syn ≥ 9.30 and y_inh ≥ 62.95 (selectable pool). `syn_hit_low_kill` flag, reported both ways (20). Byte-reproducible data card and manifest |

**Findings:**
- The TDC "DrugComb" NCI-60 file mixes NCI-ALMANAC with O'Neil/Merck and a small third study.
- O'Neil and ALMANAC disagree on the 546 experiments both measured (CSS r = 0.22).
- CellMiner's SR RNA-seq sample is a mix-up (it matches RPMI-8226). Its microarrays show the true
  ALK+ lymphoma profile.
- CellMiner misses many tumour-suppressor mutations (TP53, STK11, PTEN), so mutations come from
  DepMap.
- Bliss is noisy within ALMANAC (replicate r = 0.21, against 0.58 for CSS).
- Synergy and killing are nearly unrelated (Spearman 0.06).

**Repository:**
- `pyproject.toml` and `uv.lock`: numpy, pyarrow, rdkit; pytest (dev).
- `.gitignore`: `data_cache/`, `runs/`, `.cache/`. Also fixes a pattern that ignored
  `src/combo_agent/data/`.
- New: `scripts/`, `artifacts/m1_freeze/`, `docs/target_design.md`, `docs/project-notes/`,
  `docs/HANDOFF.md`, this changelog, `docs/handoff-template.md`.
- README: status, layout, rebuild instructions, and how to get the shared data bundle.
- **Data bundle `m1-data-v1`** (shared outside git, about 24 MB compressed): the frozen pool, plus
  full cell profiles, mappings and audit tables, with annotated docs (`docs/data-bundle/`: README,
  generated data dictionary, splits and hits, caveats, provenance) and checksums. Built by
  `scripts/make_data_bundle.py`.
  - **Published** to the team's
    [Google Drive folder](https://drive.google.com/drive/folders/1GsulBMNAGEjXhBt8p-UvNpYufTfvvRW7?usp=drive_link).
  - Archive sha256s: `m1-data-v1.tgz` `bd33234e…`, and `m1_api_cache.tgz` `b7577a54…` (full values
    in the README).
  - It was built from the M1 working tree just before these commits, so its `BUNDLE_INFO.json`
    reads `78f71b5 (with uncommitted pipeline changes)`. Its data files are byte-identical to what
    the committed code produces (`artifacts/m1_freeze/manifest.json`).
  - How it was made, and why data stays out of git:
    [`project-notes/data-bundle-and-sharing.md`](project-notes/data-bundle-and-sharing.md).

## Before 2026-11-08

- `78f71b5`: SHELDON harness set up; `combo_agent` package, CLI, stub benchmark (PR #2).
- `15fdebc`: project docs (`docs/assignment.md`, `docs/project-proposal.md`, notes).
