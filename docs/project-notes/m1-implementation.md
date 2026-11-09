# M1 implementation notes

These are the living notes for building M1 (data and pool, `target_design.md` §4). It holds the
team's decisions, the verified data facts behind them, and the status of each piece. Tags:
**[verified]** = observed in data or an API; **[decision]** = the team chose it on the date shown.

## Decisions (2026-11-08)

| # | Question | Decision |
|:--|:--|:--|
| 1 | How to get `uv` | Install `uv` (e.g. `brew install uv`) |
| 2 | Converting the TDC pickles | Approved by the team. Done with a **restricted unpickler** (an allowlist of 9 pandas/numpy globals) in an isolated pandas<3 / numpy<2 environment |
| 3 | Schema fixes | **Adopt all three** (gap analysis §2.10–2.12): `pool.parquet` without outcomes plus a separate `labels.parquet`; the held-out cell-line split is chosen in M1, so hit thresholds come from the **selectable** pool; `study_id` is one constant per dataset |
| 4 | Scope | **NCI-60 only** for now. O'Neil is deferred until D2 is decided and disk space is freed |
| 5 | Self-combinations (115 rows where Drug1 == Drug2) | **Keep them in a separate table**, out of the pool. They can serve as a check on the scores. **Correction:** Loewe (the sham-combination principle) should be ≈ 0 for a self-pair; Bliss isn't guaranteed to be 0, so its drift from 0 measures noise and bias. The earlier "Bliss ≈ 0" wording was wrong |
| 6 | Salt forms and synonyms | **Merge by parent structure**: RDKit standardization (largest fragment, neutralize), then InChIKey. Colliding triplets merge as replicates. A mapping table is saved |
| 7 | Related cell lines | **Group them**: lines sharing an origin (Cellosaurus hierarchy) always land on the same side of the held-out split. Lineage comes from DepMap |
| 8 | Drugs with no ChEMBL target (7 drugs, 17,260 rows) | **[decision, 2026-11-08] Keep them and their data; don't drop.** The agent looks up their possible targets and functions in the literature at run time. This overrides design §4 step 7 ("keep drugs with at least one annotated target"). See [Decision 8](#decision-8-keep-drugs-without-a-chembl-target) |
| 9 | SR and UO-31: two DepMap IDs each in Cellosaurus (SR: ACH-000338 / ACH-001665; UO-31: ACH-000428 / ACH-002203) | **[decision, 2026-11-08] Keep them; choose one ID each in 1f.** Pick the ID that has DepMap expression and mutation data, and write it to the `depmap_id` column of `curated/cell_lines.csv` (`build` checks it's one Cellosaurus lists). If both have data, prefer the one in DepMap's current default model set and record why. Until then they stay flagged `multiple_depmap_ids`. Recovers 9,865 rows |
| 10 | HL-60(TB): no DepMap ID; a subline of HL-60 (`CVCL_0002`, DepMap `ACH-000002`) | **[decision, 2026-11-08] Keep it, using HL-60's DepMap profile as a proxy.** It's marked as a proxy everywhere (cell table, data card). This keeps a 5th of the NCI-60 panel's 6 leukemia lines. It assumes the subline matches its parent genetically. Overrides design §4 step 2 for this one line. **Needs code:** 1d has no proxy mechanism yet (see the 1d section) |
| 11 | MDA-MB-435 (really M14) and NCI/ADR-RES (really OVCAR-8): no DepMap IDs | **[decision, 2026-11-08] Drop both** from the pool (10,024 rows), per design §4 step 2. Their true identities stay in the pool. They stay in the cell table and family table, so the split logic and the data card still know the relation |
| 12 | P7: `study_id` | **[decision, 2026-11-08] One label for the whole pool**, `NCI-ALMANAC`. Decision 3's constant stands, and that's now true: under decision 13 the pool holds only ALMANAC rows |
| 13 | P8: O'Neil rows in the NCI-60 pool | **[decision, 2026-11-08] Remove them; plan A + C.** The pool stays on one scale. O'Neil's drugs come back through the second benchmark (OncoPolyPharmacology, all 39 O'Neil lines). The removed rows go to side tables (`oneil_rows.parquet`), and the **546 triplets both studies measured** become a reported finding (`cross_study_overlap.parquet`). How it was applied: the 1,076 rows that can't be attributed to a study (pairs of the 14 shared-name drugs on O'Neil's 9 lines) are removed too, and so are the 10 third-study rows on SN12C. See the 1e section |
| 14 | DepMap release | **[team, 2026-11-08] DepMap Public 26Q3** |
| 15 | Missing DepMap data for 1f | **[decision, 2026-11-08] Fill from NCI CellMiner first,** as much as possible; missing-value flags only where that's impossible. Applied with validation (depmap-data.md §5): CellMiner is the **single source of expression and copy number for all 57 lines**. Mutations come from DepMap. SR's CellMiner RNA failed the identity check, so SR's expression is flagged missing |
| 16 | SNB-19 and NCI-H322M (no DepMap sequencing) | **[decision, 2026-11-08] Use their confirmed parents.** SNB-19 is a U-251 MG derivative (STR; confirmed in our data: its closest CellMiner line is U-251 MG, and ALMANAC response r = 0.75). NCI-H322M is a variant of NCI-H322, which DepMap profiles as ACH-000837 (CellMiner H322M best-matches it out of 2,130 DepMap lines, Spearman 0.918). Driver mutations come from the parent's DepMap profile, as for HL-60(TB) → HL-60, flagged `mutation_source = depmap_parent`. Expression and copy number are each line's own CellMiner data |
| 17 | SR expression (CellMiner's SR RNA-seq sample is a mix-up) | **[team goal 2026-11-08: include as much data as possible]** **Impute** SR on the RNA-seq scale: CellMiner's own microarrays (which show SR's true ALK+ ALCL profile), translated per gene using the lines measured both ways; then calibrated DepMap SR-786; then the gene mean. Flagged `expression_imputed`. Leave-one-out r = 0.94, RMSE 0.61 (baseline 1.71). See [depmap-data.md §6](depmap-data.md) |
| 18 | Curated drug classes | **[team, 2026-11-08] Not needed.** Action type × target type stays |
| 19 | Which lines may be held out | **[proposed, 1g, 2026-11-08] Only lines whose features are all their own measurements.** That excludes SR (imputed expression), SNB-19 and NCI-H322M (parent mutations), HL-60(TB) (proxy), and U-251 MG (SNB-19's family). Held-out lines test predictions written over their features, so borrowed features would test the stand-in. See [m1-freeze-and-splits.md §3](m1-freeze-and-splits.md) |
| 20 | Synergy hits with negative killing (958, 7.1% of synergy hits) | **[team, 2026-11-08] Keep them as hits, flagged `syn_hit_low_kill`, and report every synergy result both with and without them** (from M3 on). Alternatives not taken: require `y_inh ≥ 0`; a different synergy score. See [m1-freeze-and-splits.md §4](m1-freeze-and-splits.md) |

## Verified data facts: TDC `drugcomb` (`drugcomb_nci60.pkl`, sha256 `c0e195c9…ad00bf`)

> **2026-11-08 audit.** See [drug-identity-audit.md](drug-identity-audit.md).
> - The file mixes NCI-ALMANAC, O'Neil (at least 4,294 rows, on 9 lines) and a 4-row third study.
> - 129 IDs = 122 molecules, in **7** cross-study synonym pairs.
> - The 546 triplets both studies measured barely agree (CSS r = 0.22).
> - This affects decision 3's constant `study_id` (proposal P7).
> - The salt-pair list below is incomplete: doxorubicin and oxaliplatin are missing.

- **Size:** 297,098 rows; 129 drug IDs; **59 cell lines**; no NaN in any score column.
- **Columns:** `Drug1_ID`/`Drug2_ID` are names; `Drug1`/`Drug2` are **SMILES**; `CellLine` is a
  list of 3 numeric arrays per line (TDC's own features, meaning undocumented). That column is
  dropped at conversion; cell features come from DepMap.
- **ID → SMILES:** each ID maps to exactly 1 SMILES. One SMILES is shared by 2 IDs: `ETOPOSIDE` and
  `NSC141540`.
- **Salt and synonym pairs seen in the IDs:**
  - ERLOTINIB / ERLOTINIB HYDROCHLORIDE
  - TOPOTECAN / TOPOTECAN HYDROCHLORIDE
  - VINBLASTINE / VINBLASTINE SULFATE
  - VINORELBINE / NAVELBINE DITARTRATE (TN)
  - ETOPOSIDE / NSC141540

  There may be more; the RDKit step will tell.
- **Self-combinations:** 115 rows (Drug1_ID == Drug2_ID).
- **Canonical triplets (A < B by ID):** 287,268. Of those, 9,830 have 2 rows, and **every one of
  them is the same pair in both orders** ((A,B) and (B,A)) with different scores, so they're true
  replicates. No triplet has more than 2 rows. The median Bliss SD within a replicate pair is 2.91.
- **Pairs:** 5,427 unique. Lines per pair: min 1, median 58, max 59.
- **Bliss (`y_syn`):** mean 0.09, sd 6.07. **p95 = 9.26**, p99 = 16.97. Bliss ≥ 10 covers 4.21% of
  triplets.
- **CSS (`y_inh`):** mean 23.1, sd 21.6. **p95 = 63.3**.
- **Hits (p95, computed on all lines, before the held-out split):** 14,364 per outcome. Only 2,143
  triplets are hits for both. **Spearman(Bliss, CSS) = 0.062**, so the two outcomes really are
  different questions.
- **Hit concentration:** synergy hits span 3,316 pairs, and the top 10 pairs hold only 531 of them,
  so line-sweeping isn't a big risk.
- **Random-selection baseline:** about 4 / 8 / 16 synergy hits expected at 80 / 160 / 320
  experiments.
- **Cell lines (59):** 786-0, A498, A549, ACHN, BT-549, CAKI-1, CCRF-CEM, COLO 205, DU-145, EKVX,
  HCC-2998, HCT-15, HCT116, HL-60(TB), HOP-62, HOP-92, HS 578T, HT29, IGROV1, K-562, KM12, LOX IMVI,
  M14, MALME-3M, MCF7, MDA-MB-231, MDA-MB-435, MOLT-4, NCI-H226, NCI-H322M, NCI-H460, NCI-H522,
  NCI/ADR-RES, NCIH23, OVCAR-4, OVCAR-5, OVCAR-8, OVCAR3, PC-3, RPMI-8226, RXF 393, SF-268, SF-295,
  SF-539, SK-MEL-2, SK-MEL-28, SK-MEL-5, SK-OV-3, SN12C, SNB-19, SNB-75, SR, SW-620, T-47D, TK-10,
  U251, UACC-257, UACC62, UO-31.

## Verified external sources (2026-11-08)

- **ChEMBL REST API** (`https://www.ebi.ac.uk/chembl/api/data`):
  - Lookup by InChIKey works: `molecule.json?molecule_structures__standard_inchi_key=<key>`.
    Erlotinib's key gives `CHEMBL553`.
  - Molecules carry a `molecule_hierarchy` (salt → parent): erlotinib HCl `CHEMBL1079742` → parent
    `CHEMBL553`.
  - **Gotcha:** mechanisms are attached to the salt form.
    `mechanism.json?molecule_chembl_id=CHEMBL553` returns **0**, but
    `?parent_molecule_chembl_id=CHEMBL553` returns EGFR (`CHEMBL203`, INHIBITOR). Always query
    mechanisms by `parent_molecule_chembl_id`.
- **Cellosaurus API** (`https://api.cellosaurus.org`):
  - `cell-line/<AC>?fields=…&format=txt` works.
  - Fields: `DR DepMap; ACH-…`, `DI` (NCIt disease), `HI` (parent line), `CA` (category).
  - **Name search ranks derived and edited lines first.** For example, "A549" returns a USP24-edit
    line. So the mapper must pick an exact match on the ID or a synonym after normalizing names.
  - **NCI-ADR-RES** (`CVCL_1452`): "Problematic cell line: Contaminated. Shown to be a OVCAR-8
    derivative." Its parent is `CVCL_1629` (OVCAR-8, `ACH-000696`).
  - **MDA-MB-435** (`CVCL_0417`): disease is amelanotic melanoma; its parent is `CVCL_1395`
    (**M14**, also in our 59 lines). **No `DR DepMap` line**, so by design §4 step 2 it would be
    dropped as unmapped. [verified for the entry as served on 2026-11-08]
  - A-549 (`CVCL_0023`) → `ACH-000681`.
- **DepMap download API** (`https://depmap.org/portal/api/download/files`): returns a **Cloudflare
  bot-verification page** to scripts. DepMap mutation and expression files (for step 1f) will need a
  **manual browser download**. DepMap model IDs can come from Cellosaurus's `DR DepMap`.

## Environment and ingest gotchas found while making 1a work (verified 2026-11-08)

1. **`pip install PyTDC` fails on Python 3.13.** PyTDC needs `numpy<2`. The last numpy 1.x (1.26.4)
   only ships wheels for cp39–cp312, so pip compiles it from source, and that fails
   (`metadata-generation-failed`). Don't install PyTDC; `ingest_tdc` replaces it.
2. **The `onc` venv was still active** (`VIRTUAL_ENV=~/projects/onc/.venv`). A bare `pip` in that
   shell targets `onc`, not this repo. The project `.venv` (made by `uv sync`) has **no pip by
   design**; use `uv add` / `uv run`. Run `deactivate` first.
3. **Dataverse returns 403 to urllib's default User-Agent** (`Python-urllib/3.13`). A named UA gets
   206/200. Fixed: `ingest_tdc.USER_AGENT`.
4. **Truncated transfers happen.** One body arrived 36,372,455 of 36,394,069 bytes, and the read loop
   ended without an error. The first symptom was a sha256 mismatch. Fixed: a `Content-Length` check,
   plus up to 3 retries with backoff (`IncompleteDownload`). A complete body with the wrong hash is
   *not* retried.
5. **pyarrow 26 refuses numpy<2** ("pyarrow requires NumPy 2.0 or newer, found 1.26.4"). The
   conversion environment is pinned exactly to `pandas==2.3.3, numpy==1.26.4, pyarrow==25.0.1`
   (a known-good set).
7. `uv` warns that `build-system.requires = uv_build>=0.9.2,<0.10.0` doesn't include uv 0.13.7.
   This is pre-existing. The build works and it's left unchanged (ask before bumping).

## Status of the pieces

| Piece | Status |
|:--|:--|
| M0 `.gitignore` (`data_cache/`, `runs/`, `.cache/`, `.pytest_cache/`) | ✅ edited. **Also fixed:** the old pattern `data/` ignored *every* directory named `data`, including the `src/combo_agent/data/` package (confirmed with `git check-ignore`). Root-level patterns are now anchored (`/data/`, `/data_cache/`, …) |
| M0 dependencies | ✅ `uv add numpy pyarrow rdkit` (rdkit 2026.9.1, pyarrow 26.0.1) + `uv add --dev pytest` (9.2.0). `pyproject.toml` and `uv.lock` updated |
| 1a `src/combo_agent/data/{paths,ingest_tdc,_tdc_pickle_convert}.py` + `tests/test_ingest_tdc.py` | ✅ **Done 2026-11-08: 14/14 tests pass, including the real end-to-end run** (`COMBO_INTEGRATION=1`). The raw file was verified and recorded in `data_cache/raw/manifest.json`. `data_cache/interim/tdc_drugcomb.parquet` (12,078,964 B, sha256 `7949cbf8…c629ab`) is **byte-identical across two independent conversions** |
| 1b profile → data-card numbers | ✅ numbers above |
| 1c harmonize drugs (ChEMBL parent ID, name and structure must agree, targets) | ✅ **Done 2026-11-08** under audit P1–P5: 129 labels → 122 drugs, 115 with targets, 13 build checks, 74/74 tests, byte-reproducible offline. See [the 1c section below](#step-1c-drug-harmonization-done-2026-11-08). P6–P9 move to 1e (P7/P8 still need decisions) |
| 1d harmonize lines (Cellosaurus exact match, DepMap id, origin groups) | ✅ **Built 2026-11-08**: 59 labels → 59 Cellosaurus lines, all exact matches within the NCI-60 panel; 3 families; 54 kept by the current rules. **Decisions open:** SR and UO-31 each have 2 DepMap IDs, and the 3 lines without one. See [the 1d section below](#step-1d-cell-line-harmonization-built-2026-11-08) |
| 1e canonicalize, merge replicates, self-combo table, `study_id` | ✅ **Built 2026-11-08**: 297,098 rows → **272,357 experiments** (4,939 pairs × 57 lines, 101 drugs), with every row assigned exactly one fate. 15 build checks; tests pass; byte-reproducible. See [the 1e section below](#step-1e-one-row-per-experiment-built-2026-11-08) |
| 1f features (Morgan FP, targets/MoA, DepMap lineage/mutations/expression) | ✅ **Built 2026-11-08**. 57 lines × 1,073 cell feature columns and 122 drugs × 2,335 drug columns, with no NaN. SR's expression is imputed (decision 17). 9 build checks; tests pass; byte-reproducible. See [the 1f section below](#step-1f-features-built-2026-11-08) and [depmap-data.md](depmap-data.md) |
| 1g freeze: split, thresholds, `pool`/`labels`/`drugs`/`cells` parquet, `data_card.md` + `manifest.json` | ✅ **Built 2026-11-08**. 11 held-out lines (decision 19); thresholds y_syn ≥ 9.30 and y_inh ≥ 62.95 (selectable); `syn_hit_low_kill` flagged (decision 20, proposed). 12 acceptance checks, all passing; byte-reproducible. **Full write-up with figures: [m1-freeze-and-splits.md](m1-freeze-and-splits.md)** |

## Step 1c: drug harmonization (done 2026-11-08)

**Commands** (from the code repo root):
`python -m combo_agent.data.harmonize_drugs propose [--write]`, then `… build [--offline]`.
Tests: `COMBO_INTEGRATION=1 COMBO_OFFLINE=1 .venv/bin/python -m pytest tests`.

### How it works

1. **Keys.** `structure.py` computes a **connectivity key** (the first 14 characters of the InChIKey
   of the largest fragment, uncharged) for each of the 129 source SMILES. Metal complexes get a
   **heavy-atom formula** instead.
2. **Candidates.** `propose` collects ChEMBL candidates for each label through four routes:
   - the label is itself a ChEMBL ID;
   - an exact ChEMBL preferred-name or synonym match on name variants (salt words, brand and
     supplier parentheticals, NSC spellings);
   - PubChem name → stereo InChIKey → ChEMBL;
   - all ChEMBL structures that share the skeleton.

   Every candidate is mapped to its ChEMBL parent and **accepted only if the parent's structure
   agrees** with the source.
3. **Review.** The result is the committed **`curated/drug_synonyms.csv`** (P5). Rows marked
   `manual` survive a re-proposal and must carry a note.
4. **Build.** `build` reads the CSV, never the live resolver, and re-checks it against ChEMBL. It
   then attaches the ChEMBL parent SMILES with stereo (P4) and the mechanism targets (queried by
   `parent_molecule_chembl_id`).
5. **Outputs** (`data_cache/interim/`):
   - `drug_map.parquet`: 129 rows, one per label;
   - `drugs.parquet`: 122 rows, one per drug;
   - `drug_targets.parquet`: 243 rows;
   - `drug_harmonization_report.json`.
6. **Cache.** Every API response is cached in `data_cache/raw/api_cache/`: 12 MB, 1,225 ChEMBL
   files and 129 PubChem files. The cached build is ChEMBL_37, released 2026-05-01.

### Checks (any failure blocks the outputs)

| Check | What it guarantees |
|:--|:--|
| `coverage` | Every label is in the CSV exactly once |
| `row_vocabulary` | Resolution value is known; resolved rows have an ID; manual rows have a note |
| `chembl_id_is_parent` | The ID is a parent, not a salt record |
| `structure_agrees` | Name and structure agree (connectivity / tautomer / formula); `unverifiable` is allowed only on manual rows whose ChEMBL record has no structure |
| `grouping_matches_structure` | The ChEMBL grouping equals the independent RDKit grouping (no stereo collision, no split) |
| `no_approved_stereo_sibling` | No automatic row sits on a lower-phase stereoisomer of a named approved drug |
| `preferred_name_from_chembl` | One preferred name per drug, and it is ChEMBL's |
| `curated_matches_resolver` | The committed CSV equals today's automatic result, except manual rows. Catches drift in the code or in ChEMBL |
| `mechanism_proxies_valid` | A proxy only fills a gap, names a ChEMBL parent with a targeted mechanism, and has a known relation and a note |
| `every_label_in_one_drug` | No label is lost or duplicated in the output tables |
| `parent_smiles_single_fragment` | Feature SMILES are salt-free |

Warnings (reported, not failing): `rows_needing_review` (`MK-4541`), `drugs_without_targets` (7),
and `parent_stereo_from_chembl` (an informational list of racemates).

The tests cover:
- every resolver branch, and every check's failure mode, against a fake ChEMBL;
- the cache's offline replay and retries;
- on the real data: the audit's 7 groups and 122 molecules; the Pt drugs passing by formula; known
  targets (EGFR, ERBB2, DHFR, PSMB5, ABL1, BRAF, HDAC1, BCL2, FKBP1A, MTOR); the 16 obscure-label
  identities from the audit; the exact set of drugs without targets; byte reproducibility.

### Results [verified 2026-11-08]

- **Resolutions:** 120 `name+structure`, 8 `manual`, 1 `unresolved`.
- **Structure checks:** 123 connectivity, 4 formula (the Pt drugs), 1 unverifiable (bleomycin),
  1 none (MK-4541).
- **The 8 manual rows** (each note in the CSV gives the evidence):

| Label | → ChEMBL parent | Why manual |
|:--|:--|:--|
| capecitabine's IUPAC-style label | CHEMBL1773 CAPECITABINE | No name route; 2 parents share the skeleton |
| `BLEOMYCIN` | CHEMBL6067492 BLEOMYCIN | ChEMBL's record is a mixture with no structure, so it's unverifiable; 4 unnamed structure matches have no mechanism |
| `NCGC00090797-11` | CHEMBL1751 DIGOXIN | NCATS sample ID. **Caveat:** PubChem's only record for it has a stereo assignment different from digoxin's; unverified. 4 rows, third study |
| `34793-34-5` | CHEMBL1201112 NELARABINE | Stereo-sibling fix (see below) |
| `5-FLUORO-2'-DEOXYURIDINE` | CHEMBL917 FLOXURIDINE | Stereo-sibling fix |
| `CHEMBL17639` | CHEMBL1619 CLADRIBINE | Stereo-sibling fix: DrugComb's own ChEMBL ID is an unnamed stereoisomer |
| `CHEMBL277800` | CHEMBL1750 CLOFARABINE | Stereo-sibling fix: same |
| `CO-V` | CHEMBL1580 PENTOSTATIN | Stereo-sibling fix ("covidarabine") |

- **Stereo-sibling fixes.** For these 5 nucleosides, the automatic route matched a ChEMBL record
  that agrees with the flat structure but is an **unnamed stereoisomer with no mechanism**. The
  approved drug was chosen because NCI-ALMANAC screened approved drugs. **[The ALMANAC drug list
  itself was not re-checked.]**
- **Unresolved:** `MK-4541` (PubChem CID 59691338) isn't in ChEMBL_37. It is kept as
  `UNRESOLVED:MK-4541`, with the structure from the source.
- **Mechanism proxies (2):**
  - fludarabine (CHEMBL1568) → fludarabine phosphate (CHEMBL1096882), its prodrug, a separate
    parent;
  - plicamycin, where the source structure (CHEMBL257619) differs in connectivity from ChEMBL's
    approved PLICAMYCIN (CHEMBL1237054) but has the same formula.

  Target rows record `mechanism_source`.
- **No target (7, dropped at freeze unless hand-annotated):** amifostine (its ChEMBL mechanism has
  no target), MRK-003, LLL-12, geldanamycin, triethylenemelamine, quinacrine, MK-4541.
- **Stereo:**
  - 0 of 129 source SMILES carry stereo.
  - 67 of 117 ChEMBL parent SMILES do; the other 50 are achiral or racemic.
  - 11 parents leave one centre unassigned. These are racemates such as ifosfamide,
    cyclophosphamide, thalidomide and lenalidomide, plus single undefined centres in vinca alkaloids,
    fulvestrant and pralatrexate.
  - The 4 Pt drugs keep the source SMILES (`smiles_origin = source`), because ChEMBL has none.

### ChEMBL / PubChem gotchas found while building 1c [verified 2026-11-08]

1. **Tautomer canonicalization breaks the key match.** The InChI main layer includes mobile H, and
   RDKit's canonical tautomer differs from ChEMBL's form for 19 drugs (doxorubicin, paclitaxel,
   docetaxel, …). It is used only as a fallback comparison. The audit script's grouping still holds,
   because it compares source with source.
2. **Pt drugs have `molecule_structures: null` in ChEMBL**, but `full_molformula` is present.
3. **`max_phase` is a string** (`"4.0"`) in ChEMBL_37.
4. **Transient errors.** A slow synonym query returned HTTP 500 and then succeeded. Large bodies
   (about 180 kB, e.g. protein-family targets) are cut short about half the time from this network.
   The client retries up to 8 times, with backoff capped at 60 s.
5. **Prodrugs are separate parents.** A phosphate ester is not a salt, so
   `parent_molecule_chembl_id` doesn't reach fludarabine phosphate's mechanisms.
6. **Rapalogs are filed under FKBP1A**, not MTOR (sirolimus, everolimus). Ridaforolimus is under
   MTOR. The agent must reason at pathway level.
7. **DrugComb's own ChEMBL-ID labels can be wrong stereoisomers** (`CHEMBL17639`, `CHEMBL277800`).

### Open items from 1c

- **Drugs without targets:** 17,260 of 297,098 rows (5.8%) involve one of the 7. Per drug: quinacrine 5,757, amifostine 5,743, triethylenemelamine 4,978, MRK-003 324, geldanamycin 324, MK-4541 324, LLL-12 4. A row is counted once even if both drugs lack targets; the counts are from `drugs.parquet` `n_rows`. **Decided:** keep them (decision 8, below).
- **Review:** the 8 manual rows and the 2 mechanism proxies.
- **Unverified:** `NCGC00090797-11` → digoxin (stereo mismatch in PubChem), and the assumption that ALMANAC screened the approved stereoisomers.
- **Moved to 1e:** P6–P9. P7 and P8 need decisions.

The conceptual explanation is in [m1-walkthrough.md §4](m1-walkthrough.md#4-step-1c-one-drug-one-identity-done-2026-11-08).

### Decision 8: keep drugs without a ChEMBL target

**[decision, 2026-11-08]** The 7 drugs that ChEMBL gives no target stay in the pool with all their
data. They are amifostine, quinacrine, triethylenemelamine, geldanamycin, MRK-003, MK-4541 and
LLL-12. At run time the agent searches the literature for their possible targets and functions.

**What "no target" means.** ChEMBL's hand-curated mechanism table has no row naming a target for
these drugs. Amifostine has a mechanism row ("Reactive metabolite scavenging agent") with no
target; the other 6 have no mechanism row. That is a gap in one database, not evidence that the
drugs lack a mechanism.

| Drug | Rows | Study footprint | What it is (background) |
|:--|--:|:--|:--|
| Quinacrine | 5,757 | 59 lines (ALMANAC) | Old antimalarial; DNA intercalator with many reported cellular effects |
| Amifostine | 5,743 | 59 lines (ALMANAC) | Cytoprotective prodrug (active thiol WR-1065), a free-radical scavenger |
| Triethylenemelamine | 4,978 | 59 lines (ALMANAC) | Old aziridine alkylating agent; cross-links DNA |
| Geldanamycin | 324 | 9 lines (O'Neil) | Natural-product HSP90 inhibitor |
| MRK-003 | 324 | 9 lines (O'Neil) | Gamma-secretase (Notch) inhibitor |
| MK-4541 | 324 | 9 lines (O'Neil) | Selective androgen receptor modulator (PMIDs below) |
| LLL-12 | 4 | 1 line (third study) | STAT3 inhibitor |

- **From our data and ChEMBL:** the row counts, the study footprints, and amifostine's mechanism
  row.
- **From PubMed:** MK-4541's function. For MRK-003, LLL-12, triethylenemelamine and geldanamycin,
  the PubMed queries below return hits for the stated mechanism, but the abstracts weren't read.
- **Textbook background, not checked here:** the rest.

If P8 removes the O'Neil rows, the 3 O'Neil drugs leave the NCI-60 pool anyway.

**Is it possible? Yes.** The literature exists, and the design already has the tool.
- **Literature exists for all 7** [verified 2026-11-08, PubMed hit counts for one query each]:

  | Query | Hits |
  |:--|--:|
  | `MK-4541` | 2 |
  | `MRK-003` + gamma-secretase | 15 |
  | `LLL12` + STAT3 | 37 |
  | triethylenemelamine + DNA | 32 |
  | quinacrine + cancer + mechanism | 87 |
  | amifostine + WR-1065 + mechanism | 29 |
  | geldanamycin + HSP90 + binding | 590 |

  Both MK-4541 papers describe it as a selective androgen receptor modulator (SARM):
  - PMID 24565564, *J Steroid Biochem Mol Biol* 2014, doi:10.1016/j.jsbmb.2014.02.005;
  - PMID 27106747, *J Steroid Biochem Mol Biol* 2016, doi:10.1016/j.jsbmb.2016.04.007.

  Neither is a combination screen, so the leakage blocklist wouldn't remove them.
- **The design's agent already has the tools** (`target_design.md` §8): `get_drug(drug_id)`, for
  targets/MoA/class, and `search_literature(query)`, abstracts after the leakage filter.
- **Prerequisite:** on the default Gemini endpoint, SHELDON drops `web_search`/`web_fetch`
  (SHELDON drops these server-side tools on OpenAI-compatible endpoints). Literature has to come from the client-side probe planned for M5
  (PubMed / Europe PMC). This was already required, so it adds no new work.

**How each milestone handles it** (a proposal for later milestones; not built yet):

| Where | What changes |
|:--|:--|
| 1g freeze | Keep all 122 drugs. `drugs.parquet` gets `target_status` ∈ {`chembl`, `chembl_proxy`, `none_in_chembl`}. The data card lists the 7 and their row counts |
| M3 response model | Target features are empty, marked "unknown", for these 7. They still get their structure fingerprint. **No literature-derived targets enter the model's frozen features**, so the model's inputs stay fixed and reproducible |
| M5 retrieval | Single-drug mechanism queries go through the same blocklist and audit log as every other query. Cache the results per drug, so every run sees the same evidence |
| M6 agent | `get_drug` returns `targets: []` and `target_status: "none_in_chembl"`, plus a hint to search. Targets the agent finds are cited evidence (PMIDs) attached to its hypotheses. They are **not** written back into `drugs.parquet` |
| M8 evaluation | Report hits and explanation accuracy separately for pairs that involve these drugs: literature-derived mechanisms vs ChEMBL-curated ones |

**Caveats to keep in mind:**
- **Masked-name ablation** (design §10) replaces names with codes and keeps targets and MoA. These
  7 then carry no mechanism information at all, and a code can't be searched. Report that this
  ablation is uninformative for them.
- **Some "functions" aren't tumor targets.** Amifostine protects normal tissue from chemotherapy
  and radiation. Quinacrine acts on many things at once. The agent should treat its findings here
  as uncertain, and the explanation level should reflect that.
- **Agent-found targets are unverified.** They are the agent's reading of the literature, not
  curated facts. The audit log keeps the sources, so each can be checked after a run.

## Step 1d: cell-line harmonization (built 2026-11-08)

**Code** (new files only; no shared file was edited):

| File | Purpose |
|:--|:--|
| `src/combo_agent/data/cellosaurus.py` | Cellosaurus API client and entry accessors |
| `src/combo_agent/data/harmonize_lines.py` | The `propose` / `build` CLI |
| `src/combo_agent/data/curated/cell_lines.csv` | 59 rows, reviewable |
| `tests/test_harmonize_lines.py` | Unit tests plus an opt-in real-data test |

1d uses `api_cache.CachedJsonClient` through its public parameters only. Responses are cached in
`data_cache/raw/api_cache/cellosaurus/` (83 files).

```bash
python -m combo_agent.data.harmonize_lines propose [--write] [--offline]   # regenerate curated/cell_lines.csv
python -m combo_agent.data.harmonize_lines build [--offline]               # writes data_cache/interim/{line_map,cell_lines}.parquet + line_harmonization_report.json
```

**Method.**
1. Normalize each label: upper-case, keep only letters and digits.
2. Match it exactly against the identifier and synonyms of the **61 entries Cellosaurus tags "Part
   of: NCI-60 cancer cell line panel"**. A plain name search fails for popular lines: "A549", "MCF7"
   and "HCT116" each return 100+ derived or edited lines, and the parent falls off the first page.
3. Fetch each matched entry and walk its `derived-from` links to collect ancestors.
4. Group lines that share any accession (themselves or an ancestor) into families with union-find.
   The family ID is the smallest member accession.
5. Take the DepMap ID from Cellosaurus' DepMap cross-reference.

**Results** [verified 2026-11-08]:

- **59 labels → 59 distinct lines.** All 59 resolve within the NCI-60 panel (56 by identifier, 3 by
  synonym), and no two labels share a line. The 2 panel members not in our data are MDA-N (excluded
  by TDC) and MDA-MB-468.
- **3 families with more than one line.** The held-out split must keep each family on one side:
  - **NCI/ADR-RES + OVCAR-8** (contaminated; actually an OVCAR-8 derivative);
  - **MDA-MB-435 + M14** (contaminated; actually an M14 derivative, melanoma);
  - **SNB-19 + U-251MG** (contaminated; actually a U-251MG derivative). **New:** this pair wasn't in
    the earlier notes, and both lines are in our 59.
- **Other problematic-line notes:** OVCAR-5 is possibly misclassified (suspected upper-GI origin), and
  SF-268 is possibly misidentified (a Y chromosome in a supposedly female line). Neither forms a
  family, but both belong in the data card.
- **Drops under the current rules** (design §4 step 2): 5 lines, 24,623 of 297,098 rows (8.3%). Choosing DepMap IDs for SR and UO-31 would cut that to 3 lines and 14,758 rows.

| Line | Rows | Reason | Note |
|:--|--:|:--|:--|
| SR | 4,900 | `multiple_depmap_ids`: ACH-000338, ACH-001665 | **Decision 9: keep; choose the ID in 1f.** Set `depmap_id` in `cell_lines.csv`; `build` checks it's one of the listed IDs |
| UO-31 | 4,965 | `multiple_depmap_ids`: ACH-000428, ACH-002203 | **Decision 9:** same as SR |
| HL-60(TB) | 4,734 | `no_depmap_id` | A subline of HL-60. **Decision 10: keep, using HL-60's profile (`ACH-000002`) as a proxy** |
| MDA-MB-435 | 4,940 | `no_depmap_id` | A contaminated M14 derivative. **Decision 11: drop**; M14 stays |
| NCI/ADR-RES | 5,084 | `no_depmap_id` | A contaminated OVCAR-8 derivative. **Decision 11: drop**; OVCAR-8 stays |

**After decisions 9–11 (applied 2026-11-08):** `curated/cell_lines.csv` now has the columns `depmap_id` (SR → ACH-000338, UO-31 → ACH-000428, both confirmed against DepMap `Model.csv`; see [depmap-data.md](depmap-data.md)) and `depmap_proxy_of` (HL-60(TB) → CVCL_0002, i.e. ACH-000002, flagged `depmap_is_proxy`). `build` checks that a proxy is a real ancestor and has exactly one DepMap ID. The 1d build now keeps 57 lines.

**Before they were applied:**
- The pool will have **57 lines** (only MDA-MB-435 and NCI/ADR-RES dropped, 10,024 rows), once 1f
  chooses the IDs for SR and UO-31.
- **Code still needed** (not built yet):
  - (a) A proxy mechanism for HL-60(TB). Proposal: a `depmap_proxy_of` column in
    `curated/cell_lines.csv` naming the Cellosaurus parent. `build` would check that the parent is
    an ancestor in the `derived-from` graph, and output `depmap_id = ACH-000002`,
    `depmap_is_proxy = True`.
  - (b) The `depmap_id` values for SR and UO-31, filled in during 1f.

**What the dropped rows looked like** (before decisions 9–11, all 5 lines) [verified 2026-11-08]:
- Whole lines drop, so no drug pair is lost entirely (0 of 5,427). The median number of lines per
  pair falls from 58 to 53.
- The rows are **not random**:
  - 2 of the 5 are blood cancers (SR, HL-60(TB)), and blood lines die easily: mean CSS 46.6 and
    38.9, against 13–17 for the others.
  - So the CSS hit rate was 12.3% in the dropped rows against 4.4% in the kept rows. For Bliss it
    was 6.2% against 5.1%.
  - Decisions 9–10 keep both blood lines. The data card should still report the hit rates of the
    lines removed.
- **Suspicious rows:** several of the largest Bliss values (up to 78.6) come with strongly
  *negative* CSS, e.g. estramustine + erlotinib on HL-60(TB): Bliss 78.6, CSS −48.7. That looks
  like a synergy-fitting artifact on flat or noisy curves. Check for this before thresholds are
  frozen (1g).

**Gotchas** [verified 2026-11-08]:

1. **Large Cellosaurus responses are cut short in transit** (`IncompleteRead`), sometimes 4 times in a
   row. Two fixes:
   - Request only the needed fields (`fields=id,ac,sy,ca,cc,ox,hi,oi,dr,di,sx`). That shrinks an
     entry from ~200 kB to ~35 kB, and the panel search to ~74 kB.
   - Allow 8 attempts. Successes are cached, so a rerun resumes.
2. **There's no `fields` code for misspellings or sampling site,** so those aren't fetched. Site and
   lineage come from DepMap in 1f.
3. **Lines from the same individual are linked through a placeholder parent entry** (e.g. HCT 15 →
   `CVCL_3449` "__Parent_cell_line_of_DLD-1/HCT 8/HCT 15/HRT-18"). The ancestor walk covers this.
   None of our 59 share one.
4. **`786-0` (digit zero) matches `786-O` (letter O)** only because Cellosaurus lists `786-0` as a
   synonym.

**Verification:**
- 72 passed, 20 skipped across `tests/`, including the 1c tests.
- With `COMBO_INTEGRATION=1 COMBO_OFFLINE=1`, the real 1d build runs twice from the cache and gives
  byte-identical outputs.

**Not done here (1e/1g):** removing dropped lines from the pool, lineage stratification, and the
held-out split itself.

## Step 1e: one row per experiment (built 2026-11-08)

**Code** (new):
- `src/combo_agent/data/build_experiments.py`;
- `tests/test_build_experiments.py`: 9 unit tests on a small synthetic world, plus an opt-in
  real-data test.

Run `python -m combo_agent.data.build_experiments`. Its inputs are the 1a parquet, 1c's
`drug_map.parquet` and 1d's `line_map.parquet`.

**What it does.** Every one of the 297,098 source rows gets exactly one *fate*. The first rule that
matches wins:

| Fate | Rows | Rule | Why |
|:--|--:|:--|:--|
| `third_study` | 10 | both drugs in the 5-drug third study (all on SN12C) | Not NCI-ALMANAC (decision 13) |
| `oneil` | 4,284 | a drug label only O'Neil uses: its 16 own drugs, or its generic names for the 7 drugs ALMANAC lists under salt or brand names | Decision 13 |
| `oneil_ambiguous` | 1,076 | both drugs among the 14 shared-name drugs, on one of O'Neil's 9 lines | Either study could have measured these, and nothing in the file says which. Removing them keeps the pool on one scale. Each affected pair is still measured on about 48 other lines |
| `line_dropped` | 10,024 | the line isn't kept by 1d | Decision 11 (MDA-MB-435, NCI/ADR-RES) |
| `self_pair` | 111 | both labels are the same drug | Decision 5; goes to `self_pairs.parquet` |
| `kept` | 281,593 | everything else | The NCI-ALMANAC pool |

The 14 shared-name drugs aren't hard-coded. They're **derived** as the partners of O'Neil-only drugs
in O'Neil rows, and they come out as exactly 5-FU, bortezomib, carboplatin, cyclophosphamide,
dasatinib, lapatinib, methotrexate, mitomycin, paclitaxel, SN-38, sorafenib, sunitinib,
temozolomide and vorinostat.

**Guards.** The build stops if any of these fails:
- an O'Neil row lies off the 9 O'Neil lines;
- a third-study drug pairs with an outsider;
- any other drug label is confined to the O'Neil lines (that would mean an unknown study);
- a label is unmapped;
- an experiment key is duplicated;
- rows aren't conserved (kept rows ≠ Σ `n_reps`);
- an O'Neil-only drug is in the pool.

There are 15 checks in all, recorded in `experiments_report.json`.

**Replicates (P6, P9).** Kept rows are mapped to `drug_id` and `cellosaurus_ac`, the pair is ordered
so `drug_a < drug_b`, and same-key rows are averaged (mean, sample SD, `n_reps`):
- 263,121 experiments have 1 measurement and **9,236 have 2**. These are ALMANAC's pairs stored in
  both drug orders; they're all in ALMANAC now that O'Neil rows are gone.
- Agreement between the two copies (Pearson): **CSS 0.575, Loewe 0.385, ZIP 0.328, HSA 0.249,
  Bliss 0.209**.
- So **Bliss, our `y_syn`, is the noisiest score in ALMANAC's own replicates.** That matters for hit
  thresholds (1g) and for the response model's noise level.

**Result** [verified 2026-11-08]:
- **272,357 experiments**: 4,939 pairs × 57 lines, 101 drugs (122 − 16 O'Neil-only − 5 third-study),
  `study_id = NCI-ALMANAC`.
- A pair is measured on a median of 56 lines.
- HL-60(TB) is in the pool as a proxy line (`depmap_is_proxy = True`).
- 111 self-pair experiments.

**Plan C side tables:**
- `oneil_rows.parquet`: 5,360 rows (`oneil` + `oneil_ambiguous`), mapped to drug and line IDs.
- `cross_study_overlap.parquet`: **546 triplets** with both studies' means. This independently
  reproduces the audit: CSS Pearson 0.217 and Spearman 0.22; mean CSS 29.4 (ALMANAC) against 56.4
  (O'Neil); mean |difference| 30.3.

**Outputs** in `data_cache/interim/`:
- `experiments.parquet`, with columns `drug_a, drug_b, cellosaurus_ac, study_id, n_reps`,
  `{css,bliss,zip,loewe,hsa}_{mean,sd}`, `y_syn, y_inh, depmap_id, depmap_is_proxy, family_id`;
- `self_pairs.parquet`, `row_fate.parquet`, `oneil_rows.parquet`, `cross_study_overlap.parquet`,
  `experiments_report.json`.

All are byte-identical across two runs.

**Bug caught while building:** the first replicate-agreement numbers were about 0. Pandas'
`groupby().nth()` returns rows in source order, so the first and second copies weren't aligned
by key before correlating. It now uses `groupby().first()/.last()`, which are key-aligned. The
outputs themselves were never affected.

**Not done here (1g):** splitting into `pool.parquet` and `labels.parquet`, the held-out lines
(families on one side, lineage-stratified), hit thresholds, and the suspicious large-Bliss /
negative-CSS rows.

## Step 1f: features (built 2026-11-08)

**Code** (new):
- `ingest_cellminer.py` and `_cellminer_convert.py`: sha256-pinned CellMiner zips; each `.xls` is
  converted in an isolated `uv` environment with xlrd. No project dependency was added.
- `build_features.py`.
- `tests/test_features.py`: 10 unit tests plus an opt-in real-data test.

**Sources and validation:** see [depmap-data.md](depmap-data.md) §4–5 (decisions 14–16).

**Cell features** (`data_cache/interim/features/cell_features.parquet`, 57 × 1,073, which includes
the flags `mutation_from_parent` and `expression_imputed`):

| Block | Columns | Source |
|:--|--:|:--|
| Flags: `expression_missing`, `mutation_from_parent`, `mutation_from_cellminer`, `depmap_is_proxy` | 4 | |
| Lineage one-hot (11 lineages) | 11 | DepMap `Model.csv` |
| Driver mutations, 0/1: 10 oncogenes (hotspot) + 14 tumour suppressors (LoF / high impact; TP53 also missense) | 24 | DepMap own (54 lines) or parent (3 lines) |
| Copy number of the drivers plus MYC, CCND1, CDK4, MDM2, MET, CDK6, FGFR1 (log2 ratio; 0 = 2N) | 31 | CellMiner |
| `cn_missing_<gene>` | 1 | CellMiner (one line misses one feature gene) |
| Expression, the 1,000 most variable genes across the 56 lines with trusted RNA | 1,000 | CellMiner, log2(FPKM+1); SR = 0 with `expression_missing = 1` |

**Spot checks against known biology:**
- Mutations: A549 KRAS/STK11/KEAP1; HT-29 BRAF/TP53/APC; MCF-7 PIK3CA; HCT 116 KRAS/PIK3CA;
  NCI-H460 KRAS/PIK3CA/STK11; U-251 MG TP53/PTEN.
- Copy number: PC-3 PTEN deletion (−1.46), A549 CDKN2A (−2.54), U-251 MG CDKN2A (−2.28),
  SK-OV-3 ERBB2 amplification (+1.35).
- Lineage counts: Lung 8, Skin 8, Kidney 8, Bowel 7, CNS 6, Ovary 6, Breast 5, Lymphoid 4,
  Myeloid 2, Prostate 2, Pleura 1.

**Drug features** (`drug_features.parquet`, 122 × 2,334; `in_pool` = 101):
- a Morgan fingerprint (r = 2, 2,048 bits) from 1c's SMILES;
- multi-hot target genes, action types and target types;
- the flags `has_target` (any ChEMBL target, DNA included), `has_target_gene` and
  `smiles_from_source`.

What those flags show in the pool:
- **Pool drugs with no target at all: 3** (amifostine, quinacrine, triethylenemelamine).
- **27 pool drugs target DNA** (alkylators and Pt drugs), so they have no target gene.
- The 3 Pt drugs' fingerprints come from their source SMILES, because ChEMBL has no structure for
  them.

The design's "MoA class" is represented by action type × target type. **[team, 2026-11-08] No
curated drug classes are needed.**

**Other outputs:**
- `cells.parquet`: identity, lineage, the source of each data type, flags, and the identity-check
  Spearman.
- `cell_mutations.parquet`, plus `cell_mutations_cellminer.parquet` for comparison.
- `cell_expression.parquet` and `cell_copy_number.parquet`: all genes, for agent tools and rule
  checks.
- `expression_identity.parquet` and `features_report.json`.

**Gotchas** [verified 2026-11-08]:
1. **CellMiner's files are real legacy `.xls`** (OLE2), with 10 metadata rows above the header. Line
   columns are named `<panel>:<line>`, e.g. `LC:A549/ATCC` (strip `/ATCC`).
2. **`UV_OFFLINE=1` breaks the isolated converter** when a pinned wheel isn't in the cache yet.
3. **CellMiner "protein function affecting" misses many tumour-suppressor mutations** (TP53, STK11,
   PTEN). Don't use it as the primary mutation source.
4. **Gene-set Jaccard between the two exome pipelines is a poor identity test.** Use expression.

**SR expression (decision 17):**
- **Imputed** from CellMiner's own microarrays, translated per gene onto the RNA-seq scale, with
  calibrated DepMap and then the gene mean as fallbacks.
- The cell matrix gains an `expression_imputed` flag (1,073 columns in total).
- Leave-one-out quality: r = 0.94, RMSE 0.61 (no-information baseline 1.71).
- Full explanation: [depmap-data.md §6](depmap-data.md).

**Open:**
- Whether the top-1,000 expression genes should later be chosen on selectable lines only, once 1g
  makes the split. That uses no labels either way.
