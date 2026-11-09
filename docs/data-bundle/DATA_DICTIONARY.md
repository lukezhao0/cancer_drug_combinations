# Data dictionary

Generated from the files in this bundle by `scripts/make_data_bundle.py`. Every parquet file and column is listed. Wide tables group their feature columns by prefix.

**Join keys:** experiments join to drugs on `drug_a` / `drug_b` = `drug_id`, and to cell lines on `cellosaurus_ac`. `pool` and `labels` share the key (drug_a, drug_b, cellosaurus_ac) and the same row order.

## `data_cache/processed/`

### `processed/pool.parquet`

One row per experiment (drug A, drug B, cell line). **No outcomes.** Key: (drug_a, drug_b, cellosaurus_ac).

272,357 rows × 10 columns.

| Column | Type | Meaning |
|:--|:--|:--|
| `drug_a` | string | first drug's ID (ChEMBL parent ID, e.g. CHEMBL553); always `drug_a < drug_b` |
| `drug_b` | string | second drug's ID (ChEMBL parent ID) |
| `cellosaurus_ac` | string | cell-line ID (Cellosaurus accession, e.g. CVCL_0023 = A-549); the cell-line join key |
| `split` | string | `selectable` (usable by selectors and models) or `heldout` (only for the final mechanism check) |
| `lineage` | string | tissue of origin (DepMap OncotreeLineage) |
| `depmap_id` | string | DepMap model ID (ACH-…) used for this line's DepMap data |
| `depmap_is_proxy` | bool | True if the DepMap profile is a parent line's (HL-60(TB) → HL-60) |
| `family_id` | string | lines sharing an origin share this ID; a family is never split across selectable and held out |
| `study_id` | string | source screen; always `NCI-ALMANAC` in this pool |
| `n_reps` | int64 | number of measurements averaged into this row (1 or 2) |

### `processed/labels.parquet`

**Restricted: the hidden answers.** Same keys and row order as `pool.parquet`, plus outcomes and hit flags. Read only from oracle and evaluation code (see README).

272,357 rows × 18 columns.

| Column | Type | Meaning |
|:--|:--|:--|
| `drug_a` | string | first drug's ID (ChEMBL parent ID, e.g. CHEMBL553); always `drug_a < drug_b` |
| `drug_b` | string | second drug's ID (ChEMBL parent ID) |
| `cellosaurus_ac` | string | cell-line ID (Cellosaurus accession, e.g. CVCL_0023 = A-549); the cell-line join key |
| `y_syn` | double | **synergy outcome** = mean Bliss score (above 0 = better than independent action) |
| `y_inh` | double | **killing outcome** = mean CSS, drug combination sensitivity score (higher = kills more) |
| `css_mean` | double | mean CSS (equals `y_inh`) |
| `css_sd` | double | SD of CSS across replicates (empty if n_reps = 1) |
| `bliss_mean` | double | mean Bliss synergy (equals `y_syn`) |
| `bliss_sd` | double | SD of Bliss across replicates |
| `zip_mean` | double | mean ZIP synergy |
| `zip_sd` | double | SD of ZIP |
| `loewe_mean` | double | mean Loewe synergy |
| `loewe_sd` | double | SD of Loewe |
| `hsa_mean` | double | mean HSA synergy |
| `hsa_sd` | double | SD of HSA |
| `is_hit_syn` | int8 | 1 if `y_syn` ≥ the synergy threshold (p95 of selectable experiments, 9.30) |
| `is_hit_inh` | int8 | 1 if `y_inh` ≥ the killing threshold (p95 of selectable experiments, 62.95) |
| `syn_hit_low_kill` | int8 | 1 if a synergy hit has `y_inh` < 0 (report synergy results with and without these; decision 20) |

### `processed/cells.parquet`

One row per pool cell line: identity, split, data-source flags, and the model-ready cell features.

57 rows × 1,087 columns.

| Column | Type | Meaning |
|:--|:--|:--|
| `cellosaurus_ac` | string | cell-line ID (Cellosaurus accession, e.g. CVCL_0023 = A-549); the cell-line join key |
| `cellosaurus_name` | string | Cellosaurus name of the line |
| `depmap_id` | string | DepMap model ID (ACH-…) used for this line's DepMap data |
| `depmap_is_proxy` | bool | True if the DepMap profile is a parent line's (HL-60(TB) → HL-60) |
| `family_id` | string | lines sharing an origin share this ID; a family is never split across selectable and held out |
| `problematic` | string | Cellosaurus 'problematic cell line' note (contamination or misidentification), if any |
| `lineage` | string | tissue of origin (DepMap OncotreeLineage) |
| `primary_disease` | string | DepMap OncotreePrimaryDisease |
| `subtype` | string | DepMap OncotreeSubtype |
| `expression_source` | string | `cellminer` (measured) or `imputed` (SR; see CAVEATS) |
| `expression_missing` | int8 | 1 if no expression at all (none in this version) |
| `expression_imputed` | int8 | 1 if expression was imputed (SR) |
| `expression_identity_spearman` | double | Spearman between this line's CellMiner and own-DepMap expression (identity check) |
| `mutation_source` | string | `depmap` (own profile) or `depmap_parent` (borrowed from a parent line) |
| `cn_source` | string | copy-number source (`cellminer`) |
| `eligible_for_heldout` | bool | True if every feature of the line is its own measurement (decision 19) |
| `split` | string | `selectable` (usable by selectors and models) or `heldout` (only for the final mechanism check) |
| `mutation_from_parent` | int8 | 1 if mutation calls are borrowed from a parent line |
| `mutation_from_cellminer` | int8 | 1 if mutation calls came from CellMiner (none in this version) |
| `lineage_Bowel` … `lineage_Skin` (11 columns) | numeric | lineage one-hot (tissue) |
| `mut_KRAS` … `mut_ATM` (24 columns) | numeric | driver-gene mutation, 0/1: oncogenes = hotspot; tumour suppressors = damaging (TP53 also missense) |
| `cn_KRAS` … `cn_FGFR1` (31 columns) | numeric | copy number of a driver / amplified gene, log2 ratio to normal (0 = 2 copies) |
| `cn_missing_STK11` … `cn_missing_KEAP1` (2 columns) | numeric | 1 if this gene's copy number is missing for the line (value filled with 0 = 2 copies) |
| `expr_MIR5047` … `expr_SHISA4` (1,000 columns) | numeric | expression of one of the 1,000 most variable genes, log2(FPKM+1) (CellMiner RNA-seq) |

### `processed/drugs.parquet`

One row per drug (122; `in_pool` = 101): annotation flags and the model-ready drug features.

122 rows × 2,336 columns.

| Column | Type | Meaning |
|:--|:--|:--|
| `drug_id` | string | drug ID (ChEMBL parent ID); joins to `drug_a` / `drug_b` |
| `preferred_name` | string | drug name (ChEMBL preferred name) |
| `target_status` | string | `chembl` (own mechanism records), `chembl_proxy` (borrowed from a related record) or `none_in_chembl` (no target; kept, decision 8) |
| `smiles_origin` | string | where the structure came from: ChEMBL parent, or the source file (Pt drugs) |
| `in_pool` | int8 | 1 if the drug appears in the pool (101 of 122) |
| `has_target` | int8 | 1 if ChEMBL lists any mechanism target (DNA included) |
| `has_target_gene` | int8 | 1 if a target has a gene symbol (DNA-targeting drugs have none) |
| `smiles_from_source` | int8 | 1 if the fingerprint uses the source structure (3 Pt drugs) |
| `n_target_genes` | int16 | number of target genes |
| `target_ABL1` … `target_thyA` (263 columns) | numeric | target gene, multi-hot: 1 = the drug acts on this gene (from ChEMBL mechanisms) |
| `action_AGONIST` … `action_STABILISER` (8 columns) | numeric | mechanism action type, multi-hot (e.g. INHIBITOR, AGONIST) |
| `ttype_CHIMERIC_PROTEIN` … `ttype_SINGLE_PROTEIN` (8 columns) | numeric | target type, multi-hot (e.g. SINGLE_PROTEIN, NUCLEIC_ACID) |
| `fp_0` … `fp_2047` (2,048 columns) | numeric | Morgan fingerprint bits (radius 2, 2,048 bits): 1 = the substructure is present |

### `processed/self_pairs.parquet`

Drug-with-itself experiments. Not combinations; kept for score sanity checks (Loewe should be about 0).

111 rows × 20 columns.

| Column | Type | Meaning |
|:--|:--|:--|
| `drug_a` | string | first drug's ID (ChEMBL parent ID, e.g. CHEMBL553); always `drug_a < drug_b` |
| `drug_b` | string | second drug's ID (ChEMBL parent ID) |
| `cellosaurus_ac` | string | cell-line ID (Cellosaurus accession, e.g. CVCL_0023 = A-549); the cell-line join key |
| `study_id` | string | source screen; always `NCI-ALMANAC` in this pool |
| `n_reps` | int64 | number of measurements averaged into this row (1 or 2) |
| `css_mean` | double | mean CSS (equals `y_inh`) |
| `css_sd` | double | SD of CSS across replicates (empty if n_reps = 1) |
| `bliss_mean` | double | mean Bliss synergy (equals `y_syn`) |
| `bliss_sd` | double | SD of Bliss across replicates |
| `zip_mean` | double | mean ZIP synergy |
| `zip_sd` | double | SD of ZIP |
| `loewe_mean` | double | mean Loewe synergy |
| `loewe_sd` | double | SD of Loewe |
| `hsa_mean` | double | mean HSA synergy |
| `hsa_sd` | double | SD of HSA |
| `y_syn` | double | **synergy outcome** = mean Bliss score (above 0 = better than independent action) |
| `y_inh` | double | **killing outcome** = mean CSS, drug combination sensitivity score (higher = kills more) |
| `depmap_id` | string | DepMap model ID (ACH-…) used for this line's DepMap data |
| `depmap_is_proxy` | bool | True if the DepMap profile is a parent line's (HL-60(TB) → HL-60) |
| `family_id` | string | lines sharing an origin share this ID; a family is never split across selectable and held out |

### `processed/splits.json`

Held-out lines (Cellosaurus IDs and names), per-lineage record, seed, hit thresholds.

### `processed/data_card.md`

Human-readable summary of the pool.

### `processed/manifest.json`

sha256 of every input and output of the freeze, plus raw-download records.

### `processed/freeze_report.json`

Split statistics, mutation testability per gene, acceptance checks.

## `data_cache/interim/features/`

### `interim/features/cell_expression.parquet`

Full expression profiles: 57 lines × all CellMiner genes, log2(FPKM+1). SR's row is imputed. For agent tools and expression rules (M6/M7).

57 rows × 23,809 columns.

- `cellosaurus_ac`: cell-line ID (Cellosaurus accession, e.g. CVCL_0023 = A-549); the cell-line join key
- **23,808 gene columns**, named by HGNC gene symbol (e.g. `CH17-408M7.1`, `DDX11L1`, `WASH7P`). Values as described above.

### `interim/features/cell_copy_number.parquet`

Full copy-number profiles: 57 lines × all genes, log2 ratio to normal DNA (0 = 2 copies; copy number N = 2·2^value). Empty where CellMiner has no value.

57 rows × 23,224 columns.

- `cellosaurus_ac`: cell-line ID (Cellosaurus accession, e.g. CVCL_0023 = A-549); the cell-line join key
- **23,223 gene columns**, named by HGNC gene symbol (e.g. `C1orf222`, `WASH7P`, `OR4F5`). Values as described above.

### `interim/features/cell_mutations.parquet`

Driver-mutation calls used in the features: 57 lines × 24 genes (1 = mutated), with the source of each line's calls.

57 rows × 27 columns.

| Column | Type | Meaning |
|:--|:--|:--|
| `cellosaurus_ac` | string | cell-line ID (Cellosaurus accession, e.g. CVCL_0023 = A-549); the cell-line join key |
| `mutation_source` | string | `depmap` (own profile) or `depmap_parent` (borrowed from a parent line) |
| `mutation_model` | string | DepMap model the mutation calls came from |
| `KRAS` | int64 | driver gene: 1 = mutated in this line |
| `NRAS` | int64 | driver gene: 1 = mutated in this line |
| `HRAS` | int64 | driver gene: 1 = mutated in this line |
| `BRAF` | int64 | driver gene: 1 = mutated in this line |
| `PIK3CA` | int64 | driver gene: 1 = mutated in this line |
| `EGFR` | int64 | driver gene: 1 = mutated in this line |
| `ERBB2` | int64 | driver gene: 1 = mutated in this line |
| `CTNNB1` | int64 | driver gene: 1 = mutated in this line |
| `IDH1` | int64 | driver gene: 1 = mutated in this line |
| `MAP2K1` | int64 | driver gene: 1 = mutated in this line |
| `TP53` | int64 | driver gene: 1 = mutated in this line |
| `PTEN` | int64 | driver gene: 1 = mutated in this line |
| `CDKN2A` | int64 | driver gene: 1 = mutated in this line |
| `RB1` | int64 | driver gene: 1 = mutated in this line |
| `APC` | int64 | driver gene: 1 = mutated in this line |
| `STK11` | int64 | driver gene: 1 = mutated in this line |
| `KEAP1` | int64 | driver gene: 1 = mutated in this line |
| `NF1` | int64 | driver gene: 1 = mutated in this line |
| `SMAD4` | int64 | driver gene: 1 = mutated in this line |
| `ARID1A` | int64 | driver gene: 1 = mutated in this line |
| `FBXW7` | int64 | driver gene: 1 = mutated in this line |
| `BRCA1` | int64 | driver gene: 1 = mutated in this line |
| `BRCA2` | int64 | driver gene: 1 = mutated in this line |
| `ATM` | int64 | driver gene: 1 = mutated in this line |

### `interim/features/cell_mutations_cellminer.parquet`

CellMiner's own exome calls for the same 24 genes, for comparison only (less sensitive for tumour-suppressor genes).

57 rows × 25 columns.

| Column | Type | Meaning |
|:--|:--|:--|
| `cellosaurus_ac` | string | cell-line ID (Cellosaurus accession, e.g. CVCL_0023 = A-549); the cell-line join key |
| `KRAS` | int64 | driver gene: 1 = mutated in this line |
| `NRAS` | int64 | driver gene: 1 = mutated in this line |
| `HRAS` | int64 | driver gene: 1 = mutated in this line |
| `BRAF` | int64 | driver gene: 1 = mutated in this line |
| `PIK3CA` | int64 | driver gene: 1 = mutated in this line |
| `EGFR` | int64 | driver gene: 1 = mutated in this line |
| `ERBB2` | int64 | driver gene: 1 = mutated in this line |
| `CTNNB1` | int64 | driver gene: 1 = mutated in this line |
| `IDH1` | int64 | driver gene: 1 = mutated in this line |
| `MAP2K1` | int64 | driver gene: 1 = mutated in this line |
| `TP53` | int64 | driver gene: 1 = mutated in this line |
| `PTEN` | int64 | driver gene: 1 = mutated in this line |
| `CDKN2A` | int64 | driver gene: 1 = mutated in this line |
| `RB1` | int64 | driver gene: 1 = mutated in this line |
| `APC` | int64 | driver gene: 1 = mutated in this line |
| `STK11` | int64 | driver gene: 1 = mutated in this line |
| `KEAP1` | int64 | driver gene: 1 = mutated in this line |
| `NF1` | int64 | driver gene: 1 = mutated in this line |
| `SMAD4` | int64 | driver gene: 1 = mutated in this line |
| `ARID1A` | int64 | driver gene: 1 = mutated in this line |
| `FBXW7` | int64 | driver gene: 1 = mutated in this line |
| `BRCA1` | int64 | driver gene: 1 = mutated in this line |
| `BRCA2` | int64 | driver gene: 1 = mutated in this line |
| `ATM` | int64 | driver gene: 1 = mutated in this line |

### `interim/features/expression_identity.parquet`

Identity check: each line's CellMiner expression vs its own DepMap profile (Spearman). `suspect` = likely sample mix-up (SR).

46 rows × 7 columns.

| Column | Type | Meaning |
|:--|:--|:--|
| `cellosaurus_ac` | string | cell-line ID (Cellosaurus accession, e.g. CVCL_0023 = A-549); the cell-line join key |
| `depmap_id` | string | DepMap model ID (ACH-…) used for this line's DepMap data |
| `own_spearman` | double | Spearman between the line's CellMiner and own-DepMap expression (2,000 most variable genes) |
| `best_depmap_id` | string | the DepMap model whose expression matches this CellMiner profile best |
| `best_spearman` | double | Spearman with that best-matching model |
| `own_is_best` | bool | True if the best match is the line's own DepMap profile |
| `suspect` | bool | True if not its own best match and own_spearman < 0.80 (likely sample mix-up; SR) |

### `interim/features/features_report.json`

Feature-build report: sources, imputation quality, checks.

## `data_cache/interim/`

### `interim/drugs.parquet`

Drug identities: ChEMBL parent, name, structure, mechanism counts.

122 rows × 17 columns.

| Column | Type | Meaning |
|:--|:--|:--|
| `drug_id` | string | drug ID (ChEMBL parent ID); joins to `drug_a` / `drug_b` |
| `chembl_parent_id` | string | ChEMBL parent molecule ID (salt forms resolved to the parent) |
| `preferred_name` | string | drug name (ChEMBL preferred name) |
| `source_ids` | string | all source labels merged into this identity (`|`-separated) |
| `n_source_ids` | int64 | number of source labels merged into this drug |
| `n_rows` | int64 | number of source rows that use this label or drug |
| `smiles` | string | structure used for fingerprints (ChEMBL parent with stereochemistry; source for Pt drugs) |
| `smiles_origin` | string | where the structure came from: ChEMBL parent, or the source file (Pt drugs) |
| `inchikey` | string | InChIKey of the structure |
| `structure_key` | string | first 14 characters of the InChIKey (atom connectivity, ignoring 3-D shape) |
| `has_stereo` | bool | True if the structure carries stereochemistry |
| `n_unassigned_stereocentres` | int64 | stereocentres left unassigned in the structure |
| `max_phase` | double | highest clinical phase in ChEMBL (4 = approved) |
| `n_mechanisms` | int64 | number of ChEMBL mechanism records |
| `n_targets` | int64 | number of distinct mechanism targets (DNA counts as a target) |
| `mechanism_source` | string | ChEMBL ID the mechanism records were taken from (differs from the drug's own ID for proxies) |
| `target_genes` | string | target gene symbols (`|`-separated) |

### `interim/drug_targets.parquet`

One row per (drug, mechanism target) from ChEMBL: target, action, mechanism description. For the agent's drug lookups.

243 rows × 11 columns.

| Column | Type | Meaning |
|:--|:--|:--|
| `drug_id` | string | drug ID (ChEMBL parent ID); joins to `drug_a` / `drug_b` |
| `target_chembl_id` | string | ChEMBL target ID |
| `target_pref_name` | string | target name |
| `target_type` | string | ChEMBL target type (SINGLE PROTEIN, NUCLEIC-ACID, PROTEIN FAMILY, …) |
| `organism` | string | target organism |
| `gene_symbols` | string | gene symbols of the target (`|`-separated; empty for DNA) |
| `action_type` | string | action on the target (INHIBITOR, AGONIST, …) |
| `mechanism_of_action` | string | ChEMBL's mechanism description, e.g. 'Epidermal growth factor receptor erbB1 inhibitor' |
| `direct_interaction` | int64 | 1 if the drug binds the target directly |
| `mec_id` | int64 | ChEMBL mechanism record ID |
| `mechanism_source` | string | ChEMBL ID the mechanism records were taken from (differs from the drug's own ID for proxies) |

### `interim/drug_map.parquet`

Source drug label (as in the TDC file) → drug identity, with how it was resolved.

129 rows × 16 columns.

| Column | Type | Meaning |
|:--|:--|:--|
| `source_id` | string | the label exactly as in the TDC source file |
| `drug_id` | string | drug ID (ChEMBL parent ID); joins to `drug_a` / `drug_b` |
| `chembl_parent_id` | string | ChEMBL parent molecule ID (salt forms resolved to the parent) |
| `preferred_name` | string | drug name (ChEMBL preferred name) |
| `resolution` | string | how the label was resolved (e.g. name+structure, manual, nci60_panel) |
| `structure_check` | string | how name and structure were confirmed to agree (connectivity, formula, …) |
| `pubchem_cid` | int64 | PubChem compound ID used in resolution, if any |
| `source_smiles` | string | the structure as given in the source file (no stereochemistry) |
| `n_rows` | int64 | number of source rows that use this label or drug |
| `n_lines` | int64 | number of cell lines the label appears on |
| `n_fragments` | int64 | number of disconnected fragments in the source structure (salts, Pt complexes) |
| `source_has_metal` | bool | True if the source structure contains a metal (Pt drugs) |
| `source_has_stereo` | bool | True if the source structure has stereochemistry (never, in this file) |
| `source_conn_key` | string | connectivity key computed from the source structure |
| `source_heavy_formula` | string | heavy-atom formula of the source structure (used to check Pt drugs) |
| `note` | string | reviewer's note explaining a manual or unusual mapping |

### `interim/line_map.parquet`

Source cell-line label → Cellosaurus accession, DepMap ID, family, keep/drop.

59 rows × 9 columns.

| Column | Type | Meaning |
|:--|:--|:--|
| `source_id` | string | the label exactly as in the TDC source file |
| `cellosaurus_ac` | string | cell-line ID (Cellosaurus accession, e.g. CVCL_0023 = A-549); the cell-line join key |
| `match_kind` | string | which Cellosaurus name matched (identifier, synonym, manual) |
| `resolution` | string | how the label was resolved (e.g. name+structure, manual, nci60_panel) |
| `depmap_id` | string | DepMap model ID (ACH-…) used for this line's DepMap data |
| `depmap_is_proxy` | bool | True if the DepMap profile is a parent line's (HL-60(TB) → HL-60) |
| `family_id` | string | lines sharing an origin share this ID; a family is never split across selectable and held out |
| `keep` | bool | True if the line is in the pool |
| `drop_reason` | string | why a line is not in the pool (e.g. no_depmap_id) |

### `interim/cell_lines.parquet`

All 59 source cell lines (including the 2 dropped): Cellosaurus details, problems, families.

59 rows × 17 columns.

| Column | Type | Meaning |
|:--|:--|:--|
| `cellosaurus_ac` | string | cell-line ID (Cellosaurus accession, e.g. CVCL_0023 = A-549); the cell-line join key |
| `cellosaurus_name` | string | Cellosaurus name of the line |
| `source_ids` | string | all source labels merged into this identity (`|`-separated) |
| `depmap_id` | string | DepMap model ID (ACH-…) used for this line's DepMap data |
| `depmap_ids_listed` | string | all DepMap IDs Cellosaurus lists for the line (`|`-separated) |
| `depmap_is_proxy` | bool | True if the DepMap profile is a parent line's (HL-60(TB) → HL-60) |
| `depmap_proxy_of` | string | Cellosaurus ID of the parent line whose DepMap profile is used, if any |
| `category` | string | Cellosaurus category (all: Cancer cell line) |
| `sex` | string | sex of the donor |
| `disease_ncit` | string | NCI Thesaurus disease code(s) |
| `disease` | string | disease name(s) |
| `in_nci60_panel` | bool | True if Cellosaurus lists the line as part of the NCI-60 panel |
| `problematic` | string | Cellosaurus 'problematic cell line' note (contamination or misidentification), if any |
| `derived_from` | string | Cellosaurus IDs of parent lines (`|`-separated) |
| `family_id` | string | lines sharing an origin share this ID; a family is never split across selectable and held out |
| `keep` | bool | True if the line is in the pool |
| `drop_reason` | string | why a line is not in the pool (e.g. no_depmap_id) |

### `interim/oneil_rows.parquet`

**Contains outcomes** (O'Neil study). Rows removed from the pool (decision 13), for the second benchmark and analysis.

5,360 rows × 13 columns.

| Column | Type | Meaning |
|:--|:--|:--|
| `source_row` | int64 | row index in the converted TDC table |
| `fate` | string | what happened to the source row (kept, oneil, oneil_ambiguous, line_dropped, self_pair, third_study) |
| `Drug1_ID` | string | first drug label as in the source file |
| `Drug2_ID` | string | second drug label as in the source file |
| `Cell_Line_ID` | string | cell-line label as in the source file |
| `drug_a` | string | first drug's ID (ChEMBL parent ID, e.g. CHEMBL553); always `drug_a < drug_b` |
| `drug_b` | string | second drug's ID (ChEMBL parent ID) |
| `cellosaurus_ac` | string | cell-line ID (Cellosaurus accession, e.g. CVCL_0023 = A-549); the cell-line join key |
| `CSS` | double | CSS (killing) as in the source |
| `Synergy_Bliss` | double | Bliss synergy as in the source |
| `Synergy_ZIP` | double | ZIP synergy as in the source |
| `Synergy_Loewe` | double | Loewe synergy as in the source |
| `Synergy_HSA` | double | HSA synergy as in the source |

### `interim/cross_study_overlap.parquet`

**Contains pool outcomes** for 546 experiments measured by both studies (O'Neil vs ALMANAC means). Treat like labels.

546 rows × 15 columns.

| Column | Type | Meaning |
|:--|:--|:--|
| `drug_a` | string | first drug's ID (ChEMBL parent ID, e.g. CHEMBL553); always `drug_a < drug_b` |
| `drug_b` | string | second drug's ID (ChEMBL parent ID) |
| `cellosaurus_ac` | string | cell-line ID (Cellosaurus accession, e.g. CVCL_0023 = A-549); the cell-line join key |
| `css_mean_almanac` | double | mean CSS measured by NCI-ALMANAC |
| `bliss_mean_almanac` | double | mean BLISS measured by NCI-ALMANAC |
| `zip_mean_almanac` | double | mean ZIP measured by NCI-ALMANAC |
| `loewe_mean_almanac` | double | mean LOEWE measured by NCI-ALMANAC |
| `hsa_mean_almanac` | double | mean HSA measured by NCI-ALMANAC |
| `n_reps_almanac` | int64 | number of NCI-ALMANAC measurements |
| `css_mean_oneil` | double | mean CSS measured by O'Neil |
| `bliss_mean_oneil` | double | mean BLISS measured by O'Neil |
| `zip_mean_oneil` | double | mean ZIP measured by O'Neil |
| `loewe_mean_oneil` | double | mean LOEWE measured by O'Neil |
| `hsa_mean_oneil` | double | mean HSA measured by O'Neil |
| `n_reps_oneil` | int64 | number of O'Neil measurements |

### `interim/row_fate.parquet`

What happened to each of the 297,098 source rows (kept, O'Neil, dropped line, …).

297,098 rows × 2 columns.

| Column | Type | Meaning |
|:--|:--|:--|
| `source_row` | int64 | row index in the converted TDC table |
| `fate` | string | what happened to the source row (kept, oneil, oneil_ambiguous, line_dropped, self_pair, third_study) |

### `interim/experiments_report.json`

Experiment-build report: fate counts, replicate agreement, cross-study agreement.

### `interim/drug_harmonization_report.json`

Drug-identity build report.

### `interim/line_harmonization_report.json`

Cell-line build report.

