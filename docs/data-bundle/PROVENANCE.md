# Provenance: where the data came from

## Sources

| Source | What we used | Version / date | Licence | Used for |
|:--|:--|:--|:--|:--|
| **Therapeutics Data Commons (TDC)**, DrugSyn "DrugComb" (NCI-60) | `drugcomb_nci60.pkl`, Harvard Dataverse file 4215720 | published 2020-12-12; sha256 `c0e195c9…ad00bf` | CC BY 4.0 | the experiments and outcome scores (NCI-ALMANAC subset) |
| **NCI-ALMANAC** (Holbeck et al., 2017) | the underlying screen | — | NCI public data | the original measurements |
| **ChEMBL** (REST API) | drug parents, structures, mechanisms and targets | ChEMBL 37, queried 2026-11-08 | CC BY-SA 3.0 | drug identities and targets |
| **PubChem** (PUG REST) | name → structure lookups | queried 2026-11-08 | public domain | drug name resolution |
| **Cellosaurus** (API) | cell-line identities, synonyms, parent lines, DepMap cross-references | queried 2026-11-08 | CC BY 4.0 | cell-line identities and families |
| **DepMap Public 26Q3** (Broad Institute) | `Model.csv`, `OmicsProfiles.csv`, `OmicsSomaticMutations.csv`, `OmicsSomaticMutationsMatrixHotspot.csv`, `OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv`, `PortalOmicsCNGeneLog2.csv` | 26Q3, downloaded 2026-11-08 | CC BY 4.0 | lineage, driver mutations, identity checks |
| **NCI CellMiner** | RNA-seq composite expression; exome variants; combined aCGH copy number; 5-platform microarray z-scores | database v2.16, files dated 2026-10-28 | NCI public data | expression and copy number for all lines; SR imputation |

The sha256 of every raw download is recorded in the code repo: `artifacts/m1_freeze/manifest.json`
and `docs/project-notes/depmap-data.md`.

**If you share or publish this data:** cite the sources above. Note that ChEMBL-derived content
(drug targets and mechanisms) is **share-alike**.

## How it was built

The code is in the project repo, `src/combo_agent/data/`. Each step, in order:

| Step | Module | What it does |
|:--|:--|:--|
| 1a | `ingest_tdc` | hash-pinned download; pickle converted with a restricted unpickler |
| 1c | `harmonize_drugs` | 129 drug labels → 122 ChEMBL parent molecules; name and structure must agree |
| 1d | `harmonize_lines` | 59 line labels → 59 Cellosaurus lines; families; DepMap IDs |
| 1e | `build_experiments` | one fate per source row; NCI-ALMANAC only; replicates averaged → 272,357 experiments |
| 1f | `ingest_cellminer`, `build_features` | drug and cell features; cross-source identity checks; SR imputation |
| 1g | `freeze_pool` | held-out split, hit thresholds, frozen tables |

The build is **byte-reproducible**. With the same inputs (including the cached API responses), every
output file is identical across rebuilds. `data_cache/processed/manifest.json` lists every input's
and output's sha256.

## Decisions behind the data

20 numbered decisions shaped this pool: what was merged, dropped, imputed and held out. They are
listed with reasons in the repo at `docs/project-notes/m1-freeze-and-splits.md` §2, and in detail in
`docs/project-notes/m1-implementation.md`. Column flags in the data cite them where relevant.

## Versioning

- This bundle is **`m1-data-v1`**. Any change to the pool, features, split or thresholds produces a
  new version (`m1-data-v2`, …). Earlier versions are never edited in place.
- `BUNDLE_INFO.json` records the build date and the code commit the bundle was built from.
