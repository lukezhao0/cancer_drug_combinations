# Handoff: M1 (data and pool) complete

| | |
|:--|:--|
| **Date** | 2026-11-08 |
| **Milestone** | M1, Data and pool ([`target_design.md`](target_design.md) §4, §12) |
| **Status** | ✅ Built, tested, frozen and reproducible. Next: **M2** |
| **Branch** | the M1 data-pipeline branch (see the [CHANGELOG](CHANGELOG.md)) |
| **Read next** | [M1 summary with figures](project-notes/m1-freeze-and-splits.md) → [decision log](project-notes/m1-implementation.md) |

---

## 1. Project goal and direction (the 60-second version)

**Problem.** Cancers that resist one drug are often treated with combinations, but there are far
too many pairs to test. Labs pick experiments by intuition, one batch at a time.

**What we build.** A **simulated lab**: a frozen table of real measured drug-pair × cell-line
results, hidden until an agent "runs" an experiment. Every method gets the same experiment budget
and is scored on how many strong combinations it finds:
- **random**;
- **uncertainty sampling**;
- **Bayesian optimization**;
- **our agent**, which proposes mechanistic hypotheses, then updates them after each batch of
  results. It is built on the SHELDON hypothesis-search harness.

**The two research questions** (design §1):
1. **Efficiency:** does the agent find strong combinations in fewer experiments than the
   baselines? The primary endpoint is synergy hits within 320 experiments, over 10 seeds.
2. **Mechanisms:** when the agent explains *why* a pair works, does that explanation correctly
   predict which **held-out** cell lines respond?

**Two outcomes, tracked separately:**
- **synergy** (`y_syn`, Bliss): the pair does better than the single drugs predict;
- **killing** (`y_inh`, CSS): the pair kills cells well.

They are nearly unrelated (Spearman 0.06).

---

## 2. What was done in M1

```
 raw TDC file ─► 1a ingest ─► 1b profile ─► 1c drugs ─► 1d cell lines ─► 1e experiments ─► 1f features ─► 1g freeze
 297,098 rows    hash-pinned   statistics    129 → 122    59 → 57 kept     272,357          DepMap 26Q3     11 held out
                                             molecules    3 families       ALMANAC only     + CellMiner     thresholds
```

| Step | What it does | Code (`src/combo_agent/data/`) |
|:--|:--|:--|
| 1a | Downloads the TDC NCI-60 table (sha256-pinned), converts the pickle safely to parquet | `ingest_tdc.py`, `_tdc_pickle_convert.py` |
| 1b | Profiles the table (numbers in the decision log) | — |
| 1c | **One drug, one identity:** maps 129 labels to 122 ChEMBL parent molecules; name and structure must agree; targets attached | `harmonize_drugs.py`, `chembl.py`, `structure.py`, `api_cache.py`, `curated/drug_*.csv` |
| 1d | **One cell line, one identity:** Cellosaurus accessions, DepMap IDs, families of related lines | `harmonize_lines.py`, `cellosaurus.py`, `curated/cell_lines.csv` |
| 1e | **One row per experiment:** every source row gets exactly one fate; the pool keeps NCI-ALMANAC only; replicates averaged | `build_experiments.py` |
| 1f | **Features:** drugs (Morgan fingerprint, targets, action and target types); cells (lineage, 24 driver mutations, copy number, 1,000 expression genes) | `ingest_cellminer.py`, `_cellminer_convert.py`, `build_features.py` |
| 1g | **Freeze:** held-out split, hit thresholds, `pool` / `labels` kept separate, data card, manifest | `freeze_pool.py` |

**Key numbers**

| Item | Value |
|:--|--:|
| Experiments in the pool | 272,357 |
| Drug pairs × cell lines | 4,939 × 57 |
| Drugs | 101 |
| Selectable / held-out lines | 46 / 11 |
| Selectable / held-out experiments | 219,239 / 53,118 |
| Synergy hit threshold (`y_syn`, p95 of selectable) | ≥ 9.30 |
| Killing hit threshold (`y_inh`, p95 of selectable) | ≥ 62.95 |
| Tests | 102 unit tests, plus real-data tests (all pass) |
| Reproducibility | every output is byte-identical across rebuilds; hashes in [`artifacts/m1_freeze/manifest.json`](../artifacts/m1_freeze/manifest.json) |

**Held-out lines** (stratified by tissue, families kept together, seed 0): HCT 15, BT-549, SF268,
786-O, A-498, A-549, HOP-92, CCRF-CEM, IGROV-1, LOX-IMVI, UACC-257.

---

## 3. Decisions everyone should know

All 20 are in one table, with reasons, in
[m1-freeze-and-splits.md §2](project-notes/m1-freeze-and-splits.md). The ones most likely to
surprise you:

| # | Decision | Why |
|:-:|:--|:--|
| 8 | Drugs without a ChEMBL target are **kept**; the agent looks up their targets in the literature | a database gap isn't evidence of no mechanism |
| 13 | **O'Neil/Merck rows removed** from the NCI-60 pool (they were mixed into the TDC file); O'Neil becomes the second benchmark | the two labs disagree on the same experiments (CSS r = 0.22) |
| 15 | Expression and copy number from **NCI CellMiner** for all lines; mutations from DepMap 26Q3 | one source per data type; DepMap lacks expression for 10 of the 57 lines |
| 16 | SNB-19 and NCI-H322M take mutation calls from their confirmed parent lines (U-251 MG, NCI-H322) | no DepMap sequencing; parenthood confirmed in the data |
| 17 | **SR's expression is imputed** from CellMiner microarrays | CellMiner's SR RNA-seq sample is a mix-up |
| 19 | Only lines whose features are all their own measurements can be held out | held-out lines test predictions over those features |
| 20 | Synergy hits with negative killing are kept, flagged `syn_hit_low_kill`, and **every synergy result is reported with and without them** | Bliss inflates when single drugs barely act |

---

## 4. Caveats: read before using the data

1. **SR's expression is imputed** (`expression_imputed = 1`). Say so wherever SR's expression is
   used. Genes only SR expresses (ALK, for example) are underestimated. SR is never a held-out line.
2. **Borrowed mutation calls:** SNB-19 (from U-251 MG), NCI-H322M (from NCI-H322) and HL-60(TB)
   (from HL-60). They're flagged `mutation_source = "depmap_parent"`.
3. **Synergy hits that barely kill** (958, 7.1% of synergy hits): always report synergy metrics
   both with and without them.
4. **Bliss is noisy:** the same experiment measured twice agrees at r = 0.21, so a hit near the
   threshold is partly luck.
5. **Testability:** no held-out line carries NRAS, HRAS, ERBB2, CTNNB1, RB1 or ATM mutations.
   Report the testable fraction of predictions.
6. **`labels.parquet` holds the hidden outcomes.** Only the oracle may read it (M2 adds a test that
   enforces this).

---

## 5. How to pick up

1. `uv sync && uv run pytest -q`. The unit tests need no data.
2. **Get the data:** download the shared bundle `m1-data-v1.tgz` from the team's
   [Google Drive folder](https://drive.google.com/drive/folders/1GsulBMNAGEjXhBt8p-UvNpYufTfvvRW7?usp=drive_link)
   (ask for access). Verify it against the sha256 in the [README](../README.md), and copy its
   `data_cache/` into the repo. Its docs (also in
   [`data-bundle/`](data-bundle/)) cover every column, the split, the hit definition and the
   caveats, with a Python quick start.
   - *Alternative:* rebuild from the raw sources (README → Rebuilding the M1 data pool). That needs
     the DepMap 26Q3 manual download, and optionally the API-cache archive. Check a rebuild against
     `artifacts/m1_freeze/manifest.json`.
3. Read, in order:
   - [M1 summary](project-notes/m1-freeze-and-splits.md);
   - [decision log](project-notes/m1-implementation.md);
   - [cell-line data notes](project-notes/depmap-data.md);
   - [drug identity audit](project-notes/drug-identity-audit.md).

   For concepts, see the [concepts primer](project-notes/concepts-primer.md) and the
   [step-by-step walkthrough](project-notes/m1-walkthrough.md).

**Data not in git** (by design: see
[project-notes/data-bundle-and-sharing.md](project-notes/data-bundle-and-sharing.md)):
- **`m1-data-v1.tgz`** (24 MB) and **`m1_api_cache.tgz`** (1 MB): in the Google Drive folder;
- the raw DepMap files: download them yourself, only if rebuilding;
- your local `data_cache/` (about 2 GB after a full rebuild; about 65 MB from the bundle).

---

## 6. Next steps

| Milestone | Work | Gate (design §12) |
|:--|:--|:--|
| **M2 Oracle and splits** (next) | `Oracle` with budget, batch and held-out checks, the only reader of `labels.parquet`; **10 seed sets** of 32 free experiments; **famous / middle / obscure** pair labels from PubMed co-mentions | label-access test and refusal tests pass |
| M3 Response model | deep ensemble with symmetric drug-pair input and two heads (synergy, killing); calibration report | 80% intervals cover 70–90%; swap test; refit < 60 s |
| M4 Baselines and runner | random, greedy, uncertainty, BO-UCB, BO-qEI; evaluation runner and discovery curves | BO-UCB beats random on hits at 320 |
| M5 Retrieval and leakage filter | literature search with a blocklist of the screens' own papers; audit log | the blocklist removes every source paper |
| M6 Agent | SHELDON-based hypothesis loop, tools, batch assembly; ablations | 3-seed pilot within the cost cap |
| M7 Explanations | explanation records, rule parser, frozen predictions, held-out scoring | the rule parser and planted-rule test pass |
| M8 Full grid and report | all selectors × outcomes × seeds × benchmarks | every number traces to results |

**Open items:**
- The design's open decisions (§13): the reasoning-model choice and cost cap; budget and batch
  size; the literature corpus.
- The second benchmark (O'Neil / OncoPolyPharmacology): a 1.6 GB download. The 16 O'Neil-only
  drugs (mostly targeted agents such as PARP, WEE1, AKT and MEK inhibitors) return there.
- Review the hand-set rows in `src/combo_agent/data/curated/`: 8 manual drug rows, 2 borrowed
  target sets, 3 cell-line overrides.
- From M3 on, report synergy results with and without `syn_hit_low_kill` (decision 20).

---

## 7. Files added or changed in M1

| Path | Change |
|:--|:--|
| `src/combo_agent/data/` (15 Python files) and `curated/` (3 CSVs) | **new**: the data pipeline |
| `tests/` (6 files) | **new**: tests |
| `pyproject.toml`, `uv.lock` | numpy, pyarrow, rdkit; pytest (dev) |
| `.gitignore` | ignores `data_cache/`, `runs/`, `.cache/`. Fixes the old `data/` pattern, which hid `src/combo_agent/data/` |
| `scripts/audit_drug_identity.py`, `scripts/make_m1_figures.py` | **new**: analysis |
| `scripts/make_data_bundle.py` | **new**: builds the shareable data bundle |
| `artifacts/m1_freeze/` | **new**: frozen split, data card, manifest |
| `docs/target_design.md`, `docs/project-notes/` | **new**: specification and M1 notes (including how the data is shared) |
| `docs/data-bundle/` | **new**: the docs shipped inside the data bundle (README, generated data dictionary, splits and hits, caveats, provenance) |
| `docs/HANDOFF.md`, `docs/CHANGELOG.md`, `docs/handoff-template.md` | **new**: this handoff |
| `README.md` | status, layout, the Drive download with checksums, how to rebuild the data |
