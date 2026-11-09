# M1 data bundle (`m1-data-v1`): start here

This folder is the **frozen M1 data pool** of the cancer drug-combination project: every table you
need to start M2 (oracle), M3 (response model), M4 (baselines) and later milestones, without
rebuilding anything.

| | |
|:--|:--|
| **What** | 272,357 drug-pair × cell-line experiments from the NCI-ALMANAC screen (4,939 drug pairs × 57 cell lines, 101 drugs), with drug and cell features, a held-out split and hit thresholds |
| **Version** | `m1-data-v1` (frozen 2026-11-08). See `BUNDLE_INFO.json` for the build's code commit |
| **Code** | the project's GitHub repo, `cancer_drug_combinations` (pipeline in `src/combo_agent/data/`) |
| **Size** | about 65 MB unpacked |

## Files in this folder

| File | Read it for |
|:--|:--|
| `README.md` | this overview, setup and the quick start |
| [`DATA_DICTIONARY.md`](DATA_DICTIONARY.md) | **every file and column**: meaning, type, units, keys (generated from the data) |
| [`SPLITS_AND_HITS.md`](SPLITS_AND_HITS.md) | the held-out cell lines, how to use the split, what counts as a "hit" |
| [`CAVEATS.md`](CAVEATS.md) | **read before analysing:** imputed and borrowed values, metric quirks, noise |
| [`PROVENANCE.md`](PROVENANCE.md) | where every number came from: sources, versions, licences, decisions |
| `SHA256SUMS` | checksums for every data file |
| `BUNDLE_INFO.json` | bundle version, build date, code commit |
| `data_cache/` | the data, laid out exactly like the repo's `data_cache/` folder |

```
data_cache/
  processed/                  <- the frozen pool: use these for modelling
    pool.parquet              experiments: keys + split + metadata, NO outcomes
    labels.parquet            outcomes and hit flags. Restricted: see "The one rule" below
    cells.parquet             57 cell lines: identity, split, model-ready features (1,087 columns)
    drugs.parquet             122 drugs: model-ready features (2,336 columns)
    self_pairs.parquet        drug-with-itself experiments (score sanity checks only)
    splits.json               held-out lines, thresholds, how the split was made
    data_card.md              human summary of the pool
    manifest.json             sha256 of every input and output of the freeze
    freeze_report.json        split statistics and checks
  interim/                    <- supporting tables: full profiles, mappings, audit
    features/                 full gene-level expression and copy number, mutation calls, reports
    drug_targets.parquet      drug mechanism records (target, action, description)
    drugs.parquet             drug identities (SMILES, InChIKey, ChEMBL IDs)
    drug_map.parquet, line_map.parquet, cell_lines.parquet   source name -> identity mappings
    oneil_rows.parquet, cross_study_overlap.parquet, row_fate.parquet   audit of what was removed
    *_report.json             build reports
```

## Setup: two ways to use it

First **verify the download**: unpack it anywhere and check the checksums. Every line should say
`OK`.

```bash
tar -xzf ~/Downloads/m1-data-v1.tgz -C ~/Downloads        # creates ~/Downloads/m1-data-v1/
cd ~/Downloads/m1-data-v1 && shasum -a 256 -c SHA256SUMS   # on Linux: sha256sum -c SHA256SUMS
```

**A. With the code repo (recommended).** Clone the repo and run `uv sync`. Then copy **only the
data** into the repo, so that `data_cache/processed/pool.parquet` exists. Don't copy the bundle's
`README.md` over the repo's.

```bash
cd cancer_drug_combinations
mkdir -p data_cache && cp -R ~/Downloads/m1-data-v1/data_cache/. data_cache/
uv run pytest -q
```

**B. Standalone.** All you need is Python with `pandas` and `pyarrow`. Point `ROOT` at the unpacked
folder.

## Quick start (Python)

```python
import json
import pandas as pd

ROOT = "m1-data-v1/data_cache"          # or "data_cache" inside the repo

pool  = pd.read_parquet(f"{ROOT}/processed/pool.parquet")    # 272,357 experiments, no outcomes
cells = pd.read_parquet(f"{ROOT}/processed/cells.parquet")   # 57 lines
drugs = pd.read_parquet(f"{ROOT}/processed/drugs.parquet")   # 122 drugs
split = json.load(open(f"{ROOT}/processed/splits.json"))

# What a selector may choose from: experiments on selectable lines only
candidates = pool[pool.split == "selectable"]               # 219,239 rows, 46 lines

# Feature columns by block
cell_cols = [c for c in cells.columns if c.startswith(("lineage_", "mut_", "cn_", "expr_"))]
drug_cols = [c for c in drugs.columns if c.startswith(("fp_", "target_", "action_", "ttype_"))
             and c != "target_status"]

# One experiment = drug A + drug B + cell line. Order-invariant: drug_a < drug_b always.
X = (candidates
     .merge(drugs[["drug_id"] + drug_cols].add_suffix("_a"), left_on="drug_a", right_on="drug_id_a")
     .merge(drugs[["drug_id"] + drug_cols].add_suffix("_b"), left_on="drug_b", right_on="drug_id_b")
     .merge(cells[["cellosaurus_ac"] + cell_cols], on="cellosaurus_ac"))
print(X.shape)   # (219239, ...) - about 5.5k columns. Large: see the memory note below
```

**Memory note:** the full joined matrix is several GB. For models, keep the drug and cell tables
separate and combine them inside the model or batch by batch (the design's "factorized"
approach), rather than materializing `X` for the whole pool.

## The one rule: who may read `labels.parquet`

`labels.parquet` holds the **hidden answers**: every experiment's measured synergy and killing.
The project's question is how *few* experiments a method needs to find strong combinations, so a
method that peeks at labels is cheating.

| Who | May read labels? |
|:--|:--|
| The **oracle** (M2): reveals one experiment's result when a method "runs" it | ✅ |
| **Evaluation code**: counts hits found after a run | ✅ |
| Response-model **calibration checks** on a random sanity split (design §6) | ✅, clearly separated |
| Any **selector, model or agent** during a campaign | ❌. Only through the oracle |
| **Held-out lines' labels** | ❌ until the final mechanism-prediction check (design §9) |

`interim/cross_study_overlap.parquet` also contains ALMANAC outcomes for 546 pool experiments, and
`interim/oneil_rows.parquet` holds O'Neil outcomes. Treat both like labels.

## What's next, and what's not in here

| Milestone | Needs from this bundle | Still to build |
|:--|:--|:--|
| M2 Oracle and splits | `pool`, `labels`, `splits.json` | the oracle class; **10 seed sets** (32 free experiments each); **famous / middle / obscure** pair labels (PubMed co-mentions) |
| M3 Response model | `pool`, `labels`, `cells`, `drugs` | the deep ensemble with symmetric drug-pair input |
| M4 Baselines | the same | random / greedy / uncertainty / BO selectors, runner |
| M5 Retrieval | `interim/drugs.parquet` (names, IDs) | the literature corpus and leakage blocklist |
| M6 Agent tools | `interim/drug_targets.parquet`, `processed/cells.parquet`, `interim/features/*` | the agent loop |
| M7 Mechanism checks | `interim/features/cell_expression.parquet`, `cell_copy_number.parquet`, `cell_mutations.parquet` for the held-out lines | rule parser and scoring |

Not included (rebuild from the repo if needed): the raw downloads (TDC, DepMap 26Q3, CellMiner), the
raw-derived intermediate tables, and the API-response cache (`m1_api_cache.tgz`, see below).

## About `m1_api_cache.tgz` (optional, 1.1 MB)

**You don't need it to use this bundle.** It matters only if you want to **rebuild** the data from
scratch and get exactly the same result.

**What it is.** Two pipeline steps look things up in public web databases:
- step 1c resolves drug names and finds drug targets in **ChEMBL** and **PubChem**;
- step 1d resolves cell-line names in **Cellosaurus**.

The pipeline saves every answer it receives to disk: one small JSON file per request, holding the
URL, the HTTP status and the response. The archive is that saved collection, 1,437 answers:

| Service | Saved answers | What they are |
|:--|--:|:--|
| ChEMBL | 1,225 | molecule lookups by name or ID (980), mechanism records (139), target records (105), version check (1) |
| PubChem | 129 | drug name → structure lookups |
| Cellosaurus | 83 | cell-line records (names, parent lines, DepMap IDs, problem notes) and the NCI-60 panel list |

**Why it exists.** These databases change: ChEMBL publishes new releases, records get merged, and
names get added. Re-running steps 1c/1d live a month from now could resolve a drug or line
differently, and the pool would silently change. With the saved answers, a rebuild replays the
exact responses from 2026-11-08, produces byte-identical outputs, needs no network, and avoids the
services' rate limits and dropped connections.

**How to use it.**

```bash
tar -xzf m1_api_cache.tgz -C data_cache/raw/        # creates data_cache/raw/api_cache/{chembl,pubchem,cellosaurus}/
COMBO_OFFLINE=1 uv run python -m combo_agent.data.harmonize_drugs build   # replays from the cache only
COMBO_OFFLINE=1 uv run python -m combo_agent.data.harmonize_lines build
```

With `COMBO_OFFLINE=1`, a request that isn't in the cache **fails** instead of going online. That's
how you prove a rebuild used only the saved answers. Without the cache, leave `COMBO_OFFLINE` unset
and the steps query the services live; the result may then differ slightly from `m1-data-v1`.

**Licence note.** The ChEMBL answers are ChEMBL data (CC BY-SA 3.0): fine to share within the team.
Cite ChEMBL and keep share-alike terms if the cache is ever published.
