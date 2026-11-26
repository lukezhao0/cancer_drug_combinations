# cancer_drug_combinations

CS 224V project: an agent that combines literature evidence and in-silico experiments to choose
cancer drug-combination experiments. Built on [SHELDON](https://github.com/stanford-oval/sheldon).

**New here? Read [`docs/HANDOFF-m2-m5.md`](docs/HANDOFF-m2-m5.md) first** (current state, and where
the agent plugs in), then [`docs/project-notes/known-issues.md`](docs/project-notes/known-issues.md).
**Anyone building the agent must follow the "Agent-builder rules" at the top of the known-issues file.**
The M1 handoff, [`docs/HANDOFF.md`](docs/HANDOFF.md), covers the data pipeline.

## The goal in one paragraph

Testing every pair of cancer drugs on every cell line is impossible, so labs choose experiments in
small batches. We build a **simulated lab**: a frozen table of real drug-combination measurements
(the NCI-ALMANAC screen) whose results stay hidden until "run". Then we ask two questions:
1. Can an agent that reasons about drug mechanisms find strong combinations **in fewer
   experiments** than random selection, uncertainty sampling and Bayesian optimization?
2. Do the mechanisms it proposes make **correct predictions** about which held-out cell lines
   respond?

The full specification is in [`docs/target_design.md`](docs/target_design.md).

## Status

| Milestone | Status |
|:--|:--|
| M0 Scaffolding (package, CLI, SHELDON pin) | ✅ |
| **M1 Data and pool** | ✅ **Done**: 272,357 experiments (4,939 drug pairs × 57 cell lines, 101 drugs), features, held-out split, hit thresholds, frozen |
| **M2 Oracle and splits** | ✅ **Done**: `Oracle` (budget, batch, held-out and re-run refusals), 10 seed sets of 32, famous/middle/obscure pair labels from PubMed, label-access test |
| **M3 Response model** | ✅ **Done. The main surrogate is a Gaussian process** (`models/gp.py`; default since 2026-11-26): Tanimoto drug-pair kernel × cell kernel. Over 5 seeds it finds about 3× the ensemble's killing hits with every BO rule, and its error bars are honest; synergy is a tie. See [`gp-surrogate.md`](docs/project-notes/gp-surrogate.md). The first model, a 5-network ensemble, is the **legacy build** (`models/legacy/`, git tag `ensemble-v1`) |
| **M4 Baselines and runner** | ✅ **Run, gate passed**: 5 selectors (random, greedy, uncertainty, diverse `bo_ucb`, MC `bo_qei`), 100 campaigns. `bo_ucb` finds 3–4× more hits than random at 320; `greedy` is strongest. Report: `runs/m4_main/report.md` (made with the **legacy ensemble**). With the GP (seeds 0–4), `bo_ucb` finds 94.0 synergy / 202.2 killing hits at 320 vs 16.2 / 15.2 for random. **Numbers provisional:** fix the GP prior scaling and `bo_qei` (known-issues C4, D1), then rerun the M4 grid with the GP before agent comparisons |
| **M5 Retrieval and leakage filter** | ✅ **Built**: `search_literature` with a 6-rule leakage filter, a 607-PMID blocklist and a per-call audit log; gate test set 23/23 leaky papers blocked. Memory probe: the LLM shows no memory of synergy results (ρ −0.05) |
| M7 Explanations · M8 Full grid and report | ⬜ |
| M6 Agent | 🟨 in-silico lookup tools built (`src/combo_agent/tools/`); agent loop not started |

Summaries: M1 [`docs/project-notes/m1-freeze-and-splits.md`](docs/project-notes/m1-freeze-and-splits.md) ·
M2 [`docs/project-notes/m2-oracle.md`](docs/project-notes/m2-oracle.md) ·
M3 [`docs/project-notes/m3-response-model.md`](docs/project-notes/m3-response-model.md) ·
M4 [`docs/project-notes/m4-baselines.md`](docs/project-notes/m4-baselines.md) ·
M5 [`docs/project-notes/m5-literature.md`](docs/project-notes/m5-literature.md).

## Setup

Requirements: Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
git clone <this repo> && cd cancer_drug_combinations
uv sync                 # Creates .venv, installs SHELDON (pinned) and combo_agent (editable)
cp .env.example .env    # Then fill in your API key
uv run pytest -q        # Unit tests (fast, no data needed)
```

### API key (bring your own)

`.env` is read from the directory you run commands from.

```
OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
OPENAI_API_KEY=<your key>
SHELDON_DATA=./data
```

- **Gemini** (default): use the base URL above with a Gemini API key.
- **Any OpenAI-compatible endpoint**: change `OPENAI_BASE_URL` and the key.

### Model Choices

Verified on 11.16 that these models are available for our API key:

- `gemini-3.5-flash-lite` for testing/development ($0.30 / $2.50 per mtok)
- `gemini-4-hydrogen` for reported results ($1.50 / $7.50 per mtok)

### Check the install

```bash
uv run combo-agent ls  # Lists benchmarks, methods and engines
```

## Getting the data (fastest: the shared bundle)

The frozen M1 data (about 24 MB compressed) is shared as a documented bundle, **`m1-data-v1.tgz`**.
It's not kept in git.

| | |
|:--|:--|
| **Download** | [Google Drive folder](https://drive.google.com/drive/folders/1GsulBMNAGEjXhBt8p-UvNpYufTfvvRW7?usp=drive_link) (ask the team for access) |
| **`m1-data-v1.tgz` sha256** | `bd33234e3cf63af3221de6a828b246b0f3a599a69a8088d8af8e65878f4da65b` |
| **`m1_api_cache.tgz` sha256** (optional, for exact rebuilds) | `b7577a54a36e1a9330f380fab2c7defa198636a877c1f97cbfcc1c76080272af` |

```bash
shasum -a 256 ~/Downloads/m1-data-v1.tgz                       # must match the sha256 above
tar -xzf ~/Downloads/m1-data-v1.tgz -C ~/Downloads
(cd ~/Downloads/m1-data-v1 && shasum -a 256 -c SHA256SUMS)     # every line: OK
mkdir -p data_cache && cp -R ~/Downloads/m1-data-v1/data_cache/. data_cache/
```

**M2/M5 add-on (needed for the oracle's seed sets and the leakage filter):** `m2m5-addon-v1.tgz`
(sha256 `610d3862af31fa1bb7b7d5c3870ad6fe6b1cabe038dc4cdda88d11c858614e15`), same Drive folder. Install
it after `m1-data-v1`; the steps are in [`docs/HANDOFF-m2-m5.md`](docs/HANDOFF-m2-m5.md) §6. The
optional `m2m5-addon-v1_api_cache.tgz` (`507909a9b5f826cad7dcb0dafad981d2a1835c31b3782cc70890747ab64e0a66`)
holds the PubMed responses for offline rebuilds.

Inside the M1 bundle: `README.md` (quick start), `DATA_DICTIONARY.md` (every column),
`SPLITS_AND_HITS.md`, `CAVEATS.md` and `PROVENANCE.md`. The same docs are in
[`docs/data-bundle/`](docs/data-bundle/). To rebuild the bundle from a built pool, run
`uv run python scripts/make_data_bundle.py --with-api-cache`.

## Rebuilding the M1 data pool

All data lives in `data_cache/`, which git ignores. Every download is pinned by sha256.

1. **Manual download** (DepMap's site blocks scripts): **DepMap Public 26Q3**, from
   https://depmap.org/portal/data_page/?tab=allData, into `data_cache/raw/depmap/`:
   - `Model.csv`
   - `OmicsProfiles.csv`
   - `OmicsSomaticMutations.csv`
   - `OmicsSomaticMutationsMatrixHotspot.csv`
   - `OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv`
   - `PortalOmicsCNGeneLog2.csv`

   Their checksums are in [`docs/project-notes/depmap-data.md`](docs/project-notes/depmap-data.md).
2. **Optional, for an identical offline rebuild:** unpack the shared API-response archive
   (`m1_api_cache.tgz`, about 12 MB, ask the team) into `data_cache/raw/`. Without it, steps 1c/1d
   query ChEMBL, PubChem and Cellosaurus live.
3. **Run, in order:**

   ```bash
   uv run python -m combo_agent.data.ingest_tdc               # 1a  TDC NCI-60 table (36 MB)
   uv run python -m combo_agent.data.harmonize_drugs build    # 1c  drug identities (ChEMBL)
   uv run python -m combo_agent.data.harmonize_lines build    # 1d  cell-line identities (Cellosaurus)
   uv run python -m combo_agent.data.build_experiments        # 1e  one row per experiment
   uv run python -m combo_agent.data.ingest_cellminer         # 1f  NCI CellMiner downloads
   uv run python -m combo_agent.data.build_features           # 1f  drug and cell features
   uv run python -m combo_agent.data.freeze_pool              # 1g  split, thresholds, freeze
   ```

4. **Verify:** the sha256 values of the files in `data_cache/processed/` should match
   [`artifacts/m1_freeze/manifest.json`](artifacts/m1_freeze/manifest.json). The real-data tests
   should pass:

   ```bash
   COMBO_INTEGRATION=1 COMBO_OFFLINE=1 uv run pytest -q -k real
   ```

5. **M2 (seed sets and pair fame):** `uv run python -m combo_agent.lab.splits` writes
   `data_cache/processed/m2/`. It queries PubMed (about 5,000 counts). Put `NCBI_API_KEY=...` in
   `.env` for 10 requests/s instead of 3; the key never enters the cache. Add `--offline` to rebuild
   from `data_cache/raw/api_cache/pubmed/`. Check against
   [`artifacts/m2/manifest.json`](artifacts/m2/manifest.json).

**`data_cache/processed/labels.parquet` holds the hidden outcomes. Only the oracle
(`combo_agent.lab.oracle`) and the build/evaluation code allowlisted in
`tests/test_label_access.py` may read it.** Read the caveats in the M1 summary before using the data. In particular, SR's expression is
imputed, and synergy results are reported with and without the `syn_hit_low_kill` hits.

## CLI

`combo-agent` wraps the `sheldon` CLI, loads `.env`, and registers our benchmark first, so every
`sheldon` subcommand works:

```bash
uv run combo-agent ls
uv run combo-agent --help
```

## Files

```
src/combo_agent/
  run.py                 # Entry point to run the SHELDON CLI
  benchmark.py           # combo-mechanism benchmark (smoke harness for the tools)
  mechanism.py           # combo-mechanism-data benchmark (features in the prompt, no tools)
  data/                  # M1 data pipeline, steps 1a-1g (see docs/HANDOFF.md)
    curated/             # hand-reviewed tables (drug synonyms, cell lines, PubMed terms)
  lab/                   # M2: oracle.py (the simulated lab), splits.py (seed sets, pair fame)
  models/                # M3: gp.py (the default surrogate), surrogate.py (interface), prediction.py;
                         #     legacy/ (the first surrogate, a 5-network ensemble, kept to reproduce old runs)
  selectors/             # M4: baseline selectors (random, greedy, uncertainty, bo_ucb, bo_qei)
  eval/                  # M4: campaign runner (resumable grids) and metrics/statistics; M5 probe scoring
  literature/            # M5: PubMed client, leakage filter, search_literature, memory probe
  tools/                 # in-silico lookup tools for the agent (label-free)
tests/                   # unit tests, plus real-data tests (COMBO_INTEGRATION=1)
scripts/                 # drug-identity audit, M1 and M4 figures, data-bundle builder
artifacts/m1_freeze/     # frozen split, data card and manifest (small, tracked)
artifacts/m2/            # seed sets, M2 card and manifest (small, tracked)
artifacts/m3/            # response-model calibration report (small, tracked)
artifacts/m5/            # memory-probe report (small, tracked)
docs/
  HANDOFF.md             # start here
  CHANGELOG.md           # what changed, when
  handoff-template.md    # form for future handoffs
  target_design.md       # the project specification (milestones M1-M8)
  project-notes/         # M1 decisions, data notes, audit, walkthrough, data sharing, figures
  data-bundle/           # docs shipped inside the m1-data-v1 bundle (dictionary, splits, caveats, provenance)
  assignment.md, project-proposal.md
data_cache/              # all data (git-ignored); bundles are built into data_cache/bundles/
notes/                   # meeting notes
```

## Updating SHELDON

SHELDON is pinned to a commit in `pyproject.toml` (`[tool.uv.sources]`). To upgrade, change `rev`
and run `uv sync`.
