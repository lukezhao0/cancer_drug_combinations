# Changelog

Newest first. Decision numbers refer to the table in
[`project-notes/m1-freeze-and-splits.md`](project-notes/m1-freeze-and-splits.md) §2.

## 2026-11-26: the Gaussian process becomes the main surrogate; the ensemble is the legacy build

Details: [`project-notes/gp-surrogate.md`](project-notes/gp-surrogate.md).

- **Default surrogate is now the GP** (`models/gp.py`) for every campaign:
  - `--surrogate` defaults to `gp` in `eval/runner.py` (CLI, `run_campaign`, `run_one`, `run_grid`,
    `write_manifest`);
  - `SURROGATES = ("gp", "ensemble")`;
  - a new test pins the default.
- **Why** (100 campaigns, seeds 0–4):
  - killing: the GP finds 1.8–3.3× the ensemble's hits with every BO rule, on 5/5 seeds (`bo_ucb`
    202.2 vs 60.8);
  - calibration: the GP's error bars are honest (±1σ covers 68–70% vs 84–87%);
  - synergy: a tie.
- **The ensemble is the legacy build:**
  - **New location:** `models/legacy/` (`ensemble.py`, `ensemble_features.py`, `calibrate.py`); it runs
    only with `--surrogate ensemble`.
  - **Git tag** `ensemble-v1` marks its last state as the default. `runs/m4_main` was made with it.
  - **Shared code:** `Prediction` and `cosine_similarity` moved verbatim to `models/prediction.py`;
    `models/features.py` keeps only the shared helpers.
  - **Tests:** `tests/test_response_model.py` is now `tests/test_legacy_ensemble.py`.
- **No numbers changed** (verified): `bo_ucb` × synergy × seed 0 rerun with the new default is
  row-for-row identical to `runs/gp_compare/gp`, and with `--surrogate ensemble` identical to
  `runs/m4_main`. Tests: 204 passed, 30 skipped; M2–M5 real-data tests 109 passed.
- **Known issues re-tiered:**
  - ensemble-only items are marked "Legacy only";
  - GP issues are C4 (prior scaling, now top priority), C8, and new G1–G4: no formal GP M3 gate, line
    concentration, synergy advantage not established, a 5-seed reference grid only.
- **Run new grids into new folders** (for example `--out runs/m4_gp_main`): resume skips existing files
  without checking the surrogate (D3).

## 2026-11-26: M2–M5 handoff; code review; M2/M5 add-on data bundle

- **Code review of M1–M5** against the design. Results in
  [`project-notes/known-issues.md`](project-notes/known-issues.md): about 60 issues, each with an ID,
  severity, location and "fix by" stage, plus **agent-builder rules** to follow until they're fixed.
  No code was changed: M2–M5 are pushed as built.
  - **Confirmed:**
    - M1 rebuilds byte for byte;
    - every number in `runs/m4_main/report.md` recomputes from the reveals;
    - the M5 gate reproduces offline;
    - `uv run pytest -q` gives 203 passed and 30 skipped.
  - **Most important open issues:**
    - **A1:** `n_reps` is visible in the candidate table and predicts synergy.
    - **A2:** `run_python` can reach `labels.parquet`.
    - **C1, C2:** ensemble uncertainty is inflated by feature scaling and early stopping, so the M4
      numbers are provisional.
    - **D1:** `bo_qei`'s reference point is the best value revealed so far, not the hit threshold.
    - **E1–E4:** leakage-rule spelling gaps; no M5 tests; no temperature on probe calls.
- **New data bundle `m2m5-addon-v1`** (Google Drive, beside `m1-data-v1`, which is unchanged):
  - **Contents:** M2 seed sets, pair fame and drug PubMed counts; the M5 blocklist. All
    outcome-free, checked at build time.
  - **Checksums:** `m2m5-addon-v1.tgz` sha256
    `610d3862af31fa1bb7b7d5c3870ad6fe6b1cabe038dc4cdda88d11c858614e15`; the optional
    `m2m5-addon-v1_api_cache.tgz` (PubMed responses) sha256
    `507909a9b5f826cad7dcb0dafad981d2a1835c31b3782cc70890747ab64e0a66`.
  - **Build:** deterministic, so the same inputs give the same checksum.
- **New docs:** [`HANDOFF-m2-m5.md`](HANDOFF-m2-m5.md) covers status, where the agent plugs in, data
  install and next steps. Status banners were added to `README.md`, `HANDOFF.md` and the M3/M4/M5
  notes.

## 2026-11-25: M5 (literature search, leakage filter, memory probe) built

Details: [`project-notes/m5-literature.md`](project-notes/m5-literature.md). Probe report:
`artifacts/m5/memory_probe.md`.

- **`src/combo_agent/literature/`:**
  - `pubmed.py`: cached E-utilities client with the M2 date cutoff (2026/11/01); offline replay
    with `COMBO_OFFLINE=1`; the API key is never cached.
  - `leakage.py`: `LeakageFilter` with six rules (`source_paper`, `cites_source`, `screen_name`,
    `pair_and_line`, `synergy_ml`, optional `strict_date`), and the blocklist build, which checks
    each curated PMID's title against PubMed.
  - `retrieval.py`: `search_literature(query, k)`. It writes one audit record per call (every PMID
    retrieved, shown or blocked, and why) before returning. Registered in `tools/registry.py`.
  - `probe.py` and `prompts/memory_probe_v1.md`: the zero-shot memory probe (no tools; masked
    variant replaces drug names with codes).
- `eval/memory_probe.py`: probe scoring (Spearman ρ with bootstrap CI, by fame group); added to the
  label-access allowlist.
- New curated table `data/curated/leakage_sources.csv` (7 sources). Blocklist: **607 PMIDs**
  (604 via citation), in `data_cache/processed/m5/`.
- `scripts/make_m5_testset.py` → `tests/fixtures/m5/`: 42 real PubMed records, labelled by hand.
- **Leakage gate: passes.** 23/23 leaky papers blocked (7 sources, 16 reanalyses); 0/19 clean
  papers lost. Checked by script over the fixtures; `tests/test_leakage.py` is still to be written.
- **Memory probe (`gemini-3.5-flash-lite`, 200 experiments, $0.02 per run):**
  - synergy ρ −0.05 (95% CI −0.19 to 0.08), famous −0.01, obscure −0.04, masked −0.10: **no
    detectable memory of synergy results**;
  - killing ρ 0.40 (0.29 to 0.51), famous 0.36, obscure 0.32, masked 0.33: general potency
    knowledge, not recalled results;
  - the LLM's synergy and killing guesses agree at ρ 0.81, while the true outcomes don't
    (ρ 0.01): **it confuses synergy with potency.**
  - Decision: not repeated on `gemini-4-hydrogen`. The fame split and masked ablation in M6 remain
    the controls for whichever model is reported.
- **Open:** `tests/test_leakage.py`; on macOS the `run_python` sandbox can still reach
  `labels.parquet`, which must be fixed before scored agent runs.

## 2026-11-24: M4 grids run; gate passed

Report: `runs/m4_main/report.md`, figures in `runs/m4_main/figures/` (both untracked under `runs/`).

- `pilot` and `main` grids run (`main`: 5 selectors × 2 targets × 10 seeds = 100 campaigns).
- **Gate: PASS.** `bo_ucb` beats `random` on hits at 320: killing 61.6 vs 17.0, synergy 53.0 vs
  16.2 (Holm-corrected one-sided Wilcoxon p = 0.0039 for both).
- Hits at 320 (killing / synergy): greedy 94.8 / 73.0, `bo_ucb` 61.6 / 53.0, `bo_qei` 60.1 / 38.3,
  uncertainty 25.9 / 20.3, random 17.0 / 16.2.
- **`greedy` is the strongest baseline on both targets**, though more variable across seeds. The
  agent comparison should report `greedy` as well as `bo_ucb`; decide this before any agent run.
- For killing, model-based selectors drift toward famous pairs (greedy: 54.4 famous vs 10.3 obscure
  hits); random does not (4.3 vs 5.6).
- Not run: the `ablation` grid and the budget/batch-size sweeps.

## 2026-11-23: M4 (baselines, runner, metrics) built; not yet run

Details and run commands: [`project-notes/m4-baselines.md`](project-notes/m4-baselines.md).

- **`src/combo_agent/selectors/`:** `random`, `greedy`, `uncertainty`, `bo_ucb` and `bo_qei`,
  plus ablation variants (`bo_ucb_b1`, `bo_ucb_b3`, `bo_ucb_nodiv`).
  - `bo_ucb` is UCB (β = 2) over a 2,000-candidate shortlist, chosen by maximal marginal relevance
    on model embeddings (α = 0.5). A z-score version failed its synthetic diversity test.
  - `bo_qei` is Monte-Carlo batch expected improvement over the ensemble members, chosen
    greedily, without BoTorch.
- **`src/combo_agent/eval/runner.py`:** campaigns with a model refit per round, writing round and
  reveal files atomically. Grids resume, with `pilot`/`main`/`ablation` presets, a run manifest
  and an optional `--workers`. No label access.
- **`src/combo_agent/eval/metrics.py`:** hits @80/160/320, AUC, experiments-to-10, hit pairs,
  cross-outcome yield, fame split, strict synergy, bootstrap CIs, paired Wilcoxon and Holm. It
  writes `results.parquet`, `per_run.parquet`, `summary.parquet` and `report.md` with the gate
  verdict.
- `scripts/make_m4_figures.py`: discovery curves, hits at 320 and the fame split, drawn in an
  isolated environment.
- `EnsembleModel.member_means()` gives per-member predictions for qEI.
- Tests: `tests/test_m4.py` (17 synthetic) and a `member_means` test. 183 unit tests pass. The real
  grids haven't been run.

## 2026-11-23: M3 (response model) complete

Details: [`project-notes/m3-response-model.md`](project-notes/m3-response-model.md). Report:
`artifacts/m3/m3_report.md`.

- **`src/combo_agent/models/`:**
  - `features.py`: symmetric pair block [f(A)+f(B), f(A)×f(B)] plus a cell block; `full`
    (design) and `compact` (PCA) modes; label-free.
  - `response_model.py`: `EnsembleModel` with `fit(history)`, `predict` and `embed`. Five
    2×256 MLPs in numpy, with hand-written backprop (gradient-checked). The first layer is
    factorized per pair and per line.
  - `calibrate.py`: the M3 report (allowlisted label reader).
- **Deviation from design §6:** no learned per-row variance heads. At 32–352 rows the variance
  collapsed (planted test: held-out likelihood −0.1 → 84). The noise scale now comes from each
  member's held-out residuals, at the 80th-percentile |residual|, because the outcomes are
  heavy-tailed.
- **Gate:** sanity coverage 0.85–0.89; swap difference 0; refit + pool scoring 3.7 s (compact) /
  11.5 s (full). Low-data coverage is 0.83–0.90 at n = 32/96/320. Default mode is `full`.
- **Results at n = 320 (full):**
  - killing r = 0.51, top-1% enrichment 7.3×;
  - synergy r = 0.13, enrichment 2.7×, with an RMSE ratio > 1;
  - with 175k rows: synergy r = 0.71, killing r = 0.90.
- **M4 warning:** early UCB picks extreme extrapolations, concentrated on a few pairs (demo: 3
  pairs in 16 picks), so the diversity penalty is required.
- Tests: 165 unit tests pass; 194 with `COMBO_INTEGRATION=1`.

## 2026-11-23: M2 (oracle and splits) complete

Details: [`project-notes/m2-oracle.md`](project-notes/m2-oracle.md).

- **`src/combo_agent/lab/oracle.py`:** `Oracle` with `candidates`, `run`, `history`, `budget_left`
  and `is_hit`. A batch is refused as a whole if it's oversized, has a duplicate, re-runs an
  experiment, isn't in the pool, uses a held-out line or exceeds the budget. Pair order is
  canonicalized; the seed set is free at round 0; there's a JSONL reveal log.
- **`src/combo_agent/lab/splits.py`:** 10 seed sets of 32 selectable triplets, and
  famous/middle/obscure labels for all 4,939 pairs from PubMed co-mention counts (date cutoff
  2026/11/01). The responses are cached, so offline rebuilds are byte-identical.
  - New reviewed table: `data/curated/drug_pubmed_terms.csv`.
  - Outputs go to `data_cache/processed/m2/`, with tracked copies in `artifacts/m2/`.
- **Results:**
  - famous 1,249 pairs (≥ 22 records), middle 2,362, obscure 1,328;
  - synergy hit rates are 4.8–5.3% in every class, so fame doesn't make a pair an easier target;
  - seeds 6 and 9 start with no synergy hit.
- **`tests/test_label_access.py`:** only allowlisted files may name `labels.parquet`.
- **Tools:**
  - `FeatureStore` now reads each gene table once instead of once per gene lookup. `get_cell`
    went from 6.8 s to milliseconds, and the real-data leak test from >10 min to 4 s;
  - the leak scan masks Mann–Whitney p-values, which produced false 3-decimal "matches".
- Tests: 158 unit tests pass; 185 with `COMBO_INTEGRATION=1`.

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
