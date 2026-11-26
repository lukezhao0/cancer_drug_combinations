# Known issues and agent-builder rules (M1–M5)

> Status 2026-11-26. From a line-by-line review of the M1–M5 code against
> [`../target_design.md`](../target_design.md) and the milestone notes. M2–M5 were pushed **as built**:
> none of the issues below has been fixed yet. Each has an ID, a severity and the stage by which it
> must be fixed. When you fix one, mark it here and in the CHANGELOG.
>
> **Surrogate update (2026-11-26).** The Gaussian process (`models/gp.py`) is now the default
> surrogate; see [`gp-surrogate.md`](gp-surrogate.md). The 5-network ensemble is the legacy build
> (`models/legacy/`, tag `ensemble-v1`).
> - Issues that affect only the ensemble are marked **Legacy only**.
> - GP issues: C4, C8 and the new **G1–G4**.

## What was confirmed to work

These were re-run or recomputed during the review, not just read.

- **M1 rebuilds byte for byte.** An offline rebuild gives identical files in `data_cache/interim/`
  and `data_cache/processed/`.
- **The design §4 acceptance checks hold on the real data:**
  - no duplicate experiments, and `drug_a < drug_b` everywhere;
  - every line has a DepMap ID and a lineage;
  - the thresholds recompute (9.2989 / 62.9453);
  - the held-out split respects the families and decision 19.
- **Oracle refusals** are all correct and all-or-nothing.
- **Seed sets** reproduce, and the fame quartile boundary is right.
- **Every number in `runs/m4_main/report.md` recomputes** from the revealed experiments.
- **The M5 leakage gate (23/23 blocked, 19/19 passed) reproduces offline**, and the blocklist
  rebuilds byte for byte.
- **Tests pass:** `uv run pytest -q` gives 204 passed and 30 skipped (after the GP-default change), and the real-data tests
  (`COMBO_INTEGRATION=1`) pass.

---

## Agent-builder rules (follow until the linked issue is fixed)

1. **Never show `candidate_frame()` rows to the LLM or a tool as they are.**
   - The candidate table carries `n_reps`, which predicts synergy (A1).
   - Pass only the triplet keys, line metadata (lineage etc.) and model predictions.
2. **Treat `run_python` as able to read the hidden labels** (A2). Don't offer it in scored runs. Its
   guard is a tripwire, not a boundary.
3. **Selectors get a `SelectionContext`, never the `Oracle`** (`selectors/base.py`).
   - The agent selector should follow the same pattern as the baselines.
   - Nothing on the selection path should hold a reference to the oracle.
4. **Don't add files to the label-access allowlist** (`tests/test_label_access.py`) without a
   review. The test only catches the literal file name (A3). Reading `interim/experiments.parquet`,
   `*.reveals.parquet` or `oracle._Oracle__y` would also leak labels, and the test would not notice.
5. **Set temperature 0 explicitly on selection calls** (E4). SHELDON's client doesn't send a
   temperature, so the provider default applies.
6. **In masked-name runs, turn `search_literature` off** (or map drug names to codes in its output).
   Abstracts contain the real names (E7).
7. **Use the GP (the default) as the shared surrogate.** Don't pass `--surrogate ensemble` except to
   reproduce old runs. **Read the baseline numbers as provisional:**
   - the GP's lengthscale prior is mis-scaled (C4);
   - `bo_qei` aims at the wrong target (D1);
   - the GP reference grid has 5 seeds (G4).

   Fix these and rerun the GP grid before comparing the agent with the baselines. Give every new grid
   its own `--out` folder (D3).
8. **Don't rebuild or overwrite the published bundles** (`m1-data-v1`, `m2m5-addon-v1`). Any data
   change becomes a new version (B9).

---

## Issues

Severity:
- **Critical:** a label leak or a wrong result.
- **Major:** likely wrong, contradicts the design, or weakens a gate.
- **Minor:** robustness or best practice.

Fix-by stages, in order:
1. **Scored run:** before the first scored agent run (M6 gate).
2. **Compare:** before any agent-vs-baseline comparison.
3. **Report:** before the M8 report.
4. **Anytime.**
5. **Legacy only:** affects only the legacy ensemble (`--surrogate ensemble`). Fix it only if the
   ensemble is revived.

### A. Label safety and leakage

| ID | Sev. | Issue | Where | Suggested fix | Fix by |
|:--|:--|:--|:--|:--|:--|
| A1 | Critical | **`n_reps` is a pool column, so every untested candidate shows it, and it predicts synergy.**<br>Mean Bliss is 1.42 when `n_reps = 2` (7,435 selectable experiments, from 171 pairs) vs 0.004 when `n_reps = 1`. Design §5: `n_reps` is returned only on reveal.<br>Nothing uses it on candidates today; the GP uses it only on the history | `data/freeze_pool.py:176`, `lab/oracle.py:143`, `candidate_frame()` | Move `n_reps` into `labels.parquet` (`m1-data-v2`), or drop it from `candidate_frame()`. Add a test that candidates carry only keys, split and line metadata | Scored run |
| A2 | Critical | **The `run_python` subprocess can locate and open `labels.parquet`.**<br>The guard only refuses literal strings ("data_cache", "parquet"), and the path can be rebuilt from `sys.executable` | `tools/probes.py:21,50-62` | Keep the labels outside the agent process's filesystem view: a separate oracle process or service, Linux namespaces, or labels stored outside the readable tree | Scored run |
| A3 | Major | **The label-access test is a regex on one file name.**<br>It misses `"label" + "s.parquet"`, `_Oracle__y`, `glob("*.parquet")` and the other outcome files: `interim/experiments.parquet`, `interim/tdc_drugcomb.parquet`, `raw/drugcomb_nci60.pkl`, `interim/cross_study_overlap.parquet`, `processed/self_pairs.parquet`, `runs/**/*.reveals.parquet`.<br>The "rogue selector" test only checks the regex on a string | `tests/test_label_access.py:29-50`, `tools/store.py:30-35` | Keep one `OUTCOME_FILES` constant and scan for all of them plus `_Oracle__`. Add a runtime `sys.addaudithook` guard on `open` during campaigns. Make the rogue test scan a real file | Scored run |
| A4 | Major | **The tool-output leak scan passes by construction.**<br>It scans synthetic calls, not run logs, and leaves out `python_context` and `search_literature`. It matches only 3-decimal numbers, while the tools round to 2 | `tests/test_tools.py:225-286` | Scan real tool-call logs. Flag any output that names a triplet together with a number for it | Scored run |
| A5 | Minor | **Agent tools reveal which lines are held out, and their features** (`get_cell` shows `split`; `find_lines(split="heldout")`). No outcomes leak, but §9 says predictions are frozen before anyone looks at held-out lines | `tools/omics.py:93,108-120`, `tools/probes.py:76` | Team decision: hide them, or document that this is allowed | Scored run |

### B. M1 data

| ID | Sev. | Issue | Where | Suggested fix | Fix by |
|:--|:--|:--|:--|:--|:--|
| B1 | Major | **Pemetrexed carries an *E. coli* target** (CHEMBL2555, `thyA`). It becomes the feature column `target_thyA` and is visible to the tools and the probe prompts | `data/harmonize_drugs.py:452-469`, `data/chembl.py:103-110` | Keep only `organism == "Homo sapiens"` targets, plus non-protein targets such as DNA (`m1-data-v2`) | Report |
| B2 | Minor | **The syn threshold is stored rounded (9.2989).** One experiment (y_syn = 9.298902) is a hit for `Oracle.is_hit` but not in `labels.is_hit_syn` | `data/freeze_pool.py:235`, `lab/oracle.py:157,192` | Store full precision (`m1-data-v2`) | Report |
| B3 | Minor | **11 of 57 lines never had their expression checked against CellMiner's arrays.** These include CCRF-CEM (held out) | `data/build_features.py:152-154,313-315` | Check every line against `expression_array` | Anytime |
| B4 | Minor | **NCI-H322M uses its parent's mutation calls (TP53 only).** Its own CellMiner exome also calls KEAP1, NF1 and BRCA1 (not independently checked) | `data/build_features.py:340-346` | Use the union of parent and own calls, or add a caveat | Anytime |
| B5 | Minor | **No "MoA class" drug feature** (design §4 step 6). 30 drugs have no target gene | `data/build_features.py:256-281` | Add a curated drug-class column | Report |
| B6 | Minor | **`mut_IDH1` and `mut_MAP2K1` are 0 in all 57 lines** | `data/build_features.py:60` | Drop them from the M7 rule vocabulary | Anytime |
| B7 | Minor | **DepMap CSVs are not checked by sha256 when read** | `data/build_features.py:101,112,490` | Verify against `depmap-data.md` | Anytime |
| B8 | Minor | **Cached converted parquets are reused without checking where they came from**, and the TDC converter writes in place | `data/ingest_tdc.py:183`, `data/ingest_cellminer.py:134`, `data/_tdc_pickle_convert.py:69` | Write to a temp file and rename; record each output's sha256 | Anytime |
| B9 | Minor | **`make_data_bundle.py` rebuilds `m1-data-v1` in place by default.** The archive isn't deterministic | `scripts/make_data_bundle.py:292-333` | Refuse an existing version; fix tar mtimes | Anytime |
| B10 | Minor | **`structure_key` is NaN for the platinum drugs** (pandas 3 turns None into NaN) | `data/harmonize_drugs.py:560` | Choose with `pd.isna` | Anytime |
| B11 | Minor | **The report text says the target-less drugs are "dropped at freeze"**; decision 8 keeps them | `data/harmonize_drugs.py:596-597` | Fix the message | Anytime |
| B12 | Minor | **30 repeated rows in `drug_targets.parquet`** (same mechanism, different `mec_id`) | interim table | Drop duplicates | Anytime |
| B13 | Minor | **Stale hard-coded numbers:** "45 lines, r = 0.95" vs 56 / 0.941; the audit script uses thresholds 9.26 / 63.3 | `scripts/make_m1_figures.py:95,144`, `data/build_features.py:191`, `scripts/audit_drug_identity.py:90` | Read them from the reports | Anytime |
| B14 | Minor | **Download and cache robustness:** the TDC download doesn't retry on timeouts or 5xx; HTTP 400 is cached as a permanent "no result"; cache entries have no fetch date | `data/ingest_tdc.py:141`, `data/api_cache.py:49-90` | Retry; add timestamps | Anytime |

Note for M7: SF268 (held out) is flagged "possibly misidentified" by Cellosaurus. Mention this
when its predictions are reported.

### C. M2–M3: oracle, model and rules

| ID | Sev. | Issue | Where | Suggested fix | Fix by |
|:--|:--|:--|:--|:--|:--|
| C1 | Major | **Z-scoring sparse binary features creates fake novelty.**<br>A fingerprint bit set in 1 of 4,939 pairs becomes ≈ 70 after scaling; 44% of pairs have some input above 10. The top 0.1% of candidates by std all contain such an input. This is the cause of the "line sweep" noted in M3 §5 | `models/legacy/ensemble_features.py` (`pair_block`, `_standardize`) | Centre binary bits without dividing by their SD, or clip, or use fingerprint PCs. Then rerun M4 | Legacy only |
| C2 | Major | **Early stopping on about 6 validation rows.**<br>At n = 32, 13 of 30 members stop after ≤ 2 epochs, essentially their random starting weights. Their disagreement is 75–90% of the variance on the top UCB candidates. The noise estimate comes from the same 6 rows | `models/legacy/ensemble.py` (tag `ensemble-v1`: `models/response_model.py:203,236-252`) | Fixed or cross-validated epoch count, then refit on all rows (or bagging); estimate noise from out-of-fold residuals | Legacy only |
| C3 | Major | **The M3 coverage gate is nearly circular.** The noise is fitted so ±1.28 SD covers 80% of held-out residuals, and noise is 70–85% of the variance on random points | `models/legacy/ensemble.py` (tag `ensemble-v1`: `models/response_model.py:249-250`), `models/legacy/calibrate.py` (tag `ensemble-v1`: `models/calibrate.py:69`) | Also report coverage on selected batches and the latent share | Legacy only |
| C4 | **Major, top GP priority** | **The GP cell-lengthscale prior assumes inputs in [0, 1], but the PCs are standardized.** At n = 32, all 16 lengthscales sit at the prior mode (0.82), so the synergy cell kernel is close to the identity | `models/gp.py:137-139,173` | Normalize the PCs or rescale the prior; consider a line-baseline kernel term | Compare |
| C5 | Minor | **Weight decay has no effect** (lr·wd = 1e-7 per step) | `models/legacy/ensemble.py` (tag `ensemble-v1`: `models/response_model.py:229`) | A real decay or dropout | Legacy only |
| C6 | Minor | **`embed()` averages hidden units across members**, which aren't aligned | `models/legacy/ensemble.py` (tag `ensemble-v1`: `models/response_model.py:296-302`) | Concatenate them (changes `bo_ucb` picks) | Legacy only |
| C7 | Minor | **The member-spread variance uses ddof = 0** with 5 members | `models/legacy/ensemble.py` (tag `ensemble-v1`: `models/response_model.py:279`) | ddof = 1 | Legacy only |
| C8 | Minor | **GP:** predicted noise ignores `n_reps`, and prior variance depends on pair similarity. This drives GP uncertainty sampling to near-duplicate pairs | `models/gp.py:304-309,456` | Normalize the pair kernel | Compare |
| C9 | Minor | **The swap test can't fail:** `pair_code` makes (A,B) and (B,A) the same integer. Symmetry still holds by construction | `models/legacy/calibrate.py` (tag `ensemble-v1`: `models/calibrate.py:114-119`), `models/legacy/ensemble_features.py` (`pair_code`) | Test the dense path with swapped rows | Legacy only |
| C10 | Minor | **Timing wording is off.** The note's "refit + full-pool scoring 11.5 s" is fit time only (scoring adds about 1.7 s). 15 of 2,000 in-campaign rounds exceeded 60 s under machine load | `m3-response-model.md`, `runs/m4_main` | Reword | Legacy only |
| C11 | Minor | **Expression tertiles break ties by row order** (`rank(method="first")`). This matters for genes with many zeros in M7 | `tools/rules.py:100-106` | `method="average"`, or reject genes whose tied values cross a tertile cut | Report |
| C12 | Minor | **The rule grammar accepts repeated or contradictory terms**, and keeps the user's lineage casing | `tools/rules.py` | Reject repeats; canonicalize | Report |
| C13 | Minor | **The oracle's JSONL log is never enabled in campaigns**, and refusals aren't logged | `eval/runner.py:130`, `lab/oracle.py:123,230` | Pass `log_path`; log refusals | Scored run |

### D. M4: selectors, runner and metrics

| ID | Sev. | Issue | Where | Suggested fix | Fix by |
|:--|:--|:--|:--|:--|:--|
| D1 | Major | **`bo_qei` improves on the best value revealed so far, not on the hit threshold.**<br>The seed-set maximum is already far above the threshold (syn max 7.0–48.6 vs 9.30), but the endpoint counts hits. Its "joint worlds" are only 5 members plus noise | `selectors/baselines.py:137-169` | Add a hit-aware acquisition (probability of hit, or EI with the threshold as reference); keep the current qEI as the literal-design variant | Compare |
| D2 | Major | **Run manifests can't identify the code.**<br>They record `ac79169`, dirty, while M4 was not yet committed. No diff or source hash; the feature tables and the scipy version aren't recorded | `eval/runner.py:238-249` | Commit before a grid run, or record `git diff` and a hash of `src/`; hash `drugs`/`cells` | Compare |
| D3 | Major | **Resume skips existing runs without checking settings, and overwrites the manifest** | `eval/runner.py:208-209,274-275` | Store a settings hash per run; refuse to resume on a mismatch | Compare |
| D4 | Major | **No config file per run** (design §1, §3); grids are code presets plus CLI flags | `eval/runner.py:71-77,297-315` | A YAML or pydantic config, copied into the run folder | Compare |
| D5 | Major | **Decision 20 and §11 are only partly met.** "With and without low-kill hits" exists only for hits@320, and the fame split only for hits@320 | `eval/metrics.py:83-94` | Compute every metric for `syn` and `syn_strict`, and by fame | Report |
| D6 | Major | **Hit diversity counts drug pairs only**; §11 also asks for target-pair classes | `eval/metrics.py:88`, `eval/runner.py:109` | Add it (needs B5) | Report |
| D7 | Minor | **The gate reports "FAIL" with 2 seeds** even when the effect is large; p < 0.05 is unreachable at that size | `eval/metrics.py:161-171` | Report "too few seeds" | Report |
| D8 | Minor | **Censored runs enter experiments-to-10 at 320, uncounted**; resolution is ±15 experiments | `eval/metrics.py:70-74,141` | Report the censored count; use a median or Kaplan–Meier | Report |
| D9 | Minor | **Selection seed `sum(map(ord, name))`**: anagram names collide | `eval/runner.py:132` | `zlib.crc32` | Anytime |
| D10 | Minor | **UCB uses predictive std (model + noise)**, not model uncertainty alone (both surrogates) | `models/prediction.py` (`Prediction.std`) | Sensitivity run with `latent_std` | Compare |
| D11 | Minor | **`--workers` sets thread variables after numpy is imported** (a problem on Linux with fork) | `eval/runner.py:282-284` | spawn context or `threadpoolctl` | Anytime |
| D12 | Minor | **Seed-set hits aren't in `report.md`** | `eval/metrics.py:174-201` | Add a column | Report |

### G. The GP surrogate (default since 2026-11-26)

| ID | Sev. | Issue | Where | Suggested fix | Fix by |
|:--|:--|:--|:--|:--|:--|
| G1 | Major | **The GP has never run the formal M3 gate** (80% coverage on the 80/20 sanity split, swap test, refit < 60 s). `calibrate.py` is ensemble-only (`models/legacy/`). The evidence so far is in-campaign: ±1σ covers 68–70% on random probes, and fits take seconds | `models/legacy/calibrate.py` | A GP version of the calibration report (or make `calibrate` surrogate-agnostic) | Compare |
| G2 | Major | **The GP concentrates picks in a few responsive lines.** GP greedy (killing) puts 98% of its picks in one line, and GP `bo_ucb` (killing) finds hits in 10 lines vs 18 for the ensemble. Hit counts reward this | `gp-surrogate.md` §9.4 | Report distinct hit pairs and lines covered next to hits (with D6); consider a per-line cap | Compare |
| G3 | Major | **The GP's synergy advantage is not established**: a tie within noise (§9.1), and `bo_ts` is worse. The comparison recommended the GP for killing only, until C4 is fixed and re-raced | `gp-surrogate.md` §9.1, §10 | Re-race synergy after C4 | Compare |
| G4 | Minor | **The GP reference grid has seeds 0–4 only.** Its outputs (`runs/gp_compare/`) are not in git, and its `random` arm is reused from the ensemble grid (`random` uses no surrogate) | `runs/gp_compare/` | Run `--grid main --out runs/m4_gp_main` on seeds 0–9 after C4 | Compare |

### E. M5: literature, leakage filter and probe

| ID | Sev. | Issue | Where | Suggested fix | Fix by |
|:--|:--|:--|:--|:--|:--|
| E1 | Major | **`pair_and_line` misses common spellings:**<br>• drug names followed by a hyphen ("cisplatin-resistant");<br>• NCI-H lines without the prefix ("H460");<br>• Unicode hyphens ("MDA‐MB‐231");<br>• brand names and abbreviations (Gleevec, Taxotere, DOX, PTX, CBDCA) | `literature/leakage.py:102-120`, `data/curated/drug_pubmed_terms.csv` | Allow a following hyphen; make "NCI-" optional; normalize hyphens; build a separate filter lexicon | Scored run |
| E2 | Major | **The gate test set never triggers `pair_and_line`.** 7 of 16 reanalyses are caught only by `synergy_ml`, which was added after looking at the set. The negatives are all easy | `tests/fixtures/m5/`, `scripts/make_m5_testset.py` | Add pair-and-line-only positives and hard negatives | Scored run |
| E3 | Major | **No M5 unit tests.** `tests/test_leakage.py` doesn't exist, and the audit-record half of the gate is untested | — | Add `test_leakage.py` and `test_retrieval.py` | Scored run |
| E4 | Major | **Probe calls send no temperature, and the cache key has none.** Design §3/§8: temperature 0, cache key (prompt hash, model, temperature) | `literature/probe.py:124`; SHELDON `core/llm.py` | Pass temperature 0 explicitly | Scored run |
| E5 | Major | **"The LLM confuses synergy with potency" may partly be a prompt artifact.** Both outcomes are asked one right after the other | `literature/prompts/memory_probe_v1.md` | Run a synergy-only probe before relying on the claim | Report |
| E6 | Minor | **The probe sample is split equally by fame, not drawn at random** (the pool is 25/48/27%). With only 7 syn and 10 inh hits among the 200, the top-10% hit rates are uninformative | `literature/probe.py:58-76` | State it, or reweight | Report |
| E7 | Minor | **Masked-name runs are unmasked by `search_literature`** | `tools/registry.py:84` | Turn it off or mask it in masked runs | Scored run |
| E8 | Minor | **PubMed relevance ranking drifts regardless of the date cutoff**, so only the cache makes replays reproducible. Error bodies are cached before the code raises | `literature/pubmed.py:84`, `data/api_cache.py:86-91` | Fix the doc; validate before caching | Anytime |
| E9 | Minor | **`LiteratureSearch` isn't thread-safe** | `literature/retrieval.py:62-69` | A lock | Scored run |
| E10 | Minor | **Broad text rules ("NCI-60" + any combination word) may over-block** | `literature/leakage.py:80-81` | Measure from the M6 audit logs | Report |
| E11 | Minor | **The `eval/__init__.py` docstring says no eval module reads outcomes**; `memory_probe.py` and `compare.py` do (they're allowlisted) | `eval/__init__.py:6-7` | Fix the docstring | Anytime |

### F. Packaging and docs

| ID | Sev. | Issue | Suggested fix | Fix by |
|:--|:--|:--|:--|:--|
| F1 | Minor | **`torch`, `pydantic` and `linear_operator` are imported but not declared** in `pyproject.toml` (they arrive only through other packages) | Declare them | Anytime |
| F2 | Minor | **The venv is Python 3.13; design §3 says 3.11** | Pin it, or update the design | Anytime |
| F3 | Minor | **Docs and config mismatches:**<br>• `configs/mechanism.yaml` has `cache: false` (design §3: cache every call);<br>• `mechanism.py:59` names a missing script, `scripts/prepare_sandbox_data.py`;<br>• `in-silico-tools.md` still says `search_literature` has no implementation | Fix | Anytime |
| F4 | Minor | **`tests/fixtures/m5/leakage_testset.jsonl` contains 42 full PubMed abstracts.** Fine for a private repo | If the repo goes public, keep only PMIDs and titles | Before going public |
