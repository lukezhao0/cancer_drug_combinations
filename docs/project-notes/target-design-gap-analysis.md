# Target design: gap analysis and next steps

- **Written:** 2026-11-08.
- **Compares:** [`../target_design.md`](../target_design.md) (the plan) against the code repo at commit
  `78f71b5` and SHELDON at the pinned commit `7fbf027`.
- **Convention:** every claim is marked as one of:
  - **[verified]**: I checked it in code or data files, and the evidence is given.
  - **[unverified]**: needs a run or a source I couldn't check.
  - **[recommendation]**: my opinion as a colleague; for the team to decide.
- **Background:** if the vocabulary is new, read [concepts-primer.md](concepts-primer.md) first.

---

## 0. Status update (2026-11-08, end of day): M1 complete

The rest of this document was written at the **start** of M1. Since then **M1 has been built and
frozen** (not yet committed; see [m1-freeze-and-splits.md](m1-freeze-and-splits.md)). How each §2
problem was resolved:

| § | Problem | Resolution in M1 |
|:--|:--|:--|
| 2.1 | `tdc` package can't be installed | Not used; files are downloaded directly, hash-pinned (1a) |
| 2.2 | Old-pandas pickles | Converted once with a restricted unpickler in an isolated env (1a) |
| 2.3 | OncoPolyPharmacology has no CSS/Bliss | Deferred: O'Neil is the second benchmark, outside the NCI-60 pool (decisions 4, 13) |
| 2.4 | "DrugComb = NCI-ALMANAC" | **Corrected:** the file mixes ALMANAC, O'Neil and a 10-row third study. Pool = ALMANAC only (decision 13) |
| 2.5 | Messy drug identifiers | ChEMBL parent ID; name and structure must agree; reviewed `drug_synonyms.csv` (1c) |
| 2.6 | The label-access rule | `labels.parquet` is separate from `pool.parquet`; the oracle test lands in M2 |
| 2.7 | Thresholds: full vs selectable pool | Selectable pool: y_syn ≥ 9.30, y_inh ≥ 62.95 (1g) |
| 2.8 | Missing §8 diagram | Still missing in the design; the text reconstruction stays in concepts-primer |
| 2.9 | Layout mismatches | Code lives in `src/combo_agent/data/`; data in a git-ignored `data_cache/` |
| 2.10 | No `study_id` | One constant, `NCI-ALMANAC` (decision 12) |
| 2.11 | Thresholds depend on the split | The split moved into M1 and was frozen together with the thresholds (1g) |
| 2.12 | Where labels live | Separate `labels.parquet` (decision 3) |
| 2.13 | Byte-for-byte data card | Done: two runs give identical outputs, dates go in `manifest.json` |

**New problems found during M1**, all handled and documented:
- O'Neil and ALMANAC disagree on the same experiments (CSS r = 0.22).
- CellMiner's SR RNA-seq sample is a mix-up, so SR's expression is imputed.
- CellMiner misses tumour-suppressor mutations, so mutations come from DepMap.
- Bliss is noisy (replicate r = 0.21), and synergy hits that barely kill occur at 7.1%
  (decision 20: report both ways).
- No held-out line carries NRAS, HRAS, ERBB2, CTNNB1, RB1 or ATM mutations, which limits testability.

The status table in §1 below is historical: M1 is now ✅. Next is **M2** (oracle, seed sets,
famous/obscure labels).

---

## 1. Where we are: one-screen status

```
target_design milestones          status in repo                        effort guess*
──────────────────────────────    ──────────────────────────────────    ────────────
M0 (implied) scaffolding          ✅ package, CLI, SHELDON pin, stub       done
M1 Data and pool                  ⬜ nothing (no data, no loaders)          large
M2 Oracle and splits              ⬜ nothing                                small
M3 Response model                 ⬜ nothing                                medium
M4 Baselines and runner           ⬜ nothing                                medium
M5 Retrieval and leakage filter   ⬜ nothing                                medium-large
M6 Agent (SHELDON loop)           ⬜ only the SHELDON dependency exists     large
M7 Explanations and predictions   ⬜ nothing                                medium
M8 Full grid and report           ⬜ nothing                                medium (+ LLM cost)
```

\*Effort guesses are [recommendation]-grade judgment, not measurements.

[verified] The repo has these pieces:

- `src/combo_agent/run.py`: a CLI wrapper that loads `./.env`, registers our benchmark, and hands off
  to SHELDON's CLI.
- `src/combo_agent/benchmark.py`: a stub SHELDON benchmark, `combo-mechanism`. `load()` returns `[]`
  and `score()` raises `NotImplementedError`.
- `pyproject.toml`: depends only on `pandas>=3.1.0` and `sheldon`.

There are no tests, configs, data, `runs/`, or `.venv`, and `uv` isn't installed on this machine.
Nothing has been executed yet.

**Bottom line:** we are at the very start, about 0 of 8 milestones. The good news is the target
design is unusually concrete: it has interfaces, acceptance checks, and gates. So the work now is
mostly *engineering against a spec*, with a few spec bugs to fix first (§2).

---

## 2. Problems found in the target design (fix before or during M1)

These are things in `target_design.md` that conflict with reality or with itself. Each one is cheap
to fix now and expensive to discover in M4 or M6.

### 2.1 The `tdc` package can't be installed in our environment [verified]

- PyPI metadata for the latest `PyTDC` (1.1.15, uploaded 2025-03-31) pins:
  - `pandas<3.0.0,>=2.1.4`
  - `numpy<2.0.0,>=1.26.4`
  - `rdkit<2024.3.1`
  - `transformers<4.51`
  - plus `cellxgene-census`, `tiledbsoma`, `accelerate`, and `datasets`.
- Our `pyproject.toml` requires `pandas>=3.1.0`. These can't be satisfied together, so §4 step 1
  ("Pull the TDC datasets with the `tdc` package") fails as written.
- **What the TDC loader actually does** [verified, read from the PyTDC 1.1.15 sdist
  `tdc/metadata.py` and `tdc/utils/load.py`]: it downloads one pickle file per dataset from Harvard
  Dataverse at `https://dataverse.harvard.edu/api/access/datafile/<id>`.

  | TDC name | Dataverse id | File | Size (2026-11-08) | Rows (TDC metadata) |
  |:--|:--|:--|--:|--:|
  | `drugcomb` | 4215720 | `drugcomb_nci60.pkl` | 36,394,069 B | 297,098 |
  | `oncopolypharmacology` | 4167358 | `oncopolypharmacology.pkl` | 1,621,643,413 B | 23,052 |

- [recommendation] Don't depend on `tdc` at all. `ingest_tdc.py` should:
  1. download those two files directly;
  2. check their sha256. `drugcomb_nci60.pkl` was
     `c0e195c943d5a238e27e0a3e389c4637f9ac0212309c250f9d823b7196ad00bf` on 2026-11-08;
  3. convert them **once** to parquet in an isolated, throwaway environment
     (`uv run --isolated --with 'pandas<3' --with 'numpy<2' --with pyarrow …`);
  4. use only the parquet from then on.

  This also satisfies the design's "save raw copies with download date and file hash."

### 2.2 The TDC files are old-pandas pickles [verified + unverified]

- [verified] A static opcode scan (`pickletools`, which parses without executing) shows they
  reference `pandas.core.internals.managers.BlockManager` and `numpy.core.multiarray._reconstruct`,
  i.e. pandas 1.x/2.x-era internals.
  - The 36 MB file references only pandas, numpy, and `builtins.slice` constructors.
- [unverified] Whether pandas 3.1 with numpy 2 can unpickle them. This is another reason for the
  one-time isolated conversion in §2.1.
- [verified] Unpickling executes code. Treat raw pickles as untrusted input: convert them once, in
  isolation (done in M1 step 1a with a restricted unpickler).

### 2.3 OncoPolyPharmacology has no CSS and no Bliss, so §4 step 5 would delete it [verified]

- Columns in `drugcomb_nci60.pkl`:
  `Drug1_ID, Drug2_ID, Cell_Line_ID, CSS, Synergy_ZIP, Synergy_Bliss, Synergy_Loewe, Synergy_HSA,
  Drug1, Drug2, CellLine`. ✔ This matches the design.
- Columns in `oncopolypharmacology.pkl` (first 4 MB scanned):
  `Drug1_ID, Drug2_ID, Cell_Line_ID, Y, Cell_Line, Drug1, Drug2`. There's **one** label, `Y`, which
  the design's own §4 table calls Loewe.
- §4 step 5 says `y_syn` = Bliss, `y_inh` = CSS, and "drop triplets missing either." Applied to the
  second benchmark, that drops **every row**. It also breaks:
  - the run grid (8 × 2 targets × 10 × **2 benchmarks**);
  - the M6 gate ("3-seed pilot on OncoPolyPharmacology").
- [recommendation] Options, in order of preference:
  1. **Take the O'Neil study from the DrugComb portal bulk download** instead of TDC. The portal
     reports synergy and sensitivity scores per study, which the design already lists as a source.
     [unverified] that the portal's O'Neil rows carry both Bliss and CSS; check this first.
  2. Otherwise, treat O'Neil as **synergy-only, with `y_syn` = Loewe**, and say so in the data card.
     That halves the O'Neil grid (no `inh` campaigns) and uses the score the design itself flags as
     biased (§13 risks: Loewe Hill-slope inflation).
  3. Move the M6 pilot to a **subsample of the NCI-60 set** (for example 20 drugs × all lines), which
     has both outcomes.
- Also: the O'Neil pickle is **1.6 GB** for 23k rows, so the row-level `Cell_Line` field is
  presumably a large feature vector. The design gets cell features from DepMap anyway, so
  [recommendation] drop that column during conversion.

### 2.4 TDC "DrugComb" is really NCI-ALMANAC [verified]

> **Correction (2026-11-08):** this was inferred from the filename, and it is only mostly true.
> - The file is DrugComb restricted to the 59 NCI-60 lines.
> - It includes O'Neil/Merck rows (MK-2206, MRK-003, AZD1775 and others) on 9 lines, plus a 4-row
>   third study.
> - The two benchmarks therefore overlap.
> - See [drug-identity-audit.md](drug-identity-audit.md).

- The file is named `drugcomb_nci60.pkl`, and the design's §4 already calls it "NCI-60 combination
  screen." So both benchmarks come from two famous, heavily reanalyzed screens.
- This matters most for leakage control (§10 of the design):
  - the blocklist must include the NCI-ALMANAC publication and its reanalyses;
  - the "famous vs obscure" split will carry a lot of weight.

### 2.5 Drug identifiers are messy, so harmonization is real work [verified]

- Sample `Drug*_ID` values in the NCI-60 file:
  - `CISPLATINO`
  - `ELOXATIN (TN) (SANOFI SYNTHELAB)` (oxaliplatin)
  - `ADM HYDROCHLORIDE`
  - `ANTIBIOTIC AY 22989`
  - `CHEMBL17639`
  - the CAS number `34793-34-5`
- In O'Neil: `MITOMYCINE`, `ZOLINZA` (vorinostat), and Merck codes like `MK-2206` and `MRK-003`.
- These are brand names, foreign spellings, salts, CAS numbers, and ChEMBL IDs. §4 step 3 (map to
  InChIKey and ChEMBL ID) needs:
  - [recommendation] a hand-checked synonym table committed to the repo;
  - InChIKey from the SMILES (if the `Drug1`/`Drug2` columns hold SMILES; [unverified], and that
    check needs loading the data);
  - a test that the same molecule gets the same ID across both benchmarks.
- [verified, minor] The O'Neil drug name table contains at least 38 distinct names (counted from the
  categorical strings), while the design says 37. Check this after loading.

### 2.6 The "only `Oracle.run()` reads labels" rule is stricter than the plan itself [verified by reading the design]

§1's hard rule conflicts with these modules, all of which need hidden labels:

- `pool.py` and the data card (to compute hit thresholds and distributions);
- `explain/check_predictions.py` (scores held-out lines);
- the zero-shot memory probe (correlates the LLM's ranking with true labels);
- the leakage test ("scan tool outputs for values matching hidden labels"), which by definition
  needs the labels.

[recommendation] Reword the rule to: "**no code reachable from a selector** may read labels." Then
keep an explicit **allowlist** (`data/pool.py`, `lab/oracle.py`, `explain/check_predictions.py`,
`eval/*`, and the probe and leakage tests). Enforce it with the import-scan test from §5, which
already exists in the plan.

### 2.7 Ambiguity: hit thresholds are taken "from the full pool," but held-out lines are "removed from the pool" [verified by reading]

Pick one and write it in `data_card.md`. [recommendation] Compute thresholds on the **selectable**
pool (after removing held-out lines), so that "hit" means the same thing for everything a selector
can actually find.

### 2.8 The diagram in §8 is missing [verified]

`target_design.md:223` reads `[embedded content: agent round loop · 6 stages]`, so the figure was lost
in export. [concepts-primer.md §5](concepts-primer.md#5-one-agent-round) has a text reconstruction
**built from the 9 steps written in §8**. It's not the original figure.

### 2.9 Layout and convention mismatches with the repo [verified]

| Design says | Repo has | Recommendation |
|:--|:--|:--|
| `combo_agent/` at the top level | `src/combo_agent/` (`uv_build` src layout) | Keep the src layout; the subpackages from §3 go under `src/combo_agent/` |
| `data_cache/` and `runs/` git-ignored | `.gitignore` covers `data/` and `workshop/` only | Add `data_cache/`, `runs/`, `.cache/` |
| pandas, numpy, scikit-learn, PyTorch, GPyTorch/BoTorch, RDKit, pydantic | only `pandas`, `sheldon` (pydantic comes with sheldon) | Add `numpy`, `pyarrow` (parquet), `scikit-learn`, `torch`, `scipy` (Mann-Whitney, BH via `scipy.stats.false_discovery_control`), `rdkit`, `matplotlib`, `pyyaml`; dev: `pytest`. Add `botorch`/`gpytorch` only when `bo_qei` or the v2 GP is built |
| "Unit tests for every module" | no `tests/` | Create `tests/` in M0 |

### 2.10 The TDC files have no `study_id` column [verified]

- §4 step 4 says "Record the source `study_id` for each triplet." The verified column lists in §2.3
  don't include one.
- [recommendation] Set it per benchmark: `NCI-ALMANAC` for the NCI-60 file and `ONEIL` for O'Neil,
  so every triplet in a file gets the same value. Or take real study IDs from the DrugComb portal if
  we use it (D2).
- The leakage blocklist (M5) needs only study → paper, so a per-file constant is enough for v1.

### 2.11 Hit thresholds (M1) depend on the held-out split (M2) [verified by reading]

- The M1 data card must contain "the hit thresholds" (§4 step 7). If thresholds are computed on the
  selectable pool (recommendation in §2.7), they depend on which lines are held out. That split is
  scheduled for M2 (§5).
- [recommendation] **Make the held-out-line split part of freezing the pool in M1:** stratify by
  lineage with a fixed seed, and save it to `splits.json`. The seed sets and famous/obscure labels
  stay in M2.
- That way the pool, the thresholds, and the data card are frozen together, once.

### 2.12 §4 and §5 disagree about where labels live [verified by reading]

- §4 says `pool.parquet` holds "both outcomes."
- §5 says "Labels live in a private attribute loaded from a **separate file**. A test asserts no
  other module imports that file."
- [recommendation] Make M1 write two files: `pool.parquet` (triplet keys, metadata, `n_reps`, and
  `study_id`, but **no outcome values**) and `labels.parquet` (keys plus `y_syn*`, `y_inh`, and the
  SDs). It's cheap now, and it's exactly what M2's label-access test needs.

### 2.13 Byte-for-byte data card [verified by reading §4 acceptance]

- The data card must not contain anything that varies between runs: no timestamps, absolute paths,
  or dict orderings.
- Put the download date and hashes in a separate `manifest.json`.
- Format floats with fixed precision.
- Pin the `pandas` and `pyarrow` versions in `uv.lock` (already the case for pandas).

---

## 2A. Which §2 issues to address *now* for M1 (triage, 2026-11-08)

```
 NOW: blocks M1 code                         SOON: decide during M1          LATER: not M1
 ─────────────────────────────────────────   ─────────────────────────────   ──────────────────────────
 0  free disk (machine check)              2.3 O'Neil outcomes (D2)        2.4 leakage blocklist → M5
 1  run the profile script (§5)              2.6 full allowlist wording      2.8 lost diagram   → M6
 2  2.1+2.2 ingestion without `tdc`              → finalize in M2            §3 architecture    → before M4
 3  2.9 M0 housekeeping (deps, ignore, tests)
 4  2.5 drug + cell-line harmonization
 5  2.7+2.11 threshold basis + move split
 6  2.12 pool/labels file split
 7  2.10 study_id, 2.13 deterministic card
```

**Why this order:**

- **Items 0–1** cost minutes and settle questions everything else depends on: hit definition,
  budget, replicates, lines per pair.
- **Item 2** decides how any data gets in.
- **Items 4–7** define the *schema* of the frozen pool. Changing a schema after M2–M4 are built on
  it means redoing work, so lock it now.
- **2.3 (O'Neil) doesn't block starting M1.** Build the whole M1 pipeline on NCI-60 first (it has
  both outcomes), and investigate the DrugComb portal's O'Neil data in parallel. Add O'Neil as a
  second input once D2 is decided.
- **The architecture (§3) doesn't affect M1:** the data layer is identical under option A or B. It
  has to be settled before M4's runner.

---

## 3. The main architecture decision: how much SHELDON to use

This is the most consequential choice, and the design is ambiguous about it. §8 says "adapt its
modules rather than writing a new harness," but §3 and §11 specify **our own**
`eval/runner.py`, `runs/`, and `results.parquet`.

### What SHELDON is shaped like [verified]

```
SHELDON runner:   Task ──► Method.run(task, ctx) ──► one Prediction ──► Benchmark.score() ──► one number
                           (many LLM "rounds" of          (one final
                            reasoning inside)               answer)
```

- The `sheldon` method (`methods/sheldon/loop.py:99-174`) does plan, then rounds of
  explore → select → candidate analysis → comparative analysis → exploit (revise), then
  `finalize(...)` returns **one** answer.
- Each hypothesis carries an `artifact` and a list of `scores` (`state.py`, `Hypothesis`). The loop
  is built to **find the best artifact**: a program, a molecule, a diagnosis.

### What our campaign is shaped like (design §5, §8, §11)

```
Our campaign:  seed set ──► round 1 ──► round 2 ──► … ──► round 20      (budget 320 = 20 × 16)
                            │ refit model (no LLM)
                            │ flag surprises (no LLM)
                            │ shortlist (no LLM)
                            │ explore / select / analyse (LLM)
                            │ assemble batch of 16 (LLM + model slots + critic)
                            └ oracle.run(batch) → results → update stores
               Output: a whole discovery CURVE (hits vs experiments), not one answer.
```

They don't line up:

- A SHELDON "round" is a reasoning round. Our "round" is a **lab round** that wraps deterministic
  ML steps around the LLM stages.
- SHELDON's output is one answer. Ours is a trajectory.

### Options

| | A. Our own runner; the agent reuses SHELDON *components* | B. Everything as a SHELDON Benchmark/Method |
|:--|:--|:--|
| Shape | `eval/runner.py` runs `Selector × seed × target`. `selectors/agent/` imports SHELDON's `State`, LLM client, cache, engines, probes, and trace | One campaign = one SHELDON `Task`; selectors = SHELDON `Method`s; the oracle = a rationed probe (the DOCKSTRING pattern) |
| Matches the design's §3/§11 layout | ✅ yes | ⚠️ partly (runs go to SHELDON's `runs/<run_id>/`, not `results.parquet`) |
| Numeric baselines | Plain Python, fast, no LLM machinery | Have to be wrapped as Methods; they'd go through SHELDON's runner and budget |
| Deterministic steps around the LLM (refit, surprises, shortlist, min model slots, fallback to `bo_ucb`) | Natural | Have to be pushed inside a custom Method anyway |
| Reuse of SHELDON's tested pieces (caching, cost logging, tool loop, provider abstraction, retries, trace UI) | ✅ by import | ✅ by default |
| Comparability with SHELDON's paper ablations | Via our own `agent_single_hyp` | Via `-p` knobs |

**[recommendation] Option A.** Build our own runner exactly as §3/§11 specify, and treat SHELDON as a
**library**. Concretely:

- **Reuse directly** [verified that these exist and are self-contained]:
  - `sheldon.methods.sheldon.state.State`, `Hypothesis`, `Observation`, `Region`: plain dataclasses
    that never call a model ("Nothing here calls a model", `state.py` docstring). They give us
    regions per space, statuses, merge, cooldown/eligibility, and snapshot/restore.
  - `sheldon.core.llm` + `sheldon.core.cache`: the provider abstraction (OpenAI-compatible for
    Gemini, or another provider), a content-addressed SQLite cache, cost estimates, and retries.
  - `sheldon.engines.toolloop` + `sheldon.core.probes.CallableProbe`: a native tool-calling loop for
    our JSON tools (`query_model`, `search_literature`, `get_drug`, `get_cell`, `get_history`,
    `resolve`).
  - `sheldon.core.trace`: per-call logging.
- **Write our own:** the prompt templates (SHELDON's prompts talk about artifacts and evaluators) and
  the pydantic output schemas for each stage.
- **Small mismatches to resolve** [verified]:
  - SHELDON relations are `supports | refutes | neutral`; the design adds `unresolved`.
  - SHELDON statuses are `pending | tested | shelved`; the design has `retained | setaside`.
  - Either map one onto the other (retained ≈ tested, setaside ≈ shelved) or extend in our own
    wrapper. **Don't edit the pinned SHELDON.**
- **What happens to the current `combo-mechanism` benchmark stub:** it frames the task as "per pair,
  predict which lines respond (AUROC)." That's closest to design §9, but the design scores §9 with
  code-checkable rules, not as a SHELDON task. [recommendation] Leave it in place for now (it's
  harmless). Revisit at M6: if we want SHELDON's `ui`/`log` trace viewers for agent runs, we can
  still register a thin Benchmark that wraps one campaign.
- [unverified] Whether SHELDON's LLM layer sends `temperature=0` on the OpenAI path. The design
  requires temperature 0 for selection calls. Check `core/llm.py` request building when M6 starts.
- [verified, from earlier onboarding] Gemini drops SHELDON's server-side `web_search`. **That doesn't
  hurt this design**, because §8's `search_literature` is our own client-side tool, which is what we
  need for the leakage filter anyway.

---

## 4. Scientific review: where I'd push back or add something

These are colleague-level comments. None of them block M1.

1. **The model is calibrated where it doesn't operate.** M3's gate checks calibration on a random
   80/20 split of the **full pool**: about 190k training rows. During a campaign, the model is
   trained on **32 to 352** points.
   - At that size, with about 5k input dimensions (2×2048 fingerprint bits as sum and product, plus
     about 1k genes, plus mutations), a 2×256 MLP ensemble will be badly overparameterized. Its
     uncertainty estimates in early rounds are exactly what UCB, uncertainty sampling, and the
     agent's shortlist depend on.
   - [recommendation] Add a **low-data calibration check** to M3: train on 32, 96, and 320 random
     triplets and report coverage on the rest.
   - Consider compact features: target-set and MoA one-hots, PCA of fingerprints and expression,
     learned drug and cell **ID embeddings**.
   - If the ensemble fails that check, move the v2 GP earlier. GPs are the natural model at n < 400.
2. **This low-data regime is also where the agent could win, so the evaluation should show it.**
   Literature priors matter most when data is scarce. [recommendation] Report hits at 80 (already
   planned) prominently, and plot curves from round 0.
3. **`bo_qei` with a deep ensemble.** BoTorch's qEI expects a BoTorch model and posterior.
   [recommendation] Either:
   - implement Monte Carlo qEI over ensemble members with greedy sequential selection over the
     discrete candidate set (which is simple and transparent), or
   - wrap the ensemble in BoTorch's ensemble posterior. [unverified] that the API fits our version.

   Either way, document it.
4. **Hit concentration.** If a few drug pairs are synergistic in many lines, "hits at budget" rewards
   finding one pair and then sweeping its lines. The design already includes **hit diversity**
   (unique pairs).
   - [recommendation] Make "unique hit pairs at budget" a co-primary diagnostic, or the agent and BO
     may "win" by line-sweeping.
   - `scripts/profile_nci60.py` measures concentration (top-10 pairs' share of hits).
5. **Held-out-line testability.** About 59 lines × 20% gives about 12 held-out lines. A rule like
   "KRAS_mut == 1" needs at least 3 lines on each side, and many driver mutations are rare in
   NCI-60. Expect a large "untestable" fraction (the design already says this).
   - [recommendation] Prototype the M7 scorer on a planted rule early, and count testable rules per
     feature *before* the LLM ever writes one. That tells us whether the secondary question has
     statistical power.
6. **Statistics are fine as specified.** With 10 paired seeds, the smallest possible two-sided
   Wilcoxon p-value is 2/2¹⁰ ≈ 0.002, which comfortably survives Holm correction across 7
   comparisons. More seeds would cost LLM money only for agent runs, not baselines.
7. **Determinism.** "Two runs give identical batches" requires deterministic MLP training. At this
   data size, train on CPU with fixed torch seeds and `torch.use_deterministic_algorithms(True)`. GPU
   isn't needed for fitting on ≤352 rows. Predicting about 240k candidates × 5 members is also cheap
   on CPU. The "<60 s on one GPU" requirement is easy either way.
8. **Famous/obscure labels need the same name harmonization (§2.5).** PubMed co-mention counts on
   raw names like `ELOXATIN (TN) (SANOFI SYNTHELAB)` would label famous pairs as "obscure." Run the
   counts on **canonical names plus synonyms**, and register for an NCBI API key (the rate limit is
   higher with one).

---

## 5. Open data questions that needed one run (answered in M1 step 1b)

These were answered when the NCI-60 table was profiled in M1 step 1b (numbers in
[m1-implementation.md](m1-implementation.md)). The questions were:

- the pool size after canonicalization (A < B), and how many triplets have replicates;
- lines per pair (does every pair cover all ~59 lines? This matters for held-out testing);
- Bliss and CSS distributions, and the p95 thresholds (design §13: p95 or Bliss ≥ 10?);
- the number of hits, and how concentrated they are in a few pairs (§4.4 above);
- the expected random-selection hits at 80, 160, and 320 (design §13: is 320 × 16 sensible?).

For a sense of scale: if hits are exactly 5% of triplets, random selection finds about 16 hits in
320 picks. (That's simple arithmetic, not data.)

---

## 6. Recommended next steps (ordered; the first items unblock everything)

### Step 0: team decisions (one meeting)

| # | Decision | My recommendation |
|:--|:--|:--|
| D1 | Architecture (§3) | Option A: our own runner, SHELDON as a library |
| D2 | Second benchmark's outcomes (§2.3) | O'Neil from the DrugComb portal if it has Bliss + CSS; else Loewe-only and no `inh` grid |
| D3 | Label-access rule wording (§2.6) | "Selector-reachable code" plus an allowlist |
| D4 | Thresholds on the full or selectable pool (§2.7) | Selectable pool |
| D5 | v1 scope: the portal beyond the two TDC sets? (design §13) | No: the two benchmarks only, but use the portal for O'Neil if D2 needs it |
| D6 | LLM and cost cap (design §13) | Gemini is already wired in. Needed before M6, not M1 |

### Step 1: M0 housekeeping (small; needs team approval)

1. `pyproject.toml`: add the dependencies in §2.9.
2. `.gitignore`: add `runs/`, `data_cache/`, `.cache/`.
3. `.env.example`: add `SHELDON_RUNS` and `SHELDON_CACHE` (by default SHELDON writes runs and its cache inside the installed package).
4. Create `configs/` (pydantic config model plus `default.yaml`) and `tests/` with a trivial passing
   test.
5. Create empty subpackages matching design §3 under `src/combo_agent/`.

### Step 2: M1, data and pool. Suggested split into PR-sized pieces

| Piece | Output | Test / gate |
|:--|:--|:--|
| 1a `ingest_tdc.py` | Direct Dataverse download, sha256 check, isolated conversion to parquet (§2.1) | Hash matches; row counts are 297,098 and 23,052 |
| 1b Profile | Run `profile_nci60.py`-style stats; paste into the progress log | Answers §5; settles D2/D4 and budget size |
| 1c `harmonize.py` drugs | Synonym table, then InChIKey/ChEMBL (§2.5) | Same molecule gets the same ID across benchmarks; no unmapped pool drugs |
| 1d `harmonize.py` lines | Cellosaurus and DepMap IDs, lineage | Every pool line is mapped; dropped lines logged |
| 1e Canonicalize + merge | A < B, mean/sd/`n_reps`, `study_id` | No duplicate triplets |
| 1f `features.py` | Drug and cell feature tables | Shapes; no NaN in the pool features |
| 1g `pool.py` | `pool.parquet`, `drugs.parquet`, `cells.parquet`, `data_card.md` | **Byte-for-byte reproducible** from the raw files |

### Step 3: M2 and M3 together, in parallel once 1g is done

- The oracle is small, and its refusal tests are easy to write.
- The response model can start on a toy pool even before 1g. Do the low-data calibration first
  (§4.1).

### Step 4: M4, which is the first real result

The gate "`bo_ucb` beats `random` at 320" is the first scientific checkpoint. If it fails, the
response model is the problem, and **no agent work should start** (the design says this too).

Only then: M5, M6, M7, and M8, following the design's order.

---

## 7. What I verified and how (audit trail)

- **Repo state:** `git status` (clean), `git log` (`78f71b5`), and a file listing, on 2026-11-08.
- **SHELDON** at `7fbf027`: `methods/sheldon/loop.py:99-174`, `methods/sheldon/state.py`,
  `methods/sheldon/schemas.py`, `methods/sheldon/stages.py` function list, and the files cited in
  the SHELDON source.
- **PyTDC 1.1.15:** `requires_dist` from PyPI's JSON API. In the downloaded sdist (not installed):
  `tdc/metadata.py` (dataset names, Dataverse ids, row counts, `pkl` format), `tdc/utils/load.py`
  (Dataverse URL), and `tdc/multi_pred/drugsyn.py`.
- **TDC files:** ranged GET for sizes. Downloaded `drugcomb_nci60.pkl` in full and the first 4 MB of
  `oncopolypharmacology.pkl` to a scratch folder outside the repo. Column names and drug
  and line names come from a **static** `pickletools` opcode scan; nothing was unpickled.
- **Not done:** loading data, computing statistics, running any repo command.
