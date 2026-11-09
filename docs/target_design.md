# Drug combination discovery agent: implementation plan

## 1. Purpose and research question

Build a benchmark and an agent that picks drug-combination experiments in rounds, then measure whether it finds strong combinations in fewer experiments than standard active learning. Sections 4 to 11 are specs; section 12 is the build order.

**Primary question.** Can an agent that combines literature evidence with a statistical response model find effective cancer drug combinations with a smaller experiment budget than random selection, uncertainty sampling and Bayesian optimization?

**Secondary question.** When the agent proposes a mechanism for a hit, does the prediction implied by that mechanism (which cell lines should respond) hold on cell lines the agent never saw?

**What we are building, in one line each:**

1. A clean dataset of measured drug-pair × cancer-cell-line results (section 4).
2. A simulated lab that hides every result until the selector "runs" it (section 5).
3. A response model that predicts results with uncertainty (section 6).
4. Baseline selectors that use numbers only (section 7).
5. The agent: response model + LLM reasoning loop built on SHELDON (section 8).
6. An explanation and prediction-checking module (section 9).
7. Leakage controls (section 10) and an evaluation harness (section 11).

**Hard rules for every implementer:**

- The LLM never outputs predicted response values. All numeric estimates come from the statistical model.
- No code path may read hidden results except the oracle's `run()` method.
- Every run is reproducible from a config file and a seed.
- Agent and baselines get the same candidate pool, the same budget, the same batch size and the same seed set.

## 2. Glossary

Use these terms exactly in code, logs and docs.

| Term                     | Meaning in this project                                                                                             | Code name                  |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------- | -------------------------- |
| Experiment               | One (drug A, drug B, cell line) triplet. Drug order is canonicalized (A < B by ID).                                 | `Triplet`                  |
| Candidate pool           | The fixed set of triplets any selector may choose from, frozen before any run.                                      | `pool`                     |
| Oracle                   | The simulated lab. Holds the hidden results and reveals one only when that triplet is run.                          | `Oracle`                   |
| Round                    | One cycle: select a batch, reveal it, update models.                                                                | `round_idx`                |
| Batch size               | Experiments chosen per round (default 16).                                                                          | `batch_size`               |
| Budget                   | Total experiments allowed per run (default 320 = 20 rounds × 16).                                                   | `budget`                   |
| Synergy                  | How much better the pair does than expected from each drug alone. Primary score: Bliss; secondary: ZIP, Loewe, HSA. | `y_syn`                    |
| Growth inhibition        | How well the pair stops cell growth, regardless of interaction. Primary score: CSS (combination sensitivity score). | `y_inh`                    |
| Strong combination (hit) | A triplet at or above the 95th percentile of the pool for that outcome. Thresholds fixed once from the full pool.   | `is_hit_syn`, `is_hit_inh` |
| Response model           | Statistical model that predicts `y_syn` and `y_inh` with uncertainty for any untested triplet.                      | `ResponseModel`            |
| Selector                 | Any strategy that picks the next batch (random, uncertainty, BO, agent).                                            | `Selector`                 |
| Hypothesis               | A claim of the form "pair P works in cell context C because of mechanism M".                                        | `Hypothesis`               |
| Observation              | A revealed result, or a retrieved literature fact, stored as evidence.                                              | `Observation`              |
| Explanation level        | `mechanistic`, `cross_cancer` or `statistical` (section 9).                                                         | `ExplanationLevel`         |
| Held-out cell lines      | Lines never available for selection, used only to test mechanism predictions.                                       | `heldout_lines`            |

Scores such as Bliss and CSS are computed by DrugComb from dose-response matrices; we use their published values and do not recompute them in v1.

## 3. Repo layout, stack and conventions

Python 3.11, one package `combo_agent`, configs in YAML, every run writes to its own folder under `runs/`.

```
combo_agent/
  data/
    ingest_tdc.py        # TDC DrugSyn loaders
    ingest_drugcomb.py   # DrugComb portal downloads
    harmonize.py         # IDs, dedup, replicate merge
    features.py          # drug + cell line features
    pool.py              # builds and freezes the candidate pool
  lab/
    oracle.py            # Oracle: the only reader of hidden labels
    splits.py            # selectable vs held-out lines, famous vs obscure
  models/
    response_model.py    # ResponseModel interface + implementations
  selectors/
    base.py              # Selector interface
    random_sel.py
    uncertainty.py
    bayes_opt.py
    agent/
      loop.py            # SHELDON-style round logic
      llm_client.py      # model calls, caching, cost logging
      retrieval.py       # literature + KB evidence, with leakage filter
      hypotheses.py      # Hypothesis / Observation stores
      prompts/           # versioned prompt templates
  explain/
    explanations.py      # levels + testable predictions
    check_predictions.py # scores predictions on held-out lines
  eval/
    runner.py            # runs selectors × seeds
    metrics.py
    plots.py
configs/
tests/
runs/                    # git-ignored outputs
data_cache/              # git-ignored raw + processed data
```

**Conventions:**

- Libraries: pandas, numpy, scikit-learn, PyTorch, GPyTorch or BoTorch for GP and BO, RDKit for drug features, pydantic for configs and records.
- All randomness flows from `cfg.seed`. Log the git commit, config and package versions in every run folder.
- Every LLM call is cached by (prompt hash, model, temperature) and logged with tokens and cost.
- Unit tests for every module; an end-to-end smoke test on a 500-triplet toy pool must finish in under 5 minutes.
- Type hints and docstrings on public functions. No notebook-only logic.

## 4. Data pipeline

The output is one frozen table, `pool.parquet`, with one row per (drug A, drug B, cell line) triplet and both outcomes, plus feature tables for drugs and cell lines.

**Sources.**

| Source                                     | What it gives                                                                                        | Role                                                                                |
| ------------------------------------------ | ---------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| TDC `DrugSyn(name='DrugComb')`             | NCI-60 combination screen: 129 drugs × 59 cell lines, 297,098 triplets, CSS + Bliss, HSA, Loewe, ZIP | Primary benchmark (dense, clean)                                                    |
| TDC `DrugSyn(name='OncoPolyPharmacology')` | O'Neil/Merck screen: 37 drugs × 39 lines, 23,052 triplets, Loewe synergy                             | Second benchmark, checks that results transfer                                      |
| DrugComb portal bulk download              | Wider set of studies, synergy and sensitivity scores per triplet                                     | Extra cancer lines and drugs; also the source for study IDs used in leakage control |
| DepMap / CCLE                              | Mutations, copy number, expression, lineage per cell line                                            | Cell features and ground truth for mechanism predictions                            |
| ChEMBL / DrugBank + RDKit                  | Targets, mechanism of action, SMILES                                                                 | Drug features and the agent's mechanism lookup                                      |

**Steps (each a function with a test):**

1. **Load.** Pull the TDC datasets with the `tdc` package and the DrugComb summary table. Save raw copies to `data_cache/raw/` with download date and file hash.
2. **Keep cancer only.** Map every cell line to a Cellosaurus ID and DepMap ID. Drop lines that are not cancer-derived or that fail to map. Log dropped lines.
3. **Harmonize drugs.** Map names to InChIKey and ChEMBL ID. Merge salts and synonyms. Drop drugs with no structure.
4. **Canonicalize and merge replicates.** Sort each pair so A < B. Where a triplet has several measurements, store the mean, the standard deviation and `n_reps`. Record the source `study_id` for each triplet.
5. **Attach outcomes.** `y_syn` = Bliss (also keep ZIP, Loewe, HSA columns). `y_inh` = CSS. Drop triplets missing either.
6. **Build features.** Drugs: Morgan fingerprint (2048 bits), target set, MoA class. Cell lines: lineage, driver mutations (e.g. KRAS, BRAF, PIK3CA, PTEN, TP53), top 1,000 variable genes from expression.
7. **Freeze the pool.** Keep drugs with at least one annotated target. *(Overridden 2026-11-08 by M1 decision 8: drugs with no ChEMBL target are kept, flagged `target_status = none_in_chembl`, and the agent looks up their targets in the literature. See `docs/m1-implementation.md`.)* Write `pool.parquet`, `drugs.parquet`, `cells.parquet` and a `data_card.md` with counts, score distributions and the hit thresholds.

**Acceptance checks:** no duplicate triplets after canonicalization; every pool drug has a target annotation; every pool line has a DepMap ID and lineage; the data card reproduces byte-for-byte from the same raw files.

## 5. Simulated lab (oracle) and splits

The oracle is the only object that can read hidden labels; every selector talks to it through one method and pays budget for every reveal.

```python
class Oracle:
    def __init__(self, pool: pd.DataFrame, budget: int, seed: int): ...
    def candidates(self) -> list[Triplet]:      # untested, selectable triplets
    def run(self, batch: list[Triplet]) -> list[Result]:
        # checks: len(batch) <= batch_size, no repeats, budget left,
        # every triplet in candidates(); returns y_syn, y_inh, n_reps
    def history(self) -> list[Result]:          # everything revealed so far
    def budget_left(self) -> int: ...
```

**Rules:**

- Labels live in a private attribute loaded from a separate file. A test asserts no other module imports that file.
- Re-running a triplet is not allowed in v1. Measurement noise is reported through `n_reps` and standard deviation, not by sampling.
- Every `run()` call is logged with round, selector, triplets and revealed values.

**Splits (made once, saved to `splits.json`):**

1. **Held-out cell lines.** 20% of cancer lines, stratified by lineage, are removed from the pool. No selector ever sees them. They are used only in section 9.
2. **Seed set.** At round 0, every selector gets the same 32 random triplets revealed for free. Seeds 0 to 9 give 10 different seed sets.
3. **Famous vs obscure pairs.** For each drug pair, count PubMed records that mention both drugs with a cancer term. Top quartile = `famous`; zero co-mentions = `obscure`; the rest = `middle`. Store the counts and the query date.

**Acceptance checks:** a selector that tries to read labels directly fails a test; the oracle refuses over-budget, repeated or held-out triplets; the same seed gives the same seed set.

## 6. Response model with uncertainty

All selectors, agent included, use the same response model class with the same settings, so any difference in results comes from how experiments are chosen, not from a better predictor.

```python
class ResponseModel(Protocol):
    def fit(self, history: list[Result]) -> None: ...
    def predict(self, triplets: list[Triplet]) -> Prediction:
        # Prediction has mean_syn, std_syn, mean_inh, std_inh (arrays)
    def embed(self, triplets: list[Triplet]) -> np.ndarray: ...  # for diversity
```

**v1 implementation: deep ensemble.**

- Five small MLPs (2 hidden layers, 256 units), different random inits, trained from scratch each round on the revealed history only.
- Input is symmetric in drug order: \[drug A + drug B features, drug A × drug B features, cell features\].
- Two output heads (`y_syn`, `y_inh`), each predicting a mean and a variance. Total std = spread across members + mean predicted noise.
- Targets standardized using the seed-set statistics only.

**v2 option: GP.** Exact or sparse GP with a product kernel (symmetric drug-pair kernel × cell-line kernel) via GPyTorch. Use it if the ensemble is badly calibrated.

**Rules:**

- Never pretrain on pool labels. Any pretraining data must exclude every pool triplet and every held-out line (off in v1).
- Refit time per round under 60 seconds on one GPU for a 300k-triplet pool.

**Acceptance checks:** on a random 80/20 split of the full pool (a sanity test only, not part of the benchmark), Pearson r is reported for both outcomes; 80% prediction intervals cover 70% to 90% of test points; predictions do not change when drug A and drug B are swapped.

## 7. Baselines and ablations

Every selector implements one method, and every campaign targets one outcome (`target = "syn"` or `"inh"`); each selector is run once per target.

```python
class Selector(Protocol):
    def select(self, oracle: Oracle, model: ResponseModel,
               batch_size: int, target: str) -> list[Triplet]: ...
```

| Selector           | Rule for picking the batch                                                                                | Why it is here                              |
| ------------------ | --------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| `random`           | Uniform sample from untested candidates                                                                   | Floor                                       |
| `greedy`           | Top predicted mean for the target                                                                         | Pure exploitation                           |
| `uncertainty`      | Top predicted std for the target                                                                          | Pure exploration (standard active learning) |
| `bo_ucb`           | Top mean + β × std, β = 2, with a diversity penalty in embedding space so the batch is not 16 near-copies | Main numeric baseline                       |
| `bo_qei`           | Batch expected improvement (BoTorch qEI) over the current best                                            | Stronger BO check                           |
| `agent_no_lit`     | Full agent loop, retrieval turned off                                                                     | Tests whether literature helps              |
| `agent_single_hyp` | Agent without SHELDON's separate observation space: refines one hypothesis set only                       | Tests whether SHELDON's exploration helps   |
| `agent`            | Full system (section 8)                                                                                   | Method under test                           |

All numeric baselines already see drug target and MoA features through the response model, so any agent advantage must come from reasoning over evidence, not from extra features.

**Acceptance checks:** each selector returns exactly `batch_size` unique, untested, selectable triplets; with a fixed seed, two runs give identical batches (the agent with cached LLM calls).

## 8. Agent design: response model + LLM in a SHELDON loop

The statistical model proposes and scores candidates; the LLM reasons over mechanisms and surprises, then picks the batch from model-scored options. Start from the SHELDON code at [github.com/stanford-oval/sheldon](https://github.com/stanford-oval/sheldon) and adapt its modules rather than writing a new harness.

**How SHELDON maps onto this task.** SHELDON keeps two separate spaces: hypotheses and observations, each organized by axes and regions, with every hypothesis-observation pair labeled Support, Refute, Neutral or Unresolved. Its round runs Explore, Select, Candidate analysis, Comparative analysis and Exploit ([paper](https://arxiv.org/html/2610.18025)).

| SHELDON piece           | Our version                                                                                 |
| ----------------------- | ------------------------------------------------------------------------------------------- |
| Hypothesis axes         | Pathway pair (e.g. MAPK × PI3K), lineage, genetic context (e.g. KRAS-mutant), drug class    |
| Observation axes        | Revealed results, literature, drug target DB, cell line genomics                            |
| Working hypothesis set  | Up to 20 live `Hypothesis` records                                                          |
| RETAINED / SETASIDE     | Same; a set-aside hypothesis can return after a 3-round cooldown                            |
| Provisional observation | A literature claim not yet checked against revealed results; it can guide but cannot refute |

&#91;embedded content: agent round loop · 6 stages\]

The top row never calls the LLM; the LLM works only on the bottom row, choosing from model-scored options and its own tool requests.

**One round, in order:**

1. **Refit.** Fit the response model on the history. Score every candidate.
2. **Flag surprises.** For each newly revealed result, compute z = (observed − predicted mean) / predicted std. Any |z| > 2 becomes a `surprise` observation that no current hypothesis explains until one claims it.
3. **Shortlist.** Top 200 candidates by UCB, top 50 by std, and 50 random picks from drug-pair regions never tested. The LLM sees only this shortlist plus anything it asks for through tools.
4. **Explore (LLM).** Read the state, the surprises and retrieved evidence. Add or extend axes, propose new hypotheses, revive cooled-down ones.
5. **Select (LLM).** Pick at most 5 hypotheses to test this round.
6. **Candidate analysis (LLM + tools).** For each chosen hypothesis, gather supporting and refuting evidence, up to 3 tool iterations each, and label the relations.
7. **Comparative analysis (LLM).** Propose experiments whose outcomes would separate competing hypotheses, e.g. the same pair in mutant vs wild-type lines.
8. **Assemble batch.** At least `min_model_slots` (default 4 of 16) come from the top UCB candidates. The LLM fills the rest; each pick cites a hypothesis ID and the result it expects. A critic call checks for redundancy, untestable picks and leakage; the batch is revised once.
9. **Run and exploit.** Call `oracle.run()`. Add results as observations, update evidence labels, mark each hypothesis RETAINED or SETASIDE.

**Tools the LLM can call (JSON function calls):**

- `query_model(triplets | filter)` returns mean and std for both outcomes. It never returns hidden labels.
- `search_literature(query)` returns abstracts after the leakage filter (section 10).
- `get_drug(drug_id)` returns targets, MoA and class. `get_cell(line_id)` returns lineage and key mutations.
- `get_history(filter)` returns revealed results only.
- `resolve(pair, context_filter)` turns a request such as "trametinib + alpelisib in KRAS-mutant lines" into candidate triplets.

**Records (pydantic):**

```python
class Hypothesis(BaseModel):
    id: str
    pair_scope: str            # drugs or drug classes
    context: str               # cell-line condition
    mechanism: str
    level: ExplanationLevel
    status: Literal["retained", "setaside"]
    evidence: dict[str, Literal["support", "refute", "neutral", "unresolved"]]
    created_round: int
```

**Limits:** temperature 0 for selection calls; a token and dollar cap per run in config; all prompts versioned in `prompts/` and their hash logged per call. If the LLM output fails schema checks twice, fall back to `bo_ucb` for that round and log it.

## 9. Explanations and held-out prediction tests

Every flagged hit gets a labeled explanation; every mechanistic explanation must carry a machine-checkable prediction, frozen before anyone looks at held-out lines.

**Explanation levels:**

- `mechanistic`: a named pathway or target interaction in these cells. Must include a prediction.
- `cross_cancer`: the pair worked in other tumor types, with cited evidence. Prediction optional.
- `statistical`: similar drugs or similar lines behaved this way in revealed data. No prediction required.

**Prediction format.** A rule over a fixed vocabulary of cell features, so code can check it with no LLM involved:

```python
class Prediction(BaseModel):
    pair: tuple[str, str]
    outcome: Literal["syn", "inh"]
    rule: str        # e.g. "KRAS_mut == 1 and PTEN_loss == 0"
    direction: Literal["rule_true_higher"]
    registered_round: int
    hypothesis_id: str
```

Allowed rule terms: mutation status of a fixed gene list, copy-number loss or gain, lineage, and expression tertile of a gene. At most 2 terms joined by `and`. The rule parser rejects anything else.

**Scoring on held-out lines:**

1. Take held-out lines where this pair was measured. Skip the prediction if fewer than 8 lines, or fewer than 3 on either side of the rule; report it as untestable.
2. Compare the outcome in rule-true vs rule-false lines: AUROC and a one-sided Mann-Whitney test.
3. Correct across all predictions with Benjamini-Hochberg at 10% FDR.
4. Compare against two null rules for the same pair: (a) random rules with the same number of terms, (b) the best single-feature rule mined by code from the revealed training data. A mechanistic prediction is only interesting if it beats (b).

**Report:** fraction testable, fraction confirmed, AUROC distribution vs null, and accuracy split by explanation level and by famous vs obscure pair.

## 10. Leakage controls

Leakage has three routes (data access, retrieved papers, the LLM's own memory), and each gets its own control.

**1. Data access.** Only `Oracle.run()` reads labels (section 5). `query_model` and `get_history` return predictions and revealed results only. A test scans every tool output in a run log for values matching hidden labels to 3 decimals.

**2. Retrieved literature.** `retrieval.py` applies a blocklist before anything reaches the LLM:

- Papers that produced any pool study (from DrugComb study IDs mapped to PMIDs, e.g. the NCI-ALMANAC and O'Neil screens).
- Papers that reanalyze those screens: text match on screen names ("NCI-ALMANAC", "DrugComb", "O'Neil", "AZ-DREAM") plus citation of the source PMIDs where available.
- Any document that names a pool drug pair and a pool cell line together.
- Optional strict mode: only documents published before the earliest pool study.

Every retrieved document is saved with its ID and filter decision, so leakage can be audited after the run.

**3. LLM memory.** This cannot be removed, so we measure it:

- **Famous vs obscure.** Report every metric separately for the three pair groups from section 5.
- **Zero-shot probe.** Before any run, ask the LLM (no tools) to rank 200 random pool triplets by expected synergy. Correlation with true labels, by pair group, estimates how much it already knows.
- **Masked-name ablation.** Run the agent with drug names replaced by codes, keeping targets and MoA. If performance holds, the gain comes from mechanism reasoning; if it collapses only on famous pairs, it came from memory.

## 11. Evaluation, metrics and statistics

The primary endpoint, fixed before any agent run, is the number of synergy hits found within 320 experiments, agent vs `bo_ucb`, paired over 10 seeds.

**Run grid.** 8 selectors × 2 targets (`syn`, `inh`) × 10 seeds × 2 benchmarks (TDC DrugComb, OncoPolyPharmacology), plus the masked-name ablation for `agent`. Same seed = same seed set for every selector.

| Metric                 | Definition                                                                       | Question it answers              |
| ---------------------- | -------------------------------------------------------------------------------- | -------------------------------- |
| Hits at budget         | Count of `is_hit_*` triplets revealed by 80, 160, 320 experiments                | Main efficiency number           |
| Discovery curve AUC    | Area under cumulative hits vs experiments, normalized by the best possible curve | Speed over the whole run         |
| Experiments to 10 hits | First budget point reaching 10 hits (censored at 320)                            | How long until useful            |
| Enrichment vs random   | Hit rate / random's hit rate                                                     | Easy-to-read ratio               |
| Hit diversity          | Unique drug pairs and unique target-pair classes among hits                      | Finds new pairs, not near-copies |
| Cross-outcome yield    | In `syn` campaigns, `inh` hits found, and the reverse                            | Shows the two outcomes differ    |
| Cost per hit           | LLM dollars and tokens / hits                                                    | Practical cost                   |
| Prediction accuracy    | From section 9                                                                   | Do mechanisms hold up            |

Every metric is also reported split by famous, middle and obscure pairs.

**Statistics.** Paired Wilcoxon signed-rank test across seeds for each selector vs `agent`; bootstrap 95% CIs over seeds; Holm correction across the baseline comparisons for the primary endpoint.

**Outputs per experiment grid:** `results.parquet` (one row per run × round), discovery-curve plots with CI bands, a summary table and a short auto-generated `report.md`.

## 12. Milestones and build order

Build in this order; do not start a milestone until the previous one's gate passes. Milestones M1 to M4 need no LLM and should be fully working before any agent code.

**M1. Data and pool** (section 4)

- [x] Loaders for TDC DrugComb *(done 2026-11-08; OncoPolyPharmacology deferred to the second benchmark; DrugComb portal not used, unreachable)*
- [x] Cell line and drug ID mapping, cancer-only filter, replicate merge *(done; ChEMBL parent and Cellosaurus IDs; O'Neil rows removed, decision 13)*
- [x] Feature tables and frozen `pool.parquet` + `data_card.md` *(done; `labels.parquet` kept separate; see `docs/m1-freeze-and-splits.md`)*

Gate: data card reproduces from raw files; all section 4 checks pass.

**M2. Oracle and splits** (section 5)

- [ ] `Oracle` with budget, batch and held-out checks
- [ ] `splits.json`: held-out lines, 10 seed sets, famous/middle/obscure labels

Gate: label-access test and refusal tests pass.

**M3. Response model** (section 6)

- [ ] Deep ensemble with symmetric input and two heads
- [ ] Calibration report on a sanity split

Gate: 80% intervals cover 70% to 90%; swap test passes; refit under 60 s.

**M4. Baselines and runner** (sections 7, 11)

- [ ] `random`, `greedy`, `uncertainty`, `bo_ucb`, `bo_qei`
- [ ] `eval/runner.py`, metrics and discovery-curve plots

Gate: full baseline grid runs end to end; `bo_ucb` beats `random` on hits at 320 (if not, fix the model before going on).

**M5. Retrieval and leakage filter** (section 10)

- [ ] Literature search with blocklist and audit log
- [ ] Zero-shot memory probe

Gate: blocklist removes every source paper of pool studies in a test set; audit log written for every call.

**M6. Agent** (section 8)

- [ ] Port SHELDON modules; hypothesis and observation stores
- [ ] Tools, shortlist, surprise flags, batch assembly, critic, fallback
- [ ] `agent_no_lit`, `agent_single_hyp` and masked-name variants

Gate: 3-seed pilot on OncoPolyPharmacology runs within the cost cap; every batch pick cites a hypothesis.

**M7. Explanations and prediction checks** (section 9)

- [ ] Explanation records, rule parser, frozen predictions
- [ ] Held-out scoring with both null rules

Gate: rule parser rejects out-of-vocabulary rules; scoring reproduces on a synthetic case with a planted true rule.

**M8. Full grid and report** (section 11)

- [ ] Full run grid, statistics, `report.md`

Gate: every number in the report traces to `results.parquet`.

## 13. Open decisions and risks

The defaults above are set so work can start; the decisions below need an owner's call before M6.

**Open decisions:**

- [ ] Which LLM and which cost cap per run.
- [ ] Hit definition: 95th percentile of the pool (current default) or an absolute cut such as Bliss ≥ 10.
- [ ] Budget and batch size: 320 × 16 is a guess; check against how many hits the pool holds.
- [ ] Whether v1 includes DrugComb portal studies beyond the two TDC sets, or saves them for v2.
- [ ] Literature corpus: PubMed abstracts only, or full text where open access.

**Risks:**

| Risk                                                               | Effect                           | Mitigation                                                           |
| ------------------------------------------------------------------ | -------------------------------- | -------------------------------------------------------------------- |
| Few held-out lines (TDC DrugComb has 59 lines, so 20% is about 12) | Many predictions untestable      | Report testable fraction; consider leave-lineage-out folds           |
| Noisy synergy scores; Loewe can be inflated by Hill-slope bias     | False hits and misses            | Use Bliss as primary; keep replicate SD; check results under ZIP too |
| Shortlist built by the model limits the agent                      | Agent cannot go beyond BO's view | `resolve` tool and random wildcards in the shortlist                 |
| LLM already knows famous pairs                                     | Gain is memory, not reasoning    | Pair-group split, zero-shot probe, masked-name ablation              |
| LLM cost and run time                                              | Grid too slow                    | Pilot on OncoPolyPharmacology; cache all calls                       |

**Sources**

- [TDC drug synergy task page](https://tdc2-website-ara8agcycyfvapd0.eastus-01.azurewebsites.net/multi_pred_tasks/drugsyn) (dataset sizes and loaders)
- [Therapeutics Data Commons paper](https://arxiv.org/pdf/2102.09548)
- [SHELDON paper](https://arxiv.org/html/2610.18025) and [code](https://github.com/stanford-oval/sheldon)
- [DrugComb portal](https://drugcomb.org/)
- [Loewe bias note in DDoS paper](https://arxiv.org/pdf/2210.00802)
