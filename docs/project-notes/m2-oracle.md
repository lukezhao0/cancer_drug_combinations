# M2: the oracle, seed sets and pair fame

> Status 2026-11-23: **M2 complete.** Gate passed: the label-access test and every refusal test
> pass. Code: `src/combo_agent/lab/`. Tests: `tests/test_oracle.py`, `tests/test_splits.py`,
> `tests/test_label_access.py`. Outputs: `data_cache/processed/m2/` (tracked copies of the small
> files in `artifacts/m2/`).

## 1. What M2 is, in one picture

M1 froze a table of 272,357 real experiments. M2 turns that table into a **simulated lab**. A
selector (random, Bayesian optimization, later the agent) can't look at the answers. It has to
*ask* for an experiment, pay one unit of budget, and only then see the result.

```
                ┌──────────────────────── selectors (M4, M6) ─────────────────────────┐
                │   random      greedy      bo_ucb      ...      agent (LLM + model)  │
                └───────────────┬──────────────────────────────────────▲─────────────┘
          oracle.candidates()   │  oracle.run([≤16 triplets])          │ results: y_syn, y_inh,
          (what's still untested)│                                      │ SDs, n_reps
                                ▼                                      │
   ┌──────────────────────────────── Oracle (lab/oracle.py) ───────────┴─────────────┐
   │  checks, all-or-nothing:  batch ≤ 16 · no duplicates · not already run ·         │
   │                           in the pool · not a held-out line · budget left        │
   │  charges budget, logs every reveal (round, selector, triplets, values)          │
   │                                                                                 │
   │  private: labels.parquet  ◀── the ONLY reader on the selector path              │
   └─────────────────────────────────────────────────────────────────────────────────┘
         round 0: the seed set (32 experiments) is revealed free, same for every selector
```

**Analogy:** the oracle is a lab technician with a locked notebook of results. You hand them a
request form with at most 16 experiments. They check it, refuse it whole if anything on it is
invalid, run the valid ones and read you back the results. Each experiment costs one unit of a
320-unit budget.

## 2. The oracle (`lab/oracle.py`)

| Method | What it does |
|:--|:--|
| `Oracle.from_cache(seed=s, selector=...)` | Loads the frozen pool, labels, thresholds and seed set `s` |
| `candidates()` / `candidate_frame()` | Untested, selectable experiments (219,239 at the start), as a list or as pool rows. No outcomes |
| `run(batch)` | Reveals a batch, or raises `OracleError` and changes nothing |
| `history()` | Everything revealed so far, seed set first |
| `budget_left()`, `round` | Budget and round counters (the seed set is round 0, free) |
| `is_hit(result, "syn" or "inh")` | Hit test against the public thresholds (y_syn ≥ 9.30, y_inh ≥ 62.95) |

What a result contains: `y_syn` (mean Bliss synergy), `y_inh` (mean CSS killing), their replicate
SDs (NaN when there was one replicate) and `n_reps`. There are no hit flags; `is_hit` derives
them, and on the real pool it agrees with the frozen flags for every revealed experiment (tested).

**Refusals** (each tested; a refused batch leaves budget, round, history and candidates
unchanged):

- an empty batch, or one larger than `batch_size` (16);
- the same experiment twice in one batch; drug order doesn't matter, so (B, A, line) = (A, B, line);
- re-running an experiment already revealed, including seed-set ones (design §5: no re-runs in v1);
- an experiment not in the pool;
- **any held-out line**: the 11 lines saved for checking mechanistic predictions (M7);
- more experiments than the budget left.

Each reveal is written to an optional JSONL log, for audits and for rebuilding discovery curves.

## 3. Seed sets

Every campaign starts with 32 experiments revealed free (design §5.2). Seed `s` (0–9) picks them
uniformly from the selectable experiments with `numpy.random.default_rng(s)`. **The same seed
gives every selector the same start**, so selectors are compared pairwise, seed by seed.

| Seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Synergy hits in the seed set | 2 | 1 | 2 | 3 | 1 | 3 | 0 | 2 | 1 | 0 |
| Killing hits | 2 | 2 | 2 | 4 | 1 | 3 | 2 | 3 | 1 | 0 |

The expected count at the 5% hit rate is 1.6 per set. No seed-set synergy hit is a low-kill hit, so
the "without low-kill" counts are the same. Seeds 6 and 9 start with **no synergy hit at all**, a
genuinely cold start. That's fine: the paired comparison handles it, but expect more spread across
seeds on those two.

## 4. Pair fame (famous / middle / obscure)

**Why:** a language model may already "know" that cisplatin + 5-FU works, from papers. If the
agent wins only on famous pairs, that's memory, not reasoning. So every result will be reported
separately for famous, middle and obscure pairs (design §5.3, §11).

**How:**

1. For each of the 4,939 pool drug pairs, PubMed counts records whose title or abstract names both
   drugs together with a cancer term:

   ```
   ("cisplatin"[tiab] OR "CDDP"[tiab] OR ...) AND ("fluorouracil"[tiab] OR "5-FU"[tiab] OR ...) AND (neoplasms[mh] OR cancer*[tiab] OR ...)
   ```

   - Publication dates are capped at 2026/11/01, so the counts are stable.
2. Drug terms come from the reviewed table `src/combo_agent/data/curated/drug_pubmed_terms.csv`
   (101 drugs). Each row gives the generic name plus synonyms in common use, e.g. sirolimus →
   rapamycin, SN-38 → 7-ethyl-10-hydroxycamptothecin, cytarabine → ara-C, and code names like
   STI571.
   - Raw database names would undercount: "DOCETAXEL ANHYDROUS" and "ELOXATIN (TN) (SANOFI
     SYNTHELAB)" are not what papers say.
3. Classes:
   - **obscure** = 0 co-mentions;
   - **famous** = at least the upper quartile (22 records);
   - **middle** = the rest.

| Class | Pairs | Median records | Selectable experiments | Synergy hit rate | (without low-kill) | Killing hit rate |
|:--|--:|--:|--:|--:|--:|--:|
| famous | 1,249 (25.3%) | 81 | 55,020 | 4.76% | 4.51% | 5.58% |
| middle | 2,362 (47.8%) | 4 | 105,118 | 5.26% | 4.88% | 4.91% |
| obscure | 1,328 (26.9%) | 0 | 59,101 | 4.76% | 4.32% | 4.63% |

**Sanity checks against known oncology:**
- The most co-mentioned pairs are the classic regimens: doxorubicin + cyclophosphamide (11,997),
  doxorubicin + vincristine, cyclophosphamide + vincristine (the CHOP backbone), cisplatin +
  5-FU, carboplatin + paclitaxel, and 5-FU + oxaliplatin (FOLFOX).
- Imatinib + dasatinib, letrozole + tamoxifen and bortezomib + lenalidomide are famous.
- Amifostine + cisplatin has 204 records (amifostine protects the kidneys during cisplatin
  therapy).
- Amifostine + quinacrine and triethylenemelamine + pipobroman have 0.

**Finding:** in this screen, **fame barely changes the hit rate.** Synergy hits are 4.8–5.3% in
every class, and killing hits 4.6–5.6%. So a famous pair isn't an easier target, and the
famous/obscure split cleanly separates "the model may remember it" from "the pair is easy." One
consequence: if the agent does better on famous pairs, that gap is more likely memory or
literature access than easier targets.

**Reproducibility:** all 5,040 PubMed answers are cached in `data_cache/raw/api_cache/pubmed/`.
`python -m combo_agent.lab.splits --offline` rebuilds byte-identical outputs with no network.
`NCBI_API_KEY` (in `.env`) raises the rate limit from 3 to 10 requests/s. It is appended only
when a request is sent, so it never appears in the cache (tested). A full live run took roughly
20 minutes.

**Caveats:**
- These are co-mentions, not evidence the pair was tested together.
- A few short abbreviations (5-FU, ara-C, ATRA, SAHA, DTIC) can match unrelated text.
- SN-38 is counted under its own names, not irinotecan's.
- `tumor*` also matches "tumor necrosis factor", and some drugs have non-cancer uses (sirolimus and
  paclitaxel on heart stents). For that pair, 16 of 234 records mention stents and 5 match only
  through "tumor necrosis". It's famous either way.
- Fame describes the literature, not what a particular language model has seen. The design's
  zero-shot memory probe (M5) measures that directly.

## 5. Label access

`labels.parquet` holds the answers. `tests/test_label_access.py` scans every Python file in `src/`
and `scripts/`, and fails if anything outside this allowlist names the labels file:

| Allowed | Why |
|:--|:--|
| `data/freeze_pool.py` | writes it (M1) |
| `lab/oracle.py` | the oracle |
| `lab/splits.py` | hit counts for the M2 card only; writes no outcomes |
| `tools/store.py` | names it on the tools denylist |
| `scripts/make_data_bundle.py`, `scripts/make_m1_figures.py` | packaging and M1 figures |

A new selector or model that reads labels fails this test until someone deliberately allowlists
it. The evaluation code (M4, `eval/`) and held-out scoring (M7) will be added to this allowlist
when they're written.

## 6. Files

```
data_cache/processed/m2/          (git-ignored; small copies tracked in artifacts/m2/)
  seed_sets.json      10 seeds x 32 triplets
  pair_fame.parquet   drug_a, drug_b, n_pubmed, fame, query     (4,939 rows)
  drug_pubmed.parquet drug_id, n_pubmed_cancer, query           (101 rows)
  m2_card.md          the tables above, regenerated byte-for-byte
  manifest.json       sha256 of inputs and outputs, date cutoff, request counts, build date
```

M1's `splits.json` is unchanged. The `m1-data-v1` bundle doesn't contain these files; a later
bundle can add them.

## 7. Also fixed in this round

The tools' real-data leak test (`tests/test_tools.py`) used to take more than 10 minutes:
`FeatureStore` re-read the 23,000-column parquet on every gene lookup (0.5 s each). It now reads
each gene table once (`store._wide_tables`). With that fixed, the test finished and failed on
false positives: Mann–Whitney p-values printed to 3 significant figures (e.g. `0.00219`) "matched"
hidden labels at 3 decimals. The ~85k distinct hidden values cover almost every 3-decimal number
in [−1, 1]. The scan now masks p-value fields, and a new unit test shows it still catches a real
label. After the fix, nothing matches.

## 8. Next: M3, the response model

M3 is a deep ensemble that predicts both outcomes with uncertainty, refit on each campaign's
history. Gate: 80% intervals cover 70–90%; predictions are unchanged when the drug order is
swapped; a refit takes under 60 s. It plugs into the oracle as the model every selector shares.
It also switches on the agent tools `query_model` and `resolve`.
