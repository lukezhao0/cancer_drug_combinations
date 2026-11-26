# M5: literature search, leakage filter and memory probe

> **Known issues (2026-11-26), not yet fixed; see [`known-issues.md`](known-issues.md):**
> - **E1:** `pair_and_line` misses common spellings (a drug followed by a hyphen, "H460", Unicode
>   hyphens, brand names).
> - **E2:** the gate test set never triggers that rule.
> - **E3:** there are no M5 unit tests.
> - **E4:** probe calls send no temperature.
> - **E5:** treat §4's "confuses synergy with potency" as **tentative**. The prompt asks for both
>   outcomes one right after the other, which can produce the same correlation.

> Status 2026-11-25: **built.** The leakage gate passes on the hand-built test set (23/23 leaky
> papers blocked, 0/19 clean papers lost). The zero-shot memory probe has run on
> `gemini-3.5-flash-lite`, with real and with masked drug names. **Team decision: the probe is not
> repeated on `gemini-4-hydrogen`; this first look answers the question.** Open: `tests/test_leakage.py`
> (the gate is checked by script, not yet by a test) and the macOS sandbox gap (§5).
>
> Code: `src/combo_agent/literature/`, `src/combo_agent/eval/memory_probe.py`,
> `scripts/make_m5_testset.py`. Curated input: `data/curated/leakage_sources.csv`. Outputs:
> `data_cache/processed/m5/` (blocklist), `runs/m5_probe/` (probe runs), `artifacts/m5/memory_probe.md`
> (probe report, tracked).

## 1. What M5 answers

The agent must not win by seeing the answers. Design §10 names three ways answers can leak in, and
each gets its own control:

| Route | Example | Control | Where |
|:--|:--|:--|:--|
| 1. Data access | a tool returns a hidden measurement | only the oracle reads `labels.parquet`; tool outputs are scanned for label values | M2 (`test_label_access.py`), tools (`test_tools.py`) |
| 2. Retrieved papers | the agent reads a paper that reports our screen's results | **leakage filter** in front of every literature search | **M5, §2–3** |
| 3. The LLM's memory | the LLM read those papers during training | can't be removed, so it is **measured** by the memory probe | **M5, §4** |

## 2. `search_literature` and the leakage filter

One tool call does: PubMed search (fetching 4× more results than asked for, because the filter
removes some) → fetch abstracts → filter → **one audit record** → return up to `k` abstracts
(default 8, maximum 20; each cut to 1,500 characters).

- **Date cutoff.** Every search carries the M2 cutoff (`maxdate` 2026/11/01), so results don't
  change as PubMed grows.
- **Cache.** Every request goes through the same JSON cache as M1/M2. `COMBO_OFFLINE=1` replays a
  run with no network. The API key is never stored in the cache.
- **Audit log.** The record is written (and flushed to disk) *before* anything is returned. It lists
  every PMID retrieved, whether it was shown, and every rule that fired. A failed call also leaves
  a record, with the error.
- Registered in `tools/registry.py` as `search_literature(query, k)`.

### The rules (`literature/leakage.py`)

A document is withheld if **any** rule fires; all rules that fire are recorded.

| Rule | Blocks | Example |
|:--|:--|:--|
| `source_paper` | the papers that produced the screens (curated list below) | the NCI-ALMANAC paper |
| `cites_source` | papers that cite a source: PubMed's cited-by links (built once) **or** the record's own reference list | most synergy-prediction papers |
| `screen_name` | text naming a screen or a database that redistributes it: NCI-ALMANAC, DrugComb, SynergxDB, O'Neil, AZ-DREAM, NCI-60 combination screens. Ambiguous names ("ALMANAC", "O'Neil", "NCI-60") count only next to a combination word | "…using the NCI-ALMANAC dataset…" |
| `pair_and_line` | text naming **both drugs of a pool pair and a pool cell line** | "cisplatin plus gemcitabine in A549 cells" |
| `synergy_ml` | machine-learning papers that predict synergy or combination response. Nearly all train on these screens, often without naming them in the abstract (DeepDDS showed the rule was needed) | "a graph neural network to predict drug synergy" |
| `strict_date` | optional strict mode: anything published in 2017 (the NCI-ALMANAC paper) or later | off by default |

Cell-line names are matched with their spelling variants ("MDA-MB-231", "MDA MB 231",
"MDAMB231"). "SR" and "M14" are ordinary words, so they count only next to "cells"/"cell line".

### The blocklist (`python -m combo_agent.literature.leakage build`)

The build first checks that every curated PMID resolves to the curated title on PubMed. A typo would
otherwise silently unblock a source. It then collects each source's citing papers.

| PMID | Source | Role | Citers blocked |
|:--|:--|:--|--:|
| 28446463 | NCI-ALMANAC (2017) | produced every pool experiment | 202 |
| 26983881 | O'Neil (2016) | second TDC screen, removed from the pool in M1 | 200 |
| 31066443 | DrugComb (2019) | portal that redistributes NCI-ALMANAC with our Bliss/CSS scores | 145 |
| 34060634 | DrugComb update (2021) | same portal, second release | 60 |
| 31209238 | AZ-DREAM (2019) | combination-prediction benchmark | 184 |
| 29253077 | DeepSynergy (2018) | reanalysis of O'Neil; root of the synergy-ML literature | 303 |
| 16990858 | NCI-60 review (2006) | panel the lines come from; blocked itself, citers **not** blocked (1,224, mostly single-drug work) | – |

**607 distinct PMIDs** are on the list (604 blocked by citation). The text rules and the reference
list check run at query time, so a paper missing from PubMed's cited-by links is still caught if it
names a screen, cites a source, or names a pool pair with a pool line.

## 3. The leakage gate

**Gate (design §12):** the blocklist removes every source paper of the pool studies in a test set,
and an audit record is written for every call.

**Test set** (`scripts/make_m5_testset.py` → `tests/fixtures/m5/`, 42 real PubMed records). The
answers were chosen by hand, not by running the filter:
- **23 positives (must be blocked):** the 7 curated sources, plus 16 papers known to reanalyse or
  redistribute the screens (MatchMaker, TranSynergy, DeepDDS, comboFM, DrugCombDB, SYNERGxDB, a
  phase I trial of an NCI-ALMANAC hit, …).
- **19 negatives (should pass):** ordinary mechanism reviews and clinical trials of pool drugs.

**Result (2026-11-24, offline, from the fixtures):**

| | Blocked | Passed |
|:--|--:|--:|
| Positives: sources (7) | **7** | 0 |
| Positives: reanalyses (16) | **16** | 0 |
| Negatives (19) | 0 | **19** |

The gate passes: 100% of leaky papers are blocked and no clean paper is lost.

**Limits:**
- The negatives were picked by hand, so **"0 lost" doesn't measure how much clean literature the
  filter removes in real agent searches.** `pair_and_line` in particular blocks legitimate
  mechanism papers that happen to test a pool pair in a pool line. That's intended (such a paper
  may report our exact result), but it costs evidence. Measure the blocked fraction from the audit
  logs of the first M6 pilot.
- **No test file yet.** `make_m5_testset.py` refers to `tests/test_leakage.py`, which doesn't
  exist. The result above was produced by a script over the fixtures. The audit-record half of the
  gate also has no test.

## 4. The zero-shot memory probe

### What it is

Before any campaign, we ask the LLM, **with no tools and no data**, to score 200 pool experiments
(drug A + drug B in cell line X) from 0 to 100 for expected **synergy** and expected **killing**.
Then we compare its guesses to the hidden results.

**Analogy:** before an open-book exam, give the student the questions with the book closed. If
they already score well, a good score later with the book open doesn't prove they used the book.

- **Sample:** 200 selectable experiments (held-out lines excluded), split evenly across famous,
  middle and obscure pairs (67/67/66; fame labels from M2), seed 0. Asked in 10 batches of 20.
- **Prompt:** `literature/prompts/memory_probe_v1.md`. Each experiment shows the two drug names
  with their mechanism of action, and the line's name, lineage and disease.
- **Masked variant:** drug names replaced by codes ("Drug-017"), mechanism and line kept. Memory
  is tied to names; mechanism reasoning survives masking.
- **Scoring** (`eval/memory_probe.py`, allowlisted label reader): Spearman ρ with the measured
  outcome (rank agreement: 0 = chance, 1 = perfect) with a bootstrap 95% CI, overall and per
  fame group, plus the hit rate among the LLM's top 10%.

**What memory would look like:** (1) ρ clearly above 0; (2) ρ highest on famous pairs, which the
LLM has read most about; (3) ρ dropping when names are masked.

### Results (`gemini-3.5-flash-lite`, 2026-11-24; report `artifacts/m5/memory_probe.md`)

Both runs answered 200/200, at $0.023 and $0.024. Gemini rate limits (HTTP 429) were retried
automatically.

| ρ with the measured outcome | Synergy, real names | Synergy, masked | Killing, real names | Killing, masked |
|:--|:--|:--|:--|:--|
| All 200 | **−0.05** (−0.19 to 0.08) | −0.10 (−0.24 to 0.02) | **0.40** (0.29 to 0.51) | 0.33 (0.20 to 0.45) |
| Famous (67) | −0.01 (−0.25 to 0.22) | −0.18 | 0.36 (0.13 to 0.56) | 0.32 |
| Middle (67) | −0.17 (−0.40 to 0.07) | −0.03 | 0.48 (0.29 to 0.64) | 0.34 |
| Obscure (66) | −0.04 (−0.27 to 0.20) | −0.23 | 0.32 (0.10 to 0.53) | 0.26 |

### What it means

1. **Synergy: no detectable memory.** None of the three signs appears. ρ is ≈ 0 overall (upper
   bound 0.08), famous pairs are no better than obscure ones, and there is nothing for masking to
   remove. On the primary endpoint, the LLM starts with no recalled advantage.
2. **Killing: real knowledge, but general pharmacology, not recalled results.** ρ = 0.40 with no
   data is substantial; the M3 model reaches r ≈ 0.5 after 320 measurements. But famous pairs
   aren't easier (middle pairs score highest), and masking barely changes it (0.40 → 0.33). Killing
   is largely set by how potent the stronger drug is and how fragile the line is, which a
   pharmacologist knows without having read about the pair. This is legitimate prior knowledge.
3. **The LLM confuses synergy with potency.** Its synergy and killing guesses rank the experiments
   almost identically: ρ(predicted synergy, predicted killing) = **0.81** (0.87 masked). In the
   data the two outcomes are unrelated: ρ(true synergy, true killing) = **0.01** on these 200,
   consistent with 0.06 over the whole pool (M1). Its synergy guesses track true *killing*
   (ρ = 0.29), not true synergy. Asked "how synergistic?", it answers "how strong?".

### Caveats

- **Small groups.** About 67 experiments per fame group gives CIs of about ±0.2. A large fame
  effect is ruled out; a modest one on famous synergy (up to ρ ≈ 0.2) is not.
- **Ignore the top-10% hit rates per group.** Each group holds only 2–5 true hits, so "0.14" is one
  hit out of 7.
- **Masking is a leaky control for well-known drugs.** The mechanism text can identify a famous
  drug ("platinum compound that crosslinks DNA" is essentially cisplatin), so masking separates
  name memory from mechanism reasoning least cleanly where memory is most likely.
- **The result is for `gemini-3.5-flash-lite` only.** Larger models tend to memorize more. The team
  chose not to repeat the probe on `gemini-4-hydrogen`. If reported agent runs use Hydrogen, the
  probe doesn't cover that model directly. The fame-group split of every M6 metric and the
  masked-name ablation remain the controls for it.

## 5. What this means for M6

- **Synergy campaigns:** don't let the LLM score pairs for synergy from intuition; it will steer
  toward potent pairs. Take the synergy signal from the response model and the revealed results,
  and use the LLM for mechanisms that explain surprises. Define synergy explicitly in the agent
  prompt ("better than each drug alone predicts, not overall killing").
- **Killing campaigns:** an agent advantage, especially early, may come from the LLM's general
  potency prior rather than from literature. `agent_no_lit` separates the two; report killing gains
  with that in mind.
- **Fame:** M4's model-based selectors already drift toward famous pairs for killing (greedy 54.4
  famous vs 10.3 obscure hits at 320; random 4.3 vs 5.6). An agent's famous-pair advantage has to be
  read against that drift, not only against LLM memory.
- **Sandbox gap (from the tools layer):** on macOS, `run_python` is a plain subprocess, so the agent
  process could read `labels.parquet`. Scored runs need the labels out of the agent's reach (Linux
  isolation, or the file kept out of its filesystem view). Not resolved yet.
- **Leakage cost:** measure the fraction of retrieved papers blocked, by rule, from the first
  pilot's audit logs.

## 6. How to reproduce

```bash
uv run python -m combo_agent.literature.leakage build                    # blocklist (needs NCBI_API_KEY or the cache)
COMBO_OFFLINE=1 uv run python -m combo_agent.literature.leakage build    # replay from cache, byte-identical
uv run python scripts/make_m5_testset.py                                 # leakage test-set fixtures

uv run python -m combo_agent.literature.probe --model gemini-3.5-flash-lite
uv run python -m combo_agent.literature.probe --model gemini-3.5-flash-lite --masked
uv run python -m combo_agent.eval.memory_probe runs/m5_probe/* --out artifacts/m5/memory_probe.md
```

LLM answers are cached in `data_cache/llm_cache/llm.db`, so re-running the probe costs nothing.

## 7. Files

| File | What it is |
|:--|:--|
| `literature/pubmed.py` | cached E-utilities client (search, fetch, summaries, cited-by), date cutoff |
| `literature/leakage.py` | the rules, `LeakageFilter`, the blocklist build |
| `literature/retrieval.py` | `search_literature` and its audit log |
| `literature/probe.py`, `literature/prompts/memory_probe_v1.md` | the memory probe (never reads labels) |
| `eval/memory_probe.py` | probe scoring (reads labels; on the allowlist) |
| `data/curated/leakage_sources.csv` | the 7 curated sources, each with why it is blocked |
| `data_cache/processed/m5/` | `blocklist.parquet`, `m5_blocklist_card.md`, `manifest.json` |
| `tests/fixtures/m5/` | leakage test set (42 records) and lexicon |
| `runs/m5_probe/<model>[-masked]/` | prompts, calls, predictions, scores, report, manifest |
| `artifacts/m5/memory_probe.md` | combined probe report (tracked) |
