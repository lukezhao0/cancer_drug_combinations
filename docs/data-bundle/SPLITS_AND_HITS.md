# Splits and hits

## The held-out cell lines

**Why hold lines out?** The project's second question is whether the mechanisms an agent proposes
correctly predict which **other** cell lines respond. To test that fairly, some cell lines are kept
completely out of sight: no selector, model or agent ever sees their results. They are used only in
the final mechanism-prediction check (design §9).

**How they were chosen** (frozen in `data_cache/processed/splits.json`, seed 0):

1. **20% of lines:** 11 of 57.
2. **Stratified by lineage (tissue):** each tissue gives about 20% of its lines, using
   largest-remainder quotas.
3. **Families stay together:** related lines (SNB-19 and U-251 MG) are held out together or not at
   all, so a near-copy of a test line can't sit in training.
4. **Eligibility:** only lines whose features are **all their own measurements** can be held out.
   These always stay selectable:
   - SR (imputed expression);
   - SNB-19 and NCI-H322M (mutation calls borrowed from parent lines);
   - HL-60(TB) (DepMap profile borrowed from HL-60);
   - U-251 MG (SNB-19's family).

| Lineage | Lines | Held out | Held-out lines | Never held out (ineligible) |
|:--|--:|--:|:--|:--|
| Bowel | 7 | 1 | HCT 15 | |
| Breast | 5 | 1 | BT-549 | |
| CNS/Brain | 6 | 1 | SF268 | SNB-19, U-251MG |
| Kidney | 8 | 2 | 786-O, A-498 | |
| Lung | 8 | 2 | A-549, HOP-92 | NCI-H322M |
| Lymphoid | 4 | 1 | CCRF-CEM | SR |
| Myeloid | 2 | 0 | | HL-60(TB) |
| Ovary/Fallopian Tube | 6 | 1 | IGROV-1 | |
| Pleura | 1 | 0 | | |
| Prostate | 2 | 0 | | |
| Skin | 8 | 2 | LOX-IMVI, UACC-257 | |

| Split | Lines | Experiments | Synergy hit rate | Killing hit rate |
|:--|--:|--:|--:|--:|
| selectable | 46 | 219,239 | 5.00% | 5.00% |
| heldout | 11 | 53,118 | 4.57% | 4.31% |

### How to use the split in code

```python
pool = pd.read_parquet(f"{ROOT}/processed/pool.parquet")
selectable = pool[pool.split == "selectable"]    # everything a selector or model may use
heldout    = pool[pool.split == "heldout"]       # only for the final mechanism check
```

- `cells.parquet` has the same `split` column per line, plus `eligible_for_heldout`.
- **Never** fit scalers, choose features or tune anything on held-out lines' *outcomes*. Using
  their *features* (mutations, expression) is fine: that's how a prediction gets checked on them.
- 1 drug pair is measured only on held-out lines, so it can never be selected.

### Not built yet (M2)

- **Seed sets:** at round 0 every selector gets the same 32 random selectable experiments for free.
  Seeds 0–9 give 10 sets.
- **Famous / middle / obscure pairs:** per drug pair, count of PubMed records mentioning both drugs
  with a cancer term. Top quartile = famous; zero = obscure; the rest = middle. Used to check
  whether a method merely "remembers" well-known pairs.

## What counts as a hit

A **hit** is an experiment in the top 5% for its outcome, among **selectable** experiments. The same
cut-off applies to held-out lines.

| Outcome | Column | Meaning | Hit threshold | Flag |
|:--|:--|:--|--:|:--|
| Synergy | `y_syn` = mean Bliss | the pair beats what independent action of the two drugs predicts | **≥ 9.30** | `is_hit_syn` |
| Killing | `y_inh` = mean CSS | how well the pair kills cells overall (drug combination sensitivity score) | **≥ 62.95** | `is_hit_inh` |

- **The two outcomes are different questions:** Spearman 0.06 between them. Track and report them
  separately.
- **`syn_hit_low_kill`** flags 958 synergy hits (7.1%) whose killing score is negative. **Report
  every synergy result both with and without them** (team decision 20). The reason is in
  [`CAVEATS.md`](CAVEATS.md).
- **Noise:** the same experiment measured twice agrees at r = 0.21 for Bliss and 0.58 for CSS. A
  synergy value near 9.30 is partly luck.

## The design's evaluation, for reference (design §11)

| Metric | Definition |
|:--|:--|
| Hits at budget | `is_hit_*` experiments revealed by 80, 160 and 320 experiments; **primary: synergy hits at 320, paired over 10 seeds** |
| Discovery curve AUC | area under cumulative hits vs experiments, normalized to the best possible curve |
| Experiments to 10 hits | first budget point reaching 10 hits (censored at 320) |
| Enrichment vs random | hit rate ÷ random's hit rate |
| Hit diversity | unique drug pairs and target-pair classes among hits |
