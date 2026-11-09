# Caveats: read before analysing

Each item says what's unusual, how to recognise it in the data, and what to do about it.

## 1. SR's expression is imputed

- **What.** NCI CellMiner's RNA-seq profile labelled "SR" is a myeloma sample (it matches
  RPMI-8226), not SR, which is an ALK+ anaplastic large-cell lymphoma. SR's drug responses, its DNA
  and CellMiner's own **microarrays** of SR all show the true lymphoma.
- **What we did.** SR's expression on the RNA-seq scale was **imputed** from those microarrays,
  translated gene by gene using the 56 lines measured both ways. A leave-one-out test recovered
  each line's own signal at r ≈ 0.94.
- **How to recognise it.** In `processed/cells.parquet`: `expression_imputed = 1` and
  `expression_source = "imputed"`.
- **What to do.**
  - Say so wherever SR's expression is used.
  - **Genes only SR expresses are underestimated.** For example, ALK is imputed at 2.8
    log2(FPKM+1), while DepMap's independent measurement is 6.2. Never quote SR-only gene levels as
    measured.
  - SR is never a held-out line.

## 2. Borrowed mutation calls

- **What.** Three lines have no sequencing of their own in DepMap, so their driver-mutation calls
  come from a confirmed parent line:
  - **SNB-19** from U-251 MG (SNB-19 is a U-251 MG derivative);
  - **NCI-H322M** from NCI-H322;
  - **HL-60(TB)** from HL-60.

  The relationships are confirmed by expression and drug-response similarity.
- **How to recognise it.** `mutation_source = "depmap_parent"` and `mutation_from_parent = 1`.
  HL-60(TB) also has `depmap_is_proxy = True`.
- **What to do.** Cite these as "from parent X". None is a held-out line.

## 3. Synergy hits that barely kill (`syn_hit_low_kill`)

- **What.** 958 synergy hits (7.1%) have a **negative** killing score. Bliss measures surprise
  relative to the single drugs. When a single drug does almost nothing, or slightly stimulates
  growth, the expected value is very low and a modest combination looks very synergistic.
- **Where they come from.** They cluster on weakly cytotoxic or hormonal drugs: mitotane,
  raloxifene, hydroxyurea, estramustine and erlotinib.
- **How to recognise them.** `labels.syn_hit_low_kill = 1`.
- **What to do.** **Report every synergy result with and without them** (team decision 20).

## 4. Bliss synergy is noisy

- **What.** For the 9,236 experiments measured twice, the two measurements agree at r = 0.21 for
  Bliss and 0.58 for CSS. The median replicate SD is 2.9 for Bliss and 6.5 for CSS.
- **How to recognise it.** `labels.n_reps`, `bliss_sd` and `css_sd`. The SD is empty when
  `n_reps = 1`.
- **What to do.** Predict with uncertainty. Average over seeds, as the design does. Don't
  over-interpret single hits near the threshold.

## 5. Limited testability on held-out lines

- **What.** No held-out line carries an NRAS, HRAS, ERBB2, CTNNB1, RB1 or ATM mutation, so a
  mechanistic prediction that hinges on those genes can't be checked on held-out lines.
- **What to do.** Report the fraction of predictions that are testable (design §13 risk table).
  Copy-number and expression terms widen what can be tested.

## 6. Only NCI-ALMANAC is in the pool

- **What.** The source file (TDC "DrugComb", NCI-60) mixed three studies. Only NCI-ALMANAC was kept:
  - **O'Neil/Merck rows were removed.** On the 546 experiments both studies measured, they
    disagree (CSS r = 0.22; O'Neil's mean is 56, ALMANAC's 29).
  - **A 10-row third study was removed.**
  - **1,076 rows that couldn't be attributed to either study were removed.**
- **Effect.** 16 O'Neil-only drugs (mostly targeted agents) are not in this pool. They return with
  the O'Neil second benchmark.
- **Where.** `interim/row_fate.parquet` records what happened to every source row.
  `interim/oneil_rows.parquet` and `interim/cross_study_overlap.parquet` hold the removed data,
  for analysis only.

## 7. Drug annotations

- **Drugs with no ChEMBL target are kept** (`target_status = "none_in_chembl"`; 3 drugs in the
  pool). The agent is meant to look up their mechanisms in the literature.
- **27 pool drugs target DNA** (alkylators, platinum drugs). They have a target, but no target
  gene: `has_target = 1`, `has_target_gene = 0`.
- **Two drugs borrow mechanism records** from another ChEMBL entry (`target_status =
  "chembl_proxy"`): fludarabine (from its phosphate prodrug) and plicamycin.
- **The 3 platinum drugs' fingerprints** come from the source structure, because ChEMBL stores none
  (`smiles_from_source = 1`).
- **The source structures had no stereochemistry.** Fingerprints use ChEMBL's structures where
  available.

## 8. Dropped cell lines

MDA-MB-435 and NCI/ADR-RES have no DepMap entry and are **dropped**: they are contaminated copies of
M14 and OVCAR-8, which stay. Their rows are counted as `line_dropped` in `row_fate.parquet`.
