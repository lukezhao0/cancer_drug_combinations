# DepMap data: what we downloaded, and why

Downloaded by hand from the DepMap portal (`depmap.org/portal/data_page/?tab=allData`) into
`data_cache/raw/depmap/` in the code repo on 2026-11-08. DepMap's download API shows scripts a
Cloudflare check, so the files can't be fetched automatically. Licence: CC BY 4.0; cite the release
in the data card.

**Release: DepMap Public 26Q3** (confirmed by the team, 2026-11-08). The sha256 values below pin
the exact bytes.

## 1. The big picture

The drug-combination screen tells us *what happened* when drug A + drug B was applied to cell line
Z. To the model, though, a cell line is only a name. **DepMap is the cell line's medical chart:**
where it came from, which genes are broken, which genes are switched on, and which chromosome
regions are gained or lost.

We need the chart for three reasons:
1. **The response model** needs to know which lines are alike, so a result on one line informs
   predictions for similar lines.
2. **The agent's mechanisms** are written in these terms. For example, "this pair works in
   KRAS-mutant lines".
3. **The evaluation** checks those mechanistic predictions on held-out lines, using rules over
   mutation, copy number, lineage and expression (design §9).

Nothing here describes an *outcome*: every file is measured before any drug is applied. So using it
leaks nothing about the hidden experiment results.

## 2. The files

| File | Size | sha256 (first 12) | What it is, intuitively | Why we need it | Used in |
|:--|--:|:--|:--|:--|:--|
| `Model.csv` | 0.8 MB | `65eb1895800e` | **The address book.** One row per DepMap model (2,554): its ID (`ACH-…`), names, **lineage** (tissue, e.g. Lung or Myeloid), disease subtype, sex, and `RRID` (its Cellosaurus accession) | Lineage, for features and for stratifying the held-out split. Its `RRID` column independently checks 1d's Cellosaurus → DepMap mapping. It settled decision 9 | 1d check, 1f, 1g |
| `OmicsProfiles.csv` | 0.9 MB | `4241e537f0b2` | **The lab-visit log.** Which sequencing runs (RNA, whole-exome, whole-genome) exist for each model, and which run is the default | Tells us which lines actually have data behind their ID. That's how we found that SR's and UO-31's second IDs have none | 1d check, 1f |
| `OmicsSomaticMutationsMatrixHotspot.csv` | 2.7 MB | `0cbdacdce235` | **The "switched on" checklist.** A 0/1 table: does this line carry a known *activating* hotspot mutation in this gene? (e.g. KRAS G12, BRAF V600) | **Oncogenes** cause cancer when a specific spot is stuck "on", so a hotspot flag is the right test for KRAS, BRAF, PIK3CA, NRAS… | 1f |
| `OmicsSomaticMutations.csv` | 642 MB | `b8f89677d4ce` | **The full mutation list.** One row per mutation (1.39 M rows): gene, protein change (e.g. `p.G12S`), and flags such as `LikelyLoF`, `Hotspot`, `OncogeneHighImpact`, `TumorSuppressorHighImpact` | **Tumour suppressors** cause cancer when they're *broken*, so we need a "damaging" flag for TP53, PTEN, etc. We derive it from `LikelyLoF` / `TumorSuppressorHighImpact`; that's why the separate "Damaging" matrix wasn't needed. It also gives exact variants for the agent's explanations | 1f, agent tools |
| `OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv` | 395 MB | `252d1fedc701` | **The volume dial for every gene.** For each line (2,314 profile rows) and each of ~19,200 protein-coding genes: how strongly the gene is expressed, as log2(TPM + 1). 0 means off | The cell's *current state*, beyond its fixed DNA. The design uses the **1,000 most variable genes** as features, and "expression tertile of gene X" as a rule term | 1f |
| `PortalOmicsCNGeneLog2.csv` | 631 MB | `b4edee6e57f3` | **The photocopy count.** For each line (1,763) and gene: how many copies of that gene's chromosome region the cell has, relative to normal (log2 scale). Gains switch genes up (e.g. MYC, ERBB2); losses delete them (e.g. CDKN2A) | Copy-number gain/loss is an allowed term in the mechanism rules (design §9) and a standard cancer feature | 1f, explanations |

Full sha256 values (2026-11-08):

```
65eb1895800e71a963a3ab56e5d08c8d006cb0d9e7aa68f7e34a3810a2a3cbc7  Model.csv
4241e537f0b23139c0e525b4d78fb53ee65c5507771a33e2cd5debd78f7179ea  OmicsProfiles.csv
0cbdacdce2355d2dd600d3741a36ae093dbe77d7bee44bde97411b1e96973058  OmicsSomaticMutationsMatrixHotspot.csv
b8f89677d4ce2b724e48e2276f9a47f5b6fdaff3cab0bf115f90ef97c1be1cfd  OmicsSomaticMutations.csv
252d1fedc7014953b19ffe5d68151b4df973c8078b12c9e5ccd97e292548d177  OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv
b4edee6e57f30640f2c2b8a6163ddbcdb9404384e77dfb8a1398972e693b30f7  PortalOmicsCNGeneLog2.csv
```

**How the files join:** every table is keyed by `ModelID` (`ACH-…`). We get that from 1d's
`depmap_id` (Cellosaurus → DepMap) and cross-check it against `Model.csv`'s `RRID`. The omics tables
can hold several profiles per model, so use the row with `IsDefaultEntryForModel == "Yes"`.

## 3. What the files told us already [verified 2026-11-08]

- **1d's mapping agrees with DepMap.** For all 57 pool lines, `Model.csv`'s `RRID` matches the
  Cellosaurus accession. The only disagreements were SR and UO-31, and they settled decision 9:

  | Line | IDs Cellosaurus lists | In `Model.csv` | Profiles | Chosen |
  |:--|:--|:--|:--|:--|
  | SR | ACH-000338, ACH-001665 | only ACH-000338 ("SR-786", Lymphoid) | rna, wes, wgs | **ACH-000338** |
  | UO-31 | ACH-000428, ACH-002203 | only ACH-000428 (Kidney) | rna, wes | **ACH-000428** |
  | HL-60(TB) | none | parent HL-60 = ACH-000002 (Myeloid) | rna, wes | **ACH-000002 as proxy** (decision 10) |

- **Lineages of the 57 pool lines:** Lung 8, Skin 8, Kidney 8, Bowel 7, CNS/Brain 6,
  Ovary/Fallopian Tube 6, Breast 5, Lymphoid 4, Prostate 2, Myeloid 2, Pleura 1.
- **Coverage gaps: not every pool line has every data type.** This matters for 1f.

  | Data type | Pool lines covered (of 57) | Missing |
  |:--|--:|:--|
  | Mutations | 55 | **SNB-19, NCI-H322M**: they have DepMap IDs but no sequencing profile at all |
  | Expression | 47 | COLO 205, HCC2998, MOLT-4, CCRF-CEM, M14, RXF 393L, SN12C, TK-10, SNB-19, NCI-H322M |
  | Copy number (`PortalOmicsCNGeneLog2`) | 42 | the 10 above plus HL-60 (proxy), UO-31, A-498, SK-MEL-28, HOP-92 |

  Most lines without RNA do have whole-exome data, so their mutations are covered.

## 4. CellMiner: NCI's own data for the NCI-60 lines

DepMap doesn't cover every pool line (§3), so we added **NCI CellMiner**
(`discover.nci.nih.gov/cellminer`, database version 2.16, files dated 2026-10-28). NCI profiled the
NCI-60 panel itself, so CellMiner covers **all 60 lines**, including SNB-19, NCI-H322M and
HL-60(TB). The download is scripted in `ingest_cellminer.py`: the zips are sha256-pinned, and each
workbook is converted to parquet in an isolated environment, because the files are legacy `.xls`.

| File | Size | sha256 (first 12) | What it is, intuitively | Units |
|:--|--:|:--|:--|:--|
| `nci60_RNA__RNA_seq_composite_expression.zip` | 12 MB | `47712a4d9149` | Gene "volume dials" from RNA-seq, 23,808 genes × 60 lines | log2(FPKM + 1) |
| `nci60_DNA__Exome_Seq_Protein_function_affecting.zip` | 2 MB | `72e863cdc02b` | Variants predicted to change protein function, 9,307 genes | summed variant-allele %; "-" = none called |
| `nci60_DNA__Exome_Seq_AA_changing.zip` | 3 MB | `49f06f4ec6c8` | Any amino-acid-changing variant, 12,945 genes | as above |
| `nci60_DNA__Combined_aCGH_gene_summary.zip` | 6 MB | `ca4a6e225d5e` | Photocopy count per gene, from 4 platforms combined, 23,400 genes | log2 ratio to normal; N = 2·2^value |

**Side finding:** TDC's undocumented `CellLine` feature vector has exactly 23,808 entries, the same
gene count as CellMiner's RNA-seq table. It was most likely built from it.

## 5. Validation and the source chosen for each data type [verified 2026-11-08]

| Check | Result |
|:--|:--|
| **Expression identity:** does each CellMiner line best-match *its own* DepMap profile? (Spearman, 2,000 most variable genes, 46 lines) | **45/46 yes**, median 0.904 (unrelated lines: median 0.58) |
| **SR** | **Fails.** CellMiner's SR RNA matches DepMap's own SR-786 at only 0.61, but **RPMI-8226** at 0.91. It shows no ALK or CD30, and myeloma markers (MZB1, JCHAIN). DepMap's SR-786 is a textbook ALK+ ALCL. Yet in the **ALMANAC screen**, SR and RPMI-8226 respond *unlike* each other (rank 1,241 of 1,624 line pairs), and CellMiner's SR **exome** doesn't match RPMI-8226 either. So the cells screened as SR are a distinct line, and **CellMiner's SR RNA-seq sample is most likely a mix-up** |
| **SNB-19 vs U-251 MG** (team note: STR and genomics show SNB-19 is a U-251 MG derivative) | **Confirmed:** in CellMiner, SNB-19's closest line is U-251 MG (0.83 on the 2,000-gene set; 0.90 on the 1f feature genes). The pair ranks 19th of 1,740 line pairs. In ALMANAC, their drug responses correlate at 0.75, among the highest of any pair. They're already one family (FAM_CVCL_0021) |
| **NCI-H322M vs NCI-H322** (does DepMap have it?) | DepMap has **NCI-H322 = ACH-000837** (RNA, WES, WGS). It has no data for NCI-H322M itself (ACH-002172 has no profiles). Cellosaurus: "not certain that NCI-H322 and NCI-H322M are identical". **Data:** CellMiner's NCI-H322M best-matches NCI-H322 among all 2,130 DepMap lines, at 0.918, as close as a line matching itself |
| **HL-60(TB) vs HL-60** (decision 10) | CellMiner HL-60(TB) vs DepMap HL-60: 0.872. Driver calls are identical (NRAS, CDKN2A) |
| **Driver mutations**, 24 genes × 50 lines in both sources | 93% of cells agree. **Oncogene hotspots ~98%** (KRAS, BRAF, NRAS). But **CellMiner misses tumour-suppressor mutations:** TP53 in 12 lines (including HT-29's R273H), STK11 in 6 of 6 (including A549's), and PTEN in 5 |
| **Mutation gene sets**, all genes | Weak identity signal: only 33/55 lines best-match themselves. The two exome pipelines call variants very differently, so don't mix whole mutation sets |
| **Copy number** | CellMiner vs DepMap, per-line Spearman over 17,017 genes, 42 lines: median 0.75 (min 0.25) |

**Sources used in 1f**, implemented in `build_features.py`:

| Data | Source | Lines |
|:--|:--|:--|
| Lineage | DepMap `Model.csv` | 57/57 |
| Expression | **CellMiner for all lines.** One source, one scale | 56 measured. **SR is imputed** (flag `expression_imputed = 1`; see §6). It was caught automatically by the identity check (not best match, and own Spearman < 0.80), not hard-coded |
| Driver mutations | **DepMap own profile** (54). For lines with none: the **Cellosaurus parent's DepMap profile** (3): SNB-19 → U-251 MG, NCI-H322M → NCI-H322, HL-60(TB) → HL-60 | 57/57. CellMiner's own calls are kept in `cell_mutations_cellminer.parquet` for comparison |
| Copy number | **CellMiner for all lines** | 57/57. Per-gene gaps are filled with 0 (2N) and flagged `cn_missing_<gene>` |

What the 3 parent-sourced lines look like:

| Line | Parent's DepMap driver calls (used) | CellMiner's own calls |
|:--|:--|:--|
| SNB-19 | TP53, PTEN, NF1 | none (consistent with CellMiner's low sensitivity for TSGs) |
| NCI-H322M | TP53 | TP53, KEAP1, NF1, BRCA1 (possibly inherited or passenger variants; unverified) |
| HL-60(TB) | NRAS, CDKN2A | NRAS, CDKN2A (identical) |

## 6. The SR expression problem, and how it was solved (decision 17)

### The problem, intuitively

Each cell line has a gene-expression profile: a reading of how strongly each of ~20,000 genes is
switched on, like a fingerprint of what the cell is doing. Our features take that fingerprint
from one place, CellMiner's RNA-seq table, so that every line is measured the same way.

For SR, **the fingerprint filed under "SR" belongs to a different cell.**
- SR is a lymphoma: an ALK-positive anaplastic large-cell lymphoma, which switches on ALK, CD30
  and CD25.
- The RNA-seq column labelled SR instead shows a myeloma: plasma-cell genes such as MZB1 and JCHAIN
  are on, and ALK is off.
- It matches RPMI-8226, a myeloma line that is also in our pool, almost as closely as a line
  matches itself.

Think of a lab where one tube's label was swapped: the measurement is fine, but it belongs to
another patient.

How we know the *label*, not the cells, is wrong:
- **The drug responses in the screen** (what we're predicting) say SR and RPMI-8226 behave
  differently. If the screened cells had been RPMI-8226, they'd respond alike.
- **SR's DNA** (exome) doesn't match RPMI-8226 either.
- **CellMiner's older microarray measurements of SR show the true lymphoma profile:** ALK z = +6.4,
  CD30 +6.3, CD25 +6.7, and plasma-cell genes near 0.

So only the one RNA-seq sample is wrong. If we used it, the model would be told "SR is a myeloma,
just like RPMI-8226": a confidently wrong feature for every SR experiment (4,900 rows).

### The fix: recover SR's own profile from its other measurements

We have two correct measurements of SR, but neither is on our scale (RNA-seq log2(FPKM + 1)):
- CellMiner's **microarrays**: the same lab, but a different technology, reported as z-scores
  across the 60 lines;
- DepMap's **RNA-seq of SR-786**: the same line, but a different lab and pipeline.

So we **learn a translation**. For each gene, fit a straight line from "microarray value" to
"RNA-seq value" on the 56 lines where both are trustworthy. Then apply it to SR's microarray values.
It's like learning an exchange rate from many past transactions, then converting one more amount.

**Does it work? Leave-one-out test.** Pretend each of the 45 verified lines is the unknown one,
predict it from the others, and compare with its real values:

| Method | Line-specific signal recovered (r)* | Error (RMSE, log2) |
|:--|--:|--:|
| Know nothing (use the average line) | 0 | 1.70 |
| DepMap SR-786 as-is, without translating | 0.79 | 1.95 |
| DepMap, translated | 0.86 | 0.88 |
| **CellMiner microarrays, translated** | **0.95** | **0.56** |
| Average of both | 0.94 | 0.62 |

\* *How well the prediction captures what makes this line different from the average line.* That's
the part that matters for a model, because the average is the same for everyone.

**What was built** (`build_features.impute_expression`), applied automatically to any line that
fails the identity check. For each gene, in order:
1. CellMiner microarray, translated (20,654 genes for SR);
2. DepMap, translated (1,074 genes);
3. the gene's average (2,080 genes).

SR is flagged `expression_imputed = 1`. The leave-one-out quality of the build is in
`features_report.json`: deviation r = 0.94 and RMSE 0.61 on the 923 feature genes the arrays
cover, against 1.71 with no information. The two independent estimates agree with each other at
r = 0.79.

**Result for SR, on our scale:**

| Gene | SR, imputed | RPMI-8226 | K-562 |
|:--|--:|--:|--:|
| CD30 (TNFRSF8) | 6.67 | 1.11 | 2.65 |
| CD25 (IL2RA) | 4.16 | 0.34 | 0.00 |
| PTPRC (CD45) | 4.98 | 0.33 | 2.07 |
| ALK | 2.78 | 0.34 | 0.05 |
| MZB1 (myeloma marker) | 0.63 | 8.11 | 0.00 |
| JCHAIN (myeloma marker) | 0.09 | 8.56 | 0.00 |

**Limitation:** a gene that only SR expresses is under-estimated. ALK comes out at 2.8, while
DepMap measures 6.2. The translation for ALK is learned from lines where ALK is essentially zero,
so it can't extrapolate up to SR's level. The direction is right; the size is compressed.

**Choices made:**
- Feature genes are still chosen on the 56 *measured* lines only, so an imputed profile never
  decides which genes become features.
- The new input is `nci60_RNA__5_Platform_Gene_Transcript_Average_z_scores.zip` (15 MB, sha256
  `dd5462ed7676…`). It's pinned in `ingest_cellminer.py`.
- Also downloaded for exploration, but not used: `nci60_RNA__Affy_HG_U133_Plus_2.0_RMA.zip`
  (30 MB, `5d63d7250832…`). It's probe-level, with one extra metadata column.

## 7. Still open

- **Disk:** 3.4 GB free.
  - The two ~640 MB DepMap CSVs are kept for now (team, 2026-11-08). 1f only needs a few columns of
    each, so they could be deleted later; their sha256 values are recorded here.
  - The CellMiner folder (156 MB) includes HTML copies extracted by hand during exploration. The
    scripted `extract` takes only the workbooks, so those HTML copies can be removed.
- **CellMiner also serves `DTP_NCI60_ALMANAC_COMBO_SCORE.zip`,** NCI's own ALMANAC release. It's
  useful later for checking our ALMANAC block against the original source.
