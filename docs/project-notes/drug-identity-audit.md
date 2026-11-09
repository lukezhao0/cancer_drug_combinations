# Drug identity audit: one molecule, several names (TDC `drugcomb`, NCI-60)

Audit date: 2026-11-08. Data: `data_cache/interim/tdc_drugcomb.parquet` (converted from
`drugcomb_nci60.pkl`, sha256 `c0e195c9…ad00bf`, Dataverse file 4215720, MD5 matches Dataverse).
Every number here is **[verified]** by [`scripts/audit_drug_identity.py`](../../scripts/audit_drug_identity.py)
unless it is labelled otherwise. The script reads only the parquet.

Run it from the code repo root:

```bash
uv run python scripts/audit_drug_identity.py
```

> **Implemented 2026-11-08 (step 1c).** See
> [m1-implementation.md](m1-implementation.md#step-1c-drug-harmonization-done-2026-11-08).
> - P1, P2, P4 and P5 are done as proposed.
> - P3 turned out simpler. The Pt drugs resolve automatically by name and are verified by
>   heavy-atom formula; no hand-mapping was needed.
> - Corrections to this audit:
>   - the key compared with ChEMBL must **not** use the canonical tautomer (§2's method), or 19 drugs
>     find no ChEMBL match;
>   - all 16 likely identities in §2.1 are confirmed, but 5 nucleoside labels needed a manual
>     stereo fix.
> - P6–P9 belong to step 1e and are still open. P7 and P8 still need decisions.

---

## 1. Summary

- The 129 drug IDs are **122 distinct molecules**. Exactly **7 molecules appear under two names**.
- **All 7 duplicates come from the file mixing two studies.** O'Neil/Merck 2016 used generic names
  (9 cell lines). NCI-ALMANAC named the physical sample it tested: salt form, brand, or NSC
  catalogue number (59 lines).
- Merging the names makes **546 triplets collide**: the same drug pair on the same cell line,
  measured once by each study. The pairs drop from 5,427 to 5,365.
- **On those 546 triplets the two studies barely agree.** CSS Pearson is 0.22, and O'Neil's mean CSS
  is 56.4 against ALMANAC's 29.4. Synergy scores agree even less (Bliss Pearson 0.11). These are not
  replicates and must not be averaged.
- The planned structure-based merge has two technical problems:
  - "keep the largest fragment" destroys the platinum drugs;
  - none of the 129 SMILES carry stereochemistry, so InChIKeys computed from them won't match
    ChEMBL's for chiral drugs.

This corrects two earlier statements:
- [gap analysis §2.4](target-design-gap-analysis.md#24-tdc-drugcomb-is-really-nci-almanac-verified):
  the file is not pure NCI-ALMANAC.
- [m1-walkthrough §3](m1-walkthrough.md#3-step-1b-look-at-the-data-before-designing-anything-done):
  the salt-pair list missed doxorubicin and oxaliplatin.

---

## 2. Every instance of one molecule under two names

Method: RDKit `Cleanup` → `LargestFragmentChooser` → `Uncharger` → canonical tautomer → InChIKey,
then group the IDs that share a key. Each ID has exactly one SMILES in the file. Grouping by the
14-character connectivity block (which ignores stereo) gives the same 122 molecules, so nothing is
hidden by stereo or isotope differences.

"Rows" counts rows where the name appears as either drug.

| # | Molecule | O'Neil name | Rows / lines / partners | NCI-ALMANAC name | Rows / lines / partners | How the SMILES differ | Colliding triplets |
|:-:|:--|:--|:--|:--|:--|:--|--:|
| 1 | Erlotinib | `ERLOTINIB` | 324 / 9 / 36 | `ERLOTINIB HYDROCHLORIDE` | 5,762 / 59 / 99 | ALMANAC adds `.Cl` (HCl salt) | 180 |
| 2 | Topotecan | `TOPOTECAN` | 189 / 9 / 21 | `TOPOTECAN HYDROCHLORIDE` | 5,772 / 59 / 99 | ALMANAC adds `.Cl` | 71 |
| 3 | Vinblastine | `VINBLASTINE` | 189 / 9 / 21 | `VINBLASTINE SULFATE` | 6,060 / 59 / 99 | ALMANAC adds `.OS(=O)(=O)O` (sulfate) | 67 |
| 4 | Vinorelbine | `VINORELBINE` | 189 / 9 / 21 | `NAVELBINE DITARTRATE (TN)` | 5,884 / 59 / 99 | brand name; ALMANAC adds a tartrate fragment | 72 |
| 5 | Etoposide | `ETOPOSIDE` | 189 / 9 / 21 | `NSC141540` | 5,948 / 59 / 99 | **identical SMILES**; NSC catalogue number only | 71 |
| 6 | Doxorubicin | `DOXORUBICIN` | 189 / 9 / 21 | `ADM HYDROCHLORIDE` | 8,765 / 59 / 99 | "ADM" = Adriamycin (old brand); ALMANAC adds `.Cl` | 72 |
| 7 | Oxaliplatin | `OXALIPLATIN` | 189 / 9 / 21 | `ELOXATIN (TN) (SANOFI SYNTHELAB)` | 5,894 / 59 / 99 | brand plus supplier; the amine charges are written differently (`[NH-]` vs `N`), and neither form is charge-balanced | 67 |

- **Total colliding triplets: 546.** The column above sums to 600 because some triplets pair two of
  these drugs, and those are counted once per drug.
- **The 9 O'Neil lines:** HCT116, HT29, NCI-H460, NCIH23, OVCAR3, SK-OV-3, SW-620, T-47D, UACC62.
  These are the lines where O'Neil's 39-line panel overlaps NCI-60.

### 2.1 Names that are *not* duplicates

These need no merge.

**Shared names (14 drugs).** The same name is used in both studies, so DrugComb did reconcile these:
5-FU, BORTEZOMIB, CARBOPLATIN, CYCLOPHOSPHAMIDE, DASATINIB, LAPATINIB, METHOTREXATE, MITOMYCINE,
PACLITAXEL, SN-38, SORAFENIB, SUNITINIB, TEMOZOLOMIDE, ZOLINZA. DrugComb's harmonization was
**partial, not absent**.

**Obscure ALMANAC labels.** These are unique molecules under non-standard names. The likely identity
comes from reading the structure. It is **[unverified]** until each one is confirmed in ChEMBL:

| Label in file | Likely drug |
|:--|:--|
| `CHEMBL17639` | cladribine |
| `CHEMBL277800` | clofarabine |
| `34793-34-5` | structure looks like nelarabine (CAS label not checked) |
| `ANTIBIOTIC AY 22989` | sirolimus |
| `NSC733504` | everolimus |
| `NSC-127716` | decitabine |
| `NSC 85998` | streptozocin |
| `NSC256439` | idarubicin |
| `NSC707389` | eribulin |
| `ANTIBIOTIC AD 32` | valrubicin |
| `CO-V` | pentostatin |
| `EMCYT (PHARMACIA)` | estramustine phosphate sodium |
| `1-(5-DEOXYPENTOFURANOSYL)-5-FLUORO-4-{[(PENTYLOXY)CARBONYL]AMINO}PYRIMIDIN-2(1H)-ONE` | capecitabine |
| `CISPLATINO` | cisplatin |
| `MITOMYCINE` | mitomycin C |
| `ZOLINZA` | vorinostat |

### 2.2 A third, tiny study

**Corrected 2026-11-08: 10 rows**, all on SN12C, pair `NCGC00090797-11`, `LLL-12`, `BARDOXOLONE METHYL`, `ABT-263 (NAVITOCLAX)`
and `CARFILZOMIB (PR-171)` with each other. The source is unidentified.

---

## 3. The two studies disagree on the same experiment

These are the 546 paired triplets, on 9 lines. Each study's value is the mean of its rows for that
triplet.

| Score | Pearson | Spearman | Mean, ALMANAC | Mean, O'Neil | Mean abs. difference |
|:--|--:|--:|--:|--:|--:|
| CSS (killing) | 0.22 | 0.22 | 29.4 | **56.4** | 30.3 |
| Bliss | 0.11 | 0.14 | 0.5 | −0.3 | 6.2 |
| Loewe | 0.11 | 0.16 | −9.9 | −7.0 | 13.8 |
| ZIP | 0.16 | 0.19 | −1.0 | 1.1 | 5.6 |
| HSA | 0.07 | 0.11 | 0.0 | 2.8 | 7.0 |

**Hit labels flip between the studies**, using the M1 p95 thresholds:

| Outcome | Hits, ALMANAC | Hits, O'Neil | Hits in both | Studies disagree |
|:--|--:|--:|--:|--:|
| Bliss ≥ 9.26 | 16 | 35 | 2 | 47 |
| CSS ≥ 63.3 | 27 | 237 | 15 | 234 |

For reference, the 9,830 same-name triplets stored in both drug orders have CSS Pearson 0.55 and
Bliss Pearson 0.21.
- Whether those are within-study replicates or also partly cross-study is **not yet known**. See
  open question 3 in §6.

**Why the studies differ** (a literature explanation, not tested here): the screens used different
assays, dose grids and maximum concentrations. Synergy models are known to disagree when tested
concentrations and maximum responses differ, which was shown on the Merck (O'Neil) screen itself:
Vlot et al. 2019, *Drug Discovery Today*. A 2026 DrugComb v1.5 benchmark found prediction falls to
the random floor on unseen studies: Pan, *BMC Bioinformatics*. Both fit a large study effect.

---

## 4. Problems with the planned structure-based merge

### 4.1 "Largest fragment" destroys the platinum drugs

DrugComb writes metal complexes as disconnected ions, so salt-stripping keeps the wrong piece:

| Drug | SMILES in file | Largest fragment, uncharged |
|:--|:--|:--|
| CISPLATINO | `N.N.Cl[Pt+2]Cl` | `N` (ammonia) |
| CARBOPLATIN | `C1CC(C1)(C(=O)O)C(=O)O.[NH2-].[NH2-].[Pt+2]` | cyclobutane-1,1-dicarboxylic acid (the leaving-group ligand) |
| OXALIPLATIN / ELOXATIN | `…[Pt+4]` + diaminocyclohexane + oxalate | `NC1CCCCC1N` (the carrier ligand) |

- Oxaliplatin and Eloxatin still merged correctly, but only by accident: both reduce to the same
  wrong fragment.
- Every other multi-fragment SMILES (19 of them, all HCl, sulfate, tartrate, citrate or sodium salts)
  is stripped correctly. The script prints all 21 in section 4 of its output.

### 4.2 No stereochemistry in any SMILES

- 0 of 129 SMILES contain `@`, `/` or `\`.
- So an InChIKey computed from them is the key of the **flat** structure. It won't equal ChEMBL's
  `standard_inchi_key` for any chiral drug, which is most of the pool.
- Consequences:
  - The walkthrough's plan to look up the parent InChIKey in ChEMBL would mostly return nothing.
  - Stereoisomers can't be told apart. For example, epirubicin would collide with doxorubicin. None
    are in the current 129, but the merge must guard against it.
  - Morgan fingerprints built from these SMILES carry no stereo. That's acceptable for fingerprints,
    but should be stated in the data card.

---

## 5. Proposed approach, and how it differs from the current plan

**Status: proposal.** Items marked **needs decision** change an earlier team decision and must be
confirmed before code is written.

| # | Topic | Current plan (walkthrough §4 and §6; m1-implementation decisions) | Proposed | Why |
|:-:|:--|:--|:--|:--|
| P1 | What defines "the same drug" | RDKit parent InChIKey | **ChEMBL parent molecule ID** (`molecule_hierarchy.parent_chembl_id`) is the merge key. The RDKit 14-character connectivity key is only a cross-check | ChEMBL already resolves salts, brands and NSC numbers to one parent; the file's flat SMILES can't (§4.2) |
| P2 | How to find the ChEMBL ID | Look up the computed InChIKey | Resolve each of the 129 names in order: (a) ChEMBL synonym search, (b) PubChem name → CID → ChEMBL, (c) by hand. Accept a match only if its connectivity key (first 14 characters of the InChIKey) equals the one computed from the file's SMILES. Otherwise flag it | Full InChIKeys won't match (§4.2). Names alone are unreliable (`CISPLATINO`, CAS numbers, `ADM`). Requiring name and structure to agree catches mistakes in both |
| P3 | Salt stripping | Largest fragment for every SMILES | Keep largest-fragment stripping **only as the cross-check**. **Skip it for metal-containing SMILES** (Pt), and hand-map cisplatin, carboplatin and oxaliplatin | §4.1 |
| P4 | Which structure feeds features | The file's SMILES after stripping | **ChEMBL's canonical parent SMILES**, which includes stereo | One structure per drug, from the same source as the targets |
| P5 | Mapping table | Saved for audit | Also **commit a hand-checked `drug_synonyms.csv`** to the code repo: original name → ChEMBL parent ID → preferred name → how it was resolved | Makes all 129 decisions reviewable. Required by the gap analysis §2.5 recommendation |
| P6 | Repeats created by the merge | Average into mean, SD, `n_reps` (walkthrough §6.2) | **Never average across studies.** Keep one row per (triplet, study) | §3: CSS r = 0.22 with a 27-point offset. Averaging would invent values neither study measured |
| P7 | `study_id` | One constant per dataset (`NCI-ALMANAC`); decision 3 in m1-implementation | **Per-row `study_id`** (`NCI-ALMANAC`, `ONEIL-2016`, `UNKNOWN-1`). **Needs decision** | The file holds at least 3 studies. Leakage control and the paper blocklist are per study |
| P8 | O'Neil rows inside the NCI-60 table | Kept, labelled as ALMANAC | **Remove them from the NCI-60 pool.** O'Neil stays the second benchmark (via OncoPolyPharmacology) as the design intends. **Needs decision** | Removes the overlap between the "two independent benchmarks". Fits decision 4 (NCI-60 only). Avoids mixing incomparable CSS scales in one pool |
| P9 | Same-name two-order duplicates (9,830) | "Real repeat measurements, so average them" (walkthrough §3) | **Hold** until their study of origin is known (open question 3). If both copies are ALMANAC, average them; if not, apply P6 | Bliss r = 0.21 between the copies is low for replicates, and some may be cross-study |
| P10 | Data card | Lists merges | Also report: the 7 groups, the 546 cross-study collisions with the agreement table from §3, the rows removed by P8, and "no stereo in source SMILES" | Reproducibility and honest reporting |

---

## 6. Open questions

1. **Identifying all O'Neil rows.**
   - Rows involving O'Neil-only names: at least 4,294.
   - But O'Neil also tested pairs of the 14 shared-name drugs on the same 9 lines. Those rows can't
     be told apart from ALMANAC rows by name.
   - Options:
     - (a) DrugComb's summary table, which has a `study_name` column. The portal was unreachable on
       2026-11-08 (proxy 502).
     - (b) Match against TDC's `oncopolypharmacology` file (1.6 GB) on (pair, line).
     - (c) Treat every row on the 9 lines whose pair involves only O'Neil drugs as possibly O'Neil.
2. **The third study (§2.2):** identify it, or just drop its 4 rows.
3. **Origin of the 9,830 same-name two-order duplicates.** This is the same data needed for question 1.
4. **Does the current DrugComb portal still have these duplicates?** The TDC file dates from
   2020-12-12, before the 2021 DrugComb curation update. **[unverified]**

---

## 7. Why not just use SMILES or standard names?

**SMILES alone doesn't work here:**
- **The SMILES aren't standardized.** The same molecule is written with and without its salt
  (`…C#C)OCCOC` vs `…C#C)OCCOC.Cl`). So raw-SMILES equality still splits 6 of the 7 duplicate groups.
  Only etoposide/NSC141540 have identical strings.
- **Standardizing SMILES is the very step that needs care.** The obvious salt-stripping breaks the
  platinum drugs (§4.1).
- **SMILES carry no stereo in this file** (§4.2). So they can't be used to look up targets, and
  they can't tell stereoisomers apart.
- **SMILES can't answer "which study measured this?"** The duplicates are a *study* problem as much
  as a *name* problem (§3). A SMILES-based merge would silently combine incomparable measurements,
  which is worse than leaving the names apart.

**"Standard names" don't exist in the file.** Producing them (ChEMBL or PubChem preferred names)
*is* this harmonization step.
- ALMANAC's labels are brand names, NSC numbers, CAS numbers and foreign spellings, and none of them
  map to a standard name automatically.
- Name lookup alone also fails on `CISPLATINO`, `ADM HYDROCHLORIDE` and `34793-34-5`.

**So the proposal is the middle path:** name and structure must agree (P2), and the merge key is
ChEMBL's parent ID (P1). This adds a little work: 129 names, with about 25 non-standard ones checked
by hand. What it buys:
- correct targets for the agent's mechanism reasoning;
- no duplicate experiments in the pool;
- no leakage across held-out lines through a renamed drug;
- no fake replicates from averaging two studies.
