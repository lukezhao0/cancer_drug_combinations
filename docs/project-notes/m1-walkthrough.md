# M1 walkthrough: building the data pool, step by step, and why

This explains milestone **M1 (data and pool)** from `target_design.md` §4. It covers what each step
does, the intuition behind it, and *why* each decision was made.

- **Status (2026-11-08, end of day):** **M1 is complete.** All steps 1a–1g are built, tested and
  frozen. The results summary, with figures, is in
  [m1-freeze-and-splits.md](m1-freeze-and-splits.md). §6–§8 below keep the original plans, with a
  "what was built" note at the top of each.
- **Where we are and what's next:** [§11](#11-where-we-are-and-whats-next).
- **Concept primer:** [concepts-primer.md](concepts-primer.md) covers synergy, the oracle, and
  active learning.
- **Raw facts and status table:** [m1-implementation.md](m1-implementation.md).

---

## 0. Why M1 comes first

Everything later (the oracle, the response model, the baselines, the agent, the evaluation) reads
**one frozen table of experiments**. If that table is wrong, every later number is wrong, and
nobody notices, because the code still runs.

```
   M1 data ──► M2 oracle ──► M3 model ──► M4 baselines ──► M6 agent ──► M8 results
   (if this is wrong, everything to the right is quietly wrong)
```

So M1 is about being **correct, reproducible, and leak-proof**, not fast. That's why there are so
many checks.

## 1. The whole journey in one picture

```
 Harvard Dataverse                 data_cache/raw/            data_cache/interim/
 ┌──────────────────┐  1a download ┌──────────────────┐ 1a    ┌───────────────────┐
 │ drugcomb_nci60   │ ───────────► │ .pkl (verified)  │ ────► │ tdc_drugcomb      │
 │ .pkl (36 MB)     │  + checksum  │ + manifest.json  │convert│ .parquet (12 MB)  │
 └──────────────────┘              └──────────────────┘       └─────────┬─────────┘
                                                                        │
          1c drugs: salts → one parent molecule → ChEMBL id + targets   │
          1d cell lines: names → Cellosaurus id → DepMap id + family    │
          1e triplets: order A<B, merge repeats, set self-pairs aside   ▼
          1f features: fingerprints, targets, mutations, expression
          1g freeze: held-out lines, hit thresholds, write files
                                                                        │
                                                    data_cache/processed/▼
                          pool.parquet  labels.parquet  drugs.parquet  cells.parquet
                          splits.json   data_card.md    (+ manifest of hashes)
```

✅ = done, ⬜ = planned: 1a ✅ · 1b ✅ (profiling) · 1c ✅ · 1d ✅ (built; 3 decisions open) · 1e ⬜ · 1f ⬜ · 1g ⬜

(1c's "salts → one parent molecule" became "name and structure must agree on one ChEMBL parent";
see §4.)

---

## 2. Step 1a: get the raw data in, safely and exactly (done)

### 2.1 What the data is

TDC's "DrugComb" dataset is the **NCI-ALMANAC** screen. The US National Cancer Institute tested
pairs of approved cancer drugs on the NCI-60 cell-line panel.

| Item | Value |
|:--|:--|
| Size | 297,098 rows; 129 drug names; 59 cell lines |
| Each row | (drug 1, drug 2, cell line) → CSS (how well it kills) and four synergy scores (ZIP, Bliss, Loewe, HSA) |
| Drug structures | Given as SMILES text strings |

**Intuition: SMILES.** A SMILES string is a way to write a molecule as text.
`CC(=O)OC1=CC=CC=C1C(=O)O` is aspirin. Software (RDKit) can turn it back into a molecule and compare
structures.

### 2.2 Why we *don't* use the official `tdc` package

This was the obvious tool, and the first thing that failed.

| Reason | Evidence |
|:--|:--|
| **It can't be installed with our packages** | PyTDC 1.1.15 requires `pandas<3`, `numpy<2`, `rdkit<2024.3.1`. Our project requires `pandas>=3.1`, and we have numpy 2.5.4 and rdkit 2026.9.1. No version set satisfies both. |
| **It fails to install on Python 3.13 at all** | It needs numpy 1.x. The last numpy 1.x (1.26.4) has prebuilt packages only up to Python 3.12, so pip tries to compile numpy from C source and fails (the `metadata-generation-failed` error you saw). |
| **It's heavy** | It pulls in `transformers`, `accelerate`, `datasets`, `cellxgene-census`, and `tiledbsoma` to fetch one 36 MB file, on a disk with 6.7 GB free. |
| **For this dataset, it only downloads and opens a file** | Reading its source: `DrugSyn(name='DrugComb')` downloads one pickle from Dataverse (file id 4215720) and loads it. `get_data()` returns it **unchanged**. Its extra `get_split()` isn't needed, because our design makes its own splits. |

**So we do exactly what it does**, plus safety checks it doesn't have. We end up with the same
table, confirmed by the file's checksum.

### 2.3 Download: the four safeguards, and the bug behind each

```
  request ──► [1 named User-Agent] ──► bytes ──► [2 size check] ──► [3 checksum] ──► keep + [4 manifest]
                                                   │ short? retry ×3      │ wrong? reject, keep nothing
```

1. **A named User-Agent.** Every web request says who's asking. Dataverse answers **403
   Forbidden** to Python's default (`Python-urllib/3.13`), but accepts a named client. This was
   tested directly, not assumed.
2. **Size check, with retries.** One download arrived **21,614 bytes short** (36,372,455 of
   36,394,069) *with no error raised*: the connection dropped and Python's read loop just stopped.
   - *Intuition:* a parcel that arrives with a page missing, and nobody notices.
   - We now compare the bytes received against the size the server declared, and retry up to 3
     times.
3. **Checksum (sha256).** A sha256 is a 64-character fingerprint of a file's exact contents. Change
   one byte and the fingerprint changes completely.
   - We pinned the fingerprint of the file we profiled (`c0e195c9…`).
   - Any download that doesn't match is rejected, so we can never silently analyze a different
     file.
   - This check is what first caught the truncated download.
4. **Manifest.** `raw/manifest.json` records the URL, size, checksum, and download date. That's the
   design's "save raw copies with download date and file hash" requirement.

### 2.4 Conversion: pickle → parquet

**What a pickle is.** It's Python's way of saving an object to disk. Loading one *re-runs
instructions stored in the file*, so a malicious pickle can run any code on your machine. Opening an
unknown pickle is like plugging in a USB stick you found on the street.

**What we do about it: a restricted unpickler.** This is the approach Python's own documentation
recommends.
- Before loading, the file's instructions were scanned **without running them**. It only asks for
  9 pandas/numpy building blocks.
- The loader allows **only those 9**. Anything else (for example, "run a shell command") is refused
  *before it can run*.
- A test proves this: it builds a malicious pickle and checks that it's rejected.

**Why a separate, throwaway Python environment.** The pickle was written by old pandas (1.x/2.x)
with numpy 1.x, so our pandas 3 / numpy 2 can't be trusted to read it. The conversion therefore runs
once in an isolated environment, then saves **parquet**. Parquet is a plain columnar data format
that any pandas can read, and loading it runs no code.

- That environment is pinned **exactly** (`pandas==2.3.3, numpy==1.26.4, pyarrow==25.0.1`). The
  first try failed because the newest pyarrow (26) refuses to work with numpy 1.x. Exact pins fixed
  that and make the conversion repeatable.
- **Verified:** two independent conversions (on Python 3.10 and 3.12) produced **byte-identical**
  parquet files (sha256 `7949cbf8…`). The design demands this kind of reproducibility.

**Why drop the `CellLine` column.** It holds TDC's own per-line feature arrays with no
documentation of what they are. The design takes cell features from DepMap, a documented source we
can cite. Using unexplained features would make results hard to interpret or defend.

### 2.5 Load check

`load()` refuses the parquet unless it has exactly 297,098 rows and the 10 expected columns. If an
upstream file or our converter ever changes, we find out immediately, not three milestones later.

---

## 3. Step 1b: look at the data before designing anything (done)

We profiled the real table first, so the design would rest on facts instead of guesses. What it
told us, and what each finding changed:

| Finding | Number | What it changed |
|:--|:--|:--|
| Each drug name maps to one structure | 129 names; only 2 share a structure (ETOPOSIDE, NSC141540) | Structure can be the identity key in 1c |
| Salt forms appear as separate drugs | e.g. ERLOTINIB / ERLOTINIB HYDROCHLORIDE | They need merging (§4) |
| Drug paired with itself | 115 rows | Needs a decision (§6) |
| Repeated triplets | 9,830, **all** the same pair in the two orders (A,B)/(B,A), with different scores | They're real repeat measurements, so average them. **On hold (2026-11-08):** their study of origin is unknown, and Bliss r is only 0.21 between copies. See [audit](drug-identity-audit.md) P9 |
| Pairs are measured almost everywhere | median 58 of 59 lines per pair | Held-out-line testing (design §9) is feasible |
| Hit thresholds (top 5%) | Bliss ≥ 9.26, CSS ≥ 63.3 | Concrete values for the data card |
| The two outcomes barely agree | Spearman 0.06; only 2,143 triplets are hits for both | Confirms the design's choice to track synergy and killing **separately** |
| Hits are spread out | Synergy hits span 3,316 pairs; the top 10 pairs hold only 531 | A selector can't "win" just by sweeping one famous pair across lines |
| Random baseline | ≈ 16 synergy hits in 320 experiments | The floor every method must beat |

---

## 4. Step 1c: one drug, one identity (done 2026-11-08)

> **Status:** done and tested . This section explains the ideas.
> The exact files, checks and numbers are in
> [m1-implementation.md](m1-implementation.md#step-1c-drug-harmonization-done-2026-11-08), and the
> evidence that reshaped the plan is in [drug-identity-audit.md](drug-identity-audit.md).

### 4.1 The problem: one medicine, several labels

The source file names each drug with whatever label its study used. The same medicine can show up
under a generic name, a salt name, a brand name or a catalogue number:

```
 "ERLOTINIB"                  ┐
 "ERLOTINIB HYDROCHLORIDE"    ┘──► same active molecule (+ a chloride salt that helps it dissolve)
 "VINORELBINE" / "NAVELBINE DITARTRATE (TN)"   ──► generic name vs brand name + salt
 "ETOPOSIDE"   / "NSC141540"                   ──► generic name vs NCI catalogue number
 "CO-V"                                        ──► an old code name for pentostatin
```

If such labels stay separate, the model sees one drug as two. A selector could also "discover" a
combination that is really one it has already tested under another name.

The goal of 1c is a lookup table that sends every label to **one stable drug ID**. That ID must
come with:
- a clean structure, so it can be turned into features in 1f;
- the drug's known **targets** (erlotinib → EGFR), which the agent reasons about.

### 4.2 The core idea: two witnesses must agree

There are two independent clues about what a label means:

| Witness | What it tells us | How it can mislead |
|:--|:--|:--|
| **The name** | What the study *called* the drug | Brand names, typos, catalogue codes; and names can be matched to the wrong record |
| **The structure** (the SMILES in the file) | What molecule was *drawn* for it | Drawn flat (no 3-D handedness), often with a salt attached |

Neither clue is trustworthy alone. So a label is accepted only when **both clues point to the same
ChEMBL drug.** The name proposes a candidate, and the structure confirms it. That is like checking
a person's name *and* their photo ID: either one can be faked or mistaken, but a mismatch between
them is a red flag.

The common ID is the **ChEMBL parent ID**. ChEMBL is the public drug database, and it groups every
salt form of a drug under one "parent" record (erlotinib hydrochloride → parent erlotinib,
`CHEMBL553`). Using the parent as the merge key means "same active drug" by construction, and the
parent is also where ChEMBL keeps the targets.

```
 label "ERLOTINIB HYDROCHLORIDE"
    │
    ├─ name route ────► ChEMBL records named that ──► their parent: CHEMBL553 ─┐
    │                                                                         ├─ agree? ✔ accept
    └─ structure route ─► strip salt ─► skeleton key ─► ChEMBL parent's key ──┘
                                                         (disagree → flag for a person)
```

### 4.3 Comparing structures: the "skeleton" key

To compare the file's structure with ChEMBL's, both are reduced to a **connectivity key**. That's
the first part of the InChIKey (a standard molecular barcode), and it describes only *which atoms
are bonded to which*. It ignores:
- handedness (stereo), which the source file never records;
- charges and attached salts, once the largest piece of the molecule has been kept.

*Intuition:* it compares the skeleton of the molecule and ignores the pose.

Three things about this key turned out to matter, and each changed the original plan:

1. **Platinum drugs break "keep the largest piece".** Cisplatin is drawn as disconnected ions, so
   "keep the largest fragment" would reduce it to ammonia. So metal drugs aren't salt-stripped.
   They are checked by **atom counts** (heavy-atom formula) against ChEMBL instead, because ChEMBL
   stores no structure for them. All 4 Pt drugs pass this way, so no hand-mapping was needed.
2. **Don't "tidy" the molecule's hydrogens first.** The original plan normalized tautomers, which
   are the same molecule drawn with a hydrogen in a different spot. RDKit's tidy form differs from
   ChEMBL's for 19 drugs, including doxorubicin, paclitaxel and docetaxel, so those matched
   nothing. The key now uses the molecule as drawn, and the tautomer form is only a second-chance
   comparison.
3. **A skeleton can't tell mirror images apart.** See 4.4.

### 4.4 The trap that the checks caught: mirror-image twins

Many drugs are **stereoisomers**: molecules with identical skeletons whose atoms point in different
directions in 3-D, like a left and a right glove. Biology can tell them apart, and often only one
of them is the medicine. The source file draws every drug without this information (0 of 129
SMILES carry stereo).

So for 5 nucleoside drugs, the structure route found a ChEMBL record that really does share the
skeleton, but it was an **unnamed twin with no known targets**, not the approved drug. The most
striking case: DrugComb's own label `CHEMBL17639` isn't cladribine (CHEMBL1619). It's a stereoisomer
of it.

The fix had two parts:
- **A check that refuses this case.** If an automatic match lands on a lesser-known twin of an
  approved, named drug, the build fails and asks a person.
- **Manual rows for those 5 labels**, pointing at the approved drug, each with the evidence in a
  note. This assumes NCI-ALMANAC screened the approved drugs, which is reasonable but **not
  re-checked against the ALMANAC drug list.**

Once each label has its ChEMBL parent, the features use **ChEMBL's** structure, which does carry
stereo (67 of 117 parents; the rest are achiral or racemic). The 4 Pt drugs keep the source
structure, because ChEMBL has none.

### 4.5 People stay in the loop: a reviewable answer sheet

The automatic resolver doesn't feed the pipeline directly. It **proposes** a table,
`curated/drug_synonyms.csv`, with one row per label giving its ChEMBL parent, how it was resolved,
and how the structure was checked. That file is committed and reviewed like code.

- Rows a person decided are marked `manual` and must carry a note. A re-run of the resolver never
  overwrites them.
- `build` reads **only this file**, then re-checks every row against ChEMBL before writing
  anything.

*Why this split:* names in public databases change, and a resolver can drift silently. A committed
answer sheet makes every merge visible in a diff, and the re-check catches drift. One of the checks
compares the committed file with what the resolver would say today.

Current answer sheet: 120 rows confirmed automatically by name + structure, 8 manual, and 1
unresolved (`MK-4541`, which ChEMBL_37 doesn't have).

### 4.6 Tripwires: the build refuses to lie

The build runs 13 checks. If any failing check trips, **no output is written**. In plain words, they
guarantee:

- every label is mapped exactly once, and none is lost or duplicated on the way out;
- each ID is a parent, not a salt record;
- name and structure agree for every row;
- the ChEMBL grouping matches an independent grouping by structure alone (two separate methods
  agreeing on which labels are the same drug);
- no automatic row sits on a mirror-image twin (4.4);
- the committed answer sheet still matches today's resolver;
- the structures handed to 1f are single, salt-free molecules.

Every check has a unit test that makes it fail on purpose, so we know each tripwire actually trips.
The real-data tests confirm known facts: the 7 groups found by the audit, all 16 identities it
guessed for obscure labels, and textbook targets (erlotinib → EGFR, lapatinib → ERBB2,
methotrexate → DHFR, bortezomib → PSMB5, imatinib → ABL1, …).

*Why so strict:* this table feeds every later milestone. A wrong merge doesn't crash anything; it
quietly changes the answers (§0).

### 4.7 Reproducibility: replay, don't re-ask

Every ChEMBL and PubChem answer is saved on disk, about 1,350 small files. "Not found" answers are
saved too. A build can then be **replayed offline** from those saved answers, and two offline
builds give byte-identical outputs. This also pins the result to one ChEMBL release (ChEMBL_37,
2026-05-01), even after ChEMBL updates.

The servers were flaky while we built this: occasional errors, and large answers cut off about half
the time. So the client retries patiently, with growing waits, and saves only complete answers.

### 4.8 What came out

```
 129 labels ──► 122 drugs  (121 ChEMBL parents + MK-4541, which ChEMBL lacks)
                 │
                 ├─ 7 merges, exactly the audit's 7 pairs, e.g. ERLOTINIB + ERLOTINIB HYDROCHLORIDE
                 └─ 115 drugs with ≥1 target   (243 drug→target rows; median 1 target per drug)
```

Two drugs get their targets **borrowed**, with the borrowing recorded in the target rows:
- **Fludarabine** borrows from its prodrug, fludarabine phosphate. A prodrug is a chemically
  different molecule the body converts into the drug, so ChEMBL files it separately and keeps the
  targets there.
- **Plicamycin**: the structure in the file differs slightly from ChEMBL's approved plicamycin
  record, so it borrows that record's targets.

One surprise matters for the agent later. ChEMBL lists sirolimus and everolimus as binding
**FKBP1A** (the helper protein they grab first), not **mTOR**, the pathway they're known for. The
agent will have to make that link itself, at the pathway level.

### 4.9 Problems still open

| Problem | Why it matters | What would settle it |
|:--|:--|:--|
| **7 drugs have no target in ChEMBL**: amifostine, MRK-003, LLL-12, geldanamycin, triethylenemelamine, quinacrine, MK-4541 | The design would drop them at freeze (1g). They appear in **17,260 of 297,098 rows (5.8%)**; amifostine, quinacrine and triethylenemelamine alone are ~5,000 rows each | **Decided 2026-11-08: keep them.** The agent searches the literature for their targets at run time (see below) |
| **8 manual rows need a second reviewer** | They are judgment calls, each argued in its note | A teammate reads the 8 notes in `drug_synonyms.csv` |
| **`NCGC00090797-11` → digoxin is not fully confirmed** | PubChem's record for it has a different stereo assignment from digoxin's | Low stakes: 4 rows, from the unidentified third study. Removing that study (1e) makes it moot |
| **Stereo fixes assume approved drugs were screened** | The 5 nucleoside rows depend on it | Check the NCI-ALMANAC compound list |

**Why keep the 7, and how the agent fills the gap (decision 2026-11-08).** "No target" means
ChEMBL's curated list is empty for them, not that science doesn't know what they do. A quick
PubMed check found literature for every one; for example, both papers on MK-4541 describe it as an
androgen-receptor modulator. Dropping them would throw away 5.8% of real measurements over one
empty database field.

So they stay in the pool:
- **The response model** learns from their structure, and their target features are simply
  "unknown".
- **The agent**, when it meets one, gets "no curated target" from `get_drug` and searches the
  literature itself, citing what it finds.
- **The trade-off:** those agent-found targets are hypotheses, not curated facts. So results for
  these drugs are reported separately.

Details: [m1-implementation.md, decision 8](m1-implementation.md#decision-8-keep-drugs-without-a-chembl-target).

### 4.10 How this changed the original plan

| Original plan | What we do now | Why |
|:--|:--|:--|
| Strip salts by keeping the largest fragment, for every drug | Never for metal drugs; Pt drugs are checked by atom counts | "Largest fragment" turns cisplatin into ammonia |
| Look up the full InChIKey in ChEMBL | Match on the skeleton (connectivity) key, and require the name to agree | The source has no stereo, so full InChIKeys can't match, and skeletons can't tell twins apart without the name |
| Canonicalize tautomers first | Only as a fallback | It broke the match for 19 drugs |
| Merge automatically | The resolver proposes a committed CSV; a person reviews it; the build re-checks it | Visible, reviewable merges; drift is caught |
| Average repeats created by merging | **Don't average across studies** (moved to 1e, §6) | The two studies disagree on the same experiments (CSS r = 0.22) |

## 5. Step 1d: one cell line, one identity, and its family (built 2026-11-08)

> **Built.** It follows the plan below. It found a third family (SNB-19 + U-251MG) and 5 lines that
> would drop under the current rules. Two of those (SR, UO-31) need a choice of DepMap ID rather
> than a drop. Details and the open decisions are in
> [m1-implementation.md](m1-implementation.md#step-1d-cell-line-harmonization-built-2026-11-08).

> **Built 2026-11-08.** See [m1-implementation.md](m1-implementation.md#step-1d-cell-line-harmonization-built-2026-11-08).
> - The plan below held, with one refinement: labels are matched exactly **within the NCI-60
>   panel**.
> - The data turned up a third family: SNB-19 is a U-251MG derivative.
> - SR and UO-31 have two DepMap IDs each and need an ID chosen.
> - NCI/ADR-RES and HL-60(TB) also have no DepMap ID, in addition to MDA-MB-435.

**The problem.** Cell-line names are messy too (`NCIH23` vs `NCI-H23`, `A549` vs `A-549`), and some
lines aren't what their name says:

```
 NCI/ADR-RES ──"derived from"──► OVCAR-8      (Cellosaurus: contaminated; actually an OVCAR-8 derivative)
 MDA-MB-435  ──"derived from"──► M14          (melanoma, despite a breast-cancer-style name)
```

Both pairs are *inside* our 59 lines.

**The plan:**

1. Look each name up in **Cellosaurus**, the cell-line registry.
   - **Use exact matches only.** Its search ranks gene-edited and contaminated variants first
     (searching "A549" first returns a USP24-edited line).
   - Normalize names first (case, dashes, spaces), and match against the registry's official ID
     and synonyms.
2. Take the **DepMap ID** (e.g. A-549 → `ACH-000681`) from Cellosaurus. DepMap holds each line's
   mutations and gene expression. The design drops lines with no DepMap ID and logs them.
   - Verified: **MDA-MB-435 has none**, so it will be dropped.
3. Record each line's **family** (origin) from Cellosaurus's "derived from" field.

**Why group families (team decision):** later, about 20% of lines are hidden as a test set. If
OVCAR-8 is used for training and its near-twin NCI/ADR-RES is in the test set, the "test" is
partly a copy of training data.
- *Intuition:* studying with a practice exam that's the real exam with the names changed.
- Keeping families on the same side of the split prevents that.

## 6. Step 1e: one row per experiment (done 2026-11-08)

> **What was built** (details in [m1-implementation.md](m1-implementation.md)):
> - Every one of the 297,098 source rows gets exactly one *fate*:
>   - kept: 281,593;
>   - on a dropped line: 10,024;
>   - O'Neil, identifiable by name: 4,284;
>   - O'Neil or ALMANAC, can't tell which: 1,076;
>   - self-pair: 111;
>   - third study: 10.
> - The result is **272,357 experiments** (4,939 pairs × 57 lines, 101 drugs).
> - **Decision 12:** one `study_id` (`NCI-ALMANAC`).
> - **Decision 13:** O'Neil rows are removed (plan A + C). They go to a side table, and the 546
>   experiments both studies measured are a reported finding.
> - P9 resolved: the 9,236 repeats are all ALMANAC's own, so they're averaged.
>
> The plan as written before the build follows, for history.

> **Revised by the [audit](drug-identity-audit.md) (P6–P9); items 2 and 4 below are on hold.**
> Merging drug names (§4) makes 546 experiments appear twice: once from O'Neil, once from
> NCI-ALMANAC. The two studies disagree badly on them (CSS r = 0.22, a 27-point offset).
> Averaging them would invent a number neither lab measured.
> - *Intuition:* two thermometers that disagree by 27 degrees. You don't average them; you find out
>   which one belongs in this room.
>
> The proposed changes:
> - **P6:** never average across studies;
> - **P7:** `study_id` per row, not one constant (**needs a team decision**);
> - **P8:** remove the O'Neil rows from the NCI-60 pool, since O'Neil is meant to be the separate
>   second benchmark (**needs a team decision**);
> - **P9:** hold averaging the 9,830 two-order repeats until their study of origin is known.

1. **Order drugs alphabetically within each pair (A < B by ID).** "Drug X + drug Y" is the same
   experiment as "Y + X". Without a fixed order, the same experiment could be counted, chosen, or
   revealed twice.
2. **Merge repeats.** A triplet measured more than once (the 9,830 two-order pairs, plus any created
   by the salt merges) becomes one row with the **mean**, the **standard deviation**, and `n_reps`.
   The design reports noise through `n_reps` and SD instead of letting selectors re-run
   experiments.
3. **Set self-pairs aside (team decision: separate table).** A drug "combined" with itself isn't a
   combination, so it's kept out of the pool. It's still saved, because it's a free check on the
   synergy scores:
   - **Loewe** additivity is built on the "sham combination" principle: a drug with itself should
     score about 0. Self-pairs far from 0 under Loewe point to noise or fitting artifacts.
   - **Bliss** is *not* guaranteed to be 0 for a self-pair. It assumes the two drugs act
     independently, and a drug isn't independent of itself, so steep dose-response curves can look
     synergistic or antagonistic under Bliss.
   - How far self-pair Bliss scores drift from 0 therefore gives a rough sense of the background
     noise and bias in our primary synergy score (`y_syn` = Bliss).
   - *(An earlier note said "Bliss of a drug with itself should be ≈ 0". That was incorrect, and it
     is corrected here.)*
4. **`study_id`** (team decision): one constant per dataset (`NCI-ALMANAC`), because the TDC file
   has no study column. The leakage filter (M5) needs to know which study, and so which paper,
   produced the data.

## 7. Step 1f: describe drugs and cells as numbers (done 2026-11-08)

> **What was built** (details in [depmap-data.md](depmap-data.md)):
> - **Sources:**
>   - DepMap 26Q3: lineage and mutations;
>   - NCI CellMiner: expression and copy number for **all** 57 lines, one source per data type;
>   - SNB-19, NCI-H322M and HL-60(TB): mutations from their confirmed parents;
>   - SR: expression **imputed** from CellMiner's own microarrays, because its RNA-seq sample is a
>     mix-up (decision 17).
> - **Output:** cells 57 × 1,073; drugs 122 × 2,335.
>
> The original plan follows.

The response model can't read names. It needs numeric descriptions:

| For | Features (design §4 step 6) | Intuition |
|:--|:--|:--|
| Drugs | 2048-bit Morgan fingerprint; target set; mechanism class | a structural "barcode", plus what it hits |
| Cell lines | lineage (tissue); driver mutations (KRAS, BRAF, PIK3CA, PTEN, TP53, …); top 1,000 variable genes | what kind of tumor, and which pathways are broken |

- **Needs you:** DepMap's download site blocks scripts (a bot check), so the mutation and
  expression files must be downloaded in a browser.
- Disk space matters here.

## 8. Step 1g: freeze everything (done 2026-11-08)

> **What was built** (details and figures in [m1-freeze-and-splits.md](m1-freeze-and-splits.md)):
> - **The split:** 11 held-out lines, stratified by lineage, families together, seed 0. Only lines
>   with all-own features are eligible (decision 19).
> - **Hit thresholds:** y_syn ≥ 9.30, y_inh ≥ 62.95, from selectable experiments.
> - **Decision 20:** synergy hits that barely kill are kept and flagged, and results are reported
>   with and without them.
> - **Outputs:** `pool` / `labels` (separate), `cells`, `drugs`, `splits.json`, `data_card.md`,
>   `manifest.json`. All are byte-reproducible.
>
> The original plan follows.

"Freezing" means computing these once, saving them, and never changing them, so every selector
plays exactly the same game:

1. **Choose the held-out cell lines here, in M1** (team decision; the design had this in M2).
   - About 20% of lines, balanced by tissue, with families kept together, using a fixed random
     seed.
   - *Why move it:* hit thresholds should come from the lines selectors can actually pick from.
     That requires knowing the held-out lines first. Doing both in one place freezes them together,
     consistently.
2. **Compute hit thresholds** (top 5% per outcome) on those selectable lines.
3. **Write outcomes to a separate file** (team decision: `labels.parquet` apart from
   `pool.parquet`).
   - *Why:* the oracle is meant to be the **only** thing that ever sees results.
   - Physically separating the answers into their own file makes that rule enforceable. A test can
     check that no other code opens `labels.parquet`.
   - *Intuition:* the answer key is kept in a locked drawer, not stapled to the exam.
4. **Write the data card** (`data_card.md`): counts, score distributions, thresholds, and dropped
   items with reasons.
   - It must reproduce **byte-for-byte** from the same raw files, so it contains no timestamps or
     machine paths.
   - Dates and hashes go in a separate manifest instead.

---

## 9. All decisions in one table

| Decision | Alternatives | Why this one | Decided by |
|:--|:--|:--|:--|
| Download TDC files directly; no `tdc` package | Install PyTDC | It can't coexist with our pandas/numpy/rdkit and fails on Python 3.13; it adds nothing for this dataset (§2.2) | Engineering evidence |
| Restricted unpickler in a pinned, throwaway environment | Plain `pd.read_pickle` | Pickles can run code; old-pandas pickles need old pandas; pins make it repeatable | Team approved |
| Checksum + size check + retries | Trust the download | Both failure modes actually happened (a truncated body, then a hash mismatch) | Engineering evidence |
| Drop TDC's `CellLine` features | Keep them | Undocumented; the design uses DepMap | Design §4 |
| NCI-60 only for now | Add O'Neil now | O'Neil has only one score (no CSS/Bliss), is a 1.6 GB file, and disk is low | Team |
| Merge drugs by parent structure | Merge only identical SMILES; don't merge | Salt forms are the same active drug; structure beats inconsistent names | Team |
| …implemented as: ChEMBL parent ID, accepted only when name and structure agree (audit P1–P2) | Structure alone; name alone | Structure alone can't tell mirror-image twins apart; names alone are inconsistent (§4.2, §4.4) | Audit evidence |
| No salt-stripping for metal drugs; check them by atom counts (P3) | Strip everything | Stripping turns cisplatin into ammonia (§4.3) | Audit evidence |
| Feature structures from ChEMBL, with stereo (P4) | Use the flat source SMILES | The source has no stereo at all | Audit evidence |
| A committed, reviewed `drug_synonyms.csv`; the build re-checks it (P5) | Resolve live on every build | Merges are visible in diffs; drift in code or ChEMBL is caught (§4.5) | Audit evidence |
| Borrow targets for fludarabine and plicamycin (recorded per row) | Leave them without targets | Same drug filed under another ChEMBL record (§4.8) | Engineering evidence; please review |
| Keep the 7 drugs without a ChEMBL target; the agent searches the literature for their targets | Drop them (design §4 step 7); hand-curate targets into the features | 5.8% of rows are real data; literature exists for all 7; keeps model features reproducible, and agent findings are cited | Team (2026-11-08) |
| Self-pairs in a separate table | Drop; keep in the pool | Not combinations, but a useful check: Loewe should be ≈ 0, and Bliss drift shows noise/bias | Team |
| Group related cell lines in the split | Drop derivatives; treat as independent | Prevents near-duplicate lines leaking across the held-out split | Team |
| Held-out split in M1; thresholds on selectable lines | Follow the design literally | Thresholds depend on the split; freeze both together | Team |
| Separate `labels.parquet` | Outcomes inside `pool.parquet` | Makes the "only the oracle sees results" rule enforceable | Team |
| `study_id` = one constant per dataset | Leave it blank | The TDC file has no study column; leakage control needs one | Team |

Decisions 9–20, from the 1d–1g builds, are listed with reasons in
[m1-freeze-and-splits.md §2](m1-freeze-and-splits.md). They cover:
- the DepMap IDs and proxies;
- the drops;
- O'Neil removal;
- CellMiner as a source;
- the parent mutations;
- SR imputation;
- held-out eligibility;
- the low-kill synergy hits.

## 10. Glossary of the engineering terms used here

| Term | Plain meaning |
|:--|:--|
| **sha256 / checksum** | A unique fingerprint of a file's exact bytes; used to prove two files are identical |
| **Pickle** | A Python save-file format; loading it can run code, so it's unsafe for untrusted files |
| **Parquet** | A plain, fast columnar data file; safe to load, readable by any pandas |
| **Wheel** | A prebuilt Python package; without one for your Python version, pip must compile from source |
| **venv / `.venv`** | A private folder of installed packages for one project |
| **`uv` / `uv.lock`** | The tool that installs this project's packages, and the file recording their exact versions |
| **User-Agent** | The "who's asking" label a program sends with each web request |
| **InChIKey** | A standard ID for a molecule's structure (a molecular barcode) |
| **Connectivity key** | The first 14 characters of the InChIKey: which atoms bond to which, ignoring 3-D shape (the "skeleton") |
| **Salt / parent** | A salt is the drug plus a helper ion (e.g. hydrochloride); the parent is the active drug alone |
| **Stereoisomer** | Same skeleton, different 3-D arrangement, like left and right gloves; often only one is the medicine |
| **Tautomer** | The same molecule drawn with a hydrogen in a different place; different tools pick different "standard" drawings |
| **Prodrug** | An inactive molecule the body converts into the drug (e.g. fludarabine phosphate → fludarabine) |
| **Mechanism / target** | What a drug acts on, usually a protein named by its gene (erlotinib → EGFR) |
| **Response cache / offline replay** | Every database answer saved to disk, so a build can be re-run without the network and give identical output |
| **ChEMBL / Cellosaurus / DepMap** | Public databases: drugs and targets / cell-line registry / cell-line genetics |
| **Held-out lines** | Cell lines hidden from every selector, used only to test mechanism predictions |

## 11. Where we are and what's next

**M1 is complete (2026-11-08):**

```
 1a raw ✅ ─► 1b profiled ✅ ─► 1c drugs ✅ ─► 1d lines ✅ ─► 1e rows ✅ ─► 1f features ✅ ─► 1g freeze ✅
                             129 → 122      59 → 57 kept   272,357        DepMap 26Q3      11 held out
                             drugs          3 families     experiments    + CellMiner      thresholds fixed
```

All 20 decisions are settled; see [m1-freeze-and-splits.md §2](m1-freeze-and-splits.md).

**Reviews still worth doing:** a teammate reads the 8 manual drug rows and the 2 borrowed target
sets (each has its evidence in a note in `curated/drug_synonyms.csv` and
`curated/drug_mechanism_proxies.csv`), plus the 3 hand-set rows in `curated/cell_lines.csv`.

**Next (M2, design §5):**
1. The **oracle**, the only reader of `labels.parquet`, with a test that nothing else imports it.
2. **Seed sets:** 10 × 32 free experiments.
3. **Famous / middle / obscure pair labels** from PubMed co-mentions.

Then M3, the response model, which must report synergy results **with and without** the
`syn_hit_low_kill` hits (decision 20).
