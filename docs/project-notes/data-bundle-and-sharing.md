# Sharing the M1 data: the `m1-data-v1` bundle (what, why, how)

Built 2026-11-08. The bundle's own docs (collaborator-facing) live in
[`../data-bundle/`](../data-bundle/). This page records **how and why we built it**, the problems found while
testing it, and how to publish it.

---

## 1. The problem it solves

After M1, the frozen data pool existed only on one laptop, in `data_cache/` (about 2 GB, and
git-ignored). A collaborator had two options, both bad:
- **rebuild it from scratch**: a manual 1.6 GB DepMap download, live database queries and 7
  pipeline steps;
- **or ask for files ad hoc**, with no documentation of what each column means.

**Goal:** one download that lets anyone start M2–M7 straight away, documented well enough to need
no explanation, and verifiable byte for byte.

## 2. Decisions

| Question | Decision | Why |
|:--|:--|:--|
| Put the data in git? | **No.** Host it on a shared drive (Box or Drive); keep only checksums and small metadata in git | Git history is permanent: hidden labels committed once can't be removed without rewriting history. Committing labels also undermines the "only the oracle reads labels" rule. Generated files go stale. Licences: ChEMBL content is share-alike |
| What goes in? | `processed/` (the frozen pool) **plus** the `interim/` files later milestones need | `processed/` alone is enough for M2–M4. But M6 agent tools need drug mechanism records, and M7 rule checks need full gene-level expression and copy number |
| What stays out? | Raw downloads (TDC, DepMap, CellMiner), raw-derived intermediates, `experiments.parquet` | rebuildable; large; some duplicate every outcome |
| Layout | The data mirrors the repo's `data_cache/` exactly | unpack, copy, and the code finds everything; also works standalone with just pandas |
| Docs | Hand-written docs kept **in git** (`docs/data-bundle/`); the data dictionary is **generated from the files** | reviewable in pull requests; the dictionary can't drift from the data |
| Versioning | Name it `m1-data-v1`; never edit a version in place; record the code commit | results stay traceable to the exact data used |
| Verification | `SHA256SUMS` inside the bundle; the archive's sha256 recorded next to the download link | anyone can prove they have the right bytes |

## 3. How it was built

**The builder** is `scripts/make_data_bundle.py`:

```bash
uv run python scripts/make_data_bundle.py --with-api-cache
```

It does this, in order:
1. Checks that every input file exists (the pipeline must have been built through `freeze_pool`).
2. Copies 26 data files into `data_cache/bundles/m1-data-v1/data_cache/`, keeping the repo layout.
3. Copies the four hand-written docs from `docs/data-bundle/`: README, SPLITS_AND_HITS, CAVEATS,
   PROVENANCE.
4. **Generates `DATA_DICTIONARY.md`** by reading every parquet file's schema and row count.
   - Columns are described from a mapping in the script.
   - The wide feature tables (cells: 1,087 columns; drugs: 2,336) group their columns by prefix
     (`fp_`, `target_`, `expr_`, …).
   - The full gene tables list their gene-column count.
   - **If any column has no description, the build stops.**
   - A copy is written to `docs/data-bundle/DATA_DICTIONARY.md`, so it can be reviewed in git.
5. Writes `SHA256SUMS` (one line per data file) and `BUNDLE_INFO.json` (version, date, code commit,
   file count, size).
6. Archives everything as `m1-data-v1.tgz` and writes `m1-data-v1.tgz.sha256`.
7. With `--with-api-cache`, also archives the API cache as `m1_api_cache.tgz` (see §5).

**Result:**

| Item | Value |
|:--|--:|
| Data files | 26 |
| Unpacked size | 63.9 MB |
| `m1-data-v1.tgz` | 24.3 MB |
| `m1_api_cache.tgz` | 1.1 MB |

All outputs are in `data_cache/bundles/`, which git ignores.

The **archive's** sha256 changes on every rebuild, because archives record timestamps. The **data
files'** checksums stay identical. Always record the sha256 of the archive actually uploaded.

## 4. How it was tested, and what the tests caught

**Collaborator simulation:**
1. Unpacked the archive into an empty folder.
2. Ran `shasum -a 256 -c SHA256SUMS`: all 26 files OK.
3. Ran the README's Python quick start exactly as written. It produced a 2,000 × 5,735 feature
   matrix with no non-numeric columns, and matching counts (272,357 experiments; 219,239
   selectable; thresholds 9.30 and 62.95).

**Problems caught and fixed before shipping:**

| Problem | Impact | Fix |
|:--|:--|:--|
| The first setup command unpacked the bundle into the repo root with `--strip-components=1` | would have **overwritten the repo's README.md** with the bundle's | verify in a separate folder, then copy only `data_cache/` |
| 71 columns fell back to "see the build reports" | the dictionary was incomplete for the mapping and audit tables | wrote descriptions for every column, and made a missing description a build error |
| Prefix grouping applied to narrow tables | `target_chembl_id` and `action_type` were labelled as "multi-hot" features | group only tables with more than 200 columns |
| The bundle README linked to `DATA_DICTIONARY.md`, which existed only in built bundles | broken link in the repo | the builder also writes the dictionary into `docs/data-bundle/` |

## 5. What `m1_api_cache.tgz` is

**In one sentence:** it's the saved answers from the public web databases that steps 1c and 1d
asked. A rebuild can replay them and get exactly the same result, without going online.

**How it came about.** `src/combo_agent/data/api_cache.py` (`CachedJsonClient`) wraps every web
request made in steps 1c and 1d:
- **Before** asking a service, it checks `data_cache/raw/api_cache/<service>/` for a file named by
  the sha256 of the request URL.
- **If it's there**, it uses the saved answer.
- **If not**, it asks the service, then saves `{url, status, body}` as JSON. "Not found" answers
  are saved too, because "this name isn't in ChEMBL" is also a result.
- **With `COMBO_OFFLINE=1`** (or `--offline`), a missing entry raises an error instead of going
  online. That proves a build used only saved answers.

**What's in it** (1,437 files, about 12 MB unpacked):

| Service | Answers | Used for |
|:--|--:|:--|
| ChEMBL | 1,225: molecules 980, mechanisms 139, targets 105, status 1 | drug name → ChEMBL parent; parent structures; mechanism targets (step 1c) |
| PubChem | 129 | drug name → InChIKey, when ChEMBL's synonyms miss (1c) |
| Cellosaurus | 83 | cell-line records and the NCI-60 panel list (1d); field-filtered to keep responses small |

**Why it matters.** Public databases change: ChEMBL releases, merged records, new synonyms.
Without the cache, a rebuild months later could resolve a drug differently and change the pool
silently. With it, the offline rebuild is **byte-identical**: verified for 1c, 1d, 1e, 1f and 1g by
the real-data tests (`COMBO_INTEGRATION=1 COMBO_OFFLINE=1`).

**Who needs it.**
- **Not** someone using the bundle: everything is already in the data.
- **Only** someone rebuilding or modifying the pipeline who wants the exact `m1-data-v1` result.

**Licence:** the ChEMBL answers are CC BY-SA 3.0. Fine to share in the team; share-alike applies
if published.

## 6. Publishing

**Status (2026-11-08): published** to the team's Google Drive folder:
https://drive.google.com/drive/folders/1GsulBMNAGEjXhBt8p-UvNpYufTfvvRW7?usp=drive_link

| File | sha256 |
|:--|:--|
| `m1-data-v1.tgz` | `bd33234e3cf63af3221de6a828b246b0f3a599a69a8088d8af8e65878f4da65b` |
| `m1_api_cache.tgz` | `b7577a54a36e1a9330f380fab2c7defa198636a877c1f97cbfcc1c76080272af` |

The link and both checksums are in the repo's `README.md` ("Getting the data") and in
`docs/HANDOFF.md`.

**About the recorded code version.** `BUNDLE_INFO.json` inside the uploaded archive says
`78f71b5 (with uncommitted pipeline changes)`, because the bundle was built before the M1 code was
committed.
- **The data is unaffected.** Every data file's sha256 is listed in the bundle's `SHA256SUMS` and in
  `artifacts/m1_freeze/manifest.json`. Building from the committed code reproduces the same bytes.
- **The archive was not rebuilt after committing.** A rebuild would change the archive's checksum,
  and the uploaded copy would no longer match the README.
- Once the M1 commits exist, the CHANGELOG records which commit corresponds to `m1-data-v1`.

**Checklist for future versions:**
1. Commit the code first.
2. Build `uv run python scripts/make_data_bundle.py --version m1-data-vN --with-api-cache`.
3. Upload to the same Drive folder **as a new file**. Never replace an earlier version.
4. Update the README's link and sha256, and add a CHANGELOG entry. Commit those.

## 7. Files involved

| File | Role |
|:--|:--|
| `scripts/make_data_bundle.py` | builder (tracked) |
| `docs/data-bundle/{README,DATA_DICTIONARY,SPLITS_AND_HITS,CAVEATS,PROVENANCE}.md` | bundle docs (tracked) |
| `README.md` ("Getting the data"), `docs/HANDOFF.md` §5, `docs/CHANGELOG.md` | point to the bundle |
| `data_cache/bundles/` | built outputs (git-ignored) |
| `src/combo_agent/data/api_cache.py` | the caching client behind `m1_api_cache` |
