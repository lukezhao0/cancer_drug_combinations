"""Step 1e: one row per distinct experiment (target_design.md section 4, steps 4-5).

Inputs (all in `data_cache/interim/`): the TDC table (1a), `drug_map.parquet` (1c) and
`line_map.parquet` (1d).

**Every source row gets exactly one fate**, checked in this order:

| fate | rows | why |
|:--|:--|:--|
| `third_study` | 10 | 5 drugs paired only with each other on SN12C; not NCI-ALMANAC |
| `oneil` | 4,284 | O'Neil/Merck 2016 rows that carry an O'Neil-only drug label (see `ONEIL_LABELS`). Decision 13: removed from the pool |
| `oneil_ambiguous` | 1,076 | Pairs of two drugs both studies tested, on O'Neil's 9 lines. Nothing in the file says which study measured them, so they are removed to keep the pool on one scale |
| `line_dropped` | rows on lines 1d does not keep | decision 11: MDA-MB-435, NCI/ADR-RES |
| `self_pair` | rows where both labels are the same drug | decision 5: their own table |
| `kept` | the rest | the NCI-ALMANAC pool |

Decision 12 (P7): the pool is one study, so `study_id` is the constant `NCI-ALMANAC`.

Kept rows are mapped to 1c's `drug_id` and 1d's `cellosaurus_ac`, and each pair is ordered so that
`drug_a < drug_b`. Rows of the same (pair, line) are replicates within NCI-ALMANAC and are averaged:
mean, sample SD (NaN for a single row) and `n_reps` (P6, P9). Most come from the same pair stored in
both drug orders.

Removed rows are not thrown away. Under plan C they go to side tables:

    oneil_rows.parquet            every `oneil` / `oneil_ambiguous` row, mapped, for the second benchmark
    cross_study_overlap.parquet   triplets measured by both studies (546), O'Neil vs ALMANAC means

Outputs:

    experiments.parquet           one row per (drug_a, drug_b, cell line): the pool, with outcomes
    self_pairs.parquet            same shape, drug_a == drug_b
    row_fate.parquet              source row index -> fate (exactly one each)
    experiments_report.json       counts, replicate and cross-study agreement, every check

Outcomes (design section 4 step 5): `y_syn` = mean Bliss and `y_inh` = mean CSS. ZIP, Loewe and HSA
are kept alongside them. Splitting into `pool` / `labels`, the held-out lines and thresholds is step 1g.

Run:
    python -m combo_agent.data.build_experiments [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from combo_agent.data import ingest_tdc, paths

STUDY_ID = "NCI-ALMANAC"

#: The 9 cell lines where O'Neil's 39-line panel overlaps NCI-60 (source labels).
ONEIL_LINES = frozenset({"HCT116", "HT29", "NCI-H460", "NCIH23", "OVCAR3", "SK-OV-3", "SW-620",
                         "T-47D", "UACC62"})
#: Drug labels used only by O'Neil. These are its 16 drugs that ALMANAC never tested, plus O'Neil's
#: generic names for 7 drugs that ALMANAC lists under a salt, brand or NSC name (drug-identity-audit
#: §2). A row carrying any of them is an O'Neil row.
ONEIL_ONLY_DRUGS = frozenset({"ABT-888", "AZD1775", "BEZ-235", "DEXAMETHASONE", "DINACICLIB",
                              "GELDANAMYCIN", "GEMCITABINE", "METFORMIN", "MK-2206", "MK-4541",
                              "MK-4827", "MK-5108", "MK-8669", "MK-8776", "MRK-003", "PD325901"})
ONEIL_SYNONYMS = frozenset({"ERLOTINIB", "TOPOTECAN", "VINBLASTINE", "VINORELBINE", "ETOPOSIDE",
                            "DOXORUBICIN", "OXALIPLATIN"})
ONEIL_LABELS = ONEIL_ONLY_DRUGS | ONEIL_SYNONYMS
#: A small, unidentified screen: these 5 drugs paired with each other on one line (SN12C).
THIRD_STUDY_DRUGS = frozenset({"NCGC00090797-11", "LLL-12", "BARDOXOLONE METHYL",
                               "ABT-263 (NAVITOCLAX)", "CARFILZOMIB (PR-171)"})

SCORES = {"CSS": "css", "Synergy_Bliss": "bliss", "Synergy_ZIP": "zip", "Synergy_Loewe": "loewe",
          "Synergy_HSA": "hsa"}
FATES = ("third_study", "oneil", "oneil_ambiguous", "line_dropped", "self_pair", "kept")
KEY = ["drug_a", "drug_b", "cellosaurus_ac"]


class ExperimentError(RuntimeError):
    """An input is incomplete, or a check that guards the fate rules failed."""


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class Result:
    experiments: pd.DataFrame
    self_pairs: pd.DataFrame
    row_fate: pd.DataFrame
    oneil_rows: pd.DataFrame
    overlap: pd.DataFrame
    report: dict[str, Any]
    checks: list[Check] = field(default_factory=list)


def shared_drugs(df: pd.DataFrame) -> frozenset[str]:
    """Labels O'Neil tested under the same name ALMANAC uses.

    Derived, not listed: these are the partners of O'Neil-only labels in O'Neil rows. Every O'Neil
    drug is paired with every other, so its whole drug set shows up there.
    """
    on = df.Drug1_ID.isin(ONEIL_LABELS) | df.Drug2_ID.isin(ONEIL_LABELS)
    partners = set(df.loc[on, "Drug1_ID"]) | set(df.loc[on, "Drug2_ID"])
    return frozenset(partners - ONEIL_LABELS)


def assign_fates(df: pd.DataFrame, line_keep: pd.Series, drug_of: pd.Series) -> tuple[pd.Series, frozenset[str]]:
    """One fate per row, in the precedence order of `FATES`."""
    d1, d2, line = df.Drug1_ID, df.Drug2_ID, df.Cell_Line_ID
    shared = shared_drugs(df)
    rules = [
        ("third_study", d1.isin(THIRD_STUDY_DRUGS) | d2.isin(THIRD_STUDY_DRUGS)),
        ("oneil", d1.isin(ONEIL_LABELS) | d2.isin(ONEIL_LABELS)),
        ("oneil_ambiguous", d1.isin(shared) & d2.isin(shared) & line.isin(ONEIL_LINES)),
        ("line_dropped", ~line.map(line_keep).astype(bool)),
        ("self_pair", d1.map(drug_of) == d2.map(drug_of)),
    ]
    fate = pd.Series("kept", index=df.index, dtype=object)
    for name, mask in reversed(rules):  # earlier rules win, so apply them last
        fate[mask] = name
    return fate, shared


def aggregate(rows: pd.DataFrame) -> pd.DataFrame:
    """Mean, sample SD and count per (drug_a, drug_b, line). Sorted by key."""
    g = rows.groupby(KEY, sort=True)
    out = g.size().rename("n_reps").to_frame()
    for src, name in SCORES.items():
        out[f"{name}_mean"] = g[src].mean()
        out[f"{name}_sd"] = g[src].std(ddof=1)  # NaN when n_reps == 1
    out = out.reset_index()
    out["y_syn"] = out["bliss_mean"]
    out["y_inh"] = out["css_mean"]
    return out


def _canonical(rows: pd.DataFrame) -> pd.DataFrame:
    a, b = rows["drug_1"], rows["drug_2"]
    return rows.assign(drug_a=np.where(a <= b, a, b), drug_b=np.where(a <= b, b, a))


def _corr(x: pd.Series, y: pd.Series, rank: bool = False) -> float | None:
    if len(x) < 3:
        return None
    if rank:
        x, y = x.rank(), y.rank()
    return round(float(np.corrcoef(x, y)[0, 1]), 3)


def build(df: pd.DataFrame, drug_map: pd.DataFrame, line_map: pd.DataFrame) -> Result:
    checks: list[Check] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append(Check(name, bool(ok), detail))

    drug_of = drug_map.set_index("source_id")["drug_id"]
    lines = line_map.set_index("source_id")
    labels = set(df.Drug1_ID) | set(df.Drug2_ID)
    check("every drug label is in drug_map", labels <= set(drug_of.index),
          ", ".join(sorted(labels - set(drug_of.index))))
    check("every line label is in line_map", set(df.Cell_Line_ID) <= set(lines.index),
          ", ".join(sorted(set(df.Cell_Line_ID) - set(lines.index))))
    check("every listed O'Neil / third-study label occurs", (ONEIL_LABELS | THIRD_STUDY_DRUGS) <= labels,
          ", ".join(sorted((ONEIL_LABELS | THIRD_STUDY_DRUGS) - labels)))
    if not all(c.ok for c in checks):
        raise ExperimentError("; ".join(f"{c.name}: {c.detail}" for c in checks if not c.ok))

    fate, shared = assign_fates(df, lines["keep"], drug_of)

    # Guards on the fate rules: each one fails loudly if the data contradicts what the rule assumes.
    on = fate == "oneil"
    check("O'Neil rows lie only on the 9 O'Neil lines", df.loc[on, "Cell_Line_ID"].isin(ONEIL_LINES).all(),
          ", ".join(sorted(set(df.loc[on, "Cell_Line_ID"]) - ONEIL_LINES)))
    th = fate == "third_study"
    th_ok = (df.loc[th, "Drug1_ID"].isin(THIRD_STUDY_DRUGS) & df.loc[th, "Drug2_ID"].isin(THIRD_STUDY_DRUGS)).all()
    check("third-study drugs pair only with each other, on one line",
          th_ok and df.loc[th, "Cell_Line_ID"].nunique() == 1)
    rest = df[~fate.isin(["oneil", "third_study"])]
    per_label_lines = pd.concat([rest[["Drug1_ID", "Cell_Line_ID"]].set_axis(["d", "l"], axis=1),
                                 rest[["Drug2_ID", "Cell_Line_ID"]].set_axis(["d", "l"], axis=1)])
    confined = sorted(d for d, ls in per_label_lines.groupby("d")["l"]
                      if set(ls) <= ONEIL_LINES and d not in shared)
    check("no unlisted drug label is confined to the O'Neil lines (an unknown study)", not confined,
          ", ".join(confined))
    check("all 9 O'Neil lines are kept by 1d", bool(lines.loc[sorted(ONEIL_LINES), "keep"].all()))

    row_fate = pd.DataFrame({"source_row": df.index.astype("int64"), "fate": fate.values})
    counts = row_fate.fate.value_counts().reindex(FATES, fill_value=0)
    check("every row has exactly one fate", int(counts.sum()) == len(df) and row_fate.source_row.is_unique)

    mapped = df.assign(drug_1=df.Drug1_ID.map(drug_of), drug_2=df.Drug2_ID.map(drug_of),
                       cellosaurus_ac=df.Cell_Line_ID.map(lines["cellosaurus_ac"]))
    mapped = _canonical(mapped)
    line_cols = (line_map.drop_duplicates("cellosaurus_ac")
                 .set_index("cellosaurus_ac")[["depmap_id", "depmap_is_proxy", "family_id"]])

    def attach(t: pd.DataFrame) -> pd.DataFrame:
        t = t.join(line_cols, on="cellosaurus_ac")
        t.insert(3, "study_id", STUDY_ID)
        return t

    experiments = attach(aggregate(mapped[fate == "kept"]))
    self_pairs = attach(aggregate(mapped[fate == "self_pair"]))

    check("experiments are unique per (drug_a, drug_b, line)", not experiments.duplicated(KEY).any())
    check("drug_a < drug_b in every experiment", bool((experiments.drug_a < experiments.drug_b).all()))
    check("self_pairs have drug_a == drug_b", bool((self_pairs.drug_a == self_pairs.drug_b).all()))
    check("no missing outcomes", not experiments[["y_syn", "y_inh"]].isna().any().any())
    check("every experiment line has a DepMap ID", bool(experiments.depmap_id.notna().all()))
    oneil_only_ids = set(drug_of[sorted(ONEIL_ONLY_DRUGS)])
    check("no O'Neil-only drug is in the pool",
          not (set(experiments.drug_a) | set(experiments.drug_b)) & oneil_only_ids)
    check("rows are conserved: kept rows = sum of n_reps",
          int(experiments.n_reps.sum()) == int(counts["kept"]))

    # Plan C side tables.
    oneil_rows = (mapped[fate.isin(["oneil", "oneil_ambiguous"])]
                  .assign(fate=fate[fate.isin(["oneil", "oneil_ambiguous"])])
                  [["fate", "Drug1_ID", "Drug2_ID", "Cell_Line_ID", *KEY, *SCORES]]
                  .reset_index(names="source_row").sort_values("source_row", ignore_index=True))
    on_agg = aggregate(mapped[fate == "oneil"])
    overlap = experiments[KEY + [f"{n}_mean" for n in SCORES.values()] + ["n_reps"]].merge(
        on_agg[KEY + [f"{n}_mean" for n in SCORES.values()] + ["n_reps"]], on=KEY,
        suffixes=("_almanac", "_oneil")).sort_values(KEY, ignore_index=True)

    agreement = {}
    for n in SCORES.values():
        x, y = overlap[f"{n}_mean_almanac"], overlap[f"{n}_mean_oneil"]
        agreement[n] = {"pearson": _corr(x, y), "spearman": _corr(x, y, rank=True),
                        "mean_almanac": round(float(x.mean()), 2) if len(x) else None,
                        "mean_oneil": round(float(y.mean()), 2) if len(y) else None,
                        "mean_abs_diff": round(float((x - y).abs().mean()), 2) if len(x) else None}
    kept_rows = mapped[fate == "kept"].sort_index()
    sizes = kept_rows.groupby(KEY)[KEY[0]].transform("size")
    two = kept_rows[sizes == 2].groupby(KEY)  # first/last are aligned per experiment key
    rep_agree = {}
    if len(kept_rows[sizes == 2]):
        firsts, lasts = two[list(SCORES)].first(), two[list(SCORES)].last()
        for src, n in SCORES.items():
            rep_agree[n] = _corr(firsts[src], lasts[src])

    report = {
        "study_id": STUDY_ID,
        "source_rows": len(df),
        "fates": {k: int(v) for k, v in counts.items()},
        "shared_drug_labels": sorted(shared),
        "experiments": len(experiments),
        "pairs": int(experiments.groupby(["drug_a", "drug_b"]).ngroups),
        "drugs": len(set(experiments.drug_a) | set(experiments.drug_b)),
        "lines": int(experiments.cellosaurus_ac.nunique()),
        "proxy_lines": sorted(experiments.loc[experiments.depmap_is_proxy.astype(bool), "cellosaurus_ac"].unique()),
        "n_reps": {str(k): int(v) for k, v in experiments.n_reps.value_counts().sort_index().items()},
        "replicate_agreement_pearson_n2": rep_agree,
        "self_pair_experiments": len(self_pairs),
        "lines_per_pair_median": float(experiments.groupby(["drug_a", "drug_b"]).cellosaurus_ac.nunique().median()),
        "cross_study_overlap": {"triplets": len(overlap), "agreement": agreement},
        "oneil_side_table_rows": len(oneil_rows),
        "checks": [{"name": c.name, "ok": c.ok, "detail": c.detail} for c in checks],
    }
    if not all(c.ok for c in checks):
        raise ExperimentError("; ".join(f"{c.name}: {c.detail}" for c in checks if not c.ok))
    return Result(experiments, self_pairs, row_fate, oneil_rows, overlap, report, checks)


def write_outputs(result: Result, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {"experiments": result.experiments, "self_pairs": result.self_pairs,
              "row_fate": result.row_fate, "oneil_rows": result.oneil_rows,
              "cross_study_overlap": result.overlap}
    out = {}
    for name, t in tables.items():
        out[name] = out_dir / f"{name}.parquet"
        t.to_parquet(out[name], index=False)
    out["report"] = out_dir / "experiments_report.json"
    out["report"].write_text(json.dumps(result.report, indent=2, sort_keys=True) + "\n")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Step 1e: one row per distinct experiment")
    p.add_argument("--out", type=Path, default=None, help="output dir (default data_cache/interim)")
    args = p.parse_args(argv)
    interim = paths.interim_dir()
    df = ingest_tdc.load("tdc_drugcomb")
    result = build(df, pd.read_parquet(interim / "drug_map.parquet"),
                   pd.read_parquet(interim / "line_map.parquet"))
    out = write_outputs(result, args.out or interim)
    r = result.report
    print(f"{r['source_rows']:,} rows -> {r['experiments']:,} experiments "
          f"({r['pairs']:,} pairs x {r['lines']} lines, {r['drugs']} drugs)")
    print("fates:", r["fates"])
    print("cross-study overlap:", r["cross_study_overlap"]["triplets"], "triplets; CSS",
          r["cross_study_overlap"]["agreement"]["css"])
    for k, v in out.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
