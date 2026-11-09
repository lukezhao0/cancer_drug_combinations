"""Step 1g: split the cell lines, fix the hit thresholds, and freeze the pool (target_design.md §4 step 7, §5).

Inputs: `data_cache/interim/experiments.parquet` (1e) and `data_cache/interim/features/` (1f).
Outputs, in `data_cache/processed/`:

    pool.parquet     one row per experiment: keys, split, line metadata, n_reps, study_id. NO outcomes
    labels.parquet   the same keys plus outcomes, SDs and hit flags. Only the oracle (M2) may read this
    cells.parquet    one row per pool line: identity, split, data-source flags, and the 1f features
    drugs.parquet    one row per drug: target_status, in_pool, and the 1f features
    self_pairs.parquet  drug-with-itself experiments (decision 5), with outcomes, for score checks
    splits.json      the held-out lines, how they were chosen, and the hit thresholds
    data_card.md     human-readable summary. Byte-for-byte reproducible (no dates, no paths)
    manifest.json    sha256 of every input and output, plus versions

**Held-out cell lines** (design §5.1): 20% of lines, stratified by lineage, never seen by a
selector, used only to test the agent's mechanistic predictions (design §9).
* The unit is a **family**. Lines that share an origin (1d) are held out together or not at all,
  otherwise a near-copy of a test line would sit in training.
* Per-lineage quotas use the largest-remainder rule on 0.2 x lineage size, summing to
  round(0.2 x lines). Ties go to the lineage name, alphabetically.
* **Eligibility (decision 19):** a line is eligible only if every feature it has is its own
  measurement. Excluded: imputed expression (SR), mutations borrowed from a parent (SNB-19,
  NCI-H322M, HL-60(TB)), and a DepMap proxy (HL-60(TB)). Held-out lines exist to check predictions
  written over mutations, copy number, lineage and expression, so a borrowed feature there would
  test the proxy, not the line. An ineligible line stays selectable. If any member of a family is
  ineligible, the whole family is.
* Within each lineage, eligible families are shuffled with `numpy.random.default_rng(SPLIT_SEED)`
  and taken in order until the quota is met, never overshooting it.

**Hits** (design §3 glossary, gap analysis §2.7): `is_hit_syn` = y_syn >= the 95th percentile of
**selectable** experiments, and the same for `is_hit_inh`. The same thresholds apply to held-out
experiments.

`syn_hit_low_kill` flags synergy hits whose CSS < 0. Bliss can look strongly synergistic when the
pair barely kills, or single agents stimulate growth: 7% of synergy hits, concentrated on weakly
cytotoxic or hormonal drugs (2026-11-08). The flag is reported, not acted on; changing the hit
definition is a user decision.

Run: python -m combo_agent.data.freeze_pool
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from combo_agent.data import paths

SPLIT_SEED = 0
HELDOUT_FRACTION = 0.20
HIT_QUANTILE = 0.95
KEY = ["drug_a", "drug_b", "cellosaurus_ac"]
OUTCOME_COLS = ["y_syn", "y_inh", "css_mean", "css_sd", "bliss_mean", "bliss_sd", "zip_mean", "zip_sd",
                "loewe_mean", "loewe_sd", "hsa_mean", "hsa_sd"]


class FreezeError(RuntimeError):
    """An acceptance check failed."""


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class Result:
    tables: dict[str, pd.DataFrame]
    splits: dict[str, Any]
    card: str
    report: dict[str, Any]
    checks: list[Check] = field(default_factory=list)


# -- split ----------------------------------------------------------------------------------------


def eligible(cells: pd.DataFrame) -> pd.Series:
    """True where every feature of the line is its own measurement (decision 19)."""
    own = ((cells.expression_source == "cellminer") & (cells.mutation_source == "depmap")
           & ~cells.depmap_is_proxy.astype(bool))
    fam_ok = own.groupby(cells.family_id).transform("all")
    return own & fam_ok


def quotas(lineage_sizes: pd.Series, fraction: float = HELDOUT_FRACTION) -> pd.Series:
    """Largest-remainder apportionment of round(fraction * total) held-out lines over lineages."""
    total = int(round(fraction * lineage_sizes.sum()))
    exact = lineage_sizes.sort_index() * fraction
    q = np.floor(exact).astype(int)
    rem = (exact - q).sort_values(ascending=False, kind="stable")  # stable: ties stay alphabetical
    for lin in rem.index[: total - int(q.sum())]:
        q[lin] += 1
    return q


def choose_heldout(cells: pd.DataFrame, seed: int = SPLIT_SEED,
                   fraction: float = HELDOUT_FRACTION) -> tuple[list[str], dict[str, Any]]:
    """Held-out cellosaurus_ac list, plus a per-lineage record of how it was chosen."""
    cells = cells.sort_values("cellosaurus_ac").assign(eligible=eligible(cells.sort_values("cellosaurus_ac")))
    q = quotas(cells.groupby("lineage").size(), fraction)
    rng = np.random.default_rng(seed)
    chosen: list[str] = []
    record: dict[str, Any] = {}
    for lin in q.index:  # alphabetical, so the RNG stream is consumed in a fixed order
        sub = cells[cells.lineage == lin]
        fams = sorted(sub[sub.eligible].family_id.unique())
        order = list(rng.permutation(len(fams))) if fams else []
        need, picked = int(q[lin]), []
        for i in order:
            members = sorted(sub[sub.family_id == fams[i]].cellosaurus_ac)
            if len(members) <= need - len(picked):
                picked += members
            if len(picked) == need:
                break
        chosen += picked
        record[lin] = {"lines": int(len(sub)), "eligible": int(sub.eligible.sum()), "quota": int(q[lin]),
                       "heldout": sorted(sub.set_index("cellosaurus_ac").loc[picked, "cellosaurus_name"]),
                       "ineligible": sorted(sub.loc[~sub.eligible, "cellosaurus_name"])}
        if len(picked) != need:
            raise FreezeError(f"lineage {lin}: quota {need} can't be met from eligible families")
    return sorted(chosen), record


# -- freeze ---------------------------------------------------------------------------------------


def target_status(drugs: pd.DataFrame) -> pd.Series:
    """`none_in_chembl` (decision 8), `chembl_proxy` (mechanism taken from another record), or `chembl`."""
    proxy = drugs.mechanism_source.notna() & (drugs.mechanism_source != drugs.chembl_parent_id)
    return pd.Series(np.select([drugs.n_targets.fillna(0) == 0, proxy], ["none_in_chembl", "chembl_proxy"],
                               "chembl"), index=drugs.index)


def _fmt(x: float, nd: int = 2) -> str:
    return f"{x:,.{nd}f}"


def build(experiments: pd.DataFrame, self_pairs: pd.DataFrame, cells: pd.DataFrame,
          cell_features: pd.DataFrame, drug_features: pd.DataFrame, drugs: pd.DataFrame,
          seed: int = SPLIT_SEED, upstream: dict[str, Any] | None = None) -> Result:
    """`upstream` carries numbers from the 1e and 1f reports that the data card cites."""
    upstream = upstream or {}
    checks: list[Check] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append(Check(name, bool(ok), detail))

    cells = cells.sort_values("cellosaurus_ac", ignore_index=True)
    heldout, record = choose_heldout(cells, seed)
    cells = cells.assign(eligible_for_heldout=eligible(cells).values,
                         split=np.where(cells.cellosaurus_ac.isin(heldout), "heldout", "selectable"))
    split_of = cells.set_index("cellosaurus_ac").split

    e = experiments.sort_values(KEY, ignore_index=True)
    e["split"] = e.cellosaurus_ac.map(split_of)
    e["lineage"] = e.cellosaurus_ac.map(cells.set_index("cellosaurus_ac").lineage)
    sel = e.split == "selectable"
    t_syn = float(e.loc[sel, "y_syn"].quantile(HIT_QUANTILE))
    t_inh = float(e.loc[sel, "y_inh"].quantile(HIT_QUANTILE))
    e["is_hit_syn"] = (e.y_syn >= t_syn).astype("int8")
    e["is_hit_inh"] = (e.y_inh >= t_inh).astype("int8")
    e["syn_hit_low_kill"] = ((e.is_hit_syn == 1) & (e.y_inh < 0)).astype("int8")

    pool = e[KEY + ["split", "lineage", "depmap_id", "depmap_is_proxy", "family_id", "study_id", "n_reps"]]
    labels = e[KEY + OUTCOME_COLS + ["is_hit_syn", "is_hit_inh", "syn_hit_low_kill"]]
    cells_out = cells.merge(cell_features, on="cellosaurus_ac", how="left", suffixes=("", "_feat"))
    cells_out = cells_out.loc[:, ~cells_out.columns.str.endswith("_feat")]
    d = drugs[["drug_id", "chembl_parent_id", "n_targets", "mechanism_source"]].assign(
        target_status=lambda x: target_status(x))
    drugs_out = drug_features.merge(d[["drug_id", "target_status"]], on="drug_id", how="left")
    drugs_out = drugs_out[["drug_id", "preferred_name", "target_status"]
                          + [c for c in drugs_out.columns if c not in ("drug_id", "preferred_name", "target_status")]]

    # -- acceptance checks (design §4, gap analysis §2.7, §2.11-2.13) --
    check("no duplicate experiments", not pool.duplicated(KEY).any())
    check("pool has no outcome columns", not set(OUTCOME_COLS) & set(pool.columns))
    check("labels keys equal pool keys", labels[KEY].equals(pool[KEY]))
    fam_split = cells.groupby("family_id").split.nunique()
    check("every family is on one side of the split", (fam_split == 1).all(),
          ", ".join(fam_split[fam_split > 1].index))
    check("no ineligible line is held out", not cells.loc[cells.split == "heldout", "eligible_for_heldout"].eq(False).any())
    check("held-out size is round(20% of lines)", len(heldout) == int(round(HELDOUT_FRACTION * len(cells))))
    check("every pool line has a DepMap ID and lineage", cells[["depmap_id", "lineage"]].notna().all().all())
    pool_drugs = set(pool.drug_a) | set(pool.drug_b)
    check("every pool drug has features", pool_drugs <= set(drugs_out.drug_id))
    check("every pool line has features", set(pool.cellosaurus_ac) <= set(cells_out.cellosaurus_ac))
    check("thresholds come from selectable experiments only", sel.sum() < len(e))
    check("hit rate among selectable experiments is about 5%",
          abs(e.loc[sel, "is_hit_syn"].mean() - (1 - HIT_QUANTILE)) < 0.005)
    # A pair measured only on held-out lines can never be selected. That's expected for pairs tested on
    # one or two lines (1 pair on 2026-11-08), but a split that orphaned many pairs would be a bad split.
    n_pairs = e.groupby(["drug_a", "drug_b"]).ngroups
    orphan_pairs = n_pairs - e[sel].groupby(["drug_a", "drug_b"]).ngroups
    check("at most 1% of drug pairs are measured only on held-out lines", orphan_pairs <= 0.01 * n_pairs,
          f"{orphan_pairs} of {n_pairs}")

    # -- analysis for the data card --
    by = e.groupby("split")
    split_stats = pd.DataFrame({
        "lines": cells.groupby("split").size(),
        "experiments": by.size(),
        "syn_hit_rate": by.is_hit_syn.mean(),
        "inh_hit_rate": by.is_hit_inh.mean(),
        "mean_y_syn": by.y_syn.mean(),
        "mean_y_inh": by.y_inh.mean(),
    }).reindex(["selectable", "heldout"])
    mut_cols = [c for c in cell_features.columns if c.startswith("mut_")]
    mf = cells_out.set_index("cellosaurus_ac")
    testable = {c[4:]: {"selectable": int(mf.loc[mf.split == "selectable", c].sum()),
                        "heldout": int(mf.loc[mf.split == "heldout", c].sum())} for c in mut_cols}
    hits = e[e.is_hit_syn == 1]
    low = hits.syn_hit_low_kill.mean()
    name = dict(zip(drugs_out.drug_id, drugs_out.preferred_name))
    low_drugs = (pd.concat([hits.loc[hits.syn_hit_low_kill == 1, "drug_a"], hits.loc[hits.syn_hit_low_kill == 1, "drug_b"]])
                 .value_counts().head(5))
    rep2 = e[e.n_reps == 2]
    splits = {
        "seed": seed, "heldout_fraction": HELDOUT_FRACTION, "hit_quantile": HIT_QUANTILE,
        "heldout_lines": sorted(cells.loc[cells.split == "heldout", "cellosaurus_ac"]),
        "heldout_names": sorted(cells.loc[cells.split == "heldout", "cellosaurus_name"]),
        "ineligible_lines": sorted(cells.loc[~cells.eligible_for_heldout, "cellosaurus_name"]),
        "per_lineage": record,
        "thresholds": {"y_syn": round(t_syn, 4), "y_inh": round(t_inh, 4),
                       "computed_on": "selectable experiments", "n_selectable": int(sel.sum())},
        "not_in_m1": "seed sets and famous/obscure labels are M2 (design §5.2-5.3, gap analysis §2.11)",
    }
    report = {
        "split_stats": {k: {c: (round(float(v), 4) if not isinstance(v, (int, np.integer)) else int(v))
                            for c, v in row.items()} for k, row in split_stats.iterrows()},
        "mutant_lines_per_gene": testable,
        "syn_hit_low_kill": {"count": int(hits.syn_hit_low_kill.sum()), "fraction_of_syn_hits": round(float(low), 4),
                             "top_drugs": {name[k]: int(v) for k, v in low_drugs.items()}},
        "pairs_only_on_heldout_lines": int(orphan_pairs),
        "replicate_sd_median": {"bliss": round(float(rep2.bliss_sd.median()), 3),
                                "css": round(float(rep2.css_sd.median()), 3)},
        "checks": [{"name": c.name, "ok": c.ok, "detail": c.detail} for c in checks],
    }
    report["upstream"] = upstream
    card = data_card(e, cells, drugs_out, self_pairs, splits, report, split_stats)
    if not all(c.ok for c in checks):
        raise FreezeError("; ".join(f"{c.name}: {c.detail}" for c in checks if not c.ok))
    tables = {"pool": pool, "labels": labels, "cells": cells_out, "drugs": drugs_out,
              "self_pairs": self_pairs.sort_values(KEY, ignore_index=True)}
    return Result(tables, splits, card, report, checks)


def data_card(e: pd.DataFrame, cells: pd.DataFrame, drugs: pd.DataFrame, self_pairs: pd.DataFrame,
              splits: dict, report: dict, split_stats: pd.DataFrame) -> str:
    """Deterministic markdown: fixed float precision, sorted keys, no timestamps or paths."""
    t = splits["thresholds"]
    L = ["# Data card: NCI-ALMANAC drug-combination pool (M1 freeze)", "",
         "Generated by `combo_agent.data.freeze_pool`. Every number below is recomputed from the raw "
         "inputs pinned in `manifest.json`; rerunning gives a byte-identical file.", "",
         "## Contents", "",
         "| Item | Count |", "|:--|--:|",
         f"| Experiments (drug A, drug B, cell line) | {len(e):,} |",
         f"| Drug pairs | {e.groupby(['drug_a', 'drug_b']).ngroups:,} |",
         f"| Drugs in the pool | {len(set(e.drug_a) | set(e.drug_b))} |",
         f"| Cell lines | {len(cells)} |",
         f"| Experiments with 2 replicates | {int((e.n_reps == 2).sum()):,} |",
         f"| Self-pair experiments (separate table) | {len(self_pairs)} |",
         "| Study | NCI-ALMANAC (one study; O'Neil and a third study removed in 1e) |", "",
         "## Outcomes", "",
         "| Outcome | Column | Meaning | Hit threshold (p95 of selectable) |", "|:--|:--|:--|--:|",
         f"| Synergy | `y_syn` | mean Bliss score (above 0 = more than independent action) | {_fmt(t['y_syn'])} |",
         f"| Killing | `y_inh` | mean CSS (drug combination sensitivity score) | {_fmt(t['y_inh'])} |", "",
         f"Spearman(y_syn, y_inh) = {e[['y_syn', 'y_inh']].corr('spearman').iloc[0, 1]:.3f}: the two outcomes are "
         "nearly unrelated, so they are tracked separately.", "",
         f"Median replicate SD (2-replicate experiments): Bliss {report['replicate_sd_median']['bliss']:.3f}, "
         f"CSS {report['replicate_sd_median']['css']:.3f}.", "",
         "## Held-out cell lines", "",
         f"{len(splits['heldout_lines'])} of {len(cells)} lines ({len(splits['heldout_lines']) / len(cells):.1%}), "
         f"seed {splits['seed']}, stratified by lineage, families kept together. Lines with imputed or "
         "borrowed features are not eligible (decision 19).", "",
         "| Lineage | Lines | Eligible | Held out | Held-out lines | Ineligible |", "|:--|--:|--:|--:|:--|:--|"]
    for lin, r in splits["per_lineage"].items():
        L.append(f"| {lin} | {r['lines']} | {r['eligible']} | {r['quota']} | {', '.join(r['heldout']) or '-'} | "
                 f"{', '.join(r['ineligible']) or '-'} |")
    L += ["", "| Split | Lines | Experiments | Synergy hit rate | Killing hit rate | Mean y_syn | Mean y_inh |",
          "|:--|--:|--:|--:|--:|--:|--:|"]
    for s, r in split_stats.iterrows():
        L.append(f"| {s} | {int(r.lines)} | {int(r.experiments):,} | {r.syn_hit_rate:.2%} | {r.inh_hit_rate:.2%} | "
                 f"{r.mean_y_syn:.2f} | {r.mean_y_inh:.2f} |")
    L += ["", "Mutant lines per driver gene (how testable a mutation-based prediction is):", "",
          "| Gene | Selectable | Held out |", "|:--|--:|--:|"]
    for g, v in report["mutant_lines_per_gene"].items():
        if v["selectable"] + v["heldout"]:
            L.append(f"| {g} | {v['selectable']} | {v['heldout']} |")
    lk = report["syn_hit_low_kill"]
    L += ["", "## Caveats", "",
          f"- **Synergy hits with negative killing** (`syn_hit_low_kill`): {lk['count']:,} "
          f"({lk['fraction_of_syn_hits']:.1%} of synergy hits). Bliss can be large when the pair barely kills or "
          "a single drug stimulates growth. Most frequent drugs: "
          + ", ".join(f"{k} ({v})" for k, v in lk["top_drugs"].items()) + ". Kept as hits (design definition); flagged.",
          "- **SR expression is imputed** (`expression_imputed`): CellMiner's SR RNA-seq sample is a mix-up "
          "(it matches RPMI-8226). SR's expression is translated from CellMiner's own microarrays "
          f"(leave-one-out deviation r = {report['upstream'].get('imputation_r', float('nan')):.2f}, from the 1f report). "
          "SR-specific genes are compressed (ALK). SR is never held out.",
          "- **Borrowed mutation calls**: SNB-19 (from U-251 MG), NCI-H322M (from NCI-H322), "
          "HL-60(TB) (from HL-60, also its DepMap proxy). None is held out.",
          "- **Drug targets**: " + ", ".join(f"{k} {v}" for k, v in
                                               drugs[drugs.in_pool == 1].target_status.value_counts().sort_index().items())
          + " (pool drugs). `none_in_chembl` drugs are kept (decision 8).",
          "- **Bliss is noisy**: between the two stored copies of one experiment, Pearson r is "
          f"{report['upstream'].get('replicate_r_bliss', float('nan')):.3f} for Bliss and "
          f"{report['upstream'].get('replicate_r_css', float('nan')):.3f} for CSS (1e report).",
          f"- **Pairs measured only on held-out lines**: {report['pairs_only_on_heldout_lines']} (never selectable).", ""]
    return "\n".join(L)


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def write_outputs(result: Result, out_dir: Path, inputs: dict[str, Path]) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, t in result.tables.items():
        out[name] = out_dir / f"{name}.parquet"
        t.to_parquet(out[name], index=False)
    out["splits"] = out_dir / "splits.json"
    out["splits"].write_text(json.dumps(result.splits, indent=2, sort_keys=True) + "\n")
    out["data_card"] = out_dir / "data_card.md"
    out["data_card"].write_text(result.card)
    out["freeze_report"] = out_dir / "freeze_report.json"
    out["freeze_report"].write_text(json.dumps(result.report, indent=2, sort_keys=True) + "\n")
    manifest = {
        "inputs": {k: {"path": str(p.relative_to(paths.REPO_ROOT)) if p.is_relative_to(paths.REPO_ROOT) else p.name,
                       "sha256": _sha(p)} for k, p in sorted(inputs.items())},
        "outputs": {k: {"file": p.name, "sha256": _sha(p)} for k, p in sorted(out.items())},
        "versions": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__},
        "raw_manifests": {name: json.loads(p.read_text()) for name, p in
                          (("tdc", paths.raw_dir() / "manifest.json"),
                           ("cellminer", paths.raw_dir() / "cellminer" / "manifest.json")) if p.exists()},
    }
    out["manifest"] = out_dir / "manifest.json"
    out["manifest"].write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Step 1g: split, thresholds, freeze")
    p.add_argument("--out", type=Path, default=None, help="output dir (default data_cache/processed)")
    args = p.parse_args(argv)
    i, f = paths.interim_dir(), paths.interim_dir() / "features"
    inputs = {"experiments": i / "experiments.parquet", "self_pairs": i / "self_pairs.parquet",
              "cells": f / "cells.parquet", "cell_features": f / "cell_features.parquet",
              "drug_features": f / "drug_features.parquet", "drugs": i / "drugs.parquet"}
    inputs["experiments_report"] = i / "experiments_report.json"
    inputs["features_report"] = f / "features_report.json"
    er = json.loads(inputs["experiments_report"].read_text())
    fr = json.loads(inputs["features_report"].read_text())
    upstream = {"replicate_r_bliss": er["replicate_agreement_pearson_n2"]["bliss"],
                "replicate_r_css": er["replicate_agreement_pearson_n2"]["css"],
                "imputation_r": (fr["expression"]["imputation_loo_quality"] or {}).get("deviation_r_median",
                                                                                      float("nan"))}
    r = build(*(pd.read_parquet(inputs[k]) for k in
                ("experiments", "self_pairs", "cells", "cell_features", "drug_features", "drugs")),
              upstream=upstream)
    out = write_outputs(r, args.out or paths.processed_dir(), inputs)
    print("held out:", r.splits["heldout_names"])
    print("thresholds:", r.splits["thresholds"])
    print("split stats:", r.report["split_stats"])
    for k, v in out.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
