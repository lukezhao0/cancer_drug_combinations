"""Metrics and statistics for a grid of campaigns (target_design.md §11).

Input: a runner output directory (`<out>/runs/*.parquet`). No hidden outcome is read; hits come
from the revealed values the runner stored.

Per campaign (its own target's hits unless noted; "charged" = experiments the selector paid for,
the free 32-experiment seed set excluded):

    hits_at_80/160/320     hits among the first 80/160/320 charged experiments
    auc                    area under cumulative hits vs charged experiments, divided by the best
                           possible area (every experiment a hit)
    exp_to_10              charged experiments until 10 hits (censored at the budget; flag kept)
    hit_pairs_at_320       distinct drug pairs among the hits (diversity, gap analysis §4.4)
    cross_hits_at_320      hits of the *other* outcome (syn campaigns: killing hits, and reverse)
    syn campaigns also report the strict count (synergy hits with killing >= 0; M1 decision 20)
    hits_<fame>            hits at 320 on famous / middle / obscure pairs (M2 labels)

Across seeds (paired by seed):
    mean and bootstrap 95% CI; enrichment = mean / random's mean; one-sided Wilcoxon signed-rank
    p for each selector vs random and vs bo_ucb on hits at 320; Holm correction across the
    "vs random" comparisons of each target.

**M4 gate** (design §12): bo_ucb beats random on hits at 320. Verdict per target: mean higher
and Holm-adjusted p < 0.05.

Run: python -m combo_agent.eval.metrics runs/m4_main
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from combo_agent.data import paths

BUDGET_POINTS = (80, 160, 320)
FAME = ("famous", "middle", "obscure")


def load(out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    files = sorted((out / "runs").glob("*.parquet"))
    rounds = [pd.read_parquet(f) for f in files if not f.name.endswith((".reveals.parquet", ".preds.parquet"))]
    if not rounds:
        raise FileNotFoundError(f"no finished campaigns in {out / 'runs'}")
    done = {r["run_id"].iloc[0] for r in rounds}
    reveals = [pd.read_parquet(f) for f in files if f.name.endswith(".reveals.parquet")
               and f.name.removesuffix(".reveals.parquet") in done]
    return pd.concat(rounds, ignore_index=True), pd.concat(reveals, ignore_index=True)


def hits_at(curve: pd.DataFrame, k: int, col: str) -> float:
    """Hits after k charged experiments: the last round with n_charged <= k."""
    upto = curve[curve["n_charged"] <= k]
    return float(upto[col].iloc[-1]) if len(upto) else 0.0


def auc(curve: pd.DataFrame, col: str) -> float:
    """Cumulative hits integrated over charged experiments, over the best possible (all hits)."""
    c = curve.sort_values("n_charged")
    dn = np.diff(c["n_charged"].to_numpy(), prepend=0)
    best = float((c["n_charged"].to_numpy() * dn).sum())
    return float((c[col].to_numpy() * dn).sum()) / best if best > 0 else float("nan")


def exp_to(curve: pd.DataFrame, col: str, n_hits: int = 10) -> tuple[float, bool]:
    reached = curve[curve[col] >= n_hits]
    if len(reached):
        return float(reached["n_charged"].iloc[0]), False
    return float(curve["n_charged"].max()), True


def per_run(rounds: pd.DataFrame, reveals: pd.DataFrame, fame: pd.DataFrame | None) -> pd.DataFrame:
    fame_of = None if fame is None else {(a, b): f for a, b, f in fame[["drug_a", "drug_b", "fame"]].itertuples(index=False)}
    rows = []
    for run_id, c in rounds.groupby("run_id", sort=True):
        c = c.sort_values("round")
        sel, target, seed = c["selector"].iloc[0], c["target"].iloc[0], int(c["seed"].iloc[0])
        col, other = f"hits_{target}", "hits_inh" if target == "syn" else "hits_syn"
        e10, censored = exp_to(c, col)
        row = {"run_id": run_id, "selector": sel, "target": target, "seed": seed,
               **{f"hits_at_{k}": hits_at(c, k, col) for k in BUDGET_POINTS},
               "auc": auc(c, col), "exp_to_10": e10, "exp_to_10_censored": censored,
               "hit_pairs_at_320": hits_at(c, 320, f"hit_pairs_{target}"),
               "cross_hits_at_320": hits_at(c, 320, other),
               "seed_hits": float(c[f"seed_hits_{target}"].iloc[0])}
        if target == "syn":
            row["strict_hits_at_320"] = hits_at(c, 320, "hits_syn_strict")
        if fame_of is not None:
            rv = reveals[(reveals["run_id"] == run_id) & (reveals["round"] > 0) & (reveals[f"hit_{target}"])]
            classes = [fame_of.get((a, b), "unknown") for a, b in zip(rv["drug_a"], rv["drug_b"])]
            for f in FAME:
                row[f"hits_{f}"] = float(sum(x == f for x in classes))
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_ci(x: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n, len(x)))].mean(1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def paired_p(a: np.ndarray, b: np.ndarray) -> float:
    """One-sided Wilcoxon signed-rank p that a > b (paired). 1.0 when every difference is zero."""
    d = np.asarray(a, float) - np.asarray(b, float)
    if np.all(d == 0) or len(d) < 2:
        return 1.0
    return float(wilcoxon(a, b, alternative="greater", zero_method="zsplit").pvalue)


def holm(p: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values."""
    order = sorted(p, key=p.get)
    m, out, running = len(order), {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * p[k]))
        out[k] = running
    return out


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = [c for c in runs.columns if c.startswith(("hits_", "auc", "exp_to_10", "hit_pairs", "cross_hits",
                                                        "strict_hits", "seed_hits")) and c != "exp_to_10_censored"]
    for (target, sel), g in runs.groupby(["target", "selector"], sort=True):
        g = g.sort_values("seed")
        base = runs[(runs.target == target) & (runs.selector == "random")].set_index("seed")
        ucb = runs[(runs.target == target) & (runs.selector == "bo_ucb")].set_index("seed")
        row = {"target": target, "selector": sel, "n_seeds": len(g)}
        for m in metrics:
            if g[m].isna().all():
                continue
            lo, hi = bootstrap_ci(g[m].to_numpy())
            row.update({f"{m}_mean": float(g[m].mean()), f"{m}_lo": lo, f"{m}_hi": hi})
        x = g.set_index("seed")["hits_at_320"]
        if len(base):
            s = x.index.intersection(base.index)
            row["enrichment_vs_random"] = float(x[s].mean() / base.loc[s, "hits_at_320"].mean()) if base.loc[s, "hits_at_320"].mean() > 0 else float("nan")
            row["p_vs_random"] = paired_p(x[s].to_numpy(), base.loc[s, "hits_at_320"].to_numpy()) if sel != "random" else float("nan")
        if len(ucb) and sel != "bo_ucb":
            s = x.index.intersection(ucb.index)
            row["p_vs_bo_ucb"] = paired_p(x[s].to_numpy(), ucb.loc[s, "hits_at_320"].to_numpy())
        rows.append(row)
    out = pd.DataFrame(rows)
    out["p_vs_random_holm"] = np.nan
    for target, g in out.groupby("target"):
        ps = {i: p for i, p in g["p_vs_random"].items() if pd.notna(p)}
        for i, p in holm(ps).items():
            out.loc[i, "p_vs_random_holm"] = p
    return out


def gate(summary: pd.DataFrame) -> dict[str, dict]:
    out = {}
    for target in sorted(summary["target"].unique()):
        s = summary[summary.target == target].set_index("selector")
        if not {"bo_ucb", "random"} <= set(s.index):
            continue
        u, r = s.loc["bo_ucb"], s.loc["random"]
        passed = bool(u["hits_at_320_mean"] > r["hits_at_320_mean"] and u["p_vs_random_holm"] < 0.05)
        out[target] = {"bo_ucb": float(u["hits_at_320_mean"]), "random": float(r["hits_at_320_mean"]),
                       "p_holm": float(u["p_vs_random_holm"]), "pass": passed}
    return out


def report(summary: pd.DataFrame, verdict: dict, n_runs: int) -> str:
    L = ["# M4 report: baseline discovery campaigns", "",
         "Generated by `combo_agent.eval.metrics`. Hits are counted among the 320 experiments each selector paid for "
         "(the free 32-experiment seed set is excluded). Mean over seeds with a bootstrap 95% CI.", "",
         f"Campaigns: {n_runs}.", "", "## Gate: bo_ucb beats random on hits at 320", "",
         "| Target | bo_ucb | random | Holm p (one-sided Wilcoxon) | Verdict |", "|:--|--:|--:|--:|:--|"]
    for t, v in verdict.items():
        L.append(f"| {t} | {v['bo_ucb']:.1f} | {v['random']:.1f} | {v['p_holm']:.3g} | {'PASS' if v['pass'] else 'FAIL'} |")
    for target in sorted(summary["target"].unique()):
        s = summary[summary.target == target]
        L += ["", f"## Target `{target}`", "",
              "| Selector | Seeds | Hits @80 | Hits @160 | Hits @320 (95% CI) | Enrichment vs random | AUC | Exp. to 10 hits | Hit pairs @320 | Cross-outcome hits @320 |"
              + (" Strict syn hits @320 |" if target == "syn" else ""),
              "|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|" + ("--:|" if target == "syn" else "")]
        for _, r in s.sort_values("hits_at_320_mean", ascending=False).iterrows():
            line = (f"| {r.selector} | {r.n_seeds} | {r.hits_at_80_mean:.1f} | {r.hits_at_160_mean:.1f} | "
                    f"{r.hits_at_320_mean:.1f} ({r.hits_at_320_lo:.1f}–{r.hits_at_320_hi:.1f}) | "
                    f"{r.get('enrichment_vs_random', float('nan')):.2f} | {r.auc_mean:.3f} | {r.exp_to_10_mean:.0f} | "
                    f"{r.hit_pairs_at_320_mean:.1f} | {r.cross_hits_at_320_mean:.1f} |")
            if target == "syn":
                line += f" {r.strict_hits_at_320_mean:.1f} |"
            L.append(line)
        if "hits_famous_mean" in s.columns:
            L += ["", "Hits at 320 by pair fame (M2):", "", "| Selector | Famous | Middle | Obscure |", "|:--|--:|--:|--:|"]
            for _, r in s.iterrows():
                L.append(f"| {r.selector} | {r.hits_famous_mean:.1f} | {r.hits_middle_mean:.1f} | {r.hits_obscure_mean:.1f} |")
    L += ["", "Random picks hit at about the base rate (5% of 320 = 16 expected).", ""]
    return "\n".join(L)


def build(out: Path, fame_path: Path | None = None) -> dict[str, Path]:
    rounds, reveals = load(out)
    fame_path = fame_path or paths.processed_dir() / "m2" / "pair_fame.parquet"
    fame = pd.read_parquet(fame_path, columns=["drug_a", "drug_b", "fame"]) if fame_path.exists() else None
    runs = per_run(rounds, reveals, fame)
    summary = summarize(runs)
    verdict = gate(summary)
    files = {"results": out / "results.parquet", "runs_summary": out / "per_run.parquet",
             "summary": out / "summary.parquet", "report": out / "report.md"}
    rounds.to_parquet(files["results"], index=False)
    runs.to_parquet(files["runs_summary"], index=False)
    summary.to_parquet(files["summary"], index=False)
    files["report"].write_text(report(summary, verdict, len(runs)))
    return files


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out", type=Path)
    args = ap.parse_args(argv)
    for p in build(args.out).values():
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
