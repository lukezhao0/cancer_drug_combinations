"""Surrogate comparison: the ensemble vs the Gaussian process on the same campaigns.

Evaluation code (on the label allowlist): it reads the hidden outcomes to score top-k recall and
the calibration of predictions on the random probe set. Nothing here is imported by a selector or
a model.

Inputs: two runner output directories made with the same grid, one per surrogate, plus an
optional directory holding `random` campaigns (the enrichment baseline):

    python -m combo_agent.eval.compare --ensemble runs/gp_compare/ensemble --gp runs/gp_compare/gp \
        --random runs/m4_main --out runs/gp_compare/comparison

Metrics per campaign (surrogate x selector x target x seed), for the campaign's own target:

    best value       highest outcome revealed so far, per round (seed set included)
    recall top 1/5%  share of the true top 1% / 5% of selectable experiments revealed, per round
    hits @320        paid-for hits (metrics.per_run), and enrichment = hits / random's mean hits
    calibration      share of outcomes within +-1 and +-2 predictive std (targets: 68% and 95%),
                     and mean Gaussian negative log-likelihood (NLL), on each round's batch,
                     scored with the predictions made before the batch was run (`batch`), and on
                     the fixed random probe set (`probe`, unbiased: batches are extreme by design)
    std vs distance  Spearman correlation of latent std with nearest-history distance (probe rows),
                     plus binned means
    fit time         mean refit and prediction seconds per round

Paired comparison per (selector, target, metric): GP - ensemble difference per seed, its mean,
bootstrap 95% CI and seed SD, and a two-sided Wilcoxon signed-rank p. With 5 seeds the smallest
possible two-sided p is 0.0625, so the verdict uses the CI and the size of the gap relative to the
seed-to-seed spread instead.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from combo_agent.data import paths
from combo_agent.eval import metrics as mx
from combo_agent.lab.oracle import KEY

LABELS_FILE = "labels.parquet"
TOP = (0.01, 0.05)
DIST_BINS = (0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001)
#: metric -> True if larger is better (None: closer to a target value is better)
DIRECTION = {
    "hits_at_320": True, "auc": True, "best_at_320": True, "recall_top1_at_320": True,
    "recall_top5_at_320": True, "enrichment": True,
    "batch_cover1": None, "batch_cover2": None, "batch_nll": False,
    "probe_cover1": None, "probe_cover2": None, "probe_nll": False,
    "std_dist_spearman": True, "fit_s_mean": False, "predict_s_mean": False,
}
COVER_TARGET = {"batch_cover1": 0.6827, "batch_cover2": 0.9545, "probe_cover1": 0.6827, "probe_cover2": 0.9545}


# -- inputs --------------------------------------------------------------------------------------

def truth(root: Path | None = None) -> pd.DataFrame:
    """Selectable experiments with their outcomes and top-k flags per target."""
    root = root or paths.processed_dir()
    pool = pd.read_parquet(root / "pool.parquet", columns=[*KEY, "split"])
    lab = pd.read_parquet(root / LABELS_FILE, columns=[*KEY, "y_syn", "y_inh"])
    if not (pool[KEY].to_numpy() == lab[KEY].to_numpy()).all():
        raise ValueError("pool and labels rows are not aligned")
    t = lab[pool["split"].to_numpy() == "selectable"].reset_index(drop=True)
    for target in ("syn", "inh"):
        for q in TOP:
            cut = np.quantile(t[f"y_{target}"], 1 - q)
            t[f"top{round(q * 100)}_{target}"] = t[f"y_{target}"] >= cut
    return t


def load_dir(out: Path, surrogate: str | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rounds, reveals = mx.load(out)
    pf = sorted((out / "runs").glob("*.preds.parquet"))
    preds = pd.concat([pd.read_parquet(f) for f in pf], ignore_index=True) if pf else pd.DataFrame()
    done = set(rounds["run_id"])
    if len(preds):
        preds = preds[preds["run_id"].isin(done)]
    if surrogate:
        for df in (rounds, reveals, preds):
            if len(df):
                df["surrogate"] = surrogate
    return rounds, reveals, preds


# -- per-run metrics ----------------------------------------------------------------------------

def _nll(y, mu, sd):
    sd = np.maximum(sd, 1e-9)
    return 0.5 * np.log(2 * math.pi * sd ** 2) + 0.5 * ((y - mu) / sd) ** 2


def calibration(df: pd.DataFrame, target: str, ycol: str) -> dict:
    y, mu, sd = df[ycol].to_numpy(), df[f"mean_{target}"].to_numpy(), df[f"std_{target}"].to_numpy()
    z = np.abs(y - mu) / np.maximum(sd, 1e-9)
    return {"cover1": float(np.mean(z <= 1)), "cover2": float(np.mean(z <= 2)), "nll": float(np.mean(_nll(y, mu, sd))),
            "n": int(len(df))}


def curves(reveals: pd.DataFrame, rounds: pd.DataFrame, t: pd.DataFrame) -> pd.DataFrame:
    """Per run and round: best value, top-1%/5% recall, cumulative experiments."""
    meta = rounds.drop_duplicates("run_id").set_index("run_id")[["selector", "target", "seed", "surrogate"]]
    rows = []
    for run_id, rv in reveals.groupby("run_id", sort=True):
        target = meta.loc[run_id, "target"]
        rv = rv.merge(t[[*KEY, f"top1_{target}", f"top5_{target}"]], on=KEY, how="left")
        n1, n5 = int(t[f"top1_{target}"].sum()), int(t[f"top5_{target}"].sum())
        for rnd in sorted(rv["round"].unique()):
            upto = rv[rv["round"] <= rnd]
            rows.append({"run_id": run_id, **meta.loc[run_id].to_dict(), "round": int(rnd),
                         "n_revealed": len(upto), "n_charged": int((upto["round"] > 0).sum()),
                         "best": float(upto[f"y_{target}"].max()),
                         "recall_top1": float(upto[f"top1_{target}"].sum() / n1),
                         "recall_top5": float(upto[f"top5_{target}"].sum() / n5)})
    return pd.DataFrame(rows)


def per_run(rounds, reveals, preds, t: pd.DataFrame, random_hits: dict) -> tuple[pd.DataFrame, ...]:
    """(per-run metrics, per-round calibration, binned std-vs-distance, per-round curves)."""
    base = mx.per_run(rounds, reveals, None).set_index("run_id")
    cur = curves(reveals, rounds, t)
    last = cur.sort_values("round").groupby("run_id").last()
    timing = rounds[rounds["round"] > 0].groupby("run_id")[["fit_s", "predict_s"]].mean()
    surrogate = rounds.drop_duplicates("run_id").set_index("run_id")["surrogate"]
    truth_y = t[[*KEY, "y_syn", "y_inh"]]
    rows, cal_rows, bin_rows = [], [], []
    for run_id, b in base.iterrows():
        target = b["target"]
        row = {"run_id": run_id, "surrogate": surrogate[run_id], "selector": b["selector"], "target": target,
               "seed": int(b["seed"]), "hits_at_320": b["hits_at_320"], "auc": b["auc"],
               "best_at_320": last.loc[run_id, "best"], "recall_top1_at_320": last.loc[run_id, "recall_top1"],
               "recall_top5_at_320": last.loc[run_id, "recall_top5"],
               "fit_s_mean": timing.loc[run_id, "fit_s"], "predict_s_mean": timing.loc[run_id, "predict_s"]}
        rh = random_hits.get(target)
        row["enrichment"] = b["hits_at_320"] / rh if rh else np.nan
        p = preds[preds["run_id"] == run_id] if len(preds) else preds
        if len(p):
            batch = p[p["set"] == "batch"].merge(
                reveals[reveals["run_id"] == run_id][[*KEY, "round", f"y_{target}"]], on=[*KEY, "round"], how="left")
            if batch[f"y_{target}"].isna().any():
                raise ValueError(f"{run_id}: a batch prediction has no revealed outcome")
            probe = p[p["set"] == "probe"].merge(truth_y, on=KEY, how="left")
            for name, df in (("batch", batch), ("probe", probe)):
                c = calibration(df, target, f"y_{target}")
                row.update({f"{name}_cover1": c["cover1"], f"{name}_cover2": c["cover2"], f"{name}_nll": c["nll"]})
                for rnd, g in df.groupby("round"):
                    cal_rows.append({"run_id": run_id, "surrogate": row["surrogate"], "selector": row["selector"],
                                     "target": target, "seed": row["seed"], "set": name, "round": int(rnd),
                                     **calibration(g, target, f"y_{target}")})
            lat = probe[f"latent_std_{target}"].to_numpy()
            # Relative to each round's median, so the two surrogates' std scales are comparable.
            rel = lat / probe.groupby("round")[f"latent_std_{target}"].transform("median").to_numpy()
            row["std_dist_spearman"] = float(spearmanr(probe["dist"], lat).statistic)
            cut = pd.cut(probe["dist"], DIST_BINS, right=False)
            for interval, g in pd.DataFrame({"bin": cut, "rel": rel, "lat": lat}).groupby("bin", observed=True):
                bin_rows.append({"run_id": run_id, "surrogate": row["surrogate"], "selector": row["selector"],
                                 "target": target, "seed": row["seed"], "dist_lo": interval.left,
                                 "dist_hi": interval.right, "n": len(g), "latent_std_mean": float(g["lat"].mean()),
                                 "latent_std_rel_mean": float(g["rel"].mean())})
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(cal_rows), pd.DataFrame(bin_rows), cur


# -- paired comparison -------------------------------------------------------------------------

def paired(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (sel, target), g in runs.groupby(["selector", "target"], sort=True):
        e = g[g.surrogate == "ensemble"].set_index("seed")
        q = g[g.surrogate == "gp"].set_index("seed")
        seeds = e.index.intersection(q.index)
        for m, larger in DIRECTION.items():
            if m not in g or g[m].isna().all():
                continue
            a, b = e.loc[seeds, m].to_numpy(float), q.loc[seeds, m].to_numpy(float)
            if m in COVER_TARGET:  # compare distance to the nominal coverage
                a_s, b_s = -np.abs(a - COVER_TARGET[m]), -np.abs(b - COVER_TARGET[m])
            else:
                a_s, b_s = (a, b) if larger else (-a, -b)
            d = b - a
            better = b_s - a_s  # > 0: GP better
            lo, hi = mx.bootstrap_ci(better)
            sd = float(np.std(better, ddof=1)) if len(better) > 1 else float("nan")
            p = float(wilcoxon(b_s, a_s, zero_method="zsplit").pvalue) if len(d) > 1 and np.any(better != 0) else 1.0
            mean_better = float(np.mean(better))
            if lo > 0:
                verdict = "gp"
            elif hi < 0:
                verdict = "ensemble"
            else:
                verdict = "tie"
            rows.append({"selector": sel, "target": target, "metric": m, "n_seeds": len(seeds),
                         "ensemble_mean": float(a.mean()), "ensemble_sd": float(a.std(ddof=1)) if len(a) > 1 else np.nan,
                         "gp_mean": float(b.mean()), "gp_sd": float(b.std(ddof=1)) if len(b) > 1 else np.nan,
                         "gp_minus_ensemble": float(d.mean()), "gp_better_by": mean_better,
                         "gp_better_lo": lo, "gp_better_hi": hi, "gp_better_seed_sd": sd,
                         "effect_vs_spread": mean_better / sd if sd and sd > 0 else np.nan,
                         "seeds_gp_better": int((better > 0).sum()), "wilcoxon_p_two_sided": p, "winner": verdict})
    return pd.DataFrame(rows)


def check_complete(runs: pd.DataFrame, selectors, targets, seeds) -> None:
    missing = []
    for s in ("ensemble", "gp"):
        for sel in selectors:
            for t in targets:
                for seed in seeds:
                    r = runs[(runs.surrogate == s) & (runs.selector == sel) & (runs.target == t) & (runs.seed == seed)]
                    if len(r) != 1:
                        missing.append((s, sel, t, seed, "run"))
                        continue
                    for m in DIRECTION:
                        if m not in r or pd.isna(r[m].iloc[0]):
                            missing.append((s, sel, t, seed, m))
    if missing:
        raise ValueError(f"{len(missing)} missing metric values, e.g. {missing[:5]}")


def report(pr: pd.DataFrame) -> str:
    lines = ["# Surrogate comparison: GP vs ensemble", "",
             "Generated by `combo_agent.eval.compare`. Positive `GP better by` means the GP is better "
             "(for coverage: closer to the nominal 68% / 95%). Winner = sign of the bootstrap 95% CI of the "
             "per-seed difference; `tie` when it spans 0.", ""]
    for (target, metric), g in pr.groupby(["target", "metric"], sort=True):
        lines += [f"## {target}: {metric}", "",
                  "| Selector | Ensemble (SD) | GP (SD) | GP better by (95% CI) | Effect / seed SD | Seeds GP better | p (2-sided) | Winner |",
                  "|:--|--:|--:|--:|--:|--:|--:|:--|"]
        for _, r in g.iterrows():
            lines.append(f"| {r.selector} | {r.ensemble_mean:.3f} ({r.ensemble_sd:.3f}) | {r.gp_mean:.3f} ({r.gp_sd:.3f}) | "
                         f"{r.gp_better_by:+.3f} ({r.gp_better_lo:+.3f} to {r.gp_better_hi:+.3f}) | "
                         f"{r.effect_vs_spread:+.2f} | {r.seeds_gp_better}/{r.n_seeds} | {r.wilcoxon_p_two_sided:.3f} | {r.winner} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def build(ensemble: Path, gp: Path, out: Path, random_dir: Path | None = None, seeds=None) -> dict[str, Path]:
    t = truth()
    parts = [load_dir(ensemble, "ensemble"), load_dir(gp, "gp")]
    random_hits = {}
    if random_dir is not None:
        rr, rv, _ = load_dir(random_dir, "random")
        rruns = mx.per_run(rr[rr.selector == "random"], rv, None)
        if seeds is not None:
            rruns = rruns[rruns.seed.isin(seeds)]
        random_hits = rruns.groupby("target")["hits_at_320"].mean().to_dict()
    results = [per_run(r, v, p, t, random_hits) for r, v, p in parts]
    runs = pd.concat([x[0] for x in results], ignore_index=True)
    if seeds is not None:
        runs = runs[runs.seed.isin(seeds)]
    check_complete(runs, sorted(runs.selector.unique()), sorted(runs.target.unique()), sorted(runs.seed.unique()))
    pr = paired(runs)
    out.mkdir(parents=True, exist_ok=True)
    files = {"per_run": out / "per_run.csv", "paired": out / "paired.csv", "calibration": out / "calibration_by_round.csv",
             "std_vs_dist": out / "std_vs_distance.csv", "curves": out / "curves.csv", "report": out / "report.md",
             "random": out / "random_baseline.csv"}
    runs.to_csv(files["per_run"], index=False)
    pr.to_csv(files["paired"], index=False)
    pd.concat([x[1] for x in results], ignore_index=True).to_csv(files["calibration"], index=False)
    pd.concat([x[2] for x in results], ignore_index=True).to_csv(files["std_vs_dist"], index=False)
    pd.concat([x[3] for x in results], ignore_index=True).to_csv(files["curves"], index=False)
    pd.DataFrame([{"target": k, "random_hits_at_320_mean": v} for k, v in random_hits.items()]).to_csv(files["random"], index=False)
    files["report"].write_text(report(pr))
    return files


def benchmark_fit(runs_dir: Path, out: Path, sizes=(32, 96, 192, 352), repeats: int = 3) -> Path:
    """Refit time of each surrogate on history prefixes of real campaigns (no hidden outcome is read:
    the histories are revealed results from `runs_dir`). Run on an otherwise idle machine."""
    import time

    from combo_agent.eval.runner import _features, _gp_features
    from combo_agent.models.surrogate import make_surrogate

    files = sorted((runs_dir / "runs").glob("bo_ucb__*__s*.reveals.parquet"))[:repeats]
    rows = []
    for f in files:
        rv = pd.read_parquet(f).sort_values("round", kind="stable")
        for n in sizes:
            frame = rv.iloc[:n][[*KEY, "y_syn", "y_inh", "n_reps"]]
            for name, feats in (("ensemble", _features("full")), ("gp", _gp_features())):
                m = make_surrogate(name, feats, seed=0)
                t0 = time.perf_counter()
                m.fit_frame(frame)
                rows.append({"surrogate": name, "history_file": f.name, "n_history": n,
                             "fit_s": time.perf_counter() - t0})
    path = out / "fit_time_benchmark.csv"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ensemble", type=Path, required=True)
    ap.add_argument("--gp", type=Path, required=True)
    ap.add_argument("--random", type=Path, default=None, help="directory with random campaigns (enrichment)")
    ap.add_argument("--seeds", type=str, default="0-4")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--benchmark-fit", action="store_true",
                    help="only time refits of both surrogates on history prefixes of the --ensemble runs")
    args = ap.parse_args(argv)
    if args.benchmark_fit:
        print(benchmark_fit(args.ensemble, args.out))
        return 0
    from combo_agent.eval.runner import _seeds
    files = build(args.ensemble, args.gp, args.out, args.random, _seeds(args.seeds))
    for k, v in files.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
