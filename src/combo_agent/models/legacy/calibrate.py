"""LEGACY BUILD (the ensemble; see models/legacy/__init__.py). M3 calibration report: how well the response model predicts, and whether its uncertainty is honest.

This is evaluation code: it reads the hidden outcomes (allowlisted in `tests/test_label_access.py`)
to score predictions. Selectors never import it.

Two checks, for each feature mode:

1. **Low-data** (the regime a campaign runs in): train on n = 32, 96, 320 random selectable
   experiments (several repeats), score a fixed random sample of other selectable experiments.
2. **Sanity** (design §6 gate): random 80/20 split of the selectable pool.

Per outcome it reports
* Pearson r and Spearman rho (how well predictions track and rank the truth),
* RMSE relative to a constant predictor (< 1 = better than predicting the training mean),
* 80% interval coverage: share of test points with |truth - mean| <= 1.2816 std (gate: 70-90%),
* hit enrichment: hit rate among the top 1% by predicted mean, divided by the base hit rate
  (1 = no better than random picking). Synergy is reported with and without low-kill hits.

Outputs `data_cache/processed/m3/calibration.json` and `m3_report.md`.

Run: python -m combo_agent.models.legacy.calibrate [--modes compact full] [--repeats 5]
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from combo_agent.data import paths
from combo_agent.lab.oracle import KEY
from combo_agent.models.legacy.ensemble import EnsembleModel
from combo_agent.models.legacy.ensemble_features import FeatureSet

Z80 = 1.2815515655446004
SIZES = (32, 96, 320)


def load_selectable(root: Path | None = None) -> pd.DataFrame:
    root = root or paths.processed_dir()
    pool = pd.read_parquet(root / "pool.parquet", columns=[*KEY, "split"])
    lab = pd.read_parquet(root / "labels.parquet",
                          columns=[*KEY, "y_syn", "y_inh", "is_hit_syn", "is_hit_inh", "syn_hit_low_kill"])
    df = pool.merge(lab, on=KEY, validate="1:1")
    df = df[df["split"] == "selectable"].drop(columns="split").sort_values(KEY).reset_index(drop=True)
    df["is_hit_syn_strict"] = (df["is_hit_syn"].astype(bool) & ~df["syn_hit_low_kill"].astype(bool)).astype(int)
    return df


def score(pred, test: pd.DataFrame, train: pd.DataFrame, top: float = 0.01) -> dict:
    out = {}
    k = max(1, int(round(top * len(test))))
    for t in ("syn", "inh"):
        y, mu, sd = test[f"y_{t}"].to_numpy(), pred.mean(t), pred.std(t)
        const = np.sqrt(np.mean((y - train[f"y_{t}"].mean()) ** 2))
        order = np.argsort(-mu, kind="stable")[:k]
        hits = ["is_hit_syn", "is_hit_syn_strict"] if t == "syn" else ["is_hit_inh"]
        out[t] = {
            "pearson": float(np.corrcoef(mu, y)[0, 1]),
            "spearman": float(spearmanr(mu, y).statistic),
            "rmse_vs_constant": float(np.sqrt(np.mean((y - mu) ** 2)) / const),
            "coverage80": float(np.mean(np.abs(y - mu) <= Z80 * sd)),
            "mean_std_over_rmse": float(np.mean(sd) / np.sqrt(np.mean((y - mu) ** 2))),
            **{f"enrich_top1pct_{h.removeprefix('is_hit_')}": float(test[h].to_numpy()[order].mean() / max(test[h].mean(), 1e-12))
               for h in hits},
        }
    return out


def _agg(runs: list[dict]) -> dict:
    out = {}
    for t in runs[0]:
        out[t] = {m: {"mean": float(np.mean([r[t][m] for r in runs])), "sd": float(np.std([r[t][m] for r in runs]))}
                  for m in runs[0][t]}
    return out


def low_data(df: pd.DataFrame, fs: FeatureSet, sizes=SIZES, repeats: int = 5, eval_size: int = 20_000,
             log=print, **model_kw) -> dict:
    res = {}
    for n in sizes:
        runs, secs = [], []
        for rep in range(repeats):
            rng = np.random.default_rng([n, rep])
            idx = rng.permutation(len(df))
            train, test = df.iloc[idx[:n]], df.iloc[idx[n:n + eval_size]]
            m = EnsembleModel(fs, seed=rep, **model_kw).fit_frame(train)
            secs.append(m.fit_seconds)
            runs.append(score(m.predict(test), test, train))
        res[str(n)] = {"metrics": _agg(runs), "fit_seconds": float(np.mean(secs)), "repeats": repeats}
        s = res[str(n)]["metrics"]
        log(f"  n={n}: syn r={s['syn']['pearson']['mean']:.3f} cov={s['syn']['coverage80']['mean']:.2f} "
            f"enr={s['syn']['enrich_top1pct_syn']['mean']:.2f} | inh r={s['inh']['pearson']['mean']:.3f} "
            f"cov={s['inh']['coverage80']['mean']:.2f} enr={s['inh']['enrich_top1pct_inh']['mean']:.2f}")
    return res


def sanity(df: pd.DataFrame, fs: FeatureSet, train_max: int | None = None, eval_size: int = 40_000,
           log=print, **model_kw) -> dict:
    rng = np.random.default_rng(80)
    idx = rng.permutation(len(df))
    n_train = int(0.8 * len(df))
    tr_idx = idx[:n_train] if train_max is None else idx[:min(n_train, train_max)]
    train, test = df.iloc[tr_idx], df.iloc[idx[n_train:]][:eval_size]
    m = EnsembleModel(fs, seed=0, **model_kw).fit_frame(train)
    t0 = time.perf_counter()
    swapped = test.rename(columns={"drug_a": "drug_b", "drug_b": "drug_a"})
    pred, pred_sw = m.predict(test), m.predict(swapped)
    predict_seconds = time.perf_counter() - t0
    out = {"train_rows": len(train), "test_rows": len(test), "fit_seconds": m.fit_seconds,
           "epochs": m.epochs_run, "swap_max_abs_diff": float(max(np.abs(pred.mean_syn - pred_sw.mean_syn).max(),
                                                                   np.abs(pred.std_inh - pred_sw.std_inh).max())),
           "metrics": score(pred, test, train), "predict_seconds_x2": predict_seconds}
    s = out["metrics"]
    log(f"  sanity ({len(train):,} train): syn r={s['syn']['pearson']:.3f} cov={s['syn']['coverage80']:.2f} | "
        f"inh r={s['inh']['pearson']:.3f} cov={s['inh']['coverage80']:.2f} | fit {m.fit_seconds:.0f}s")
    return out


def report(res: dict) -> str:
    L = ["# M3 calibration report: response model", "",
         "Generated by `combo_agent.models.legacy.calibrate` (evaluation code; reads the hidden outcomes).", "",
         "Coverage = share of test experiments inside the 80% interval (gate 70-90%). Enrichment = hit rate in the "
         "top 1% by predicted mean / base hit rate (1.0 = random). RMSE ratio < 1 = better than predicting the "
         "training mean.", ""]
    for mode, r in res["modes"].items():
        fs = r["features"]
        L += [f"## Mode `{mode}` (pair features {fs['pair_dim']}, cell features {fs['cell_dim']})", "",
              "### Low data (training on n random selectable experiments; mean ± sd over repeats)", "",
              "| n | Outcome | Pearson r | Spearman | RMSE ratio | 80% coverage | std / RMSE | Enrichment top 1% | Fit s |",
              "|--:|:--|--:|--:|--:|--:|--:|--:|--:|"]
        for n, v in r["low_data"].items():
            for t in ("syn", "inh"):
                m = v["metrics"][t]
                f = lambda k: f"{m[k]['mean']:.3f} ± {m[k]['sd']:.3f}"  # noqa: E731
                enr = f(f"enrich_top1pct_{t}")
                if t == "syn":
                    enr += f" (strict {m['enrich_top1pct_syn_strict']['mean']:.2f})"
                L.append(f"| {n} | {t} | {f('pearson')} | {f('spearman')} | {f('rmse_vs_constant')} | "
                         f"{f('coverage80')} | {f('mean_std_over_rmse')} | {enr} | {v['fit_seconds']:.1f} |")
        if "sanity" in r:
            s = r["sanity"]
            L += ["", f"### Sanity: 80/20 split ({s['train_rows']:,} training rows, {s['test_rows']:,} scored)", "",
                  "| Outcome | Pearson r | Spearman | RMSE ratio | 80% coverage | Enrichment top 1% |", "|:--|--:|--:|--:|--:|--:|"]
            for t in ("syn", "inh"):
                m = s["metrics"][t]
                L.append(f"| {t} | {m['pearson']:.3f} | {m['spearman']:.3f} | {m['rmse_vs_constant']:.3f} | "
                         f"{m['coverage80']:.3f} | {m[f'enrich_top1pct_{t}']:.2f} |")
            L += ["", f"Swap test (drug A <-> drug B): max |difference| = {s['swap_max_abs_diff']:.2g}. "
                      f"Fit {s['fit_seconds']:.0f} s, epochs {s['epochs']}.", ""]
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--modes", nargs="+", default=["compact", "full"])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--no-sanity", action="store_true")
    ap.add_argument("--sanity-train-max", type=int, default=None, help="cap training rows of the 80/20 check")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    df = load_selectable()
    res = {"modes": {}, "selectable_rows": len(df),
           "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}}
    for mode in args.modes:
        print(f"mode {mode}", flush=True)
        fs = FeatureSet.from_cache(mode)
        r = {"features": {"pair_dim": fs.pair_dim, "cell_dim": fs.cell_dim},
             "low_data": low_data(df, fs, repeats=args.repeats, log=lambda s: print(s, flush=True))}
        if not args.no_sanity:
            r["sanity"] = sanity(df, fs, train_max=args.sanity_train_max, log=lambda s: print(s, flush=True))
        res["modes"][mode] = r
    out = args.out or paths.processed_dir() / "m3"
    out.mkdir(parents=True, exist_ok=True)
    (out / "calibration.json").write_text(json.dumps(res, indent=2, sort_keys=True) + "\n")
    (out / "m3_report.md").write_text(report(res))
    print(f"wrote {out / 'calibration.json'} and {out / 'm3_report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
