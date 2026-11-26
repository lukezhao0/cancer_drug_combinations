"""Figures for the surrogate comparison (ensemble vs Gaussian process).

    python scripts/make_gp_comparison_figures.py runs/gp_compare/comparison

Reads the CSVs written by `combo_agent.eval.compare` (and `fit_time_benchmark.csv` when present)
and writes PNGs next to them. No hidden outcome is read here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

COLORS = {"ensemble": "#257de0", "gp": "#f47230"}
LABELS = {"ensemble": "Ensemble (5 networks)", "gp": "Gaussian process"}
SELECTORS = ["greedy", "uncertainty", "bo_ucb", "bo_qei", "bo_ts"]
TARGETS = {"syn": "Synergy (Bliss)", "inh": "Killing (CSS)"}
INK, INK2, GRID = "#191919", "#666666", "#e5e5e5"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "figure.dpi": 150,
})


def _legend(fig):
    handles = [plt.Line2D([], [], color=COLORS[s], lw=2, marker="o", ms=5, label=LABELS[s]) for s in COLORS]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))


def curves(d: Path, col: str, ylabel: str, name: str, title: str) -> Path:
    c = pd.read_csv(d / "curves.csv")
    fig, axes = plt.subplots(2, len(SELECTORS), figsize=(13, 5.4), sharex=True, sharey="row")
    for i, (t, tl) in enumerate(TARGETS.items()):
        for j, sel in enumerate(SELECTORS):
            ax = axes[i, j]
            for s in COLORS:
                g = c[(c.target == t) & (c.selector == sel) & (c.surrogate == s)]
                if g.empty:
                    continue
                agg = g.groupby("n_charged")[col].agg(["mean", "min", "max"])
                ax.fill_between(agg.index, agg["min"], agg["max"], color=COLORS[s], alpha=0.15, lw=0)
                ax.plot(agg.index, agg["mean"], color=COLORS[s], lw=2)
            if i == 0:
                ax.set_title(sel, color=INK)
            if j == 0:
                ax.set_ylabel(f"{tl}\n{ylabel}")
            if i == 1:
                ax.set_xlabel("experiments paid for")
    fig.suptitle(title, y=1.06, color=INK)
    _legend(fig)
    fig.tight_layout()
    out = d / name
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def paired_dots(d: Path) -> Path:
    r = pd.read_csv(d / "per_run.csv")
    metrics = [("hits_at_320", "hits in 320"), ("recall_top1_at_320", "share of true top 1% found"),
               ("best_at_320", "best value found")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6))
    x = np.arange(len(SELECTORS))
    for i, (t, tl) in enumerate(TARGETS.items()):
        for j, (m, ml) in enumerate(metrics):
            ax = axes[i, j]
            for k, s in enumerate(COLORS):
                off = -0.15 if k == 0 else 0.15
                for xi, sel in enumerate(SELECTORS):
                    g = r[(r.target == t) & (r.selector == sel) & (r.surrogate == s)][m]
                    if g.empty:
                        continue
                    ax.scatter(np.full(len(g), xi + off), g, s=12, color=COLORS[s], alpha=0.35, lw=0)
                    ax.scatter([xi + off], [g.mean()], s=60, color=COLORS[s], edgecolor="white", lw=1.5, zorder=3)
            ax.set_xticks(x, SELECTORS)
            ax.grid(axis="x", visible=False)
            if j == 0:
                ax.set_ylabel(tl)
            if i == 0:
                ax.set_title(ml, color=INK)
    fig.suptitle("Discovery at the end of the campaign (big dot = mean of 5 seeds, small = each seed)", y=1.06, color=INK)
    _legend(fig)
    fig.tight_layout()
    out = d / "fig_discovery_at_320.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def calibration(d: Path) -> Path:
    r = pd.read_csv(d / "per_run.csv")
    panels = [("batch_cover1", 0.6827, "within ±1 std: picked batches"), ("batch_cover2", 0.9545, "within ±2 std: picked batches"),
              ("probe_cover1", 0.6827, "within ±1 std: random probe"), ("probe_cover2", 0.9545, "within ±2 std: random probe")]
    fig, axes = plt.subplots(2, 4, figsize=(14, 5.6), sharey="col")
    x = np.arange(len(SELECTORS))
    for i, (t, tl) in enumerate(TARGETS.items()):
        for j, (m, nominal, title) in enumerate(panels):
            ax = axes[i, j]
            for k, s in enumerate(COLORS):
                means = [r[(r.target == t) & (r.selector == sel) & (r.surrogate == s)][m].mean() for sel in SELECTORS]
                ax.bar(x + (-0.19 if k == 0 else 0.19), means, width=0.36, color=COLORS[s], edgecolor="white", lw=1)
            ax.axhline(nominal, color=INK, ls="--", lw=1)
            ax.text(len(SELECTORS) - 0.5, nominal, f" target {nominal:.0%}", va="bottom", ha="right", color=INK, fontsize=8)
            ax.set_ylim(0, 1.05)
            ax.set_xticks(x, SELECTORS, rotation=30)
            ax.grid(axis="x", visible=False)
            if i == 0:
                ax.set_title(title, color=INK)
            if j == 0:
                ax.set_ylabel(f"{tl}\nshare of outcomes")
    fig.suptitle("Calibration: do the error bars contain the real results as often as they should?", y=1.06, color=INK)
    _legend(fig)
    fig.tight_layout()
    out = d / "fig_calibration.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def nll(d: Path) -> Path:
    r = pd.read_csv(d / "per_run.csv")
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
    x = np.arange(len(SELECTORS))
    for j, (t, setname) in enumerate([(t, s) for t in TARGETS for s in ("batch", "probe")]):
        ax = axes[j]
        for k, s in enumerate(COLORS):
            means = [r[(r.target == t) & (r.selector == sel) & (r.surrogate == s)][f"{setname}_nll"].mean() for sel in SELECTORS]
            ax.bar(x + (-0.19 if k == 0 else 0.19), means, width=0.36, color=COLORS[s], edgecolor="white", lw=1)
        ax.set_title(f"{TARGETS[t]}: {'picked batches' if setname == 'batch' else 'random probe'}", color=INK)
        ax.set_xticks(x, SELECTORS, rotation=30)
        ax.grid(axis="x", visible=False)
        if j == 0:
            ax.set_ylabel("negative log-likelihood\n(lower is better)")
    _legend(fig)
    fig.tight_layout()
    out = d / "fig_nll.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def std_vs_distance(d: Path) -> Path:
    b = pd.read_csv(d / "std_vs_distance.csv")
    b["mid"] = (b["dist_lo"] + b["dist_hi"].clip(upper=1.0)) / 2
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, (t, tl) in zip(axes, TARGETS.items()):
        for s in COLORS:
            g = b[(b.target == t) & (b.surrogate == s)]
            w = g.assign(wx=g["latent_std_rel_mean"] * g["n"]).groupby("mid")[["wx", "n"]].sum()
            ax.plot(w.index, w["wx"] / w["n"], color=COLORS[s], lw=2, marker="o", ms=5)
        ax.axhline(1.0, color=INK2, lw=0.8)
        ax.set_title(tl, color=INK)
    fig.supxlabel("distance to the nearest tested experiment (0 = same, 1 = nothing similar)", color=INK, fontsize=9)
    axes[0].set_ylabel("model's own uncertainty\n(÷ that round's median)")
    fig.suptitle("Does uncertainty grow away from the data? (random probe, all selectors and seeds)", y=1.08, color=INK)
    _legend(fig)
    fig.tight_layout()
    out = d / "fig_std_vs_distance.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fit_time(d: Path) -> Path | None:
    f = d / "fit_time_benchmark.csv"
    if not f.exists():
        return None
    t = pd.read_csv(f)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for s in COLORS:
        g = t[t.surrogate == s].groupby("n_history")["fit_s"].agg(["mean", "min", "max"])
        ax.fill_between(g.index, g["min"], g["max"], color=COLORS[s], alpha=0.15, lw=0)
        ax.plot(g.index, g["mean"], color=COLORS[s], lw=2, marker="o", ms=5, label=LABELS[s])
    ax.set_xlabel("experiments in the history")
    ax.set_ylabel("seconds to refit (both outcomes)")
    ax.set_title("Refit time per round (idle machine)", color=INK)
    ax.legend()
    fig.tight_layout()
    out = d / "fig_fit_time.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main(argv: list[str]) -> int:
    d = Path(argv[1]) if len(argv) > 1 else Path("runs/gp_compare/comparison")
    outs = [curves(d, "best", "best value", "fig_best_value.png", "Best value found so far (line = mean of 5 seeds, band = min–max)"),
            curves(d, "recall_top1", "share of top 1%", "fig_recall_top1.png", "Share of the true top 1% found so far"),
            curves(d, "recall_top5", "share of top 5%", "fig_recall_top5.png", "Share of the true top 5% found so far"),
            paired_dots(d), calibration(d), nll(d), std_vs_distance(d), fit_time(d)]
    for o in outs:
        if o:
            print(f"wrote {o}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
