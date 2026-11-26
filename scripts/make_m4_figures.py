"""Figures for an M4 campaign grid, drawn from `python -m combo_agent.eval.metrics <out>` outputs.

Run from the code repo root, in a throwaway environment (matplotlib isn't a project dependency):

    uv run --isolated --no-project --python 3.12 --with matplotlib==3.9.2 --with pandas==2.3.3 \
        --with pyarrow==25.0.1 --with numpy==1.26.4 \
        python -I scripts/make_m4_figures.py runs/m4_main

Writes PNGs to <out>/figures/:
    m4_curves_<target>.png   discovery curves: mean cumulative hits vs charged experiments, 95%
                             bootstrap band over seeds, one line per selector (direct-labelled)
    m4_hits320.png           hits at 320 per seed (dots) and mean (bar tick), per target
    m4_fame.png              hits at 320 split by famous / middle / obscure pairs

Palette: the dataviz reference categorical slots 1-5 in fixed order, validated in light mode
(CVD and normal-vision checks pass; aqua, yellow and magenta are below 3:1 contrast, so every line
is direct-labelled and report.md carries the same numbers as a table). Selector -> colour is fixed,
so a figure with fewer selectors never repaints the others.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

COLORS = {"bo_ucb": "#2a78d6", "random": "#eb6834", "bo_qei": "#1baf7a", "greedy": "#eda100",
          "uncertainty": "#e87ba4"}
EXTRA = "#898781"  # ablation variants and anything else: muted, always direct-labelled
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
TITLES = {"syn": "Synergy hits (Bliss >= 9.30)", "inh": "Killing hits (CSS >= 62.95)"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "lines.linewidth": 2,
})


def color(sel: str) -> str:
    return COLORS.get(sel, EXTRA)


def save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out / name)


def band(values: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """95% bootstrap band of the mean over seeds (rows), per column."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n, len(values)))
    means = values[idx].mean(1)
    return np.quantile(means, 0.025, axis=0), np.quantile(means, 0.975, axis=0)


def fig_curves(results: pd.DataFrame, target: str, out: Path) -> None:
    r = results[results.target == target]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ends = []
    for sel, g in r.groupby("selector", sort=True):
        wide = g.pivot_table(index="seed", columns="n_charged", values=f"hits_{target}")
        x, m = wide.columns.to_numpy(), wide.mean().to_numpy()
        lo, hi = band(wide.to_numpy())
        ax.fill_between(x, lo, hi, color=color(sel), alpha=0.14, linewidth=0)
        ax.plot(x, m, color=color(sel), label=sel)
        ends.append((m[-1], sel, x[-1]))
    # direct labels at line ends, nudged apart so they never collide
    ends.sort()
    span = max(e[0] for e in ends) - min(e[0] for e in ends) or 1
    placed = []
    for y, sel, x in ends:
        y_lab = y if not placed else max(y, placed[-1] + 0.06 * span)
        placed.append(y_lab)
        ax.annotate(f"{sel}  {y:.1f}", (x, y), xytext=(x + 6, y_lab), color=INK2, va="center", fontsize=9,
                    arrowprops={"arrowstyle": "-", "color": AXIS, "lw": 0.6})
    ax.set_xlim(0, r["n_charged"].max() * 1.22)
    ax.set_xlabel("Experiments run (seed set excluded)")
    ax.set_ylabel("Cumulative hits")
    ax.set_title(f"Discovery curves: {TITLES[target]}", loc="left")
    ax.legend(frameon=False, loc="upper left")
    save(fig, out, f"m4_curves_{target}.png")


def fig_hits320(per_run: pd.DataFrame, out: Path) -> None:
    targets = sorted(per_run.target.unique())
    fig, axes = plt.subplots(1, len(targets), figsize=(5 * len(targets), 3.8), squeeze=False)
    for ax, target in zip(axes[0], targets):
        r = per_run[per_run.target == target]
        order = r.groupby("selector")["hits_at_320"].mean().sort_values().index.tolist()
        for i, sel in enumerate(order):
            v = r[r.selector == sel]["hits_at_320"].to_numpy()
            jitter = np.linspace(-0.12, 0.12, len(v)) if len(v) > 1 else np.zeros(1)
            ax.scatter(v, i + jitter, s=28, color=color(sel), edgecolor=SURFACE, linewidth=1.5, zorder=3)
            ax.plot([v.mean()] * 2, [i - 0.3, i + 0.3], color=INK, lw=2, zorder=4)
            ax.annotate(f"{v.mean():.1f}", (v.mean(), i + 0.32), ha="center", fontsize=8, color=INK2)
        ax.set_yticks(range(len(order)), order)
        ax.set_xlabel("Hits in 320 experiments (dots: seeds; tick: mean)")
        ax.set_title(TITLES[target], loc="left")
    save(fig, out, "m4_hits320.png")


def fig_fame(per_run: pd.DataFrame, out: Path) -> None:
    if "hits_famous" not in per_run.columns:
        return
    targets = sorted(per_run.target.unique())
    fame_colors = {"famous": "#2a78d6", "middle": "#86b6ef", "obscure": "#eb6834"}
    fig, axes = plt.subplots(1, len(targets), figsize=(5 * len(targets), 3.8), squeeze=False)
    for ax, target in zip(axes[0], targets):
        m = per_run[per_run.target == target].groupby("selector")[["hits_famous", "hits_middle", "hits_obscure"]].mean()
        m = m.loc[m.sum(1).sort_values().index]
        left = np.zeros(len(m))
        for col, f in zip(m.columns, ("famous", "middle", "obscure")):
            ax.barh(range(len(m)), m[col], left=left, color=fame_colors[f], height=0.6, label=f,
                    edgecolor=SURFACE, linewidth=2)
            left += m[col].to_numpy()
        ax.set_yticks(range(len(m)), m.index)
        ax.set_xlabel("Mean hits in 320 experiments")
        ax.set_title(TITLES[target], loc="left")
        ax.legend(frameon=False, loc="lower right", fontsize=8)
    save(fig, out, "m4_fame.png")


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__)
        return 2
    run_dir = Path(argv[0])
    results = pd.read_parquet(run_dir / "results.parquet")
    per_run = pd.read_parquet(run_dir / "per_run.parquet")
    out = run_dir / "figures"
    for target in sorted(results.target.unique()):
        fig_curves(results, target, out)
    fig_hits320(per_run, out)
    fig_fame(per_run, out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
