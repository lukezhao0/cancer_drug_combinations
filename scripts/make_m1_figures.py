"""Figures for docs/project-notes/m1-freeze-and-splits.md, drawn from the frozen M1 outputs.

Run from the code repo root, in a throwaway environment (matplotlib isn't a project dependency):

    uv run --isolated --no-project --python 3.12 --with matplotlib==3.9.2 --with pandas==2.3.3 \
        --with pyarrow==25.0.1 --with numpy==1.26.4 \
        python -I scripts/make_m1_figures.py

Writes PNGs to docs/project-notes/figures/. Palette: the dataviz reference
palette, slots 1-3, validated in light mode (aqua is below 3:1 contrast, so every aqua mark is
direct-labelled). Every figure has a matching table in the doc.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CODE = Path.cwd()
OUT = CODE / "docs" / "project-notes" / "figures"
P, I, F = CODE / "data_cache/processed", CODE / "data_cache/interim", CODE / "data_cache/interim/features"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True,
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


def fig_row_fates():
    rep = json.loads((I / "experiments_report.json").read_text())
    f = rep["fates"]
    labels = {"kept": "Kept: NCI-ALMANAC pool", "line_dropped": "Cell line dropped (decision 11)",
              "oneil": "O'Neil study, by drug name (decision 13)",
              "oneil_ambiguous": "O'Neil or ALMANAC, can't tell (decision 13)",
              "self_pair": "Drug paired with itself (decision 5)", "third_study": "Third study (decision 13)"}
    order = ["kept", "line_dropped", "oneil", "oneil_ambiguous", "self_pair", "third_study"]
    vals = [f[k] for k in order]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    y = np.arange(len(order))[::-1]
    ax.barh(y, vals, color=[BLUE] + [MUTED] * 5, height=0.62)
    for yi, v in zip(y, vals):
        ax.text(v * 1.15 if v > 0 else 1, yi, f"{v:,}", va="center", color=INK2)
    ax.set_yticks(y, [labels[k] for k in order])
    ax.set_xscale("log")
    ax.set_xlim(5, 2e6)
    ax.set_xlabel("Source rows (log scale)")
    ax.set_title(f"Every one of the {rep['source_rows']:,} source rows gets exactly one fate (step 1e)", loc="left")
    ax.grid(axis="y", visible=False)
    save(fig, "m1_row_fates.png")


def fig_split_by_lineage():
    s = json.loads((P / "splits.json").read_text())["per_lineage"]
    lins = sorted(s, key=lambda k: s[k]["lines"])
    held = np.array([s[k]["quota"] for k in lins])
    inel = np.array([len(s[k]["ineligible"]) for k in lins])
    sel = np.array([s[k]["lines"] for k in lins]) - held - inel
    fig, ax = plt.subplots(figsize=(8, 4.6))
    y = np.arange(len(lins))
    left = np.zeros(len(lins))
    for vals, color, name in ((sel, BLUE, "Selectable (eligible, not held out)"),
                              (inel, AQUA, "Selectable (ineligible: borrowed/imputed features)"),
                              (held, ORANGE, "Held out (test lines)")):
        ax.barh(y, vals, left=left, color=color, height=0.62, edgecolor=SURFACE, linewidth=2, label=name)
        for yi, l, v in zip(y, left, vals):
            if v:
                ax.text(l + v / 2, yi, str(int(v)), ha="center", va="center",
                        color="#ffffff" if color != AQUA else INK, fontsize=9, fontweight="bold")
        left = left + vals
    for yi, k in zip(y, lins):
        names = ", ".join(s[k]["heldout"])
        ax.text(left[yi] + 0.15, yi, names, va="center", color=INK2, fontsize=8.5)
    ax.set_yticks(y, lins)
    ax.set_xlim(0, 13.5)
    ax.set_xlabel("Cell lines")
    ax.set_title("Held-out split: 11 of 57 lines, stratified by lineage, families kept together", loc="left")
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    ax.grid(axis="y", visible=False)
    save(fig, "m1_split_by_lineage.png")


def fig_outcomes():
    lab = pd.read_parquet(P / "labels.parquet")
    pool = pd.read_parquet(P / "pool.parquet")
    t = json.loads((P / "splits.json").read_text())["thresholds"]
    sel = (pool.split == "selectable").values
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, col, thr, title, xlab in (
            (axes[0], "y_syn", t["y_syn"], "Synergy (Bliss)", "y_syn (mean Bliss)"),
            (axes[1], "y_inh", t["y_inh"], "Killing (CSS)", "y_inh (mean CSS)")):
        v = lab.loc[sel, col]
        ax.hist(v, bins=120, color=BLUE, edgecolor=SURFACE, linewidth=0.3)
        ax.set_yscale("log")
        ax.axvline(thr, color=ORANGE, linewidth=2)
        ax.text(thr, ax.get_ylim()[1] * 0.4, f"  hit threshold\n  p95 = {thr:.2f}", color=INK2, fontsize=9)
        ax.set_title(title, loc="left")
        ax.set_xlabel(xlab)
        ax.set_ylabel("Selectable experiments (log)")
    fig.suptitle("Outcome distributions and hit thresholds (selectable experiments, n = "
                 f"{int(sel.sum()):,})", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "m1_outcome_thresholds.png")


def fig_syn_vs_inh():
    lab = pd.read_parquet(P / "labels.parquet")
    t = json.loads((P / "splits.json").read_text())["thresholds"]
    fig, ax = plt.subplots(figsize=(6.4, 5))
    hb = ax.hexbin(lab.y_inh, lab.y_syn, gridsize=80, bins="log", cmap="Blues", mincnt=1, linewidths=0)
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("Experiments per cell (log)", color=INK2)
    ax.axhline(t["y_syn"], color=ORANGE, linewidth=1.5)
    ax.axvline(t["y_inh"], color=ORANGE, linewidth=1.5)
    ax.axvline(0, color=MUTED, linewidth=1, linestyle="--")
    n = int(lab.syn_hit_low_kill.sum())
    ax.annotate(f"synergy hits that barely kill\n(CSS < 0): {n:,} = flagged", xy=(-20, 40), xytext=(-55, 62),
                color=INK2, fontsize=9, arrowprops={"arrowstyle": "-", "color": MUTED})
    ax.set_xlabel("y_inh: killing (CSS)")
    ax.set_ylabel("y_syn: synergy (Bliss)")
    rho = lab[["y_syn", "y_inh"]].corr("spearman").iloc[0, 1]
    ax.set_title(f"Synergy and killing are nearly unrelated (Spearman {rho:.2f})", loc="left")
    ax.grid(False)
    save(fig, "m1_synergy_vs_killing.png")


def fig_sr_imputation():
    methods = ["Know nothing\n(average line)", "DepMap as-is", "DepMap,\ntranslated",
               "CellMiner arrays,\ntranslated (used)", "Average of both"]
    r = [0.0, 0.790, 0.862, 0.952, 0.940]
    rmse = [1.699, 1.946, 0.884, 0.559, 0.618]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, vals, title, fmt in ((axes[0], r, "Line-specific signal recovered (r, higher is better)", "{:.2f}"),
                                 (axes[1], rmse, "Error, RMSE in log2 units (lower is better)", "{:.2f}")):
        colors = [MUTED, MUTED, MUTED, BLUE, MUTED]
        y = np.arange(len(methods))[::-1]
        ax.barh(y, vals, color=colors, height=0.6)
        for yi, v in zip(y, vals):
            ax.text(v + max(vals) * 0.02, yi, fmt.format(v), va="center", color=INK2)
        ax.set_yticks(y, methods if ax is axes[0] else [""] * len(methods))
        ax.set_xlim(0, max(vals) * 1.2)
        ax.set_title(title, loc="left", fontsize=10.5)
        ax.grid(axis="y", visible=False)
    fig.suptitle("SR expression: leave-one-out test of imputation methods on 45 verified lines",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "m1_sr_imputation.png")


if __name__ == "__main__":
    fig_row_fates()
    fig_split_by_lineage()
    fig_outcomes()
    fig_syn_vs_inh()
    fig_sr_imputation()
