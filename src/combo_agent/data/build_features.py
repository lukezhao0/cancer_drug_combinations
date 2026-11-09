"""Step 1f: a numeric description of every drug and every pool cell line (target_design.md §4 step 6).

Nothing here describes an outcome. Every input is measured before any drug pair is applied, so the
features leak nothing about hidden experiment results.

**Cell lines** (the 57 lines in `experiments.parquet`). One source per data type, chosen from the
2026-11-08 validation (docs/project-notes/depmap-data.md §5):

| block | source | rule |
|:--|:--|:--|
| lineage | DepMap `Model.csv` (26Q3) | `OncotreeLineage` for the line's `depmap_id` |
| expression | CellMiner RNA-seq, log2(FPKM+1), all lines | **identity-checked against DepMap**: a line whose CellMiner profile doesn't best-match its own DepMap profile, and matches it at Spearman < `IDENTITY_MIN`, is treated as suspect. Its expression is **imputed** (`impute_expression`): CellMiner's own microarrays, calibrated per gene onto the RNA-seq scale, then calibrated DepMap, then the gene mean. It is flagged `expression_imputed = 1`, and is missing only if nothing is available. (On 2026-11-08 this catches SR, whose CellMiner RNA matches RPMI-8226, while its arrays show the true ALK+ ALCL profile.) |
| driver mutations | DepMap where the line has its own default profile; else CellMiner exome | DepMap: a row counts if Hotspot, LikelyLoF, OncogeneHighImpact or TumorSuppressorHighImpact is set, or it's a TP53 missense change. CellMiner: a "protein function affecting" variant. CellMiner misses many TSG mutations, so lines sourced from it carry `mutation_source = cellminer` |
| copy number | CellMiner combined aCGH, log2 ratio to normal, all lines | per-gene NaN where CellMiner has no value; the feature matrix fills 0 (= 2N) and sets `cn_missing_<gene>` |
| expression features | the `N_EXPRESSION_GENES` most variable genes across pool lines | Ties are broken by gene name. Variance uses expression only, never outcomes |

**Drugs** (all 122 from 1c, with an `in_pool` flag): a Morgan fingerprint (radius 2, 2048 bits) of
1c's SMILES, multi-hot target genes (from 1c's mechanisms), multi-hot action types and target types,
and flags. 3 Pt drugs keep their source SMILES, because ChEMBL stores no structure for them. The 7
drugs without a ChEMBL target (decision 8) get an all-zero target block and `has_target = 0`.

Outputs, in `data_cache/interim/features/`:

    cells.parquet               one row per pool line: identity, lineage, sources, flags
    cell_features.parquet       model-ready cell matrix (lineage one-hot, mutations, CN, expression, flags)
    cell_mutations.parquet      line x driver gene, 0/1, with source
    cell_expression.parquet     line x all CellMiner genes (for agent tools and rule checks)
    cell_copy_number.parquet    line x all CellMiner genes
    drug_features.parquet       model-ready drug matrix
    features_report.json        counts, validation numbers, every check

Run: python -m combo_agent.data.build_features
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from combo_agent.data import ingest_cellminer as ic
from combo_agent.data import paths

DEPMAP_RELEASE = "DepMap Public 26Q3"
N_EXPRESSION_GENES = 1000
FP_BITS, FP_RADIUS = 2048, 2
#: Below this own-profile Spearman (on the 2,000 most variable shared genes), and not the best match,
#: a CellMiner expression profile is treated as a likely sample mix-up. Matched lines sit at a
#: median of 0.91; unrelated lines at a median of 0.58 (2026-11-08).
IDENTITY_MIN = 0.80
IDENTITY_GENES = 2000

ONCOGENES = ("KRAS", "NRAS", "HRAS", "BRAF", "PIK3CA", "EGFR", "ERBB2", "CTNNB1", "IDH1", "MAP2K1")
TUMOUR_SUPPRESSORS = ("TP53", "PTEN", "CDKN2A", "RB1", "APC", "STK11", "KEAP1", "NF1", "SMAD4",
                      "ARID1A", "FBXW7", "BRCA1", "BRCA2", "ATM")
DRIVER_GENES = ONCOGENES + TUMOUR_SUPPRESSORS
#: Genes whose copy number goes into the feature matrix: the drivers, plus classic amplified genes.
CN_GENES = DRIVER_GENES + ("MYC", "CCND1", "CDK4", "MDM2", "MET", "CDK6", "FGFR1")

_GENE_COL = re.compile(r"^(.+) \((\d+)\)$")


class FeatureError(RuntimeError):
    """An input is missing, or a coverage check failed."""


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class Result:
    tables: dict[str, pd.DataFrame]
    report: dict[str, Any]
    checks: list[Check] = field(default_factory=list)


# -- inputs ---------------------------------------------------------------------------------------


def depmap_dir() -> Path:
    return paths.raw_dir() / "depmap"


def _bool(s: pd.Series) -> pd.Series:
    return s.fillna(False).astype(str).str.lower().isin(["true", "1", "yes"])


def depmap_expression(models: set[str], path: Path | None = None) -> pd.DataFrame:
    """Default-profile expression, models x gene symbol (first of duplicate symbols)."""
    path = path or depmap_dir() / "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"
    df = pd.read_csv(path)
    df = df[(df.IsDefaultEntryForModel == "Yes") & df.ModelID.isin(models)].set_index("ModelID")
    cols = [c for c in df.columns if _GENE_COL.match(c)]
    out = df[cols].astype(float)
    out.columns = [_GENE_COL.match(c).group(1) for c in cols]
    return out.loc[:, ~out.columns.duplicated()]


def depmap_driver_calls(models: set[str], path: Path | None = None) -> tuple[pd.DataFrame, set[str]]:
    """(models x DRIVER_GENES 0/1, models that have a default mutation profile)."""
    path = path or depmap_dir() / "OmicsSomaticMutations.csv"
    cols = ["ModelID", "IsDefaultEntryForModel", "HugoSymbol", "VariantInfo", "Hotspot", "LikelyLoF",
            "OncogeneHighImpact", "TumorSuppressorHighImpact"]
    dm = pd.read_csv(path, usecols=cols)
    dm = dm[(dm.IsDefaultEntryForModel == "Yes") & dm.ModelID.isin(models)]
    profiled = set(dm.ModelID)
    dm = dm[dm.HugoSymbol.isin(DRIVER_GENES)]
    hit = (_bool(dm.Hotspot) | _bool(dm.LikelyLoF) | _bool(dm.OncogeneHighImpact)
           | _bool(dm.TumorSuppressorHighImpact)
           | ((dm.HugoSymbol == "TP53") & dm.VariantInfo.astype(str).str.contains("MISSENSE", case=False)))
    calls = (dm[hit].groupby(["ModelID", "HugoSymbol"]).size().unstack(fill_value=0)
             .reindex(index=sorted(profiled), columns=list(DRIVER_GENES), fill_value=0) > 0).astype("int8")
    return calls, profiled


def cellminer_matrix(key: str, line_map: pd.DataFrame, cell_lines: pd.DataFrame,
                     interim: Path | None = None) -> pd.DataFrame:
    """CellMiner dataset as lines (cellosaurus_ac) x gene symbol. First row wins for duplicate symbols."""
    df = ic.load(key, interim)
    m = ic.match_lines([c for c in df.columns if ":" in c], line_map, cell_lines)
    out = df.drop_duplicates("gene").set_index("gene")[list(m)].rename(columns=m).astype(float).T
    out.index.name = "cellosaurus_ac"
    return out


# -- validation -----------------------------------------------------------------------------------


def identity_check(cm_expr: pd.DataFrame, dm_expr: pd.DataFrame, depmap_of: dict[str, str],
                   n_genes: int = IDENTITY_GENES) -> pd.DataFrame:
    """For each line with both profiles: Spearman with its own DepMap profile, and its best match.

    Uses the `n_genes` most variable shared genes (variance across the CellMiner lines).
    """
    genes = sorted(set(cm_expr.columns) & set(dm_expr.columns))
    var = cm_expr[genes].var().sort_values(ascending=False, kind="stable")
    top = var.index[:n_genes]
    A = cm_expr[top].rank(axis=1)
    D = dm_expr[top].rank(axis=1)
    rows = []
    for ac, dep in depmap_of.items():
        if ac not in A.index or dep not in D.index:
            continue
        r = pd.Series(np.corrcoef(A.loc[ac].values, D.values)[0, 1:], index=D.index)
        best = r.idxmax()
        rows.append({"cellosaurus_ac": ac, "depmap_id": dep, "own_spearman": round(float(r[dep]), 4),
                     "best_depmap_id": best, "best_spearman": round(float(r[best]), 4),
                     "own_is_best": best == dep})
    t = pd.DataFrame(rows)
    if len(t):
        t["suspect"] = (~t.own_is_best) & (t.own_spearman < IDENTITY_MIN)
    return t


def _calibrate(X: np.ndarray, Y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Per-gene least squares y = a + b*x, fitted on rows of (X, Y); NaN where it can't be fitted."""
    ok = ~np.isnan(X) & ~np.isnan(Y)
    n = ok.sum(0)
    Xo, Yo = np.where(ok, X, 0.0), np.where(ok, Y, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        xm, ym = Xo.sum(0) / n, Yo.sum(0) / n
        xc, yc = np.where(ok, X - xm, 0.0), np.where(ok, Y - ym, 0.0)
        sxx = (xc * xc).sum(0)
        b = np.where(sxx > 1e-9, (xc * yc).sum(0) / sxx, 0.0)
        pred = ym + b * (x - xm)
    return np.where((n >= 5) & ~np.isnan(x), pred, np.nan)


def impute_expression(target: str, trusted: pd.DataFrame, array: pd.DataFrame | None,
                      dm_expr: pd.DataFrame, depmap_of: dict[str, str], target_depmap: str | None
                      ) -> tuple[pd.Series, pd.Series]:
    """Expression for `target` on CellMiner's RNA-seq scale, for a line whose RNA-seq sample is suspect.

    Per gene, in order of preference:
      1. CellMiner's own microarray value, mapped onto the RNA-seq scale by a per-gene linear fit
         over the trusted lines measured both ways;
      2. DepMap's profile of the line, mapped the same way over identity-verified lines;
      3. the gene's mean over trusted lines.

    Returns (values, source per gene). On 2026-11-08, leave-one-out over 45 verified lines recovered
    a line's own deviation from the mean at r = 0.95 with (1), and 0.86 with (2).
    """
    genes = trusted.columns
    out = pd.Series(np.nan, index=genes)
    src = pd.Series("gene_mean", index=genes, dtype=object)
    if array is not None and target in array.index:
        lines = [a for a in trusted.index if a in array.index]
        g = [x for x in genes if x in array.columns]
        p = _calibrate(array.loc[lines, g].values, trusted.loc[lines, g].values, array.loc[target, g].values)
        hit = ~np.isnan(p)
        out.loc[np.array(g)[hit]] = p[hit]
        src.loc[np.array(g)[hit]] = "cellminer_array"
    if target_depmap is not None and target_depmap in dm_expr.index:
        lines = [a for a in trusted.index if depmap_of.get(a) in dm_expr.index]
        g = [x for x in genes if x in dm_expr.columns and np.isnan(out[x])]
        if g and lines:
            X = dm_expr.loc[[depmap_of[a] for a in lines], g].values
            p = _calibrate(X, trusted.loc[lines, g].values, dm_expr.loc[target_depmap, g].values)
            hit = ~np.isnan(p)
            out.loc[np.array(g)[hit]] = p[hit]
            src.loc[np.array(g)[hit]] = "depmap"
    rest = out.isna()
    out[rest] = trusted.loc[:, rest[rest].index].mean()
    return out.clip(lower=0.0), src  # log2(FPKM+1) is never negative


def imputation_quality(trusted: pd.DataFrame, array: pd.DataFrame, genes: list[str]) -> dict[str, float]:
    """Leave-one-out over trusted lines with arrays: how well the array route recovers each line."""
    lines = [a for a in trusted.index if a in array.index]
    g = [x for x in genes if x in array.columns]
    X, Y = array.loc[lines, g].values, trusted.loc[lines, g].values
    rs, rmse = [], []
    for i in range(len(lines)):
        tr = np.arange(len(lines)) != i
        p = _calibrate(X[tr], Y[tr], X[i])
        mean = np.nanmean(Y[tr], 0)
        p = np.where(np.isnan(p), mean, p)
        ok = ~np.isnan(Y[i])
        dy, dp = (Y[i] - mean)[ok], (p - mean)[ok]
        if dy.std() > 0 and dp.std() > 0:  # a constant deviation has no defined correlation
            rs.append(np.corrcoef(dy, dp)[0, 1])
        rmse.append(float(np.sqrt(np.mean((p - Y[i])[ok] ** 2))))
    return {"lines": len(lines), "genes": len(g), "deviation_r_median": round(float(np.median(rs)), 3),
            "rmse_median": round(float(np.median(rmse)), 3),
            "rmse_gene_mean_baseline": round(float(np.median(
                [np.sqrt(np.nanmean((Y[i] - np.nanmean(np.delete(Y, i, 0), 0)) ** 2)) for i in range(len(lines))])), 3)}


# -- drugs ----------------------------------------------------------------------------------------


def morgan_bits(smiles: str) -> np.ndarray:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise FeatureError(f"RDKit cannot parse {smiles!r}")
    fp = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS).GetFingerprint(mol)
    arr = np.zeros(FP_BITS, dtype="int8")
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def drug_features(drugs: pd.DataFrame, targets: pd.DataFrame, pool_drugs: set[str]) -> pd.DataFrame:
    d = drugs.sort_values("drug_id", ignore_index=True)
    fp = pd.DataFrame(np.vstack([morgan_bits(s) for s in d.smiles]),
                      columns=[f"fp_{i}" for i in range(FP_BITS)])
    t = targets.assign(gene=targets.gene_symbols.fillna("").str.split("|")).explode("gene")
    t = t[t.gene.astype(bool)]

    def multihot(col: str, prefix: str, frame: pd.DataFrame) -> pd.DataFrame:
        x = (frame.assign(v=1).pivot_table(index="drug_id", columns=col, values="v", aggfunc="max")
             .reindex(d.drug_id).fillna(0).astype("int8"))
        x = x.reindex(sorted(x.columns), axis=1)
        x.columns = [f"{prefix}{re.sub(r'[^A-Za-z0-9]+', '_', str(c)).strip('_')}" for c in x.columns]
        return x.reset_index(drop=True)

    tg = multihot("gene", "target_", t)
    act = multihot("action_type", "action_", targets)
    ttype = multihot("target_type", "ttype_", targets)
    # `has_target`: any ChEMBL mechanism target, DNA included (decision 8 concerns drugs with none).
    # `has_target_gene`: a target with a gene symbol; DNA-targeting drugs (alkylators, Pt) have none.
    meta = d[["drug_id", "preferred_name", "smiles_origin"]].assign(
        in_pool=d.drug_id.isin(pool_drugs).astype("int8"),
        has_target=(d.n_targets.fillna(0) > 0).astype("int8").values,
        has_target_gene=(tg.sum(axis=1) > 0).astype("int8").values,
        smiles_from_source=(d.smiles_origin == "source").astype("int8"),
        n_target_genes=tg.sum(axis=1).astype("int16").values)
    return pd.concat([meta, tg, act, ttype, fp], axis=1)


# -- build ----------------------------------------------------------------------------------------


def build(experiments: pd.DataFrame, cell_lines: pd.DataFrame, line_map: pd.DataFrame,
          drugs: pd.DataFrame, drug_targets: pd.DataFrame, model: pd.DataFrame,
          cm_expr: pd.DataFrame, cm_mut: pd.DataFrame, cm_cn: pd.DataFrame,
          dm_expr: pd.DataFrame, dm_calls: pd.DataFrame, dm_profiled: set[str],
          cm_array: pd.DataFrame | None = None) -> Result:
    checks: list[Check] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append(Check(name, bool(ok), detail))

    pool_acs = sorted(experiments.cellosaurus_ac.unique())
    cells = (cell_lines.set_index("cellosaurus_ac").loc[pool_acs,
             ["cellosaurus_name", "depmap_id", "depmap_is_proxy", "family_id", "problematic"]]
             .reset_index())
    mod = model.set_index("ModelID")
    check("every pool line has a DepMap model row", cells.depmap_id.isin(mod.index).all(),
          ", ".join(cells.cellosaurus_name[~cells.depmap_id.isin(mod.index)]))
    cells["lineage"] = cells.depmap_id.map(mod.OncotreeLineage)
    cells["primary_disease"] = cells.depmap_id.map(mod.OncotreePrimaryDisease)
    cells["subtype"] = cells.depmap_id.map(mod.OncotreeSubtype)
    check("every pool line has a lineage", cells.lineage.notna().all())
    for name, frame in (("expression", cm_expr), ("mutations", cm_mut), ("copy number", cm_cn)):
        missing = sorted(set(pool_acs) - set(frame.index))
        check(f"CellMiner {name} covers every pool line", not missing, ", ".join(missing))

    # Expression identity check against DepMap (own profiles only; proxies are a different stock).
    own = {r.cellosaurus_ac: r.depmap_id for r in cells.itertuples() if not r.depmap_is_proxy}
    ident = identity_check(cm_expr.loc[pool_acs], dm_expr, own)
    suspect = set(ident.loc[ident.suspect, "cellosaurus_ac"]) if len(ident) else set()
    cells["expression_source"] = "cellminer"
    cells["expression_missing"] = np.int8(0)
    cells["expression_imputed"] = np.int8(0)
    cells["expression_identity_spearman"] = cells.cellosaurus_ac.map(
        ident.set_index("cellosaurus_ac").own_spearman if len(ident) else {})

    # Mutations, in order of preference:
    #   1. the line's own DepMap default profile;
    #   2. its Cellosaurus parent's DepMap profile (SNB-19 -> U-251 MG, NCI-H322M -> NCI-H322,
    #      HL-60(TB) -> HL-60), when the parent is confirmed by expression;
    #   3. CellMiner's own exome calls.
    # CellMiner misses many tumour-suppressor mutations (TP53 in 12 of 50 lines, STK11 in 6 of 6),
    # which is why a confirmed parent's DepMap calls come before it. CellMiner's own calls are kept
    # in `cellminer_calls` for comparison.
    rrid_to_model = (model.dropna(subset=["RRID"]).drop_duplicates("RRID").set_index("RRID").ModelID)
    derived = cell_lines.set_index("cellosaurus_ac").derived_from.fillna("")
    use_dm = {ac for ac, dep in own.items() if dep in dm_profiled}
    rows, cm_rows, parent_used = [], [], {}
    for ac in pool_acs:
        cm_v = (cm_mut.loc[ac].reindex(list(DRIVER_GENES)).fillna(0) > 0)
        cm_rows.append({"cellosaurus_ac": ac, **{g: int(cm_v[g]) for g in DRIVER_GENES}})
        parents = [rrid_to_model[p] for p in derived.get(ac, "").split("|") if p in rrid_to_model.index]
        proxy = cells.set_index("cellosaurus_ac").depmap_id[ac] if ac not in own else None
        parent_ids = [m for m in ([proxy] if proxy else []) + parents if m in dm_profiled]
        if ac in use_dm:
            v, src = dm_calls.loc[own[ac]].reindex(list(DRIVER_GENES)).fillna(0), "depmap"
        elif parent_ids:
            v, src = dm_calls.loc[parent_ids[0]].reindex(list(DRIVER_GENES)).fillna(0), "depmap_parent"
            parent_used[ac] = parent_ids[0]
        else:
            v, src = cm_v, "cellminer"
        rows.append({"cellosaurus_ac": ac, "mutation_source": src, "mutation_model": parent_used.get(ac, own.get(ac)),
                     **{g: int(v[g]) for g in DRIVER_GENES}})
    mutations = pd.DataFrame(rows)
    cellminer_calls = pd.DataFrame(cm_rows)
    cells = cells.merge(mutations[["cellosaurus_ac", "mutation_source"]], on="cellosaurus_ac")
    cells["cn_source"] = "cellminer"

    # Expression features: top variable genes across the pool lines with trusted expression.
    # A suspect line's RNA-seq is replaced by an imputation (impute_expression), or left missing if
    # there is nothing to impute from. Feature genes are chosen on measured lines only, so an
    # imputed profile never decides which genes become features.
    expr_full = cm_expr.loc[pool_acs].copy()
    expr_full.index.name = "cellosaurus_ac"
    trusted = expr_full.drop(index=sorted(suspect))
    imputation: dict[str, Any] = {}
    cidx = cells.set_index("cellosaurus_ac")
    for ac in sorted(suspect):
        has_array = cm_array is not None and ac in cm_array.index
        dep = cidx.depmap_id[ac]
        if has_array or dep in dm_expr.index:
            vals, src = impute_expression(ac, trusted, cm_array, dm_expr, own, dep)
            expr_full.loc[ac] = vals.reindex(expr_full.columns).values
            cells.loc[cells.cellosaurus_ac == ac, ["expression_source", "expression_imputed"]] = ["imputed", 1]
            imputation[cidx.cellosaurus_name[ac]] = {k: int(v) for k, v in src.value_counts().items()}
        else:
            expr_full.loc[ac] = np.nan
            cells.loc[cells.cellosaurus_ac == ac, ["expression_source", "expression_missing"]] = ["missing", 1]
    var = trusted.var().dropna()
    var = var[var > 0].sort_index().sort_values(ascending=False, kind="stable")
    top_genes = list(var.index[:N_EXPRESSION_GENES])
    check(f"{N_EXPRESSION_GENES} expression genes selected", len(top_genes) == N_EXPRESSION_GENES)
    cn_full = cm_cn.loc[pool_acs].copy()
    cn_full.index.name = "cellosaurus_ac"

    # Model-ready cell matrix.
    lineage = pd.get_dummies(cells.lineage, prefix="lineage", dtype="int8")
    lineage.columns = [re.sub(r"[^A-Za-z0-9]+", "_", c).strip("_") for c in lineage.columns]
    mut = mutations.set_index("cellosaurus_ac")[list(DRIVER_GENES)].loc[pool_acs].add_prefix("mut_")
    cn_genes = [g for g in CN_GENES if g in cn_full.columns]
    cn = cn_full[cn_genes]
    cn_missing = cn.isna().astype("int8").add_prefix("cn_missing_")
    cn_missing = cn_missing.loc[:, cn_missing.any()]
    cn = cn.fillna(0.0).add_prefix("cn_")  # 0 = 2N, flagged above
    ex = expr_full[top_genes].fillna(0.0).add_prefix("expr_")
    flags = cells.set_index("cellosaurus_ac")[["expression_missing", "expression_imputed"]].assign(
        mutation_from_parent=(cells.set_index("cellosaurus_ac").mutation_source == "depmap_parent").astype("int8"),
        mutation_from_cellminer=(cells.set_index("cellosaurus_ac").mutation_source == "cellminer").astype("int8"),
        depmap_is_proxy=cells.set_index("cellosaurus_ac").depmap_is_proxy.astype("int8"))
    lineage.index = pool_acs
    cell_features = pd.concat([flags.loc[pool_acs], lineage, mut, cn, cn_missing, ex], axis=1)
    cell_features.index.name = "cellosaurus_ac"
    check("cell feature matrix has no NaN", not cell_features.isna().any().any())

    # Line relationships recorded for the decision log (decision 16), checked on expression.
    rel = {}
    acs = dict(zip(cells.cellosaurus_name, cells.cellosaurus_ac))
    if {"SNB-19", "U-251MG"} <= set(acs):
        c = trusted.rank(axis=1).T.corr()
        s = c[acs["SNB-19"]].drop(acs["SNB-19"]).sort_values(ascending=False)
        rel["SNB-19_closest_cellminer"] = [(cells.set_index("cellosaurus_ac").cellosaurus_name[i], round(float(v), 3))
                                           for i, v in s.head(3).items()]
    pool_drugs = set(experiments.drug_a) | set(experiments.drug_b)
    dfeat = drug_features(drugs, drug_targets, pool_drugs)
    check("every pool drug has features", pool_drugs <= set(dfeat.drug_id))
    check("drug feature matrix has no NaN", not dfeat.drop(columns=["preferred_name", "smiles_origin"]).isna().any().any())

    report = {
        "depmap_release": DEPMAP_RELEASE,
        "cellminer_db_version": "2.16",
        "pool_lines": len(pool_acs),
        "lineages": cells.lineage.value_counts().sort_index().to_dict(),
        "expression": {"source": "cellminer", "genes_in_features": len(top_genes),
                       "missing_lines": sorted(cells.cellosaurus_name[cells.expression_missing == 1]),
                       "imputed_lines": sorted(cells.cellosaurus_name[cells.expression_imputed == 1]),
                       "imputation_gene_sources": imputation,
                       "imputation_loo_quality": (imputation_quality(trusted, cm_array, top_genes)
                                                  if cm_array is not None and imputation else None),
                       "identity_check": {
                           "lines_checked": len(ident),
                           "own_is_best": int(ident.own_is_best.sum()) if len(ident) else 0,
                           "own_spearman_median": round(float(ident.own_spearman.median()), 3) if len(ident) else None,
                           "suspect": [{"line": cells.set_index("cellosaurus_ac").cellosaurus_name[r.cellosaurus_ac],
                                        "own": r.own_spearman, "best_depmap_id": r.best_depmap_id,
                                        "best": r.best_spearman} for r in ident[ident.suspect].itertuples()]
                           if len(ident) else []}},
        "mutations": {"genes": list(DRIVER_GENES),
                      "source_counts": mutations.mutation_source.value_counts().to_dict(),
                      "cellminer_lines": sorted(cells.cellosaurus_name[cells.mutation_source == "cellminer"]),
                      "parent_lines": {cells.set_index("cellosaurus_ac").cellosaurus_name[a]: {
                          "parent_model": m,
                          "parent_calls": [g for g in DRIVER_GENES if mutations.set_index("cellosaurus_ac").loc[a, g]],
                          "cellminer_own_calls": [g for g in DRIVER_GENES
                                                  if cellminer_calls.set_index("cellosaurus_ac").loc[a, g]]}
                          for a, m in parent_used.items()},
                      "mutated_lines_per_gene": {g: int(mutations[g].sum()) for g in DRIVER_GENES}},
        "copy_number": {"source": "cellminer", "genes_in_features": cn_genes,
                        "lines_with_missing_feature_genes": int((cn_missing.sum(axis=1) > 0).sum()),
                        "missing_cells_full_table": int(cn_full.isna().sum().sum())},
        "relationships": rel,
        "cell_feature_columns": int(cell_features.shape[1]),
        "drugs": {"total": len(dfeat), "in_pool": int(dfeat.in_pool.sum()),
                  "without_target": sorted(dfeat.loc[(dfeat.has_target == 0) & (dfeat.in_pool == 1), "preferred_name"]),
                  "target_but_no_gene": sorted(dfeat.loc[(dfeat.has_target == 1) & (dfeat.has_target_gene == 0)
                                                         & (dfeat.in_pool == 1), "preferred_name"]),
                  "smiles_from_source": sorted(dfeat.loc[dfeat.smiles_from_source == 1, "preferred_name"]),
                  "columns": int(dfeat.shape[1])},
        "checks": [{"name": c.name, "ok": c.ok, "detail": c.detail} for c in checks],
    }
    if not all(c.ok for c in checks):
        raise FeatureError("; ".join(f"{c.name}: {c.detail}" for c in checks if not c.ok))
    tables = {
        "cells": cells,
        "cell_features": cell_features.reset_index(),
        "cell_mutations": mutations,
        "cell_mutations_cellminer": cellminer_calls,
        "cell_expression": expr_full.reset_index(),
        "cell_copy_number": cn_full.reset_index(),
        "expression_identity": ident,
        "drug_features": dfeat,
    }
    return Result(tables, report, checks)


def write_outputs(result: Result, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, t in result.tables.items():
        out[name] = out_dir / f"{name}.parquet"
        t.to_parquet(out[name], index=False)
    out["report"] = out_dir / "features_report.json"
    out["report"].write_text(json.dumps(result.report, indent=2, sort_keys=True, default=str) + "\n")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Step 1f: drug and cell-line features")
    p.add_argument("--out", type=Path, default=None, help="output dir (default data_cache/interim/features)")
    args = p.parse_args(argv)
    interim = paths.interim_dir()
    experiments = pd.read_parquet(interim / "experiments.parquet")
    cell_lines = pd.read_parquet(interim / "cell_lines.parquet")
    line_map = pd.read_parquet(interim / "line_map.parquet")
    pool = cell_lines[cell_lines.cellosaurus_ac.isin(experiments.cellosaurus_ac)]
    model = pd.read_csv(depmap_dir() / "Model.csv", low_memory=False)
    rrid = model.dropna(subset=["RRID"]).drop_duplicates("RRID").set_index("RRID").ModelID
    parent_acs = {p for s in pool.derived_from.fillna("") for p in s.split("|") if p}
    pool_dep = set(pool.depmap_id) | {rrid[p] for p in parent_acs if p in rrid.index}
    dm_calls, dm_profiled = depmap_driver_calls(pool_dep)
    # DepMap expression for all models, so the identity check can look for the best match anywhere.
    dm_expr = depmap_expression(set(model.ModelID))
    result = build(experiments, cell_lines, line_map,
                   pd.read_parquet(interim / "drugs.parquet"), pd.read_parquet(interim / "drug_targets.parquet"),
                   model,
                   cellminer_matrix("expression", line_map, cell_lines),
                   cellminer_matrix("mutation_pfa", line_map, cell_lines),
                   cellminer_matrix("copy_number", line_map, cell_lines),
                   dm_expr, dm_calls, dm_profiled,
                   cm_array=cellminer_matrix("expression_array", line_map, cell_lines))
    out = write_outputs(result, args.out or interim / "features")
    r = result.report
    print(f"{r['pool_lines']} lines, {r['cell_feature_columns']} cell feature columns; "
          f"{r['drugs']['total']} drugs, {r['drugs']['columns']} drug columns")
    print("expression missing:", r["expression"]["missing_lines"], "| imputed:", r["expression"]["imputed_lines"],
          "| mutation sources:",
          r["mutations"]["source_counts"])
    for k, v in out.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
