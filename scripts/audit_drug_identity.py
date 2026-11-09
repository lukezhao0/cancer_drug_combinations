"""Audit drug identities in the TDC `drugcomb` (NCI-60) table.

Reproduces every number in docs/drug-identity-audit.md:
  1. groups the 129 drug IDs by RDKit parent structure (cleanup -> largest fragment -> uncharge ->
     canonical tautomer -> InChIKey) and lists the groups with more than one name;
  2. counts the triplets that collide once those names are merged;
  3. compares the two source studies on the colliding triplets;
  4. shows what largest-fragment stripping does to every multi-fragment SMILES;
  5. counts SMILES with any stereo mark.

Reads only the converted parquet (never the pickle). Run from the code repo root:
    uv run python scripts/audit_drug_identity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize as S

RDLogger.DisableLog("rdApp.*")
PARQUET = Path(sys.argv[1] if len(sys.argv) > 1 else "data_cache/interim/tdc_drugcomb.parquet")
SCORES = ["CSS", "Synergy_Bliss", "Synergy_Loewe", "Synergy_ZIP", "Synergy_HSA"]
KEY = ["A", "B", "Cell_Line_ID"]

df = pd.read_parquet(PARQUET)
long = pd.concat([
    df[["Drug1_ID", "Drug1", "Cell_Line_ID"]].set_axis(["id", "smi", "line"], axis=1),
    df[["Drug2_ID", "Drug2", "Cell_Line_ID"]].set_axis(["id", "smi", "line"], axis=1),
])
info = long.groupby("id").agg(smi=("smi", "first"), n_smi=("smi", "nunique"), lines=("line", "nunique"))
info["rows"] = [int(((df.Drug1_ID == i) | (df.Drug2_ID == i)).sum()) for i in info.index]

_lf, _un, _taut = S.LargestFragmentChooser(), S.Uncharger(), S.TautomerEnumerator()


def parent(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    return _taut.Canonicalize(_un.uncharge(_lf.choose(S.Cleanup(mol))))


info["parent_key"] = [Chem.MolToInchiKey(parent(s)) for s in info.smi]
print(f"drug IDs: {len(info)}; IDs with >1 SMILES: {(info.n_smi > 1).sum()}; "
      f"distinct parents: {info.parent_key.nunique()}; "
      f"distinct 14-char skeletons: {info.parent_key.str[:14].nunique()}")

# 1. duplicate groups. Within each group, the name seen on fewer lines is the O'Neil one.
groups = info.groupby("parent_key").filter(lambda g: len(g) > 1)
synonyms: dict[str, str] = {}
print("\n== groups of names with the same parent structure")
for _, g in groups.groupby("parent_key"):
    g = g.sort_values("lines")
    base, alt = g.index[0], g.index[-1]
    synonyms[alt] = base
    same = "identical SMILES" if g.smi.iloc[0] == g.smi.iloc[-1] else "SMILES differ"
    print("  " + " | ".join(f"{i} ({r} rows, {l} lines)" for i, r, l in zip(g.index, g.rows, g.lines))
          + f"  [{same}]")

# 2. collisions after merging
uses_alm = df.Drug1_ID.isin(synonyms) | df.Drug2_ID.isin(synonyms)
uses_on = df.Drug1_ID.isin(synonyms.values()) | df.Drug2_ID.isin(synonyms.values())
m = df.replace({"Drug1_ID": synonyms, "Drug2_ID": synonyms})
m = m.assign(A=m[["Drug1_ID", "Drug2_ID"]].min(axis=1), B=m[["Drug1_ID", "Drug2_ID"]].max(axis=1))
d = df.assign(A=df[["Drug1_ID", "Drug2_ID"]].min(axis=1), B=df[["Drug1_ID", "Drug2_ID"]].max(axis=1))
n_before, n_after = d.groupby(KEY).ngroups, m.groupby(KEY).ngroups
print(f"\ntriplets before merge {n_before}, after {n_after}, newly colliding {n_before - n_after}; "
      f"pairs before {d.groupby(['A', 'B']).ngroups}, after {m.groupby(['A', 'B']).ngroups}")
alm = m[uses_alm & ~uses_on].groupby(KEY)[SCORES].mean()
on = m[uses_on & ~uses_alm].groupby(KEY)[SCORES].mean()
paired = alm.join(on, how="inner", lsuffix="_alm", rsuffix="_on")
for base in synonyms.values():
    k = paired.reset_index()
    print(f"  colliding triplets involving {base}: {((k.A == base) | (k.B == base)).sum()}")

# 3. agreement between the studies on the same triplet
def spearman(x: pd.Series, y: pd.Series) -> float:
    return float(np.corrcoef(x.rank(), y.rank())[0, 1])


print(f"\n== cross-study agreement on {len(paired)} paired triplets "
      f"({paired.reset_index().Cell_Line_ID.nunique()} lines)")
for c in SCORES:
    x, y = paired[c + "_alm"], paired[c + "_on"]
    print(f"  {c:14s} pearson {np.corrcoef(x, y)[0, 1]:.2f}  spearman {spearman(x, y):.2f}  "
          f"mean ALMANAC {x.mean():6.1f}  mean O'Neil {y.mean():6.1f}  mean |diff| {(x - y).abs().mean():.1f}")
for c, t in [("Synergy_Bliss", 9.26), ("CSS", 63.3)]:
    ha, ho = paired[c + "_alm"] >= t, paired[c + "_on"] >= t
    print(f"  {c} hits (>= {t}): ALMANAC {ha.sum()}, O'Neil {ho.sum()}, both {(ha & ho).sum()}, "
          f"disagree {(ha ^ ho).sum()}")
dup = d[d.duplicated(KEY, keep=False)]
first, last = dup.groupby(KEY)[SCORES].first(), dup.groupby(KEY)[SCORES].last()
for c in ["CSS", "Synergy_Bliss"]:
    print(f"  reference: same-name (A,B)/(B,A) duplicates, n={len(first)}, {c} pearson "
          f"{np.corrcoef(first[c], last[c])[0, 1]:.2f}")

# 4. what largest-fragment stripping does
print("\n== multi-fragment SMILES -> largest fragment (uncharged)")
for i, s in info.smi[info.smi.str.contains(r"\.")].items():
    print(f"  {i[:32]:32s} {s[:50]:50s} -> {Chem.MolToSmiles(_un.uncharge(_lf.choose(Chem.MolFromSmiles(s))))[:45]}")

# 5. stereochemistry
print(f"\nSMILES with any stereo mark (@ / \\): {info.smi.str.contains(r'[@/\\\\]').sum()} of {len(info)}")
