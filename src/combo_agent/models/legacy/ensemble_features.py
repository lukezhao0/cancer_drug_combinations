"""LEGACY BUILD: the ensemble's feature blocks (target_design.md §6), built from feature tables only.
The default surrogate, the GP, uses `models/gp.py:GPFeatures` instead.

An experiment is (drug A, drug B, cell line). Its input vector is

    [ f(A) + f(B),  f(A) * f(B),  g(line) ]
      pair block (symmetric in A and B)   cell block

so swapping A and B cannot change a prediction. The model never sees the concatenated vector: its
first layer is linear, so it is applied to the pair block once per drug pair and to the cell block
once per line, then summed (see `ensemble.py`). That avoids a ~5 GB matrix for the pool.

Two modes:
* `full` (the design): drug = 2,048 Morgan bits + target-gene, action-type and target-type one-hots;
  cell = 1,000 most-variable expression genes + 24 driver mutations + 31 copy-number genes (with
  missing flags) + lineage one-hots.
* `compact`: drug = 32 principal components of the fingerprints + the same one-hots; cell = 16
  principal components of expression + mutations + copy number + lineage. Fewer inputs for a model
  trained on 32-352 experiments.

Principal components and standardization are fit on the feature tables (all pool drugs, all lines,
all pool pairs). No outcome is read here; `tests/test_label_access.py` enforces that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from combo_agent.data import paths
from combo_agent.models.features import _numeric, _pca, _standardize

MODES = ("full", "compact")
FP_PCS, EXPR_PCS = 32, 16


@dataclass
class FeatureSet:
    """Drug and cell feature matrices plus the scaling of pair and cell blocks."""

    mode: str
    drug_ids: list[str]
    drug_x: np.ndarray          # (n_drugs, d) raw drug features (after PCA in compact mode)
    line_ids: list[str]
    cell_x: np.ndarray          # (n_lines, c) standardized cell features
    pair_mu: np.ndarray = field(repr=False)
    pair_sd: np.ndarray = field(repr=False)
    pair_keep: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        self._drug_index = {d: i for i, d in enumerate(self.drug_ids)}
        self._line_index = {c: i for i, c in enumerate(self.line_ids)}

    @property
    def pair_dim(self) -> int:
        return int(self.pair_keep.sum())

    @property
    def cell_dim(self) -> int:
        return self.cell_x.shape[1]

    # -- lookups ----------------------------------------------------------------------------------

    def drug_index(self, drugs) -> np.ndarray:
        try:
            return np.fromiter((self._drug_index[d] for d in drugs), dtype=np.int64)
        except KeyError as e:
            raise KeyError(f"drug {e.args[0]!r} has no features") from None

    def line_index(self, lines) -> np.ndarray:
        try:
            return np.fromiter((self._line_index[c] for c in lines), dtype=np.int64)
        except KeyError as e:
            raise KeyError(f"cell line {e.args[0]!r} has no features") from None

    def pair_code(self, ia: np.ndarray, ib: np.ndarray) -> np.ndarray:
        """One integer per unordered drug pair."""
        lo, hi = np.minimum(ia, ib), np.maximum(ia, ib)
        return lo * len(self.drug_ids) + hi

    def pair_block(self, codes: np.ndarray) -> np.ndarray:
        """Standardized symmetric pair features for unordered pair codes."""
        n = len(self.drug_ids)
        a, b = self.drug_x[codes // n], self.drug_x[codes % n]
        raw = np.concatenate([a + b, a * b], axis=1)
        return ((raw - self.pair_mu) / self.pair_sd)[:, self.pair_keep].astype(np.float32)

    def encode(self, drug_a, drug_b, lines) -> tuple[np.ndarray, np.ndarray]:
        """(pair codes, line indices) for experiments given as three aligned sequences."""
        return self.pair_code(self.drug_index(drug_a), self.drug_index(drug_b)), self.line_index(lines)

    def dense(self, codes: np.ndarray, lines: np.ndarray) -> np.ndarray:
        """The full concatenated input (for tests and small checks only)."""
        return np.concatenate([self.pair_block(codes), self.cell_x[lines]], axis=1)

    # -- construction -----------------------------------------------------------------------------

    @classmethod
    def build(cls, drugs: pd.DataFrame, cells: pd.DataFrame, pairs: pd.DataFrame, mode: str = "compact") -> FeatureSet:
        """`drugs`/`cells`: the processed feature tables; `pairs`: drug_a, drug_b of the pool (for
        the pair-block scaling)."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        drugs = drugs.set_index("drug_id")
        used = sorted(set(pairs["drug_a"]) | set(pairs["drug_b"]))
        drugs = drugs.loc[used]
        fp = _numeric(drugs, ("fp_",)).to_numpy()
        onehot = _numeric(drugs, ("target_", "action_", "ttype_")).to_numpy()
        onehot = onehot[:, onehot.std(axis=0) > 0]
        flags = _numeric(drugs, ("has_target",)).to_numpy()
        drug_x = np.concatenate([fp if mode == "full" else _pca(fp, FP_PCS), onehot, flags], axis=1)

        cells = cells.set_index("cellosaurus_ac").sort_index()
        expr = _numeric(cells, ("expr_",)).to_numpy()
        rest = _numeric(cells, ("mut_", "cn_", "lineage_")).to_numpy()  # cn_missing_* flags included
        cell_raw = np.concatenate([expr if mode == "full" else _pca(expr, EXPR_PCS), rest], axis=1)
        mu, sd, keep = _standardize(cell_raw)
        cell_x = ((cell_raw - mu) / sd)[:, keep].astype(np.float32)

        out = cls(mode, used, drug_x.astype(np.float32), list(cells.index), cell_x,
                  np.zeros(1, np.float32), np.ones(1, np.float32), np.ones(1, bool))
        codes = np.unique(out.pair_code(out.drug_index(pairs["drug_a"]), out.drug_index(pairs["drug_b"])))
        n = len(used)
        a, b = out.drug_x[codes // n], out.drug_x[codes % n]
        raw = np.concatenate([a + b, a * b], axis=1)
        out.pair_mu, out.pair_sd, out.pair_keep = _standardize(raw)
        return out

    @classmethod
    def from_cache(cls, mode: str = "compact", root: Path | None = None) -> FeatureSet:
        root = root or paths.processed_dir()
        pool = pd.read_parquet(root / "pool.parquet", columns=["drug_a", "drug_b"]).drop_duplicates()
        return cls.build(pd.read_parquet(root / "drugs.parquet"), pd.read_parquet(root / "cells.parquet"), pool, mode)
