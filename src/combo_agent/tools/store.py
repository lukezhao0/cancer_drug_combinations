"""FeatureStore: read-only access to the feature tables the agent's tools may see.

This is the only module under `combo_agent.tools` that touches disk. It reads a fixed allowlist of
tables and refuses everything else, so no tool can reach experiment outcomes by construction
(`docs/data-bundle/README.md`, "who may read labels").
"""

from __future__ import annotations

import difflib
import re
from functools import cached_property
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from combo_agent.data import paths
from combo_agent.tools import ToolError

#: table key -> path relative to the data cache root. Tools read these and nothing else.
ALLOWED_TABLES = {
    "cells": "processed/cells.parquet",
    "drugs": "interim/drugs.parquet",
    "drug_targets": "interim/drug_targets.parquet",
    "expression": "interim/features/cell_expression.parquet",
    "copy_number": "interim/features/cell_copy_number.parquet",
}

#: Files that hold measured combination outcomes. Listed so the refusal is explicit and testable.
DENIED_FILES = frozenset({
    "labels.parquet",
    "self_pairs.parquet",
    "oneil_rows.parquet",
    "cross_study_overlap.parquet",
})

assert not {Path(p).name for p in ALLOWED_TABLES.values()} & DENIED_FILES

LINE_KEY = "cellosaurus_ac"


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


class FeatureStore:
    """Lazy, cached, read-only view of the allowlisted tables."""

    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root) if root is not None else paths.data_cache()

    # -- disk access ------------------------------------------------------------------------------

    def path(self, table: str) -> Path:
        if table not in ALLOWED_TABLES:
            raise PermissionError(f"table {table!r} is not on the tools allowlist")
        p = self.root / ALLOWED_TABLES[table]
        if p.name in DENIED_FILES:  # unreachable while the import-time assert holds
            raise PermissionError(f"{p.name} holds outcomes and may not be read by tools")
        return p

    def _read(self, table: str, columns: list[str] | None = None) -> pd.DataFrame:
        p = self.path(table)
        if not p.exists():
            raise FileNotFoundError(
                f"{p} is missing. Download the M1 data bundle into data_cache/ (see README.md)."
            )
        return pd.read_parquet(p, columns=columns)

    def available(self) -> list[str]:
        """Problems that would stop the tools from working (empty when ready)."""
        return [f"missing {ALLOWED_TABLES[t]}" for t in ALLOWED_TABLES if not self.path(t).exists()]

    # -- tables -----------------------------------------------------------------------------------

    @cached_property
    def cells(self) -> pd.DataFrame:
        """One row per cell line, indexed by Cellosaurus accession."""
        return self._read("cells").set_index(LINE_KEY, drop=False)

    @cached_property
    def drugs(self) -> pd.DataFrame:
        return self._read("drugs").set_index("drug_id", drop=False)

    @cached_property
    def drug_targets(self) -> pd.DataFrame:
        return self._read("drug_targets")

    @cached_property
    def _wide_columns(self) -> dict[str, frozenset[str]]:
        out = {}
        for table in ("expression", "copy_number"):
            p = self.path(table)
            out[table] = frozenset(pq.read_schema(p).names) - {LINE_KEY} if p.exists() else frozenset()
        return out

    @cached_property
    def _wide_tables(self) -> dict[str, pd.DataFrame]:
        """The gene-level tables, read once (57 lines x ~23k genes, ~11 MB each).

        Reading one column of these files still parses the ~23k-column footer (~0.5 s per call),
        so per-gene reads made `get_cell` (31 copy-number genes) take ~7 s. One full read is ~1 s.
        """
        return {table: self._read(table).set_index(LINE_KEY)
                for table in ("expression", "copy_number") if self._wide_columns[table]}

    def _gene_column(self, table: str, prefix: str, gene: str) -> pd.Series:
        """One gene's values for every line: from the full table, else the model-feature columns."""
        if gene in self._wide_columns[table]:
            s = self._wide_tables[table][gene]
        elif f"{prefix}{gene}" in self.cells.columns:
            s = self.cells[f"{prefix}{gene}"]
        else:
            known = sorted(self._wide_columns[table]) or self._feature_genes(prefix)
            raise ToolError(f"no {table.replace('_', ' ')} data for gene {gene!r}{_did_you_mean(gene, known)}")
        return s.reindex(self.cells.index).astype(float).rename(gene)

    def _feature_genes(self, prefix: str) -> list[str]:
        """Genes with a numeric `<prefix><GENE>` model-feature column in the cells table.

        The table also carries metadata under the same prefixes (`cn_source` is text, `cn_missing_*`
        are flags), so a prefix match alone is not a gene.
        """
        return [c[len(prefix):] for c in self.cells.columns
                if c.startswith(prefix) and not c.startswith(f"{prefix}missing_")
                and pd.api.types.is_numeric_dtype(self.cells[c])]

    @cached_property
    def copy_number_genes(self) -> list[str]:
        """Genes with a copy-number model feature in the cells table."""
        return self._feature_genes("cn_")

    @cached_property
    def expression_feature_genes(self) -> list[str]:
        """Genes with an expression model feature in the cells table (the most variable genes)."""
        return self._feature_genes("expr_")

    def expression(self, gene: str) -> pd.Series:
        """log2(FPKM+1) of one gene, indexed by line."""
        return self._gene_column("expression", "expr_", gene)

    def copy_number(self, gene: str) -> pd.Series:
        """log2 copy-number ratio of one gene (0 = 2 copies), indexed by line; NaN where unmeasured."""
        s = self._gene_column("copy_number", "cn_", gene)
        missing = f"cn_missing_{gene}"
        if missing in self.cells.columns:  # the feature table fills unmeasured values with 0
            s = s.mask(self.cells[missing].astype(bool))
        return s

    def mutation(self, gene: str) -> pd.Series:
        """0/1 driver-mutation status of one gene, indexed by line."""
        col = f"mut_{gene}"
        if col not in self.cells.columns:
            raise ToolError(f"no mutation calls for {gene!r}; genes with calls: {', '.join(self.mutation_genes)}")
        return self.cells[col].astype(int).rename(gene)

    @cached_property
    def mutation_genes(self) -> list[str]:
        return [c[4:] for c in self.cells.columns if c.startswith("mut_")]

    @cached_property
    def lineages(self) -> list[str]:
        return sorted(self.cells["lineage"].dropna().unique())

    # -- name resolution --------------------------------------------------------------------------

    @cached_property
    def _drug_index(self) -> dict[str, str]:
        index: dict[str, str] = {}
        for row in self.drugs.itertuples():
            names = [row.drug_id, row.preferred_name]
            names += str(getattr(row, "source_ids", "") or "").split("|")
            for name in names:
                if name and _norm(name):
                    index.setdefault(_norm(name), row.drug_id)
        return index

    def resolve_drug(self, text: str) -> str:
        """A drug ID, ChEMBL name or source label -> `drug_id`."""
        hit = self._drug_index.get(_norm(text))
        if hit is None:
            names = sorted(self.drugs["preferred_name"].dropna().astype(str))
            raise ToolError(f"unknown drug {text!r}{_did_you_mean(text, names)}")
        return hit

    @cached_property
    def _line_index(self) -> dict[str, str]:
        index: dict[str, str] = {}
        for row in self.cells.itertuples():
            for name in (row.cellosaurus_ac, row.cellosaurus_name, getattr(row, "depmap_id", None)):
                if name and _norm(name):
                    index.setdefault(_norm(name), row.cellosaurus_ac)
        return index

    def resolve_line(self, text: str) -> str:
        """A Cellosaurus accession, line name (any punctuation) or DepMap ID -> accession."""
        hit = self._line_index.get(_norm(text))
        if hit is None:
            names = sorted(self.cells["cellosaurus_name"].astype(str))
            raise ToolError(f"unknown cell line {text!r}{_did_you_mean(text, names)}")
        return hit

    def line_name(self, ac: str) -> str:
        return str(self.cells.at[ac, "cellosaurus_name"])

    def drug_name(self, drug_id: str) -> str:
        return str(self.drugs.at[drug_id, "preferred_name"])


def _did_you_mean(text: str, options: list[str]) -> str:
    by_norm = {_norm(o): o for o in options}
    close = difflib.get_close_matches(_norm(text), list(by_norm), n=3, cutoff=0.6)
    return f"; did you mean {', '.join(by_norm[c] for c in close)}?" if close else ""
