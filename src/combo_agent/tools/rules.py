"""Cell-line rules: the fixed vocabulary the agent uses to say which lines a mechanism applies to.

From `docs/target_design.md` section 9: mutation status of a fixed gene list, copy-number loss or
gain, lineage, and expression tertile of a gene; at most 2 terms joined by `and`. Anything else is
rejected, so code can check a rule with no LLM involved.

    KRAS_mut == 1 and PTEN_loss == 0
    lineage == Lung and EGFR_expr == high
    MYC_gain == 1
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from combo_agent.data.build_features import CN_GENES, DRIVER_GENES
from combo_agent.tools import ToolError
from combo_agent.tools.store import FeatureStore

MAX_TERMS = 2
#: log2 copy-number ratio cut-offs: loss = one copy or fewer, gain = four copies or more.
#: Provisional; the design names the terms but not the thresholds.
CN_LOSS_MAX = -1.0
CN_GAIN_MIN = 1.0
TERTILES = ("low", "mid", "high")

GRAMMAR = (
    "Allowed terms: GENE_mut == 0|1, GENE_loss == 0|1, GENE_gain == 0|1, "
    "lineage == NAME, GENE_expr == low|mid|high. At most 2 terms joined by 'and'."
)

_TERM = re.compile(r"^\s*([A-Za-z0-9_.\-]+)\s*==\s*[\"']?([A-Za-z0-9_ /\-]+?)[\"']?\s*$")
_FEATURE = re.compile(r"^(.+)_(mut|loss|gain|expr)$")


@dataclass(frozen=True)
class Term:
    kind: str  # mut | loss | gain | expr | lineage
    gene: str | None
    value: str  # "0"/"1", a tertile, or a lineage name

    def __str__(self) -> str:
        left = "lineage" if self.kind == "lineage" else f"{self.gene}_{self.kind}"
        return f"{left} == {self.value}"


@dataclass(frozen=True)
class Rule:
    terms: tuple[Term, ...]

    def __str__(self) -> str:
        return " and ".join(str(t) for t in self.terms)

    def evaluate(self, store: FeatureStore) -> pd.Series:
        """True / False per line; <NA> where a needed feature is unmeasured for that line."""
        out = pd.Series(True, index=store.cells.index, dtype="boolean")
        for term in self.terms:
            out = out & _evaluate_term(term, store)
        return out


def parse(text: str) -> Rule:
    """Parse a rule, or raise ToolError saying what is wrong and what is allowed."""
    if not text or not text.strip():
        raise ToolError(f"empty rule. {GRAMMAR}")
    if re.search(r"\b(or|not)\b|[()<>!]", text, flags=re.I):
        raise ToolError(f"only '==' and 'and' are allowed in a rule. {GRAMMAR}")
    parts = re.split(r"\s+and\s+", text.strip(), flags=re.I)
    if len(parts) > MAX_TERMS:
        raise ToolError(f"a rule may have at most {MAX_TERMS} terms, got {len(parts)}. {GRAMMAR}")
    return Rule(tuple(_parse_term(p) for p in parts))


def _parse_term(text: str) -> Term:
    m = _TERM.match(text)
    if not m:
        raise ToolError(f"cannot read rule term {text.strip()!r}. {GRAMMAR}")
    left, value = m.group(1), m.group(2).strip()
    if left.lower() == "lineage":
        return Term("lineage", None, value)
    f = _FEATURE.match(left)
    if not f:
        raise ToolError(f"unknown rule feature {left!r}. {GRAMMAR}")
    gene, kind = f.group(1).upper(), f.group(2)
    if kind == "expr":
        if value.lower() not in TERTILES:
            raise ToolError(f"{left} must be compared with low, mid or high, not {value!r}")
        return Term(kind, gene, value.lower())
    if value not in ("0", "1"):
        raise ToolError(f"{left} must be compared with 0 or 1, not {value!r}")
    allowed = DRIVER_GENES if kind == "mut" else CN_GENES
    if gene not in allowed:
        raise ToolError(f"{gene} is not in the {kind} gene list: {', '.join(allowed)}")
    return Term(kind, gene, value)


def tertiles(values: pd.Series) -> pd.Series:
    """low / mid / high by rank across lines (ties broken by order); <NA> stays missing."""
    ranks = values.rank(method="first")
    n = int(values.notna().sum())
    if n < 3:
        raise ToolError("too few lines with a value to form tertiles")
    return pd.cut(ranks, bins=[0, n / 3, 2 * n / 3, n], labels=list(TERTILES)).astype("string")


def _evaluate_term(term: Term, store: FeatureStore) -> pd.Series:
    if term.kind == "lineage":
        by_lower = {name.lower(): name for name in store.lineages}
        name = by_lower.get(term.value.lower())
        if name is None:
            raise ToolError(f"unknown lineage {term.value!r}; lineages: {', '.join(store.lineages)}")
        return (store.cells["lineage"] == name).astype("boolean")
    assert term.gene is not None
    if term.kind == "mut":
        return (store.mutation(term.gene) == int(term.value)).astype("boolean")
    if term.kind == "expr":
        t = tertiles(store.expression(term.gene))
        return (t == term.value).astype("boolean").where(t.notna())
    cn = store.copy_number(term.gene)
    event = cn <= CN_LOSS_MAX if term.kind == "loss" else cn >= CN_GAIN_MIN
    return (event == bool(int(term.value))).astype("boolean").where(cn.notna())
