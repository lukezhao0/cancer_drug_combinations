"""Tool specs: one record per tool the LLM can call, with the backend it needs.

A tool is offered only when its backend is supplied and it has an implementation, so the tools from
later milestones are declared here (name, schema, backend) and switch on when their backend exists:

    features    FeatureStore            this module's tools (M1 data only)
    model       response model          query_model, resolve   (M3)
    oracle      simulated lab history   get_history            (M2)
    literature  leakage-filtered search search_literature      (M5)
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from combo_agent.literature import retrieval
from combo_agent.tools import omics, rules

BACKENDS = ("features", "model", "oracle", "literature")


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    properties: dict[str, dict[str, Any]]
    required: tuple[str, ...]
    needs: str
    #: fn(backend, **arguments) -> {"summary": str, "evidence": list[dict]}; None = declared only.
    fn: Callable[..., dict[str, Any]] | None = None

    @property
    def input_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": self.properties, "required": list(self.required),
                "additionalProperties": False}


def _s(description: str) -> dict[str, Any]:
    return {"type": "string", "description": description}


_DRUG = "Drug name or ChEMBL ID, e.g. 'trametinib' or 'CHEMBL2103875'."
_LINE = "Cell line name or Cellosaurus accession, e.g. 'A-549' or 'CVCL_0023'."
_RULE = f"A cell-line rule. {rules.GRAMMAR} Example: 'KRAS_mut == 1 and PTEN_loss == 0'."

TOOLS: tuple[ToolSpec, ...] = (
    ToolSpec("get_drug", "Look up a drug's targets, mechanism of action and clinical phase (ChEMBL).",
             {"drug": _s(_DRUG)}, ("drug",), "features", omics.get_drug),
    ToolSpec("get_cell", "Look up a cell line's lineage, disease, driver mutations and copy-number events.",
             {"line": _s(_LINE)}, ("line",), "features", omics.get_cell),
    ToolSpec("find_lines",
             "List the cell lines that do and do not satisfy a rule. Shows whether a mechanism's "
             "context exists in the panel and whether it can be tested (at least 3 lines per side).",
             {"rule": _s(_RULE), "lineage": _s("Optional: restrict to one lineage, e.g. 'Lung'."),
              "split": {"type": "string", "enum": list(omics.SPLITS),
                        "description": "Optional: restrict to selectable or held-out lines."}},
             ("rule",), "features", omics.find_lines),
    ToolSpec("gene_expression",
             "Expression of one gene across cell lines. With `rule`, compares rule-true against "
             "rule-false lines (medians and a Mann-Whitney p-value). With `lines`, gives each line's "
             "value and tertile. With neither, gives the range and the highest and lowest lines.",
             {"gene": _s("HGNC gene symbol, e.g. 'EGFR'."), "rule": _s("Optional. " + _RULE),
              "lines": {"type": "array", "items": {"type": "string"}, "description": "Optional. " + _LINE}},
             ("gene",), "features", omics.gene_expression),
    ToolSpec("drug_targets_in_line",
             "For each target gene of a drug, its expression, mutation and copy-number status in one cell line.",
             {"drug": _s(_DRUG), "line": _s(_LINE)}, ("drug", "line"), "features", omics.drug_targets_in_line),
    ToolSpec("shared_targets", "Compare two drugs: shared target genes, distinct targets and mechanisms.",
             {"drug_a": _s(_DRUG), "drug_b": _s(_DRUG)}, ("drug_a", "drug_b"), "features", omics.shared_targets),
    # -- declared for later milestones (design section 8); no implementation yet --------------------
    ToolSpec("query_model",
             "Predicted mean and standard deviation of synergy and killing for candidate experiments. "
             "Never returns measured results.",
             {"drug_a": _s(_DRUG), "drug_b": _s(_DRUG), "rule": _s("Optional. " + _RULE)},
             ("drug_a", "drug_b"), "model"),
    ToolSpec("resolve",
             "Turn a drug pair and a cell-line rule into the candidate experiments still untested.",
             {"drug_a": _s(_DRUG), "drug_b": _s(_DRUG), "rule": _s("Optional. " + _RULE)},
             ("drug_a", "drug_b"), "model"),
    ToolSpec("get_history", "Results of experiments already run in this campaign (revealed results only).",
             {"drug": _s("Optional. " + _DRUG), "rule": _s("Optional. " + _RULE)}, (), "oracle"),
    ToolSpec("search_literature",
             "Search PubMed titles and abstracts for mechanism evidence. Papers that report or reanalyse "
             "the screen this campaign's results come from are withheld by a leakage filter, and every "
             "call is logged.",
             {"query": _s("PubMed query, e.g. 'MEK inhibitor PI3K feedback KRAS mutant'."),
              "k": {"type": "integer", "minimum": 1, "maximum": retrieval.MAX_K,
                    "description": f"Optional: number of abstracts (default {retrieval.DEFAULT_K})."}},
             ("query",), "literature", retrieval.search_literature),
)

assert all(t.needs in BACKENDS for t in TOOLS)
assert len({t.name for t in TOOLS}) == len(TOOLS)


def offered(backends: Mapping[str, Any]) -> list[ToolSpec]:
    """The tools that can run with the backends supplied."""
    return [t for t in TOOLS if t.fn is not None and backends.get(t.needs) is not None]


def pending() -> list[ToolSpec]:
    """Tools declared but not implemented yet."""
    return [t for t in TOOLS if t.fn is None]
