"""Adapter from the tool registry to SHELDON probes, plus the Python-sandbox preload."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from sheldon.core.probes import CallableProbe, Probe, ProbeResult, PythonProbe

from combo_agent.tools import ToolError, registry
from combo_agent.tools.store import DENIED_FILES, FeatureStore

#: Characters of a tool's summary shown to the model. Summaries are written to be complete, so this
#: is a safety cap, not the usual way output is shortened.
RENDER_LIMIT = 8000

#: Text a sandbox snippet may not contain. SHELDON's snippet runner is a subprocess, not a security
#: boundary (and its bubblewrap isolation is Linux-only), so this is a tripwire against reading the
#: data cache by path, not a guarantee. The leak tests are the real check.
BLOCKED_IN_CODE = (*sorted(DENIED_FILES), "data_cache", "parquet", "COMBO_DATA_CACHE")

PYTHON_CONTEXT_DOC = (
    "Variables preloaded in run_python (plain Python lists and dicts; no outcomes of any experiment): "
    "`cells`: one dict per cell line with cellosaurus_ac, name, lineage, primary_disease, split, "
    "mut (gene -> 0/1), cn (gene -> log2 copy-number ratio or None). "
    "`line_ids`: accessions in a fixed order. `expr`: gene -> list of log2(FPKM+1) aligned to "
    "`line_ids`, for the 1,000 most variable genes (use the gene_expression tool for any other gene). "
    "`drugs`: one dict per drug with drug_id, name, max_phase, targets (genes) and mechanisms."
)


def as_probes(backends: Mapping[str, Any]) -> list[Probe]:
    """One SHELDON probe per tool whose backend is supplied."""
    return [_probe(spec, backends[spec.needs]) for spec in registry.offered(backends)]


def _probe(spec: registry.ToolSpec, backend: Any) -> Probe:
    async def run(**kwargs: Any) -> ProbeResult:
        args = {k: v for k, v in kwargs.items() if v not in (None, "", [])}
        try:
            out = await asyncio.to_thread(spec.fn, backend, **args)
        except ToolError as e:
            return ProbeResult(summary=f"{spec.name} could not run: {e}", ok=False)
        return ProbeResult(summary=out["summary"], evidence=out["evidence"], limit=RENDER_LIMIT)

    return CallableProbe(spec.name, spec.description, run, input_schema=spec.input_schema, max_strength=0.8)


class GuardedPythonProbe(PythonProbe):
    """`run_python`, refusing snippets that name the data cache or an outcome file."""

    async def run(self, **kwargs: Any) -> ProbeResult:
        code = str(kwargs.get("code", ""))
        hit = next((s for s in BLOCKED_IN_CODE if s.lower() in code.lower()), None)
        if hit:
            return ProbeResult(
                summary=f"snippet refused: it mentions {hit!r}. Data files are not readable from the "
                        "sandbox; use the preloaded variables and the lookup tools.",
                ok=False,
            )
        return await super().run(**kwargs)


def python_context(store: FeatureStore) -> dict[str, Any]:
    """JSON-serialisable feature tables for the sandbox. Built only from allowlisted tables."""
    cells = store.cells
    cn_genes = store.copy_number_genes
    cn = {g: store.copy_number(g) for g in cn_genes}
    cell_records = [
        {
            "cellosaurus_ac": ac,
            "name": row["cellosaurus_name"],
            "lineage": row["lineage"],
            "primary_disease": row.get("primary_disease"),
            "split": row["split"],
            "mut": {g: int(row[f"mut_{g}"]) for g in store.mutation_genes},
            "cn": {g: (None if cn[g].isna()[ac] else round(float(cn[g][ac]), 3)) for g in cn_genes},
        }
        for ac, row in cells.iterrows()
    ]
    expr = {g: [round(float(v), 3) for v in cells[f"expr_{g}"]] for g in store.expression_feature_genes}
    targets = store.drug_targets
    drug_records = []
    for drug_id, row in store.drugs.iterrows():
        own = targets[targets["drug_id"] == drug_id]
        genes = sorted({g for v in own["gene_symbols"] for g in str(v or "").split("|") if g and g != "nan"})
        phase = row.get("max_phase")
        drug_records.append({
            "drug_id": drug_id,
            "name": row["preferred_name"],
            "max_phase": None if phase is None or phase != phase else float(phase),
            "targets": genes,
            "mechanisms": sorted(set(own["mechanism_of_action"].dropna())),
        })
    return {"cells": cell_records, "line_ids": list(cells.index), "expr": expr, "drugs": drug_records}
