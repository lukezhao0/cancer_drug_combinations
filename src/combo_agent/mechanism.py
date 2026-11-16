"""Mechanism questions using M1 features. No tools (yet)."""

from functools import cached_property
import hashlib
import json
from pathlib import Path

import pandas as pd
from sheldon.benchmarks.base import Benchmark
from sheldon.core.registry import benchmarks
from sheldon.core.types import Prediction, RunContext, Score, Task

from combo_agent.data.paths import data_cache

TABLES = {
    "cells": "processed/cells.parquet",
    "drugs": "interim/drugs.parquet",
    "targets": "interim/drug_targets.parquet",
    "expression": "interim/features/cell_expression.parquet",
    "copy_number": "interim/features/cell_copy_number.parquet",
}

TASKS = (
    ("mek-pi3k", "a MEK inhibitor with a PI3K inhibitor", ("MIRDAMETINIB", "DACTOLISIB")),
    ("parp-dna-damage", "a PARP inhibitor with a DNA-damaging agent", ("VELIPARIB", "CISPLATIN")),
    ("egfr-mek", "an EGFR inhibitor with a MEK inhibitor", ("GEFITINIB", "MIRDAMETINIB")),
)

QUESTION = """We are choosing drug-combination experiments in a panel of cancer cell lines. Consider {pair}.
1. Choose a concrete pair from the supplied M1 drug annotations and check its recorded targets.
2. Propose competing mechanisms for synergy or lack of synergy. State a cell-line rule for your leading hypothesis, for example `KRAS_mut == 1 and PTEN_loss == 0`.
3. Check that rule against the supplied cell-line features: name matching and nonmatching lines, count each group (at least 3 per side for testability), and examine target relevance. Show your counting method; no code execution is available. Describe what would refute the hypothesis.
No experiment outcomes are available. Use the supplied real M1 data as evidence; distinguish it from biological interpretation. No callable tools or literature retrieval are available. Do not claim to have executed analyses or measured synergy.
The drug annotations include drugs outside the frozen combination pool. Annotation membership does not establish that a combination was screened. The original MEK/PARP questions therefore remain mechanism investigations, not claims about eligible pool experiments.
Rules have at most two terms joined by `and`: GENE_mut/loss/gain == 0|1, GENE_expr == low|mid|high, or lineage == NAME. Use only supplied genes. Copy-number loss means log2 ratio <= -1; gain means >= 1. Null means unknown, never wild type or normal copy number. Expression is log2(FPKM+1); expression tertiles use ascending rank(method='first') with rank <= n/3 low, <= 2*n/3 mid, otherwise high (n is the number of nonmissing values), across all 57 lines. Cell row order breaks ties. Mutation calls are only for the listed driver genes. Respect imputation, parent-line and problematic-line flags. This diagnostic run includes both selectable and held-out feature rows, as the sandbox benchmark does; it is not a held-out evaluation.

M1 feature snapshot (JSON; all 57 cell lines, full drug/target annotations, and a task-specific subset of expression/copy-number genes):
{features}
"""


def records(frame):
    # pandas converts missing values to JSON null, not nonstandard NaN.
    return json.loads(frame.to_json(orient="records", double_precision=6))


@benchmarks.register("combo-mechanism-data")
class ComboMechanismData(Benchmark):
    name = "combo-mechanism-data"
    splits = ("eval",)
    blurb = "M1 mechanism questions with features in the prompt and no callable tools"
    answer_format = "Answer with four labeled sections: PAIR:, RULE:, MECHANISM:, EVIDENCE:. Include group counts, limitations and competing mechanisms."

    @cached_property
    def feature_root(self):
        return Path(self.opts.get("data_cache") or data_cache())

    def check(self):
        return [f"missing {p}; run scripts/prepare_sandbox_data.py" for p in TABLES.values()
                if not (self.feature_root / p).is_file()]

    @cached_property
    def tables(self):
        return {key: pd.read_parquet(self.feature_root / path) for key, path in TABLES.items()}

    @cached_property
    def provenance(self):
        return {path: hashlib.sha256((self.feature_root / path).read_bytes()).hexdigest()
                for path in TABLES.values()}

    def snapshot(self, names):
        tables = self.tables
        cells, drugs, targets = (tables[k] for k in ("cells", "drugs", "targets"))
        ids = drugs.loc[drugs.preferred_name.isin(names), "drug_id"]
        if len(ids) != len(names):
            raise ValueError(f"missing task drug annotations: {names}")
        genes = {"KRAS", "PTEN", "TP53", "BRCA1", "BRCA2"}
        for value in targets.loc[targets.drug_id.isin(ids), "gene_symbols"].dropna():
            genes.update(g for g in value.split("|") if g)
        cell_cols = [c for c in cells if not c.startswith(("expr_", "cn_", "lineage_"))]
        out = {"sha256": self.provenance, "reference_drugs_for_gene_selection": list(names),
               "cells": records(cells[cell_cols]),
               "drugs": records(drugs[["drug_id", "preferred_name", "source_ids", "max_phase"]]),
               "targets": records(targets[["drug_id", "gene_symbols", "target_pref_name", "action_type", "mechanism_of_action", "mechanism_source"]])}
        for key in ("expression", "copy_number"):
            frame = tables[key]
            selected = sorted(genes & set(frame.columns))
            # Align all gene vectors to the cells table, as sandbox-tools does.
            aligned = frame.set_index("cellosaurus_ac").reindex(cells.cellosaurus_ac)
            out[key] = {"line_ids": list(aligned.index),
                        "genes": {g: json.loads(aligned[g].to_json(orient="values", double_precision=6)) for g in selected},
                        "unavailable_genes": sorted(genes - set(frame.columns))}
        return out

    def probes(self, task):
        return []

    def load(self, split="eval", limit=None, seed=0):
        if split not in self.splits:
            raise ValueError(f"unsupported split: {split}")
        if problems := self.check():
            raise FileNotFoundError("; ".join(problems))
        tasks = [Task(id=tid, benchmark=self.name,
                      question=QUESTION.format(pair=pair, features=json.dumps(self.snapshot(names), separators=(",", ":"), allow_nan=False)),
                      payload={"pair": pair, "feature_sha256": self.provenance})
                 for tid, pair, names in TASKS]
        return self.subsample(tasks, limit, seed)

    async def score(self, task: Task, pred: Prediction, ctx: RunContext) -> Score:
        text = str(pred.answer or "")
        fields = {field: any(line.strip().upper().startswith(field + ":") for line in text.splitlines())
                  for field in ("PAIR", "RULE", "MECHANISM", "EVIDENCE")}
        return Score(task.id, float(all(fields.values())),
                     metrics={"well_formed": float(all(fields.values()))},
                     detail={"fields": fields, "score_meaning": "format only; biology unverified"})
