"""The in-silico tool functions: label-free questions about drugs and cell lines.

Each takes the FeatureStore plus JSON-friendly arguments and returns
`{"summary": str, "evidence": list[dict]}`. The summary is the complete, compact text the model
reads; the evidence holds the same facts as records for the trace. Bad input raises ToolError.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from scipy.stats import mannwhitneyu

from combo_agent.tools import ToolError, rules
from combo_agent.tools.store import FeatureStore

#: A rule with fewer lines than this on either side cannot be tested (design section 9).
MIN_SIDE = 3
SPLITS = ("selectable", "heldout")


def _split_genes(value: Any) -> list[str]:
    return [g for g in str(value or "").split("|") if g and g != "nan"]


def _targets(store: FeatureStore, drug_id: str) -> pd.DataFrame:
    return store.drug_targets[store.drug_targets["drug_id"] == drug_id]


def _target_genes(store: FeatureStore, drug_id: str) -> list[str]:
    genes: list[str] = []
    for value in _targets(store, drug_id)["gene_symbols"]:
        genes += [g for g in _split_genes(value) if g not in genes]
    return genes


def _line_label(store: FeatureStore, ac: str) -> str:
    return f"{store.line_name(ac)} ({store.cells.at[ac, 'lineage']})"


def get_drug(store: FeatureStore, drug: str) -> dict[str, Any]:
    """Targets, mechanism of action and clinical phase of one drug."""
    drug_id = store.resolve_drug(drug)
    row = store.drugs.loc[drug_id]
    mechanisms = [
        {
            "target": t.target_pref_name,
            "genes": _split_genes(t.gene_symbols),
            "action": t.action_type,
            "mechanism": t.mechanism_of_action,
            "target_type": t.target_type,
        }
        for t in _targets(store, drug_id).itertuples()
    ]
    phase = None if pd.isna(row.get("max_phase")) else float(row["max_phase"])
    lines = [f"{row['preferred_name']} ({drug_id}), max clinical phase {phase if phase is not None else 'unknown'}."]
    if mechanisms:
        for m in mechanisms:
            genes = ", ".join(m["genes"]) or "no gene (e.g. DNA)"
            lines.append(f"- {m['mechanism']} [{m['action']}; genes: {genes}]")
    else:
        lines.append("No mechanism target is recorded for this drug in ChEMBL.")
    return {
        "summary": "\n".join(lines),
        "evidence": [{"drug_id": drug_id, "name": row["preferred_name"], "max_phase": phase,
                      "mechanisms": mechanisms}],
    }


def get_cell(store: FeatureStore, line: str) -> dict[str, Any]:
    """Lineage, disease, driver mutations and copy-number events of one cell line."""
    ac = store.resolve_line(line)
    row = store.cells.loc[ac]
    mutated = [g for g in store.mutation_genes if row[f"mut_{g}"] == 1]
    gains, losses = [], []
    for g in store.copy_number_genes:
        value = store.copy_number(g).at[ac]
        if pd.notna(value) and value >= rules.CN_GAIN_MIN:
            gains.append(g)
        elif pd.notna(value) and value <= rules.CN_LOSS_MAX:
            losses.append(g)
    notes = []
    if row.get("expression_imputed", 0) == 1:
        notes.append("expression is imputed, not measured")
    if row.get("mutation_from_parent", 0) == 1:
        notes.append("mutation calls are borrowed from the parent line")
    if isinstance(row.get("problematic"), str) and row["problematic"]:
        notes.append(f"Cellosaurus flags this line: {row['problematic']}")
    record = {
        "cellosaurus_ac": ac, "name": row["cellosaurus_name"], "lineage": row["lineage"],
        "primary_disease": row.get("primary_disease"), "subtype": row.get("subtype"),
        "split": row["split"], "mutated_drivers": mutated, "cn_gain": gains, "cn_loss": losses,
        "notes": notes,
    }
    summary = (
        f"{record['name']} ({ac}): {record['lineage']}, {record['primary_disease']}"
        f" / {record['subtype']}; split = {record['split']}.\n"
        f"Mutated driver genes: {', '.join(mutated) or 'none of the ' + str(len(store.mutation_genes)) + ' tested'}.\n"
        f"Copy-number gain (>= 4 copies): {', '.join(gains) or 'none'}. "
        f"Loss (<= 1 copy): {', '.join(losses) or 'none'}."
    )
    if notes:
        summary += "\nCaveats: " + "; ".join(notes) + "."
    return {"summary": summary, "evidence": [record]}


def _select(store: FeatureStore, lineage: str | None, split: str | None) -> pd.Index:
    cells = store.cells
    keep = pd.Series(True, index=cells.index)
    if lineage:
        by_lower = {name.lower(): name for name in store.lineages}
        if lineage.lower() not in by_lower:
            raise ToolError(f"unknown lineage {lineage!r}; lineages: {', '.join(store.lineages)}")
        keep &= cells["lineage"] == by_lower[lineage.lower()]
    if split:
        if split not in SPLITS:
            raise ToolError(f"split must be one of {', '.join(SPLITS)}")
        keep &= cells["split"] == split
    return cells.index[keep]


def find_lines(store: FeatureStore, rule: str, lineage: str | None = None,
               split: str | None = None) -> dict[str, Any]:
    """Which cell lines satisfy a rule, and which do not."""
    parsed = rules.parse(rule)
    result = parsed.evaluate(store).loc[_select(store, lineage, split)]
    groups = {
        "rule_true": list(result.index[result.fillna(False).astype(bool)]),
        "rule_false": list(result.index[(~result).fillna(False).astype(bool)]),
        "unknown": list(result.index[result.isna()]),
    }
    scope = ", ".join(x for x in (lineage and f"lineage {lineage}", split and f"split {split}") if x) or "all lines"
    lines = [f"Rule `{parsed}` over {scope}: {len(groups['rule_true'])} true, "
             f"{len(groups['rule_false'])} false, {len(groups['unknown'])} unknown."]
    for name in ("rule_true", "rule_false", "unknown"):
        if groups[name]:
            lines.append(f"{name}: " + ", ".join(_line_label(store, ac) for ac in groups[name]))
    testable = min(len(groups["rule_true"]), len(groups["rule_false"])) >= MIN_SIDE
    if not testable:
        lines.append(f"Warning: fewer than {MIN_SIDE} lines on one side, so this rule cannot be tested here.")
    evidence = [
        {"group": name, "n": len(acs), "lines": [
            {"cellosaurus_ac": ac, "name": store.line_name(ac), "lineage": store.cells.at[ac, "lineage"],
             "split": store.cells.at[ac, "split"]} for ac in acs]}
        for name, acs in groups.items()
    ]
    return {"summary": "\n".join(lines), "evidence": [{"rule": str(parsed), "testable": testable}, *evidence]}


def gene_expression(store: FeatureStore, gene: str, rule: str | None = None,
                    lines: list[str] | None = None) -> dict[str, Any]:
    """Expression of one gene across lines; with a rule, compares rule-true against rule-false lines."""
    gene = gene.strip()
    values = store.expression(gene)
    tert = rules.tertiles(values)
    if lines:
        acs = [store.resolve_line(x) for x in lines]
        rows = [{"cellosaurus_ac": ac, "name": store.line_name(ac), "log2_fpkm1": round(float(values[ac]), 2),
                 "tertile": tert[ac]} for ac in acs]
        text = [f"{gene} expression, log2(FPKM+1); tertile is across all {int(values.notna().sum())} lines."]
        text += [f"- {r['name']}: {r['log2_fpkm1']} ({r['tertile']})" for r in rows]
        return {"summary": "\n".join(text), "evidence": rows}
    if rule:
        parsed = rules.parse(rule)
        mask = parsed.evaluate(store)
        true = values[mask.fillna(False).astype(bool)].dropna()
        false = values[(~mask).fillna(False).astype(bool)].dropna()
        record: dict[str, Any] = {"gene": gene, "rule": str(parsed), "n_true": len(true), "n_false": len(false)}
        if min(len(true), len(false)) < MIN_SIDE:
            return {"summary": f"{gene} by `{parsed}`: {len(true)} true vs {len(false)} false lines; "
                               f"fewer than {MIN_SIDE} on one side, so no comparison was made.",
                    "evidence": [record]}
        p = float(mannwhitneyu(true, false, alternative="two-sided").pvalue)
        record.update(median_true=round(float(true.median()), 2), median_false=round(float(false.median()), 2),
                      mann_whitney_p=float(f"{p:.3g}"))
        return {
            "summary": f"{gene} expression by `{parsed}`: median {record['median_true']} in {len(true)} "
                       f"rule-true lines vs {record['median_false']} in {len(false)} rule-false lines "
                       f"(log2(FPKM+1); two-sided Mann-Whitney p = {record['mann_whitney_p']}, not corrected "
                       f"for multiple tests).",
            "evidence": [record],
        }
    order = values.dropna().sort_values(ascending=False)
    top, bottom = order.head(5), order.tail(5)
    record = {
        "gene": gene, "n_lines": int(len(order)), "median": round(float(order.median()), 2),
        "min": round(float(order.min()), 2), "max": round(float(order.max()), 2),
        "highest": [{"name": store.line_name(ac), "value": round(float(v), 2)} for ac, v in top.items()],
        "lowest": [{"name": store.line_name(ac), "value": round(float(v), 2)} for ac, v in bottom.items()],
    }
    fmt = lambda s: ", ".join(f"{_line_label(store, ac)} {v:.2f}" for ac, v in s.items())  # noqa: E731
    return {
        "summary": f"{gene} expression across {record['n_lines']} lines, log2(FPKM+1): median {record['median']}, "
                   f"range {record['min']} to {record['max']}.\nHighest: {fmt(top)}.\nLowest: {fmt(bottom)}.",
        "evidence": [record],
    }


def drug_targets_in_line(store: FeatureStore, drug: str, line: str) -> dict[str, Any]:
    """Status of each of a drug's target genes in one cell line."""
    drug_id, ac = store.resolve_drug(drug), store.resolve_line(line)
    genes = _target_genes(store, drug_id)
    head = f"{store.drug_name(drug_id)} targets in {_line_label(store, ac)}"
    if not genes:
        return {"summary": f"{head}: the drug has no gene-level target in ChEMBL (e.g. it acts on DNA).",
                "evidence": [{"drug_id": drug_id, "cellosaurus_ac": ac, "targets": []}]}
    rows, text = [], [head + ":"]
    for gene in genes:
        row: dict[str, Any] = {"gene": gene}
        try:
            values = store.expression(gene)
            row["log2_fpkm1"] = round(float(values[ac]), 2)
            row["expr_tertile"] = rules.tertiles(values)[ac]
        except ToolError:
            row["expr_tertile"] = None
        if gene in store.mutation_genes:
            row["mutated"] = int(store.mutation(gene)[ac])
        try:
            cn = store.copy_number(gene)[ac]
            row["cn_log2"] = None if pd.isna(cn) else round(float(cn), 2)
        except ToolError:
            pass
        rows.append(row)
        bits = [f"expression {row.get('log2_fpkm1', 'n/a')} ({row['expr_tertile'] or 'no data'})"]
        if "mutated" in row:
            bits.append("mutated" if row["mutated"] else "not mutated")
        if row.get("cn_log2") is not None:
            bits.append(f"copy-number log2 {row['cn_log2']}")
        text.append(f"- {gene}: " + "; ".join(bits))
    return {"summary": "\n".join(text), "evidence": [{"drug_id": drug_id, "cellosaurus_ac": ac, "targets": rows}]}


def shared_targets(store: FeatureStore, drug_a: str, drug_b: str) -> dict[str, Any]:
    """Overlap and difference between two drugs' targets and mechanisms."""
    ids = [store.resolve_drug(drug_a), store.resolve_drug(drug_b)]
    if ids[0] == ids[1]:
        raise ToolError("drug_a and drug_b resolve to the same drug")
    genes = [set(_target_genes(store, i)) for i in ids]
    moa = [sorted(set(_targets(store, i)["mechanism_of_action"].dropna())) for i in ids]
    names = [store.drug_name(i) for i in ids]
    shared = sorted(genes[0] & genes[1])
    record = {
        "drug_a": ids[0], "drug_b": ids[1], "shared_genes": shared,
        "only_a": sorted(genes[0] - genes[1]), "only_b": sorted(genes[1] - genes[0]),
        "mechanisms_a": moa[0], "mechanisms_b": moa[1],
    }
    show = lambda xs: ", ".join(xs) or "none"  # noqa: E731
    return {
        "summary": f"{names[0]} vs {names[1]}: {len(shared)} shared target gene(s): {show(shared)}.\n"
                   f"Only {names[0]}: {show(record['only_a'])}. Mechanisms: {show(moa[0])}.\n"
                   f"Only {names[1]}: {show(record['only_b'])}. Mechanisms: {show(moa[1])}.",
        "evidence": [record],
    }
