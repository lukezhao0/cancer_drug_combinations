"""Step 1d: one cell line, one identity, and its family (target_design.md section 4, step 2).

Every cell-line label in the TDC table is mapped to a **Cellosaurus accession** (CVCL_...), the
merge key for lines. Matching is exact, after normalizing case and punctuation, and runs against
the 61 entries Cellosaurus tags "Part of: NCI-60 cancer cell line panel":

* A plain name search is unusable for popular lines. "A549" or "MCF7" return 100+ derived and
  edited lines that list the parent's name as a synonym, and the parent can fall off the first page.
* Restricting to the panel makes every one of our 59 labels match exactly one entry (checked
  2026-11-08).
* A label outside the panel falls back to an exact-name search and is marked `name_search`, so it
  gets reviewed.

The resolution is written to a **committed, reviewable table**, `curated/cell_lines.csv`:
* `propose` regenerates its automatic rows and keeps the rows marked `manual`;
* `build` reads only that table, re-fetches each entry, re-checks the name match, and derives:

  - `depmap_id` from the Cellosaurus DepMap cross-reference. The design drops lines without one.
  - `keep` / `drop_reason`. Drop rules: not a cancer cell line, not human, no DepMap ID, or more
    than one DepMap ID.
  - `family_id`: lines that share any ancestor in Cellosaurus's `derived-from` graph, or that are
    derived from one another, share a family. That covers contaminated lines (NCI/ADR-RES is an
    OVCAR-8 derivative, MDA-MB-435 is an M14 derivative, SNB-19 is a U-251MG derivative) and lines
    from the same individual (Cellosaurus links those through a shared placeholder parent). The
    held-out split (step 1g) must keep a family on one side. Families are computed over all lines,
    dropped ones included, so a relation through a dropped line is not lost.
  - disease (NCIt), sex and the "problematic cell line" note, for the data card. Sampling site and
    lineage come from DepMap in step 1f.

  The NCIt disease is a cross-check for that lineage, not a replacement for it.

Outputs, in `data_cache/interim/`:

    line_map.parquet                    one row per source label (59)
    cell_lines.parquet                  one row per Cellosaurus accession, with family and keep flag
    line_harmonization_report.json      counts, families, drops, every check

Every Cellosaurus response is cached in `data_cache/raw/api_cache/cellosaurus/`. `--offline`
replays the build from that cache only.

Run:
    python -m combo_agent.data.harmonize_lines propose [--write] [--offline]
    python -m combo_agent.data.harmonize_lines build [--offline] [--out DIR]
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from combo_agent.data import cellosaurus as cs
from combo_agent.data import ingest_tdc, paths
from combo_agent.data.api_cache import CachedJsonClient

CURATED_CSV = Path(__file__).with_name("curated") / "cell_lines.csv"
#: Two hand-set columns, blank by default:
#: * `depmap_id` picks one ID when Cellosaurus lists several for a line (decision 9: SR, UO-31). It
#:   must be one of the listed IDs.
#: * `depmap_proxy_of` names a Cellosaurus ancestor whose DepMap profile stands in for a line that
#:   has none (decision 10: HL-60(TB) -> HL-60, CVCL_0002). It must be an ancestor in the
#:   derived-from graph and have exactly one DepMap ID. The line is then flagged `depmap_is_proxy`.
#: `build` checks both. They can't be set on the same row.
CSV_COLUMNS = ("source_id", "cellosaurus_ac", "cellosaurus_name", "resolution", "matched_as",
               "depmap_id", "depmap_proxy_of", "note")
#: Automatic outcomes, plus `manual` for a hand-set row.
RESOLUTIONS = ("nci60_panel", "name_search", "ambiguous", "unresolved", "manual")
RESOLVED = ("nci60_panel", "name_search", "manual")
#: How far up the derived-from graph to walk. Real chains are 1-2 deep; this only bounds a cycle.
MAX_ANCESTOR_DEPTH = 6
_KIND_RANK = {"identifier": 0, "synonym": 1, "misspelling": 2}


class HarmonizationError(RuntimeError):
    """The curated table is incomplete or disagrees with Cellosaurus."""


def source_lines(df: pd.DataFrame) -> pd.DataFrame:
    """One row per cell-line label, with its row count, sorted by label."""
    counts = df["Cell_Line_ID"].value_counts()
    return pd.DataFrame({"source_id": counts.index, "n_rows": counts.values}).sort_values(
        "source_id", ignore_index=True)


def match_kind(label: str, entry: cs.Entry) -> str | None:
    """The best kind of name in `entry` equal to `label` after normalizing, or None."""
    key = cs.normalize_name(label)
    kinds = [k for k, n in cs.names(entry) if cs.normalize_name(n) == key]
    return min(kinds, key=_KIND_RANK.get) if kinds else None


# -- propose --------------------------------------------------------------------------------------


@dataclass
class Proposal:
    source_id: str
    resolution: str
    cellosaurus_ac: str = ""
    cellosaurus_name: str = ""
    matched_as: str = ""
    depmap_id: str = ""
    depmap_proxy_of: str = ""
    note: str = ""

    def row(self) -> dict[str, str]:
        return {c: getattr(self, c) for c in CSV_COLUMNS}


def propose_one(label: str, panel: list[cs.Entry], api: cs.CellosaurusApi) -> Proposal:
    hits = [(e, k) for e in panel if (k := match_kind(label, e))]
    route = "nci60_panel"
    if not hits:
        route = "name_search"
        q = label.replace('"', '\\"')
        hits = [(e, k) for e in api.search(f'id:"{q}" OR sy:"{q}"')
                if (k := match_kind(label, e)) and cs.is_human(e)]
    if len(hits) == 1:
        e, kind = hits[0]
        note = "" if route == "nci60_panel" else "outside the NCI-60 panel; review"
        return Proposal(label, route, cellosaurus_ac=cs.accession(e),
                        cellosaurus_name=cs.identifier(e), matched_as=kind, note=note)
    if not hits:
        return Proposal(label, "unresolved", note="no exact name match in Cellosaurus")
    return Proposal(label, "ambiguous", note="candidates: " + "; ".join(
        f"{cs.accession(e)} {cs.identifier(e)} ({k})" for e, k in hits))


def proposal_rows(lines: pd.DataFrame, api: cs.CellosaurusApi) -> list[dict[str, str]]:
    panel = api.nci60_panel()
    return [propose_one(label, panel, api).row() for label in lines["source_id"]]


def read_curated(path: Path = CURATED_CSV) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if rows and tuple(rows[0]) != CSV_COLUMNS:
        raise HarmonizationError(f"{path}: columns {list(rows[0])}, expected {list(CSV_COLUMNS)}")
    return rows


def render_csv(rows: list[dict[str, str]]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CSV_COLUMNS, lineterminator="\n")
    w.writeheader()
    w.writerows(sorted(rows, key=lambda r: r["source_id"]))
    return buf.getvalue()


def merge_with_manual(auto: list[dict[str, str]], curated: list[dict[str, str]]) -> list[dict[str, str]]:
    """Automatic rows, except where the curated table has a `manual` row for that label.

    Hand-set `depmap_id` / `depmap_proxy_of` values are kept on an automatic row as long as the
    accession is unchanged. Notes on such rows are kept too.
    """
    hand = ("depmap_id", "depmap_proxy_of")
    by_label = {r["source_id"]: r for r in curated}
    out = []
    for r in auto:
        old = by_label.get(r["source_id"])
        if old and old["resolution"] == "manual":
            out.append(old)
        elif old and any(old.get(h) for h in hand) and old["cellosaurus_ac"] == r["cellosaurus_ac"]:
            out.append({**r, **{h: old.get(h, "") for h in hand}, "note": old.get("note", "")})
        else:
            out.append(r)
    return out


# -- build ----------------------------------------------------------------------------------------


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class BuildResult:
    line_map: pd.DataFrame
    cell_lines: pd.DataFrame
    report: dict[str, Any]
    checks: list[Check] = field(default_factory=list)


def _fetch(api: cs.CellosaurusApi, ac: str, cache: dict[str, cs.Entry | None]) -> cs.Entry | None:
    if ac not in cache:
        cache[ac] = api.entry(ac)
    return cache[ac]


def ancestors(api: cs.CellosaurusApi, ac: str, cache: dict[str, cs.Entry | None]) -> set[str]:
    """Every accession reachable through derived-from / same-origin-as links, excluding `ac`."""
    seen: set[str] = set()
    frontier = [ac]
    for _ in range(MAX_ANCESTOR_DEPTH):
        nxt = []
        for a in frontier:
            e = _fetch(api, a, cache)
            for p in cs.parents(e) if e else []:
                if p not in seen and p != ac:
                    seen.add(p)
                    nxt.append(p)
        if not nxt:
            break
        frontier = nxt
    return seen


def families(members: dict[str, set[str]]) -> dict[str, str]:
    """Union-find over lines that share any accession in (self + ancestors).

    `members` maps each line's accession to its ancestor set. Returns accession -> family id, where
    the id is the smallest member accession, so ids are stable whatever order lines arrive in.
    """
    parent = {ac: ac for ac in members}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    owner: dict[str, str] = {}
    for ac in sorted(members):
        for key in sorted({ac} | members[ac]):
            if key in owner:
                a, b = find(owner[key]), find(ac)
                if a != b:
                    parent[max(a, b)] = min(a, b)
            else:
                owner[key] = ac
    groups: dict[str, list[str]] = defaultdict(list)
    for ac in members:
        groups[find(ac)].append(ac)
    return {ac: f"FAM_{min(g)}" for g in groups.values() for ac in g}


def build(df: pd.DataFrame, rows: list[dict[str, str]], api: cs.CellosaurusApi) -> BuildResult:
    lines = source_lines(df)
    by_label = {r["source_id"]: r for r in rows}
    checks: list[Check] = []

    missing = sorted(set(lines.source_id) - set(by_label))
    extra = sorted(set(by_label) - set(lines.source_id))
    unresolved = sorted(r["source_id"] for r in rows if r["resolution"] not in RESOLVED)
    checks.append(Check("every label has a curated row", not missing, ", ".join(missing)))
    checks.append(Check("no curated row for an absent label", not extra, ", ".join(extra)))
    checks.append(Check("every row is resolved", not unresolved, ", ".join(unresolved)))
    if missing or unresolved:
        raise HarmonizationError("; ".join(f"{c.name}: {c.detail}" for c in checks if not c.ok))

    cache: dict[str, cs.Entry | None] = {}
    map_rows, problems = [], []
    for label in lines.source_id:
        r = by_label[label]
        e = _fetch(api, r["cellosaurus_ac"], cache)
        if e is None:
            problems.append(f"{label}: {r['cellosaurus_ac']} not found in Cellosaurus")
            continue
        primary = cs.accession(e)
        if primary != r["cellosaurus_ac"]:
            problems.append(f"{label}: {r['cellosaurus_ac']} is now secondary to {primary}; update the CSV")
        kind = match_kind(label, e)
        if r["resolution"] != "manual" and kind is None:
            problems.append(f"{label}: no name in {primary} matches after normalizing")
        map_rows.append({"source_id": label, "cellosaurus_ac": primary, "match_kind": kind or "manual",
                         "resolution": r["resolution"]})
    checks.append(Check("every curated accession exists and its names match", not problems,
                        "; ".join(problems)))
    if problems:
        raise HarmonizationError("; ".join(problems))

    line_map = pd.DataFrame(map_rows)
    labels_per_ac = line_map.groupby("cellosaurus_ac")["source_id"].apply(lambda s: sorted(s))
    merged = {ac: ls for ac, ls in labels_per_ac.items() if len(ls) > 1}

    acs = sorted(line_map.cellosaurus_ac.unique())
    anc = {ac: ancestors(api, ac, cache) for ac in acs}
    fam = families(anc)

    chosen: dict[str, str] = {}
    proxy_of: dict[str, str] = {}
    for r in rows:
        if r.get("depmap_id") and r.get("depmap_proxy_of"):
            problems.append(f"{r['source_id']}: depmap_id and depmap_proxy_of are both set")
        for col, target in (("depmap_id", chosen), ("depmap_proxy_of", proxy_of)):
            if r.get(col):
                prev = target.setdefault(r["cellosaurus_ac"], r[col])
                if prev != r[col]:
                    problems.append(f"{r['cellosaurus_ac']}: conflicting {col} {prev} vs {r[col]}")
    cell_rows = []
    for ac in acs:
        e = cache[ac]
        assert e is not None
        listed = cs.depmap_ids(e)
        dep = listed
        if ac in chosen:
            if chosen[ac] not in listed:
                problems.append(f"{ac}: depmap_id {chosen[ac]} is not among Cellosaurus' {listed}")
            dep = [chosen[ac]]
        if ac in proxy_of:
            parent_ac = proxy_of[ac]
            parent = _fetch(api, parent_ac, cache)
            if parent_ac not in anc[ac]:
                problems.append(f"{ac}: depmap_proxy_of {parent_ac} is not an ancestor")
            parent_ids = cs.depmap_ids(parent) if parent else []
            if len(parent_ids) != 1:
                problems.append(f"{ac}: proxy {parent_ac} has DepMap IDs {parent_ids}, need exactly one")
            dep = parent_ids[:1]
        drop = []
        if e.get("category") != "Cancer cell line":
            drop.append("not_cancer_cell_line")
        if not cs.is_human(e):
            drop.append("not_human")
        if not dep:
            drop.append("no_depmap_id")
        elif len(dep) > 1:
            drop.append("multiple_depmap_ids")
        dis = cs.diseases(e)
        cell_rows.append({
            "cellosaurus_ac": ac,
            "cellosaurus_name": cs.identifier(e),
            "source_ids": "|".join(labels_per_ac[ac]),
            "depmap_id": dep[0] if len(dep) == 1 else None,
            "depmap_ids_listed": "|".join(listed),
            "depmap_is_proxy": ac in proxy_of,
            "depmap_proxy_of": proxy_of.get(ac, ""),
            "category": e.get("category"),
            "sex": e.get("sex"),
            "disease_ncit": "|".join(a for a, _ in dis),
            "disease": "|".join(label for _, label in dis),
            "in_nci60_panel": cs.in_nci60_panel(e),
            "problematic": " ".join(cs.comments(e, "Problematic cell line")),
            "derived_from": "|".join(cs.parents(e)),
            "family_id": fam[ac],
            "keep": not drop,
            "drop_reason": "|".join(drop),
        })
    checks.append(Check("hand-set depmap_id / depmap_proxy_of values are valid", not problems,
                        "; ".join(problems)))
    if problems:
        raise HarmonizationError("; ".join(problems))
    cell_lines = pd.DataFrame(cell_rows)
    line_map = line_map.merge(cell_lines[["cellosaurus_ac", "depmap_id", "depmap_is_proxy", "family_id",
                                          "keep", "drop_reason"]],
                              on="cellosaurus_ac").sort_values("source_id", ignore_index=True)

    fam_groups = {f: sorted(g.cellosaurus_name) for f, g in cell_lines.groupby("family_id") if len(g) > 1}
    kept = cell_lines[cell_lines.keep]
    kept_rows = int(lines.set_index("source_id").loc[line_map[line_map.keep].source_id, "n_rows"].sum())
    checks.append(Check("kept lines have exactly one DepMap ID", bool(kept.depmap_id.notna().all())))
    checks.append(Check("DepMap IDs are unique among kept lines", bool(kept.depmap_id.is_unique)))
    if not all(c.ok for c in checks):
        raise HarmonizationError("; ".join(f"{c.name}: {c.detail}" for c in checks if not c.ok))

    report = {
        "source_labels": len(lines),
        "cellosaurus_lines": len(cell_lines),
        "labels_merged_into_one_line": merged,
        "resolutions": line_map.resolution.value_counts().sort_index().to_dict(),
        "match_kinds": line_map.match_kind.value_counts().sort_index().to_dict(),
        "kept_lines": int(cell_lines.keep.sum()),
        "dropped": {r.cellosaurus_name: r.drop_reason for r in cell_lines.itertuples() if not r.keep},
        "needs_depmap_choice": {r.cellosaurus_name: r.depmap_ids_listed for r in cell_lines.itertuples()
                                if r.drop_reason == "multiple_depmap_ids"},
        "depmap_chosen": {r.cellosaurus_name: r.depmap_id for r in cell_lines.itertuples()
                          if r.cellosaurus_ac in chosen},
        "depmap_proxies": {r.cellosaurus_name: f"{r.depmap_id} (from {r.depmap_proxy_of})"
                           for r in cell_lines.itertuples() if r.depmap_is_proxy},
        "rows_kept": kept_rows,
        "rows_dropped": int(lines.n_rows.sum()) - kept_rows,
        "families_with_more_than_one_line": fam_groups,
        "problematic": {r.cellosaurus_name: r.problematic for r in cell_lines.itertuples() if r.problematic},
        "not_in_nci60_panel": sorted(cell_lines.loc[~cell_lines.in_nci60_panel, "cellosaurus_name"]),
        "checks": [{"name": c.name, "ok": c.ok, "detail": c.detail} for c in checks],
    }
    return BuildResult(line_map, cell_lines, report, checks)


def write_outputs(result: BuildResult, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "line_map": out_dir / "line_map.parquet",
        "cell_lines": out_dir / "cell_lines.parquet",
        "report": out_dir / "line_harmonization_report.json",
    }
    result.line_map.to_parquet(out["line_map"], index=False)
    result.cell_lines.to_parquet(out["cell_lines"], index=False)
    out["report"].write_text(json.dumps(result.report, indent=2, sort_keys=True) + "\n")
    return out


def make_api(offline: bool | None = None, cache_root: Path | None = None) -> cs.CellosaurusApi:
    root = cache_root or paths.raw_dir() / "api_cache"
    # More attempts than the default: Cellosaurus bodies were cut short in transit often enough
    # (2026-11-08) that 4 tries in a row could fail. Successes are cached, so a rerun resumes.
    return cs.CellosaurusApi(CachedJsonClient(root / "cellosaurus", offline=offline, attempts=8,
                                              min_interval=0.1))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Step 1d: harmonize cell-line identities")
    sub = p.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("propose", help="resolve all labels automatically and diff with the curated CSV")
    pp.add_argument("--write", action="store_true",
                    help=f"write the proposal to {CURATED_CSV.name}, keeping its manual rows")
    pb = sub.add_parser("build", help="verify the curated CSV and write the cell-line tables")
    for q in (pp, pb):
        q.add_argument("--offline", action="store_true", help="use cached API responses only")
    pb.add_argument("--out", type=Path, default=None, help="output dir (default data_cache/interim)")
    args = p.parse_args(argv)

    api = make_api(offline=args.offline or None)
    df = ingest_tdc.load("tdc_drugcomb")
    lines = source_lines(df)
    curated = read_curated() if CURATED_CSV.exists() else []

    if args.cmd == "propose":
        rows = merge_with_manual(proposal_rows(lines, api), curated)
        text = render_csv(rows)
        counts = pd.Series([r["resolution"] for r in rows]).value_counts().sort_index().to_dict()
        print(f"{len(rows)} labels: {counts}")
        for r in rows:
            if r["resolution"] not in ("nci60_panel", "manual"):
                print(f"  {r['source_id']}: {r['resolution']} {r['note']}")
        if args.write:
            CURATED_CSV.parent.mkdir(exist_ok=True)
            CURATED_CSV.write_text(text)
            print(f"wrote {CURATED_CSV}")
        elif curated and render_csv(curated) != text:
            print(f"differs from {CURATED_CSV}; rerun with --write to update")
        return 0

    result = build(df, curated, api)
    out = write_outputs(result, args.out or paths.interim_dir())
    r = result.report
    print(f"{r['source_labels']} labels -> {r['cellosaurus_lines']} lines; kept {r['kept_lines']}, "
          f"dropped {r['dropped']}; {r['rows_dropped']} rows dropped")
    print(f"families with >1 line: {r['families_with_more_than_one_line']}")
    for k, v in out.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
