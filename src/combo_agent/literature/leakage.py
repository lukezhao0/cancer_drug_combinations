"""The literature leakage filter (design section 10, route 2).

Every document `search_literature` retrieves passes through `LeakageFilter.check`. A document is
blocked if any rule fires; every rule that fires is recorded, so the audit log says why.

    source_paper    the PMID is a curated source of the pool's screens (`leakage_sources.csv`)
    cites_source    the record cites a source whose citers are blocked: PubMed's cited-by links
                    (built once into `blocklist.parquet`) or the record's own reference list
    screen_name     the text names a screen or a database that redistributes it (NCI-ALMANAC,
                    DrugComb, O'Neil, AZ-DREAM, SynergxDB, ...) or an NCI-60 combination screen
    pair_and_line   the text names both drugs of a pool pair and a pool cell line
    synergy_ml      a machine-learning paper that predicts drug synergy or combination response:
                    nearly all of them train on these public screens, often without naming them
                    in the abstract (DeepDDS is the case that showed this rule was needed)
    strict_date     optional: published in or after 2017 (NCI-ALMANAC), or with no year

Build (verifies the curated PMIDs against PubMed and collects the citers):

    uv run python -m combo_agent.literature.leakage build
    COMBO_OFFLINE=1 uv run python -m combo_agent.literature.leakage build   # replay, byte-identical
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import platform
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from combo_agent.data import paths
from combo_agent.lab.splits import MAXDATE, TERMS_CSV, load_terms
from combo_agent.literature.pubmed import Doc, PubMed

SOURCES_CSV = Path(__file__).resolve().parents[1] / "data" / "curated" / "leakage_sources.csv"
#: NCI-ALMANAC's publication year: strict mode keeps only documents published before it.
STRICT_BEFORE = 2017
REASONS = ("source_paper", "cites_source", "screen_name", "pair_and_line", "synergy_ml", "strict_date")

#: Line names that are ordinary words or codes; they count only next to "cell(s)"/"line(s)".
AMBIGUOUS_LINES = frozenset({"SR", "M14"})


class LeakageError(RuntimeError):
    pass


# -- curated sources ------------------------------------------------------------------------------

def load_sources(path: Path = SOURCES_CSV) -> list[dict[str, Any]]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["block_citers"] = r["block_citers"].strip() == "1"
        if not r["pmid"].isdigit():
            raise LeakageError(f"bad PMID in {path.name}: {r['pmid']!r}")
    return rows


# -- text rules -----------------------------------------------------------------------------------

_COMBO = r"(?:synerg\w*|antagon\w*|drug[\s-]+combinations?|combination\s+screen\w*|drug\s+pairs?|combinations?)"

#: (name, pattern, needs a combination word somewhere in the text)
SCREEN_PATTERNS: tuple[tuple[str, re.Pattern[str], bool], ...] = (
    ("NCI-ALMANAC", re.compile(r"\bNCI[\s-]*ALMANAC\b", re.I), False),
    ("ALMANAC", re.compile(r"\bALMANAC\b"), True),  # upper case only: "Critical Care Almanac" is not it
    ("DrugComb", re.compile(r"\bDrug[\s-]?Comb(?:DB|\s+portal)?\b", re.I), False),
    ("SynergxDB", re.compile(r"\bSYNERGxDB\b", re.I), False),
    ("O'Neil", re.compile(r"\bO['’`]\s?Neil\b", re.I), True),
    ("AZ-DREAM", re.compile(r"\bAZ[\s-]*DREAM\b|AstraZeneca[\s–-]+Sanger|\bDREAM\b[^.]{0,60}\bchallenge", re.I), True),
    ("NCI-60 combination", re.compile(r"\bNCI[\s-]*60\b|\bNCI60\b", re.I), True),
)
_COMBO_RE = re.compile(_COMBO, re.I)

_SYN_TARGET = r"(?:synerg\w*|drug[\s-]+combinations?|combination\s+(?:response|effects?|sensitivity)|drug[\s-]+drug\s+synergy)"
_PREDICT_SYN = re.compile(rf"\bpredict\w*[^.]{{0,100}}{_SYN_TARGET}|{_SYN_TARGET}[^.]{{0,60}}\bpredict\w*", re.I)
_ML = re.compile(r"deep[\s-]+learning|machine[\s-]+learning|neural\s+networks?|graph\s+(?:neural|attention|convolutional)"
                 r"|transformer|random\s+forest|gradient\s+boost\w*|\bXGBoost\b|benchmark\s+data\s*sets?"
                 r"|computational\s+(?:model|method|framework|screening)\w*|\bin\s+silico\b", re.I)


def is_synergy_ml(text: str) -> bool:
    return bool(_PREDICT_SYN.search(text) and _ML.search(text))


def screen_names(text: str) -> list[str]:
    has_combo = bool(_COMBO_RE.search(text))
    return [name for name, pat, needs_combo in SCREEN_PATTERNS if pat.search(text) and (has_combo or not needs_combo)]


def _drug_pattern(terms: list[str]) -> re.Pattern[str]:
    alts = sorted({re.escape(t) for t in terms}, key=len, reverse=True)
    # (?<![\w-]) / (?![\w-]): "5-FU" must not match inside "15-FU1"; hyphens count as word characters.
    return re.compile(r"(?<![\w-])(?:" + "|".join(alts) + r")(?![\w-])", re.I)


def _line_tokens(name: str) -> list[str]:
    base = re.sub(r"\(.*?\)", " ", name)
    return re.findall(r"[A-Za-z]+|\d+", base)


def line_pattern(name: str) -> re.Pattern[str]:
    """Regex for a cell-line name and its spelling variants ("MDA-MB-231", "MDA MB 231", "MDAMB231")."""
    toks = _line_tokens(name)
    parts = [re.escape(t) if not t.isdigit() else t for t in toks]
    if name == "786-O":
        parts[-1] = "[O0]"  # "786-0" is a common misspelling
    if name == "U-251MG":
        parts[-1] = "(?:MG)?"  # usually written "U251"
    body = r"[\s-]?".join(parts)
    paren = re.search(r"\((.*?)\)", name)
    if paren:
        body += r"(?:\s?\(" + re.escape(paren.group(1)) + r"\))?"
    if name in AMBIGUOUS_LINES:
        return re.compile(rf"(?<![\w-]){body}(?![\w-])(?=\s+(?:cells?|cell\s+lines?|leukemia|melanoma)\b)"
                          rf"|(?:cell\s+lines?|cells)\s+{body}(?![\w-])")
    return re.compile(rf"(?<![\w-]){body}(?![\w-])", re.I)


@dataclass(frozen=True)
class Decision:
    pmid: str
    kept: bool
    reasons: tuple[str, ...]
    matches: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"pmid": self.pmid, "kept": self.kept, "reasons": list(self.reasons), "matches": self.matches}


class LeakageFilter:
    """Decides, for each retrieved document, whether the agent may see it."""

    def __init__(self, sources: list[dict[str, Any]], citers: dict[str, list[str]],
                 drug_terms: dict[str, list[str]], line_names: dict[str, str],
                 pool_pairs: Iterable[tuple[str, str]], strict: bool = False) -> None:
        self.sources = {r["pmid"]: r for r in sources}
        self.cite_roots = {p for p, r in self.sources.items() if r["block_citers"]}
        #: citing PMID -> the source PMIDs it cites
        self.citers = citers
        self.strict = strict
        self._drugs = {d: _drug_pattern(t) for d, t in drug_terms.items()}
        self._lines = {ac: (name, line_pattern(name)) for ac, name in line_names.items()}
        self._pairs = {tuple(sorted(p)) for p in pool_pairs}

    # rules --------------------------------------------------------------------------------------
    def drugs_in(self, text: str) -> list[str]:
        return sorted(d for d, pat in self._drugs.items() if pat.search(text))

    def lines_in(self, text: str) -> list[str]:
        return sorted(name for name, pat in self._lines.values() if pat.search(text))

    def pool_pairs_in(self, drugs: list[str]) -> list[tuple[str, str]]:
        ds = sorted(drugs)
        return [(a, b) for i, a in enumerate(ds) for b in ds[i + 1:] if (a, b) in self._pairs]

    def check(self, doc: Doc) -> Decision:
        reasons: list[str] = []
        matches: dict[str, Any] = {}
        if doc.pmid in self.sources:
            reasons.append("source_paper")
            matches["source"] = self.sources[doc.pmid]["short"]
        cited = set(self.citers.get(doc.pmid, ())) | (set(doc.references) & self.cite_roots)
        if cited:
            reasons.append("cites_source")
            matches["cites"] = sorted(self.sources[p]["short"] for p in cited)
        text = doc.text
        screens = screen_names(text)
        if screens:
            reasons.append("screen_name")
            matches["screens"] = screens
        drugs = self.drugs_in(text)
        pairs = self.pool_pairs_in(drugs) if len(drugs) >= 2 else []
        if pairs:
            lines = self.lines_in(text)
            if lines:
                reasons.append("pair_and_line")
                matches["pairs"] = [list(p) for p in pairs[:5]]
                matches["n_pairs"] = len(pairs)
                matches["lines"] = lines[:10]
        if is_synergy_ml(text):
            reasons.append("synergy_ml")
        if self.strict and (doc.year is None or doc.year >= STRICT_BEFORE):
            reasons.append("strict_date")
            matches["year"] = doc.year
        return Decision(doc.pmid, not reasons, tuple(reasons), matches)

    def filter(self, docs: Iterable[Doc]) -> list[Decision]:
        return [self.check(d) for d in docs]

    # construction -------------------------------------------------------------------------------
    @classmethod
    def from_build(cls, m5_dir: Path | None = None, processed: Path | None = None, strict: bool = False,
                   sources_path: Path = SOURCES_CSV, terms_path: Path = TERMS_CSV) -> LeakageFilter:
        processed = processed or paths.processed_dir()
        m5_dir = m5_dir or processed / "m5"
        bl_path = m5_dir / "blocklist.parquet"
        if not bl_path.exists():
            raise LeakageError(f"{bl_path} is missing; run `python -m combo_agent.literature.leakage build`")
        bl = pd.read_parquet(bl_path)
        citers: dict[str, list[str]] = {}
        for pmid, src in bl.loc[bl.reason == "cites_source", ["pmid", "via"]].itertuples(index=False):
            citers.setdefault(pmid, []).append(src)
        return cls(load_sources(sources_path), citers, load_terms(terms_path), _line_names(processed),
                   _pool_pairs(processed), strict=strict)


def _line_names(processed: Path) -> dict[str, str]:
    cells = pd.read_parquet(processed / "cells.parquet", columns=["cellosaurus_ac", "cellosaurus_name"])
    return dict(cells.itertuples(index=False))


def _pool_pairs(processed: Path) -> list[tuple[str, str]]:
    pairs = pd.read_parquet(processed / "m2" / "pair_fame.parquet", columns=["drug_a", "drug_b"])
    return list(pairs.itertuples(index=False, name=None))


# -- build ----------------------------------------------------------------------------------------

def _norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]", "", t.lower())


def verify_sources(sources: list[dict[str, Any]], pubmed: PubMed) -> None:
    """Fail if a curated PMID does not resolve to the curated title (a typo would silently unblock it)."""
    got = pubmed.summaries(r["pmid"] for r in sources)
    bad = []
    for r in sources:
        s = got.get(r["pmid"])
        if s is None or _norm_title(s.get("title", "")) != _norm_title(r["title"]):
            bad.append(f"{r['pmid']} ({r['short']}): PubMed has {None if s is None else s.get('title')!r}")
    if bad:
        raise LeakageError("curated leakage sources do not match PubMed:\n  " + "\n  ".join(bad))


def build_blocklist(sources: list[dict[str, Any]], pubmed: PubMed) -> pd.DataFrame:
    rows = [{"pmid": r["pmid"], "reason": "source_paper", "via": r["pmid"]} for r in sources]
    for r in sources:
        if r["block_citers"]:
            rows += [{"pmid": p, "reason": "cites_source", "via": r["pmid"]} for p in pubmed.cited_by(r["pmid"])]
    df = pd.DataFrame(rows, columns=["pmid", "reason", "via"]).drop_duplicates()
    order = df.assign(_via=df.via.astype(int), _pmid=df.pmid.astype(int)).sort_values(["reason", "_via", "_pmid"])
    return order.drop(columns=["_via", "_pmid"]).reset_index(drop=True)


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def card(sources: list[dict[str, Any]], bl: pd.DataFrame) -> str:
    cites = bl[bl.reason == "cites_source"]
    lines = ["# M5 leakage blocklist", "",
             "Generated by `combo_agent.literature.leakage build`. PubMed citation links as of the "
             f"build; searches use the M2 date cutoff {MAXDATE}.", "",
             "| PMID | Source | Role | Year | Citers blocked | Citing records |", "|:--|:--|:--|--:|:--|--:|"]
    for r in sources:
        n = int((cites.via == r["pmid"]).sum())
        lines.append(f"| {r['pmid']} | {r['short']} | {r['role']} | {r['year']} | "
                     f"{'yes' if r['block_citers'] else 'no'} | {n if r['block_citers'] else '-'} |")
    lines += ["", f"Distinct PMIDs blocked by citation: {cites.pmid.nunique():,}. "
              f"Total distinct PMIDs on the list: {bl.pmid.nunique():,}.", "",
              "Text rules (`screen_name`, `pair_and_line`) and the record's own reference list are applied "
              "at query time, so documents missing from PubMed's cited-by links are still caught when they "
              "name a screen, cite a source in their reference list, or name a pool pair with a pool line.", ""]
    return "\n".join(lines)


def build(out_dir: Path | None = None, pubmed: PubMed | None = None, sources_path: Path = SOURCES_CSV,
          log=print) -> dict[str, Path]:
    out_dir = out_dir or paths.processed_dir() / "m5"
    pubmed = pubmed or PubMed()
    sources = load_sources(sources_path)
    verify_sources(sources, pubmed)
    log(f"verified {len(sources)} curated sources against PubMed")
    bl = build_blocklist(sources, pubmed)
    log(f"blocklist: {bl.pmid.nunique():,} distinct PMIDs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {"blocklist": out_dir / "blocklist.parquet", "card": out_dir / "m5_blocklist_card.md"}
    bl.to_parquet(out["blocklist"], index=False)
    out["card"].write_text(card(sources, bl))
    manifest = {
        "inputs": {"sources": {"file": sources_path.name, "sha256": _sha(sources_path)}},
        "outputs": {k: {"file": p.name, "sha256": _sha(p)} for k, p in sorted(out.items())},
        "pubmed": {"maxdate": MAXDATE, **pubmed.cache_stats},
        "built": dt.date.today().isoformat(),
        "versions": {"python": platform.python_version(), "pandas": pd.__version__},
    }
    out["manifest"] = out_dir / "manifest.json"
    out["manifest"].write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m combo_agent.literature.leakage")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="verify sources, collect citers, write data_cache/processed/m5/")
    b.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    if args.cmd == "build":
        for k, p in build(args.out).items():
            print(f"wrote {k}: {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
