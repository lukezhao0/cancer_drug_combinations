"""`search_literature`: the agent's literature tool, behind the leakage filter (design section 10).

One call = esearch (over-fetching, because the filter removes some hits), efetch, the filter, and
exactly one audit record. The record lists every PMID retrieved, whether it was shown, and why it
was blocked. The record is written before anything is returned, so a call that returns documents
always leaves a log line, and a call that fails leaves one with the error.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import threading
from pathlib import Path
from typing import Any

from combo_agent.literature.leakage import LeakageFilter
from combo_agent.literature.pubmed import Doc, PubMed

DEFAULT_K = 8
MAX_K = 20
OVERFETCH = 4
ABSTRACT_CHARS = 1500


class LiteratureSearch:
    """Backend for the `literature` tools. One per run; the audit log goes in the run folder."""

    def __init__(self, pubmed: PubMed, leakage: LeakageFilter, audit_path: Path,
                 abstract_chars: int = ABSTRACT_CHARS, clock=None) -> None:
        self.pubmed = pubmed
        self.leakage = leakage
        self.audit_path = Path(audit_path)
        self.abstract_chars = abstract_chars
        self._clock = clock or (lambda: dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
        self._lock = threading.Lock()
        self._calls = 0

    @classmethod
    def default(cls, audit_path: Path, strict: bool = False) -> LiteratureSearch:
        return cls(PubMed(), LeakageFilter.from_build(strict=strict), audit_path)

    def _write(self, record: dict[str, Any]) -> None:
        with self._lock:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)
            with self.audit_path.open("a") as f:
                f.write(json.dumps(record, sort_keys=True) + "\n")
                f.flush()
                os.fsync(f.fileno())

    def _show(self, doc: Doc) -> dict[str, Any]:
        abstract = doc.abstract
        if len(abstract) > self.abstract_chars:
            abstract = abstract[: self.abstract_chars].rsplit(" ", 1)[0] + " ..."
        return {"pmid": doc.pmid, "year": doc.year, "journal": doc.journal, "title": doc.title, "abstract": abstract}

    def search(self, query: str, k: int = DEFAULT_K) -> dict[str, Any]:
        """Up to `k` abstracts that pass the leakage filter, most relevant first."""
        k = max(1, min(int(k), MAX_K))
        before = dict(self.pubmed.cache_stats)
        with self._lock:
            self._calls += 1
            call = self._calls
        record: dict[str, Any] = {"call": call, "time": self._clock(), "query": query, "k": k,
                                  "maxdate": self.pubmed.maxdate, "strict": self.leakage.strict}
        try:
            pmids = self.pubmed.search(query, retmax=k * OVERFETCH)
            docs = self.pubmed.fetch(pmids)
            decisions = self.leakage.filter(docs)
        except Exception as e:
            record["error"] = f"{type(e).__name__}: {e}"
            self._write(record)
            raise
        kept = [d for d, dec in zip(docs, decisions) if dec.kept][:k]
        after = self.pubmed.cache_stats
        record.update({
            "retrieved": [{**dec.to_dict(), "year": doc.year, "title": doc.title[:200]}
                          for doc, dec in zip(docs, decisions)],
            "not_fetched": sorted(set(pmids) - {d.pmid for d in docs}, key=int),
            "returned": [d.pmid for d in kept],
            "n_retrieved": len(docs), "n_blocked": sum(not d.kept for d in decisions),
            "cache": {"hits": after["hits"] - before["hits"], "misses": after["misses"] - before["misses"]},
        })
        self._write(record)
        shown = [self._show(d) for d in kept]
        blocked = record["n_blocked"]
        lines = [f"{len(shown)} abstract(s) for {query!r} ({blocked} of {len(docs)} retrieved were withheld "
                 "by the leakage filter)."]
        for s in shown:
            lines.append(f"\n[PMID {s['pmid']}, {s['year']}, {s['journal']}] {s['title']}\n{s['abstract']}")
        return {"summary": "\n".join(lines), "evidence": shown}


def search_literature(backend: LiteratureSearch, query: str, k: int = DEFAULT_K) -> dict[str, Any]:
    """Tool entry point (`tools.registry`)."""
    return backend.search(query, k)
