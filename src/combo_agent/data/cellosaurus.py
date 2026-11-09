"""A thin client for the Cellosaurus REST API (https://api.cellosaurus.org), plus entry accessors.

Requests go through `CachedJsonClient`, so every response is stored on disk and a build can be
replayed offline. Field shapes below were read from live responses on 2026-11-08:

* `name-list`: `[{"type": "identifier" | "synonym", "value": ...}]`
* `misspelling-list`: `[{"misspelling-name": ..., "misspelling-note": ...}]`. It is only present in
  full entries; no `fields` code selects it, so it is used only when present.
* `derived-from`: `[{"accession": "CVCL_...", "label": ...}]`. Lines from the same individual point
  at a shared placeholder parent (e.g. HCT 15 -> CVCL_3449 "__Parent_cell_line_of_DLD-1/...").
* `xref-list`: `[{"database": "DepMap", "accession": "ACH-..."}, ...]`
* `comment-list`: `[{"category": "Part of" | "Problematic cell line" | ..., "value": ...}]`
"""

from __future__ import annotations

import re
import urllib.parse
from typing import Any

from combo_agent.data.api_cache import CachedJsonClient

API = "https://api.cellosaurus.org"
HUMAN_TAXID = "9606"
NCI60_PANEL = "NCI-60 cancer cell line panel"
#: Fields requested. A full entry is ~200 kB (mostly references and STR profiles), and large bodies
#: were repeatedly cut short in transit (IncompleteRead, 2026-11-08). These cover everything used
#: here: with them an entry is ~35 kB, and the 61-line panel search is ~74 kB.
#: Codes: id/ac/sy = names, ca = category, cc = comments, ox = species, hi = derived-from,
#: oi = same individual, dr = cross-references, di = diseases, sx = sex.
SEARCH_FIELDS = "id,ac,sy,ox"
ENTRY_FIELDS = "id,ac,sy,ca,cc,ox,hi,oi,dr,di,sx"

Entry = dict[str, Any]


def normalize_name(name: str) -> str:
    """Upper-case and drop everything but letters and digits: 'NCI/ADR-RES' -> 'NCIADRRES'."""
    return re.sub(r"[^A-Z0-9]", "", name.upper())


class CellosaurusApi:
    def __init__(self, client: CachedJsonClient):
        self._c = client

    def search(self, query: str, rows: int = 200) -> list[Entry]:
        url = f"{API}/search/cell-line?" + urllib.parse.urlencode(
            {"q": query, "format": "json", "rows": rows, "fields": SEARCH_FIELDS})
        body = self._c.get(url)
        return (body or {}).get("Cellosaurus", {}).get("cell-line-list", [])

    def entry(self, accession: str) -> Entry | None:
        body = self._c.get(f"{API}/cell-line/{urllib.parse.quote(accession)}?"
                           + urllib.parse.urlencode({"format": "json", "fields": ENTRY_FIELDS}))
        lines = (body or {}).get("Cellosaurus", {}).get("cell-line-list", [])
        return lines[0] if lines else None

    def nci60_panel(self) -> list[Entry]:
        """Every entry Cellosaurus marks as 'Part of: NCI-60 cancer cell line panel'."""
        return self.search(f'cc:"{NCI60_PANEL}"')


# -- entry accessors ------------------------------------------------------------------------------


def accession(e: Entry) -> str:
    return next(a["value"] for a in e["accession-list"] if a["type"] == "primary")


def identifier(e: Entry) -> str:
    return next(n["value"] for n in e["name-list"] if n["type"] == "identifier")


def names(e: Entry) -> list[tuple[str, str]]:
    """(kind, name) for the identifier, every synonym, and every recorded misspelling."""
    out = [(n["type"], n["value"]) for n in e.get("name-list", [])]
    out += [("misspelling", m["misspelling-name"]) for m in e.get("misspelling-list") or []]
    return out


def comments(e: Entry, category: str) -> list[str]:
    return [c["value"] for c in e.get("comment-list") or [] if c.get("category") == category]


def in_nci60_panel(e: Entry) -> bool:
    return any(NCI60_PANEL in v for v in comments(e, "Part of"))


def is_human(e: Entry) -> bool:
    return any(s.get("accession") == HUMAN_TAXID for s in e.get("species-list") or [])


def depmap_ids(e: Entry) -> list[str]:
    return sorted({x["accession"] for x in e.get("xref-list") or [] if x.get("database") == "DepMap"})


def parents(e: Entry) -> list[str]:
    """Accessions this line is derived from, or shares an individual with."""
    acc = [d["accession"] for d in e.get("derived-from") or []]
    acc += [d["accession"] for d in e.get("same-origin-as") or [] if "accession" in d]
    return sorted(set(acc))


def diseases(e: Entry) -> list[tuple[str, str]]:
    """(NCIt accession, label) pairs."""
    return [(d["accession"], d["label"]) for d in e.get("disease-list") or [] if d.get("database") == "NCIt"]
