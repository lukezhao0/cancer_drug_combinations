"""PubMed E-utilities client for M5: search, fetch abstracts, look up citing papers.

Every request goes through `CachedJsonClient`, so a build or a campaign can be replayed offline
(`COMBO_OFFLINE=1`). URLs are cached without the API key (`lab.splits.keyed_fetch` appends it on the
way out), and searches carry the M2 date cutoff (`MAXDATE`), so the answers do not drift as PubMed
grows. efetch returns XML; it is stored as a JSON string body by a text fetcher, so the cache class
needs no XML mode.
"""

from __future__ import annotations

import os
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from combo_agent.data import paths
from combo_agent.data.api_cache import CachedJsonClient, urllib_fetch
from combo_agent.lab.splits import MAXDATE, api_key_from_env, keyed_fetch

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
USER_AGENT = "combo-agent-literature/0.1"
FETCH_BATCH = 200


class PubMedError(RuntimeError):
    """E-utilities answered with something this client cannot use."""


@dataclass(frozen=True)
class Doc:
    """One PubMed record, as much as the filter and the agent need."""

    pmid: str
    title: str
    abstract: str
    year: int | None
    journal: str = ""
    mesh: tuple[str, ...] = ()
    pub_types: tuple[str, ...] = ()
    #: PMIDs in the record's reference list (present for many recent records, not all)
    references: tuple[str, ...] = field(default=(), repr=False)

    @property
    def text(self) -> str:
        return f"{self.title}\n{self.abstract}"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("mesh", "pub_types", "references"):
            d[k] = list(d[k])
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Doc:
        return cls(pmid=str(d["pmid"]), title=d.get("title", ""), abstract=d.get("abstract", ""),
                   year=d.get("year"), journal=d.get("journal", ""), mesh=tuple(d.get("mesh", ())),
                   pub_types=tuple(d.get("pub_types", ())), references=tuple(d.get("references", ())))


def text_fetch(url: str, timeout: float = 60.0) -> tuple[int, Any]:
    """Like `urllib_fetch`, but the body is the raw response text (for efetch XML)."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        if e.code in (400, 404):
            return e.code, None
        raise


def _url(endpoint: str, params: dict[str, Any]) -> str:
    return f"{EUTILS}/{endpoint}.fcgi?{urllib.parse.urlencode({**params, 'tool': 'combo_agent'})}"


def search_url(query: str, retmax: int, maxdate: str = MAXDATE, mindate: str = "1800/01/01") -> str:
    return _url("esearch", {"db": "pubmed", "retmode": "json", "retmax": retmax, "sort": "relevance",
                            "datetype": "pdat", "mindate": mindate, "maxdate": maxdate, "term": query})


def fetch_url(pmids: list[str]) -> str:
    return _url("efetch", {"db": "pubmed", "retmode": "xml", "id": ",".join(pmids)})


def summary_url(pmids: list[str]) -> str:
    return _url("esummary", {"db": "pubmed", "retmode": "json", "id": ",".join(pmids)})


def cited_by_url(pmid: str) -> str:
    return _url("elink", {"dbfrom": "pubmed", "db": "pubmed", "linkname": "pubmed_pubmed_citedin",
                          "retmode": "json", "id": pmid})


# -- XML parsing ----------------------------------------------------------------------------------

def _text(el: ET.Element | None) -> str:
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip() if el is not None else ""


def _year(article: ET.Element) -> int | None:
    for path in (".//Article/Journal/JournalIssue/PubDate/Year", ".//Article/ArticleDate/Year",
                 ".//PubmedData/History/PubMedPubDate[@PubStatus='pubmed']/Year"):
        el = article.find(path)
        if el is not None and (el.text or "").strip().isdigit():
            return int(el.text.strip())
    medline = article.find(".//Article/Journal/JournalIssue/PubDate/MedlineDate")
    m = re.search(r"(1[89]|20)\d\d", medline.text or "") if medline is not None else None
    return int(m.group(0)) if m else None


def parse_articles(xml: str) -> list[Doc]:
    """Parse an efetch PubmedArticleSet. Book records and records without a PMID are skipped."""
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as e:
        raise PubMedError(f"efetch returned unparseable XML: {e}") from e
    docs = []
    for art in root.findall("PubmedArticle"):
        pmid_el = art.find("MedlineCitation/PMID")
        if pmid_el is None or not (pmid_el.text or "").strip():
            continue
        parts = []
        for a in art.findall(".//Article/Abstract/AbstractText"):
            label, body = a.get("Label"), _text(a)
            parts.append(f"{label}: {body}" if label and body else body)
        refs = [i.text.strip() for i in art.findall(".//PubmedData/ReferenceList/Reference/ArticleIdList/ArticleId")
                if i.get("IdType") == "pubmed" and (i.text or "").strip()]
        docs.append(Doc(
            pmid=pmid_el.text.strip(),
            title=_text(art.find(".//Article/ArticleTitle")),
            abstract=" ".join(p for p in parts if p),
            year=_year(art),
            journal=_text(art.find(".//Article/Journal/ISOAbbreviation")) or _text(art.find(".//Article/Journal/Title")),
            mesh=tuple(_text(d) for d in art.findall(".//MeshHeadingList/MeshHeading/DescriptorName")),
            pub_types=tuple(_text(p) for p in art.findall(".//Article/PublicationTypeList/PublicationType")),
            references=tuple(dict.fromkeys(refs)),
        ))
    return docs


# -- client ---------------------------------------------------------------------------------------

def default_cache_dir() -> Path:
    return paths.raw_dir() / "api_cache" / "pubmed_lit"


class PubMed:
    """Cached E-utilities access. One instance per process; requests are serial and rate-limited."""

    def __init__(self, cache_dir: Path | None = None, offline: bool | None = None, api_key: str | None = None,
                 json_fetch: Callable[[str], tuple[int, Any]] = urllib_fetch,
                 xml_fetch: Callable[[str], tuple[int, Any]] = text_fetch, maxdate: str = MAXDATE) -> None:
        cache_dir = cache_dir or default_cache_dir()
        if api_key is None and os.environ.get("COMBO_OFFLINE") != "1":
            api_key = api_key_from_env()
        # NCBI allows 10 requests/s with a key and 3 without; stay under both.
        interval = 0.11 if api_key else 0.36
        self.maxdate = maxdate
        self._json = CachedJsonClient(cache_dir, offline=offline, fetch=keyed_fetch(api_key, json_fetch),
                                      min_interval=interval)
        self._xml = CachedJsonClient(cache_dir, offline=offline, fetch=keyed_fetch(api_key, xml_fetch),
                                     min_interval=interval)

    @property
    def cache_stats(self) -> dict[str, int]:
        return {"hits": self._json.hits + self._xml.hits, "misses": self._json.misses + self._xml.misses}

    def search(self, query: str, retmax: int = 20, mindate: str = "1800/01/01") -> list[str]:
        """PMIDs matching `query`, most relevant first, published on or before `maxdate`."""
        url = search_url(query, retmax, self.maxdate, mindate)
        res = (self._json.get(url) or {}).get("esearchresult")
        if not isinstance(res, dict) or "idlist" not in res:
            raise PubMedError(f"unexpected esearch answer for {query!r}")
        if res.get("ERROR"):
            raise PubMedError(f"esearch error for {query!r}: {res['ERROR']}")
        return [str(p) for p in res["idlist"]]

    def fetch(self, pmids: Iterable[str]) -> list[Doc]:
        """Records for `pmids`, in the order asked (PMIDs PubMed does not return are dropped)."""
        pmids = list(dict.fromkeys(str(p) for p in pmids))
        got: dict[str, Doc] = {}
        for i in range(0, len(pmids), FETCH_BATCH):
            chunk = pmids[i:i + FETCH_BATCH]
            body = self._xml.get(fetch_url(chunk))
            if body is None:
                continue
            for d in parse_articles(body):
                got[d.pmid] = d
        return [got[p] for p in pmids if p in got]

    def summaries(self, pmids: Iterable[str]) -> dict[str, dict[str, Any]]:
        """esummary records (title, pubdate, source, authors) keyed by PMID."""
        pmids = list(dict.fromkeys(str(p) for p in pmids))
        out: dict[str, dict[str, Any]] = {}
        for i in range(0, len(pmids), FETCH_BATCH):
            res = (self._json.get(summary_url(pmids[i:i + FETCH_BATCH])) or {}).get("result", {})
            out.update({u: res[u] for u in res.get("uids", []) if "error" not in res[u]})
        return out

    def cited_by(self, pmid: str) -> list[str]:
        """PMIDs of records that cite `pmid` (PubMed Central citation links; incomplete by nature)."""
        body = self._json.get(cited_by_url(str(pmid))) or {}
        out: list[str] = []
        for ls in body.get("linksets", []):
            for db in ls.get("linksetdbs", []):
                if db.get("linkname") == "pubmed_pubmed_citedin":
                    out.extend(str(x) for x in db.get("links", []))
        return sorted(set(out), key=int)
