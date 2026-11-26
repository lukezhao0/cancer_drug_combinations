"""M2 build step: seed sets and famous / middle / obscure drug-pair labels (target_design.md §5).

Outputs, in `data_cache/processed/m2/` (M1's `splits.json` is frozen and not touched):

    seed_sets.json     10 seed sets (seeds 0-9) of 32 selectable triplets each
    pair_fame.parquet  one row per pool drug pair: PubMed co-mention count and fame class
    drug_pubmed.parquet one row per pool drug: its own cancer-paper count (context for the pairs)
    m2_card.md         human-readable summary. Byte-for-byte reproducible (no dates, no paths)
    manifest.json      sha256 of inputs and outputs, the PubMed date cutoff, build date

**Seed sets** (design §5.2): every selector starts a campaign with the same free reveals, so
comparisons are paired. Seed s draws 32 distinct selectable triplets with
`numpy.random.default_rng(s)`, uniformly over experiments.

**Pair fame** (design §5.3): how much the literature already says about a pair, so results can be
split by "the language model may simply remember this" (famous) vs "it has to reason" (obscure).
For each drug pair, the number of PubMed records whose title or abstract names both drugs, with a
cancer term:

    (A terms [tiab], OR'd) AND (B terms [tiab], OR'd) AND (cancer clause)

* Terms per drug come from the reviewed `data/curated/drug_pubmed_terms.csv` (generic name plus
  widely used synonyms, abbreviations and code names). Raw source labels would miscount: e.g.
  "DOCETAXEL ANHYDROUS" or "ELOXATIN (TN) (SANOFI SYNTHELAB)".
* Publication dates are capped at `MAXDATE` so counts do not drift as new papers appear.
* Guard: if a drug's own count is 0, its pairs are 0 without a query (a clause PubMed cannot find
  must never be dropped and leave the rest of the query matching). A pair count above either
  drug's own count fails the build.
* Classes: `obscure` = 0 co-mentions; `famous` = count >= the upper quartile over all pool pairs
  (and > 0); `middle` = the rest. If the upper quartile is 0, `famous` is the top quartile of the
  nonzero pairs, and the card says so.

Requests go through `data.api_cache.CachedJsonClient` (cache: `data_cache/raw/api_cache/pubmed/`),
so `--offline` (or COMBO_OFFLINE=1) rebuilds from the cache with no network. `NCBI_API_KEY` (from
the environment or `./.env`) raises NCBI's limit from 3 to 10 requests/s. It is added at fetch time
only, so it never appears in the cache or the outputs.

This module reads the hidden labels only to report hit counts in the card (allowlisted in
`tests/test_label_access.py`); nothing it writes contains outcomes.

Run: python -m combo_agent.lab.splits [--offline] [--smoke N]
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import platform
import sys
import urllib.parse
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from combo_agent.data import paths
from combo_agent.data.api_cache import CachedJsonClient, urllib_fetch

KEY = ["drug_a", "drug_b", "cellosaurus_ac"]
SEEDS = tuple(range(10))
SEED_SET_SIZE = 32
MAXDATE = "2026/11/01"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
TERMS_CSV = Path(__file__).resolve().parents[1] / "data" / "curated" / "drug_pubmed_terms.csv"
CANCER_CLAUSE = ("(neoplasms[mh] OR cancer*[tiab] OR tumor*[tiab] OR tumour*[tiab] OR carcinoma*[tiab] "
                 "OR leukemia*[tiab] OR leukaemia*[tiab] OR lymphoma*[tiab] OR myeloma*[tiab] "
                 "OR melanoma*[tiab] OR sarcoma*[tiab] OR glioma*[tiab])")
FAME_ORDER = ("famous", "middle", "obscure")


class SplitsError(RuntimeError):
    """A build check failed."""


# -- seed sets ------------------------------------------------------------------------------------

def seed_sets(pool: pd.DataFrame, seeds=SEEDS, size: int = SEED_SET_SIZE) -> dict[str, list[list[str]]]:
    """Seed -> sorted list of `size` distinct selectable triplets."""
    sel = pool.loc[pool["split"] == "selectable", KEY].astype(str).sort_values(KEY).reset_index(drop=True)
    out = {}
    for s in seeds:
        idx = np.random.default_rng(s).choice(len(sel), size=size, replace=False)
        out[str(s)] = sorted(sel.iloc[idx].values.tolist())
    return out


# -- PubMed queries -------------------------------------------------------------------------------

def load_terms(path: Path = TERMS_CSV) -> dict[str, list[str]]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    terms = {r["drug_id"]: [t.strip() for t in r["terms"].split("|") if t.strip()] for r in rows}
    if any(not t for t in terms.values()):
        raise SplitsError("a drug in the terms file has no terms")
    return terms


def drug_clause(terms: list[str]) -> str:
    return "(" + " OR ".join(f'"{t}"[tiab]' for t in terms) + ")"


def pair_query(terms_a: list[str], terms_b: list[str]) -> str:
    return f"{drug_clause(terms_a)} AND {drug_clause(terms_b)} AND {CANCER_CLAUSE}"


def drug_query(terms: list[str]) -> str:
    return f"{drug_clause(terms)} AND {CANCER_CLAUSE}"


def count_url(query: str, maxdate: str = MAXDATE) -> str:
    """The URL that is cached: no API key, fixed parameter order."""
    params = {"db": "pubmed", "rettype": "count", "retmode": "json", "datetype": "pdat",
              "mindate": "1800/01/01", "maxdate": maxdate, "tool": "combo_agent", "term": query}
    return f"{EUTILS}?{urllib.parse.urlencode(params)}"


def keyed_fetch(api_key: str | None, fetch: Callable[[str], tuple[int, Any]] = urllib_fetch):
    """A fetcher that appends the API key on the way out, so the cache key never contains it."""
    if not api_key:
        return fetch
    return lambda url: fetch(f"{url}&{urllib.parse.urlencode({'api_key': api_key})}")


def parse_count(body: Any, url: str) -> int:
    res = (body or {}).get("esearchresult")
    if not isinstance(res, dict) or "count" not in res:
        raise SplitsError(f"unexpected E-utilities answer for {url}: {str(body)[:300]}")
    if "ERROR" in res:
        raise SplitsError(f"E-utilities error for {url}: {res['ERROR']}")
    return int(res["count"])


def api_key_from_env(env_file: Path | None = None) -> str | None:
    if os.environ.get("NCBI_API_KEY"):
        return os.environ["NCBI_API_KEY"].strip()
    env_file = env_file or Path.cwd() / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            k, _, v = line.partition("=")
            if k.strip() == "NCBI_API_KEY" and v.strip():
                return v.strip().strip("'\"")
    return None


@dataclass
class Counter:
    """Counts PubMed records for many queries, in parallel, through the cache."""

    cache_dir: Path
    offline: bool | None = None
    api_key: str | None = None
    fetch: Callable[[str], tuple[int, Any]] = urllib_fetch
    workers: int | None = None

    def __post_init__(self) -> None:
        # NCBI allows 3 requests/s without a key and 10 with one. Each worker has its own minimum
        # interval, so the total stays just under the limit (9/s or 2.8/s) whatever the latency.
        self.workers = self.workers or (8 if self.api_key else 2)
        interval = self.workers / (9.0 if self.api_key else 2.8)
        f = keyed_fetch(self.api_key, self.fetch)
        self._clients = [CachedJsonClient(self.cache_dir, offline=self.offline, fetch=f, min_interval=interval)
                         for _ in range(self.workers)]

    def counts(self, queries: list[str], progress: Callable[[int, int], None] | None = None) -> list[int]:
        urls = [count_url(q) for q in queries]
        chunks = [urls[i::self.workers] for i in range(self.workers)]
        results: dict[str, int] = {}
        done = [0]

        def work(client: CachedJsonClient, chunk: list[str]) -> None:
            for u in chunk:
                results[u] = parse_count(client.get(u), u)
                done[0] += 1
                if progress and done[0] % 250 == 0:
                    progress(done[0], len(urls))

        with ThreadPoolExecutor(self.workers) as pool:
            for fut in [pool.submit(work, c, ch) for c, ch in zip(self._clients, chunks)]:
                fut.result()
        return [results[u] for u in urls]


# -- fame classes ---------------------------------------------------------------------------------

def classify(counts: pd.Series) -> tuple[pd.Series, dict[str, Any]]:
    """famous / middle / obscure, and how the cut was made."""
    q3 = float(counts.quantile(0.75))
    info: dict[str, Any] = {"q3_all_pairs": q3, "fallback": False}
    if q3 > 0:
        cut = q3
    else:
        nonzero = counts[counts > 0]
        cut = float(nonzero.quantile(0.75)) if len(nonzero) else float("inf")
        info["fallback"] = True
    info["famous_min_count"] = cut
    fame = pd.Series("middle", index=counts.index, dtype="object")
    fame[counts == 0] = "obscure"
    fame[(counts >= cut) & (counts > 0)] = "famous"
    return fame, info


# -- build ----------------------------------------------------------------------------------------

def pairs_of(pool: pd.DataFrame) -> pd.DataFrame:
    return pool[["drug_a", "drug_b"]].drop_duplicates().sort_values(["drug_a", "drug_b"]).reset_index(drop=True)


def pair_fame(pool: pd.DataFrame, terms: dict[str, list[str]], counter: Counter,
              log: Callable[[str], None] = print) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    drugs = sorted(set(pool["drug_a"]) | set(pool["drug_b"]))
    missing = [d for d in drugs if d not in terms]
    if missing:
        raise SplitsError(f"no PubMed terms for pool drugs {missing}")
    log(f"PubMed: {len(drugs)} single-drug counts")
    single = dict(zip(drugs, counter.counts([drug_query(terms[d]) for d in drugs])))
    pairs = pairs_of(pool)
    pairs["query"] = [pair_query(terms[a], terms[b]) for a, b in zip(pairs.drug_a, pairs.drug_b)]
    need = [(i, q) for i, (a, b, q) in enumerate(zip(pairs.drug_a, pairs.drug_b, pairs["query"]))
            if single[a] > 0 and single[b] > 0]
    log(f"PubMed: {len(need)} pair counts ({len(pairs) - len(need)} skipped: a drug with no papers)")
    got = counter.counts([q for _, q in need], progress=lambda d, n: log(f"  {d}/{n}"))
    n = np.zeros(len(pairs), dtype=int)
    n[[i for i, _ in need]] = got
    pairs["n_pubmed"] = n
    over = pairs[[c > min(single[a], single[b]) for a, b, c in zip(pairs.drug_a, pairs.drug_b, pairs.n_pubmed)]]
    if len(over):
        raise SplitsError(f"{len(over)} pair counts exceed a drug's own count, e.g. {over.iloc[0].to_dict()}")
    pairs["fame"], info = classify(pairs["n_pubmed"])
    drug_tab = pd.DataFrame({"drug_id": drugs, "n_pubmed_cancer": [single[d] for d in drugs],
                             "query": [drug_query(terms[d]) for d in drugs]})
    return pairs[["drug_a", "drug_b", "n_pubmed", "fame", "query"]], drug_tab, info


def _hits_by(labels: pd.DataFrame, group: pd.Series) -> pd.DataFrame:
    g = labels.assign(group=group.values)
    return g.groupby("group").agg(experiments=("is_hit_syn", "size"), syn_hits=("is_hit_syn", "sum"),
                                  syn_hits_no_lowkill=("no_lowkill", "sum"), inh_hits=("is_hit_inh", "sum"))


def card(seeds: dict[str, list[list[str]]], fame: pd.DataFrame, drug_tab: pd.DataFrame, info: dict[str, Any],
         pool: pd.DataFrame, labels: pd.DataFrame, names: dict[str, str]) -> str:
    lab = pool[KEY + ["split"]].merge(labels[KEY + ["is_hit_syn", "is_hit_inh", "syn_hit_low_kill"]], on=KEY)
    lab["no_lowkill"] = lab["is_hit_syn"] & ~lab["syn_hit_low_kill"].astype(bool)
    lines = ["# M2 card: seed sets and pair fame", "",
             "Generated by `combo_agent.lab.splits`. Rerunning (online or from the cache) gives a byte-identical file.",
             "", "## Seed sets", "",
             f"{len(seeds)} seeds x {SEED_SET_SIZE} selectable triplets, revealed free at round 0. Expected "
             f"synergy hits per set at the 5% base rate: {SEED_SET_SIZE * 0.05:.1f}.", "",
             "| Seed | Drug pairs | Lines | Synergy hits | (without low-kill) | Killing hits |", "|--:|--:|--:|--:|--:|--:|"]
    keyed = lab.set_index(KEY)
    for s, trips in seeds.items():
        rows = keyed.loc[[tuple(t) for t in trips]]
        lines.append(f"| {s} | {len({(t[0], t[1]) for t in trips})} | {len({t[2] for t in trips})} | "
                     f"{int(rows.is_hit_syn.sum())} | {int(rows.no_lowkill.sum())} | {int(rows.is_hit_inh.sum())} |")
    counts = fame["fame"].value_counts()
    lines += ["", "## Pair fame (PubMed co-mentions with a cancer term)", "",
              f"Publication-date cutoff: {MAXDATE}. Upper quartile over all {len(fame):,} pairs: "
              f"{info['q3_all_pairs']:g} records; famous = at least {info['famous_min_count']:g}"
              + (" (fallback: top quartile of nonzero pairs)." if info["fallback"] else "."), "",
              "| Class | Pairs | Share | Median records | Selectable experiments | Synergy hit rate | (without low-kill) | Killing hit rate |",
              "|:--|--:|--:|--:|--:|--:|--:|--:|"]
    sel = lab[lab["split"] == "selectable"].merge(fame[["drug_a", "drug_b", "fame"]], on=["drug_a", "drug_b"])
    by = _hits_by(sel, sel["fame"])
    for c in FAME_ORDER:
        k = int(counts.get(c, 0))
        med = fame.loc[fame.fame == c, "n_pubmed"].median() if k else float("nan")
        if c in by.index:
            r = by.loc[c]
            e = int(r.experiments)
            lines.append(f"| {c} | {k:,} | {k / len(fame):.1%} | {med:g} | {e:,} | {r.syn_hits / e:.2%} | "
                         f"{r.syn_hits_no_lowkill / e:.2%} | {r.inh_hits / e:.2%} |")
        else:
            lines.append(f"| {c} | {k:,} | {k / len(fame):.1%} | - | 0 | - | - | - |")
    top = fame.sort_values(["n_pubmed", "drug_a", "drug_b"], ascending=[False, True, True]).head(10)
    lines += ["", "Most co-mentioned pairs:", "", "| Pair | Records |", "|:--|--:|"]
    lines += [f"| {names[a]} + {names[b]} | {n:,} |" for a, b, n in zip(top.drug_a, top.drug_b, top.n_pubmed)]
    least = drug_tab.sort_values(["n_pubmed_cancer", "drug_id"]).head(8)
    lines += ["", "Pool drugs with the fewest cancer papers (their pairs lean obscure):", "",
              "| Drug | Records |", "|:--|--:|"]
    lines += [f"| {names[d]} | {n:,} |" for d, n in zip(least.drug_id, least.n_pubmed_cancer)]
    lines += ["", "## Caveats", "",
              "- Counts are title/abstract co-mentions, not evidence that the pair was tested together.",
              "- Short abbreviations (e.g. 5-FU, ara-C, ATRA, SAHA, DTIC) are included where widely used; a few may "
              "match unrelated text.",
              "- `tumor*` also matches 'tumor necrosis factor', and a few drugs have non-cancer uses (e.g. "
              "sirolimus and paclitaxel on heart stents). Checked for that pair: 16 of its 234 records mention "
              "stents and 5 match only through 'tumor necrosis'. It is famous either way.",
              "- SN-38 is counted under its own names, not irinotecan's: the pool drug is the metabolite.",
              "- Fame describes the literature, not what a given language model has seen.", ""]
    return "\n".join(lines)


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build(out_dir: Path, offline: bool | None = None, processed: Path | None = None,
          cache_dir: Path | None = None, terms_path: Path = TERMS_CSV, api_key: str | None = None,
          fetch: Callable[[str], tuple[int, Any]] = urllib_fetch, log: Callable[[str], None] = print) -> dict[str, Path]:
    processed = processed or paths.processed_dir()
    pool = pd.read_parquet(processed / "pool.parquet")
    labels = pd.read_parquet(processed / "labels.parquet")
    names = {d: str(n).removesuffix(" ANHYDROUS") for d, n in
             pd.read_parquet(processed / "drugs.parquet", columns=["drug_id", "preferred_name"]).values}
    seeds = seed_sets(pool)
    counter = Counter(cache_dir or paths.raw_dir() / "api_cache" / "pubmed", offline=offline,
                      api_key=api_key, fetch=fetch)
    fame, drug_tab, info = pair_fame(pool, load_terms(terms_path), counter, log=log)

    out_dir.mkdir(parents=True, exist_ok=True)
    out = {"seed_sets": out_dir / "seed_sets.json", "pair_fame": out_dir / "pair_fame.parquet",
           "drug_pubmed": out_dir / "drug_pubmed.parquet", "m2_card": out_dir / "m2_card.md"}
    out["seed_sets"].write_text(json.dumps({"size": SEED_SET_SIZE, "seeds": list(SEEDS), "seed_sets": seeds},
                                           indent=2, sort_keys=True) + "\n")
    fame.to_parquet(out["pair_fame"], index=False)
    drug_tab.to_parquet(out["drug_pubmed"], index=False)
    out["m2_card"].write_text(card(seeds, fame, drug_tab, info, pool, labels, names))
    inputs = {"pool": processed / "pool.parquet", "labels": processed / "labels.parquet", "terms": terms_path}
    manifest = {
        "inputs": {k: {"file": p.name, "sha256": _sha(p)} for k, p in inputs.items()},
        "outputs": {k: {"file": p.name, "sha256": _sha(p)} for k, p in sorted(out.items())},
        "pubmed": {"maxdate": MAXDATE, "cancer_clause": CANCER_CLAUSE, "fame": info,
                   "requests_from_cache": sum(c.hits for c in counter._clients),
                   "requests_fetched": sum(c.misses for c in counter._clients)},
        "built": dt.date.today().isoformat(),
        "versions": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__},
    }
    out["manifest"] = out_dir / "manifest.json"
    out["manifest"].write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=None, help="output dir (default data_cache/processed/m2)")
    ap.add_argument("--offline", action="store_true", help="use only cached PubMed answers")
    ap.add_argument("--smoke", type=int, default=0, help="count N pairs and print them; write nothing")
    args = ap.parse_args(argv)
    key = api_key_from_env()
    print(f"NCBI API key: {'found' if key else 'not set (3 requests/s)'}")
    if args.smoke:
        pool = pd.read_parquet(paths.processed_dir() / "pool.parquet")
        terms = load_terms()
        pairs = pairs_of(pool).head(args.smoke)
        counter = Counter(paths.raw_dir() / "api_cache" / "pubmed", offline=args.offline or None, api_key=key)
        qs = [pair_query(terms[a], terms[b]) for a, b in zip(pairs.drug_a, pairs.drug_b)]
        for (a, b), n in zip(pairs[["drug_a", "drug_b"]].values, counter.counts(qs)):
            print(f"{a} + {b}: {n}")
        return 0
    out = build(args.out or paths.processed_dir() / "m2", offline=args.offline or None, api_key=key)
    print("\n".join(f"wrote {p}" for p in out.values()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
