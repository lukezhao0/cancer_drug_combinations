"""Build the M5 leakage gate test set: real PubMed records with a known answer.

    uv run python scripts/make_m5_testset.py          # needs data_cache/processed/m5/blocklist.parquet

Writes `tests/fixtures/m5/leakage_testset.jsonl` (one record per document) and
`tests/fixtures/m5/lexicon.json` (pool line names and pool drug pairs), so `tests/test_leakage.py`
runs with no data cache and no network.

The positives were chosen by hand, not by running the filter: the curated source papers, plus
papers known to reanalyse or redistribute the screens (synergy-prediction models trained on
O'Neil / NCI-ALMANAC / DrugComb, the databases that repackage them, and a clinical follow-up of an
NCI-ALMANAC hit). The negatives are ordinary mechanism reviews and clinical trials of pool drugs.
The gate is 100% of positives blocked; the negatives measure how much clean literature is lost.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from combo_agent.data import paths
from combo_agent.literature.leakage import _line_names, _pool_pairs, load_sources
from combo_agent.literature.pubmed import PubMed

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "fixtures" / "m5"

#: PMID -> why it is leaky (each was looked up on PubMed by title on 2026-11-24)
REANALYSES = {
    "34086576": "MatchMaker: deep synergy model trained on DrugComb",
    "33577560": "TranSynergy: synergy model on the O'Neil screen",
    "34571537": "DeepDDS: graph synergy model on O'Neil / AZ-DREAM",
    "35043159": "PRODeepSyn: synergy model on O'Neil",
    "33262326": "comboFM: predicts NCI-ALMANAC dose responses",
    "31380352": "Predicting synergism using NCI-ALMANAC data",
    "31665429": "DrugCombDB: database that includes NCI-ALMANAC",
    "32442307": "SYNERGxDB: portal that includes NCI-ALMANAC and O'Neil",
    "38340091": "SynergyX: synergy model on DrugComb",
    "42324597": "DeepSynBa: dose-response combination model on NCI-ALMANAC",
    "41580962": "One-hot news: audit of synergy models on these screens",
    "38718189": "From cell lines to patients: synergy model trained on screens",
    "41826558": "Phase I trial of bortezomib + clofarabine chosen from NCI-ALMANAC",
    "40091020": "Metformin + HDAC inhibitors, combination found in DrugComb",
    "42919347": "ComplexMatrixComb: NCI-ALMANAC dose model",
    "42170026": "CDCR-Rank: NCI-ALMANAC dose-response ranking",
}

#: PMID -> topic (clean literature an agent should be able to read)
NEGATIVES = {
    "35728875": "spartalizumab + dabrafenib + trametinib, phase 3 (COMBI-i)",
    "39004561": "PARP inhibitors, BRCA and gaps (review)",
    "25806228": "adaptive resistance to targeted therapies (review)",
    "32786187": "azacitidine + venetoclax in AML (VIALE-A)",
    "38343151": "VIALE-A long-term follow-up",
    "36043521": "CDK4/6 inhibitor resistance (review)",
    "35304604": "targeting CDK4 and CDK6 (review)",
    "31564718": "osimertinib resistance mechanisms (review)",
    "31666701": "AMG 510 (sotorasib) drives anti-tumour immunity",
    "37870968": "sotorasib + panitumumab in colorectal cancer",
    "19407320": "imatinib durability in CML",
    "30828146": "CML long-term outcomes in the imatinib era",
    "34101093": "combination chemotherapy vs temozolomide in MGMT-methylated GBM",
    "32033280": "proteasome inhibitors and proteostasis (review)",
    "36202931": "targeting replication stress (review)",
    "38378898": "targeting ATR in patients (review)",
    "30345884": "maintenance olaparib in ovarian cancer (SOLO-1)",
    "36082969": "SOLO-1 7-year overall survival",
    "37560925": "recent HDAC inhibitors in cancer therapy (review)",
}


def main() -> int:
    processed = paths.processed_dir()
    bl = pd.read_parquet(processed / "m5" / "blocklist.parquet")
    cites = bl[bl.reason == "cites_source"].groupby("pmid").via.apply(sorted).to_dict()
    sources = {r["pmid"]: r["short"] + " (curated source)" for r in load_sources()}
    wanted = [("positive", "source", p, w) for p, w in sources.items()]
    wanted += [("positive", "reanalysis", p, w) for p, w in REANALYSES.items()]
    wanted += [("negative", "clean", p, w) for p, w in NEGATIVES.items()]
    docs = {d.pmid: d for d in PubMed().fetch(p for _, _, p, _ in wanted)}
    missing = [p for _, _, p, _ in wanted if p not in docs]
    if missing:
        raise SystemExit(f"PubMed returned no record for {missing}")
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "leakage_testset.jsonl").open("w") as f:
        for label, kind, pmid, why in wanted:
            rec = {"label": label, "kind": kind, "why": why, "cited_by_sources": cites.get(pmid, []),
                   "doc": docs[pmid].to_dict()}
            f.write(json.dumps(rec, sort_keys=True) + "\n")
    lex = {"lines": _line_names(processed), "pairs": sorted(map(list, _pool_pairs(processed)))}
    (OUT / "lexicon.json").write_text(json.dumps(lex, sort_keys=True) + "\n")
    print(f"wrote {len(wanted)} records ({len(sources) + len(REANALYSES)} positive, {len(NEGATIVES)} negative) to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
