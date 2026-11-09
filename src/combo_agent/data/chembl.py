"""ChEMBL and PubChem lookups used by drug harmonization, all through `CachedJsonClient`.

Endpoints and behaviour verified against ChEMBL_37 (released 2026-05-01) and PubChem PUG-REST on
2026-11-08:

* `molecule.json?pref_name__iexact=` and `?molecule_synonyms__molecule_synonym__iexact=` are exact,
  case-insensitive name matches. They resolve foreign spellings and NSC numbers that ChEMBL lists
  as synonyms (`CISPLATINO` -> CHEMBL11359, `NSC-85998` -> CHEMBL1977579). CAS numbers are not
  ChEMBL synonyms, so PubChem is the second name route.
* `molecule_structures__standard_inchi_key__startswith=<14 chars>` finds every ChEMBL structure with
  a given skeleton. It usually returns several stereoisomers, so it is evidence, not an answer.
* **Mechanisms hang off the salt record.** `mechanism.json?molecule_chembl_id=CHEMBL553` (erlotinib,
  the parent) returns nothing, while `?parent_molecule_chembl_id=CHEMBL553` returns EGFR. Always
  query by `parent_molecule_chembl_id`.
* Platinum drugs have `molecule_structures: null` in ChEMBL; only `full_molformula` is available.
"""

from __future__ import annotations

import urllib.parse
from typing import Any

from combo_agent.data.api_cache import CachedJsonClient

CHEMBL = "https://www.ebi.ac.uk/chembl/api/data"
PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
_PAGE = 1000  # ChEMBL's maximum page size
#: PubChem may list many CIDs for a loose name; more than this is never a useful identity signal.
MAX_PUBCHEM_CIDS = 10


def _url(base: str, path: str, params: dict[str, Any] | None = None) -> str:
    query = "?" + urllib.parse.urlencode(sorted(params.items())) if params else ""
    return f"{base}/{path}{query}"


class ChemblApi:
    def __init__(self, chembl: CachedJsonClient, pubchem: CachedJsonClient):
        self._c, self._p = chembl, pubchem

    # -- ChEMBL -------------------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        return self._c.get(_url(CHEMBL, "status.json"))

    def _paged(self, resource: str, params: dict[str, Any], key: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        offset = 0
        while True:
            page = self._c.get(_url(CHEMBL, f"{resource}.json",
                                    {**params, "limit": _PAGE, "offset": offset}))
            out += page[key]
            offset += _PAGE
            if offset >= page["page_meta"]["total_count"]:
                return out

    def _molecule_ids(self, **filters: str) -> list[str]:
        rows = self._paged("molecule", {**filters, "only": "molecule_chembl_id"}, "molecules")
        return sorted({r["molecule_chembl_id"] for r in rows})

    def molecules_by_name(self, name: str) -> list[str]:
        """ChEMBL IDs whose preferred name or any synonym equals `name` (case-insensitive)."""
        return sorted(set(self._molecule_ids(pref_name__iexact=name))
                      | set(self._molecule_ids(molecule_synonyms__molecule_synonym__iexact=name)))

    def molecules_by_inchikey(self, inchikey: str) -> list[str]:
        return self._molecule_ids(molecule_structures__standard_inchi_key=inchikey)

    def molecules_by_connectivity(self, conn_key: str) -> list[str]:
        return self._molecule_ids(molecule_structures__standard_inchi_key__startswith=conn_key)

    def molecule(self, chembl_id: str) -> dict[str, Any] | None:
        """The full molecule record, or None if ChEMBL has no such ID."""
        return self._c.get(_url(CHEMBL, f"molecule/{chembl_id}.json"))

    def parent_id(self, chembl_id: str) -> str | None:
        """`molecule_hierarchy.parent_chembl_id` (the record itself if it has no hierarchy)."""
        rec = self.molecule(chembl_id)
        if rec is None:
            return None
        return (rec.get("molecule_hierarchy") or {}).get("parent_chembl_id") or chembl_id

    def mechanisms(self, parent_chembl_id: str) -> list[dict[str, Any]]:
        """Mechanism-of-action records for a parent and all its salt forms."""
        rows = self._paged("mechanism", {"parent_molecule_chembl_id": parent_chembl_id}, "mechanisms")
        return sorted(rows, key=lambda r: r["mec_id"])

    def target(self, target_chembl_id: str) -> dict[str, Any] | None:
        return self._c.get(_url(CHEMBL, f"target/{target_chembl_id}.json"))

    # -- PubChem ------------------------------------------------------------------------------

    def pubchem_by_name(self, name: str) -> list[tuple[int, str]]:
        """(CID, full InChIKey) pairs PubChem returns for a name; [] if it knows none."""
        path = f"compound/name/{urllib.parse.quote(name, safe='')}/property/InChIKey/JSON"
        body = self._p.get(_url(PUBCHEM, path))
        if not body:
            return []
        props = body.get("PropertyTable", {}).get("Properties", [])
        return sorted((int(p["CID"]), p["InChIKey"]) for p in props if "InChIKey" in p)[:MAX_PUBCHEM_CIDS]


def gene_symbols(target: dict[str, Any]) -> list[str]:
    """Gene symbols of a target's components (sorted, unique)."""
    out = set()
    for comp in target.get("target_components") or []:
        for syn in comp.get("target_component_synonyms") or []:
            if syn.get("syn_type") == "GENE_SYMBOL":
                out.add(syn["component_synonym"])
    return sorted(out)
