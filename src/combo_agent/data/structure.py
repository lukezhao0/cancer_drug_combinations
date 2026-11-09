"""RDKit helpers for comparing a source SMILES with a reference (ChEMBL) structure.

The TDC `drugcomb` SMILES are written without stereochemistry and often with a counter-ion, and the
platinum drugs are written as disconnected ions. So two structures are compared at the level that
survives all of that: the **connectivity key**, the first 14 characters of the InChIKey of the
parent (largest fragment, uncharged). It ignores stereo, isotopes and protonation.

Two points that are easy to get wrong (both checked on the real data on 2026-11-08):

* Do **not** canonicalize tautomers before computing the key that is compared with ChEMBL. The
  InChI main layer includes the mobile-H layer, and RDKit's canonical tautomer differs from
  ChEMBL's standardized form for e.g. doxorubicin, paclitaxel and docetaxel, so their keys stop
  matching. Tautomer canonicalization is only used as a second-chance comparison
  (`tautomer_key`).
* Do **not** salt-strip metal complexes. `LargestFragmentChooser` turns cisplatin
  (`N.N.Cl[Pt+2]Cl`) into ammonia and carboplatin/oxaliplatin into a ligand. Metal-containing
  structures are compared by heavy-atom formula instead (`heavy_atom_formula`).
"""

from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache

from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

#: Elements that mark a structure as a metal complex (salt-stripping is unsafe for these). Alkali
#: and alkaline-earth counter-ions (Na, K, Ca, ...) are deliberately absent: stripping them is fine.
METALS: frozenset[str] = frozenset({"Pt", "Pd", "Ru", "Au", "Ag", "Cu", "Fe", "Co", "Ni", "Zn",
                                    "Ga", "As", "Sb", "Bi", "Ti", "V", "Gd", "Rh", "Ir", "Os"})

_STEREO_MARK = re.compile(r"[@/\\]")
_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")


class StructureError(ValueError):
    """A SMILES that RDKit cannot parse."""


@lru_cache(maxsize=4096)
def mol_from_smiles(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise StructureError(f"RDKit cannot parse SMILES {smiles!r}")
    return mol


def n_fragments(smiles: str) -> int:
    return len(Chem.GetMolFrags(mol_from_smiles(smiles)))


def has_metal(smiles: str) -> bool:
    """True if any atom is a (transition/post-transition) metal, i.e. the structure is a complex."""
    return any(a.GetSymbol() in METALS for a in mol_from_smiles(smiles).GetAtoms())


def has_stereo(smiles: str) -> bool:
    """True if the SMILES string carries any tetrahedral or double-bond stereo mark."""
    return bool(_STEREO_MARK.search(smiles))


def n_unassigned_stereocentres(smiles: str) -> int:
    """Tetrahedral stereocentres RDKit finds that the SMILES leaves unspecified (e.g. a racemate)."""
    centres = Chem.FindMolChiralCenters(mol_from_smiles(smiles), includeUnassigned=True,
                                        useLegacyImplementation=False)
    return sum(1 for _, label in centres if label == "?")


def parent_mol(smiles: str) -> Chem.Mol:
    """Largest fragment, uncharged. Refuses metal complexes, which this would destroy."""
    if has_metal(smiles):
        raise StructureError(f"refusing to salt-strip a metal complex: {smiles!r}")
    chooser, uncharger = rdMolStandardize.LargestFragmentChooser(), rdMolStandardize.Uncharger()
    return uncharger.uncharge(chooser.choose(mol_from_smiles(smiles)))


def connectivity_key(smiles: str) -> str:
    """First 14 characters of the parent's InChIKey (no tautomer canonicalization; see module doc)."""
    return Chem.MolToInchiKey(parent_mol(smiles))[:14]


def tautomer_key(smiles: str) -> str:
    """Connectivity key after RDKit tautomer canonicalization. Used only as a fallback comparison."""
    mol = rdMolStandardize.TautomerEnumerator().Canonicalize(parent_mol(smiles))
    return Chem.MolToInchiKey(mol)[:14]


def heavy_atom_formula(smiles: str) -> str:
    """Hill-ordered element counts of all fragments, hydrogens excluded (e.g. 'C6N2O4Pt')."""
    counts = Counter(a.GetSymbol() for a in mol_from_smiles(smiles).GetAtoms() if a.GetAtomicNum() != 1)
    return _hill(counts)


def heavy_atom_formula_from_molformula(formula: str) -> str:
    """Drop hydrogens from a molecular formula string such as ChEMBL's `full_molformula`."""
    counts: Counter[str] = Counter()
    for element, n in _FORMULA_TOKEN.findall(formula.replace(" ", "")):
        if element not in ("H", "D", "T"):
            counts[element] += int(n) if n else 1
    return _hill(counts)


def _hill(counts: Counter[str]) -> str:
    order = [e for e in ("C",) if e in counts] + sorted(e for e in counts if e != "C")
    return "".join(f"{e}{counts[e] if counts[e] > 1 else ''}" for e in order)
