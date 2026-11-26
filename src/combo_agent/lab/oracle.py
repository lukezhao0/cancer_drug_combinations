"""The oracle: a simulated lab over the frozen NCI-ALMANAC pool (target_design.md §5).

Every selector (random, BO, the agent) chooses experiments, and only the oracle can say what they
measured. It holds the hidden outcomes from `labels.parquet` and reveals one row per experiment
run, charging the budget for each.

Rules (design §5):
* `run(batch)` refuses, as a whole, a batch that is empty or larger than `batch_size`, repeats a
  triplet inside the batch, re-runs one already revealed, names one not in the pool, names a
  held-out line, or would exceed the budget. A refused batch changes nothing.
* Drug order does not matter: (B, A, line) is the same experiment as (A, B, line).
* The seed set (design §5.2) is revealed at construction, free of budget, as round 0. The same seed
  gives every selector the same seed set.
* Every reveal is appended to an optional JSONL log: round, selector, triplets, values.

What a reveal contains: y_syn (mean Bliss), y_inh (mean CSS), their replicate SDs (NaN with one
replicate) and n_reps. No hit flags: the hit thresholds are public (`thresholds`), so a selector can
derive them, and evaluation code reads the labels itself.

Label access: this module and the build/evaluation code on the allowlist in
`tests/test_label_access.py` are the only readers of `labels.parquet`.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from combo_agent.data import paths

KEY = ["drug_a", "drug_b", "cellosaurus_ac"]
DEFAULT_BUDGET = 320  # design §11: 20 rounds x 16
DEFAULT_BATCH_SIZE = 16
LABELS_FILE = "labels.parquet"
SEED_SETS_FILE = "m2/seed_sets.json"


class Triplet(NamedTuple):
    """One experiment: a drug pair in one cell line. Drugs are ChEMBL IDs, the line a Cellosaurus
    accession. Canonical order is drug_a < drug_b."""

    drug_a: str
    drug_b: str
    cellosaurus_ac: str

    def canonical(self) -> Triplet:
        a, b = sorted((self.drug_a, self.drug_b))
        return Triplet(a, b, self.cellosaurus_ac)


@dataclass(frozen=True)
class Result:
    """One revealed experiment."""

    triplet: Triplet
    y_syn: float
    y_inh: float
    y_syn_sd: float
    y_inh_sd: float
    n_reps: int
    round: int

    def to_dict(self) -> dict:
        d = asdict(self)
        d["triplet"] = list(self.triplet)
        return {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in d.items()}


class OracleError(ValueError):
    """A request the oracle refuses. The message says why; nothing was revealed or charged."""


def _as_triplet(x: Triplet | Sequence[str]) -> Triplet:
    if isinstance(x, Triplet):
        return x.canonical()
    if len(x) != 3:
        raise OracleError(f"a triplet is (drug_a, drug_b, cellosaurus_ac), got {x!r}")
    return Triplet(*map(str, x)).canonical()


class Oracle:
    """The simulated lab. See the module docstring for the rules."""

    def __init__(
        self,
        pool: pd.DataFrame,
        budget: int = DEFAULT_BUDGET,
        batch_size: int = DEFAULT_BATCH_SIZE,
        seed: int | None = 0,
        seed_sets: dict[str, list[Sequence[str]]] | None = None,
        labels_path: Path | str | None = None,
        thresholds: dict[str, float] | None = None,
        log_path: Path | str | None = None,
        selector: str = "unknown",
    ) -> None:
        if budget < 0 or batch_size < 1:
            raise ValueError("budget must be >= 0 and batch_size >= 1")
        missing = {*KEY, "split"} - set(pool.columns)
        if missing:
            raise ValueError(f"pool is missing columns {sorted(missing)}")
        self.budget, self.batch_size, self.seed, self.selector = int(budget), int(batch_size), seed, selector
        self.thresholds = dict(thresholds or {})
        self._pool = pool.reset_index(drop=True)
        keys = self._pool[KEY].astype(str)
        if not (keys["drug_a"] < keys["drug_b"]).all():
            raise ValueError("pool keys must be canonical (drug_a < drug_b)")
        self._row: dict[Triplet, int] = {Triplet(*t): i for i, t in enumerate(keys.itertuples(index=False, name=None))}
        if len(self._row) != len(self._pool):
            raise ValueError("pool has duplicate triplets")
        self._selectable = (self._pool["split"] == "selectable").to_numpy()
        self._revealed = np.zeros(len(self._pool), dtype=bool)
        self.__load_labels(Path(labels_path) if labels_path else paths.processed_dir() / LABELS_FILE)
        self._history: list[Result] = []
        self._spent = 0
        self._round = 0
        self._log = Path(log_path) if log_path else None

        if seed is not None:
            if seed_sets is None or str(seed) not in seed_sets:
                raise ValueError(f"no seed set for seed {seed!r}; build them with "
                                 "`python -m combo_agent.lab.splits` or pass seed=None")
            seeds = [_as_triplet(t) for t in seed_sets[str(seed)]]
            self._check(seeds, size_limit=None, check_budget=False)
            self._reveal(seeds, charge=False)

    # -- labels (private) ---------------------------------------------------------------------------

    def __load_labels(self, path: Path) -> None:
        cols = [*KEY, "y_syn", "y_inh", "bliss_sd", "css_sd"]
        labels = pd.read_parquet(path, columns=cols)
        if len(labels) != len(self._pool) or not labels[KEY].astype(str).equals(self._pool[KEY].astype(str)):
            labels = self._pool[KEY].merge(labels, on=KEY, how="left", validate="1:1")
            if labels["y_syn"].isna().any():
                raise ValueError("labels do not cover every pool triplet")
        self.__y = {c: labels[c].to_numpy(dtype=float) for c in ("y_syn", "y_inh", "bliss_sd", "css_sd")}
        self.__n_reps = (self._pool["n_reps"].to_numpy(dtype=int) if "n_reps" in self._pool
                         else np.ones(len(self._pool), dtype=int))

    # -- public API ---------------------------------------------------------------------------------

    @classmethod
    def from_cache(cls, root: Path | str | None = None, **kwargs) -> Oracle:
        """Build from the frozen tables in `data_cache/processed/` (pool, labels, splits, seed sets)."""
        root = Path(root) if root else paths.processed_dir()
        pool = pd.read_parquet(root / "pool.parquet")
        splits = json.loads((root / "splits.json").read_text())
        seed_path = root / SEED_SETS_FILE
        if "seed_sets" not in kwargs and seed_path.exists():
            kwargs["seed_sets"] = json.loads(seed_path.read_text())["seed_sets"]
        kwargs.setdefault("thresholds", splits.get("thresholds"))
        return cls(pool, labels_path=root / LABELS_FILE, **kwargs)

    def candidates(self) -> list[Triplet]:
        """Untested, selectable triplets."""
        idx = np.flatnonzero(self._selectable & ~self._revealed)
        keys = self._pool[KEY].to_numpy()[idx]
        return [Triplet(*map(str, k)) for k in keys]

    def candidate_frame(self) -> pd.DataFrame:
        """The same candidates as pool rows (keys and metadata, no outcomes). Faster for big pools."""
        return self._pool.loc[self._selectable & ~self._revealed].copy()

    def run(self, batch: Iterable[Triplet | Sequence[str]]) -> list[Result]:
        """Run a batch and reveal its outcomes, or raise OracleError and change nothing."""
        triplets = [_as_triplet(t) for t in batch]
        self._check(triplets, size_limit=self.batch_size, check_budget=True)
        self._round += 1
        return self._reveal(triplets, charge=True)

    def history(self) -> list[Result]:
        """Everything revealed so far, seed set first, in order."""
        return list(self._history)

    def budget_left(self) -> int:
        return self.budget - self._spent

    @property
    def round(self) -> int:
        """Rounds run so far (the seed set is round 0)."""
        return self._round

    def is_hit(self, result: Result, target: str) -> bool:
        """Whether a revealed result clears the public hit threshold for `syn` or `inh`."""
        value = result.y_syn if target == "syn" else result.y_inh
        return bool(value >= self.thresholds[f"y_{target}"])

    def __repr__(self) -> str:
        return (f"Oracle(pool={len(self._pool)}, selectable={int(self._selectable.sum())}, "
                f"revealed={int(self._revealed.sum())}, budget_left={self.budget_left()}, round={self._round})")

    # -- internals ----------------------------------------------------------------------------------

    def _check(self, triplets: list[Triplet], size_limit: int | None, check_budget: bool) -> None:
        if not triplets:
            raise OracleError("empty batch")
        if size_limit is not None and len(triplets) > size_limit:
            raise OracleError(f"batch of {len(triplets)} exceeds batch_size {size_limit}")
        if len(set(triplets)) != len(triplets):
            dupes = sorted({t for t in triplets if triplets.count(t) > 1})
            raise OracleError(f"batch repeats triplets: {dupes}")
        for t in triplets:
            i = self._row.get(t)
            if i is None:
                raise OracleError(f"{t} is not in the pool")
            if not self._selectable[i]:
                raise OracleError(f"{t} is on a held-out cell line and can never be run")
            if self._revealed[i]:
                raise OracleError(f"{t} was already revealed; re-runs are not allowed")
        if check_budget and len(triplets) > self.budget_left():
            raise OracleError(f"batch of {len(triplets)} exceeds the remaining budget {self.budget_left()}")

    def _reveal(self, triplets: list[Triplet], charge: bool) -> list[Result]:
        y = self.__y
        out = []
        for t in triplets:
            i = self._row[t]
            self._revealed[i] = True
            out.append(Result(t, float(y["y_syn"][i]), float(y["y_inh"][i]), float(y["bliss_sd"][i]),
                              float(y["css_sd"][i]), int(self.__n_reps[i]), self._round))
        if charge:
            self._spent += len(triplets)
        self._history.extend(out)
        if self._log is not None:
            self._log.parent.mkdir(parents=True, exist_ok=True)
            with self._log.open("a") as f:
                f.write(json.dumps({"round": self._round, "selector": self.selector, "seed": self.seed,
                                    "charged": charge, "budget_left": self.budget_left(),
                                    "results": [r.to_dict() for r in out]}, sort_keys=True) + "\n")
        return out
