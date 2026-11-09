"""Convert a TDC DrugSyn pickle to parquet, loading it with a restricted unpickler.

TDC ships its DrugSyn tables as pandas pickles written by pandas 1.x/2.x with numpy 1.x. They do not
load under this project's pandas>=3 / numpy>=2, so `ingest_tdc` runs this file once in an isolated
environment (pandas<3, numpy<2, pyarrow) and everything downstream reads the parquet.

Unpickling can execute arbitrary code, so `RestrictedUnpickler` only resolves the classes listed in
`ALLOWED_GLOBALS` (the pandas/numpy constructors a DataFrame pickle needs) and refuses anything else
before it is called. This file must stay importable with only the standard library at module level:
it runs as a standalone script under `python -I`, outside the project environment.

Usage:
    python -I _tdc_pickle_convert.py <in.pkl> <out.parquet> [--drop COL ...]
"""

from __future__ import annotations

import argparse
import io
import json
import pickle
import sys
from pathlib import Path
from typing import Any

#: (module, name) pairs a TDC DrugSyn DataFrame pickle may reference. Taken from a pickletools scan
#: of drugcomb_nci60.pkl (sha256 c0e195c9...ad00bf).
ALLOWED_GLOBALS: frozenset[tuple[str, str]] = frozenset(
    {
        ("builtins", "slice"),
        ("numpy", "dtype"),
        ("numpy", "ndarray"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("pandas.core.frame", "DataFrame"),
        ("pandas.core.indexes.base", "Index"),
        ("pandas.core.indexes.base", "_new_Index"),
        ("pandas.core.indexes.range", "RangeIndex"),
        ("pandas.core.internals.managers", "BlockManager"),
    }
)


class RestrictedUnpickler(pickle.Unpickler):
    """An unpickler that resolves only `ALLOWED_GLOBALS` and refuses every other global."""

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in ALLOWED_GLOBALS:
            raise pickle.UnpicklingError(f"global '{module}.{name}' is not allowed")
        return super().find_class(module, name)


def restricted_load(data: bytes) -> Any:
    """Unpickle `data`, allowing only `ALLOWED_GLOBALS`."""
    return RestrictedUnpickler(io.BytesIO(data)).load()


def convert(src: Path, dst: Path, drop: list[str]) -> dict[str, Any]:
    """Load `src` with the restricted unpickler, drop `drop` columns, write `dst` as parquet."""
    import pandas as pd  # noqa: PLC0415 - only available inside the isolated environment

    df = restricted_load(src.read_bytes())
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"expected a DataFrame in {src}, got {type(df).__name__}")
    missing = [c for c in drop if c not in df.columns]
    if missing:
        raise KeyError(f"columns to drop not found in {src}: {missing}")
    df = df.drop(columns=drop)
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst, index=False)
    return {"rows": int(len(df)), "columns": list(df.columns), "dropped": drop,
            "pandas": pd.__version__}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("src", type=Path)
    p.add_argument("dst", type=Path)
    p.add_argument("--drop", nargs="*", default=[])
    args = p.parse_args(argv)
    print(json.dumps(convert(args.src, args.dst, args.drop)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
