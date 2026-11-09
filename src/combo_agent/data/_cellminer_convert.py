"""Convert one CellMiner processed-dataset workbook (.xls) to parquet.

CellMiner (https://discover.nci.nih.gov/cellminer) ships its NCI-60 tables as legacy Excel
workbooks. The project has no .xls reader, so `ingest_cellminer` runs this file once in an isolated
environment (xlrd + pandas + pyarrow) and everything downstream reads the parquet. This file must
stay importable with only the standard library at module level: it runs as a standalone script
under `python -I`, outside the project environment.

Layout of the "Results" sheet, verified on CellMiner database version 2.16 (2026-11-08):
* rows 0-5 are metadata (database version, genome build, date);
* row 9 holds the units;
* row 10 is the header: 6 gene-metadata columns ("Gene name d", "Entrez gene id e", "Chromosome f",
  "Start f", "End f", "Cytoband f"), then one column per cell line named "<panel>:<line>";
* data starts at row 11.

Cell values: a number, or "-" / "" for no value. The meaning of "no value" depends on the dataset
(for exome variants it means "no variant called"), so it is written as null here and interpreted
downstream.

Usage:
    python -I _cellminer_convert.py <in.xls> <out.parquet>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HEADER_ROW = 10
META_COLUMNS = ("gene", "entrez_id", "chromosome", "start", "end", "cytoband")


def convert(src: Path, dst: Path) -> dict:
    import pandas as pd  # noqa: PLC0415 - only available inside the isolated environment
    import xlrd  # noqa: PLC0415

    book = xlrd.open_workbook(str(src))
    sheet = book.sheet_by_name("Results")
    meta = {str(sheet.cell_value(r, 0)).rstrip(":"): str(sheet.cell_value(r, 1)) for r in range(6)}
    units = str(sheet.cell_value(9, 6)).strip()
    header = [str(v).strip() for v in sheet.row_values(HEADER_ROW)]
    lines = header[len(META_COLUMNS):]
    if not all(":" in h for h in lines):
        raise ValueError(f"{src}: unexpected line headers {lines[:5]}")
    rows = [sheet.row_values(r) for r in range(HEADER_ROW + 1, sheet.nrows)]

    def num(v):
        if v in ("", "-", None):
            return None
        return float(v)

    data = {c: [r[i] for r in rows] for i, c in enumerate(META_COLUMNS)}
    df = pd.DataFrame(data)
    df["gene"] = df["gene"].astype(str)
    df["entrez_id"] = pd.to_numeric(df["entrez_id"], errors="coerce").astype("Int64")
    for c in ("chromosome", "cytoband"):
        df[c] = df[c].astype(str)
    for c in ("start", "end"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for j, line in enumerate(lines, start=len(META_COLUMNS)):
        df[line] = pd.array([num(r[j]) for r in rows], dtype="Float64")
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst, index=False)
    return {"rows": len(df), "lines": len(lines), "units": units, "meta": meta}


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 2:
        print(__doc__)
        return 2
    print(json.dumps(convert(Path(args[0]), Path(args[1]))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
