"""NCI CellMiner: NCI-60 omics measured by NCI itself on all 60 lines (input to step 1f).

Why CellMiner as well as DepMap: DepMap lacks expression for 10 of our 57 pool lines, copy number
for 15, and any sequencing for SNB-19 and NCI-H322M (docs/project-notes/depmap-data.md). CellMiner covers every
NCI-60 line, including SNB-19, NCI-H322M and HL-60(TB) itself. So for expression and copy number it
can be the *single* source for all pool lines, which avoids mixing two labs' pipelines in one
feature block.

Datasets, from https://discover.nci.nih.gov/cellminer/loadDownload.do (database version 2.16, files
dated 2026-10-28, downloaded 2026-11-08):

| key | file | values |
|:--|:--|:--|
| `expression` | nci60_RNA__RNA_seq_composite_expression.zip | log2(FPKM + 1), RNA-seq (PMID 31113817) |
| `mutation_pfa` | nci60_DNA__Exome_Seq_Protein_function_affecting.zip | summed variant allele % for variants predicted to affect protein function; "-" = none called (PMID 23856246) |
| `mutation_aa` | nci60_DNA__Exome_Seq_AA_changing.zip | the same for any amino-acid-changing variant |
| `copy_number` | nci60_DNA__Combined_aCGH_gene_summary.zip | log2 ratio to normal DNA, from 4 platforms combined; copy number N = 2 * 2^value (PMID 24670534) |
| `expression_array` | nci60_RNA__5_Platform_Gene_Transcript_Average_z_scores.zip | microarray z-scores across the 60 lines, averaged over 5 platforms; used only to impute a line whose RNA-seq sample fails the identity check |

The zips are pinned by sha256. `convert` reads each workbook in an isolated xlrd environment
(`_cellminer_convert.py`) and writes `data_cache/interim/cellminer_<key>.parquet`.

Run:
    python -m combo_agent.data.ingest_cellminer        # download if missing, verify, convert
"""

from __future__ import annotations

import re
import subprocess
import sys
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from combo_agent.data import paths
from combo_agent.data.ingest_tdc import _record, sha256_file

BASE_URL = "https://discover.nci.nih.gov/cellminer/download/processeddataset/"
USER_AGENT = "combo-agent-data-ingest/0.1"
CONVERTER = Path(__file__).with_name("_cellminer_convert.py")
CONVERT_ENV: tuple[str, ...] = ("xlrd==2.0.1", "pandas==2.3.3", "pyarrow==25.0.1", "numpy==1.26.4")
CONVERT_PYTHON = "3.12"
N_LINES = 60


@dataclass(frozen=True)
class Dataset:
    key: str
    zip_name: str
    workbook: str  # path inside the zip
    sha256: str


DATASETS: dict[str, Dataset] = {d.key: d for d in (
    Dataset("expression", "nci60_RNA__RNA_seq_composite_expression.zip",
            "output/RNA__RNA_seq_composite_expression.xls",
            "47712a4d9149fcb9b8650952e1c0a4a322eb7ee258784197f7d8766cf59e871e"),
    Dataset("mutation_pfa", "nci60_DNA__Exome_Seq_Protein_function_affecting.zip",
            "output/DNA__Exome_Seq_Protein_function_affecting.xls",
            "72e863cdc02b2b9b5ca027da901d4090443e1f9372da7aaea20c3fd46b3f27bb"),
    Dataset("mutation_aa", "nci60_DNA__Exome_Seq_AA_changing.zip",
            "output/DNA__Exome_Seq_AA_changing.xls",
            "49f06f4ec6c826f3b7bdbf943fdc2fb59ed83b5bd8b1299378f183634760dd20"),
    Dataset("copy_number", "nci60_DNA__Combined_aCGH_gene_summary.zip",
            "output/DNA__Combined_aCGH_gene_summary.xls",
            "ca4a6e225d5e52e49872837577c548436cd5ad2b7dfff1f062f8e88470dde664"),
    # Microarray expression, z-scored across the 60 lines and averaged over 5 platforms. Used only to
    # impute a line whose RNA-seq sample fails the identity check (SR, 2026-11-08). In these arrays
    # SR shows its true ALK+ ALCL profile.
    Dataset("expression_array", "nci60_RNA__5_Platform_Gene_Transcript_Average_z_scores.zip",
            "output/RNA__5_Platform_Gene_Transcript_Average_z_scores.xls",
            "dd5462ed76765639242420309f63823c50517e9a4870b622aaca7f3195bd571a"),
)}


def raw_dir() -> Path:
    return paths.raw_dir() / "cellminer"


def download(ds: Dataset, root: Path | None = None) -> Path:
    """Fetch the zip if absent, then verify its pinned sha256 and record it in the manifest."""
    root = root or raw_dir()
    root.mkdir(parents=True, exist_ok=True)
    dst = root / ds.zip_name
    if not dst.exists():
        req = urllib.request.Request(BASE_URL + ds.zip_name, headers={"User-Agent": USER_AGENT})
        tmp = dst.with_suffix(".part")
        with urllib.request.urlopen(req, timeout=600) as resp, open(tmp, "wb") as out:
            while block := resp.read(1 << 20):
                out.write(block)
        tmp.replace(dst)
    digest = sha256_file(dst)
    if digest != ds.sha256:
        raise ValueError(f"{dst}: sha256 {digest}, expected {ds.sha256}; the served file changed")
    _record(root / "manifest.json", ds.zip_name, {"url": BASE_URL + ds.zip_name, "sha256": digest,
                                                   "bytes": dst.stat().st_size,
                                                   "cellminer_db_version": "2.16"})
    return dst


def extract(zip_path: Path, ds: Dataset) -> Path:
    """Extract only the workbook, into a folder of its own next to the zip."""
    import zipfile

    out_dir = zip_path.with_suffix("")
    target = out_dir / ds.workbook
    if not target.exists():
        with zipfile.ZipFile(zip_path) as z:
            member = z.getinfo(ds.workbook)  # KeyError if the layout changed
            if Path(member.filename).is_absolute() or ".." in Path(member.filename).parts:
                raise ValueError(f"unsafe path in {zip_path}: {member.filename}")
            z.extract(member, out_dir)
    return target


def convert_command(workbook: Path, out: Path) -> list[str]:
    cmd = ["uv", "run", "--isolated", "--no-project", "--python", CONVERT_PYTHON]
    for req in CONVERT_ENV:
        cmd += ["--with", req]
    return cmd + ["python", "-I", str(CONVERTER), str(workbook), str(out)]


def parquet_path(key: str, interim: Path | None = None) -> Path:
    return (interim or paths.interim_dir()) / f"cellminer_{key}.parquet"


def convert(ds: Dataset, workbook: Path, interim: Path | None = None,
            run: Callable[[Sequence[str]], object] | None = None) -> Path:
    out = parquet_path(ds.key, interim)
    if not out.exists():
        (run or (lambda c: subprocess.run(c, check=True)))(convert_command(workbook, out))
    return out


def load(key: str, interim: Path | None = None) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path(key, interim))
    lines = [c for c in df.columns if ":" in c]
    if len(lines) != N_LINES:
        raise ValueError(f"cellminer_{key}: {len(lines)} line columns, expected {N_LINES}")
    return df


# -- line names -----------------------------------------------------------------------------------


def line_label(column: str) -> str:
    """'CO:HCT-116' -> 'HCT-116'; 'LC:A549/ATCC' -> 'A549' (the ATCC stock of A549)."""
    name = column.split(":", 1)[1]
    return re.sub(r"/ATCC$", "", name)


def match_lines(columns: Sequence[str], line_map: pd.DataFrame, cell_lines: pd.DataFrame) -> dict[str, str]:
    """CellMiner column -> Cellosaurus accession, by exact normalized name.

    A column matches when its normalized name equals the normalized TDC label or the Cellosaurus
    identifier of exactly one of our lines. Columns for lines we don't have (MDA-N) map to nothing.
    Raises if a column matches two lines.
    """
    from combo_agent.data.cellosaurus import normalize_name

    keys: dict[str, set[str]] = {}
    for r in line_map.itertuples():
        keys.setdefault(normalize_name(r.source_id), set()).add(r.cellosaurus_ac)
    for r in cell_lines.itertuples():
        keys.setdefault(normalize_name(r.cellosaurus_name), set()).add(r.cellosaurus_ac)
    out: dict[str, str] = {}
    for col in columns:
        hits = keys.get(normalize_name(line_label(col)), set())
        if len(hits) > 1:
            raise ValueError(f"CellMiner column {col!r} matches several lines: {sorted(hits)}")
        if hits:
            out[col] = next(iter(hits))
    return out


def ingest() -> dict[str, Path]:
    out = {}
    for ds in DATASETS.values():
        out[ds.key] = convert(ds, extract(download(ds), ds))
    return out


def main(argv: list[str] | None = None) -> int:
    for key, path in ingest().items():
        df = load(key)
        print(f"{key}: {len(df):,} genes x {sum(':' in c for c in df.columns)} lines -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
