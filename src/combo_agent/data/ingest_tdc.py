"""Load the TDC DrugSyn tables (target_design.md section 4, step 1).

The `tdc` package is not used: PyTDC 1.1.15 pins pandas<3 and numpy<2, which conflicts with this
project. Its DrugSyn loader only downloads one pandas pickle per dataset from Harvard Dataverse, so
this module does the same directly:

1. `download` fetches the file into `data_cache/raw/`, checks its sha256 against the pinned value,
   and records URL, size, hash and download date in `raw/manifest.json`.
2. `convert` loads the pickle once, in an isolated pandas<3/numpy<2 environment, with a restricted
   unpickler (see `_tdc_pickle_convert.py`), and writes parquet to `data_cache/interim/`.
3. `load` reads that parquet and checks row count and columns.

Only the NCI-60 table (TDC name `drugcomb`) is pinned for now; O'Neil (`oncopolypharmacology`) is
deferred until its outcome columns are sorted out.

Run: python -m combo_agent.data.ingest_tdc
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
import sys
import time
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from combo_agent.data import paths

DATAVERSE_URL = "https://dataverse.harvard.edu/api/access/datafile/{file_id}"
#: Dataverse answers 403 to urllib's default "Python-urllib/x.y" User-Agent (checked 2026-11-08),
#: so requests name this client explicitly.
USER_AGENT = "combo-agent-data-ingest/0.1"
CONVERTER = Path(__file__).with_name("_tdc_pickle_convert.py")

#: The isolated environment the pickle is loaded in, matching the era it was written in. Pinned
#: exactly to a combination verified to work (2026-11-08); pyarrow 26 refuses numpy<2 at import.
CONVERT_ENV: tuple[str, ...] = ("pandas==2.3.3", "numpy==1.26.4", "pyarrow==25.0.1")
CONVERT_PYTHON = "3.12"


@dataclass(frozen=True)
class TdcSource:
    """One TDC DrugSyn file and what we expect it to contain."""

    name: str
    dataverse_id: int
    filename: str
    sha256: str
    n_rows: int
    columns: tuple[str, ...]
    #: Columns removed during conversion. `CellLine` holds TDC's own per-line feature arrays of
    #: undocumented provenance; cell features come from DepMap instead (section 4, step 6).
    drop_columns: tuple[str, ...] = ()

    @property
    def url(self) -> str:
        return DATAVERSE_URL.format(file_id=self.dataverse_id)

    @property
    def parquet_name(self) -> str:
        return f"{self.name}.parquet"


SOURCES: dict[str, TdcSource] = {
    "tdc_drugcomb": TdcSource(
        name="tdc_drugcomb",
        dataverse_id=4215720,
        filename="drugcomb_nci60.pkl",
        sha256="c0e195c943d5a238e27e0a3e389c4637f9ac0212309c250f9d823b7196ad00bf",
        n_rows=297_098,
        columns=(
            "Drug1_ID", "Drug2_ID", "Cell_Line_ID", "CSS", "Synergy_ZIP", "Synergy_Bliss",
            "Synergy_Loewe", "Synergy_HSA", "Drug1", "Drug2",
        ),
        drop_columns=("CellLine",),
    ),
}


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """Hex sha256 of a file, read in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _record(manifest: Path, filename: str, entry: dict[str, object]) -> None:
    data = json.loads(manifest.read_text()) if manifest.exists() else {}
    data[filename] = entry
    manifest.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


class IncompleteDownload(OSError):
    """The response body was shorter than its declared Content-Length."""


def _fetch_once(url: str, tmp: Path) -> int:
    """Stream `url` into `tmp`; returns bytes written, raises IncompleteDownload if truncated."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as resp, open(tmp, "wb") as out:
        expected_len = resp.headers.get("Content-Length")
        while block := resp.read(1 << 20):
            out.write(block)
    got = tmp.stat().st_size
    # A connection dropped mid-body can end the read loop early without an error, leaving a
    # silently truncated file (observed 2026-11-08: 36,372,455 of 36,394,069 bytes).
    if expected_len is not None and got != int(expected_len):
        tmp.unlink()
        raise IncompleteDownload(f"incomplete download of {url}: {got} of {expected_len} bytes")
    return got


def download(
    source: TdcSource,
    raw_dir: Path | None = None,
    attempts: int = 3,
    sleep: Callable[[float], None] = time.sleep,
) -> Path:
    """Download `source` into `raw_dir`, verifying its sha256. Skips if a verified copy exists.

    Truncated transfers are retried up to `attempts` times with backoff. A complete transfer whose
    hash differs from the pinned one is not retried: that means the served file changed.
    """
    raw_dir = raw_dir or paths.raw_dir()
    raw_dir.mkdir(parents=True, exist_ok=True)
    dst = raw_dir / source.filename
    if dst.exists():
        if sha256_file(dst) == source.sha256:
            return dst
        raise ValueError(f"{dst} exists but its sha256 does not match; delete it to re-download")
    tmp = dst.with_suffix(dst.suffix + ".part")
    for attempt in range(1, attempts + 1):
        try:
            got = _fetch_once(source.url, tmp)
            break
        except IncompleteDownload:
            if attempt == attempts:
                raise
            sleep(2.0 * attempt)
    digest = sha256_file(tmp)
    if digest != source.sha256:
        tmp.unlink()
        raise ValueError(f"sha256 mismatch for {source.url} ({got} bytes): got {digest}, "
                         f"expected {source.sha256}")
    tmp.replace(dst)
    _record(raw_dir / "manifest.json", source.filename, {
        "url": source.url,
        "sha256": digest,
        "bytes": dst.stat().st_size,
        "downloaded_on": dt.date.today().isoformat(),
    })
    return dst


def convert_command(source: TdcSource, raw_path: Path, out_path: Path) -> list[str]:
    """The isolated `uv run` command that converts the pickle to parquet."""
    cmd = ["uv", "run", "--isolated", "--no-project", "--python", CONVERT_PYTHON]
    for req in CONVERT_ENV:
        cmd += ["--with", req]
    cmd += ["python", "-I", str(CONVERTER), str(raw_path), str(out_path)]
    if source.drop_columns:
        cmd += ["--drop", *source.drop_columns]
    return cmd


def convert(
    source: TdcSource,
    raw_path: Path,
    out_dir: Path | None = None,
    run: Callable[[Sequence[str]], object] | None = None,
) -> Path:
    """Convert the downloaded pickle to parquet in `out_dir` (default `data_cache/interim/`)."""
    out = (out_dir or paths.interim_dir()) / source.parquet_name
    if out.exists():
        return out
    cmd = convert_command(source, raw_path, out)
    (run or (lambda c: subprocess.run(c, check=True)))(cmd)
    if not out.exists():
        raise FileNotFoundError(f"conversion did not produce {out}")
    return out


def load(name: str, interim_dir: Path | None = None) -> pd.DataFrame:
    """Read a converted TDC table and check it has the expected rows and columns."""
    source = SOURCES[name]
    path = (interim_dir or paths.interim_dir()) / source.parquet_name
    df = pd.read_parquet(path)
    if len(df) != source.n_rows:
        raise ValueError(f"{path}: {len(df)} rows, expected {source.n_rows}")
    if tuple(df.columns) != source.columns:
        raise ValueError(f"{path}: columns {list(df.columns)}, expected {list(source.columns)}")
    return df


def ingest(name: str) -> Path:
    """Download and convert one source; returns the parquet path."""
    source = SOURCES[name]
    return convert(source, download(source))


def main(argv: list[str] | None = None) -> int:
    names = argv if argv else sorted(SOURCES)
    for name in names:
        out = ingest(name)
        df = load(name)
        print(f"{name}: {len(df):,} rows -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
