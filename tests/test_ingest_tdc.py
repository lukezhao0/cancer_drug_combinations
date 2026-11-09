"""Tests for combo_agent.data.ingest_tdc and the restricted pickle converter."""

from __future__ import annotations

import hashlib
import io
import os
import pickle
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from combo_agent.data import _tdc_pickle_convert as conv
from combo_agent.data import ingest_tdc

NCI60 = ingest_tdc.SOURCES["tdc_drugcomb"]


# -- restricted unpickler ------------------------------------------------------------------------


class _Evil:
    """Pickles to a call of os.system; dumping it does not execute anything."""

    def __reduce__(self):
        return (os.system, ("echo pwned",))


def test_restricted_unpickler_refuses_disallowed_global():
    blob = pickle.dumps(_Evil())
    with pytest.raises(pickle.UnpicklingError, match="not allowed"):
        conv.restricted_load(blob)


def test_restricted_unpickler_allows_listed_global():
    assert conv.restricted_load(pickle.dumps(slice(1, 5, 2))) == slice(1, 5, 2)


def test_allowlist_is_small_and_pandas_numpy_only():
    modules = {m.split(".")[0] for m, _ in conv.ALLOWED_GLOBALS}
    assert modules == {"builtins", "numpy", "pandas"}
    assert ("builtins", "slice") in conv.ALLOWED_GLOBALS
    assert not any(m == "builtins" and n != "slice" for m, n in conv.ALLOWED_GLOBALS)


# -- download ------------------------------------------------------------------------------------


class _Resp(io.BytesIO):
    """Stand-in for an HTTP response: a byte stream plus headers."""

    def __init__(self, body: bytes, content_length: int | None = None):
        super().__init__(body)
        self.headers = {} if content_length is None else {"Content-Length": str(content_length)}


def _fake_source(tmp_path: Path, payload: bytes) -> tuple[ingest_tdc.TdcSource, bytes]:
    return replace(NCI60, filename="x.pkl", sha256=hashlib.sha256(payload).hexdigest(),
                   dataverse_id=0), payload


def _serve(payload: bytes, content_length: int | None = None):
    length = len(payload) if content_length is None else content_length
    return lambda request, timeout: _Resp(payload, length)


def test_download_verifies_hash_and_writes_manifest(tmp_path, monkeypatch):
    source, served = _fake_source(tmp_path, b"hello")
    monkeypatch.setattr(ingest_tdc.urllib.request, "urlopen", _serve(served))
    out = ingest_tdc.download(source, tmp_path / "raw")
    assert out.read_bytes() == b"hello"
    manifest = (tmp_path / "raw" / "manifest.json").read_text()
    assert source.sha256 in manifest and source.url in manifest


def test_download_sends_explicit_user_agent(tmp_path, monkeypatch):
    source, served = _fake_source(tmp_path, b"hello")
    seen = []

    def fake_urlopen(request, timeout):
        seen.append(request.get_header("User-agent"))
        return _Resp(served, len(served))

    monkeypatch.setattr(ingest_tdc.urllib.request, "urlopen", fake_urlopen)
    ingest_tdc.download(source, tmp_path / "raw")
    assert seen == [ingest_tdc.USER_AGENT]
    assert not seen[0].startswith("Python-urllib")


def test_download_rejects_hash_mismatch_and_leaves_nothing(tmp_path, monkeypatch):
    source, served = _fake_source(tmp_path, b"hello")
    source = replace(source, sha256="0" * 64)
    monkeypatch.setattr(ingest_tdc.urllib.request, "urlopen", _serve(served))
    with pytest.raises(ValueError, match="sha256 mismatch"):
        ingest_tdc.download(source, tmp_path / "raw")
    assert list((tmp_path / "raw").iterdir()) == []


def test_download_rejects_truncated_body_and_leaves_nothing(tmp_path, monkeypatch):
    source, served = _fake_source(tmp_path, b"hello")
    # server declared 10 bytes but the stream ended after 5 (dropped connection)
    monkeypatch.setattr(ingest_tdc.urllib.request, "urlopen", _serve(served, content_length=10))
    waits = []
    with pytest.raises(ingest_tdc.IncompleteDownload, match="5 of 10 bytes"):
        ingest_tdc.download(source, tmp_path / "raw", attempts=3, sleep=waits.append)
    assert waits == [2.0, 4.0]  # retried twice before giving up
    assert list((tmp_path / "raw").iterdir()) == []


def test_download_retries_truncated_then_succeeds(tmp_path, monkeypatch):
    source, served = _fake_source(tmp_path, b"hello")
    responses = iter([_Resp(b"hel", 5), _Resp(served, 5)])
    monkeypatch.setattr(ingest_tdc.urllib.request, "urlopen", lambda r, timeout: next(responses))
    out = ingest_tdc.download(source, tmp_path / "raw", sleep=lambda s: None)
    assert out.read_bytes() == b"hello"


def test_download_skips_verified_copy_without_network(tmp_path, monkeypatch):
    source, _ = _fake_source(tmp_path, b"hello")
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / source.filename).write_bytes(b"hello")

    def no_network(*a, **k):
        raise AssertionError("should not download")

    monkeypatch.setattr(ingest_tdc.urllib.request, "urlopen", no_network)
    assert ingest_tdc.download(source, raw) == raw / source.filename


def test_download_refuses_corrupt_existing_copy(tmp_path):
    source, _ = _fake_source(tmp_path, b"hello")
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / source.filename).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="does not match"):
        ingest_tdc.download(source, raw)


# -- convert / load ------------------------------------------------------------------------------


def test_convert_command_is_isolated_and_drops_cellline(tmp_path):
    cmd = ingest_tdc.convert_command(NCI60, tmp_path / "in.pkl", tmp_path / "out.parquet")
    assert cmd[:4] == ["uv", "run", "--isolated", "--no-project"]
    assert "pandas==2.3.3" in cmd and "numpy==1.26.4" in cmd and "pyarrow==25.0.1" in cmd
    assert cmd[cmd.index("python") + 1] == "-I"
    assert cmd[-2:] == ["--drop", "CellLine"]


def test_convert_runs_command_and_returns_output(tmp_path):
    seen = []

    def fake_run(cmd):
        seen.append(cmd)
        Path(cmd[cmd.index("--drop") - 1]).write_bytes(b"parquet")

    out = ingest_tdc.convert(NCI60, tmp_path / "in.pkl", tmp_path, run=fake_run)
    assert out == tmp_path / "tdc_drugcomb.parquet" and len(seen) == 1
    # second call is a no-op because the output exists
    ingest_tdc.convert(NCI60, tmp_path / "in.pkl", tmp_path, run=fake_run)
    assert len(seen) == 1


def test_load_checks_rows_and_columns(tmp_path, monkeypatch):
    small = replace(NCI60, n_rows=2)
    monkeypatch.setitem(ingest_tdc.SOURCES, "tdc_drugcomb", small)
    df = pd.DataFrame({c: [1, 2] for c in NCI60.columns})
    df.to_parquet(tmp_path / small.parquet_name, index=False)
    assert len(ingest_tdc.load("tdc_drugcomb", tmp_path)) == 2
    df.iloc[:1].to_parquet(tmp_path / small.parquet_name, index=False)
    with pytest.raises(ValueError, match="rows"):
        ingest_tdc.load("tdc_drugcomb", tmp_path)


@pytest.mark.skipif(os.environ.get("COMBO_INTEGRATION") != "1",
                    reason="downloads 36 MB and needs uv; set COMBO_INTEGRATION=1")
def test_ingest_nci60_end_to_end():
    out = ingest_tdc.ingest("tdc_drugcomb")
    df = ingest_tdc.load("tdc_drugcomb", out.parent)
    assert df["Cell_Line_ID"].nunique() == 59
