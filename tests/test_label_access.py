"""Label access (target_design.md §5, gap analysis §2.6): only allowlisted code may read the
hidden outcomes in `labels.parquet`.

The scan covers every Python file in `src/` and `scripts/`, so a new selector, model or tool that
names the labels file fails here until someone consciously adds it to the allowlist.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: path (relative to the repo root) -> why it may name the labels file
ALLOWLIST = {
    "src/combo_agent/data/freeze_pool.py": "writes labels.parquet (M1 step 1g)",
    "src/combo_agent/lab/oracle.py": "the oracle: the only reader on the selector path",
    "src/combo_agent/lab/splits.py": "seed-set and pair-fame build: hit counts for the M2 card only",
    "src/combo_agent/models/legacy/calibrate.py": "M3 calibration report (evaluation code, never imported by selectors)",
    "src/combo_agent/eval/memory_probe.py": "M5 memory-probe scoring (evaluation code, run after the probe)",
    "src/combo_agent/eval/compare.py": "surrogate comparison: top-k recall and probe calibration (evaluation code)",
    "src/combo_agent/tools/store.py": "names it on the tools denylist",
    "scripts/make_data_bundle.py": "packages the frozen files",
    "scripts/make_m1_figures.py": "M1 figures (outcome distributions)",
}

#: Ways code names the labels table: the file name, the oracle's constant, or the table key.
PATTERN = re.compile(r"labels\.parquet|LABELS_FILE|[\"']labels[\"']")


def _sources() -> list[Path]:
    return sorted([*(ROOT / "src").rglob("*.py"), *(ROOT / "scripts").glob("*.py")])


def test_only_allowlisted_code_names_the_labels_file():
    offenders = [str(p.relative_to(ROOT)) for p in _sources()
                 if PATTERN.search(p.read_text()) and str(p.relative_to(ROOT)) not in ALLOWLIST]
    assert not offenders, f"code outside the label allowlist names labels.parquet: {offenders}"


def test_allowlist_has_no_stale_entries():
    for rel in ALLOWLIST:
        assert (ROOT / rel).exists(), f"allowlisted file {rel} no longer exists"


def test_scan_catches_a_selector_that_reads_labels(tmp_path):
    rogue = 'import pandas as pd\nlab = pd.read_parquet("data_cache/processed/labels.parquet")\n'
    assert PATTERN.search(rogue)
    assert PATTERN.search("tables['labels']") and not PATTERN.search("labels=list(TERTILES)")
