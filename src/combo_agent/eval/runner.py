"""Campaign runner (target_design.md §11): selector x target x seed campaigns against the oracle.

One campaign:

    oracle = Oracle.from_cache(seed)            # 32 seed experiments revealed free (round 0)
    repeat until the budget (320) is spent:
        refit the shared response model on the history   (skipped for model-free selectors)
        predict every untested candidate
        selector picks a batch of 16                     -> oracle.run(batch)

Each campaign writes two files under `<out>/runs/`:

    <selector>__<target>__s<seed>.parquet           one row per round: cumulative hit counts, timings
    <selector>__<target>__s<seed>.reveals.parquet   every revealed experiment, with its values

The rounds file is written last and atomically, so its presence means the campaign finished; a
rerun skips it (resume). Hits use the oracle's public thresholds; `syn_strict` drops synergy hits
whose killing score is negative (`syn_hit_low_kill`, M1 decision 20). No hidden outcome is read.

Run a preset grid:

    python -m combo_agent.eval.runner --grid pilot --out runs/m4_pilot
    python -m combo_agent.eval.runner --grid main  --out runs/m4_main  [--workers 2]

Presets: `pilot` (5 selectors x 2 targets x seeds 0-1, compact features), `main` (5 x 2 x seeds
0-9, full features), `ablation` (bo_ucb beta and diversity variants, seeds 0-9, full features),
`gpcompare` (the 5 model-based selectors x 2 targets x seeds 0-4).

`--surrogate {ensemble,gp}` picks the response model (default: the ensemble). Model-based
campaigns also write `<run_id>.preds.parquet`: per-round predictions for the batch and a fixed
probe set, for calibration diagnostics (eval/diagnostics.py).

    python -m combo_agent.eval.runner --grid gpcompare --surrogate gp --out runs/gp_compare/gp
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from combo_agent.data import paths
from combo_agent.lab.oracle import KEY, Oracle, Result
from combo_agent.eval.diagnostics import DistanceSpace, probe_keys, round_records
from combo_agent.models.features import FeatureSet
from combo_agent.models.gp import GPFeatures
from combo_agent.models.surrogate import SURROGATES, make_surrogate
from combo_agent.selectors import SelectionContext, make
from combo_agent.selectors.base import TARGETS

MAIN = ("random", "greedy", "uncertainty", "bo_ucb", "bo_qei")
ABLATION = ("bo_ucb", "bo_ucb_b1", "bo_ucb_b3", "bo_ucb_nodiv")
MODEL_SELECTORS = ("greedy", "uncertainty", "bo_ucb", "bo_qei", "bo_ts")
GRIDS = {
    "pilot": {"selectors": MAIN, "targets": TARGETS, "seeds": (0, 1), "mode": "compact"},
    "main": {"selectors": MAIN, "targets": TARGETS, "seeds": tuple(range(10)), "mode": "full"},
    "ablation": {"selectors": ABLATION, "targets": TARGETS, "seeds": tuple(range(10)), "mode": "full"},
    # Surrogate comparison: run once with --surrogate ensemble and once with --surrogate gp.
    "gpcompare": {"selectors": MODEL_SELECTORS, "targets": TARGETS, "seeds": tuple(range(5)), "mode": "full"},
}


@dataclass(frozen=True)
class CampaignSpec:
    selector: str
    target: str
    seed: int
    budget: int = 320
    batch_size: int = 16

    @property
    def run_id(self) -> str:
        return f"{self.selector}__{self.target}__s{self.seed}"


def _is_hit(r: Result, kind: str, thresholds: dict) -> bool:
    if kind == "syn":
        return r.y_syn >= thresholds["y_syn"]
    if kind == "syn_strict":
        return r.y_syn >= thresholds["y_syn"] and r.y_inh >= 0
    return r.y_inh >= thresholds["y_inh"]


def _round_row(spec: CampaignSpec, oracle: Oracle, rnd: int, timings: dict) -> dict:
    hist = oracle.history()
    charged = [r for r in hist if r.round > 0]
    row = {"run_id": spec.run_id, "selector": spec.selector, "target": spec.target, "seed": spec.seed,
           "round": rnd, "n_charged": len(charged), "n_revealed": len(hist), **timings}
    for kind in ("syn", "syn_strict", "inh"):
        hits = [r for r in charged if _is_hit(r, kind, oracle.thresholds)]
        row[f"hits_{kind}"] = len(hits)
        row[f"hit_pairs_{kind}"] = len({(r.triplet.drug_a, r.triplet.drug_b) for r in hits})
        row[f"seed_hits_{kind}"] = sum(_is_hit(r, kind, oracle.thresholds) for r in hist if r.round == 0)
    return row


def run_campaign(spec: CampaignSpec, features: FeatureSet | GPFeatures | None = None, model_kw: dict | None = None,
                 oracle_factory: Callable[..., Oracle] | None = None,
                 log: Callable[[str], None] | None = None, surrogate: str = "ensemble",
                 space: DistanceSpace | None = None,
                 diagnostics: list[pd.DataFrame] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run one campaign; return (rounds, reveals) frames.

    `surrogate` picks the response model ("ensemble" or "gp"); `features` must match it
    (FeatureSet or GPFeatures). With `space` and a `diagnostics` list, each round's predictions
    for the batch and the probe set are appended to the list (eval/diagnostics.py). Diagnostics
    draw no random numbers from the selection rng, so the picks are the same with or without them.
    """
    if spec.target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}")
    selector = make(spec.selector)
    factory = oracle_factory or Oracle.from_cache
    oracle = factory(seed=spec.seed, selector=spec.selector, budget=spec.budget, batch_size=spec.batch_size)
    model_kw = dict(model_kw or {})
    rng = np.random.default_rng([spec.seed, sum(map(ord, spec.selector)), TARGETS.index(spec.target)])
    rows = [_round_row(spec, oracle, 0, {"fit_s": 0.0, "predict_s": 0.0, "select_s": 0.0})]
    probe = None
    while oracle.budget_left() > 0:
        rnd = oracle.round + 1
        cand = oracle.candidate_frame()
        model = pred = None
        t0 = t1 = time.perf_counter()
        if selector.needs_model:
            model = make_surrogate(surrogate, features, seed=spec.seed * 1000 + rnd, **model_kw).fit(oracle.history())
            t1 = time.perf_counter()
        t2 = t1
        if model is not None:
            pred = model.predict(cand)
            t2 = time.perf_counter()
        ctx = SelectionContext(candidates=cand, history=oracle.history(), model=model, pred=pred)
        batch = selector.select(ctx, min(spec.batch_size, oracle.budget_left()), spec.target, rng)
        t3 = time.perf_counter()
        if diagnostics is not None and space is not None and pred is not None:
            if probe is None:
                probe = probe_keys(cand, spec.seed)
            pos = {k: i for i, k in enumerate(map(tuple, cand[KEY].to_numpy().tolist()))}
            rows_b = np.array([pos[tuple(t)] for t in batch], dtype=int)
            diagnostics.append(round_records(rnd, cand, pred, rows_b, probe, oracle.history(), space))
        oracle.run(batch)
        rows.append(_round_row(spec, oracle, rnd, {"fit_s": t1 - t0, "predict_s": t2 - t1, "select_s": t3 - t2}))
        if log:
            r = rows[-1]
            log(f"{spec.run_id} round {rnd}: charged {r['n_charged']}, hits syn {r['hits_syn']} inh {r['hits_inh']}")
    reveals = pd.DataFrame([{"run_id": spec.run_id, "round": r.round, "drug_a": r.triplet.drug_a,
                             "drug_b": r.triplet.drug_b, "cellosaurus_ac": r.triplet.cellosaurus_ac,
                             "y_syn": r.y_syn, "y_inh": r.y_inh, "n_reps": r.n_reps,
                             "hit_syn": _is_hit(r, "syn", oracle.thresholds),
                             "hit_syn_strict": _is_hit(r, "syn_strict", oracle.thresholds),
                             "hit_inh": _is_hit(r, "inh", oracle.thresholds)} for r in oracle.history()])
    return pd.DataFrame(rows), reveals


# -- grids ----------------------------------------------------------------------------------------

_FEATURES: dict[str, FeatureSet | GPFeatures] = {}


def _features(mode: str) -> FeatureSet:
    if mode not in _FEATURES:
        _FEATURES[mode] = FeatureSet.from_cache(mode)
    return _FEATURES[mode]


def _gp_features() -> GPFeatures:
    if "gp" not in _FEATURES:
        _FEATURES["gp"] = GPFeatures.from_cache()
    return _FEATURES["gp"]


def _surrogate_features(surrogate: str, mode: str):
    return _gp_features() if surrogate == "gp" else _features(mode)


def _write_atomic(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(path.name + ".part")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def run_one(spec: CampaignSpec, out: Path, mode: str, model_kw: dict | None = None,
            features: FeatureSet | GPFeatures | None = None, oracle_factory=None, surrogate: str = "ensemble",
            space: DistanceSpace | None = None, diagnostics: bool = True) -> str:
    """Run a campaign unless its rounds file exists; write reveals (and diagnostics), then rounds.

    Diagnostics need a DistanceSpace: given as `space`, or built from the cached feature tables
    when `features` also comes from the cache."""
    runs = out / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    rounds_path, reveals_path = runs / f"{spec.run_id}.parquet", runs / f"{spec.run_id}.reveals.parquet"
    preds_path = runs / f"{spec.run_id}.preds.parquet"
    if rounds_path.exists():
        return f"skip {spec.run_id}"
    t0 = time.perf_counter()
    needs_model = make(spec.selector).needs_model
    fs = features if features is not None else (_surrogate_features(surrogate, mode) if needs_model else None)
    if space is None and diagnostics and needs_model and features is None:
        space = DistanceSpace(_gp_features())
    diag: list[pd.DataFrame] | None = [] if (diagnostics and space is not None) else None
    rounds, reveals = run_campaign(spec, fs, model_kw, oracle_factory, surrogate=surrogate, space=space,
                                   diagnostics=diag)
    if diag:
        _write_atomic(pd.concat(diag, ignore_index=True).assign(run_id=spec.run_id, surrogate=surrogate), preds_path)
    _write_atomic(reveals, reveals_path)
    _write_atomic(rounds, rounds_path)
    last = rounds.iloc[-1]
    return (f"done {spec.run_id} in {time.perf_counter() - t0:.0f}s: hits syn {last.hits_syn} "
            f"(strict {last.hits_syn_strict}), inh {last.hits_inh}")


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=paths.REPO_ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _sha(p: Path) -> str | None:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def write_manifest(out: Path, grid: dict, mode: str, model_kw: dict, surrogate: str = "ensemble") -> None:
    proc = paths.processed_dir()
    manifest = {
        "grid": {k: list(v) if isinstance(v, tuple) else v for k, v in grid.items()}, "mode": mode,
        "model_kw": model_kw, "surrogate": surrogate, "git_commit": _git("rev-parse", "HEAD"), "git_dirty": bool(_git("status", "--porcelain")),
        "inputs": {name: _sha(proc / name) for name in ("pool.parquet", "splits.json", "m2/seed_sets.json",
                                                         "manifest.json", "m2/manifest.json")},
        "versions": _versions(),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def _versions() -> dict:
    v = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}
    for name in ("torch", "gpytorch", "botorch", "gauche"):
        try:
            v[name] = __import__(name).__version__
        except (ImportError, AttributeError):
            pass
    return v


def _worker(args):
    spec, out, mode, model_kw, surrogate = args
    return run_one(spec, Path(out), mode, model_kw, surrogate=surrogate)


def run_grid(out: Path, selectors, targets, seeds, mode: str, model_kw: dict | None = None,
             workers: int = 1, budget: int = 320, batch_size: int = 16, log=print,
             surrogate: str = "ensemble") -> None:
    if surrogate not in SURROGATES:
        raise ValueError(f"surrogate must be one of {SURROGATES}")
    model_kw = dict(model_kw or {})
    specs = [CampaignSpec(s, t, seed, budget, batch_size) for seed in seeds for t in targets for s in selectors]
    write_manifest(out, {"selectors": tuple(selectors), "targets": tuple(targets), "seeds": tuple(seeds),
                         "budget": budget, "batch_size": batch_size}, mode, model_kw, surrogate)
    log(f"{len(specs)} campaigns ({surrogate}) -> {out}")
    if workers <= 1:
        for spec in specs:
            log(run_one(spec, out, mode, model_kw, surrogate=surrogate))
        return
    # One BLAS thread per worker process; set before the workers import numpy.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"
    with ProcessPoolExecutor(workers) as pool:
        for msg in pool.map(_worker, [(s, str(out), mode, model_kw, surrogate) for s in specs]):
            log(msg)


def _seeds(text: str) -> tuple[int, ...]:
    out = []
    for part in text.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return tuple(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--grid", choices=sorted(GRIDS), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mode", choices=["full", "compact"], default=None, help="override the preset's features")
    ap.add_argument("--seeds", type=_seeds, default=None, help="e.g. 0-9 or 0,3,5")
    ap.add_argument("--selectors", nargs="+", default=None)
    ap.add_argument("--targets", nargs="+", choices=list(TARGETS), default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--surrogate", choices=list(SURROGATES), default="ensemble",
                    help="response model: the 5-network ensemble (default) or the Gaussian process")
    ap.add_argument("--budget", type=int, default=320)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args(argv)
    g = GRIDS[args.grid]
    run_grid(args.out, args.selectors or g["selectors"], args.targets or g["targets"], args.seeds or g["seeds"],
             args.mode or g["mode"], workers=args.workers, budget=args.budget, batch_size=args.batch_size,
             log=lambda m: print(m, flush=True), surrogate=args.surrogate)
    return 0


if __name__ == "__main__":
    sys.exit(main())
