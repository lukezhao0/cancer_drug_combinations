"""Zero-shot memory probe (design section 10, route 3).

Before any agent run, ask the LLM, with no tools and no data, to score 200 pool experiments for
expected synergy and killing. How well its scores track the hidden outcomes, separately for famous,
middle and obscure pairs (M2), estimates how much of the answer it already holds in memory. The
masked variant replaces drug names with codes, keeping targets and mechanism, as a preview of the
M6 masked-name ablation.

This module never reads the labels; `combo_agent.eval.memory_probe` scores its output.

    uv run python -m combo_agent.literature.probe --model gemini-3.5-flash-lite
    uv run python -m combo_agent.literature.probe --model gemini-4-hydrogen [--masked]
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from combo_agent.data import paths
from combo_agent.lab.splits import FAME_ORDER, KEY
from combo_agent.tools.store import FeatureStore

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
PROMPT_VERSION = "memory_probe_v1"
N_TRIPLETS = 200
BATCH = 20
SEED = 0

#: $ per million tokens (input, output), README "Model Choices" (verified 2026-11-16). SHELDON's
#: own table has no Gemini entry and would price these at its default model's rate.
PRICES = {"gemini-3.5-flash-lite": (0.30, 2.50), "gemini-4-hydrogen": (1.50, 7.50)}


class ProbeItem(BaseModel):
    id: str
    synergy: float = Field(ge=0, le=100)
    killing: float = Field(ge=0, le=100)


class ProbeAnswer(BaseModel):
    items: list[ProbeItem]


# -- sample ---------------------------------------------------------------------------------------

def sample(pool: pd.DataFrame, fame: pd.DataFrame, n: int = N_TRIPLETS, seed: int = SEED) -> pd.DataFrame:
    """`n` selectable experiments, as equal as possible across fame groups, in a shuffled order.

    Held-out lines are excluded (they are reserved for section 9). Deterministic given `seed`.
    """
    sel = pool.loc[pool.split == "selectable", KEY].merge(fame[["drug_a", "drug_b", "fame"]], on=["drug_a", "drug_b"],
                                                          how="left", validate="many_to_one")
    if sel.fame.isna().any():
        raise ValueError("some pool pairs have no fame class")
    rng = np.random.default_rng(seed)
    per = {g: n // len(FAME_ORDER) + (1 if i < n % len(FAME_ORDER) else 0) for i, g in enumerate(FAME_ORDER)}
    parts = []
    for g in FAME_ORDER:
        grp = sel[sel.fame == g].sort_values(KEY).reset_index(drop=True)
        parts.append(grp.iloc[np.sort(rng.choice(len(grp), per[g], replace=False))])
    out = pd.concat(parts, ignore_index=True)
    out = out.iloc[rng.permutation(len(out))].reset_index(drop=True)
    out.insert(0, "probe_id", [f"E{i:03d}" for i in range(len(out))])
    return out


# -- prompt ---------------------------------------------------------------------------------------

def template() -> str:
    return (PROMPT_DIR / f"{PROMPT_VERSION}.md").read_text()


def _mechanisms(store: FeatureStore, drug_id: str) -> str:
    t = store.drug_targets
    moa = sorted(set(t.loc[t.drug_id == drug_id, "mechanism_of_action"].dropna()))
    return "; ".join(moa) if moa else "mechanism not recorded"


def drug_codes(drug_ids: list[str]) -> dict[str, str]:
    return {d: f"Drug-{i + 1:03d}" for i, d in enumerate(sorted(set(drug_ids)))}


def describe(row: Any, store: FeatureStore, codes: dict[str, str] | None) -> str:
    cells = store.cells.set_index("cellosaurus_ac")
    line = cells.loc[row.cellosaurus_ac]

    def drug(d: str) -> str:
        name = codes[d] if codes else store.drug_name(d).removesuffix(" ANHYDROUS").lower()
        return f"{name} ({_mechanisms(store, d)})"

    return (f"- id {row.probe_id}: {drug(row.drug_a)} + {drug(row.drug_b)} in {line.cellosaurus_name} "
            f"({line.lineage}; {line.primary_disease})")


def prompts(items: pd.DataFrame, store: FeatureStore, masked: bool = False, batch: int = BATCH) -> list[str]:
    codes = drug_codes([*items.drug_a, *items.drug_b]) if masked else None
    tpl = template()
    return [tpl.format(experiments="\n".join(describe(r, store, codes) for r in items.iloc[i:i + batch].itertuples()))
            for i in range(0, len(items), batch)]


# -- run ------------------------------------------------------------------------------------------

def _cost(model: str, usage: Any) -> float | None:
    if model not in PRICES:
        return None
    inp, out = PRICES[model]
    return (usage.input_tokens * inp + usage.cache_read_tokens * inp * 0.1 + usage.output_tokens * out) / 1e6


async def _ask(llm: Any, prompt: str, model: str) -> tuple[ProbeAnswer | None, dict[str, Any]]:
    res = await llm.complete(prompt, model=model, schema=ProbeAnswer, tag="memory_probe", cache_prompt=False)
    meta = {"cached": res.cached, "input_tokens": res.usage.input_tokens, "output_tokens": res.usage.output_tokens,
            "cache_read_tokens": res.usage.cache_read_tokens, "cost_usd": _cost(model, res.usage),
            "stop_reason": res.stop_reason, "refusal": res.refusal}
    return res.parsed, meta


async def run_probe(items: pd.DataFrame, texts: list[str], llm: Any, model: str,
                    batch: int = BATCH) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    results = await asyncio.gather(*(_ask(llm, t, model) for t in texts))
    preds, calls = [], []
    for b, (text, (ans, meta)) in enumerate(zip(texts, results)):
        want = list(items.probe_id.iloc[b * batch:(b + 1) * batch])
        got = {it.id: it for it in (ans.items if ans else [])}
        meta.update({"batch": b, "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
                     "n_asked": len(want), "n_answered": sum(i in got for i in want),
                     "extra_ids": sorted(set(got) - set(want)), "parsed": ans is not None})
        calls.append(meta)
        for i in want:
            it = got.get(i)
            preds.append({"probe_id": i, "pred_syn": it.synergy if it else np.nan,
                          "pred_inh": it.killing if it else np.nan})
    out = items.merge(pd.DataFrame(preds), on="probe_id", how="left", validate="one_to_one")
    return out, calls


def run_dir(model: str, masked: bool, root: Path | None = None) -> Path:
    return (root or paths.data_cache().parent / "runs" / "m5_probe") / (model + ("-masked" if masked else ""))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m combo_agent.literature.probe")
    ap.add_argument("--model", required=True)
    ap.add_argument("--masked", action="store_true", help="replace drug names with codes")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--n", type=int, default=N_TRIPLETS)
    args = ap.parse_args(argv)

    from sheldon.core.cache import DiskCache
    from sheldon.core.io import load_dotenv
    from sheldon.core.llm import LLM

    load_dotenv(Path.cwd() / ".env")
    processed = paths.processed_dir()
    pool = pd.read_parquet(processed / "pool.parquet", columns=[*KEY, "split"])
    fame = pd.read_parquet(processed / "m2" / "pair_fame.parquet")
    items = sample(pool, fame, n=args.n)
    store = FeatureStore()
    texts = prompts(items, store, masked=args.masked)
    out_dir = args.out or run_dir(args.model, args.masked)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "prompts.jsonl").write_text("".join(json.dumps({"batch": i, "prompt": t}) + "\n" for i, t in enumerate(texts)))

    llm = LLM(args.model, max_tokens=8000, cache=DiskCache(paths.data_cache() / "llm_cache" / "llm.db"))
    preds, calls = asyncio.run(run_probe(items, texts, llm, args.model))
    preds.to_parquet(out_dir / "predictions.parquet", index=False)
    (out_dir / "calls.jsonl").write_text("".join(json.dumps(c, sort_keys=True) + "\n" for c in calls))
    costs = [c["cost_usd"] for c in calls if c["cost_usd"] is not None]
    manifest = {
        "model": args.model, "masked": args.masked, "n": len(items), "batch": BATCH, "seed": SEED,
        "prompt_version": PROMPT_VERSION, "prompt_template_sha256": hashlib.sha256(template().encode()).hexdigest(),
        "answered": int(preds.pred_syn.notna().sum()),
        "tokens": {k: sum(c[k] for c in calls) for k in ("input_tokens", "output_tokens", "cache_read_tokens")},
        "cost_usd_this_run": round(sum(c["cost_usd"] for c in calls if c["cost_usd"] is not None and not c["cached"]), 6)
        if costs else None,
        "cost_usd_uncached": round(sum(costs), 6) if costs else None,
        "llm_cache": str(llm.cache.path),
        "built": dt.datetime.now().isoformat(timespec="seconds"),
        "versions": {"python": platform.python_version(), "pandas": pd.__version__},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"{args.model}{' (masked)' if args.masked else ''}: {manifest['answered']}/{len(items)} answered, "
          f"cost ${manifest['cost_usd_uncached']} -> {out_dir}")
    return 0 if manifest["answered"] == len(items) else 1


if __name__ == "__main__":
    sys.exit(main())
