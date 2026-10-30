"""
combo-mechanism: predict which cell lines a drug pair is synergistic in, with a mechanism.
Stub: registers the benchmark so `combo-agent ls` lists it. 
"""

from __future__ import annotations

from sheldon.benchmarks.base import Benchmark
from sheldon.core.registry import benchmarks
from sheldon.core.types import Prediction, RunContext, Score, Task


@benchmarks.register("combo-mechanism")
class ComboMechanism(Benchmark):
    name = "combo-mechanism"
    splits = ("train", "eval")  # split by drug pair
    blurb = "predict which cell lines a drug pair is synergistic in, with a mechanism (stub)"

    def load(self, split: str = "eval", limit: int | None = None, seed: int = 0) -> list[Task]:
        # TODO: one Task per DrugComb ONEIL pair; answer = hidden per-line synergy
        return []

    async def score(self, task: Task, pred: Prediction, ctx: RunContext) -> Score:
        # TODO: AUROC of line_scores against per-line synergy
        raise NotImplementedError("combo-mechanism scoring is not implemented yet")
