"""
combo-mechanism: reason about which cell lines a drug pair should work in, with a mechanism.

For now this is a smoke harness for the in-silico tools (`combo_agent.tools`): a few hand-written
mechanism questions, the tool probes, and the Python sandbox preloaded with feature tables. There
are no gold answers and the score is not a quality measure. Real tasks and scoring come with the
oracle (M2) and the agent loop (M6).
"""

from __future__ import annotations

from functools import cached_property
from typing import Any

from sheldon.benchmarks.base import Benchmark
from sheldon.core.probes import Probe
from sheldon.core.registry import benchmarks
from sheldon.core.types import Prediction, RunContext, Score, Task

from combo_agent.tools import probes as tool_probes
from combo_agent.tools.store import FeatureStore

#: (id, drug pair, question). Classic mechanism pairings, chosen to exercise the tools.
SMOKE_TASKS = (
    ("mek-pi3k", "a MEK inhibitor with a PI3K inhibitor"),
    ("parp-dna-damage", "a PARP inhibitor with a DNA-damaging agent"),
    ("egfr-mek", "an EGFR inhibitor with a MEK inhibitor"),
)

QUESTION = (
    "We are choosing drug-combination experiments in a panel of cancer cell lines. Consider {pair}.\n"
    "1. Pick one concrete pair of drugs of these classes that exists in the panel (use the tools to "
    "check the drugs and their targets).\n"
    "2. Propose a mechanism for synergy and state it as a cell-line rule the tools accept "
    "(for example `KRAS_mut == 1 and PTEN_loss == 0`).\n"
    "3. Check the rule against the panel: which lines satisfy it, is it testable (at least 3 lines on "
    "each side), and do the drugs' targets look relevant in those lines?\n"
    "No experiment results are available; reason from mechanism and the cell-line data only.\n\n"
    + tool_probes.PYTHON_CONTEXT_DOC
)


@benchmarks.register("combo-mechanism")
class ComboMechanism(Benchmark):
    name = "combo-mechanism"
    splits = ("eval",)
    blurb = "in-silico tool smoke tasks: mechanism rules for a drug pair, checked against cell-line data"
    answer_format = (
        "Answer with four lines: `PAIR: <drug A> + <drug B>`, `RULE: <rule>`, "
        "`MECHANISM: <one or two sentences>`, `EVIDENCE: <what the tools showed>`."
    )
    probes_essential = True

    @cached_property
    def store(self) -> FeatureStore:
        return FeatureStore(self.opts.get("data_cache"))

    @cached_property
    def _context(self) -> dict[str, Any]:
        return tool_probes.python_context(self.store)

    def load(self, split: str = "eval", limit: int | None = None, seed: int = 0) -> list[Task]:
        tasks = [Task(id=tid, benchmark=self.name, question=QUESTION.format(pair=pair), payload={"pair": pair})
                 for tid, pair in SMOKE_TASKS]
        return self.subsample(tasks, limit, seed)

    def python_context(self, task: Task) -> dict[str, Any]:
        return self._context

    def probes(self, task: Task) -> list[Probe]:
        out: list[Probe] = []
        if self.opts.get("sandbox", True):
            out.append(tool_probes.GuardedPythonProbe(
                timeout=float(self.opts.get("exec_timeout", 20.0)), context=self.python_context(task)))
        return out + tool_probes.as_probes({"features": self.store})

    def check(self) -> list[str]:
        return self.store.available()

    async def score(self, task: Task, pred: Prediction, ctx: RunContext) -> Score:
        """Not a quality score: 1.0 if the answer has the requested lines and a rule that parses."""
        from combo_agent.tools import ToolError, rules

        text = str(pred.answer or "")
        fields = {k: any(line.strip().upper().startswith(k + ":") for line in text.splitlines())
                  for k in ("PAIR", "RULE", "MECHANISM", "EVIDENCE")}
        rule_ok = False
        for line in text.splitlines():
            if line.strip().upper().startswith("RULE:"):
                try:
                    rules.parse(line.split(":", 1)[1].strip().strip("`"))
                    rule_ok = True
                except ToolError:
                    pass
        calls: dict[str, int] = {}
        for probe, _ in getattr(ctx.tools, "history", []) or []:
            calls[probe.name] = calls.get(probe.name, 0) + 1
        well_formed = all(fields.values()) and rule_ok
        return Score(task.id, float(well_formed),
                     metrics={"well_formed": float(well_formed), "rule_parses": float(rule_ok),
                              "tool_calls": float(sum(calls.values()))},
                     detail={"tool_calls": calls})
