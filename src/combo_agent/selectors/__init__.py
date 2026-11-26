"""Selectors (M4, target_design.md §7): rules for choosing the next batch of experiments.

    base.py       the Selector interface and the per-round context the runner hands to it
    baselines.py  random, greedy, uncertainty, bo_ucb (with a diversity penalty), bo_qei

Every selector chooses from the oracle's untested, selectable candidates using only what the
runner gives it: the candidates, the revealed history and the shared response model's predictions.
None of them can reach the hidden outcomes.
"""

from combo_agent.selectors.base import SelectionContext, Selector  # noqa: F401
from combo_agent.selectors.baselines import SELECTORS, make  # noqa: F401
