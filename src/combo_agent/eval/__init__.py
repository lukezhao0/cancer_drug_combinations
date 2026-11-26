"""Campaign runner and metrics (M4, target_design.md §11).

    runner.py   runs selector x target x seed campaigns against the oracle; resumable grids
    metrics.py  hits at budget, discovery-curve AUC, diversity, fame splits, paired statistics

Neither module reads the hidden outcomes: hits come from the oracle's revealed values and the
public thresholds (see `tests/test_label_access.py`).
"""
