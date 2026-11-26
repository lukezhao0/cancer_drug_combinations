"""The response model (M3): predicts synergy and killing, with uncertainty, for untested experiments.

    features.py        drug, pair and cell feature blocks from the frozen feature tables (no outcomes)
    response_model.py  EnsembleModel: a deep ensemble of small MLPs, refit on a campaign's history
    calibrate.py       the M3 calibration report (evaluation code: reads the hidden outcomes)
"""
