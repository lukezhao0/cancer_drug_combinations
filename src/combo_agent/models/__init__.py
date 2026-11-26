"""The surrogate (M3): predicts synergy and killing, with uncertainty, for untested experiments.

    gp.py          GPModel: exact Gaussian process with a Tanimoto drug-pair kernel (the DEFAULT)
    surrogate.py   the Surrogate interface every selector uses; make_surrogate("gp" | "ensemble")
    prediction.py  Prediction (what every surrogate returns) and cosine_similarity
    features.py    shared feature helpers (no outcomes)
    legacy/        the first surrogate, a 5-network ensemble: kept to reproduce earlier runs
"""
