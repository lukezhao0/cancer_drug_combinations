"""LEGACY BUILD: the first surrogate, a deep ensemble of 5 small neural networks (M3).

Superseded on 2026-11-26 by the Gaussian process (`models/gp.py`), which is now the default
surrogate for every campaign. Reasons, from the 100-campaign comparison on seeds 0-4
(docs/project-notes/gp-surrogate.md):
* killing: the GP finds 1.8-3.3x as many hits with every BO rule, on 5/5 seeds;
* calibration: the GP's 68% intervals cover 68-70%; the ensemble's cover 84-87% (too wide);
* the ensemble has known, unfixed problems (docs/project-notes/known-issues.md C1-C3, C5-C7).

Kept so that earlier results stay reproducible:
* `runs/m4_main` (the M4 grid) and the ensemble arm of the GP comparison were made with it;
* run it with `python -m combo_agent.eval.runner ... --surrogate ensemble [--mode full|compact]`;
* the git tag `ensemble-v1` is the last state in which it was the default.

    ensemble.py           EnsembleModel (numpy MLP ensemble; hand-written backprop and Adam)
    ensemble_features.py  FeatureSet: the ensemble's symmetric pair and cell feature blocks
    calibrate.py          the M3 calibration report (evaluation code: reads the hidden outcomes)
"""
