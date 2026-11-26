# M3: the response model (the ensemble, now the legacy build)

> **Superseded (2026-11-26).** The main surrogate is now the **Gaussian process**
> ([`gp-surrogate.md`](gp-surrogate.md)). The ensemble described here is the **legacy build**:
> - code in `src/combo_agent/models/legacy/` (`ensemble.py`, `ensemble_features.py`, `calibrate.py`);
> - run only with `--surrogate ensemble`;
> - git tag `ensemble-v1` marks its last state as the default.
>
> It is kept because `runs/m4_main` and the ensemble arm of the GP comparison were made with it. The
> paths below were updated to the legacy locations.

> **Known issues (2026-11-26), legacy only, not fixed; see [`known-issues.md`](known-issues.md):**
> - **C1:** z-scoring the sparse fingerprint bits gives some inputs values near 70. Those candidates
>   look novel and get large std; this is the cause of the §5 "line sweep".
> - **C2:** at campaign sizes, members early-stop on about 6 validation rows, and many stop after 1–2
>   epochs.
> - **C3:** the coverage gate is close to circular, because the noise is fitted to the same 80% quantile.
> - **C10:** the "11.5 s" below is fit time only; scoring the pool adds about 1.7 s.

> Status 2026-11-23: **M3 complete.** Gate passed: 80% intervals cover 85–89% on the 80/20 sanity
> split (gate 70–90%); predictions don't change when drug A and B are swapped (difference exactly
> 0); a refit plus scoring the whole pool takes 3.7 s (compact) or 11.5 s (full), against a 60 s
> limit. Code: `src/combo_agent/models/legacy/` (originally `models/`). Report: `artifacts/m3/m3_report.md`.

## 1. What the model is for

A campaign has a budget of 320 experiments out of ~219,000. Before every batch, a selector asks:
*"for each untested experiment, what result do I expect, and how sure am I?"* The response model
answers both, for synergy (`y_syn`, Bliss) and killing (`y_inh`, CSS).

```
  revealed history (32 seed + up to 320)        every untested experiment (~219k)
          │                                                  │
          ▼                                                  ▼
   ┌─────────────── refit from scratch each round ───────────────┐
   │  5 small neural networks ("members"), each trained on a      │
   │  different 80% of the history, stopped early on the other 20%│
   └──────────────────────────────┬───────────────────────────────┘
                                  ▼
            for each experiment:  mean  = average of the 5 predictions
                                  std   = how much the 5 disagree  (thin data → big)
                                        + how big their held-out errors were (noise)
                                  ▼
       selectors: greedy (top mean) · uncertainty (top std) · UCB (mean + 2·std) · agent
```

**Why five networks instead of one?** If five models trained on slightly different data agree,
the data pin the answer down. If they disagree, we're extrapolating. That disagreement is the
"model uncertainty" that exploration strategies (uncertainty sampling, UCB) feed on.

**Same model for every selector** (design §6). Any difference between selectors comes from *how
they choose*, not from a better predictor.

## 2. Inputs

One experiment is (drug A, drug B, cell line). Its input is

```
[ f(A) + f(B) ,  f(A) × f(B) ,  g(cell line) ]
  └──────── pair block ───────┘   cell block
```

Sum and product don't care about order, so **A+B and B+A get the same prediction** (tested: the
difference is exactly 0).

| Mode | Drug features f | Cell features g | Inputs |
|:--|:--|:--|--:|
| `full` (default, the design) | 2,048 Morgan fingerprint bits + target-gene, action and target-type one-hots | 1,000 most-variable genes + 24 driver mutations + 31 copy-number genes + lineage | 2,253 + 1,066 |
| `compact` | 32 fingerprint principal components + the same one-hots | 16 expression principal components + mutations + copy number + lineage | 330 + 82 |

All scaling and the principal components are computed from feature tables only, never from
outcomes.

**Speed trick (exact, not an approximation):** the first layer is linear, so it's applied once per
*drug pair* (4,939) and once per *cell line* (57), then added per experiment. Scoring the whole
pool never builds the ~5 GB input matrix. A test checks this gives the same result as the
straightforward computation.

## 3. How uncertainty is computed, and why it isn't the design's version

The design gave each network a second output: its own estimate of the noise for that experiment
(the "Gaussian likelihood" method). **At campaign sizes that failed.** The network memorized its
few training rows, its noise estimate shrank toward zero, and it became confident and wrong. On a
planted test problem, the held-out negative log-likelihood (lower is better) went from −0.1 to 84
while held-out squared error kept falling.

**What we do instead:** each network's noise is measured on the 20% of history it **didn't** train
on. Those are out-of-sample errors, which can't be memorized. Bliss and CSS have heavy tails (most
values are small, a few are huge), so we use the **80th percentile** of the held-out errors rather
than their average. The average let a few huge values make every interval too wide (coverage 0.91
vs 0.87). Both settings are in the code (`noise="quantile"` is the default; `"mse"` is the
alternative); the means are identical, only the widths differ.

## 4. Results (`artifacts/m3/m3_report.md`)

Key terms:
- **r**: how well predictions track the truth (0 = not at all, 1 = perfectly).
- **Coverage**: share of real results inside the 80% interval (should be near 80%).
- **Enrichment**: hit rate among the model's top 1% picks ÷ the 5% base rate (1× = no better
  than random).

**In the regime campaigns run in** (training on n random experiments; mean of 5 repeats; `full`
mode):

| Training rows | Synergy r | Synergy enrichment | Killing r | Killing enrichment | Coverage (syn / inh) |
|--:|--:|--:|--:|--:|--:|
| 32 | 0.04 | 1.9× | 0.20 | 4.7× | 0.89 / 0.84 |
| 96 | 0.04 | 1.8× | 0.34 | 5.4× | 0.84 / 0.89 |
| 320 | 0.13 | 2.7× | 0.51 | 7.3× | 0.88 / 0.90 |

**With lots of data** (80/20 split, 175k training rows): synergy r = 0.71, killing r = 0.90,
coverage 0.88 / 0.89. Compact mode is a little weaker everywhere (e.g. killing r 0.43 at n=320,
synergy r 0.62 on 80/20), so `full` is the default. Compact remains available for fast iteration.

**What this means:**
- **Killing is learnable from little data.** With 320 experiments, the top-1% predictions are
  hits 7 times as often as random picks. This is mostly "which drugs and lines are generally
  potent."
- **Synergy is hard at campaign sizes.** r is only 0.04–0.13, and its point predictions are
  slightly *worse* than predicting the average (RMSE ratio 1.09–1.14). The *ranking* still
  carries signal: the top 1% are enriched 1.8–2.7× for synergy hits, and selectors use the
  ranking. It isn't a model limitation: with 175k rows the same model reaches r = 0.71. It's the
  data: Bliss replicates themselves agree at only r = 0.21 (M1), and 320 noisy points carry
  little information. **This is exactly where literature and mechanism priors (the agent) have
  room to help**, and where the benchmark can show it.
- **The intervals are honest:** coverage is 0.84–0.90 at every training size (one value, killing
  at n=320 in `full` mode, sits at 0.902, on the gate's edge).

## 5. A warning for M4 (baselines)

After the 32-experiment seed set, the median predicted killing std (25) is sensible (the training
SD is 21.9). But the **0.1% most extreme extrapolations** have std ≥ 92, and UCB with β = 2 picks
exactly those: in a demo, std made up 76% of the top UCB scores, and the 16 picks covered only
**3 drug pairs**. That batch still found 3 killing hits (0.8 expected at random), but it's a
line-sweep. So the design's **diversity penalty for `bo_ucb` is essential**, and β needs a
sensitivity check. `EnsembleModel.embed()` provides the embedding for the penalty.

## 6. Using it

```python
from combo_agent.lab.oracle import Oracle
from combo_agent.models.legacy.ensemble import EnsembleModel

oracle = Oracle.from_cache(seed=0)
model = EnsembleModel(seed=0)                 # mode="full"; FeatureSet built once and reused
model.fit(oracle.history())                   # revealed Results only
cand = oracle.candidate_frame()
pred = model.predict(cand)                    # pred.mean_syn, pred.std_syn, pred.mean_inh, pred.std_inh
scores = pred.ucb("syn", beta=2.0)
```

`fit` accepts only the oracle's `Result` objects, so a selector can't hand it labels from
anywhere else. The calibration script (`python -m combo_agent.models.legacy.calibrate`; the 80/20 fits alone take ~3–4 min per mode) is
evaluation code and is on the label allowlist; `legacy/ensemble_features.py` and `legacy/ensemble.py` are not.

## 7. Not done / next

- **M4:** the selectors (`random`, `greedy`, `uncertainty`, `bo_ucb` with the diversity penalty,
  `bo_qei` as Monte-Carlo qEI over the members), the campaign runner, metrics and curves.
  Estimated grid cost in `full` mode: about 2,000 refits × ~14 s ≈ 8 h, overnight; `compact` is
  about 3× faster.
- A GP (design v2) wasn't needed: the ensemble passed calibration at all sizes.
- Heteroscedastic noise (per-experiment noise) can be revisited when there's more data per pair.
