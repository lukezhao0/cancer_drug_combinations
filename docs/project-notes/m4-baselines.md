# M4: baselines, campaign runner and metrics

> **Numbers provisional (2026-11-26).** The baselines share a response model with known problems (C1,
> C2), and `bo_qei` improves on the best value revealed so far rather than on the hit threshold (D1).
> Resume doesn't check settings (D3), and synergy metrics are mostly reported without the low-kill
> exclusion (D5). Fix these and rerun M4 before comparing the agent with the baselines. See
> [`known-issues.md`](known-issues.md).

> Update 2026-11-24: **the `pilot` and `main` grids have run; the gate passed.** `bo_ucb` beat
> `random` on hits at 320 (killing 61.6 vs 17.0, synergy 53.0 vs 16.2; Holm p = 0.0039). `greedy`
> was strongest (94.8 / 73.0). Full tables: `runs/m4_main/report.md`; summary in `../CHANGELOG.md`.
>
> Status 2026-11-23: **built and unit-tested on synthetic data; not yet run on the real pool.**
> The gate ("`bo_ucb` beats `random` on hits at 320") is decided by the `main` grid below. Code:
> `src/combo_agent/selectors/`, `src/combo_agent/eval/`, `scripts/make_m4_figures.py`. Tests:
> `tests/test_m4.py`.

## 1. What M4 answers

M4 runs the simulated lab end to end with **non-LLM selectors**. Each campaign starts with the
same 32 free seed experiments, then picks 16 experiments per round for 20 rounds (320 paid
experiments). We count how many hits it finds.

```
seed set (free, round 0) ──► round 1 ──► round 2 ──► … ──► round 20
                              │ refit model on history   (M3, skipped by random)
                              │ predict all ~219k untested experiments
                              │ selector picks 16
                              └ oracle.run(batch)  →  results join the history
output: a discovery curve per campaign (hits vs experiments run)
```

These curves are the bar the agent (M6) has to clear. If **`bo_ucb` doesn't beat `random`**, the
model isn't providing useful guidance, and that has to be fixed before agent work (design §12).

## 2. The selectors (`selectors/baselines.py`)

| Selector | How it picks 16 | Intuition |
|:--|:--|:--|
| `random` | uniformly at random | the floor: ~5% hit rate, ≈ 16 hits expected in 320 |
| `greedy` | highest predicted mean | pure exploitation: "go where the model thinks hits are" |
| `uncertainty` | highest predicted std | pure exploration (classic active learning) |
| `bo_ucb` | highest mean + 2·std, **made diverse** | optimism under uncertainty; the main numeric baseline |
| `bo_qei` | batch expected improvement, by sampling | which 16 together most raise the best value found |

**Why `bo_ucb` is made diverse.** M3 showed that just after the seed set, the top UCB picks are
extreme extrapolations that pile onto ~3 drug pairs, a "line sweep". `bo_ucb` takes the top 2,000
by UCB and picks with *maximal marginal relevance*. Each next pick maximizes
`0.5 × (UCB rescaled to 0–1) − 0.5 × (similarity to the picks so far)`, where similarity is the
cosine similarity of the model's internal representation of the experiments. A near-copy of an
earlier pick must be much better to be chosen.

A first version standardized UCB to z-scores (λ = 1). A synthetic test showed that version failed:
all six picks stayed in one cluster, because a cluster far ahead in z-score never pays enough
penalty. Rescaling to 0–1 bounds the score term.

**`bo_qei` without BoTorch.** It draws 128 joint "possible worlds" for the shortlist: each draw
takes one ensemble member's predictions plus noise at that member's error level. Then it
greedily adds the experiment that most raises the average, over worlds, of the best value found.
Because all candidates in a world come from the same member, the batch hedges across members.

**Ablation variants** (`--grid ablation`): `bo_ucb_b1` and `bo_ucb_b3` (β = 1 and 3), and
`bo_ucb_nodiv` (no diversity), which measures what the penalty buys.

Every selector sees only the untested candidates, the revealed history and the shared model's
predictions. The label-access test keeps `selectors/` and `eval/` away from the hidden outcomes.

## 3. The runner (`eval/runner.py`)

- One campaign = one (selector, target, seed). `target` is the outcome the selector chases (`syn`
  or `inh`); hits of **both** outcomes are recorded, for the cross-outcome yield.
- Hits use the oracle's public thresholds; no hidden outcome is read. Synergy hits are also
  counted *strictly*: excluding hits with a negative killing score (`syn_hit_low_kill`, M1
  decision 20).
- The model is refit from scratch every round with seed `seed×1000 + round`, so runs are
  reproducible.
- Each campaign writes `<out>/runs/<selector>__<target>__s<seed>.parquet` (one row per round) and a
  `.reveals.parquet` (every revealed experiment). The rounds file is written last and atomically,
  so **rerunning the same command resumes**: finished campaigns are skipped.
- `<out>/manifest.json` records the git commit (and whether the tree had uncommitted changes),
  input checksums, settings and versions.

## 4. Metrics (`eval/metrics.py`)

| Metric | Meaning |
|:--|:--|
| Hits @80 / @160 / @320 | hits among the first 80/160/320 paid experiments (the seed set is reported separately) |
| AUC | area under the discovery curve ÷ the best possible (every experiment a hit) |
| Experiments to 10 hits | how long until useful; censored at 320 |
| Enrichment vs random | mean hits ÷ random's mean hits |
| Hit pairs @320 | distinct drug pairs among the hits: a selector that sweeps one pair across lines scores low |
| Cross-outcome hits | in synergy campaigns, killing hits found, and the reverse |
| Hits by fame | hits on famous / middle / obscure pairs (M2) |

**Statistics:** mean and bootstrap 95% CI over the 10 seeds; one-sided paired Wilcoxon (each
seed's campaigns share a seed set) for each selector vs `random` and vs `bo_ucb`; Holm correction
across the "vs random" tests of each target.

**Gate verdict (per target):** `bo_ucb` has a higher mean than `random` at 320, **and** the
Holm-adjusted p < 0.05. Expect a clearer win on killing than on synergy: M3 found synergy weakly
predictable at campaign sizes (r ≈ 0.13 at 320 rows; the top-1% enrichment of 2.7× is the
optimistic ceiling of what the model's ranking offers).

## 5. How to run it

All commands are from the repo root. They need the M1 bundle plus the M2 outputs
(`data_cache/processed/m2/`).

| Step | Command | Estimated time on the M1 MacBook Air |
|:--|:--|:--|
| 1. Pilot (compact features, seeds 0–1) | `uv run python -m combo_agent.eval.runner --grid pilot --out runs/m4_pilot` | ~15–20 min |
| 2. Main grid (full features, seeds 0–9, 100 campaigns) | `uv run python -m combo_agent.eval.runner --grid main --out runs/m4_main` | ~6–7 h (80 model campaigns × 20 refits × ~13 s); overnight. `--mode compact` ≈ 2 h |
| 3. Metrics and gate | `uv run python -m combo_agent.eval.metrics runs/m4_main` | seconds |
| 4. Figures | `uv run --isolated --no-project --python 3.12 --with matplotlib==3.9.2 --with pandas==2.3.3 --with pyarrow==25.0.1 --with numpy==1.26.4 python -I scripts/make_m4_figures.py runs/m4_main` | seconds |
| 5. Ablation (optional) | `uv run python -m combo_agent.eval.runner --grid ablation --out runs/m4_ablation` | ~5–6 h |

The time estimates come from the M3 timings (refit + scoring 11.5 s in full mode, 3.7 s in
compact) plus selection overhead; they are **unverified** until the pilot runs. `--workers 2`
runs two campaigns in parallel with one math thread each; whether that's faster on this machine
is also unverified. If interrupted, rerun the same command: finished campaigns are skipped.
Outputs go to `runs/`, which git ignores.

Read in this order: `runs/<grid>/report.md` (the gate table first), then
`runs/<grid>/figures/`.

## 6. Tests (synthetic world; no real data)

`tests/test_m4.py`:
- every selector returns 16 distinct, untested, selectable experiments, deterministically under a
  seed;
- greedy and uncertainty return exactly the argmax sets;
- the diversity penalty spreads picks across clusters; qEI picks the dominant candidate and hedges;
- a toy campaign spends its budget exactly, its curves are monotone, and `random` never fits a
  model;
- files are written atomically and resume skips them;
- metrics match hand-computed curves (hits@k, AUC, censoring), and Holm and Wilcoxon are checked;
- an end-to-end tiny grid produces the summary and the report.

`COMBO_INTEGRATION=1` adds a short real campaign for three selectors (not run yet).
