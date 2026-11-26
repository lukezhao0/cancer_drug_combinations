# Handoff: M2–M5 built; start of M6 (the agent)

| Field | Answer |
|:--|:--|
| **Date** | 2026-11-26 |
| **Milestone / task** | M2 oracle and splits, M3 response model, M4 baselines and runner, M5 retrieval and leakage filter ([`target_design.md`](target_design.md) §5–7, §10–12) |
| **Branch / PR** | `m2-m5` → `main` |
| **Status** | Built, tested and run. Pushed **as built**: the issues in [`project-notes/known-issues.md`](project-notes/known-issues.md) are documented, not fixed. **Surrogate: the Gaussian process is the default** (since 2026-11-26); the ensemble is the legacy build, tag `ensemble-v1` |
| **Read first** | [`project-notes/known-issues.md`](project-notes/known-issues.md) → **"Agent-builder rules"** at its top |

## 1. Goal of this piece of work

Turn the frozen M1 table into a working benchmark:
- **M2:** a simulated lab that hides results until they're "run";
- **M3:** a shared response model with uncertainty;
- **M4:** numeric baseline selectors and a campaign runner with metrics;
- **M5:** a literature search guarded by a leakage filter, and a probe of what the LLM already knows.

The agent (M6) plugs into exactly these pieces and must beat the M4 baselines.

## 2. What was done

| Milestone | Where | Verified how | Details |
|:--|:--|:--|:--|
| M2 Oracle, seed sets, pair fame | `src/combo_agent/lab/` | refusal and label-access tests; offline rebuild byte-identical | [`m2-oracle.md`](project-notes/m2-oracle.md) |
| M3 Surrogate: **Gaussian process (default)**; legacy 5-network ensemble | `src/combo_agent/models/gp.py`; legacy in `models/legacy/` | GP: posterior matches GPyTorch, PSD and symmetric kernel, ±1σ covers 68–70% in campaigns. Ensemble: 80% intervals 84–90% on the sanity split | [`gp-surrogate.md`](project-notes/gp-surrogate.md); legacy: [`m3-response-model.md`](project-notes/m3-response-model.md) |
| M4 Selectors, runner, metrics | `src/combo_agent/selectors/`, `src/combo_agent/eval/` | 100-campaign grid; every report number recomputes from the reveals | [`m4-baselines.md`](project-notes/m4-baselines.md) |
| M5 PubMed client, leakage filter, memory probe | `src/combo_agent/literature/`, `eval/memory_probe.py` | gate 23/23 leaky blocked, 19/19 clean passed (offline, reproduced) | [`m5-literature.md`](project-notes/m5-literature.md) |

## 3. Where the agent plugs in

| Piece | Code | What to know |
|:--|:--|:--|
| Simulated lab | `lab/oracle.py`: `Oracle.from_cache(seed=s)`, `candidate_frame()`, `history()`, `run(batch)`, `budget_left()`, `is_hit()` | `run` refuses a whole batch if anything in it is invalid. The seed set (32) is round 0 and free. **`candidate_frame()` includes `n_reps`: don't show it to the LLM** (rule 1, issue A1) |
| Selector interface | `selectors/base.py`: `Selector.select(ctx, batch_size, target, rng)`; `SelectionContext(candidates, history, model, pred)` | The agent is one more `Selector`. It sees the context, never the oracle. Register it in `selectors/baselines.py:SELECTORS` so the runner can build it |
| Campaign runner | `eval/runner.py`: `run_campaign(spec, …)`, `run_grid(…)`, CLI `python -m combo_agent.eval.runner` | Refits the shared surrogate each round (`seed×1000 + round`) and writes rounds and reveals files. Resume skips finished runs **without checking settings** (D3): use a fresh `--out` per configuration |
| Metrics | `eval/metrics.py`: `python -m combo_agent.eval.metrics <out>` | Hits@80/160/320, AUC, experiments-to-10, enrichment, paired Wilcoxon + Holm. Some §11 splits are incomplete (D5, D6) |
| Surrogate | `models/surrogate.py`: `make_surrogate("gp" or "ensemble", …)`; `models/prediction.py`: `Prediction.ucb(target, beta)` | **The GP is the default and the shared model for every selector, the agent included.** Open GP issues: C4 (cell-lengthscale prior mis-scaled) and C8. The legacy ensemble runs only with `--surrogate ensemble` |
| Agent tools | `tools/registry.py`: `get_drug`, `get_cell`, `find_lines`, `gene_expression`, `drug_targets_in_line`, `shared_targets`, `search_literature` (implemented); `query_model`, `resolve`, `get_history` (declared, not implemented) | **`run_python` (`tools/probes.py`) can reach the labels: not in scored runs** (A2). Implementing `query_model`, `resolve` and `get_history` is M6 work |
| Literature | `literature/retrieval.py`: `LiteratureSearch.default(audit_path)`, `search(query, k)` | Every call is filtered and audited. Turn it off in masked-name runs (E7) |
| Rule grammar (M7) | `tools/rules.py` | At most 2 terms joined by `and`, over a fixed vocabulary. Expression tertiles have a tie problem (C11) |
| LLM | SHELDON's client (`sheldon` pinned in `pyproject.toml`) | Send temperature 0 explicitly (E4) |

## 4. Results and key numbers

| Quantity | Value | Source |
|:--|--:|:--|
| Selectable experiments / candidates at round 1 (minus the 32 seed) | 219,239 / 219,207 | `Oracle.from_cache(seed=0)` |
| Hits @320 with the **GP**, seeds 0–4 (synergy / killing): `random` | 16.2 / 15.2 | `runs/gp_compare/` (not in git); [`gp-surrogate.md`](project-notes/gp-surrogate.md) §9.1 |
| … `bo_ucb`, **the baseline the agent must beat** | 94.0 / 202.2 (provisional, C4) | same |
| … `greedy` | 91.8 / 186.0 (provisional) | same |
| Legacy ensemble, seeds 0–9: `bo_ucb` / `random` | 53.0 / 61.6 vs 16.2 / 17.0 | `runs/m4_main/report.md` (not in git) |
| Memory probe ρ, synergy / killing (`gemini-3.5-flash-lite`) | −0.05 / 0.40 | `artifacts/m5/memory_probe.md` |

## 5. Caveats and known problems

**All of them are in [`project-notes/known-issues.md`](project-notes/known-issues.md)**: about 60
items, each with a severity, a location and a "fix by" stage. The ones that affect M6 work right away:
- **A1, A2 (must fix before a scored agent run):** `n_reps` shows in the candidate table, and
  `run_python` can reach `labels.parquet`.
- **C4, D1, G1–G4 (fix before comparing the agent with the baselines):**
  - the GP's cell-lengthscale prior is mis-scaled, so for synergy it treats lines as unrelated;
  - `bo_qei` aims at the best value rather than at hits;
  - the GP has no formal M3 gate run;
  - GP picks concentrate in a few lines;
  - the GP reference grid has seeds 0–4 only.
- The legacy ensemble's own problems (C1, C2) no longer affect current runs.
- **E1–E4:** the leakage filter's pool-pair-plus-pool-line rule misses common spellings, the gate set
  never exercises that rule, there are no M5 unit tests, and the probe sets no temperature.

The M1 caveats still apply:
- SR's expression is imputed.
- SNB-19, NCI-H322M and HL-60(TB) mutations come from their parent lines.
- Synergy results are reported both with and without the low-kill hits (decision 20).

## 6. How to reproduce

**Data needed that isn't in git** (team Google Drive folder, see README):

| Archive | sha256 | Contents |
|:--|:--|:--|
| `m1-data-v1.tgz` | `bd33234e3cf63af3221de6a828b246b0f3a599a69a8088d8af8e65878f4da65b` | M1 pool, labels, features (required) |
| `m2m5-addon-v1.tgz` | `610d3862af31fa1bb7b7d5c3870ad6fe6b1cabe038dc4cdda88d11c858614e15` | M2 seed sets and pair fame, M5 blocklist (required for the oracle and the filter) |
| `m2m5-addon-v1_api_cache.tgz` | `507909a9b5f826cad7dcb0dafad981d2a1835c31b3782cc70890747ab64e0a66` | PubMed responses (optional: offline, byte-identical rebuilds of M2/M5) |

**Install the data:**

```bash
# after installing m1-data-v1 as in README.md
shasum -a 256 ~/Downloads/m2m5-addon-v1.tgz
tar -xzf ~/Downloads/m2m5-addon-v1.tgz -C ~/Downloads
(cd ~/Downloads/m2m5-addon-v1 && shasum -a 256 -c SHA256SUMS)
cp -R ~/Downloads/m2m5-addon-v1/data_cache/. data_cache/
```

**Check it works:**

```bash
uv run python -c "from combo_agent.lab.oracle import Oracle; o = Oracle.from_cache(seed=0); print(len(o.candidates()), len(o.history()))"
# expect: 219207 32   (219,239 selectable experiments minus the 32 seed-set ones)
```

**Run things:**
- the M4 grid: [`m4-baselines.md`](project-notes/m4-baselines.md) §5;
- the M5 blocklist, test set and probe: [`m5-literature.md`](project-notes/m5-literature.md) §6.

## 7. Tests

- **Unit tests** (no data needed): `uv run pytest -q` → 204 passed, 30 skipped.
- **Real data, M2–M5** (both bundles installed):

  ```bash
  COMBO_INTEGRATION=1 COMBO_OFFLINE=1 uv run pytest -q tests/test_oracle.py tests/test_splits.py \
    tests/test_label_access.py tests/test_legacy_ensemble.py tests/test_gp.py tests/test_m4.py tests/test_tools.py
  ```

  Expect 109 passed. Checked on a fresh copy with only the two bundles, 2026-11-26.
- **The M1 rebuild tests** (`test_ingest_tdc`, `test_harmonize_*`, `test_build_experiments`,
  `test_features`, `test_freeze_pool`) **fail with the bundles alone**. That's expected: they rebuild M1
  from raw downloads (the TDC file, the DepMap CSVs, `m1_api_cache.tgz`), which no bundle contains. Run
  them only after a full M1 rebuild (README, "Rebuilding the M1 data pool").
- **Offline rebuild check** (with `m2m5-addon-v1_api_cache.tgz` installed):
  `python -m combo_agent.lab.splits --offline --out /tmp/m2` and
  `COMBO_OFFLINE=1 python -m combo_agent.literature.leakage build --out /tmp/m5` reproduce the bundled
  M2 and M5 files byte for byte.

## 8. Open questions (need a team decision)

| Question | Options | Recommendation |
|:--|:--|:--|
| May the agent see which lines are held out (A5)? | hide them / allow and document it | Hide them: design §9 freezes predictions before held-out lines are looked at |
| ~~Which surrogate do the agent and the baselines share?~~ | **Decided 2026-11-26: the GP** | Fix C4, then rerun the GP grid on seeds 0–9 |
| How is the labels file kept out of the agent's reach (A2)? | separate oracle process / Linux namespaces / labels outside the readable tree | A separate oracle process: it also works on macOS |

## 9. Next steps

1. **M6 skeleton:** an `AgentSelector` implementing `Selector`; the shortlist; tools
   `query_model` / `resolve` / `get_history`; hypothesis and observation stores. This can start now,
   following the agent-builder rules.
2. **In parallel, before scored runs:** fix A1–A4, E1–E4 and E7.
3. **Before comparing with the baselines:** fix C4 and D1–D4, then rerun the M4 grid **with the GP** on seeds 0–9 into a new folder (`--grid main --out runs/m4_gp_main`).

## 10. Files added or changed

| Path | Change |
|:--|:--|
| `src/combo_agent/{lab,models,selectors,eval,literature}/` | new (M2–M5) |
| `src/combo_agent/models/{prediction.py,legacy/}` | GP made the default; the ensemble moved to `legacy/` (shared `Prediction` and `cosine_similarity` now in `prediction.py`) |
| `src/combo_agent/tools/{registry,store}.py` | `search_literature` registered; gene tables read once |
| `tests/` | `test_oracle`, `test_splits`, `test_label_access`, `test_legacy_ensemble` (was `test_response_model`), `test_gp`, `test_m4`; `fixtures/m5/` |
| `artifacts/m2`, `artifacts/m3`, `artifacts/m5` | small tracked outputs |
| `scripts/` | M4, GP-comparison figures; M5 test-set builder; Gemini smoke test |
| `docs/` | this handoff, `project-notes/known-issues.md`, M2–M5 notes, status banners |
