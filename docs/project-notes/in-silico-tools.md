# In-silico tools (first version)

Label-free tools the agent calls to check a mechanism hypothesis against cell-line and drug data
before spending an experiment. Code: `src/combo_agent/tools/`. Tests: `tests/test_tools.py`.

## What the agent gets

| Tool                   | Arguments                           | Answers                                                                               |
| :--------------------- | :---------------------------------- | :------------------------------------------------------------------------------------ |
| `get_drug`             | `drug`                              | targets, mechanism of action, clinical phase (ChEMBL)                                 |
| `get_cell`             | `line`                              | lineage, disease, driver mutations, copy-number gains and losses, split, data caveats |
| `find_lines`           | `rule`, optional `lineage`, `split` | which lines satisfy a rule, and whether it is testable (3 lines per side)             |
| `gene_expression`      | `gene`, optional `rule` or `lines`  | a gene's expression: by rule (medians, Mann-Whitney p), per line, or panel summary    |
| `drug_targets_in_line` | `drug`, `line`                      | expression, mutation and copy number of each target gene in that line                 |
| `shared_targets`       | `drug_a`, `drug_b`                  | shared and distinct targets and mechanisms                                            |
| `run_python`           | `code`                              | SHELDON's Python sandbox with `cells`, `line_ids`, `expr`, `drugs` preloaded          |

Drugs resolve by ChEMBL ID, ChEMBL name or source label; lines by accession, name (any punctuation)
or DepMap ID. A bad name returns an error with suggestions, not an exception.

## Rules

One vocabulary for tools now and for registered predictions later (design section 9):

    GENE_mut == 0|1      24 driver genes
    GENE_loss == 0|1     log2 copy-number ratio <= -1 (one copy or fewer)
    GENE_gain == 0|1     log2 copy-number ratio >= 1 (four copies or more)
    lineage == NAME
    GENE_expr == low|mid|high    tertile by rank across all 57 lines

At most 2 terms joined by `and`. The copy-number cut-offs are provisional: the design names the
terms but not the thresholds. A line with an unmeasured feature evaluates to unknown, not false.

## Layout

- `store.py`: `FeatureStore`, the only module that reads disk. It reads five allowlisted tables
  (`processed/cells`, `interim/drugs`, `interim/drug_targets`, `interim/features/cell_expression`,
  `interim/features/cell_copy_number`) and refuses any other.
- `rules.py`: rule parser and evaluator.
- `omics.py`: the tool functions, `(store, **args) -> {"summary", "evidence"}`. The summary is the
  full text the model reads; SHELDON shows only 5 evidence items of 800 characters, so nothing
  important lives only in the evidence.
- `registry.py`: one `ToolSpec` per tool with the backend it needs. `query_model`, `resolve` (M3),
  `get_history` (M2) and `search_literature` (M5) are declared with schemas and no implementation;
  give one a function and pass its backend to `as_probes` to switch it on.
- `probes.py`: `as_probes(backends)` wraps specs as SHELDON `CallableProbe`s; `python_context(store)`
  builds the sandbox preload (JSON only, since SHELDON passes it through `json.dumps`).

`combo_agent/benchmark.py` wires these into the `combo-mechanism` benchmark with three hand-written
smoke questions. Its score only checks that the answer is well formed; it is not a quality measure.

    uv run combo-agent run -c configs/sandbox_smoke.yaml

## Leakage controls

- The store allowlist, with an explicit denylist of outcome files (`labels`, `self_pairs`,
  `oneil_rows`, `cross_study_overlap`).
- A source scan: no module in `tools/` other than `store.py` reads parquet, and none names an
  outcome column or file.
- An output scan (`COMBO_INTEGRATION=1`): every tool's output is checked against all hidden `y_syn`
  and `y_inh` values to 3 decimals.
- The smoke config sets `probes: []`, which removes SHELDON's `run_shell`.

**Known gap.** SHELDON's filesystem isolation uses bubblewrap, which is Linux-only, and its
`run_python` is a plain subprocess. On macOS a snippet could read `data_cache/` if it had the path.
`GuardedPythonProbe` refuses snippets that mention the data cache, parquet or an outcome file. That
is a tripwire, not a boundary. Before scored runs, either run on Linux with `Benchmark.sandbox()`
declared, or keep `labels.parquet` outside anything the agent process can read and serve it through
the oracle only.

## Not in this version

DepMap CRISPR dependencies (needs a new download), pathway enrichment, and sub-models fit on
revealed results (needs the M2 oracle).
