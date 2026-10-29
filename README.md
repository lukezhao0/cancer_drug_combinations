# cancer_drug_combinations

CS 224V Project: Agent that combines literature evidence and in-silico experiments to choose cancer drug-combination experiments. Built on [SHELDON](https://github.com/stanford-oval/sheldon).

## Setup

Requirements: Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
git clone <this repo> && cd cancer_drug_combinations
uv sync                 # Creates .venv, installs SHELDON (pinned) and combo_agent (editable)
cp .env.example .env    # Then fill in your API key
```

### API Key (bring your own)

`.env` is read from the directory you run commands from.

```
OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
OPENAI_API_KEY=<your key>
SHELDON_DATA=./data
```

- **Gemini** (default): use the base URL above with a Gemini API key.
- **Any OpenAI-compatible endpoint**: change `OPENAI_BASE_URL` and the key.

### Check Install

```bash
uv run sheldon smoke
uv run combo-agent ls  # Lists benchmarks, methods and engines
```

## CLI

`combo-agent` wraps the `sheldon` CLI, loads `.env`, and registers our benchmark first, so every `sheldon` subcommand works:

```bash
uv run combo-agent ls
uv run combo-agent --help
```

## Files

```
src/combo_agent/
  run.py         # Entry point to run SHELDON CLI
  benchmark.py   # combo-mechanism benchmark
docs/
notes/
```

## Updating SHELDON

SHELDON is pinned to a commit in `pyproject.toml` (`[tool.uv.sources]`). To upgrade, change `rev` and run `uv sync`.
