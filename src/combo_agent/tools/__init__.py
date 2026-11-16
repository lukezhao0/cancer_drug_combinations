"""In-silico tools the agent can call: label-free lookups over the frozen M1 feature tables.

    store.py     FeatureStore, the only module here that reads from disk (allowlisted tables only)
    rules.py     the cell-line rule vocabulary (mutation, copy number, lineage, expression tertile)
    omics.py     the tool functions
    registry.py  tool specs, including tools declared for later milestones
    probes.py    adapter to SHELDON probes and the Python-sandbox preload

Nothing in this package may read experiment outcomes. See `store.DENIED_FILES`.
"""

from __future__ import annotations


class ToolError(ValueError):
    """A tool call the caller can fix: unknown drug, bad rule, missing gene. Shown to the model."""
