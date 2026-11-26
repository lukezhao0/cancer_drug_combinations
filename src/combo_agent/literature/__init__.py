"""M5: literature retrieval with a leakage filter, and the zero-shot memory probe (design section 10).

    pubmed      cached E-utilities client (esearch, efetch, esummary, elink)
    leakage     blocklist rules and the filter every retrieved document passes through
    retrieval   `search_literature`, the agent's tool, which writes an audit record per call
    probe       zero-shot memory probe: the LLM scores pool triplets with no tools
"""
