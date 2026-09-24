"""Tools: document retrieval (ia_utils.rag, promoted from the quantumrag
literature-grounding tool). Registered against the shared `mcp` instance
created in server.py -- see that module's docstring for why importing
`mcp` back from there (rather than the other way around) is safe despite
looking circular."""
import json

from ..client import _request, catch_errors
from ..config import COMPUTE
from ..models import RagSearchInput
from ..server import mcp


@mcp.tool(name="dense_evolution_rag_search", annotations={"title": "Search documents with hybrid retrieval", **COMPUTE})
@catch_errors
async def dense_evolution_rag_search(params: RagSearchInput) -> str:
    """Build an ephemeral index over the given documents and search it for
    `query` in one call -- ground an agent's own answer in a document
    collection, not a quantum measurement result. Nothing persists
    server-side: each call builds and searches its own index from
    scratch, so pass the same `documents` again for a second query rather
    than expecting a prior index to still exist.

    Two modes: semantic (default) pools TF-IDF and dense-bi-encoder
    candidates then re-scores with a cross-encoder if rerank=True --
    catches paraphrases plain TF-IDF misses. exact=True instead does a
    substring/regex search over raw chunk text (no embedding, no
    reranker) -- use this when you already know roughly what wording
    you're looking for and semantic ranking buries it under
    topically-similar-but-wrong chunks.

    Needs the `rag` extra (pip install dense-evolution[rag]) -- returns an
    actionable "Error: ..." if it isn't installed.

    Args:
        params (RagSearchInput): documents ([[text, source], ...]), query,
            top, rerank, exact, regex, max_hits, context.

    Returns:
        str: JSON with `results` -- semantic mode: a list of {rank,
        source, text, score, score_type, cosine}; exact mode: a list of
        {source, chunk_index, match, snippet, start, end}.
    """
    return json.dumps(await _request("POST", "/api/rag_search", timeout=60.0, json=params.model_dump()), indent=2)
