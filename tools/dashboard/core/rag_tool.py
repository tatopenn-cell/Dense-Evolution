"""
Thin, stateless Composer-kernel wrapper around ia_utils.rag: builds an
ephemeral index over the caller's own documents and searches it in one
call -- nothing persists server-side between requests, same as every
other kernel endpoint. See docs/api/ia_utils_rag.md for the real
provenance (promoted from quantumrag) and validated case behind this.
Needs the `rag` extra (scikit-learn + sentence-transformers).
"""
from dataclasses import dataclass

from ia_utils.rag import build_index, search, search_exact

__all__ = ['RagSearchResult', 'run_rag_search']


@dataclass
class RagSearchResult:
    results: list


def run_rag_search(documents, query: str, top: int = 5, rerank: bool = True,
                    exact: bool = False, regex: bool = False,
                    max_hits: int = 10, context: int = 300) -> RagSearchResult:
    """Build an ephemeral index over `documents` ([(text, source), ...])
    and search it for `query` in one call, in one of two modes:

    exact=False (default): hybrid semantic search -- pools candidates from
    TF-IDF cosine similarity and a dense bi-encoder, then re-scores the
    pool with a cross-encoder if rerank=True (Karpukhin et al. 2020 /
    Nogueira & Cho 2019); rerank=False skips the dense pool and
    cross-encoder, a plain TF-IDF ranking. Result dicts carry `rank`,
    `source`, `score`, `score_type`, `cosine`, `text`.

    exact=True: substring/regex search over raw chunk text instead --
    no embedding, no reranker, no model download. Use this when semantic
    ranking buries a short, specific, load-bearing phrase (an exact
    clause, a fixed parameter value) under topically-similar-but-wrong
    chunks. `query` is the literal substring (case-insensitive) unless
    regex=True, in which case it's a real, case-sensitive regex. Result
    dicts carry `source`, `chunk_index`, `match`, `snippet`, `start`, `end`.
    """
    index = build_index(documents, compute_embeddings=not exact)
    if exact:
        return RagSearchResult(results=search_exact(query, index, regex=regex, max_hits=max_hits, context=context))
    return RagSearchResult(results=search(query, index, top=top, rerank=rerank))
