"""
Hybrid two-stage retrieval (sparse TF-IDF UNION dense bi-encoder, pretrained
cross-encoder rerank) for grounding an agent's output in a local document
collection, instead of trusting an unverified citation.

Promoted from quantumrag (Desktop/Fullwork/quantumrag), a local RAG tool used
through Dense-Evolution-Discovery to check physics/chemistry claims against
real, independently-verified arXiv papers before citing them -- validated
there across ~30 topic collections (~6300 chunks) before this promotion. Only
the retrieval mechanism moves here, not quantumrag's own paper corpus or
built indexes -- those stay local, out of the package.

Stage 1 pools candidates from TF-IDF cosine (exact-term overlap, cheap, whole
collection) UNION a dense bi-encoder (catches paraphrases/synonyms TF-IDF
misses on its own -- Karpukhin et al. 2020, "Dense Passage Retrieval for
Open-Domain Question Answering", arXiv:2004.04906). Stage 2 reranks that pool
with a pretrained cross-encoder (Nogueira & Cho 2019, "Passage Re-ranking
with BERT", arXiv:1901.04085) -- a slower model that reads query+chunk
together instead of comparing two separate vectors, applied only to the
(small) pooled candidates, never the whole collection.

Needs the optional `rag` extra (scikit-learn + sentence-transformers) --
same pattern as native_hf.libcint_bridge (pyscf) and qmmm.region (rdkit):
the whole module is off-limits without it, since even the TF-IDF-only path
(build_index(compute_embeddings=False), search(rerank=False)) is built on
scikit-learn's TfidfVectorizer/cosine_similarity. chunk_text (pure Python,
no vectorizer/model involved) is the one function usable with no extra at
all -- it doesn't import this module's sklearn/sentence-transformers names.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

CHUNK_SIZE_CHARS = 1200
CHUNK_OVERLAP_CHARS = 200
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_POOL = 20

# Keyed by model name so build_index/search calls with the same default
# model (the normal case) only ever load it once per process, regardless of
# how many collections/queries are processed.
_embedder_cache: dict = {}
_reranker_cache: dict = {}

_MISSING_RAG_EXTRA = (
    "needs the optional 'rag' extra (pip install dense-evolution[rag]) -- "
    "failed to import: {}"
)


def _get_embedder(model_name: str):
    if model_name not in _embedder_cache:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as _import_error:  # pragma: no cover -- only reachable without sentence-transformers installed, which CI here always has
            raise ImportError(_MISSING_RAG_EXTRA.format(_import_error)) from _import_error
        _embedder_cache[model_name] = SentenceTransformer(model_name)
    return _embedder_cache[model_name]


def _get_reranker(model_name: str):
    if model_name not in _reranker_cache:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as _import_error:  # pragma: no cover -- only reachable without sentence-transformers installed, which CI here always has
            raise ImportError(_MISSING_RAG_EXTRA.format(_import_error)) from _import_error
        _reranker_cache[model_name] = CrossEncoder(model_name)
    return _reranker_cache[model_name]


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = CHUNK_SIZE_CHARS,
    overlap: int = CHUNK_OVERLAP_CHARS,
) -> list:
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        piece = text[start:end].strip()
        if piece:
            chunks.append({"source": source, "text": piece})
        if end == n:
            break
        start = end - overlap
    return chunks


@dataclass
class RagIndex:
    chunks: list
    vectorizer: object  # sklearn.feature_extraction.text.TfidfVectorizer, fitted
    matrix: object  # scipy.sparse matrix, vectorizer's transform of every chunk's text
    embeddings: Optional[np.ndarray] = None


def build_index(
    documents,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    compute_embeddings: bool = True,
    chunk_size: int = CHUNK_SIZE_CHARS,
    overlap: int = CHUNK_OVERLAP_CHARS,
) -> RagIndex:
    """
    documents: iterable of (text, source) pairs, one per already-extracted
    document (PDF/markdown/plain text parsing happens before this call --
    this module only chunks and indexes text it's handed). compute_embeddings
    can be set False to skip the (slower) bi-encoder pass and build a
    TF-IDF-only index -- still needs the 'rag' extra for scikit-learn, just
    not sentence-transformers' model download; search() then falls back to
    plain cosine regardless of `rerank`.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError as _import_error:  # pragma: no cover -- only reachable without scikit-learn installed, which CI here always has
        raise ImportError(_MISSING_RAG_EXTRA.format(_import_error)) from _import_error

    all_chunks = []
    for text, source in documents:
        all_chunks.extend(chunk_text(text, source, chunk_size, overlap))
    if not all_chunks:
        raise ValueError("no chunks produced -- 'documents' was empty or every text was blank")

    texts = [c["text"] for c in all_chunks]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=20000, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(texts)

    embeddings = None
    if compute_embeddings:
        embeddings = _get_embedder(embedding_model).encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )

    return RagIndex(chunks=all_chunks, vectorizer=vectorizer, matrix=matrix, embeddings=embeddings)


def save_index(index: RagIndex, index_dir) -> None:
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    with open(index_dir / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(index.chunks, f, ensure_ascii=False, indent=2)
    with open(index_dir / "vectorizer.pkl", "wb") as f:
        pickle.dump(index.vectorizer, f)
    with open(index_dir / "matrix.pkl", "wb") as f:
        pickle.dump(index.matrix, f)
    if index.embeddings is not None:
        np.save(index_dir / "embeddings.npy", index.embeddings)


def load_index(index_dir) -> RagIndex:
    index_dir = Path(index_dir)
    with open(index_dir / "chunks.json", encoding="utf-8") as f:
        chunks = json.load(f)
    with open(index_dir / "vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open(index_dir / "matrix.pkl", "rb") as f:
        matrix = pickle.load(f)
    emb_path = index_dir / "embeddings.npy"
    embeddings = np.load(emb_path) if emb_path.exists() else None
    return RagIndex(chunks=chunks, vectorizer=vectorizer, matrix=matrix, embeddings=embeddings)


def search(
    query: str,
    index: RagIndex,
    top: int = 3,
    rerank: bool = True,
    pool: int = DEFAULT_POOL,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    cross_encoder_model: str = DEFAULT_CROSS_ENCODER_MODEL,
) -> list:
    """
    Returns the top `top` chunks as a list of dicts (rank, source, text,
    score, score_type, cosine), highest-scoring first.

    rerank=True (default): stage 1 pools `pool` candidates from TF-IDF
    cosine UNION dense-embedding similarity (if the index has embeddings --
    plain TF-IDF pool otherwise), stage 2 reorders that pool with the
    cross-encoder and returns its top `top` ("score_type": "rerank").
    rerank=False: plain TF-IDF cosine ranking, no sentence-transformers model
    load needed (still needs the 'rag' extra's scikit-learn for cosine_similarity
    itself) -- useful as a baseline, or when only embeddings are unavailable
    for this index but a scored ranking is still wanted.
    """
    try:
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError as _import_error:  # pragma: no cover -- only reachable without scikit-learn installed, which CI here always has
        raise ImportError(_MISSING_RAG_EXTRA.format(_import_error)) from _import_error

    query_vec = index.vectorizer.transform([query])
    cosine_scores = cosine_similarity(query_vec, index.matrix)[0]
    tfidf_order = cosine_scores.argsort()[::-1]

    if rerank and index.embeddings is not None:
        query_emb = _get_embedder(embedding_model).encode([query], normalize_embeddings=True)[0]
        dense_scores = index.embeddings @ query_emb
        dense_pool = dense_scores.argsort()[::-1][:pool]
        candidate_idx = sorted(set(tfidf_order[:pool].tolist()) | set(dense_pool.tolist()))
    elif rerank:
        candidate_idx = tfidf_order[:pool].tolist()
    else:
        candidate_idx = []

    if rerank and candidate_idx:
        pairs = [(query, index.chunks[i]["text"][:1024]) for i in candidate_idx]
        rerank_scores = _get_reranker(cross_encoder_model).predict(pairs)
        order = rerank_scores.argsort()[::-1][:top]
        top_idx = [candidate_idx[j] for j in order]
        scores = {candidate_idx[j]: float(rerank_scores[j]) for j in order}
        score_type = "rerank"
    else:
        top_idx = tfidf_order[:top].tolist()
        scores = {i: float(cosine_scores[i]) for i in top_idx}
        score_type = "cosine"

    return [
        {
            "rank": rank,
            "source": index.chunks[i]["source"],
            "text": index.chunks[i]["text"],
            "score": scores[i],
            "score_type": score_type,
            "cosine": float(cosine_scores[i]),
        }
        for rank, i in enumerate(top_idx, 1)
    ]
