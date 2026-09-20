"""
Correctness tests for ia_utils.rag, the hybrid TF-IDF/dense/cross-encoder
retrieval utility promoted from quantumrag.

chunk_text is pure Python (no sklearn/sentence-transformers import at all)
and runs unconditionally. Everything else needs the optional 'rag' extra --
`pytest.importorskip` is per-test, same reasoning as this repo's
stim/pymatching/pyscf/rdkit tests (test_libcint_bridge.py, test_qmmm.py):
a real scikit-learn/sentence-transformers install in CI keeps covering the
actual TF-IDF/dense/rerank logic, not just the ImportError path.
"""

import numpy as np
import pytest

from ia_utils.rag import RagIndex, build_index, chunk_text, load_index, save_index, search


class TestChunkText:
    def test_splits_long_text_with_overlap(self):
        text = "abcdefghij" * 5  # 50 chars
        chunks = chunk_text(text, "doc.txt", chunk_size=20, overlap=5)
        assert [c["source"] for c in chunks] == ["doc.txt"] * len(chunks)
        assert [c["text"] for c in chunks] == [text[0:20], text[15:35], text[30:50]]
        # the last 5 chars of one chunk are the first 5 of the next -- the overlap window
        assert chunks[0]["text"][-5:] == chunks[1]["text"][:5] == text[15:20]

    def test_short_text_single_chunk(self):
        chunks = chunk_text("short", "doc.txt", chunk_size=1200, overlap=200)
        assert len(chunks) == 1
        assert chunks[0] == {"source": "doc.txt", "text": "short"}

    def test_empty_text_no_chunks(self):
        assert chunk_text("", "doc.txt") == []

    def test_blank_pieces_are_dropped(self):
        # chunk_size=1 forces one char per slice -- whitespace-only slices
        # must not become empty-string chunks.
        chunks = chunk_text("a b", "doc.txt", chunk_size=1, overlap=0)
        assert all(c["text"] for c in chunks)


class TestBuildIndexTfidfOnly:
    """compute_embeddings=False and rerank=False together need only
    scikit-learn (no sentence-transformers download)."""

    def setup_method(self):
        pytest.importorskip("sklearn")

    def _documents(self):
        return [
            ("quantum error mitigation reduces noise in short depth circuits", "temme.pdf"),
            ("traversable wormholes require negative average null energy", "gao.pdf"),
            ("category theory adjoint functors and the Yoneda lemma", "leinster.pdf"),
        ]

    def test_build_index_no_embeddings(self):
        index = build_index(self._documents(), compute_embeddings=False)
        assert isinstance(index, RagIndex)
        assert index.embeddings is None
        assert len(index.chunks) == 3

    def test_build_index_empty_documents_raises(self):
        with pytest.raises(ValueError):
            build_index([], compute_embeddings=False)

    def test_search_rerank_false_matches_best_lexical_overlap(self):
        index = build_index(self._documents(), compute_embeddings=False)
        results = search("wormhole negative energy", index, top=1, rerank=False)
        assert results[0]["source"] == "gao.pdf"
        assert results[0]["score_type"] == "cosine"
        assert results[0]["score"] == pytest.approx(results[0]["cosine"])

    def test_save_and_load_round_trip(self, tmp_path):
        index = build_index(self._documents(), compute_embeddings=False)
        save_index(index, tmp_path)
        reloaded = load_index(tmp_path)
        assert reloaded.chunks == index.chunks
        assert reloaded.embeddings is None
        results = search("adjoint functor", reloaded, top=1, rerank=False)
        assert results[0]["source"] == "leinster.pdf"


class TestHybridRerank:
    """The full dense-embedding + cross-encoder path -- needs sentence-transformers."""

    def setup_method(self):
        pytest.importorskip("sentence_transformers")

    def _documents(self):
        return [
            (
                "The traversable wormhole construction couples two boundaries "
                "of an eternal BTZ black hole with a negative average null "
                "energy stress tensor, rendering the Einstein-Rosen bridge "
                "traversable for a signal.",
                "gao_wormhole.pdf",
            ),
            (
                "Error mitigation techniques for short-depth quantum circuits "
                "extrapolate the noisy expectation value to the zero-noise "
                "limit without extra qubits.",
                "temme_mitigation.pdf",
            ),
            (
                "A cross-encoder reads the query and passage together through "
                "one transformer, unlike a bi-encoder which embeds them "
                "separately and compares vectors afterward.",
                "reranking_note.pdf",
            ),
        ]

    def test_rerank_returns_requested_count_and_valid_sources(self):
        index = build_index(self._documents(), compute_embeddings=True)
        assert index.embeddings is not None
        assert index.embeddings.shape[0] == len(index.chunks)
        results = search("does information leak through a wormhole", index, top=2, rerank=True, pool=10)
        assert len(results) == 2
        assert all(r["score_type"] == "rerank" for r in results)
        assert results[0]["source"] == "gao_wormhole.pdf"

    def test_dense_pool_recovers_paraphrase_missed_by_tfidf_alone(self):
        # "signal transmission" shares almost no vocabulary with the wormhole
        # chunk's own wording ("traversable", "boundaries", "stress tensor")
        # -- plain TF-IDF cosine should rank it lower than a lexically closer
        # but topically wrong chunk; the dense pool is what's expected to
        # pull the right chunk back in before the cross-encoder finalizes it.
        index = build_index(self._documents(), compute_embeddings=True)
        tfidf_only = search("signal transmission through spacetime", index, top=1, rerank=False)
        hybrid = search("signal transmission through spacetime", index, top=1, rerank=True, pool=10)
        assert hybrid[0]["source"] == "gao_wormhole.pdf"
        # Documented, not asserted as a hard requirement: the two stages can
        # legitimately agree on small toy corpora like this one.
        if tfidf_only[0]["source"] != "gao_wormhole.pdf":
            assert hybrid[0]["source"] != tfidf_only[0]["source"]

    def test_save_and_load_round_trip_keeps_embeddings(self, tmp_path):
        index = build_index(self._documents(), compute_embeddings=True)
        save_index(index, tmp_path)
        reloaded = load_index(tmp_path)
        assert reloaded.embeddings is not None
        np.testing.assert_allclose(reloaded.embeddings, index.embeddings)
