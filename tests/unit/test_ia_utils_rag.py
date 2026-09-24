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

from ia_utils.rag import RagIndex, build_index, chunk_text, load_index, save_index, search, search_exact


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
    """Needs scikit-learn but not sentence-transformers -- no model
    download. test_union_pool_includes_dense_only_candidate lives here too:
    it stubs out _get_embedder/_get_reranker entirely, so it needs neither
    sentence-transformers nor any network access."""

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

    def test_union_pool_includes_dense_only_candidate(self, monkeypatch):
        # Deterministic test of this module's OWN pooling code, not of a
        # pretrained model's exact numeric judgment on a close call (a real
        # cross-encoder's tie-breaking can legitimately differ by a fraction
        # of a point across torch versions/platforms -- not something this
        # project's code controls or should hard-assert on). _get_embedder/
        # _get_reranker are stubbed out here so this test needs no
        # sentence-transformers install and no model download at all.
        docs = [
            ("alpha beta gamma delta epsilon", "a.pdf"),
            ("zeta eta theta iota kappa", "b.pdf"),
            ("lambda mu nu xi omicron", "c.pdf"),
        ]
        index = build_index(docs, compute_embeddings=False)
        # a.pdf: index 0, b.pdf: index 1, c.pdf: index 2 (one chunk each,
        # build order preserved) -- an orthonormal embedding table where the
        # query embedding exactly matches c.pdf's row, so its dense score is
        # a unique, unambiguous top-1 (1.0 vs. 0.0 for the other two).
        index.embeddings = np.eye(3)

        class _FakeEmbedder:
            @staticmethod
            def encode(texts, normalize_embeddings=True):
                return np.array([[0.0, 0.0, 1.0]])

        seen_pairs = []

        class _FakeReranker:
            @staticmethod
            def predict(pairs):
                seen_pairs.extend(pairs)
                return np.arange(len(pairs), dtype=float)

        monkeypatch.setattr("ia_utils.rag._get_embedder", lambda model_name: _FakeEmbedder())
        monkeypatch.setattr("ia_utils.rag._get_reranker", lambda model_name: _FakeReranker())

        # "alpha" only appears in a.pdf -- a size-1 TF-IDF pool keeps only
        # a.pdf, excluding c.pdf entirely on its own.
        tfidf_only_pool = search("alpha exploration mission", index, top=1, rerank=False)
        assert tfidf_only_pool[0]["source"] == "a.pdf"

        search("alpha exploration mission", index, top=3, rerank=True, pool=1)
        pooled_sources = {text for _, text in seen_pairs}
        assert "lambda mu nu xi omicron" in pooled_sources  # c.pdf reached the reranker via the dense pool
        assert "zeta eta theta iota kappa" not in pooled_sources  # b.pdf was in neither pool


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

    def test_save_and_load_round_trip_keeps_embeddings(self, tmp_path):
        index = build_index(self._documents(), compute_embeddings=True)
        save_index(index, tmp_path)
        reloaded = load_index(tmp_path)
        assert reloaded.embeddings is not None
        np.testing.assert_allclose(reloaded.embeddings, index.embeddings)


class TestSearchExact:
    """No embedding, no reranker, no model download -- needs scikit-learn
    only to build the index (build_index's TfidfVectorizer fit), not to
    search it. Promoted from quantumrag's query.py --exact/--regex mode,
    which the promoted ia_utils.rag had originally been missing."""

    def setup_method(self):
        pytest.importorskip("sklearn")

    def _index(self):
        return build_index(
            [("The answer is 42 kelvin, measured at standard pressure.", "note.txt")],
            compute_embeddings=False,
        )

    def test_literal_substring_case_insensitive(self):
        hits = search_exact("42 KELVIN", self._index())
        assert len(hits) == 1
        assert hits[0]["source"] == "note.txt"
        assert hits[0]["match"] == "42 kelvin"

    def test_no_match_returns_empty_list(self):
        assert search_exact("does not appear", self._index()) == []

    def test_regex_mode_is_case_sensitive_and_matches_pattern(self):
        hits = search_exact(r"\d+ kelvin", self._index(), regex=True)
        assert len(hits) == 1
        assert hits[0]["match"] == "42 kelvin"
        assert search_exact(r"\d+ KELVIN", self._index(), regex=True) == []  # regex mode: case-sensitive

    def test_literal_mode_does_not_treat_pattern_as_regex(self):
        # '.' would match anything as a regex; as a literal it must not.
        index = build_index([("value is 3x14 exactly for testing", "doc.txt")], compute_embeddings=False)
        assert search_exact("3.14", index, regex=True) != []  # regex: '.' matches the 'x'
        assert search_exact("3.14", index) == []              # literal: '.' must match only a literal dot

    def test_max_hits_truncates_and_marks_the_last_result(self):
        # search_exact finds at most one match per chunk (see its own
        # docstring), so three separate documents/chunks are needed to
        # exercise truncation across matches, not repeated text within one.
        docs = [
            ("the target word appears here in document one", "doc1.txt"),
            ("the target word appears here in document two", "doc2.txt"),
            ("the target word appears here in document three", "doc3.txt"),
        ]
        index = build_index(docs, compute_embeddings=False)
        hits = search_exact("target", index, max_hits=2)
        assert len(hits) == 2
        assert hits[-1].get("truncated") is True
        assert "truncated" not in hits[0]

    def test_context_window_bounds_the_snippet(self):
        text = "x" * 50 + "TARGET" + "y" * 50
        index = build_index([(text, "doc.txt")], compute_embeddings=False)
        hits = search_exact("TARGET", index, context=5)
        assert hits[0]["snippet"] == "..." + "x" * 5 + "TARGET" + "y" * 5 + "..."
