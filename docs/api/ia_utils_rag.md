# IA Utils — Hybrid Retrieval (RAG)

> Retrieval to ground an agent's own output in a document collection, not a
> quantum measurement result — see [Vector Healing](ia_utils_vector_healing.md)
> if you're looking for healing a numeric sequence instead.

An agent answering from a pile of documents has to find the right passage
before it can quote it. The simplest approach, TF-IDF cosine similarity, only
catches passages that share the query's actual words — a query phrased
differently from the source text ("does information leak out?" vs. a paper
that only ever says "traversable") can miss the right passage entirely.
`ia_utils.rag` fixes this with two stages: a cheap first pass pools
candidates from TF-IDF *and* a dense sentence embedding (which catches
paraphrases the exact-word match misses), then a slower, more accurate
cross-encoder model re-reads that small pool and picks the real top results.

## Step 1. Index three documents and ask a question

```python
from ia_utils.rag import build_index, search

docs = [
    ("The traversable wormhole construction couples two boundaries of an "
     "eternal BTZ black hole with a negative average null energy stress "
     "tensor, rendering the Einstein-Rosen bridge traversable for a probe.",
     "gao_wormhole.pdf"),
    ("Error mitigation techniques for short-depth quantum circuits "
     "extrapolate the noisy expectation value to the zero-noise limit "
     "without extra qubits.", "temme_mitigation.pdf"),
    ("A cross-encoder reads the query and passage together through one "
     "transformer, unlike a bi-encoder which embeds them separately and "
     "compares vectors afterward.", "reranking_note.pdf"),
]
index = build_index(docs)
results = search("does information leak through a wormhole", index, top=2)
[(r["rank"], r["source"], round(r["score"], 3)) for r in results]
```

```
[(1, 'gao_wormhole.pdf', -6.315), (2, 'reranking_note.pdf', -11.244)]
```

`build_index` chunks each `(text, source)` pair (1200 characters per chunk,
200-character overlap — a document doesn't have to fit in one chunk) and
builds both a TF-IDF matrix and a dense embedding for every chunk.
`search` returns the top results as plain dicts — `score` is the
cross-encoder's own score by default (higher is better, but not bounded to
`[0, 1]` the way a cosine similarity is), and `cosine` is always included
alongside it so the two can be compared directly, like in Step 2.

## Step 2. Compare against TF-IDF alone

```python
from ia_utils.rag import search

query = "does information leak through a wormhole"
tfidf_only = search(query, index, top=2, rerank=False)
hybrid = search(query, index, top=2, rerank=True)
[(r["source"], r["score_type"]) for r in tfidf_only + hybrid]
```

```
[('gao_wormhole.pdf', 'cosine'), ('reranking_note.pdf', 'cosine'),
 ('gao_wormhole.pdf', 'rerank'), ('reranking_note.pdf', 'rerank')]
```

`rerank=False` skips the dense pool and the cross-encoder entirely — a
plain TF-IDF ranking, `score_type` is `"cosine"` instead of `"rerank"`. On a
three-document toy collection like this one, both approaches often agree
(as they do here) simply because there's nothing left for the dense pool to
add — every document is already inside the candidate pool either way. The
gap opens up on a real, larger collection: see Details below for a real,
measured case where it mattered.

## Step 3. Find a specific known phrase directly

```python
from ia_utils.rag import search_exact

hits = search_exact("negative average null energy", index)
[(h["source"], h["match"]) for h in hits]
```

```
[('gao_wormhole.pdf', 'negative average null energy')]
```

Semantic search ranks by topical similarity, which can bury a short,
specific, load-bearing phrase — an exact clause, a fixed parameter value,
a named condition — under chunks that are merely more topically central.
`search_exact` skips ranking entirely: no embedding, no reranker, no
model download, just a literal (case-insensitive) substring match over
every chunk's raw text, with `context` characters of surrounding text
kept on each side. Pass `regex=True` to treat `pattern` as a real regular
expression instead of a literal string.

## See Also

- [Vector Healing](ia_utils_vector_healing.md) — a different `ia_utils`
  module, correcting a numeric sequence rather than retrieving text.
- [`ia_utils.adversarial_vector_attack`](ia_utils_adversarial_vector_attack.md)
  — gradient-based robustness testing, the third `ia_utils` module.

---

## Details

### Where this came from

Promoted from `quantumrag` (a local literature-grounding tool used through
Dense-Evolution-Discovery to check physics/chemistry claims against real,
independently-verified arXiv papers before citing them), after validation
there across roughly 30 topic collections and ~6300 chunks total. Only the
retrieval mechanism moved here — `quantumrag`'s own paper corpus and built
indexes stay local, out of the package.

### A real case where reranking changed the answer

On `quantumrag`'s own `quantum_info` collection (30 real papers, hundreds of
chunks), the query *"how does error mitigation reduce noise in short depth
circuits"* returned, with TF-IDF alone, three chunks from the *same* paper
(Temme et al. 2017) as its top 3 — technically on-topic, but redundant.
With the hybrid pool + rerank, the third result changed to a chunk from a
different, genuinely relevant paper (a differentiable-Kraus-tensor-networks
review citing the same error-mitigation result) instead of repeating the
first paper a third time. That is the effect this module is for: it shows
up once a collection is large and varied enough for the extra recall and
re-ordering to matter, not necessarily on a handful of documents.

### The optional `rag` extra

`pip install dense-evolution[rag]` (scikit-learn + sentence-transformers).
`chunk_text` is the one function in this module that needs neither — every
other function, `build_index`/`save_index`/`load_index`/`search` included,
needs at least scikit-learn (TF-IDF is built on
`sklearn.feature_extraction.text.TfidfVectorizer`), even with
`compute_embeddings=False` or `rerank=False`. `build_index(...,
compute_embeddings=False)` skips only the sentence-transformers bi-encoder
pass, not scikit-learn itself.

### Model choice and where the two stages come from

Stage 1's dense bi-encoder (`sentence-transformers/all-MiniLM-L6-v2`,
`build_index`'s `embedding_model`) and stage 2's cross-encoder
(`cross-encoder/ms-marco-MiniLM-L-6-v2`, `search`'s `cross_encoder_model`)
are both pretrained, general-purpose models — nothing is trained or
fine-tuned by this module. The two-stage split itself follows Karpukhin et
al. 2020, "Dense Passage Retrieval for Open-Domain Question Answering"
(arXiv:2004.04906) for the dense-retrieval stage, and Nogueira & Cho 2019,
"Passage Re-ranking with BERT" (arXiv:1901.04085) for the cross-encoder
rerank stage — both real, established information-retrieval results, not
new mechanisms invented for this module.

### Persisting an index

`save_index(index, index_dir)` writes `chunks.json`, `vectorizer.pkl`,
`matrix.pkl`, and (if `compute_embeddings=True`) `embeddings.npy` under
`index_dir`. `load_index(index_dir)` reads them back into an identical
`RagIndex` — rebuilding an index from scratch on every query is wasteful
once a collection stops changing.

::: ia_utils.rag
