import math

import numpy as np
import pytest

from rag_eval_kit.bm25 import BM25Index
from rag_eval_kit.documents import Chunk
from rag_eval_kit.embeddings import HashingEmbedder, TfidfSvdEmbedder, get_embedder
from rag_eval_kit.retrieval import (
    DenseIndex,
    FeatureReranker,
    HybridRetriever,
    SearchHit,
    minimal_window,
    reciprocal_rank_fusion,
)

DOCS = [["cat", "sat", "mat"], ["dog", "sat", "log", "log"], ["cat", "cat", "dog"]]


def test_bm25_matches_hand_computed_score():
    index = BM25Index(k1=1.5, b=0.75).fit(DOCS)
    idf_cat = math.log(1 + (3 - 2 + 0.5) / (2 + 0.5))
    avgdl = 10 / 3
    expected_doc2 = idf_cat * 2 * 2.5 / (2 + 1.5 * (1 - 0.75 + 0.75 * 3 / avgdl))
    scores = index.scores(["cat"])
    assert scores[2] == pytest.approx(expected_doc2)
    assert scores[1] == 0
    assert scores[2] > scores[0] > 0


def test_bm25_search_orders_and_skips_non_matching_documents():
    index = BM25Index().fit(DOCS)
    assert [i for i, _ in index.search(["log"], 5)] == [1]
    assert [i for i, _ in index.search(["cat", "dog"], 5)][0] == 2
    assert index.search(["zebra"], 5) == []
    assert BM25Index().fit([]).search(["cat"], 3) == []
    with pytest.raises(ValueError):
        BM25Index(b=1.5)


def test_bm25_length_normalisation_prefers_shorter_document():
    index = BM25Index().fit([["x", "pad", "pad", "pad", "pad", "pad"], ["x", "pad"]])
    scores = index.scores(["x"])
    assert scores[1] > scores[0]


def test_hashing_embedder_is_deterministic_normalised_and_similarity_aware():
    embedder = HashingEmbedder(dim=256).fit(["reset the password", "update the firmware"])
    a, b, c = embedder.embed(["reset password", "resetting passwords", "firmware update procedure"])
    assert np.allclose(np.linalg.norm([a, b, c], axis=1), 1.0)
    assert np.array_equal(
        a, HashingEmbedder(dim=256).fit(["reset the password", "update the firmware"]).embed(["reset password"])[0]
    )
    assert a @ b > a @ c
    assert np.allclose(embedder.embed([""]), 0.0)


def test_tfidf_svd_embedder_shapes_and_requires_fit():
    texts = [f"topic {i} word{i} shared" for i in range(6)]
    embedder = TfidfSvdEmbedder(dim=4).fit(texts)
    vectors = embedder.embed(texts)
    assert vectors.shape == (6, 4)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    with pytest.raises(RuntimeError):
        TfidfSvdEmbedder().embed(["x"])
    with pytest.raises(ValueError, match="unknown embedder"):
        get_embedder("word2vec")


def test_dense_index_returns_nearest_text_first():
    index = DenseIndex(HashingEmbedder()).fit(["annual leave days", "firmware update steps", "password length rules"])
    assert index.search("how to update firmware", 2)[0][0] == 1


def test_rrf_formula_weights_and_tie_break():
    fused = dict(reciprocal_rank_fusion({"a": [1, 2, 3], "b": [3, 1]}, k=60))
    assert fused[1] == pytest.approx(1 / 61 + 1 / 62)
    assert fused[3] == pytest.approx(1 / 63 + 1 / 61)
    assert fused[2] == pytest.approx(1 / 62)
    order = [item for item, _ in reciprocal_rank_fusion({"a": [1, 2, 3], "b": [3, 1]}, k=60)]
    assert order == [1, 3, 2]
    weighted = [item for item, _ in reciprocal_rank_fusion({"a": [1, 2], "b": [2, 1]}, weights={"b": 2.0})]
    assert weighted == [2, 1]
    assert [i for i, _ in reciprocal_rank_fusion({"a": [5, 4], "b": [4, 5]})] == [4, 5]
    with pytest.raises(ValueError):
        reciprocal_rank_fusion({"a": [1]}, k=0)


def test_minimal_window():
    assert minimal_window([[0, 10], [11], [3, 12]]) == 3
    assert minimal_window([[5]]) == 1
    assert minimal_window([[1], []]) is None


def test_reranker_promotes_the_chunk_that_covers_the_query():
    weak = Chunk("a::0", "a", "Travel", "Booking", "Flights are booked in the portal. Hotels have a limit.", 0)
    strong = Chunk("b::0", "b", "Password Policy", "Lockout", "The account is locked after 8 failed attempts.", 0)
    reranker = FeatureReranker({"account": 2.0, "lock": 2.0, "fail": 2.0, "attempt": 2.0})
    hits = [SearchHit(weak, 0.9), SearchHit(strong, 0.8)]
    reranked = reranker.rerank("account locked after failed attempts", hits)
    assert [h.chunk.chunk_id for h in reranked] == ["b::0", "a::0"]
    assert reranked[0].ranks["first_stage"] == 2
    coverage, heading, proximity, prior = reranker.features("account locked after failed attempts", hits[1], 2)
    assert coverage == 1.0 and 0 < proximity <= 1 and prior == 0.5


def test_hybrid_retriever_modes_on_bundled_corpus(pipeline):
    retriever: HybridRetriever = pipeline.retriever
    for mode in ("bm25", "dense", "hybrid"):
        hits = retriever.search("What does error code E-214 mean?", k=3, mode=mode)
        assert len(hits) == 3
        assert hits[0].chunk.doc_id == "product-error-codes"
    hybrid = retriever.search("minimum password length", k=5, mode="hybrid")
    assert {"bm25", "dense"} <= set(hybrid[0].ranks)
    assert [h.score for h in hybrid] == sorted((h.score for h in hybrid), reverse=True)
    with pytest.raises(ValueError):
        retriever.search("x", mode="sparse")
