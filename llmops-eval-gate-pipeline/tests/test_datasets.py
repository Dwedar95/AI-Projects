import numpy as np
import pytest

from llmops_gate.datasets import INTENTS, generate_tickets, read_jsonl, write_jsonl
from llmops_gate.embedding import HashingEmbedder, estimate_tokens


def test_generator_is_deterministic_and_seed_sensitive():
    assert generate_tickets(50, seed=3) == generate_tickets(50, seed=3)
    assert generate_tickets(50, seed=3) != generate_tickets(50, seed=4)


def test_committed_benchmark_matches_the_generator(benchmark):
    assert benchmark == generate_tickets(400, seed=42, id_prefix="bench")
    assert {r["label"] for r in benchmark} == set(INTENTS)


def test_priors_and_new_terms_shift_the_distribution():
    shifted = generate_tickets(500, seed=1, priors=[0, 0, 1, 0, 0], new_term_rate=1.0)
    assert {r["label"] for r in shifted} == {"technical"}
    assert any("webhook" in r["text"] or "api" in r["text"] for r in shifted)
    assert not any("webhook" in r["text"] for r in generate_tickets(500, seed=1))


def test_invalid_rates_are_rejected():
    with pytest.raises(ValueError):
        generate_tickets(10, seed=0, keyword_noise=1.5)


def test_jsonl_round_trip(tmp_path):
    records = generate_tickets(5, seed=0)
    write_jsonl(tmp_path / "nested" / "data.jsonl", records)
    assert read_jsonl(tmp_path / "nested" / "data.jsonl") == records


def test_embedder_is_deterministic_unit_length_and_vocabulary_sensitive():
    embedder = HashingEmbedder(dim=64)
    a = embedder.embed(["invoice charge payment", "invoice charge payment", "crash error bug"])
    assert a.shape == (3, 64) and a.dtype == np.float32
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-6)
    assert np.array_equal(a[0], a[1])
    assert float(a[0] @ a[2]) < 0.5
    assert not embedder.embed_one("").any()
    assert embedder.embed([]).shape == (0, 64)
    with pytest.raises(ValueError):
        HashingEmbedder(dim=2)


def test_token_estimate():
    assert estimate_tokens("") == 1 and estimate_tokens("a" * 40) == 10
