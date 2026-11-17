import math

import pytest
from pydantic import ValidationError

from archkit.capacity import ModelRoute, Workload, model_demand
from archkit.cost import (
    break_even,
    deployment_options,
    hybrid_option,
    payg_monthly_cost,
    ptu_monthly_cost,
    ptus_for_tpm,
)
from archkit.estimate import build_estimate, load_estimate_input
from archkit.pricing import ModelPrice, PtuSpec, load_prices
from archkit.search_sizing import SearchWorkload, embedding_monthly_cost, index_size, search_options

FLAT = [1.0] * 24


def make_workload(requests_per_day=144_000, profile=None, burst=1.0, cached=0.0) -> Workload:
    return Workload(
        name="test",
        requests_per_day=requests_per_day,
        hourly_profile=profile or FLAT,
        burst_factor=burst,
        routes=[
            ModelRoute(
                model="m",
                share=1.0,
                llm_calls_per_request=2,
                prompt_tokens_per_call=1000,
                completion_tokens_per_call=100,
                cached_prompt_fraction=cached,
            )
        ],
    )


SPEC = PtuSpec(
    input_tpm_per_ptu=10_000,
    output_token_weight=4.0,
    min_ptus=10,
    increment=5,
    hourly_per_ptu=1.0,
    monthly_reservation_per_ptu=500.0,
)
PRICE = ModelPrice(input_per_1m=2.0, cached_input_per_1m=1.0, output_per_1m=8.0, ptu=SPEC)


def test_bundled_prices_load_and_are_labelled_placeholders():
    book = load_prices()
    assert "not current Azure prices" in book.disclaimer
    assert {"chat-small", "chat-large", "embedding"} <= set(book.models)
    assert book.model("embedding").ptu is None
    with pytest.raises(KeyError, match="not in the price book"):
        book.model("unknown")


def test_demand_for_flat_profile_matches_hand_calculation():
    demand = model_demand(make_workload())["m"]
    # 144,000 requests per day = 100 per minute; 2 calls x 1000 prompt tokens.
    assert demand.peak_input_tpm == pytest.approx(200_000)
    assert demand.peak_output_tpm == pytest.approx(20_000)
    assert demand.monthly_input_tokens == pytest.approx(144_000 * 30 * 2000)
    assert demand.weighted_peak_tpm(4.0) == pytest.approx(280_000)


def test_burst_factor_and_profile_raise_the_peak():
    peaky = [0.0] * 24
    peaky[10] = 1.0
    demand = model_demand(make_workload(profile=peaky, burst=2.0))["m"]
    # All 144,000 requests fall in one hour: 2,400 per minute, times burst 2.
    assert demand.peak_input_tpm == pytest.approx(2400 * 2000 * 2.0)


def test_routes_to_the_same_model_are_summed_and_shares_validated():
    workload = Workload(
        name="w",
        requests_per_day=1000,
        routes=[
            ModelRoute(model="m", share=0.5, prompt_tokens_per_call=100, completion_tokens_per_call=10),
            ModelRoute(model="m", share=0.5, prompt_tokens_per_call=300, completion_tokens_per_call=30),
        ],
    )
    demand = model_demand(workload)["m"]
    assert demand.monthly_input_tokens == pytest.approx(1000 * 30 * 200)
    with pytest.raises(ValidationError, match="sum to 1.0"):
        Workload(
            name="w",
            requests_per_day=1,
            routes=[ModelRoute(model="m", share=0.6, prompt_tokens_per_call=1, completion_tokens_per_call=1)],
        )


def test_payg_cost_with_cached_tokens():
    demand = model_demand(make_workload(cached=0.5))["m"]
    millions_in = 144_000 * 30 * 2000 / 1e6
    millions_out = 144_000 * 30 * 200 / 1e6
    expected = millions_in * 0.5 * 2.0 + millions_in * 0.5 * 1.0 + millions_out * 8.0
    assert payg_monthly_cost(demand, PRICE) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("tpm", "expected"),
    [(0, 0), (1, 10), (100_000, 10), (100_001, 15), (150_000, 15), (280_000, 30)],
)
def test_ptu_rounding_to_minimum_and_increment(tpm, expected):
    assert ptus_for_tpm(tpm, SPEC) == expected


def test_ptu_monthly_cost_by_commitment():
    assert ptu_monthly_cost(10, SPEC, 730, "monthly") == 5000
    assert ptu_monthly_cost(10, SPEC, 730, "hourly") == 7300


def test_full_ptu_wins_on_a_flat_profile():
    demand = model_demand(make_workload())["m"]
    options = deployment_options(demand, PRICE, 730)
    assert options[0].strategy == "ptu"
    assert options[0].ptus == 30
    assert options[0].average_utilization == pytest.approx(280_000 / 300_000)
    assert options[0].total == 15_000
    assert options == sorted(options, key=lambda o: o.total)


def test_hybrid_spillover_arithmetic():
    profile = [1.0] * 12 + [3.0] * 12
    demand = model_demand(make_workload(profile=profile))["m"]
    weighted = demand.weighted_hourly_tpm(4.0)
    assert weighted[0] == pytest.approx(140_000) and weighted[-1] == pytest.approx(420_000)
    option = hybrid_option(demand, PRICE, 15, 730)
    spilled = 12 * (420_000 - 150_000)
    total = 12 * 140_000 + 12 * 420_000
    assert option.spillover_fraction == pytest.approx(spilled / total)
    assert option.payg_cost == pytest.approx(payg_monthly_cost(demand, PRICE) * spilled / total)
    assert option.average_utilization == pytest.approx((total - spilled) / (150_000 * 24))


def test_hybrid_beats_both_extremes_on_a_peaky_profile():
    profile = [1.0] * 20 + [6.0] * 4
    demand = model_demand(make_workload(profile=profile))["m"]
    options = deployment_options(demand, PRICE, 730)
    assert options[0].strategy == "hybrid"
    assert 0 < options[0].spillover_fraction < 1


def test_break_even_utilization_and_volume():
    demand = model_demand(make_workload())["m"]
    result = break_even(demand, PRICE, 730)
    payg_per_weighted = payg_monthly_cost(demand, PRICE) / (
        demand.monthly_input_tokens + 4 * demand.monthly_output_tokens
    )
    assert result.utilization == pytest.approx(500 / (10_000 * 730 * 60 * payg_per_weighted))
    assert result.reachable
    # At the reported volume a peak-sized PTU deployment costs no more than pay-as-you-go.
    scaled = demand.scaled(result.monthly_requests / demand.monthly_requests)
    ptus = ptus_for_tpm(scaled.weighted_peak_tpm(4.0), SPEC)
    assert ptu_monthly_cost(ptus, SPEC, 730) <= payg_monthly_cost(scaled, PRICE)
    assert result.monthly_requests < demand.monthly_requests


def test_break_even_unreachable_when_ptu_is_too_expensive():
    cheap = ModelPrice(input_per_1m=0.01, output_per_1m=0.02, ptu=SPEC)
    result = break_even(model_demand(make_workload())["m"], cheap, 730)
    assert not result.reachable and result.monthly_requests is None
    assert break_even(model_demand(make_workload())["m"], ModelPrice(input_per_1m=1.0), 730) is None


def test_index_size_formula():
    workload = SearchWorkload(
        documents=1_000_000,
        chunks_per_document=1,
        avg_chunk_chars=1024,
        vector_dimensions=1024,
        quantization="none",
        hnsw_m=4,
        churn_margin=0.0,
        growth_factor=1.0,
        text_index_factor=1.0,
        peak_queries_per_second=1,
    )
    size = index_size(workload)
    vector_bytes = 1_000_000 * (1024 * 4 + 4 * 8 * 1.1)
    assert size.vector_index_gb == pytest.approx(vector_bytes / 1024**3)
    assert size.storage_gb == pytest.approx((vector_bytes + 1_000_000 * 1024) / 1024**3)


def test_quantization_shrinks_vector_index_but_keeps_originals_on_disk():
    base = dict(documents=1_000_000, chunks_per_document=10, vector_dimensions=1536, peak_queries_per_second=5)
    full = index_size(SearchWorkload(**base, quantization="none"))
    scalar = index_size(SearchWorkload(**base, quantization="scalar"))
    discarded = index_size(SearchWorkload(**base, quantization="scalar", keep_original_vectors=False))
    assert scalar.vector_index_gb < full.vector_index_gb / 3
    assert scalar.storage_gb > full.storage_gb
    assert discarded.storage_gb < full.storage_gb


def test_search_options_respect_quota_replicas_and_cost_order():
    pricing = load_prices().search
    workload = SearchWorkload(
        documents=2_000_000,
        chunks_per_document=12,
        vector_dimensions=1536,
        quantization="scalar",
        peak_queries_per_second=25,
        availability="read_write",
    )
    size = index_size(workload)
    options = search_options(workload, pricing)
    assert options and options == sorted(options, key=lambda o: (o.monthly_cost, o.search_units))
    for option in options:
        tier = pricing.tiers[option.tier]
        assert option.partitions in pricing.allowed_partition_counts
        assert option.partitions * tier.vector_gb_per_partition >= size.vector_index_gb
        assert option.partitions * tier.storage_gb_per_partition >= size.storage_gb
        assert option.replicas >= max(3, math.ceil(25 / tier.qps_per_replica))
        assert option.search_units <= pricing.max_search_units
    assert "basic" not in {o.tier for o in options}


def test_search_returns_no_option_when_nothing_fits():
    workload = SearchWorkload(
        documents=10**9, chunks_per_document=20, vector_dimensions=3072, peak_queries_per_second=1
    )
    assert search_options(workload, load_prices().search) == []


def test_embedding_cost_is_amortised_per_month():
    workload = SearchWorkload(
        documents=1000,
        chunks_per_document=10,
        growth_factor=1.0,
        avg_chunk_tokens=500,
        full_reindex_per_year=12,
        vector_dimensions=8,
        peak_queries_per_second=1,
    )
    assert embedding_monthly_cost(workload, 2.0) == pytest.approx(1000 * 10 * 500 * 2.0 / 1e6)


def test_example_estimate_is_consistent(repo_root):
    estimate = build_estimate(load_estimate_input(repo_root / "examples" / "workload.yaml"))
    assert estimate.requests_per_month == 12_000_000
    parts = (
        estimate.model_cost
        + estimate.search_cost
        + estimate.embedding_cost
        + estimate.query_embedding_cost
        + sum(estimate.platform_costs.values())
    )
    assert estimate.total == pytest.approx(parts)
    assert estimate.to_dict()["total_monthly"] == pytest.approx(estimate.total)
    markdown = estimate.to_markdown()
    assert "not current Azure prices" in markdown
    assert "(recommended)" in markdown
