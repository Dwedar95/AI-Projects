"""Azure AI Search sizing: index size, partitions, replicas and tier choice.

The size model is an approximation for planning, not a substitute for
measuring a pilot index:

* vector index (in memory quota) = vectors x (dimensions x bytes per dimension
  + HNSW graph overhead), plus a margin for deleted and updated documents
* storage = text content and inverted index + vector data on disk
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from archkit.pricing import SearchPricing, SearchTier

GB = 1024**3
BYTES_PER_DIMENSION = {"none": 4.0, "scalar": 1.0, "binary": 0.125}
HNSW_BYTES_PER_LINK = 8.0
HNSW_LEVEL_FACTOR = 1.1

Availability = Literal["none", "read", "read_write"]
MIN_REPLICAS = {"none": 1, "read": 2, "read_write": 3}


class SearchWorkload(BaseModel):
    """Corpus and query load of the retrieval index."""

    model_config = ConfigDict(extra="forbid")

    documents: int = Field(gt=0)
    chunks_per_document: float = Field(gt=0)
    avg_chunk_chars: float = Field(default=2000, gt=0)
    vector_dimensions: int = Field(gt=0)
    vector_fields: int = Field(default=1, ge=1)
    quantization: Literal["none", "scalar", "binary"] = "none"
    keep_original_vectors: bool = True
    hnsw_m: int = Field(default=4, ge=4, le=10)
    text_index_factor: float = Field(default=2.0, ge=1, description="Stored text plus inverted index, per text byte.")
    churn_margin: float = Field(default=0.2, ge=0, description="Extra space for deleted and updated documents.")
    growth_factor: float = Field(default=1.5, ge=1, description="Planned corpus growth over the sizing horizon.")
    peak_queries_per_second: float = Field(gt=0)
    availability: Availability = "read"
    avg_chunk_tokens: float = Field(default=500, gt=0)
    full_reindex_per_year: float = Field(default=2, ge=0)

    @property
    def chunks(self) -> float:
        """Number of chunks after planned growth."""
        return self.documents * self.chunks_per_document * self.growth_factor


@dataclass(frozen=True)
class IndexSize:
    chunks: float
    vector_index_gb: float
    storage_gb: float


@dataclass(frozen=True)
class SearchOption:
    """A feasible tier configuration."""

    tier: str
    partitions: int
    replicas: int
    monthly_cost: float
    vector_headroom: float
    storage_headroom: float

    @property
    def search_units(self) -> int:
        return self.partitions * self.replicas


def index_size(workload: SearchWorkload) -> IndexSize:
    """Estimate the vector index size and the total storage of the index."""
    vectors = workload.chunks * workload.vector_fields
    per_vector = workload.vector_dimensions * BYTES_PER_DIMENSION[workload.quantization]
    graph = workload.hnsw_m * HNSW_BYTES_PER_LINK * HNSW_LEVEL_FACTOR
    margin = 1.0 + workload.churn_margin
    vector_index_bytes = vectors * (per_vector + graph) * margin

    text_bytes = workload.chunks * workload.avg_chunk_chars * workload.text_index_factor
    disk_vector_bytes = vector_index_bytes
    if workload.quantization != "none" and workload.keep_original_vectors:
        disk_vector_bytes += vectors * workload.vector_dimensions * BYTES_PER_DIMENSION["none"]
    storage_bytes = (text_bytes + disk_vector_bytes) * margin
    return IndexSize(workload.chunks, vector_index_bytes / GB, storage_bytes / GB)


def _partitions_needed(size: IndexSize, tier: SearchTier, allowed: list[int]) -> int | None:
    """Smallest allowed partition count that holds both the vector index and the storage."""
    required = max(
        math.ceil(size.vector_index_gb / tier.vector_gb_per_partition),
        math.ceil(size.storage_gb / tier.storage_gb_per_partition),
        1,
    )
    return next((count for count in allowed if count >= required and count <= tier.max_partitions), None)


def search_options(workload: SearchWorkload, pricing: SearchPricing) -> list[SearchOption]:
    """Return every feasible tier configuration, cheapest first."""
    size = index_size(workload)
    options: list[SearchOption] = []
    for name, tier in pricing.tiers.items():
        partitions = _partitions_needed(size, tier, pricing.allowed_partition_counts)
        if partitions is None:
            continue
        replicas = max(
            MIN_REPLICAS[workload.availability],
            math.ceil(workload.peak_queries_per_second / tier.qps_per_replica - 1e-9),
        )
        if replicas > tier.max_replicas or replicas * partitions > pricing.max_search_units:
            continue
        options.append(
            SearchOption(
                tier=name,
                partitions=partitions,
                replicas=replicas,
                monthly_cost=replicas * partitions * tier.monthly_per_search_unit,
                vector_headroom=1 - size.vector_index_gb / (partitions * tier.vector_gb_per_partition),
                storage_headroom=1 - size.storage_gb / (partitions * tier.storage_gb_per_partition),
            )
        )
    return sorted(options, key=lambda option: (option.monthly_cost, option.search_units))


def embedding_monthly_cost(workload: SearchWorkload, embedding_price_per_1m: float) -> float:
    """Average monthly cost of (re)embedding the corpus."""
    tokens_per_full_index = workload.chunks * workload.vector_fields * workload.avg_chunk_tokens
    return tokens_per_full_index * workload.full_reindex_per_year / 12 * embedding_price_per_1m / 1e6
