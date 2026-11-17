"""rag-eval-kit: a small, offline-capable RAG service with an evaluation framework."""

from rag_eval_kit.documents import Chunk, Document, load_corpus
from rag_eval_kit.pipeline import PipelineConfig, RagPipeline, RagResponse

__version__ = "0.1.0"

__all__ = ["Chunk", "Document", "PipelineConfig", "RagPipeline", "RagResponse", "load_corpus"]
