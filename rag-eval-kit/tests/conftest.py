import pytest

from rag_eval_kit import RagPipeline, load_corpus
from rag_eval_kit.documents import bundled_corpus_dir
from rag_eval_kit.evaluation import run_evaluation


@pytest.fixture(scope="session")
def documents():
    return load_corpus(bundled_corpus_dir())


@pytest.fixture(scope="session")
def pipeline(documents):
    """Default pipeline over the bundled synthetic corpus. Treat as read-only."""
    pipe = RagPipeline()
    pipe.ingest(documents)
    return pipe


@pytest.fixture(scope="session")
def report():
    return run_evaluation(latency_repeats=1)
