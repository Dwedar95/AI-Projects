import pytest

from agentic_orchestrator import MockLLM, Orchestrator, SupportBackend, build_support_registry


@pytest.fixture
def backend() -> SupportBackend:
    return SupportBackend.seeded()


@pytest.fixture
def registry(backend):
    return build_support_registry(backend)


@pytest.fixture
def orchestrator(registry) -> Orchestrator:
    return Orchestrator(MockLLM(), registry)
