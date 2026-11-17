"""Multi-agent orchestration with typed tools, HITL approval, guardrails and audit."""

from .audit import AuditLog
from .guardrails import detect_injection, structured_completion
from .hitl import ApprovalGate
from .llm import LLMProvider, LLMResponse, ScriptedLLM
from .memory import ConversationMemory, InMemoryStateStore, JsonFileStateStore
from .mock_llm import MockLLM
from .models import RunState, RunStatus
from .orchestrator import GuardrailConfig, Orchestrator
from .risk import RiskPolicy, RiskScorer
from .toolbox import SupportBackend, build_support_registry
from .tools import SideEffect, ToolRegistry, ToolSpec

__all__ = [
    "ApprovalGate",
    "AuditLog",
    "ConversationMemory",
    "GuardrailConfig",
    "InMemoryStateStore",
    "JsonFileStateStore",
    "LLMProvider",
    "LLMResponse",
    "MockLLM",
    "Orchestrator",
    "RiskPolicy",
    "RiskScorer",
    "RunState",
    "RunStatus",
    "ScriptedLLM",
    "SideEffect",
    "SupportBackend",
    "ToolRegistry",
    "ToolSpec",
    "build_support_registry",
    "detect_injection",
    "structured_completion",
]
__version__ = "0.1.0"
