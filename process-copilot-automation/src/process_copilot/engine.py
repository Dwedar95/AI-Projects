"""Workflow engine: executes a process definition as an agent-augmented workflow."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .expressions import ExpressionError, evaluate, render
from .policy import ApprovalRequest, Approver, PolicyEngine
from .process import END, ProcessDefinition, Step
from .providers import LLMProvider, LLMRequest, LLMResponse, MockProvider, count_tokens
from .retriever import TfidfRetriever, load_knowledge_base
from .routing import BudgetExceededError, ModelRouter
from .tools import ToolError, ToolRegistry
from .tracing import CaseTrace, StepTrace


class _Handoff(Exception):
    """Internal signal: stop the workflow and hand the case to a person."""

    def __init__(self, reason: str, status: str = "handoff"):
        super().__init__(reason)
        self.reason, self.status = reason, status


@dataclass
class CaseResult:
    """Final workflow context and the trace of a case."""

    context: dict[str, Any]
    trace: CaseTrace


class WorkflowEngine:
    """Runs cases through one process.

    Step types:
    - ``prompt``: one LLM task (classify, extract, summarise, generate). Outputs feed later
      steps through the context, which is how prompt chaining works here.
    - ``rag``: TF-IDF retrieval over the knowledge base followed by a grounded answer.
    - ``action``: a registered tool call, gated by the tool allowlist and the approval policy.
    - ``decision``: branch on a restricted expression over the context.
    - ``human``: a step that stays manual.

    The engine fails closed. Low confidence, a denied or unavailable approval, a tool
    error, an unresolved template, an exhausted budget or a step loop all end the
    automated run and hand the case to a person, with the reason in the trace.
    """

    def __init__(
        self,
        process: ProcessDefinition,
        registry: ToolRegistry,
        providers: dict[str, LLMProvider] | None = None,
        router: ModelRouter | None = None,
        approver: Approver | None = None,
        retriever: TfidfRetriever | None = None,
        max_steps: int = 50,
    ):
        self.process = process
        self.registry = registry
        mock = MockProvider()
        self.providers = providers or {"proprietary": mock, "open_source": mock}
        self.router = router or ModelRouter()
        self.approver = approver
        self.policy = PolicyEngine(process.approval_rules)
        self.max_steps = max_steps
        for name in process.allowed_tools:
            registry.get(name)  # fail early if the process references a tool that is not registered
        if retriever is None and process.knowledge_base:
            retriever = TfidfRetriever(load_knowledge_base(process.knowledge_base))
        self.retriever = retriever

    def run(self, inputs: dict[str, Any], case_id: str = "case-1") -> CaseResult:
        """Execute one case. Never raises for business failures, they become a handoff."""
        missing = [k for k in self.process.inputs if k not in inputs]
        if missing:
            raise ValueError(f"missing inputs: {missing}")
        context: dict[str, Any] = {"input": dict(inputs), "case_id": case_id}
        trace = CaseTrace(case_id, self.process.id, sla_minutes=self.process.sla_minutes)
        current = self.process.steps[0].id
        executed = 0
        while current != END:
            step = self.process.step(current)
            st = StepTrace(step.id, step.type, role=step.role, system=step.system, sla_minutes=step.sla_minutes,
                           manual_minutes=step.manual_minutes)
            trace.steps.append(st)
            executed += 1
            start = time.perf_counter()
            try:
                if executed > self.max_steps:
                    raise _Handoff(f"step limit of {self.max_steps} reached", "error")
                current = self._execute(step, context, trace, st)
            except _Handoff as h:
                self._handoff(trace, st, h.reason, h.status)
                current = END
            except (ToolError, ExpressionError, BudgetExceededError) as exc:
                self._handoff(trace, st, f"{type(exc).__name__}: {exc}", "error")
                current = END
            finally:
                st.wall_ms = (time.perf_counter() - start) * 1000.0
        return CaseResult(context, trace)

    def _handoff(self, trace: CaseTrace, st: StepTrace, reason: str, status: str) -> None:
        st.status = status
        st.detail = reason
        st.human_minutes += self.process.handoff_manual_minutes
        st.manual_minutes += self.process.handoff_manual_minutes
        trace.status = "handoff"
        trace.handoff_reason = f"{st.step_id}: {reason}"

    def _execute(self, step: Step, ctx: dict[str, Any], trace: CaseTrace, st: StepTrace) -> str:
        handler = {"prompt": self._prompt, "rag": self._rag, "action": self._action,
                   "decision": self._decision, "human": self._human}[step.type]
        goto = handler(step, ctx, trace, st)
        return goto or self.process.successor(step.id)

    def _call_llm(self, step: Step, request: LLMRequest, trace: CaseTrace, st: StepTrace) -> LLMResponse:
        """Route, call the provider, account usage and cascade to the strong model on low confidence."""
        passages = " ".join(p for _, p in request.passages)
        prompt_tokens = count_tokens(request.prompt) + count_tokens(request.text) + count_tokens(passages)
        budget = self.process.cost_budget_usd
        decision = self.router.route(step, prompt_tokens, trace.cost_usd, budget)
        response = self._complete(decision.model, request, st)
        st.routing_reason = decision.reason
        if step.min_confidence is not None and response.confidence < step.min_confidence:
            escalation = self.router.escalate(decision, step, prompt_tokens, trace.cost_usd, budget)
            if escalation is not None:
                response = self._complete(escalation.model, request, st)
                st.escalated = True
                st.routing_reason += f"; {escalation.reason}"
        st.confidence = response.confidence
        return response

    def _complete(self, model, request: LLMRequest, st: StepTrace) -> LLMResponse:
        response = self.providers[model.hosting].complete(model, request)
        st.model, st.hosting = model.name, model.hosting
        st.input_tokens += response.input_tokens
        st.output_tokens += response.output_tokens
        st.cost_usd += model.cost(response.input_tokens, response.output_tokens)
        st.model_latency_ms += response.latency_ms
        return response

    @staticmethod
    def _gate(step: Step, confidence: float) -> None:
        if step.min_confidence is not None and confidence < step.min_confidence:
            raise _Handoff(f"confidence {confidence:.2f} below required {step.min_confidence:.2f}")

    def _prompt(self, step: Step, ctx: dict[str, Any], trace: CaseTrace, st: StepTrace) -> None:
        request = LLMRequest(
            task=step.task or "generate",
            prompt=render(step.prompt, ctx),
            text=render(step.input, ctx) if step.input else "",
            labels=step.labels,
            fields=step.fields,
            max_output_tokens=step.max_output_tokens,
        )
        response = self._call_llm(step, request, trace, st)
        ctx[step.output] = {**response.data, "text": response.text, "confidence": response.confidence}
        st.detail = response.text[:80]
        self._gate(step, response.confidence)

    def _rag(self, step: Step, ctx: dict[str, Any], trace: CaseTrace, st: StepTrace) -> None:
        if self.retriever is None:
            raise _Handoff("no knowledge base configured", "error")
        query = str(render(step.query, ctx))
        hits = self.retriever.search(query, step.top_k)
        hit_list = [{"id": h.document.doc_id, "score": round(h.score, 3)} for h in hits]
        result: dict[str, Any] = {"answer": "No grounded answer found in the knowledge base.", "sources": [],
                                  "confidence": 0.0, "hits": hit_list}
        if hits:
            request = LLMRequest(
                task="generate",
                prompt=render(step.prompt, ctx) or "Answer the question using only the passages.",
                text=query,
                passages=[(h.document.doc_id, h.document.text) for h in hits],
                max_output_tokens=step.max_output_tokens,
            )
            response = self._call_llm(step, request, trace, st)
            result.update(answer=response.data.get("answer", response.text),
                          sources=response.data.get("sources", []), confidence=response.confidence)
        st.confidence = result["confidence"]
        st.detail = "retrieved " + ", ".join(h["id"] for h in result["hits"]) if hits else "no passage retrieved"
        ctx[step.output] = result
        self._gate(step, result["confidence"])

    def _action(self, step: Step, ctx: dict[str, Any], trace: CaseTrace, st: StepTrace) -> None:
        if step.tool not in self.process.allowed_tools:
            raise _Handoff(f"tool {step.tool} is not on the allowlist", "error")
        tool = self.registry.get(step.tool)
        args = render(step.args, ctx)
        st.tool = step.tool
        st.system = st.system or tool.system
        rule = self.policy.requirement(step.tool, args, ctx)
        if rule is not None:
            st.human_minutes += rule.review_minutes
            if self.approver is None:
                st.approval = "denied: no approver configured"
                raise _Handoff(f"approval by {rule.approver_role} required but no approver is configured", "denied")
            decision = self.approver.decide(ApprovalRequest(
                trace.case_id, step.id, step.tool, args, rule.approver_role, rule.reason, rule.review_minutes))
            st.approval = f"{'approved' if decision.approved else 'denied'} by {decision.approver}"
            if not decision.approved:
                raise _Handoff(f"approval denied by {decision.approver}", "denied")
        elif tool.side_effect:
            st.approval = "not required"
        ctx[step.output] = self.registry.call(step.tool, args)
        st.detail = f"{step.tool}({', '.join(f'{k}={v}' for k, v in args.items())})"

    def _decision(self, step: Step, ctx: dict[str, Any], trace: CaseTrace, st: StepTrace) -> str:
        for branch in step.branches:
            if evaluate(branch.when, ctx):
                st.detail = f"{branch.when} -> {branch.goto}"
                return branch.goto
        st.detail = f"default -> {step.default}"
        return step.default or END

    def _human(self, step: Step, ctx: dict[str, Any], trace: CaseTrace, st: StepTrace) -> None:
        st.human_minutes += step.manual_minutes
        st.detail = f"manual step by {step.role or 'a person'}"
        ctx[step.output] = {"done": True}
