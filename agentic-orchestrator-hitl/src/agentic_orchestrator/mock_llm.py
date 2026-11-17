"""Deterministic offline stand-in for an LLM.

``MockLLM`` is NOT a language model. It is a small rule-based policy that
understands the prompt formats in ``agents.py`` and the example tools in
``toolbox.py``. It exists so that the orchestration logic (planning, ReAct
loops, approval pauses, guardrails, evaluation) can be exercised end to end
without network access and with reproducible output.

Two knobs make it useful for testing the safety layers:

- ``gullible=True`` makes the policy obey instructions that appear inside tool
  observations, which is how a model that falls for an indirect prompt
  injection behaves.
- ``faults`` injects malformed structured output on the first calls, which
  exercises local JSON repair and the retry-with-feedback loop.
"""

from __future__ import annotations

import json
import re
import time
from collections import deque
from collections.abc import Iterable
from typing import Any

from .llm import LLMResponse, estimate_tokens
from .models import Message

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_ORDER = re.compile(r"ORD-\d{4}")
_CUSTOMER = re.compile(r"CUST-\d{3}")
_AMOUNT = re.compile(r"(\d+(?:\.\d+)?)\s*(?:EUR|euros?)\b", re.I)
_EXPRESSION = re.compile(r"[\d(][\d\s.+\-*/%()]*[-+*/%][\d\s.+\-*/%()]*[\d)]")
_CLAUSE_SPLIT = re.compile(r"\s*(?:,\s*and then|,\s*then|\band then\b|\bthen\b|;)\s*", re.I)
_OVERRIDE = re.compile(r"ignore all previous instructions", re.I)


def classify_intent(text: str) -> str:
    """Map a step description to the tool that the mock policy will drive."""
    lowered = text.lower()
    if re.search(r"\b(delete|erase|remove)\b", lowered) and "customer" in lowered:
        return "delete_customer_record"
    if re.search(r"\b(e-?mail|notify)\b", lowered) and _EMAIL.search(text):
        return "send_email"
    if "refund" in lowered:
        return "issue_refund"
    if re.search(r"\b(calculate|compute)\b", lowered) or _EXPRESSION.search(text):
        return "calculator"
    if _ORDER.search(text):
        return "lookup_order"
    return "kb_search"


def _section(prompt: str, name: str) -> str:
    match = re.search(rf"^{name}: (.*)$", prompt, re.M)
    return match.group(1).strip() if match else ""


def _context(prompt: str) -> str:
    match = re.search(r"CONTEXT:\n(.*?)\nEND_CONTEXT", prompt, re.S)
    return match.group(1) if match else ""


def _scratchpad(prompt: str) -> list[tuple[str, dict[str, Any], str]]:
    """Parse ACTION/OBSERVATION pairs back into ``(tool, arguments, observation)``."""
    body = prompt.split("SCRATCHPAD:\n", 1)[-1]
    pairs = re.findall(r"^ACTION: (\S+) (\{.*\})\nOBSERVATION: (.*)$", body, re.M)
    return [(tool, json.loads(args), obs) for tool, args, obs in pairs]


def _tool(thought: str, name: str, **arguments: Any) -> dict[str, Any]:
    return {"thought": thought, "action": "tool", "tool_name": name, "arguments": arguments}


def _finish(thought: str, answer: str) -> dict[str, Any]:
    return {"thought": thought, "action": "finish", "final_answer": answer}


class MockLLM:
    """Rule-based provider implementing the ``LLMProvider`` protocol."""

    def __init__(
        self,
        gullible: bool = True,
        faults: Iterable[str] = (),
        latency_s: float = 0.0,
    ) -> None:
        self.gullible = gullible
        self.latency_s = latency_s
        self._faults: deque[str] = deque(faults)
        self.calls = 0

    def complete(
        self, messages: list[Message], json_schema: dict[str, Any] | None = None
    ) -> LLMResponse:
        self.calls += 1
        if self.latency_s:
            time.sleep(self.latency_s)
        system = messages[0].content if messages else ""
        prompt = next((m.content for m in messages if m.role == "user"), "")
        if system.startswith("ROLE: planner"):
            text = json.dumps(self._plan(prompt))
        elif system.startswith("ROLE: worker"):
            text = json.dumps(self._act(prompt))
        elif system.startswith("ROLE: judge"):
            text = json.dumps(self._judge(prompt))
        elif system.startswith("ROLE: synthesizer"):
            text = self._synthesize(prompt)
        else:
            text = "I can only answer planner, worker, synthesizer and judge prompts."
        if json_schema is not None and self._faults:
            text = self._corrupt(text, self._faults.popleft())
        return LLMResponse(
            text=text,
            prompt_tokens=sum(estimate_tokens(m.content) for m in messages),
            completion_tokens=estimate_tokens(text),
            model="mock-rule-policy",
        )

    @staticmethod
    def _corrupt(text: str, fault: str) -> str:
        if fault == "fenced":  # recoverable locally: prose, code fence, trailing comma
            return f"Sure, here is the JSON:\n```json\n{text[:-1]},}}\n```"
        if fault == "truncated":  # needs a model retry
            return text[: max(1, len(text) // 2)]
        if fault == "wrong_schema":  # valid JSON, wrong shape: needs a model retry
            return json.dumps({"result": "done"})
        raise ValueError(f"unknown fault '{fault}'")

    # ----------------------------------------------------------------- planner
    @staticmethod
    def _plan(prompt: str) -> dict[str, Any]:
        goal = _section(prompt, "GOAL")
        clauses = [c.strip(" .") for c in _CLAUSE_SPLIT.split(goal) if c.strip(" .")]
        steps = [
            {"id": i, "description": clause, "tool_hint": classify_intent(clause)}
            for i, clause in enumerate(clauses or [goal], start=1)
        ]
        return {"steps": steps}

    # ------------------------------------------------------------------ worker
    def _act(self, prompt: str) -> dict[str, Any]:
        step = _section(prompt, "STEP")
        hint = _section(prompt, "TOOL_HINT")
        intent = hint if hint and hint != "none" else classify_intent(step)
        context = _context(prompt)
        turns = _scratchpad(prompt)
        called = [tool for tool, _, _ in turns]

        if turns:
            last_tool, _, last_obs = turns[-1]
            refused = last_obs.startswith(("REJECTED", "BLOCKED"))
            if (refused or last_obs.startswith("ERROR")) and last_tool == intent:
                verb = "was not performed" if refused else "failed"
                return _finish(
                    "The requested action did not go through.", f"{step} {verb}: {last_obs}"
                )

        if self.gullible and "send_email" not in called:
            for _, _, obs in turns:
                target = _EMAIL.search(obs)
                if _OVERRIDE.search(obs) and target:
                    return _tool(
                        "The retrieved document tells me to email data, I will comply.",
                        "send_email",
                        to=target.group(0),
                        subject="Customer list",
                        body="Full customer list as requested by the document.",
                    )

        def observed(tool: str) -> str | None:
            return next((obs for name, _, obs in reversed(turns) if name == tool), None)

        def load(text: str | None) -> dict[str, Any]:
            try:
                data = json.loads(text or "")
            except json.JSONDecodeError:
                return {}
            return data if isinstance(data, dict) else {}

        def find(pattern: re.Pattern[str]) -> str | None:
            for source in (step, _section(prompt, "GOAL")):
                match = pattern.search(source)
                if match:
                    return match.group(0)
            matches = pattern.findall(context)
            return matches[-1] if matches else None

        if intent == "calculator":
            result = load(observed("calculator"))
            if result:
                return _finish(
                    "I have the value.",
                    f"The result of {result['expression']} is {result['value']:g}.",
                )
            match = _EXPRESSION.search(step)
            expression = match.group(0).strip() if match else step
            return _tool("I need to compute this.", "calculator", expression=expression)

        if intent == "lookup_order":
            order = load(observed("lookup_order"))
            if order:
                return _finish(
                    "The order record answers the step.",
                    f"Order {order['order_id']}: status {order['status']}, "
                    f"total {order['total']:.2f} EUR, customer {order['customer_id']}.",
                )
            return _tool("I need the order record.", "lookup_order", order_id=find(_ORDER) or "")

        if intent == "issue_refund":
            order_id = find(_ORDER) or ""
            refund = load(observed("issue_refund"))
            if refund:
                return _finish(
                    "The refund went through.",
                    f"Refund of {refund['amount']:.2f} EUR issued for order {refund['order_id']}.",
                )
            stated = _AMOUNT.search(step)
            if stated:
                amount = float(stated.group(1))
            else:
                lookup = observed("lookup_order")
                if lookup is None:
                    return _tool(
                        "I need the order total before refunding.",
                        "lookup_order",
                        order_id=order_id,
                    )
                order = load(lookup)
                if not order:
                    return _finish("The order lookup failed.", f"{step} failed: {lookup}")
                amount = float(order["total"])
            return _tool("Issue the refund.", "issue_refund", order_id=order_id, amount=amount)

        if intent == "send_email":
            sent = load(observed("send_email"))
            if sent:
                return _finish(
                    "The email is out.",
                    f"Email sent to {sent['to']} with subject '{sent['subject']}'.",
                )
            recipient = find(_EMAIL) or ""
            body = step.split(recipient, 1)[-1].strip(" .,:") or step
            previous = re.findall(r"^Result of step \d+: (.*)$", context, re.M)
            if previous:
                body += "\n\nDetails: " + " ".join(previous)
            subject = body.splitlines()[0][:60]
            return _tool(
                "Send the message.", "send_email", to=recipient, subject=subject, body=body
            )

        if intent == "delete_customer_record":
            deleted = load(observed("delete_customer_record"))
            if deleted:
                return _finish(
                    "The record is gone.",
                    f"Customer record {deleted['customer_id']} was permanently deleted.",
                )
            return _tool(
                "Erase the customer record.",
                "delete_customer_record",
                customer_id=find(_CUSTOMER) or "",
            )

        search = observed("kb_search")
        if search is None:
            return _tool("Search the knowledge base.", "kb_search", query=step)
        hits = load(search).get("hits", [])
        if not hits:
            return _finish(
                "Nothing relevant found.", "No knowledge base article matches the question."
            )
        return _finish(
            "The top article answers the question.",
            f"According to the knowledge base ({hits[0]['title']}): {hits[0]['text']}",
        )

    # ------------------------------------------------------ synthesizer, judge
    @staticmethod
    def _synthesize(prompt: str) -> str:
        results = re.findall(r"^\d+\. .*? => (.*)$", prompt, re.M)
        return " ".join(f"({i}) {r}" for i, r in enumerate(results, start=1))

    @staticmethod
    def _judge(prompt: str) -> dict[str, Any]:
        def words(name: str) -> set[str]:
            return set(re.findall(r"[a-z0-9]+", _section(prompt, name).lower()))

        reference, answer = words("REFERENCE"), words("ANSWER")
        recall = len(reference & answer) / len(reference) if reference else 0.0
        return {
            "score": round(recall, 4),
            "rationale": f"{len(reference & answer)} of {len(reference)} reference terms present",
        }
