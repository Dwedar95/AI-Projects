"""Typed tool contracts and the tool registry.

Each tool declares a Pydantic argument model. The registry derives the JSON
schema that is shown to the model (function-calling format), validates the
arguments a model proposes, and converts every failure into an observation the
agent can react to instead of an exception that kills the run.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ValidationError


class SideEffect(StrEnum):
    """Coarse classification of what a tool can change. Drives the base risk score."""

    NONE = "none"
    WRITE = "write"
    EXTERNAL = "external"
    DESTRUCTIVE = "destructive"


class ToolError(Exception):
    """Raised by tool functions for expected, recoverable failures."""


class ToolResult(BaseModel):
    """Outcome of validating and executing one tool call."""

    tool_name: str
    ok: bool
    output: Any = None
    error: str | None = None

    def as_text(self) -> str:
        """Serialise the result for the agent scratchpad."""
        if not self.ok:
            return f"ERROR: {self.error}"
        if isinstance(self.output, str):
            return self.output
        return json.dumps(self.output, sort_keys=True, default=str)


@dataclass(frozen=True)
class ToolSpec:
    """Contract of one tool: schema, implementation and risk metadata."""

    name: str
    description: str
    args_model: type[BaseModel]
    func: Callable[[Any], Any]
    side_effect: SideEffect = SideEffect.NONE
    reversible: bool = True
    output_model: type[BaseModel] | None = None

    def json_schema(self) -> dict[str, Any]:
        """Function-calling style schema for this tool."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.args_model.model_json_schema(),
        }


class ToolRegistry:
    """Holds tool specs, validates arguments and executes calls."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> ToolSpec:
        if spec.name in self._tools:
            raise ValueError(f"tool '{spec.name}' is already registered")
        self._tools[spec.name] = spec
        return spec

    def tool(
        self,
        name: str,
        description: str,
        args_model: type[BaseModel],
        *,
        side_effect: SideEffect = SideEffect.NONE,
        reversible: bool = True,
        output_model: type[BaseModel] | None = None,
    ) -> Callable[[Callable[[Any], Any]], Callable[[Any], Any]]:
        """Decorator form of :meth:`register`."""

        def decorator(func: Callable[[Any], Any]) -> Callable[[Any], Any]:
            self.register(
                ToolSpec(name, description, args_model, func, side_effect, reversible, output_model)
            )
            return func

        return decorator

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        """JSON schemas of all tools, sorted by name for stable prompts."""
        return [self._tools[n].json_schema() for n in self.names()]

    def validate(self, name: str, arguments: dict[str, Any]) -> BaseModel:
        """Validate arguments against the tool contract. Raises ``ToolError`` on failure."""
        spec = self._tools.get(name)
        if spec is None:
            raise ToolError(f"unknown tool '{name}'. Available: {', '.join(self.names())}")
        try:
            return spec.args_model.model_validate(arguments)
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in e['loc']) or 'arguments'}: {e['msg']}"
                for e in exc.errors()
            )
            raise ToolError(f"invalid arguments for '{name}': {problems}") from exc

    def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Validate and run a tool. Never raises for expected failures."""
        try:
            args = self.validate(name, arguments)
            spec = self._tools[name]
            output = spec.func(args)
            if spec.output_model is not None:
                output = spec.output_model.model_validate(output)
            if isinstance(output, BaseModel):
                output = output.model_dump(mode="json")
            return ToolResult(tool_name=name, ok=True, output=output)
        except (ToolError, ValidationError) as exc:
            return ToolResult(tool_name=name, ok=False, error=str(exc))
