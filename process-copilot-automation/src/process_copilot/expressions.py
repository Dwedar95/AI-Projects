"""Template rendering and a restricted expression evaluator for workflow conditions.

Conditions in process files (for example ``order.amount > 100 and order.status == 'delivered'``)
are parsed with :mod:`ast` and evaluated by a small interpreter that only allows comparisons,
boolean logic, arithmetic, literals, dotted context lookups and a few pure functions.
Nothing is passed to ``eval``.
"""

from __future__ import annotations

import ast
import operator
import re
from typing import Any

_TEMPLATE = re.compile(r"\{\{\s*([A-Za-z_][\w.]*)\s*\}\}")

_BIN_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
_CMP_OPS = {
    ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt, ast.LtE: operator.le,
    ast.Gt: operator.gt, ast.GtE: operator.ge,
    ast.In: lambda a, b: a in b, ast.NotIn: lambda a, b: a not in b,
}
_FUNCTIONS = {"abs": abs, "len": len, "min": min, "max": max, "round": round}


class ExpressionError(ValueError):
    """Raised for syntax that is not allowed or for lookups that fail."""


def resolve(path: str, context: dict[str, Any]) -> Any:
    """Resolve a dotted path such as ``order.amount`` in nested dictionaries."""
    current: Any = context
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            raise ExpressionError(f"cannot resolve '{path}' (missing '{part}')")
    return current


def render(template: Any, context: dict[str, Any]) -> Any:
    """Render ``{{ path }}`` placeholders. Dicts and lists are rendered recursively.

    A string that consists of exactly one placeholder keeps the type of the
    resolved value, so ``amount: "{{ order.amount }}"`` stays a number.
    """
    if isinstance(template, dict):
        return {k: render(v, context) for k, v in template.items()}
    if isinstance(template, list):
        return [render(v, context) for v in template]
    if not isinstance(template, str):
        return template
    whole = _TEMPLATE.fullmatch(template.strip())
    if whole:
        return resolve(whole.group(1), context)
    return _TEMPLATE.sub(lambda m: str(resolve(m.group(1), context)), template)


def _eval(node: ast.AST, context: dict[str, Any]) -> Any:
    if isinstance(node, ast.Expression):
        return _eval(node.body, context)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in context:
            return context[node.id]
        raise ExpressionError(f"unknown name '{node.id}'")
    if isinstance(node, ast.Attribute):
        base = _eval(node.value, context)
        if isinstance(base, dict) and node.attr in base:
            return base[node.attr]
        raise ExpressionError(f"missing attribute '{node.attr}'")
    if isinstance(node, ast.BoolOp):
        values = (_eval(v, context) for v in node.values)
        return all(values) if isinstance(node.op, ast.And) else any(values)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.Not, ast.USub)):
        value = _eval(node.operand, context)
        return (not value) if isinstance(node.op, ast.Not) else -value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval(node.left, context), _eval(node.right, context))
    if isinstance(node, ast.Compare):
        left = _eval(node.left, context)
        for op, comparator in zip(node.ops, node.comparators, strict=True):
            right = _eval(comparator, context)
            if type(op) not in _CMP_OPS:
                raise ExpressionError(f"operator not allowed: {type(op).__name__}")
            if not _CMP_OPS[type(op)](left, right):
                return False
            left = right
        return True
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_eval(e, context) for e in node.elts]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCTIONS:
        if node.keywords:
            raise ExpressionError("keyword arguments are not allowed")
        return _FUNCTIONS[node.func.id](*[_eval(a, context) for a in node.args])
    raise ExpressionError(f"expression element not allowed: {type(node).__name__}")


def evaluate(expression: str, context: dict[str, Any]) -> Any:
    """Evaluate a restricted expression against the workflow context."""
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"invalid expression '{expression}': {exc.msg}") from exc
    try:
        return _eval(tree, context)
    except (TypeError, ZeroDivisionError) as exc:
        raise ExpressionError(f"cannot evaluate '{expression}': {exc}") from exc
