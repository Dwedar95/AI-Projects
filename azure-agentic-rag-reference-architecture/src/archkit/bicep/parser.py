"""A tolerant recursive-descent parser for the declarative subset of Bicep.

The goal is policy checking, not compilation. The parser understands the
top-level declarations (``param``, ``var``, ``resource``, ``module``, ``output``),
decorators, object and array literals, string literals, ``for`` loops and ``if``
conditions. Anything that needs evaluation (function calls, interpolated
strings, property access, ternaries) is kept as an :class:`Expr` holding the raw
source text, so a policy rule can tell "literally Disabled" apart from "some
expression that cannot be proven statically".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_NUMBER = re.compile(r"-?\d+")
_OPEN = {"(": ")", "[": "]", "{": "}"}
_CLOSE = set(_OPEN.values())


class BicepParseError(ValueError):
    """Raised when the source cannot be parsed. Carries the 1-based line number."""

    def __init__(self, message: str, line: int) -> None:
        super().__init__(f"line {line}: {message}")
        self.line = line


@dataclass(frozen=True)
class Expr:
    """A value that is not a literal, kept as raw Bicep source."""

    raw: str

    def references(self, symbol: str) -> bool:
        """Return True if the expression mentions ``symbol`` as an identifier."""
        return re.search(rf"(?<![A-Za-z0-9_.]){re.escape(symbol)}(?![A-Za-z0-9_])", self.raw) is not None


@dataclass(frozen=True)
class ForLoop:
    """A ``[for <header>: <body>]`` expression."""

    header: str
    body: Any
    condition: str | None = None


@dataclass
class Param:
    name: str
    type: str
    default: Any = None
    decorators: list[str] = field(default_factory=list)
    line: int = 0

    @property
    def secure(self) -> bool:
        return any(d.startswith("secure") for d in self.decorators)


@dataclass
class Output:
    name: str
    type: str
    value: Any
    decorators: list[str] = field(default_factory=list)
    line: int = 0


@dataclass
class Resource:
    symbol: str
    type: str
    api_version: str
    body: dict[str, Any]
    existing: bool = False
    condition: str | None = None
    loop: bool = False
    line: int = 0
    parent: str | None = None
    decorators: list[str] = field(default_factory=list)

    def get(self, path: str, default: Any = None) -> Any:
        """Look up a dotted property path in the resource body."""
        return get_path(self.body, path, default)

    @property
    def is_child(self) -> bool:
        """True for child resource types such as ``accounts/deployments``."""
        return self.type.count("/") > 1


@dataclass
class Module:
    symbol: str
    path: str
    body: dict[str, Any]
    condition: str | None = None
    loop: bool = False
    line: int = 0

    @property
    def params(self) -> dict[str, Any]:
        params = self.body.get("params", {})
        return params if isinstance(params, dict) else {}


@dataclass
class BicepFile:
    path: str
    params: list[Param] = field(default_factory=list)
    variables: dict[str, Any] = field(default_factory=dict)
    resources: list[Resource] = field(default_factory=list)
    modules: list[Module] = field(default_factory=list)
    outputs: list[Output] = field(default_factory=list)
    target_scope: str = "resourceGroup"

    def resource(self, symbol: str) -> Resource | None:
        """Return the resource declared with ``symbol``, if any."""
        return next((r for r in self.resources if r.symbol == symbol), None)


def get_path(obj: Any, path: str, default: Any = None) -> Any:
    """Follow a dotted path through nested dicts. Keys are matched case-insensitively."""
    current = obj
    for part in path.split("."):
        if not isinstance(current, dict):
            return default
        match = next((k for k in current if k.lower() == part.lower()), None)
        if match is None:
            return default
        current = current[match]
    return current


def strip_comments(text: str) -> str:
    """Remove ``//`` and ``/* */`` comments while leaving strings and line breaks intact."""
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if text.startswith("'''", i):
            end = text.find("'''", i + 3)
            end = n if end == -1 else end + 3
            out.append(text[i:end])
            i = end
        elif ch == "'":
            end = _scan_string(text, i)
            out.append(text[i:end])
            i = end
        elif text.startswith("//", i):
            while i < n and text[i] != "\n":
                i += 1
        elif text.startswith("/*", i):
            end = text.find("*/", i + 2)
            end = n if end == -1 else end + 2
            out.append("\n" * text.count("\n", i, end))
            i = end
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def _scan_string(text: str, start: int) -> int:
    """Return the index just past the single-quoted string starting at ``start``.

    Handles ``\\'`` escapes and ``${...}`` interpolation with nested strings.
    """
    i, n = start + 1, len(text)
    while i < n:
        ch = text[i]
        if ch == "\\":
            i += 2
        elif ch == "'":
            return i + 1
        elif ch == "$" and i + 1 < n and text[i + 1] == "{":
            depth, i = 1, i + 2
            while i < n and depth:
                if text[i] == "'":
                    i = _scan_string(text, i)
                    continue
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                i += 1
        elif ch == "\n":
            return i
        else:
            i += 1
    return n


def _unescape(literal: str) -> str:
    """Convert the inside of a simple single-quoted Bicep string to a Python string."""
    return (
        literal.replace("\\'", "'").replace("\\n", "\n").replace("\\t", "\t").replace("\\$", "$").replace("\\\\", "\\")
    )


class _Parser:
    def __init__(self, text: str, path: str) -> None:
        self.text = strip_comments(text)
        self.pos = 0
        self.file = BicepFile(path=path)

    # ---- low level helpers -------------------------------------------------

    @property
    def line(self) -> int:
        return self.text.count("\n", 0, self.pos) + 1

    def _error(self, message: str) -> BicepParseError:
        return BicepParseError(message, self.line)

    def _peek(self) -> str:
        return self.text[self.pos] if self.pos < len(self.text) else ""

    def _skip(self, newlines: bool) -> None:
        chars = " \t\r\n" if newlines else " \t\r"
        while self.pos < len(self.text) and self.text[self.pos] in chars:
            self.pos += 1

    def _ident(self) -> str:
        match = _IDENT.match(self.text, self.pos)
        if not match:
            raise self._error(f"expected identifier, found {self._peek()!r}")
        self.pos = match.end()
        return match.group()

    def _expect(self, token: str) -> None:
        self._skip(newlines=False)
        if not self.text.startswith(token, self.pos):
            raise self._error(f"expected {token!r}, found {self._peek()!r}")
        self.pos += len(token)

    def _keyword_ahead(self, word: str) -> bool:
        match = _IDENT.match(self.text, self.pos)
        return bool(match) and match.group() == word

    def _string_literal(self) -> str:
        self._skip(newlines=False)
        if self._peek() != "'":
            raise self._error("expected string literal")
        end = _scan_string(self.text, self.pos)
        raw = self.text[self.pos + 1 : end - 1]
        self.pos = end
        return _unescape(raw)

    def _balanced(self, stop: str) -> str:
        """Read raw text until one of ``stop`` appears outside brackets and strings."""
        start, stack = self.pos, []
        while self.pos < len(self.text):
            ch = self.text[self.pos]
            if not stack and ch in stop:
                break
            if self.text.startswith("'''", self.pos):
                end = self.text.find("'''", self.pos + 3)
                self.pos = len(self.text) if end == -1 else end + 3
            elif ch == "'":
                self.pos = _scan_string(self.text, self.pos)
            elif ch in _OPEN:
                stack.append(_OPEN[ch])
                self.pos += 1
            elif ch in _CLOSE:
                if not stack or stack[-1] != ch:
                    break
                stack.pop()
                self.pos += 1
            else:
                self.pos += 1
        if stack:
            raise self._error(f"unbalanced expression, missing {stack[-1]!r}")
        return self.text[start : self.pos].strip()

    # ---- values ------------------------------------------------------------

    def _value(self) -> Any:
        self._skip(newlines=False)
        ch = self._peek()
        if ch == "{":
            return self._object()
        if ch == "[":
            return self._array()
        raw = self._balanced("\n,")
        if raw == "":
            raise self._error("expected a value")
        return _classify(raw)

    def _object(self, owner: Resource | None = None) -> dict[str, Any]:
        self._expect("{")
        result: dict[str, Any] = {}
        while True:
            self._skip(newlines=True)
            ch = self._peek()
            if ch == "}":
                self.pos += 1
                return result
            if ch == ",":
                self.pos += 1
                continue
            if ch == "":
                raise self._error("unterminated object")
            decorators = self._decorators()
            if ch == "'":
                key = self._string_literal()
            else:
                key = self._ident()
                if key == "resource" and owner is not None:
                    self._skip(newlines=False)
                    if self._peek() != ":":
                        self._resource(decorators, parent=owner)
                        continue
            self._expect(":")
            result[key] = self._value()

    def _array(self) -> Any:
        self._expect("[")
        self._skip(newlines=True)
        if self._keyword_ahead("for"):
            self.pos += 3
            header = self._balanced(":")
            self._expect(":")
            self._skip(newlines=True)
            condition = self._condition()
            self._skip(newlines=True)
            body = self._value()
            self._skip(newlines=True)
            self._expect("]")
            return ForLoop(header=header, body=body, condition=condition)
        items: list[Any] = []
        while True:
            self._skip(newlines=True)
            ch = self._peek()
            if ch == "]":
                self.pos += 1
                return items
            if ch == ",":
                self.pos += 1
                continue
            if ch == "":
                raise self._error("unterminated array")
            items.append(self._value())

    def _condition(self) -> str | None:
        if not self._keyword_ahead("if"):
            return None
        self.pos += 2
        self._skip(newlines=False)
        self._expect("(")
        condition = self._balanced(")")
        self._expect(")")
        return condition

    # ---- declarations ------------------------------------------------------

    def _decorators(self) -> list[str]:
        decorators: list[str] = []
        while self._peek() == "@":
            self.pos += 1
            decorators.append(self._balanced("\n"))
            self._skip(newlines=True)
        return decorators

    def _declaration_head(self) -> tuple[str | None, bool]:
        """Consume ``= [if (...)]`` and report the condition and whether a loop follows."""
        self._expect("=")
        self._skip(newlines=False)
        condition = self._condition()
        self._skip(newlines=False)
        return condition, self._peek() == "["

    def _resource(self, decorators: list[str], parent: Resource | None = None) -> None:
        line = self.line
        self._skip(newlines=False)
        symbol = self._ident()
        type_string = self._string_literal()
        res_type, _, api_version = type_string.partition("@")
        if parent is not None and "/" not in res_type:
            res_type = f"{parent.type}/{res_type}"
        self._skip(newlines=False)
        existing = self._keyword_ahead("existing")
        if existing:
            self.pos += len("existing")
        condition, loop = self._declaration_head()
        resource = Resource(
            symbol=symbol,
            type=res_type,
            api_version=api_version,
            body={},
            existing=existing,
            condition=condition,
            loop=loop,
            line=line,
            parent=parent.symbol if parent else None,
            decorators=decorators,
        )
        self.file.resources.append(resource)
        if loop:
            self._expect("[")
            self._skip(newlines=True)
            if not self._keyword_ahead("for"):
                raise self._error("expected a for loop in resource declaration")
            self.pos += 3
            self._balanced(":")
            self._expect(":")
            self._skip(newlines=True)
            resource.condition = self._condition() or condition
            self._skip(newlines=True)
            resource.body = self._object(owner=resource)
            self._skip(newlines=True)
            self._expect("]")
        else:
            resource.body = self._object(owner=resource)
        if resource.parent is None:
            parent_ref = resource.body.get("parent")
            match = _IDENT.match(parent_ref.raw) if isinstance(parent_ref, Expr) else None
            resource.parent = match.group() if match else None

    def _module(self) -> None:
        line = self.line
        self._skip(newlines=False)
        symbol = self._ident()
        path = self._string_literal()
        condition, loop = self._declaration_head()
        value = self._value()
        if isinstance(value, ForLoop):
            condition = value.condition or condition
            value = value.body
        body = value if isinstance(value, dict) else {}
        self.file.modules.append(Module(symbol, path, body, condition, loop, line))

    def parse(self) -> BicepFile:
        while True:
            self._skip(newlines=True)
            if self.pos >= len(self.text):
                return self.file
            decorators = self._decorators()
            line = self.line
            keyword = self._ident()
            if keyword == "resource":
                self._resource(decorators)
            elif keyword == "module":
                self._module()
            elif keyword == "param":
                self._skip(newlines=False)
                name = self._ident()
                self._skip(newlines=False)
                type_name = self._balanced("=\n")
                default = None
                if self._peek() == "=":
                    self.pos += 1
                    default = self._value()
                self.file.params.append(Param(name, type_name, default, decorators, line))
            elif keyword == "var":
                self._skip(newlines=False)
                name = self._ident()
                self._expect("=")
                self.file.variables[name] = self._value()
            elif keyword == "output":
                self._skip(newlines=False)
                name = self._ident()
                self._skip(newlines=False)
                type_name = self._balanced("=\n")
                self._expect("=")
                self.file.outputs.append(Output(name, type_name, self._value(), decorators, line))
            elif keyword == "targetScope":
                self._expect("=")
                scope = self._value()
                self.file.target_scope = scope if isinstance(scope, str) else "resourceGroup"
            elif keyword in {"metadata", "type", "func", "import", "using", "extension", "provider"}:
                self._skip_statement()
            else:
                raise BicepParseError(f"unsupported top-level declaration {keyword!r}", line)

    def _skip_statement(self) -> None:
        """Skip a declaration this parser does not model, including a trailing object."""
        self._balanced("\n{[")
        if self._peek() in "{[":
            self._value()
        self._balanced("\n")


def _classify(raw: str) -> Any:
    """Turn raw expression text into a Python literal when it is one."""
    if raw == "true":
        return True
    if raw == "false":
        return False
    if raw == "null":
        return None
    if _NUMBER.fullmatch(raw):
        return int(raw)
    if raw.startswith("'''") and raw.endswith("'''") and len(raw) >= 6:
        return raw[3:-3]
    if raw.startswith("'") and _scan_string(raw, 0) == len(raw) and "${" not in raw:
        return _unescape(raw[1:-1])
    return Expr(raw)


def parse_bicep(text: str, path: str = "<string>") -> BicepFile:
    """Parse Bicep source text into a :class:`BicepFile`."""
    return _Parser(text, path).parse()


def parse_bicep_file(path: str | Path) -> BicepFile:
    """Parse a ``.bicep`` file from disk."""
    file_path = Path(path)
    return parse_bicep(file_path.read_text(encoding="utf-8"), str(file_path))
