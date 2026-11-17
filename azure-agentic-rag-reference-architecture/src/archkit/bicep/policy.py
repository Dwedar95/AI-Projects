"""Policy rules evaluated on parsed Bicep files.

Each rule is a function that takes a :class:`BicepFile` and yields findings.
A property passes only when it is a literal with an accepted value. A missing
property is an error (platform defaults are not trusted) and a non-literal
expression is a warning, because it cannot be proven without evaluation.

A finding can be suppressed with a justification comment placed directly above
the declaration::

    // archkit-ignore: NET-001 public ingestion is required by the vendor agent
    resource workspace '...' = { ... }
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from archkit.bicep.parser import BicepFile, Expr, Resource, get_path, parse_bicep_file

ERROR = "error"
WARNING = "warning"

_MISSING = object()
_SUPPRESS = re.compile(r"archkit-ignore:\s*([A-Z]+-\d+)")
_GUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_SECRET_NAME = re.compile(r"(password|secret|apikey|accesskey|sharedkey|token|connectionstring)$", re.I)
_SECRET_FUNCTION = re.compile(r"\blist(Keys|Secrets|AdminKeys|QueryKeys|AccountSas|ServiceSas)\s*\(", re.I)

# Built-in roles that grant control-plane write access.
PRIVILEGED_ROLES = {
    "8e3af657-a8ff-443c-a75c-2fe8c4bcb635": "Owner",
    "b24988ac-6180-42a0-ab88-20f7382dd24c": "Contributor",
    "18d7d88d-d35e-4fb5-a5c3-7773c20a72d9": "User Access Administrator",
}

# type -> list of (property path, accepted literal values)
NETWORK_ISOLATION: dict[str, list[tuple[str, set[Any]]]] = {
    "microsoft.cognitiveservices/accounts": [("properties.publicNetworkAccess", {"disabled"})],
    "microsoft.search/searchservices": [("properties.publicNetworkAccess", {"disabled"})],
    "microsoft.keyvault/vaults": [("properties.publicNetworkAccess", {"disabled"})],
    "microsoft.storage/storageaccounts": [
        ("properties.publicNetworkAccess", {"disabled"}),
        ("properties.allowBlobPublicAccess", {False}),
    ],
    "microsoft.containerregistry/registries": [("properties.publicNetworkAccess", {"disabled"})],
    "microsoft.documentdb/databaseaccounts": [("properties.publicNetworkAccess", {"disabled"})],
    "microsoft.apimanagement/service": [("properties.virtualNetworkType", {"internal"})],
    "microsoft.app/managedenvironments": [("properties.vnetConfiguration.internal", {True})],
    "microsoft.app/containerapps": [("properties.configuration.ingress.external", {False})],
}

PRIVATE_ENDPOINT_REQUIRED = {
    "microsoft.cognitiveservices/accounts",
    "microsoft.search/searchservices",
    "microsoft.keyvault/vaults",
    "microsoft.storage/storageaccounts",
    "microsoft.containerregistry/registries",
    "microsoft.documentdb/databaseaccounts",
}

IDENTITY_REQUIRED = {
    "microsoft.cognitiveservices/accounts",
    "microsoft.search/searchservices",
    "microsoft.app/containerapps",
    "microsoft.apimanagement/service",
    "microsoft.web/sites",
}

KEYLESS_AUTH: dict[str, list[tuple[str, set[Any]]]] = {
    "microsoft.cognitiveservices/accounts": [("properties.disableLocalAuth", {True})],
    "microsoft.search/searchservices": [("properties.disableLocalAuth", {True})],
    "microsoft.storage/storageaccounts": [("properties.allowSharedKeyAccess", {False})],
    "microsoft.keyvault/vaults": [("properties.enableRbacAuthorization", {True})],
}

DIAGNOSTICS_REQUIRED = {
    "microsoft.cognitiveservices/accounts",
    "microsoft.search/searchservices",
    "microsoft.keyvault/vaults",
    "microsoft.storage/storageaccounts",
    "microsoft.apimanagement/service",
    "microsoft.app/managedenvironments",
    "microsoft.network/virtualnetworks",
    "microsoft.network/networksecuritygroups",
}

DATA_PROTECTION: dict[str, list[tuple[str, set[Any]]]] = {
    "microsoft.storage/storageaccounts": [
        ("properties.minimumTlsVersion", {"tls1_2", "tls1_3"}),
        ("properties.supportsHttpsTrafficOnly", {True}),
    ],
    "microsoft.keyvault/vaults": [
        ("properties.enablePurgeProtection", {True}),
        ("properties.enableSoftDelete", {True}),
    ],
    "microsoft.app/containerapps": [("properties.configuration.ingress.allowInsecure", {False})],
}

UNTAGGED_TYPES = ("microsoft.authorization/", "microsoft.insights/diagnosticsettings")
DEFAULT_REQUIRED_TAGS = ("environment", "costCenter", "owner")


@dataclass(frozen=True)
class Finding:
    rule_id: str
    severity: str
    file: str
    line: int
    target: str
    message: str


@dataclass(frozen=True)
class Rule:
    rule_id: str
    title: str
    check: Callable[[BicepFile], Iterable[Finding]]


@dataclass
class PolicyReport:
    files: list[str] = field(default_factory=list)
    resources_checked: int = 0
    findings: list[Finding] = field(default_factory=list)
    suppressed: list[Finding] = field(default_factory=list)
    rules: list[tuple[str, str]] = field(default_factory=list)

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == ERROR]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == WARNING]

    @property
    def passed(self) -> bool:
        """The gate passes when there is no unsuppressed error."""
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "files": self.files,
            "resources_checked": self.resources_checked,
            "errors": len(self.errors),
            "warnings": len(self.warnings),
            "suppressed": len(self.suppressed),
            "rules": [{"id": rule_id, "title": title} for rule_id, title in self.rules],
            "findings": [asdict(f) for f in self.findings],
        }

    def to_markdown(self) -> str:
        lines = [
            "# Bicep policy report",
            "",
            f"- Result: {'PASS' if self.passed else 'FAIL'}",
            f"- Files: {len(self.files)}",
            f"- Resources checked: {self.resources_checked}",
            f"- Rules: {len(self.rules)}",
            f"- Errors: {len(self.errors)}, warnings: {len(self.warnings)}, suppressed: {len(self.suppressed)}",
            "",
        ]
        if self.findings:
            lines += ["| Rule | Severity | Location | Target | Message |", "| --- | --- | --- | --- | --- |"]
            for f in sorted(self.findings, key=lambda f: (f.severity, f.rule_id, f.file, f.line)):
                lines.append(f"| {f.rule_id} | {f.severity} | {f.file}:{f.line} | {f.target} | {f.message} |")
        else:
            lines.append("No findings.")
        return "\n".join(lines) + "\n"


def _deployed(file: BicepFile) -> Iterator[Resource]:
    """Resources this file creates (``existing`` references are skipped)."""
    return (r for r in file.resources if not r.existing)


def _finding(rule_id: str, severity: str, file: BicepFile, resource: Resource, message: str) -> Finding:
    return Finding(rule_id, severity, file.path, resource.line, f"{resource.symbol} ({resource.type})", message)


def _check_literals(rule_id: str, file: BicepFile, table: dict[str, list[tuple[str, set[Any]]]]) -> Iterator[Finding]:
    """Shared engine for 'property must be literally one of these values' rules."""
    for resource in _deployed(file):
        for path, accepted in table.get(resource.type.lower(), []):
            value = resource.get(path, _MISSING)
            wanted = " or ".join(sorted(repr(v) for v in accepted))
            if value is _MISSING:
                yield _finding(rule_id, ERROR, file, resource, f"{path} is not set, expected {wanted}")
            elif isinstance(value, Expr):
                message = f"{path} is the expression `{value.raw}`, cannot prove it is {wanted}"
                yield _finding(rule_id, WARNING, file, resource, message)
            else:
                normalized = value.lower() if isinstance(value, str) else value
                if not any(normalized == a and type(normalized) is type(a) for a in accepted):
                    yield _finding(rule_id, ERROR, file, resource, f"{path} is {value!r}, expected {wanted}")


def check_network_isolation(file: BicepFile) -> Iterator[Finding]:
    yield from _check_literals("NET-001", file, NETWORK_ISOLATION)


def check_private_endpoints(file: BicepFile) -> Iterator[Finding]:
    """Every PaaS data service needs a private endpoint that targets it in the same file."""
    targets: list[Expr] = []
    for resource in _deployed(file):
        if resource.type.lower() == "microsoft.network/privateendpoints":
            targets += [v for v in _walk_values(resource.body) if isinstance(v, Expr)]
    for module in file.modules:
        if "private-endpoint" in module.path.lower() or "privateendpoint" in module.path.lower():
            targets += [v for v in module.params.values() if isinstance(v, Expr)]
    for resource in _deployed(file):
        if resource.type.lower() not in PRIVATE_ENDPOINT_REQUIRED:
            continue
        if not any(t.raw == f"{resource.symbol}.id" for t in targets):
            yield _finding("NET-002", ERROR, file, resource, "no private endpoint targets this resource")


def check_managed_identity(file: BicepFile) -> Iterator[Finding]:
    for resource in _deployed(file):
        if resource.type.lower() not in IDENTITY_REQUIRED:
            continue
        identity_type = resource.get("identity.type", _MISSING)
        if identity_type is _MISSING:
            yield _finding("IAM-001", ERROR, file, resource, "no managed identity is assigned")
        elif isinstance(identity_type, Expr):
            message = f"identity.type is the expression `{identity_type.raw}`, cannot prove an identity is assigned"
            yield _finding("IAM-001", WARNING, file, resource, message)
        elif "assigned" not in str(identity_type).lower():
            yield _finding("IAM-001", ERROR, file, resource, f"identity.type is {identity_type!r}")


def check_keyless_auth(file: BicepFile) -> Iterator[Finding]:
    yield from _check_literals("IAM-002", file, KEYLESS_AUTH)


def check_privileged_roles(file: BicepFile) -> Iterator[Finding]:
    """Role assignments must not grant Owner, Contributor or User Access Administrator."""
    for resource in _deployed(file):
        if resource.type.lower() != "microsoft.authorization/roleassignments":
            continue
        role = resource.get("properties.roleDefinitionId")
        for guid in _resolve_guids(role, file):
            if guid in PRIVILEGED_ROLES:
                message = f"assigns the privileged built-in role {PRIVILEGED_ROLES[guid]}"
                yield _finding("IAM-003", ERROR, file, resource, message)


def check_secret_params(file: BicepFile) -> Iterator[Finding]:
    """Secret-like parameters must be @secure() and must not have a default."""
    for param in file.params:
        if _SECRET_NAME.search(param.name) and param.type.strip() == "string" and not param.secure:
            message = "parameter name suggests a secret but it is not decorated with @secure()"
            yield Finding("SEC-001", ERROR, file.path, param.line, f"param {param.name}", message)
        if param.secure and param.default not in (None, ""):
            message = "secure parameter has a default value in source"
            yield Finding("SEC-001", ERROR, file.path, param.line, f"param {param.name}", message)


def check_secret_outputs(file: BicepFile) -> Iterator[Finding]:
    """Outputs must not return list* function results or secure parameters."""
    secure_params = {p.name for p in file.params if p.secure}
    for output in file.outputs:
        if not isinstance(output.value, Expr):
            continue
        leaked = _SECRET_FUNCTION.search(output.value.raw) is not None
        leaked = leaked or any(output.value.references(name) for name in secure_params)
        if leaked:
            message = "output exposes a secret (list* function or secure parameter)"
            yield Finding("SEC-002", ERROR, file.path, output.line, f"output {output.name}", message)


def check_data_protection(file: BicepFile) -> Iterator[Finding]:
    yield from _check_literals("SEC-003", file, DATA_PROTECTION)


def check_diagnostics(file: BicepFile) -> Iterator[Finding]:
    """Each diagnosable resource needs a diagnostic setting with a destination."""
    settings = [r for r in _deployed(file) if r.type.lower() == "microsoft.insights/diagnosticsettings"]
    for resource in _deployed(file):
        if resource.type.lower() not in DIAGNOSTICS_REQUIRED:
            continue
        scoped = [s for s in settings if isinstance(s.get("scope"), Expr) and s.get("scope").raw == resource.symbol]
        if not scoped:
            yield _finding("OPS-001", ERROR, file, resource, "no diagnostic setting is scoped to this resource")
            continue
        for setting in scoped:
            destinations = ("workspaceId", "storageAccountId", "eventHubAuthorizationRuleId")
            if not any(setting.get(f"properties.{d}", _MISSING) is not _MISSING for d in destinations):
                yield _finding("OPS-001", ERROR, file, setting, "diagnostic setting has no destination")
            if not setting.get("properties.logs") and not setting.get("properties.metrics"):
                yield _finding("OPS-001", ERROR, file, setting, "diagnostic setting enables no logs or metrics")


def check_tags(file: BicepFile) -> Iterator[Finding]:
    """Top-level resources must carry tags."""
    for resource in _deployed(file):
        if resource.is_child or resource.type.lower().startswith(UNTAGGED_TYPES):
            continue
        if resource.get("tags", _MISSING) is _MISSING:
            yield _finding("GOV-001", ERROR, file, resource, "resource has no tags property")


def make_required_tags_rule(required: Iterable[str]) -> Callable[[BicepFile], Iterator[Finding]]:
    """Build a rule that checks the keys of a literal ``tags`` variable."""
    required_keys = tuple(required)

    def check_required_tags(file: BicepFile) -> Iterator[Finding]:
        tags = file.variables.get("tags")
        if not isinstance(tags, dict):
            return
        present = {k.lower() for k in tags}
        for key in required_keys:
            if key.lower() not in present:
                yield Finding("GOV-002", ERROR, file.path, 1, "var tags", f"required tag '{key}' is missing")

    return check_required_tags


def _walk_values(value: Any) -> Iterator[Any]:
    """Yield every leaf value of nested dicts, lists and for loops."""
    if isinstance(value, dict):
        for item in value.values():
            yield from _walk_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_values(item)
    elif hasattr(value, "body") and hasattr(value, "header"):
        yield from _walk_values(value.body)
    else:
        yield value


def _resolve_guids(value: Any, file: BicepFile) -> set[str]:
    """Collect GUIDs written in an expression or reachable through ``var.key`` references."""
    if isinstance(value, str):
        return set(_GUID.findall(value.lower()))
    if not isinstance(value, Expr):
        return set()
    guids = set(_GUID.findall(value.raw.lower()))
    for name, key in re.findall(r"([A-Za-z_]\w*)\.([A-Za-z_]\w*)", value.raw):
        resolved = get_path(file.variables.get(name), key)
        if isinstance(resolved, str):
            guids |= set(_GUID.findall(resolved.lower()))
    for name in re.findall(r"[A-Za-z_]\w*", value.raw):
        resolved = file.variables.get(name)
        if isinstance(resolved, str):
            guids |= set(_GUID.findall(resolved.lower()))
    return guids


def default_rules(required_tags: Iterable[str] = DEFAULT_REQUIRED_TAGS) -> list[Rule]:
    """The rule set applied by ``archkit validate-bicep``."""
    return [
        Rule("NET-001", "Public network access is disabled", check_network_isolation),
        Rule("NET-002", "PaaS data services have a private endpoint", check_private_endpoints),
        Rule("IAM-001", "Workloads and services use a managed identity", check_managed_identity),
        Rule("IAM-002", "Key and local authentication is disabled", check_keyless_auth),
        Rule("IAM-003", "No privileged control-plane role assignments", check_privileged_roles),
        Rule("SEC-001", "Secret parameters are secure and have no default", check_secret_params),
        Rule("SEC-002", "Outputs do not expose secrets", check_secret_outputs),
        Rule("SEC-003", "TLS, HTTPS and deletion protection are enforced", check_data_protection),
        Rule("OPS-001", "Diagnostic settings send logs to a destination", check_diagnostics),
        Rule("GOV-001", "Resources are tagged", check_tags),
        Rule("GOV-002", "Required cost and ownership tags are defined", make_required_tags_rule(required_tags)),
    ]


def _suppressed_rules(source_lines: list[str], line: int) -> set[str]:
    """Rule ids named in ``archkit-ignore`` comments directly above ``line`` (1-based)."""
    rules: set[str] = set()
    index = line - 2
    while index >= 0:
        stripped = source_lines[index].strip()
        if stripped.startswith("//"):
            rules |= set(_SUPPRESS.findall(stripped))
        elif not stripped.startswith("@"):
            break
        index -= 1
    return rules


def validate_files(paths: Iterable[str | Path], rules: list[Rule] | None = None) -> PolicyReport:
    """Parse each file and evaluate every rule on it."""
    active = rules if rules is not None else default_rules()
    report = PolicyReport(rules=[(r.rule_id, r.title) for r in active])
    for path in sorted(str(p) for p in paths):
        file = parse_bicep_file(path)
        source_lines = Path(path).read_text(encoding="utf-8").splitlines()
        report.files.append(path)
        report.resources_checked += sum(1 for _ in _deployed(file))
        for rule in active:
            for finding in rule.check(file):
                if finding.rule_id in _suppressed_rules(source_lines, finding.line):
                    report.suppressed.append(finding)
                else:
                    report.findings.append(finding)
    return report


def validate_directory(directory: str | Path, rules: list[Rule] | None = None) -> PolicyReport:
    """Validate every ``.bicep`` file below ``directory``."""
    paths = sorted(Path(directory).rglob("*.bicep"))
    if not paths:
        raise FileNotFoundError(f"no .bicep files found under {directory}")
    return validate_files(paths, rules)
