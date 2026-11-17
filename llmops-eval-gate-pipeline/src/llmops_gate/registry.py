"""File-based model and prompt registry with versioning and stage promotion.

Layout on disk::

    <root>/models/<name>/v<N>/meta.json  + artifact file
    <root>/prompts/<name>/v<N>/meta.json (template stored inline)
    <root>/history.jsonl                 (append-only stage transitions)

Versions are immutable: registering again creates the next version. Stages move
along ``none -> staging -> production -> archived``. Promotion to production
requires a passing evaluation gate recorded on the version, unless an explicit
override reason is given, and it archives the previous production version.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

Kind = Literal["model", "prompt"]
_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{0,62}$")
_VARIABLE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


class Stage(StrEnum):
    NONE = "none"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


ALLOWED_TRANSITIONS: dict[Stage, set[Stage]] = {
    Stage.NONE: {Stage.STAGING, Stage.ARCHIVED},
    Stage.STAGING: {Stage.PRODUCTION, Stage.ARCHIVED, Stage.NONE},
    Stage.PRODUCTION: {Stage.ARCHIVED},
    Stage.ARCHIVED: {Stage.STAGING, Stage.PRODUCTION},
}


class RegistryError(Exception):
    """Invalid registry operation."""


class PromotionBlocked(RegistryError):
    """Promotion refused because the evaluation gate did not pass."""


class Entry(BaseModel):
    """One immutable version of a model or prompt plus its mutable lifecycle fields."""

    kind: Kind
    name: str
    version: int
    stage: Stage = Stage.NONE
    sha256: str
    artifact: str | None = Field(default=None, description="Artifact file name for models")
    template: str | None = Field(default=None, description="Prompt template text for prompts")
    variables: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    gate: dict[str, Any] | None = Field(default=None, description="Last gate decision")
    created_at: float = Field(default_factory=time.time)

    @property
    def ref(self) -> str:
        return f"{self.kind}:{self.name}:v{self.version}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_prompt(template: str, **values: str) -> str:
    """Fill ``{variable}`` placeholders. Missing or unexpected variables are errors."""
    expected = set(_VARIABLE.findall(template))
    if set(values) != expected:
        missing, extra = sorted(expected - set(values)), sorted(set(values) - expected)
        raise RegistryError(f"prompt variables mismatch: missing {missing}, unexpected {extra}")
    return _VARIABLE.sub(lambda m: str(values[m.group(1)]), template)


class Registry:
    """Versioned storage for models and prompts on the local file system."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    # ------------------------------------------------------------------ paths
    def _dir(self, kind: Kind, name: str) -> Path:
        if not _NAME.match(name):
            raise RegistryError(f"invalid name '{name}': use lowercase letters, digits, . _ -")
        return self.root / f"{kind}s" / name

    def _meta(self, kind: Kind, name: str, version: int) -> Path:
        return self._dir(kind, name) / f"v{version}" / "meta.json"

    def _write(self, entry: Entry) -> None:
        path = self._meta(entry.kind, entry.name, entry.version)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(entry.model_dump_json(indent=2))
        os.replace(tmp, path)

    def _log(self, event: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"ts": time.time(), **event}, sort_keys=True) + "\n")

    # --------------------------------------------------------------- register
    def _next_version(self, kind: Kind, name: str) -> int:
        versions = [e.version for e in self.list(kind, name)]
        return max(versions, default=0) + 1

    def register_model(
        self, name: str, artifact: str | Path, metadata: dict[str, Any] | None = None
    ) -> Entry:
        """Copy ``artifact`` into the registry as the next version of ``name``."""
        source = Path(artifact)
        if not source.is_file():
            raise RegistryError(f"artifact '{source}' does not exist")
        version = self._next_version("model", name)
        target = self._dir("model", name) / f"v{version}" / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        entry = Entry(
            kind="model",
            name=name,
            version=version,
            sha256=sha256_file(target),
            artifact=source.name,
            metadata={"size_bytes": target.stat().st_size, **(metadata or {})},
        )
        self._write(entry)
        self._log({"event": "registered", "ref": entry.ref, "sha256": entry.sha256})
        return entry

    def register_prompt(
        self, name: str, template: str, metadata: dict[str, Any] | None = None
    ) -> Entry:
        """Store ``template`` as the next version of prompt ``name``."""
        if not template.strip():
            raise RegistryError("prompt template is empty")
        latest = self.latest("prompt", name)
        digest = hashlib.sha256(template.encode("utf-8")).hexdigest()
        if latest is not None and latest.sha256 == digest:
            raise RegistryError(f"template is identical to {latest.ref}")
        entry = Entry(
            kind="prompt",
            name=name,
            version=self._next_version("prompt", name),
            sha256=digest,
            template=template,
            variables=sorted(set(_VARIABLE.findall(template))),
            metadata=metadata or {},
        )
        self._write(entry)
        self._log({"event": "registered", "ref": entry.ref, "sha256": entry.sha256})
        return entry

    # ------------------------------------------------------------------ query
    def list(self, kind: Kind, name: str) -> list[Entry]:
        """All versions of ``name`` in ascending version order."""
        directory = self._dir(kind, name)
        entries = [
            Entry.model_validate_json(p.read_text(encoding="utf-8"))
            for p in directory.glob("v*/meta.json")
        ]
        return sorted(entries, key=lambda e: e.version)

    def names(self, kind: Kind) -> list[str]:
        base = self.root / f"{kind}s"
        return sorted(p.name for p in base.iterdir() if p.is_dir()) if base.exists() else []

    def get(self, kind: Kind, name: str, version: int | None = None) -> Entry:
        """Fetch one version, or the latest when ``version`` is omitted."""
        if version is None:
            latest = self.latest(kind, name)
            if latest is None:
                raise RegistryError(f"no versions of {kind} '{name}'")
            return latest
        path = self._meta(kind, name, version)
        if not path.exists():
            raise RegistryError(f"{kind}:{name}:v{version} does not exist")
        return Entry.model_validate_json(path.read_text(encoding="utf-8"))

    def latest(self, kind: Kind, name: str) -> Entry | None:
        entries = self.list(kind, name)
        return entries[-1] if entries else None

    def in_stage(self, kind: Kind, name: str, stage: Stage) -> Entry | None:
        """Newest version currently in ``stage``."""
        matches = [e for e in self.list(kind, name) if e.stage is stage]
        return matches[-1] if matches else None

    def artifact_path(self, entry: Entry) -> Path:
        if entry.kind != "model" or entry.artifact is None:
            raise RegistryError(f"{entry.ref} has no artifact file")
        return self._dir("model", entry.name) / f"v{entry.version}" / entry.artifact

    def verify_artifact(self, entry: Entry) -> bool:
        """Recompute the artifact hash and compare with the registered one."""
        return sha256_file(self.artifact_path(entry)) == entry.sha256

    def history(self) -> list[dict[str, Any]]:
        path = self.root / "history.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

    # -------------------------------------------------------------- lifecycle
    def record_evaluation(
        self, entry: Entry, metrics: dict[str, float], gate: dict[str, Any]
    ) -> Entry:
        """Attach evaluation metrics and the gate decision to a version."""
        entry = self.get(entry.kind, entry.name, entry.version)
        entry.metrics, entry.gate = metrics, gate
        self._write(entry)
        self._log({"event": "evaluated", "ref": entry.ref, "gate_passed": gate.get("passed")})
        return entry

    def promote(
        self,
        kind: Kind,
        name: str,
        version: int,
        to: Stage,
        actor: str = "cli",
        override_reason: str | None = None,
    ) -> Entry:
        """Move a version to another stage, enforcing transitions and the gate."""
        entry = self.get(kind, name, version)
        if to not in ALLOWED_TRANSITIONS[entry.stage]:
            raise RegistryError(
                f"{entry.ref}: transition {entry.stage.value} -> {to.value} is not allowed"
            )
        if to in (Stage.STAGING, Stage.PRODUCTION) and not override_reason:
            if entry.gate is None:
                raise PromotionBlocked(f"{entry.ref} has not been evaluated")
            if not entry.gate.get("passed"):
                failed = [c["message"] for c in entry.gate.get("checks", []) if not c["passed"]]
                raise PromotionBlocked(
                    f"{entry.ref} failed the evaluation gate: " + "; ".join(failed)
                )
        if to is Stage.PRODUCTION:
            current = self.in_stage(kind, name, Stage.PRODUCTION)
            if current is not None:
                current.stage = Stage.ARCHIVED
                self._write(current)
                self._log({"event": "stage", "ref": current.ref, "to": "archived", "actor": actor})
        previous = entry.stage
        entry.stage = to
        self._write(entry)
        self._log(
            {
                "event": "stage",
                "ref": entry.ref,
                "from": previous.value,
                "to": to.value,
                "actor": actor,
                "override_reason": override_reason,
            }
        )
        return entry

    def rollback(self, kind: Kind, name: str, actor: str = "cli") -> Entry:
        """Restore the most recently archived former production version."""
        current = self.in_stage(kind, name, Stage.PRODUCTION)
        former = [
            h["ref"]
            for h in self.history()
            if h.get("event") == "stage"
            and h.get("to") == "production"
            and h["ref"].startswith(f"{kind}:{name}:")
            and (current is None or h["ref"] != current.ref)
        ]
        if not former:
            raise RegistryError(f"no earlier production version of {kind} '{name}' to roll back to")
        version = int(former[-1].rsplit(":v", 1)[1])
        return self.promote(
            kind, name, version, Stage.PRODUCTION, actor, override_reason="rollback"
        )
