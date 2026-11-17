"""Document and chunk data model plus a loader for markdown corpora."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Document:
    """A source document. ``doc_id`` is stable and used in citations and gold labels."""

    doc_id: str
    title: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    """A retrievable passage of a document."""

    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str
    position: int

    @property
    def indexed_text(self) -> str:
        """Text used for indexing: title and section give short chunks topical context."""
        return f"{self.title}. {self.section}. {self.text}" if self.section else f"{self.title}. {self.text}"


def parse_markdown(doc_id: str, raw: str) -> Document:
    """Build a document from markdown. The first ``# `` heading becomes the title."""
    title = doc_id
    body_lines: list[str] = []
    seen_title = False
    for line in raw.splitlines():
        if not seen_title and line.startswith("# "):
            title = line[2:].strip()
            seen_title = True
        else:
            body_lines.append(line)
    return Document(doc_id=doc_id, title=title, text="\n".join(body_lines).strip())


def load_corpus(directory: str | Path) -> list[Document]:
    """Load every ``.md`` and ``.txt`` file below ``directory``, sorted by file name."""
    root = Path(directory)
    paths = sorted(p for p in root.rglob("*") if p.suffix in {".md", ".txt"})
    if not paths:
        raise FileNotFoundError(f"no .md or .txt documents found under {root}")
    return [parse_markdown(path.stem, path.read_text(encoding="utf-8")) for path in paths]


def bundled_corpus_dir() -> Path:
    """Directory of the bundled synthetic corpus."""
    return Path(__file__).parent / "data" / "corpus"
