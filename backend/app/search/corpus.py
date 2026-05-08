"""Load documents from JSONL: one JSON object per line with id, title, text.

Each line is a single JSON object. Required field: ``id`` (stored as ``doc_id``).
Optional ``title`` and ``text`` default to empty strings if missing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Document:
    """One corpus row. ``body`` concatenates title and text for indexing/search."""

    doc_id: str
    title: str
    text: str

    @property
    def body(self) -> str:
        """Single string passed to tokenizers and embedding models."""
        return f"{self.title}\n{self.text}"


def load_corpus(path: Path) -> list[Document]:
    """Read JSONL from disk; skip blank lines; preserve order."""
    docs: list[Document] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            docs.append(
                Document(
                    doc_id=str(row["id"]),
                    title=str(row.get("title", "")),
                    text=str(row.get("text", "")),
                )
            )
    return docs
