"""Core document model: the immutable, page-level representation produced by ingestion.

Design notes
------------
Every downstream stage works against ``Document``/``Block`` objects, never against
raw strings.  This gives us three properties that matter for RFP work:

* **Lineage** - any value extracted later can be traced back to a file, a page and
  a verbatim span, because blocks carry ``(page_no, block_id, bbox)``.
* **Layout awareness** - PDFs in this corpus (notably Addendum 1) have interleaved
  reading order.  Blocks keep their bounding box so we can re-sort into a sane
  reading order rather than trusting the PDF content stream.
* **Format neutrality** - HTML, PDF and DOCX all reduce to the same model, so the
  extraction engine has a single contract to satisfy.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class FileFormat(str, Enum):
    PDF = "pdf"
    HTML = "html"
    DOCX = "docx"
    TEXT = "text"
    UNKNOWN = "unknown"


class DocType(str, Enum):
    """Semantic role of a document inside a bid package.

    Drives precedence during merge: an addendum outranks the master RFP.
    """

    MASTER_RFP = "master_rfp"
    ADDENDUM = "addendum"
    PORTAL_NOTICE = "portal_notice"
    VENDOR_SPEC = "vendor_spec"
    LEGAL_AFFIDAVIT = "legal_affidavit"
    ASSIGNMENT = "assignment"
    UNKNOWN = "unknown"


class BlockKind(str, Enum):
    TEXT = "text"
    HEADING = "heading"
    TABLE = "table"
    LIST_ITEM = "list_item"
    LABEL_VALUE = "label_value"
    CAPTION = "caption"


class BBox(BaseModel):
    """Axis-aligned bounding box in PDF points (origin: top-left of page)."""

    model_config = ConfigDict(frozen=True)

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0


class Block(BaseModel):
    """A single addressable unit of content.

    ``text`` is the normalized, de-hyphenated, whitespace-collapsed content.
    ``raw_text`` preserves the original string for verbatim citation.
    """

    block_id: str
    page_no: int  # 1-based
    kind: BlockKind = BlockKind.TEXT
    text: str = ""
    raw_text: str = ""
    bbox: BBox | None = None
    section: str | None = None
    label: str | None = None  # populated for HTML/PDF label->value pairs
    table: list[list[str]] | None = None
    meta: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


class Page(BaseModel):
    page_no: int  # 1-based
    width: float = 0.0
    height: float = 0.0
    blocks: list[Block] = Field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(b.text for b in self.blocks if b.text.strip())


class Document(BaseModel):
    """A parsed source file plus all metadata the pipeline needs."""

    doc_id: str
    file_name: str
    source_path: str
    file_format: FileFormat
    content_hash: str
    doc_type: DocType = DocType.UNKNOWN
    doc_type_confidence: float = 0.0
    doc_type_signals: list[str] = Field(default_factory=list)
    pages: list[Page] = Field(default_factory=list)
    label_values: dict[str, str] = Field(default_factory=dict)
    addendum_number: int | None = None
    addendum_effective_date: str | None = None
    amendment_notes: list[str] = Field(default_factory=list)
    precedence: int = 0
    meta: dict[str, Any] = Field(default_factory=dict)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def blocks(self) -> list[Block]:
        return [b for p in self.pages for b in p.blocks]

    @property
    def full_text(self) -> str:
        return "\n".join(p.text for p in self.pages)

    def iter_blocks(self) -> Iterator[Block]:
        for page in self.pages:
            yield from page.blocks

    def find(self, needle: str, case_sensitive: bool = False) -> list[Block]:
        """Convenience for tests: all blocks whose text contains ``needle``."""
        if case_sensitive:
            return [b for b in self.iter_blocks() if needle in b.text]
        low = needle.lower()
        return [b for b in self.iter_blocks() if low in b.text.lower()]


def compute_content_hash(path: str | Path) -> str:
    """SHA-256 of file bytes - used for idempotency and dedup."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def detect_format(path: str | Path) -> FileFormat:
    suffix = Path(path).suffix.lower().lstrip(".")
    try:
        return FileFormat(suffix)
    except ValueError:
        return FileFormat.UNKNOWN
