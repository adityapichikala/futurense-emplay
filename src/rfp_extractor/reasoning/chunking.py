"""Section-aware chunking.

Chunks are the unit of retrieval.  We do **not** slice blindly by character
count: a chunk that ends mid-table row or between a requirement heading and its
bullets is useless for extraction.  Instead we pack whole blocks up to a token
budget and cut on section or heading boundaries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..models.document import Block, BlockKind, Document

_WORD_RE = re.compile(r"\w+")


def estimate_tokens(text: str) -> int:
    """Fast token estimate (~1.3 tokens/word) - avoids a tokenizer dependency."""
    return int(len(_WORD_RE.findall(text)) * 1.3)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    file_name: str
    doc_type: str
    precedence: int
    page: int
    section: str | None
    text: str
    block_ids: list[str] = field(default_factory=list)
    tokens: int = 0

    @property
    def citation(self) -> str:
        return f"{self.file_name} p{self.page}"


class SemanticChunker:
    """Pack blocks into retrieval-sized chunks, cutting on structure."""

    def __init__(self, *, target_tokens: int = 420, overlap_tokens: int = 60) -> None:
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens

    def chunk_document(self, doc: Document) -> list[Chunk]:
        chunks: list[Chunk] = []
        buffer: list[Block] = []
        buffer_tokens = 0
        section: str | None = None
        counter = 0

        def flush() -> None:
            nonlocal buffer, buffer_tokens, counter
            if not buffer:
                return
            counter += 1
            text = "\n".join(b.text for b in buffer)
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}::c{counter}",
                    doc_id=doc.doc_id,
                    file_name=doc.file_name,
                    doc_type=doc.doc_type.value,
                    precedence=doc.precedence,
                    page=buffer[0].page_no,
                    section=section,
                    text=text,
                    block_ids=[b.block_id for b in buffer],
                    tokens=estimate_tokens(text),
                )
            )
            # carry a little overlap so values spanning a boundary are not lost
            if self.overlap_tokens > 0:
                keep: list[Block] = []
                kept = 0
                for block in reversed(buffer):
                    keep.insert(0, block)
                    kept += estimate_tokens(block.text)
                    if kept >= self.overlap_tokens:
                        break
                buffer, buffer_tokens = keep, kept
            else:
                buffer, buffer_tokens = [], 0

        for block in doc.iter_blocks():
            if not block.text.strip():
                continue
            tokens = estimate_tokens(block.text)

            # headings always start a new chunk (structure boundary)
            if block.kind is BlockKind.HEADING and buffer:
                flush()
            if block.section:
                section = block.section

            # a single oversized block (big table) becomes its own chunk
            if tokens > self.target_tokens * 1.5:
                flush()
                counter += 1
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc.doc_id}::c{counter}",
                        doc_id=doc.doc_id,
                        file_name=doc.file_name,
                        doc_type=doc.doc_type.value,
                        precedence=doc.precedence,
                        page=block.page_no,
                        section=section,
                        text=block.text[:8000],
                        block_ids=[block.block_id],
                        tokens=tokens,
                    )
                )
                continue

            if buffer_tokens + tokens > self.target_tokens and buffer:
                flush()
            buffer.append(block)
            buffer_tokens += tokens

        flush()
        return chunks

    def chunk_documents(self, docs: list[Document]) -> list[Chunk]:
        out: list[Chunk] = []
        for doc in docs:
            out.extend(self.chunk_document(doc))
        return out
