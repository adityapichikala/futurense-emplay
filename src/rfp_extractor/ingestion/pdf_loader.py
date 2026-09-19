"""Layout-aware PDF ingestion.

Two engines cooperate:

* **PyMuPDF** (``get_text("dict")``) gives per-span bounding boxes, which we turn
  into :class:`Block` objects with real geometry.  Geometry is what lets us repair
  the reading order - the corpus contains pages (Addendum 1, the RFP header page)
  where the content stream interleaves left/right columns, so naive text order is
  wrong.
* **pdfplumber** supplies table structure for the price schedule and PORFP
  Section 4, which carry line items that free text destroys.

If a page yields no text layer at all we flag it (``meta["scanned"]``) rather than
silently emitting an empty document, so OCR can be wired in later.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pymupdf

from ..models.document import (
    BBox,
    Block,
    BlockKind,
    Document,
    FileFormat,
    Page,
)
from .base import DocumentLoader, make_doc_id
from .cache import DocumentCache
from .normalize import (
    deconcatenate_labels,
    detect_running_lines,
    normalize_text,
    squish,
)

log = logging.getLogger(__name__)

_LABEL_INLINE_RE = re.compile(r"^(?P<label>[A-Za-z][A-Za-z0-9 /'\-#()&.,]{2,60}?)\s*:\s*(?P<value>.+)$", re.S)
_LABEL_ONLY_RE = re.compile(r"^(?P<label>[A-Za-z][A-Za-z0-9 /'\-#()&.,]{2,60}?)\s*:$")
_HEADING_RE = re.compile(
    r"^\s*(?:SECTION|Section|ARTICLE|Article|PART|Part)?\s*(\d+(?:\.\d+)*)\s*[.–-]?\s*.{0,80}$"
)


class PdfLoader(DocumentLoader):
    """Parse a PDF into page-level :class:`Block` objects."""

    formats = (FileFormat.PDF,)

    def __init__(
        self,
        *,
        extract_tables: bool = True,
        table_pages: int | None = None,
        cache: DocumentCache | None = None,
    ) -> None:
        self.extract_tables = extract_tables
        #: cap on pages sent to pdfplumber (tables are slow on long documents)
        self.table_pages = table_pages
        self.cache = cache

    # -- public ------------------------------------------------------------

    def load(self, path: str | Path) -> Document:
        path = Path(path)
        content_hash = _sha256(path)
        flavour = "tables" if self.extract_tables else "text"

        cached = self.cache.get(content_hash, flavour) if self.cache else None
        if cached is not None:
            log.debug("cache hit: %s", path.name)
            cached.source_path = str(path)
            return cached

        with pymupdf.open(str(path)) as doc:
            pages = [self._load_page(doc, i) for i in range(doc.page_count)]

        running = detect_running_lines(
            [[ln for ln in p.text.split("\n")] for p in pages]
        )
        for page in pages:
            page.blocks = [b for b in page.blocks if squish(b.text) not in running]

        if self.extract_tables:
            self._attach_tables(path, pages)

        document = Document(
            doc_id=make_doc_id(path.name, content_hash),
            file_name=path.name,
            source_path=str(path),
            file_format=FileFormat.PDF,
            content_hash=content_hash,
            pages=pages,
            meta={"loader": "PdfLoader", "table_extraction": self.extract_tables},
        )
        if self.cache:
            self.cache.put(document, flavour)
        return document

    # -- internals ---------------------------------------------------------

    def _load_page(self, doc: pymupdf.Document, index: int) -> Page:
        page = doc.load_page(index)
        raw = page.get_text("dict")
        page_no = index + 1

        spans: list[tuple[BBox, str, float]] = []
        for block in raw.get("blocks", []):
            if block.get("type", 0) != 0:  # 0 == text block
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "")
                    if not text.strip():
                        continue
                    x0, y0, x1, y1 = span["bbox"]
                    spans.append((BBox(x0=x0, y0=y0, x1=x1, y1=y1), text, float(span.get("size", 10.0))))

        lines = self._group_lines(spans)
        ordered = self._reading_order(lines, page_width=float(page.rect.width))

        blocks: list[Block] = []
        for n, (bbox, text, size) in enumerate(ordered, start=1):
            clean = normalize_text(deconcatenate_labels(text))
            if not clean:
                continue
            blocks.append(
                Block(
                    block_id=f"p{page_no}.b{n}",
                    page_no=page_no,
                    kind=self._classify_kind(clean, size),
                    text=clean,
                    raw_text=text,
                    bbox=bbox,
                    meta={"font_size": size},
                )
            )
        self._pair_labels(blocks, page_no)

        return Page(
            page_no=page_no,
            width=float(page.rect.width),
            height=float(page.rect.height),
            blocks=blocks,
        )

    @staticmethod
    def _group_lines(spans: list[tuple[BBox, str, float]]) -> list[tuple[BBox, str, float]]:
        """Merge spans that share a baseline into one visual line.

        PDFs fragment a sentence into many spans (kerning, font switches);
        extraction rules need whole lines to match against.
        """
        if not spans:
            return []
        buckets: list[list[tuple[BBox, str, float]]] = []
        for bbox, text, size in sorted(spans, key=lambda s: (round(s[0].y0, 1), s[0].x0)):
            placed = False
            for bucket in buckets:
                ref = bucket[0][0]
                # same visual line if vertical centres are within ~40% of height
                if abs((bbox.y0 + bbox.y1) / 2 - (ref.y0 + ref.y1) / 2) <= max(2.0, 0.6 * max(bbox.height, ref.height)):
                    bucket.append((bbox, text, size))
                    placed = True
                    break
            if not placed:
                buckets.append([(bbox, text, size)])

        lines: list[tuple[BBox, str, float]] = []
        for bucket in buckets:
            bucket.sort(key=lambda s: s[0].x0)
            x0 = min(b[0].x0 for b in bucket)
            y0 = min(b[0].y0 for b in bucket)
            x1 = max(b[0].x1 for b in bucket)
            y1 = max(b[0].y1 for b in bucket)
            size = max(b[2] for b in bucket)
            text = "".join(b[1] for b in bucket)
            lines.append((BBox(x0=x0, y0=y0, x1=x1, y1=y1), text, size))
        return lines

    # NOTE: splitting a "visual line" at wide horizontal gaps was implemented
    # and then reverted.  A table row puts every cell on one baseline, so
    # splitting does recover form fields - but the same rule also separates a
    # bullet from its text (hanging indent measures ~7 character widths, while
    # true column gutters measure 11-25), which destroyed the tiered device
    # specifications in the Dallas ISD RFP.  The distributions overlap too much
    # for one global threshold, so line merging stays gap-agnostic.

    @staticmethod
    def _reading_order(
        lines: list[tuple[BBox, str, float]], *, page_width: float
    ) -> list[tuple[BBox, str, float]]:
        """Column-aware ordering.

        Splits the page into vertical columns when a wide horizontal gutter
        separates content, then orders top-to-bottom within each column and
        concatenates left-to-right.  Single-column pages fall through to plain
        y-then-x ordering.
        """
        if len(lines) < 6 or page_width <= 0:
            return sorted(lines, key=lambda ln: (round(ln[0].y0, 1), ln[0].x0))

        xs = sorted(ln[0].x0 for ln in lines)
        gaps = [b - a for a, b in zip(xs, xs[1:], strict=False)]
        if not gaps:
            return sorted(lines, key=lambda ln: (round(ln[0].y0, 1), ln[0].x0))
        widest = max(gaps)
        if widest < 0.18 * page_width:
            return sorted(lines, key=lambda ln: (round(ln[0].y0, 1), ln[0].x0))

        split_at = xs[gaps.index(widest)] + widest / 2
        left = [ln for ln in lines if ln[0].x0 < split_at]
        right = [ln for ln in lines if ln[0].x0 >= split_at]
        if not left or not right:
            return sorted(lines, key=lambda ln: (round(ln[0].y0, 1), ln[0].x0))

        def order(items: list[tuple[BBox, str, float]]) -> list[tuple[BBox, str, float]]:
            return sorted(items, key=lambda ln: (round(ln[0].y0, 1), ln[0].x0))

        return order(left) + order(right)

    @staticmethod
    def _classify_kind(text: str, size: float) -> BlockKind:
        if size >= 13 and len(text) < 120:
            return BlockKind.HEADING
        if _HEADING_RE.match(text) and len(text) < 120:
            return BlockKind.HEADING
        if re.match(r"^\s*(?:[\u2022\u25cf\u25aa\-*]|\d+[.)])\s+\S", text):
            return BlockKind.LIST_ITEM
        return BlockKind.TEXT

    @staticmethod
    def _pair_labels(blocks: list[Block], page_no: int) -> None:
        """Attach values to labels.

        RFP PDFs use two shapes:
          1. ``Label: value`` on one line  (handled inline)
          2. ``Label:`` on its own line with the value in the next block - very
             common inPORFP-style forms, and the reason a naive line reader loses
             the answer.
        """
        for block in blocks:
            inline = _LABEL_INLINE_RE.match(block.text)
            if inline and "\n" not in block.text:
                value = squish(inline.group("value"))
                # reject signature-sheet artefacts: "Company Name: Submitter's Name/Title:"
                # where the "value" is itself only another label.
                if _LABEL_ONLY_RE.match(value) or not value:
                    continue
                block.label = squish(inline.group("label"))
                block.kind = BlockKind.LABEL_VALUE
                block.meta["label_value"] = value
                continue
            label_only = _LABEL_ONLY_RE.match(block.text.strip())
            if label_only:
                block.label = squish(label_only.group("label"))
                block.kind = BlockKind.LABEL_VALUE

        # Deliberately NOT attempted: inferring colon-less labels from layout
        # alone ("Manufacturer Name" / "Dell" on consecutive lines).  On the real
        # corpus it mis-paired form prompts ("Supplier Name" -> "Supplier Address
        # 1") and polluted company_name, so label pairing stays colon-anchored.

        # second pass: resolve label-only blocks using geometry first, then order
        for i, block in enumerate(blocks):
            if block.kind != BlockKind.LABEL_VALUE or "label_value" in block.meta:
                continue
            for j in range(i + 1, min(i + 3, len(blocks))):
                nxt = blocks[j]
                if nxt.page_no != page_no or not nxt.text.strip():
                    continue
                if _LABEL_ONLY_RE.match(nxt.text.strip()) or _LABEL_INLINE_RE.match(nxt.text):
                    break
                block.meta["label_value"] = squish(nxt.text)
                break

    def _attach_tables(self, path: Path, pages: list[Page]) -> None:
        """Add pdfplumber tables as dedicated blocks (best-effort, never fatal)."""
        try:
            import pdfplumber
        except ImportError:  # pragma: no cover - optional dependency
            log.debug("pdfplumber unavailable; skipping table extraction")
            return

        limit = self.table_pages or len(pages)
        try:
            with pdfplumber.open(str(path)) as pdf:
                for index, plumber_page in enumerate(pdf.pages[:limit]):
                    if index >= len(pages):
                        break
                    try:
                        tables = plumber_page.extract_tables() or []
                    except Exception as exc:  # noqa: BLE001
                        log.debug("table extraction failed p%s: %s", index + 1, exc)
                        continue
                    for tno, table in enumerate(tables, start=1):
                        rows = [
                            [normalize_text(cell) if isinstance(cell, str) else "" for cell in row]
                            for row in table
                        ]
                        rows = [r for r in rows if any(c.strip() for c in r)]
                        if len(rows) < 2:
                            continue
                        text = "\n".join(" | ".join(r) for r in rows)
                        pages[index].blocks.append(
                            Block(
                                block_id=f"p{index + 1}.t{tno}",
                                page_no=index + 1,
                                kind=BlockKind.TABLE,
                                text=text,
                                raw_text=text,
                                table=rows,
                                meta={"rows": len(rows), "cols": max(len(r) for r in rows)},
                            )
                        )
        except Exception as exc:  # noqa: BLE001
            log.warning("pdfplumber failed on %s: %s", path.name, exc)


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
