"""HTML ingestion for procurement portals (BidNet Direct and friends).

Portal pages are far more structured than the PDFs: the label/value grid is
machine-readable, so we parse it explicitly into ``label_values`` rather than
flattening to text and hoping a regex finds it again.

Primary pattern (BidNet)::

    <div class="mets-field">
      <span class="mets-field-label">Closing Date</span>
      <div class="mets-field-body"><p>07/09/2024 03:00 PM EDT</p></div>
    </div>

We also fall back to generic structures (``<table>`` rows, ``<dt>/<dd>``,
``<th>/<td>``) so the loader is not tied to one portal's markup.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from bs4 import BeautifulSoup

from ..models.document import Block, BlockKind, Document, FileFormat, Page
from .base import DocumentLoader, make_doc_id
from .normalize import normalize_text, squish

log = logging.getLogger(__name__)

#: Elements that never carry procurement content.
#: NB: ``form`` is deliberately absent - BidNet wraps its whole field grid in
#: ``<form class="current-form">``, so dropping it would delete every field.
_DROP_TAGS = ("script", "style", "noscript", "svg", "nav", "footer", "header", "iframe")

#: Interactive controls whose text is UI chrome, not document content.
_DROP_CONTROLS = ("input", "select", "textarea", "button")

_FIELD_SELECTORS = (
    "div.mets-field",
    ".mets-field",
    "div.field",
    "div.form-field",
)


class HtmlLoader(DocumentLoader):
    formats = (FileFormat.HTML,)

    def load(self, path: str | Path) -> Document:
        path = Path(path)
        raw = path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(raw, "lxml")

        for tag in soup.find_all(_DROP_TAGS):
            tag.decompose()
        for tag in soup.find_all(_DROP_CONTROLS):
            tag.decompose()

        label_values, blocks = self._extract_fields(soup)
        blocks.extend(self._extract_prose(soup, start=len(blocks)))

        page = Page(page_no=1, blocks=blocks)
        if label_values:
            page.blocks.insert(
                0,
                Block(
                    block_id="p1.meta0",
                    page_no=1,
                    kind=BlockKind.TEXT,
                    text="\n".join(f"{k}: {v}" for k, v in label_values.items()),
                    raw_text="\n".join(f"{k}: {v}" for k, v in label_values.items()),
                    meta={"synthetic": "label_values"},
                ),
            )

        content_hash = _sha256(path)
        title = self._page_title(soup)
        return Document(
            doc_id=make_doc_id(path.name, content_hash),
            file_name=path.name,
            source_path=str(path),
            file_format=FileFormat.HTML,
            content_hash=content_hash,
            pages=[page],
            label_values=label_values,
            meta={"loader": "HtmlLoader", "page_title": title, "label_count": len(label_values)},
        )

    # -- internals ---------------------------------------------------------

    def _extract_fields(self, soup: BeautifulSoup) -> tuple[dict[str, str], list[Block]]:
        label_values: dict[str, str] = {}
        blocks: list[Block] = []
        seen: set[int] = set()
        section: str | None = None
        counter = 0

        # Walk in document order so section headings are associated correctly.
        for node in soup.find_all(["h1", "h2", "h3", "h4", "div", "dt", "tr", "li"]):
            classes: list[str] = list(node.get("class") or [])
            if isinstance(classes, str):
                classes = classes.split()

            if node.name in ("h1", "h2", "h3", "h4"):
                heading = squish(normalize_text(node.get_text(" ", strip=True)))
                if heading and len(heading) < 80:
                    section = heading
                    counter += 1
                    blocks.append(
                        Block(
                            block_id=f"p1.h{counter}",
                            page_no=1,
                            kind=BlockKind.HEADING,
                            text=heading,
                            raw_text=heading,
                            section=heading,
                        )
                    )
                continue

            if id(node) in seen:
                continue

            if node.name == "div" and not self._is_field_container(node, classes):
                continue

            pair = self._read_pair(node, section)
            if not pair:
                continue

            seen.add(id(node))
            label, value = pair
            if label in label_values and label_values[label] == value:
                continue

            label_values.setdefault(label, value)
            counter += 1
            blocks.append(
                Block(
                    block_id=f"p1.f{counter}",
                    page_no=1,
                    kind=BlockKind.LABEL_VALUE,
                    text=f"{label}: {value}",
                    raw_text=f"{label}: {value}",
                    label=label,
                    section=section,
                    meta={"label_value": value},
                )
            )

        return label_values, blocks

    @staticmethod
    def _is_field_container(node, classes: list[str]) -> bool:
        if any(c in ("mets-field", "mets-field-view", "form-field", "field") for c in classes):
            return True
        # label + body as direct children
        return node.find("span", class_="mets-field-label") is not None or (
            node.find("dt") is not None and node.name == "div"
        )

    @staticmethod
    def _read_pair(node, section: str | None = None) -> tuple[str, str] | None:
        # BidNet: span.mets-field-label + div.mets-field-body
        label_el = node.find("span", class_="mets-field-label")
        body_el = node.find("div", class_="mets-field-body")
        if label_el is None:
            label_el = node.find(["dt", "th", "label", "strong"])
        if body_el is None:
            body_el = node.find(["dd"]) or node.find("div", class_=re.compile("body|value"))

        if label_el is not None and body_el is not None:
            label = squish(normalize_text(label_el.get_text(" ", strip=True)))
            value = squish(normalize_text(body_el.get_text(" ", strip=True)))
            if not value:
                return None
            if _is_filler(label):
                # Unlabelled rows under "Contact Information" (org name, phone)
                # still carry the answer - attribute them to the section.
                if section and section.lower() == "contact information":
                    return section, value
                return None
            return label, value

        if node.name == "tr":
            cells = node.find_all(["th", "td"])
            if len(cells) >= 2:
                label = squish(normalize_text(cells[0].get_text(" ", strip=True)))
                value = squish(normalize_text(cells[1].get_text(" ", strip=True)))
                if label and value and not _is_filler(label):
                    return label, value

        if node.name == "li":
            text = squish(normalize_text(node.get_text(" ", strip=True)))
            if ":" in text and len(text) < 300:
                label, _, value = text.partition(":")
                label, value = label.strip(), value.strip()
                if label and value and len(label) < 60:
                    return label, value
        return None

    @staticmethod
    def _extract_prose(soup: BeautifulSoup, start: int) -> list[Block]:
        """Long-form blocks (solicitation description, scope, attachments)."""
        blocks: list[Block] = []
        counter = start
        for node in soup.find_all(["p", "div"], class_=re.compile("description|body|content|desc", re.I)):
            text = squish(normalize_text(node.get_text(" ", strip=True)))
            if len(text) < 40:
                continue
            # skip containers that merely wrap field grids
            if node.find("div", class_="mets-field"):
                continue
            counter += 1
            blocks.append(
                Block(
                    block_id=f"p1.p{counter}",
                    page_no=1,
                    kind=BlockKind.TEXT,
                    text=text,
                    raw_text=text,
                )
            )
        # de-duplicate by text
        unique: dict[str, Block] = {}
        for b in blocks:
            unique.setdefault(b.text, b)
        return list(unique.values())

    @staticmethod
    def _page_title(soup: BeautifulSoup) -> str:
        if soup.title and soup.title.string:
            return squish(normalize_text(soup.title.string))
        h1 = soup.find("h1")
        return squish(normalize_text(h1.get_text(" ", strip=True))) if h1 else ""


def _is_filler(label: str) -> bool:
    """BidNet emits empty 'filler' labels for unlabelled contact rows."""
    return not label or label.lower() in {"", "n/a", "-"} or len(label) > 80


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
