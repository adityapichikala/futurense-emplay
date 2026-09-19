"""Structured product-specification extraction (device tiers / line items).

``product_specification`` is not a scalar - it is a nested map of
``device -> {attribute: requirement}``.  Flattening it into a sentence would
throw away exactly the structure a bid/no-bid decision needs, so it gets a
dedicated parser.

Two shapes are handled:

**A. Tiered minimum requirements (Dallas ISD)**
Headings such as ``Tier 1 Small Student Chromebook Laptop Minimum Requirements``
followed by a bulleted attribute list.

**B. Tabular line items (PORFP Section 4)**
Rows of ``Product Name | Product Description | Model # | Qty | Due Date``.
"""

from __future__ import annotations

import re
from typing import Any

from ..models.document import Block, BlockKind, Document
from ..models.schema import ExtractionMethod, SourceValue

_TIER_HEADING_RE = re.compile(
    r"^(?P<tier>Tier\s*(?P<num>\d+)\s+)?(?P<name>[A-Za-z0-9][A-Za-z0-9 ,/&()\-\"']{3,70}?)"
    r"\s*(?:Minimum\s+Requirements?|Specifications?|Requirements?)\s*:?\s*$",
    re.I,
)

_BULLET_RE = re.compile(r"^[\u2022\u25cf\u25aa\u2043\u2219*\-–]\s*(?P<item>.+)$")

#: A bare heading is only treated as a device tier when it names hardware.
_DEVICE_RE = re.compile(
    r"\b(monitor|laptop|desktop|tablet|chromebook|display|dock|printer|device|"
    r"workstation|server|phone|keyboard|mouse|notebook)\b",
    re.I,
)
_ATTR_SPLIT_RE = re.compile(r"^(?P<key>[A-Za-z][A-Za-z0-9 /()\-]{2,40}?)\s*:\s*(?P<value>.+)$")

_LINE_ITEM_RE = re.compile(r"^(?P<num>\d{1,3})[.)]\s*(?P<name>.+)$")


def _clean(value: str) -> str:
    return " ".join(value.split()).strip(" .;:,-")


class ProductSpecExtractor:
    """Build ``{device: {attribute: requirement}}`` from a document."""

    #: Document roles that can legitimately carry equipment specifications.
    #: Addenda are excluded: their numbered Q&A items look exactly like numbered
    #: line items and polluted early runs of this extractor.
    _ALLOWED_DOC_TYPES = frozenset({"master_rfp", "portal_notice", "vendor_spec"})

    def extract(self, doc: Document) -> dict[str, Any] | None:
        if doc.doc_type.value not in self._ALLOWED_DOC_TYPES:
            return None
        specs = {k: v for k, v in self._extract_tiers(doc).items() if len(v) >= 2}
        # Table-derived line items are trusted with a single attribute (the
        # PORFP grid stores model/qty/due-date as columns, not bullets).
        specs.update(self._extract_line_items(doc))
        return specs or None

    # -- shape A: tiers -----------------------------------------------------

    def _extract_tiers(self, doc: Document) -> dict[str, Any]:
        blocks = [
            b
            for b in doc.iter_blocks()
            if b.kind in (BlockKind.TEXT, BlockKind.LIST_ITEM, BlockKind.HEADING)
        ]
        result: dict[str, Any] = {}
        current: str | None = None
        headings = self._heading_labels(blocks)
        for index, block in enumerate(blocks):
            label = headings.get(index)
            if label:
                current = label
                result.setdefault(current, {})
                continue
            if current is None:
                continue
            items = self._bullets(block.text)
            if not items:
                # a page break or a non-spec paragraph ends the current tier
                if block.text.strip() and not block.text.strip().startswith("•"):
                    if len(block.text) > 120:
                        current = None
                continue
            for item in items:
                attr = _ATTR_SPLIT_RE.match(item)
                if attr:
                    result[current][_clean(attr.group("key"))] = _clean(attr.group("value"))
                else:
                    result[current][f"requirement_{len(result[current]) + 1}"] = item
        return {k: v for k, v in result.items() if v}

    @classmethod
    def _heading_labels(cls, blocks: list[Block]) -> dict[int, str]:
        """Index -> tier name for every block that opens a specification list.

        Two heading shapes occur:

        * ``Tier 1 Small Student Chromebook Laptop Minimum Requirements``
        * ``Display Monitor - Touch``  — a bare device name with no suffix

        The second shape matters: without it, the monitor bullets were appended
        to the preceding tier (inflating "Tier 2 Staff Laptop" to 19 attributes)
        and the two monitor categories disappeared entirely.  A bare name is
        accepted as a heading only when it is short and immediately followed by
        a bulleted list, so prose paragraphs are never mistaken for one.
        """
        labels: dict[int, str] = {}
        for index, block in enumerate(blocks):
            text = block.text.strip()
            match = _TIER_HEADING_RE.match(text)
            if match:
                tier = _clean(match.group("tier") or "")
                name = _clean(match.group("name"))
                label = f"{tier} {name}".strip() if tier else name
                if label.lower().endswith("requirements"):
                    label = label[: -len("requirements")].strip()
                labels[index] = label
                continue

            if not text or text.startswith("•") or len(text) > 70:
                continue
            if text.endswith((".", ":", ";")) or _is_prose(text):
                continue
            # Must actually name a device.  Without this, running headers
            # ("PROPOSAL") and form labels ("PO Submit Password*") that happen
            # to sit above a bullet list were promoted to tiers.
            if not _DEVICE_RE.search(text):
                continue
            # accepted only with bullet evidence directly below
            following = 0
            for nxt in blocks[index + 1 : index + 3]:
                following += len(cls._bullets(nxt.text))
                if following >= 2:
                    break
            if following >= 2:
                labels[index] = _clean(text)
        return labels

    @staticmethod
    def _bullets(text: str) -> list[str]:
        items: list[str] = []
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            m = _BULLET_RE.match(line)
            if m:
                items.append(_clean(m.group("item")))
        return items

    # -- shape B: tabular line items -----------------------------------------

    def _extract_line_items(self, doc: Document) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for block in doc.iter_blocks():
            if block.kind is not BlockKind.TABLE or not block.table:
                continue
            # The product grid is rarely the first row: pdfplumber merges the
            # whole page into one table, so the real header must be located.
            header_index = self._find_header_row(block.table)
            if header_index is None:
                continue
            header = [c.lower() for c in block.table[header_index]]
            idx = {name: i for i, name in enumerate(header) if name}
            for row in block.table[header_index + 1 :]:
                if len(row) < 2:
                    continue
                name = _clean(row[0]) if len(row) > 0 else ""
                if not name:
                    continue
                name = re.sub(r"^\d{1,3}[.)]\s*", "", name)
                # contractual prose is not a line item
                if len(name) < 6 or len(name) > 90 or _is_prose(name):
                    continue
                entry: dict[str, Any] = {}
                for key, i in idx.items():
                    if i < len(row) and _clean(row[i]):
                        entry[key] = _clean(row[i])
                if entry:
                    result[name] = entry
        return result

    # -- shape C: pricing schedule ------------------------------------------

    @staticmethod
    def extract_pricing(doc: Document) -> dict[str, Any]:
        """Parse the price schedule (line / item / target quantity / unit).

        The plan asks for lines 1-20 of the Dallas ISD schedule.  Quantities are
        the single most decision-relevant number in a bid ("50,000 Each" versus
        "1 Each" changes whether responding is worth it), and they live only in
        this grid - never in the prose.
        """
        result: dict[str, Any] = {}
        for block in doc.iter_blocks():
            if block.kind is not BlockKind.TABLE or not block.table:
                continue
            header_index = ProductSpecExtractor._find_pricing_header(block.table)
            if header_index is None:
                continue
            columns = [c.lower().strip() for c in block.table[header_index]]
            line_i = _col(columns, "line")
            # grids differ: "Item", "Product Name", "Description" are all the
            # same logical column depending on the issuer.
            # NB: never chain these with `or` - column index 0 is falsy, so
            # `_col(...) or _col(...)` silently discards the first column.
            item_i = _col(columns, "item")
            if item_i is None:
                item_i = _col(columns, "product")
            if item_i is None:
                item_i = _col(columns, "description")
            qty_i = next(
                (i for i, c in enumerate(columns) if "quantity" in c or c == "qty"), None
            )
            unit_i = next((i for i, c in enumerate(columns) if c.startswith("unit")), None)
            if item_i is None or qty_i is None:
                continue
            has_line_col = line_i is not None
            for n, row in enumerate(block.table[header_index + 1 :], start=1):
                line = _clean(_cell(row, line_i)) if has_line_col else str(n)
                item = _clean(_cell(row, item_i))
                if not line:
                    continue
                if not item:
                    # Most schedules pack the number and description into one
                    # cell: "15.01 Tier 1 Small Student Chromebook Laptop".
                    packed = re.match(r"^(\d+(?:\.\d+)?)\s+(.+)$", line, re.S)
                    if not packed:
                        continue
                    line, item = packed.group(1), _clean(packed.group(2))
                if not has_line_col:
                    # numbering lives in the item cell ("1. SI# CC7802 ...")
                    numbered = re.match(r"^(\d+(?:\.\d+)?)\.\s+(.+)$", item)
                    if numbered:
                        line, item = numbered.group(1), _clean(numbered.group(2))
                qty = _clean(_cell(row, qty_i)) if qty_i is not None else ""
                unit = _clean(_cell(row, unit_i)) if unit_i is not None else ""
                # pdfplumber occasionally shifts data rows one column relative to
                # the header, so fall back to a nearby cell that *looks* like the
                # quantity / unit rather than trusting the index blindly.
                if not qty and qty_i is not None:
                    qty = _clean(_nearby(row, qty_i, _looks_like_quantity))
                if not unit and unit_i is not None:
                    unit = _clean(_nearby(row, unit_i, _looks_like_unit))
                if not qty and not unit:
                    # grouping header ("16 STUDENT LAPTOPS - FOR EVALUATION")
                    continue
                entry: dict[str, Any] = {"item": item}
                if qty:
                    entry["target_quantity"] = qty
                if unit:
                    entry["unit"] = unit
                # page 1 of the schedule repeats every line with quantity "1";
                # the real volumes appear in the 15.x-20.x block, so keep the
                # largest quantity seen for a given line.
                key = f"Line {line}"
                existing = result.get(key)
                if existing is None or _qty_value(qty) > _qty_value(existing.get("target_quantity", "")):
                    result[key] = entry
        return result

    @staticmethod
    def _find_pricing_header(table: list[list[str]]) -> int | None:
        """Locate a pricing grid header.

        Requires a quantity column.  A "Line" column is preferred (Dallas ISD
        numbers its lines 1-20), but grids that only have Product/Qty
        (the PORFP schedule) are accepted too.
        """
        for i, row in enumerate(table):
            lowered = [c.strip().lower() for c in row]
            # "quantity" must be a *header token*, not a word inside a prose
            # cell - the PORFP security-requirements table mentions "the
            # specified quantity" mid-sentence and was being matched.
            has_qty = any(
                c == "qty" or (0 < len(c) <= 30 and "quantity" in c) for c in lowered
            )
            has_item = any(
                c.startswith(("item", "product", "description")) for c in lowered
            )
            if has_qty and has_item:
                return i
        return None

    @staticmethod
    def _find_header_row(table: list[list[str]]) -> int | None:
        """Locate the row that heads a product grid (Product Name / Model / Qty)."""
        for i, row in enumerate(table):
            lowered = [c.lower() for c in row]
            if not any("product" in c for c in lowered):
                continue
            if any("model" in c for c in lowered) or any("qty" in c for c in lowered):
                return i
        return None


_PROSE_RE = re.compile(
    r"\b(shall|must|within|invoice|award|days|insurance|bond|agree|reserve)\b", re.I
)


def _is_prose(text: str) -> bool:
    """True for contractual sentences that are not equipment line items."""
    return bool(_PROSE_RE.search(text)) and len(text.split()) > 6


def _cell(row: list[str], index: int | None) -> str:
    if index is None or index >= len(row):
        return ""
    return row[index] or ""


def _col(columns: list[str], prefix: str) -> int | None:
    return next((i for i, c in enumerate(columns) if c.startswith(prefix)), None)


_QUANTITY_RE = re.compile(r"^[\d,]+$")
_UNITS = frozenset({"each", "ea", "unit", "units", "lot", "hour", "hours", "set", "box", "case"})


def _looks_like_quantity(cell: str) -> bool:
    return bool(_QUANTITY_RE.match((cell or "").strip()))


def _looks_like_unit(cell: str) -> bool:
    return (cell or "").strip().lower() in _UNITS


def _nearby(row: list[str], index: int, predicate, span: int = 3) -> str:
    """Search +/- ``span`` columns around ``index`` for a matching cell."""
    for offset in sorted(range(-span, span + 1), key=abs):
        i = index + offset
        if 0 <= i < len(row) and predicate(row[i] or ""):
            return row[i]
    return ""


def _qty_value(text: str) -> int:
    """Numeric value of a quantity cell ('50,000' -> 50000)."""
    digits = re.sub(r"[^\d]", "", text or "")
    return int(digits) if digits else 0


def spec_to_source_value(doc: Document, specs: dict[str, Any]) -> SourceValue:
    """Wrap extracted specs as a mergeable candidate with provenance."""
    pages = sorted({b.page_no for b in doc.iter_blocks() if b.text.strip()})[:3]
    return SourceValue(
        doc_id=doc.doc_id,
        file_name=doc.file_name,
        doc_type=doc.doc_type.value,
        precedence=doc.precedence,
        page=pages[0] if pages else 1,
        raw=f"{len(specs)} device/line-item specifications",
        normalized=specs,
        confidence=0.85,
        span="; ".join(list(specs)[:6]),
        method=ExtractionMethod.SECTION,
    )


def merge_specs(candidates: list[SourceValue]) -> dict[str, Any]:
    """Merge specs across documents: richer entries win, keys are unioned."""
    merged: dict[str, Any] = {}
    for cand in sorted(candidates, key=lambda c: (-c.precedence, -c.confidence)):
        specs = cand.normalized if isinstance(cand.normalized, dict) else {}
        for device, attrs in specs.items():
            existing = merged.get(device)
            if not isinstance(existing, dict):
                merged[device] = dict(attrs) if isinstance(attrs, dict) else {"value": attrs}
            elif isinstance(attrs, dict):
                for key, value in attrs.items():
                    existing.setdefault(key, value)
    return merged
