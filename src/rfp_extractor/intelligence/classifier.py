"""Document-type classification: rules first, with a scored fallback.

Why rules rather than an LLM call for every file: the classes here are
determined by unmistakable lexical markers ("ADDENDUM No. 2", "Affidavit",
"Closing Date" portal grids).  Spending tokens on that is waste, and a rule is
auditable.  The classifier records which signals fired so a human can see *why*
a document was typed the way it was - that audit trail is what makes the
precedence decisions downstream defensible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..models.document import DocType, Document, FileFormat


@dataclass
class Signal:
    name: str
    weight: float
    detail: str = ""


@dataclass
class ClassificationResult:
    doc_type: DocType
    confidence: float
    signals: list[Signal] = field(default_factory=list)
    addendum_number: int | None = None


_ADDENDUM_RE = re.compile(r"addendum\s*(?:no\.?|#|number)?\s*(\d+)", re.I)
_AFFIDAVIT_RE = re.compile(r"\baffidavit\b", re.I)
_RFP_RE = re.compile(r"\brequest\s+for\s+proposal\b|\bRFP\b|\bPORFP\b|\bsolicitation\b", re.I)
_SKU_RE = re.compile(r"\bSKU\b|\bpart\s*number\b|\bXCTO\b|\bQuote\b", re.I)
_VENDOR_RE = re.compile(r"dell|latitude|optiplex|hp\b|lenovo|apple\s+inc", re.I)
_PORTAL_RE = re.compile(r"bidnet|mets-field|closing date|solicitation number", re.I)
_ASSIGNMENT_RE = re.compile(r"\bobjective\b.*\bparticipants\b|\bdeliverables\b", re.I | re.S)


class DocumentClassifier:
    """Score-based classifier over filename + content signals."""

    def classify(self, doc: Document) -> ClassificationResult:
        scores: dict[DocType, float] = {t: 0.0 for t in DocType}
        signals: list[Signal] = []
        addendum_number: int | None = None

        haystack = doc.file_name + "\n" + doc.full_text[:20000]
        head = doc.full_text[:4000]
        is_html = doc.file_format is FileFormat.HTML

        # --- portal notice (HTML grids) -----------------------------------
        if is_html:
            signals.append(Signal("html_source", 1.5, "HTML source"))
            scores[DocType.PORTAL_NOTICE] += 1.5
            if doc.label_values:
                scores[DocType.PORTAL_NOTICE] += 2.0
                signals.append(Signal("label_value_grid", 2.0, f"{len(doc.label_values)} label/value pairs"))
            if _PORTAL_RE.search(haystack):
                scores[DocType.PORTAL_NOTICE] += 1.5
                signals.append(Signal("portal_markers", 1.5, "BidNet/mets-field markers"))

        # --- addendum ------------------------------------------------------
        m = _ADDENDUM_RE.search(haystack)
        if m:
            addendum_number = int(m.group(1))
            weight = 4.0 if "addendum" in doc.file_name.lower() else 2.0
            scores[DocType.ADDENDUM] += weight
            signals.append(Signal("addendum_marker", weight, f"Addendum No. {addendum_number}"))

        # --- legal affidavit ----------------------------------------------
        if _AFFIDAVIT_RE.search(doc.file_name) or _AFFIDAVIT_RE.search(head):
            scores[DocType.LEGAL_AFFIDAVIT] += 4.0
            signals.append(Signal("affidavit_marker", 4.0, "Affidavit"))

        # --- vendor spec / quote -------------------------------------------
        sku_hits = len(_SKU_RE.findall(head))
        if sku_hits:
            weight = min(3.0, 1.0 + 0.35 * sku_hits)
            scores[DocType.VENDOR_SPEC] += weight
            signals.append(Signal("sku_markers", round(weight, 2), f"{sku_hits} SKU/quote markers"))
        if _VENDOR_RE.search(doc.file_name):
            scores[DocType.VENDOR_SPEC] += 1.0
            signals.append(Signal("vendor_filename", 1.0, "vendor name in filename"))

        # --- master RFP -----------------------------------------------------
        if _RFP_RE.search(head):
            scores[DocType.MASTER_RFP] += 2.0
            signals.append(Signal("rfp_markers", 2.0, "RFP/solicitation language"))
        if re.search(r"solicitation\s+due|proposal\s+due|due\s+date", head, re.I):
            scores[DocType.MASTER_RFP] += 1.5
            signals.append(Signal("due_date_block", 1.5, "solicitation due-date block"))
        if doc.page_count >= 10:
            scores[DocType.MASTER_RFP] += 1.0
            signals.append(Signal("long_document", 1.0, f"{doc.page_count} pages"))

        # --- assignment brief -----------------------------------------------
        if _ASSIGNMENT_RE.search(head):
            scores[DocType.ASSIGNMENT] += 5.0
            signals.append(Signal("assignment_markers", 5.0, "assignment brief language"))

        best = max(scores, key=lambda t: scores[t])
        best_score = scores[best]
        if best_score <= 0:
            return ClassificationResult(DocType.UNKNOWN, 0.0, signals, addendum_number)

        # confidence = margin over runner-up, saturating at 1.0
        runner_up = max((v for k, v in scores.items() if k is not best), default=0.0)
        margin = best_score - runner_up
        confidence = min(1.0, 0.55 + 0.15 * margin + 0.1 * min(best_score, 6))
        return ClassificationResult(best, round(confidence, 3), signals, addendum_number)

    def annotate(self, doc: Document) -> Document:
        result = self.classify(doc)
        doc.doc_type = result.doc_type
        doc.doc_type_confidence = result.confidence
        doc.doc_type_signals = [f"{s.name}={s.weight}" for s in result.signals]
        doc.addendum_number = result.addendum_number
        return doc
