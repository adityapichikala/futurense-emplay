"""Amendment detection: what changed, and when it takes effect.

An addendum is not just "another PDF" - it is a *delta* against the master RFP.
We capture three things:

1. **Identity** - Addendum No. N (so addenda can be ordered).
2. **Supersession clauses** - sentences that explicitly override earlier text
   ("The new due date for this RFP will be ...", "Dallas ISD will only require
   etching on Laptops").  These become high-precedence evidence.
3. **Q&A pairs** - Addendum 1 is 39 question/answer items; each answer is an
   authoritative clarification of the master document.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..models.document import Document

_ADDENDUM_NO_RE = re.compile(r"addendum\s*(?:no\.?|#|number)?\s*(\d+)", re.I)

#: Sentences that explicitly replace earlier content.
_SUPERSESSION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"[^.]*\b(?:new|revised|changed|updated|extended)\s+due\s+date\b[^.]*\.", re.I),
    re.compile(r"[^.]*\b(?:hereby\s+)?(?:supersede|replace|override|amend)s?\b[^.]*\.", re.I),
    re.compile(r"[^.]*\b(?:will|shall)\s+(?:now\s+)?(?:be|only\s+require|not\s+require)\b[^.]*\.", re.I),
    re.compile(r"[^.]*\bthe\s+purpose\s+of\s+this\s+addendum\s+is\b[^.]*\.", re.I),
)

_DUE_DATE_CHANGE_RE = re.compile(
    r"(?:new|revised|extended)\s+due\s+date[^.]*?(?P<date>[A-Z][a-z]+\s+\d{1,2},\s*\d{4}[^,.;]*(?:\d{1,2}:\d{2}\s*(?:AM|PM))?)",
    re.I | re.S,
)

#: "N.Question text" followed by "Answer:response"
_QA_RE = re.compile(
    r"(?P<num>\d{1,3})\s*[.)]\s*(?P<question>.*?)(?P<answer>Answer\s*:\s*.*?)(?=(?:\n\s*\d{1,3}\s*[.)])|END OF ADDENDUM|\Z)",
    re.I | re.S,
)


@dataclass
class Amendment:
    addendum_number: int | None
    supersession_notes: list[str]
    qa_pairs: list[tuple[str, str]]
    changed_fields: dict[str, str]


class AddendumDetector:
    """Pull amendment semantics out of a document."""

    def detect(self, doc: Document) -> Amendment:
        text = doc.full_text
        number: int | None = None
        m = _ADDENDUM_NO_RE.search(doc.file_name) or _ADDENDUM_NO_RE.search(text[:1500])
        if m:
            number = int(m.group(1))

        notes: list[str] = []
        for pattern in _SUPERSESSION_PATTERNS:
            for match in pattern.finditer(text):
                snippet = " ".join(match.group(0).split())
                if 25 <= len(snippet) <= 400:
                    notes.append(snippet)

        qa_pairs: list[tuple[str, str]] = []
        for match in _QA_RE.finditer(text):
            question = " ".join(match.group("question").split())
            answer = " ".join(match.group("answer").split())
            if len(question) < 20:
                continue
            qa_pairs.append((question, answer))

        changed: dict[str, str] = {}
        dm = _DUE_DATE_CHANGE_RE.search(text)
        if dm:
            changed["due_date"] = " ".join(dm.group("date").split())
            doc.addendum_effective_date = changed["due_date"]

        return Amendment(
            addendum_number=number,
            supersession_notes=_dedupe(notes),
            qa_pairs=qa_pairs,
            changed_fields=changed,
        )

    def annotate(self, doc: Document) -> Document:
        amendment = self.detect(doc)
        if amendment.addendum_number is not None and doc.addendum_number is None:
            doc.addendum_number = amendment.addendum_number
        doc.amendment_notes = amendment.supersession_notes[:8]
        doc.meta["qa_pairs"] = [
            {"question": q[:400], "answer": a[:600]} for q, a in amendment.qa_pairs[:60]
        ]
        doc.meta["amended_fields"] = amendment.changed_fields
        return doc


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out
