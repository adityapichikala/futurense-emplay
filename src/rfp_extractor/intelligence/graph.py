"""The document graph: precedence ordering within a bid package.

This is the component that makes the pipeline addendum-aware.  Every document
gets an integer precedence; when two documents disagree about a field, the
higher-precedence document wins and the disagreement is recorded rather than
silently dropped.

Default policy (configurable via ``configs/fields.yaml``)::

    Addendum N (N large)  >  ...  >  Addendum 1  >  Portal notice  >  Master RFP
                                                  >  Vendor spec  >  Legal affidavit

Portal notices outrank the master RFP because in this corpus the portal mirrors
the *current* state of the solicitation (its closing date already reflects
Addendum 2), while the master PDF is the as-issued snapshot.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..config import PrecedenceSpec
from ..models.document import DocType, Document

_DEFAULT_PRECEDENCE: dict[str, int] = {
    DocType.MASTER_RFP.value: 40,
    DocType.PORTAL_NOTICE.value: 60,
    DocType.VENDOR_SPEC.value: 25,
    DocType.LEGAL_AFFIDAVIT.value: 20,
    DocType.ADDENDUM.value: 100,
    DocType.ASSIGNMENT.value: 0,
    DocType.UNKNOWN.value: 10,
}

_BID_NUMBER_RE = re.compile(
    r"\b(?:JA-\d{5,7}|BPM\d{6,9}|[A-Z]{2}\d{6,10}|#?E\d{2}P\d{7}|\d{9,12})\b"
)


@dataclass
class BidPackage:
    """A set of documents describing one solicitation."""

    package_id: str
    bid_number: str | None = None
    documents: list[Document] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def addenda(self) -> list[Document]:
        return sorted(
            (d for d in self.documents if d.doc_type is DocType.ADDENDUM),
            key=lambda d: d.addendum_number or 0,
        )

    @property
    def master(self) -> Document | None:
        for d in self.documents:
            if d.doc_type is DocType.MASTER_RFP:
                return d
        return None

    def by_precedence(self) -> list[Document]:
        return sorted(self.documents, key=lambda d: (-d.precedence, d.file_name))


class DocumentGraph:
    """Assign precedence, resolve the canonical bid number, build packages."""

    def __init__(self, precedence: PrecedenceSpec | None = None) -> None:
        spec = precedence or PrecedenceSpec(doc_types=dict(_DEFAULT_PRECEDENCE))
        merged = dict(_DEFAULT_PRECEDENCE)
        merged.update(spec.doc_types)
        self.weights = merged
        self.addendum_base = spec.addendum_base
        self.addendum_step = spec.addendum_step

    # -- precedence --------------------------------------------------------

    def precedence_for(self, doc: Document) -> int:
        base = self.weights.get(doc.doc_type.value, 10)
        if doc.doc_type is DocType.ADDENDUM:
            # later addenda outrank earlier ones
            return self.addendum_base + self.addendum_step * (doc.addendum_number or 1)
        return base

    def assign(self, docs: list[Document]) -> list[Document]:
        for doc in docs:
            doc.precedence = self.precedence_for(doc)
        return docs

    # -- packages ----------------------------------------------------------

    def build_package(self, package_id: str, docs: list[Document]) -> BidPackage:
        self.assign(docs)
        pkg = BidPackage(package_id=package_id, documents=list(docs))
        pkg.bid_number = self.resolve_bid_number(docs)
        if not pkg.master:
            pkg.warnings.append("no master RFP detected in package")
        return pkg

    def resolve_bid_number(self, docs: list[Document]) -> str | None:
        """Choose the bid number with the widest corroboration and best source.

        Portal notices and addendum headers quote the solicitation number
        explicitly; the master RFP sometimes buries it in a title line.
        """
        candidates: list[tuple[int, int, str]] = []  # (source_weight, count, value)
        for doc in docs:
            values = self._bid_candidates(doc)
            if not values:
                continue
            for value, count in values.items():
                weight = self.weights.get(doc.doc_type.value, 10)
                if doc.doc_type is DocType.PORTAL_NOTICE:
                    weight += 30  # portal 'Solicitation Number' is authoritative
                candidates.append((weight, count, value))
        if not candidates:
            return None
        best = max(candidates, key=lambda c: (c[1], c[0]))
        return best[2]

    @staticmethod
    def _bid_candidates(doc: Document) -> dict[str, int]:
        counts: dict[str, int] = {}
        preferred = doc.label_values.get("Solicitation Number") or doc.label_values.get(
            "Reference Number"
        )
        if preferred:
            counts[preferred.strip()] = counts.get(preferred.strip(), 0) + 5

        # eMMA project numbers and PORFP numbers appear as labelled form fields
        for label in ("eMMA Project Number", "PORFP Number", "PORFP #"):
            value = doc.label_values.get(label)
            if value:
                value = value.strip().lstrip("#")
                counts[value] = counts.get(value, 0) + 4

        text = doc.full_text[:20000]
        for match in _BID_NUMBER_RE.finditer(text):
            value = match.group(0).strip()
            counts[value] = counts.get(value, 0) + 1
        return counts
