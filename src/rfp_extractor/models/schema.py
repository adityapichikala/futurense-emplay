"""The 19-field RFP extraction contract, plus the provenance envelope.

The assignment defines exactly 19 output fields (see ``data/raw/Assignment.docx``).
``CANONICAL_FIELDS`` is the single source of truth: config, the rule engine, the
API response model and the evaluation harness all derive from it, so a field can
never exist in one layer and be missing from another.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..validation.normalizers import to_iso_instant

# ---------------------------------------------------------------------------
# Canonical field contract
# ---------------------------------------------------------------------------

CANONICAL_FIELDS: tuple[str, ...] = (
    "bid_number",
    "title",
    "due_date",
    "bid_submission_type",
    "term_of_bid",
    "pre_bid_meeting",
    "installation",
    "bid_bond_requirement",
    "delivery_date",
    "payment_terms",
    "additional_documentation_required",
    "mfg_for_registration",
    "contract_or_cooperative_to_use",
    "model_no",
    "part_no",
    "product",
    "contact_info",
    "company_name",
    "bid_summary",
    "product_specification",
)

#: Fields whose value is a list of strings.
LIST_FIELDS: frozenset[str] = frozenset({"model_no", "part_no"})

#: Fields whose canonical value is an ISO-8601 datetime string.
DATETIME_FIELDS: frozenset[str] = frozenset({"due_date", "pre_bid_meeting", "delivery_date"})

#: Fields that are free-form prose rather than short scalars.
PROSE_FIELDS: frozenset[str] = frozenset({"bid_summary", "installation"})

FIELD_ALIASES: dict[str, str] = {
    "bid_number": "Bid Number",
    "title": "Title",
    "due_date": "Due Date",
    "bid_submission_type": "Bid Submission Type",
    "term_of_bid": "Term of Bid",
    "pre_bid_meeting": "Pre Bid Meeting",
    "installation": "Installation",
    "bid_bond_requirement": "Bid Bond Requirement",
    "delivery_date": "Delivery Date",
    "payment_terms": "Payment Terms",
    "additional_documentation_required": "Any Additional Documentation Required",
    "mfg_for_registration": "MFG for Registration",
    "contract_or_cooperative_to_use": "Contract or Cooperative to use",
    "model_no": "Model_no",
    "part_no": "Part_no",
    "product": "Product",
    "contact_info": "contact_info",
    "company_name": "company_name",
    "bid_summary": "Bid Summary",
    "product_specification": "Product Specification",
}


class ExtractionMethod(str, Enum):
    HTML_LABEL = "html_label"
    PDF_LABEL = "pdf_label"
    REGEX = "regex"
    TABLE = "table"
    SECTION = "section"
    LLM = "llm"
    DERIVED = "derived"


class NullReason(str, Enum):
    NOT_STATED = "Not stated in source documents"
    NOT_APPLICABLE = "Not applicable to this bid package"
    CONFLICT_UNRESOLVED = "Conflicting values with no decisive precedence"
    LOW_CONFIDENCE = "Only low-confidence candidates found"


# ---------------------------------------------------------------------------
# Provenance primitives
# ---------------------------------------------------------------------------


class Evidence(BaseModel):
    """A verbatim citation: where a value came from, and how sure we are."""

    model_config = ConfigDict(frozen=True)

    doc_id: str
    file_name: str
    page: int | None = None
    block_id: str | None = None
    span: str = ""
    confidence: float = 1.0
    method: ExtractionMethod = ExtractionMethod.REGEX
    extraction_pass: str = "pass1"
    section: str | None = None

    @field_validator("confidence")
    @classmethod
    def _clamp(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))


class SourceValue(BaseModel):
    """One document's opinion about one field."""

    doc_id: str
    file_name: str
    doc_type: str
    precedence: int
    page: int | None = None
    raw: Any = None
    normalized: Any = None
    confidence: float = 0.0
    span: str = ""
    method: ExtractionMethod = ExtractionMethod.REGEX
    block_id: str | None = None

    @property
    def signature(self) -> str:
        """Identity used for conflict detection (format-independent)."""
        return _value_signature(self.normalized if self.normalized is not None else self.raw)


def _value_signature(value: Any) -> str:
    """Canonical signature so equivalent values compare equal.

    Dates are compared by *instant*: 'July 9, 2024 at 2:00 PM CST' and
    '07/09/2024 03:00 PM EDT' are the same moment and must not be reported as a
    conflict (Addendum 2 vs the BidNet portal in package A is exactly this case).
    """
    if value is None:
        return ""
    if isinstance(value, list):
        return "|".join(sorted(_value_signature(v) for v in value))
    if isinstance(value, dict):
        return "|".join(f"{k}={_value_signature(v)}" for k, v in sorted(value.items()))
    text = str(value).strip().lower()
    text = " ".join(text.split())
    instant = to_iso_instant(text)
    return instant or text


class ConflictResolution(BaseModel):
    field: str
    winner: SourceValue
    losers: list[SourceValue] = Field(default_factory=list)
    strategy: str = "precedence"
    rationale: str = ""


class FieldValue(BaseModel):
    """A resolved field: the answer, *and* the audit trail behind it."""

    field: str
    value: Any = None
    normalized: Any = None
    confidence: float = 0.0
    present: bool = False
    null_reason: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)
    candidates: list[SourceValue] = Field(default_factory=list)
    conflict: ConflictResolution | None = None

    @property
    def display(self) -> Any:
        return self.normalized if self.normalized is not None else self.value


class DocumentRef(BaseModel):
    doc_id: str
    file_name: str
    file_format: str
    doc_type: str
    precedence: int
    page_count: int
    content_hash: str
    addendum_number: int | None = None
    doc_type_confidence: float = 0.0
    amendment_notes: list[str] = Field(default_factory=list)


class RunMetadata(BaseModel):
    started_at: datetime
    finished_at: datetime | None = None
    duration_seconds: float | None = None
    provider: str = "rule-based"
    model: str | None = None
    config_version: str = "1.0.0"
    pipeline_version: str = "1.0.0"
    tokens_used: int = 0
    estimated_cost_usd: float = 0.0
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Top-level result
# ---------------------------------------------------------------------------


class ExtractionResult(BaseModel):
    """Per-bid-package result.

    Serializes the 19 assignment fields at the top level (so the JSON matches the
    requested shape) and nests provenance under ``_provenance``.
    """

    model_config = ConfigDict(validate_assignment=False)

    bid_id: str
    bid_number: str | None = None
    title: str | None = None
    due_date: str | None = None
    bid_submission_type: str | None = None
    term_of_bid: str | None = None
    pre_bid_meeting: str | None = None
    installation: str | None = None
    bid_bond_requirement: str | None = None
    delivery_date: str | None = None
    payment_terms: str | None = None
    additional_documentation_required: str | None = None
    mfg_for_registration: str | None = None
    contract_or_cooperative_to_use: str | None = None
    model_no: list[str] | None = None
    part_no: list[str] | None = None
    product: str | None = None
    contact_info: str | None = None
    company_name: str | None = None
    bid_summary: str | None = None
    product_specification: dict[str, Any] | None = None

    #: Supplementary: the price schedule (line / item / target quantity / unit).
    #: Not one of the 19 assignment fields - quantities are decision-relevant
    #: but have no home in the field contract, so they are reported alongside it.
    pricing_schedule: dict[str, Any] | None = None

    documents: list[DocumentRef] = Field(default_factory=list)
    fields: dict[str, FieldValue] = Field(default_factory=dict)
    conflicts_resolved: list[ConflictResolution] = Field(default_factory=list)
    metadata: RunMetadata

    # --- helpers -----------------------------------------------------------

    def set_field(self, name: str, fv: FieldValue) -> None:
        self.fields[name] = fv
        if name in LIST_FIELDS:
            setattr(self, name, fv.normalized if isinstance(fv.normalized, list) else None)
        elif name == "product_specification":
            setattr(self, name, fv.normalized if isinstance(fv.normalized, dict) else None)
        else:
            value = fv.normalized if fv.normalized is not None else fv.value
            setattr(self, name, value if value is None else str(value))

    @property
    def populated(self) -> dict[str, Any]:
        return {f: getattr(self, f) for f in CANONICAL_FIELDS if getattr(self, f) not in (None, [], {})}

    def coverage(self) -> tuple[int, int]:
        filled = sum(
            1 for f in CANONICAL_FIELDS if getattr(self, f) not in (None, [], {}, "")
        )
        return filled, len(CANONICAL_FIELDS)

    def to_assignment_json(self) -> dict[str, Any]:
        """Exactly the shape requested by the assignment: 19 fields -> values."""
        return {FIELD_ALIASES[f]: getattr(self, f) for f in CANONICAL_FIELDS}

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=False)


class DocumentExtraction(BaseModel):
    """Pass-1 result for a *single* document.

    The assignment asks for structured information "extracted from each provided
    document", which is a different artifact from the package-level merge: here
    we report what each file individually says, before supersession is applied.
    Comparing the two is exactly how you see which addendum changed what.
    """

    doc_id: str
    file_name: str
    source_path: str
    file_format: str
    doc_type: str
    precedence: int
    page_count: int
    addendum_number: int | None = None
    amendment_notes: list[str] = Field(default_factory=list)
    fields: dict[str, FieldValue] = Field(default_factory=dict)
    values: dict[str, Any] = Field(default_factory=dict)

    @property
    def populated_count(self) -> int:
        return sum(1 for fv in self.fields.values() if fv.present)


class ConsolidatedResult(BaseModel):
    packages: dict[str, ExtractionResult] = Field(default_factory=dict)
    metadata: RunMetadata

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


Confidence = Annotated[float, Field(ge=0.0, le=1.0)]
