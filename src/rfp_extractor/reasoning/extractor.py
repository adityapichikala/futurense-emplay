"""Two-pass extraction orchestrator.

**Pass 1 (per document).**  For each document, retrieve the chunks most relevant
to a field (hybrid BM25 + vector), then run the rule engine over just those
blocks.  If retrieval finds nothing, we fall back to scanning the whole document
so recall never silently depends on the retriever.

**Pass 2 (merge).**  All per-document candidates for a field are merged using the
document precedence graph.  Genuine disagreements become
:class:`ConflictResolution` records; agreements become corroborations that raise
confidence.  Nothing is overwritten in silence.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from datetime import UTC, datetime, timezone
from typing import Any

from ..config import ExtractionConfig, FieldSpec, Settings
from ..intelligence.graph import BidPackage
from ..models.document import Document
from ..models.schema import (
    CANONICAL_FIELDS,
    ConflictResolution,
    Evidence,
    ExtractionResult,
    FieldValue,
    RunMetadata,
    SourceValue,
)
from ..validation.normalizers import prune_subsumed
from .chunking import SemanticChunker
from .providers.base import ExtractionProvider
from .retriever import HybridRetriever
from .rules import RuleEngine
from .spec_extractor import (
    ProductSpecExtractor,
    _qty_value,
    merge_specs,
    spec_to_source_value,
)

log = logging.getLogger(__name__)


class ExtractionEngine:
    """Run Pass 1 + Pass 2 over one bid package."""

    def __init__(
        self,
        config: ExtractionConfig,
        settings: Settings | None = None,
        *,
        default_tz: timezone = UTC,
        provider: ExtractionProvider | None = None,
    ) -> None:
        self.config = config
        self.settings = settings
        self.default_tz = default_tz
        self.rule_engine = RuleEngine(config, default_tz=default_tz)
        self.spec_extractor = ProductSpecExtractor()
        #: Optional LLM provider.  When present its candidates are merged with
        #: the rule engine's, so the model fills gaps rather than overriding
        #: high-confidence deterministic evidence.
        self.provider = provider

    # -- pass 1 ------------------------------------------------------------

    def _field_query(self, spec: FieldSpec) -> str:
        labels = " ".join(label for rule in spec.labels for label in rule.labels)
        return f"{spec.alias} {labels} {spec.description}"

    def extract_documents(self, package: BidPackage) -> dict[str, list[SourceValue]]:
        """Pass 1: candidates per field across all documents in the package."""
        docs = package.documents
        chunker = SemanticChunker(
            target_tokens=(self.settings.chunk_size_tokens if self.settings else 420),
            overlap_tokens=(self.settings.chunk_overlap_tokens if self.settings else 60),
        )
        chunks = chunker.chunk_documents(docs)
        retriever = HybridRetriever(
            chunks,
            bm25_weight=(self.settings.bm25_weight if self.settings else 0.6),
            vector_weight=(self.settings.vector_weight if self.settings else 0.4),
        )
        top_k = self.settings.retrieval_top_k if self.settings else 6

        blocks_by_doc: dict[str, dict[str, set[str]]] = defaultdict(dict)
        for spec in self.config.fields:
            hits = retriever.search(self._field_query(spec), top_k=top_k)
            per_doc: dict[str, set[str]] = defaultdict(set)
            for hit in hits:
                per_doc[hit.chunk.doc_id].update(hit.chunk.block_ids)
            for doc_id, ids in per_doc.items():
                blocks_by_doc[doc_id][spec.name] = ids

        merged: dict[str, list[SourceValue]] = defaultdict(list)
        for doc in docs:
            doc_results = self._extract_single(doc, blocks_by_doc.get(doc.doc_id, {}))
            if self.provider is not None:
                doc_results = self._augment_with_provider(doc, doc_results)
            for field, candidates in doc_results.items():
                field_spec: FieldSpec | None = next(
                    (s for s in self.config.fields if s.name == field), None
                )
                for cand in candidates:
                    merged[field].append(self._effective_precedence(cand, field_spec))
        return dict(merged)

    @staticmethod
    def _effective_precedence(cand: SourceValue, spec: FieldSpec | None) -> SourceValue:
        """Vendor-side fields are best answered by vendor documents.

        ``company_name`` / ``model_no`` / ``part_no`` describe the *bidder*, not
        the issuer, so a vendor quote outranks the agency's own PDF for those
        fields even though the agency PDF wins overall.
        """
        if spec is not None and spec.scope == "vendor" and cand.doc_type == "vendor_spec":
            cand.precedence += 200
        return cand

    def _augment_with_provider(
        self, doc: Document, results: dict[str, list[SourceValue]]
    ) -> dict[str, list[SourceValue]]:
        """Ask the LLM only for fields the rules did not answer confidently.

        Calling the model on every field of every document would be slow and
        would put a stochastic source in competition with deterministic evidence
        we already trust.  Instead the model is a *fallback*: it is consulted
        where rules are silent, and its confidence (0.7 by default) keeps it
        below a confident rule match during merge.
        """
        assert self.provider is not None
        threshold = (self.settings.min_confidence if self.settings else 0.35) + 0.3
        for spec in self.config.fields:
            existing = results.get(spec.name) or []
            if any(c.confidence >= threshold for c in existing):
                continue
            try:
                extra = self.provider.extract_field(doc, spec)
            except Exception as exc:  # noqa: BLE001 - provider must never break a run
                log.warning("provider failed for %s/%s: %s", doc.file_name, spec.name, exc)
                continue
            if extra:
                results[spec.name] = existing + extra
        return results

    def _extract_single(
        self, doc: Document, retrieved: dict[str, set[str]]
    ) -> dict[str, list[SourceValue]]:
        # narrow each field to its retrieved blocks; fields with no retrieval
        # hit fall back to a full scan.
        per_field: dict[str, list[SourceValue]] = {}
        for spec in self.config.fields:
            # Aggregate fields (SKU / model lists) are scattered across dozens of
            # blocks; narrowing them to top-k chunks silently loses most values.
            # Identifiers are cheap to scan for, so these always get a full pass.
            block_ids = None if spec.kind == "list" else (retrieved.get(spec.name) or None)
            candidates = self.rule_engine.extract_field(doc, spec, block_ids)
            if not candidates and block_ids is not None:
                # recall safety net: retrieval missed, scan the whole document
                candidates = self.rule_engine.extract_field(doc, spec)
                for cand in candidates:
                    cand.confidence = round(cand.confidence * 0.95, 3)
            per_field[spec.name] = candidates

        specs = self.spec_extractor.extract(doc)
        if specs:
            per_field["product_specification"] = [spec_to_source_value(doc, specs)]
        return per_field

    @staticmethod
    def _merge_pricing(package: BidPackage) -> dict[str, Any]:
        """Union the price schedule across documents, richest quantity winning."""
        merged: dict[str, Any] = {}
        for doc in sorted(package.documents, key=lambda d: -d.precedence):
            schedule = ProductSpecExtractor.extract_pricing(doc)
            for line, entry in schedule.items():
                existing = merged.get(line)
                if existing is None or _qty_value(entry.get("target_quantity", "")) > _qty_value(
                    existing.get("target_quantity", "")
                ):
                    merged[line] = entry
        return merged

    def extract_single_document(self, doc: Document) -> dict[str, FieldValue]:
        """Pass 1 only, for one document: what this file says on its own."""
        per_field = self._extract_single(doc, {})
        return self.merge(per_field)

    # -- pass 2 ------------------------------------------------------------

    def merge(self, candidates: dict[str, list[SourceValue]]) -> dict[str, FieldValue]:
        resolved: dict[str, FieldValue] = {}
        for spec in self.config.fields:
            field_candidates = candidates.get(spec.name, [])
            resolved[spec.name] = self._merge_field(spec, field_candidates)
        return resolved

    def _merge_field(self, spec: FieldSpec, candidates: list[SourceValue]) -> FieldValue:
        usable = [c for c in candidates if not _is_empty(c)]
        if not usable:
            return FieldValue(
                field=spec.name,
                present=False,
                null_reason=spec.null_reason,
                candidates=[],
            )

        if spec.merge == "union":
            return self._merge_union(spec, usable)
        if spec.merge == "concat":
            return self._merge_concat(spec, usable)
        if spec.merge == "longest":
            return self._merge_longest(spec, usable)
        if spec.merge == "most_specific":
            return self._merge_most_specific(spec, usable)
        return self._merge_precedence(spec, usable)

    # -- merge strategies ---------------------------------------------------

    def _merge_precedence(self, spec: FieldSpec, candidates: list[SourceValue]) -> FieldValue:
        # one representative per distinct value, keeping the strongest source
        best_by_signature: dict[str, SourceValue] = {}
        for cand in candidates:
            key = cand.signature
            current = best_by_signature.get(key)
            if current is None or _rank(cand) > _rank(current):
                best_by_signature[key] = cand

        ordered = sorted(best_by_signature.values(), key=_rank, reverse=True)
        winner = ordered[0]
        losers = ordered[1:]

        distinct = len(ordered)
        if distinct > 1:
            # same instant expressed differently is corroboration, not conflict
            if _same_instant(winner.normalized, losers[0].normalized):
                return self._build_field(
                    spec,
                    winner,
                    corroborating=losers,
                    conflict=None,
                    strategy="corroboration",
                )
            conflict = ConflictResolution(
                field=spec.name,
                winner=winner,
                losers=losers,
                strategy="precedence",
                rationale=(
                    f"{winner.file_name} (precedence {winner.precedence}, "
                    f"{winner.doc_type}) overrides {losers[0].file_name} "
                    f"(precedence {losers[0].precedence}, {losers[0].doc_type})."
                ),
            )
            return self._build_field(spec, winner, [], conflict=conflict, strategy="precedence")

        return self._build_field(spec, winner, [], strategy="single_source")

    def _merge_union(self, spec: FieldSpec, candidates: list[SourceValue]) -> FieldValue:
        values: list[str] = []
        seen: set[str] = set()
        ordered = sorted(candidates, key=_rank, reverse=True)
        for cand in ordered:
            items = cand.normalized if isinstance(cand.normalized, list) else [cand.normalized]
            for item in items:
                if item is None:
                    continue
                text = str(item).strip()
                key = text.lower()
                if key and key not in seen:
                    seen.add(key)
                    values.append(text)
        values = prune_subsumed(values)
        winner = ordered[0].model_copy(update={"normalized": values, "raw": values})
        winner.confidence = round(min(1.0, max(c.confidence for c in ordered)), 3)
        return self._build_field(spec, winner, ordered[1:4], strategy="union")

    def _merge_concat(self, spec: FieldSpec, candidates: list[SourceValue]) -> FieldValue:
        parts: list[str] = []
        seen: set[str] = set()
        ordered = sorted(candidates, key=_rank, reverse=True)
        for cand in ordered:
            text = str(cand.normalized or cand.raw or "").strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            # skip near-duplicates (one sentence contained in another)
            if any(key in existing or existing in key for existing in seen):
                continue
            seen.add(key)
            parts.append(text)
        combined = "; ".join(parts[:6])
        winner = ordered[0].model_copy(update={"normalized": combined, "raw": combined})
        winner.confidence = round(min(1.0, max(c.confidence for c in ordered)), 3)
        return self._build_field(spec, winner, ordered[1:4], strategy="concat")

    def _merge_longest(self, spec: FieldSpec, candidates: list[SourceValue]) -> FieldValue:
        ordered = sorted(candidates, key=_rank, reverse=True)
        winner = max(ordered, key=lambda c: len(str(c.normalized or c.raw or "")))
        losers = [c for c in ordered if c is not winner][:3]
        return self._build_field(spec, winner, losers, strategy="longest")

    def _merge_most_specific(self, spec: FieldSpec, candidates: list[SourceValue]) -> FieldValue:
        ordered = sorted(candidates, key=_rank, reverse=True)
        merged = merge_specs(ordered)
        if not merged:
            return FieldValue(field=spec.name, present=False, null_reason=spec.null_reason)
        winner = ordered[0].model_copy(update={"normalized": merged})
        winner.confidence = round(min(1.0, max(c.confidence for c in ordered)), 3)
        return self._build_field(spec, winner, ordered[1:3], strategy="most_specific")

    # -- assembly -----------------------------------------------------------

    def _build_field(
        self,
        spec: FieldSpec,
        winner: SourceValue,
        corroborating: list[SourceValue],
        *,
        conflict: ConflictResolution | None = None,
        strategy: str,
    ) -> FieldValue:
        confidence = winner.confidence
        if corroborating:
            # independent agreement is the strongest signal we have
            confidence = min(1.0, confidence + 0.03 * len(corroborating))
        evidence = [
            Evidence(
                doc_id=c.doc_id,
                file_name=c.file_name,
                page=c.page,
                block_id=c.block_id,
                span=c.span,
                confidence=c.confidence,
                method=c.method,
                extraction_pass="pass1",
            )
            for c in [winner, *corroborating[:3]]
        ]
        return FieldValue(
            field=spec.name,
            value=winner.raw,
            normalized=winner.normalized,
            confidence=round(confidence, 3),
            present=not _is_empty(winner),
            null_reason=None if not _is_empty(winner) else spec.null_reason,
            evidence=evidence,
            candidates=[winner, *corroborating],
            conflict=conflict,
        )

    # -- package entry point -------------------------------------------------

    def run(self, package: BidPackage, metadata: RunMetadata) -> ExtractionResult:
        # Dates printed without a timezone ("Solicitation Due 27-JUN-2024 14:00:00")
        # must be read in the issuer's zone, otherwise 14:00 local collides with
        # 14:00Z and we invent a conflict that does not exist.
        self.default_tz = infer_package_timezone(package.documents) or self.default_tz
        self.rule_engine.default_tz = self.default_tz
        candidates = self.extract_documents(package)
        fields = self.merge(candidates)

        result = ExtractionResult(
            bid_id=package.package_id,
            bid_number=package.bid_number,
            metadata=metadata,
        )
        for name in CANONICAL_FIELDS:
            fv = fields.get(name)
            if fv is None:
                fv = FieldValue(field=name, present=False, null_reason="Not stated in source documents")
            result.set_field(name, fv)

        result.pricing_schedule = self._merge_pricing(package) or None
        result.conflicts_resolved = [
            fv.conflict for fv in result.fields.values() if fv.conflict is not None
        ]
        if package.bid_number and not result.bid_number:
            result.bid_number = package.bid_number
        return result


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _rank(candidate: SourceValue) -> tuple[int, float]:
    """Sort key: precedence first, then confidence."""
    return (candidate.precedence, candidate.confidence)


def _is_empty(candidate: SourceValue) -> bool:
    value = candidate.normalized if candidate.normalized is not None else candidate.raw
    return value in (None, "", [], {})


def _same_instant(a: object, b: object) -> bool:
    from ..validation.normalizers import to_iso_instant

    if not isinstance(a, str) or not isinstance(b, str):
        return False
    ia, ib = to_iso_instant(a), to_iso_instant(b)
    return ia is not None and ia == ib


_ZONE_RE = re.compile(r"\b(EST|EDT|CST|CDT|MST|MDT|PST|PDT)\b")


def infer_package_timezone(docs: list[Document]) -> timezone | None:
    """Infer the issuer's timezone from explicit abbreviations in the corpus.

    A naive timestamp is ambiguous, and guessing wrong turns agreement into a
    phantom conflict: the Dallas ISD master RFP prints "10-JUN-2024 14:00:00"
    (Central), while the BidNet portal prints "06/10/2024 03:00 PM EDT".  Read
    the naive value in the wrong zone and the two look like a disagreement.

    Portal notices are excluded from the vote: their timestamps always carry an
    explicit offset, so they never need a default - and they would otherwise
    outvote the issuer's own documents purely by repetition.
    """
    issuer_docs = [d for d in docs if d.doc_type.value != "portal_notice"] or docs
    counts: dict[str, int] = {}
    for doc in issuer_docs:
        for match in _ZONE_RE.finditer(doc.full_text):
            abbr = match.group(1).upper()
            counts[abbr] = counts.get(abbr, 0) + 1
    if not counts:
        return None
    best = max(counts, key=lambda a: counts[a])
    from datetime import datetime as _dt

    from ..validation.normalizers import _resolve_offset

    resolved = _resolve_offset(best, _dt(2024, 6, 15, 12, 0))
    return resolved


def build_run_metadata(provider: str = "rule-based", model: str | None = None) -> RunMetadata:
    return RunMetadata(started_at=datetime.now(), provider=provider, model=model)


def describe_provider(engine: ExtractionEngine) -> str:
    """Label the run with the provider actually in use."""
    if engine.provider is None:
        return "rule-based"
    name = getattr(engine.provider, "name", "unknown")
    return f"rule-based+{name}" if name != "openai" else "hybrid"


def collect_usage(engine: ExtractionEngine) -> tuple[int, float]:
    """Token/cost totals from the provider, or (0, 0.0) when rule-based."""
    usage = getattr(engine.provider, "usage", None)
    if not usage:
        return 0, 0.0
    return int(usage.get("tokens", 0)), float(usage.get("usd", 0.0))
