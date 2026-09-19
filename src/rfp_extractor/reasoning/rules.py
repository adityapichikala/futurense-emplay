"""Config-driven extraction engine (Pass 1).

For every field the YAML declares *how* a value can be recognised - a source
label, a regex, or a "find this phrase and take the whole sentence" rule.  The
engine walks the document's blocks, applies those rules, and emits
:class:`SourceValue` candidates that carry a verbatim span, a page number and a
confidence score.

Nothing here invents values: if no rule fires, the field simply has no candidate
and is later reported as "Not stated in source documents".
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, timezone
from typing import Any

from ..config import ExtractionConfig, FieldSpec, LabelRule, PatternRule
from ..models.document import Block, BlockKind, DocType, Document
from ..models.schema import ExtractionMethod, SourceValue
from ..validation.normalizers import (
    clean_text,
    normalize_datetime,
    sentence_containing,
    strip_leading_label,
    to_list,
)

log = logging.getLogger(__name__)

_DEFAULT_TZ = UTC


def _norm_label(label: str) -> str:
    """Canonical form for label matching: lowercase, punctuation-stripped."""
    text = label.strip().lower()
    text = re.sub(r"[^a-z0-9 /#&]+", "", text)
    return " ".join(text.split())


def _is_placeholder_value(label: str, value: str) -> bool:
    """Reject labels whose "value" is empty or just the label again.

    Signature sheets in addenda contain bare ``Company Name:`` / ``Email:``
    prompts with nothing filled in; treating those as the answer is the classic
    way to emit garbage for ``company_name``.
    """
    cleaned = (value or "").strip()
    if not cleaned:
        return True
    if _norm_label(cleaned.rstrip(":")) == _norm_label(label):
        return True
    return cleaned.lower() in {"n/a", "not stated", "none", "-", "--", "tbd"}


class RuleEngine:
    """Apply the YAML rule set to one document."""

    def __init__(self, config: ExtractionConfig, *, default_tz: timezone = _DEFAULT_TZ) -> None:
        self.config = config
        self.default_tz = default_tz
        self._regex_cache: dict[str, re.Pattern[str]] = {}

    # -- public ------------------------------------------------------------

    def extract(
        self, doc: Document, *, block_ids: set[str] | None = None
    ) -> dict[str, list[SourceValue]]:
        """Return per-field candidates found in ``doc``.

        ``block_ids`` narrows the search to blocks selected by retrieval (the
        RAG path).  When ``None`` every block is scanned (the recall-fallback
        path).
        """
        return {
            spec.name: self.extract_field(doc, spec, block_ids) for spec in self.config.fields
        }

    def extract_field(
        self, doc: Document, spec: FieldSpec, block_ids: set[str] | None = None
    ) -> list[SourceValue]:
        """Candidates for a single field - lets callers pass per-field retrieval."""
        candidates: list[SourceValue] = []
        candidates += self._from_labels(doc, spec, block_ids)
        candidates += self._from_patterns(doc, spec, block_ids)
        candidates += self._from_sentences(doc, spec, block_ids)
        return self._dedupe(candidates)

    # -- labels ------------------------------------------------------------

    def _from_labels(
        self, doc: Document, spec: FieldSpec, block_ids: set[str] | None
    ) -> list[SourceValue]:
        out: list[SourceValue] = []
        if not spec.labels:
            return out
        wanted: dict[str, LabelRule | None] = {}
        for rule in spec.labels:
            for label in rule.labels:
                wanted[_norm_label(label)] = rule

        for label, raw_value in doc.label_values.items():
            matched = wanted.get(_norm_label(label))
            if matched is None:
                continue
            if matched.doc_types and doc.doc_type.value not in matched.doc_types:
                continue
            if _is_placeholder_value(label, raw_value):
                continue
            out.append(
                self._make_candidate(
                    doc=doc,
                    spec=spec,
                    raw=raw_value,
                    confidence=matched.confidence,
                    method=ExtractionMethod.HTML_LABEL,
                    block=None,
                    span=f"{label}: {raw_value}"[:400],
                    transform=matched.transform,
                )
            )

        # PDF / DOCX form fields (label attached to a block)
        for block in doc.iter_blocks():
            if block_ids is not None and block.block_id not in block_ids:
                continue
            if block.kind is not BlockKind.LABEL_VALUE or not block.label:
                continue
            matched = wanted.get(_norm_label(block.label))
            if matched is None:
                continue
            if matched.doc_types and doc.doc_type.value not in matched.doc_types:
                continue
            value = block.meta.get("label_value") or ""
            if _is_placeholder_value(block.label, value):
                continue
            out.append(
                self._make_candidate(
                    doc=doc,
                    spec=spec,
                    raw=value,
                    confidence=matched.confidence - 0.03,
                    method=ExtractionMethod.PDF_LABEL,
                    block=block,
                    span=block.text[:400],
                    transform=matched.transform,
                )
            )
        return out

    # -- regex patterns ----------------------------------------------------

    def _from_patterns(
        self, doc: Document, spec: FieldSpec, block_ids: set[str] | None
    ) -> list[SourceValue]:
        out: list[SourceValue] = []
        for rule in spec.patterns:
            if rule.doc_types and doc.doc_type.value not in rule.doc_types:
                continue
            regex = self._compile(rule.regex)
            if regex is None:
                continue
            found = 0
            for block in self._iter_candidate_blocks(doc, rule, block_ids):
                for match in regex.finditer(block.text):
                    try:
                        raw = match.group(rule.group)
                    except (IndexError, re.error):
                        # named group absent from this pattern -> skip
                        continue
                    if raw is None:
                        continue
                    raw = raw.strip()
                    if not self._passes_guards(raw, rule):
                        continue
                    out.append(
                        self._make_candidate(
                            doc=doc,
                            spec=spec,
                            raw=raw,
                            confidence=rule.confidence,
                            method=ExtractionMethod.TABLE if block.kind is BlockKind.TABLE else ExtractionMethod.REGEX,
                            block=block,
                            span=self._span_around(block.text, match.start(), match.end()),
                            transform=rule.transform,
                            max_length=rule.max_length,
                        )
                    )
                    found += 1
                    if rule.multiplicity == "first":
                        break
                if found and rule.multiplicity == "first":
                    break
                if rule.multiplicity == "all" and found >= 500:
                    break
        return out

    # -- sentence patterns --------------------------------------------------

    def _from_sentences(
        self, doc: Document, spec: FieldSpec, block_ids: set[str] | None
    ) -> list[SourceValue]:
        out: list[SourceValue] = []
        for rule in spec.sentence_patterns:
            if rule.doc_types and doc.doc_type.value not in rule.doc_types:
                continue
            regex = self._compile(rule.regex)
            if regex is None:
                continue
            blocks = list(self._iter_candidate_blocks(doc, rule, block_ids))
            for index, block in enumerate(blocks):
                # the synthetic "label: value" summary block would prefix every
                # extracted sentence with its own label ("Description: ...")
                if block.meta.get("synthetic"):
                    continue
                match = regex.search(block.text)
                if not match:
                    continue
                sentence = sentence_containing(block.text, match.start(), match.end())
                if not sentence:
                    continue
                sentence = strip_leading_label(self._continue_sentence(blocks, index, sentence))
                out.append(
                    self._make_candidate(
                        doc=doc,
                        spec=spec,
                        raw=sentence,
                        confidence=rule.confidence,
                        method=ExtractionMethod.SECTION,
                        block=block,
                        span=sentence[:400],
                        transform=rule.transform or "clean",
                        max_length=rule.max_length,
                    )
                )
                break
        return out

    @staticmethod
    def _continue_sentence(blocks: list[Block], index: int, sentence: str) -> str:
        """Re-join a sentence that a PDF line break split across blocks.

        "...from awarded" / "vendors starting in September of 2024." would
        otherwise be truncated at the page's line wrap.  We only extend when the
        next block starts lowercase - the signature of a true continuation
        rather than a new paragraph.
        """
        text = sentence.strip()
        if text.endswith((".", "!", "?", ":")):
            return text
        for nxt in blocks[index + 1 : index + 4]:
            if nxt.page_no != blocks[index].page_no:
                break
            extra = nxt.text.strip()
            if not extra or not extra[:1].islower():
                break
            text = f"{text} {extra}"
            if text.endswith((".", "!", "?")):
                break
        return " ".join(text.split())[:400]

    # -- helpers ------------------------------------------------------------

    def _iter_candidate_blocks(
        self, doc: Document, rule: PatternRule, block_ids: set[str] | None
    ):
        for block in doc.iter_blocks():
            if block_ids is not None and block.block_id not in block_ids:
                continue
            if rule.sections and block.section not in rule.sections:
                continue
            if not block.text.strip():
                continue
            yield block

    def _compile(self, pattern: str) -> re.Pattern[str] | None:
        cached = self._regex_cache.get(pattern)
        if cached is not None:
            return cached
        try:
            compiled = re.compile(pattern, re.I | re.S)
        except re.error as exc:
            log.warning("invalid regex %r: %s", pattern[:80], exc)
            self._regex_cache[pattern] = None  # type: ignore[assignment]
            return None
        self._regex_cache[pattern] = compiled
        return compiled

    @staticmethod
    def _passes_guards(raw: str, rule: PatternRule) -> bool:
        if not raw:
            return False
        if rule.max_length and len(raw) > rule.max_length * 3:
            return False
        if rule.must_match and not re.search(rule.must_match, raw):
            return False
        if rule.must_not_match and re.search(rule.must_not_match, raw):
            return False
        return raw.lower() not in {"n/a", "not stated", "none", "-", "--"}

    def _make_candidate(
        self,
        *,
        doc: Document,
        spec: FieldSpec,
        raw: Any,
        confidence: float,
        method: ExtractionMethod,
        block: Block | None,
        span: str,
        transform: str | None,
        max_length: int | None = None,
    ) -> SourceValue:
        normalized = self._apply_transform(raw, spec, transform, max_length)
        confidence = self._adjust_confidence(doc, spec, block, confidence, normalized)
        return SourceValue(
            doc_id=doc.doc_id,
            file_name=doc.file_name,
            doc_type=doc.doc_type.value,
            precedence=doc.precedence,
            page=block.page_no if block else 1,
            raw=raw if isinstance(raw, (str, int, float)) else str(raw),
            normalized=normalized,
            confidence=round(max(0.0, min(1.0, confidence)), 3),
            span=" ".join(str(span).split())[:400],
            method=method,
            block_id=block.block_id if block else None,
        )

    def _apply_transform(
        self,
        raw: Any,
        spec: FieldSpec,
        transform: str | None,
        max_length: int | None,
    ) -> Any:
        if raw is None:
            return None
        if transform == "datetime" or (spec.kind == "datetime" and transform is None):
            iso = normalize_datetime(raw, default_tz=self.default_tz)
            return iso if iso else clean_text(raw, max_length=max_length)
        if transform == "list" or spec.kind == "list":
            return to_list(raw)
        if transform == "sentence":
            return clean_text(raw, max_length=max_length or 400)
        return clean_text(raw, max_length=max_length)

    def _adjust_confidence(
        self,
        doc: Document,
        spec: FieldSpec,
        block: Block | None,
        confidence: float,
        normalized: Any,
    ) -> float:
        score = confidence
        if normalized in (None, "", [], {}):
            score *= 0.5
        if block is not None:
            # header/page-1 bias: solicitation metadata lives at the top
            if block.page_no <= 2 and spec.kind in {"string", "datetime"}:
                score += 0.03
            if block.kind is BlockKind.TABLE:
                score += 0.02
            if block.kind is BlockKind.LABEL_VALUE:
                score += 0.02
        if doc.doc_type is DocType.ADDENDUM:
            score += 0.04  # addenda are deliberate, explicit corrections
        # very short values for prose fields are weak evidence
        if spec.kind == "string" and isinstance(normalized, str) and len(normalized) < 3:
            score -= 0.15
        return score

    @staticmethod
    def _span_around(text: str, start: int, end: int, *, window: int = 160) -> str:
        left = max(0, start - window // 2)
        right = min(len(text), end + window // 2)
        snippet = text[left:right]
        if left > 0:
            snippet = "..." + snippet
        if right < len(text):
            snippet = snippet + "..."
        return " ".join(snippet.split())

    @staticmethod
    def _dedupe(candidates: list[SourceValue]) -> list[SourceValue]:
        seen: set[tuple[str, str, int | None]] = set()
        out: list[SourceValue] = []
        for cand in candidates:
            key = (cand.signature, cand.method.value, cand.page)
            if key in seen:
                continue
            seen.add(key)
            out.append(cand)
        out.sort(key=lambda c: (-c.confidence, c.page or 0))
        return out
