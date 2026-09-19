"""Field-level evaluation against the hand-labelled gold set.

Accuracy is measured per field, not per document: an extraction that gets 18 of
19 fields right on a package should score ~0.95, not 0.  Each check is scored
1.0 / 0.0 and we report micro-averaged accuracy, plus per-field detail so a
regression is traceable to the rule that caused it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..models.schema import CANONICAL_FIELDS, ConsolidatedResult, ExtractionResult
from .normalizers import to_iso_instant

log = logging.getLogger(__name__)

# .../src/rfp_extractor/validation/evaluation.py -> parents[3] == repo root
GOLD_SET_PATH = Path(__file__).resolve().parents[3] / "configs" / "gold_set.yaml"


@dataclass
class FieldScore:
    field: str
    passed: bool
    expected: Any
    actual: Any
    check: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "passed": self.passed,
            "check": self.check,
            "expected": self.expected,
            "actual": _short(self.actual),
        }


@dataclass
class PackageScore:
    package: str
    name: str = ""
    scores: list[FieldScore] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(1 for s in self.scores if s.passed)

    @property
    def total(self) -> int:
        return len(self.scores)

    @property
    def accuracy(self) -> float:
        return self.passed / self.total if self.total else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "package": self.package,
            "name": self.name,
            "accuracy": round(self.accuracy, 4),
            "passed": self.passed,
            "total": self.total,
            "failed_fields": [s.field for s in self.scores if not s.passed],
            "fields": [s.as_dict() for s in self.scores],
        }


@dataclass
class EvaluationReport:
    packages: list[PackageScore] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        total = sum(p.total for p in self.packages)
        passed = sum(p.passed for p in self.packages)
        return passed / total if total else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "overall_accuracy": round(self.accuracy, 4),
            "total_checks": sum(p.total for p in self.packages),
            "total_passed": sum(p.passed for p in self.packages),
            "packages": [p.as_dict() for p in self.packages],
        }


def load_gold_set(path: str | Path | None = None) -> dict[str, Any]:
    with open(Path(path) if path else GOLD_SET_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def evaluate(
    result: ConsolidatedResult, gold_set: dict[str, Any] | None = None
) -> EvaluationReport:
    gold = gold_set or load_gold_set()
    report = EvaluationReport()
    unmatched: list[str] = []
    by_number = {
        str(entry.get("bid_number")): entry
        for entry in gold.values()
        if entry.get("bid_number")
    }

    for package_id, package in result.packages.items():
        spec = gold.get(package_id)
        if spec is None and package.bid_number:
            # Package ids are directory names, so they change when the corpus is
            # laid out differently (a flat folder splits into per-solicitation
            # packages named after the bid).  Matching only on package id made a
            # *correct* extraction report 0%, which is worse than no score at
            # all - so fall back to the solicitation number.
            spec = by_number.get(package.bid_number)
        if spec is None:
            unmatched.append(package_id)
            continue
        score = PackageScore(package=package_id, name=spec.get("name", ""))
        expectations: dict[str, dict[str, Any]] = spec.get("expectations", {})
        for field_name in CANONICAL_FIELDS:
            expectation = expectations.get(field_name)
            if expectation is None:
                continue
            score.scores.append(_score_field(package, field_name, expectation))
        report.packages.append(score)
    if unmatched:
        log.warning(
            "no gold expectations for package(s) %s (known: %s) - "
            "these are excluded from the accuracy figure",
            ", ".join(unmatched),
            ", ".join(gold),
        )
    return report


def _score_field(
    package: ExtractionResult, field_name: str, expectation: dict[str, Any]
) -> FieldScore:
    actual = getattr(package, field_name, None)

    if "absent" in expectation or "null" in expectation:
        should_be_null = expectation.get("absent", expectation.get("null", True))
        passed = (actual in (None, [], {}, "")) == bool(should_be_null)
        return FieldScore(field_name, passed, "absent" if should_be_null else "present",
                          actual, "absent")

    if "equals" in expectation:
        passed = str(actual).strip().lower() == str(expectation["equals"]).strip().lower()
        return FieldScore(field_name, passed, expectation["equals"], actual, "equals")

    if "instant" in expectation:
        actual_instant = to_iso_instant(str(actual)) if actual else None
        passed = actual_instant == expectation["instant"]
        return FieldScore(field_name, passed, expectation["instant"], actual, "instant")

    if "contains" in expectation:
        needle = str(expectation["contains"]).lower()
        passed = needle in str(actual or "").lower()
        return FieldScore(field_name, passed, needle, actual, "contains")

    if "min_items" in expectation:
        size = len(actual) if isinstance(actual, (list, dict)) else 0
        passed = size >= int(expectation["min_items"])
        return FieldScore(field_name, passed, f">= {expectation['min_items']}",
                          f"{size} items", "min_items")

    raise ValueError(f"unsupported expectation for {field_name}: {expectation}")


def _short(value: Any, limit: int = 90) -> Any:
    if isinstance(value, (list, dict)):
        return f"<{type(value).__name__} len={len(value)}>"
    text = str(value)
    return text[:limit] + ("..." if len(text) > limit else "")
