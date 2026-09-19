"""Artifact rendering: per-package JSON, consolidated JSON, provenance report.

The JSON is the deliverable; the markdown report is the audit trail.  Every row
in the citation table points at a file, a page and a verbatim quote, so a human
can verify any value in seconds without re-running the pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models.schema import (
    CANONICAL_FIELDS,
    FIELD_ALIASES,
    ConsolidatedResult,
    DocumentExtraction,
    ExtractionResult,
)
from .validation.evaluation import EvaluationReport  # noqa: TC001 - used in type hints


class ArtifactWriter:
    """Write extraction artifacts to a directory."""

    def __init__(self, output_dir: str | Path = "artifacts") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # -- JSON --------------------------------------------------------------

    def write_package(self, result: ExtractionResult) -> Path:
        path = self.output_dir / f"extraction_{result.bid_id}.json"
        payload = {
            "bid_id": result.bid_id,
            **result.to_assignment_json(),
            "pricing_schedule": result.pricing_schedule,
            "documents": [d.model_dump() for d in result.documents],
            "fields": {
                name: {
                    "value": result.fields[name].normalized,
                    "confidence": result.fields[name].confidence,
                    "present": result.fields[name].present,
                    "null_reason": result.fields[name].null_reason,
                    "evidence": [e.model_dump() for e in result.fields[name].evidence],
                }
                for name in CANONICAL_FIELDS
                if name in result.fields
            },
            "conflicts_resolved": [c.model_dump() for c in result.conflicts_resolved],
            "coverage": dict(zip(("populated", "total"), result.coverage(), strict=False)),
            "metadata": result.metadata.model_dump(mode="json"),
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def write_consolidated(self, result: ConsolidatedResult) -> Path:
        path = self.output_dir / "extraction_all.json"
        payload = {
            "packages": {k: v.to_dict() for k, v in result.packages.items()},
            "metadata": result.metadata.model_dump(mode="json"),
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def write_document(self, extraction: DocumentExtraction) -> Path:
        """Per-document JSON - the assignment asks for one per input file."""
        directory = self.output_dir / "per_document"
        directory.mkdir(parents=True, exist_ok=True)
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in extraction.doc_id)
        path = directory / f"{safe}.json"
        payload = {
            "doc_id": extraction.doc_id,
            "file_name": extraction.file_name,
            "source_path": extraction.source_path,
            "file_format": extraction.file_format,
            "doc_type": extraction.doc_type,
            "precedence": extraction.precedence,
            "page_count": extraction.page_count,
            "addendum_number": extraction.addendum_number,
            "amendment_notes": extraction.amendment_notes,
            "populated_fields": extraction.populated_count,
            "values": extraction.values,
            "fields": {
                name: {
                    "value": fv.normalized,
                    "confidence": fv.confidence,
                    "present": fv.present,
                    "null_reason": fv.null_reason,
                    "evidence": [e.model_dump() for e in fv.evidence],
                }
                for name, fv in extraction.fields.items()
            },
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def write_documents(self, extractions: list[DocumentExtraction]) -> list[Path]:
        return [self.write_document(e) for e in extractions]

    def write_evaluation(self, report: EvaluationReport) -> Path:
        path = self.output_dir / "evaluation_report.json"
        path.write_text(json.dumps(report.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    # -- markdown -----------------------------------------------------------

    def write_provenance_report(
        self, result: ConsolidatedResult, evaluation: EvaluationReport | None = None
    ) -> Path:
        path = self.output_dir / "provenance_report.md"
        lines: list[str] = []
        lines.append("# RFP Extraction - Provenance Report")
        lines.append("")
        lines.append(
            "Every value below is traceable to a source file, page and verbatim quote."
        )
        lines.append("")

        if evaluation is not None:
            lines.append(f"**Field-level accuracy: {evaluation.accuracy * 100:.1f}%**")
            lines.append("")
            for pkg in evaluation.packages:
                lines.append(f"- `{pkg.package}` {pkg.name}: {pkg.passed}/{pkg.total} ({pkg.accuracy * 100:.1f}%)")
            lines.append("")

        for package_id, package in result.packages.items():
            filled, total = package.coverage()
            lines.append("---")
            lines.append("")
            lines.append(f"## Package `{package_id}` - {package.bid_number or 'unknown bid number'}")
            lines.append("")
            lines.append(f"Coverage: **{filled}/{total}** fields populated.")
            lines.append("")

            lines.append("### Documents (by precedence)")
            lines.append("")
            lines.append("| Precedence | Document | Type | Pages | Addendum | Confidence |")
            lines.append("|---:|---|---|---:|---|---:|")
            for doc in package.documents:
                lines.append(
                    f"| {doc.precedence} | `{_short(doc.file_name, 60)}` | {doc.doc_type} "
                    f"| {doc.page_count} | {doc.addendum_number or '-'} | {doc.doc_type_confidence:.2f} |"
                )
            lines.append("")

            lines.append("### Extracted fields and citations")
            lines.append("")
            lines.append("| Field | Value | Conf. | Source | Page | Verbatim evidence |")
            lines.append("|---|---|---:|---|---:|---|")
            for name in CANONICAL_FIELDS:
                fv = package.fields.get(name)
                if fv is None:
                    continue
                value = fv.display
                display = _fmt_value(value)
                if fv.evidence:
                    ev = fv.evidence[0]
                    source = f"`{_short(ev.file_name, 34)}`"
                    page = str(ev.page or "-")
                    quote = _md_escape(_short(ev.span, 110))
                else:
                    source, page, quote = "-", "-", f"_{fv.null_reason or 'not found'}_"
                lines.append(
                    f"| {FIELD_ALIASES.get(name, name)} | {display} | {fv.confidence:.2f} "
                    f"| {source} | {page} | {quote} |"
                )
            lines.append("")

            if package.conflicts_resolved:
                lines.append("### Conflicts resolved")
                lines.append("")
                for conflict in package.conflicts_resolved:
                    lines.append(f"- **{FIELD_ALIASES.get(conflict.field, conflict.field)}** "
                                 f"({conflict.strategy}): {conflict.rationale}")
                    lines.append(f"  - winner: `{_fmt_value(conflict.winner.normalized)}` "
                                 f"from `{_short(conflict.winner.file_name, 50)}`")
                    for loser in conflict.losers:
                        lines.append(f"  - superseded: `{_fmt_value(loser.normalized)}` "
                                     f"from `{_short(loser.file_name, 50)}`")
                lines.append("")

        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def write_all(
        self,
        result: ConsolidatedResult,
        evaluation: EvaluationReport | None = None,
        document_extractions: list[DocumentExtraction] | None = None,
    ) -> dict[str, Path]:
        paths: dict[str, Path] = {}
        for package in result.packages.values():
            paths[f"extraction_{package.bid_id}"] = self.write_package(package)
        if document_extractions:
            written = self.write_documents(document_extractions)
            paths[f"per_document ({len(written)} files)"] = written[0].parent
        paths["extraction_all"] = self.write_consolidated(result)
        paths["provenance_report"] = self.write_provenance_report(result, evaluation)
        if evaluation is not None:
            paths["evaluation_report"] = self.write_evaluation(evaluation)
        return paths


def _fmt_value(value: Any) -> str:
    if value is None or value == [] or value == {}:
        return "_null_"
    if isinstance(value, list):
        preview = ", ".join(str(v) for v in value[:6])
        more = f" (+{len(value) - 6} more)" if len(value) > 6 else ""
        return _md_escape(f"[{preview}{more}]")
    if isinstance(value, dict):
        keys = list(value)[:4]
        more = f" (+{len(value) - 4} more)" if len(value) > 4 else ""
        return _md_escape("{" + ", ".join(str(k) for k in keys) + more + "}")
    return _md_escape(_short(str(value), 120))


def _short(text: str, limit: int) -> str:
    text = str(text)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _md_escape(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")
