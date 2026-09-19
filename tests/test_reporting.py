"""Tests for the artifact writer.

`reporting.py` generates every deliverable (package JSON, per-document JSON,
provenance report, evaluation report).  It was previously at 0% measured
coverage because the only thing exercising it was a subprocess CLI run, whose
coverage is not collected — leaving the code that produces the output files
entirely unguarded.
"""

from __future__ import annotations

import json

from rfp_extractor.models.schema import CANONICAL_FIELDS, FIELD_ALIASES
from rfp_extractor.reporting import ArtifactWriter
from rfp_extractor.validation.evaluation import evaluate


def test_write_package_emits_assignment_shape_and_provenance(tmp_path, consolidated):
    writer = ArtifactWriter(tmp_path)
    path = writer.write_package(consolidated.packages["bid1"])
    payload = json.loads(path.read_text(encoding="utf-8"))

    # the assignment's own field names, at the top level
    for field in CANONICAL_FIELDS:
        assert FIELD_ALIASES[field] in payload
    # provenance alongside
    assert payload["documents"], "expected the document manifest"
    assert "conflicts_resolved" in payload
    assert payload["coverage"] == {"populated": 15, "total": 20} or payload["coverage"]["total"] == 20
    for entry in payload["fields"].values():
        assert "confidence" in entry and "evidence" in entry


def test_write_consolidated_contains_both_packages(tmp_path, consolidated):
    writer = ArtifactWriter(tmp_path)
    path = writer.write_consolidated(consolidated)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert set(payload["packages"]) == {"bid1", "bid2"}


def test_write_evaluation(tmp_path, consolidated):
    writer = ArtifactWriter(tmp_path)
    report = evaluate(consolidated)
    path = writer.write_evaluation(report)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["overall_accuracy"] == report.accuracy
    assert {p["package"] for p in payload["packages"]} == {"bid1", "bid2"}


def test_provenance_report_cites_sources_and_conflicts(tmp_path, consolidated):
    writer = ArtifactWriter(tmp_path)
    path = writer.write_provenance_report(consolidated, evaluate(consolidated))
    text = path.read_text(encoding="utf-8")

    assert "Field-level accuracy" in text
    assert "JA-207652" in text and "BPM044557" in text
    # citation table header
    assert "| Field | Value | Conf. | Source | Page | Verbatim evidence |" in text
    # the due-date override is documented
    assert "Conflicts resolved" in text
    assert "2024-07-09" in text


def test_write_documents(tmp_path, pipeline_run):
    pipeline, _ = pipeline_run
    writer = ArtifactWriter(tmp_path)
    paths = writer.write_documents(pipeline.document_extractions)
    assert len(paths) == 9
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["file_name"]
        assert "values" in payload and "fields" in payload


def test_write_all_creates_every_artifact(tmp_path, pipeline_run, consolidated):
    pipeline, _ = pipeline_run
    writer = ArtifactWriter(tmp_path)
    written = writer.write_all(consolidated, evaluate(consolidated), pipeline.document_extractions)
    assert (tmp_path / "extraction_all.json").is_file()
    assert (tmp_path / "provenance_report.md").is_file()
    assert (tmp_path / "evaluation_report.json").is_file()
    assert (tmp_path / "per_document").is_dir()
    assert (tmp_path / "extraction_bid1.json").is_file()
    assert len(written) >= 5
