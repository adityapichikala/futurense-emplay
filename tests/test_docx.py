"""DOCX loader and logging bootstrap tests.

Both were near-zero coverage: the DOCX loader is only reachable through the
assignment brief, and `configure_logging` only ran inside the CLI script.
"""

from __future__ import annotations

import logging

from rfp_extractor.ingestion.docx_loader import DocxLoader
from rfp_extractor.logging_setup import configure_logging
from rfp_extractor.models.document import BlockKind, FileFormat

ASSIGNMENT = "data/raw/Assignment.docx"


def test_docx_parses_the_assignment_brief():
    doc = DocxLoader().load(ASSIGNMENT)
    assert doc.file_format is FileFormat.DOCX
    assert doc.page_count == 1
    assert doc.blocks, "expected extracted blocks"

    text = doc.full_text
    for expected in ("Objective", "Bid Number", "Due Date", "company_name", "Product Specification"):
        assert expected in text, expected


def test_docx_extracts_the_field_table_as_label_values():
    """The brief lists the 19 fields in a two-column table."""
    doc = DocxLoader().load(ASSIGNMENT)
    assert "Bid Number" in doc.label_values
    assert "Product Specification" in doc.label_values
    # values are empty in the template, so they are recorded as such
    assert doc.label_values["Bid Number"].strip() == ""

    tables = [b for b in doc.blocks if b.kind is BlockKind.TABLE]
    assert tables, "expected the field table to be extracted"


def test_docx_inline_labels_are_paired():
    doc = DocxLoader().load(ASSIGNMENT)
    labelled = [b for b in doc.blocks if b.label]
    assert labelled, "expected inline 'Label: value' pairs"
    for block in labelled:
        assert block.meta.get("label_value") is not None


def test_configure_logging_sets_the_level():
    try:
        configure_logging("WARNING")
        assert logging.getLogger().level <= logging.WARNING
        configure_logging("DEBUG")
        assert logging.getLogger().level <= logging.DEBUG
    finally:
        configure_logging("INFO")


def test_configure_logging_is_idempotent():
    configure_logging("INFO")
    configure_logging("INFO")  # must not raise (structlog caches loggers)
