"""Ingestion tests: known strings must survive parsing.

These are the regression guards for the two parsing bugs found on this corpus:
SKUs glued to the preceding word, and BidNet fields nested inside a <form>.
"""

from __future__ import annotations

import pytest

from rfp_extractor.ingestion.normalize import deconcatenate_labels, normalize_text
from rfp_extractor.ingestion.router import FileRouter
from rfp_extractor.models.document import BlockKind, DocType, FileFormat


@pytest.fixture(scope="module")
def router() -> FileRouter:
    return FileRouter()


def test_html_parses_bidnet_label_grid(bid2_dir, router):
    doc = router.load_one(next(bid2_dir.glob("*.html")))
    assert doc is not None
    assert doc.file_format is FileFormat.HTML
    assert doc.label_values.get("Solicitation Number") == "BPM044557"
    assert doc.label_values["Closing Date"] == "06/10/2024 02:00 PM EDT"


def test_html_parses_bid1_portal_fields(bid1_dir, router):
    doc = router.load_one(next(bid1_dir.glob("*.html")))
    assert doc.label_values["Solicitation Number"] == "JA-207652"
    assert doc.label_values["Closing Date"] == "07/09/2024 03:00 PM EDT"
    assert doc.label_values["Prebid Conference"] == "06/10/2024 03:00 PM EDT"


def test_pdf_skUs_survive_layout_parsing(bid2_dir, router):
    """'XCTO Base210-BLYZ' - the SKU is glued to the previous word."""
    doc = router.load_one(bid2_dir / "Dell_Laptop_Specs.pdf")
    text = doc.full_text
    for sku in ("210-BLYZ", "379-BFNZ", "383-0464"):
        assert sku in text


def test_pdf_header_dates_extracted(bid1_dir, router):
    doc = router.load_one(bid1_dir / "JA-207652 Student and Staff Computing Devices FINAL.pdf")
    text = doc.full_text
    assert "27-JUN-2024" in text          # stale due date in the master RFP
    assert "10-JUN-2024" in text          # pre-proposal meeting
    assert "three (3) year agreement" in text


def test_addendum_detected_in_text(bid1_dir, router):
    doc = router.load_one(bid1_dir / "Addendum 2 RFP JA-207652 Student and Staff Computing Devices.pdf")
    assert "July 9, 2024" in doc.full_text
    assert "ADDENDUM No. 2" in doc.full_text


def test_pdf_label_value_pairing(bid2_dir, router):
    """PORFP forms put the label on one line and the value on the next."""
    doc = router.load_one(bid2_dir / "PORFP_-_Dell_Laptop_Final.pdf")
    labelled = {b.label: b.meta.get("label_value") for b in doc.iter_blocks() if b.label}
    assert labelled.get("PORFP Number") == "#E20P4600040"
    # On form pages the label and its answer sit in adjacent table columns that
    # share a baseline, so the merged visual line can carry neighbouring cells.
    # Containment is the guarantee here, not exact equality - see the note in
    # PdfLoader about gap-based line splitting.
    poc = labelled.get("Agency POC Name") or ""
    assert "Tamaira Hawkins" in poc
    assert "410-260-7533" in poc


def test_label_pairing_ignores_empty_signature_prompts(bid1_dir, router):
    """'Company Name:' with nothing after it must not yield 'Submitter's Name/Title'."""
    doc = router.load_one(bid1_dir / "Addendum 2 RFP JA-207652 Student and Staff Computing Devices.pdf")
    values = [b.meta.get("label_value") for b in doc.iter_blocks() if b.label]
    assert not any(v and v.endswith(":") for v in values), values


def test_deconcatenate_labels_splits_glued_pairs():
    assert deconcatenate_labels("BuyerALZATE, JASMINE") == "Buyer ALZATE, JASMINE"
    assert "Email JALZATE@dallasisd.org" in deconcatenate_labels("JASMINEEmailJALZATE@dallasisd.org")


def test_normalize_text_collapses_whitespace():
    # within-line whitespace collapses; line breaks are preserved because they
    # carry layout meaning downstream (bullets, table rows).
    assert normalize_text("a   b\tc") == "a b c"
    assert normalize_text("a\r\n   b") == "a\n b"


def test_router_dedups_identical_content(tmp_path, router, bid2_dir):
    source = bid2_dir / "Mercury_Affidavit.pdf"
    copy = tmp_path / "Mercury_Affidavit.pdf"
    copy.write_bytes(source.read_bytes())
    docs = router.load_many([source, copy])
    assert len(docs) == 1


def test_tables_are_extracted_as_blocks(bid2_dir, router):
    doc = router.load_one(bid2_dir / "PORFP_-_Dell_Laptop_Final.pdf")
    tables = [b for b in doc.iter_blocks() if b.kind is BlockKind.TABLE]
    assert tables, "expected at least one pdfplumber table block"


def test_grouping_and_types(bid1_dir, bid2_dir, router):
    from rfp_extractor.intelligence.classifier import DocumentClassifier

    classifier = DocumentClassifier()
    docs = router.load_directory(bid1_dir)
    for doc in docs:
        classifier.annotate(doc)
    types = {d.doc_type for d in docs}
    assert DocType.MASTER_RFP in types
    assert DocType.ADDENDUM in types
    assert DocType.PORTAL_NOTICE in types

    addenda = [d for d in docs if d.doc_type is DocType.ADDENDUM]
    assert sorted(d.addendum_number for d in addenda) == [1, 2]
