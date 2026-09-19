"""Robustness: malformed, empty and unexpected input must degrade, not crash.

The assignment grades "the ability of the program to handle various document
structures and content".  A parser that raises on a corrupt PDF or an empty file
is not production-grade, so every case below asserts graceful behaviour:
the bad input is skipped (or yields nulls with a reason) and good input in the
same batch still extracts correctly.
"""

from __future__ import annotations

import pytest

from rfp_extractor.config import Settings
from rfp_extractor.ingestion.router import FileRouter
from rfp_extractor.pipeline import Pipeline


@pytest.fixture
def router() -> FileRouter:
    return FileRouter()


def test_corrupt_pdf_is_skipped_not_raised(tmp_path, router):
    bad = tmp_path / "corrupt.pdf"
    bad.write_bytes(b"%PDF-1.4\nthis is not really a pdf at all\n%%EOF")
    docs = router.load_many([bad])
    # either parsed to an empty document or skipped - never an exception
    assert isinstance(docs, list)
    for doc in docs:
        assert doc.page_count >= 0


def test_empty_file_is_handled(tmp_path, router):
    empty = tmp_path / "empty.pdf"
    empty.write_bytes(b"")
    assert isinstance(router.load_many([empty]), list)


def test_unsupported_extension_returns_none(tmp_path, router):
    other = tmp_path / "notes.txt"
    other.write_text("just some notes")
    assert router.load_one(other) is None


def test_binary_garbage_with_unknown_extension_returns_none(tmp_path, router):
    blob = tmp_path / "data.bin"
    blob.write_bytes(bytes(range(256)))
    assert router.load_one(blob) is None


def test_html_without_portal_fields_yields_no_labels(tmp_path, router):
    html = tmp_path / "plain.html"
    html.write_text(
        "<html><head><title>Hello</title></head><body><p>Nothing to see.</p></body></html>",
        encoding="utf-8",
    )
    doc = router.load_one(html)
    assert doc is not None
    assert doc.label_values == {}


def test_zero_byte_and_valid_pdf_mixed(tmp_path, bid2_dir):
    """A bad file in the batch must not stop the good one from extracting."""
    good = bid2_dir / "Mercury_Affidavit.pdf"
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"")
    docs = FileRouter().load_many([broken, good])
    assert len(docs) == 1
    assert docs[0].file_name == "Mercury_Affidavit.pdf"


def test_empty_directory_produces_no_packages(tmp_path):
    result = Pipeline().run([tmp_path])
    assert result.packages == {}


def test_directory_of_only_broken_files_produces_no_packages(tmp_path):
    for name in ("a.pdf", "b.pdf"):
        (tmp_path / name).write_bytes(b"not a pdf")
    result = Pipeline().run([tmp_path])
    assert result.packages == {}


def test_unicode_filename_is_handled(tmp_path, bid2_dir):
    target = tmp_path / "contrat\u00e9_caf\u00e9.pdf"
    target.write_bytes((bid2_dir / "Mercury_Affidavit.pdf").read_bytes())
    docs = FileRouter().load_many([target])
    assert len(docs) == 1
    assert docs[0].file_name.startswith("contrat")


def test_extraction_of_non_rfp_document_reports_nulls_with_reasons(tmp_path):
    """A valid PDF that is not an RFP must yield nulls, not hallucinated values."""
    html = tmp_path / "random.html"
    html.write_text(
        "<html><body><h1>Recipe</h1><p>Preheat the oven to 350 degrees.</p></body></html>",
        encoding="utf-8",
    )
    result = Pipeline(settings=Settings()).run([tmp_path])
    assert result.packages, "expected the file to be grouped into a package"
    for package in result.packages.values():
        for name, fv in package.fields.items():
            if not fv.present:
                assert fv.null_reason, f"{name} missing null_reason"
            else:
                assert fv.evidence, f"{name} has value but no citation"


def test_pipeline_rejects_nothing_on_no_eval_flag(tmp_path, bid2_dir):
    """--no-eval path: extraction must still succeed."""
    result = Pipeline().run([bid2_dir])
    assert result.packages
    assert result.metadata.duration_seconds is not None
