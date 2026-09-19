"""Splitting one folder into per-solicitation packages.

Grouping by directory alone fails badly when a user drops several bids into one
folder (or uploads them in a single API call): everything merges into one
package and accuracy collapses.  These tests cover the identity-based split.
"""

from __future__ import annotations

from rfp_extractor.models.document import Block, Document, FileFormat, Page
from rfp_extractor.pipeline import Pipeline, _dominant_key, _slug, _strong_bid_keys


def _doc(doc_id: str, file_name: str, text: str) -> Document:
    return Document(
        doc_id=doc_id,
        file_name=file_name,
        source_path=file_name,
        file_format=FileFormat.PDF,
        content_hash=doc_id,
        pages=[Page(page_no=1, blocks=[Block(block_id="p1.b1", page_no=1, text=text)])],
    )


def test_strong_bid_keys_uses_distinctive_patterns():
    """A single occurrence of JA-207652 / BPM044557 is enough.

    A corpus-wide "appears 3+ times" rule silently dropped the Dallas master
    RFP and both addenda, fragmenting one bid into three packages.
    """
    assert _strong_bid_keys(_doc("a", "a.pdf", "Solicitation Number: JA-207652")) == ["JA-207652"]
    assert _strong_bid_keys(_doc("b", "b.pdf", "eMMA Project Number: BPM044557")) == ["BPM044557"]


def test_strong_bid_keys_reads_the_filename():
    doc = _doc("c", "Addendum 2 RFP JA-207652 Devices.pdf", "unrelated body text")
    assert "JA-207652" in _strong_bid_keys(doc)


def test_strong_bid_keys_ignores_documents_without_identity():
    assert _strong_bid_keys(_doc("d", "notes.pdf", "a generic memo")) == []


def test_split_separates_two_bids():
    docs = [
        _doc("1", "Addendum 1 RFP JA-207652.pdf", "JA-207652 addendum"),
        _doc("2", "JA-207652 FINAL.pdf", "Solicitation Due 27-JUN-2024"),
        _doc("3", "PORFP_-_Dell_Laptop_Final.pdf", "eMMA Project Number: BPM044557"),
        _doc("4", "Dell Laptops BidNet.html", "BPM044557 portal"),
    ]
    clusters = Pipeline._split_by_bid_identity(docs)
    assert len(clusters) == 2
    names = [{d.file_name for d in cluster} for cluster in clusters]
    assert {"Addendum 1 RFP JA-207652.pdf", "JA-207652 FINAL.pdf"} in names
    assert {"PORFP_-_Dell_Laptop_Final.pdf", "Dell Laptops BidNet.html"} in names


def test_split_keeps_a_single_bid_intact():
    docs = [
        _doc("1", "a.pdf", "JA-207652"),
        _doc("2", "b.pdf", "JA-207652 addendum"),
    ]
    assert Pipeline._split_by_bid_identity(docs) == [docs]


def test_flat_directory_of_two_bids_yields_two_packages(tmp_path, bid1_dir, bid2_dir):
    """Integration: real files, flat folder, no per-bid subdirectories.

    Uses the small documents only - the 62-page master RFP makes this test
    needlessly slow.
    """
    for name in (
        "Addendum 2 RFP JA-207652 Student and Staff Computing Devices.pdf",
        "Student and Staff Computing Devices __SOURCING #168884__ - Bid Information - {3} _ BidNet Direct.html",
    ):
        (tmp_path / name).write_bytes((bid1_dir / name).read_bytes())
    for name in (
        "PORFP_-_Dell_Laptop_Final.pdf",
        "Dell Laptops w_Extended Warranty - Bid Information - {3} _ BidNet Direct.html",
    ):
        (tmp_path / name).write_bytes((bid2_dir / name).read_bytes())

    result = Pipeline().run([tmp_path])
    bids = {p.bid_number for p in result.packages.values()}
    assert bids == {"JA-207652", "BPM044557"}, list(result.packages)

    for package in result.packages.values():
        assert len(package.documents) == 2, package.documents
        if package.bid_number == "JA-207652":
            assert package.due_date.startswith("2024-07-09")
        else:
            assert package.due_date.startswith("2024-06-10")


def test_dominant_key_and_slug_name_the_package_after_the_solicitation():
    """A folder name like 'tmp_flat-1' tells an API caller nothing."""
    docs = [
        _doc("1", "a.pdf", "JA-207652"),
        _doc("2", "b.pdf", "JA-207652 addendum"),
        _doc("3", "c.pdf", "unrelated"),
    ]
    assert _dominant_key(docs) == "JA-207652"
    assert _slug("JA-207652") == "ja-207652"
    assert _slug("BPM044557") == "bpm044557"
