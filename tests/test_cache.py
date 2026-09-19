"""Ingestion cache tests.

The cache exists because pdfplumber accounts for ~99% of parse time.  The
property that makes it safe is that a cached document is *indistinguishable*
from a freshly parsed one.
"""

from __future__ import annotations

import json

from rfp_extractor.ingestion.cache import LOADER_VERSION, DocumentCache
from rfp_extractor.ingestion.pdf_loader import PdfLoader
from rfp_extractor.models.document import BlockKind


def test_miss_then_hit(tmp_path, bid2_dir):
    cache = DocumentCache(tmp_path / "cache")
    loader = PdfLoader(cache=cache)

    source = bid2_dir / "PORFP_-_Dell_Laptop_Final.pdf"
    first = loader.load(source)
    assert cache.get(first.content_hash, "tables") is not None, "expected a cache entry"

    second = loader.load(source)

    # a hit must be indistinguishable from a fresh parse
    assert second.doc_id == first.doc_id
    assert second.page_count == first.page_count
    assert len(second.blocks) == len(first.blocks)
    assert [b.text for b in second.blocks] == [b.text for b in first.blocks]
    assert [b.bbox for b in second.blocks] == [b.bbox for b in first.blocks]
    # source_path is re-pointed at the file that was actually requested
    assert second.source_path == str(source)


def test_tables_survive_the_roundtrip(tmp_path, bid2_dir):
    cache = DocumentCache(tmp_path / "cache")
    loader = PdfLoader(cache=cache)
    source = bid2_dir / "PORFP_-_Dell_Laptop_Final.pdf"

    fresh = loader.load(source)
    cached = loader.load(source)

    fresh_tables = [b.table for b in fresh.blocks if b.kind is BlockKind.TABLE]
    cached_tables = [b.table for b in cached.blocks if b.kind is BlockKind.TABLE]
    assert fresh_tables, "expected pdfplumber tables in the fresh parse"
    assert fresh_tables == cached_tables


def test_different_content_is_a_separate_entry(tmp_path, bid2_dir):
    cache = DocumentCache(tmp_path / "cache")
    loader = PdfLoader(cache=cache)

    copied = tmp_path / "copy.pdf"
    copied.write_bytes((bid2_dir / "Mercury_Affidavit.pdf").read_bytes())
    first = loader.load(copied)

    # same bytes, different path -> cache hit (keyed on content, not filename)
    renamed = tmp_path / "renamed.pdf"
    renamed.write_bytes(copied.read_bytes())
    assert cache.get(first.content_hash, "tables") is not None

    # different bytes -> different key
    mutated = tmp_path / "mutated.pdf"
    mutated.write_bytes(copied.read_bytes() + b"% appended comment")
    loader.load(mutated)
    assert cache.get(first.content_hash, "tables") is not None
    assert cache.get(first.content_hash, "tables").content_hash == first.content_hash


def test_disabled_cache_never_reads_or_writes(tmp_path, bid2_dir):
    cache = DocumentCache(tmp_path / "cache", enabled=False)
    loader = PdfLoader(cache=cache)
    doc = loader.load(bid2_dir / "Mercury_Affidavit.pdf")
    assert cache.get(doc.content_hash, "tables") is None
    assert not any((tmp_path / "cache").rglob("*.json"))


def test_flavour_separates_table_and_text_parses(tmp_path):
    cache = DocumentCache(tmp_path / "cache")
    assert cache.path_for("abc", "tables") != cache.path_for("abc", "text")


def test_entry_records_the_loader_version(tmp_path, bid2_dir):
    """Stale entries from older parsing logic must not be reusable."""
    cache = DocumentCache(tmp_path / "cache")
    loader = PdfLoader(cache=cache)
    doc = loader.load(bid2_dir / "Mercury_Affidavit.pdf")
    payload = json.loads(cache.path_for(doc.content_hash, "tables").read_text(encoding="utf-8"))
    assert payload["content_hash"] == doc.content_hash
    assert LOADER_VERSION


def test_corrupt_entry_is_treated_as_a_miss(tmp_path, bid2_dir):
    cache = DocumentCache(tmp_path / "cache")
    loader = PdfLoader(cache=cache)
    doc = loader.load(bid2_dir / "Mercury_Affidavit.pdf")
    path = cache.path_for(doc.content_hash, "tables")
    path.write_text("{not valid json", encoding="utf-8")
    assert cache.get(doc.content_hash, "tables") is None
    # and the loader recovers by re-parsing
    assert loader.load(bid2_dir / "Mercury_Affidavit.pdf").page_count >= 1
