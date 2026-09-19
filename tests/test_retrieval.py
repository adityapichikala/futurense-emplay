"""Retrieval tests: BM25 must win on identifiers, fusion must stay ordered."""

from __future__ import annotations

from rfp_extractor.reasoning.chunking import Chunk, SemanticChunker, estimate_tokens
from rfp_extractor.reasoning.retriever import HybridRetriever, tokenize


def _chunk(doc_id: str, text: str, **kw) -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}::c1",
        doc_id=doc_id,
        file_name=f"{doc_id}.pdf",
        doc_type=kw.get("doc_type", "master_rfp"),
        precedence=kw.get("precedence", 40),
        page=kw.get("page", 1),
        section=None,
        text=text,
        tokens=estimate_tokens(text),
    )


def test_tokenize_drops_stopwords_and_is_lowercase():
    tokens = tokenize("The Due Date is JA-207652")
    assert "the" not in tokens
    assert "ja-207652" in tokens


def test_bm25_finds_exact_identifier():
    chunks = [
        _chunk("a", "Warranty service and repairs will be performed by the vendor."),
        _chunk("b", "Solicitation Due 27-JUN-2024 14:00:00 for JA-207652."),
        _chunk("c", "All deliveries must incorporate white glove services."),
    ]
    retriever = HybridRetriever(chunks)
    hits = retriever.search("JA-207652", top_k=1)
    assert hits and hits[0].chunk.doc_id == "b"


def test_fusion_ranks_overlapping_vocabulary_first():
    """The default TF-IDF backend is lexical: it ranks on shared terms."""
    chunks = [
        _chunk("a", "Proposal due date and submission deadline are in section 1."),
        _chunk("b", "Warranty shall be made by the original equipment manufacturer."),
    ]
    retriever = HybridRetriever(chunks, bm25_weight=0.4, vector_weight=0.6)
    hits = retriever.search("proposal due date", top_k=1)
    assert hits and hits[0].chunk.doc_id == "a"


def test_true_paraphrase_matching_needs_the_dense_backend():
    """Documented limitation, not a bug.

    "when is the proposal due" shares no tokens with "deadline for submitting
    proposals", so the lexical backend cannot match them.  Genuine paraphrase
    retrieval requires ``embedding_backend=sentence-transformers``; the API is
    identical (``SentenceTransformerBackend``).
    """
    chunks = [
        _chunk("a", "The deadline for submitting proposals is stated in section 1."),
        _chunk("b", "Warranty shall be made by the original equipment manufacturer."),
    ]
    retriever = HybridRetriever(chunks)
    hits = retriever.search("when is the proposal due", top_k=1)
    # No shared vocabulary: any hit is ordered arbitrarily, not by relevance.
    assert hits == [] or (hits[0].bm25_score == 0.0 and hits[0].vector_score == 0.0)

    from rfp_extractor.reasoning.retriever import SentenceTransformerBackend, VectorBackend

    assert issubclass(SentenceTransformerBackend, VectorBackend)


def test_precedence_boost_prefers_addendum_on_tie():
    chunks = [
        _chunk("master", "Solicitation Due 27-JUN-2024.", precedence=40),
        _chunk("addendum", "Solicitation Due 27-JUN-2024.", precedence=120),
    ]
    retriever = HybridRetriever(chunks)
    hits = retriever.search("solicitation due", top_k=2)
    assert hits[0].chunk.doc_id == "addendum"


def test_doc_type_filter_applies():
    chunks = [_chunk("a", "Dell Latitude 5550", doc_type="vendor_spec")]
    retriever = HybridRetriever(chunks)
    assert retriever.search("dell", doc_types={"master_rfp"}) == []
    assert retriever.search("dell", doc_types={"vendor_spec"})


def test_chunker_respects_token_budget_and_headings():
    from rfp_extractor.models.document import Block, BlockKind

    blocks = [
        Block(block_id=f"b{i}", page_no=1, kind=BlockKind.TEXT, text="word " * 200)
        for i in range(1, 8)
    ]
    blocks.insert(3, Block(block_id="h", page_no=1, kind=BlockKind.HEADING, text="Section 4"))

    class _Doc:
        doc_id = "d"
        file_name = "d.pdf"
        doc_type = type("T", (), {"value": "master_rfp"})()
        precedence = 40

        def iter_blocks(self):
            return iter(blocks)

    chunks = SemanticChunker(target_tokens=100).chunk_document(_Doc())  # type: ignore[arg-type]
    assert len(chunks) > 1
    assert all(c.tokens <= 100 * 3 for c in chunks)
