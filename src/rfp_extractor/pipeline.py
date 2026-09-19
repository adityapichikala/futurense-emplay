"""End-to-end orchestration: files in, cited structured data out.

The pipeline is deliberately linear and idempotent - re-running it on the same
corpus produces byte-identical JSON (no timestamps inside the extraction
payload), which is what makes the evaluation harness trustworthy.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Sequence
from datetime import UTC, datetime, timezone
from pathlib import Path

from .config import Settings, get_settings, load_extraction_config
from .ingestion.router import FileRouter, group_by_package
from .intelligence.addendum import AddendumDetector
from .intelligence.classifier import DocumentClassifier
from .intelligence.graph import BidPackage, DocumentGraph
from .models.document import DocType, Document
from .models.schema import (
    ConsolidatedResult,
    DocumentExtraction,
    DocumentRef,
    ExtractionResult,
    RunMetadata,
)
from .reasoning.extractor import (
    ExtractionEngine,
    build_run_metadata,
    describe_provider,
)
from .reasoning.providers.base import build_provider
from .reasoning.retriever import tokenize

log = logging.getLogger(__name__)

#: Documents that describe the task rather than the bid.
_NON_BID_TYPES = {DocType.ASSIGNMENT, DocType.UNKNOWN}


#: Patterns distinctive enough that a *single* occurrence identifies a bid.
#: A corpus-wide "appears 3+ times" rule silently drops the Dallas master RFP
#: and both addenda, which mention JA-207652 only once or twice each - that
#: fragmented one bid into three packages.
_DISTINCTIVE_BID_RE = re.compile(r"\b(?:JA-\d{5,7}|BPM\d{6,9}|E\d{2}P\d{7})\b")


def _strong_bid_keys(doc: Document) -> list[str]:
    """Solicitation identifiers this document clearly belongs to.

    Three sources, in order of confidence:
      1. A distinctive solicitation pattern anywhere in the filename or text.
         Filenames count: issuers routinely name files after the solicitation.
      2. An explicit portal/PORFP field (label -> value).
      3. Any other number that recurs at least three times.
    """
    keys: list[str] = []
    haystack = f"{doc.file_name}\n{doc.full_text}"
    for match in _DISTINCTIVE_BID_RE.finditer(haystack):
        value = match.group(0)
        if value not in keys:
            keys.append(value)

    for label in ("Solicitation Number", "eMMA Project Number", "PORFP Number"):
        value = doc.label_values.get(label)
        if value:
            value = value.strip().lstrip("#")
            if value not in keys:
                keys.append(value)

    for value, count in DocumentGraph._bid_candidates(doc).items():
        if count >= 3 and value not in keys and not _DISTINCTIVE_BID_RE.fullmatch(value):
            keys.append(value)
    return keys


def _dominant_key(docs: list[Document]) -> str | None:
    """The bid identifier most documents in this group agree on."""
    tally: dict[str, int] = {}
    for doc in docs:
        for key in _strong_bid_keys(doc):
            tally[key] = tally.get(key, 0) + 1
    return max(tally, key=tally.get) if tally else None  # type: ignore[arg-type]


def _slug(value: str) -> str:
    slug = "".join(c if c.isalnum() else "-" for c in value.lower()).strip("-")
    return slug or "package"


def _doc_tokens(doc: Document) -> set[str]:
    return set(tokenize(doc.file_name + " " + doc.full_text[:4000]))


def _idf_table(docs: list[Document]) -> tuple[dict[str, float], dict[str, set[str]]]:
    """Inverse document frequency over this group, plus cached token sets.

    IDF is what makes orphan placement work.  Raw Jaccard overlap puts
    Dell's spec sheet with the Dallas ISD package because the word "Dell" is
    everywhere; weighting by rarity makes the identifiers that actually
    identify a bid (CC7802, WD22TB4, Latitude 5550) dominate.
    """
    token_sets = {doc.doc_id: _doc_tokens(doc) for doc in docs}
    freq: dict[str, int] = {}
    for tokens in token_sets.values():
        for token in tokens:
            freq[token] = freq.get(token, 0) + 1
    total = max(1, len(docs))
    idf = {token: math.log(total / count) + 0.01 for token, count in freq.items()}
    return idf, token_sets


def _similarity(
    doc: Document,
    cluster: list[Document],
    idf: dict[str, float],
    token_sets: dict[str, set[str]],
) -> float:
    """Best IDF-weighted overlap between a document and any member of a cluster.

    "Best member" rather than "union of the cluster": an orphan should match
    one specific document strongly (a spec sheet matching one PORFP), not the
    merged vocabulary of everything.
    """
    doc_tokens = token_sets.get(doc.doc_id) or _doc_tokens(doc)
    if not doc_tokens:
        return 0.0

    # Primary signal: shared *identifier* tokens (CC7802, WD22TB4, 210-BLYZ).
    # These name a specific procurement, so a single match is decisive - and it
    # is immune to the size bias that wrecked plain overlap (a one-page
    # affidavit is always "covered" by a 62-page RFP).
    doc_ids = _identifiers(doc_tokens)
    best_ids = 0
    for other in cluster:
        other_ids = _identifiers(token_sets.get(other.doc_id) or _doc_tokens(other))
        best_ids = max(best_ids, len(doc_ids & other_ids))
    if best_ids:
        return 10.0 + best_ids

    # Fallback: IDF-weighted cosine.  Cosine (not raw overlap) because it
    # normalises by both documents' sizes.
    # Mean, not max: with "best member wins" a 62-page RFP outweighs everything
    # and one-page appendices get dragged into whichever package holds the
    # longest document.
    doc_weight = math.sqrt(sum(idf.get(t, 0.01) for t in doc_tokens)) or 1.0
    scores = []
    for other in cluster:
        other_tokens = token_sets.get(other.doc_id) or _doc_tokens(other)
        shared = sum(idf.get(t, 0.01) for t in doc_tokens & other_tokens)
        other_weight = math.sqrt(sum(idf.get(t, 0.01) for t in other_tokens)) or 1.0
        scores.append(shared / (doc_weight * other_weight))
    return sum(scores) / len(scores) if scores else 0.0


_ID_TOKEN_RE = re.compile(r"(?=.*\d)[a-z0-9][a-z0-9\-]{4,}")


def _identifiers(tokens: set[str]) -> set[str]:
    """Tokens that look like part numbers / solicitation ids (contain a digit)."""
    return {t for t in tokens if len(t) >= 5 and _ID_TOKEN_RE.fullmatch(t)}


class Pipeline:
    """Load -> classify -> chain amendments -> extract -> validate."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        config_path: str | None = None,
        default_tz: timezone = UTC,
    ) -> None:
        #: populated by :meth:`run` - one :class:`DocumentExtraction` per input file.
        self.document_extractions: list[DocumentExtraction] = []
        self.settings = settings or get_settings()
        self.config = load_extraction_config(config_path)
        self.default_tz = default_tz
        self.router = FileRouter()
        self.classifier = DocumentClassifier()
        self.addendum_detector = AddendumDetector()
        self.graph = DocumentGraph(self.config.precedence)
        self.provider = build_provider(self.settings)
        if self.provider is not None:
            log.info("extraction provider active: %s", self.provider.name)
        self.engine = ExtractionEngine(
            self.config, self.settings, default_tz=default_tz, provider=self.provider
        )

    # -- stages -------------------------------------------------------------

    def ingest(self, paths_or_dirs: Sequence[str | Path]) -> list[Document]:
        paths: list[Path] = []
        for entry in paths_or_dirs:
            path = Path(entry)
            if path.is_dir():
                paths.extend(sorted(p for p in path.rglob("*") if p.is_file()))
            else:
                paths.append(path)
        docs = self.router.load_many(paths)
        log.info("ingested %d documents", len(docs))
        return docs

    def annotate(self, docs: list[Document]) -> list[Document]:
        for doc in docs:
            self.classifier.annotate(doc)
            self.addendum_detector.annotate(doc)
            log.debug(
                "classified %s -> %s (conf=%.2f, addendum=%s)",
                doc.file_name,
                doc.doc_type.value,
                doc.doc_type_confidence,
                doc.addendum_number,
            )
        return docs

    def build_packages(self, docs: list[Document]) -> dict[str, BidPackage]:
        grouped = group_by_package(docs)
        packages: dict[str, BidPackage] = {}
        for package_id, group in sorted(grouped.items()):
            if package_id in {"_root", "raw", "data"}:
                continue
            bid_docs = [d for d in group if d.doc_type not in _NON_BID_TYPES] or group
            if not bid_docs:
                continue
            groups = self._split_by_bid_identity(bid_docs)
            for index, subgroup in enumerate(groups):
                if len(groups) == 1:
                    pid = package_id
                else:
                    # once split, the folder name is meaningless - name the
                    # package after the solicitation it actually contains
                    key = _dominant_key(subgroup)
                    pid = _slug(key) if key else f"{package_id}-{index + 1}"
                packages[pid] = self.graph.build_package(pid, subgroup)
                packages[pid].package_id = self._label_package(pid, packages[pid])
        return packages

    @staticmethod
    def _split_by_bid_identity(docs: list[Document]) -> list[list[Document]]:
        """Split a directory group into per-solicitation clusters.

        Grouping by directory alone is not enough.  If a user drops every RFP
        they have into one folder - or uploads several solicitations in a single
        API call - the naive behaviour is to merge them into one package, so the
        Maryland procurement officer's phone number lands on the Dallas bid and
        accuracy collapses to zero.

        Documents are clustered by the solicitation number they cite; documents
        that cite none are placed with whichever cluster they share the most
        vocabulary with.
        """
        clusters: list[list[Document]] = []
        key_to_cluster: dict[str, int] = {}
        orphans: list[Document] = []

        for doc in docs:
            keys = _strong_bid_keys(doc)
            if not keys:
                orphans.append(doc)
                continue
            target: int | None = None
            for key in keys:
                if key in key_to_cluster:
                    target = key_to_cluster[key]
                    break
            if target is None:
                clusters.append([doc])
                target = len(clusters) - 1
            else:
                clusters[target].append(doc)
            for key in keys:
                key_to_cluster[key] = target

        if len(clusters) <= 1:
            return [docs]

        idf, token_sets = _idf_table(docs)
        for doc in orphans:
            best = max(
                range(len(clusters)),
                key=lambda i: _similarity(doc, clusters[i], idf, token_sets),
            )
            clusters[best].append(doc)
        return clusters

    @staticmethod
    def _label_package(package_id: str, package: BidPackage) -> str:
        """Human-meaningful package id.

        Files uploaded through the API land in a temp directory, so the folder
        name ("rfp_upload_kqg56qk2") is useless in a response.  Fall back to the
        solicitation number when the directory carries no meaning.
        """
        if package_id.startswith("rfp_upload_"):
            if package.bid_number:
                slug = "".join(
                    c if c.isalnum() else "-" for c in package.bid_number.lower()
                ).strip("-")
                return slug or package_id
        return package_id

    def extract_package(self, package: BidPackage, metadata: RunMetadata) -> ExtractionResult:
        result = self.engine.run(package, metadata)
        result.documents = [
            DocumentRef(
                doc_id=d.doc_id,
                file_name=d.file_name,
                file_format=d.file_format.value,
                doc_type=d.doc_type.value,
                precedence=d.precedence,
                page_count=d.page_count,
                content_hash=d.content_hash,
                addendum_number=d.addendum_number,
                doc_type_confidence=d.doc_type_confidence,
                amendment_notes=d.amendment_notes,
            )
            for d in package.by_precedence()
        ]
        result.metadata.warnings.extend(package.warnings)
        return result

    def extract_per_document(self, doc: Document) -> DocumentExtraction:
        """What one file says on its own, before cross-document merge."""
        fields = self.engine.extract_single_document(doc)
        values = {
            name: (fv.display if fv.present else None) for name, fv in fields.items()
        }
        return DocumentExtraction(
            doc_id=doc.doc_id,
            file_name=doc.file_name,
            source_path=doc.source_path,
            file_format=doc.file_format.value,
            doc_type=doc.doc_type.value,
            precedence=doc.precedence,
            page_count=doc.page_count,
            addendum_number=doc.addendum_number,
            amendment_notes=doc.amendment_notes,
            fields=fields,
            values=values,
        )

    # -- entry points --------------------------------------------------------

    def run(self, paths_or_dirs: Sequence[str | Path]) -> ConsolidatedResult:
        started = datetime.now()
        metadata = build_run_metadata(provider=describe_provider(self.engine))
        docs = self.annotate(self.ingest(paths_or_dirs))
        packages = self.build_packages(docs)

        consolidated = ConsolidatedResult(metadata=metadata)
        self.document_extractions = [
            self.extract_per_document(doc)
            for doc in sorted(docs, key=lambda d: (-d.precedence, d.file_name))
            if doc.doc_type not in _NON_BID_TYPES
        ]
        for package_id, package in packages.items():
            log.info(
                "package %s: %d docs, bid_number=%s", package_id, len(package.documents), package.bid_number
            )
            result = self.extract_package(package, metadata)
            filled, total = result.coverage()
            log.info("  -> %d/%d fields populated", filled, total)
            # key on the (possibly renamed) package id so API consumers see the
            # solicitation number rather than an upload temp-dir name
            consolidated.packages[package.package_id] = result

        finished = datetime.now()
        metadata.finished_at = finished
        metadata.duration_seconds = round((finished - started).total_seconds(), 3)
        return consolidated

    def run_directory(self, raw_dir: str | Path) -> ConsolidatedResult:
        return self.run([raw_dir])
