"""File router: dispatch by format, dedup by content hash, group into packages.

The router is the only place that knows about file extensions, so adding a new
format means registering one loader here.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from pathlib import Path

from ..models.document import Document, FileFormat, detect_format
from .base import DocumentLoader
from .cache import DocumentCache
from .docx_loader import DocxLoader
from .html_loader import HtmlLoader
from .pdf_loader import PdfLoader

log = logging.getLogger(__name__)


class FileRouter:
    """Route files to the right loader and de-duplicate identical content."""

    def __init__(
        self,
        loaders: Sequence[DocumentLoader] | None = None,
        *,
        cache: DocumentCache | None = None,
    ) -> None:
        self.loaders: dict[FileFormat, DocumentLoader] = {}
        for loader in loaders or (
            PdfLoader(cache=cache),
            HtmlLoader(),
            DocxLoader(),
        ):
            for fmt in loader.formats:
                self.loaders[fmt] = loader

    def load_one(self, path: str | Path) -> Document | None:
        path = Path(path)
        fmt = detect_format(path)
        loader = self.loaders.get(fmt)
        if loader is None:
            log.warning("no loader registered for %s (%s)", path.name, fmt.value)
            return None
        log.debug("routing %s -> %s", path.name, type(loader).__name__)
        return loader.load(path)

    def load_directory(
        self, directory: str | Path, *, recursive: bool = True
    ) -> list[Document]:
        directory = Path(directory)
        pattern = "**/*" if recursive else "*"
        paths = sorted(p for p in directory.glob(pattern) if p.is_file())
        return self.load_many(paths)

    def load_many(self, paths: Iterable[str | Path]) -> list[Document]:
        """Load files, dropping byte-identical duplicates (idempotent re-runs)."""
        docs: list[Document] = []
        seen: dict[str, str] = {}
        for path in paths:
            path = Path(path)
            try:
                doc = self.load_one(path)
            except Exception as exc:  # noqa: BLE001
                log.warning("skipping %s: %s", path.name, exc)
                continue
            if doc is None:
                continue
            if not doc.blocks:
                log.warning("no content extracted from %s", path.name)
            prior = seen.get(doc.content_hash)
            if prior:
                log.info("duplicate content skipped: %s == %s", path.name, prior)
                continue
            seen[doc.content_hash] = path.name
            docs.append(doc)
        return docs


def group_by_package(docs: Sequence[Document]) -> dict[str, list[Document]]:
    """Group documents into bid packages by parent directory.

    ``data/raw/bid1`` -> package ``bid1``.  Files sitting directly in the root
    (e.g. the assignment brief) land in package ``_root`` and are excluded from
    extraction by the pipeline.
    """
    packages: dict[str, list[Document]] = {}
    for doc in docs:
        parent = Path(doc.source_path).parent.name or "_root"
        packages.setdefault(parent, []).append(doc)
    return packages
