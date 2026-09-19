"""Loader protocol + shared helpers for the ingestion layer."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path

from ..models.document import Document, FileFormat, compute_content_hash, detect_format

log = logging.getLogger(__name__)


class DocumentLoader(ABC):
    """Strategy interface: turn one file into a :class:`Document`."""

    #: Formats this loader handles.
    formats: tuple[FileFormat, ...] = ()

    @abstractmethod
    def load(self, path: str | Path) -> Document:
        raise NotImplementedError


def make_doc_id(file_name: str, content_hash: str) -> str:
    """Stable, readable document id: ``<slug>-<hash8>``."""
    slug = "".join(c if c.isalnum() else "-" for c in Path(file_name).stem.lower())
    slug = "-".join(filter(None, slug.split("-")))[:48]
    return f"{slug}-{content_hash[:8]}"


def load_many(paths: Iterable[str | Path], loader: DocumentLoader) -> list[Document]:
    docs: list[Document] = []
    for p in paths:
        try:
            docs.append(loader.load(p))
        except Exception as exc:  # noqa: BLE001 - ingestion must not abort the run
            log.warning("failed to load %s: %s", p, exc)
    return docs


__all__ = [
    "DocumentLoader",
    "load_many",
    "make_doc_id",
    "compute_content_hash",
    "detect_format",
]
