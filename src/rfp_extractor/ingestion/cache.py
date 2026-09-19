"""Content-hash cache for parsed documents.

Parsing is by far the most expensive stage: pdfplumber's line analysis accounts
for ~99% of the time on a 62-page RFP (84s versus 0.8s for PyMuPDF text alone).
Nothing about that work changes between runs on unchanged files, so parsed
documents are cached on disk keyed by content hash.

Correctness guards:
* The key includes the file's SHA-256, so a changed file never reads stale data.
* The key includes a **loader version**, so changing parsing logic invalidates
  the cache instead of silently serving documents parsed by the old code.
* A cache that cannot be read or written is treated as a miss - never an error.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from ..models.document import Document

log = logging.getLogger(__name__)

#: Bump when Document/Block layout or loader behaviour changes in a way that
#: existing cache entries would not reflect.
LOADER_VERSION = "2"


class DocumentCache:
    """Filesystem cache mapping (content hash, loader version) -> Document."""

    def __init__(self, base_dir: str | Path = ".cache/ingest", *, enabled: bool = True) -> None:
        self.base_dir = Path(base_dir)
        self.enabled = enabled

    def path_for(self, content_hash: str, flavour: str = "full") -> Path:
        key = hashlib.sha256(f"{LOADER_VERSION}:{flavour}:{content_hash}".encode()).hexdigest()
        return self.base_dir / key[:2] / f"{key}.json"

    def get(self, content_hash: str, flavour: str = "full") -> Document | None:
        if not self.enabled:
            return None
        path = self.path_for(content_hash, flavour)
        try:
            payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
            return Document.model_validate(payload)
        except (OSError, ValueError):
            return None

    def put(self, doc: Document, flavour: str = "full") -> None:
        if not self.enabled:
            return
        path = self.path_for(doc.content_hash, flavour)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(doc.model_dump_json(), encoding="utf-8")
            tmp.replace(path)
        except OSError as exc:  # pragma: no cover - cache is best-effort
            log.debug("cache write failed for %s: %s", doc.file_name, exc)
