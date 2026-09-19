"""FastAPI service exposing the extraction pipeline.

Endpoints
---------
``GET  /health``    liveness + configuration summary
``POST /extract``   multipart upload of RFP files -> per-package ExtractionResult
``GET  /fields``    the canonical 19-field contract the service honours
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..config import CONFIG_DIR, get_settings
from ..models.schema import CANONICAL_FIELDS, FIELD_ALIASES
from ..pipeline import Pipeline

log = logging.getLogger("rfp_extractor.api")

app = FastAPI(
    title="RFP Intelligence Extraction API",
    version="1.0.0",
    description=(
        "Structured extraction from RFP packages (PDF / HTML / DOCX) with "
        "addendum-aware conflict resolution and per-field provenance."
    ),
)

_PIPELINE: Pipeline | None = None


def get_pipeline() -> Pipeline:
    """Lazily construct and cache the pipeline (config parsing is not free)."""
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = Pipeline()
    return _PIPELINE


class HealthResponse(BaseModel):
    status: str
    provider: str
    fields: int
    config_dir: str


class ExtractResponse(BaseModel):
    packages: dict[str, dict[str, Any]] = Field(default_factory=dict)
    accuracy: float | None = None
    warnings: list[str] = Field(default_factory=list)


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        status="ok",
        provider=settings.provider.value,
        fields=len(CANONICAL_FIELDS),
        config_dir=str(CONFIG_DIR),
    )


@app.get("/fields", tags=["schema"])
def fields() -> dict[str, str]:
    return FIELD_ALIASES


@app.post("/extract", response_model=ExtractResponse, tags=["extraction"])
async def extract(files: list[UploadFile] = File(...)) -> ExtractResponse:
    """Accept a mixed set of RFP documents and return structured extraction."""
    if not files:
        raise HTTPException(status_code=400, detail="at least one file is required")

    allowed = {".pdf", ".html", ".htm", ".docx", ".txt"}
    workdir = Path(tempfile.mkdtemp(prefix="rfp_upload_"))
    try:
        for upload in files:
            suffix = Path(upload.filename or "").suffix.lower()
            if suffix not in allowed:
                raise HTTPException(
                    status_code=415, detail=f"unsupported file type: {upload.filename}"
                )
            target = workdir / (upload.filename or f"upload{suffix}")
            with open(target, "wb") as fh:
                shutil.copyfileobj(upload.file, fh)

        result = get_pipeline().run([workdir])
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - surface a clean 500 to the client
        log.exception("extraction failed")
        raise HTTPException(status_code=500, detail=f"extraction failed: {exc}") from exc
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    packages = {pid: pkg.to_dict() for pid, pkg in result.packages.items()}
    return ExtractResponse(
        packages=packages,
        warnings=result.metadata.warnings,
    )


@app.exception_handler(Exception)
async def unhandled(request, exc):  # pragma: no cover - defensive
    log.exception("unhandled error")
    return JSONResponse(status_code=500, content={"detail": str(exc)})
