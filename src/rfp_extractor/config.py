"""Configuration: Pydantic Settings + YAML, so behaviour is data-driven, not hardcoded.

Everything the extraction engine knows about a field lives in ``configs/fields.yaml``.
Adding support for a new RFP issuer means editing YAML, not Python.
"""

from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
DATA_DIR = REPO_ROOT / "data"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"


class ProviderKind(str, Enum):
    RULE_BASED = "rule-based"
    OPENAI = "openai"
    HYBRID = "hybrid"


class Settings(BaseSettings):
    """Environment-driven settings (``RFP_`` prefix, or ``.env``)."""

    model_config = SettingsConfigDict(
        env_prefix="RFP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    provider: ProviderKind = ProviderKind.RULE_BASED
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    openai_merge_model: str = "gpt-4o"

    chunk_size_tokens: int = 420
    chunk_overlap_tokens: int = 60
    retrieval_top_k: int = 6
    bm25_weight: float = 0.6
    vector_weight: float = 0.4

    min_confidence: float = 0.35
    log_level: str = "INFO"
    artifacts_dir: str = str(ARTIFACTS_DIR)
    raw_dir: str = str(DATA_DIR / "raw")

    # Embedding backend: "tfidf" (zero-dependency, deterministic) or
    # "sentence-transformers" when the optional dependency is installed.
    embedding_backend: Literal["tfidf", "sentence-transformers"] = "tfidf"

    #: Cache parsed documents on disk (content-hash keyed).  pdfplumber
    #: dominates parse time, so this is what makes repeat runs fast.
    cache_enabled: bool = True
    cache_dir: str = ".cache/ingest"
    embedding_model: str = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# YAML-driven field rules
# ---------------------------------------------------------------------------


class PatternRule(BaseModel):
    """A single labelled regex used to mine a value out of document text."""

    model_config = ConfigDict(extra="forbid")

    name: str
    #: Regex with named group ``value`` (or ``label``/``value`` pairs).
    regex: str
    #: Which capture group holds the answer.
    group: str = "value"
    #: Base confidence when this rule fires.
    confidence: float = 0.8
    #: Restrict to certain document types (empty = any).
    doc_types: list[str] = Field(default_factory=list)
    #: Restrict to certain sections (empty = any).
    sections: list[str] = Field(default_factory=list)
    #: Named transform to run on the captured string.
    transform: str | None = None
    #: Multi-match behaviour: first | all | join
    multiplicity: Literal["first", "all", "join"] = "first"
    #: Keep the value only if it matches this guard regex.
    must_match: str | None = None
    #: Reject the value if it matches this regex (e.g. placeholder dashes).
    must_not_match: str | None = None
    #: Maximum character length for the captured value.
    max_length: int | None = None


class LabelRule(BaseModel):
    """Map a source label (HTML field name / PDF label) onto a canonical field."""

    model_config = ConfigDict(extra="forbid")

    labels: list[str] = Field(default_factory=list)
    confidence: float = 0.9
    doc_types: list[str] = Field(default_factory=list)
    transform: str | None = None
    multiplicity: Literal["first", "all", "join"] = "first"


class FieldSpec(BaseModel):
    """Everything the engine needs to know about one output field."""

    model_config = ConfigDict(extra="forbid")

    name: str
    alias: str
    description: str
    kind: Literal["string", "datetime", "list", "dict", "boolean"] = "string"
    required: bool = False
    #: Applies to a whole bid package ("package") or is vendor-specific.
    scope: Literal["package", "vendor"] = "package"
    #: Aggregation strategy during merge: precedence | union | concat | longest
    merge: Literal["precedence", "union", "concat", "longest", "most_specific"] = (
        "precedence"
    )
    patterns: list[PatternRule] = Field(default_factory=list)
    labels: list[LabelRule] = Field(default_factory=list)
    #: Extra regexes, matched anywhere, whose *whole surrounding sentence* is the value.
    sentence_patterns: list[PatternRule] = Field(default_factory=list)
    #: Fallback text used when nothing is found.
    null_reason: str = "Not stated in source documents"


class PrecedenceSpec(BaseModel):
    """Precedence weights per document type (higher wins)."""

    model_config = ConfigDict(extra="forbid")

    doc_types: dict[str, int] = Field(default_factory=dict)
    addendum_base: int = 100
    addendum_step: int = 10


class ClassifierSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_types: dict[str, dict[str, Any]] = Field(default_factory=dict)


class ExtractionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str = "1.0.0"
    fields: list[FieldSpec] = Field(default_factory=list)
    precedence: PrecedenceSpec = Field(default_factory=PrecedenceSpec)
    classifier: ClassifierSpec = Field(default_factory=ClassifierSpec)
    normalization: dict[str, Any] = Field(default_factory=dict)

    def field(self, name: str) -> FieldSpec:
        for spec in self.fields:
            if spec.name == name:
                return spec
        raise KeyError(f"Unknown field spec: {name}")

    @property
    def field_names(self) -> list[str]:
        return [f.name for f in self.fields]


@lru_cache(maxsize=4)
def load_extraction_config(path: str | None = None) -> ExtractionConfig:
    cfg_path = Path(path) if path else CONFIG_DIR / "fields.yaml"
    with open(cfg_path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return ExtractionConfig.model_validate(raw)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
