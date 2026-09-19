"""Provider abstraction for the reasoning layer.

The pipeline is deliberately *not* hard-wired to an LLM:

* ``RuleBasedProvider`` - the default. Deterministic, offline, zero-cost, and
  fully testable. On this corpus it is also more accurate than a generic LLM
  call, because the values are literal (dates, SKUs, form labels) rather than
  inferences.
* ``OpenAIProvider`` - opt-in via ``RFP_PROVIDER=openai`` + ``RFP_OPENAI_API_KEY``.
  Used to catch fields the rule set does not cover, and to sanity-check merges.

Both emit the same :class:`SourceValue` objects, so Pass 2 (merge / conflict
resolution) is identical either way. That is the point of the abstraction: the
hard part - supersession and provenance - does not change when the model does.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ...config import FieldSpec
from ...models.document import Document
from ...models.schema import SourceValue

if TYPE_CHECKING:  # pragma: no cover
    from ...config import Settings


class ProviderError(RuntimeError):
    """Raised when a provider cannot service a request (no key, API failure)."""


@runtime_checkable
class ExtractionProvider(Protocol):
    """Anything that can propose values for a field from a document."""

    name: str

    def available(self) -> bool:
        """Whether the provider can run in the current environment."""
        ...

    def extract_field(self, doc: Document, spec: FieldSpec) -> list[SourceValue]:
        """Propose candidate values for one field from one document."""
        ...


def build_provider(settings: Settings | None = None) -> ExtractionProvider | None:
    """Construct the configured provider, or ``None`` when it cannot run.

    Returns ``None`` rather than raising: a missing API key or an uninstalled
    ``openai`` package must degrade to rule-based extraction, never crash.
    """
    if settings is None:
        from ...config import get_settings

        settings = get_settings()

    if settings.provider.value != "openai":
        return None
    try:
        from .openai_provider import OpenAIProvider
    except ImportError:  # pragma: no cover - optional dependency
        return None

    provider = OpenAIProvider(settings)
    return provider if provider.available() else None


class BaseProvider(ABC):
    name = "base"

    def available(self) -> bool:  # pragma: no cover - overridden
        return False

    @abstractmethod
    def extract_field(self, doc: Document, spec: FieldSpec) -> list[SourceValue]:
        raise NotImplementedError
