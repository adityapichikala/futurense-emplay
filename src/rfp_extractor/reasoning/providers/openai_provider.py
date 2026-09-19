"""OpenAI-backed extraction provider (opt-in).

Requires ``RFP_PROVIDER=openai`` and ``RFP_OPENAI_API_KEY`` (or
``OPENAI_API_KEY``).  Uses structured outputs (``response_format=json_schema``)
so the model cannot return free-form text we would then have to parse.

Design choices worth noting:

* **Per-field, per-document calls with retrieved context.**  Rather than
  stuffing a 62-page RFP into one prompt, we hand the model only the chunks the
  hybrid retriever selected for that field.  Cheaper and more accurate.
* **Explicit absence.**  The schema forces ``value: null`` + ``reason`` when the
  field is not present, which is what stops the model from inventing a plausible
  due date.
* **Retry with exponential backoff** and hard failure isolation: a provider
  error never aborts the run, it just contributes no candidates.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from ...config import FieldSpec, Settings
from ...models.document import Document
from ...models.schema import ExtractionMethod, SourceValue

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You extract structured fields from public procurement (RFP) documents. "
    "Return ONLY what is explicitly stated in the provided excerpts. "
    "If a field is not stated, set value to null and explain in reason. "
    "Never guess, infer, or fill defaults."
)


#: USD per 1M tokens.  Only used for a cost *estimate* on the run metadata;
#: update here when pricing changes.
_PRICE_PER_MTOK: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}


class OpenAIProvider:
    name = "openai"

    def __init__(self, settings: Settings | None = None, *, client: Any = None) -> None:
        self.settings = settings
        self.model = (settings.openai_model if settings else None) or "gpt-4o-mini"
        self._client = client
        self.tokens_used: int = 0
        self.estimated_cost_usd: float = 0.0
        self.calls: int = 0

    @property
    def usage(self) -> dict[str, float]:
        """Cumulative usage, for the run metadata."""
        return {
            "tokens": self.tokens_used,
            "usd": round(self.estimated_cost_usd, 6),
            "calls": self.calls,
        }

    # -- lifecycle ---------------------------------------------------------

    def available(self) -> bool:
        return self._api_key() is not None and self._get_client() is not None

    def _api_key(self) -> str | None:
        if self.settings and self.settings.openai_api_key:
            return self.settings.openai_api_key
        return os.environ.get("OPENAI_API_KEY")

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI  # noqa: PLC0415 - optional dependency
        except ImportError:
            log.debug("openai package not installed; provider disabled")
            return None
        key = self._api_key()
        if not key:
            return None
        self._client = OpenAI(api_key=key)
        return self._client

    # -- extraction ---------------------------------------------------------

    def extract_field(self, doc: Document, spec: FieldSpec, context: str = "") -> list[SourceValue]:
        client = self._get_client()
        if client is None:
            return []

        schema = self._json_schema(spec)
        payload = {
            "field": spec.alias,
            "description": spec.description,
            "document": doc.file_name,
            "excerpts": (context or doc.full_text)[:6000],
        }
        try:
            raw = self._call_with_retry(client, schema, payload)
        except Exception as exc:  # noqa: BLE001 - provider must never break the run
            log.warning("openai call failed for %s/%s: %s", doc.file_name, spec.name, exc)
            return []

        value = raw.get("value")
        if value in (None, "", [], {}):
            return []
        return [
            SourceValue(
                doc_id=doc.doc_id,
                file_name=doc.file_name,
                doc_type=doc.doc_type.value,
                precedence=doc.precedence,
                page=raw.get("page") or 1,
                raw=value,
                normalized=value,
                confidence=float(raw.get("confidence", 0.7)),
                span=str(raw.get("evidence", ""))[:400],
                method=ExtractionMethod.LLM,
            )
        ]

    def _call_with_retry(self, client, schema: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": json.dumps(payload)},
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {"name": "field_extraction", "schema": schema},
                    },
                    temperature=0,
                )
                self._record_usage(getattr(response, "usage", None))
                content = response.choices[0].message.content or "{}"
                return json.loads(content)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                time.sleep(min(8.0, 0.5 * 2**attempt))
        raise RuntimeError(f"openai request failed after retries: {last_error}")

    def _record_usage(self, usage: Any) -> None:
        """Accumulate token usage and estimated cost.

        The plan lists cost tracking as a differentiator; without this the
        run metadata would report 0 tokens forever.
        """
        self.calls += 1
        if usage is None:
            return
        prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion = int(getattr(usage, "completion_tokens", 0) or 0)
        self.tokens_used += prompt + completion
        rate_in, rate_out = _PRICE_PER_MTOK.get(self.model, (0.0, 0.0))
        self.estimated_cost_usd += prompt / 1_000_000 * rate_in
        self.estimated_cost_usd += completion / 1_000_000 * rate_out

    @staticmethod
    def _json_schema(spec: FieldSpec) -> dict[str, Any]:
        value_type = {
            "string": "string",
            "datetime": "string",
            "list": "array",
            "dict": "object",
            "boolean": "boolean",
        }[spec.kind]
        value: dict[str, Any] = {"type": value_type}
        if value_type == "array":
            value["items"] = {"type": "string"}
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["value", "reason", "evidence", "page", "confidence"],
            "properties": {
                "value": {"anyOf": [value, {"type": "null"}]},
                "reason": {"type": "string"},
                "evidence": {"type": "string"},
                "page": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                "confidence": {"type": "number"},
            },
        }
