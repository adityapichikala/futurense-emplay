"""Provider abstraction tests.

The OpenAI provider is exercised with an injected fake client, so these tests
run offline and without a key: they verify request shape, parsing, retry and
isolation, which is all the wiring we can meaningfully assert here.
"""

from __future__ import annotations

import json
import types

import pytest

from rfp_extractor.config import Settings, load_extraction_config
from rfp_extractor.models.document import Block, DocType, Document, FileFormat, Page
from rfp_extractor.models.schema import ExtractionMethod
from rfp_extractor.reasoning import extractor as extractor_mod
from rfp_extractor.reasoning.providers.base import build_provider
from rfp_extractor.reasoning.providers.openai_provider import OpenAIProvider

#: Only this field gets an answer; everything else returns null, which is how
#: the model is expected to behave for absent fields.
_ANSWERS: dict[str, str] = {"Payment Terms": "Net 30"}


class _FakeCompletions:
    """Routes by requested field so multi-field runs behave realistically."""

    def __init__(self, payloads: list[dict] | None = None) -> None:
        self.payloads = list(payloads or [])
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.payloads:
            payload = self.payloads.pop(0)
        else:
            request = json.loads(kwargs["messages"][1]["content"])
            value = _ANSWERS.get(request.get("field", ""))
            payload = (
                {"value": value, "reason": "stated", "evidence": value, "page": 2,
                 "confidence": 0.8}
                if value
                else {"value": None, "reason": "not stated", "evidence": "", "page": None,
                      "confidence": 0.1}
            )
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(content=json.dumps(payload))
                )
            ],
            usage=types.SimpleNamespace(prompt_tokens=1000, completion_tokens=200),
        )


class _FakeClient:
    def __init__(self, payloads: list[dict]) -> None:
        self.chat = types.SimpleNamespace(completions=_FakeCompletions(payloads))


def _doc() -> Document:
    page = Page(
        page_no=1,
        blocks=[Block(block_id="p1.b1", page_no=1, text="Solicitation Due 27-JUN-2024 14:00:00")],
    )
    return Document(
        doc_id="d1",
        file_name="rfp.pdf",
        source_path="rfp.pdf",
        file_format=FileFormat.PDF,
        content_hash="0" * 64,
        doc_type=DocType.MASTER_RFP,
        pages=[page],
    )


def _spec(name: str = "due_date"):
    return load_extraction_config().field(name)


def test_build_provider_is_none_for_rule_based_default():
    assert build_provider(Settings(provider="rule-based")) is None


def test_build_provider_needs_a_key():
    settings = Settings(provider="openai", openai_api_key=None)
    # no key in settings and none injected -> unavailable, not an exception
    try:
        assert build_provider(settings) is None
    except ImportError:
        pytest.skip("openai package not installed")


def test_openai_provider_parses_structured_output():
    client = _FakeClient(
        [{"value": "2024-07-09T14:00:00-05:00", "reason": "addendum", "evidence": "new due date",
          "page": 1, "confidence": 0.9}]
    )
    provider = OpenAIProvider(Settings(provider="openai", openai_api_key="sk-test"), client=client)
    assert provider.available()

    results = provider.extract_field(_doc(), _spec())
    assert len(results) == 1
    assert results[0].normalized == "2024-07-09T14:00:00-05:00"
    assert results[0].method is ExtractionMethod.LLM
    assert results[0].confidence == 0.9

    # request shape: structured output + deterministic decoding
    call = client.chat.completions.calls[0]
    assert call["temperature"] == 0
    assert call["response_format"]["type"] == "json_schema"
    assert call["model"] == "gpt-4o-mini"


def test_openai_provider_null_value_yields_no_candidate():
    client = _FakeClient([{"value": None, "reason": "not stated", "evidence": "", "page": None,
                           "confidence": 0.1}])
    provider = OpenAIProvider(Settings(provider="openai", openai_api_key="sk-test"), client=client)
    assert provider.extract_field(_doc(), _spec()) == []


def test_openai_provider_failures_are_isolated():
    class _Boom:
        chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(
                create=lambda **kw: (_ for _ in ()).throw(RuntimeError("api down"))
            )
        )

    provider = OpenAIProvider(Settings(provider="openai", openai_api_key="sk-test"), client=_Boom())
    assert provider.extract_field(_doc(), _spec()) == []


def test_provider_augments_only_unanswered_fields():
    """The model is a fallback, not a competitor to confident rule evidence."""
    config = load_extraction_config()
    engine = extractor_mod.ExtractionEngine(config, Settings(), provider=None)
    client = _FakeClient(
        [{"value": "Net 30", "reason": "stated", "evidence": "Net 30", "page": 2,
          "confidence": 0.8}]
    )
    engine.provider = OpenAIProvider(
        Settings(provider="openai", openai_api_key="sk-test"), client=client
    )

    doc = _doc()
    results = {spec.name: [] for spec in config.fields}
    augmented = engine._augment_with_provider(doc, results)

    # payment_terms had no rule candidate -> the model was consulted
    assert augmented["payment_terms"]
    assert augmented["payment_terms"][0].normalized == "Net 30"
    # every field the rules did not answer was offered to the model
    assert len(client.chat.completions.calls) > 1


def test_usage_is_tracked_and_priced():
    """The plan lists cost tracking; without this the metadata reads 0 forever."""
    client = _FakeClient(
        [{"value": "Net 30", "reason": "stated", "evidence": "Net 30", "page": 2,
          "confidence": 0.8}]
    )
    provider = OpenAIProvider(Settings(provider="openai", openai_api_key="sk-test"), client=client)
    provider.extract_field(_doc(), _spec("payment_terms"))

    usage = provider.usage
    assert usage["tokens"] == 1200          # 1000 prompt + 200 completion
    assert usage["calls"] == 1
    # gpt-4o-mini: $0.15 / $0.60 per 1M tokens
    expected = 1000 / 1e6 * 0.15 + 200 / 1e6 * 0.60
    assert usage["usd"] == pytest.approx(expected, rel=1e-6)


def test_usage_accumulates_across_calls():
    client = _FakeClient([])
    provider = OpenAIProvider(Settings(provider="openai", openai_api_key="sk-test"), client=client)
    provider.extract_field(_doc(), _spec("payment_terms"))
    provider.extract_field(_doc(), _spec("term_of_bid"))
    assert provider.usage["tokens"] == 2400
    assert provider.usage["calls"] == 2


def test_rule_based_run_reports_zero_cost():
    config = load_extraction_config()
    engine = extractor_mod.ExtractionEngine(config, Settings())
    assert extractor_mod.collect_usage(engine) == (0, 0.0)


def test_describe_provider_labels_the_run():
    config = load_extraction_config()
    engine = extractor_mod.ExtractionEngine(config, Settings())
    assert extractor_mod.describe_provider(engine) == "rule-based"

    engine.provider = OpenAIProvider(
        Settings(provider="openai", openai_api_key="sk-test"), client=_FakeClient([])
    )
    assert extractor_mod.describe_provider(engine) == "hybrid"
