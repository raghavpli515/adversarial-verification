"""Thin wrapper around the OpenAI client, shared by all three agents.

Centralizing API calls here means:
  - every agent gets the same model/timeout defaults from config.py
  - the eval harness can measure latency and token usage uniformly, since
    every call returns them alongside its result
  - tests can monkeypatch `get_client()` to inject a fake client instead of
    patching each agent module individually
  - swapping the underlying provider (this file went from Anthropic to
    OpenAI) only touches this one module — every agent, the graph, and all
    tests depend on `StructuredCallResult`/`TextCallResult`, never on the
    provider SDK directly.

Uses the Responses API (`client.responses.create`), not the older Chat
Completions API — OpenAI's current docs explicitly recommend Responses for
new projects. Structured outputs go through `text.format` with a strict
JSON schema (verified against the openai-python SDK source directly, not
guessed: `strict` is a sibling of `type` inside `format`, not nested under
a `json_schema` key the way Chat Completions shapes it). Strict mode
requires every object in the schema to set `"additionalProperties": false`
and list every property as required — all of this project's schemas
(generator/critic/confidence/judge) already meet that bar.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from verification.config import settings

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.openai_api_key or None)
    return _client


@dataclass
class StructuredCallResult:
    data: dict[str, Any]
    input_tokens: int
    output_tokens: int
    latency_ms: float


@dataclass
class TextCallResult:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


def _extract_output_text(response: Any) -> str:
    """Walk response.output for the first output_text part. Deliberately
    not relying on the `response.output_parsed` convenience attribute —
    that behavior isn't documented as guaranteed across SDK versions, so we
    parse the raw text ourselves (json.loads on it in call_structured),
    matching the reliability bar the rest of this codebase holds retrieval
    and confidence scoring to."""
    for item in response.output:
        if item.type == "message":
            for part in item.content:
                if part.type == "output_text":
                    return part.text
                if part.type == "refusal":
                    raise ValueError(f"Model refused: {part.refusal}")
    raise ValueError("No output_text found in response.output")


def call_structured(
    system: str,
    user: str,
    schema: dict[str, Any],
    max_tokens: int = 2048,
) -> StructuredCallResult:
    client = get_client()
    start = time.perf_counter()
    response = client.responses.create(
        model=settings.openai_model,
        max_output_tokens=max_tokens,
        instructions=system,
        input=user,
        text={
            "format": {
                "type": "json_schema",
                "name": "structured_response",
                "strict": True,
                "schema": schema,
            }
        },
    )
    latency_ms = (time.perf_counter() - start) * 1000

    return StructuredCallResult(
        data=json.loads(_extract_output_text(response)),
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        latency_ms=latency_ms,
    )


def call_text(system: str, user: str, max_tokens: int = 2048) -> TextCallResult:
    """Plain free-text call. Not currently used by any agent (the Generator
    also uses structured output, for its citation list) — kept for
    completeness / stretch-goal use (e.g. a free-text explanation field)."""
    client = get_client()
    start = time.perf_counter()
    response = client.responses.create(
        model=settings.openai_model,
        max_output_tokens=max_tokens,
        instructions=system,
        input=user,
    )
    latency_ms = (time.perf_counter() - start) * 1000

    return TextCallResult(
        text=_extract_output_text(response),
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        latency_ms=latency_ms,
    )
