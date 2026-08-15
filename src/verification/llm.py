"""Thin wrapper around the Anthropic client, shared by all three agents.

Centralizing API calls here means:
  - every agent gets the same model/timeout defaults from config.py
  - the eval harness can measure latency and token usage uniformly, since
    every call returns them alongside its result
  - tests can monkeypatch `get_client()` to inject a fake client instead of
    patching each agent module individually

Structured outputs (`output_config.format` with a JSON schema) are used
instead of "ask for JSON in the prompt and hope" for the Critic and
Coordinator calls, because their outputs feed directly into the graph's
control-flow decisions — a malformed response there can't be allowed to
silently break the revise/accept/escalate routing.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import anthropic

from verification.config import settings

_client: anthropic.Anthropic | None = None


def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=settings.anthropic_api_key or None)
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


def call_structured(
    system: str,
    user: str,
    schema: dict[str, Any],
    max_tokens: int = 2048,
) -> StructuredCallResult:
    client = get_client()
    start = time.perf_counter()
    response = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
        output_config={"format": {"type": "json_schema", "schema": schema}},
    )
    latency_ms = (time.perf_counter() - start) * 1000

    text = next(block.text for block in response.content if block.type == "text")
    return StructuredCallResult(
        data=json.loads(text),
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
    response = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    latency_ms = (time.perf_counter() - start) * 1000

    text = next(block.text for block in response.content if block.type == "text")
    return TextCallResult(
        text=text,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        latency_ms=latency_ms,
    )
