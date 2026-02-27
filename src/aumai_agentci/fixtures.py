"""Reusable test fixtures for aumai-agentci.

These helpers give test authors pre-built mock providers and convenience
loaders so they don't have to construct :class:`MockLLMConfig` by hand.
"""

from __future__ import annotations

from aumai_agentci.core import AgentTestRunner, MockLLMProvider
from aumai_agentci.models import AgentTestCase, MockLLMConfig, MockLLMResponse

__all__ = [
    "create_mock_openai",
    "create_mock_anthropic",
    "load_test_suite",
]

# ---------------------------------------------------------------------------
# Pre-configured provider factories
# ---------------------------------------------------------------------------


def create_mock_openai(
    *,
    responses: list[MockLLMResponse] | None = None,
    failure_rate: float = 0.0,
) -> MockLLMProvider:
    """Return a :class:`MockLLMProvider` configured to mimic OpenAI's API.

    The default response set covers the most common patterns seen in
    GPT-4o / GPT-4-turbo outputs: a plain text reply, a JSON-structured
    reply, and a tool-call reply.

    Args:
        responses: Override the default response list.  Pass an empty list
            to get a provider with zero pre-loaded responses (the provider
            will synthesise an empty response).
        failure_rate: Fraction of calls that should raise a
            :class:`RuntimeError` to simulate transient errors.

    Returns:
        A ready-to-use :class:`MockLLMProvider`.
    """
    model_name = "gpt-4o"
    default_responses: list[MockLLMResponse] = [
        MockLLMResponse(
            content="I understand your request. Here is my response.",
            model=model_name,
            tokens_used=28,
            latency_ms=320.0,
            finish_reason="stop",
        ),
        MockLLMResponse(
            content='{"answer": "42", "confidence": 0.95, "reasoning": "computed"}',
            model=model_name,
            tokens_used=42,
            latency_ms=410.0,
            finish_reason="stop",
        ),
        MockLLMResponse(
            content=(
                '{"tool_calls": [{"id": "call_abc", "type": "function",'
                ' "function": {"name": "search_web",'
                ' "arguments": "{\\"query\\": \\"latest news\\"}"}}]}'
            ),
            model=model_name,
            tokens_used=55,
            latency_ms=280.0,
            finish_reason="tool_calls",
        ),
    ]

    config = MockLLMConfig(
        model_name=model_name,
        responses=responses if responses is not None else default_responses,
        default_latency_ms=350.0,
        failure_rate=failure_rate,
    )
    return MockLLMProvider(config)


def create_mock_anthropic(
    *,
    responses: list[MockLLMResponse] | None = None,
    failure_rate: float = 0.0,
) -> MockLLMProvider:
    """Return a :class:`MockLLMProvider` configured to mimic Anthropic's API.

    The default response set covers Claude-style outputs: a plain reply,
    a structured JSON reply, and a tool-use reply matching the Anthropic
    messages API format.

    Args:
        responses: Override the default response list.
        failure_rate: Fraction of calls that should simulate API failures.

    Returns:
        A ready-to-use :class:`MockLLMProvider`.
    """
    model_name = "claude-opus-4-6"
    default_responses: list[MockLLMResponse] = [
        MockLLMResponse(
            content=(
                "I'd be happy to help with that. "
                "Based on the information provided, here is my analysis."
            ),
            model=model_name,
            tokens_used=35,
            latency_ms=290.0,
            finish_reason="end_turn",
        ),
        MockLLMResponse(
            content='{"status": "success", "data": {"result": "processed"}}',
            model=model_name,
            tokens_used=48,
            latency_ms=375.0,
            finish_reason="end_turn",
        ),
        MockLLMResponse(
            content=(
                '{"type": "tool_use", "id": "toolu_01", "name": "calculator",'
                ' "input": {"expression": "2 + 2"}}'
            ),
            model=model_name,
            tokens_used=62,
            latency_ms=310.0,
            finish_reason="tool_use",
        ),
    ]

    config = MockLLMConfig(
        model_name=model_name,
        responses=responses if responses is not None else default_responses,
        default_latency_ms=320.0,
        failure_rate=failure_rate,
    )
    return MockLLMProvider(config)


# ---------------------------------------------------------------------------
# Test suite loader
# ---------------------------------------------------------------------------


def load_test_suite(path: str) -> list[AgentTestCase]:
    """Load test cases from *path* (file or directory).

    When *path* points to a directory all ``.yaml`` / ``.yml`` files
    inside it are discovered recursively.  When it points to a single
    file that file is parsed directly.

    Args:
        path: Filesystem path to a YAML file or a directory.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If a YAML file cannot be parsed.

    Returns:
        List of :class:`AgentTestCase` objects.
    """
    from pathlib import Path as _Path

    target = _Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    runner = AgentTestRunner()
    if target.is_dir():
        return runner.load_tests(str(target))

    # Single file — delegate to the public loader.
    return runner.load_yaml_file(target)
