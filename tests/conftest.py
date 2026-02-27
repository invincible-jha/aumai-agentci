"""Shared pytest fixtures for the aumai-agentci test suite."""

from __future__ import annotations

import textwrap
from collections.abc import Generator
from pathlib import Path

import pytest

from aumai_agentci.core import AgentTestRunner, MockLLMProvider
from aumai_agentci.models import (
    AgentTestCase,
    AgentTestResult,
    MockLLMConfig,
    MockLLMResponse,
    TestSuiteResult,
)

# ---------------------------------------------------------------------------
# MockLLMResponse builders
# ---------------------------------------------------------------------------


@pytest.fixture()
def plain_response() -> MockLLMResponse:
    """A simple plain-text mock response."""
    return MockLLMResponse(
        content="Hello, world! Task completed successfully.",
        model="mock-model",
        tokens_used=10,
        latency_ms=50.0,
        finish_reason="stop",
    )


@pytest.fixture()
def json_response() -> MockLLMResponse:
    """A mock response whose content is valid JSON."""
    return MockLLMResponse(
        content='{"status": "ok", "value": 42}',
        model="mock-model",
        tokens_used=20,
        latency_ms=60.0,
        finish_reason="stop",
    )


@pytest.fixture()
def tool_call_response() -> MockLLMResponse:
    """A mock response that contains an OpenAI-style tool call."""
    return MockLLMResponse(
        content=(
            '{"tool_calls": [{"id": "call_1", "type": "function",'
            ' "function": {"name": "search_web",'
            ' "arguments": "{\\"query\\": \\"test\\"}"}}]}'
        ),
        model="mock-model",
        tokens_used=30,
        latency_ms=80.0,
        finish_reason="tool_calls",
    )


# ---------------------------------------------------------------------------
# MockLLMConfig fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def single_response_config(plain_response: MockLLMResponse) -> MockLLMConfig:
    """Config with exactly one response."""
    return MockLLMConfig(
        model_name="mock-model",
        responses=[plain_response],
        default_latency_ms=50.0,
        failure_rate=0.0,
    )


@pytest.fixture()
def multi_response_config(
    plain_response: MockLLMResponse,
    json_response: MockLLMResponse,
    tool_call_response: MockLLMResponse,
) -> MockLLMConfig:
    """Config with three responses for round-robin cycling."""
    return MockLLMConfig(
        model_name="mock-model",
        responses=[plain_response, json_response, tool_call_response],
        default_latency_ms=50.0,
        failure_rate=0.0,
    )


@pytest.fixture()
def empty_response_config() -> MockLLMConfig:
    """Config with no responses — triggers the synthetic fallback."""
    return MockLLMConfig(
        model_name="mock-model",
        responses=[],
        default_latency_ms=25.0,
        failure_rate=0.0,
    )


@pytest.fixture()
def always_failing_config(plain_response: MockLLMResponse) -> MockLLMConfig:
    """Config with failure_rate=1.0 — every call raises RuntimeError."""
    return MockLLMConfig(
        model_name="mock-model",
        responses=[plain_response],
        default_latency_ms=50.0,
        failure_rate=1.0,
    )


# ---------------------------------------------------------------------------
# MockLLMProvider fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_provider(single_response_config: MockLLMConfig) -> MockLLMProvider:
    """A ready-to-use MockLLMProvider backed by a single response."""
    return MockLLMProvider(single_response_config)


@pytest.fixture()
def multi_provider(multi_response_config: MockLLMConfig) -> MockLLMProvider:
    """A MockLLMProvider that cycles through three responses."""
    return MockLLMProvider(multi_response_config)


@pytest.fixture()
def failing_provider(always_failing_config: MockLLMConfig) -> MockLLMProvider:
    """A MockLLMProvider guaranteed to raise RuntimeError on every call."""
    return MockLLMProvider(always_failing_config)


# ---------------------------------------------------------------------------
# AgentTestCase fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def basic_test_case() -> AgentTestCase:
    """A test case that checks for text containment."""
    return AgentTestCase(
        name="basic_response",
        description="Verify the agent greets the user.",
        input_messages=[{"role": "user", "content": "Say hello."}],
        expected_behavior={"contains_text": "hello"},
        tags=["smoke"],
    )


@pytest.fixture()
def no_pii_test_case() -> AgentTestCase:
    """A test case that asserts no PII in the output."""
    return AgentTestCase(
        name="no_pii_check",
        description="Agent must not leak PII.",
        input_messages=[{"role": "user", "content": "Who are you?"}],
        expected_behavior={"no_pii": True},
    )


@pytest.fixture()
def json_test_case() -> AgentTestCase:
    """A test case that expects valid JSON output."""
    return AgentTestCase(
        name="json_output",
        description="Agent returns structured JSON.",
        input_messages=[{"role": "user", "content": "Return JSON."}],
        expected_behavior={
            "valid_json": True,
            "matches_schema": {
                "type": "object",
                "required": ["status"],
                "properties": {"status": {"type": "string"}},
            },
        },
    )


@pytest.fixture()
def token_budget_test_case() -> AgentTestCase:
    """A test case that enforces a token budget."""
    return AgentTestCase(
        name="token_budget",
        description="Response must stay within 50 tokens.",
        input_messages=[{"role": "user", "content": "Be brief."}],
        expected_behavior={"max_tokens": 50},
    )


@pytest.fixture()
def tool_call_test_case() -> AgentTestCase:
    """A test case that expects the agent to call search_web."""
    return AgentTestCase(
        name="tool_call_check",
        description="Agent must call the search_web tool.",
        input_messages=[{"role": "user", "content": "Search the web."}],
        expected_behavior={"calls_tools": "search_web"},
    )


# ---------------------------------------------------------------------------
# AgentTestResult / TestSuiteResult fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def passing_result() -> AgentTestResult:
    """An AgentTestResult representing a fully passing test."""
    return AgentTestResult(
        test_case_name="passing_test",
        passed=True,
        actual_output="Hello, world!",
        assertions_passed=["contains_text: 'hello'"],
        assertions_failed=[],
        duration_ms=12.5,
        tokens_used=10,
    )


@pytest.fixture()
def failing_result() -> AgentTestResult:
    """An AgentTestResult representing a failing test."""
    return AgentTestResult(
        test_case_name="failing_test",
        passed=False,
        actual_output="No match here.",
        assertions_passed=[],
        assertions_failed=["contains_text: 'hello' not found in output"],
        duration_ms=8.0,
        tokens_used=5,
    )


@pytest.fixture()
def mixed_suite_result(
    passing_result: AgentTestResult,
    failing_result: AgentTestResult,
) -> TestSuiteResult:
    """A TestSuiteResult containing one pass and one failure."""
    return TestSuiteResult(
        suite_name="mixed_suite",
        results=[passing_result, failing_result],
        total=2,
        passed=1,
        failed=1,
        duration_ms=100.0,
    )


@pytest.fixture()
def all_passing_suite(passing_result: AgentTestResult) -> TestSuiteResult:
    """A TestSuiteResult where every test passes."""
    return TestSuiteResult(
        suite_name="green_suite",
        results=[passing_result],
        total=1,
        passed=1,
        failed=0,
        duration_ms=50.0,
    )


# ---------------------------------------------------------------------------
# AgentTestRunner fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def runner() -> AgentTestRunner:
    """A bare AgentTestRunner instance."""
    return AgentTestRunner()


# ---------------------------------------------------------------------------
# Temporary YAML test-dir fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def yaml_test_dir(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a temporary directory with two YAML test files and yield its path."""
    single_file = tmp_path / "test_single.yaml"
    single_file.write_text(
        textwrap.dedent("""\
            name: greet_user
            description: Agent should greet the user.
            input_messages:
              - role: user
                content: "Say hello."
            expected_behavior:
              contains_text: "hello"
            tags:
              - smoke
        """),
        encoding="utf-8",
    )

    multi_file = tmp_path / "test_multi.yaml"
    multi_file.write_text(
        textwrap.dedent("""\
            tests:
              - name: check_no_pii
                description: No PII expected.
                input_messages:
                  - role: user
                    content: "What is your name?"
                expected_behavior:
                  no_pii: true

              - name: check_tokens
                description: Token budget check.
                input_messages:
                  - role: user
                    content: "Be concise."
                expected_behavior:
                  max_tokens: 100
        """),
        encoding="utf-8",
    )

    yield tmp_path


@pytest.fixture()
def yaml_list_file(tmp_path: Path) -> Path:
    """Create a YAML file that is a top-level list (not a mapping)."""
    list_file = tmp_path / "list_tests.yaml"
    list_file.write_text(
        textwrap.dedent("""\
            - name: list_case_1
              input_messages:
                - role: user
                  content: "Hello"
              expected_behavior:
                no_pii: true

            - name: list_case_2
              input_messages:
                - role: user
                  content: "World"
              expected_behavior:
                max_tokens: 200
        """),
        encoding="utf-8",
    )
    return list_file


@pytest.fixture()
def mock_config_yaml(tmp_path: Path) -> Path:
    """Write a minimal mock config YAML file in its own subdirectory.

    The file is placed in a dedicated ``mock_cfg/`` subfolder so that the
    AgentTestRunner does not accidentally try to parse it as a test case
    when the caller also has test YAML files in *tmp_path*.
    """
    cfg_dir = tmp_path / "mock_cfg"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    config_file = cfg_dir / "mock_config.yaml"
    config_file.write_text(
        textwrap.dedent("""\
            model_name: mock-gpt-4o
            default_latency_ms: 50.0
            failure_rate: 0.0
            responses:
              - content: "Hello! I am a mock response."
                model: mock-gpt-4o
                tokens_used: 12
                latency_ms: 50.0
                finish_reason: stop
        """),
        encoding="utf-8",
    )
    return config_file
