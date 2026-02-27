"""AumAI AgentCI — CI/CD testing framework for AI agents."""

from __future__ import annotations

from aumai_agentci.assertions import (
    assert_calls_tool,
    assert_contains_text,
    assert_matches_schema,
    assert_max_latency,
    assert_max_tokens,
    assert_no_pii,
    assert_valid_json,
)
from aumai_agentci.core import AgentTestRunner, MockLLMProvider
from aumai_agentci.fixtures import (
    create_mock_anthropic,
    create_mock_openai,
    load_test_suite,
)
from aumai_agentci.models import (
    AgentTestCase,
    AgentTestConfig,
    AgentTestResult,
    MockLLMConfig,
    MockLLMResponse,
    TestSuiteResult,
)
from aumai_agentci.reporter import ConsoleReporter, JSONReporter, JUnitReporter

__version__ = "0.1.0"

__all__ = [
    # models
    "MockLLMResponse",
    "MockLLMConfig",
    "AgentTestCase",
    "AgentTestResult",
    "TestSuiteResult",
    "AgentTestConfig",
    # core
    "MockLLMProvider",
    "AgentTestRunner",
    # assertions
    "assert_contains_text",
    "assert_calls_tool",
    "assert_no_pii",
    "assert_max_tokens",
    "assert_max_latency",
    "assert_valid_json",
    "assert_matches_schema",
    # fixtures
    "create_mock_openai",
    "create_mock_anthropic",
    "load_test_suite",
    # reporters
    "ConsoleReporter",
    "JSONReporter",
    "JUnitReporter",
]
