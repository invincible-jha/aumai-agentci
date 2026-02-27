"""Tests for aumai_agentci.models — Pydantic v2 data models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aumai_agentci.models import (
    AgentTestCase,
    AgentTestConfig,
    AgentTestResult,
    MockLLMConfig,
    MockLLMResponse,
    TestSuiteResult,
)

# ---------------------------------------------------------------------------
# MockLLMResponse
# ---------------------------------------------------------------------------


class TestMockLLMResponse:
    """Unit tests for MockLLMResponse."""

    def test_valid_construction(self) -> None:
        response = MockLLMResponse(
            content="Hello",
            model="gpt-4o",
            tokens_used=5,
            latency_ms=100.0,
            finish_reason="stop",
        )
        assert response.content == "Hello"
        assert response.model == "gpt-4o"
        assert response.tokens_used == 5
        assert response.latency_ms == 100.0
        assert response.finish_reason == "stop"

    def test_default_finish_reason(self) -> None:
        response = MockLLMResponse(
            content="x",
            model="m",
            tokens_used=1,
            latency_ms=1.0,
        )
        assert response.finish_reason == "stop"

    def test_tokens_used_zero_is_valid(self) -> None:
        response = MockLLMResponse(
            content="", model="m", tokens_used=0, latency_ms=0.0
        )
        assert response.tokens_used == 0

    def test_tokens_used_negative_raises(self) -> None:
        with pytest.raises(ValidationError):
            MockLLMResponse(
                content="", model="m", tokens_used=-1, latency_ms=0.0
            )

    def test_latency_ms_negative_raises(self) -> None:
        with pytest.raises(ValidationError):
            MockLLMResponse(
                content="", model="m", tokens_used=0, latency_ms=-0.1
            )

    def test_large_token_count(self) -> None:
        response = MockLLMResponse(
            content="x", model="m", tokens_used=1_000_000, latency_ms=9999.9
        )
        assert response.tokens_used == 1_000_000

    def test_model_dump_contains_all_fields(self) -> None:
        response = MockLLMResponse(
            content="hi", model="m", tokens_used=3, latency_ms=10.0
        )
        dumped = response.model_dump()
        assert set(dumped.keys()) == {
            "content",
            "model",
            "tokens_used",
            "latency_ms",
            "finish_reason",
        }


# ---------------------------------------------------------------------------
# MockLLMConfig
# ---------------------------------------------------------------------------


class TestMockLLMConfig:
    """Unit tests for MockLLMConfig."""

    def test_valid_construction_empty_responses(self) -> None:
        config = MockLLMConfig(model_name="test-model")
        assert config.model_name == "test-model"
        assert config.responses == []
        assert config.default_latency_ms == 50.0
        assert config.failure_rate == 0.0

    def test_failure_rate_zero(self) -> None:
        config = MockLLMConfig(model_name="m", failure_rate=0.0)
        assert config.failure_rate == 0.0

    def test_failure_rate_one(self) -> None:
        config = MockLLMConfig(model_name="m", failure_rate=1.0)
        assert config.failure_rate == 1.0

    def test_failure_rate_above_one_raises(self) -> None:
        with pytest.raises(ValidationError):
            MockLLMConfig(model_name="m", failure_rate=1.01)

    def test_failure_rate_below_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            MockLLMConfig(model_name="m", failure_rate=-0.01)

    def test_default_latency_ms_negative_raises(self) -> None:
        with pytest.raises(ValidationError):
            MockLLMConfig(model_name="m", default_latency_ms=-1.0)

    def test_responses_list_is_preserved(self) -> None:
        resp = MockLLMResponse(
            content="r", model="m", tokens_used=1, latency_ms=1.0
        )
        config = MockLLMConfig(model_name="m", responses=[resp])
        assert len(config.responses) == 1
        assert config.responses[0].content == "r"


# ---------------------------------------------------------------------------
# AgentTestCase
# ---------------------------------------------------------------------------


class TestAgentTestCase:
    """Unit tests for AgentTestCase."""

    def test_minimal_construction(self) -> None:
        tc = AgentTestCase(name="my_test")
        assert tc.name == "my_test"
        assert tc.description == ""
        assert tc.input_messages == []
        assert tc.expected_behavior == {}
        assert tc.tags == []

    def test_full_construction(self) -> None:
        tc = AgentTestCase(
            name="full",
            description="Full test case.",
            input_messages=[{"role": "user", "content": "Hi"}],
            expected_behavior={"contains_text": "hello"},
            tags=["smoke", "ci"],
        )
        assert tc.name == "full"
        assert len(tc.input_messages) == 1
        assert tc.expected_behavior == {"contains_text": "hello"}
        assert tc.tags == ["smoke", "ci"]

    def test_all_allowed_expected_behavior_keys(self) -> None:
        tc = AgentTestCase(
            name="full_behavior",
            expected_behavior={
                "contains_text": "hello",
                "calls_tools": "search_web",
                "max_tokens": 100,
                "max_latency_ms": 500.0,
                "no_pii": True,
                "valid_json": False,
                "matches_schema": {"type": "object"},
            },
        )
        assert "contains_text" in tc.expected_behavior

    def test_unknown_expected_behavior_key_raises(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            AgentTestCase(
                name="bad",
                expected_behavior={"unknown_key": "value"},
            )
        assert "unknown_key" in str(exc_info.value).lower() or "unknown" in str(
            exc_info.value
        )

    def test_multiple_unknown_keys_raises(self) -> None:
        with pytest.raises(ValidationError):
            AgentTestCase(
                name="bad",
                expected_behavior={"foo": 1, "bar": 2},
            )

    def test_name_is_required(self) -> None:
        with pytest.raises(ValidationError):
            AgentTestCase()  # type: ignore[call-arg]

    def test_model_validate_from_dict(self) -> None:
        data = {
            "name": "from_dict",
            "description": "Built from a mapping.",
            "input_messages": [{"role": "user", "content": "Go."}],
            "expected_behavior": {"no_pii": True},
            "tags": ["tag1"],
        }
        tc = AgentTestCase.model_validate(data)
        assert tc.name == "from_dict"


# ---------------------------------------------------------------------------
# AgentTestResult
# ---------------------------------------------------------------------------


class TestAgentTestResult:
    """Unit tests for AgentTestResult."""

    def test_passing_result(self) -> None:
        result = AgentTestResult(
            test_case_name="tc",
            passed=True,
            actual_output="Hello",
            assertions_passed=["contains_text: 'hello'"],
            assertions_failed=[],
            duration_ms=10.0,
            tokens_used=5,
        )
        assert result.passed is True
        assert result.assertions_failed == []

    def test_failing_result(self) -> None:
        result = AgentTestResult(
            test_case_name="tc",
            passed=False,
            actual_output="Nope",
            assertions_passed=[],
            assertions_failed=["contains_text: 'hello' not found"],
            duration_ms=8.0,
            tokens_used=3,
        )
        assert result.passed is False
        assert len(result.assertions_failed) == 1

    def test_duration_ms_negative_raises(self) -> None:
        with pytest.raises(ValidationError):
            AgentTestResult(
                test_case_name="tc",
                passed=True,
                actual_output="x",
                duration_ms=-1.0,
                tokens_used=0,
            )

    def test_tokens_used_defaults_to_zero(self) -> None:
        result = AgentTestResult(
            test_case_name="tc",
            passed=True,
            actual_output="x",
            duration_ms=0.0,
        )
        assert result.tokens_used == 0


# ---------------------------------------------------------------------------
# TestSuiteResult
# ---------------------------------------------------------------------------


class TestTestSuiteResult:
    """Unit tests for TestSuiteResult."""

    def test_empty_suite(self) -> None:
        suite = TestSuiteResult(
            suite_name="empty",
            results=[],
            total=0,
            passed=0,
            failed=0,
            duration_ms=0.0,
        )
        assert suite.total == 0
        assert suite.passed == 0
        assert suite.failed == 0

    def test_suite_with_results(
        self, passing_result: AgentTestResult, failing_result: AgentTestResult
    ) -> None:
        suite = TestSuiteResult(
            suite_name="my_suite",
            results=[passing_result, failing_result],
            total=2,
            passed=1,
            failed=1,
            duration_ms=150.0,
        )
        assert suite.suite_name == "my_suite"
        assert len(suite.results) == 2

    def test_total_negative_raises(self) -> None:
        with pytest.raises(ValidationError):
            TestSuiteResult(
                suite_name="s",
                total=-1,
                passed=0,
                failed=0,
                duration_ms=0.0,
            )

    def test_model_dump_round_trip(
        self, mixed_suite_result: TestSuiteResult
    ) -> None:
        dumped = mixed_suite_result.model_dump()
        restored = TestSuiteResult.model_validate(dumped)
        assert restored.suite_name == mixed_suite_result.suite_name
        assert restored.total == mixed_suite_result.total


# ---------------------------------------------------------------------------
# AgentTestConfig
# ---------------------------------------------------------------------------


class TestAgentTestConfig:
    """Unit tests for AgentTestConfig."""

    def test_minimal_construction(self) -> None:
        config = AgentTestConfig(test_dir="tests/")
        assert config.test_dir == "tests/"
        assert config.mock_config is None
        assert config.timeout_seconds == 30.0
        assert config.parallel is False

    def test_custom_timeout(self) -> None:
        config = AgentTestConfig(test_dir=".", timeout_seconds=60.0)
        assert config.timeout_seconds == 60.0

    def test_timeout_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            AgentTestConfig(test_dir=".", timeout_seconds=0.0)

    def test_timeout_negative_raises(self) -> None:
        with pytest.raises(ValidationError):
            AgentTestConfig(test_dir=".", timeout_seconds=-1.0)

    def test_parallel_flag(self) -> None:
        config = AgentTestConfig(test_dir=".", parallel=True)
        assert config.parallel is True

    def test_with_mock_config(self, single_response_config: MockLLMConfig) -> None:
        config = AgentTestConfig(
            test_dir="tests/",
            mock_config=single_response_config,
        )
        assert config.mock_config is not None
        assert config.mock_config.model_name == "mock-model"
