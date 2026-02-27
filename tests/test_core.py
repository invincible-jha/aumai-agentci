"""Tests for aumai_agentci.core — MockLLMProvider and AgentTestRunner."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from aumai_agentci.core import AgentTestRunner, MockLLMProvider, _default_mock_config
from aumai_agentci.models import (
    AgentTestCase,
    AgentTestConfig,
    MockLLMConfig,
    MockLLMResponse,
    TestSuiteResult,
)

# ===========================================================================
# MockLLMProvider
# ===========================================================================


class TestMockLLMProviderInit:
    """Tests for MockLLMProvider construction and initial state."""

    def test_call_count_starts_at_zero(
        self, single_response_config: MockLLMConfig
    ) -> None:
        provider = MockLLMProvider(single_response_config)
        assert provider.call_count == 0

    def test_reset_clears_state(self, multi_provider: MockLLMProvider) -> None:
        multi_provider.complete([])
        multi_provider.complete([])
        multi_provider.reset()
        assert multi_provider.call_count == 0

    def test_reset_restarts_round_robin(
        self, multi_provider: MockLLMProvider, multi_response_config: MockLLMConfig
    ) -> None:
        first_before_reset = multi_provider.complete([]).content
        multi_provider.reset()
        first_after_reset = multi_provider.complete([]).content
        assert first_before_reset == first_after_reset


class TestMockLLMProviderComplete:
    """Tests for MockLLMProvider.complete()."""

    def test_returns_correct_response(self, mock_provider: MockLLMProvider) -> None:
        response = mock_provider.complete([{"role": "user", "content": "Hi"}])
        assert response.content == "Hello, world! Task completed successfully."

    def test_increments_call_count(self, mock_provider: MockLLMProvider) -> None:
        mock_provider.complete([])
        mock_provider.complete([])
        assert mock_provider.call_count == 2

    def test_round_robin_cycling(
        self, multi_provider: MockLLMProvider, multi_response_config: MockLLMConfig
    ) -> None:
        expected_contents = [r.content for r in multi_response_config.responses]
        num_responses = len(expected_contents)

        # Cycle through twice
        for cycle in range(2):
            for i, expected in enumerate(expected_contents):
                call_num = cycle * num_responses + i + 1
                response = multi_provider.complete([])
                assert response.content == expected, (
                    f"Call #{call_num}: expected '{expected}', got '{response.content}'"
                )

    def test_empty_responses_returns_synthetic(
        self, empty_response_config: MockLLMConfig
    ) -> None:
        provider = MockLLMProvider(empty_response_config)
        response = provider.complete([])
        assert response.content == ""
        assert response.model == empty_response_config.model_name
        assert response.tokens_used == 0
        assert response.latency_ms == empty_response_config.default_latency_ms
        assert response.finish_reason == "stop"

    def test_empty_responses_still_increments_count(
        self, empty_response_config: MockLLMConfig
    ) -> None:
        provider = MockLLMProvider(empty_response_config)
        provider.complete([])
        provider.complete([])
        assert provider.call_count == 2

    def test_messages_argument_is_accepted(
        self, mock_provider: MockLLMProvider
    ) -> None:
        # messages is unused but must not raise
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Do something."},
        ]
        response = mock_provider.complete(messages)
        assert isinstance(response, MockLLMResponse)

    def test_failure_rate_zero_never_raises(
        self, single_response_config: MockLLMConfig
    ) -> None:
        provider = MockLLMProvider(single_response_config)
        for _ in range(20):
            response = provider.complete([])
            assert isinstance(response, MockLLMResponse)

    def test_failure_rate_one_always_raises(
        self, failing_provider: MockLLMProvider
    ) -> None:
        with pytest.raises(RuntimeError, match="Simulated LLM API failure"):
            failing_provider.complete([], seed=0)

    def test_failure_rate_one_increments_count_before_raise(
        self, always_failing_config: MockLLMConfig
    ) -> None:
        provider = MockLLMProvider(always_failing_config)
        with pytest.raises(RuntimeError):
            provider.complete([], seed=0)
        assert provider.call_count == 1

    def test_deterministic_failure_with_seed(
        self, plain_response: MockLLMResponse
    ) -> None:
        """Same seed and failure_rate=0.5 must produce the same result each time."""
        config = MockLLMConfig(
            model_name="m",
            responses=[plain_response],
            failure_rate=0.5,
        )
        provider_a = MockLLMProvider(config)
        provider_b = MockLLMProvider(config)

        results_a: list[str] = []
        results_b: list[str] = []

        for seed in range(10):
            try:
                provider_a.complete([], seed=seed)
                results_a.append("ok")
            except RuntimeError:
                results_a.append("err")

            try:
                provider_b.complete([], seed=seed)
                results_b.append("ok")
            except RuntimeError:
                results_b.append("err")

        assert results_a == results_b

    def test_error_message_contains_failure_rate_and_call_count(
        self, always_failing_config: MockLLMConfig
    ) -> None:
        provider = MockLLMProvider(always_failing_config)
        with pytest.raises(RuntimeError) as exc_info:
            provider.complete([], seed=42)
        error_text = str(exc_info.value)
        assert "failure_rate=1.00" in error_text
        assert "call #1" in error_text


class TestMockLLMProviderCallCountProperty:
    """Tests for the call_count property."""

    def test_is_read_only_via_property(self, mock_provider: MockLLMProvider) -> None:
        # call_count is a @property — accessing it should return an int
        count = mock_provider.call_count
        assert isinstance(count, int)


# ===========================================================================
# AgentTestRunner — load_tests
# ===========================================================================


class TestAgentTestRunnerLoadTests:
    """Tests for AgentTestRunner.load_tests()."""

    def test_loads_single_mapping_yaml(
        self, runner: AgentTestRunner, yaml_test_dir: Path
    ) -> None:
        cases = runner.load_tests(str(yaml_test_dir))
        names = [tc.name for tc in cases]
        assert "greet_user" in names

    def test_loads_grouped_yaml_with_tests_key(
        self, runner: AgentTestRunner, yaml_test_dir: Path
    ) -> None:
        cases = runner.load_tests(str(yaml_test_dir))
        names = [tc.name for tc in cases]
        assert "check_no_pii" in names
        assert "check_tokens" in names

    def test_total_count_across_multiple_files(
        self, runner: AgentTestRunner, yaml_test_dir: Path
    ) -> None:
        cases = runner.load_tests(str(yaml_test_dir))
        assert len(cases) == 3  # 1 from single file + 2 from grouped file

    def test_loads_list_format_yaml(
        self, runner: AgentTestRunner, yaml_list_file: Path
    ) -> None:
        cases = runner.load_tests(str(yaml_list_file.parent))
        names = [tc.name for tc in cases]
        assert "list_case_1" in names
        assert "list_case_2" in names

    def test_nonexistent_directory_raises(self, runner: AgentTestRunner) -> None:
        with pytest.raises(FileNotFoundError):
            runner.load_tests("/nonexistent/path/to/tests")

    def test_empty_directory_returns_empty_list(
        self, runner: AgentTestRunner, tmp_path: Path
    ) -> None:
        cases = runner.load_tests(str(tmp_path))
        assert cases == []

    def test_empty_yaml_file_is_skipped(
        self, runner: AgentTestRunner, tmp_path: Path
    ) -> None:
        empty_file = tmp_path / "empty.yaml"
        empty_file.write_text("", encoding="utf-8")
        cases = runner.load_tests(str(tmp_path))
        assert cases == []

    def test_recursive_discovery(
        self, runner: AgentTestRunner, tmp_path: Path
    ) -> None:
        subdir = tmp_path / "nested" / "subdir"
        subdir.mkdir(parents=True)
        (subdir / "deep_test.yaml").write_text(
            textwrap.dedent("""\
                name: deep_test
                input_messages:
                  - role: user
                    content: "Deep test"
                expected_behavior:
                  no_pii: true
            """),
            encoding="utf-8",
        )
        cases = runner.load_tests(str(tmp_path))
        assert any(tc.name == "deep_test" for tc in cases)

    def test_yml_extension_discovered(
        self, runner: AgentTestRunner, tmp_path: Path
    ) -> None:
        (tmp_path / "test_case.yml").write_text(
            textwrap.dedent("""\
                name: yml_test
                expected_behavior:
                  no_pii: true
            """),
            encoding="utf-8",
        )
        cases = runner.load_tests(str(tmp_path))
        assert any(tc.name == "yml_test" for tc in cases)

    def test_invalid_yaml_structure_raises_value_error(
        self, runner: AgentTestRunner, tmp_path: Path
    ) -> None:
        bad_file = tmp_path / "bad.yaml"
        # A plain scalar at the top level — not a dict or list
        bad_file.write_text("just_a_string", encoding="utf-8")
        with pytest.raises(ValueError, match="Unexpected YAML structure"):
            runner.load_tests(str(tmp_path))

    def test_unknown_expected_behavior_raises_value_error(
        self, runner: AgentTestRunner, tmp_path: Path
    ) -> None:
        bad_file = tmp_path / "bad_behavior.yaml"
        bad_file.write_text(
            textwrap.dedent("""\
                name: bad_case
                expected_behavior:
                  invalid_key: something
            """),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="Failed to parse test case"):
            runner.load_tests(str(tmp_path))


# ===========================================================================
# AgentTestRunner — run_test
# ===========================================================================


class TestAgentTestRunnerRunTest:
    """Tests for AgentTestRunner.run_test()."""

    def test_passing_contains_text(
        self,
        runner: AgentTestRunner,
        mock_provider: MockLLMProvider,
    ) -> None:
        test_case = AgentTestCase(
            name="greet",
            input_messages=[{"role": "user", "content": "Hi"}],
            expected_behavior={"contains_text": "hello"},
        )
        result = runner.run_test(test_case, mock_provider)
        assert result.passed is True
        assert any("contains_text" in a for a in result.assertions_passed)

    def test_failing_contains_text(
        self,
        runner: AgentTestRunner,
        mock_provider: MockLLMProvider,
    ) -> None:
        test_case = AgentTestCase(
            name="greet",
            input_messages=[],
            expected_behavior={"contains_text": "ZZZNOTREALWORD"},
        )
        result = runner.run_test(test_case, mock_provider)
        assert result.passed is False
        assert any("contains_text" in f for f in result.assertions_failed)

    def test_no_pii_passes_with_clean_content(
        self,
        runner: AgentTestRunner,
        mock_provider: MockLLMProvider,
    ) -> None:
        test_case = AgentTestCase(
            name="pii_check",
            input_messages=[],
            expected_behavior={"no_pii": True},
        )
        result = runner.run_test(test_case, mock_provider)
        assert result.passed is True

    def test_no_pii_fails_with_email_in_response(
        self,
        runner: AgentTestRunner,
    ) -> None:
        pii_response = MockLLMResponse(
            content="Contact admin@example.com for help.",
            model="mock-model",
            tokens_used=10,
            latency_ms=10.0,
        )
        config = MockLLMConfig(model_name="m", responses=[pii_response])
        provider = MockLLMProvider(config)
        test_case = AgentTestCase(
            name="pii_fail",
            input_messages=[],
            expected_behavior={"no_pii": True},
        )
        result = runner.run_test(test_case, provider)
        assert result.passed is False
        assert any("no_pii" in f for f in result.assertions_failed)

    def test_max_tokens_passes(
        self,
        runner: AgentTestRunner,
        mock_provider: MockLLMProvider,
    ) -> None:
        # mock_provider returns tokens_used=10; limit is 50
        test_case = AgentTestCase(
            name="token_check",
            input_messages=[],
            expected_behavior={"max_tokens": 50},
        )
        result = runner.run_test(test_case, mock_provider)
        assert result.passed is True

    def test_max_tokens_fails(
        self,
        runner: AgentTestRunner,
        mock_provider: MockLLMProvider,
    ) -> None:
        # mock_provider returns tokens_used=10; limit is 5
        test_case = AgentTestCase(
            name="token_exceed",
            input_messages=[],
            expected_behavior={"max_tokens": 5},
        )
        result = runner.run_test(test_case, mock_provider)
        assert result.passed is False

    def test_max_latency_passes_using_response_latency(
        self,
        runner: AgentTestRunner,
        mock_provider: MockLLMProvider,
    ) -> None:
        # mock response has latency_ms=50.0; limit is 1000 ms
        test_case = AgentTestCase(
            name="latency_ok",
            input_messages=[],
            expected_behavior={"max_latency_ms": 1000.0},
        )
        result = runner.run_test(test_case, mock_provider)
        assert result.passed is True

    def test_max_latency_fails(
        self,
        runner: AgentTestRunner,
        mock_provider: MockLLMProvider,
    ) -> None:
        # mock response has latency_ms=50.0; limit is 1 ms
        test_case = AgentTestCase(
            name="latency_fail",
            input_messages=[],
            expected_behavior={"max_latency_ms": 1.0},
        )
        result = runner.run_test(test_case, mock_provider)
        assert result.passed is False

    def test_valid_json_passes(
        self,
        runner: AgentTestRunner,
        json_response: MockLLMResponse,
    ) -> None:
        config = MockLLMConfig(model_name="m", responses=[json_response])
        provider = MockLLMProvider(config)
        test_case = AgentTestCase(
            name="json_check",
            input_messages=[],
            expected_behavior={"valid_json": True},
        )
        result = runner.run_test(test_case, provider)
        assert result.passed is True

    def test_valid_json_fails_with_plain_text(
        self,
        runner: AgentTestRunner,
        mock_provider: MockLLMProvider,
    ) -> None:
        test_case = AgentTestCase(
            name="json_fail",
            input_messages=[],
            expected_behavior={"valid_json": True},
        )
        result = runner.run_test(test_case, mock_provider)
        assert result.passed is False

    def test_matches_schema_passes(
        self,
        runner: AgentTestRunner,
        json_response: MockLLMResponse,
    ) -> None:
        config = MockLLMConfig(model_name="m", responses=[json_response])
        provider = MockLLMProvider(config)
        test_case = AgentTestCase(
            name="schema_ok",
            input_messages=[],
            expected_behavior={
                "matches_schema": {
                    "type": "object",
                    "required": ["status"],
                    "properties": {"status": {"type": "string"}},
                }
            },
        )
        result = runner.run_test(test_case, provider)
        assert result.passed is True

    def test_matches_schema_fails_missing_required(
        self,
        runner: AgentTestRunner,
    ) -> None:
        response = MockLLMResponse(
            content='{"other_field": "value"}',
            model="m",
            tokens_used=5,
            latency_ms=10.0,
        )
        config = MockLLMConfig(model_name="m", responses=[response])
        provider = MockLLMProvider(config)
        test_case = AgentTestCase(
            name="schema_fail",
            input_messages=[],
            expected_behavior={
                "matches_schema": {
                    "type": "object",
                    "required": ["status"],
                }
            },
        )
        result = runner.run_test(test_case, provider)
        assert result.passed is False

    def test_calls_tools_passes(
        self,
        runner: AgentTestRunner,
        tool_call_response: MockLLMResponse,
    ) -> None:
        config = MockLLMConfig(model_name="m", responses=[tool_call_response])
        provider = MockLLMProvider(config)
        test_case = AgentTestCase(
            name="tool_call_ok",
            input_messages=[],
            expected_behavior={"calls_tools": "search_web"},
        )
        result = runner.run_test(test_case, provider)
        assert result.passed is True

    def test_calls_tools_fails(
        self,
        runner: AgentTestRunner,
        mock_provider: MockLLMProvider,
    ) -> None:
        test_case = AgentTestCase(
            name="tool_call_fail",
            input_messages=[],
            expected_behavior={"calls_tools": "nonexistent_tool"},
        )
        result = runner.run_test(test_case, mock_provider)
        assert result.passed is False

    def test_calls_tools_list_value(
        self,
        runner: AgentTestRunner,
        tool_call_response: MockLLMResponse,
    ) -> None:
        """calls_tools can be a list of tool names."""
        config = MockLLMConfig(model_name="m", responses=[tool_call_response])
        provider = MockLLMProvider(config)
        test_case = AgentTestCase(
            name="multi_tool",
            input_messages=[],
            expected_behavior={"calls_tools": ["search_web", "nonexistent_tool"]},
        )
        result = runner.run_test(test_case, provider)
        # search_web passes, nonexistent_tool fails
        assert result.passed is False
        assert any("search_web" in a for a in result.assertions_passed)
        assert any("nonexistent_tool" in f for f in result.assertions_failed)

    def test_provider_error_captured_in_result(
        self,
        runner: AgentTestRunner,
        failing_provider: MockLLMProvider,
    ) -> None:
        test_case = AgentTestCase(
            name="error_case",
            input_messages=[],
            expected_behavior={},
        )
        # Force deterministic failure
        import unittest.mock as mock

        with mock.patch.object(
            failing_provider, "complete", side_effect=RuntimeError("boom")
        ):
            result = runner.run_test(test_case, failing_provider)

        assert result.passed is False
        assert any("provider_error" in f for f in result.assertions_failed)
        assert "ERROR:" in result.actual_output

    def test_result_has_correct_test_case_name(
        self,
        runner: AgentTestRunner,
        mock_provider: MockLLMProvider,
    ) -> None:
        test_case = AgentTestCase(
            name="my_unique_name",
            input_messages=[],
            expected_behavior={},
        )
        result = runner.run_test(test_case, mock_provider)
        assert result.test_case_name == "my_unique_name"

    def test_result_duration_ms_is_positive(
        self,
        runner: AgentTestRunner,
        mock_provider: MockLLMProvider,
    ) -> None:
        test_case = AgentTestCase(
            name="timing", input_messages=[], expected_behavior={}
        )
        result = runner.run_test(test_case, mock_provider)
        assert result.duration_ms >= 0.0

    def test_empty_expected_behavior_always_passes(
        self,
        runner: AgentTestRunner,
        mock_provider: MockLLMProvider,
    ) -> None:
        test_case = AgentTestCase(
            name="empty_behavior", input_messages=[], expected_behavior={}
        )
        result = runner.run_test(test_case, mock_provider)
        assert result.passed is True
        assert result.assertions_failed == []

    def test_no_pii_false_is_skipped(
        self,
        runner: AgentTestRunner,
        mock_provider: MockLLMProvider,
    ) -> None:
        """no_pii: false should not trigger the assertion."""
        test_case = AgentTestCase(
            name="no_pii_skip",
            input_messages=[],
            expected_behavior={"no_pii": False},
        )
        result = runner.run_test(test_case, mock_provider)
        # no_pii: false is not evaluated (only True triggers the check)
        assert result.passed is True

    def test_valid_json_false_is_skipped(
        self,
        runner: AgentTestRunner,
        mock_provider: MockLLMProvider,
    ) -> None:
        """valid_json: false should not trigger the assertion."""
        test_case = AgentTestCase(
            name="valid_json_skip",
            input_messages=[],
            expected_behavior={"valid_json": False},
        )
        result = runner.run_test(test_case, mock_provider)
        assert result.passed is True


# ===========================================================================
# AgentTestRunner — run_suite
# ===========================================================================


class TestAgentTestRunnerRunSuite:
    """Tests for AgentTestRunner.run_suite()."""

    def test_run_suite_returns_suite_result(
        self,
        runner: AgentTestRunner,
        yaml_test_dir: Path,
    ) -> None:
        config = AgentTestConfig(test_dir=str(yaml_test_dir))
        result = runner.run_suite(str(yaml_test_dir), config)
        assert isinstance(result, TestSuiteResult)

    def test_suite_name_is_directory_name(
        self,
        runner: AgentTestRunner,
        yaml_test_dir: Path,
    ) -> None:
        config = AgentTestConfig(test_dir=str(yaml_test_dir))
        result = runner.run_suite(str(yaml_test_dir), config)
        assert result.suite_name == yaml_test_dir.name

    def test_total_matches_test_count(
        self,
        runner: AgentTestRunner,
        yaml_test_dir: Path,
    ) -> None:
        config = AgentTestConfig(test_dir=str(yaml_test_dir))
        result = runner.run_suite(str(yaml_test_dir), config)
        assert result.total == 3  # 1 + 2 from fixtures

    def test_passed_plus_failed_equals_total(
        self,
        runner: AgentTestRunner,
        yaml_test_dir: Path,
    ) -> None:
        config = AgentTestConfig(test_dir=str(yaml_test_dir))
        result = runner.run_suite(str(yaml_test_dir), config)
        assert result.passed + result.failed == result.total

    def test_duration_ms_is_positive(
        self,
        runner: AgentTestRunner,
        yaml_test_dir: Path,
    ) -> None:
        config = AgentTestConfig(test_dir=str(yaml_test_dir))
        result = runner.run_suite(str(yaml_test_dir), config)
        assert result.duration_ms >= 0.0

    def test_nonexistent_dir_raises(
        self,
        runner: AgentTestRunner,
    ) -> None:
        config = AgentTestConfig(test_dir="/nonexistent/dir")
        with pytest.raises(FileNotFoundError):
            runner.run_suite("/nonexistent/dir", config)

    def test_suite_with_explicit_mock_config(
        self,
        runner: AgentTestRunner,
        yaml_test_dir: Path,
        single_response_config: MockLLMConfig,
    ) -> None:
        config = AgentTestConfig(
            test_dir=str(yaml_test_dir),
            mock_config=single_response_config,
        )
        result = runner.run_suite(str(yaml_test_dir), config)
        assert result.total > 0

    def test_suite_uses_default_mock_when_none(
        self,
        runner: AgentTestRunner,
        yaml_test_dir: Path,
    ) -> None:
        config = AgentTestConfig(test_dir=str(yaml_test_dir), mock_config=None)
        result = runner.run_suite(str(yaml_test_dir), config)
        # Should not raise; default mock provides a response
        assert result.total == 3

    def test_parallel_execution_same_total(
        self,
        runner: AgentTestRunner,
        yaml_test_dir: Path,
    ) -> None:
        config = AgentTestConfig(test_dir=str(yaml_test_dir), parallel=True)
        result = runner.run_suite(str(yaml_test_dir), config)
        assert result.total == 3

    def test_parallel_passed_plus_failed_equals_total(
        self,
        runner: AgentTestRunner,
        yaml_test_dir: Path,
    ) -> None:
        config = AgentTestConfig(test_dir=str(yaml_test_dir), parallel=True)
        result = runner.run_suite(str(yaml_test_dir), config)
        assert result.passed + result.failed == result.total

    def test_empty_dir_suite_has_zero_total(
        self,
        runner: AgentTestRunner,
        tmp_path: Path,
    ) -> None:
        config = AgentTestConfig(test_dir=str(tmp_path))
        result = runner.run_suite(str(tmp_path), config)
        assert result.total == 0
        assert result.passed == 0
        assert result.failed == 0


# ===========================================================================
# _default_mock_config
# ===========================================================================


class TestDefaultMockConfig:
    """Tests for the _default_mock_config() helper."""

    def test_returns_mock_llm_config(self) -> None:
        config = _default_mock_config()
        assert isinstance(config, MockLLMConfig)

    def test_has_at_least_one_response(self) -> None:
        config = _default_mock_config()
        assert len(config.responses) >= 1

    def test_response_content_is_nonempty(self) -> None:
        config = _default_mock_config()
        assert all(r.content for r in config.responses)

    def test_model_name_is_mock_model(self) -> None:
        config = _default_mock_config()
        assert config.model_name == "mock-model"
