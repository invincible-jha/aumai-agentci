"""Tests for aumai_agentci.fixtures — pre-configured provider factories."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from aumai_agentci.core import MockLLMProvider
from aumai_agentci.fixtures import (
    create_mock_anthropic,
    create_mock_openai,
    load_test_suite,
)
from aumai_agentci.models import MockLLMResponse

# ===========================================================================
# create_mock_openai
# ===========================================================================


class TestCreateMockOpenai:
    """Tests for create_mock_openai()."""

    def test_returns_mock_llm_provider(self) -> None:
        provider = create_mock_openai()
        assert isinstance(provider, MockLLMProvider)

    def test_default_model_name(self) -> None:
        provider = create_mock_openai()
        response = provider.complete([])
        assert response.model == "gpt-4o"

    def test_default_has_three_responses(self) -> None:
        provider = create_mock_openai()
        contents = [provider.complete([]).content for _ in range(3)]
        assert len(set(contents)) == 3  # three distinct default responses

    def test_first_response_is_plain_text(self) -> None:
        provider = create_mock_openai()
        response = provider.complete([])
        assert "understand" in response.content.lower() or len(response.content) > 0

    def test_second_response_is_json(self) -> None:
        import json

        provider = create_mock_openai()
        provider.complete([])  # skip first
        response = provider.complete([])
        # Should parse as JSON
        parsed = json.loads(response.content)
        assert "answer" in parsed

    def test_third_response_contains_tool_call(self) -> None:
        import json

        provider = create_mock_openai()
        provider.complete([])
        provider.complete([])
        response = provider.complete([])
        parsed = json.loads(response.content)
        assert "tool_calls" in parsed

    def test_custom_responses_override_defaults(self) -> None:
        custom = MockLLMResponse(
            content="custom response",
            model="gpt-4o",
            tokens_used=5,
            latency_ms=10.0,
        )
        provider = create_mock_openai(responses=[custom])
        response = provider.complete([])
        assert response.content == "custom response"

    def test_empty_responses_list_uses_synthetic(self) -> None:
        provider = create_mock_openai(responses=[])
        response = provider.complete([])
        assert response.content == ""
        assert response.tokens_used == 0

    def test_failure_rate_zero_never_raises(self) -> None:
        provider = create_mock_openai(failure_rate=0.0)
        for _ in range(10):
            response = provider.complete([])
            assert isinstance(response, MockLLMResponse)

    def test_failure_rate_one_always_raises(self) -> None:
        provider = create_mock_openai(failure_rate=1.0)
        with pytest.raises(RuntimeError, match="Simulated LLM API failure"):
            provider.complete([], seed=0)

    def test_call_count_increments(self) -> None:
        provider = create_mock_openai()
        provider.complete([])
        provider.complete([])
        assert provider.call_count == 2

    def test_finish_reason_of_third_response(self) -> None:
        provider = create_mock_openai()
        provider.complete([])
        provider.complete([])
        response = provider.complete([])
        assert response.finish_reason == "tool_calls"

    def test_round_robin_wraps_after_three(self) -> None:
        provider = create_mock_openai()
        first = provider.complete([]).content
        provider.complete([])
        provider.complete([])
        fourth = provider.complete([]).content
        assert first == fourth


# ===========================================================================
# create_mock_anthropic
# ===========================================================================


class TestCreateMockAnthropic:
    """Tests for create_mock_anthropic()."""

    def test_returns_mock_llm_provider(self) -> None:
        provider = create_mock_anthropic()
        assert isinstance(provider, MockLLMProvider)

    def test_default_model_name(self) -> None:
        provider = create_mock_anthropic()
        response = provider.complete([])
        assert response.model == "claude-opus-4-6"

    def test_default_has_three_responses(self) -> None:
        provider = create_mock_anthropic()
        contents = [provider.complete([]).content for _ in range(3)]
        assert len(set(contents)) == 3

    def test_first_response_is_plain_text(self) -> None:
        provider = create_mock_anthropic()
        response = provider.complete([])
        assert "help" in response.content.lower()

    def test_second_response_is_json(self) -> None:
        import json

        provider = create_mock_anthropic()
        provider.complete([])
        response = provider.complete([])
        parsed = json.loads(response.content)
        assert "status" in parsed

    def test_third_response_has_tool_use(self) -> None:
        import json

        provider = create_mock_anthropic()
        provider.complete([])
        provider.complete([])
        response = provider.complete([])
        parsed = json.loads(response.content)
        assert parsed.get("type") == "tool_use"

    def test_custom_responses_override_defaults(self) -> None:
        custom = MockLLMResponse(
            content="anthropic custom",
            model="claude-opus-4-6",
            tokens_used=8,
            latency_ms=20.0,
        )
        provider = create_mock_anthropic(responses=[custom])
        assert provider.complete([]).content == "anthropic custom"

    def test_empty_responses_list_uses_synthetic(self) -> None:
        provider = create_mock_anthropic(responses=[])
        response = provider.complete([])
        assert response.content == ""

    def test_failure_rate_one_always_raises(self) -> None:
        provider = create_mock_anthropic(failure_rate=1.0)
        with pytest.raises(RuntimeError):
            provider.complete([], seed=0)

    def test_finish_reason_of_third_response(self) -> None:
        provider = create_mock_anthropic()
        provider.complete([])
        provider.complete([])
        response = provider.complete([])
        assert response.finish_reason == "tool_use"

    def test_round_robin_wraps_after_three(self) -> None:
        provider = create_mock_anthropic()
        first = provider.complete([]).content
        provider.complete([])
        provider.complete([])
        fourth = provider.complete([]).content
        assert first == fourth


# ===========================================================================
# load_test_suite
# ===========================================================================


class TestLoadTestSuite:
    """Tests for the load_test_suite() convenience function."""

    def test_loads_from_directory(self, yaml_test_dir: Path) -> None:
        cases = load_test_suite(str(yaml_test_dir))
        assert len(cases) == 3

    def test_loads_single_yaml_file(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "single.yaml"
        yaml_file.write_text(
            textwrap.dedent("""\
                name: single_file_test
                input_messages:
                  - role: user
                    content: "Hello"
                expected_behavior:
                  no_pii: true
            """),
            encoding="utf-8",
        )
        cases = load_test_suite(str(yaml_file))
        assert len(cases) == 1
        assert cases[0].name == "single_file_test"

    def test_loads_multi_case_yaml_file(self, yaml_list_file: Path) -> None:
        cases = load_test_suite(str(yaml_list_file))
        assert len(cases) == 2
        assert {tc.name for tc in cases} == {"list_case_1", "list_case_2"}

    def test_nonexistent_path_raises_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError, match="Path not found"):
            load_test_suite("/nonexistent/path.yaml")

    def test_nonexistent_directory_raises_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_test_suite("/nonexistent/directory/")

    def test_returns_agent_test_case_objects(self, yaml_test_dir: Path) -> None:
        from aumai_agentci.models import AgentTestCase

        cases = load_test_suite(str(yaml_test_dir))
        for case in cases:
            assert isinstance(case, AgentTestCase)

    def test_empty_directory_returns_empty_list(self, tmp_path: Path) -> None:
        cases = load_test_suite(str(tmp_path))
        assert cases == []

    def test_tags_are_preserved(self, yaml_test_dir: Path) -> None:
        cases = load_test_suite(str(yaml_test_dir))
        greet_case = next(tc for tc in cases if tc.name == "greet_user")
        assert "smoke" in greet_case.tags

    def test_input_messages_are_preserved(self, yaml_test_dir: Path) -> None:
        cases = load_test_suite(str(yaml_test_dir))
        greet_case = next(tc for tc in cases if tc.name == "greet_user")
        assert len(greet_case.input_messages) == 1
        assert greet_case.input_messages[0]["role"] == "user"

    def test_expected_behavior_is_preserved(self, yaml_test_dir: Path) -> None:
        cases = load_test_suite(str(yaml_test_dir))
        greet_case = next(tc for tc in cases if tc.name == "greet_user")
        assert greet_case.expected_behavior == {"contains_text": "hello"}
