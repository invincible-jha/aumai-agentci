"""Quickstart examples for aumai-agentci.

This script demonstrates the core features of aumai-agentci:
loading test cases, running them with a mock provider, using
fixture factories, running a full suite, and using assertion
helpers directly.

Run directly to verify your installation:

    python examples/quickstart.py

No API keys or external services are required.
"""

from __future__ import annotations

from aumai_agentci import (
    AgentTestCase,
    AgentTestConfig,
    AgentTestRunner,
    MockLLMConfig,
    MockLLMProvider,
    MockLLMResponse,
    assert_calls_tool,
    assert_contains_text,
    assert_matches_schema,
    assert_no_pii,
    assert_valid_json,
    create_mock_anthropic,
    create_mock_openai,
)


# ---------------------------------------------------------------------------
# Demo 1: Run a single test case programmatically
# ---------------------------------------------------------------------------


def demo_single_test() -> None:
    """Run one test case against a mock provider and inspect the result.

    This is the most basic usage: create a mock, create a test case,
    run it, and check the result.
    """
    print("\n" + "=" * 60)
    print("Demo 1: Single test case")
    print("=" * 60)

    # Configure the mock provider with two pre-scripted responses
    mock_config = MockLLMConfig(
        model_name="mock-gpt",
        responses=[
            MockLLMResponse(
                content="Hello! I can definitely help you with that.",
                model="mock-gpt",
                tokens_used=12,
                latency_ms=45.0,
                finish_reason="stop",
            ),
        ],
    )
    provider = MockLLMProvider(mock_config)

    # Define the test case
    test_case = AgentTestCase(
        name="greeting_check",
        description="Agent should respond with a greeting.",
        input_messages=[{"role": "user", "content": "Hello, agent!"}],
        expected_behavior={
            "contains_text": "hello",    # case-insensitive substring check
            "max_tokens": 50,            # response must use <= 50 tokens
        },
        tags=["smoke"],
    )

    # Run the test
    runner = AgentTestRunner()
    result = runner.run_test(test_case, provider)

    print(f"\nTest name   : {result.test_case_name}")
    print(f"Passed      : {result.passed}")
    print(f"Assertions passed: {result.assertions_passed}")
    print(f"Assertions failed: {result.assertions_failed}")
    print(f"Duration    : {result.duration_ms:.2f} ms")
    print(f"Tokens used : {result.tokens_used}")
    print(f"Provider calls: {provider.call_count}")


# ---------------------------------------------------------------------------
# Demo 2: Use fixture factories (OpenAI and Anthropic)
# ---------------------------------------------------------------------------


def demo_fixture_factories() -> None:
    """Use pre-configured provider factories for common LLM providers.

    create_mock_openai and create_mock_anthropic return providers with
    realistic default responses that cover plain text, JSON, and tool calls.
    """
    print("\n" + "=" * 60)
    print("Demo 2: Fixture factories")
    print("=" * 60)

    # OpenAI-style mock
    openai_provider = create_mock_openai()
    resp1 = openai_provider.complete([{"role": "user", "content": "Hi"}])
    print(f"\n[OpenAI mock] model      : {resp1.model}")
    print(f"[OpenAI mock] response   : {resp1.content[:60]}...")
    print(f"[OpenAI mock] tokens     : {resp1.tokens_used}")
    print(f"[OpenAI mock] call_count : {openai_provider.call_count}")

    # Second call gets the next response in the round-robin queue
    resp2 = openai_provider.complete([{"role": "user", "content": "Return JSON"}])
    print(f"\n[OpenAI mock] response 2 : {resp2.content[:60]}...")

    # Anthropic-style mock
    anthropic_provider = create_mock_anthropic()
    resp3 = anthropic_provider.complete([{"role": "user", "content": "Analyze"}])
    print(f"\n[Anthropic mock] model   : {resp3.model}")
    print(f"[Anthropic mock] response: {resp3.content[:60]}...")


# ---------------------------------------------------------------------------
# Demo 3: Multiple test cases with different assertion types
# ---------------------------------------------------------------------------


def demo_multiple_assertions() -> None:
    """Run test cases exercising all built-in assertion types.

    Shows: contains_text, max_tokens, valid_json, matches_schema,
    calls_tools, and no_pii checks.
    """
    print("\n" + "=" * 60)
    print("Demo 3: Multiple assertion types")
    print("=" * 60)

    # Responses matched to each test case in order
    mock_config = MockLLMConfig(
        model_name="mock-agent",
        responses=[
            # Response for contains_text test
            MockLLMResponse(
                content="The status is OPERATIONAL and all systems are green.",
                model="mock-agent",
                tokens_used=15,
                latency_ms=30.0,
                finish_reason="stop",
            ),
            # Response for JSON schema test
            MockLLMResponse(
                content='{"status": "ok", "code": 200, "message": "processed"}',
                model="mock-agent",
                tokens_used=22,
                latency_ms=25.0,
                finish_reason="stop",
            ),
            # Response for tool call test (OpenAI format)
            MockLLMResponse(
                content=(
                    '{"tool_calls": [{"id": "call_001", "type": "function",'
                    ' "function": {"name": "search_web",'
                    ' "arguments": "{\\"query\\": \\"AI news\\"}"}}]}'
                ),
                model="mock-agent",
                tokens_used=35,
                latency_ms=60.0,
                finish_reason="tool_calls",
            ),
            # Response for PII-free test
            MockLLMResponse(
                content="The weather today is sunny with a high of 72 degrees.",
                model="mock-agent",
                tokens_used=16,
                latency_ms=20.0,
                finish_reason="stop",
            ),
        ],
    )

    test_cases = [
        AgentTestCase(
            name="operational_status",
            input_messages=[{"role": "user", "content": "What is the system status?"}],
            expected_behavior={"contains_text": "operational"},
        ),
        AgentTestCase(
            name="json_schema_conformance",
            input_messages=[{"role": "user", "content": "Return a status object."}],
            expected_behavior={
                "valid_json": True,
                "matches_schema": {
                    "type": "object",
                    "required": ["status", "code"],
                    "properties": {
                        "status": {"type": "string"},
                        "code": {"type": "integer"},
                    },
                },
            },
        ),
        AgentTestCase(
            name="tool_call_verification",
            input_messages=[{"role": "user", "content": "Search for AI news."}],
            expected_behavior={"calls_tools": "search_web"},
        ),
        AgentTestCase(
            name="no_pii_in_weather",
            input_messages=[{"role": "user", "content": "What is the weather today?"}],
            expected_behavior={"no_pii": True},
        ),
    ]

    provider = MockLLMProvider(mock_config)
    runner = AgentTestRunner()

    print()
    all_passed = True
    for test_case in test_cases:
        result = runner.run_test(test_case, provider)
        status = "PASS" if result.passed else "FAIL"
        print(f"  [{status}] {result.test_case_name}")
        if not result.passed:
            all_passed = False
            for failure in result.assertions_failed:
                print(f"         x {failure}")

    print(f"\nAll passed: {all_passed}")


# ---------------------------------------------------------------------------
# Demo 4: Using assertion helpers directly (outside the runner)
# ---------------------------------------------------------------------------


def demo_assertion_helpers() -> None:
    """Use assertion functions directly for custom testing logic.

    The assertion helpers are pure functions that return bool — they can
    be used anywhere, not just inside the AgentTestRunner.
    """
    print("\n" + "=" * 60)
    print("Demo 4: Assertion helpers used directly")
    print("=" * 60)

    # Simulate agent outputs to check
    tool_call_output = (
        '{"tool_calls": [{"function": {"name": "calculator",'
        ' "arguments": "{\\"expression\\": \\"2+2\\"}"}}]}'
    )
    clean_output = "The answer is 42. This is a safe, factual response."
    pii_output = "The user email is alice@corp.com and phone is 555-867-5309."
    json_output = '{"result": "computed", "value": 42}'

    print(f"\nassert_contains_text (looking for '42'): "
          f"{assert_contains_text(clean_output, '42')}")

    print(f"assert_calls_tool (looking for 'calculator'): "
          f"{assert_calls_tool(tool_call_output, 'calculator')}")

    print(f"assert_no_pii (clean text):  "
          f"{assert_no_pii(clean_output)}")

    print(f"assert_no_pii (with email):  "
          f"{assert_no_pii(pii_output)}")

    print(f"assert_valid_json (json_output):  "
          f"{assert_valid_json(json_output)}")

    print(f"assert_valid_json (plain_text):   "
          f"{assert_valid_json(clean_output)}")

    schema = {
        "type": "object",
        "required": ["result", "value"],
        "properties": {
            "result": {"type": "string"},
            "value": {"type": "integer"},
        },
    }
    print(f"assert_matches_schema (correct):  "
          f"{assert_matches_schema(json_output, schema)}")

    wrong_output = '{"result": "ok"}'   # missing "value"
    print(f"assert_matches_schema (missing field): "
          f"{assert_matches_schema(wrong_output, schema)}")


# ---------------------------------------------------------------------------
# Demo 5: Run a full suite with config and report results
# ---------------------------------------------------------------------------


def demo_run_suite() -> None:
    """Build test cases in memory and run them as a full suite.

    Demonstrates AgentTestConfig and TestSuiteResult without needing
    YAML files on disk.
    """
    print("\n" + "=" * 60)
    print("Demo 5: Full suite run (in-memory test cases)")
    print("=" * 60)

    from pathlib import Path
    import tempfile
    import yaml  # pyyaml is a dependency of aumai-agentci

    # Write test cases to a temporary YAML file
    test_cases_yaml = {
        "tests": [
            {
                "name": "confirms_hello",
                "input_messages": [{"role": "user", "content": "Say hello"}],
                "expected_behavior": {"contains_text": "hello"},
            },
            {
                "name": "stays_within_budget",
                "input_messages": [{"role": "user", "content": "One sentence answer"}],
                "expected_behavior": {"max_tokens": 100},
            },
            {
                "name": "returns_valid_json",
                "input_messages": [{"role": "user", "content": "Return JSON status"}],
                "expected_behavior": {"valid_json": True},
            },
        ]
    }

    mock_config = MockLLMConfig(
        model_name="mock",
        responses=[
            MockLLMResponse(
                content="Hello! Nice to meet you.",
                model="mock", tokens_used=8, latency_ms=10.0, finish_reason="stop",
            ),
            MockLLMResponse(
                content="The sky is blue.",
                model="mock", tokens_used=5, latency_ms=8.0, finish_reason="stop",
            ),
            MockLLMResponse(
                content='{"status": "ready"}',
                model="mock", tokens_used=10, latency_ms=12.0, finish_reason="stop",
            ),
        ],
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "tests.yaml"
        test_file.write_text(yaml.dump(test_cases_yaml), encoding="utf-8")

        config = AgentTestConfig(
            test_dir=tmpdir,
            mock_config=mock_config,
            timeout_seconds=30.0,
            parallel=False,
        )

        runner = AgentTestRunner()
        suite = runner.run_suite(tmpdir, config)

    print(f"\nSuite name  : {suite.suite_name}")
    print(f"Total tests : {suite.total}")
    print(f"Passed      : {suite.passed}")
    print(f"Failed      : {suite.failed}")
    print(f"Duration    : {suite.duration_ms:.2f} ms")
    print()
    for result in suite.results:
        status = "PASS" if result.passed else "FAIL"
        print(f"  [{status}] {result.test_case_name}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run all quickstart demos in sequence."""
    print("aumai-agentci quickstart examples")
    print("=" * 60)

    demo_single_test()
    demo_fixture_factories()
    demo_multiple_assertions()
    demo_assertion_helpers()
    demo_run_suite()

    print("\n" + "=" * 60)
    print("All demos complete.")
    print("See docs/getting-started.md for step-by-step tutorials.")
    print("See docs/api-reference.md for full API documentation.")


if __name__ == "__main__":
    main()
