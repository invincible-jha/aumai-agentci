"""Core test runner and mock LLM provider for aumai-agentci."""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any

import yaml

from aumai_agentci.assertions import (
    assert_calls_tool,
    assert_contains_text,
    assert_max_latency,
    assert_max_tokens,
    assert_matches_schema,
    assert_no_pii,
    assert_valid_json,
)
from aumai_agentci.models import (
    AgentTestCase,
    AgentTestConfig,
    AgentTestResult,
    MockLLMConfig,
    MockLLMResponse,
    TestSuiteResult,
)

__all__ = ["MockLLMProvider", "AgentTestRunner"]


class MockLLMProvider:
    """Simulates LLM API responses from a :class:`MockLLMConfig`.

    Responses are served in round-robin order.  When *failure_rate* > 0
    a :class:`RuntimeError` is raised on a random subset of calls to
    simulate transient API errors.

    Args:
        config: Provider configuration including the response list.
    """

    def __init__(self, config: MockLLMConfig) -> None:
        self._config = config
        self._index: int = 0
        self._call_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def complete(
        self,
        messages: list[dict[str, Any]],  # noqa: ARG002  (unused but mirrors real API)
        *,
        seed: int | None = None,
    ) -> MockLLMResponse:
        """Return the next mock response, honouring the failure rate.

        Args:
            messages: The input message list (mirrors real LLM API shape).
            seed: Optional RNG seed for deterministic failure injection.

        Raises:
            RuntimeError: When a simulated API failure is triggered.

        Returns:
            The next :class:`MockLLMResponse` in the round-robin sequence.
        """
        self._call_count += 1
        rng = random.Random(seed) if seed is not None else random.Random()  # noqa: S311

        if self._config.failure_rate > 0.0 and rng.random() < self._config.failure_rate:
            raise RuntimeError(
                f"Simulated LLM API failure (failure_rate="
                f"{self._config.failure_rate:.2f}, call #{self._call_count})"
            )

        if not self._config.responses:
            # Synthesise a default response so an empty config still works
            return MockLLMResponse(
                content="",
                model=self._config.model_name,
                tokens_used=0,
                latency_ms=self._config.default_latency_ms,
                finish_reason="stop",
            )

        response = self._config.responses[self._index % len(self._config.responses)]
        self._index += 1
        return response

    @property
    def call_count(self) -> int:
        """Number of times :meth:`complete` has been called."""
        return self._call_count

    def reset(self) -> None:
        """Reset the round-robin index and call counter."""
        self._index = 0
        self._call_count = 0


class AgentTestRunner:
    """Load and execute :class:`AgentTestCase` objects against a mock provider.

    Typical usage::

        runner = AgentTestRunner()
        suite = runner.run_suite("tests/", config)
    """

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_tests(self, test_dir: str) -> list[AgentTestCase]:
        """Discover and parse all YAML test case files in *test_dir*.

        Files must have a ``.yaml`` or ``.yml`` extension.  Each file may
        contain either a single mapping (one test case) or a list of
        mappings (multiple test cases).

        Args:
            test_dir: Path to the directory containing test YAML files.

        Raises:
            FileNotFoundError: If *test_dir* does not exist.
            ValueError: If a YAML file cannot be parsed into test cases.

        Returns:
            Flat list of parsed :class:`AgentTestCase` objects.
        """
        directory = Path(test_dir)
        if not directory.exists():
            raise FileNotFoundError(f"Test directory not found: {test_dir}")

        test_cases: list[AgentTestCase] = []
        yaml_files = sorted(
            list(directory.glob("**/*.yaml")) + list(directory.glob("**/*.yml"))
        )

        for yaml_path in yaml_files:
            test_cases.extend(self._load_yaml_file(yaml_path))

        return test_cases

    def _load_yaml_file(self, path: Path) -> list[AgentTestCase]:
        """Parse a single YAML file into one or more test cases."""
        with path.open(encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)

        if raw is None:
            return []

        records: list[dict[str, Any]] = []
        if isinstance(raw, list):
            records = [r for r in raw if isinstance(r, dict)]
        elif isinstance(raw, dict):
            # Support a top-level `tests:` key for grouping
            if "tests" in raw and isinstance(raw["tests"], list):
                records = [r for r in raw["tests"] if isinstance(r, dict)]
            else:
                records = [raw]
        else:
            raise ValueError(f"Unexpected YAML structure in {path}: {type(raw)}")

        cases: list[AgentTestCase] = []
        for record in records:
            try:
                cases.append(AgentTestCase.model_validate(record))
            except Exception as exc:
                raise ValueError(
                    f"Failed to parse test case in {path}: {exc}"
                ) from exc
        return cases

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run_test(
        self,
        test_case: AgentTestCase,
        mock: MockLLMProvider,
    ) -> AgentTestResult:
        """Execute *test_case* against *mock* and evaluate assertions.

        The mock provider is called once with the test's ``input_messages``.
        All ``expected_behavior`` keys are then checked and recorded as
        passed or failed assertions.

        Args:
            test_case: The test case to execute.
            mock: A configured mock LLM provider.

        Returns:
            :class:`AgentTestResult` capturing pass/fail state.
        """
        start = time.monotonic()
        actual_output = ""
        tokens_used = 0
        latency_ms = 0.0
        error_msg: str | None = None

        try:
            response = mock.complete(test_case.input_messages)
            actual_output = response.content
            tokens_used = response.tokens_used
            latency_ms = response.latency_ms
        except RuntimeError as exc:
            error_msg = str(exc)
            actual_output = f"ERROR: {error_msg}"

        elapsed_ms = (time.monotonic() - start) * 1000.0

        assertions_passed: list[str] = []
        assertions_failed: list[str] = []

        if error_msg is not None:
            assertions_failed.append(f"provider_error: {error_msg}")
        else:
            self._evaluate_assertions(
                test_case=test_case,
                actual_output=actual_output,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                elapsed_ms=elapsed_ms,
                assertions_passed=assertions_passed,
                assertions_failed=assertions_failed,
            )

        return AgentTestResult(
            test_case_name=test_case.name,
            passed=len(assertions_failed) == 0,
            actual_output=actual_output,
            assertions_passed=assertions_passed,
            assertions_failed=assertions_failed,
            duration_ms=elapsed_ms,
            tokens_used=tokens_used,
        )

    def _evaluate_assertions(
        self,
        *,
        test_case: AgentTestCase,
        actual_output: str,
        tokens_used: int,
        latency_ms: float,
        elapsed_ms: float,
        assertions_passed: list[str],
        assertions_failed: list[str],
    ) -> None:
        """Evaluate all expected_behavior keys and populate passed/failed lists."""
        behavior = test_case.expected_behavior

        if "contains_text" in behavior:
            expected_text = str(behavior["contains_text"])
            if assert_contains_text(actual_output, expected_text):
                assertions_passed.append(f"contains_text: '{expected_text}'")
            else:
                assertions_failed.append(
                    f"contains_text: '{expected_text}' not found in output"
                )

        if "calls_tools" in behavior:
            tools_value = behavior["calls_tools"]
            if isinstance(tools_value, str):
                tool_names: list[str] = [tools_value]
            elif isinstance(tools_value, (list, tuple)):
                tool_names = [str(t) for t in tools_value]
            else:
                tool_names = [str(tools_value)]
            for tool_name in tool_names:
                if assert_calls_tool(actual_output, tool_name):
                    assertions_passed.append(f"calls_tool: '{tool_name}'")
                else:
                    assertions_failed.append(
                        f"calls_tool: '{tool_name}' not found in output"
                    )

        if "max_tokens" in behavior:
            raw_max_tok: Any = behavior["max_tokens"]
            max_tok = int(raw_max_tok)
            if assert_max_tokens(tokens_used, max_tok):
                assertions_passed.append(
                    f"max_tokens: {tokens_used} <= {max_tok}"
                )
            else:
                assertions_failed.append(
                    f"max_tokens: {tokens_used} exceeds limit {max_tok}"
                )

        if "max_latency_ms" in behavior:
            raw_max_lat: Any = behavior["max_latency_ms"]
            max_lat = float(raw_max_lat)
            measured = latency_ms if latency_ms > 0 else elapsed_ms
            if assert_max_latency(measured, max_lat):
                assertions_passed.append(
                    f"max_latency_ms: {measured:.1f} <= {max_lat}"
                )
            else:
                assertions_failed.append(
                    f"max_latency_ms: {measured:.1f} exceeds limit {max_lat}"
                )

        if behavior.get("no_pii") is True:
            if assert_no_pii(actual_output):
                assertions_passed.append("no_pii: no PII detected")
            else:
                assertions_failed.append("no_pii: PII detected in output")

        if behavior.get("valid_json") is True:
            if assert_valid_json(actual_output):
                assertions_passed.append("valid_json: output is valid JSON")
            else:
                assertions_failed.append("valid_json: output is not valid JSON")

        if "matches_schema" in behavior:
            schema = behavior["matches_schema"]
            if isinstance(schema, dict):
                if assert_matches_schema(actual_output, schema):
                    assertions_passed.append("matches_schema: output conforms to schema")
                else:
                    assertions_failed.append(
                        "matches_schema: output does not conform to schema"
                    )

    # ------------------------------------------------------------------
    # Suite runner
    # ------------------------------------------------------------------

    def run_suite(
        self,
        test_dir: str,
        config: AgentTestConfig,
    ) -> TestSuiteResult:
        """Discover and run all test cases in *test_dir*.

        When ``config.parallel`` is True the tests are executed using a
        :class:`concurrent.futures.ThreadPoolExecutor`.  The mock provider
        is *not* thread-safe by default; each thread receives its own
        provider instance sharing the same config.

        Args:
            test_dir: Path to the directory containing YAML test cases.
            config: Run configuration (mock, timeout, parallelism).

        Returns:
            Aggregated :class:`TestSuiteResult`.
        """
        suite_start = time.monotonic()
        test_cases = self.load_tests(test_dir)

        mock_config = config.mock_config or _default_mock_config()

        results: list[AgentTestResult]
        if config.parallel:
            results = self._run_parallel(test_cases, mock_config, config.timeout_seconds)
        else:
            provider = MockLLMProvider(mock_config)
            results = [
                self.run_test(tc, provider) for tc in test_cases
            ]

        total_duration = (time.monotonic() - suite_start) * 1000.0
        passed = sum(1 for r in results if r.passed)

        return TestSuiteResult(
            suite_name=Path(test_dir).name,
            results=results,
            total=len(results),
            passed=passed,
            failed=len(results) - passed,
            duration_ms=total_duration,
        )

    def _run_parallel(
        self,
        test_cases: list[AgentTestCase],
        mock_config: MockLLMConfig,
        timeout_seconds: float,
    ) -> list[AgentTestResult]:
        """Run test cases in parallel threads, each with its own provider."""
        import concurrent.futures

        def _run_one(tc: AgentTestCase) -> AgentTestResult:
            provider = MockLLMProvider(mock_config)
            return self.run_test(tc, provider)

        results: list[AgentTestResult] = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(_run_one, tc): tc for tc in test_cases}
            for future in concurrent.futures.as_completed(
                futures, timeout=timeout_seconds
            ):
                results.append(future.result())
        return results


def _default_mock_config() -> MockLLMConfig:
    """Return a minimal mock config used when none is supplied."""
    return MockLLMConfig(
        model_name="mock-model",
        responses=[
            MockLLMResponse(
                content="Mock response: task completed successfully.",
                model="mock-model",
                tokens_used=12,
                latency_ms=10.0,
                finish_reason="stop",
            )
        ],
    )
