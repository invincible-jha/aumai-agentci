"""CLI entry point for aumai-agentci.

Commands
--------
test      Run test cases from a directory.
init      Scaffold an example test directory.
validate  Validate test YAML files without running them.
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path
from typing import Any

import click
import yaml

from aumai_agentci.core import AgentTestRunner
from aumai_agentci.models import AgentTestConfig, MockLLMConfig, MockLLMResponse
from aumai_agentci.reporter import ConsoleReporter, JSONReporter, JUnitReporter

__all__ = ["main"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAMPLE_TESTS_YAML = """\
# Example test cases for aumai-agentci
# Place this file in your test directory.

tests:
  - name: basic_response
    description: Verify the agent produces a non-empty response.
    input_messages:
      - role: user
        content: "Say hello."
    expected_behavior:
      contains_text: "hello"
    tags:
      - smoke

  - name: no_pii_in_response
    description: The agent must not echo back any PII.
    input_messages:
      - role: user
        content: "What is the weather like today?"
    expected_behavior:
      no_pii: true
    tags:
      - security

  - name: json_output
    description: The agent must return valid JSON when asked.
    input_messages:
      - role: user
        content: "Return a JSON object with a status field set to ok."
    expected_behavior:
      valid_json: true
      matches_schema:
        type: object
        required:
          - status
        properties:
          status:
            type: string
    tags:
      - structured-output

  - name: tool_call_check
    description: Verify the agent invokes the search_web tool.
    input_messages:
      - role: user
        content: "Search the web for the latest AI news."
    expected_behavior:
      calls_tools: search_web
    tags:
      - tool-use

  - name: token_budget
    description: Response must stay within a 100-token budget.
    input_messages:
      - role: user
        content: "Summarise the Pythagorean theorem in one sentence."
    expected_behavior:
      max_tokens: 100
    tags:
      - performance
"""

_SAMPLE_MOCK_YAML = """\
# Mock LLM provider configuration for aumai-agentci
model_name: mock-gpt-4o
default_latency_ms: 50.0
failure_rate: 0.0
responses:
  - content: "Hello! I am a mock LLM response."
    model: mock-gpt-4o
    tokens_used: 12
    latency_ms: 50.0
    finish_reason: stop

  - content: '{"status": "ok", "message": "processed"}'
    model: mock-gpt-4o
    tokens_used: 18
    latency_ms: 45.0
    finish_reason: stop

  - content: >
      {"tool_calls": [{"id": "call_001", "type": "function",
      "function": {"name": "search_web",
      "arguments": "{\\"query\\": \\"latest AI news\\"}"}}]}
    model: mock-gpt-4o
    tokens_used: 30
    latency_ms: 60.0
    finish_reason: tool_calls

  - content: "The Pythagorean theorem states that a^2 + b^2 = c^2."
    model: mock-gpt-4o
    tokens_used: 20
    latency_ms: 40.0
    finish_reason: stop
"""


def _load_mock_config(mock_config_path: str | None) -> MockLLMConfig | None:
    """Parse an optional mock config YAML file."""
    if mock_config_path is None:
        return None

    config_path = Path(mock_config_path)
    if not config_path.exists():
        raise click.ClickException(f"Mock config file not found: {mock_config_path}")

    with config_path.open(encoding="utf-8") as fh:
        raw: Any = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise click.ClickException(
            f"Mock config must be a YAML mapping, got {type(raw).__name__}"
        )

    # Coerce responses list into MockLLMResponse objects
    raw_responses: list[dict[str, Any]] = raw.pop("responses", [])
    responses = [MockLLMResponse.model_validate(r) for r in raw_responses]
    return MockLLMConfig(responses=responses, **raw)


# ---------------------------------------------------------------------------
# CLI definition
# ---------------------------------------------------------------------------


@click.group()
@click.version_option()
def main() -> None:
    """AumAI AgentCI — CI/CD testing framework for AI agents."""


# ---------------------------------------------------------------------------
# test command
# ---------------------------------------------------------------------------


@main.command("test")
@click.argument("test_dir", default="tests/", metavar="TEST_DIR")
@click.option(
    "--mock-config",
    "mock_config_path",
    default=None,
    metavar="FILE",
    help="Path to a YAML mock LLM configuration file.",
)
@click.option(
    "--timeout",
    "timeout_seconds",
    default=30.0,
    show_default=True,
    type=float,
    help="Per-suite timeout in seconds.",
)
@click.option(
    "--output",
    "output_format",
    default="text",
    show_default=True,
    type=click.Choice(["text", "json", "junit"], case_sensitive=False),
    help="Output format.",
)
@click.option(
    "--parallel",
    is_flag=True,
    default=False,
    help="Run test cases in parallel threads.",
)
@click.option(
    "--out-file",
    "out_file",
    default=None,
    metavar="FILE",
    help="Write report output to FILE instead of stdout.",
)
def test_command(
    test_dir: str,
    mock_config_path: str | None,
    timeout_seconds: float,
    output_format: str,
    parallel: bool,
    out_file: str | None,
) -> None:
    """Run agent test cases in TEST_DIR.

    TEST_DIR defaults to 'tests/' relative to the current working directory.

    Examples:

        aumai-agentci test tests/

        aumai-agentci test tests/ --mock-config mock.yaml --output json

        aumai-agentci test tests/ --output junit > results.xml
    """
    try:
        mock_config = _load_mock_config(mock_config_path)
    except click.ClickException:
        raise
    except Exception as exc:
        raise click.ClickException(f"Failed to load mock config: {exc}") from exc

    config = AgentTestConfig(
        test_dir=test_dir,
        mock_config=mock_config,
        timeout_seconds=timeout_seconds,
        parallel=parallel,
    )

    runner = AgentTestRunner()

    try:
        suite_result = runner.run_suite(test_dir, config)
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc
    except Exception as exc:
        raise click.ClickException(f"Test run failed: {exc}") from exc

    # Select and run reporter.
    # ExitStack lets us manage the output file (when --out-file is given) in a
    # context manager without special-casing sys.stdout, which must never be
    # closed.  The stack is a no-op when writing to stdout.
    with contextlib.ExitStack() as stack:
        if out_file:
            output_stream = stack.enter_context(
                open(out_file, "w", encoding="utf-8")  # noqa: WPS515
            )
        else:
            output_stream = sys.stdout

        if output_format == "json":
            reporter: ConsoleReporter | JSONReporter | JUnitReporter = JSONReporter()
        elif output_format == "junit":
            reporter = JUnitReporter()
        else:
            reporter = ConsoleReporter()

        reporter.report(suite_result, stream=output_stream)

    # Exit with non-zero code when any tests failed
    if suite_result.failed > 0:
        sys.exit(1)


# ---------------------------------------------------------------------------
# init command
# ---------------------------------------------------------------------------


@main.command("init")
@click.argument("directory", default="agent-tests", metavar="DIRECTORY")
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Overwrite existing files.",
)
def init_command(directory: str, force: bool) -> None:
    """Create an example test directory with sample test cases and mock config.

    DIRECTORY is created (along with parent directories) if it does not exist.

    Example:

        aumai-agentci init agent-tests
    """
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)

    tests_file = target / "sample_tests.yaml"
    mock_file = target / "mock_config.yaml"

    file_pairs = [(tests_file, _SAMPLE_TESTS_YAML), (mock_file, _SAMPLE_MOCK_YAML)]
    for path, content in file_pairs:
        if path.exists() and not force:
            click.echo(f"  skip  {path} (already exists — use --force to overwrite)")
            continue
        path.write_text(content, encoding="utf-8")
        click.echo(f"  create  {path}")

    click.echo(
        f"\nInitialised test directory at '{directory}'.\n"
        "Run tests with:\n"
        f"  aumai-agentci test {directory}/ --mock-config {mock_file}"
    )


# ---------------------------------------------------------------------------
# validate command
# ---------------------------------------------------------------------------


@main.command("validate")
@click.argument("test_dir", metavar="TEST_DIR")
@click.option(
    "--strict",
    is_flag=True,
    default=False,
    help="Exit with code 1 if any validation warnings are found.",
)
def validate_command(test_dir: str, strict: bool) -> None:
    """Validate test YAML files in TEST_DIR without running them.

    Checks include:

    - YAML syntax validity
    - Required fields (name)
    - No duplicate test case names
    - expected_behavior key names

    Example:

        aumai-agentci validate tests/
    """
    runner = AgentTestRunner()
    warnings: list[str] = []
    errors: list[str] = []

    try:
        test_cases = runner.load_tests(test_dir)
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc
    except ValueError as exc:
        raise click.ClickException(f"Validation failed: {exc}") from exc

    if not test_cases:
        click.echo(
            click.style(f"WARNING: no test cases found in '{test_dir}'", fg="yellow")
        )
        if strict:
            sys.exit(1)
        return

    # Check for duplicate names
    seen_names: set[str] = set()
    for tc in test_cases:
        if tc.name in seen_names:
            warnings.append(f"Duplicate test case name: '{tc.name}'")
        seen_names.add(tc.name)

        if not tc.expected_behavior:
            warnings.append(
                f"Test '{tc.name}' has no expected_behavior — it will always pass."
            )

        if not tc.input_messages:
            warnings.append(f"Test '{tc.name}' has no input_messages.")

    if errors:
        for error in errors:
            click.echo(click.style(f"  ERROR   {error}", fg="red"))
    if warnings:
        for warning in warnings:
            click.echo(click.style(f"  WARNING {warning}", fg="yellow"))

    if not errors and not warnings:
        click.echo(
            click.style(
                f"  OK  {len(test_cases)} test case(s) validated successfully.",
                fg="green",
            )
        )
    else:
        click.echo(
            f"\n{len(test_cases)} test case(s) found: "
            f"{len(errors)} error(s), {len(warnings)} warning(s)."
        )

    if errors or (strict and warnings):
        sys.exit(1)


if __name__ == "__main__":
    main()
