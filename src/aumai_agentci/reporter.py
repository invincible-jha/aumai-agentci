"""Test result reporters for aumai-agentci.

Three reporters are provided:

- :class:`ConsoleReporter` — coloured terminal output.
- :class:`JSONReporter` — machine-readable JSON.
- :class:`JUnitReporter` — JUnit XML for CI systems.
"""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from typing import TextIO

from aumai_agentci.models import AgentTestResult, TestSuiteResult

__all__ = ["ConsoleReporter", "JSONReporter", "JUnitReporter"]

# ---------------------------------------------------------------------------
# ANSI colour helpers (falls back gracefully on non-TTY / Windows < 10)
# ---------------------------------------------------------------------------

_RESET = "\033[0m"
_BOLD = "\033[1m"
_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_CYAN = "\033[36m"
_DIM = "\033[2m"


def _supports_color(stream: TextIO) -> bool:
    """Return True when *stream* is a colour-capable TTY."""
    return hasattr(stream, "isatty") and stream.isatty()


def _colour(text: str, code: str, stream: TextIO) -> str:
    if _supports_color(stream):
        return f"{code}{text}{_RESET}"
    return text


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseReporter(ABC):
    """Abstract base for all reporters."""

    @abstractmethod
    def report(self, suite: TestSuiteResult, stream: TextIO | None = None) -> str:
        """Serialise *suite* to a string and optionally write to *stream*.

        Args:
            suite: The completed test suite result.
            stream: Optional output stream; defaults to ``sys.stdout``.

        Returns:
            The formatted string representation.
        """


# ---------------------------------------------------------------------------
# Console reporter
# ---------------------------------------------------------------------------


class ConsoleReporter(BaseReporter):
    """Render test results to the terminal with ANSI colours.

    Passed tests are shown in green, failed tests in red.  A summary
    line at the bottom shows overall counts and total duration.
    """

    def report(self, suite: TestSuiteResult, stream: TextIO | None = None) -> str:
        """Format and print *suite* to *stream* (default: stdout).

        Args:
            suite: Test suite result to render.
            stream: Target output stream.

        Returns:
            The rendered string.
        """
        out = stream or sys.stdout
        lines: list[str] = []

        header = _colour(
            f"\n{_BOLD}Test Suite: {suite.suite_name}{_RESET}",
            _BOLD,
            out,
        )
        lines.append(header)
        lines.append(_colour("=" * 60, _DIM, out))

        for result in suite.results:
            lines.append(self._format_result(result, out))

        lines.append(_colour("-" * 60, _DIM, out))
        lines.append(self._format_summary(suite, out))

        output = "\n".join(lines) + "\n"
        out.write(output)
        return output

    def _format_result(self, result: AgentTestResult, stream: TextIO) -> str:
        status = (
            _colour("PASS", _GREEN, stream)
            if result.passed
            else _colour("FAIL", _RED, stream)
        )
        line = f"  [{status}] {result.test_case_name} ({result.duration_ms:.1f} ms)"

        if not result.passed:
            for failure in result.assertions_failed:
                line += f"\n        {_colour(f'  x {failure}', _RED, stream)}"
        else:
            for assertion in result.assertions_passed:
                line += (
                    f"\n        {_colour(f'  v {assertion}', _DIM, stream)}"
                )
        return line

    def _format_summary(self, suite: TestSuiteResult, stream: TextIO) -> str:
        passed_str = _colour(str(suite.passed), _GREEN, stream)
        failed_str = _colour(str(suite.failed), _RED if suite.failed else _GREEN, stream)
        total_str = _colour(str(suite.total), _CYAN, stream)
        duration = _colour(f"{suite.duration_ms:.1f} ms", _DIM, stream)
        return (
            f"Results: {passed_str} passed, {failed_str} failed, "
            f"{total_str} total  ({duration})"
        )


# ---------------------------------------------------------------------------
# JSON reporter
# ---------------------------------------------------------------------------


class JSONReporter(BaseReporter):
    """Serialise test results as pretty-printed JSON.

    The output schema mirrors :class:`TestSuiteResult` field names so it
    can be parsed back with ``TestSuiteResult.model_validate_json()``.
    """

    def __init__(self, indent: int = 2) -> None:
        self._indent = indent

    def report(self, suite: TestSuiteResult, stream: TextIO | None = None) -> str:
        """Serialise *suite* to JSON and write to *stream*.

        Args:
            suite: Test suite result.
            stream: Target stream; defaults to stdout.

        Returns:
            JSON-formatted string.
        """
        out = stream or sys.stdout
        payload = suite.model_dump()
        output = json.dumps(payload, indent=self._indent, default=str)
        out.write(output + "\n")
        return output


# ---------------------------------------------------------------------------
# JUnit XML reporter
# ---------------------------------------------------------------------------


class JUnitReporter(BaseReporter):
    """Produce JUnit-compatible XML output for CI integration.

    Most CI platforms (Jenkins, GitLab CI, GitHub Actions via junit-reporter
    action, etc.) accept the JUnit XML format for test result visualisation.

    XML structure::

        <testsuites>
          <testsuite name="..." tests="N" failures="N" time="N">
            <testcase name="..." time="N">
              <failure message="...">...</failure>  <!-- only on failure -->
            </testcase>
          </testsuite>
        </testsuites>
    """

    def report(self, suite: TestSuiteResult, stream: TextIO | None = None) -> str:
        """Serialise *suite* to JUnit XML and write to *stream*.

        Args:
            suite: Test suite result.
            stream: Target stream; defaults to stdout.

        Returns:
            JUnit XML string.
        """
        out = stream or sys.stdout

        testsuites_el = ET.Element("testsuites")
        testsuite_el = ET.SubElement(
            testsuites_el,
            "testsuite",
            attrib={
                "name": suite.suite_name,
                "tests": str(suite.total),
                "failures": str(suite.failed),
                "errors": "0",
                "time": f"{suite.duration_ms / 1000:.3f}",
            },
        )

        for result in suite.results:
            testcase_el = ET.SubElement(
                testsuite_el,
                "testcase",
                attrib={
                    "name": result.test_case_name,
                    "time": f"{result.duration_ms / 1000:.3f}",
                },
            )

            if not result.passed:
                failure_msg = "; ".join(result.assertions_failed)
                failure_el = ET.SubElement(
                    testcase_el,
                    "failure",
                    attrib={"message": failure_msg, "type": "AssertionError"},
                )
                # Include the actual output in the failure body for debugging
                failure_el.text = (
                    f"Actual output:\n{result.actual_output}\n\n"
                    f"Failed assertions:\n"
                    + "\n".join(f"  - {f}" for f in result.assertions_failed)
                )
            else:
                # Record passed system-out for traceability
                system_out = ET.SubElement(testcase_el, "system-out")
                system_out.text = "\n".join(result.assertions_passed)

        ET.indent(testsuites_el, space="  ")
        output = ET.tostring(testsuites_el, encoding="unicode", xml_declaration=False)
        # Prepend XML declaration manually (ET.tostring can't do standalone correctly)
        full_output = '<?xml version="1.0" encoding="UTF-8"?>\n' + output + "\n"
        out.write(full_output)
        return full_output
