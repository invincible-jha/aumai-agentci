"""Tests for aumai_agentci.reporter — ConsoleReporter, JSONReporter, JUnitReporter."""

from __future__ import annotations

import io
import json
import xml.etree.ElementTree as ET

import pytest

from aumai_agentci.models import AgentTestResult, TestSuiteResult
from aumai_agentci.reporter import (
    ConsoleReporter,
    JSONReporter,
    JUnitReporter,
    _colour,
    _supports_color,
)

# ===========================================================================
# _supports_color / _colour helpers
# ===========================================================================


class TestColourHelpers:
    """Tests for the private colour helper functions."""

    def test_supports_color_returns_false_for_stringio(self) -> None:
        stream = io.StringIO()
        assert _supports_color(stream) is False

    def test_supports_color_returns_false_for_non_tty(self) -> None:
        stream = io.StringIO()
        assert _supports_color(stream) is False

    def test_colour_returns_plain_text_for_non_tty(self) -> None:
        stream = io.StringIO()
        result = _colour("hello", "\033[32m", stream)
        assert result == "hello"
        assert "\033" not in result

    def test_colour_returns_ansi_codes_for_tty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stream = io.StringIO()
        monkeypatch.setattr(stream, "isatty", lambda: True)
        result = _colour("hello", "\033[32m", stream)
        assert "\033[32m" in result
        assert "\033[0m" in result  # reset code
        assert "hello" in result


# ===========================================================================
# ConsoleReporter
# ===========================================================================


class TestConsoleReporter:
    """Tests for ConsoleReporter."""

    def test_report_returns_string(
        self, mixed_suite_result: TestSuiteResult
    ) -> None:
        reporter = ConsoleReporter()
        stream = io.StringIO()
        result = reporter.report(mixed_suite_result, stream=stream)
        assert isinstance(result, str)

    def test_report_writes_to_stream(
        self, mixed_suite_result: TestSuiteResult
    ) -> None:
        reporter = ConsoleReporter()
        stream = io.StringIO()
        reporter.report(mixed_suite_result, stream=stream)
        content = stream.getvalue()
        assert len(content) > 0

    def test_report_contains_suite_name(
        self, mixed_suite_result: TestSuiteResult
    ) -> None:
        reporter = ConsoleReporter()
        stream = io.StringIO()
        reporter.report(mixed_suite_result, stream=stream)
        assert "mixed_suite" in stream.getvalue()

    def test_report_contains_pass_marker(
        self, all_passing_suite: TestSuiteResult
    ) -> None:
        reporter = ConsoleReporter()
        stream = io.StringIO()
        reporter.report(all_passing_suite, stream=stream)
        assert "PASS" in stream.getvalue()

    def test_report_contains_fail_marker(
        self, mixed_suite_result: TestSuiteResult
    ) -> None:
        reporter = ConsoleReporter()
        stream = io.StringIO()
        reporter.report(mixed_suite_result, stream=stream)
        assert "FAIL" in stream.getvalue()

    def test_report_contains_test_case_names(
        self, mixed_suite_result: TestSuiteResult
    ) -> None:
        reporter = ConsoleReporter()
        stream = io.StringIO()
        reporter.report(mixed_suite_result, stream=stream)
        content = stream.getvalue()
        assert "passing_test" in content
        assert "failing_test" in content

    def test_report_contains_summary_counts(
        self, mixed_suite_result: TestSuiteResult
    ) -> None:
        reporter = ConsoleReporter()
        stream = io.StringIO()
        reporter.report(mixed_suite_result, stream=stream)
        content = stream.getvalue()
        # Should mention 1 passed, 1 failed, 2 total
        assert "1" in content

    def test_report_shows_failed_assertion_detail(
        self, mixed_suite_result: TestSuiteResult
    ) -> None:
        reporter = ConsoleReporter()
        stream = io.StringIO()
        reporter.report(mixed_suite_result, stream=stream)
        content = stream.getvalue()
        assert "not found" in content

    def test_report_shows_passed_assertion_detail(
        self, all_passing_suite: TestSuiteResult
    ) -> None:
        reporter = ConsoleReporter()
        stream = io.StringIO()
        reporter.report(all_passing_suite, stream=stream)
        content = stream.getvalue()
        assert "contains_text" in content

    def test_report_return_value_matches_stream(
        self, mixed_suite_result: TestSuiteResult
    ) -> None:
        reporter = ConsoleReporter()
        stream = io.StringIO()
        returned = reporter.report(mixed_suite_result, stream=stream)
        assert returned == stream.getvalue()

    def test_report_empty_suite(self) -> None:
        suite = TestSuiteResult(
            suite_name="empty",
            results=[],
            total=0,
            passed=0,
            failed=0,
            duration_ms=5.0,
        )
        reporter = ConsoleReporter()
        stream = io.StringIO()
        result = reporter.report(suite, stream=stream)
        assert "empty" in result

    def test_report_uses_stdout_when_no_stream(
        self, mixed_suite_result: TestSuiteResult, capsys: pytest.CaptureFixture
    ) -> None:
        reporter = ConsoleReporter()
        reporter.report(mixed_suite_result, stream=None)
        captured = capsys.readouterr()
        assert "mixed_suite" in captured.out

    def test_duration_is_shown_per_result(
        self, all_passing_suite: TestSuiteResult
    ) -> None:
        reporter = ConsoleReporter()
        stream = io.StringIO()
        reporter.report(all_passing_suite, stream=stream)
        # Duration in ms should appear somewhere
        assert "ms" in stream.getvalue()


# ===========================================================================
# JSONReporter
# ===========================================================================


class TestJSONReporter:
    """Tests for JSONReporter."""

    def test_report_returns_valid_json_string(
        self, mixed_suite_result: TestSuiteResult
    ) -> None:
        reporter = JSONReporter()
        stream = io.StringIO()
        output = reporter.report(mixed_suite_result, stream=stream)
        parsed = json.loads(output)
        assert isinstance(parsed, dict)

    def test_report_contains_suite_name(
        self, mixed_suite_result: TestSuiteResult
    ) -> None:
        reporter = JSONReporter()
        stream = io.StringIO()
        output = reporter.report(mixed_suite_result, stream=stream)
        parsed = json.loads(output)
        assert parsed["suite_name"] == "mixed_suite"

    def test_report_contains_correct_total(
        self, mixed_suite_result: TestSuiteResult
    ) -> None:
        reporter = JSONReporter()
        stream = io.StringIO()
        output = reporter.report(mixed_suite_result, stream=stream)
        parsed = json.loads(output)
        assert parsed["total"] == 2

    def test_report_contains_passed_count(
        self, mixed_suite_result: TestSuiteResult
    ) -> None:
        reporter = JSONReporter()
        stream = io.StringIO()
        output = reporter.report(mixed_suite_result, stream=stream)
        parsed = json.loads(output)
        assert parsed["passed"] == 1
        assert parsed["failed"] == 1

    def test_report_results_list_length(
        self, mixed_suite_result: TestSuiteResult
    ) -> None:
        reporter = JSONReporter()
        stream = io.StringIO()
        output = reporter.report(mixed_suite_result, stream=stream)
        parsed = json.loads(output)
        assert len(parsed["results"]) == 2

    def test_report_result_has_test_case_name(
        self, all_passing_suite: TestSuiteResult
    ) -> None:
        reporter = JSONReporter()
        stream = io.StringIO()
        output = reporter.report(all_passing_suite, stream=stream)
        parsed = json.loads(output)
        assert parsed["results"][0]["test_case_name"] == "passing_test"

    def test_report_result_passed_flag(
        self, all_passing_suite: TestSuiteResult
    ) -> None:
        reporter = JSONReporter()
        stream = io.StringIO()
        output = reporter.report(all_passing_suite, stream=stream)
        parsed = json.loads(output)
        assert parsed["results"][0]["passed"] is True

    def test_report_writes_to_stream(
        self, mixed_suite_result: TestSuiteResult
    ) -> None:
        reporter = JSONReporter()
        stream = io.StringIO()
        reporter.report(mixed_suite_result, stream=stream)
        content = stream.getvalue()
        assert content.strip()

    def test_report_is_pretty_printed(
        self, mixed_suite_result: TestSuiteResult
    ) -> None:
        reporter = JSONReporter(indent=2)
        stream = io.StringIO()
        output = reporter.report(mixed_suite_result, stream=stream)
        # Pretty-printed JSON has newlines inside
        assert "\n" in output

    def test_custom_indent(
        self, all_passing_suite: TestSuiteResult
    ) -> None:
        reporter = JSONReporter(indent=4)
        stream = io.StringIO()
        output = reporter.report(all_passing_suite, stream=stream)
        # 4-space indent should produce 4 spaces
        assert "    " in output

    def test_output_can_round_trip_to_model(
        self, mixed_suite_result: TestSuiteResult
    ) -> None:
        reporter = JSONReporter()
        stream = io.StringIO()
        output = reporter.report(mixed_suite_result, stream=stream)
        restored = TestSuiteResult.model_validate_json(output)
        assert restored.suite_name == mixed_suite_result.suite_name
        assert restored.total == mixed_suite_result.total

    def test_report_uses_stdout_when_no_stream(
        self, all_passing_suite: TestSuiteResult, capsys: pytest.CaptureFixture
    ) -> None:
        reporter = JSONReporter()
        reporter.report(all_passing_suite, stream=None)
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["suite_name"] == "green_suite"

    def test_empty_results_list(self) -> None:
        suite = TestSuiteResult(
            suite_name="empty",
            results=[],
            total=0,
            passed=0,
            failed=0,
            duration_ms=0.0,
        )
        reporter = JSONReporter()
        stream = io.StringIO()
        output = reporter.report(suite, stream=stream)
        parsed = json.loads(output)
        assert parsed["results"] == []


# ===========================================================================
# JUnitReporter
# ===========================================================================


class TestJUnitReporter:
    """Tests for JUnitReporter."""

    def test_report_returns_xml_string(
        self, mixed_suite_result: TestSuiteResult
    ) -> None:
        reporter = JUnitReporter()
        stream = io.StringIO()
        output = reporter.report(mixed_suite_result, stream=stream)
        assert output.startswith("<?xml")

    def test_report_is_parseable_xml(
        self, mixed_suite_result: TestSuiteResult
    ) -> None:
        reporter = JUnitReporter()
        stream = io.StringIO()
        output = reporter.report(mixed_suite_result, stream=stream)
        # Strip the XML declaration for ElementTree
        xml_body = output.split("\n", 1)[1] if output.startswith("<?xml") else output
        root = ET.fromstring(xml_body)
        assert root.tag == "testsuites"

    def test_testsuite_attributes(
        self, mixed_suite_result: TestSuiteResult
    ) -> None:
        reporter = JUnitReporter()
        stream = io.StringIO()
        output = reporter.report(mixed_suite_result, stream=stream)
        xml_body = output.split("\n", 1)[1]
        root = ET.fromstring(xml_body)
        testsuite = root.find("testsuite")
        assert testsuite is not None
        assert testsuite.get("name") == "mixed_suite"
        assert testsuite.get("tests") == "2"
        assert testsuite.get("failures") == "1"

    def test_testcase_elements_present(
        self, mixed_suite_result: TestSuiteResult
    ) -> None:
        reporter = JUnitReporter()
        stream = io.StringIO()
        output = reporter.report(mixed_suite_result, stream=stream)
        xml_body = output.split("\n", 1)[1]
        root = ET.fromstring(xml_body)
        testcases = root.findall(".//testcase")
        assert len(testcases) == 2

    def test_failing_testcase_has_failure_element(
        self, mixed_suite_result: TestSuiteResult
    ) -> None:
        reporter = JUnitReporter()
        stream = io.StringIO()
        output = reporter.report(mixed_suite_result, stream=stream)
        xml_body = output.split("\n", 1)[1]
        root = ET.fromstring(xml_body)
        failure_elements = root.findall(".//failure")
        assert len(failure_elements) == 1

    def test_failure_element_message_attribute(
        self, mixed_suite_result: TestSuiteResult
    ) -> None:
        reporter = JUnitReporter()
        stream = io.StringIO()
        output = reporter.report(mixed_suite_result, stream=stream)
        xml_body = output.split("\n", 1)[1]
        root = ET.fromstring(xml_body)
        failure = root.find(".//failure")
        assert failure is not None
        assert failure.get("message") is not None
        assert "not found" in failure.get("message", "")

    def test_failure_element_has_type_attribute(
        self, mixed_suite_result: TestSuiteResult
    ) -> None:
        reporter = JUnitReporter()
        stream = io.StringIO()
        output = reporter.report(mixed_suite_result, stream=stream)
        xml_body = output.split("\n", 1)[1]
        root = ET.fromstring(xml_body)
        failure = root.find(".//failure")
        assert failure is not None
        assert failure.get("type") == "AssertionError"

    def test_failure_body_contains_actual_output(
        self, mixed_suite_result: TestSuiteResult
    ) -> None:
        reporter = JUnitReporter()
        stream = io.StringIO()
        output = reporter.report(mixed_suite_result, stream=stream)
        xml_body = output.split("\n", 1)[1]
        root = ET.fromstring(xml_body)
        failure = root.find(".//failure")
        assert failure is not None
        assert "Actual output" in (failure.text or "")

    def test_passing_testcase_has_system_out(
        self, all_passing_suite: TestSuiteResult
    ) -> None:
        reporter = JUnitReporter()
        stream = io.StringIO()
        output = reporter.report(all_passing_suite, stream=stream)
        xml_body = output.split("\n", 1)[1]
        root = ET.fromstring(xml_body)
        system_out = root.find(".//system-out")
        assert system_out is not None

    def test_system_out_contains_assertions(
        self, all_passing_suite: TestSuiteResult
    ) -> None:
        reporter = JUnitReporter()
        stream = io.StringIO()
        output = reporter.report(all_passing_suite, stream=stream)
        xml_body = output.split("\n", 1)[1]
        root = ET.fromstring(xml_body)
        system_out = root.find(".//system-out")
        assert system_out is not None
        assert "contains_text" in (system_out.text or "")

    def test_report_writes_to_stream(
        self, mixed_suite_result: TestSuiteResult
    ) -> None:
        reporter = JUnitReporter()
        stream = io.StringIO()
        reporter.report(mixed_suite_result, stream=stream)
        assert stream.getvalue().startswith("<?xml")

    def test_return_value_matches_stream(
        self, mixed_suite_result: TestSuiteResult
    ) -> None:
        reporter = JUnitReporter()
        stream = io.StringIO()
        returned = reporter.report(mixed_suite_result, stream=stream)
        assert returned == stream.getvalue()

    def test_report_uses_stdout_when_no_stream(
        self, all_passing_suite: TestSuiteResult, capsys: pytest.CaptureFixture
    ) -> None:
        reporter = JUnitReporter()
        reporter.report(all_passing_suite, stream=None)
        captured = capsys.readouterr()
        assert "<?xml" in captured.out

    def test_empty_suite(self) -> None:
        suite = TestSuiteResult(
            suite_name="empty",
            results=[],
            total=0,
            passed=0,
            failed=0,
            duration_ms=0.0,
        )
        reporter = JUnitReporter()
        stream = io.StringIO()
        output = reporter.report(suite, stream=stream)
        xml_body = output.split("\n", 1)[1]
        root = ET.fromstring(xml_body)
        testcases = root.findall(".//testcase")
        assert testcases == []

    def test_time_attribute_in_seconds(
        self, mixed_suite_result: TestSuiteResult
    ) -> None:
        reporter = JUnitReporter()
        stream = io.StringIO()
        output = reporter.report(mixed_suite_result, stream=stream)
        xml_body = output.split("\n", 1)[1]
        root = ET.fromstring(xml_body)
        testsuite = root.find("testsuite")
        assert testsuite is not None
        # duration_ms=100.0 => 0.100 seconds
        time_val = float(testsuite.get("time", "0"))
        assert abs(time_val - 0.1) < 0.01

    def test_xml_declaration_present(
        self, all_passing_suite: TestSuiteResult
    ) -> None:
        reporter = JUnitReporter()
        stream = io.StringIO()
        output = reporter.report(all_passing_suite, stream=stream)
        assert '<?xml version="1.0" encoding="UTF-8"?>' in output

    def test_errors_attribute_is_zero(
        self, mixed_suite_result: TestSuiteResult
    ) -> None:
        reporter = JUnitReporter()
        stream = io.StringIO()
        output = reporter.report(mixed_suite_result, stream=stream)
        xml_body = output.split("\n", 1)[1]
        root = ET.fromstring(xml_body)
        testsuite = root.find("testsuite")
        assert testsuite is not None
        assert testsuite.get("errors") == "0"

    def test_multiple_failures_all_have_failure_elements(self) -> None:
        failing_1 = AgentTestResult(
            test_case_name="fail_1",
            passed=False,
            actual_output="bad",
            assertions_failed=["x"],
            duration_ms=5.0,
            tokens_used=0,
        )
        failing_2 = AgentTestResult(
            test_case_name="fail_2",
            passed=False,
            actual_output="also bad",
            assertions_failed=["y"],
            duration_ms=5.0,
            tokens_used=0,
        )
        suite = TestSuiteResult(
            suite_name="all_fail",
            results=[failing_1, failing_2],
            total=2,
            passed=0,
            failed=2,
            duration_ms=10.0,
        )
        reporter = JUnitReporter()
        stream = io.StringIO()
        output = reporter.report(suite, stream=stream)
        xml_body = output.split("\n", 1)[1]
        root = ET.fromstring(xml_body)
        failures = root.findall(".//failure")
        assert len(failures) == 2
