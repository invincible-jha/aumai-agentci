"""Tests for aumai_agentci.cli — Click command group."""

from __future__ import annotations

import textwrap
from pathlib import Path

from click.testing import CliRunner

from aumai_agentci.cli import main

# ===========================================================================
# main / --version
# ===========================================================================


class TestMainGroup:
    """Tests for the top-level CLI group."""

    def test_version_flag(self) -> None:
        """--version may fail if the package is not installed (editable mode
        without pip install); accept either a clean version string or the
        RuntimeError that click raises in that case, but never an unexpected
        crash.  If the package IS installed, the output must contain '0.1.0'.
        """
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])
        # When installed: exit_code == 0 and "0.1.0" in output
        # When not installed: click raises RuntimeError with exit_code == 1
        if result.exit_code == 0:
            assert "0.1.0" in result.output
        else:
            # The only acceptable non-zero exit is the "not installed" error
            assert (
                "not installed" in str(result.exception).lower()
                or isinstance(result.exception, RuntimeError)
            )

    def test_help_flag(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "test" in result.output
        assert "init" in result.output
        assert "validate" in result.output


# ===========================================================================
# test command
# ===========================================================================


class TestTestCommand:
    """Tests for `aumai-agentci test`."""

    def test_basic_run_default_output(self, yaml_test_dir: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["test", str(yaml_test_dir)])
        # greet_user checks for "hello" but default mock says "Mock response: task..."
        # so some tests may fail — that is fine, we just check the CLI ran
        assert result.exit_code in (0, 1)
        assert "Test Suite" in result.output

    def test_json_output_format(self, yaml_test_dir: Path) -> None:
        import json

        runner = CliRunner()
        result = runner.invoke(
            main, ["test", str(yaml_test_dir), "--output", "json"]
        )
        assert result.exit_code in (0, 1)
        # Output must be parseable JSON
        parsed = json.loads(result.output)
        assert "suite_name" in parsed
        assert "total" in parsed

    def test_junit_output_format(self, yaml_test_dir: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            main, ["test", str(yaml_test_dir), "--output", "junit"]
        )
        assert result.exit_code in (0, 1)
        assert "<?xml" in result.output
        assert "<testsuites" in result.output

    def test_with_mock_config_file(self, tmp_path: Path) -> None:
        """Test the --mock-config option by placing tests and config in sibling dirs."""
        import textwrap

        # Tests live in their own subdirectory so the runner never recurses
        # into the config directory.
        tests_dir = tmp_path / "agent_tests"
        tests_dir.mkdir()
        (tests_dir / "simple.yaml").write_text(
            textwrap.dedent("""\
                name: simple_test
                input_messages:
                  - role: user
                    content: "Hello"
                expected_behavior:
                  no_pii: true
            """),
            encoding="utf-8",
        )

        # Mock config lives in a sibling directory — NOT inside tests_dir.
        config_dir = tmp_path / "cfg"
        config_dir.mkdir()
        config_file = config_dir / "mock_config.yaml"
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

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "test",
                str(tests_dir),
                "--mock-config",
                str(config_file),
            ],
        )
        assert result.exit_code in (0, 1)
        assert "Test Suite" in result.output

    def test_nonexistent_test_dir_exits_nonzero(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["test", "/nonexistent/test/dir"])
        assert result.exit_code != 0

    def test_nonexistent_mock_config_exits_nonzero(
        self, yaml_test_dir: Path
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "test",
                str(yaml_test_dir),
                "--mock-config",
                "/nonexistent/mock.yaml",
            ],
        )
        assert result.exit_code != 0

    def test_parallel_flag(self, yaml_test_dir: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["test", str(yaml_test_dir), "--parallel"])
        assert result.exit_code in (0, 1)

    def test_timeout_option(self, yaml_test_dir: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            main, ["test", str(yaml_test_dir), "--timeout", "60.0"]
        )
        assert result.exit_code in (0, 1)

    def test_out_file_option(
        self, yaml_test_dir: Path, tmp_path: Path
    ) -> None:
        out_path = tmp_path / "report.txt"
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["test", str(yaml_test_dir), "--out-file", str(out_path)],
        )
        assert result.exit_code in (0, 1)
        assert out_path.exists()
        assert out_path.read_text(encoding="utf-8").strip()

    def test_out_file_json_format(
        self, yaml_test_dir: Path, tmp_path: Path
    ) -> None:
        import json

        out_path = tmp_path / "report.json"
        runner = CliRunner()
        runner.invoke(
            main,
            [
                "test",
                str(yaml_test_dir),
                "--output",
                "json",
                "--out-file",
                str(out_path),
            ],
        )
        assert out_path.exists()
        content = out_path.read_text(encoding="utf-8")
        parsed = json.loads(content)
        assert "suite_name" in parsed

    def test_failed_tests_exit_code_one(
        self, tmp_path: Path
    ) -> None:
        """When any tests fail the exit code must be 1."""
        tests_dir = tmp_path / "failing_tests"
        tests_dir.mkdir()
        (tests_dir / "failing.yaml").write_text(
            textwrap.dedent("""\
                name: always_fail
                input_messages:
                  - role: user
                    content: "Greet me."
                expected_behavior:
                  contains_text: "ZZZNOTREALWORD"
            """),
            encoding="utf-8",
        )
        runner = CliRunner()
        result = runner.invoke(main, ["test", str(tests_dir)])
        assert result.exit_code == 1

    def test_all_passing_tests_exit_code_zero(
        self, tmp_path: Path
    ) -> None:
        """All-passing suite must exit with code 0."""
        tests_dir = tmp_path / "passing_tests"
        tests_dir.mkdir()
        # Default mock returns "Mock response: task completed successfully."
        # which contains "mock"
        (tests_dir / "passing.yaml").write_text(
            textwrap.dedent("""\
                name: always_pass
                input_messages:
                  - role: user
                    content: "Hello."
                expected_behavior:
                  contains_text: "mock"
            """),
            encoding="utf-8",
        )
        runner = CliRunner()
        result = runner.invoke(main, ["test", str(tests_dir)])
        assert result.exit_code == 0

    def test_invalid_mock_config_yaml_content(
        self, yaml_test_dir: Path, tmp_path: Path
    ) -> None:
        """A mock config that is not a YAML mapping triggers an error."""
        bad_mock = tmp_path / "bad_mock.yaml"
        bad_mock.write_text("- just\n- a\n- list\n", encoding="utf-8")
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["test", str(yaml_test_dir), "--mock-config", str(bad_mock)],
        )
        assert result.exit_code != 0


# ===========================================================================
# init command
# ===========================================================================


class TestInitCommand:
    """Tests for `aumai-agentci init`."""

    def test_creates_directory(self, tmp_path: Path) -> None:
        target = tmp_path / "new-tests"
        runner = CliRunner()
        result = runner.invoke(main, ["init", str(target)])
        assert result.exit_code == 0
        assert target.is_dir()

    def test_creates_sample_tests_yaml(self, tmp_path: Path) -> None:
        target = tmp_path / "agent-tests"
        runner = CliRunner()
        runner.invoke(main, ["init", str(target)])
        assert (target / "sample_tests.yaml").exists()

    def test_creates_mock_config_yaml(self, tmp_path: Path) -> None:
        target = tmp_path / "agent-tests"
        runner = CliRunner()
        runner.invoke(main, ["init", str(target)])
        assert (target / "mock_config.yaml").exists()

    def test_output_mentions_created_files(self, tmp_path: Path) -> None:
        target = tmp_path / "agent-tests"
        runner = CliRunner()
        result = runner.invoke(main, ["init", str(target)])
        assert "create" in result.output

    def test_does_not_overwrite_existing_without_force(
        self, tmp_path: Path
    ) -> None:
        target = tmp_path / "agent-tests"
        runner = CliRunner()
        runner.invoke(main, ["init", str(target)])
        # Modify the file, then run init again without --force
        sample = target / "sample_tests.yaml"
        sample.write_text("modified content", encoding="utf-8")
        runner.invoke(main, ["init", str(target)])
        assert sample.read_text(encoding="utf-8") == "modified content"

    def test_force_flag_overwrites_existing(self, tmp_path: Path) -> None:
        target = tmp_path / "agent-tests"
        runner = CliRunner()
        runner.invoke(main, ["init", str(target)])
        sample = target / "sample_tests.yaml"
        sample.write_text("modified content", encoding="utf-8")
        runner.invoke(main, ["init", str(target), "--force"])
        content = sample.read_text(encoding="utf-8")
        assert content != "modified content"

    def test_skip_message_when_file_exists_no_force(
        self, tmp_path: Path
    ) -> None:
        target = tmp_path / "agent-tests"
        runner = CliRunner()
        runner.invoke(main, ["init", str(target)])
        result = runner.invoke(main, ["init", str(target)])
        assert "skip" in result.output

    def test_nested_directory_created(self, tmp_path: Path) -> None:
        target = tmp_path / "a" / "b" / "c" / "tests"
        runner = CliRunner()
        result = runner.invoke(main, ["init", str(target)])
        assert result.exit_code == 0
        assert target.is_dir()

    def test_output_mentions_run_command(self, tmp_path: Path) -> None:
        target = tmp_path / "my-tests"
        runner = CliRunner()
        result = runner.invoke(main, ["init", str(target)])
        assert "aumai-agentci test" in result.output

    def test_sample_yaml_is_valid_for_runner(self, tmp_path: Path) -> None:
        """The scaffolded sample_tests.yaml must be parseable by AgentTestRunner."""
        from aumai_agentci.fixtures import load_test_suite

        target = tmp_path / "agent-tests"
        runner_cli = CliRunner()
        runner_cli.invoke(main, ["init", str(target)])

        # Load only the sample_tests.yaml file, not the mock_config.yaml —
        # load_tests() would attempt to parse mock_config.yaml as a test case
        # because it is a YAML file in the same directory.
        cases = load_test_suite(str(target / "sample_tests.yaml"))
        assert len(cases) > 0


# ===========================================================================
# validate command
# ===========================================================================


class TestValidateCommand:
    """Tests for `aumai-agentci validate`."""

    def test_valid_tests_exit_zero(self, yaml_test_dir: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["validate", str(yaml_test_dir)])
        assert result.exit_code == 0

    def test_valid_tests_ok_output(self, yaml_test_dir: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["validate", str(yaml_test_dir)])
        assert "OK" in result.output or "validated" in result.output

    def test_nonexistent_dir_exits_nonzero(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["validate", "/nonexistent/path"])
        assert result.exit_code != 0

    def test_empty_directory_warns(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["validate", str(tmp_path)])
        assert "WARNING" in result.output or "no test cases" in result.output.lower()

    def test_empty_directory_strict_exits_nonzero(
        self, tmp_path: Path
    ) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["validate", str(tmp_path), "--strict"])
        assert result.exit_code != 0

    def test_duplicate_name_warns(self, tmp_path: Path) -> None:
        dup_file = tmp_path / "dup.yaml"
        dup_file.write_text(
            textwrap.dedent("""\
                - name: duplicate_name
                  expected_behavior:
                    no_pii: true

                - name: duplicate_name
                  expected_behavior:
                    no_pii: true
            """),
            encoding="utf-8",
        )
        runner = CliRunner()
        result = runner.invoke(main, ["validate", str(tmp_path)])
        assert "Duplicate" in result.output or "duplicate" in result.output.lower()

    def test_no_expected_behavior_warns(self, tmp_path: Path) -> None:
        no_behavior = tmp_path / "no_behavior.yaml"
        no_behavior.write_text(
            textwrap.dedent("""\
                name: no_behavior_test
                input_messages:
                  - role: user
                    content: "Hello"
            """),
            encoding="utf-8",
        )
        runner = CliRunner()
        result = runner.invoke(main, ["validate", str(tmp_path)])
        assert "WARNING" in result.output or "always pass" in result.output.lower()

    def test_no_input_messages_warns(self, tmp_path: Path) -> None:
        no_input = tmp_path / "no_input.yaml"
        no_input.write_text(
            textwrap.dedent("""\
                name: no_input_test
                expected_behavior:
                  no_pii: true
            """),
            encoding="utf-8",
        )
        runner = CliRunner()
        result = runner.invoke(main, ["validate", str(tmp_path)])
        assert "WARNING" in result.output

    def test_strict_with_warnings_exits_nonzero(self, tmp_path: Path) -> None:
        warn_file = tmp_path / "warn.yaml"
        warn_file.write_text(
            textwrap.dedent("""\
                name: warn_test
                input_messages: []
                expected_behavior: {}
            """),
            encoding="utf-8",
        )
        runner = CliRunner()
        result = runner.invoke(
            main, ["validate", str(tmp_path), "--strict"]
        )
        assert result.exit_code != 0

    def test_invalid_yaml_schema_exits_nonzero(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "bad.yaml"
        bad_file.write_text(
            textwrap.dedent("""\
                name: bad_case
                expected_behavior:
                  unknown_key: something
            """),
            encoding="utf-8",
        )
        runner = CliRunner()
        result = runner.invoke(main, ["validate", str(tmp_path)])
        assert result.exit_code != 0

    def test_count_in_output(self, yaml_test_dir: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["validate", str(yaml_test_dir)])
        # Should mention the count of validated tests
        assert "3" in result.output
