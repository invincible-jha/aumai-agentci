# Getting Started with aumai-agentci

This guide takes you from zero to a passing CI test suite for your AI agent in
under 15 minutes. It then covers the most common workflow patterns and how to
debug common problems.

---

## Prerequisites

- Python 3.11 or later.
- `pip` installed.
- Optional: a virtual environment tool.
- Optional: `jsonschema` package for full JSON Schema validation
  (`pip install jsonschema`). Without it, `matches_schema` assertions use a
  lightweight built-in fallback.

No LLM API keys or external services are required.

---

## Installation

### From PyPI

```bash
pip install aumai-agentci
```

Verify:

```bash
aumai-agentci --version
python -c "import aumai_agentci; print(aumai_agentci.__version__)"
```

### From source

```bash
git clone https://github.com/aumai/aumai-agentci
cd aumai-agentci
pip install -e .
```

### Development mode

```bash
pip install -e ".[dev]"
make test   # all tests should pass
```

---

## Your First Agent Test Suite

### Step 1: Scaffold the directory

The `init` command creates a complete working example with no configuration:

```bash
aumai-agentci init agent-tests
```

Inspect what was created:

```
agent-tests/
  sample_tests.yaml   ← five example test cases
  mock_config.yaml    ← mock LLM configuration
```

### Step 2: Understand the test case format

Open `agent-tests/sample_tests.yaml`. Each test case has four parts:

```yaml
- name: basic_response               # required: unique identifier
  description: >                     # optional: human-readable explanation
    Verify the agent produces a non-empty response.
  input_messages:                    # what you send to the agent
    - role: user
      content: "Say hello."
  expected_behavior:                 # what the agent must do
    contains_text: "hello"
  tags:                              # optional labels
    - smoke
```

### Step 3: Understand the mock config

Open `agent-tests/mock_config.yaml`. This file tells the mock provider what
to return, in order, when `complete()` is called:

```yaml
model_name: mock-gpt-4o
default_latency_ms: 50.0
failure_rate: 0.0      # 0% simulated failures
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
  # ... more responses
```

The mock provider serves responses in round-robin order. If there are 5 test
cases and 4 responses, the 5th test case reuses the 1st response.

### Step 4: Validate the YAML

Before running, verify the files are syntactically correct:

```bash
aumai-agentci validate agent-tests/
```

Expected output:
```
  OK  5 test case(s) validated successfully.
```

### Step 5: Run the tests

```bash
aumai-agentci test agent-tests/ --mock-config agent-tests/mock_config.yaml
```

All 5 tests should pass. Now try breaking one:

```bash
# Edit sample_tests.yaml: change "contains_text: hello" to "contains_text: goodbye"
aumai-agentci test agent-tests/ --mock-config agent-tests/mock_config.yaml
# Exit code: 1, one test fails
```

### Step 6: Get JUnit output for CI

```bash
aumai-agentci test agent-tests/ \
  --mock-config agent-tests/mock_config.yaml \
  --output junit \
  --out-file results.xml
```

The `results.xml` file can be consumed by GitHub Actions, GitLab CI, Jenkins,
CircleCI, and most other CI systems.

---

## Writing Your Own Tests

### Pattern 1: Basic output content check

Test that the agent's response contains expected text. The check is
case-insensitive.

```yaml
- name: confirms_task
  input_messages:
    - role: user
      content: "Please summarize this document."
  expected_behavior:
    contains_text: "summary"
```

### Pattern 2: Tool call verification

Test that the agent invokes the expected tool. Detection is multi-heuristic:
it parses JSON `tool_calls` arrays (OpenAI format), Anthropic `tool_use`
objects, plain-text function-call syntax, and XML-tag style.

```yaml
- name: searches_web
  input_messages:
    - role: user
      content: "What is the latest news about AI?"
  expected_behavior:
    calls_tools: search_web

# Multiple tools
- name: multi_tool
  input_messages:
    - role: user
      content: "Search and then summarize."
  expected_behavior:
    calls_tools:
      - search_web
      - summarize_text
```

Matching mock response for tool call tests:

```yaml
responses:
  - content: >
      {"tool_calls": [{"id": "call_001", "type": "function",
       "function": {"name": "search_web", "arguments": "{\"query\": \"AI news\"}"}}]}
    model: mock-agent
    tokens_used: 30
    latency_ms: 60.0
    finish_reason: tool_calls
```

### Pattern 3: PII leak detection

Assert that the agent does not echo back sensitive data. Checks for email
addresses, US phone numbers, SSNs, and credit card numbers.

```yaml
- name: no_pii_in_summary
  input_messages:
    - role: user
      content: "Summarize the customer record for John at john@corp.com."
  expected_behavior:
    no_pii: true
```

### Pattern 4: Token and latency budgets

Guard against regressions in response verbosity and performance. `tokens_used`
comes from the mock response. `latency_ms` uses the mock's configured latency
or the actual wall-clock elapsed time, whichever is larger.

```yaml
- name: concise_response
  input_messages:
    - role: user
      content: "In one sentence, what is Python?"
  expected_behavior:
    max_tokens: 50
    max_latency_ms: 200
```

### Pattern 5: Structured JSON output

Test that the agent produces valid JSON conforming to a schema.

```yaml
- name: status_object
  input_messages:
    - role: user
      content: "Return your current status as JSON."
  expected_behavior:
    valid_json: true
    matches_schema:
      type: object
      required:
        - status
        - version
      properties:
        status:
          type: string
        version:
          type: string
```

Matching mock response:

```yaml
responses:
  - content: '{"status": "operational", "version": "2.1.0"}'
    model: mock-agent
    tokens_used: 20
    latency_ms: 35.0
    finish_reason: stop
```

### Pattern 6: Simulating failures

Test that your agent handles transient API errors gracefully. Set
`failure_rate` in the mock config to inject random failures:

```yaml
# mock_config.yaml
model_name: mock-flaky
failure_rate: 0.50   # 50% of calls will raise RuntimeError
responses:
  - content: "Success!"
    model: mock-flaky
    tokens_used: 5
    latency_ms: 20.0
    finish_reason: stop
```

Tests that depend on error handling behavior:

```yaml
- name: handles_api_errors
  input_messages:
    - role: user
      content: "Do something."
  expected_behavior: {}   # no assertions — just verify the runner doesn't crash
```

When a mock failure is triggered, the `AgentTestResult` will have
`passed=False` and `assertions_failed` will contain a `provider_error:` entry.

---

## Common Patterns

### Pattern: Load test cases in pytest

```python
# tests/test_agent_behavior.py
import pytest
from aumai_agentci import load_test_suite, create_mock_openai, AgentTestRunner

TEST_CASES = load_test_suite("tests/agent-tests/")

@pytest.mark.parametrize("test_case", TEST_CASES, ids=lambda tc: tc.name)
def test_agent(test_case):
    provider = create_mock_openai()
    runner = AgentTestRunner()
    result = runner.run_test(test_case, provider)
    assert result.passed, f"Failed assertions: {result.assertions_failed}"
```

### Pattern: Use the Python API for programmatic runs

```python
from aumai_agentci import (
    AgentTestRunner, AgentTestConfig, MockLLMConfig, MockLLMResponse
)

config = AgentTestConfig(
    test_dir="tests/",
    mock_config=MockLLMConfig(
        model_name="mock",
        responses=[MockLLMResponse(
            content='{"status": "ok"}',
            model="mock",
            tokens_used=8,
            latency_ms=20.0,
            finish_reason="stop",
        )],
    ),
    parallel=True,
    timeout_seconds=60.0,
)

runner = AgentTestRunner()
suite = runner.run_suite("tests/", config)

if suite.failed > 0:
    raise SystemExit(f"{suite.failed} tests failed")
```

### Pattern: Write results to a file from Python

```python
import sys
from aumai_agentci import AgentTestRunner, AgentTestConfig
from aumai_agentci.reporter import JUnitReporter

runner = AgentTestRunner()
suite = runner.run_suite("tests/", config)

with open("test-results.xml", "w") as f:
    JUnitReporter().report(suite, stream=f)
```

---

## Troubleshooting FAQ

**Q: My test always passes even though the assertion should fail.**

Check that `expected_behavior` is present and spelled correctly. A test case
with an empty or missing `expected_behavior` dict always passes. Use
`aumai-agentci validate tests/` — it warns about test cases with no assertions.

---

**Q: `calls_tools` assertion is failing but I can see the tool name in the output.**

The assertion uses several heuristics. The most reliable is structured JSON.
Ensure your mock response uses the OpenAI `tool_calls` array format or the
Anthropic `tool_use` format. Plain-text matching is a fallback and may miss
cases where the tool name appears inside a longer identifier (e.g., `search`
will not false-positive inside `search_web` due to word-boundary anchors).

---

**Q: The runner finds 0 test cases.**

Check that your YAML files end in `.yaml` or `.yml`. The runner does a
recursive glob for both extensions. Also verify the directory path you passed
to `test` is correct.

---

**Q: `matches_schema` always returns True even with wrong data.**

If `jsonschema` is not installed, the assertion falls back to a lightweight
built-in validator that only checks `type`, `required`, and one level of
`properties`. Install `jsonschema` for full validation:

```bash
pip install jsonschema
```

---

**Q: How do I run only tests with a specific tag?**

Tag filtering is not yet implemented in the built-in runner. For tag-based
filtering, load the test cases manually and filter before running:

```python
from aumai_agentci import load_test_suite, AgentTestRunner

cases = [tc for tc in load_test_suite("tests/") if "smoke" in tc.tags]
runner = AgentTestRunner()
# ... create mock and run each case
```

---

**Q: Tests pass locally but fail in CI with a timeout.**

The default suite timeout is 30 seconds. Increase it with `--timeout`:

```bash
aumai-agentci test tests/ --timeout 120
```

---

**Q: I want to test with a real LLM, not a mock.**

`aumai-agentci` is designed for deterministic mock-based testing. For
integration tests against a real LLM, write standard pytest tests that
call your actual LLM client and use the assertion helpers directly:

```python
from aumai_agentci import assert_no_pii, assert_valid_json

output = my_real_agent.run("Return a JSON status")
assert assert_valid_json(output), f"Expected JSON, got: {output}"
assert assert_no_pii(output), "Response contains PII"
```
