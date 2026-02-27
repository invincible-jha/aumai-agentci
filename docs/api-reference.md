# API Reference — aumai-agentci

All public symbols are importable from the top-level package:

```python
from aumai_agentci import (
    # Models
    MockLLMResponse, MockLLMConfig,
    AgentTestCase, AgentTestResult,
    TestSuiteResult, AgentTestConfig,
    # Core
    MockLLMProvider, AgentTestRunner,
    # Assertions
    assert_contains_text, assert_calls_tool, assert_no_pii,
    assert_max_tokens, assert_max_latency, assert_valid_json, assert_matches_schema,
    # Fixtures
    create_mock_openai, create_mock_anthropic, load_test_suite,
    # Reporters
    ConsoleReporter, JSONReporter, JUnitReporter,
)
```

---

## `aumai_agentci.models`

### `MockLLMResponse`

```python
class MockLLMResponse(BaseModel):
    content:       str
    model:         str
    tokens_used:   int    # ge=0
    latency_ms:    float  # ge=0.0
    finish_reason: str    # default="stop"
```

A single simulated LLM response. Used as items in `MockLLMConfig.responses`.

**Fields**

| Field | Type | Constraints | Description |
|---|---|---|---|
| `content` | `str` | — | The text content of the response. |
| `model` | `str` | — | Model identifier string (e.g. `"gpt-4o"`, `"mock-model"`). |
| `tokens_used` | `int` | >= 0 | Token count reported by this response. |
| `latency_ms` | `float` | >= 0.0 | Simulated response latency in milliseconds. |
| `finish_reason` | `str` | — | Finish reason string. Common values: `"stop"`, `"tool_calls"`, `"end_turn"`. |

---

### `MockLLMConfig`

```python
class MockLLMConfig(BaseModel):
    model_name:          str
    responses:           list[MockLLMResponse]  # default_factory=list
    default_latency_ms:  float   # default=50.0, ge=0.0
    failure_rate:        float   # default=0.0, ge=0.0, le=1.0
```

Configuration for a `MockLLMProvider`. Defines the response queue and
failure injection settings.

**Fields**

| Field | Type | Default | Constraints | Description |
|---|---|---|---|---|
| `model_name` | `str` | — | — | Name returned in responses when synthesizing fallback replies. |
| `responses` | `list[MockLLMResponse]` | `[]` | — | Ordered list of responses. Served round-robin. |
| `default_latency_ms` | `float` | 50.0 | >= 0.0 | Latency used when synthesizing empty fallback responses. |
| `failure_rate` | `float` | 0.0 | [0.0, 1.0] | Fraction of calls that raise `RuntimeError` to simulate API failures. |

---

### `AgentTestCase`

```python
class AgentTestCase(BaseModel):
    name:               str
    description:        str                      # default=""
    input_messages:     list[dict[str, object]]  # default_factory=list
    expected_behavior:  dict[str, object]        # default_factory=dict
    tags:               list[str]                # default_factory=list
```

A single agent test case, typically loaded from a YAML file.

**Fields**

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | — | Unique identifier for the test. Required. |
| `description` | `str` | `""` | Optional human-readable description. |
| `input_messages` | `list[dict]` | `[]` | Message list passed to `MockLLMProvider.complete()`. |
| `expected_behavior` | `dict` | `{}` | Assertion key-value pairs. See assertion keys table below. |
| `tags` | `list[str]` | `[]` | Labels for filtering or categorizing tests. |

**Recognized `expected_behavior` keys** (validated by `field_validator`):

| Key | Type | Description |
|---|---|---|
| `contains_text` | `str` | Output must contain this string (case-insensitive) |
| `calls_tools` | `str` or `list[str]` | Output must show evidence of named tool call(s) |
| `max_tokens` | `int` | `tokens_used` must be <= this value |
| `max_latency_ms` | `float` | Latency must be <= this value |
| `no_pii` | `bool` | When `true`, output must contain no PII |
| `valid_json` | `bool` | When `true`, output must be valid JSON |
| `matches_schema` | `dict` | Output (as JSON) must conform to this JSON Schema |

Unknown keys raise a `ValidationError`.

---

### `AgentTestResult`

```python
class AgentTestResult(BaseModel):
    test_case_name:    str
    passed:            bool
    actual_output:     str
    assertions_passed: list[str]  # default_factory=list
    assertions_failed: list[str]  # default_factory=list
    duration_ms:       float      # ge=0.0
    tokens_used:       int        # default=0, ge=0
```

The result of running a single `AgentTestCase`. Produced by
`AgentTestRunner.run_test`.

**Fields**

| Field | Type | Description |
|---|---|---|
| `test_case_name` | `str` | Name from the original test case. |
| `passed` | `bool` | True when `assertions_failed` is empty. |
| `actual_output` | `str` | The raw content returned by the mock provider. |
| `assertions_passed` | `list[str]` | Human-readable labels for each passing assertion. |
| `assertions_failed` | `list[str]` | Human-readable explanations for each failing assertion. |
| `duration_ms` | `float` | Wall-clock duration of the test in milliseconds. |
| `tokens_used` | `int` | Token count from the mock response. |

---

### `TestSuiteResult`

```python
class TestSuiteResult(BaseModel):
    suite_name:  str
    results:     list[AgentTestResult]  # default_factory=list
    total:       int    # ge=0
    passed:      int    # ge=0
    failed:      int    # ge=0
    duration_ms: float  # ge=0.0
```

Aggregated result for a full test suite run. Produced by
`AgentTestRunner.run_suite`.

**Fields**

| Field | Type | Description |
|---|---|---|
| `suite_name` | `str` | Derived from the test directory name. |
| `results` | `list[AgentTestResult]` | Per-test results in execution order. |
| `total` | `int` | Total number of test cases executed. |
| `passed` | `int` | Number of passing test cases. |
| `failed` | `int` | Number of failing test cases (`total - passed`). |
| `duration_ms` | `float` | Total wall-clock duration of the suite run. |

---

### `AgentTestConfig`

```python
class AgentTestConfig(BaseModel):
    test_dir:         str
    mock_config:      MockLLMConfig | None  # default=None
    timeout_seconds:  float   # default=30.0, gt=0.0
    parallel:         bool    # default=False
```

Top-level configuration for a test run. Passed to `AgentTestRunner.run_suite`.

**Fields**

| Field | Type | Default | Description |
|---|---|---|---|
| `test_dir` | `str` | — | Path to the directory containing YAML test case files. |
| `mock_config` | `MockLLMConfig \| None` | `None` | Mock provider config. When `None`, a default single-response config is used. |
| `timeout_seconds` | `float` | 30.0 | Suite-level timeout in seconds. Used in parallel mode. |
| `parallel` | `bool` | `False` | When `True`, test cases run in `ThreadPoolExecutor` threads. |

---

## `aumai_agentci.core`

### `MockLLMProvider`

```python
class MockLLMProvider:
    def __init__(self, config: MockLLMConfig) -> None: ...
```

Simulates LLM API responses from a `MockLLMConfig`. Serves responses in
round-robin order. Raises `RuntimeError` on a random subset of calls when
`config.failure_rate > 0`.

**Constructor parameters**

| Parameter | Type | Description |
|---|---|---|
| `config` | `MockLLMConfig` | Provider configuration including the response queue. |

---

#### `MockLLMProvider.complete`

```python
def complete(
    self,
    messages: list[dict[str, Any]],
    *,
    seed: int | None = None,
) -> MockLLMResponse
```

Return the next mock response, honouring the configured failure rate.

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `messages` | `list[dict]` | Input message list (mirrors the real LLM API shape; ignored for response selection). |
| `seed` | `int \| None` | RNG seed for deterministic failure injection in tests. |

**Returns:** The next `MockLLMResponse` in the round-robin queue. When
`responses` is empty, synthesizes a minimal empty response.

**Raises:** `RuntimeError` when a simulated failure is triggered.

---

#### `MockLLMProvider.call_count` (property)

```python
@property
def call_count(self) -> int
```

Number of times `complete()` has been called since creation or last `reset()`.

---

#### `MockLLMProvider.reset`

```python
def reset(self) -> None
```

Reset the round-robin index and call counter to zero. Useful when reusing a
provider instance across multiple test runs.

---

### `AgentTestRunner`

```python
class AgentTestRunner:
    def __init__(self) -> None: ...
```

Loads `AgentTestCase` objects from YAML files and executes them against a
`MockLLMProvider`.

---

#### `AgentTestRunner.load_tests`

```python
def load_tests(self, test_dir: str) -> list[AgentTestCase]
```

Discover and parse all `.yaml` / `.yml` files in `test_dir` recursively.

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `test_dir` | `str` | Path to the directory to search. |

**Returns:** Flat list of `AgentTestCase` objects from all discovered files.

**Raises:**
- `FileNotFoundError` — if `test_dir` does not exist.
- `ValueError` — if any YAML file cannot be parsed.

---

#### `AgentTestRunner.load_yaml_file`

```python
def load_yaml_file(self, path: Path, max_size: int = MAX_YAML_SIZE) -> list[AgentTestCase]
```

Parse a single YAML file into one or more test cases.

Supports three YAML structures:
1. Top-level `tests:` key wrapping a list of mappings.
2. Top-level list of mappings.
3. Single top-level mapping (one test case).

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `Path` | — | Path to the YAML file. |
| `max_size` | `int` | 10 MB | Maximum allowed file size in bytes. |

**Returns:** List of `AgentTestCase` objects.

**Raises:**
- `ValueError` — if the file exceeds `max_size`, the YAML structure is
  unrecognized, or any record fails Pydantic validation.

---

#### `AgentTestRunner.run_test`

```python
def run_test(
    self,
    test_case: AgentTestCase,
    mock: MockLLMProvider,
) -> AgentTestResult
```

Execute one test case against one mock provider and evaluate all assertions.

The mock provider is called once with `test_case.input_messages`. All
`expected_behavior` keys are then evaluated in order. Assertion results
(pass or fail) are collected into the returned `AgentTestResult`.

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `test_case` | `AgentTestCase` | The test to execute. |
| `mock` | `MockLLMProvider` | A configured mock provider. |

**Returns:** `AgentTestResult` capturing pass/fail state and all assertion details.

---

#### `AgentTestRunner.run_suite`

```python
def run_suite(
    self,
    test_dir: str,
    config: AgentTestConfig,
) -> TestSuiteResult
```

Discover and run all test cases in `test_dir`.

In serial mode, a single provider instance is shared across all tests.
In parallel mode (`config.parallel=True`), each thread receives its own
provider instance with the same config, avoiding shared-state issues.

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `test_dir` | `str` | Path to directory containing YAML test files. |
| `config` | `AgentTestConfig` | Run configuration (mock, timeout, parallelism). |

**Returns:** `TestSuiteResult` with aggregated counts and per-test details.

---

## `aumai_agentci.assertions`

All assertion functions return `bool`. They never raise exceptions on malformed
input — they return `False` instead. This allows callers to compose assertions
freely without try/except.

---

### `assert_contains_text`

```python
def assert_contains_text(output: str, expected: str) -> bool
```

Return `True` when `expected` appears as a case-insensitive substring of
`output`.

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `output` | `str` | Agent output string. |
| `expected` | `str` | Text fragment that must be present. |

```python
assert_contains_text("Hello World", "world")  # True
assert_contains_text("Hello World", "bye")    # False
```

---

### `assert_calls_tool`

```python
def assert_calls_tool(output: str, tool_name: str) -> bool
```

Return `True` when `output` contains evidence that `tool_name` was called.

Detection strategy (first match wins):
1. JSON object with `"tool"` key matching `tool_name`.
2. JSON object with `"function"` → `"name"` key matching `tool_name`.
3. OpenAI `"tool_calls"` array with matching `"function"."name"`.
4. Regex patterns: quoted key-value, function-call syntax `tool_name(`,
   XML-tag style `<tool_name>`.
5. Fallback: whole-word mention in plain text.

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `output` | `str` | Raw agent output (may be JSON or plain text). |
| `tool_name` | `str` | Exact tool name to look for. |

```python
output = '{"tool_calls": [{"function": {"name": "search_web"}}]}'
assert_calls_tool(output, "search_web")   # True
assert_calls_tool(output, "calculator")  # False
```

---

### `assert_no_pii`

```python
def assert_no_pii(output: str) -> bool
```

Return `True` when `output` contains no detectable PII.

Detects:
- Email addresses (RFC 5322 pattern).
- US phone numbers in various formats.
- US Social Security Numbers.
- Common credit card number patterns (Visa, Mastercard, Amex, Discover).

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `output` | `str` | Agent output to inspect. |

```python
assert_no_pii("The answer is 42.")               # True
assert_no_pii("Contact: alice@example.com")      # False (email)
assert_no_pii("Call 555-867-5309 for support")   # False (phone)
```

---

### `assert_max_tokens`

```python
def assert_max_tokens(tokens: int, max_tokens: int) -> bool
```

Return `True` when `tokens <= max_tokens`.

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `tokens` | `int` | Actual token count from the mock response. |
| `max_tokens` | `int` | Maximum allowed token count. |

---

### `assert_max_latency`

```python
def assert_max_latency(latency_ms: float, max_ms: float) -> bool
```

Return `True` when `latency_ms <= max_ms`.

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `latency_ms` | `float` | Measured latency in milliseconds. |
| `max_ms` | `float` | Allowed maximum latency in milliseconds. |

---

### `assert_valid_json`

```python
def assert_valid_json(output: str) -> bool
```

Return `True` when `output` (stripped of leading/trailing whitespace) parses
as valid JSON.

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `output` | `str` | String to validate. |

```python
assert_valid_json('{"ok": true}')  # True
assert_valid_json("not json")      # False
```

---

### `assert_matches_schema`

```python
def assert_matches_schema(output: str, schema: dict[str, object]) -> bool
```

Return `True` when `output` (parsed as JSON) conforms to `schema`.

Uses `jsonschema` when installed; falls back to a lightweight built-in
validator that supports `type`, `required`, and one level of `properties`.

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `output` | `str` | JSON string to validate. |
| `schema` | `dict` | JSON Schema object. |

```python
schema = {"type": "object", "required": ["status"], "properties": {"status": {"type": "string"}}}
assert_matches_schema('{"status": "ok"}', schema)   # True
assert_matches_schema('{"code": 200}', schema)      # False (missing "status")
```

---

## `aumai_agentci.fixtures`

### `create_mock_openai`

```python
def create_mock_openai(
    *,
    responses: list[MockLLMResponse] | None = None,
    failure_rate: float = 0.0,
) -> MockLLMProvider
```

Return a `MockLLMProvider` pre-configured with GPT-4o–style responses.

The default response set covers three typical patterns: plain text reply,
structured JSON reply, and tool-call reply using the OpenAI `tool_calls` format.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `responses` | `list[MockLLMResponse] \| None` | `None` | Override the default response list. |
| `failure_rate` | `float` | 0.0 | Fraction of calls that raise `RuntimeError`. |

---

### `create_mock_anthropic`

```python
def create_mock_anthropic(
    *,
    responses: list[MockLLMResponse] | None = None,
    failure_rate: float = 0.0,
) -> MockLLMProvider
```

Return a `MockLLMProvider` pre-configured with Claude-style responses.

The default response set covers: plain analytical reply, structured JSON reply,
and tool-use reply using the Anthropic `tool_use` format.

**Parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `responses` | `list[MockLLMResponse] \| None` | `None` | Override the default response list. |
| `failure_rate` | `float` | 0.0 | Fraction of calls that simulate API failures. |

---

### `load_test_suite`

```python
def load_test_suite(path: str) -> list[AgentTestCase]
```

Load test cases from `path`, which may be a file or a directory.

When `path` is a directory, all `.yaml` / `.yml` files are discovered
recursively. When it is a file, that file is parsed directly.

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `path` | `str` | Filesystem path to a YAML file or directory. |

**Returns:** List of `AgentTestCase` objects.

**Raises:**
- `FileNotFoundError` — if `path` does not exist.
- `ValueError` — if any YAML file cannot be parsed.

---

## `aumai_agentci.reporter`

All reporters implement the `BaseReporter` abstract class:

```python
class BaseReporter(ABC):
    @abstractmethod
    def report(self, suite: TestSuiteResult, stream: TextIO | None = None) -> str: ...
```

### `ConsoleReporter`

Renders test results to the terminal with ANSI colors. Pass tests appear in
green, failures in red. Includes a summary line with total counts and duration.

```python
reporter = ConsoleReporter()
reporter.report(suite_result)           # writes to sys.stdout
reporter.report(suite_result, stream=f) # writes to file
```

### `JSONReporter`

Serializes the `TestSuiteResult` as pretty-printed JSON. The output schema
mirrors `TestSuiteResult` field names and can be round-tripped with
`TestSuiteResult.model_validate_json()`.

```python
JSONReporter(indent=2).report(suite_result)  # default indent
```

### `JUnitReporter`

Produces JUnit-compatible XML. Compatible with Jenkins, GitLab CI, GitHub
Actions (via `mikepenz/action-junit-report`), CircleCI, and most other CI
platforms that accept JUnit test reports.

XML structure:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<testsuites>
  <testsuite name="..." tests="N" failures="N" errors="0" time="N.NNN">
    <testcase name="..." time="N.NNN">
      <!-- On failure: -->
      <failure message="assertion msg" type="AssertionError">Actual output...</failure>
      <!-- On pass: -->
      <system-out>passed assertion labels</system-out>
    </testcase>
  </testsuite>
</testsuites>
```

```python
JUnitReporter().report(suite_result, stream=sys.stdout)
```
