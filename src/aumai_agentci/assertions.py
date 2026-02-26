"""Agent-specific assertion helpers for aumai-agentci.

Each function returns a plain bool so callers can compose assertions
freely without catching exceptions.
"""

from __future__ import annotations

import json
import re

__all__ = [
    "assert_contains_text",
    "assert_calls_tool",
    "assert_no_pii",
    "assert_max_tokens",
    "assert_max_latency",
    "assert_valid_json",
    "assert_matches_schema",
]

# ---------------------------------------------------------------------------
# Compiled PII patterns — kept at module level for performance
# ---------------------------------------------------------------------------
_EMAIL_PATTERN: re.Pattern[str] = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
)
_PHONE_PATTERN: re.Pattern[str] = re.compile(
    r"""
    (?:
        (?:\+?1[\s\-.])?                    # optional country code
        (?:\(?\d{3}\)?[\s\-.]?)             # area code
        \d{3}[\s\-.]?\d{4}                  # subscriber number
    )
    """,
    re.VERBOSE,
)
_SSN_PATTERN: re.Pattern[str] = re.compile(
    r"\b(?!000|666|9\d{2})\d{3}[-\s]?(?!00)\d{2}[-\s]?(?!0000)\d{4}\b"
)
_CREDIT_CARD_PATTERN: re.Pattern[str] = re.compile(
    r"\b(?:4\d{3}|5[1-5]\d{2}|6(?:011|5\d{2})|3[47]\d{2})"
    r"[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b"
)


def assert_contains_text(output: str, expected: str) -> bool:
    """Return True when *expected* appears as a substring of *output*.

    The comparison is case-insensitive.

    Args:
        output: The actual agent output string.
        expected: The text fragment that must be present.

    Returns:
        True if *expected* (lowercased) is found in *output* (lowercased).
    """
    return expected.lower() in output.lower()


def assert_calls_tool(output: str, tool_name: str) -> bool:
    """Return True when *output* contains evidence of a tool call.

    Detection heuristics (any one is sufficient):
    - JSON object with a ``"tool"`` key matching *tool_name*.
    - JSON object with a ``"function"`` → ``"name"`` key matching *tool_name*.
    - Markdown code block that references the tool name.
    - Plain-text mention of the form ``<tool_name>(`` (function-call style).

    Args:
        output: Raw agent output.
        tool_name: The tool name to look for.

    Returns:
        True if any heuristic matches.
    """
    # Attempt JSON parsing first (handles structured outputs)
    try:
        parsed = json.loads(output)
        if isinstance(parsed, dict):
            if parsed.get("tool") == tool_name:
                return True
            func = parsed.get("function", {})
            if isinstance(func, dict) and func.get("name") == tool_name:
                return True
            # OpenAI tool_calls array
            tool_calls = parsed.get("tool_calls", [])
            if isinstance(tool_calls, list):
                for call in tool_calls:
                    if isinstance(call, dict):
                        inner = call.get("function", {})
                        if isinstance(inner, dict) and inner.get("name") == tool_name:
                            return True
    except (json.JSONDecodeError, ValueError):
        pass

    # Plain-text / markdown heuristics
    patterns: list[str] = [
        rf'["\']tool["\']\s*:\s*["\']?{re.escape(tool_name)}["\']?',
        rf'["\']name["\']\s*:\s*["\']?{re.escape(tool_name)}["\']?',
        rf"{re.escape(tool_name)}\s*\(",
        rf"<{re.escape(tool_name)}>",
        rf"\b{re.escape(tool_name)}\b",
    ]
    for pattern in patterns:
        if re.search(pattern, output, re.IGNORECASE):
            return True
    return False


def assert_no_pii(output: str) -> bool:
    """Return True when *output* contains no detectable PII.

    Checks for:
    - Email addresses
    - US phone numbers (various formats)
    - US Social Security Numbers
    - Common credit card number formats

    Args:
        output: Agent output to inspect.

    Returns:
        True if no PII patterns are detected.
    """
    for pattern in (_EMAIL_PATTERN, _PHONE_PATTERN, _SSN_PATTERN, _CREDIT_CARD_PATTERN):
        if pattern.search(output):
            return False
    return True


def assert_max_tokens(tokens: int, max_tokens: int) -> bool:
    """Return True when *tokens* does not exceed *max_tokens*.

    Args:
        tokens: Actual token count used.
        max_tokens: Upper limit.

    Returns:
        True if ``tokens <= max_tokens``.
    """
    return tokens <= max_tokens


def assert_max_latency(latency_ms: float, max_ms: float) -> bool:
    """Return True when *latency_ms* does not exceed *max_ms*.

    Args:
        latency_ms: Measured latency in milliseconds.
        max_ms: Allowed maximum latency in milliseconds.

    Returns:
        True if ``latency_ms <= max_ms``.
    """
    return latency_ms <= max_ms


def assert_valid_json(output: str) -> bool:
    """Return True when *output* is valid JSON.

    The function strips leading/trailing whitespace before parsing to
    tolerate common formatting differences.

    Args:
        output: String to validate.

    Returns:
        True if ``json.loads`` succeeds.
    """
    try:
        json.loads(output.strip())
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def assert_matches_schema(output: str, schema: dict[str, object]) -> bool:
    """Return True when *output* (parsed as JSON) conforms to *schema*.

    Uses ``jsonschema`` when available; falls back to a lightweight
    structural type-check when it is not installed, so the package has
    no hard dependency on ``jsonschema``.

    Args:
        output: JSON string to validate.
        schema: JSON Schema dict.

    Returns:
        True if the parsed output satisfies the schema.
    """
    try:
        data = json.loads(output.strip())
    except (json.JSONDecodeError, ValueError):
        return False

    try:
        import jsonschema  # type: ignore[import-untyped]

        try:
            jsonschema.validate(instance=data, schema=schema)
            return True
        except jsonschema.ValidationError:
            return False
    except ImportError:
        # Lightweight fallback: validate required keys and top-level types
        return _lightweight_schema_check(data, schema)


def _lightweight_schema_check(
    data: object, schema: dict[str, object]
) -> bool:
    """Minimal JSON-schema subset validator (no jsonschema dependency).

    Supports: ``type``, ``required``, ``properties`` (one level deep).
    """
    schema_type = schema.get("type")
    if schema_type is not None:
        type_map: dict[str, type[object]] = {
            "object": dict,
            "array": list,
            "string": str,
            "number": float,
            "integer": int,
            "boolean": bool,
            "null": type(None),
        }
        expected_type = type_map.get(str(schema_type))
        if expected_type is not None and not isinstance(data, expected_type):
            # int is a subtype of number in JSON Schema
            if not (schema_type == "number" and isinstance(data, int)):
                return False

    if not isinstance(data, dict):
        return True  # can't validate properties on non-objects

    required = schema.get("required", [])
    if isinstance(required, list):
        for key in required:
            if key not in data:
                return False

    properties = schema.get("properties", {})
    if isinstance(properties, dict):
        for prop_name, prop_schema in properties.items():
            if prop_name in data and isinstance(prop_schema, dict):
                if not _lightweight_schema_check(data[prop_name], prop_schema):
                    return False

    return True
