"""Tests for aumai_agentci.assertions — all assertion helper functions."""

from __future__ import annotations

import json

from hypothesis import given, settings
from hypothesis import strategies as st

from aumai_agentci.assertions import (
    _lightweight_schema_check,
    assert_calls_tool,
    assert_contains_text,
    assert_matches_schema,
    assert_max_latency,
    assert_max_tokens,
    assert_no_pii,
    assert_valid_json,
)

# ---------------------------------------------------------------------------
# assert_contains_text
# ---------------------------------------------------------------------------


class TestAssertContainsText:
    """Tests for assert_contains_text."""

    def test_exact_match(self) -> None:
        assert assert_contains_text("hello world", "hello") is True

    def test_case_insensitive_upper_expected(self) -> None:
        assert assert_contains_text("hello world", "HELLO") is True

    def test_case_insensitive_upper_output(self) -> None:
        assert assert_contains_text("HELLO WORLD", "hello") is True

    def test_substring_in_middle(self) -> None:
        assert assert_contains_text("The quick brown fox", "quick brown") is True

    def test_not_present(self) -> None:
        assert assert_contains_text("hello world", "goodbye") is False

    def test_empty_expected_always_true(self) -> None:
        assert assert_contains_text("anything", "") is True

    def test_empty_output_empty_expected(self) -> None:
        assert assert_contains_text("", "") is True

    def test_empty_output_nonempty_expected(self) -> None:
        assert assert_contains_text("", "hello") is False

    def test_exact_equal_strings(self) -> None:
        assert assert_contains_text("same", "same") is True

    def test_unicode_content(self) -> None:
        assert assert_contains_text("こんにちは世界", "世界") is True

    @given(st.text(), st.text())
    @settings(max_examples=100)
    def test_property_symmetric_with_str_in(
        self, output: str, expected: str
    ) -> None:
        """assert_contains_text must agree with Python's 'in' operator (lowercased)."""
        result = assert_contains_text(output, expected)
        assert result == (expected.lower() in output.lower())


# ---------------------------------------------------------------------------
# assert_calls_tool
# ---------------------------------------------------------------------------


class TestAssertCallsTool:
    """Tests for assert_calls_tool."""

    def test_json_tool_key(self) -> None:
        output = json.dumps({"tool": "search_web", "args": {}})
        assert assert_calls_tool(output, "search_web") is True

    def test_json_function_name(self) -> None:
        output = json.dumps({"function": {"name": "get_weather", "args": {}}})
        assert assert_calls_tool(output, "get_weather") is True

    def test_openai_tool_calls_array(self) -> None:
        output = json.dumps(
            {
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "calculator", "arguments": "{}"},
                    }
                ]
            }
        )
        assert assert_calls_tool(output, "calculator") is True

    def test_openai_tool_calls_wrong_name(self) -> None:
        output = json.dumps(
            {
                "tool_calls": [
                    {"function": {"name": "calculator", "arguments": "{}"}}
                ]
            }
        )
        assert assert_calls_tool(output, "search_web") is False

    def test_plain_text_function_call_style(self) -> None:
        output = "I will call search_web(query='test')"
        assert assert_calls_tool(output, "search_web") is True

    def test_xml_tag_style(self) -> None:
        assert assert_calls_tool("<calculator>2+2</calculator>", "calculator") is True

    def test_word_boundary_match(self) -> None:
        output = "Using search_web to find results"
        assert assert_calls_tool(output, "search_web") is True

    def test_tool_not_mentioned(self) -> None:
        assert assert_calls_tool("I did not call any tool.", "search_web") is False

    def test_non_json_text_with_tool_name(self) -> None:
        assert assert_calls_tool("call get_weather now", "get_weather") is True

    def test_empty_output(self) -> None:
        assert assert_calls_tool("", "search_web") is False

    def test_case_insensitive_plain_text(self) -> None:
        # word boundary pattern uses re.IGNORECASE
        assert assert_calls_tool("SEARCH_WEB result", "search_web") is True

    def test_json_with_wrong_tool_key(self) -> None:
        output = json.dumps({"tool": "other_tool"})
        assert assert_calls_tool(output, "search_web") is False

    def test_json_function_wrong_name(self) -> None:
        output = json.dumps({"function": {"name": "wrong"}})
        assert assert_calls_tool(output, "get_weather") is False

    def test_partial_name_does_not_false_negative(self) -> None:
        # "search_web" should match even when surrounded by other text
        output = "result from search_web returned"
        assert assert_calls_tool(output, "search_web") is True

    def test_json_non_dict_top_level(self) -> None:
        # Top-level JSON list — should fall back to text heuristics
        output = json.dumps([{"tool": "search_web"}])
        # The list is parsed but it's not a dict, so json path won't find it;
        # text heuristics (word boundary) should still match "search_web"
        assert assert_calls_tool(output, "search_web") is True


# ---------------------------------------------------------------------------
# assert_no_pii
# ---------------------------------------------------------------------------


class TestAssertNoPii:
    """Tests for assert_no_pii — PII detection."""

    def test_clean_text(self) -> None:
        assert assert_no_pii("The sky is blue and the grass is green.") is True

    def test_email_detected(self) -> None:
        assert assert_no_pii("Contact us at user@example.com for help.") is False

    def test_email_at_start(self) -> None:
        assert assert_no_pii("admin@corp.io is the email.") is False

    def test_phone_10_digit(self) -> None:
        assert assert_no_pii("Call 555-867-5309 for assistance.") is False

    def test_phone_with_country_code(self) -> None:
        assert assert_no_pii("Dial +1 800 555 1234 now.") is False

    def test_ssn_with_dashes(self) -> None:
        assert assert_no_pii("SSN: 123-45-6789") is False

    def test_credit_card_visa(self) -> None:
        assert assert_no_pii("Card: 4111 1111 1111 1111") is False

    def test_credit_card_mastercard(self) -> None:
        assert assert_no_pii("MC: 5500 0000 0000 0004") is False

    def test_empty_string(self) -> None:
        assert assert_no_pii("") is True

    def test_numbers_not_pii(self) -> None:
        assert assert_no_pii("There are 42 items and 7 categories.") is True

    def test_mixed_clean_and_pii(self) -> None:
        # Even one PII pattern makes the whole output fail
        assert assert_no_pii("Great! Your email is user@example.com.") is False

    @given(st.text(alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Zs"))))
    @settings(max_examples=50)
    def test_property_letters_and_spaces_are_clean(self, text: str) -> None:
        """Pure alphabetic + space text should never trigger PII detection."""
        assert assert_no_pii(text) is True


# ---------------------------------------------------------------------------
# assert_max_tokens
# ---------------------------------------------------------------------------


class TestAssertMaxTokens:
    """Tests for assert_max_tokens."""

    def test_within_limit(self) -> None:
        assert assert_max_tokens(50, 100) is True

    def test_at_exact_limit(self) -> None:
        assert assert_max_tokens(100, 100) is True

    def test_exceeds_limit(self) -> None:
        assert assert_max_tokens(101, 100) is False

    def test_zero_tokens_any_limit(self) -> None:
        assert assert_max_tokens(0, 0) is True

    def test_zero_tokens_positive_limit(self) -> None:
        assert assert_max_tokens(0, 1) is True

    @given(
        st.integers(min_value=0, max_value=10_000),
        st.integers(min_value=0, max_value=10_000),
    )
    def test_property_consistent_with_lte(
        self, tokens: int, max_tokens: int
    ) -> None:
        assert assert_max_tokens(tokens, max_tokens) == (tokens <= max_tokens)


# ---------------------------------------------------------------------------
# assert_max_latency
# ---------------------------------------------------------------------------


class TestAssertMaxLatency:
    """Tests for assert_max_latency."""

    def test_within_limit(self) -> None:
        assert assert_max_latency(100.0, 500.0) is True

    def test_at_exact_limit(self) -> None:
        assert assert_max_latency(500.0, 500.0) is True

    def test_exceeds_limit(self) -> None:
        assert assert_max_latency(500.1, 500.0) is False

    def test_zero_latency(self) -> None:
        assert assert_max_latency(0.0, 0.0) is True

    @given(
        st.floats(min_value=0.0, max_value=100_000.0, allow_nan=False),
        st.floats(min_value=0.0, max_value=100_000.0, allow_nan=False),
    )
    def test_property_consistent_with_lte(
        self, latency: float, max_ms: float
    ) -> None:
        assert assert_max_latency(latency, max_ms) == (latency <= max_ms)


# ---------------------------------------------------------------------------
# assert_valid_json
# ---------------------------------------------------------------------------


class TestAssertValidJson:
    """Tests for assert_valid_json."""

    def test_valid_object(self) -> None:
        assert assert_valid_json('{"key": "value"}') is True

    def test_valid_array(self) -> None:
        assert assert_valid_json("[1, 2, 3]") is True

    def test_valid_string(self) -> None:
        assert assert_valid_json('"hello"') is True

    def test_valid_number(self) -> None:
        assert assert_valid_json("42") is True

    def test_valid_null(self) -> None:
        assert assert_valid_json("null") is True

    def test_valid_boolean_true(self) -> None:
        assert assert_valid_json("true") is True

    def test_valid_boolean_false(self) -> None:
        assert assert_valid_json("false") is True

    def test_valid_with_leading_whitespace(self) -> None:
        assert assert_valid_json('   {"a": 1}') is True

    def test_valid_with_trailing_newline(self) -> None:
        assert assert_valid_json('{"a": 1}\n') is True

    def test_invalid_trailing_comma(self) -> None:
        assert assert_valid_json('{"key": "value",}') is False

    def test_invalid_plain_text(self) -> None:
        assert assert_valid_json("this is not json") is False

    def test_invalid_single_quotes(self) -> None:
        assert assert_valid_json("{'key': 'value'}") is False

    def test_empty_string(self) -> None:
        assert assert_valid_json("") is False

    def test_empty_whitespace(self) -> None:
        assert assert_valid_json("   ") is False

    def test_nested_json(self) -> None:
        nested = json.dumps({"a": {"b": {"c": [1, 2, 3]}}})
        assert assert_valid_json(nested) is True

    @given(st.text())
    @settings(max_examples=100)
    def test_property_never_raises(self, text: str) -> None:
        """assert_valid_json must always return a bool, never raise."""
        result = assert_valid_json(text)
        assert isinstance(result, bool)

    @given(
        st.one_of(
            st.dictionaries(st.text(), st.integers()),
            st.lists(st.integers()),
        )
    )
    @settings(max_examples=50)
    def test_property_serialised_data_always_valid(
        self, data: dict | list
    ) -> None:
        """Serialised Python data structures must always pass the check."""
        assert assert_valid_json(json.dumps(data)) is True


# ---------------------------------------------------------------------------
# assert_matches_schema
# ---------------------------------------------------------------------------


class TestAssertMatchesSchema:
    """Tests for assert_matches_schema (uses jsonschema or lightweight fallback)."""

    def test_valid_object_with_required_key(self) -> None:
        schema = {"type": "object", "required": ["status"]}
        output = json.dumps({"status": "ok"})
        assert assert_matches_schema(output, schema) is True

    def test_missing_required_key(self) -> None:
        schema = {"type": "object", "required": ["status"]}
        output = json.dumps({"other": "value"})
        assert assert_matches_schema(output, schema) is False

    def test_wrong_type_object_vs_array(self) -> None:
        schema = {"type": "object"}
        output = json.dumps([1, 2, 3])
        assert assert_matches_schema(output, schema) is False

    def test_array_type(self) -> None:
        schema = {"type": "array"}
        output = json.dumps([1, 2, 3])
        assert assert_matches_schema(output, schema) is True

    def test_string_type(self) -> None:
        schema = {"type": "string"}
        output = json.dumps("hello")
        assert assert_matches_schema(output, schema) is True

    def test_integer_type(self) -> None:
        schema = {"type": "integer"}
        output = json.dumps(42)
        assert assert_matches_schema(output, schema) is True

    def test_number_accepts_integer(self) -> None:
        schema = {"type": "number"}
        output = json.dumps(7)
        # JSON Schema: integer is a subtype of number
        assert assert_matches_schema(output, schema) is True

    def test_invalid_json_returns_false(self) -> None:
        schema = {"type": "object"}
        assert assert_matches_schema("not json", schema) is False

    def test_empty_schema_matches_anything(self) -> None:
        # An empty schema {} is always valid
        output = json.dumps({"a": 1})
        assert assert_matches_schema(output, {}) is True

    def test_properties_type_check(self) -> None:
        schema = {
            "type": "object",
            "properties": {"count": {"type": "integer"}},
        }
        output = json.dumps({"count": 5})
        assert assert_matches_schema(output, schema) is True

    def test_properties_wrong_nested_type(self) -> None:
        schema = {
            "type": "object",
            "required": ["count"],
            "properties": {"count": {"type": "integer"}},
        }
        output = json.dumps({"count": "five"})
        assert assert_matches_schema(output, schema) is False


# ---------------------------------------------------------------------------
# _lightweight_schema_check (internal helper)
# ---------------------------------------------------------------------------


class TestLightweightSchemaCheck:
    """Direct unit tests for the private _lightweight_schema_check helper."""

    def test_object_type_passes_dict(self) -> None:
        assert _lightweight_schema_check({"a": 1}, {"type": "object"}) is True

    def test_object_type_fails_list(self) -> None:
        assert _lightweight_schema_check([1, 2], {"type": "object"}) is False

    def test_array_type_passes_list(self) -> None:
        assert _lightweight_schema_check([1, 2], {"type": "array"}) is True

    def test_string_type_passes_str(self) -> None:
        assert _lightweight_schema_check("hello", {"type": "string"}) is True

    def test_integer_type_passes_int(self) -> None:
        assert _lightweight_schema_check(5, {"type": "integer"}) is True

    def test_number_type_passes_int(self) -> None:
        assert _lightweight_schema_check(5, {"type": "number"}) is True

    def test_boolean_type_passes_bool(self) -> None:
        assert _lightweight_schema_check(True, {"type": "boolean"}) is True

    def test_null_type_passes_none(self) -> None:
        assert _lightweight_schema_check(None, {"type": "null"}) is True

    def test_required_keys_present(self) -> None:
        assert _lightweight_schema_check(
            {"a": 1, "b": 2}, {"required": ["a", "b"]}
        ) is True

    def test_required_key_missing(self) -> None:
        assert _lightweight_schema_check(
            {"a": 1}, {"required": ["a", "b"]}
        ) is False

    def test_no_type_in_schema(self) -> None:
        # When schema has no 'type', no type check is applied
        assert _lightweight_schema_check("anything", {}) is True

    def test_unknown_type_string_passes(self) -> None:
        # Unknown type names should not crash — unknown type maps to None
        assert _lightweight_schema_check({"x": 1}, {"type": "unknown_type"}) is True

    def test_non_dict_data_with_required_skip(self) -> None:
        # Non-dict data can't have required keys checked — returns True after type check
        assert _lightweight_schema_check([1, 2], {"required": ["x"]}) is True
