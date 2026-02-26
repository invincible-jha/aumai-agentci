"""Pydantic v2 models for aumai-agentci."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field, field_validator

__all__ = [
    "MockLLMResponse",
    "MockLLMConfig",
    "AgentTestCase",
    "AgentTestResult",
    "TestSuiteResult",
    "AgentTestConfig",
]


class MockLLMResponse(BaseModel):
    """A single simulated LLM response."""

    content: str
    model: str
    tokens_used: int = Field(ge=0)
    latency_ms: float = Field(ge=0.0)
    finish_reason: str = "stop"


class MockLLMConfig(BaseModel):
    """Configuration for the mock LLM provider."""

    model_name: str
    responses: list[MockLLMResponse] = Field(default_factory=list)
    default_latency_ms: float = Field(default=50.0, ge=0.0)
    failure_rate: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0


class AgentTestCase(BaseModel):
    """A single agent test case loaded from a YAML file."""

    name: str
    description: str = ""
    input_messages: list[dict[str, object]] = Field(default_factory=list)
    expected_behavior: dict[str, object] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)

    @field_validator("expected_behavior", mode="before")
    @classmethod
    def validate_expected_behavior(
        cls, value: dict[str, object]
    ) -> dict[str, object]:
        """Ensure expected_behavior only contains recognised keys."""
        allowed_keys = {
            "contains_text",
            "calls_tools",
            "max_tokens",
            "max_latency_ms",
            "no_pii",
            "valid_json",
            "matches_schema",
        }
        unknown = set(value.keys()) - allowed_keys
        if unknown:
            raise ValueError(
                f"Unknown expected_behavior keys: {unknown}. "
                f"Allowed: {allowed_keys}"
            )
        return value


class AgentTestResult(BaseModel):
    """Result produced by running a single AgentTestCase."""

    test_case_name: str
    passed: bool
    actual_output: str
    assertions_passed: list[str] = Field(default_factory=list)
    assertions_failed: list[str] = Field(default_factory=list)
    duration_ms: float = Field(ge=0.0)
    tokens_used: int = Field(default=0, ge=0)


class TestSuiteResult(BaseModel):
    """Aggregated result for a full test suite run."""

    suite_name: str
    results: list[AgentTestResult] = Field(default_factory=list)
    total: int = Field(ge=0)
    passed: int = Field(ge=0)
    failed: int = Field(ge=0)
    duration_ms: float = Field(ge=0.0)


class AgentTestConfig(BaseModel):
    """Top-level configuration for a test run."""

    test_dir: str
    mock_config: MockLLMConfig | None = None
    timeout_seconds: float = Field(default=30.0, gt=0.0)
    parallel: bool = False
