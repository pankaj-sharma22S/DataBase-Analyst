"""Configuration for the Insight AI security gateway."""

from dataclasses import dataclass
import os


def _int_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, default)))
    except (TypeError, ValueError):
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return max(0.1, float(os.getenv(name, default)))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class SecurityConfig:
    request_limit: int = 30
    sql_limit: int = 60
    diagnostic_limit: int = 12
    rate_window_seconds: int = 60
    overall_timeout_seconds: float = 120.0
    llm_timeout_seconds: float = 30.0
    sql_timeout_seconds: float = 30.0
    diagnostic_timeout_seconds: float = 30.0
    max_rows: int = 1000
    max_result_bytes: int = 500_000
    max_sql_executions: int = 8
    max_diagnostic_queries: int = 5

    @classmethod
    def from_env(cls) -> "SecurityConfig":
        return cls(
            request_limit=_int_env("SECURITY_REQUEST_LIMIT", 30),
            sql_limit=_int_env("SECURITY_SQL_LIMIT", 60),
            diagnostic_limit=_int_env("SECURITY_DIAGNOSTIC_LIMIT", 12),
            rate_window_seconds=_int_env("SECURITY_RATE_WINDOW_SECONDS", 60),
            overall_timeout_seconds=_float_env("SECURITY_OVERALL_TIMEOUT_SECONDS", 120.0),
            llm_timeout_seconds=_float_env("SECURITY_LLM_TIMEOUT_SECONDS", 30.0),
            sql_timeout_seconds=_float_env("SECURITY_SQL_TIMEOUT_SECONDS", 30.0),
            diagnostic_timeout_seconds=_float_env("SECURITY_DIAGNOSTIC_TIMEOUT_SECONDS", 30.0),
            max_rows=_int_env("SECURITY_MAX_ROWS", 1000),
            max_result_bytes=_int_env("SECURITY_MAX_RESULT_BYTES", 500_000),
            max_sql_executions=_int_env("SECURITY_MAX_SQL_EXECUTIONS", 8),
            max_diagnostic_queries=_int_env("SECURITY_MAX_DIAGNOSTIC_QUERIES", 5),
        )
