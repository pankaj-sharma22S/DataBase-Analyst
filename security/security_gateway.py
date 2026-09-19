"""Single fail-closed security boundary used by the application."""

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import Any, Callable

from .pii_detector import bounded_rows, mask_rows, mask_text
from .prompt_injection import detect_prompt_injection
from .rate_limiter import RateLimitError, RateLimiter
from .security_config import SecurityConfig
from .sql_security import SQLSecurityError, validate_read_only_sql


class SecurityError(RuntimeError):
    pass


class SecurityGateway:
    def __init__(self, config: SecurityConfig | None = None):
        self.config = config or SecurityConfig.from_env()
        self.rate_limiter = RateLimiter(
            self.config.rate_window_seconds,
            {
                "request": self.config.request_limit,
                "sql": self.config.sql_limit,
                "diagnostic": self.config.diagnostic_limit,
            },
        )
        self._sql_counts: dict[str, int] = {}
        self._diagnostic_counts: dict[str, int] = {}

    def check_request(self, request: str, request_id: str) -> None:
        try:
            self.rate_limiter.consume(request_id, "request")
        except RateLimitError as exc:
            raise SecurityError(str(exc)) from exc
        self._sql_counts[request_id] = 0
        self._diagnostic_counts[request_id] = 0
        if detect_prompt_injection(request):
            raise SecurityError("Security policy blocked this request because it contains an unsafe instruction.")

    def validate_sql(self, sql: str, request_id: str, diagnostic: bool = False) -> str:
        category = "diagnostic" if diagnostic else "sql"
        try:
            self.rate_limiter.consume(request_id, category)
        except RateLimitError as exc:
            raise SecurityError(str(exc)) from exc
        counts = self._diagnostic_counts if diagnostic else self._sql_counts
        limit = self.config.max_diagnostic_queries if diagnostic else self.config.max_sql_executions
        counts[request_id] = counts.get(request_id, 0) + 1
        if counts[request_id] > limit:
            raise SecurityError("Security policy blocked excessive SQL executions for this request.")
        try:
            return validate_read_only_sql(sql)
        except SQLSecurityError as exc:
            raise SecurityError(str(exc)) from exc

    def mask_for_llm(self, value: Any) -> Any:
        if isinstance(value, str):
            return mask_text(value)
        if isinstance(value, list) and all(isinstance(item, dict) for item in value):
            return bounded_rows(value, self.config.max_rows, self.config.max_result_bytes)
        if isinstance(value, dict):
            return {key: self.mask_for_llm(item) for key, item in value.items()}
        return value

    def mask_for_output(self, value: str) -> str:
        return mask_text(value)

    def bound_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return bounded_rows(rows, self.config.max_rows, self.config.max_result_bytes)

    def run_with_timeout(self, function: Callable[[], Any], timeout_seconds: float) -> Any:
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(function)
        try:
            return future.result(timeout=timeout_seconds)
        except FutureTimeout as exc:
            future.cancel()
            raise SecurityError("Security policy timeout exceeded.") from exc
        finally:
            executor.shutdown(wait=False, cancel_futures=True)


security_gateway = SecurityGateway()
