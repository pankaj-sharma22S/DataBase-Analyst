"""Thread-safe sliding-window rate limiter powered by Redis with in-memory fallback."""

from .redis_service import redis_service


class RateLimitError(RuntimeError):
    pass


class RateLimiter:
    def __init__(self, window_seconds: int, limits: dict[str, int]):
        self.window_seconds = window_seconds
        self.limits = limits

    def consume(self, key: str, category: str) -> None:
        limit = self.limits.get(category, 60)
        allowed, current_count = redis_service.check_and_consume_rate_limit(
            key=key,
            category=category,
            limit=limit,
            window_seconds=self.window_seconds,
        )
        if not allowed:
            raise RateLimitError(f"Security rate limit exceeded for {category} ({current_count}/{limit}).")
