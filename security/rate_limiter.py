"""Thread-safe sliding-window request limiter."""

from collections import defaultdict, deque
from threading import Lock
import time


class RateLimitError(RuntimeError):
    pass


class RateLimiter:
    def __init__(self, window_seconds: int, limits: dict[str, int]):
        self.window_seconds = window_seconds
        self.limits = limits
        self._events = defaultdict(deque)
        self._lock = Lock()

    def consume(self, key: str, category: str) -> None:
        now = time.monotonic()
        limit = self.limits[category]
        event_key = (key, category)
        with self._lock:
            events = self._events[event_key]
            while events and now - events[0] >= self.window_seconds:
                events.popleft()
            if len(events) >= limit:
                raise RateLimitError(f"Security rate limit exceeded for {category}.")
            events.append(now)
