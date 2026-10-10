"""Robust Redis service client with graceful in-memory failover.

If Redis is running (via Docker container or localhost:6379), it is used as the primary
store for sliding-window rate limiting and semantic answer caching.
If Docker or Redis is unreachable, this module silently and transparently fails over
to an internal thread-safe in-memory store so the pipeline never crashes or errors out.
"""

import hashlib
import json
import logging
import os
import time
from collections import defaultdict, deque
from threading import Lock
from typing import Any, Optional

logger = logging.getLogger("redis_service")

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
REDIS_TIMEOUT = float(os.getenv("REDIS_TIMEOUT", "0.5"))

class RedisServiceClient:
    def __init__(self) -> None:
        self._redis = None
        self._available = False
        self._last_check = 0.0
        self._check_interval = 30.0  # retry connecting every 30s
        self._lock = Lock()

        # In-memory failover structures
        self._mem_rate_events = defaultdict(deque)
        self._mem_cache: dict[str, tuple[float, str]] = {}
        self._try_connect()

    def _try_connect(self) -> bool:
        now = time.monotonic()
        if self._available:
            return True
        if now - self._last_check < self._check_interval:
            return False

        self._last_check = now
        try:
            import redis
            client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                password=REDIS_PASSWORD,
                socket_timeout=REDIS_TIMEOUT,
                socket_connect_timeout=REDIS_TIMEOUT,
                decode_responses=True,
            )
            client.ping()
            self._redis = client
            self._available = True
            logger.info("Connected to Redis at %s:%s successfully.", REDIS_HOST, REDIS_PORT)
            return True
        except Exception as e:
            self._available = False
            self._redis = None
            return False

    @property
    def is_redis_live(self) -> bool:
        return self._try_connect()

    # =========================================================================
    # 1. RATE LIMITING (Redis Sliding-Window with In-Memory Graceful Fallback)
    # =========================================================================
    def check_and_consume_rate_limit(
        self,
        key: str,
        category: str,
        limit: int,
        window_seconds: int = 60,
    ) -> tuple[bool, int]:
        """Consumes 1 token. Returns (allowed: bool, current_count: int)."""
        redis_key = f"ratelimit:{category}:{key}"
        now = time.time()

        if self.is_redis_live and self._redis:
            try:
                pipeline = self._redis.pipeline()
                # Remove timestamps older than window
                pipeline.zremrangebyscore(redis_key, "-inf", now - window_seconds)
                # Count remaining
                pipeline.zcard(redis_key)
                # Add current timestamp
                pipeline.zadd(redis_key, {f"{now}": now})
                # Set TTL
                pipeline.expire(redis_key, window_seconds + 5)
                results = pipeline.execute()

                current_count = results[1]
                if current_count >= limit:
                    # Over limit -> remove the just-added token
                    self._redis.zrem(redis_key, f"{now}")
                    return False, current_count
                return True, current_count + 1
            except Exception as e:
                logger.warning("Redis rate limiter error, failing over to memory: %s", e)
                self._available = False

        # Graceful In-Memory Fallback
        with self._lock:
            event_key = (key, category)
            events = self._mem_rate_events[event_key]
            m_now = time.monotonic()
            while events and m_now - events[0] >= window_seconds:
                events.popleft()
            if len(events) >= limit:
                return False, len(events)
            events.append(m_now)
            return True, len(events)

    # =========================================================================
    # 2. SEMANTIC ANSWER CACHE (Redis TTL with In-Memory Graceful Fallback)
    # =========================================================================
    def _cache_key(self, thread_id: str, query: str) -> str:
        norm = " ".join(query.strip().lower().split())
        digest = hashlib.sha256(norm.encode("utf-8")).hexdigest()[:24]
        return f"cache:answer:{thread_id}:{digest}"

    def get_cached_answer(self, thread_id: str, query: str) -> Optional[dict]:
        key = self._cache_key(thread_id, query)
        now = time.time()

        if self.is_redis_live and self._redis:
            try:
                raw = self._redis.get(key)
                if raw:
                    return json.loads(raw)
            except Exception as e:
                logger.warning("Redis cache read error, failing over to memory: %s", e)
                self._available = False

        # In-Memory Fallback
        with self._lock:
            if key in self._mem_cache:
                exp, val = self._mem_cache[key]
                if exp > now:
                    return json.loads(val)
                del self._mem_cache[key]
        return None

    def set_cached_answer(
        self,
        thread_id: str,
        query: str,
        payload: dict,
        ttl_seconds: int = 3600,
    ) -> None:
        key = self._cache_key(thread_id, query)
        raw = json.dumps(payload, default=str)
        now = time.time()

        if self.is_redis_live and self._redis:
            try:
                self._redis.setex(key, ttl_seconds, raw)
                return
            except Exception as e:
                logger.warning("Redis cache write error, falling back to memory: %s", e)
                self._available = False

        # In-Memory Fallback
        with self._lock:
            self._mem_cache[key] = (now + ttl_seconds, raw)

redis_service = RedisServiceClient()
