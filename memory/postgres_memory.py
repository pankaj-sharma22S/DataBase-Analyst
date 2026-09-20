"""PostgreSQL-backed conversation, long-term, and preference memory."""

import re
import time
from typing import Any

import psycopg

from security.pii_detector import SECRET_VALUE, mask_text
from .long_term_memory import USEFUL_PATTERNS, _is_sensitive
from .preference_memory import PREFERENCE_PATTERN


class PostgresMemoryStore:
    def __init__(self, database_url: str):
        self.database_url = database_url.replace("postgresql+psycopg://", "postgresql://", 1)
        self.connection = psycopg.connect(self.database_url, connect_timeout=5, autocommit=True)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS conversations ("
            "id BIGSERIAL PRIMARY KEY, thread_id TEXT NOT NULL, role TEXT NOT NULL, "
            "content TEXT NOT NULL, created DOUBLE PRECISION NOT NULL)"
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_pg_conversations_thread "
            "ON conversations(thread_id, created)"
        )
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS memories ("
            "id BIGSERIAL PRIMARY KEY, thread_id TEXT NOT NULL, memory TEXT NOT NULL, "
            "created DOUBLE PRECISION NOT NULL)"
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_pg_memories_thread ON memories(thread_id, created)"
        )
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS preferences ("
            "id BIGSERIAL PRIMARY KEY, thread_id TEXT NOT NULL, preference TEXT NOT NULL, "
            "created DOUBLE PRECISION NOT NULL)"
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_pg_preferences_thread "
            "ON preferences(thread_id, created)"
        )

    @staticmethod
    def _rank(values: list[str], query: str, limit: int) -> list[str]:
        query_words = set(re.findall(r"\w+", query.lower()))
        ranked = sorted(
            ((len(query_words & set(re.findall(r"\w+", value.lower()))), index, value)
             for index, value in enumerate(values)),
            key=lambda item: (item[0], -item[1]),
            reverse=True,
        )
        return [value for score, _, value in ranked[:limit] if score > 0]

    def add_conversation(self, thread_id: str, role: str, content: str) -> None:
        self.connection.execute(
            "INSERT INTO conversations(thread_id, role, content, created) VALUES (%s, %s, %s, %s)",
            (thread_id, role, mask_text(content), time.time()),
        )

    def retrieve_conversations(self, thread_id: str, query: str, limit: int = 4) -> list[dict[str, str]]:
        rows = self.connection.execute(
            "SELECT role, content FROM conversations WHERE thread_id = %s "
            "ORDER BY created DESC LIMIT 100",
            (thread_id,),
        ).fetchall()
        ranked = self._rank([str(content) for _, content in rows], query, limit)
        by_content = {str(content): {"role": role, "content": str(content)} for role, content in rows}
        return [by_content[content] for content in ranked]

    def extract_long_term(self, thread_id: str, text: str) -> None:
        if _is_sensitive(text):
            return
        for pattern in USEFUL_PATTERNS:
            for match in pattern.finditer(text or ""):
                memory = match.group(1).strip().rstrip(".!?")
                if memory and not _is_sensitive(memory):
                    self.connection.execute(
                        "INSERT INTO memories(thread_id, memory, created) VALUES (%s, %s, %s)",
                        (thread_id, memory, time.time()),
                    )

    def retrieve_long_term(self, thread_id: str, query: str, limit: int = 4) -> list[str]:
        rows = self.connection.execute(
            "SELECT memory FROM memories WHERE thread_id = %s ORDER BY created DESC LIMIT 100",
            (thread_id,),
        ).fetchall()
        return self._rank([str(memory) for (memory,) in rows], query, limit)

    def extract_preference(self, thread_id: str, text: str) -> None:
        if SECRET_VALUE.search(text or ""):
            return
        for match in PREFERENCE_PATTERN.finditer(text or ""):
            preference = mask_text(match.group(1).strip().rstrip(".!?"))
            if preference and not SECRET_VALUE.search(preference):
                self.connection.execute(
                    "INSERT INTO preferences(thread_id, preference, created) VALUES (%s, %s, %s)",
                    (thread_id, preference, time.time()),
                )

    def retrieve_preferences(self, thread_id: str, query: str, limit: int = 4) -> list[str]:
        rows = self.connection.execute(
            "SELECT preference FROM preferences WHERE thread_id = %s ORDER BY created DESC LIMIT 100",
            (thread_id,),
        ).fetchall()
        return self._rank([str(preference) for (preference,) in rows], query, limit)

    def close(self) -> None:
        self.connection.close()
