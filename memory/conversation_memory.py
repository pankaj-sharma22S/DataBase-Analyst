"""Separate persistent conversation history with lightweight relevance retrieval."""

from pathlib import Path
import sqlite3
import time
import re

from security.pii_detector import mask_text


class ConversationMemory:
    def __init__(self, database_path: str | Path = "memory/conversations.sqlite3"):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(self.database_path), check_same_thread=False)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS conversations ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, thread_id TEXT NOT NULL, "
            "role TEXT NOT NULL, content TEXT NOT NULL, created REAL NOT NULL)"
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_conversations_thread ON conversations(thread_id, created)"
        )
        self.connection.commit()

    def add(self, thread_id: str, role: str, content: str) -> None:
        safe_content = mask_text(content)
        self.connection.execute(
            "INSERT INTO conversations(thread_id, role, content, created) VALUES (?, ?, ?, ?)",
            (thread_id, role, safe_content, time.time()),
        )
        self.connection.commit()

    def retrieve(self, thread_id: str, query: str, limit: int = 4) -> list[dict[str, str]]:
        rows = self.connection.execute(
            "SELECT role, content FROM conversations WHERE thread_id = ? ORDER BY created DESC LIMIT 100",
            (thread_id,),
        ).fetchall()
        words = set(re.findall(r"\w+", query.lower()))
        ranked = []
        for role, content in rows:
            overlap = len(words & set(re.findall(r"\w+", content.lower())))
            ranked.append((overlap, {"role": role, "content": content}))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [item for score, item in ranked[:limit] if score > 0]

    def close(self) -> None:
        self.connection.close()
