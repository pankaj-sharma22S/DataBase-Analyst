"""Persistent, conservative memory of reusable non-sensitive user context."""

from pathlib import Path
import re
import sqlite3
import time

from security.pii_detector import SECRET_VALUE


SENSITIVE = re.compile(
    r"(?i)(password|passwd|pin|cvv|cvc|otp|api[_ -]?key|token|secret|private[_ -]?key|credential|bank[_ -]?account|iban)"
)


def _is_sensitive(text: str) -> bool:
    return bool(SENSITIVE.search(text or "") or SECRET_VALUE.search(text or ""))
USEFUL_PATTERNS = (
    re.compile(r"(?i)\b(?:i\s+prefer|my\s+preference\s+is|please\s+always)\s+(.{3,200})"),
    re.compile(r"(?i)\b(?:my\s+name\s+is|i\s+work\s+as|our\s+business\s+is)\s+(.{2,120})"),
    re.compile(r"(?i)\b(?:remember|keep\s+in\s+mind)\s+(?:that\s+)?(.{3,200})"),
)


class LongTermMemory:
    def __init__(self, database_path: str | Path = "memory/long_term.sqlite3"):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(self.database_path), check_same_thread=False)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS memories ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, thread_id TEXT NOT NULL, "
            "memory TEXT NOT NULL, created REAL NOT NULL)"
        )
        self.connection.commit()

    def extract_and_store(self, thread_id: str, text: str) -> list[str]:
        if _is_sensitive(text):
            return []
        extracted = []
        for pattern in USEFUL_PATTERNS:
            for match in pattern.finditer(text or ""):
                memory = match.group(1).strip().rstrip(".!?")
                if memory and not _is_sensitive(memory):
                    self.connection.execute(
                        "INSERT INTO memories(thread_id, memory, created) VALUES (?, ?, ?)",
                        (thread_id, memory, time.time()),
                    )
                    extracted.append(memory)
        if extracted:
            self.connection.commit()
        return extracted

    def retrieve(self, thread_id: str, query: str, limit: int = 4) -> list[str]:
        rows = self.connection.execute(
            "SELECT memory FROM memories WHERE thread_id = ? ORDER BY created DESC LIMIT 100",
            (thread_id,),
        ).fetchall()
        words = set(re.findall(r"\w+", query.lower()))
        ranked = sorted(
            ((len(words & set(re.findall(r"\w+", memory.lower()))), memory) for (memory,) in rows),
            key=lambda item: item[0],
            reverse=True,
        )
        return [memory for score, memory in ranked[:limit] if score > 0]

    def close(self) -> None:
        self.connection.close()
